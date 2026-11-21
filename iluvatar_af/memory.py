"""捕获期的显存规划：一个可复用的 arena（CUDA 版）。

为什么需要它：cudaGraph 把每个 kernel 用到的缓冲地址写死，所以中间张量的
复用必须发生在**捕获期间**；而流捕获期间不允许 ``cudaMalloc``。因此捕获前
先把整张图要用的显存成块要下来，捕获时只在这几块里切。

与 ``cuda/memory.py`` 的区别只有容器：那边返回 cupy 的 ``MemoryPointer``，
这边返回只带 ``ptr`` / ``nbytes`` 的切片持有者。算法逐条相同——best-fit、
释放即合并相邻块——因为"确定性布局 + 可报告的峰值"这两条要求与厂商无关。

另有一处必要的额外步骤：``recycle()``。CUDA 侧的驱动允许在捕获期扩容
（``cudaMalloc`` 不成问题），CUDA 侧明确拒绝 ``cudaMalloc``，所以捕获前必须
把预热轮占着的地址整体收回，让捕获轮从已有的块里切。

回收点同样由执行器显式给出（``Executor.dispose`` -> ``CudaBackend.dispose``）：
``DeviceArray`` 析构时不会回调分配器，靠引用计数回收在这条路径上不成立。
"""

from __future__ import annotations

import bisect
from typing import List, Tuple

from aetherflow.backends.contracts import MemoryStats
from . import runtime

_ALIGN = 512  # cudaMalloc 至少 256 字节对齐，取 512 兼顾 float4 向量化访问
_CHUNK_START = 1 << 20  # 小块粒度 1 MiB
_CHUNK_MAX = 8 << 20  # 单块上限；更大的请求按精确大小单独成块


def _align_up(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class _Slice:
    """从 arena 切出的一块。

    刻意**不带** ``__del__``：这个地址属于整块 chunk，单独回收它会破坏
    arena 的账；真正归还由 ``Executor.dispose`` 显式触发。
    """

    __slots__ = ("ptr", "nbytes")

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = ptr
        self.nbytes = nbytes


class Arena:
    """给捕获期用的确定性分配器。

    ``malloc`` 返回 ``_Slice``，可以直接当作 ``DeviceArray`` 的持有者；
    ``free`` 只接受本 arena 发出的地址（视图传进来会被忽略）。

    视图（切片 / reshape / 恒等转置 / 恒等广播）与源数组共用同一段地址，
    而执行器按"最后一次被读"回收的是**源数组**。``alias`` / ``unalias``
    给这段地址记账：视图还活着时源数组只记成待回收，最后一个视图死掉
    才真正还块。与 CUDA 侧的 ``cuda/memory.py`` 逐条同义。
    """

    def __init__(self, chunk_bytes: int = _CHUNK_START) -> None:
        self._chunk_bytes = chunk_bytes
        self._chunks: List[Tuple[int, int]] = []  # (绝对地址, 字节数)
        self._bump: List[int] = []  # 每个 chunk 已切走的字节数
        self._free: List[Tuple[int, int]] = []  # (绝对地址, 字节数)，按地址升序
        self._live: dict = {}  # 绝对地址 -> 字节数
        self._aliases: dict = {}  # 绝对地址 -> 还活着的视图数
        self._pending: set = set()  # 源数组已请求回收、但视图还没死
        self._live_bytes = 0
        self._peak = 0
        self._requested = 0

    # ------------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        """向驱动实际申请的总字节（这是真实的显存占用）。"""
        return sum(size for _, size in self._chunks)

    @property
    def live_bytes(self) -> int:
        return self._live_bytes

    @property
    def peak_live_bytes(self) -> int:
        """历史峰值存活字节：任何分配器都绕不过去的下界。"""
        return self._peak

    @property
    def requested_bytes(self) -> int:
        return self._requested

    def reset_stats(self) -> None:
        """把峰值与请求量清零（预热轮与捕获轮不重复计数）。"""
        self._requested = 0
        self._peak = self._live_bytes

    def recycle(self) -> None:
        """把预热轮占用的地址整体收回，捕获轮从这些块里重新切。

        为什么需要这一步：图输出不会被执行器 ``dispose``（它在 ``keep`` 里），
        所以预热结束时的账本上仍有一批"活着"的块。捕获轮跑的是同一张图、
        同样的申请顺序，但 arena 认为这些地址有主，只能向驱动要新块——而
        流捕获期间 ``cudaMalloc`` 会被拒绝。调用方在预热后丢弃了整个环境，
        这些块的 Python 引用已经没了，收回它们既正确又必要。

        收回后空闲表覆盖全部 chunk，捕获轮的 best-fit 会给出与预热轮相同的
        布局，因此不会再触发任何驱动分配。
        """
        self._live.clear()
        self._aliases.clear()
        self._pending.clear()
        self._live_bytes = 0
        self._peak = 0
        # 每个 chunk 视为已经完全切走：后续分配只从空闲表取，不再推进 bump
        self._bump = [capacity for _, capacity in self._chunks]
        self._free = sorted(self._chunks)
        self._coalesce()

    def stats(self) -> MemoryStats:
        return MemoryStats(
            live_bytes=self._live_bytes,
            peak_bytes=self._peak,
            total_bytes=self.total_bytes,
            requested_bytes=self._requested,
        )

    # ------------------------------------------------------------------

    def malloc(self, size: int) -> _Slice:
        size = _align_up(max(int(size), 1))
        self._requested += size

        for i, (addr, avail) in enumerate(self._free):
            if avail >= size:
                self._free.pop(i)
                if avail > size:
                    bisect.insort(self._free, (addr + size, avail - size))
                return self._take(addr, size)

        # 没有合适的空闲块：从最后一块的尾部切，避免每次都新申请
        if self._chunks:
            idx = len(self._chunks) - 1
            addr, capacity = self._chunks[idx]
            off = self._bump[idx]
            if off + size <= capacity:
                self._bump[idx] = off + size
                return self._take(addr + off, size)

        addr = self._new_chunk(size)
        idx = len(self._chunks) - 1
        self._bump[idx] = size
        return self._take(addr, size)

    def alias(self, ptr) -> None:
        """登记一次"视图引用了这段地址"（见类说明）。"""
        addr = int(ptr)
        self._aliases[addr] = self._aliases.get(addr, 0) + 1

    def unalias(self, ptr) -> None:
        """视图死亡：扣掉一次引用；归零且源数组已经请求过回收就真正还块。"""
        addr = int(ptr)
        count = self._aliases.get(addr, 0)
        if count <= 0:
            return
        if count > 1:
            self._aliases[addr] = count - 1
            return
        self._aliases.pop(addr, None)
        if addr in self._pending:
            self._pending.discard(addr)
            self._release(addr)

    def free(self, ptr, size=None) -> None:
        addr = ptr.ptr
        if self._aliases.get(addr):
            self._pending.add(addr)  # 视图还按这段地址读，等最后一个死掉
            return
        self._release(addr)

    def pending(self, ptr) -> bool:
        """这段地址是否"已请求回收、但还等着视图死掉"（见 ``cuda/memory.py``）。"""
        return int(ptr) in self._pending

    def _release(self, addr: int) -> None:
        got = self._live.pop(addr, None)
        if got is None:  # 不是本 arena 发出的地址（例如 slice 出的视图）
            return
        self._live_bytes -= got
        bisect.insort(self._free, (addr, got))
        self._coalesce()

    def _coalesce(self) -> None:
        """把**同一个 chunk 内**地址相邻的空闲块合成一块。

        不能跨 chunk 合并：两个 chunk 是两次独立的 ``cudaMalloc``，驱动按
        分配校验拷贝范围，地址看起来相邻也不行。合并后 best-fit 会切出
        跨越分配边界的 slice，replay 时的 D2D 直接返回 code=1（实测：24 层
        模型第二段生成的第一步状态回写）。地址相邻还不够，必须同属一块。
        """
        merged: List[Tuple[int, int]] = []
        for a, n in self._free:
            if (
                merged
                and merged[-1][0] + merged[-1][1] == a
                and self._chunk_of(merged[-1][0]) == self._chunk_of(a)
            ):
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((a, n))
        self._free = merged

    def _chunk_of(self, addr: int) -> int:
        """``addr`` 属于哪一个 chunk；不属于任何一块返回 -1。"""
        for index, (start, capacity) in enumerate(self._chunks):
            if start <= addr < start + capacity:
                return index
        return -1

    # ------------------------------------------------------------------

    def _new_chunk(self, size: int) -> int:
        """向驱动直接申请一块。捕获期调用会被 runtime 拒绝并说明原因。"""
        capacity = max(size, min(self._chunk_bytes, _CHUNK_MAX))
        addr = runtime._Runtime.instance().malloc(capacity)
        self._chunks.append((addr, capacity))
        self._bump.append(0)
        return addr

    def _take(self, addr: int, size: int) -> _Slice:
        self._live[addr] = size
        self._live_bytes += size
        if self._live_bytes > self._peak:
            self._peak = self._live_bytes
        return _Slice(addr, size)

    def close(self) -> None:
        rt = runtime._Runtime.instance()
        for addr, _ in self._chunks:
            rt.free(addr)
        self._chunks.clear()
        self._bump.clear()
        self._free.clear()
        self._live.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # 解释器退出时 CUDA 上下文可能已经没了
            pass


# 执行期分层分配器的分界。小对象按 2 的幂分档池化；比 ``_LARGE_MIN`` 还大的
# 直接向驱动要、释放即还——大块常驻会推高显存峰值，而这正是普通执行与捕获
# 期最大的诉求差异（捕获期要的是"布局确定"，普通执行要的是"峰值低"）。
_SMALL_MAX = 64 << 10
_LARGE_MIN = 2 << 20

_SMALL, _ARENA, _DIRECT = 0, 1, 2


class _PooledBlock:
    """``TieredAllocator`` 发出的持有者：析构时按原路归还。

    不能复用 ``_Buffer``：它的 ``__del__`` 直接调 ``cudaFree``，池化块一析构
    就还给驱动了，池永远复用不到。``DeviceArray`` 只要求持有者带 ``ptr`` 与
    ``nbytes``，所以换一个类就够了。
    """

    __slots__ = ("ptr", "nbytes", "_owner")

    def __init__(self, ptr: int, nbytes: int, owner: "TieredAllocator") -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self._owner = owner

    def __del__(self) -> None:
        try:
            self._owner.release(self.ptr)
        except Exception:  # 解释器退出时上下文可能已经销毁
            pass


class TieredAllocator:
    """普通执行期的分层分配器（``AllocatorContract`` 的实现）。

    三层：

    - ``<= _SMALL_MAX``：按 2 的幂分档的 free list。逐元素内核一次要几 KB，
      这类块复用率最高，分档比 best-fit 查找便宜；
    - ``<= _LARGE_MIN``：交给内部 ``Arena`` 做 best-fit（它会合并相邻空闲块，
      适合卷积 im2col 这类中等大小、大小混合的请求）；
    - 更大：直接向驱动申请，释放立刻归还。注意力 / 大 matmul 的临时块属于
      这一类，留在池里只会把峰值抬上去。

    生命周期由引用计数驱动（``_PooledBlock.__del__``），不依赖执行器
    ``dispose``——普通执行没有捕获期那种"显式回收点"。
    """

    def __init__(self, chunk_bytes: int = _CHUNK_START) -> None:
        self._arena = Arena(chunk_bytes)
        self._small: dict = {}  # class 字节数 -> 空闲地址列表
        self._small_total = 0  # 小对象池向驱动申请过的总字节
        self._sizes: dict = {}  # 地址 -> (实际字节数, 层)
        self._direct: dict = {}  # 地址 -> 字节数（当前活着的直调块）
        self._live = 0
        self._peak = 0
        self._requested = 0
        self._closed = False

    # ------------------------------------------------------------------

    @property
    def live_bytes(self) -> int:
        return self._live

    @property
    def peak_live_bytes(self) -> int:
        return self._peak

    @property
    def total_bytes(self) -> int:
        """真实占用的显存：两部分池的容量 + 当前活着的直调块。"""
        return self._small_total + self._arena.total_bytes + sum(self._direct.values())

    def reset_stats(self) -> None:
        self._requested = 0
        self._peak = self._live

    def stats(self) -> MemoryStats:
        return MemoryStats(
            live_bytes=self._live,
            peak_bytes=self._peak,
            total_bytes=self.total_bytes,
            requested_bytes=self._requested,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _class_of(nbytes: int) -> int:
        size = _ALIGN
        while size < nbytes:
            size <<= 1
        return size

    def __call__(self, nbytes: int):
        """分配器的对外入口：``runtime.set_allocator`` 要的就是这个签名。"""
        nbytes = max(int(nbytes), 1)
        self._requested += nbytes

        if nbytes <= _SMALL_MAX:
            size = self._class_of(nbytes)
            free = self._small.get(size)
            if free:
                ptr = free.pop()
            else:
                ptr = runtime._Runtime.instance().malloc(size)
                self._small_total += size
            tier = _SMALL
        elif nbytes <= _LARGE_MIN:
            block = self._arena.malloc(nbytes)
            ptr, size = block.ptr, block.nbytes
            tier = _ARENA
        else:
            ptr = runtime._Runtime.instance().malloc(nbytes)
            size = nbytes
            self._direct[ptr] = nbytes
            tier = _DIRECT

        self._sizes[ptr] = (size, tier)
        self._live += size
        if self._live > self._peak:
            self._peak = self._live
        return _PooledBlock(ptr, nbytes, self)

    def release(self, ptr: int) -> None:
        """归还一块。未登记的地址（重复析构、已 close）直接忽略。"""
        if self._closed:
            return
        got = self._sizes.pop(ptr, None)
        if got is None:
            return
        size, tier = got
        self._live -= size
        if tier == _SMALL:
            self._small.setdefault(size, []).append(ptr)
        elif tier == _ARENA:
            self._arena.free(_Slice(ptr, size))
        else:
            self._direct.pop(ptr, None)
            runtime._Runtime.instance().free(ptr)

    def close(self) -> None:
        """把所有块还给驱动。之后再 ``release`` 是空操作。"""
        if self._closed:
            return
        self._closed = True
        rt = runtime._Runtime.instance()
        for ptrs in self._small.values():
            for ptr in ptrs:
                rt.free(ptr)
        self._small.clear()
        self._small_total = 0
        self._arena.close()
        for ptr in list(self._direct):
            rt.free(ptr)
        self._direct.clear()
        self._sizes.clear()
        self._live = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

