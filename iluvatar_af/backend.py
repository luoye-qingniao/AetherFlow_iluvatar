"""天数（Iluvatar / 智铠 MR）后端插件。

它不是一个"另写的后端"，而是同一批内核语义落到第三块硬件上：

- 内核源码**逐字复用** ``cuda/codegen.py`` 的模板。天数的 Clang 接受
  ``__global__`` / ``__shared__`` / ``atomicAdd`` 这些写法，唯一需要换的是
  f16 的存取（PTX 内联汇编在天数上不成立），由 ``codegen.py`` 注入
  ``_Float16`` 版本。模板只有一份，因此不存在"CUDA 修了、天数忘了"的分叉。
- 算子的形状检查、广播折叠、分块策略、能力边界与 CUDA 共用
  ``DeviceBackend``。这些是 IR 层的语义决定，与厂商无关。**天数不是
  ``CudaBackend`` 的子类**——继承会把 cupy 容器、cuDNN 假设一起带过来。
- 容器是自己写的 ``DeviceArray``（ctypes 直连天数 ``libcudart`` / ``libcuda``），
  BLAS 走天数 ``libcublas``（CUDA 10.2 ABI）。不引入 cupy / torch 任何一个。

与 NVIDIA 的实质差异：

- warp 宽度是 64。现有模板只按 block 内线程索引与 ``__syncthreads`` 归约，
  不按 32 写死，因此不需要单独的内核分支；但任何"一个 warp 内前缀和"的
  假设都不能加进来（现有代码里没有）。
- 没有 cuDNN：``conv_impl="cudnn"`` 在构造期就报错，卷积走自研的
  im2col / tiled / implicit / grouped 四条路。
- 没有 nvRTC：内核由天数 Clang 以 ``-x ivcore --cubin`` 离线编译，见
  ``rtc.py``。

当前实现边界（都明确报错，不静默降级）：

- 运行期图捕获（cudaGraph 兼容通道）只走保留式 arena：流捕获期间不允许
  ``cudaMalloc``，所以 ``reuse_memory=False`` 直接拒绝；
- attention 不走共享模板的在线 softmax 内核，而是物化成 QK^T → softmax →
  PV 三段（前两段与末段分别走 cuBLAS 与自写内核），f32/f16 都支持。
  这是本包唯一覆盖 ``_op_attention`` 结构本身（而非仅内核源码）的地方，
  原因与收益见 ``REMIND.md`` 的「GEMM 物化 attention」。

准入判据不是"能跑通几个 demo"，而是 ``run_conformance`` 全绿——见 DESIGN 的 M4。
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from aetherflow.backends.base import BackendCapabilityError
from aetherflow.backends.contracts import DeviceInfo
from aetherflow.backends.device_backend import DeviceBackend, GPU_OPS
from aetherflow.backends.device_shared import _mask_shared_over_heads, _mask_target_shape
from . import codegen, runtime
from .blas import Cublas
from .memory import Arena

# 天数没有 cuDNN（公开 SDK 里不带 cudnn）。显式把 "cudnn" 去掉，让
# conv_impl="cudnn" 在构造期就报错，而不是等到第一次卷积才发现。
_CONV_IMPLS = ("auto", "im2col", "tiled", "implicit", "grouped")

# 能力面与 CUDA 一致：同一批模板编译出来的内核承接同一批算子。声明取自
# 中立的 ``GPU_OPS``，不依赖 cuda 后端（那个模块会加载 cupy）。
ILUVATAR_OPS = GPU_OPS


class IluvatarBackend(DeviceBackend):
    """智铠 MR 上的后端。构造失败时抛出带原因的错误，不静默退回 CPU。"""

    name = "iluvatar"
    # 报错信息里称呼自己的名字，不把使用者指向 CUDA。
    _LABEL = "天数（Iluvatar）后端"
    _OPS = ILUVATAR_OPS
    CONV_IMPLS = _CONV_IMPLS
    # 隐式 GEMM 在智铠上未经实测，先沿用基类默认（按通道规模选择），
    # 等 conformance 的卷积对比跑完再决定是否覆盖成 True。
    CONV_AUTO_IMPLICIT = False
    # 执行期分层分配器惰性创建；只在显式 ``memory_mode="tiered"`` 时才分配
    # 显存。写成类属性是为了让 ``__init__`` 仍然只设置容器相关字段。
    _tiered = None
    # pinned 暂存池同样惰性创建：第一次 replay 用到主机输入时才申请页锁定
    # 内存（没有捕获计划的纯逐节点执行完全用不到它）。
    _pinned = None

    def __init__(
        self,
        block: Optional[int] = None,
        max_blocks: int = 1024,
        conv_impl: str = "auto",
        graph_mode: str = "off",
        max_graph_plans: int = 4,
        min_graph_nodes: int = 2,
        chunked_scan: Optional[int] = None,
        memory_mode: str = "container",
        sparse_impl: str = "custom",
        device: int = 0,
    ) -> None:
        # 容器与方言必须在基类构造前就位：``DeviceBackend.__init__`` 要用它们
        # 建常数标量、注册内核并与能力声明对账。
        self.cp = runtime
        self.codegen = codegen
        # 先选卡再问 SM 数：``sm_count`` 是当前设备的属性，顺序反了就会拿
        # 上一块卡的数值去铺这一块卡的网格。非法索引在这里显式报错。
        runtime.set_device(device)
        # 驱动不给设备属性时退回 1，即只剩"不派空线程"那一条规则。
        self.sm_count = max(1, runtime.sm_count())
        super().__init__(
            block=block,
            max_blocks=max_blocks,
            conv_impl=conv_impl,
            graph_mode=graph_mode,
            max_graph_plans=max_graph_plans,
            min_graph_nodes=min_graph_nodes,
            chunked_scan=chunked_scan,
            memory_mode=memory_mode,
            sparse_impl=sparse_impl,
        )

    # ------------------------------------------------------------------
    # 执行期分层内存
    # ------------------------------------------------------------------

    def _memory_scope(self):
        """把 runtime 的分配器临时换成分层分配器。

        ``runtime.set_allocator`` 是进程级的，所以这里必须严格成对还原——
        捕获期用的是同一套机制（``capture`` 里 install/restore）。

        统计不按"这次执行"清零，而是累计到分配器上，再在退出时写进
        ``_last_arena``：``memory_stats()`` 于是能像捕获期一样被直接读，
        不用新增 API。``reset_stats`` 只在第一次安装时调用，跨多次执行的
        峰值才有意义。
        """
        if self.memory_mode != "tiered":
            return nullcontext()
        if self._tiered is None:
            from .memory import TieredAllocator

            self._tiered = TieredAllocator()
            self._tiered.reset_stats()
        previous = runtime.get_allocator()
        runtime.set_allocator(self._tiered)

        @contextmanager
        def scope():
            try:
                yield
            finally:
                runtime.set_allocator(previous)
                self._last_arena = self._tiered

        return scope()

    def pinned_array(self, key: str, arr: np.ndarray, dtype: np.dtype):
        """把主机数组拷进一块复用的 pinned 缓冲，返回 ``_Pinned`` 持有者。

        与 CUDA 侧同名同义：pageable 主机内存上的 ``cudaMemcpyAsync`` 在主机
        侧仍要等驱动搬进内部暂存区，捕获流省下来的下发时间会被这次等待吃
        掉。先拷进 pinned 内存（几微秒的 CPU 拷贝），H2D 才是真异步的。

        缓冲按 ``(key, 形状, dtype)`` 复用：训练里的输入形状固定，种类有限。
        调用方要保证上一轮拷贝已经同步过（``run`` / ``run_device`` 都会同步），
        否则会覆盖正在传输的内容。

        返回的是持有者而不是 numpy 视图：CUDA 的视图本身不拥有 pinned 内存，
        由这个对象负责 ``cudaHostFree``，调用方只从它取 ``ptr``。
        """
        shape = tuple(arr.shape)
        ident = (key, shape, np.dtype(dtype))
        if self._pinned is None:
            self._pinned = {}
        holder = self._pinned.get(ident)
        if holder is None:
            holder = runtime.pinned(shape, np.dtype(dtype))
            self._pinned[ident] = holder
        np.copyto(holder.view, arr, casting="unsafe")
        return holder

    def current_stream_ptr(self) -> int:
        """当前流的地址，供 cuBLAS 绑定。"""
        return runtime.current_stream()

    # ------------------------------------------------------------------
    # 契约
    # ------------------------------------------------------------------

    def device_info(self) -> DeviceInfo:
        """``DeviceContract``：名字、CUDA 运行时版本、显存总量。"""
        name, version, total = runtime.device_info()
        return DeviceInfo(name=name, runtime=version, total_memory_bytes=total)

    def enumerate_devices(self) -> Tuple[DeviceInfo, ...]:
        """``DeviceManagerContract``：按索引顺序列出全部设备。"""
        return tuple(
            DeviceInfo(name=name, runtime=version, total_memory_bytes=total)
            for name, version, total in runtime.enumerate_devices()
        )

    def current_device(self) -> int:
        return runtime.current_device()

    def set_device(self, index: int) -> None:
        """切换当前设备，并让自动 block 的 SM 数跟着换。

        换卡之后旧设备上的一切都不能再用：已加载模块绑在旧 context，
        cudaBLAS handle 绑在设备，驻留数组带 device id，捕获计划里的地址属于
        旧卡。不清掉就会"在新卡上执行旧计划"。驻留参数直接清空——它们本来
        就在旧卡显存里，跨设备保留只会把错误推迟到下一次 execute。
        """
        previous = runtime.current_device()
        if index != previous and self._tiered is not None:
            # 分层池的块属于旧卡：必须在旧 context 里释放，切走之后
            # cudaFree 会落到新卡的上下文上。
            self._tiered.close()
            self._tiered = None
        runtime.set_device(index)
        self.sm_count = max(1, runtime.sm_count())
        if index == previous:
            return
        self._modules.clear()
        self._functions.clear()
        self._kernel_handles.clear()
        self._blas = None
        self._resident.clear()
        self._invalidate_plans()
        # 计划里缓存的常量设备副本也属于旧卡（5.172）。
        self._clear_plan_values()

    # ------------------------------------------------------------------
    # 编译与 BLAS
    # ------------------------------------------------------------------

    def _kernel(self, key, source, func: str):
        # 先按 key 查已加载的函数句柄，命中就完全不同生成源码。``source`` 允许
        # 是"返回源码的函数"，而每次生成（L2 → C）要遍历一遍表达式树（MR-V100
        # 实测单节点约 30 µs，是纯 launcher 开销的三分之一）。key 由框架保证
        # 唯一标识内核身份——CUDA 后端也是同一约定（cuda/backend.py::_kernel）。
        cache_key = (key, func)
        fn = self._functions.get(cache_key)
        if fn is not None:
            return fn
        text = source() if callable(source) else source
        fn = runtime.function(key, text, func, func)
        self._functions[cache_key] = fn
        return fn

    def _cublas(self):
        """cuBLAS handle 惰性创建：图里没有 matmul 就不加载库。"""
        if self._blas is None:
            try:
                self._blas = Cublas()
            except OSError as exc:
                raise BackendCapabilityError(
                    f"天数后端无法加载 cuBLAS：{exc}"
                ) from exc
        return self._blas

    # ------------------------------------------------------------------
    # attention：GEMM 物化路径
    # ------------------------------------------------------------------

    # 分数矩阵允许占用的上限。超过就走基类的分块 online-softmax 内核：物化
    # 路径要一块 bh*S*T（softmax 就地归一化，不再额外要一份），2048² 以上会
    # 开始吃掉可观的显存，而分块内核的显存是 O(bh*T)。
    _ATTENTION_GEMM_MAX_BYTES = 256 << 20

    def _op_attention(self, node, args, attrs):
        """QK^T -> softmax -> PV 三段 GEMM，替代逐元素融合内核。

        融合内核的瓶颈不是算力而是指令发射：每个分数要 1 次 load + 1 次
        cvt + 1 次 FMA，标量路径在 MR-V100 上只有 ~0.5 TFLOPS。同一张卡的
        cuBLAS 走 tensor core（f16 实测 ~97 TFLOPS），代价是分数矩阵要落一次
        HBM。在 ``bh*S*T`` 不大时这笔往返远比发射便宜，所以这里把 attention
        拆成两个 batched GEMM 加一次 softmax pass。

        f32/f16 都走这条路；bf16 或超过显存阈值的形状退回基类实现。
        """
        cp = self.cp
        q = args[0]
        if q.ndim < 2 or q.dtype not in (cp.float16, cp.float32):
            return super()._op_attention(node, args, attrs)
        s_len = int(q.shape[-2])
        d = int(q.shape[-1])
        t_len = int(args[1].shape[-2])
        e = int(args[2].shape[-1])
        batch = tuple(int(x) for x in q.shape[:-2])
        bh = int(np.prod(batch)) if batch else 1
        itemsize = 2 if q.dtype == cp.float16 else 4
        if (
            s_len == 0 or d == 0 or t_len == 0 or e == 0
            or bh > 65535
            or bh * s_len * t_len * itemsize > self._ATTENTION_GEMM_MAX_BYTES
        ):
            return super()._op_attention(node, args, attrs)

        out = cp.empty(batch + (s_len, e), dtype=q.dtype)
        has_mask = 0
        mask_shared = 0
        mask = q
        if len(args) > 3:
            raw_mask = args[3]
            if raw_mask.dtype != q.dtype:
                raise BackendCapabilityError(
                    f"attention 的 mask 必须与 q 同 dtype，收到 {raw_mask.dtype} 与 "
                    f"{q.dtype}（节点 %{node.output.name}）"
                )
            original = tuple(raw_mask.shape)
            target = _mask_target_shape(original, len(batch) + 2)
            if target is None:
                raise BackendCapabilityError(
                    f"attention 的 mask 前导维必须是 1，得到 {original}"
                    f"（节点 %{node.output.name}）"
                )
            if len(target) != len(original):
                raw_mask = cp.ascontiguousarray(raw_mask).reshape(target)
            if _mask_shared_over_heads(raw_mask.shape, s_len, t_len):
                mask = cp.ascontiguousarray(raw_mask).reshape(s_len, t_len)
                mask_shared = 1
            else:
                try:
                    full = cp.broadcast_to(raw_mask, batch + (s_len, t_len))
                except ValueError as exc:
                    raise BackendCapabilityError(
                        f"attention 的 mask 形状 {tuple(args[3].shape)} 无法广播到 "
                        f"{batch + (s_len, t_len)}（节点 %{node.output.name}）"
                    ) from exc
                # 内核按行偏移寻址，展平后的 (bh*S, T) 与 (bh, S, T) 等价。
                mask = cp.ascontiguousarray(full).reshape(bh * s_len, t_len)
            has_mask = 1

        q3 = cp.ascontiguousarray(q).reshape(bh, s_len, d)
        k3 = cp.ascontiguousarray(args[1]).reshape(bh, t_len, d)
        v3 = cp.ascontiguousarray(args[2]).reshape(bh, t_len, e)
        scores = cp.empty((bh, s_len, t_len), dtype=q.dtype)

        blas = self._cublas()
        blas.set_stream(self.current_stream_ptr())
        if q.dtype == cp.float16:
            blas.hgemm_batched_nt(
                q3, k3, scores, bh, s_len * d, t_len * d, s_len * t_len
            )
        else:
            blas.sgemm_batched_nt(
                q3, k3, scores, bh, s_len * d, t_len * d, s_len * t_len
            )

        scale = np.float32(float(attrs.get("scale", 1.0)))
        p_drop = float(attrs.get("dropout", 0.0))
        drop_scale = np.float32(1.0 / (1.0 - p_drop)) if p_drop > 0.0 else np.float32(1.0)
        seed = np.uint64(int(attrs.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF)
        # 整行驻留寄存器：T 恰好是 64 的倍数且不超过 1024 时，每个 lane 持有
        # 固定的 32 位字数，softmax 只需一遍 HBM 读（实测 497 GB/s，逼近
        # 522 GB/s 的纯拷贝上限）。T = 2048 时 w/mw 两个数组各 32 个字会把
        # 寄存器挤到 local memory，实测反而比两遍版本慢，所以上限停在 16。
        np_reg = t_len // 64 if (q.dtype == cp.float16 and t_len % 64 == 0
                                 and 0 < t_len // 64 <= 16) else 0
        kernel = self._kernel(
            ("attention_scores", self._kind(q), np_reg),
            lambda: self.codegen.render(
                self.codegen.attention_scores_source(np_reg), q.dtype),
            "af_attention_scores",
        )
        # 一行一个 32 线程子组，不占 smem（行靠 L1/L2 二次命中）。
        rows = bh * s_len
        subwarps = 8
        inv = cp.empty((rows,), dtype=cp.float32)
        kernel(
            (-(-rows // subwarps),),
            (32 * subwarps,),
            (
                scores,
                mask,
                scores,
                inv,
                np.int64(rows),
                np.int64(s_len),
                np.int64(t_len),
                scale,
                np.float32(p_drop),
                drop_scale,
                seed,
                np.int32(has_mask),
                np.int32(mask_shared),
            ),
        )

        if q.dtype == cp.float16:
            blas.hgemm_batched(
                scores, v3, out.reshape(bh, s_len, e),
                bh, s_len * t_len, t_len * e, s_len * e,
            )
        else:
            blas.sgemm_batched(
                scores, v3, out.reshape(bh, s_len, e),
                bh, s_len * t_len, t_len * e, s_len * e,
            )

        # (p/l) @ V == (p @ V)/l 逐行成立，所以分母的倒数放到最后乘一次：
        # 作用在 bh*S*E 上，比在 bh*S*T 上做归一化少走一个数量级的显存。
        per = 32 * subwarps
        scale_fn = self._kernel(
            ("attention_rescale", self._kind(q)),
            lambda: self.codegen.render(self.codegen.attention_rescale_source(), q.dtype),
            "af_attention_rescale",
        )
        scale_fn(
            (min(4096, max(1, -(-rows // subwarps))),),
            (per,),
            (out, inv, np.int64(rows), np.int64(e)),
        )
        return out

    # ------------------------------------------------------------------
    # 运行期计划
    # ------------------------------------------------------------------

    def new_stream(self) -> Any:
        """一条非阻塞流：捕获必须发生在非默认流上，replay 也跑在它上面。"""
        return runtime.new_stream()

    def gpu_clock(self) -> "_CudaGpuClock":
        """逐节点 GPU 计时的时钟（见 ``runtime/telemetry.py``）。

        只在显式包装成 ``InstrumentedBackend`` 时用；普通执行路径不建事件。
        """
        return _CudaGpuClock(self)

    def _alias_view(self, owner) -> None:
        """捕获期登记一次视图引用（见 ``Arena.alias``）。"""
        arena = self._active_arena
        if arena is None:
            return
        holder = owner._buffer
        if holder is not None:
            arena.alias(holder.ptr)

    def _shares_storage(self, result, source) -> bool:
        """``result`` 是否与 ``source`` 共用同一段设备内存。

        ``DeviceArray`` 没有 ``base`` 链，视图身份由构造期的 ``_view`` 标记
        给出；同一个 ``_buffer`` 只是必要条件——源数组自己也指向它。
        """
        if result is source:
            return True
        return bool(getattr(result, "_view", False)) and result._buffer is source._buffer

    def dispose(self, value: Any) -> None:
        """把中间值的地址立刻还给 arena（只在捕获期有效）。

        视图不拥有内存：按**持有者的起始地址**扣一次视图引用，源数组被回收
        时只记成待回收（``Arena.free`` 见到计数就不还块）。非视图才真正还块。
        """
        arena = self._active_arena
        if arena is None or not isinstance(value, runtime.DeviceArray):
            return
        holder = value._buffer
        if holder is None:
            return
        if value._view:
            arena.unalias(holder.ptr)
            return
        if arena.pending(holder.ptr):
            # 整段切片 / 同形状 reshape 返回的就是源数组本身：先被回收的那条
            # Value 已把地址记成待回收，这一次扣掉自己那份登记。
            arena.unalias(holder.ptr)
            return
        arena.free(holder, value.nbytes)

    def allocator(self) -> Arena:
        """``MemoryContract``：新建一个捕获期分配器（``AllocatorContract``）。

        CUDA 的 ``Arena`` 与 CUDA 那份在 ``malloc`` / ``free`` / ``stats`` /
        ``reset_stats`` / ``close`` 上同形，因此同一份契约两边都能实现；
        CUDA 多出来的 ``recycle`` 是它自己的预热收尾，不进契约。
        """
        return Arena()

    def capture(
        self,
        graph,
        sample_inputs: Mapping[str, Any],
        reuse_memory: bool = True,
        shared_buffers: Optional[Mapping[str, Any]] = None,
        stream: Optional[Any] = None,
    ) -> "CudaGraphPlan":
        """把整张图捕获成一个 cudaGraph，返回可重复 replay 的计划。

        天数兼容层没有 CUDA 那样按分配器回调工作的默认内存池，而流捕获期间
        ``cudaMalloc`` 会被驱动拒绝。所以捕获**必须**走保留式 arena：
        ``reuse_memory=False`` 在这里没有可用的实现，明确拒绝而不是假装
        成功。其余规则与 ``IluvatarBackend.capture`` 一致——名字出现在
        ``sample_inputs`` 里的输入视为可变，只从驻留表取到的视为固定。

        ``shared_buffers`` 与 CUDA 侧同义：直接采用调用方给的显存数组当静态
        输入缓冲，省掉"另申请一块再拷一次"。计划不持有这些数组的所有权。
        """
        if not reuse_memory:
            raise BackendCapabilityError(
                "天数后端的捕获必须复用内存（reuse_memory=True）：流捕获期间"
                "不允许 cudaMalloc，没有 arena 就无处安放中间张量"
            )
        cp = self.cp
        declared = [v.name for v in graph.inputs]
        shared = dict(shared_buffers or {})
        unknown = sorted(set(shared) - set(declared))
        if unknown:
            raise KeyError(f"shared_buffers 里这些名字不是图输入: {unknown}")

        buffers: Dict[str, Any] = {}
        mutable: List[str] = []
        for name in declared:
            if name in sample_inputs:
                src = self.to_array(sample_inputs[name])
                mutable.append(name)
            elif name in self._resident:
                src = self._resident[name]
            else:
                raise KeyError(f"capture 缺少图输入: {name}")
            buf = shared.get(name)
            if buf is None:
                # 静态缓冲：图里记录的是固定地址，之后每次 run 往这里拷新值
                buf = cp.empty(src.shape, dtype=src.dtype)
                buf[...] = src
            elif tuple(buf.shape) != tuple(src.shape) or buf.dtype != src.dtype:
                raise ValueError(
                    f"共享缓冲 %{name} 的形状/dtype 是 {tuple(buf.shape)}/{buf.dtype}，"
                    f"与传入值的 {tuple(src.shape)}/{src.dtype} 不一致"
                )
            buffers[name] = buf

        arena = self.allocator()
        previous = runtime.get_allocator()
        runtime.set_allocator(arena.malloc)
        stream = stream if stream is not None else runtime.new_stream()
        self._active_arena = arena
        self._last_arena = arena
        host_refs: list = []
        try:
            # 预热：编译与 cudaBLAS 初始化都不能发生在捕获期间；两轮都按
            # "用完即回收"跑，否则捕获轮拿不到可复用的地址。
            with stream:
                self.run_device(graph, buffers, release=True)
            stream.synchronize()
            # 预热轮的输出已经没人引用，但它们仍记在 arena 的账上；不收回，
            # 捕获轮第一次申请就会撞上"没有空闲块"，而捕获期不允许 cudaMalloc。
            arena.recycle()
            arena.reset_stats()
            # 常量副本在预热轮上传进了 arena，而 recycle 把预热期的地址整体
            # 收回了；复用那份副本会让捕获图读到后来写进去的中间值（实测
            # loss 从 4.37 变成 10^2 量级）。重建计划让常量在捕获轮重新上传。
            self._drop_plan(graph)
            with stream:
                with runtime.capture(stream) as session:
                    # 捕获期不允许 D2H：越界标志只登记，replay 之后由计划复查
                    guards: List[Tuple[Any, str, str]] = []
                    self._capture_guards = guards
                    self._capturing = True
                    # 图输出必须独占一块（见 ``_shared_view``）：捕获图把
                    # 地址写死，输出若与源共用一段，源被回收后地址会分给
                    # 后面的算子，两个输出就写同一块了。
                    self._capture_outputs = frozenset(v.id for v in graph.outputs)
                    try:
                        env = self.run_device(graph, buffers, release=True)
                    finally:
                        self._capturing = False
                        self._capture_outputs = frozenset()
                        self._capture_guards = None
            graph_exec = runtime.graph_exec(session.graph)
            # 捕获期 H2D 的主机源（常量、`full` 的临时 numpy）：图里只记
            # 指针，必须由计划持有到销毁，否则 replay 会读到已释放的内存。
            host_refs = session.host_refs
        finally:
            self._active_arena = None
            runtime.set_allocator(previous)

        outputs = {
            name: env[v.id] for v, name in zip(graph.outputs, graph.output_names)
        }
        return CudaGraphPlan(
            self, graph_exec, stream, buffers, outputs, mutable, arena, guards,
            host_refs,
        )

class CudaGraphPlan:
    """一次捕获、多次 replay 的运行期计划（cudaGraph 版）。

    ``arena`` 必须被计划持有：图里写死的是捕获时那些显存的地址，arena
    一旦被回收，replay 就会写到已经属于别人的内存上。
    """

    def __init__(
        self,
        backend: IluvatarBackend,
        graph,
        stream,
        buffers: Dict[str, Any],
        outputs: Dict[str, Any],
        mutable: List[str],
        arena: Arena,
        guards: Optional[List[Tuple[Any, str, str]]] = None,
        host_refs: Optional[list] = None,
    ) -> None:
        self.backend = backend
        self._cg = graph
        self._stream = stream
        self._buffers = buffers
        self._outputs = outputs
        self._mutable = tuple(mutable)
        self._arena = arena
        # 捕获期登记的越界标志，与 CUDA 侧同一个含义与复查时机。
        self._guards = tuple(guards or ())
        # 捕获期 H2D 的主机源：图里写死的是它们的地址，计划必须持有引用
        self._host_refs = tuple(host_refs or ())

    @property
    def buffers(self) -> Dict[str, Any]:
        """图里写死的输入缓冲（名字 → 数组对象），与 CUDA 侧同义。"""
        return dict(self._buffers)

    @property
    def input_shapes(self) -> Dict[str, tuple]:
        return {name: tuple(buf.shape) for name, buf in self._buffers.items()}

    @property
    def mutable_inputs(self) -> tuple:
        return self._mutable

    @property
    def memory_bytes(self) -> int:
        """这张图实际占用的显存（中间缓冲 + 输出）。"""
        return self._arena.total_bytes

    @property
    def peak_bytes(self) -> int:
        """捕获期同时存活的峰值字节，即这张图绕不过去的显存下界。"""
        return self._arena.peak_live_bytes

    def run(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """更新可变输入后 replay，返回主机侧输出。"""
        self.replay(inputs)
        # 输出由捕获流上的图写入，而 to_host 走默认流；不显式同步会读到上一轮
        self._stream.synchronize()
        self._check_guards()
        return {name: self.backend.to_host(v) for name, v in self._outputs.items()}

    def run_device(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """replay 后同步，返回**设备上**的输出，不做 D2H。"""
        self.replay(inputs)
        self._stream.synchronize()
        self._check_guards()
        return dict(self._outputs)

    def _check_guards(self) -> None:
        """复查捕获期登记的越界标志，与 CUDA 侧同一套语义。

        读回只能在 ``synchronize()`` 之后；报错前先清零，让同一计划上的
        下一次合法 replay 不误报。
        """
        for err, op, allowed in self._guards:
            if int(err.item()) == 0:
                continue
            with self._stream:
                err[...] = runtime.zeros(err.shape, dtype=err.dtype)
            self.backend._raise_bad_ids(op, allowed)

    def replay(self, inputs: Mapping[str, Any]) -> None:
        """只下拷输入并 replay，不等 GPU 结束（输出留在设备上）。

        捕获流是非阻塞流，与默认流（优化器原地更新权重的地方）之间没有
        隐式同步。进流之前记一个事件、让捕获流等它，保证 replay 读到的是
        主机侧刚刚更新过的权重（与 CUDA 侧同一个时序）。

        主机输入先拷进 pinned 暂存再异步 H2D：pageable 源的 ``cudaMemcpyAsync``
        在主机侧仍会阻塞，捕获流省下的下发时间会被抵消。设备输入走异步
        D2D，顺序由捕获流保证，不等它完成。
        """
        stream = self._stream.ptr
        previous = runtime.current_stream()
        if previous != stream:
            event = runtime.new_event()
            event.record(previous)
            runtime.stream_wait_event(stream, event)
            # 等待已经入队，事件即可释放：驱动会在等待完成后回收资源
            event.destroy()
        for name in self._mutable:
            if name not in inputs:
                raise KeyError(f"缺少图输入: {name}")
            buf = self._buffers[name]
            value = inputs[name]
            if tuple(np.shape(value)) != tuple(buf.shape):
                raise ValueError(
                    f"输入 %{name} 形状 {tuple(np.shape(value))} 与捕获时的 "
                    f"{tuple(buf.shape)} 不一致：一张 cudaGraph 只对应一组固定形状"
                )
            if value is buf:
                # 共享缓冲：调用方已经把它写在这块地址上了，不必自拷贝
                continue
            if isinstance(value, np.ndarray):
                pinned = self.backend.pinned_array(name, value, buf.dtype)
                runtime.copy_async(
                    buf.ptr, pinned.ptr, buf.nbytes,
                    runtime.HOST_TO_DEVICE, stream,
                )
                continue
            arr = self.backend.to_array(value)
            if arr.dtype != buf.dtype:
                arr = arr.astype(buf.dtype, copy=False)
            runtime.copy_async(
                buf.ptr, arr.ptr, buf.nbytes, runtime.DEVICE_TO_DEVICE, stream
            )
        self._cg.launch(stream)

    def device_outputs(self) -> Dict[str, Any]:
        """当前设备上的输出（下一次 replay 会被覆盖）。"""
        return dict(self._outputs)


class _CudaGpuClock:
    """逐节点 GPU 计时：起点/终点各记一个带计时标志的事件。

    ``elapsed`` 只在两个事件都完成之后才成立，所以调用方必须在读取前统一
    同步一次。逐节点插同步是另一回事：那样测到的是同步开销本身。
    """

    def __init__(self, backend: "IluvatarBackend") -> None:
        self._backend = backend

    def _mark(self):
        event = runtime.new_timing_event()
        event.record(self._backend.current_stream_ptr())
        return event

    def begin(self):
        return self._mark()

    def end(self):
        return self._mark()

    def synchronize(self) -> None:
        runtime.synchronize()

    def elapsed(self, start, stop) -> float:
        return stop.elapsed_ms(start)

    def release(self, start, stop) -> None:
        start.destroy()
        stop.destroy()
