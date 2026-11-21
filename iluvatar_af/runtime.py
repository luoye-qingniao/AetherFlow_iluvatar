"""天数（CoreX / 智铠 MR）运行时与设备数组（ctypes）。

天数的软件栈是一层**编译期 + ABI 级**的 CUDA 兼容层：``libcudart`` /
``libcuda`` / ``libcublas`` 都提供 CUDA 10.2 的符号，内核由天数 Clang 从
同一份 ``__global__`` 源码重新编译成 ivcore 目标（不是运行期指令翻译）。
因此这里直接按 CUDA 运行时 API 绑定，不需要 cupy、torch 或任何厂商私有库。

模块加载与内核启动走 driver API（``cuModuleLoadData`` / ``cuLaunchKernel``）：
CUDA 运行时没有 ``cudaModuleLoadData`` 这一组函数，而流句柄在 runtime 与
driver 之间是同一个底层对象，可以混用。

与 cupy 的差异只有三处，都是刻意的：

- 只支持四种 dtype：f32 / f16 / i32 / i8。它们是 IR 与内核实际用到的全部。
- 切片只支持 ``slice``（含 ``None`` 之外的省略号），不支持整数索引降维。
  后端代码只做"按轴切一段"，整数索引从不出现；遇到就明确报错。
- 没有内存池。每次 ``empty`` 直接向驱动要，``DeviceArray`` 析构时归还。
  正确性优先；峰值统计交给编译期的 ``ir/liveness.py``，不在运行期猜。

智铠的 warp 宽度是 64（NVIDIA 为 32），但本文件与内核模板都不按 warp 宽度
写死线程映射，只有 block 内 ``__syncthreads`` 归约，因此不需要分支。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib

import numpy as np

from . import rtc

_LIB_CANDIDATES = (
    "libcudart.so.10.2",
    "libcudart.so.10",
    "libcudart.so",
    "cudart.dll",
)

# driver API：模块加载与内核启动只能走这一层
_DRIVER_CANDIDATES = (
    "libcuda.so.1",
    "libcuda.so",
    "nvcuda.dll",
)

# cudaMemcpyKind
_H2D = 1
_D2H = 2
_D2D = 3

# 对外用可读名字：后端与测试都不该记数字
HOST_TO_DEVICE = _H2D
DEVICE_TO_HOST = _D2H
DEVICE_TO_DEVICE = _D2D

# 视图与 strided copy 的最大维数。超过时先折叠相邻维；仍超过则明确拒绝。
_MAX_DIMS = 4


class CudaRuntimeError(RuntimeError):
    """天数 CUDA 兼容层调用失败。"""


def _open(names, what: str) -> ctypes.CDLL:
    tried = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    raise OSError(f"找不到{what}动态库，尝试过：\n  " + "\n  ".join(tried))


def _load_library() -> ctypes.CDLL:
    names = []
    found = ctypes.util.find_library("cudart")
    if found:
        names.append(found)
    names.extend(_LIB_CANDIDATES)
    return _open(names, "天数 CUDA 运行时")


def _load_driver() -> ctypes.CDLL:
    return _open(list(_DRIVER_CANDIDATES), "天数 CUDA 驱动")


def available() -> bool:
    """进程里能否**真正用上**天数加速卡：库可加载，且至少有一个设备可枚举。

    只探测 ``CDLL`` 会给出假阳性：库在但驱动没装、或卡被别的租户占满时，
    ``cudaGetDeviceCount`` 会返回 0 个设备，后面第一次 ``cudaMalloc`` 才失败，
    而此时注册表已经把这个后端当成"可用"选走了。设备枚举失败必须落到
    "不可用"，让注册表去试下一个候选。
    """
    try:
        lib = _load_library()
    except OSError:
        return False
    count = ctypes.c_int(0)
    try:
        rc = int(lib.cudaGetDeviceCount(ctypes.byref(count)))
    except (AttributeError, OSError, ValueError):
        return False
    return rc == 0 and count.value >= 1


def device_info():
    """``(设备名, CUDA 运行时版本, 显存总量)``。"""
    return _Runtime.instance().device_info()


def sm_count() -> int:
    """当前设备的计算单元数；驱动查不到时返回 0。"""
    return _Runtime.instance().sm_count


def device_count() -> int:
    """进程可见的设备数。"""
    return _Runtime.instance().device_count


def current_device() -> int:
    """当前上下文所在的设备索引。"""
    return _Runtime.instance().current_device


def set_device(index: int) -> None:
    """切换当前设备；越界或类型不对时抛 ``CudaRuntimeError``。"""
    _Runtime.instance().set_device(index)


def enumerate_devices():
    """按索引顺序返回 ``(设备名, CUDA 运行时版本, 显存总量)``。"""
    count = device_count()
    return tuple(_Runtime.instance().device_info(i) for i in range(count))


def function(key, source: str, name: str, func: str):
    """编译并缓存一个内核，返回可直接调用的 ``CudaFunction``。"""
    return _Runtime.instance().function(key, source, name, func)


# ----------------------------------------------------------------------
# 捕获相关的薄封装：与 ``function`` / ``device_info`` 同一层，
# 后端只认这一面接口，不直接碰 ``_Runtime``。
# ----------------------------------------------------------------------


def new_stream() -> "_Stream":
    """创建一个非阻塞流；捕获必须发生在一个非默认流上。"""
    return _Stream(_Runtime.instance().create_stream())


def set_allocator(allocator) -> None:
    """把 ``empty`` 的底层分配器换成 ``allocator(nbytes) -> 持有者``。"""
    _Runtime.instance().set_allocator(allocator)


def get_allocator():
    return _Runtime.instance().get_allocator()


def capture(stream: "_Stream") -> "_CaptureSession":
    return _Runtime.instance().capture(stream.ptr)


def graph_exec(graph: int) -> "_GraphExec":
    return _Runtime.instance().graph_exec(graph)


def pinned(shape, dtype=np.float32) -> "_Pinned":
    """分配一块 pinned 主机缓冲，``.view`` 是同形同 dtype 的 numpy 视图。"""
    dt = np.dtype(dtype)
    if dt not in _SUPPORTED:
        raise CudaRuntimeError(
            f"pinned 缓冲只支持容器能承载的 dtype，收到 {dt}"
        )
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(d) for d in shape)
    holder = _Runtime.instance().alloc_pinned(_prod(shape) * dt.itemsize)
    count = _prod(shape)
    view = holder.view[: count * dt.itemsize].view(dt).reshape(shape)
    holder.view = view
    return holder


def copy_async(dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
    """把一次拷贝排进 ``stream``，不等它完成（见 ``_Runtime.memcpy_async``）。"""
    _Runtime.instance().memcpy_async(dst, src, nbytes, kind, stream)


def new_event() -> "_Event":
    """建一个禁用计时的事件；用完必须 ``destroy()``。"""
    return _Runtime.instance().new_event()


def new_timing_event() -> "_Event":
    """建一个带计时标志的事件，供逐节点 GPU 计时；用完必须 ``destroy()``。"""
    return _Runtime.instance().new_event(timing=True)


def stream_wait_event(stream: int, event: "_Event") -> None:
    """让 ``stream`` 等 ``event`` 记录的那一点完成。"""
    _Runtime.instance().stream_wait_event(stream, event)


class _Pointer:
    """``array.data`` 的返回值：只暴露 ``ptr``，与 cupy 的用法对齐。"""

    __slots__ = ("ptr",)

    def __init__(self, ptr: int) -> None:
        self.ptr = ptr


# 当前流栈：``_Stream.__enter__`` 压入、``__exit__`` 弹出，内核 launch 与
# 拷贝都从这里取流。放在模块级而不是 ``_Runtime`` 上，是因为当前流必须在
# 没有 ROCm 的机器上也能回答"默认流"，否则盯容器结构的离线测试会因加载
# 驱动失败而跑不起来。
_STREAM_STACK: list = []


def current_stream() -> int:
    """当前流的地址（没有设备时是默认流 0）。

    ``CudaBackend.current_stream_ptr`` 用它给 cudaBLAS 绑定流。以前这里是
    ``cuda.get_current_stream().ptr`` 的假命名空间，为的是让 CUDA 后端的
    ``cp.cuda`` 调用在 CUDA 上通过；拆出 ``DeviceBackend`` 之后不再需要。
    """
    return _STREAM_STACK[-1] if _STREAM_STACK else 0


def synchronize() -> None:
    """等设备把已提交的工作干完。

    cupy 后端用 ``cp.cuda.get_current_stream().synchronize()``；CUDA 的替身
    模块没有那个假命名空间（见 ``current_stream`` 的说明），所以基准工具
    的 ``_sync`` 对这里退回到设备级同步。
    """
    _Runtime.instance().synchronize()


class _Flags:
    __slots__ = ("c_contiguous",)

    def __init__(self, contiguous: bool) -> None:
        self.c_contiguous = contiguous


class _Buffer:
    """一块设备内存的真正持有者。视图只引用它，不各自释放。"""

    __slots__ = ("ptr", "nbytes")

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = ptr
        self.nbytes = nbytes

    def __del__(self) -> None:
        try:
            _Runtime.instance().free(self.ptr)
        except Exception:  # 解释器退出时驱动可能已经不在了
            pass


class _Pinned:
    """一块页锁定（pinned）主机内存与它的 numpy 视图。

    H2D 用 pageable 主机内存时，驱动要先把数据搬进内部暂存区，``cudaMemcpyAsync``
    在主机侧仍会等这次搬运；只有 pinned 源才是真异步。视图本身不持有内存，
    生命周期由这个对象负责——调用方（后端的暂存池）必须持有它。
    """

    __slots__ = ("ptr", "nbytes", "view")

    def __init__(self, ptr: int, nbytes: int, view: np.ndarray) -> None:
        self.ptr = ptr
        self.nbytes = nbytes
        self.view = view

    def __del__(self) -> None:
        try:
            _Runtime.instance().free_pinned(self.ptr)
        except Exception:  # 解释器退出时驱动可能已经不在了
            pass


def _dtype(dtype) -> np.dtype:
    dt = np.dtype(dtype)
    if dt not in _SUPPORTED:
        raise CudaRuntimeError(
            f"CUDA 容器不支持 {dt}。"
            "其它 dtype 应在进入后端前被 IR 拒绝"
        )
    return dt


def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _contiguous_strides(shape) -> tuple:
    strides = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= int(shape[i])
    return tuple(strides)


def _is_contiguous(shape, strides, itemsize: int) -> bool:
    acc = 1
    for d, s in zip(reversed(shape), reversed(strides)):
        if int(d) != 1 and int(s) != acc:
            return False
        acc *= int(d)
    return True


class DeviceArray:
    """设备上的定长数组：形状 + 步长 + dtype + 指针。

    视图（切片 / 转置 / 广播 / reshape）共享同一块 ``_Buffer``，因此生命
    周期只由最初的那次分配决定——视图对象被回收不会释放内存，也不会悬空。

    ``_view`` 标记视图身份：执行器在捕获期按"最后一次被读"回收源数组，
    视图这时可能还活着，所以 ``CudaBackend.dispose`` 必须区分"还块"与
    "扣掉一次视图引用"。``_buffer`` 相同不足以判定——源数组自己也是指向
    同一个持有者，只有构造时才知道哪个是视图。
    """

    __slots__ = ("_buffer", "ptr", "shape", "strides", "dtype", "_view")

    def __init__(
        self, buffer: _Buffer | None, ptr: int, shape, strides, dtype, view: bool = False
    ) -> None:
        self._buffer = buffer
        self.ptr = int(ptr)
        self.shape = tuple(int(d) for d in shape)
        self.strides = tuple(int(s) for s in strides)
        self.dtype = np.dtype(dtype)
        self._view = bool(view)

    # ------------------------------------------------------------------
    # 形状与容量
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return _prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def flags(self) -> _Flags:
        return _Flags(
            _is_contiguous(self.shape, self.strides, self.dtype.itemsize)
        )

    @property
    def data(self) -> _Pointer:
        return _Pointer(self.ptr)

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("0 维数组没有 len()")
        return self.shape[0]

    def __repr__(self) -> str:
        return f"iluvatar_af.array({self.shape}, {self.dtype}, 0x{self.ptr:x})"

    # ------------------------------------------------------------------
    # 视图
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """只接受 ``slice``（可带省略号），返回视图，不做拷贝。"""
        if not isinstance(key, tuple):
            key = (key,)
        # 展开省略号：把剩余的轴补成整段切片
        expanded = []
        ellipsis_seen = False
        for item in key:
            if item is Ellipsis:
                if ellipsis_seen:
                    raise IndexError("最多只能有一个省略号")
                ellipsis_seen = True
                expanded.extend([slice(None)] * (self.ndim - (len(key) - 1)))
            else:
                expanded.append(item)
        while len(expanded) < self.ndim:
            expanded.append(slice(None))
        if len(expanded) != self.ndim:
            raise IndexError(
                f"索引维数不匹配：{self.ndim} 维数组收到 {len(key)} 项"
            )

        shape = []
        strides = []
        offset = 0
        for dim, (size, step, item) in enumerate(
            zip(self.shape, self.strides, expanded)
        ):
            if isinstance(item, slice):
                start, stop, stride = item.indices(size)
                length = len(range(start, stop, stride))
                shape.append(length)
                strides.append(step * stride)
                offset += start * step
            else:
                raise TypeError(
                    "CUDA 容器只支持切片（不降维）；"
                    f"第 {dim} 轴收到 {type(item).__name__}。"
                    "整数索引会改变秩，后端不依赖该能力"
                )
        return DeviceArray(
            self._buffer, self.ptr + offset * self.dtype.itemsize,
            tuple(shape), tuple(strides), self.dtype, view=True,
        )

    def __setitem__(self, key, value) -> None:
        """整块赋值（``buf[...] = src``）或标量填充（``buf[...] = 0``）。

        标量路径是为优化器与训练循环准备的：``grad[...] = 0``、
        ``seed[...] = 1.0`` 都是"把一块缓冲刷成同一个值"，没有对应的主机
        数组可拷。数组路径要求同形同 dtype；源连续、目标可带步长（分组卷积
        的 ``out[:, a:b] = src`` 就是这种形态），源非连续时先物化一份。
        隐式广播写入仍不支持：训练路径上不存在，放开只会让形状错误更难发现。
        """
        target = self[key] if key is not Ellipsis else self
        if isinstance(value, DeviceArray):
            if target.shape != value.shape or target.dtype != value.dtype:
                raise CudaRuntimeError(
                    f"整块赋值要求同形同 dtype：{target.shape}/{target.dtype} "
                    f"收到 {value.shape}/{value.dtype}"
                )
            if not value.flags.c_contiguous:
                # 源非连续时先物化一份；写回反正也是一次拷贝，不值得为
                # "两侧都带步长"再开一条内核。
                value = value.copy()
            if target.flags.c_contiguous:
                _Runtime.instance().memcpy(
                    target.ptr, value.ptr, target.nbytes, _D2D
                )
            else:
                # 目标不连续：分组卷积按组写回 ``out[:, a:b]`` 走的就是这里。
                # 5.165 之前这里直接报错，CUDA 上只能整块重建输出。
                _strided_copy_pair(value, target)
            return
        host = np.asarray(value)
        if host.ndim != 0:
            raise CudaRuntimeError(
                f"CUDA 容器的赋值只接受同形设备数组或标量，收到形状 {host.shape}"
            )
        _fill(target, host.item())

    def copy(self) -> "DeviceArray":
        """一份连续的新数组（与 numpy / cupy 的 ``copy()`` 同义）。"""
        out = empty(self.shape, self.dtype)
        if out.size:
            _strided_copy(self, out)
        return out

    # ------------------------------------------------------------------
    # 逐元素运算（优化器与训练辅助工具所需的受控一面）
    # ------------------------------------------------------------------

    def __neg__(self) -> "DeviceArray":
        return _ew_unary(self, _NEG)

    def __add__(self, other):
        return _ew_binary(self, other, _ADD)

    def __radd__(self, other):
        return _ew_binary(self, other, _ADD, reverse=True)

    def __sub__(self, other):
        return _ew_binary(self, other, _SUB)

    def __rsub__(self, other):
        return _ew_binary(self, other, _SUB, reverse=True)

    def __mul__(self, other):
        return _ew_binary(self, other, _MUL)

    def __rmul__(self, other):
        return _ew_binary(self, other, _MUL, reverse=True)

    def __truediv__(self, other):
        return _ew_binary(self, other, _DIV)

    def __rtruediv__(self, other):
        return _ew_binary(self, other, _DIV, reverse=True)

    def __iadd__(self, other):
        _ew_binary_into(self, other, _ADD)
        return self

    def __isub__(self, other):
        _ew_binary_into(self, other, _SUB)
        return self

    def __imul__(self, other):
        _ew_binary_into(self, other, _MUL)
        return self

    def __itruediv__(self, other):
        _ew_binary_into(self, other, _DIV)
        return self

    def __pow__(self, exponent):
        # 训练层只用 ``g ** 2``（全局范数）；其余指数没有对应的内核，
        # 与其悄悄用整数幂近似，不如明确拒绝。
        if exponent == 2:
            return _ew_binary(self, self, _MUL)
        raise CudaRuntimeError(
            f"CUDA 容器只支持 2 次幂（全局范数用），收到指数 {exponent!r}"
        )

    def reshape(self, *shape):
        """连续内存上的视图重排（``a.reshape(1)`` / ``a.reshape(a, b, c)``）。"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(int(d) for d in shape)
        if -1 in shape:
            known = _prod([d for d in shape if d != -1])
            shape = tuple(
                (self.size // known if known else 0) if d == -1 else d for d in shape
            )
        if _prod(shape) != self.size:
            raise CudaRuntimeError(
                f"reshape 的元素数不一致：{self.shape} -> {shape}"
            )
        if not self.flags.c_contiguous:
            raise CudaRuntimeError(
                f"reshape 要求连续内存；{self.shape} 带步长 {self.strides}。"
                "请先调用 ascontiguousarray（显式物化，不静默拷贝）"
            )
        return DeviceArray(
            self._buffer, self.ptr, shape, _contiguous_strides(shape), self.dtype,
            view=True,
        )

    def astype(self, dtype, copy: bool = True):
        target = _dtype(dtype)
        if target == self.dtype and not copy:
            return self
        return _cast(self, target)

    def item(self):
        """读回单元素（一次 D2H 同步），供越界标志这类标量检查使用。"""
        if self.size != 1:
            raise CudaRuntimeError(f"item() 要求单元素，{self.shape} 不满足")
        host = np.empty(self.shape, dtype=self.dtype)
        _Runtime.instance().memcpy(
            host.ctypes.data, self.ptr, host.nbytes, _D2H
        )
        return host.reshape(-1)[0].item()


# 容器能**承载**的 dtype 比内核能**计算**的 dtype 宽：f64 与 bool 必须能被
# 上传，才能让"不支持"的判定发生在 ``invoke`` 的能力检查里（与 CUDA 侧一致，
# 报 BackendCapabilityError），而不是在拷贝阶段抛一个容器错误。复数同理：
# CUDA 侧没有 rocFFT 绑定，``fft`` / ``rfft`` 会在 ``invoke`` 以"未实现算子"
# 被拒；若容器先一步拒收 complex，拒绝就会变成一个裸的运行时错误。
_SUPPORTED = {
    np.dtype("float32"), np.dtype("float16"), np.dtype("float64"),
    np.dtype("complex64"), np.dtype("complex128"),
    np.dtype("int32"), np.dtype("int8"), np.dtype("bool"),
    # uint16 不是 IR 的类型，而是 bf16 的设备容器（两字节位模式）。基类在
    # ``_kind`` / ``quantize`` 里按 ``cp.uint16`` 判定 bf16，容器缺这个常量
    # 会让任何图在第一次 ``invoke`` 就 AttributeError——CUDA 的 bf16 支持面
    # 是空集，但"不支持"必须由 ``invoke`` 的能力检查给出，而不是崩在属性上。
    np.dtype("uint16"),
}

# 容器层对外暴露的 dtype 常量，与 ``cupy.float32`` 等价的用法
float32 = np.float32
float16 = np.float16
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128
int32 = np.int32
int8 = np.int8
bool_ = np.bool_
# bf16 位模式容器（见 _SUPPORTED 的说明）。
uint16 = np.uint16
ndarray = DeviceArray


class _Runtime:
    """设备、内存与内核句柄的进程级单例。"""

    _INSTANCE: "_Runtime | None" = None

    @classmethod
    def instance(cls) -> "_Runtime":
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def __init__(self) -> None:
        lib = _load_library()
        drv = _load_driver()
        vp = ctypes.c_void_p
        sz = ctypes.c_size_t
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDeviceCount.restype = ctypes.c_int
        lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaGetDevice.restype = ctypes.c_int
        lib.cudaSetDevice.argtypes = [ctypes.c_int]
        lib.cudaSetDevice.restype = ctypes.c_int
        lib.cudaGetDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.cudaGetDeviceProperties.restype = ctypes.c_int
        # 只用来问一个整数属性（SM 数）。整块 cudaDeviceProp_t 的布局随版本变，
        # 按结构体偏移读会静默读错字段；属性枚举则是稳定的 ABI。
        lib.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
        ]
        lib.cudaDeviceGetAttribute.restype = ctypes.c_int
        lib.cudaMalloc.argtypes = [ctypes.POINTER(vp), sz]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaFree.argtypes = [vp]
        lib.cudaFree.restype = ctypes.c_int
        lib.cudaMemcpy.argtypes = [vp, vp, sz, ctypes.c_int]
        lib.cudaMemcpy.restype = ctypes.c_int
        lib.cudaMemset.argtypes = [vp, ctypes.c_int, sz]
        lib.cudaMemset.restype = ctypes.c_int
        lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(sz), ctypes.POINTER(sz)]
        lib.cudaMemGetInfo.restype = ctypes.c_int
        lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaRuntimeGetVersion.restype = ctypes.c_int
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int
        lib.cudaStreamSynchronize.argtypes = [vp]
        lib.cudaStreamSynchronize.restype = ctypes.c_int
        # 模块加载与启动是 driver API：CUDA 运行时没有 cudaModule* 这组函数。
        # 天数 libcuda.so.1 提供 CUDA 10.2 的 cu* 符号，签名与运行时一一对应。
        drv.cuModuleLoadData.argtypes = [ctypes.POINTER(vp), ctypes.c_void_p]
        drv.cuModuleLoadData.restype = ctypes.c_int
        drv.cuModuleGetFunction.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_char_p]
        drv.cuModuleGetFunction.restype = ctypes.c_int
        drv.cuLaunchKernel.argtypes = [
            vp,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, vp, ctypes.POINTER(vp), ctypes.POINTER(vp),
        ]
        drv.cuLaunchKernel.restype = ctypes.c_int

        lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(vp), ctypes.c_uint]
        lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        lib.cudaStreamDestroy.argtypes = [vp]
        lib.cudaStreamDestroy.restype = ctypes.c_int
        lib.cudaMemcpyAsync.argtypes = [vp, vp, sz, ctypes.c_int, vp]
        lib.cudaMemcpyAsync.restype = ctypes.c_int
        lib.cudaMemsetAsync.argtypes = [vp, ctypes.c_int, sz, vp]
        lib.cudaMemsetAsync.restype = ctypes.c_int
        # 页锁定主机内存：replay 的 H2D 要靠它变成真异步（见 _Pinned）
        lib.cudaHostAlloc.argtypes = [ctypes.POINTER(vp), sz, ctypes.c_uint]
        lib.cudaHostAlloc.restype = ctypes.c_int
        lib.cudaFreeHost.argtypes = [vp]
        lib.cudaFreeHost.restype = ctypes.c_int
        # 事件：捕获流与默认流之间没有隐式顺序，进流前必须显式建依赖
        lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(vp), ctypes.c_uint]
        lib.cudaEventCreateWithFlags.restype = ctypes.c_int
        lib.cudaEventRecord.argtypes = [vp, vp]
        lib.cudaEventRecord.restype = ctypes.c_int
        lib.cudaStreamWaitEvent.argtypes = [vp, vp, ctypes.c_uint]
        lib.cudaStreamWaitEvent.restype = ctypes.c_int
        lib.cudaEventDestroy.argtypes = [vp]
        lib.cudaEventDestroy.restype = ctypes.c_int
        # 逐节点 GPU 计时：事件必须带计时标志，elapsed 只在两者都完成后才有效
        lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float), vp, vp,
        ]
        lib.cudaEventElapsedTime.restype = ctypes.c_int
        lib.cudaStreamBeginCapture.argtypes = [vp, ctypes.c_int]
        lib.cudaStreamBeginCapture.restype = ctypes.c_int
        lib.cudaStreamEndCapture.argtypes = [vp, ctypes.POINTER(vp)]
        lib.cudaStreamEndCapture.restype = ctypes.c_int
        lib.cudaGraphInstantiate.argtypes = [
            ctypes.POINTER(vp), vp, ctypes.POINTER(vp), ctypes.c_char_p, sz,
        ]
        lib.cudaGraphInstantiate.restype = ctypes.c_int
        lib.cudaGraphLaunch.argtypes = [vp, vp]
        lib.cudaGraphLaunch.restype = ctypes.c_int
        lib.cudaGraphExecDestroy.argtypes = [vp]
        lib.cudaGraphExecDestroy.restype = ctypes.c_int
        lib.cudaGraphDestroy.argtypes = [vp]
        lib.cudaGraphDestroy.restype = ctypes.c_int
        self._lib = lib
        self._drv = drv
        self._rc = rtc.Rtc()

        count = ctypes.c_int()
        if lib.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value < 1:
            raise CudaRuntimeError("天数驱动没有报告任何设备")
        self._device_count = count.value
        self._props: dict = {}  # 设备索引 -> 已解析的属性
        self._arch = None
        self._modules: dict = {}  # code object 的 sha1 -> 已加载模块句柄
        self._functions: dict = {}  # (模块句柄, 函数名) -> CudaFunction
        # 捕获期标志：拷贝要走异步版本，cudaMalloc 直接拒绝
        self._capturing = False
        # 捕获期被回收的指针先记账，等 EndCapture 之后再真正释放。GC 是任意
        # 时刻发生的，图外对象恰好在捕获窗口里析构时，cudaFree / cudaHostFree
        # 会让当前捕获失效（cudaErrorStreamCaptureInvalidated，901）。
        self._deferred_free: list = []
        self._deferred_pinned: list = []
        # 捕获期的 H2D 主机源暂存：图里只记指针，源数组必须活到计划销毁
        # （见 memcpy 与 _CaptureSession.host_refs）
        self._capture_host = None
        # 底层分配器：None 表示每次直接向驱动要；捕获期换成 arena
        self._allocator = None
        # SM 数按设备缓存；0 表示驱动没给出（调用方退回不按 SM 数铺块）
        self._sm_count: dict = {}

    # ------------------------------------------------------------------
    # 设备信息
    # ------------------------------------------------------------------

    @property
    def device_count(self) -> int:
        return self._device_count

    @property
    def current_device(self) -> int:
        """当前上下文所在设备的索引。

        不缓存：``cudaSetDevice`` 也可能被别的库调用，缓存会读到一个过期的
        "当前设备"，而后续的分配与内核都会落在错误的卡上。
        """
        value = ctypes.c_int()
        rc = self._lib.cudaGetDevice(ctypes.byref(value))
        if rc != 0:
            raise CudaRuntimeError(f"cudaGetDevice 失败（code={rc}）")
        return value.value

    def set_device(self, index: int) -> None:
        """切换当前设备；索引非法时在碰驱动之前就报错。"""
        if isinstance(index, bool) or not isinstance(index, int):
            raise CudaRuntimeError(f"设备索引必须是 int，收到 {index!r}")
        if not 0 <= index < self._device_count:
            raise CudaRuntimeError(
                f"设备索引 {index} 越界：进程可见 {self._device_count} 块设备"
                f"（0..{self._device_count - 1}）"
            )
        previous = self.current_device
        rc = self._lib.cudaSetDevice(index)
        if rc != 0:
            raise CudaRuntimeError(f"cudaSetDevice({index}) 失败（code={rc}）")
        if index != previous:
            # 已加载的 code object 绑在切卡前的 context 上。换卡后继续用旧
            # 句柄会启动到一个不认识它的设备，所以整块清掉、按需重新加载。
            self._modules.clear()
            self._functions.clear()

    def _parse_props(self, index: int) -> dict:
        """按索引解析设备属性；结果按设备缓存。"""
        if not 0 <= index < self._device_count:
            raise CudaRuntimeError(
                f"设备索引 {index} 越界：进程可见 {self._device_count} 块设备"
            )
        cached = self._props.get(index)
        if cached is not None:
            return cached
        buf = ctypes.create_string_buffer(8192)
        rc = self._lib.cudaGetDeviceProperties(buf, index)
        if rc != 0:
            raise CudaRuntimeError(
                f"cudaGetDeviceProperties({index}) 失败（code={rc}）"
            )
        raw = buf.raw
        # name 是结构体第一个字段（char[256]）。架构不进结构体扫描：天数
        # 报告的 compute capability 是 7.1，但目标由 ``--cuda-path`` 下的
        # ivcore 后端决定，编译期不需要 arch 字符串（见 rtc.py）。
        name = raw.split(b"\x00", 1)[0].decode(errors="replace")
        props = {"name": name, "arch": None}
        self._props[index] = props
        self._arch = None
        return props

    def _total_memory(self, index: int) -> int:
        """某块设备的显存总量。

        ``cudaMemGetInfo`` 只认当前设备，没有下标参数，所以要临时切过去再
        切回来。读失败返回 0：这里回答的是"这块卡多大"，不是一次必须成功
        的分配，让 ``device_info`` 因为一次查询失败整体崩掉没有意义。
        """
        current = self.current_device
        if index != current:
            self.set_device(index)
        try:
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            rc = self._lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            return total.value if rc == 0 else 0
        finally:
            if index != current:
                self.set_device(current)

    @property
    def arch(self) -> str | None:
        return self._parse_props(self.current_device)["arch"]

    def device_info(self, index: int | None = None):
        """``(设备名, CUDA 运行时版本, 显存总量)``；默认当前设备。"""
        if index is None:
            index = self.current_device
        props = self._parse_props(index)
        version = ctypes.c_int()
        self._lib.cudaRuntimeGetVersion(ctypes.byref(version))
        return props["name"], version.value, self._total_memory(index)

    @property
    def sm_count(self) -> int:
        """当前设备的计算单元数（CU）。查不到返回 0，不猜。

        ``cudaDevAttrMultiProcessorCount`` 的枚举值是 16（与 CUDA 的
        ``cudaDevAttrMultiProcessorCount`` 相同）。自动 block 拿它当
        "网格至少铺满一屏"的目标；0 表示这条规则不生效。
        """
        index = self.current_device
        if index not in self._sm_count:
            value = ctypes.c_int()
            rc = self._lib.cudaDeviceGetAttribute(ctypes.byref(value), 16, index)
            self._sm_count[index] = value.value if rc == 0 and value.value > 0 else 0
        return self._sm_count[index]

    # ------------------------------------------------------------------
    # 同步
    # ------------------------------------------------------------------

    def stream_synchronize(self, ptr: int = 0) -> None:
        """等某个流把活干完；``ptr == 0`` 表示默认流。"""
        rc = self._lib.cudaStreamSynchronize(ctypes.c_void_p(ptr))
        if rc != 0:
            raise CudaRuntimeError(f"cudaStreamSynchronize 失败（code={rc}）")

    # ------------------------------------------------------------------
    # 流与图捕获
    # ------------------------------------------------------------------

    @property
    def current_stream(self) -> int:
        return current_stream()

    @property
    def capturing(self) -> bool:
        return self._capturing

    def create_stream(self) -> int:
        """非阻塞流。捕获必须发生在一个非默认流上。"""
        ptr = ctypes.c_void_p()
        rc = self._lib.cudaStreamCreateWithFlags(ctypes.byref(ptr), 1)
        if rc != 0:
            raise CudaRuntimeError(f"cudaStreamCreateWithFlags 失败（code={rc}）")
        return ptr.value

    def destroy_stream(self, ptr: int) -> None:
        if ptr:
            self._lib.cudaStreamDestroy(ctypes.c_void_p(int(ptr)))

    def capture(self, stream: int) -> "_CaptureSession":
        """开始一次流捕获；用作上下文管理器，退出时拿到图句柄。"""
        return _CaptureSession(self, int(stream))

    def graph_exec(self, graph: int) -> "_GraphExec":
        """把捕获到的图实例化成可重复 launch 的执行图。"""
        exec_ptr = ctypes.c_void_p()
        rc = self._lib.cudaGraphInstantiate(
            ctypes.byref(exec_ptr), ctypes.c_void_p(int(graph)), None, None, 0
        )
        if rc != 0:
            self._lib.cudaGraphDestroy(ctypes.c_void_p(int(graph)))
            raise CudaRuntimeError(f"cudaGraphInstantiate 失败（code={rc}）")
        return _GraphExec(self, exec_ptr, int(graph))

    # ------------------------------------------------------------------
    # 分配器
    # ------------------------------------------------------------------

    def set_allocator(self, allocator) -> None:
        """替换 ``empty`` 的底层分配器；``allocator(nbytes)`` 返回持有者。"""
        self._allocator = allocator

    def get_allocator(self):
        return self._allocator

    def alloc(self, nbytes: int):
        """按当前分配器申请一块显存，返回持有者（带 ``ptr`` 与 ``nbytes``）。"""
        allocator = self._allocator
        if allocator is not None:
            return allocator(nbytes)
        return _Buffer(self.malloc(nbytes), nbytes)

    # ------------------------------------------------------------------
    # 内存
    # ------------------------------------------------------------------

    def malloc(self, nbytes: int) -> int:
        if nbytes <= 0:
            return 0
        if self._capturing:
            raise CudaRuntimeError(
                f"捕获期不能申请新显存（{nbytes} 字节）：预热轮的 arena 容量"
                "没有覆盖捕获轮。cudaMalloc 在流捕获期间不被允许"
            )
        ptr = ctypes.c_void_p()
        rc = self._lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if rc != 0:
            raise CudaRuntimeError(
                f"cudaMalloc({nbytes}) 失败（code={rc}）：显存不足或上下文失效"
            )
        return ptr.value

    def free(self, ptr: int) -> None:
        if not ptr:
            return
        if self._capturing:
            # 见 ``_deferred_free``：捕获窗口里的 cudaFree 会让捕获失效
            self._deferred_free.append(ptr)
            return
        self._lib.cudaFree(ctypes.c_void_p(ptr))

    # ------------------------------------------------------------------
    # 页锁定主机内存
    # ------------------------------------------------------------------

    def alloc_pinned(self, nbytes: int) -> _Pinned:
        """申请一块 pinned 主机内存并建好 numpy 视图（不初始化内容）。"""
        nbytes = max(int(nbytes), 1)
        ptr = ctypes.c_void_p()
        # flags=0 即 cudaHostAllocDefault：可被异步拷贝直接读取，不要求映射
        rc = self._lib.cudaHostAlloc(
            ctypes.byref(ptr), ctypes.c_size_t(nbytes), 0
        )
        if rc != 0:
            raise CudaRuntimeError(
                f"cudaHostAlloc({nbytes}) 失败（code={rc}）：页锁定内存不足"
            )
        raw = (ctypes.c_char * nbytes).from_address(ptr.value)
        view = np.ctypeslib.as_array(raw)
        return _Pinned(ptr.value, nbytes, view)

    def free_pinned(self, ptr: int) -> None:
        if not ptr:
            return
        if self._capturing:
            # cudaFreeHost 会隐式同步设备，同样不能在捕获窗口里发生
            self._deferred_pinned.append(int(ptr))
            return
        self._lib.cudaFreeHost(ctypes.c_void_p(int(ptr)))

    def drain_deferred(self) -> None:
        """把捕获期记下的释放补做掉；必须在 ``cudaStreamEndCapture`` 之后调用。"""
        pointers, self._deferred_free = self._deferred_free, []
        for ptr in pointers:
            self._lib.cudaFree(ctypes.c_void_p(ptr))
        pinned, self._deferred_pinned = self._deferred_pinned, []
        for ptr in pinned:
            self._lib.cudaFreeHost(ctypes.c_void_p(int(ptr)))

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def new_event(self, timing: bool = False) -> "_Event":
        """建一个事件。

        默认禁用计时（只用来排序）；``timing=True`` 时带计时标志，供逐节点
        GPU 计时用。两种事件的用途不能混：带计时的事件多一次时钟采样，放进
        replay 的热路径会平白增加开销。
        """
        ptr = ctypes.c_void_p()
        # cudaEventDefault = 0 / cudaEventDisableTiming = 2
        flags = 0 if timing else 2
        rc = self._lib.cudaEventCreateWithFlags(ctypes.byref(ptr), flags)
        if rc != 0:
            raise CudaRuntimeError(f"cudaEventCreateWithFlags 失败（code={rc}）")
        return _Event(self, ptr.value, timing)

    def stream_wait_event(self, stream: int, event: "_Event") -> None:
        rc = self._lib.cudaStreamWaitEvent(
            ctypes.c_void_p(int(stream)), event._event, 0
        )
        if rc != 0:
            raise CudaRuntimeError(f"cudaStreamWaitEvent 失败（code={rc}）")

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        if nbytes == 0:
            return
        stream = self.current_stream
        # 捕获期必须用异步版本：同步 ``cudaMemcpy`` 会打断捕获（CUDA 直接报错），
        # 异步版本则被记录成图里的一个拷贝节点。
        if self._capturing:
            source = src
            if kind == _H2D and self._capture_host is not None:
                # 图里记录的是**主机地址**，replay 时源必须还活着。``full`` /
                # ``array`` 的源是函数里的临时 numpy，捕获一结束就释放了；
                # 不暂存就会在 replay 时读到已释放的内存（实测常量变成垃圾，
                # 整张 events 图输出全错）。暂存块由捕获会话交给计划持有。
                buf = ctypes.create_string_buffer(nbytes)
                ctypes.memmove(buf, ctypes.c_void_p(src), nbytes)
                self._capture_host.append(buf)
                source = ctypes.addressof(buf)
            rc = self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst), ctypes.c_void_p(source),
                ctypes.c_size_t(nbytes), kind,
                ctypes.c_void_p(stream),
            )
        elif stream == 0:
            rc = self._lib.cudaMemcpy(
                ctypes.c_void_p(dst), ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes), kind,
            )
        else:
            # 处在 ``with stream:`` 里（捕获预热轮就是这样）时，同步拷贝跑在
            # 默认流上，与当前流的内核之间没有任何顺序关系。实测后果：紧跟
            # 其后的 ``spike_offsets`` 内核写完之后，默认流上被推迟的 memset
            # 才落地，把最后一个元素清零。所以这里也排进当前流，并等它完成
            # ——既保证顺序，也保证 H2D 的源在返回后可以安全释放。
            rc = self._lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst), ctypes.c_void_p(src),
                ctypes.c_size_t(nbytes), kind,
                ctypes.c_void_p(stream),
            )
            if rc == 0:
                self.stream_synchronize(stream)
        if rc != 0:
            raise CudaRuntimeError(f"cudaMemcpy 失败（code={rc}, {nbytes} 字节）")

    def memcpy_async(
        self, dst: int, src: int, nbytes: int, kind: int, stream: int
    ) -> None:
        """把一次拷贝排进指定流，**不等它完成**。

        replay 的 H2D 用这条路径：源是 pinned 缓冲（``_Pinned``），因此主机
        侧不必等待传输；顺序由流保证（紧接着的 graph launch 在同一条流上）。
        与 ``memcpy`` 的区别是这里没有"当前流"概念——调用方明确给出流地址，
        也不做任何同步。
        """
        if nbytes == 0:
            return
        rc = self._lib.cudaMemcpyAsync(
            ctypes.c_void_p(dst), ctypes.c_void_p(src),
            ctypes.c_size_t(nbytes), kind, ctypes.c_void_p(int(stream)),
        )
        if rc != 0:
            raise CudaRuntimeError(f"cudaMemcpyAsync 失败（code={rc}, {nbytes} 字节）")

    def memset(self, ptr: int, value: int, nbytes: int) -> None:
        if nbytes == 0:
            return
        stream = self.current_stream
        if self._capturing:
            rc = self._lib.cudaMemsetAsync(
                ctypes.c_void_p(ptr), value, ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
        elif stream == 0:
            rc = self._lib.cudaMemset(
                ctypes.c_void_p(ptr), value, ctypes.c_size_t(nbytes)
            )
            if rc == 0:
                # WSL/DXG 上 ``cudaMemset`` 返回时并不保证已经落地：紧接着在
                # 另一条非阻塞流上启动的内核会被它事后覆盖（实测把
                # ``spike_offsets`` 的最后一个元素清零）。补一次流同步，
                # 让"同步 memset"名副其实。
                self.stream_synchronize(0)
        else:
            # 与 memcpy 同一条理由：非默认流上的清零必须排进这条流，否则
            # 会与紧随其后的内核竞争，把内核刚写好的结果清掉。
            rc = self._lib.cudaMemsetAsync(
                ctypes.c_void_p(ptr), value, ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream),
            )
            if rc == 0:
                self.stream_synchronize(stream)
        if rc != 0:
            raise CudaRuntimeError(f"cudaMemset 失败（code={rc}）")

    def synchronize(self) -> None:
        self._lib.cudaDeviceSynchronize()

    # ------------------------------------------------------------------
    # 编译与内核
    # ------------------------------------------------------------------

    def compile(self, source: str, name: str) -> bytes:
        """编译源码。code object 按源码内容的摘要缓存，避免重复编译。"""
        digest = hashlib.sha1(source.encode()).hexdigest()
        key = (name, digest)
        cached = self._modules.get(key)
        if cached is not None:
            return cached
        code = self._rc.compile(source, name, arch=self.arch)
        self._modules[key] = code
        return code

    def function(self, key, source: str, name: str, func: str) -> "CudaFunction":
        # 缓存键里带上源码摘要：调用方给的 key（算子名 + dtype）在重构期间
        # 可能不变，但源码会变；只按 key 缓存会让开发期的修改静默失效。
        cache_key = (key, hashlib.sha1(source.encode()).hexdigest())
        cached = self._functions.get(cache_key)
        if cached is not None:
            return cached
        code = self.compile(source, name)
        buf = ctypes.create_string_buffer(code, len(code))
        module = ctypes.c_void_p()
        rc = self._drv.cuModuleLoadData(
            ctypes.byref(module), ctypes.cast(buf, ctypes.c_void_p)
        )
        if rc != 0:
            raise CudaRuntimeError(f"cuModuleLoadData 失败（code={rc}, {name}）")
        fn = ctypes.c_void_p()
        rc = self._drv.cuModuleGetFunction(ctypes.byref(fn), module, func.encode())
        if rc != 0:
            raise CudaRuntimeError(f"cuModuleGetFunction({func}) 失败（code={rc}）")
        result = CudaFunction(self, fn, module, func)
        # 模块句柄与函数句柄一起缓存：模块被回收会让已缓存的函数悬空
        self._functions[cache_key] = result
        return result

    def launch(self, fn, grid, block, params, shared_mem: int = 0) -> None:
        rc = self._drv.cuLaunchKernel(
            fn.handler,
            int(grid[0]), int(grid[1] if len(grid) > 1 else 1),
            int(grid[2] if len(grid) > 2 else 1),
            int(block[0]), int(block[1] if len(block) > 1 else 1),
            int(block[2] if len(block) > 2 else 1),
            int(shared_mem), ctypes.c_void_p(self.current_stream), params, None,
        )
        if rc != 0:
            raise CudaRuntimeError(f"cuLaunchKernel({fn.name}) 失败（code={rc}）")


class _Event:
    """一个事件：默认只用来在两条流之间建顺序，``timing=True`` 时另外能报耗时。

    ``record`` 之后立刻 ``destroy`` 是允许的——驱动会在等待它的操作都完成
    后才真正释放资源。replay 每次都要一个新的记录点，不能复用同一个事件：
    重新记录会改掉"尚未执行的等待"的语义。

    ``elapsed_ms`` 只对带计时的事件成立，且调用前必须已经同步（或由调用方
    保证两个事件都完成）；未完成时驱动返回 ``cudaErrorNotReady``，这里把它
    翻译成明确的 ``CudaRuntimeError``，不返回一个编出来的数字。
    """

    __slots__ = ("_rt", "_event", "_timing")

    def __init__(self, rt: "_Runtime", event, timing: bool = False) -> None:
        self._rt = rt
        self._event = event
        self._timing = bool(timing)

    def record(self, stream: int = 0) -> None:
        rc = self._rt._lib.cudaEventRecord(
            self._event, ctypes.c_void_p(int(stream))
        )
        if rc != 0:
            raise CudaRuntimeError(f"cudaEventRecord 失败（code={rc}）")

    def elapsed_ms(self, start: "_Event") -> float:
        """从 ``start`` 到本事件的毫秒数。两者都必须带计时标志。"""
        if not self._timing or not start._timing:
            raise CudaRuntimeError(
                "elapsed_ms 需要带计时标志的事件（new_event(timing=True)）"
            )
        value = ctypes.c_float()
        rc = self._rt._lib.cudaEventElapsedTime(
            ctypes.byref(value), start._event, self._event
        )
        if rc != 0:
            raise CudaRuntimeError(f"cudaEventElapsedTime 失败（code={rc}）")
        return float(value.value)

    def destroy(self) -> None:
        event, self._event = self._event, None
        if event is not None:
            self._rt._lib.cudaEventDestroy(event)

    def __del__(self) -> None:
        # 异常路径（invoke 中途抛出）由 GC 兜底；显式路径仍应调用 destroy()
        try:
            self.destroy()
        except Exception:  # 解释器退出时驱动可能已经不在了
            pass


class _CaptureSession:
    """一次流捕获的上下文：进入时开始，退出时结束并留下图句柄。

    捕获体里抛异常时仍然必须 ``cudaStreamEndCapture``——否则这条流会永远
    停在捕获态，之后任何 launch 都会被驱动拒绝。因此这里无条件结束捕获，
    只在没有原始异常时才把结束失败升级成错误。
    """

    __slots__ = ("_rt", "stream", "graph", "host_refs")

    def __init__(self, rt: "_Runtime", stream: int) -> None:
        self._rt = rt
        self.stream = stream
        self.graph = 0
        # 捕获期 H2D 的主机源暂存：图里只记指针，这些缓冲必须活到计划销毁
        self.host_refs: list = []

    def __enter__(self) -> "_CaptureSession":
        # cudaStreamCaptureModeGlobal = 0：同一线程里的其它流若与本流有依赖，
        # 也会被并入捕获。这里没有别的流，选最严格的一种。
        rc = self._rt._lib.cudaStreamBeginCapture(
            ctypes.c_void_p(self.stream), 0
        )
        if rc != 0:
            raise CudaRuntimeError(f"cudaStreamBeginCapture 失败（code={rc}）")
        self._rt._capturing = True
        self._rt._capture_host = self.host_refs
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._rt._capturing = False
        self._rt._capture_host = None
        ptr = ctypes.c_void_p()
        try:
            rc = self._rt._lib.cudaStreamEndCapture(
                ctypes.c_void_p(self.stream), ctypes.byref(ptr)
            )
            if rc != 0:
                if exc_type is None:
                    raise CudaRuntimeError(
                        f"cudaStreamEndCapture 失败（code={rc}）：捕获期出现了流捕获"
                        "不支持的调用（同步拷贝、宿主回读、设备分配等）"
                    )
                return False
            self.graph = ptr.value
            return False
        finally:
            # 捕获窗口里被 GC 归还的显存 / pinned 缓冲在这里补释放
            self._rt.drain_deferred()


class _GraphExec:
    """已实例化的 cudaGraph：``launch`` 一次下发整图，不再逐节点走 Python。"""

    __slots__ = ("_rt", "_exec", "_graph")

    def __init__(self, rt: "_Runtime", exec_ptr, graph: int) -> None:
        self._rt = rt
        self._exec = exec_ptr
        self._graph = int(graph)

    def launch(self, stream: int = 0) -> None:
        rc = self._rt._lib.cudaGraphLaunch(
            self._exec, ctypes.c_void_p(int(stream))
        )
        if rc != 0:
            raise CudaRuntimeError(f"cudaGraphLaunch 失败（code={rc}）")

    def close(self) -> None:
        exec_ptr, self._exec = self._exec, None
        if exec_ptr is not None:
            self._rt._lib.cudaGraphExecDestroy(exec_ptr)
        graph, self._graph = self._graph, 0
        if graph:
            self._rt._lib.cudaGraphDestroy(ctypes.c_void_p(graph))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # 解释器退出时驱动可能已经不在了
            pass


class CudaFunction:
    """一个已加载的内核。调用签名与 cupy 的 ``RawKernel`` 对齐：

    ``fn((grid,), (block,), (arg0, arg1, ...))``
    """

    __slots__ = ("_runtime", "handler", "_module", "name")

    def __init__(self, runtime: _Runtime, handler, module, name: str) -> None:
        self._runtime = runtime
        self.handler = handler
        self._module = module  # 持有引用，防止模块被卸载
        self.name = name

    def __call__(self, grid, block, args, shared_mem: int = 0) -> None:
        """``shared_mem`` 对应 CUDA 侧 ``RawKernel(..., shared_mem=N)`` 的用法：
        attention 与分块卷积按形状动态申请共享内存。"""
        holders = [_pack(a) for a in args]
        params = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
        )
        self._runtime.launch(self, grid, block, params, shared_mem)


def _pack(value):
    """把一个内核实参打包成 ctypes 值。

    CUDA 的 ``cudaModuleLaunchKernel`` 要求每一项指向"参数值"的内存：
    指针参数要传指针值本身，标量参数要传该标量的地址。因此这里返回的是
    持有者对象，而不是裸整数（裸整数会被当成 32 位截断）。
    """
    if isinstance(value, DeviceArray):
        return ctypes.c_void_p(value.ptr)
    if isinstance(value, np.generic):
        if value.dtype == np.int64:
            return ctypes.c_longlong(int(value))
        if value.dtype == np.int32:
            return ctypes.c_int(int(value))
        if value.dtype == np.float32:
            return ctypes.c_float(float(value))
        if value.dtype == np.float64:
            return ctypes.c_double(float(value))
        if value.dtype == np.uint64:
            return ctypes.c_ulonglong(int(value))
        raise TypeError(f"内核标量参数不支持 dtype {value.dtype}")
    if isinstance(value, int) and not isinstance(value, bool):
        raise TypeError(
            "内核的整数参数必须显式写成 np.int32 / np.int64："
            "裸 int 在不同内核里的宽度不同，静默选一个会产生 ABI 错位"
        )
    raise TypeError(f"内核参数类型不支持：{type(value).__name__}")


# ----------------------------------------------------------------------
# 容器层：与 cupy 同名同义的函数
# ----------------------------------------------------------------------


def empty(shape, dtype=float32) -> DeviceArray:
    dt = _dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(d) for d in shape)
    nbytes = _prod(shape) * dt.itemsize
    holder = _Runtime.instance().alloc(nbytes) if nbytes else None
    return DeviceArray(
        holder, holder.ptr if holder is not None else 0,
        shape, _contiguous_strides(shape), dt,
    )


def empty_like(a: DeviceArray) -> DeviceArray:
    return empty(a.shape, a.dtype)


def zeros(shape, dtype=float32) -> DeviceArray:
    out = empty(shape, dtype)
    _Runtime.instance().memset(out.ptr, 0, out.nbytes)
    return out


def full(shape, value, dtype=float32) -> DeviceArray:
    out = empty(shape, dtype)
    if out.size:
        host = np.full(out.shape, value, dtype=out.dtype)
        _Runtime.instance().memcpy(out.ptr, host.ctypes.data, out.nbytes, _H2D)
    return out


def ones(shape, dtype=float32) -> DeviceArray:
    return full(shape, 1, dtype)


def asarray(value) -> DeviceArray:
    if isinstance(value, DeviceArray):
        return value
    host = np.asarray(value)
    dt = _dtype(host.dtype)
    out = empty(host.shape, dt)
    host = np.ascontiguousarray(host, dtype=dt)
    if out.nbytes:
        _Runtime.instance().memcpy(out.ptr, host.ctypes.data, out.nbytes, _H2D)
    return out


def array(value, dtype=float32, copy: bool = True) -> DeviceArray:
    # ``copy`` 只为与 numpy / cupy 的调用签名对齐：进入设备本来就要一次 H2D，
    # 不存在"复用同一块显存"的零拷贝语义。
    if dtype is not None:
        value = np.asarray(value, dtype=dtype)
    return asarray(value)


def asnumpy(a: DeviceArray) -> np.ndarray:
    contiguous = ascontiguousarray(a)
    host = np.empty(contiguous.shape, dtype=contiguous.dtype)
    if host.nbytes:
        _Runtime.instance().memcpy(
            host.ctypes.data, contiguous.ptr, host.nbytes, _D2H
        )
    return host


def reshape(a: DeviceArray, shape) -> DeviceArray:
    return a.reshape(shape)


def transpose(a: DeviceArray, perm) -> DeviceArray:
    perm = tuple(int(p) % a.ndim for p in perm)
    if sorted(perm) != list(range(a.ndim)):
        raise CudaRuntimeError(f"transpose 的 perm 不是排列：{perm}")
    return DeviceArray(
        a._buffer, a.ptr,
        tuple(a.shape[p] for p in perm),
        tuple(a.strides[p] for p in perm),
        a.dtype,
        view=True,
    )


def broadcast_to(a: DeviceArray, shape) -> DeviceArray:
    shape = tuple(int(d) for d in shape)
    pad = len(shape) - a.ndim
    if pad < 0:
        raise CudaRuntimeError(f"无法把 {a.shape} 广播到 {shape}：目标秩更小")
    padded = (1,) * pad + a.shape
    strides = []
    for dim, src, step in zip(shape, padded, (0,) * pad + a.strides):
        if src == dim:
            strides.append(step)
        elif src == 1:
            strides.append(0)
        else:
            raise CudaRuntimeError(f"无法把 {a.shape} 广播到 {shape}")
    return DeviceArray(a._buffer, a.ptr, shape, tuple(strides), a.dtype, view=True)


def ascontiguousarray(a) -> DeviceArray:
    a = asarray(a)
    if a.flags.c_contiguous:
        return a
    out = empty(a.shape, a.dtype)
    if out.size:
        _strided_copy(a, out)
    return out


# ----------------------------------------------------------------------
# 辅助内核：strided copy 与 dtype 转换
# ----------------------------------------------------------------------

_COPY_SRC = r"""
extern "C" __global__ void af_copy_strided(const char* __restrict__ src,
                                           char* __restrict__ dst,
                                           long long n,
                                           int esize,
                                           long long d0, long long d1,
                                           long long d2, long long d3,
                                           long long s0, long long s1,
                                           long long s2, long long s3) {
    long long shape[4] = {d0, d1, d2, d3};
    long long steps[4] = {s0, s1, s2, s3};
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        long long rem = i, off = 0;
        for (int d = 3; d >= 0; --d) {
            long long c = rem % shape[d];
            rem /= shape[d];
            off += c * steps[d];
        }
        for (int b = 0; b < esize; ++b) { dst[i * esize + b] = src[off * esize + b]; }
    }
}
"""

# 连续源 → 带步长目标。分组卷积把每组结果写回 ``out[:, a:b]`` 时需要：
# 源是 im2col 刚算出的连续缓冲，目标在通道维上被前后两组夹着。
# 索引方式与 ``_COPY_SRC`` 相同，只是把目标偏移也按步长算一遍。
_COPY_PAIR_SRC = r"""
extern "C" __global__ void af_copy_pair(const char* __restrict__ src,
                                        char* __restrict__ dst,
                                        long long n,
                                        int esize,
                                        long long d0, long long d1,
                                        long long d2, long long d3,
                                        long long ss0, long long ss1,
                                        long long ss2, long long ss3,
                                        long long ds0, long long ds1,
                                        long long ds2, long long ds3) {
    long long shape[4] = {d0, d1, d2, d3};
    long long sstep[4] = {ss0, ss1, ss2, ss3};
    long long dstep[4] = {ds0, ds1, ds2, ds3};
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        long long rem = i, so = 0, doff = 0;
        for (int d = 3; d >= 0; --d) {
            long long c = rem % shape[d];
            rem /= shape[d];
            so += c * sstep[d];
            doff += c * dstep[d];
        }
        for (int b = 0; b < esize; ++b) { dst[doff * esize + b] = src[so * esize + b]; }
    }
}
"""

_CAST_SRC = r"""
typedef %TIN% tin_t;
typedef %TOUT% tout_t;
extern "C" __global__ void af_cast(const tin_t* __restrict__ in,
                                   tout_t* __restrict__ out,
                                   long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) { out[i] = (tout_t)in[i]; }
}
"""

# f16 用 clang 的 _Float16：它和 numpy 的 float16 是同一套 IEEE 半精度，
# 且不需要 include hip_fp16.h（cudaRTC 默认没有 include 路径，探针已确认）。
_C_TYPES = {
    np.dtype("float32"): "float",
    np.dtype("float16"): "_Float16",
    np.dtype("float64"): "double",
    np.dtype("int32"): "int",
    np.dtype("int8"): "signed char",
    np.dtype("bool"): "bool",
    # bf16 位模式：容器是 uint16，cast 内核按无符号短整型搬运位模式。
    np.dtype("uint16"): "unsigned short",
}

# 逐元素二元运算：op 0/1/2/3 = + - * /。两个操作数各带一套（已广播的）
# 形状与步长，输出永远是连续的。这与 ``_strided_copy`` 的索引方式一致。
_EW_BINARY_SRC = r"""
typedef %T% T;
extern "C" __global__ void af_ew_binary(T* __restrict__ out,
                                        const T* __restrict__ a,
                                        const T* __restrict__ b,
                                        long long n, int op,
                                        long long d0, long long d1,
                                        long long d2, long long d3,
                                        long long sa0, long long sa1,
                                        long long sa2, long long sa3,
                                        long long sb0, long long sb1,
                                        long long sb2, long long sb3) {
    long long shape[4] = {d0, d1, d2, d3};
    long long sa[4] = {sa0, sa1, sa2, sa3};
    long long sb[4] = {sb0, sb1, sb2, sb3};
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        long long rem = i, ao = 0, bo = 0;
        for (int d = 3; d >= 0; --d) {
            long long c = rem % shape[d];
            rem /= shape[d];
            ao += c * sa[d];
            bo += c * sb[d];
        }
        T x = a[ao], y = b[bo], r;
        if (op == 0) r = x + y;
        else if (op == 1) r = x - y;
        else if (op == 2) r = x * y;
        else r = x / y;
        out[i] = r;
    }
}
"""

# 数组与标量：``alpha`` 用 double 传入，到内核里再落回 T；``reverse`` 只在
# 标量在左侧（``2.0 - x``）时需要。
_EW_SCALAR_SRC = r"""
typedef %T% T;
extern "C" __global__ void af_ew_scalar(T* __restrict__ out,
                                        const T* __restrict__ a,
                                        double alpha, long long n,
                                        int op, int reverse) {
    T y = (T)alpha;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        T x = a[i];
        T l = reverse ? y : x;
        T r = reverse ? x : y;
        T v;
        if (op == 0) v = l + r;
        else if (op == 1) v = l - r;
        else if (op == 2) v = l * r;
        else v = l / r;
        out[i] = v;
    }
}
"""

# op 0 = 取负，op 1 = 开方
_EW_UNARY_SRC = r"""
typedef %T% T;
extern "C" __global__ void af_ew_unary(T* __restrict__ out,
                                       const T* __restrict__ a,
                                       long long n, int op) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        T x = a[i];
        if (op == 0) out[i] = -x;
        else if (sizeof(T) == 4) out[i] = (T)sqrtf((float)x);
        else out[i] = (T)sqrt((double)x);
    }
}
"""

_FILL_SRC = r"""
typedef %T% T;
extern "C" __global__ void af_fill(T* __restrict__ out, double value,
                                   long long n) {
    T v = (T)value;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = v;
}
"""

# 全和：先按块写成 double 分片，再由单块收尾。累加用 double 是刻意的——
# 训练层的全局范数本来就在 f64 上求平方和（见 train/grads.py）。
_SUM_PARTIAL_SRC = r"""
typedef %T% T;
extern "C" __global__ void af_sum_partial(const T* __restrict__ src,
                                          double* __restrict__ parts,
                                          long long n) {
    __shared__ double buf[256];
    double acc = 0.0;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) acc += (double)src[i];
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) parts[blockIdx.x] = buf[0];
}
"""

_SUM_FINAL_SRC = r"""
extern "C" __global__ void af_sum_final(const double* __restrict__ parts,
                                        double* __restrict__ out, int count) {
    __shared__ double buf[256];
    double acc = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) acc += parts[i];
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = buf[0];
}
"""

# 运算码
_ADD, _SUB, _MUL, _DIV = 0, 1, 2, 3
_NEG, _SQRT = 0, 1


def _collapse(shape, strides):
    """合并相邻且步长连续的维，返回 (dims, steps, ndim)。

    视图通常只在最外层不连续（转置、切片），折叠后基本都能降到 2-3 维，
    从而落在 4 维上限内。
    """
    dims = []
    steps = []
    for d, s in zip(reversed(shape), reversed(strides)):
        if dims and steps[-1] * dims[-1] == s:
            dims[-1] *= d
        else:
            dims.append(d)
            steps.append(s)
    dims.reverse()
    steps.reverse()
    if len(dims) > _MAX_DIMS:
        raise CudaRuntimeError(
            f"视图折叠后仍有 {len(dims)} 维，超过 {_MAX_DIMS}：{shape}"
        )
    while len(dims) < _MAX_DIMS:
        dims.append(1)
        steps.append(0)
    return dims, steps, len(dims)


def _collapse_pair(shape, sa, sb):
    """把同一形状与**两套**步长一起折叠，返回 (dims, sa, sb)。

    ``_collapse`` 按单个操作数折叠时，连续数组会把内层几维并成一维
    （``(2,16,4,4)`` 的连续视图折成 ``[2,256,1,1]``），而零步长的广播视图
    会保留 ``[2,16,16,1]``。内核只有一套 ``dims``，若用它去乘另一套步长，
    步长与维度就错位了——例如把 bias 的 channel 步长乘到 h*w 维上，结果是
    所有通道都读成 channel 0。两个操作数必须共用同一套分解。
    """
    dims: list = []
    left: list = []
    right: list = []
    for d, a, b in zip(reversed(shape), reversed(sa), reversed(sb)):
        if dims and left[-1] * dims[-1] == a and right[-1] * dims[-1] == b:
            dims[-1] *= d
        else:
            dims.append(d)
            left.append(a)
            right.append(b)
    dims.reverse()
    left.reverse()
    right.reverse()
    if len(dims) > _MAX_DIMS:
        raise CudaRuntimeError(
            f"两个操作数联合折叠后仍有 {len(dims)} 维，超过 {_MAX_DIMS}：{shape}"
        )
    while len(dims) < _MAX_DIMS:
        dims.append(1)
        left.append(0)
        right.append(0)
    return dims, left, right


def _strided_copy(src: DeviceArray, dst: DeviceArray) -> None:
    dims, steps, ndim = _collapse(src.shape, src.strides)
    esize = src.dtype.itemsize
    n = src.size
    block = 256
    grid = max(1, min(1024, (n + block - 1) // block))
    fn = _Runtime.instance().function(
        ("af_copy_strided",), _COPY_SRC, "af_copy", "af_copy_strided"
    )
    fn(
        (grid,), (block,),
        (
            src, dst, np.int64(n), np.int32(esize),
            *[np.int64(v) for v in dims], *[np.int64(v) for v in steps],
        ),
    )


def _strided_copy_pair(src: DeviceArray, dst: DeviceArray) -> None:
    """``src``（连续）→ ``dst``（可带步长），形状已由调用方对齐。

    与 ``_strided_copy`` 的区别只在目标：那个写连续缓冲（``dst[i]``），这个
    按目标自己的步长算地址。分组卷积把每组结果写回 ``out[:, a:b]`` 时需要
    后者——输出通道块在中间，写回去是"连续源、非连续目标"。
    """
    if src.size == 0:
        return
    dims, ssteps, dsteps = _collapse_pair(src.shape, src.strides, dst.strides)
    esize = src.dtype.itemsize
    n = src.size
    block = 256
    grid = max(1, min(1024, (n + block - 1) // block))
    fn = _Runtime.instance().function(
        ("af_copy_pair",), _COPY_PAIR_SRC, "af_copy_pair", "af_copy_pair"
    )
    fn(
        (grid,), (block,),
        (
            src, dst, np.int64(n), np.int32(esize),
            *[np.int64(v) for v in dims],
            *[np.int64(v) for v in ssteps],
            *[np.int64(v) for v in dsteps],
        ),
    )


def _cast(a: DeviceArray, dtype: np.dtype) -> DeviceArray:
    out = empty(a.shape, dtype)
    if out.size:
        key = ("af_cast", a.dtype.str, dtype.str)
        src = _CAST_SRC.replace("%TIN%", _C_TYPES[a.dtype]).replace(
            "%TOUT%", _C_TYPES[dtype]
        )
        fn = _Runtime.instance().function(key, src, "af_cast", "af_cast")
        block = 256
        grid = max(1, min(1024, (out.size + block - 1) // block))
        fn((grid,), (block,), (a, out, np.int64(out.size)))
    return out


# ----------------------------------------------------------------------
# 逐元素运算的实现（容器运算符与训练层共用）
# ----------------------------------------------------------------------

_BLOCK = 256
_MAX_BLOCKS = 1024


def _launch_shape(n: int):
    return max(1, min(_MAX_BLOCKS, (n + _BLOCK - 1) // _BLOCK)), _BLOCK


def _template(dtype: np.dtype, template: str, name: str):
    """按 dtype 展开 ``%T%`` 并编译（按 dtype 缓存，源码不同即不同函数）。"""
    text = template.replace("%T%", _C_TYPES[np.dtype(dtype)])
    return _Runtime.instance().function(
        (name, np.dtype(dtype).str), text, name, name
    )


def _broadcast_shape(a: tuple, b: tuple) -> tuple:
    """按 numpy 规则算广播后的形状；不兼容直接报错。"""
    ndim = max(len(a), len(b))
    pa = (1,) * (ndim - len(a)) + tuple(a)
    pb = (1,) * (ndim - len(b)) + tuple(b)
    out = []
    for x, y in zip(pa, pb):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise CudaRuntimeError(f"无法广播 {a} 与 {b}")
    return tuple(out)


def _broadcast_strides(arr: DeviceArray, shape: tuple) -> tuple:
    pad = len(shape) - arr.ndim
    steps = [0] * pad + list(arr.strides)
    sizes = [1] * pad + list(arr.shape)
    out = []
    for dim, size, step in zip(shape, sizes, steps):
        if size == dim:
            out.append(step)
        elif size == 1:
            out.append(0)
        else:  # _broadcast_shape 已保证到不了这里
            raise CudaRuntimeError(f"无法把 {arr.shape} 广播到 {shape}")
    return tuple(out)


def _ew_binary(
    arr: DeviceArray, other, op: int, reverse: bool = False,
    out: DeviceArray | None = None,
):
    """``arr <op> other``；``out`` 给出时写进它（原地更新）。"""
    if isinstance(other, DeviceArray):
        if other.dtype != arr.dtype:
            raise CudaRuntimeError(
                f"逐元素运算要求两侧 dtype 相同：{arr.dtype} 与 {other.dtype}"
            )
        shape = _broadcast_shape(arr.shape, other.shape)
    else:
        host = np.asarray(other)
        if host.ndim != 0:
            raise CudaRuntimeError(
                f"CUDA 容器的逐元素运算只接受同 dtype 设备数组或标量，"
                f"收到形状 {host.shape}"
            )
        shape = arr.shape

    if out is not None:
        if tuple(out.shape) != shape or out.dtype != arr.dtype:
            raise CudaRuntimeError(
                f"原地更新要求输出形状/dtype 一致：{out.shape}/{out.dtype} "
                f"收到 {shape}/{arr.dtype}"
            )
        if not out.flags.c_contiguous:
            raise CudaRuntimeError("原地更新要求目标数组连续")
    if not arr.flags.c_contiguous:
        if out is not None:
            raise CudaRuntimeError("原地更新要求源数组连续")
        arr = ascontiguousarray(arr)

    target = out if out is not None else empty(shape, arr.dtype)
    n = _prod(shape)
    if n == 0:
        return target
    grid, block = _launch_shape(n)

    if isinstance(other, DeviceArray):
        rhs = other if other.flags.c_contiguous else ascontiguousarray(other)
        dims, sa, sb = _collapse_pair(
            shape,
            _broadcast_strides(arr, shape),
            _broadcast_strides(rhs, shape),
        )
        fn = _template(arr.dtype, _EW_BINARY_SRC, "af_ew_binary")
        fn(
            (grid,), (block,),
            (
                target, arr, rhs, np.int64(n), np.int32(op),
                *[np.int64(v) for v in dims],
                *[np.int64(v) for v in sa],
                *[np.int64(v) for v in sb],
            ),
        )
        return target

    fn = _template(arr.dtype, _EW_SCALAR_SRC, "af_ew_scalar")
    fn(
        (grid,), (block,),
        (
            target, arr, np.float64(np.asarray(other).item()), np.int64(n),
            np.int32(op), np.int32(1 if reverse else 0),
        ),
    )
    return target


def _ew_binary_into(arr: DeviceArray, other, op: int) -> None:
    _ew_binary(arr, other, op, out=arr)


def _ew_unary(arr: DeviceArray, op: int) -> DeviceArray:
    src = arr if arr.flags.c_contiguous else ascontiguousarray(arr)
    out = empty(arr.shape, arr.dtype)
    n = out.size
    if n == 0:
        return out
    grid, block = _launch_shape(n)
    fn = _template(src.dtype, _EW_UNARY_SRC, "af_ew_unary")
    fn((grid,), (block,), (out, src, np.int64(n), np.int32(op)))
    return out


def _fill(arr: DeviceArray, value) -> None:
    """把整块连续缓冲刷成同一个值（标量赋值与 ``zeros`` 共用）。"""
    if not arr.flags.c_contiguous:
        raise CudaRuntimeError("标量填充要求目标数组连续")
    if arr.size == 0:
        return
    grid, block = _launch_shape(arr.size)
    fn = _template(arr.dtype, _FILL_SRC, "af_fill")
    fn((grid,), (block,), (arr, np.float64(value), np.int64(arr.size)))


def _sum_all(arr: DeviceArray) -> DeviceArray:
    """全元素和，返回 f64 的 0 维数组（两段归约，不依赖原子加）。"""
    src = arr if arr.flags.c_contiguous else ascontiguousarray(arr)
    out = empty((), np.float64)
    n = src.size
    if n == 0:
        _fill(out, 0.0)
        return out
    blocks, block = _launch_shape(n)
    parts = empty((blocks,), np.float64)
    fn = _template(src.dtype, _SUM_PARTIAL_SRC, "af_sum_partial")
    fn((blocks,), (block,), (src, parts, np.int64(n)))
    fin = _Runtime.instance().function(
        ("af_sum_final",), _SUM_FINAL_SRC, "af_sum_final", "af_sum_final"
    )
    fin((1,), (block,), (parts, out, np.int32(blocks)))
    return out


def sqrt(a) -> DeviceArray:
    """逐元素开方；只有浮点数组有意义，整型直接拒绝。"""
    a = asarray(a)
    if a.dtype.kind != "f":
        raise CudaRuntimeError(f"sqrt 只支持浮点数组，收到 {a.dtype}")
    return _ew_unary(a, _SQRT)


def zeros_like(a) -> DeviceArray:
    return zeros(a.shape, a.dtype)


def sum(a, axis=None):
    """全元素和（与 ``numpy.sum`` 的默认轴同义）。

    只做全归约：训练层用它求全局范数。带轴的归约在 IR 层由算子承接，
    容器不重复实现一套。
    """
    if axis is not None:
        raise CudaRuntimeError(f"CUDA 容器的 sum 只支持全归约，收到 axis={axis!r}")
    return _sum_all(asarray(a))


# ----------------------------------------------------------------------
# 流对象
# ----------------------------------------------------------------------


class _Stream:
    """``cp.cuda.Stream`` 的最小替身：承载指针、同步、可切换当前流。

    指针这一项是后端真正用到的（``cublas.set_stream`` 要它）；``synchronize``
    则是**测量**用到的——``benchmarks/run.py`` 的 ``_sync`` 靠它把异步执行切断，
    少了这个方法，CUDA 上每一次计时都会在第一次 ``warmup`` 之前抛
    ``AttributeError``（在 AMD 卡上实跑基准时暴露）。

    ``__enter__`` / ``__exit__`` 与 cupy 的 ``with stream:`` 对齐：后端代码
    只写 ``with stream:``，内核 launch 与拷贝都从流栈取当前流。
    """

    __slots__ = ("ptr",)

    def __init__(self, ptr: int = 0) -> None:
        self.ptr = ptr

    def synchronize(self) -> None:
        _Runtime.instance().stream_synchronize(self.ptr)

    def __enter__(self) -> "_Stream":
        _STREAM_STACK.append(self.ptr)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _STREAM_STACK.pop()
        return False
