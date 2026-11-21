"""天数（CoreX）cuBLAS 薄绑定（ctypes）。

CoreX 5.0 的 ``libcublas`` 兼容到 CUDA 10.2 的 ABI，所以这里的枚举值按
CUDA 10.2 的头文件取，而不是照抄 HIP 的数字：

- ``CUBLAS_OP_N = 0``（HIP 是 111）；
- ``CUBLAS_GEMM_DEFAULT = -1``（HIP 是 160）；
- 10.2 的 ``cublasGemmEx`` 里 computeType 仍是 ``cudaDataType``，所以
  ``CUBLAS_COMPUTE_32F`` 取 ``CUDA_R_32F = 0``；11.x 的
  ``cublasComputeType_t = 68`` 不适用于这层兼容 ABI。

只绑 float32 / float16 的**列主序** GEMM，行主序通过 ``C^T = B^T A^T``
互换 A/B 得到；f16 存储半精度、按 f32 累加。
"""

from __future__ import annotations

import ctypes
import ctypes.util

# cublasOperation_t：CUBLAS_OP_N = 0、CUBLAS_OP_T = 1
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1

# cudaDataType：CUDA_R_32F = 0、CUDA_R_16F = 2
_CUDA_R_16F = 2
_CUDA_R_32F = 0

# CUDA 10.2 的 cublasGemmEx 用 cudaDataType 作 computeType
_CUBLAS_COMPUTE_32F = _CUDA_R_32F

# cublasGemmAlgo_t：CUBLAS_GEMM_DEFAULT = -1
_CUBLAS_GEMM_DEFAULT = -1

_CANDIDATES = (
    "libcublas.so.10",
    "libcublas.so",
)

_STATUS = {
    0: "CUBLAS_STATUS_SUCCESS",
    1: "CUBLAS_STATUS_NOT_INITIALIZED",
    2: "CUBLAS_STATUS_ALLOC_FAILED",
    3: "CUBLAS_STATUS_INVALID_VALUE",
    4: "CUBLAS_STATUS_MAPPING_ERROR",
    5: "CUBLAS_STATUS_EXECUTION_FAILED",
    6: "CUBLAS_STATUS_INTERNAL_ERROR",
    7: "CUBLAS_STATUS_NOT_SUPPORTED",
    8: "CUBLAS_STATUS_ARCH_MISMATCH",
    9: "CUBLAS_STATUS_HANDLE_IS_NULLPTR",
}


class CublasError(RuntimeError):
    """cudaBLAS 返回非 SUCCESS 状态码。"""


def _load_library() -> ctypes.CDLL:
    names = []
    found = ctypes.util.find_library("cublas")
    if found:
        names.append(found)
    names.extend(_CANDIDATES)
    tried = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            tried.append(f"{name}: {exc}")
    raise OSError("找不到 cuBLAS 动态库，尝试过：\n  " + "\n  ".join(tried))


def _sym(lib: ctypes.CDLL, name: str):
    """取函数符号。

    CUDA 10.x 的 libcublas 里正式符号带 ``_v2`` 后缀（``cublasCreate_v2``
    ``cublasSgemm_v2``），旧名在头文件里只是宏。这里两个都试，命中后以
    旧名回挂，后面的 ``argtypes`` 声明不必逐个区分。
    """
    for candidate in (f"{name}_v2", name):
        try:
            return getattr(lib, candidate)
        except AttributeError:
            continue
    raise OSError(f"libcublas 里找不到符号 {name}")


def _ptr(array) -> ctypes.c_void_p:
    return ctypes.c_void_p(array.data.ptr)


class Cublas:
    """进程内复用的 cuBLAS handle；stream 绑定做缓存，避免重复下发。"""

    def __init__(self) -> None:
        self._lib = _load_library()
        self._declare()
        handle = ctypes.c_void_p()
        self._check(self._lib.cublasCreate(ctypes.byref(handle)), "cublasCreate")
        self._handle = handle
        self._stream = None

    def _declare(self) -> None:
        lib = self._lib
        vp = ctypes.c_void_p
        fp = ctypes.POINTER(ctypes.c_float)
        ci = ctypes.c_int
        cll = ctypes.c_longlong

        for name in (
            "cublasCreate", "cublasDestroy", "cublasSetStream", "cublasSgemm",
            "cublasSgemmStridedBatched", "cublasGemmEx",
            "cublasGemmStridedBatchedEx",
        ):
            setattr(lib, name, _sym(lib, name))

        lib.cublasCreate.argtypes = [ctypes.POINTER(vp)]
        lib.cublasCreate.restype = ci
        lib.cublasDestroy.argtypes = [vp]
        lib.cublasDestroy.restype = ci
        lib.cublasSetStream.argtypes = [vp, vp]
        lib.cublasSetStream.restype = ci

        lib.cublasSgemm.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, vp, ci, fp, vp, ci,
        ]
        lib.cublasSgemm.restype = ci

        lib.cublasSgemmStridedBatched.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, cll, vp, ci, cll, fp, vp, ci, cll, ci,
        ]
        lib.cublasSgemmStridedBatched.restype = ci

        # cublasGemmEx：alpha/beta 是 void*（指向 compute type 的标量），
        # A/B/C 的类型由 cudaDataType 参数给出。
        lib.cublasGemmEx.argtypes = [
            vp, ci, ci, ci, ci, ci, fp, vp, ci, ci, vp, ci, ci, fp, vp, ci, ci, ci, ci,
        ]
        lib.cublasGemmEx.restype = ci

        # cublasStride 是 int64_t；其余与 GemmEx 同序，只是多了三个 stride。
        lib.cublasGemmStridedBatchedEx.argtypes = [
            vp, ci, ci, ci, ci, ci, fp,
            vp, ci, ci, cll,
            vp, ci, ci, cll,
            fp, vp, ci, ci, cll,
            ci, ci, ci,
        ]
        lib.cublasGemmStridedBatchedEx.restype = ci

    @staticmethod
    def _check(status: int, what: str) -> None:
        if status != 0:
            raise CublasError(
                f"{what} 失败：{_STATUS.get(status, status)}（code={status}）"
            )

    def set_stream(self, ptr) -> None:
        ptr = int(ptr or 0)
        if ptr == self._stream:
            return
        self._check(
            self._lib.cublasSetStream(self._handle, ctypes.c_void_p(ptr)),
            "cublasSetStream",
        )
        self._stream = ptr

    def sgemm(self, a, b, out) -> None:
        """行主序 ``out[M, N] = a[M, K] @ b[K, N]``。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasSgemm(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha), _ptr(b), n,
                _ptr(a), k,
                ctypes.byref(beta), _ptr(out), n,
            ),
            "cublasSgemm",
        )

    def sgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasSgemmStridedBatched(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha), _ptr(b), n, stride_b,
                _ptr(a), k, stride_a,
                ctypes.byref(beta), _ptr(out), n, stride_c,
                batch,
            ),
            "cublasSgemmStridedBatched",
        )

    def hgemm(self, a, b, out) -> None:
        """行主序 f16 GEMM：存储半精度，内部按 CUBLAS_COMPUTE_32F 累加。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmEx(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _CUDA_R_16F, n,
                _ptr(a), _CUDA_R_16F, k,
                ctypes.byref(beta),
                _ptr(out), _CUDA_R_16F, n,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx",
        )

    def hgemm_batched(self, a, b, out, batch: int,
                      stride_a: int, stride_b: int, stride_c: int) -> None:
        """行主序 f16 批量 GEMM，stride 以元素为单位；0 表示整批共用一份数据。"""
        m, k = int(a.shape[-2]), int(a.shape[-1])
        n = int(out.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmStridedBatchedEx(
                self._handle, _CUBLAS_OP_N, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), _CUDA_R_16F, n, stride_b,
                _ptr(a), _CUDA_R_16F, k, stride_a,
                ctypes.byref(beta),
                _ptr(out), _CUDA_R_16F, n, stride_c,
                batch,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmStridedBatchedEx",
        )

    def _batched_nt(self, a, b, out, batch: int, stride_a: int, stride_b: int,
                    stride_c: int, rtype: int) -> None:
        """行主序 ``out[B, M, N] = a[B, M, K] @ b[B, N, K]^T``。

        attention 的 QK^T 就是这一形态：Q 与 K 都是 ``(T, D)`` 行主序，直接
        把 K 当"转置的右操作数"传即可，不需要先物化 K^T。列主序视角下
        ``C_col(N, M) = b(N, K) * a(M, K)^T``，所以 A 位传 b、OP_T，B 位传
        a、OP_N，ld 都是 K。
        """
        m, n = int(out.shape[-2]), int(out.shape[-1])
        k = int(a.shape[-1])
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._check(
            self._lib.cublasGemmStridedBatchedEx(
                self._handle, _CUBLAS_OP_T, _CUBLAS_OP_N,
                n, m, k,
                ctypes.byref(alpha),
                _ptr(b), rtype, k, stride_b,
                _ptr(a), rtype, k, stride_a,
                ctypes.byref(beta),
                _ptr(out), rtype, n, stride_c,
                batch,
                _CUBLAS_COMPUTE_32F, _CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmStridedBatchedEx",
        )

    def hgemm_batched_nt(self, a, b, out, batch: int,
                         stride_a: int, stride_b: int, stride_c: int) -> None:
        """f16 批量 GEMM，右操作数转置：``out = a @ b^T``。"""
        self._batched_nt(a, b, out, batch, stride_a, stride_b, stride_c, _CUDA_R_16F)

    def sgemm_batched_nt(self, a, b, out, batch: int,
                         stride_a: int, stride_b: int, stride_c: int) -> None:
        """f32 批量 GEMM，右操作数转置：``out = a @ b^T``。"""
        self._batched_nt(a, b, out, batch, stride_a, stride_b, stride_c, _CUDA_R_32F)

    def close(self) -> None:
        if self._handle is not None:
            self._lib.cublasDestroy(self._handle)
            self._handle = None
            self._stream = None
