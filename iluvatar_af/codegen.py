"""天数 C 源码生成：复用 CUDA 的模板，只替换 f16 的存取实现。

模板本身（逐元素 / 广播 / softmax / attention / 卷积 / 池化 / 扫描……）是
纯 C++ 加 ``__global__`` 语义，天数 Clang 完全接受。唯一不通用的是 f16 的
PTX 内联汇编：``asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h))`` 在 clang 上
报 invalid output constraint。

因此这里只做一件事：换掉 PRELUDE 里的三个 f16 辅助函数，其余常量与构造
函数原样透传（``__getattr__``）。两种后端因此**共用一份内核语义**——模板
改了，天数自动跟上。
"""

from __future__ import annotations

from aetherflow.backends.cuda import codegen as _cuda

# clang 的 _Float16 与 numpy 的 float16 是同一套 IEEE 754 半精度，逐位布局
# 一致，所以 unsigned short 与 _Float16 之间用 __builtin_bit_cast 互转即可，
# 不需要 include 任何头文件。
#
# 16 位原子加仍然用 CAS 循环，但**不能**直接调 `atomicCAS(unsigned short*)`：
# 天数 device runtime 只提供 int / unsigned int / unsigned long long 三个重载
# （实测报 "no matching function for call to 'atomicCAS'"）。这里与 CUDA sm<60
# 走同一条路：把 CAS 加宽到对齐的 32 位字，只替换目标半字。算法逐字照搬
# `aetherflow.backends.cuda.codegen` 的 `af_cas_h` 非 16 位原子分支。
_ILUVATAR_F16_HELPERS = r"""
__device__ __forceinline__ float af_ld_h(unsigned short h) {
    return (float)__builtin_bit_cast(_Float16, h);
}
__device__ __forceinline__ unsigned short af_st_h(float v) {
    return __builtin_bit_cast(unsigned short, (_Float16)v);
}
__device__ __forceinline__ unsigned short af_cas_h(unsigned short* p,
                                                   unsigned short assumed,
                                                   unsigned short v) {
    unsigned long long addr = (unsigned long long)p;
    unsigned int* base = (unsigned int*)(addr & ~2ULL);
    unsigned int shift = (addr & 2ULL) ? 16u : 0u;
    unsigned int mask = (addr & 2ULL) ? 0xFFFF0000u : 0x0000FFFFu;
    unsigned int other = *base & ~mask;
    for (;;) {
        unsigned int assumed32 = other | (((unsigned int)assumed) << shift);
        unsigned int desired = other | (((unsigned int)v) << shift);
        unsigned int prev = atomicCAS(base, assumed32, desired);
        if (prev == assumed32) return assumed;
        unsigned short got = (unsigned short)((prev & mask) >> shift);
        if (got != assumed) return got;
        other = prev & ~mask;
    }
}
__device__ __forceinline__ float af_add_h(unsigned short* p, float v) {
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        old = af_cas_h(p, assumed, af_st_h(af_ld_h(assumed) + v));
    } while (assumed != old);
    return af_ld_h(old);
}
"""

__all__ = ["render", "cast_fn"]


def render(template: str, dtype) -> str:
    """展开模板；f16 的读写与原子加换成本方言的实现。"""
    # 延迟导入：dialect 依赖 cuda.codegen，顶层导入会成环。
    from .dialect import ILUVATAR_DIALECT

    return _cuda.render(
        template, dtype, f16_helpers=_ILUVATAR_F16_HELPERS,
        dialect=ILUVATAR_DIALECT,
    )


def cast_fn(src, dst) -> str:
    """``f32 <-> f16`` 的转换源码（L2 描述与 CUDA 逐字共用）。"""
    # 延迟导入：dialect 顶层就 import 了本模块。
    from .dialect import CAST_DIALECT

    return _cuda.cast_fn(src, dst, dialect=CAST_DIALECT)


def __getattr__(name: str):
    """未定义的常量/函数一律来自 CUDA 侧的模板库（单一事实来源）。"""
    return getattr(_cuda, name)
