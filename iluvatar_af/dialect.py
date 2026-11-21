"""天数方言：与 CUDA 同一套 C 语法，只换 f16 存取。

差异只有两处：天数 Clang 不认 PTX 的 ``cvt`` 约束（``asm("cvt.f32.f16 ...")``
会报 invalid output constraint），也用不了 nvcc 的 PTX 内联汇编；f16 改用
clang 原生的 ``_Float16`` 与 ``__builtin_bit_cast``。归约辅助函数与 f32 辅助
函数逐字共用——方言的粒度就是 f16 的这三个 helper，其余都是同一层语法。
"""

from __future__ import annotations

from aetherflow.ir.types import DType
from aetherflow.lower.cstyle import CStyleDialect
from aetherflow.backends.cuda import codegen as _cuda
from . import codegen

__all__ = ["ILUVATAR_DIALECT", "CAST_DIALECT"]

ILUVATAR_DIALECT = CStyleDialect(
    "iluvatar",
    storage={DType.F32: "float", DType.F16: "unsigned short"},
    load={DType.F32: "af_ld_f", DType.F16: "af_ld_h"},
    store={DType.F32: "af_st_f", DType.F16: "af_st_h"},
    atomic={DType.F32: "af_add_f", DType.F16: "af_add_h"},
    helpers={
        DType.F32: _cuda._F32_HELPERS,
        DType.F16: codegen._ILUVATAR_F16_HELPERS,
    },
    reduce_helpers=_cuda._REDUCE,
)

# 与 CUDA 侧的 ``CAST_DIALECT`` 同构：``cast`` 的输入与输出存储宽度不同，两套
# helper 必须同时在源码里。f16 那一套是 clang 版。
CAST_DIALECT = CStyleDialect(
    "iluvatar",
    storage=ILUVATAR_DIALECT.storage,
    load=ILUVATAR_DIALECT.load,
    store=ILUVATAR_DIALECT.store,
    atomic=ILUVATAR_DIALECT.atomic,
    helpers={
        DType.F32: _cuda._F32_HELPERS + codegen._ILUVATAR_F16_HELPERS,
        DType.F16: codegen._ILUVATAR_F16_HELPERS + _cuda._F32_HELPERS,
    },
    reduce_helpers=ILUVATAR_DIALECT.reduce_helpers,
)
