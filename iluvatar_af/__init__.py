"""天数（Iluvatar / 智铠 MR）后端插件。

``ctypes`` 是标准库：本模块层面不加载任何第三方包，所以没有天数卡的机器上
``import iluvatar_af`` 仍然可用，只是 ``IluvatarBackend()`` 会带着明确原因失败。
"""

from __future__ import annotations

from . import runtime
from .backend import ILUVATAR_OPS, IluvatarBackend


def _is_device_array(value) -> bool:
    return isinstance(value, runtime.DeviceArray)


def _register_training() -> None:
    """把 ``DeviceArray`` 接入训练层的数组分派。

    设备态训练、梯度裁剪、EMA、存档都靠 ``train/_arrays.array_module`` 判断
    "这个数组在主机还是显存"。框架只硬编码了 numpy / cupy / HIP 三个来源，
    硬件包必须在导入时自己注册，否则优化器会把设备数组当成 numpy 去索引
    （``DeviceArray`` 不支持整数索引降维）。

    老框架没有这个注册点时静默跳过：主机态训练仍然可用，只是设备态的优化器
    会给出框架原本的 ``TypeError``。
    """
    try:
        from aetherflow.train._arrays import register_array_module
    except ImportError:
        return
    register_array_module(_is_device_array, runtime)


_register_training()

ILUVATAR_AVAILABLE = runtime.available()

__all__ = ["IluvatarBackend", "ILUVATAR_OPS", "ILUVATAR_AVAILABLE"]
