"""天数（Iluvatar / 智铠 MR）后端插件。

``ctypes`` 是标准库：本模块层面不加载任何第三方包，所以没有天数卡的机器上
``import iluvatar_af`` 仍然可用，只是 ``IluvatarBackend()`` 会带着明确原因失败。
"""

from __future__ import annotations

from . import runtime
from .backend import ILUVATAR_OPS, IluvatarBackend

ILUVATAR_AVAILABLE = runtime.available()

__all__ = ["IluvatarBackend", "ILUVATAR_OPS", "ILUVATAR_AVAILABLE"]
