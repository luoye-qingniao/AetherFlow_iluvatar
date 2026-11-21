"""把 ``iluvatar_af`` 打成可安装的硬件包 zip。

产物结构（与 ``aetherflow/backends/hardware.py`` 的约定一致）::

    iluvatar-0.1.0.zip
    ├── manifest.json
    └── iluvatar_af/
        ├── __init__.py
        ├── backend.py
        ├── blas.py
        ├── codegen.py
        ├── dialect.py
        ├── memory.py
        ├── rtc.py
        └── runtime.py

``files`` 摘要覆盖包里除 ``manifest.json`` 之外的全部文件，安装时会逐个核对，
所以清单必须在文件最终确定之后再生成——这正是"打包"而不是"手写清单"的理由。

用法::

    python build.py [--out dist]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

NAME = "iluvatar"
VERSION = "0.1.0"
FACTORY = "iluvatar_af.backend:IluvatarBackend"
DESCRIPTION = (
    "天数（Iluvatar CoreX / 智铠 MR）后端：ctypes 直连 CoreX 的 CUDA 兼容层，"
    "内核由天数 Clang 以 -x ivcore 编译，卷积不依赖 cuDNN。"
)

ROOT = pathlib.Path(__file__).resolve().parent
PACKAGE = ROOT / "iluvatar_af"


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources() -> dict:
    """包里参与摘要的文件（相对 POSIX 路径 -> 绝对路径）。"""
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        found[path.relative_to(ROOT).as_posix()] = path
    if not found:
        raise SystemExit(f"{PACKAGE} 下没有任何源码")
    return found


def _ops() -> list:
    """若构建机上装了框架，就把能力面写进依赖声明，换版本时能早失败。"""
    try:
        from aetherflow.backends.device_backend import GPU_OPS
    except Exception:  # 构建机没有框架时仍然可以打包
        return []
    return sorted(GPU_OPS)


def manifest() -> dict:
    return {
        "name": NAME,
        "version": VERSION,
        "aetherflow_api": "1",
        "factory": FACTORY,
        "vendor": "Iluvatar CoreX",
        "description": DESCRIPTION,
        "files": {name: _digest(path) for name, path in _sources().items()},
        "requires": {"python": ["numpy"], "ops": _ops()},
    }


def build(out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{NAME}-{VERSION}.zip"
    data = json.dumps(manifest(), ensure_ascii=False, indent=2, sort_keys=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", data)
        for name, path in _sources().items():
            zf.write(path, name)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "dist"))
    args = parser.parse_args(argv)
    target = build(pathlib.Path(args.out))
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
