"""天数 Clang 编译（子进程）。

CoreX 5.0 没有可用的运行时编译通道：``nvcc`` 是一个只打印版本号的壳，
``libnvrtc`` 也不保证接受 ivcore 目标。官方要求把同一份 CUDA C++ 源码用
天数 Clang 重新编译，因此这里把源码写到临时文件，调用

    clang++ -x ivcore --cuda-path=<corex> -O2 --cubin -o out.cubin src.cu

产出一段 cubin，再交给 ``runtime.py`` 的 ``cuModuleLoadData`` 装进上下文。
编译产物按源码摘要缓存到磁盘（``~/.cache/aetherflow/ivcore``），进程重启
后不必重复编译——这一点对"启动一次跑很多图"的用法很关键。

编译器按以下顺序确定：

1. ``AETHERFLOW_IVCORE_CLANG``（显式指定，最高优先级）；
2. ``$COREX_PATH/bin/clang++``；
3. ``/usr/local/corex-*`` 或 ``/opt/corex-*`` 下版本号最大的一套；
4. ``PATH`` 里的 ``clang++``（配合 ``$COREX_PATH`` 使用）。

``--cubin`` 与 ``-x ivcore`` 是探针验证过的组合：``-x cuda`` 会走到上游
clang 的 CUDA 目标，链接阶段缺 ``__ixRegisterFatBinary`` / ``ixLaunchKernel``
（天数把这两个符号放在自己的 device runtime 里）。
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
import tempfile

# 编译选项：``-O2`` 与探针一致；``--cubin`` 直接产出可被 cuModuleLoadData
# 加载的 ELF，不需要宿主机链接器。
_BASE_OPTIONS = ("-O2", "--cubin")


class IvcoreCompileError(RuntimeError):
    """天数 Clang 编译失败。消息里带编译器日志的尾部，便于定位。"""


def _corex_root() -> str | None:
    """按环境变量或安装目录找 CoreX 前缀。"""
    explicit = os.environ.get("COREX_PATH")
    if explicit and os.path.isdir(explicit):
        return explicit
    for base in ("/usr/local", "/opt"):
        hits = sorted(glob.glob(os.path.join(base, "corex-*")), reverse=True)
        if hits:
            return hits[0]
    return None


def _discover():
    """返回 ``(clang++, corex 前缀)``；找不到时抛 ``OSError``。"""
    root = _corex_root()
    explicit = os.environ.get("AETHERFLOW_IVCORE_CLANG")
    if explicit:
        if not os.path.exists(explicit):
            raise OSError(f"AETHERFLOW_IVCORE_CLANG 指向的文件不存在：{explicit}")
        return explicit, root
    if root:
        candidate = os.path.join(root, "bin", "clang++")
        if os.path.exists(candidate):
            return candidate, root
    found = shutil.which("clang++") or shutil.which("clang")
    if found:
        return found, root
    raise OSError(
        "找不到天数 Clang：设置 COREX_PATH 或 AETHERFLOW_IVCORE_CLANG，"
        "或把 CoreX 的 bin 目录加入 PATH"
    )


def _cache_dir() -> str:
    root = os.environ.get("AETHERFLOW_IVCORE_CACHE")
    if not root:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        root = os.path.join(base, "aetherflow", "ivcore")
    os.makedirs(root, exist_ok=True)
    return root


def available() -> bool:
    """本机是否具备"源码 -> cubin"的编译能力（不要求有 GPU）。"""
    try:
        _discover()
    except OSError:
        return False
    return True


class Rtc:
    """天数 Clang 的编译器封装；编译器路径在第一次 ``compile`` 时确定。"""

    def __init__(self) -> None:
        self._clang: str | None = None
        self._corex: str | None = None

    def _tools(self):
        if self._clang is None:
            self._clang, self._corex = _discover()
        return self._clang, self._corex

    def compile(self, source: str, name: str, arch: str | None = None) -> bytes:
        """把源码编译成 cubin 字节串。

        ``arch`` 仅为接口对齐保留：ivcore 的目标由 CoreX 的 Clang 驱动决定，
        不需要 ``sm_XX`` 这类架构名。
        """
        digest = hashlib.sha1(source.encode()).hexdigest()
        cached = os.path.join(_cache_dir(), f"{digest}.cubin")
        if os.path.exists(cached):
            with open(cached, "rb") as fh:
                return fh.read()

        clang, corex = self._tools()
        with tempfile.TemporaryDirectory(prefix="af_ivcore_") as tmp:
            src_path = os.path.join(tmp, f"{name}.cu")
            out_path = os.path.join(tmp, f"{name}.cubin")
            with open(src_path, "w", encoding="utf-8") as fh:
                fh.write(source)
            cmd = [clang, "-x", "ivcore"]
            if corex:
                cmd.append(f"--cuda-path={corex}")
            cmd.extend(_BASE_OPTIONS)
            cmd.extend(["-o", out_path, src_path])
            env = dict(os.environ)
            if corex:
                lib = os.path.join(corex, "lib")
                previous = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = lib + (":" + previous if previous else "")
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=600
            )
            if proc.returncode != 0 or not os.path.exists(out_path):
                detail = (proc.stderr or proc.stdout or "").strip()
                raise IvcoreCompileError(
                    f"天数 Clang 编译 {name} 失败（退出码 {proc.returncode}）：\n"
                    f"命令：{' '.join(cmd)}\n{detail[-4000:]}"
                )
            with open(out_path, "rb") as fh:
                data = fh.read()

        # 原子落盘：并发进程可能同时编译同一份源码
        tmp_path = f"{cached}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as fh:
            fh.write(data)
        os.replace(tmp_path, cached)
        return data
