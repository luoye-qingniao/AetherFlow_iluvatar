# AetherFlow_iluvatar

AetherFlow 的 Iluvatar CoreX（智铠 MR / MR-V100）硬件后端适配包。

## 内容

- `iluvatar_af/` — 后端插件。ctypes 直连 CoreX 的 CUDA 兼容层
  （`libcudart` / `libcuda` / `libcublas`），内核由天数 Clang 以 `-x ivcore`
  离线编译；不依赖 cupy / torch。
  - `backend.py` — `IluvatarBackend`，复用 `DeviceBackend` 的算子语义
  - `codegen.py` — 内核源码生成（f16 存取、TCU flash attention 模板）
  - `runtime.py` / `memory.py` / `blas.py` / `rtc.py` / `dialect.py`
- `bench/tcu_flash_attn.cu` — 自写 TCU flash attention 基准，独立可编译运行
- `build.py` — 打包 `dist/iluvatar-0.1.0.zip`
- `REMIND.md` — **开发记录**：实测数据、结构定案、已否决方向、踩过的坑
- `LICENSE` — Apache-2.0

## 构建与安装

```powershell
py -3.10 build.py
```

真机上安装（需要 CoreX toolkit，验证环境为 `/usr/local/corex-5.0.0`）：

```python
from aetherflow.backends import install_hardware
install_hardware("/path/to/iluvatar-0.1.0.zip", replace=True)
```

准入判据是全量 `run_conformance` 通过：104 executed / 7 refused /
0 failed，68/68 ops（MR-V100，2026-11-22）。

## 性能（512×512×64，bh=256，f16）

自写 flash attention 内核：plain 0.5996 ms、mask 0.8481、dropout 0.8812、
mask+dropout 1.2721（56~57 TFLOPS，裸 mma 上限 83.4 TFLOPS）。
