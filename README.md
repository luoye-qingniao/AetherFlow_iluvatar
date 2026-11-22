# AetherFlow_iluvatar

AetherFlow 的 Iluvatar CoreX（天数智铠 MR 系列）后端插件，让同一套计算图
可以直接跑在国产 GPU 上。

## 特性

- **一份内核语义**：与 AetherFlow 的 CUDA 后端共用模板，只替换 f16 存取等
  设备方言，避免两套实现悄悄分叉
- **零重型依赖**：通过 ctypes 直接调用 CoreX 运行时与 cuBLAS，不需要
  cupy / torch
- **融合注意力**：自写 TCU flash attention 内核，支持 mask、dropout 及其组合
- **明确的边界**：不支持的配置在构造期直接报错，不静默降级到 CPU

## 快速开始

环境要求：AetherFlow 与 Iluvatar CoreX toolkit。

```powershell
py -3.10 build.py
```

把生成的 `dist/iluvatar-0.1.0.zip` 拷到目标机器后安装：

```python
from aetherflow.backends import install_hardware
install_hardware("/path/to/iluvatar-0.1.0.zip", replace=True)
```

## 目录结构

```
iluvatar_af/          后端插件
  backend.py          后端实现，复用 AetherFlow 的算子语义
  codegen.py          内核源码生成（f16 存取、flash attention 模板）
  runtime.py          设备容器与运行时绑定
  memory.py           Arena 与分层分配器
  blas.py             cuBLAS 绑定
  rtc.py              基于 CoreX Clang 的离线编译
  dialect.py          设备方言适配
bench/
  tcu_flash_attn.cu   flash attention 基准，可独立编译运行
build.py              打包脚本
```

## 性能

在 MR-V100 上以 512×512×64、batch×heads = 256 的 f16 配置实测，flash
attention 约 0.6 ms（约 57 TFLOPS；同等规模的裸 mma 上限约 83 TFLOPS）。

## 更多

设计取舍、实测数据，以及开发过程中被否决的方案都记录在
[REMIND.md](REMIND.md)。

## 许可证

[Apache-2.0](LICENSE)
