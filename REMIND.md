# 天数（Iluvatar CoreX）硬件包 — 工作记录

## 这是什么

AetherFlow 的**独立硬件包**：一个 zip，包含 `manifest.json` 与 `iluvatar_af/` 模块。
用 `aetherflow.backends.hardware.install_hardware(zip)` 安装，之后
`get_backend("iluvatar")` 即可使用。不修改框架代码。

```
aetherflow_iluvatar/
├── iluvatar_af/        # 包内模块（必须带 iluvatar 前缀，见下）
├── build.py            # 生成 dist/iluvatar-<version>.zip
├── REMIND.md
└── dist/               # 构建产物，不入库
```

## 关键设计决定

- **不继承 `CudaBackend`**。继承会把 cupy 容器与 cuDNN 假设一起带过来。
  直接继承 `aetherflow.backends.device_backend.DeviceBackend`，算子的形状检查、
  广播折叠、分块策略全部复用；厂商层只写容器、BLAS、编译、启动。
- **内核模板逐字复用 `aetherflow.backends.cuda.codegen`**，只替换 f16 的三个
  辅助函数（PTX 内联汇编 → clang 原生 `_Float16` + `__builtin_bit_cast`）。
  `iluvatar_af.codegen.__getattr__` 把其余常量/函数全部透传给 cuda 侧，
  保证"模板只有一份"。
- **模块名必须带前缀** `iluvatar_af`：`hardware.build_factory` 会校验模块
  `__file__` 必须位于包目录内，与已安装的同名模块冲突会直接拒绝。
- **编译走子进程**：天数 `nvcc` 是只打印版本号的壳，`libnvrtc` 不保证接受
  ivcore 目标。用天数 Clang：

  ```
  clang++ -x ivcore --cuda-path=/usr/local/corex-5.0.0 -O2 --cubin -o out.cubin src.cu
  ```

  产物 cubin 交给 `cuModuleLoadData`。编译结果按源码 sha1 缓存在
  `~/.cache/aetherflow/ivcore`。
- **模块加载/启动用 driver API**（`cuModuleLoadData` / `cuModuleGetFunction` /
  `cuLaunchKernel`）：CUDA 运行时没有 `cudaModule*` 这组函数。其余（内存、流、
  事件、图捕获）用 runtime API `cuda*`。流句柄两者通用。
- **无 cuDNN**：`CONV_IMPLS` 去掉 `"cudnn"`，`conv_impl="cudnn"` 构造期报错。

## 天数实测环境事实（MR-V100）

| 项 | 值 |
| --- | --- |
| 设备 | Iluvatar MR-V100，32GB，sm=7.1 |
| **warpSize** | **64**（NVIDIA 为 32） |
| maxThreadsPerBlock | 4096 |
| regsPerBlock | 262144 |
| smemPerBlock | 131072 |
| mpCount | 16 |
| 主频 | 1500 MHz |
| 显存位宽 | 2048 bit |
| L2 | 16 MB |
| SDK | `/usr/local/corex-5.0.0`（bin/lib/include） |
| 运行时版本号 | `cudaRuntimeGetVersion()` = 10020（CUDA 10.2） |
| 库 | `libcudart.so.10.2.89`、`libcublas.so.10.2.3.254`、`libcufft`、`libcurand`、`libcusparse`、`libcusolver`、`libnvrtc`、`libcuda.so.1` |
| 缺 | **没有 cudnn** |
| `nvcc` | 假壳，只 `cat` 版本号，不编译 |

warp=64 的影响：现有模板只按 block 内线程索引 + `__syncthreads` 归约，
没有 `__shfl_down_sync`、没有 `threadIdx.x % 32` 这类假设（已 grep 确认），
所以不需要内核分支。**但以后新增任何"一个 warp 内前缀和"的优化都必须
按 `warpSize` 取，不能写 32。**

## 远端验证机

```
ssh -p 11155 ubuntu@39.156.149.45
password: K4QgJolw
```

- Ubuntu 24.04 / Python 3.12.3 / 6 核 / 9GB 内存 / 174GB 磁盘
- `ubuntu` 有 sudo 免密
- 项目：`/home/ubuntu/aetherflow`，venv：`/home/ubuntu/afenv`
- 使用前需：
  ```
  export COREX_PATH=/usr/local/corex-5.0.0
  export PATH=$COREX_PATH/bin:$PATH
  export LD_LIBRARY_PATH=$COREX_PATH/lib:$LD_LIBRARY_PATH
  ```
- 全量测试基线：`Ran 2865 tests` → `OK (skipped=540)`

## 进度

- [x] 硬件包骨架（manifest / 7 个模块）与打包脚本
- [x] 本地：zip 安装 → 注册 → 惰性工厂（无卡时明确报错）
- [x] 真机：`install_hardware` 成功，`get_backend("iluvatar")` 构造成功
      （`Iluvatar MR-V100`，32GB，runtime 10020）
- [x] 真机：最小 conformance（`add_basic` / `sub_basic`）PASS，err=0
- [ ] 真机：全量 111 用例 conformance
- [ ] 真机：框架自带的 GPU 相关测试
- [ ] 性能：`CONV_AUTO_IMPLICIT` 是否需要按天数的实测结果覆盖

## 踩过的坑

1. `-x cuda` 会走上游 clang 的 CUDA 目标，链接阶段缺
   `__ixRegisterFatBinary` / `ixLaunchKernel`（天数把这两个放在自己的 device
   runtime 里）。必须用 `-x ivcore` 且带 `--cuda-path`。
2. PowerShell 会提前展开 `$PATH`，远端命令必须整体包在单引号里（或经
   base64 传入），否则 `export PATH=...:` 会把 PATH 清空。
3. `manifest.json` 的 `aetherflow_api` 是 `"1"`（框架主版本），不是包自己的
   `0.1.0`。
4. cuBLAS 的枚举值与 hipBLAS 不同：`CUBLAS_OP_N=0`（HIP 111）、
   `CUBLAS_GEMM_DEFAULT=-1`（HIP 160）、10.2 的 `cublasGemmEx` 里
   computeType 是 `cudaDataType`（`CUDA_R_32F=0`），不是 11.x 的 68。
   正式符号带 `_v2` 后缀（`cublasCreate_v2`），`blas.py` 里两个都试。
