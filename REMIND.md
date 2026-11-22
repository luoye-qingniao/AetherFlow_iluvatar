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

### 旧机（2026-11-21，已停）

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

### 新机（2026-11-22，端口单独下发，机器随用随换）

```
ssh -p 11161 ubuntu@39.156.149.45
password: orh0GGQ7
```

- 新机是**裸镜像 + 只装了驱动**，与旧机不同，不能直接跑包：
  - `/usr/local/corex-5.0.0` 只有 18 MB：`bin`（ixsmi / logtools /
    uninstaller）、`lib` 仅 `libcuda.so*` + `libixml.so`；**没有 cudart、
    cublas、clang++、include**。
  - 症状：硬件包装得上、`get_backend("iluvatar")` 报
    `OSError: 找不到天数 CUDA 运行时动态库（libcudart.so.10.2 / .10 / .so）`。
  - 驱动安装包留在 `/opt/corex-driver-linux64-5.0.0_x86_64.run`（18 MB）。
- **缺的是 toolkit，必须单独装**（1.70 GB，实测下载 2.5 MB/s，约 12 分钟）：
  ```
  wget https://file-server-software-infra-sh.oss-cn-shanghai.aliyuncs.com/5.0.0/x86_64/sdk/corex-toolkit-linux64-5.0.0_x86_64_10.2.run
  ```
  这个 run 是 Makeself 包，直接 `sh xxx.run` 会交互。**非交互安装的正确姿势**
  （2026-11-22 已在本机跑通）：
  ```
  sudo sh corex-toolkit.run --noexec --nox11 --target /tmp/ci   # 只解包，5.5 GB
  sudo /tmp/ci/corex-installer --silent --toolkit --skip-spacecheck \
       --install-path /usr/local/corex-5.0.0
  ```
  `corex-installer` 的选项：`--silent` / `--driver` / `--toolbox` / `--toolkit` /
  `--install-path` / `--skip-spacecheck`。不要选 `--driver`——驱动已装好，
  重装驱动会重新编译 kmd 并可能重启服务。
  装完得到 5.5 GB：clang 22.1.0、`libcudart.so.10.2.89`、`libcublas.so.10.2.3.254`、
  `include/crt/iluvatar_mma.h`，`/usr/local/corex` 软链自动建好。
- 重建到"可跑包"的前置步骤（2026-11-22 全部在本机验证）：
  1. `python3 -m venv /home/ubuntu/afenv`（镜像自带 pip 24.0，无需 apt）
  2. `afenv/bin/pip install -U numpy -i https://pypi.tuna.tsinghua.edu.cn/simple`
     → numpy 2.5.5
  3. 上传主框架源码（本地 `tar -czf`，排除 `.git` / `__pycache__` / `build`）
     解到 `/home/ubuntu/aetherflow`，用 `PYTHONPATH=/home/ubuntu/aetherflow`
     即可 `import aetherflow`（版本 0.1.0，无需 pip install）
  4. 上传 `dist/iluvatar-0.1.0.zip` →
     `install_hardware("/home/ubuntu/iluvatar-0.1.0.zip", replace=True)`
     （新机上验证过：`installed: iluvatar 0.1.0`、`registry: ['iluvatar']`）
  5. 装 toolkit → `get_backend("iluvatar")` 成功：MR-V100 / 32 GB / SM 16
  6. 全量 conformance：**104 executed / 7 refused / 0 failed，68/68 ops，15.9s，
     ok=True**——与旧机基线一致，环境等价
- 机器 `ixsmi` 显示：`Iluvatar MR-V100`，32768 MiB，1500 MHz，driver 5.0.0，
  CUDA 10.2，`/dev/iluvatar0` 存在
- 注意：`tar` 解出来的源码 + venv 只有几百 MB；真正的大件是
  `/usr/local/corex-5.0.0`（5.5 GB）。换机重建的主要成本就是那 1.7 GB 下载。

## 进度

- [x] 硬件包骨架（manifest / 7 个模块）与打包脚本
- [x] 本地：zip 安装 → 注册 → 惰性工厂（无卡时明确报错）
- [x] 真机：`install_hardware` 成功，`get_backend("iluvatar")` 构造成功
      （`Iluvatar MR-V100`，32GB，runtime 10020）
- [x] 真机：最小 conformance（`add_basic` / `sub_basic`）PASS，err=0
- [x] 真机：**全量 conformance PASS**（104 executed / 7 refused / 0 failed，
      68/68 算子覆盖，耗时 15.7s）
- [x] 真机：f16 全链路（attention 含 dropout、逐元素+归约、matmul 走
      `cublasGemmEx`、`embedding_backward` 的 f16 原子加）
- [x] 真机：`memory_mode="tiered"`、`graph_mode="auto"`、`capture`（matmul 重放 /
      常量广播 / slice 视图）全 PASS
- [x] 真机：设备态训练 PASS（SGD / Adam 轨迹与主机态逐位对齐、
      `clip_global_norm`、`verify_device_state`）
- [x] 本地框架测试：`tests/test_train.py` + `tests/test_conformance.py`
      → 59 passed, 4 skipped
- [x] 性能第一轮：内核句柄缓存 + 设备内存池 + launcher 热路径
      （见"性能优化（2026-11-21）"一节）
- [x] 性能实测：`CONV_IMPLS` 的大规模形状对比（tiled/grouped 慢 23×，不推荐）
- [x] attention 内核重写（第一轮）：天数包内覆盖 `attention_source`，子组协作 +
      成对读取，512×512×64 从 18.1 ms（123 GFLOPS）降到 4.2 ms（509 GFLOPS）
- [x] attention GEMM 物化（第二轮，当前实现）：`_op_attention` 改为
      QK^T → softmax → PV → rescale，512×512×64 降到 **1.394 ms /
      24653 GFLOPS**（18.1 ms 的 13×，已达 522 GB/s 硬下限的 87.5%），
      对拍 7 项 + 全量 conformance 仍 PASS（见"GEMM 物化 attention"一节）
- [x] TCU flash attention（自写内核，`bench/tcu_flash_attn.cu`）：NW=16 + 每轮
      64 key 合并归约的 attn64 达 **0.6737 ms / 51011 GFLOPS**，为物化基线的
      2.07×（attn32 基线 0.8284 ms）；未接入包内，待验证 Δ 平移 +
      mask/dropout 后再接（见"TCU flash attention"一节）

## 训练层接入（框架侧唯一改动）

`aetherflow/train/_arrays.py` 的 `array_module` 原本硬编码 numpy / cupy / HIP，
第三方容器会落到 numpy 分支（对 `DeviceArray` 做整数索引 → TypeError）。
新增 7 行的通用注册点：

```python
register_array_module(matches, module)   # 表在 array_module 里于 numpy 兜底前查
```

`iluvatar_af/__init__.py` 在导入时注册 `(isinstance(v, DeviceArray), runtime)`。
老框架没有这个函数时静默跳过（主机态训练仍可用）。HIP / CUDA 行为不变，
`test_train` 59 项全过。

## 验证结果（2026-11-21）

```
conformance: 104 executed / 7 refused / 0 failed, 68/68 ops, 15.7s
[PASS] f16 attention plain / dropout
[PASS] f16 elementwise+reduce / matmul (cublasGemmEx)
[PASS] memory_mode=tiered / graph_mode=auto
[PASS] capture matmul replay / constant broadcast / slice view
[PASS] SGD device_state trajectory / Adam device_state trajectory
[PASS] clip_global_norm on device
```

## 性能优化（2026-11-21）

方法：先端到端、再纯内核、再 launch/图捕获/训练步，最后 cProfile 定位。
测量口径统一为 `execute_device()` + `DeviceArray` 直接喂入，排除
H2D/D2H——`execute()` 的数字被 PCIe 主导，看不出后端开销。

cProfile 在前四轮里给出两个真正的热点：

| 热点 | 单次代价 | 成因 |
| --- | --- | --- |
| `runtime.free` | ~35 µs/节点 | `cudaFree` 在天数驱动上是隐式同步点，逐节点"分配→执行→回收"= 每节点一次全设备同步 |
| `_kernel` 重建源码 | ~30 µs/节点 | key 已含 dtype/epilogue/steps/vec 等全部影响源码的因素，却每次仍遍历表达式树重新生成 L2→C |

改了三处：

1. **内核句柄缓存**（[backend.py](file:///g:/Cloumi/aetherflow_iluvatar/iluvatar_af/backend.py)）：
   `_kernel(key, source, func)` 先查 `(key, func)` → 函数句柄表，命中直接返回，
   `source` 支持传"返回源码的函数"以便真正惰性。key 由框架保证唯一标识内核身份，
   与 `cuda/backend.py::_kernel` 同一约定——不会串味（已用 add/add128/mlp 抽查，
   误差 0.0 / 0.0 / 1.2e-03）。
2. **设备内存池**（[runtime.py](file:///g:/Cloumi/aetherflow_iluvatar/iluvatar_af/runtime.py)）：
   按 512 B 对齐分桶缓存空闲块，`free(ptr, nbytes)` 有尺寸就归还池、无尺寸仍直释
   （arena/tiered 块保持原路径）；`_capturing` 期间不经过池。上限
   `AETHERFLOW_IVCORE_POOL_MB`（默认 2048，`0`=关闭），另有 `empty_cache()` /
   `pool_stats()`。图里 8 个节点的中间张量因此只申请一次。
3. **launcher 热路径**：参数打包按 `type(value)` 内联（`DeviceArray`→`c_void_p`、
   `np.float32/64/uint64` 分支），其余回退 `_pack`；`_template` 加模块级缓存，
   key 为 `(name, np.dtype(dtype).str, template)`。

实测（MR-V100，`execute_device`，µs）：

| 场景 | 优化前 | 优化后 |
| --- | --- | --- |
| relu64 eager | 117 | 32.7 |
| relu64 graph-auto | 158 | 31.7 |
| 1 × relu 1M | 124 | 39.3 |
| 2 × relu 1M | 200 | 72.1 |
| 4 × relu 1M | 374 | 120.0 |
| 8 × relu 1M | 694 | 184.2 |
| mlp eager | 237 | 103.9 |
| mlp graph-auto | 102 | 106.9 |
| train step 128×64 graph=off | 2226 | 1803 |
| train step 128×64 graph=auto | 1476 | 1322 |

图捕获已无优势（mlp 102→107）：省下来的正是池与句柄缓存已经消掉的那部分
CPU 开销，且 capture 本身有额外分支。池在基准结束时 38.9 MB / 21 块，
`empty_cache()` 后全部归还。

正确性没有放松：优化后真机全量 conformance **104 executed / 7 refused /
0 failed，68/68 ops，PASS**；本地 `tests/test_train.py tests/test_conformance.py`
59 passed, 4 skipped。

### 算力扫描（决定"下一步优化什么"）

| 项 | 形状 | 耗时 | 算力 |
| --- | --- | --- | --- |
| matmul f16 | 2048³ | 0.232 ms | 74 TFLOPS |
| matmul f16 | 4096³ | 1.413 ms | 97 TFLOPS |
| attention f16 | 4×8×512×64 | 18.1 ms | 123 GFLOPS |
| attention f16 | 16×16×512×64 | 137.9 ms | 133 GFLOPS |
| attention f16 | 2×2×2048×64 | 40.3 ms | 103 GFLOPS |
| attention f16 | 4×8×512×128 | 41.6 ms | 131 GFLOPS |

结论：

- **f16 matmul 已接近上限**（走 `cublasGemmEx`），不再动。
- **attention 是唯一的大头**，且与形状无关地停在 ~123 GFLOPS：共享模板
  `ATTENTION_SRC` 是"一行一个 block、每线程串行扫 D 维算一个 key 的 dot"，
  K 的读取步长 128 B、完全不合并；每 chunk 还有两次 `block_reduce`。
  这是 CUDA 模板的固有结构，不是天数特有缺陷。两轮优化后最终改为
  **GEMM 物化**（QK^T → softmax → PV → rescale），512×512×64 从 18.1 ms
  （123 GFLOPS）降到 **1.394 ms（24653 GFLOPS，13×）**，见「GEMM 物化
  attention」一节；中间的融合内核（4.22 ms / 509 GFLOPS）作为历史保留。
- **conv 的 tiled/grouped 实现慢 23×**（f32 8×64×112² 3×3：auto/im2col ≈5.55 ms，
  implicit 18.3 ms，tiled/grouped 129 ms），`auto` 不选它们是对的；
  `CONV_AUTO_IMPLICIT=False` 保持。小通道 C=3 时 `auto` 选了 tiled（0.827 ms）
  而 im2col 只要 0.515 ms，属已知策略缺陷，收益小、暂不改。
- **未做 Adam 融合**（26 次逐元素 launch → 2 次，训练步还能再省约 40%）：
  会改变 Adam 的数值轨迹，破坏"与主机态逐位对齐"的验收基线。风险大于收益，
  除非另建一套宽松的对齐判据。

### GEMM 物化 attention（2026-11-21，当前实现）

上一节的融合内核把 QK/PV 都手写在同一个 kernel 里，最终停在 4.22 ms /
509 GFLOPS。它已经不是带宽或 barrier 受限，而是**指令发射受限**：每个分数
需要 1 次 load + 1 次 cvt + 1 次 FMA，标量路径在 MR-V100 上封顶约 0.5 TFLOPS。
同一张卡 cuBLAS 的 f16 走 tensor core，实测 97 TFLOPS——差 190 倍。

所以改成 **materialized**（`IluvatarBackend._op_attention` 覆盖，结构级改动）：

| # | 步骤 | 实现 | 代价（bh=256, S=T=512, D=E=64, f16） |
| --- | --- | --- | --- |
| 1 | `P = QK^T` | `cublasGemmStridedBatchedEx` | 0.398 ms / 43147 GFLOPS / 420 GB/s |
| 2 | `P = softmax(P*scale + mask) [* drop]`，同时写 `inv[r]=1/l` | 自写内核，就地 | 0.549 ms / **489 GB/s** |
| 3 | `O = PV` | `cublasGemmStridedBatchedEx` | 0.418 ms / 41118 GFLOPS / 400 GB/s |
| 4 | `O *= inv[r]` | 自写内核 `af_attention_rescale` | 0.088 ms / 381 GB/s |
| | **合计** | | **1.394 ms / 24653 GFLOPS** |

首版（softmax 三遍、rescale 每元素 64 位除法）2.820 ms，上一版 1.573 ms，
当前 1.394 ms；相对融合内核的 4.22 ms 是 3.0×，相对原共享模板的 18.1 ms
是 13×。

**cuBLAS 的 NT 技巧**：框架里三个张量都是行主序 `[M,N] = a[M,K] @ b[N,K]^T`。
列主序下这等价于 `C^T = B^T A^T`，所以调用是
`cublasGemmStridedBatchedEx(OP_T, OP_N, n, m, k, A=b, lda=k, B=a, ldb=k,
C, ldc=n)`——OP 与 A/B 交换位置，其余原样。f32 走 `Sgemm *`，f16 走
`cublasGemmEx` + `CUDA_R_32F` 累加（与 `blas.py` 既有条目同一约定）。

**归一化挪到输出**：`(p/l) @ V == (p @ V)/l`，softmax 只写未归一化的 p，
每行顺手写一个 `inv[r] = 1/l`，最后在 `bh*S*E`（比分数矩阵小 `T/E` 倍）上
乘一次。省掉整整一遍 134 MB 的读写，同时让 dropout 的写法更干净（只乘分子）。

**softmax 内核的三版**（同一个 134 MB 原地读写，都是 32 线程子组一行、
不用 smem，行靠 L1/L2 二次命中）：

| 版本 | 512×512 | 2048×2048 | 说明 |
| --- | --- | --- | --- |
| 早期 smem 版 | 130 GB/s | 31 GB/s | `smem[B][T]` 中转，T=2048 时 smem 分块退化 |
| warp-per-row 两遍 | 390 GB/s | 266 GB/s | 第一遍 max、第二遍 exp+存；两遍都要回 HBM |
| **寄存器驻留单遍** | **489 GB/s** | 266 GB/s（T=2048 仍用两遍） | `AF_NP_REG` 见下 |

`AF_NP_REG = T/64`：每个 lane 持有 `T/64` 个 32 位字（f16 成对），第一遍读入
`w[]` 同时求出 max，第二遍直接从寄存器解包 exp 并成对写回，HBM 只读一次。
上限 **16**（T ≤ 1024）：T=2048 时 `w[]`/`mw[]` 各 32 字溢出到 local memory，
实测 0.326 ms / 206 GB/s，反而慢于两遍版的 0.245 ms / 274 GB/s。
`np_reg` 进 kernel key，两种版本各自缓存。

**rescale 的坑**：旧版按元素启动，每元素一次 `(i*2)/E` 的 64 位整数除法，
只有 249 GB/s。改成"一个 32 线程子组一行、行号即 `inv` 下标"后，行内偏移
全部落在 32 位常量里，0.135 → **0.079–0.088 ms（381–423 GB/s）**。

**带宽账（硬下限）**：`bh=256, S=T=512, D=E=64` 总流量 637 MB
（QK 写 134 + f16 读 33.5；softmax 读 134 + 写 134；PV 读 134 + V 16.8 +
写 16.8；rescale 33.5）。实测带宽上限（见下表）取 **522 GB/s**，
下限 ≈ **1.22 ms**。当前 1.394 ms = 下限的 **87.5%**；softmax 自身
489/522 = 94%。

带宽上限用 134 MB 行数据探针实测（f16，`copy32` 每 lane 32 位成对拷贝）：

| 访问形态 | 带宽 |
| --- | --- |
| `copy32`（32 位）T=512 / T=2048 | **522 / 427 GB/s** |
| `copy128`（16 B 向量） | 402 / 353 GB/s |
| 纯 f16↔f32 转换 | 369 GB/s |
| 加 `expf` | 368 GB/s |
| 两遍（max 预扫 + 第二遍） | 357 GB/s |
| 子组一行 + 32 位读两 f16 + 无 smem（单遍） | **464 GB/s** |

结论：**4 字节访问是最优点**，向量化在 MR-V100 上反而更差；这也是所有内核
统一用 32 位成对存取的理由。（`rt.copy_async(..., kind=1, ...)` 是 H2D，
不是 D2D，用它测带宽无效。）

**其他形状**（同版本）：

| 形状（bh, S, T, D, E） | 耗时 | 算力 |
| --- | --- | --- |
| 256, 512, 512, 64, 64 | 1.394 ms | 24653 GFLOPS |
| 256, 512, 512, 128, 128 | 1.546 ms | **44451 GFLOPS** |
| 32, 512, 512, 64, 64 | 0.250 ms | 17167 GFLOPS |
| 4, 2048, 2048, 64, 64 | 0.479 ms | 17922 GFLOPS |

小批（bh=32）与长序列（2048²）掉到 17–18 TFLOPS，原因不同：前者是 batched
GEMM 批次太小、启动与尾块占比上升；后者是 softmax 退回两遍（266 GB/s），
分数矩阵 4×2048²×2 B 也让 GEMM 的有效读写效率下降。

**显存阈值**：物化路径要一块 `bh*S*T*itemsize` 的分数矩阵
（`_ATTENTION_GEMM_MAX_BYTES = 256 MiB`，softmax 就地做，不再额外要一份）。
超过阈值或 dtype 不是 f32/f16（如 bf16）时退回基类的分块 online-softmax 内核，
显存从 O(bh*S*T) 降到 O(bh*T)。

**已否决、不要重试**：

1. **自写 tensor-core flash**：~~ivcore11 没有可用的 f16 tensor core API~~
   ——**这条结论已被推翻，见本节的「tensor core 侦察（第二轮）」**。第一轮
   的失败原因是头文件选错：`#include <mma.h>` 命中的是
   `IX/ixrt-ixsdk/mma.h`（15 行空壳），而 f16 特化在
   `<crt/iluvatar_mma.h>` → `IX/crt-ixsdk/iluvatar_mma.h/.hpp` 里。
2. **按 bh 切片做 L2 分片**：bh=32 的单片只要 0.250 ms，看起来 ×8 还能更快，
   但小批 GEMM 效率下降 + 4 次启动开销把收益全部吃掉，实测 2.0 ms。
3. **用 `|q|*max|k|` 上界替代每行 max**：省掉 max 预扫，但 Cauchy-Schwarz 上界
   太松，f16 下 `exp` 会下溢，数值不安全。
4. **T=2048 也开 `AF_NP_REG=32`**：见上，溢出 local memory 反而更慢。

**正确性**：对拍 7 项全 PASS（f16 plain 1.620e-03、mask(1,1,S,T) 1.173e-03、
mask(2,2,S,T) 1.094e-03、cross-head 8.646e-04、f32 plain 7.153e-07、
f32 mask 5.960e-07、dropout 逐位 0.0009765625 = 1/1024 量化步长）；
全量 conformance **104 executed / 7 refused / 0 failed，PASS**。

#### tensor core 侦察（第二轮，2026-11-22）

**f16 TCU 在 ivcore11 上确实存在**，第一轮「硬件层没有」的判断是错的。

- 正确的公开入口是 `#include <crt/iluvatar_mma.h>`，它转发到
  `IX/crt-ixsdk/iluvatar_mma.h`（声明）与 `.hpp`（1294 行实现）。
  之前用 `<mma.h>` 命中 `IX/ixrt-ixsdk/mma.h`（15 行空壳），所以所有 f16
  fragment 都报 undefined template。
- 存在的 f16 特化（`ixix::wmma`，编译目标 ivcore11 实测通过，产出 2832 B
  cubin）：
  `fragment<matrix_a/b, 16,16,32, __half, row/col_major>`、
  `fragment<accumulator,16,16,32,float>`，另有 32x32x16、16x32x32、32x16x32。
  `mma_sync` 落到内建 `__ivcorex_matrix_mad_f32x4_f16x4`（16x16x32 形状一次
  mma_sync 打两次该指令）。
- **但 load/store 不是 CUDA wmma 语义**：`load_matrix_sync` **完全忽略 ldm**，
  按 `CoordToOffset<16, mem_row_major|mem_col_major>` 寻址——这是 16x16 tile
  的 XOR swizzle，例如 row_major 版：
  `(i*256 + r/4*128) + ((r*32 + c*2 + (row&1) + i*64) & 127)`，
  其中 `i=col/16, c=col%16, r=row/2`。也就是要求内存**已经按该 swizzle 排好**。
- 实测（A/B 全 1，`<<<1,32>>>`）：C 一部分是 **16.1**（不是 32），其余数行是
  未初始化垃圾（29827.9 / 22484.0）。说明 ① `store_matrix_sync` 同样写
  swizzle 布局、② K=32 只累加了一半或 lane→(row,col) 映射与直觉不同。
- 结论：要用 TCU 必须逆推这套私有布局（fragment 的 lane→元素映射、mad 的
  输入语义都没有文档）。可行路径是黑盒探测：把 fragment 只置一个 1、
  直接读 `c.x[]`（绕开 store 的 swizzle），确定映射后再自己写 `CoordToOffset`
  重排 Q/K/V 的 tile。
- 收益上限：flash 融合后流量从 637 MB 降到约 67 MB（只读 Q/K/V、写 O），
  算力 8.6 GFLOP → 理论 ~0.2 ms，对比当前 1.394 ms 有 5–7 倍空间。
  但前提是布局逆推成功，且自写分块的效率能接近 cuBLAS 的 97 TFLOPS。


### attention 融合内核重写（2026-11-21，历史，已被上一节替代）

> 保留作为「指令发射上限」的量化证据：这条路线做到 509 GFLOPS 就到顶了，
> 而 cuBLAS 的 tensor core 是 97 TFLOPS，两者不在一个量级。以下内容不再维护。

共享模板 `ATTENTION_SRC` 的结构是"一行 Q 一个 block、每线程串行扫 D 维算一个
key 的 dot"，K 的读取步长 `D*sizeof(T)`，warp 内完全不合并，所以任何形状都停在
~123 GFLOPS。天数包改为在 `iluvatar_af/codegen.py::_ILUVATAR_ATTENTION_SRC` 里
覆盖 `attention_source()`，kernel 参数顺序、grid `(S,BH)`、block、smem 全部沿用
框架，语义逐项保持（在线 softmax 的 m/l/corr 顺序、dropout RNG 索引
`(bh*S+s)*T+t`、dropout 只乘分子、f16 只舍入存取）。

逐项改动与实测（f16，32 heads，`execute_device`）：

| 改动 | 512×512×64 | 说明 |
| --- | --- | --- |
| 原内核 | 18.1 ms / 123 GFLOPS | baseline |
| QK 子组协作（SG=8，lane 分块 CE=8） | 7.86 ms | 同一条 load 覆盖连续 16 B，组内 shuffle 归约 |
| warp 级块归约（2 barrier 取代 9） | 7.81 ms | barrier 从来不是瓶颈（20→6 只值 0.6%） |
| PV 分片（nparts 组 + 折 partial） | 7.84 ms | 原实现让 3/4 的 block 空转 |
| SG 改为模板参数并按 D 分档（D≥96→16） | 5.66 ms | 运行时 SG 会让 `__shfl_down_sync` 的 width 与 `threadIdx.x/SG` 失去常量折叠 |
| Q 片段寄存器化（`af_dot_reg<2/4/8>`） | 4.42 ms | 内层每元素一次 load 而非两次；窄 D 必须分档，否则 8 宽死展开让 D=8 掉 30% |
| 指针 + int 步长（QK 内层与 PV） | 4.56 ms（再经 Q 寄存器化到 4.42） | 消掉每次访问的 64 位乘加 |
| **K 成对 32 位读取（`af_ld2`）** | **4.22 ms / 509 GFLOPS** | load 指令数减半；要求 `D % (2*SG) == 0` 保证 4 字节对齐 |

其余形状（同一版本）：2048×2048×64 = 66.7 ms / 515 GFLOPS（原 2×2 heads 40.3 ms，
折算 4.8×），512×512×128 = 7.56 ms / 568 GFLOPS（原 41.6 ms，5.5×）。
诊断形状：QK-only（D=64,E=8）3.25 ms、QK-only（D=128,E=8）5.80 ms、
PV-only（D=8,E=64）2.96 ms——QK 与 PV 都已接近"每元素 1 load + 1 cvt + 1 FMA"
的发射上限。

试过但**没有收益**（不要再走一遍）：

- `__expf` 替换 `expf`：5.66 → 5.63 ms，噪声内。
- 把块归约从 20 个 barrier 压到 6 个：0.6%。
- 把展开宽度 W 或 qv 数组长度做成模板参数、按 D 在调用点分档：qv 数组长度随 W
  变化时编译器内联出十几个签名，所有形状慢 29%；数组固定 8 而只把 W 模板化时
  天数编译器在某些实例上直接崩溃（见坑 11）。最终保留"固定 8 数组 + 运行时三元
  选 W"。

正确性：全量 conformance **104 executed / 7 refused / 0 failed**
（`attention_basic` 6.0e-08、`attention_masked` 1.2e-07）；另建 7 个用例覆盖
f32/f16、multi-chunk、causal mask、dropout、E>D，误差量级与重写前一致
（f32 ≤6e-07，f16 ≤1.4e-02 且集中在 dropout 用例）。

PV 侧成对读取未做：它需要每线程负责 2 个通道，partial 数翻倍、超出框架给定的
`smem = 2*block*4 + E*4`；要扩 smem 就得在 `IluvatarBackend` 里覆盖
`_op_attention` 自己算 grid/block/smem，属结构性改动，收益上限约 7%。

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
5. **天数没有 `atomicCAS(unsigned short*)`**：device runtime 只提供
   `int` / `unsigned int` / `unsigned long long` 三个重载。f16 原子加若直接
   调 16 位 CAS，连不引用它的 attention 模板都会在函数定义处报
   "no matching function for call to 'atomicCAS'"。修法与 CUDA sm<60 相同：
   `af_cas_h` 把 CAS 加宽到对齐的 32 位字，只替换目标半字（见
   `iluvatar_af/codegen.py::_ILUVATAR_F16_HELPERS`）。
6. 远端 `nohup python /tmp/x.py` 的 `sys.path[0]` 是 `/tmp`，`import aetherflow`
   会失败——脚本里显式 `sys.path.insert(0, "/home/ubuntu/aetherflow")`。
   `pgrep -f x.py` 会匹配到轮询命令自身，判断结束要看日志里的 `DONE_MARKER`。
7. `hardware.build_factory()` 收的是 `HardwareRecord`，不是名字字符串；
   `HardwareRecord` 没有 `.path`（只有 `.root`）；`installed_hardware()`
   返回名字列表。`HardwareRecord.name` 是包名，`HardwareRecord.api` 是声明值。
8. 远端框架的 `aetherflow/train/_arrays.py` 是**手动上传补丁**覆盖的，不是
   仓库同步；重新 clone 远端框架后要先确认这个注册点还在（否则设备态训练
   会退回 TypeError）。正式发布应以本地 aetherflow 仓库的提交为准。
9. **不能用 sftp 直接覆盖已安装目录里的 `.py`**。`manifest.json` 带每个文件的
   sha256 清单，覆盖后校验失败，`installed_hardware()` 返回空，
   `get_backend("iluvatar")` 报"未注册的后端 iluvatar"（且报错点离真实原因很远）。
   任何改动都必须 `py -3.10 build.py` 重打包，再
   `install_hardware("dist/iluvatar-0.1.0.zip", replace=True)`。**换机器、
   升级框架或改天数包都适用**：不要 rsync / sftp 已安装树，也不要只替换其中
   几个 `.py`，否则 manifest 校验失败的表现同上。
   主框架 5.293 起，这种情况的报错会附带"硬件包目录存在但校验未通过：<原因>"，
   并提示用 `install_hardware(..., replace=True)` 重装；看到这句就是覆盖安装，
   不是没装。
10. `runtime.py` **模块级定义了设备函数 `sum`**（同名覆盖内置），所以文件里
    任何模块级/方法内的 `sum(...)` 都会拿到设备算子，`pool_stats` 因此报
    `int + DeviceArray`。改用显式循环计块数。`max` / `min` 未被覆盖。
    后续在此文件里写辅助逻辑时要留意同名覆盖。
11. **天数 clang 在尝试自动向量化 divergent 地址的 f16 读取时会崩**：
    `Cannot select: intrinsic %llvm.bi.ldp.vs.imm` +
    `divergent base ptr of __ivcorex_ml_mem_load_* is detected, which must be
    uniform`。触发条件是模板/数组布局一变，编译器就把相邻两次 f16 load 合并成
    向量 intrinsic，而每 lane 的基址天然 divergent。规避方式是显式写成
    "一次 32 位标量读 + 两次 bit_cast"（`af_ld2`），或保持数组形状固定、只把
    展开宽度做模板参数（后者仍会崩，所以最终只用了前者）。改 attention 模板后
    若出现这两条报错，先看是不是新增了"数组长度随模板参数变化"的写法。
12. ~~框架 `device_attention.py` 用整数索引取掉 mask 前导维~~ **已修复**
    （主框架 5.292，代码 `c4444f5`）。原写法是 `raw_mask = raw_mask[0]`：
    numpy/cupy 会降维，天数 `DeviceArray` 抛
    `TypeError: CUDA 容器只支持切片（不降维）`。现在改为纯形状推导
    `_mask_target_shape` + `reshape`，`(1,1,S,T)`、`(1,1,1,S,T)` 都能直接传，
    调用方**不再需要**先 squeeze 成 `(S,T)`。
    注意：缺陷在 aetherflow 主框架里，硬件包 zip 只含 `iluvatar_af/`；天数机器
    要把主框架更新到含 5.292 的提交，包侧未改动则不必重打包。若同时改了包内
    代码，仍按坑 9 重打包 + `install_hardware(..., replace=True)` 重装。

### 主框架两处修复的真机复验（2026-11-21）

两处都改了主框架（`aetherflow`），硬件包未动，按坑 12 的说明只需更新主框架源码。

1. **`(1,1,S,T)` mask 直传**：`bh=2×2`、`S=T=32`、`D=64`、`scale=0.125`、f16，
   与 numpy 逐点对拍 **max abs err = 8.442e-04**（f16 量级），旧代码此处抛
   `TypeError: CUDA 容器只支持切片（不降维）`。
2. **坏包诊断**（坑 9 的报错质量）：篡改已安装目录里的 `iluvatar_af/rtc.py`
   后，**新进程** `get_backend("iluvatar")` 报
   `LookupError: 未注册的后端 'iluvatar'；已注册: cuda、hip、reference。注意：
   硬件包目录 'iluvatar' 存在但校验未通过，已被跳过：'iluvatar_af/rtc.py'
   摘要不符（声明 c7f8180…，实际 5bad813…）；包在打包后被改动过。不要直接
   覆盖已安装的文件，请重新打包后用 install_hardware(<zip 路径>, replace=True)
   重装`。必须用新进程验证：同进程内 `_FACTORIES` 已缓存注册结果。
3. **回归**：文件还原后全量 `run_conformance` → `PASS`（104 executed /
   7 refused / 0 failed），attention 重写未退化。

## TCU flash attention（自写内核，进行中）

目标：绕开 HBM 物化（当前 1.394 ms 的 87.5% 已是带宽下限），直接吃 tensor
core 的 flash 路径。代码在 [bench/tcu_flash_attn.cu](file:///g:/Cloumi/aetherflow_iluvatar/bench/tcu_flash_attn.cu)，
**不在 `iluvatar_af/` 内，不会被打进硬件包**，是明天接入的基础。

### TCU（tensor core）事实（读头文件 + 实测确认）

- 头文件 `<crt/iluvatar_mma.h>` + `using namespace ixix::wmma;`。
- **一个 mma 必须 64 线程**（一个 ivcore warp，warpSize=64）；不足 64 线程
  参与会算错。
- `fragment<matrix_a,16,16,32,__half,row_major>` = `unsigned int x[4]`；
  `fragment<accumulator,16,16,32,float>` = 4 个 float；矩阵 b 同理 4 字。
- `load_matrix_sync` **忽略 ldm**，按固定 swizzle 从 smem 取值，偏移值域 0..511：
  - `off_row(row,col)`：`i=col>>4; c=col&15; r=row>>1; (i*256 + r/4*128) + ((r*32 + c*2 + (row&1) + i*64) & 127)`
  - `off_col(row,col)`：`r=row>>1; (col>>2)*128 + (r&3)*32 + ((r>>3)^(col>>3))*16 + ((r>>2&1)^(col>>2&1))*8 + (col&3)*2 + (row&1)`
- 一个 uint32 装"同列相邻两行"；gmem 相邻 half 是同一行的相邻列，**不能**
  直接读 uint32 拼 A fragment。
- 裸 mma 上限实测 **83.37 TFLOPS**（cuBLAS f16 4096³ 为 97）。
- `__shfl_xor_sync` 可用；**`__any_sync` / `__ballot_sync` 不可用**（见否决项 5）。
- 编译：`/usr/local/corex-5.0.0/bin/clang++ -x ivcore --cuda-path=$COREX_PATH -O2 -o x x.cu -L$COREX_PATH/lib -lcudart`，
  运行需 `LD_LIBRARY_PATH=$COREX_PATH/lib`。
- 占用：4 blocks/SM × 512 线程要求 regs ≤ 128（262144/(512*4)）；1024 线程
  block 则 2 blocks/SM。

### 结构（NW=16）

每个 block 1024 线程（16 个 warp），负责**整个头**的 512 行；grid=(1,bh)，
K/V 每头只搬一次到 smem。每轮一个 tile 的 key 做 8 次 mma 出 QK
（s00/s01/s10/s11），softmax 就地写进本 warp 的 P smem
（`pw = psm + warp*1024`），10 次 mma 出 PV（o0..o7 + o8/o9 = `P@ones`
求分母），末尾按 `inv = 1/den` 归一化写回。

两个内核共用这套结构，区别只在 tile 宽度：

- `attn32`：tile=32 key，每 warp 只占 2 个分数累加器 × 2 行组，smem 41984 B
  （K/V 各 2048 + P 16×1024 + ones）。数值基线 0.8284 ms。
- `attn64`：tile=64 key（两批 32），K/V 各留两批（smem 50176 B，仍满足
  2 block/SM），一次装载一次 `__syncthreads()`；两批 QK 全做完后**一次**
  4 步 shuffle 覆盖 64 个 key，批 B 的 exp 值先存 `eb[2][4][2]` 寄存器，
  批 A 的 PV 做完再写回 `pw` 复用。归约密度减半 → 0.6737 ms。
- 占用门槛：1024 线程/block 时 `regs*1024*2 ≤ 262144` 即 `regs ≤ 128`
  才能 2 block/SM；attn64 实测 regs=122，余量很小（再加一批分数就掉档，
  见否决项 6）。
- exp 走 **log2 空间**：分数直接乘 `scale*log2e`，softmax 用 `__exp2f`，
  省掉每次 exp 前的乘法；`corr` 无条件乘（不判 `!= 1.f`）。

每轮归约的写法（attn64，log2 空间，4 步覆盖 64 个 key）：

```cpp
float mn = ml[g][q], corr = 1.f;
float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
for (int off = 1; off < 16; off <<= 1)
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
if (mx > mn) { corr = __exp2f(mn - mx); mn = mx; ml[g][q] = mx; }
// e0..e3 = __exp2f(x_i - mn)（x_i 已乘 scale*log2e）
```

### 实测（512×512×64，bh=256，reps=20）

| 变体 | ms | GFLOPS |
| --- | --- | --- |
| GEMM 物化基线（当前包内实现） | 1.394 | 24653 |
| TCU flash attn32（每轮 32 key，精确归约） | 0.8284 | 41475 |
| **TCU flash attn64（每轮 64 key，两批归约合并）** | **0.6737** | **51011** |
| attn64 两批不归约（数值错，仅作上限） | 0.4804 | 71520 |
| 裸 mma 上限 | — | 83370 |

结论：attn64 **0.6737 ms = 物化基线的 2.07×**。理论上限：裸 mma 83.4 TFLOPS
里分母 mma 固定浪费 2/18 条（11%），扣完约 74 TFLOPS；attn64 的 51 TFLOPS
= 该上限的 69%，缺口是每 64 key 一次的 4 步 shuffle（8 段 × 4 步）、32 次
exp2、32 次 P 写 smem 与 2 次 load_matrix_sync，这些都在 TCU 空闲期串行执行。
注意：**同一编译单元内才有可比性**（多实例编译会整体变慢 10~20%）。

### 已否决的方向（不要再试）

1. **降频归约（MR=2/4）在真实值域下溢出**。整数 −2..2 输入下 err 仅 6.5e-4
   看不出问题；换 `std=2.0` 的 f16 正态输入后 MR=2 有 `bad=38656/131072`、
   MR=4 有 `bad=66048/131072`（inf/nan），MR=1 为 0。原因：`mn` 落后真实
   max 时 `exp` 超过 half 上限 65504（对应 x < 11.09）。
2. **双缓冲 K/V**：regs 119→131 超 128，占用掉到 3 blocks/SM，1.0733 ms 更慢。
3. **gmem 直读 A fragment**：布局不符（见上），err 3.29。
4. **K/V 预变换后 gmem 直读**：1.4955 ms，慢于 smem 重排。
5. **`__any_sync` / `__ballot_sync` 门控归约**：`<<<1,64>>>` 探针里
   `__any_sync` 正确、但 `__ballot_sync` 恒返回 0；放进大 block 后数值错
   （err 1.68593）且更慢（0.9726 vs 0.9035 ms）。天数这套 vote intrinsic 不可用。
6. **attn128（每轮 128 key，4 批合并归约）**：regs 122→154，`154*1024*2 >
   262144` → 每 SM 只剩 1 个 block，实测 0.8112 ms，比 attn64 慢 20%。
   要保 2 block/SM 必须 regs ≤ 128，而 16 个分数累加器已占 64 个。
7. **`-O3`**：0.6793 vs `-O2` 0.6767，无收益。
8. **K 两批共用 smem 省中间 `__syncthreads()`**：0.6767 vs 0.6728，噪声内略差；
   结论是块级同步不是瓶颈（代码已简化为一次装载 + 一次同步，保留）。
9. **`vbo`（P@ones 的 B 片段）外提到循环外只装一次**：regs 118→122，0.6760，
   无实质变化（保留，代码更短）。
10. **归约只做 2 步探底**：0.6391 ms（attn32 同期 0.7794）。相对 4 步只值
    0.037 ms，远小于按模型预期的 0.09 ms → 剩余时间已有相当部分不是 shuffle
    本身，继续减步数/加 tile 不划算。

### 明天接着做

- [x] ~~**验证 Δ 平移**~~ **已放弃**：exp 前统一减 `mn + Δ` 属于
      shared-upper-bound 家族，实测 sd=2 时 P 整行下溢（输出整行 0）、
      sd=4 直接 inf/nan。exact 每行每轮 max 是唯一数值安全的归约方式，
      降频（MR>1）路线整体关闭。
- [ ] 支持 mask（`[S,T]` 与 `[B,H,S,T]`）与 dropout（RNG 索引语义与现有
      实现一致）。
- [ ] 明确回退条件：S/T 非 32 倍数、D≠64 或 E≠64、f32 → 回退物化路径。
- [ ] 接入 `iluvatar_af`：`_op_attention` 增加 f16 + D=E=64 + `T%32==0`
      快路径，其余走现有物化实现。
- [ ] `py -3.10 build.py` 重打包 + 远端 `install_hardware(zip, replace=True)`；
      真机对拍 7 项 + 全量 conformance（必须 104 executed / 7 refused /
      0 failed）。

### 本轮随手记的坑

13. **`err = 0` 可能是假象**：输出全 NaN/Inf 时 `fabs(nan - ref) > max` 恒
    false。bench 里要单独数 `if (got != got) ++nbad;`，用 `bad=` 报告。
14. 探测脚本放在 `%TEMP%` 并直接用 `ssh -p 11155 ubuntu@39.156.149.45`
    + 密码推送执行；**用完即删**，结论只留在本文件。
15. 清理进程曾把 `.git/index.lock` 移进回收站。不要用工具恢复；git 报锁错误
    时直接删掉该锁文件即可。
16. **天数机器可能是"只装驱动"的裸镜像**：`/usr/local/corex-5.0.0` 仅 18 MB，
    没有 cudart / cublas / clang++ / include。此时硬件包装得上但
    `get_backend` 报找不到 `libcudart.so.10.2`。按「远端验证机 → 新机」
    的链接装 toolkit 再试，不要怀疑硬件包或 manifest。
