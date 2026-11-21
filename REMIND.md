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
- [x] attention 内核重写：天数包内覆盖 `attention_source`，子组协作 + 成对读取，
      512×512×64 从 18.1 ms（123 GFLOPS）降到 4.2 ms（509 GFLOPS），
      全量 conformance 仍 PASS（见"attention 内核重写"一节）

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
  这是 CUDA 模板的固有结构，不是天数特有缺陷。**已在天数包内覆盖
  `attention_source` 重写**：4.2–7.6 ms / 509–568 GFLOPS（4.3–5.5×），
  RNG 索引、mask 形态与在线 softmax 顺序逐项保持，详见下一小节。
- **conv 的 tiled/grouped 实现慢 23×**（f32 8×64×112² 3×3：auto/im2col ≈5.55 ms，
  implicit 18.3 ms，tiled/grouped 129 ms），`auto` 不选它们是对的；
  `CONV_AUTO_IMPLICIT=False` 保持。小通道 C=3 时 `auto` 选了 tiled（0.827 ms）
  而 im2col 只要 0.515 ms，属已知策略缺陷，收益小、暂不改。
- **未做 Adam 融合**（26 次逐元素 launch → 2 次，训练步还能再省约 40%）：
  会改变 Adam 的数值轨迹，破坏"与主机态逐位对齐"的验收基线。风险大于收益，
  除非另建一套宽松的对齐判据。

### attention 内核重写（2026-11-21）

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
