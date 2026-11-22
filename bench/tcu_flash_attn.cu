// TCU flash attention（Iluvatar MR-V100，f16，D=E=64）
//
// 独立可跑的基准程序，用来验证内核语义与性能；不是包内代码。
// 编译运行：
//   export COREX_PATH=/usr/local/corex-5.0.0
//   $COREX_PATH/bin/clang++ -x ivcore --cuda-path=$COREX_PATH -O2 \
//       -o tcu_flash_attn tcu_flash_attn.cu -L$COREX_PATH/lib -lcudart
//   LD_LIBRARY_PATH=$COREX_PATH/lib ./tcu_flash_attn 256
//
// 设计要点（详见 REMIND.md 的「TCU flash attention」一节）：
//   - 一个 ivcore warp = 64 线程，mma 的 fragment 必须由整 warp 协作装载。
//   - 每 block 16 warp × 32 行 = 512 行（整头），grid = (1, bh)：
//     K/V 每头只从 gmem 搬一次。
//   - 一个 tile = 32 个 key：8 次 mma 出 QK，10 次 mma 出 PV
//     （8 个 E 分块 + 2 个 P@ones 求分母）。
//   - softmax 的 max 不必每轮归约（见下方"惰性基准"）。
//   - 分母走 P@ones 的 mma，比 shuffle 归约 sum 快 13%。
//   - 两个内核：attn32 每轮 32 key（精确基线 0.8284 ms）；
//     attn64 每轮 64 key，两批 QK 并行取 ILP，归约用惰性基准（见下）。
//
// attn64 的算法核心是**惰性基准**：
//   基准 ml 只在"某分数超出 ml+8（log2 空间）"时才用 4 步 shuffle 重算，
//   检测就是一条 __any_sync（1 步，代替 8 段 × 4 步 shuffle）。
//   基准滞后 ≤8 → exp2 上界 256，half 不会溢出；滞后导致的极小项下溢，
//   归一化会自动抵消。sd=1/2/4/8/16 与 attn32 的 max abs err 完全一致。
//   ml 存 smem（每 warp 32 float，共 2 KB）：8 个 float 常驻会顶破
//   128 寄存器门槛（regs*1024*2 ≤ 262144），掉到 1 block/SM。
//
// 512x512x64、bh=256（实测）：
//   attn32 0.8284 ms / 41475 GFLOPS（物化基线 1.394 ms）
//   attn64 0.6725 ms / 51090 GFLOPS  <- 当前最优（regs=128 local=0）
//   裸 mma 上限 83.4 TFLOPS；扣掉 2 条分母 mma 的 11% 浪费后约 74 TFLOPS。
// 已排除：attn128（regs=154，每 SM 只剩 1 个 block，0.811 ms）、-O3、
//   K 双批省同步、vbo 外提、2 步归约（只值 0.037 ms）、
//   只用一组 S fragment（失去两批 QK 的 ILP，0.681 ms）。
// 已采用：exp 走 log2 空间（分数直接乘 scale*log2e）、corr 无条件乘、
//   b/vb 共用同一组 matrix_b fragment。

#include <crt/iluvatar_mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace ixix::wmma;

// load_matrix_sync 忽略 ldm，按 TCU 私有 swizzle 取址。两个布局的实测公式：
//   row_major 16x32 -> 0..511；col_major 32x16 -> 0..511。
__host__ __device__ __forceinline__ int off_row(int row, int col) {
  int i = col >> 4, c = col & 15, r = row >> 1;
  return (i * 256 + r / 4 * 128) + ((r * 32 + c * 2 + (row & 1) + i * 64) & 127);
}
__host__ __device__ __forceinline__ int off_col(int row, int col) {
  int r = row >> 1;
  return (col >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (col >> 3)) * 16 +
         ((r >> 2 & 1) ^ (col >> 2 & 1)) * 8 + (col & 3) * 2 + (row & 1);
}

#define NWT 16                 // warp 数；每 warp 32 行 -> 每 block 512 行
#define NTH (NWT * 64)

#define AF_LOG2E 1.4426950408889634f

// dropout 的 0/1 判定，与包内 af_drop_w / attention_rand 逐位一致：
// idx = (bh*S + s) * T + t，u 取 hash 的 24 位高段。
__device__ __forceinline__ unsigned long long af_rand(unsigned long long seed,
                                                      long long idx) {
  unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
__device__ __forceinline__ float af_drop_w(float p, float p_drop, float drop_scale,
                                           unsigned long long seed, long long idx) {
  const unsigned long long z = af_rand(seed, idx);
  const float u = (float)((z >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
  return (u >= p_drop) ? p * drop_scale : 0.f;
}

// 加性 mask 的行指针：展平 [bh*S + s, T]；mask_shared 时按 s 取模到 S 行。
// 内核里不用这个（每轮调一次太贵，改成循环外算基址），保留给 bench 参考端。
__host__ __device__ __forceinline__ const __half* mask_row(const __half* M,
                                                           int mask_shared, int bh,
                                                           int S, int gl, int T) {
  const long long r = mask_shared ? (long long)(gl % S) : ((long long)bh * S + gl);
  return M + r * T;
}

// 32 个 key 的 K/V 重排进 swizzle 布局的 smem。每次 2048 个 half（64x32）。
// nth 是 block 的线程数：每 warp 16 行的内核用 2048 线程，这里必须跟着变。
__device__ __forceinline__ void load_kv(__half* ks, __half* vs, const __half* Kb0,
                                        const __half* Vb0, int kt, int nth = NTH) {
  const long long o = (long long)kt * 2048;
  for (int idx = threadIdx.x; idx < 2048; idx += nth) {
    int t = idx >> 6, d = idx & 63;
    ks[((t >> 4) * 2 + (d >> 5)) * 512 + off_col(d & 31, t & 15)] = Kb0[o + idx];
    vs[(d >> 4) * 512 + off_col(t, d & 15)] = Vb0[o + idx];
  }
}

// 每轮 32 个 key，每行每轮一次 4 步 shuffle 的精确 max（基线 0.83 ms）。
// mask/dropout 只在 attn64 上实现，这里保留参数位以便共用 bench 调用约定。
__global__ void __launch_bounds__(NTH) attn32(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int S, int T, float scale, const __half*, float, float, unsigned long long, int,
    int) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                    // 2 x 32x16 (K 的两个 16 列组)
  __half* vsm = ksm + 2048;            // 4 x 16x16
  __half* psm = vsm + 2048;            // 每 warp 一个 16x32 的 P
  __half* ones_sm = psm + NWT * 1024;  // 16x16 全 1，用于 P@ones

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;            // 行组（0..3）
  const int c0 = lane & 15;            // 列
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (32 * NWT) + warp * 32;

  const __half* Qb = Q + (long long)bh * S * 64;
  __half* Ob = O + (long long)bh * S * 64;
  const __half* Kb0 = K + (long long)bh * T * 64;
  const __half* Vb0 = V + (long long)bh * T * 64;
  __half* pw = psm + warp * 1024;

  fragment<matrix_a, 16, 16, 32, __half, row_major> a00, a01, a10, a11, pa0, pa1;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1, b2, b3;
  fragment<matrix_b, 16, 16, 32, __half, col_major> vb0, vb1, vb2, vb3, vbo;
  fragment<accumulator, 16, 16, 32, float> s00, s01, s10, s11;
  fragment<accumulator, 16, 16, 32, float> o0, o1, o2, o3, o4, o5, o6, o7, o8, o9;

  // A fragment 必须"同列相邻两行"打包；gmem 是行内相邻列，不能直读 uint32。
  // 借 P 的 smem 分两批（各 16 行）中转。
  {
    __half* qtmp = pw;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      for (int idx = lane; idx < 1024; idx += 64) {
        int m = idx >> 6, d = idx & 63;
        qtmp[(d >> 5) * 512 + off_row(m, d & 31)] =
            Qb[(long long)(row_base + half * 16 + m) * 64 + d];
      }
      __syncwarp();
      if (half == 0) {
        load_matrix_sync(a00, qtmp, 32);
        load_matrix_sync(a01, qtmp + 512, 32);
      } else {
        load_matrix_sync(a10, qtmp, 32);
        load_matrix_sync(a11, qtmp + 512, 32);
      }
      __syncwarp();
    }
  }

  fragment<accumulator, 16, 16, 32, float>* SF[2][2] = {{&s00, &s01}, {&s10, &s11}};
  fragment<accumulator, 16, 16, 32, float>* OF[2][5] = {
      {&o0, &o1, &o2, &o3, &o8}, {&o4, &o5, &o6, &o7, &o9}};

#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 5; ++e) fill_fragment(*OF[g][e], 0.f);

  float ml[2][4];   // 每行的 running max
#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int q = 0; q < 4; ++q) ml[g][q] = -1e30f;

  for (int idx = threadIdx.x; idx < 512; idx += NTH) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[off_col(t, dd)] = __float2half(1.f);
  }

  const int nk = T >> 5;
  for (int kt = 0; kt < nk; ++kt) {
    load_kv(ksm, vsm, Kb0, Vb0, kt);
    __syncthreads();

    fill_fragment(s00, 0.f);
    fill_fragment(s01, 0.f);
    fill_fragment(s10, 0.f);
    fill_fragment(s11, 0.f);
    load_matrix_sync(b0, ksm, 32);
    load_matrix_sync(b1, ksm + 512, 32);
    load_matrix_sync(b2, ksm + 1024, 32);
    load_matrix_sync(b3, ksm + 1536, 32);
    mma_sync(s00, a00, b0, s00);
    mma_sync(s00, a01, b1, s00);
    mma_sync(s01, a00, b2, s01);
    mma_sync(s01, a01, b3, s01);
    mma_sync(s10, a10, b0, s10);
    mma_sync(s10, a11, b1, s10);
    mma_sync(s11, a10, b2, s11);
    mma_sync(s11, a11, b3, s11);

    // 每行精确 max。
#pragma unroll
    for (int g = 0; g < 2; ++g) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float x0 = SF[g][0]->x[q] * scale, x1 = SF[g][1]->x[q] * scale;
        float mn = ml[g][q], corr = 1.f;
        float mx = fmaxf(x0, x1);
#pragma unroll
        for (int off = 1; off < 16; off <<= 1)
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        if (mx > mn) {
          corr = __expf(mn - mx);
          mn = mx;
          ml[g][q] = mx;
        }
        float e0 = __expf(x0 - mn);
        float e1 = __expf(x1 - mn);
        int row = r0 + 4 * q;
        pw[g * 512 + off_row(row, c0)] = __float2half(e0);
        pw[g * 512 + off_row(row, 16 + c0)] = __float2half(e1);
        if (corr != 1.f) {
#pragma unroll
          for (int e = 0; e < 5; ++e) OF[g][e]->x[q] *= corr;
        }
      }
    }
    __syncwarp();
    load_matrix_sync(pa0, pw, 32);
    load_matrix_sync(pa1, pw + 512, 32);
    load_matrix_sync(vb0, vsm, 32);
    load_matrix_sync(vb1, vsm + 512, 32);
    load_matrix_sync(vb2, vsm + 1024, 32);
    load_matrix_sync(vb3, vsm + 1536, 32);
    load_matrix_sync(vbo, ones_sm, 32);
    mma_sync(o0, pa0, vb0, o0);
    mma_sync(o1, pa0, vb1, o1);
    mma_sync(o2, pa0, vb2, o2);
    mma_sync(o3, pa0, vb3, o3);
    mma_sync(o4, pa1, vb0, o4);
    mma_sync(o5, pa1, vb1, o5);
    mma_sync(o6, pa1, vb2, o6);
    mma_sync(o7, pa1, vb3, o7);
    mma_sync(o8, pa0, vbo, o8);   // 分母：P @ ones
    mma_sync(o9, pa1, vbo, o9);
  }

  // 归一化放在输出上：(p/l)@V == (p@V)/l，省掉一整遍分数矩阵的读写。
#pragma unroll
  for (int g = 0; g < 2; ++g) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float den = OF[g][4]->x[q];
      float inv = den != 0.f ? 1.f / den : 0.f;
      int row = row_base + g * 16 + r0 + 4 * q;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        Ob[row * 64 + e * 16 + c0] = __float2half(OF[g][e]->x[q] * inv);
    }
  }
}

// 每轮 64 个 key（两批 32）：两批 QK 背靠背发射，惰性基准只在超界时归约。
// K/V 各留两批，一次装载后只同步一次，批 A/B 之间不再有块级同步。
// mask 是加性（与原融合内核一致，行优先 [rows,T]，可 S 行共享）；
// dropout 用 af_drop_w（同一 seed 逐位一致），分母仍累加 dropout 之前的 p。
// HAS_MASK / HAS_DROP 做成模板参数：运行时 bool 会让编译器为两条路径同时
// 保活寄存器，plain 实例从 local=0 掉到 ~500 B/线程，性能崩到 13 ms。
// mask/dropout 需要额外的行指针与 64 位 hash 状态，128 寄存器装不下会
// spill 到 local（340 B/线程时慢 4.7 倍），因此这两类实例改按 1 block/SM
// 编译（上限 255 寄存器）；plain 实例仍是 2 block/SM。
template <bool HAS_MASK, bool HAS_DROP>
__global__ void __launch_bounds__(NTH, 2) attn64(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int S, int T, float scale, const __half* __restrict__ M, float p_drop,
    float drop_scale, unsigned long long seed, int /*has_mask*/, int mask_shared) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                    // 两批 K，各 2048（一次装载，免中间同步）
  __half* vsm = ksm + 4096;            // 两批 V，各 4 x 16x16
  __half* psm = vsm + 4096;            // 每 warp 一个 16x32 的 P（两批复用）
  __half* ones_sm = psm + NWT * 1024;  // 16x16 全 1，用于 P@ones
  __shared__ float mls[NWT * 32];      // 惰性基准（每 warp 32 行）
  __shared__ float ssums[NWT * 32];    // dropout 模式下的分母（每 warp 32 行）

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;
  const int c0 = lane & 15;
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (32 * NWT) + warp * 32;

  const __half* Qb = Q + (long long)bh * S * 64;
  __half* Ob = O + (long long)bh * S * 64;
  const __half* Kb0 = K + (long long)bh * T * 64;
  const __half* Vb0 = V + (long long)bh * T * 64;
  __half* pw = psm + warp * 1024;

  fragment<matrix_a, 16, 16, 32, __half, row_major> a00, a01, a10, a11, pa0, pa1;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1, b2, b3, vbo;
  fragment<accumulator, 16, 16, 32, float> s00, s01, s10, s11;  // 批 A 分数
  fragment<accumulator, 16, 16, 32, float> t00, t01, t10, t11;  // 批 B 分数
  fragment<accumulator, 16, 16, 32, float> o0, o1, o2, o3, o4, o5, o6, o7, o8, o9;

  // Q 装载（同 attn32）：借 P 的 smem 分两批中转。
  {
    __half* qtmp = pw;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      for (int idx = lane; idx < 1024; idx += 64) {
        int m = idx >> 6, d = idx & 63;
        qtmp[(d >> 5) * 512 + off_row(m, d & 31)] =
            Qb[(long long)(row_base + half * 16 + m) * 64 + d];
      }
      __syncwarp();
      if (half == 0) {
        load_matrix_sync(a00, qtmp, 32);
        load_matrix_sync(a01, qtmp + 512, 32);
      } else {
        load_matrix_sync(a10, qtmp, 32);
        load_matrix_sync(a11, qtmp + 512, 32);
      }
      __syncwarp();
    }
  }

  fragment<accumulator, 16, 16, 32, float>* SF[2][2] = {{&s00, &s01}, {&s10, &s11}};
  fragment<accumulator, 16, 16, 32, float>* SB[2][2] = {{&t00, &t01}, {&t10, &t11}};
  fragment<accumulator, 16, 16, 32, float>* OF[2][5] = {
      {&o0, &o1, &o2, &o3, &o8}, {&o4, &o5, &o6, &o7, &o9}};

#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 5; ++e) fill_fragment(*OF[g][e], 0.f);

  // 每行基准放 smem（每 warp 32 行）：8 个 float 常驻会顶破 128 寄存器门槛，
  // 那就只剩 1 个 block/SM。放 smem 只花 2 KB，每轮 16 次 LDS 完全藏得住。
  for (int idx = threadIdx.x; idx < NWT * 32; idx += NTH) {
    mls[idx] = -1e30f;
    ssums[idx] = 0.f;
  }

  for (int idx = threadIdx.x; idx < 512; idx += NTH) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[off_col(t, dd)] = __float2half(1.f);
  }
  __syncthreads();
  load_matrix_sync(vbo, ones_sm, 32);   // 分母矩阵不变，全程装一次
  // exp 走 log2 空间：分数直接乘 scale*log2e，省掉每次 exp 前的一次乘法。
  const float sc2 = scale * AF_LOG2E;
  float* mlr = mls + warp * 32;         // mlr[g*16 + r0 + 4q]
  float* ssr = ssums + warp * 32;       // dropout 分母

  // mask/dropout 的行基址只随 (g,q) 偏移，row_base+r0 是每线程常量，循环外
  // 算一次；否则每轮每个 (g,q) 都要做一次 (gl % S) 整数除法（mask 路径的
  // 主要开销，去掉后 m 模式从 2.1 ms 降到 1.3 ms 量级）。
  const __half* mbase = nullptr;
  long long dbase = 0;
  if (HAS_MASK) {
    const int mr0 = (mask_shared ? (row_base % S) : (bh * S + row_base)) + r0;
    mbase = M + (long long)mr0 * T;
  }
  if (HAS_DROP) dbase = (long long)(bh * S + row_base + r0) * T;

  for (int kt = 0; kt < T; kt += 64) {
    const bool hasB = (kt + 32 < T);
    load_kv(ksm, vsm, Kb0, Vb0, kt >> 5);
    if (hasB) load_kv(ksm + 2048, vsm + 2048, Kb0, Vb0, (kt >> 5) + 1);
    __syncthreads();

    // 批 A 的 QK
    fill_fragment(s00, 0.f);
    fill_fragment(s01, 0.f);
    fill_fragment(s10, 0.f);
    fill_fragment(s11, 0.f);
    load_matrix_sync(b0, ksm, 32);
    load_matrix_sync(b1, ksm + 512, 32);
    load_matrix_sync(b2, ksm + 1024, 32);
    load_matrix_sync(b3, ksm + 1536, 32);
    mma_sync(s00, a00, b0, s00);
    mma_sync(s00, a01, b1, s00);
    mma_sync(s01, a00, b2, s01);
    mma_sync(s01, a01, b3, s01);
    mma_sync(s10, a10, b0, s10);
    mma_sync(s10, a11, b1, s10);
    mma_sync(s11, a10, b2, s11);
    mma_sync(s11, a11, b3, s11);

    // 批 B 的 QK（T 不是 64 倍数时最后半批不存在，分数填 -1e30 使 exp 为 0）
    if (hasB) {
      fill_fragment(t00, 0.f);
      fill_fragment(t01, 0.f);
      fill_fragment(t10, 0.f);
      fill_fragment(t11, 0.f);
      load_matrix_sync(b0, ksm + 2048, 32);
      load_matrix_sync(b1, ksm + 2048 + 512, 32);
      load_matrix_sync(b2, ksm + 2048 + 1024, 32);
      load_matrix_sync(b3, ksm + 2048 + 1536, 32);
      mma_sync(t00, a00, b0, t00);
      mma_sync(t00, a01, b1, t00);
      mma_sync(t01, a00, b2, t01);
      mma_sync(t01, a01, b3, t01);
      mma_sync(t10, a10, b0, t10);
      mma_sync(t10, a11, b1, t10);
      mma_sync(t11, a10, b2, t11);
      mma_sync(t11, a11, b3, t11);
    } else {
      fill_fragment(t00, -1e30f);
      fill_fragment(t01, -1e30f);
      fill_fragment(t10, -1e30f);
      fill_fragment(t11, -1e30f);
    }

    // 风险检测单独一遍：dropout（含 mask+dropout）必须在这里、也就是写 P 和
    // 累加分母之前把基准定下来。分母 ssr 是 float 逐轮累加，若先按初始基准
    // -1e30 写 P，exp2 溢出成 inf，随后慢路径的 corr=0 又乘出 NaN；只有让
    // 预检先跑，首轮的 inf 才不会进 ssr。plain 也一样（两批 QK 已算出，检测
    // 并进写 P 的一遍会把活跃值从 8 个推到 20 个 float，local 8 → 20 B）。
    // 纯 mask（无 dropout）没有这个约束，检测并进写 P 的一遍，mask 流量从
    // 3 倍降到 1 倍，是 m 实例 1.62 → 1.10 ms 的主因。
    if (!HAS_MASK || HAS_DROP) {
      bool risk = false;
#pragma unroll
      for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = g * 16 + r0 + 4 * q;
          const float lim = mlr[ri] + 8.f;
          float x0 = SF[g][0]->x[q] * sc2, x1 = SF[g][1]->x[q] * sc2;
          float x2 = SB[g][0]->x[q] * sc2, x3 = SB[g][1]->x[q] * sc2;
          if (HAS_MASK) {
            const __half* mr = mbase + ((g * 16 + 4 * q) * T + kt);
            x0 += __half2float(mr[c0]) * AF_LOG2E;
            x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
            if (hasB) {
              x2 += __half2float(mr[32 + c0]) * AF_LOG2E;
              x3 += __half2float(mr[48 + c0]) * AF_LOG2E;
            }
          }
          risk |= fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) > lim;
        }
      if (__any_sync(0xffffffffu, risk)) {
        // 罕见慢路径：4 步 shuffle 求真实 max，整体缩放旧输出与分母。
        // 基准最多落后 8（log2 空间）→ exp2 上界 256，half 安全。
#pragma unroll
        for (int g = 0; g < 2; ++g)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int ri = g * 16 + r0 + 4 * q;
            const float mn = mlr[ri];
            float x0 = SF[g][0]->x[q] * sc2, x1 = SF[g][1]->x[q] * sc2;
            float x2 = SB[g][0]->x[q] * sc2, x3 = SB[g][1]->x[q] * sc2;
            if (HAS_MASK) {
              const __half* mr = mbase + ((g * 16 + 4 * q) * T + kt);
              x0 += __half2float(mr[c0]) * AF_LOG2E;
              x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
              if (hasB) {
                x2 += __half2float(mr[32 + c0]) * AF_LOG2E;
                x3 += __half2float(mr[48 + c0]) * AF_LOG2E;
              }
            }
            float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            float corr = 1.f;
            if (mx > mn) {
              corr = exp2f(mn - mx);
              mlr[ri] = mx;
            }
#pragma unroll
            for (int e = 0; e < 5; ++e) OF[g][e]->x[q] *= corr;
            if (HAS_DROP && c0 == 0) ssr[ri] *= corr;
          }
      }
    }

    // 单遍写 P：按当前（惰性）基准直接算 exp 并写 P，同时记录"本批是否有分数
    // 超出基准 8 以上"。风险检测并进这一遍，mask 每元素每轮只读一次——原先
    // 风险检测、快路径、批 B 各读一次，3 倍冗余的 mask 流量正是 m 实例慢
    // 2.4 倍的主因（每轮 stride 读 4 行 × 32 B，靠 L2 顶不住 196 MB/次）。
    // 慢路径只在真超界时（罕见）执行：重算基准、修正 OF/分母，并重写本批 P。
    // 加性 mask 在 exp 之内，log2 空间里同样乘 log2e。
    bool risky = false;
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = g * 16 + r0 + 4 * q;
        const float mn = mlr[ri];
        float x0 = SF[g][0]->x[q] * sc2, x1 = SF[g][1]->x[q] * sc2;
        if (HAS_MASK) {
          const __half* mr = mbase + ((g * 16 + 4 * q) * T + kt);
          x0 += __half2float(mr[c0]) * AF_LOG2E;
          x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
        }
        if (HAS_MASK && !HAS_DROP) risky |= fmaxf(x0, x1) > mn + 8.f;
        float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
        if (HAS_DROP) {
          const long long base = dbase + (g * 16 + 4 * q) * T + kt;
          float sv = p0 + p1;
          p0 = af_drop_w(p0, p_drop, drop_scale, seed, base + c0);
          p1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            sv += __shfl_xor_sync(0xffffffffu, sv, off);
          if (c0 == 0) ssr[ri] += sv;
        }
        const int row = r0 + 4 * q;
        pw[g * 512 + off_row(row, c0)] = __float2half(p0);
        pw[g * 512 + off_row(row, 16 + c0)] = __float2half(p1);
      }
    // 慢路径只服务 mask 实例（带 dropout 的实例已在预检里定好基准，写 P 时
    // 基准不会再动，不必重写 P）。
    if (HAS_MASK && !HAS_DROP) {
      if (__any_sync(0xffffffffu, risky)) {
#pragma unroll
        for (int g = 0; g < 2; ++g)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int ri = g * 16 + r0 + 4 * q;
            float x0 = SF[g][0]->x[q] * sc2, x1 = SF[g][1]->x[q] * sc2;
            const __half* mr = mbase + ((g * 16 + 4 * q) * T + kt);
            x0 += __half2float(mr[c0]) * AF_LOG2E;
            x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
            float mn = mlr[ri], corr = 1.f;
            float mx = fmaxf(x0, x1);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            if (mx > mn) {
              corr = exp2f(mn - mx);
              mlr[ri] = mx;
              mn = mx;
            }
#pragma unroll
            for (int e = 0; e < 5; ++e) OF[g][e]->x[q] *= corr;
            float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
            if (HAS_DROP) {
              // 分母：旧贡献（含本轮的旧基准值）整体乘 corr 后再加上本轮新值，
              // 与分子「旧 OF 乘 corr + 本轮新 P」完全一致。
              if (c0 == 0) ssr[ri] *= corr;
              const long long base = dbase + (g * 16 + 4 * q) * T + kt;
              float sv = p0 + p1;
              p0 = af_drop_w(p0, p_drop, drop_scale, seed, base + c0);
              p1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
              for (int off = 1; off < 16; off <<= 1)
                sv += __shfl_xor_sync(0xffffffffu, sv, off);
              if (c0 == 0) ssr[ri] += sv;
            }
            const int row = r0 + 4 * q;
            pw[g * 512 + off_row(row, c0)] = __float2half(p0);
            pw[g * 512 + off_row(row, 16 + c0)] = __float2half(p1);
          }
      }
    }
    __syncwarp();   // 慢路径写过 smem 基准，PV 前各 lane 要读到同一份 P

    // 批 A 的 PV（b0..b3 与 QK 的 K 复用：QK 已算完，可以覆盖）
    __syncwarp();
    load_matrix_sync(pa0, pw, 32);
    load_matrix_sync(pa1, pw + 512, 32);
    load_matrix_sync(b0, vsm, 32);
    load_matrix_sync(b1, vsm + 512, 32);
    load_matrix_sync(b2, vsm + 1024, 32);
    load_matrix_sync(b3, vsm + 1536, 32);
    mma_sync(o0, pa0, b0, o0);
    mma_sync(o1, pa0, b1, o1);
    mma_sync(o2, pa0, b2, o2);
    mma_sync(o3, pa0, b3, o3);
    mma_sync(o4, pa1, b0, o4);
    mma_sync(o5, pa1, b1, o5);
    mma_sync(o6, pa1, b2, o6);
    mma_sync(o7, pa1, b3, o7);
    if (!HAS_DROP) {   // dropout 时分母走 shuffle 求和，省掉这 2 条 mma
      mma_sync(o8, pa0, vbo, o8);
      mma_sync(o9, pa1, vbo, o9);
    }

    if (hasB) {
      // 批 B 的 P 在这里现算（不占暂存寄存器），pw 已空出；结构与批 A 相同。
      __syncwarp();
      risky = false;
#pragma unroll
      for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = g * 16 + r0 + 4 * q;
          const float mn = mlr[ri];
          float x0 = SB[g][0]->x[q] * sc2, x1 = SB[g][1]->x[q] * sc2;
          if (HAS_MASK) {
            const __half* mr = mbase + ((g * 16 + 4 * q) * T + kt);
            x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
            x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
          }
          if (HAS_MASK && !HAS_DROP) risky |= fmaxf(x0, x1) > mn + 8.f;
          float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
          if (HAS_DROP) {
            const long long base = dbase + (g * 16 + 4 * q) * T + kt + 32;
            float sv = p0 + p1;
            p0 = af_drop_w(p0, p_drop, drop_scale, seed, base + c0);
            p1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              sv += __shfl_xor_sync(0xffffffffu, sv, off);
            if (c0 == 0) ssr[ri] += sv;
          }
          const int row = r0 + 4 * q;
          pw[g * 512 + off_row(row, c0)] = __float2half(p0);
          pw[g * 512 + off_row(row, 16 + c0)] = __float2half(p1);
        }
      if (HAS_MASK && !HAS_DROP) {
        if (__any_sync(0xffffffffu, risky)) {
#pragma unroll
          for (int g = 0; g < 2; ++g)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int ri = g * 16 + r0 + 4 * q;
              float x0 = SB[g][0]->x[q] * sc2, x1 = SB[g][1]->x[q] * sc2;
              const __half* mr = mbase + ((g * 16 + 4 * q) * T + kt);
              x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
              x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
              float mn = mlr[ri], corr = 1.f;
              float mx = fmaxf(x0, x1);
#pragma unroll
              for (int off = 1; off < 16; off <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
              if (mx > mn) {
                corr = exp2f(mn - mx);
                mlr[ri] = mx;
                mn = mx;
              }
#pragma unroll
              for (int e = 0; e < 5; ++e) OF[g][e]->x[q] *= corr;
              float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
              if (HAS_DROP) {
                if (c0 == 0) ssr[ri] *= corr;
                const long long base = dbase + (g * 16 + 4 * q) * T + kt + 32;
                float sv = p0 + p1;
                p0 = af_drop_w(p0, p_drop, drop_scale, seed, base + c0);
                p1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
                for (int off = 1; off < 16; off <<= 1)
                  sv += __shfl_xor_sync(0xffffffffu, sv, off);
                if (c0 == 0) ssr[ri] += sv;
              }
              const int row = r0 + 4 * q;
              pw[g * 512 + off_row(row, c0)] = __float2half(p0);
              pw[g * 512 + off_row(row, 16 + c0)] = __float2half(p1);
            }
        }
      }
      __syncwarp();
      load_matrix_sync(pa0, pw, 32);
      load_matrix_sync(pa1, pw + 512, 32);
      load_matrix_sync(b0, vsm + 2048, 32);
      load_matrix_sync(b1, vsm + 2048 + 512, 32);
      load_matrix_sync(b2, vsm + 2048 + 1024, 32);
      load_matrix_sync(b3, vsm + 2048 + 1536, 32);
      mma_sync(o0, pa0, b0, o0);
      mma_sync(o1, pa0, b1, o1);
      mma_sync(o2, pa0, b2, o2);
      mma_sync(o3, pa0, b3, o3);
      mma_sync(o4, pa1, b0, o4);
      mma_sync(o5, pa1, b1, o5);
      mma_sync(o6, pa1, b2, o6);
      mma_sync(o7, pa1, b3, o7);
      if (!HAS_DROP) {
        mma_sync(o8, pa0, vbo, o8);
        mma_sync(o9, pa1, vbo, o9);
      }
    }
    __syncthreads();
  }

  // 归一化放在输出上：(p/l)@V == (p@V)/l。dropout 时 l 是 dropout 前的和。
#pragma unroll
  for (int g = 0; g < 2; ++g) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float den = HAS_DROP ? ssr[g * 16 + r0 + 4 * q] : OF[g][4]->x[q];
      float inv = den != 0.f ? 1.f / den : 0.f;
      int row = row_base + g * 16 + r0 + 4 * q;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        Ob[row * 64 + e * 16 + c0] = __float2half(OF[g][e]->x[q] * inv);
    }
  }
}

// 无 mask、无 dropout 的专用内核：把模板里 mask/dropout 的分支物理删掉。
//
// 只有**一组** S fragment（8 个 accumulator = 32 个 float）会让 plain 被挤到
// local=100 B（0.82 ms）。两批 QK 前缀能换 ILP，但代价是第二组 S 常驻，寄存器
// 越过 128 上限后就开始 spill；单组、两批串行的版本回到 local=0（0.68 ms）。
__global__ void __launch_bounds__(NTH, 2) attn64p(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O, int S, int T,
    float scale, const __half*, float, float, unsigned long long, int, int) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                    // 两批 K
  __half* vsm = ksm + 4096;            // 两批 V
  __half* psm = vsm + 4096;            // 每 warp 一个 16x32 的 P（两批复用）
  __half* ones_sm = psm + NWT * 1024;  // 16x16 全 1，用于 P@ones
  __shared__ float mls[NWT * 32];      // 惰性基准

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;
  const int c0 = lane & 15;
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (32 * NWT) + warp * 32;

  const __half* Qb = Q + (long long)bh * S * 64;
  __half* Ob = O + (long long)bh * S * 64;
  const __half* Kb0 = K + (long long)bh * T * 64;
  const __half* Vb0 = V + (long long)bh * T * 64;
  __half* pw = psm + warp * 1024;

  fragment<matrix_a, 16, 16, 32, __half, row_major> a00, a01, a10, a11, pa0, pa1;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1, b2, b3, vbo;
  fragment<accumulator, 16, 16, 32, float> s00, s01, s10, s11;  // 单组，两批复用
  fragment<accumulator, 16, 16, 32, float> o0, o1, o2, o3, o4, o5, o6, o7, o8, o9;

  {
    __half* qtmp = pw;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      for (int idx = lane; idx < 1024; idx += 64) {
        int m = idx >> 6, d = idx & 63;
        qtmp[(d >> 5) * 512 + off_row(m, d & 31)] =
            Qb[(long long)(row_base + half * 16 + m) * 64 + d];
      }
      __syncwarp();
      if (half == 0) {
        load_matrix_sync(a00, qtmp, 32);
        load_matrix_sync(a01, qtmp + 512, 32);
      } else {
        load_matrix_sync(a10, qtmp, 32);
        load_matrix_sync(a11, qtmp + 512, 32);
      }
      __syncwarp();
    }
  }

  fragment<accumulator, 16, 16, 32, float>* SF[2][2] = {{&s00, &s01}, {&s10, &s11}};
  fragment<accumulator, 16, 16, 32, float>* OF[2][5] = {
      {&o0, &o1, &o2, &o3, &o8}, {&o4, &o5, &o6, &o7, &o9}};

#pragma unroll
  for (int g = 0; g < 2; ++g)
#pragma unroll
    for (int e = 0; e < 5; ++e) fill_fragment(*OF[g][e], 0.f);

  for (int idx = threadIdx.x; idx < NWT * 32; idx += NTH) mls[idx] = -1e30f;
  for (int idx = threadIdx.x; idx < 512; idx += NTH) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[off_col(t, dd)] = __float2half(1.f);
  }
  __syncthreads();
  load_matrix_sync(vbo, ones_sm, 32);

  const float sc2 = scale * AF_LOG2E;
  float* mlr = mls + warp * 32;

  for (int kt = 0; kt < T; kt += 64) {
    const bool hasB = (kt + 32 < T);
    load_kv(ksm, vsm, Kb0, Vb0, kt >> 5);
    if (hasB) load_kv(ksm + 2048, vsm + 2048, Kb0, Vb0, (kt >> 5) + 1);
    __syncthreads();

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      if (half == 1 && !hasB) break;
      const __half* kb = (half == 0) ? ksm : ksm + 2048;
      const __half* vb = (half == 0) ? vsm : vsm + 2048;

      // QK：两批串行，S fragment 复用，寄存器压力减半。
      fill_fragment(s00, 0.f);
      fill_fragment(s01, 0.f);
      fill_fragment(s10, 0.f);
      fill_fragment(s11, 0.f);
      load_matrix_sync(b0, kb, 32);
      load_matrix_sync(b1, kb + 512, 32);
      load_matrix_sync(b2, kb + 1024, 32);
      load_matrix_sync(b3, kb + 1536, 32);
      mma_sync(s00, a00, b0, s00);
      mma_sync(s00, a01, b1, s00);
      mma_sync(s01, a00, b2, s01);
      mma_sync(s01, a01, b3, s01);
      mma_sync(s10, a10, b0, s10);
      mma_sync(s10, a11, b1, s10);
      mma_sync(s11, a10, b2, s11);
      mma_sync(s11, a11, b3, s11);

      // 风险检测：超界就 4 步 shuffle 重算基准。
      bool risky = false;
#pragma unroll
      for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = g * 16 + r0 + 4 * q;
          const float x0 = SF[g][0]->x[q] * sc2, x1 = SF[g][1]->x[q] * sc2;
          risky |= fmaxf(x0, x1) > mlr[ri] + 8.f;
        }
      if (__any_sync(0xffffffffu, risky)) {
#pragma unroll
        for (int g = 0; g < 2; ++g)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int ri = g * 16 + r0 + 4 * q;
            const float mn = mlr[ri];
            float mx = fmaxf(SF[g][0]->x[q], SF[g][1]->x[q]) * sc2;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            float corr = 1.f;
            if (mx > mn) {
              corr = exp2f(mn - mx);
              mlr[ri] = mx;
            }
#pragma unroll
            for (int e = 0; e < 5; ++e) OF[g][e]->x[q] *= corr;
          }
      }
      __syncwarp();

      // P：直接由 S fragment 写 swizzle 布局，不占暂存寄存器。
#pragma unroll
      for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float mn = mlr[g * 16 + r0 + 4 * q];
          const float p0 = exp2f(SF[g][0]->x[q] * sc2 - mn);
          const float p1 = exp2f(SF[g][1]->x[q] * sc2 - mn);
          const int row = r0 + 4 * q;
          pw[g * 512 + off_row(row, c0)] = __float2half(p0);
          pw[g * 512 + off_row(row, 16 + c0)] = __float2half(p1);
        }
      __syncwarp();
      load_matrix_sync(pa0, pw, 32);
      load_matrix_sync(pa1, pw + 512, 32);
      load_matrix_sync(b0, vb, 32);
      load_matrix_sync(b1, vb + 512, 32);
      load_matrix_sync(b2, vb + 1024, 32);
      load_matrix_sync(b3, vb + 1536, 32);
      mma_sync(o0, pa0, b0, o0);
      mma_sync(o1, pa0, b1, o1);
      mma_sync(o2, pa0, b2, o2);
      mma_sync(o3, pa0, b3, o3);
      mma_sync(o4, pa1, b0, o4);
      mma_sync(o5, pa1, b1, o5);
      mma_sync(o6, pa1, b2, o6);
      mma_sync(o7, pa1, b3, o7);
      mma_sync(o8, pa0, vbo, o8);
      mma_sync(o9, pa1, vbo, o9);
      if (half == 0) __syncwarp();
    }
    __syncthreads();
  }

#pragma unroll
  for (int g = 0; g < 2; ++g) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float den = OF[g][4]->x[q];
      const float inv = den != 0.f ? 1.f / den : 0.f;
      const int row = row_base + g * 16 + r0 + 4 * q;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        Ob[row * 64 + e * 16 + c0] = __float2half(OF[g][e]->x[q] * inv);
    }
  }
}


// 每 warp 16 行（32 warp/block = 2048 线程，1 block/SM，仍是 512 行/block）。
//
// 动机：每 warp 32 行时 OF 是 2 组 × 5 个 accumulator = 40 个 float/lane，
// 加上双批 S（16 个 float）就把 128 寄存器吃满（plain 只能退回单批串行）。
// 每 warp 16 行后 OF 降到 5 个 fragment = 20 个 float，双批 S 只 16 个 float，
// 两批 QK 可以真正背靠背，不必为了不 spill 牺牲 ILP。
//
// 代价：线程数翻倍 → 每个 lane 的活减少，`__syncthreads` 要等 16 个 warp；
// 寄存器 78 远低于 262144/(1024*2) = 128，因此按 2 block/SM 起 256 行/block：
// 两个 block 能互相掩盖 `__syncthreads` 与 K/V 装载的空泡。
// P 每 warp 只占 512 槽（16x32 的 off_row 正好覆盖 0..511），block 总 smem 33 KB。
#define NWT16 16
#define NTH16 (NWT16 * 64)

template <bool HAS_MASK, bool HAS_DROP>
__global__ void __launch_bounds__(NTH16, 3) attn64w16(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O, int S, int T,
    float scale, const __half* __restrict__ M, float p_drop, float drop_scale,
    unsigned long long seed, int /*has_mask*/, int mask_shared) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                     // 两批 K，各 2048
  __half* vsm = ksm + 4096;             // 两批 V，各 2048
  __half* psm = vsm + 4096;             // 每 warp 一个 16x32 的 P（Q 中转发复用）
  __half* ones_sm = psm + NWT16 * 512;  // 16x16 全 1，用于 P@ones
  __shared__ float mls[NWT16 * 16];     // 惰性基准（每 warp 16 行）
  __shared__ float ssums[NWT16 * 16];   // dropout 模式下的分母（每 warp 16 行）

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;
  const int c0 = lane & 15;
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (16 * NWT16) + warp * 16;

  const __half* Qb = Q + (long long)bh * S * 64;
  __half* Ob = O + (long long)bh * S * 64;
  const __half* Kb0 = K + (long long)bh * T * 64;
  const __half* Vb0 = V + (long long)bh * T * 64;
  __half* pw = psm + warp * 512;

  fragment<matrix_a, 16, 16, 32, __half, row_major> a0, a1, pa0;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1, b2, b3, vbo;
  fragment<accumulator, 16, 16, 32, float> s0, s1;  // 批 A 分数（keys 0..31）
  fragment<accumulator, 16, 16, 32, float> t0, t1;  // 批 B 分数（keys 32..63）
  fragment<accumulator, 16, 16, 32, float> o0, o1, o2, o3, o4;

  // Q 装载：16 行 × 64 列，借 P 的 512 槽分两趟（每趟 512 个 half = 16x32）。
  {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      for (int idx = lane; idx < 512; idx += 64) {
        int m = idx >> 5, d = idx & 31;
        pw[off_row(m, d)] = Qb[(long long)(row_base + m) * 64 + h * 32 + d];
      }
      __syncwarp();
      if (h == 0) load_matrix_sync(a0, pw, 32);
      else        load_matrix_sync(a1, pw, 32);
      __syncwarp();
    }
  }

  fragment<accumulator, 16, 16, 32, float>* SF[2] = {&s0, &s1};
  fragment<accumulator, 16, 16, 32, float>* TF[2] = {&t0, &t1};
  fragment<accumulator, 16, 16, 32, float>* OF[5] = {&o0, &o1, &o2, &o3, &o4};

#pragma unroll
  for (int e = 0; e < 5; ++e) fill_fragment(*OF[e], 0.f);

  for (int idx = threadIdx.x; idx < NWT16 * 16; idx += NTH16) {
    mls[idx] = -1e30f;
    ssums[idx] = 0.f;
  }
  for (int idx = threadIdx.x; idx < 512; idx += NTH16) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[off_col(t, dd)] = __float2half(1.f);
  }
  __syncthreads();
  load_matrix_sync(vbo, ones_sm, 32);   // 分母矩阵不变，全程装一次

  const float sc2 = scale * AF_LOG2E;
  float* mlr = mls + warp * 16;
  float* ssr = ssums + warp * 16;

  // mask/dropout 的行基址循环外算一次，避免每轮做 (gl % S) 的整数除法。
  const __half* mbase = nullptr;
  long long dbase = 0;
  if (HAS_MASK) {
    const int mr0 = (mask_shared ? (row_base % S) : (bh * S + row_base)) + r0;
    mbase = M + (long long)mr0 * T;
  }
  if (HAS_DROP) dbase = (long long)(bh * S + row_base + r0) * T;

  for (int kt = 0; kt < T; kt += 64) {
    const bool hasB = (kt + 32 < T);
    load_kv(ksm, vsm, Kb0, Vb0, kt >> 5, NTH16);
    if (hasB) load_kv(ksm + 2048, vsm + 2048, Kb0, Vb0, (kt >> 5) + 1, NTH16);
    __syncthreads();

    // 批 A 的 QK
    fill_fragment(s0, 0.f);
    fill_fragment(s1, 0.f);
    load_matrix_sync(b0, ksm, 32);
    load_matrix_sync(b1, ksm + 512, 32);
    load_matrix_sync(b2, ksm + 1024, 32);
    load_matrix_sync(b3, ksm + 1536, 32);
    mma_sync(s0, a0, b0, s0);
    mma_sync(s0, a1, b1, s0);
    mma_sync(s1, a0, b2, s1);
    mma_sync(s1, a1, b3, s1);

    // 批 B 的 QK（T 不是 64 倍数时最后半批不存在，分数填 -1e30 使 exp 为 0）
    if (hasB) {
      fill_fragment(t0, 0.f);
      fill_fragment(t1, 0.f);
      load_matrix_sync(b0, ksm + 2048, 32);
      load_matrix_sync(b1, ksm + 2048 + 512, 32);
      load_matrix_sync(b2, ksm + 2048 + 1024, 32);
      load_matrix_sync(b3, ksm + 2048 + 1536, 32);
      mma_sync(t0, a0, b0, t0);
      mma_sync(t0, a1, b1, t0);
      mma_sync(t1, a0, b2, t1);
      mma_sync(t1, a1, b3, t1);
    } else {
      fill_fragment(t0, -1e30f);
      fill_fragment(t1, -1e30f);
    }

    // 风险检测单独一遍：dropout（含 mask+dropout）必须在这里、写 P 和累加分母
    // 之前把基准定下来，否则首轮 exp2 溢出成 inf，corr 乘出 NaN。
    // 纯 mask 无此约束，检测并进写 P 的一遍（mask 流量 3 倍 → 1 倍）。
    if (!HAS_MASK || HAS_DROP) {
      bool risk = false;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float lim = mlr[ri] + 8.f;
        float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
        float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
        if (HAS_MASK) {
          const __half* mr = mbase + (4 * q * T + kt);
          x0 += __half2float(mr[c0]) * AF_LOG2E;
          x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
          if (hasB) {
            x2 += __half2float(mr[32 + c0]) * AF_LOG2E;
            x3 += __half2float(mr[48 + c0]) * AF_LOG2E;
          }
        }
        risk |= fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) > lim;
      }
      if (__any_sync(0xffffffffu, risk)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          const float mn = mlr[ri];
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
          if (HAS_MASK) {
            const __half* mr = mbase + (4 * q * T + kt);
            x0 += __half2float(mr[c0]) * AF_LOG2E;
            x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
            if (hasB) {
              x2 += __half2float(mr[32 + c0]) * AF_LOG2E;
              x3 += __half2float(mr[48 + c0]) * AF_LOG2E;
            }
          }
          float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
          float corr = 1.f;
          if (mx > mn) {
            corr = exp2f(mn - mx);
            mlr[ri] = mx;
          }
#pragma unroll
          for (int e = 0; e < 5; ++e) OF[e]->x[q] *= corr;
          if (HAS_DROP && c0 == 0) ssr[ri] *= corr;
        }
      }
    }

    // 写 P 的一遍：纯 mask 时把检测并进来（risky），其余情况基准已经定好。
    __syncwarp();
    bool risky = false;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int ri = r0 + 4 * q;
      const float mn = mlr[ri];
      float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
      if (HAS_MASK) {
        const __half* mr = mbase + (4 * q * T + kt);
        x0 += __half2float(mr[c0]) * AF_LOG2E;
        x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
      }
      if (HAS_MASK && !HAS_DROP) risky |= fmaxf(x0, x1) > mn + 8.f;
      float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
      if (HAS_DROP) {
        const long long base = dbase + 4 * q * T + kt;
        float sv = p0 + p1;
        p0 = af_drop_w(p0, p_drop, drop_scale, seed, base + c0);
        p1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
        for (int off = 1; off < 16; off <<= 1)
          sv += __shfl_xor_sync(0xffffffffu, sv, off);
        if (c0 == 0) ssr[ri] += sv;
      }
      const int row = r0 + 4 * q;
      pw[off_row(row, c0)] = __float2half(p0);
      pw[off_row(row, 16 + c0)] = __float2half(p1);
    }
    // 纯 mask 的慢路径：上面用旧基准算了一遍，若越界就按新基准重算重写。
    if (HAS_MASK && !HAS_DROP) {
      if (__any_sync(0xffffffffu, risky)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          const __half* mr = mbase + (4 * q * T + kt);
          x0 += __half2float(mr[c0]) * AF_LOG2E;
          x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
          float mn = mlr[ri], corr = 1.f;
          float mx = fmaxf(x0, x1);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
          if (mx > mn) {
            corr = exp2f(mn - mx);
            mlr[ri] = mx;
            mn = mx;
          }
#pragma unroll
          for (int e = 0; e < 5; ++e) OF[e]->x[q] *= corr;
          const float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
          const int row = r0 + 4 * q;
          pw[off_row(row, c0)] = __float2half(p0);
          pw[off_row(row, 16 + c0)] = __float2half(p1);
        }
      }
    }
    __syncwarp();

    // 批 A 的 PV（b0..b3 与 QK 的 K 复用：QK 已算完，可以覆盖）
    load_matrix_sync(pa0, pw, 32);
    load_matrix_sync(b0, vsm, 32);
    load_matrix_sync(b1, vsm + 512, 32);
    load_matrix_sync(b2, vsm + 1024, 32);
    load_matrix_sync(b3, vsm + 1536, 32);
    mma_sync(o0, pa0, b0, o0);
    mma_sync(o1, pa0, b1, o1);
    mma_sync(o2, pa0, b2, o2);
    mma_sync(o3, pa0, b3, o3);
    if (!HAS_DROP) mma_sync(o4, pa0, vbo, o4);   // 分母：P @ ones

    if (hasB) {
      // 批 B 的 P 在这里现算（不占暂存寄存器），pw 已空出；结构与批 A 相同。
      __syncwarp();
      risky = false;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float mn = mlr[ri];
        float x0 = TF[0]->x[q] * sc2, x1 = TF[1]->x[q] * sc2;
        if (HAS_MASK) {
          const __half* mr = mbase + (4 * q * T + kt);
          x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
          x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
        }
        if (HAS_MASK && !HAS_DROP) risky |= fmaxf(x0, x1) > mn + 8.f;
        float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
        if (HAS_DROP) {
          const long long base = dbase + 4 * q * T + kt + 32;
          float sv = p0 + p1;
          p0 = af_drop_w(p0, p_drop, drop_scale, seed, base + c0);
          p1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            sv += __shfl_xor_sync(0xffffffffu, sv, off);
          if (c0 == 0) ssr[ri] += sv;
        }
        const int row = r0 + 4 * q;
        pw[off_row(row, c0)] = __float2half(p0);
        pw[off_row(row, 16 + c0)] = __float2half(p1);
      }
      if (HAS_MASK && !HAS_DROP) {
        if (__any_sync(0xffffffffu, risky)) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int ri = r0 + 4 * q;
            float x0 = TF[0]->x[q] * sc2, x1 = TF[1]->x[q] * sc2;
            const __half* mr = mbase + (4 * q * T + kt);
            x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
            x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
            float mn = mlr[ri], corr = 1.f;
            float mx = fmaxf(x0, x1);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            if (mx > mn) {
              corr = exp2f(mn - mx);
              mlr[ri] = mx;
              mn = mx;
            }
#pragma unroll
            for (int e = 0; e < 5; ++e) OF[e]->x[q] *= corr;
            const float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
            const int row = r0 + 4 * q;
            pw[off_row(row, c0)] = __float2half(p0);
            pw[off_row(row, 16 + c0)] = __float2half(p1);
          }
        }
      }
      __syncwarp();
      load_matrix_sync(pa0, pw, 32);
      load_matrix_sync(b0, vsm + 2048, 32);
      load_matrix_sync(b1, vsm + 2048 + 512, 32);
      load_matrix_sync(b2, vsm + 2048 + 1024, 32);
      load_matrix_sync(b3, vsm + 2048 + 1536, 32);
      mma_sync(o0, pa0, b0, o0);
      mma_sync(o1, pa0, b1, o1);
      mma_sync(o2, pa0, b2, o2);
      mma_sync(o3, pa0, b3, o3);
      if (!HAS_DROP) mma_sync(o4, pa0, vbo, o4);
    }
    __syncthreads();
  }

  // 归一化放在输出上：(p/l)@V == (p@V)/l。dropout 时 l 是 dropout 前的和。
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float den = HAS_DROP ? ssr[r0 + 4 * q] : OF[4]->x[q];
    const float inv = den != 0.f ? 1.f / den : 0.f;
    const int row = row_base + r0 + 4 * q;
#pragma unroll
    for (int e = 0; e < 4; ++e)
      Ob[row * 64 + e * 16 + c0] = __float2half(OF[e]->x[q] * inv);
  }
}


// ---------------------------------------------------------------- bench

static __half* mk_gauss(size_t n, int seed, float sd) {
  __half* h = (__half*)malloc(n * sizeof(__half));
  unsigned s = 2654435761u * (unsigned)seed;
  for (size_t i = 0; i < n; ++i) {
    float acc = 0.f;
    for (int k = 0; k < 12; ++k) {
      s = s * 1664525u + 1013904223u;
      acc += (float)((int)((s >> 16) & 0xffff) - 32768) / 32768.f;
    }
    h[i] = __float2half(acc * sd);
  }
  return h;
}

static float af_drop_w_host(float p, float p_drop, float drop_scale,
                            unsigned long long seed, long long idx) {
  unsigned long long z = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z ^= z >> 31;
  const float u = (float)((z >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
  return (u >= p_drop) ? p * drop_scale : 0.f;
}

static void ref_rows(const __half* Q, const __half* K, const __half* V, int S, int T,
                     int head, int r0, int r1, double scale, double* out,
                     const __half* M = nullptr, int mask_shared = 0, float p_drop = 0.f,
                     unsigned long long seed = 0) {
  const float drop_scale = p_drop > 0.f ? 1.f / (1.f - p_drop) : 1.f;
  double* sc = (double*)malloc(sizeof(double) * T);
  for (int r = r0; r < r1; ++r) {
    const long long mrow = M ? (mask_shared ? (r % S) : ((long long)head * S + r)) : 0;
    double mx = -1e300;
    for (int t = 0; t < T; ++t) {
      double s = 0;
      for (int d = 0; d < 64; ++d)
        s += (double)__half2float(Q[((long long)head * S + r) * 64 + d]) *
             (double)__half2float(K[((long long)head * T + t) * 64 + d]);
      sc[t] = s * scale;
      if (M) sc[t] += (double)__half2float(M[mrow * T + t]);
      if (sc[t] > mx) mx = sc[t];
    }
    double sum = 0;
    for (int t = 0; t < T; ++t) {
      sc[t] = exp(sc[t] - mx);
      sum += sc[t];
      if (p_drop > 0.f)
        sc[t] = af_drop_w_host((float)sc[t], p_drop, drop_scale, seed,
                               ((long long)head * S + r) * T + t);
    }
    for (int e = 0; e < 64; ++e) {
      double acc = 0;
      for (int t = 0; t < T; ++t)
        acc += sc[t] * (double)__half2float(V[((long long)head * T + t) * 64 + e]);
      out[(r - r0) * 64 + e] = acc / sum;
    }
  }
  free(sc);
}

typedef void (*attn_kern)(const __half*, const __half*, const __half*, __half*, int, int,
                          float, const __half*, float, float, unsigned long long, int, int);

// attn32：4096(KV) + 16*1024(P) + 512(ones)
static const size_t SMEM32 = (4096 + NWT * 1024 + 512) * 2;
// attn64：两批 K 各 2048、两批 V 各 2048、P 16*1024、ones 512
static const size_t SMEM64 = (4096 + 4096 + NWT * 1024 + 512) * 2;
// attn64p16：两批 K 各 2048、两批 V 各 2048、P 每 warp 512、ones 512
static const size_t SMEM16 = (4096 + 4096 + NWT16 * 512 + 512) * 2;
static void run_one(const char* tag, attn_kern kfn, size_t smem, const __half* hQ,
                    const __half* hK, const __half* hV, int bh0, int S, int T, float scale,
                    int bench_bh, const __half* hM = nullptr, int mask_shared = 0,
                    float p_drop = 0.f, unsigned long long seed = 0,
                    int nthreads = NTH, int rows_per_block = 32 * NWT) {
  const int nrows = bh0 * S;
  size_t n = (size_t)nrows * 64, nk = (size_t)bh0 * T * 64;
  __half *dQ, *dK, *dV, *dO, *dM = nullptr;
  cudaMalloc((void**)&dQ, n * 2); cudaMalloc((void**)&dK, nk * 2);
  cudaMalloc((void**)&dV, nk * 2); cudaMalloc((void**)&dO, n * 2);
  cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, nk * 2, cudaMemcpyHostToDevice);
  const int mrows = hM ? (mask_shared ? S : nrows) : 0;
  if (hM) {
    cudaMalloc((void**)&dM, (size_t)mrows * T * 2);
    cudaMemcpy(dM, hM, (size_t)mrows * T * 2, cudaMemcpyHostToDevice);
  }
  __half* hO = (__half*)malloc(n * 2);
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, reinterpret_cast<const void*>(kfn));
  dim3 g0(S / rows_per_block, bh0);
  const float drop_scale = p_drop > 0.f ? 1.f / (1.f - p_drop) : 1.f;
  kfn<<<g0, nthreads, smem>>>(dQ, dK, dV, dO, S, T, scale, dM, p_drop, drop_scale, seed,
                              hM ? 1 : 0, mask_shared);
  printf("[%s] regs=%d local=%d sync=%d mask=%d drop=%.2f\n", tag, at.numRegs,
         (int)at.localSizeBytes, (int)cudaDeviceSynchronize(), hM ? mrows : 0, p_drop);
  cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost);

  double ref[16 * 64], mxerr = 0, mxval = 0;
  int nbad = 0, nbad2 = 0, wr_h = -1, wr_row = -1, wr_e = -1;
  int warpbad[32] = {0};
  const int wpr = rows_per_block / 16;   // 诊断粒度：每 16 行一格
  for (int h = 0; h < bh0; ++h) {
    for (int blk = 0; blk < S / 16; ++blk) {
      int r0 = blk * 16;
      ref_rows(hQ, hK, hV, S, T, h, r0, r0 + 16, scale, ref, hM, mask_shared, p_drop, seed);
      for (int i = 0; i < 16 * 64; ++i) {
        double got = (double)__half2float(hO[((long long)h * S + r0 + i / 64) * 64 + i % 64]);
        if (got != got) { ++nbad; continue; }
        double e = fabs(got - ref[i]);
        if (e > 0.5) ++warpbad[(r0 + i / 64) / 16];
        if (e > mxerr) { mxerr = e; wr_h = h; wr_row = r0 + i / 64; wr_e = i % 64; }
        if (fabs(ref[i]) > mxval) mxval = fabs(ref[i]);
        if (e > 1e-3) ++nbad2;
      }
    }
  }
  printf("[%s] max abs err = %.6g (|ref|max = %.4g) bad = %d bad2 = %d/%d\n",
         tag, mxerr, mxval, nbad, nbad2, (int)n);
  if (mxerr > 0.5) {
    printf("  [%s] worst h=%d row=%d e=%d; per-warp(16行) bad:", tag, wr_h, wr_row, wr_e);
    for (int w = 0; w < wpr; ++w) printf(" %d", warpbad[w]);
    printf("\n");
    int hh = wr_h, rr = wr_row;
    ref_rows(hQ, hK, hV, S, T, hh, rr, rr + 1, scale, ref, hM, mask_shared, p_drop, seed);
    for (int e = 0; e < 8; ++e)
      printf("    e=%2d got=%9.5g ref=%9.5g\n", e,
             (double)__half2float(hO[((long long)hh * S + rr) * 64 + e]), ref[e]);
  }
  free(hO);

  if (bench_bh > 0) {
    size_t m = (size_t)bench_bh * S * 64;
    __half *bQ = mk_gauss(m, 11, 1.f), *bK = mk_gauss(m, 12, 1.f), *bV = mk_gauss(m, 13, 1.f);
    cudaMalloc((void**)&dQ, m * 2); cudaMalloc((void**)&dK, m * 2);
    cudaMalloc((void**)&dV, m * 2); cudaMalloc((void**)&dO, m * 2);
    cudaMemcpy(dQ, bQ, m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, bK, m * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, bV, m * 2, cudaMemcpyHostToDevice);
    double fl = (double)bench_bh * 4.0 * S * T * 64 * 2.0;
    const int reps = 20;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    dim3 g(S / rows_per_block, bench_bh);
    for (int i = 0; i < 3; ++i)
      kfn<<<g, nthreads, smem>>>(dQ, dK, dV, dO, S, T, scale, dM, p_drop, drop_scale, seed,
                                 hM ? 1 : 0, hM ? 1 : 0);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i)
      kfn<<<g, nthreads, smem>>>(dQ, dK, dV, dO, S, T, scale, dM, p_drop, drop_scale, seed,
                                 hM ? 1 : 0, hM ? 1 : 0);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[%s] %.4f ms  %.2f GFLOPS\n", tag, ms / reps, fl / (ms / reps * 1e-3) / 1e9);
    free(bQ); free(bK); free(bV);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
  } else {
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
  }
  if (dM) cudaFree(dM);
}

// 随机加性 mask：一部分位置 -12（近似屏蔽），其余小值扰动
static __half* mk_mask(int rows, int T, int seed) {
  __half* h = mk_gauss((size_t)rows * T, seed, 0.5f);
  unsigned s = 2654435761u * (unsigned)seed;
  for (int r = 0; r < rows; ++r)
    for (int t = 0; t < T; ++t) {
      s = s * 1664525u + 1013904223u;
      if (((s >> 20) & 15) == 0) h[(long long)r * T + t] = __float2half(-12.f);
    }
  return h;
}

int main(int argc, char** argv) {
  const int S = 512;
  int T = argc > 3 ? atoi(argv[3]) : 512;
  int bench_bh = argc > 1 ? atoi(argv[1]) : 0;
  float sd = argc > 2 ? (float)atof(argv[2]) : 1.f;
  const char* md = argc > 4 ? argv[4] : "";
  float scale = 0.125f;
  const int bh0 = 4;
  size_t n = (size_t)bh0 * S * 64, nk = (size_t)bh0 * T * 64;
  __half *hQ = mk_gauss(n, 1, sd), *hK = mk_gauss(nk, 2, sd), *hV = mk_gauss(nk, 3, sd);
  printf("sd = %.2f  T = %d  mode = %s\n", sd, T, md[0] ? md : "plain");
  const bool use_mask = strchr(md, 'm') != nullptr;
  const bool use_drop = strchr(md, 'd') != nullptr;
  const int mask_shared = use_mask && strchr(md, 'f') == nullptr ? 1 : 0;
  __half* hM = use_mask ? mk_mask(mask_shared ? S : bh0 * S, T, 21) : nullptr;
  const float p_drop = use_drop ? 0.1f : 0.f;
  const unsigned long long seed = 12345ULL;
  if (use_mask || use_drop) {
    attn_kern k64 = use_mask ? (use_drop ? attn64<true, true> : attn64<true, false>)
                             : attn64<false, true>;
    run_one("attn64", k64, SMEM64, hQ, hK, hV, bh0, S, T, scale, bench_bh, hM,
            mask_shared, p_drop, seed);
    attn_kern kw16 = use_mask ? (use_drop ? attn64w16<true, true> : attn64w16<true, false>)
                              : attn64w16<false, true>;
    run_one("attn64w16", kw16, SMEM16, hQ, hK, hV, bh0, S, T, scale, bench_bh, hM,
            mask_shared, p_drop, seed, NTH16, 16 * NWT16);
  } else {
    run_one("attn32", attn32, SMEM32, hQ, hK, hV, bh0, S, T, scale, bench_bh);
    run_one("attn64p", attn64p, SMEM64, hQ, hK, hV, bh0, S, T, scale,
            bench_bh);
    run_one("attn64w16", attn64w16<false, false>, SMEM16, hQ, hK, hV, bh0, S, T,
            scale, bench_bh, nullptr, 0, 0.f, 0, NTH16, 16 * NWT16);
  }
  free(hQ); free(hK); free(hV);
  if (hM) free(hM);
  return 0;
}
