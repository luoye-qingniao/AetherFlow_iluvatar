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
//   - 每轮 32 个 key：8 次 mma 出 QK，softmax 就地写进本 warp 的 P smem，
//     再 10 次 mma 出 PV（8 个 E 分块 + 2 个 P@ones 求分母）。
//   - max 必须每轮归约：降频会让 exp 溢出 half 上限（std≥2 的 f16 输入实测
//     有 30%~50% 的元素变成 inf/nan）。
//   - 分母走 P@ones 的 mma，比 shuffle 归约 sum 快 13%。
//
// 512x512x64、bh=256：0.8309 ms / 41350 GFLOPS（GEMM 物化基线 1.394 ms）。

#include <crt/iluvatar_mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

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

// 32 个 key 的 K/V 重排进 swizzle 布局的 smem。每次 2048 个 half（64x32）。
__device__ __forceinline__ void load_kv(__half* ks, __half* vs, const __half* Kb0,
                                        const __half* Vb0, int kt) {
  const long long o = (long long)kt * 2048;
  for (int idx = threadIdx.x; idx < 2048; idx += NTH) {
    int t = idx >> 6, d = idx & 63;
    ks[((t >> 4) * 2 + (d >> 5)) * 512 + off_col(d & 31, t & 15)] = Kb0[o + idx];
    vs[(d >> 4) * 512 + off_col(t, d & 15)] = Vb0[o + idx];
  }
}

__global__ void __launch_bounds__(NTH) attn32(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int S, int T, float scale) {
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

#pragma unroll
    for (int g = 0; g < 2; ++g) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float x0 = SF[g][0]->x[q] * scale, x1 = SF[g][1]->x[q] * scale;
        // 每轮归约 max：降频会在 std>=2 的 f16 输入上溢出（见 REMIND）。
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

static void ref_rows(const __half* Q, const __half* K, const __half* V, int S, int T,
                     int head, int r0, int r1, double scale, double* out) {
  double* sc = (double*)malloc(sizeof(double) * T);
  for (int r = r0; r < r1; ++r) {
    double mx = -1e300;
    for (int t = 0; t < T; ++t) {
      double s = 0;
      for (int d = 0; d < 64; ++d)
        s += (double)__half2float(Q[((long long)head * S + r) * 64 + d]) *
             (double)__half2float(K[((long long)head * T + t) * 64 + d]);
      sc[t] = s * scale;
      if (sc[t] > mx) mx = sc[t];
    }
    double sum = 0;
    for (int t = 0; t < T; ++t) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
    for (int e = 0; e < 64; ++e) {
      double acc = 0;
      for (int t = 0; t < T; ++t)
        acc += sc[t] * (double)__half2float(V[((long long)head * T + t) * 64 + e]);
      out[(r - r0) * 64 + e] = acc / sum;
    }
  }
  free(sc);
}

static const size_t SMEM = (4096 + NWT * 1024 + 512) * 2;

int main(int argc, char** argv) {
  const int S = 512, T = 512;
  int bench_bh = argc > 1 ? atoi(argv[1]) : 0;
  float scale = 0.125f;
  cudaFuncAttributes at;
  cudaFuncGetAttributes(&at, (const void*)attn32);
  printf("regs=%d local=%d\n", at.numRegs, (int)at.localSizeBytes);

  const int bh0 = 4;
  size_t n = (size_t)bh0 * S * 64;
  __half *hQ = mk_gauss(n, 1, 1.f), *hK = mk_gauss(n, 2, 1.f), *hV = mk_gauss(n, 3, 1.f);
  __half *dQ, *dK, *dV, *dO;
  cudaMalloc((void**)&dQ, n * 2); cudaMalloc((void**)&dK, n * 2);
  cudaMalloc((void**)&dV, n * 2); cudaMalloc((void**)&dO, n * 2);
  cudaMemcpy(dQ, hQ, n * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, n * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, n * 2, cudaMemcpyHostToDevice);
  __half* hO = (__half*)malloc(n * 2);
  dim3 g0(S / (32 * NWT), bh0);
  attn32<<<g0, NTH, SMEM>>>(dQ, dK, dV, dO, S, T, scale);
  printf("sync err = %d\n", (int)cudaDeviceSynchronize());
  cudaMemcpy(hO, dO, n * 2, cudaMemcpyDeviceToHost);

  double ref[16 * 64], mxerr = 0, mxval = 0;
  int nbad = 0;
  for (int h = 0; h < bh0; ++h) {
    for (int blk = 0; blk < S / 16; ++blk) {
      int r0 = blk * 16;
      ref_rows(hQ, hK, hV, S, T, h, r0, r0 + 16, scale, ref);
      for (int i = 0; i < 16 * 64; ++i) {
        double got = (double)__half2float(hO[((long long)h * S + r0 + i / 64) * 64 + i % 64]);
        if (got != got) { ++nbad; continue; }
        double e = fabs(got - ref[i]);
        if (e > mxerr) mxerr = e;
        if (fabs(ref[i]) > mxval) mxval = fabs(ref[i]);
      }
    }
  }
  printf("max abs err = %.6g (|ref|max = %.4g) bad = %d/%d\n", mxerr, mxval, nbad, (int)n);
  free(hQ); free(hK); free(hV); free(hO);
  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

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
    dim3 g(S / (32 * NWT), bench_bh);
    for (int i = 0; i < 3; ++i) attn32<<<g, NTH, SMEM>>>(dQ, dK, dV, dO, S, T, scale);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) attn32<<<g, NTH, SMEM>>>(dQ, dK, dV, dO, S, T, scale);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    printf("[bench] %.4f ms  %.2f GFLOPS\n", ms / reps, fl / (ms / reps * 1e-3) / 1e9);
  }
  return 0;
}
