// D=128 单 block 结构探针：与 attn64w16 完全同构（16 warp × 16 行、每轮 64 key、
// 两批 QK 并行、惰性基准、P@ones 求分母），仅把 D 从 64 扩到 128。
// 只实现 plain（无 mask / dropout），目的是测"smem 49KB、2 block/SM"下的真实吞吐。
//
// 编译运行：
//   export COREX_PATH=/usr/local/corex-5.0.0
//   export LD_LIBRARY_PATH=$COREX_PATH/lib64:$LD_LIBRARY_PATH
//   $COREX_PATH/bin/clang++ -x ivcore --cuda-path=$COREX_PATH -O2 \
//       -o _af_d128 tcu_d128_probe.cu -L$COREX_PATH/lib -lcudart
//   ./_af_d128 256 512 20      # B S reps
//   ./_af_d128 4 512 2 8       # 第 4 个参数 = 对拍校验行数
//
// 实测（B=256 S=T=512）：1.1480 ms / 29.9 TFLOPS，rel 4.1e-4
// （物化 1.4944 ms，官方 ixattention 0.6110 ms，见 REMIND.md 第八轮）。
#include <crt/iluvatar_mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace ixix::wmma;

__host__ __device__ __forceinline__ int off_row(int row, int col) {
  int i = col >> 4, c = col & 15, r = row >> 1;
  return (i * 256 + r / 4 * 128) + ((r * 32 + c * 2 + (row & 1) + i * 64) & 127);
}
__host__ __device__ __forceinline__ int off_col(int row, int col) {
  int r = row >> 1;
  return (col >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (col >> 3)) * 16 +
         ((r >> 2 & 1) ^ (col >> 2 & 1)) * 8 + (col & 3) * 2 + (row & 1);
}

#define NWT16 16
#define NTH16 (NWT16 * 64)
#define AF_LOG2E 1.4426950408889634f

// 32 个 key、D=128 → 每批 4096 half（32x128）。K 每线程 2 个 d（一个 unsigned）；
// V 每线程 2 个 d（half2 读、标量写两次，D=64 版同构）。
__device__ __forceinline__ void load_kv128(__half* ks, __half* vs, const __half* Kb0,
                                           const __half* Vb0, int kb, int nth) {
  const long long o = (long long)kb * 4096;
  const unsigned* K2 = (const unsigned*)(Kb0 + o);
  unsigned* ks2 = (unsigned*)ks;
  for (int idx = threadIdx.x; idx < 2048; idx += nth) {
    const int t = idx >> 6, d = (idx & 63) * 2;
    ks2[(((t >> 4) * 4 + (d >> 5)) * 512 + off_col(d & 31, t & 15)) >> 1] = K2[idx];
    const int g = (t << 7) | d;
    const __half2 vv = *(const __half2*)(Vb0 + o + g);
    vs[(d >> 4) * 512 + off_col(t, d & 15)] = __low2half(vv);
    vs[(d >> 4) * 512 + off_col(t, (d + 1) & 15)] = __high2half(vv);
  }
}

__global__ void __launch_bounds__(NTH16, 2) attn128w16(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O, int S, int T,
    float scale) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                       // 两批 K，各 4096
  __half* vsm = ksm + 8192;               // 两批 V，各 4096
  __half* psm = vsm + 8192;               // 每 warp 一个 16x32 的 P
  __half* ones_sm = psm + NWT16 * 512;    // 16x16 全 1
  __shared__ float mls[NWT16 * 16];

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;
  const int c0 = lane & 15;
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (16 * NWT16) + warp * 16;

  const __half* Qb = Q + (long long)bh * S * 128;
  __half* Ob = O + (long long)bh * S * 128;
  const __half* Kb0 = K + (long long)bh * T * 128;
  const __half* Vb0 = V + (long long)bh * T * 128;
  __half* pw = psm + warp * 512;

  fragment<matrix_a, 16, 16, 32, __half, row_major> a0, a1, a2, a3, pa0;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1, b2, b3, vbo;
  fragment<accumulator, 16, 16, 32, float> s0, s1;
  fragment<accumulator, 16, 16, 32, float> t0, t1;
  fragment<accumulator, 16, 16, 32, float> o0, o1, o2, o3, o4, o5, o6, o7, o8;

  // Q 装载：16 行 × 128 列，借 P 的 512 槽分四趟（每趟 16x32）。
  {
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      for (int idx = lane; idx < 512; idx += 64) {
        int m = idx >> 5, d = idx & 31;
        pw[off_row(m, d)] = Qb[(long long)(row_base + m) * 128 + h * 32 + d];
      }
      __syncwarp();
      if (h == 0) load_matrix_sync(a0, pw, 32);
      else if (h == 1) load_matrix_sync(a1, pw, 32);
      else if (h == 2) load_matrix_sync(a2, pw, 32);
      else load_matrix_sync(a3, pw, 32);
      __syncwarp();
    }
  }

  fragment<accumulator, 16, 16, 32, float>* SF[2] = {&s0, &s1};
  fragment<accumulator, 16, 16, 32, float>* TF[2] = {&t0, &t1};
  fragment<accumulator, 16, 16, 32, float>* OF[9] = {&o0, &o1, &o2, &o3, &o4,
                                                     &o5, &o6, &o7, &o8};

#pragma unroll
  for (int e = 0; e < 9; ++e) fill_fragment(*OF[e], 0.f);

  for (int idx = threadIdx.x; idx < NWT16 * 16; idx += NTH16) mls[idx] = -1e30f;
  for (int idx = threadIdx.x; idx < 512; idx += NTH16) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[off_col(t, dd)] = __float2half(1.f);
  }
  __syncthreads();
  load_matrix_sync(vbo, ones_sm, 32);

  const float sc2 = scale * AF_LOG2E;
  float* mlr = mls + warp * 16;

  for (int kt = 0; kt < T; kt += 64) {
    const bool hasB = (kt + 32 < T);
    load_kv128(ksm, vsm, Kb0, Vb0, kt >> 5, NTH16);
    if (hasB) load_kv128(ksm + 4096, vsm + 4096, Kb0, Vb0, (kt >> 5) + 1, NTH16);
    __syncthreads();

    // 批 A 的 QK：4 段 D 维
    fill_fragment(s0, 0.f);
    fill_fragment(s1, 0.f);
    load_matrix_sync(b0, ksm, 32);
    load_matrix_sync(b1, ksm + 512, 32);
    load_matrix_sync(b2, ksm + 1024, 32);
    load_matrix_sync(b3, ksm + 1536, 32);
    mma_sync(s0, a0, b0, s0);
    mma_sync(s0, a1, b1, s0);
    mma_sync(s0, a2, b2, s0);
    mma_sync(s0, a3, b3, s0);
    load_matrix_sync(b0, ksm + 2048, 32);
    load_matrix_sync(b1, ksm + 2560, 32);
    load_matrix_sync(b2, ksm + 3072, 32);
    load_matrix_sync(b3, ksm + 3584, 32);
    mma_sync(s1, a0, b0, s1);
    mma_sync(s1, a1, b1, s1);
    mma_sync(s1, a2, b2, s1);
    mma_sync(s1, a3, b3, s1);

    if (hasB) {
      fill_fragment(t0, 0.f);
      fill_fragment(t1, 0.f);
      load_matrix_sync(b0, ksm + 4096, 32);
      load_matrix_sync(b1, ksm + 4096 + 512, 32);
      load_matrix_sync(b2, ksm + 4096 + 1024, 32);
      load_matrix_sync(b3, ksm + 4096 + 1536, 32);
      mma_sync(t0, a0, b0, t0);
      mma_sync(t0, a1, b1, t0);
      mma_sync(t0, a2, b2, t0);
      mma_sync(t0, a3, b3, t0);
      load_matrix_sync(b0, ksm + 4096 + 2048, 32);
      load_matrix_sync(b1, ksm + 4096 + 2560, 32);
      load_matrix_sync(b2, ksm + 4096 + 3072, 32);
      load_matrix_sync(b3, ksm + 4096 + 3584, 32);
      mma_sync(t1, a0, b0, t1);
      mma_sync(t1, a1, b1, t1);
      mma_sync(t1, a2, b2, t1);
      mma_sync(t1, a3, b3, t1);
    } else {
      fill_fragment(t0, -1e30f);
      fill_fragment(t1, -1e30f);
    }

    // 惰性基准预检（plain 专用 pass）
    {
      bool risk = false;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float lim = mlr[ri] + 8.f;
        float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
        float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
        risk |= fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) > lim;
      }
      if (__any_sync(0xffffffffu, risk)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          const float mn = mlr[ri];
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
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
          for (int e = 0; e < 9; ++e) OF[e]->x[q] *= corr;
        }
      }
    }

    __syncwarp();
    // 批 A 写 P
    {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float mn = mlr[ri];
        float p0 = exp2f(SF[0]->x[q] * sc2 - mn);
        float p1 = exp2f(SF[1]->x[q] * sc2 - mn);
        const int row = r0 + 4 * q;
        pw[off_row(row, c0)] = __float2half(p0);
        pw[off_row(row, 16 + c0)] = __float2half(p1);
      }
    }
    __syncwarp();

    // 批 A 的 PV：8 个 E 分块 + 分母
    load_matrix_sync(pa0, pw, 32);
    load_matrix_sync(b0, vsm, 32);
    load_matrix_sync(b1, vsm + 512, 32);
    load_matrix_sync(b2, vsm + 1024, 32);
    load_matrix_sync(b3, vsm + 1536, 32);
    mma_sync(o0, pa0, b0, o0);
    mma_sync(o1, pa0, b1, o1);
    mma_sync(o2, pa0, b2, o2);
    mma_sync(o3, pa0, b3, o3);
    load_matrix_sync(b0, vsm + 2048, 32);
    load_matrix_sync(b1, vsm + 2560, 32);
    load_matrix_sync(b2, vsm + 3072, 32);
    load_matrix_sync(b3, vsm + 3584, 32);
    mma_sync(o4, pa0, b0, o4);
    mma_sync(o5, pa0, b1, o5);
    mma_sync(o6, pa0, b2, o6);
    mma_sync(o7, pa0, b3, o7);
    mma_sync(o8, pa0, vbo, o8);

    if (hasB) {
      __syncwarp();
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float mn = mlr[ri];
        float p0 = exp2f(TF[0]->x[q] * sc2 - mn);
        float p1 = exp2f(TF[1]->x[q] * sc2 - mn);
        const int row = r0 + 4 * q;
        pw[off_row(row, c0)] = __float2half(p0);
        pw[off_row(row, 16 + c0)] = __float2half(p1);
      }
      __syncwarp();
      load_matrix_sync(pa0, pw, 32);
      load_matrix_sync(b0, vsm + 4096, 32);
      load_matrix_sync(b1, vsm + 4096 + 512, 32);
      load_matrix_sync(b2, vsm + 4096 + 1024, 32);
      load_matrix_sync(b3, vsm + 4096 + 1536, 32);
      mma_sync(o0, pa0, b0, o0);
      mma_sync(o1, pa0, b1, o1);
      mma_sync(o2, pa0, b2, o2);
      mma_sync(o3, pa0, b3, o3);
      load_matrix_sync(b0, vsm + 4096 + 2048, 32);
      load_matrix_sync(b1, vsm + 4096 + 2560, 32);
      load_matrix_sync(b2, vsm + 4096 + 3072, 32);
      load_matrix_sync(b3, vsm + 4096 + 3584, 32);
      mma_sync(o4, pa0, b0, o4);
      mma_sync(o5, pa0, b1, o5);
      mma_sync(o6, pa0, b2, o6);
      mma_sync(o7, pa0, b3, o7);
      mma_sync(o8, pa0, vbo, o8);
    }
    __syncthreads();
  }

#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float den = OF[8]->x[q];
    const float inv = den != 0.f ? 1.f / den : 0.f;
    const int row = row_base + r0 + 4 * q;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      Ob[row * 128 + e * 16 + c0] = __float2half(OF[e]->x[q] * inv);
  }
}

// ---------------------------------------------------------------- bench
static const size_t SMEM128 = (4096 + 4096 + 4096 + 4096 + NWT16 * 512 + 512) * 2;

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
      for (int d = 0; d < 128; ++d)
        s += (double)__half2float(Q[((long long)head * S + r) * 128 + d]) *
             (double)__half2float(K[((long long)head * T + t) * 128 + d]);
      sc[t] = s * scale;
      if (sc[t] > mx) mx = sc[t];
    }
    double sum = 0;
    for (int t = 0; t < T; ++t) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
    for (int e = 0; e < 128; ++e) {
      double acc = 0;
      for (int t = 0; t < T; ++t)
        acc += sc[t] * (double)__half2float(V[((long long)head * T + t) * 128 + e]);
      out[(r - r0) * 128 + e] = acc / sum;
    }
  }
  free(sc);
}

int main(int argc, char** argv) {
  const int B = argc > 1 ? atoi(argv[1]) : 256;
  const int S = argc > 2 ? atoi(argv[2]) : 512;
  const int reps = argc > 3 ? atoi(argv[3]) : 20;
  const int T = S;
  const float scale = 1.f / sqrtf(128.f);
  printf("B=%d S=%d T=%d D=128 reps=%d\n", B, S, T, reps);

  const size_t nq = (size_t)B * S * 128, nk = (size_t)B * T * 128;
  __half* hQ = mk_gauss(nq, 1, 0.5f);
  __half* hK = mk_gauss(nk, 2, 0.5f);
  __half* hV = mk_gauss(nk, 3, 0.5f);
  __half *dQ, *dK, *dV, *dO;
  cudaMalloc((void**)&dQ, nq * 2); cudaMalloc((void**)&dK, nk * 2);
  cudaMalloc((void**)&dV, nk * 2); cudaMalloc((void**)&dO, nq * 2);
  cudaMemcpy(dQ, hQ, nq * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK, nk * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, nk * 2, cudaMemcpyHostToDevice);
  cudaFuncSetAttribute(attn128w16, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM128);
  dim3 grid((S + 255) / 256, B);

  for (int i = 0; i < 3; ++i)
    attn128w16<<<grid, NTH16, SMEM128>>>(dQ, dK, dV, dO, S, T, scale);
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < reps; ++i)
    attn128w16<<<grid, NTH16, SMEM128>>>(dQ, dK, dV, dO, S, T, scale);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  ms /= reps;
  double flops = 4.0 * B * S * T * 128.0;
  printf("%.4f ms  %.1f TFLOPS\n", ms, flops / ms * 1e-9);

  if (argc > 4) {  // 对拍模式
    __half* hO = (__half*)malloc(nq * 2);
    cudaMemcpy(hO, dO, nq * 2, cudaMemcpyDeviceToHost);
    const int Nr = atoi(argv[4]);
    double* ref = (double*)malloc(sizeof(double) * Nr * 128);
    double maxe = 0, maxv = 0;
    for (int h = 0; h < (B < 4 ? B : 4); ++h) {
      ref_rows(hQ, hK, hV, S, T, h, 0, Nr, scale, ref);
      for (int r = 0; r < Nr; ++r)
        for (int e = 0; e < 128; ++e) {
          double g = __half2float(hO[((long long)h * S + r) * 128 + e]);
          double d = fabs(g - ref[(long long)r * 128 + e]);
          if (d > maxe) maxe = d;
          if (fabs(ref[(long long)r * 128 + e]) > maxv) maxv = fabs(ref[(long long)r * 128 + e]);
        }
      printf("head %d max_err=%.3e rel=%.2e\n", h, maxe, maxe / (maxv + 1e-9));
    }
    free(ref); free(hO);
  }
  free(hQ); free(hK); free(hV);
  return 0;
}
