// 纯 mma 形状微基准：比较 16x16x32 与 16x32x32 的单条成本。
// 只用 fill_fragment 造数据（不碰 smem 布局），因此不涉及 load_matrix_sync。
// 编译运行：
//   export COREX_PATH=/usr/local/corex-5.0.0
//   export LD_LIBRARY_PATH=$COREX_PATH/lib64:$COREX_PATH/lib:$LD_LIBRARY_PATH
//   $COREX_PATH/bin/clang++ -x ivcore --cuda-path=$COREX_PATH -O2 \
//       -o _af_shape tcu_mma_shape.cu -L$COREX_PATH/lib -lcudart
//   ./_af_shape 2000000
#include <crt/iluvatar_mma.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace ixix::wmma;

#define NWARP 4
#define NTH (NWARP * 64)

// VAR=0: 每轮 4 条 16x16x32（模拟 D=128 单批 QK 的依赖链）
// VAR=1: 每轮 4 条 16x32x32（若单条成本≈1 条 16x16，则 QK 指令可减半）
// VAR=2: 每轮 18 条 16x16x32（模拟 PV 全段）
// VAR=3: 每轮 4 条 32x16x32（依赖链）；FLOP 是 16x16x32 的 2 倍
// VAR=4: 每轮 4 条 32x32x16（依赖链）；FLOP 同上
// VAR=5: 每轮 4 条 32x32x16（4 个独立 acc，排除依赖链影响）
template <int VAR>
__global__ void __launch_bounds__(NTH) micro(int R, float* out) {
  fragment<accumulator, 16, 16, 32, float> c0, c1, c2, c3;
  fragment<accumulator, 16, 32, 32, float> d0;
  fragment<matrix_a, 16, 16, 32, __half, row_major> a0, a1;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1;
  fragment<matrix_a, 16, 32, 32, __half, row_major> e0;
  fragment<matrix_b, 16, 32, 32, __half, row_major> f0;
  // 2 倍 FLOP 的两种形状
  fragment<accumulator, 32, 16, 32, float> g0, g1;
  fragment<matrix_a, 32, 16, 32, __half, col_major> ga;
  fragment<matrix_b, 32, 16, 32, __half, col_major> gb;
  fragment<accumulator, 32, 32, 16, float> h0, h1, h2, h3;
  fragment<matrix_a, 32, 32, 16, __half, col_major> ha;
  fragment<matrix_b, 32, 32, 16, __half, row_major> hb;

  fill_fragment(c0, 0.f); fill_fragment(c1, 0.f);
  fill_fragment(c2, 0.f); fill_fragment(c3, 0.f);
  fill_fragment(d0, 0.f);
  fill_fragment(a0, (unsigned)0x3c003c00u);
  fill_fragment(a1, (unsigned)0x3c003c00u);
  fill_fragment(b0, (unsigned)0x3c003c00u);
  fill_fragment(b1, (unsigned)0x3c003c00u);
  fill_fragment(e0, (unsigned)0x3c003c00u);
  fill_fragment(f0, (unsigned)0x3c003c00u);
  fill_fragment(g0, 0.f); fill_fragment(g1, 0.f);
  fill_fragment(ga, (unsigned)0x3c003c00u);
  fill_fragment(gb, (unsigned)0x3c003c00u);
  fill_fragment(h0, 0.f); fill_fragment(h1, 0.f);
  fill_fragment(h2, 0.f); fill_fragment(h3, 0.f);
  fill_fragment(ha, (unsigned)0x3c003c00u);
  fill_fragment(hb, (unsigned)0x3c003c00u);

  for (int r = 0; r < R; ++r) {
    if (VAR == 0) {
      mma_sync(c0, a0, b0, c0);
      mma_sync(c0, a1, b1, c0);
      mma_sync(c0, a0, b1, c0);
      mma_sync(c0, a1, b0, c0);
    } else if (VAR == 1) {
      mma_sync(d0, e0, f0, d0);
      mma_sync(d0, e0, f0, d0);
      mma_sync(d0, e0, f0, d0);
      mma_sync(d0, e0, f0, d0);
    } else if (VAR == 2) {
      mma_sync(c0, a0, b0, c0);
      mma_sync(c1, a0, b1, c1);
      mma_sync(c2, a1, b0, c2);
      mma_sync(c3, a1, b1, c3);
      mma_sync(c0, a0, b1, c0);
      mma_sync(c1, a0, b0, c1);
      mma_sync(c2, a1, b1, c2);
      mma_sync(c3, a1, b0, c3);
      mma_sync(c0, a0, b0, c0);
      mma_sync(c1, a0, b1, c1);
      mma_sync(c2, a1, b0, c2);
      mma_sync(c3, a1, b1, c3);
      mma_sync(c0, a0, b1, c0);
      mma_sync(c1, a0, b0, c1);
      mma_sync(c2, a1, b1, c2);
      mma_sync(c3, a1, b0, c3);
      mma_sync(c0, a1, b0, c0);
      mma_sync(c1, a1, b1, c1);
    } else if (VAR == 3) {
      mma_sync(g0, ga, gb, g0);
      mma_sync(g0, ga, gb, g0);
      mma_sync(g0, ga, gb, g0);
      mma_sync(g0, ga, gb, g0);
    } else if (VAR == 4) {
      mma_sync(h0, ha, hb, h0);
      mma_sync(h0, ha, hb, h0);
      mma_sync(h0, ha, hb, h0);
      mma_sync(h0, ha, hb, h0);
    } else {
      mma_sync(h0, ha, hb, h0);
      mma_sync(h1, ha, hb, h1);
      mma_sync(h2, ha, hb, h2);
      mma_sync(h3, ha, hb, h3);
    }
  }

  // 保活
  float s = 0.f;
  s += c0.x[0] + c1.x[0] + c2.x[0] + c3.x[0] + d0.x[0];
  s += g0.x[0] + g1.x[0] + h0.x[0] + h1.x[0] + h2.x[0] + h3.x[0];
  if (threadIdx.x == 0 && s == 12345.678f) out[0] = s;
}

template <int VAR>
static void run(const char* name, int R, float* dOut, int nops) {
  const int grid = 16 * 2;  // 32 个 block，铺满 16 SM
  for (int i = 0; i < 3; ++i) micro<VAR><<<grid, NTH>>>(R, dOut);
  cudaDeviceSynchronize();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < 10; ++i) micro<VAR><<<grid, NTH>>>(R, dOut);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  ms /= 10.f;
  // 每 warp 一轮做 nops 条；全设备共 grid*NWARP 个 warp 并行。
  // 单条摊销时间（每 warp 视角）= ms / (R*nops)，再乘并发 warp 数得到设备口径。
  const double per_warp_ns = (double)ms * 1e6 / ((double)R * (double)nops);
  printf("  %-26s %.4f ms  %2d 条/轮  每warp %.4f ns/条\n", name, (double)ms,
         nops, per_warp_ns);
}

int main(int argc, char** argv) {
  const int R = argc > 1 ? atoi(argv[1]) : 2000000;
  float* dOut;
  cudaMalloc((void**)&dOut, sizeof(float));
  printf("mma 形状微基准 R=%d grid=32 NWARP=4\n", R);
  run<0>("VAR0 4x mma16x16x32", R, dOut, 4);
  run<1>("VAR1 4x mma16x32x32", R, dOut, 4);
  run<2>("VAR2 18x mma16x16x32", R, dOut, 18);
  run<3>("VAR3 4x mma32x16x32", R, dOut, 4);
  run<4>("VAR4 4x mma32x32x16", R, dOut, 4);
  run<5>("VAR5 4x mma32x32x16 独立acc", R, dOut, 4);
  cudaFree(dOut);
  return 0;
}
