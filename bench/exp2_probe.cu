// exp2 吞吐 / 精度探针（Iluvatar MR-V100）
//
// 目的：验证 flash attention 的瓶颈是否在 SFU 的 exp2。
// bench/tcu_flash_attn.cu 的 plain 每轮每线程 32 次 exp2，全问题共 67.1M 次；
// 若 SFU 吞吐 16 SM × 4/cycle × 1.5 GHz = 96 G/s，单项就要 0.70 ms，
// 与实测 0.7111 ms 相当。
//
// 同时测一个 ALU 版 exp2（floor + 4 阶 FMA + 位构造 2^n）的吞吐与误差。
// 编译：
//   clang++ -x ivcore --cuda-path=/usr/local/corex-5.0.0 -O2 \
//       -o exp2_probe exp2_probe.cu -L/usr/local/corex-5.0.0/lib -lcudart
// 运行：LD_LIBRARY_PATH=/usr/local/corex-5.0.0/lib ./exp2_probe

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// 2^x 的 ALU 版本：n = floor(x)，f = x - n ∈ [0,1) 走 4 阶多项式，
// 2^n 直接拼 float 指数位。x ∈ [-126, 0] 时与 exp2f 的相对误差 < 1e-3。
__device__ __forceinline__ float exp2_alu(float x) {
    const float n = floorf(x);
    const float f = x - n;
    float p = 0.0089893f;
    p = fmaf(p, f, 0.0555041f);
    p = fmaf(p, f, 0.2402265f);
    p = fmaf(p, f, 0.6931472f);
    p = fmaf(p, f, 1.0f);
    int e = (int)n + 127;
    e = e < 1 ? 1 : (e > 254 ? 254 : e);
    return p * __int_as_float(e << 23);
}

template <bool ALU>
__device__ __forceinline__ float ex(float x) {
    return ALU ? exp2_alu(x) : exp2f(x);
}

// 8 路独立累加，避免依赖链把延迟当成吞吐。
template <bool ALU>
__global__ void __launch_bounds__(1024, 2) k_bench(float* out, int iters) {
    const float b = -0.001f * (1.f + (float)(threadIdx.x & 7));
    float a0 = b, a1 = b - 0.002f, a2 = b - 0.004f, a3 = b - 0.006f;
    float a4 = b - 0.008f, a5 = b - 0.010f, a6 = b - 0.012f, a7 = b - 0.014f;
    for (int i = 0; i < iters; ++i) {
        a0 = ex<ALU>(a0 - 1e-3f);
        a1 = ex<ALU>(a1 - 1e-3f);
        a2 = ex<ALU>(a2 - 1e-3f);
        a3 = ex<ALU>(a3 - 1e-3f);
        a4 = ex<ALU>(a4 - 1e-3f);
        a5 = ex<ALU>(a5 - 1e-3f);
        a6 = ex<ALU>(a6 - 1e-3f);
        a7 = ex<ALU>(a7 - 1e-3f);
    }
    const float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (s == 12345.678f) { out[threadIdx.x] = s; }
}

// 最大相对误差（ppm），扫描 [lo, hi]。
__global__ void k_err(int* err, int n, float lo, float hi) {
    int m = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = lo + (hi - lo) * (float)i / (float)n;
        const float a = exp2f(x);
        const float b = exp2_alu(x);
        const int r = (int)(fabsf(b - a) / a * 1e6f);
        m = r > m ? r : m;
    }
    atomicMax(err, m);
}

template <bool ALU>
static void run(const char* name, int blocks, int threads, int iters) {
    float* d_out = nullptr;
    cudaMalloc(&d_out, threads * sizeof(float));
    k_bench<ALU><<<blocks, threads>>>(d_out, 1);  // 预热
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float best = 1e9f;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(e0);
        k_bench<ALU><<<blocks, threads>>>(d_out, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        best = ms < best ? ms : best;
    }
    const double total = (double)blocks * threads * 8.0 * iters;
    printf("%-8s %8.3f ms  %8.1f Gexp/s  (%.1f exp/SM/cycle @1.5GHz)\n",
           name, best, total / (best * 1e6),
           total / (best * 1e-3) / 16.0 / 1.5e9);
    cudaFree(d_out);
}

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    printf("device: %s  SM=%d  warp=%d  clock=%.0f MHz\n", p.name, p.multiProcessorCount,
           p.warpSize, p.clockRate / 1000.0);

    int* d_err = nullptr;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemset(d_err, 0, sizeof(int));
    k_err<<<1, 256>>>(d_err, 65536, -10.f, 0.f);
    cudaDeviceSynchronize();
    int h_err = 0;
    cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost);
    printf("exp2_alu max rel err on [-10,0]: %d ppm\n", h_err);
    cudaFree(d_err);

    const int blocks = 256, threads = 1024, iters = 128;
    run<false>("exp2f", blocks, threads, iters);
    run<true>("exp2_alu", blocks, threads, iters);
    return 0;
}
