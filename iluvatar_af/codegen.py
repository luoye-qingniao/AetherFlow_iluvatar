"""天数 C 源码生成：复用 CUDA 的模板，只替换 f16 的存取实现。

模板本身（逐元素 / 广播 / softmax / attention / 卷积 / 池化 / 扫描……）是
纯 C++ 加 ``__global__`` 语义，天数 Clang 完全接受。唯一不通用的是 f16 的
PTX 内联汇编：``asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h))`` 在 clang 上
报 invalid output constraint。

因此这里只做一件事：换掉 PRELUDE 里的三个 f16 辅助函数，其余常量与构造
函数原样透传（``__getattr__``）。两种后端因此**共用一份内核语义**——模板
改了，天数自动跟上。
"""

from __future__ import annotations

from aetherflow.backends.cuda import codegen as _cuda

# clang 的 _Float16 与 numpy 的 float16 是同一套 IEEE 754 半精度，逐位布局
# 一致，所以 unsigned short 与 _Float16 之间用 __builtin_bit_cast 互转即可，
# 不需要 include 任何头文件。
#
# 16 位原子加仍然用 CAS 循环，但**不能**直接调 `atomicCAS(unsigned short*)`：
# 天数 device runtime 只提供 int / unsigned int / unsigned long long 三个重载
# （实测报 "no matching function for call to 'atomicCAS'"）。这里与 CUDA sm<60
# 走同一条路：把 CAS 加宽到对齐的 32 位字，只替换目标半字。算法逐字照搬
# `aetherflow.backends.cuda.codegen` 的 `af_cas_h` 非 16 位原子分支。
_ILUVATAR_F16_HELPERS = r"""
__device__ __forceinline__ float af_ld_h(unsigned short h) {
    return (float)__builtin_bit_cast(_Float16, h);
}
__device__ __forceinline__ unsigned short af_st_h(float v) {
    return __builtin_bit_cast(unsigned short, (_Float16)v);
}
__device__ __forceinline__ unsigned short af_cas_h(unsigned short* p,
                                                   unsigned short assumed,
                                                   unsigned short v) {
    unsigned long long addr = (unsigned long long)p;
    unsigned int* base = (unsigned int*)(addr & ~2ULL);
    unsigned int shift = (addr & 2ULL) ? 16u : 0u;
    unsigned int mask = (addr & 2ULL) ? 0xFFFF0000u : 0x0000FFFFu;
    unsigned int other = *base & ~mask;
    for (;;) {
        unsigned int assumed32 = other | (((unsigned int)assumed) << shift);
        unsigned int desired = other | (((unsigned int)v) << shift);
        unsigned int prev = atomicCAS(base, assumed32, desired);
        if (prev == assumed32) return assumed;
        unsigned short got = (unsigned short)((prev & mask) >> shift);
        if (got != assumed) return got;
        other = prev & ~mask;
    }
}
__device__ __forceinline__ float af_add_h(unsigned short* p, float v) {
    unsigned short old = *p;
    unsigned short assumed;
    do {
        assumed = old;
        old = af_cas_h(p, assumed, af_st_h(af_ld_h(assumed) + v));
    } while (assumed != old);
    return af_ld_h(old);
}
"""

# 天数自己的融合注意力模板。
#
# 共享模板（`cuda/codegen.py::ATTENTION_SRC`）里 **每个线程负责一个 key**，
# 沿 D 维串行做点积：同一时刻 warp 内各线程读的是不同 K 行的同一个 d，地址
# 步长 D*sizeof(T)——完全不合并。MR-V100 上实测无论什么形状都停在 ~123
# GFLOPS（见 REMIND 的算力扫描）。
#
# 这里把点积改成 **子组协作**：每 SG 个连续线程处理一个 key，组内 lane 各自
# 读一段**连续**的 d（`d = lane*CE .. lane*CE+CE`），于是同一条 load 指令覆盖
# 连续的 SG*CE 个元素，读取合并、且每 lane 的串行链路上没有跨步；再用
# `__shfl_down_sync`（显式 width=SG）在组内归约。
#
# 分工方式从"跨步"（`d = lane; d += SG`）换成"分块"是有原因的：跨步时每个
# lane 每次只取 1 个元素，一条 load 指令在组内只够 SG*2 字节，而一次点积要
# log2(SG) 次 shuffle；分块后每 lane 连续取 8 个元素（f16 下正好 16 字节），
# SG 随 D 调整（D=64→8，D=128→16），shuffle 次数随之降到 3/4 次。每个 key
# 的指令数因此降了约一半。
#
# 其余部分逐字沿用共享模板，确保语义与参考实现一致：
#   - 在线 softmax 的 m / l / corr 更新顺序不变；
#   - dropout 的随机数索引仍是 `(bh*S + s)*T + t`，与参考后端逐位相同；
#   - dropout 只乘在分子上，不进入分母；
#   - f16 混合精度：点积、softmax 统计、输出累加全在 float 里，只写回时舍入。
#
# SG 上限取 32 而不取 `warpSize`：32 是 CUDA 与天数（64）的公共子宽度，
# `__shfl_down_sync` 的 width 参数负责分段，因此同一份源码在两种 warp 宽度
# 下都成立。**不要**把它换成裸的 `__shfl_down_sync(v, off)`——那次调用默认
# 跨整个 warp（天数上是 64），会把多个 key 的和串到一起。
_ILUVATAR_ATTN_HELPERS = r"""
// Block-wide max/sum with a warp-level pre-pass.
//
// The shared `block_reduce_max` / `block_reduce_sum` are pure smem trees: on a
// 256-thread block they cost log2(256) = 8 __syncthreads each, and a row calls
// them twice per chunk (plus the PV barrier) -- 20+ barriers per chunk, which
// showed up as the dominant cost once the dot products were coalesced.
// Here each 32-lane segment reduces with shuffles, one lane per segment
// publishes to smem, and every thread folds the <= 8 segment results. Two
// barriers instead of nine. Width 32 (not warpSize) keeps this valid whether
// the device warp is 32 or 64 lanes.
__device__ __forceinline__ float af_block_max(float v, float* scratch) {
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xFFFFFFFFu, v, off, 32);
        v = v > o ? v : o;
    }
    const int seg = threadIdx.x >> 5;
    const int nseg = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) { scratch[seg] = v; }
    __syncthreads();
    float r = scratch[0];
    for (int i = 1; i < nseg; ++i) {
        const float o = scratch[i];
        r = r > o ? r : o;
    }
    __syncthreads();
    return r;
}

__device__ __forceinline__ float af_block_sum(float v, float* scratch) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off, 32);
    }
    const int seg = threadIdx.x >> 5;
    const int nseg = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) { scratch[seg] = v; }
    __syncthreads();
    float r = 0.f;
    for (int i = 0; i < nseg; ++i) { r += scratch[i]; }
    __syncthreads();
    return r;
}

// dropout 掩码的 32 位 murmur3 finalizer。64 位 splitmix 每次要 3 条 64 位乘
// （天数上没有原生 64 位乘法器），每元素约 14 条指令；换成 32 位后约减半。
// 分布与相邻相关性实测与 splitmix 一致，mask 序列变化但仍是 iid Bernoulli。
__device__ __forceinline__ unsigned attention_rand(unsigned long long seed,
                                                   long long idx) {
    unsigned h = (unsigned)idx ^ (unsigned)((unsigned long long)idx >> 32) ^
                 (unsigned)(seed ^ (seed >> 32));
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}
"""

_ILUVATAR_ATTENTION_SRC = "%PRELUDE%" + _ILUVATAR_ATTN_HELPERS + r"""
// Dot product against a Q fragment already held in registers.
//
// W is the unroll width and is picked from nq so that a narrow row (D=8 gives
// nq=1, D=32 gives nq=4) does not pay for eight bound checks per key: with a
// fixed 8-wide unroll the D=8 shape alone lost ~30% to the dead iterations.
// Making W (or the array size) a template parameter and dispatching by D at the
// call site is NOT viable: the Iluvatar compiler aborts on some instantiations
// ("Cannot select: intrinsic %llvm.bi.ldp.vs.imm", divergent base pointer of
// __ivcorex_ml_mem_load_*), so the width dispatch stays a runtime ternary here.
// The array is passed by reference so that __forceinline__ keeps qv in
// registers under the constant indices below -- a pointer parameter would
// demote it to local memory and defeat the whole point.
template <int W>
__device__ __forceinline__ float af_dot_reg(const float (&qv)[8],
                                            const %T%* __restrict__ kp,
                                            int nq) {
    float dot0 = 0.f;
    float dot1 = 0.f;
#pragma unroll
    for (int i = 0; i < W; i += 2) {
        if (i + 1 < nq) {
            dot0 += qv[i] * %LD%(kp[i]);
            dot1 += qv[i + 1] * %LD%(kp[i + 1]);
        } else if (i < nq) {
            dot0 += qv[i] * %LD%(kp[i]);
        }
    }
    return dot0 + dot1;
}

// Two adjacent stored elements in one 32-bit access.
//
// The dot product costs one load + one convert + one FMA per element, and the
// load is what the instruction pipeline stalls on. Reading a pair halves that
// count (the same 4-byte word feeds two converts). Only used where the caller
// proved 4-byte alignment: D a multiple of 2*SG, so every lane's d0 is even and
// the row start is even. The access is written as a scalar 32-bit load on
// purpose -- letting the compiler auto-vectorise the same loop makes it abort
// ("Cannot select: intrinsic %llvm.bi.ldp.vs.imm", divergent base pointer of
// __ivcorex_ml_mem_load_*), so the explicit form is the only stable one.
__device__ __forceinline__ void af_ld2(const %T%* __restrict__ p, float& a, float& b) {
    if (sizeof(%T%) == 2) {
        const unsigned int w = *(const unsigned int*)(const void*)p;
        a = (float)__builtin_bit_cast(_Float16, (unsigned short)(w & 0xFFFFu));
        b = (float)__builtin_bit_cast(_Float16, (unsigned short)(w >> 16));
    } else {
        a = %LD%(p[0]);
        b = %LD%(p[1]);
    }
}

// Unchecked variant of af_dot_reg, used when nq is exactly W (and W is even):
// the caller guarantees the lane owns W elements, so no bound checks and no
// out-of-range pair reads.
template <int W>
__device__ __forceinline__ float af_dot_pair(const float (&qv)[8],
                                             const %T%* __restrict__ kp) {
    float dot0 = 0.f;
    float dot1 = 0.f;
#pragma unroll
    for (int i = 0; i < W; i += 2) {
        float k0, k1;
        af_ld2(kp + i, k0, k1);
        dot0 += qv[i] * k0;
        dot1 += qv[i + 1] * k1;
    }
    return dot0 + dot1;
}

// QK scores for one chunk, one row of Q.
//
// SG is a template parameter so the shuffle width and `threadIdx.x / SG` stay
// compile-time constants -- as a runtime value the same code measured ~7%
// slower on every shape. SG is picked by D so that a lane always owns
// CE = ceil(D/SG) contiguous elements: SG=8 gives CE=8 for D=64 (16 B per
// lane, one half-sector per key) and CE=16 for D=128, where the 16-load serial
// chain was measurably worse than using 16 lanes with CE=8.
// W is still a template argument, but it is selected by a runtime ternary over
// nq (see the key sweep), not by a call-site dispatch on D.
// PAIR is true when the caller proved D % (2*SG) == 0, so the K row is 4-byte
// aligned for every lane and adjacent elements can be fetched together.
template <int SG, bool PAIR>
__device__ __forceinline__ void af_qk_chunk(const %T%* __restrict__ qs,
                                            const %T%* __restrict__ k,
                                            const %T%* __restrict__ mrow,
                                            int has_mask,
                                            long long bh, long long T, long long D,
                                            float scale, long long t0,
                                            float* s_w) {
    const int lane = threadIdx.x & (SG - 1);
    const int ngroups = blockDim.x / SG;
    const int grp = threadIdx.x / SG;
    const long long CE = (D + SG - 1) / SG;
    const long long d0 = (long long)lane * CE;
    const long long dend = d0 + CE < D ? d0 + CE : D;
    const int nq = d0 < D ? (int)(dend - d0) : 0;

    // The Q fragment owned by this lane does not change across the key sweep,
    // so it is read once per chunk instead of once per key. nq <= 8 covers the
    // common widths (D<=64 with SG=8, 96<=D<=128 with SG=16); wider rows fall
    // back to re-reading Q, which stays correct for any D.
    float qv[8];
    const bool qreg = nq > 0 && nq <= 8;
    if (qreg) {
        const %T%* qp0 = qs + d0;
#pragma unroll
        for (int i = 0; i < 8; ++i) { qv[i] = i < nq ? %LD%(qp0[i]) : 0.f; }
    }

    // Sub-group j owns keys t0+grp, t0+grp+ngroups, ... : all ngroups
    // sub-groups walk their share in parallel, so a chunk still covers
    // blockDim.x keys. Every lane of a sub-group takes the same branch
    // (t does not depend on lane), which is what keeps the shuffles below
    // lock-step across the whole warp.
    for (long long j = grp; j < blockDim.x; j += ngroups) {
        const long long t = t0 + j;
        float dot = 0.f;
        if (t < T && nq > 0) {
            const %T%* kp = k + (bh * T + t) * D + d0;
            if (qreg) {
                // Pair reads are only valid when the lane owns an even number
                // of elements, so odd widths keep the checked path.
                if (PAIR && (nq == 2 || nq == 4 || nq == 8)) {
                    dot = nq == 2 ? af_dot_pair<2>(qv, kp)
                        : (nq == 4 ? af_dot_pair<4>(qv, kp)
                                   : af_dot_pair<8>(qv, kp));
                } else {
                    dot = nq <= 2 ? af_dot_reg<2>(qv, kp, nq)
                        : (nq <= 4 ? af_dot_reg<4>(qv, kp, nq)
                                   : af_dot_reg<8>(qv, kp, nq));
                }
            } else {
                const %T%* qp = qs + d0;
                float dot0 = 0.f;
                float dot1 = 0.f;
                int i = 0;
                for (; i + 1 < nq; i += 2) {
                    dot0 += %LD%(qp[i]) * %LD%(kp[i]);
                    dot1 += %LD%(qp[i + 1]) * %LD%(kp[i + 1]);
                }
                if (i < nq) { dot0 += %LD%(qp[i]) * %LD%(kp[i]); }
                dot = dot0 + dot1;
            }
        }
        for (int off = SG >> 1; off > 0; off >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFFu, dot, off, SG);
        }
        if (lane == 0) {
            float sc = -3.4e38f;
            if (t < T) {
                sc = dot * scale;
                if (has_mask) { sc += %LD%(mrow[t]); }
            }
            // Parking the score in s_w lets the phase below keep the
            // original "one thread per key" shape unchanged.
            s_w[j] = sc;
        }
    }
}

extern "C" __global__ void attention_fwd(const %T%* __restrict__ q,
                                         const %T%* __restrict__ k,
                                         const %T%* __restrict__ v,
                                         const %T%* __restrict__ mask,
                                         %T%* __restrict__ out,
                                         long long S, long long T,
                                         long long D, long long E,
                                         float scale,
                                         float p_drop, float drop_scale,
                                         unsigned long long seed,
                                         int has_mask,
                                         int mask_shared) {
    // Dynamic shared memory: [weights | reduce scratch | accumulator].
    extern __shared__ float smem[];
    float* s_w = smem;
    float* s_red = smem + blockDim.x;
    float* s_acc = smem + 2 * blockDim.x;

    const long long bh = blockIdx.y;
    const long long s = blockIdx.x;

    for (long long e = threadIdx.x; e < E; e += blockDim.x) { s_acc[e] = 0.f; }
    __syncthreads();

    const %T%* qs = q + (bh * S + s) * D;
    const %T%* vbase = v + bh * T * E;
    const %T%* mrow =
        has_mask ? (mask + ((mask_shared ? 0 : bh) * S + s) * T) : nullptr;

    // SG is a template parameter (see af_qk_chunk): as a runtime value the
    // width argument falls off the fast path of __shfl_down_sync and
    // threadIdx.x/SG becomes a real division (measured ~7% slower).
    // D>=96 uses 16 lanes per key so a lane still owns 8 contiguous elements.

    float m = -3.4e38f;
    float l = 0.f;

    for (long long t0 = 0; t0 < T; t0 += blockDim.x) {
        // PAIR needs D % (2*SG) == 0: that keeps every lane's d0 even and the K
        // row start even, so a 32-bit pair read cannot straddle a row.
        if (D >= 96) {
            if (D % 32 == 0) {
                af_qk_chunk<16, true>(qs, k, mrow, has_mask, bh, T, D, scale, t0, s_w);
            } else {
                af_qk_chunk<16, false>(qs, k, mrow, has_mask, bh, T, D, scale, t0, s_w);
            }
        } else if (D % 16 == 0) {
            af_qk_chunk<8, true>(qs, k, mrow, has_mask, bh, T, D, scale, t0, s_w);
        } else {
            af_qk_chunk<8, false>(qs, k, mrow, has_mask, bh, T, D, scale, t0, s_w);
        }
        __syncthreads();

        const long long t = t0 + threadIdx.x;
        float sc = -3.4e38f;
        if (t < T) { sc = s_w[threadIdx.x]; }

        const float blk_max = af_block_max(sc, s_red);
        const float m_new = m > blk_max ? m : blk_max;
        const float corr = expf(m - m_new);

        float p = 0.f;
        float w = 0.f;
        if (t < T) {
            p = expf(sc - m_new);
            w = p;
            if (p_drop > 0.f) {
                const unsigned z =
                    attention_rand(seed, (bh * S + s) * T + t);
                const float u =
                    (float)(z >> 8) * (1.0f / 16777216.0f);
                w = (u >= p_drop) ? p * drop_scale : 0.f;
            }
        }
        s_w[threadIdx.x] = w;

        const float blk_sum = af_block_sum(p, s_red);
        l = l * corr + blk_sum;

        // PV: the chunk is split across nparts thread groups. With a narrow E
        // (64 channels, 256 threads) the shared template left 3/4 of the
        // block idle here, and this stage ended up ~7x the cost of QK.
        // Each group accumulates one partial sum per channel; the first E
        // threads then fold the partials into the running accumulator.
        // s_red doubles as the scratch: nparts*E never exceeds blockDim.x.
        const int nparts = blockDim.x / (int)E;
        if (nparts > 0) {
            const int part = threadIdx.x / (int)E;
            if (part < nparts) {
                const int e = threadIdx.x % (int)E;
                // Four accumulators over the strided key sweep: the same reason
                // as the QK side -- one dependent FMA chain per thread cannot
                // keep the load unit busy. The summation order changes, which
                // only moves the low bits (online softmax statistics are
                // untouched); conformance compares against the reference with
                // rtol/atol, not bit-for-bit.
                const long long jlim =
                    (long long)blockDim.x < T - t0 ? (long long)blockDim.x : T - t0;
                // Pointer + int step: `vbase[(t0 + j) * E + e]` recomputes a
                // 64-bit multiply for every one of the ~blockDim/4 elements a
                // thread touches, which was a visible share of this stage.
                const int step = nparts * (int)E;   // <= blockDim.x, fits int
                const %T%* vp = vbase + (t0 + part) * E + e;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                long long j = part;
                for (; j + 3 * nparts < jlim; j += 4 * nparts, vp += 4 * step) {
                    a0 += s_w[j] * %LD%(vp[0]);
                    a1 += s_w[j + nparts] * %LD%(vp[step]);
                    a2 += s_w[j + 2 * nparts] * %LD%(vp[2 * step]);
                    a3 += s_w[j + 3 * nparts] * %LD%(vp[3 * step]);
                }
                for (; j < jlim; j += nparts, vp += step) {
                    a0 += s_w[j] * %LD%(vp[0]);
                }
                s_red[part * (int)E + e] = (a0 + a1) + (a2 + a3);
            }
            __syncthreads();
            if (threadIdx.x < (int)E) {
                float acc = 0.f;
                for (int q = 0; q < nparts; ++q) {
                    acc += s_red[q * (int)E + threadIdx.x];
                }
                s_acc[threadIdx.x] = s_acc[threadIdx.x] * corr + acc;
            }
        } else {
            // E is wider than the block: fall back to one thread per channel.
            for (long long e = threadIdx.x; e < E; e += blockDim.x) {
                float acc = s_acc[e] * corr;
                for (long long j = 0; j < blockDim.x && t0 + j < T; ++j) {
                    acc += s_w[j] * %LD%(vbase[(t0 + j) * E + e]);
                }
                s_acc[e] = acc;
            }
        }
        __syncthreads();
        m = m_new;
    }

    const float inv = l > 0.f ? 1.f / l : 0.f;
    %T%* orow = out + (bh * S + s) * E;
    for (long long e = threadIdx.x; e < E; e += blockDim.x) {
        orow[e] = %ST%(s_acc[e] * inv);
    }
}
"""

# 分数矩阵归一化内核：把 QK^T 的产物就地变成 P（softmax + mask + dropout）。
#
# 走 GEMM 的 attention 路径需要一次显式的 softmax pass，而共享的 softmax 内核
# 不认识 mask / dropout / attention 的 seed 约定，所以这里单独写一份。语义与
# 融合内核逐项对齐：**分母只累加 dropout 之前的 p**，dropout 只作用在写出的
# 权重上（含 1/(1-p) 缩放）；RNG 下标仍是 (bh*S + s)*T + t，保证同一 seed 下
# 与融合内核给出相同的 mask。
#
# 一行一个 32 线程子组（sub-warp），不占 smem、两遍读同一行。
#
# 三版实测（MR-V100，134 MB 行数据，见 REMIND 的带宽探针）：
#   - "一块一行 + smem 缓存"：130 GB/s。每块只有 2 KB 数据，4 次
#     __syncthreads + 归约开销把带宽吃光；smem 版本还限制了每块 32 KB 的
#     行缓存，T=2048 时只剩 6 个子组在跑，掉到 31 GB/s。
#   - "子组一行、标量逐元素、无 smem"：369 GB/s（纯拷贝上限 522 GB/s）。
#   - "子组一行、一次 32 位读两个 f16、无 smem"：464 GB/s。
# 所以这里用最后一版：一次 load 覆盖 32 lane × 4 B = 128 B，正好一条 cache
# line；第二遍重读靠 L1/L2 命中（每行只有 1~4 KB，刚读过），HBM 上仍是一次
# 读 + 一次写。
#
# `out` 允许与 `sc` 同址（原地归一化）：每个下标只在自己的那一次迭代里
# 先读后写，迭代之间下标不相交，所以不会读到被改过的值。
#
# 归约宽度取 32 而不是 `warpSize`：天数 warp 是 64，`__shfl_*_sync` 的 width
# 参数负责分段，32 是两种 warp 宽度下的公共子宽度（与融合内核同一约定）。
# `__shfl_down_sync` 只保证结果落在 0 号 lane，必须再用一次 `__shfl_sync`
# 广播回整组——漏掉这步会让 1~31 号 lane 用着残缺的 max/sum 继续算。
_ILUVATAR_SCORES_SRC = "%PRELUDE%" + _ILUVATAR_ATTN_HELPERS + r"""
// AF_NP_REG > 0 表示"整行驻留寄存器"的特化：行宽 T 恰好等于 64 * AF_NP_REG，
// 于是每个 lane 固定持有 AF_NP_REG 个 32 位字（= 2*AF_NP_REG 个 f16），
// 第一遍读进来的字直接留在寄存器里，第二遍不再碰显存。
//
// 两遍版本在 T <= 1024 时第二遍靠 L1 命中，HBM 上仍是一次读 + 一次写，但
// 省不掉第二遍的 L1 流量与地址计算；T = 2048 时一行 4 KB，多块并发把这个
// 工作集挤出 L1，第二遍变成真的 HBM 读，实测有效带宽从 390 掉到 276 GB/s。
// 寄存器版把两种情况都压到"一次 HBM 读 + 一次 HBM 写"。
//
// 上限 16 是实测的寄存器预算：AF_NP_REG = 16 时 w/mw 两个数组各 16 个字，
// 仍然全部留在寄存器里（T = 1024 实测 497 GB/s）；再往上（T = 2048）会溢出
// 到 local memory，实测 206 GB/s，反而比两遍版本（274 GB/s）更慢。
#define AF_NP_REG %NP%

__device__ __forceinline__ float af_sub_max(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xFFFFFFFFu, v, off, 32);
        v = v > o ? v : o;
    }
    return __shfl_sync(0xFFFFFFFFu, v, 0, 32);
}

__device__ __forceinline__ float af_sub_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, off, 32);
    }
    return __shfl_sync(0xFFFFFFFFu, v, 0, 32);
}

// dropout 的 0/1 判定，与融合内核同一约定（同一 seed 逐位相同）。
__device__ __forceinline__ float af_drop_w(float p, float p_drop, float drop_scale,
                                           unsigned long long seed, long long idx) {
    const unsigned z = attention_rand(seed, idx);
    const float u = (float)(z >> 8) * (1.0f / 16777216.0f);
    return (u >= p_drop) ? p * drop_scale : 0.f;
}

extern "C" __global__ void af_attention_scores(
    const %T%* __restrict__ sc,
    const %T%* __restrict__ mask,
    %T%* __restrict__ out,
    float* __restrict__ inv_out,
    long long rows, long long S, long long T, float scale,
    float p_drop, float drop_scale, unsigned long long seed,
    int has_mask, int mask_shared) {
  const int lane = threadIdx.x & 31;
  const long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (r >= rows) { return; }

  const %T%* src = sc + r * T;
  const %T%* mrow = mask + (mask_shared ? (r % S) : r) * T;
  %T%* dst = out + r * T;
#if AF_NP_REG > 0
  // 整行驻留寄存器（host 侧已保证 sizeof(T) == 2 且 T == 64 * AF_NP_REG）。
  unsigned int w[AF_NP_REG];
  unsigned int mw[AF_NP_REG];
  float mxl = -3.402823466e+38f;
#pragma unroll
  for (int i = 0; i < AF_NP_REG; ++i) {
    const long long t0 = ((long long)i * 32 + lane) * 2;
    const unsigned int wi = *(const unsigned int*)(const void*)(src + t0);
    w[i] = wi;
    float v0 = %LD%((unsigned short)(wi & 0xFFFFu)) * scale;
    float v1 = %LD%((unsigned short)(wi >> 16)) * scale;
    if (has_mask) {
      const unsigned int mi = *(const unsigned int*)(const void*)(mrow + t0);
      mw[i] = mi;
      v0 += %LD%((unsigned short)(mi & 0xFFFFu));
      v1 += %LD%((unsigned short)(mi >> 16));
    }
    mxl = v0 > mxl ? v0 : mxl;
    mxl = v1 > mxl ? v1 : mxl;
  }
  const float mxr = af_sub_max(mxl);
  float sumr = 0.f;
#pragma unroll
  for (int i = 0; i < AF_NP_REG; ++i) {
    const long long t0 = ((long long)i * 32 + lane) * 2;
    const unsigned int wi = w[i];
    float v0 = %LD%((unsigned short)(wi & 0xFFFFu)) * scale;
    float v1 = %LD%((unsigned short)(wi >> 16)) * scale;
    if (has_mask) {
      const unsigned int mi = mw[i];
      v0 += %LD%((unsigned short)(mi & 0xFFFFu));
      v1 += %LD%((unsigned short)(mi >> 16));
    }
    const float p0 = expf(v0 - mxr);
    const float p1 = expf(v1 - mxr);
    float o0 = p0;
    float o1 = p1;
    if (p_drop > 0.f) {
      const long long base = r * T + t0;
      o0 = af_drop_w(p0, p_drop, drop_scale, seed, base);
      o1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 1);
    }
    sumr += p0 + p1;
    *(unsigned int*)(void*)(dst + t0) =
        (unsigned int)%ST%(o0) | ((unsigned int)%ST%(o1) << 16);
  }
  sumr = af_sub_sum(sumr);
  if (lane == 0) { inv_out[r] = sumr > 0.f ? 1.f / sumr : 0.f; }
#else
  const long long n2 = T >> 1;
  // 成对读要求行首 4 字节对齐（T 为偶数即可，cudaMalloc 保证基址对齐）且
  // 存储宽度为 16 位。f32 或奇数 T 走标量路径，语义完全一致。
  const bool pair = sizeof(%T%) == 2 && ((T & 1) == 0);

  float mx = -3.402823466e+38f;
  if (pair) {
    for (long long i = lane; i < n2; i += 32) {
      const unsigned int w = *(const unsigned int*)(const void*)(src + i * 2);
      float v0 = %LD%((unsigned short)(w & 0xFFFFu)) * scale;
      float v1 = %LD%((unsigned short)(w >> 16)) * scale;
      if (has_mask) {
        const unsigned int mw = *(const unsigned int*)(const void*)(mrow + i * 2);
        v0 += %LD%((unsigned short)(mw & 0xFFFFu));
        v1 += %LD%((unsigned short)(mw >> 16));
      }
      mx = v0 > mx ? v0 : mx;
      mx = v1 > mx ? v1 : mx;
    }
  } else {
    for (long long t = lane; t < T; t += 32) {
      float v = %LD%(src[t]) * scale;
      if (has_mask) { v += %LD%(mrow[t]); }
      mx = v > mx ? v : mx;
    }
  }
  mx = af_sub_max(mx);

  // 分母只累加 dropout 之前的 p；dropout 只作用在写出的权重上。
  float sum = 0.f;
  if (pair) {
    for (long long i = lane; i < n2; i += 32) {
      const long long t0 = i * 2;
      const unsigned int w = *(const unsigned int*)(const void*)(src + t0);
      float v0 = %LD%((unsigned short)(w & 0xFFFFu)) * scale;
      float v1 = %LD%((unsigned short)(w >> 16)) * scale;
      if (has_mask) {
        const unsigned int mw = *(const unsigned int*)(const void*)(mrow + t0);
        v0 += %LD%((unsigned short)(mw & 0xFFFFu));
        v1 += %LD%((unsigned short)(mw >> 16));
      }
      const float p0 = expf(v0 - mx);
      const float p1 = expf(v1 - mx);
      float o0 = p0;
      float o1 = p1;
      if (p_drop > 0.f) {
        const long long base = r * T + t0;
        o0 = af_drop_w(p0, p_drop, drop_scale, seed, base);
        o1 = af_drop_w(p1, p_drop, drop_scale, seed, base + 1);
      }
      sum += p0 + p1;
      const unsigned int o = (unsigned int)%ST%(o0) |
                             ((unsigned int)%ST%(o1) << 16);
      *(unsigned int*)(void*)(dst + t0) = o;
    }
  } else {
    for (long long t = lane; t < T; t += 32) {
      float v = %LD%(src[t]) * scale;
      if (has_mask) { v += %LD%(mrow[t]); }
      const float p = expf(v - mx);
      float w = p;
      if (p_drop > 0.f) {
        w = af_drop_w(p, p_drop, drop_scale, seed, r * T + t);
      }
      sum += p;
      dst[t] = %ST%(w);
    }
  }
  sum = af_sub_sum(sum);

  // 归一化不在这里做：那要给整行再走一遍读+写。分母的倒数交给输出缩放内核，
  // 那里只有 bh*S*E 个元素（E 通常远小于 T），而且 f16 下把 [0,1] 的未归一化
  // 权重存进 HBM、再乘 1/l，比直接存 p/l 更不容易落进次正规数。
  if (lane == 0) { inv_out[r] = sum > 0.f ? 1.f / sum : 0.f; }
#endif
}
"""

# 输出行缩放：out[r, :] *= inv[r]。
#
# 分数矩阵的 softmax 只写未归一化权重，分母的倒数在这里一次性作用到输出上。
# 语义等价因为 (p/l) @ V == (p @ V)/l 对每行成立；代价是 bh*S*E 的读+写，
# 比在 bh*S*T 上多走一遍小一个数量级的量级。
_ILUVATAR_RESCALE_SRC = "%PRELUDE%" + r"""
extern "C" __global__ void af_attention_rescale(
    %T%* __restrict__ out,
    const float* __restrict__ inv,
    long long rows, long long E) {
  // 一个 32 线程子组负责一行，行号就是 inv 的下标。
  //
  // 早先的写法是扁平的 grid-stride 循环，每个元素用 ``i / E`` 反算行号：
  // 那是每元素一次 64 位整数除法（E 是运行时值，编译器消不掉），在
  // bh*S*E = 8.4M 个元素上实测只有 249 GB/s。按行分工后行号直接来自
  // 子组编号，除法整条消失，inv 每行只读一次（跨 lane 广播，命中 L1）。
  const int lane = threadIdx.x & 31;
  const int nsub = blockDim.x >> 5;
  const long long stride = (long long)gridDim.x * nsub;
  const long long start = (long long)blockIdx.x * nsub + (threadIdx.x >> 5);
  const bool pair = sizeof(%T%) == 2 && ((E & 1) == 0);
  for (long long r = start; r < rows; r += stride) {
    const float s = inv[r];
    %T%* row = out + r * E;
    if (pair) {
      const long long e2 = E >> 1;
      for (long long i = lane; i < e2; i += 32) {
        const unsigned int w = *(const unsigned int*)(const void*)(row + i * 2);
        const float v0 = %LD%((unsigned short)(w & 0xFFFFu)) * s;
        const float v1 = %LD%((unsigned short)(w >> 16)) * s;
        *(unsigned int*)(void*)(row + i * 2) =
            (unsigned int)%ST%(v0) | ((unsigned int)%ST%(v1) << 16);
      }
    } else {
      for (long long i = lane; i < E; i += 32) {
        row[i] = %ST%(%LD%(row[i]) * s);
      }
    }
  }
}
"""


# TCU flash attention（f16，D=E=64）。
#
# 与 ``attention_scores_source`` 的 GEMM 物化路径并列：Q/K/V 只在开始时读一次
# HBM，分数矩阵永不落显存。结构来自 bench/tcu_flash_attn.cu 的 attn64w16：
# 每 block 16 warp × 16 行 = 256 行、每轮 64 key（两批 32），softmax 用
# **惰性基准**（只有分数超出基准 8（log2 空间）时才 4 步 shuffle 重算），
# 分母非 dropout 时走 P@ones 的 mma、dropout 时走 warp 内 shuffle 求和。
#
# 四个模式编译成四个非模板启动器（``extern "C"``，便于 cuModuleGetFunction
# 按固定名字取），模板实例化后内联，寄存器分配与 bench 完全一致。
# 行方向允许 S 不是 256 的倍数：越界的 warp 照常参与块级同步，但只读第 0 行
# 且不写输出（S 必须是 32 的倍数才能整 warp 判定）。
_ILUVATAR_FLASH_SRC = "%PRELUDE%" + _ILUVATAR_ATTN_HELPERS + r"""
#include <crt/iluvatar_mma.h>

using namespace ixix::wmma;

#define AF_NWT16 16
#define AF_NTH16 (AF_NWT16 * 64)
#define AF_LOG2E 1.4426950408889634f

// load_matrix_sync 忽略 ldm，按 TCU 私有 swizzle 取址；这是两套布局的实测公式。
__device__ __forceinline__ int af_off_row(int row, int col) {
  int i = col >> 4, c = col & 15, r = row >> 1;
  return (i * 256 + r / 4 * 128) + ((r * 32 + c * 2 + (row & 1) + i * 64) & 127);
}
__device__ __forceinline__ int af_off_col(int row, int col) {
  int r = row >> 1;
  return (col >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (col >> 3)) * 16 +
         ((r >> 2 & 1) ^ (col >> 2 & 1)) * 8 + (col & 3) * 2 + (row & 1);
}

// 32 个 key 的 K/V 重排进 swizzle 布局的 smem（每次 2048 个 half）。
// nth 是 block 的线程数，随内核的 warp 数变化。
__device__ __forceinline__ void af_load_kv(__half* ks, __half* vs,
                                           const __half* Kb0, const __half* Vb0,
                                           int kt, int nth = AF_NTH16) {
  // 每个线程搬两个相邻 d：K 的 swizzle 布局里相邻 d 也相邻，4 字节整字写；
  // V 的布局相邻 d 差 2，只能拆成两个 half 写。gmem 侧都是 4 字节读。
  const long long o = (long long)kt * 2048;
  const unsigned* K2 = (const unsigned*)(Kb0 + o);
  unsigned* ks2 = (unsigned*)ks;
  for (int idx = threadIdx.x; idx < 1024; idx += nth) {
    const int t = idx >> 5, d = (idx & 31) * 2;
    ks2[(((t >> 4) * 2 + (d >> 5)) * 512 + af_off_col(d & 31, t & 15)) >> 1] = K2[idx];
    const int g = (t << 6) | d;
    const __half2 vv = *(const __half2*)(Vb0 + o + g);
    vs[(d >> 4) * 512 + af_off_col(t, d & 15)] = __low2half(vv);
    vs[(d >> 4) * 512 + af_off_col(t, (d + 1) & 15)] = __high2half(vv);
  }
}

// dropout 的 0/1 判定：与 attention_rand 同一 seed 约定，逐位一致。
__device__ __forceinline__ float af_flash_drop(float p, float p_drop,
                                               float drop_scale,
                                               unsigned long long seed,
                                               long long idx) {
  const unsigned z = attention_rand(seed, idx);
  const float u = (float)(z >> 8) * (1.0f / 16777216.0f);
  return (u >= p_drop) ? p * drop_scale : 0.f;
}

// ---------------------------------------------------------------------
// 一个模板覆盖四个模式：16 warp × 16 行 = 每 block 256 行，每轮 64 key
// （两批 32 背靠背发射以取 ILP）。行数减半后 OF 只剩 5 个 accumulator，
// 双批 S 能真正背靠背；`__launch_bounds__(_, 3)` 让 3 个 block 共存于
// 一个 SM，互相掩盖 __syncthreads 与 K/V 装载的空泡。
// bench 四模式（512x512x64, bh=256）最终值 / 32 行版本（HEAD b4c21a5）：
//   plain 0.5996 / 0.5998、mask 0.8481 / 0.9009、
//   drop 0.8812 / 0.8791、mask+drop 1.2721 / 1.4339 ms。
// 纯 mask 的慢路径保持展开：改成 unroll 1 会重新 spill 且慢 20%。
//
// 两种 mask/dropout 结构按 HAS_MASK 选：
//   * !HAS_MASK（plain / dropout）：风险检测先独立跑一遍——只用 SF/TF 与
//     sc2 重算 4 行分数、检测 lim = mlr + 8；越界则 shfl 归约新基准、缩放
//     OF 与 ssr。dropout 必须走这条：分母 ssr 是 float 逐轮累加，若先按
//     初始基准 -1e30 写 P，exp2 溢出成 inf，慢路径再乘 corr=0 就是 NaN；
//     预检保证首轮的 inf 不进 ssr。
//   * HAS_MASK（mask / mask+dropout）：检测并入写 P 的一遍（mask 每元素
//     每轮只读一次），命中时慢路径按新基准修正 OF / ssr 并重写 P。
//     首轮基准是 -1e30：纯 mask 跳过快路径（慢路径必然全量重算）；
//     mask+drop 不跳过——快路径的 hash 也要在慢路径重做，实测反而更快。
template <bool HAS_MASK, bool HAS_DROP>
__device__ __forceinline__ void af_attn_w16(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int S, int T, float scale, const __half* __restrict__ M, float p_drop,
    float drop_scale, unsigned long long seed, int mask_shared) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                        // 两批 K，各 2048
  __half* vsm = ksm + 4096;                // 两批 V，各 2048
  __half* psm = vsm + 4096;                // 每 warp 一个 16x32 的 P
  __half* ones_sm = psm + AF_NWT16 * 512;  // 16x16 全 1，用于 P@ones
  __shared__ float ssums[AF_NWT16 * 16];   // dropout 分母（每 warp 16 行）

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;
  const int c0 = lane & 15;
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (16 * AF_NWT16) + warp * 16;
  // 行数不是 256 的倍数时，尾部 warp 只读第 0 行、不写输出；块级同步照常。
  const int rb = (row_base < S) ? row_base : 0;

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

  // Q 装载：16 行 × 64 列，借 P 的 512 槽分两趟（每趟 16x32）。
  {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      for (int idx = lane; idx < 512; idx += 64) {
        int m = idx >> 5, d = idx & 31;
        pw[af_off_row(m, d)] = Qb[(long long)(rb + m) * 64 + h * 32 + d];
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

  for (int idx = threadIdx.x; idx < AF_NWT16 * 16; idx += AF_NTH16)
    ssums[idx] = 0.f;
  for (int idx = threadIdx.x; idx < 512; idx += AF_NTH16) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[af_off_col(t, dd)] = __float2half(1.f);
  }
  __syncthreads();
  load_matrix_sync(vbo, ones_sm, 32);   // 分母矩阵不变，全程装一次

  const float sc2 = scale * AF_LOG2E;
  // 惰性基准常驻寄存器：每 lane 4 个，对应它负责的 4 行 r0+4*q。
  // 原来每轮每行都要从 smem 读一次基准、读后立即参与 exp2 的减法，
  // 这条"读-用"依赖链是 softmax 段的主要停顿来源（真机实测省 3.1%）。
  const float ml0 = -1e30f;
  float mlr_reg[4] = {ml0, ml0, ml0, ml0};
  float* ssr = ssums + warp * 16;

  // mask/dropout 的行基址循环外算一次，避免每轮做 (gl % S) 的整数除法。
  const __half* mbase = nullptr;
  long long dbase = 0;
  if (HAS_MASK) {
    const int mr0 = (mask_shared ? (rb % S) : (bh * S + rb)) + r0;
    mbase = M + (long long)mr0 * T;
  }
  if (HAS_DROP) dbase = (long long)(bh * S + rb + r0) * T;

  for (int kt = 0; kt < T; kt += 64) {
    const bool hasB = (kt + 32 < T);
    af_load_kv(ksm, vsm, Kb0, Vb0, kt >> 5);
    if (hasB) af_load_kv(ksm + 2048, vsm + 2048, Kb0, Vb0, (kt >> 5) + 1);
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

    // 批 B 的 QK（T 不是 64 的倍数时最后半批不存在，分数填 -1e30 使 exp 为 0）
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

    // 无 mask 时风险检测单独一遍：不读 mask，只把 SF/TF 与 scale 重算一次，
    // 开销比"合并进写 P 的一遍"更低（实测 plain -5%、dropout -29%）。
    // dropout 必须走这条：分母 ssr 是 float 逐轮累加，基准必须先定下来，
    // 否则首轮 exp2 溢出成 inf，corr 乘出 NaN。
    if (!HAS_MASK) {
      bool risk = false;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float lim = mlr_reg[q] + 8.f;
        float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
        float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
        risk |= fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) > lim;
      }
      if (__any_sync(0xffffffffu, risk)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          const float mn = mlr_reg[q];
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
          float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
          float corr = 1.f;
          if (mx > mn) {
            corr = exp2f(mn - mx);
            mlr_reg[q] = mx;
          }
#pragma unroll
          for (int e = 0; e < 5; ++e) OF[e]->x[q] *= corr;
          if (HAS_DROP && c0 == 0) ssr[ri] *= corr;
        }
      }
    }

    // 有 mask 时：风险检测并入写 P 的一遍，mask 每轮只读一次。
    // 越界（旧基准滞后 > 8）时在下面的慢路径按新基准修正 OF / ssr 并重写
    // 这一批 P；快路径的 p 可能已经很大，但只写进 pw、在 load_matrix_sync
    // 之前就被覆盖，不会被消费。
    __syncwarp();
    bool risky = false;
    // 首轮基准是 -1e30，p 会溢出成 inf。纯 mask 时首轮跳过快路径最省
    // （慢路径必然全量重算）；mask+dropout 时快路径的 hash 也要重做，实测
    // "先算一遍 + 慢路径覆盖" 反而更快（寄存器分配更好），故不跳过。
    const bool firstq = (kt == 0);
    const bool first = HAS_MASK && !HAS_DROP && firstq;
    if (!first) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float mn = mlr_reg[q];
        float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
        if (HAS_MASK) {
          const __half* mr = mbase + (4 * q * T + kt);
          x0 += __half2float(mr[c0]) * AF_LOG2E;
          x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
        }
        if (HAS_MASK) risky |= fmaxf(x0, x1) > mn + 8.f;
        // 不 clamp：首轮的 inf 只会进 pw/ssr，这两处都会被慢路径覆盖；
        // 其余轮次越界幅度有限（log2 空间十几个数量级），不会溢出。
        float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
        if (HAS_DROP) {
          const long long base = dbase + 4 * q * T + kt;
          float sv = p0 + p1;
          p0 = af_flash_drop(p0, p_drop, drop_scale, seed, base + c0);
          p1 = af_flash_drop(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            sv += __shfl_xor_sync(0xffffffffu, sv, off);
          if (c0 == 0) ssr[ri] += sv;
        }
        const int row = r0 + 4 * q;
        pw[af_off_row(row, c0)] = __float2half(p0);
        pw[af_off_row(row, 16 + c0)] = __float2half(p1);
      }
    }
    // 慢路径：上面用旧基准算了一遍，若越界就按新基准修正并重写这一批 P。
    // 分母在快路径里已把旧值累加进去，乘 corr 即等于按新基准重算；
    // 首轮（firstq）快路径写进去的是 inf，必须直接取 sv 覆盖。
    // 注意：不能按 q 提前 continue——不同行组越界情况不同，跳过的 lane 会
    // 缺席后续 shfl 归约。这里所有 lane 都执行，只用 fix 谓词控制写回。
    if (HAS_MASK) {
      if (firstq || __any_sync(0xffffffffu, risky)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          {
            const __half* mr = mbase + (4 * q * T + kt);
            x0 += __half2float(mr[c0]) * AF_LOG2E;
            x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
          }
          float mn = mlr_reg[q], corr = 1.f;
          float mx = fmaxf(x0, x1);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
          const bool fix = mx > mn || first;
          if (fix) {
            corr = exp2f(mn - mx);
            mlr_reg[q] = mx;
            mn = mx;
          }
#pragma unroll
          for (int e = 0; e < 5; ++e) OF[e]->x[q] *= corr;
          float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
          if (HAS_DROP) {
            const long long base = dbase + 4 * q * T + kt;
            float sv = p0 + p1;
            p0 = af_flash_drop(p0, p_drop, drop_scale, seed, base + c0);
            p1 = af_flash_drop(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              sv += __shfl_xor_sync(0xffffffffu, sv, off);
            if (c0 == 0 && (fix || firstq)) ssr[ri] = firstq ? sv : ssr[ri] * corr;
          }
          // 首轮跳过了快路径，即使本行全是 -inf（mx 不提升）也必须写 pw。
          if (fix || firstq) {
            const int row = r0 + 4 * q;
            pw[af_off_row(row, c0)] = __float2half(p0);
            pw[af_off_row(row, 16 + c0)] = __float2half(p1);
          }
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
      if (!first) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float mn = mlr_reg[q];
        float x0 = TF[0]->x[q] * sc2, x1 = TF[1]->x[q] * sc2;
        if (HAS_MASK) {
          const __half* mr = mbase + (4 * q * T + kt);
          x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
          x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
        }
        if (HAS_MASK) risky |= fmaxf(x0, x1) > mn + 8.f;
        float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
        if (HAS_DROP) {
          const long long base = dbase + 4 * q * T + kt + 32;
          float sv = p0 + p1;
          p0 = af_flash_drop(p0, p_drop, drop_scale, seed, base + c0);
          p1 = af_flash_drop(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            sv += __shfl_xor_sync(0xffffffffu, sv, off);
          if (c0 == 0) ssr[ri] += sv;
        }
        const int row = r0 + 4 * q;
        pw[af_off_row(row, c0)] = __float2half(p0);
        pw[af_off_row(row, 16 + c0)] = __float2half(p1);
      }
      }
      if (HAS_MASK) {
        if (firstq || __any_sync(0xffffffffu, risky)) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int ri = r0 + 4 * q;
            float x0 = TF[0]->x[q] * sc2, x1 = TF[1]->x[q] * sc2;
            {
              const __half* mr = mbase + (4 * q * T + kt);
              x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
              x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
            }
            float mn = mlr_reg[q], corr = 1.f;
            float mx = fmaxf(x0, x1);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const bool fix = mx > mn || first;
            if (fix) {
              corr = exp2f(mn - mx);
              mlr_reg[q] = mx;
              mn = mx;
            }
#pragma unroll
            for (int e = 0; e < 5; ++e) OF[e]->x[q] *= corr;
            float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
            if (HAS_DROP) {
              const long long base = dbase + 4 * q * T + kt + 32;
              float sv = p0 + p1;
              p0 = af_flash_drop(p0, p_drop, drop_scale, seed, base + c0);
              p1 = af_flash_drop(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
              for (int off = 1; off < 16; off <<= 1)
                sv += __shfl_xor_sync(0xffffffffu, sv, off);
              // 批 B 的快路径一定执行过（first 只在纯 mask 首轮为真，而纯
              // mask 没有分母），且批 A 的慢路径已把基准定为有效值，快路径
              // 累加的是有效贡献 —— 这里只需缩放；若首轮再 +sv 会把本批
              // 算两遍（实测 md 误差 2.9）。
              if (c0 == 0 && (fix || firstq)) ssr[ri] = ssr[ri] * corr;
            }
            if (fix || firstq) {
              const int row = r0 + 4 * q;
              pw[af_off_row(row, c0)] = __float2half(p0);
              pw[af_off_row(row, 16 + c0)] = __float2half(p1);
            }
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
  if (row_base < S) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float den = HAS_DROP ? ssr[r0 + 4 * q] : OF[4]->x[q];
      const float inv = den != 0.f ? 1.f / den : 0.f;
      const int row = rb + r0 + 4 * q;
#pragma unroll
      for (int e = 0; e < 4; ++e)
        Ob[row * 64 + e * 16 + c0] = __float2half(OF[e]->x[q] * inv);
    }
  }
}

#define AF_FLASH_ARGS                                                        \
  const __half* __restrict__ Q, const __half* __restrict__ K,                \
      const __half* __restrict__ V, __half* __restrict__ O, int S, int T,    \
      float scale, const __half* __restrict__ M, float p_drop,               \
      float drop_scale, unsigned long long seed, int mask_shared

extern "C" __global__ void __launch_bounds__(AF_NTH16, 3) af_flash_attn_plain(
    AF_FLASH_ARGS) {
  af_attn_w16<false, false>(Q, K, V, O, S, T, scale, M, p_drop, drop_scale,
                            seed, mask_shared);
}
extern "C" __global__ void __launch_bounds__(AF_NTH16, 3) af_flash_attn_mask(
    AF_FLASH_ARGS) {
  af_attn_w16<true, false>(Q, K, V, O, S, T, scale, M, p_drop, drop_scale,
                           seed, mask_shared);
}
extern "C" __global__ void __launch_bounds__(AF_NTH16, 3) af_flash_attn_drop(
    AF_FLASH_ARGS) {
  af_attn_w16<false, true>(Q, K, V, O, S, T, scale, M, p_drop, drop_scale,
                           seed, mask_shared);
}
extern "C" __global__ void __launch_bounds__(AF_NTH16, 3) af_flash_attn_mask_drop(
    AF_FLASH_ARGS) {
  af_attn_w16<true, true>(Q, K, V, O, S, T, scale, M, p_drop, drop_scale,
                          seed, mask_shared);
}
"""


# TCU flash attention（f16，D=E=128）。
#
# D=64 版本（``_ILUVATAR_FLASH_SRC``）每个 mma 已经是 16x16x32、每 warp 16 行、
# 每轮 64 key（两批 32）；D 扩到 128 只改三处结构：
#   * QK 的 D 维从 2 段变 4 段（a0..a3，每段 16x16x32）；
#   * PV 的 E 维从 4 段变 8 段（o0..o7）加分母 o8；
#   * Q 装载从 2 趟变 4 趟（每趟 16x32，仍借 P 的 512 槽中转）。
# softmax / mask / dropout / 惰性基准的逐行结构原样沿用，所以两版数值行为一致
# （同一套 exp2 与 corr 修正顺序）。变的是并行粒度：32 warp × 16 行 =
# 每 block 512 行、2048 线程、``__launch_bounds__(_, 1)``，每头 K/V 只搬一次。
#
# bench 实测（bh=256、S=T=512、reps=20、tcu_d128_probe.cu）：
#   16 warp/2 block 1.1480 ms -> 32 warp/1 block 1.0292 ms -> 加 KV 预取 1.0176 ms。
# 这里取 32 warp 版本（不带预取：预取只值 1.2%，而它把轮末同步挪到轮首，
# 与 mask/dropout 的读路径耦合，收益不抵风险）。
# 动态 smem = 两批 K（各 8192）+ 两批 V（各 8192）+ P（32×512）+ ones（512）
# = 66560 B；天数驱动不强制 48/64 KB 的动态上限，实测可直接 launch。
_ILUVATAR_FLASH128_SRC = "%PRELUDE%" + _ILUVATAR_ATTN_HELPERS + r"""
#include <crt/iluvatar_mma.h>

using namespace ixix::wmma;

#define AF_NWT32 32
#define AF_NTH32 (AF_NWT32 * 64)
#define AF_LOG2E 1.4426950408889634f

__device__ __forceinline__ int af_off_row(int row, int col) {
  int i = col >> 4, c = col & 15, r = row >> 1;
  return (i * 256 + r / 4 * 128) + ((r * 32 + c * 2 + (row & 1) + i * 64) & 127);
}
__device__ __forceinline__ int af_off_col(int row, int col) {
  int r = row >> 1;
  return (col >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (col >> 3)) * 16 +
         ((r >> 2 & 1) ^ (col >> 2 & 1)) * 8 + (col & 3) * 2 + (row & 1);
}

// 32 个 key、D=128 → 每批 4096 half。block 固定 2048 线程，每线程正好负责
// 一个 idx（K 一个 32 位字 + V 一个 half2），不需要循环。
__device__ __forceinline__ void af_load_kv128(__half* ks, __half* vs,
                                              const __half* Kb0, const __half* Vb0,
                                              int kb) {
  const long long o = (long long)kb * 4096;
  const int idx = threadIdx.x;
  const int t = idx >> 6, d = (idx & 63) * 2;
  ((unsigned*)ks)[(((t >> 4) * 4 + (d >> 5)) * 512 + af_off_col(d & 31, t & 15)) >> 1]
      = ((const unsigned*)(Kb0 + o))[idx];
  const __half2 vv = *(const __half2*)(Vb0 + o + ((long long)t << 7) + d);
  vs[(d >> 4) * 512 + af_off_col(t, d & 15)] = __low2half(vv);
  vs[(d >> 4) * 512 + af_off_col(t, (d + 1) & 15)] = __high2half(vv);
}

__device__ __forceinline__ float af_flash_drop128(float p, float p_drop,
                                                  float drop_scale,
                                                  unsigned long long seed,
                                                  long long idx) {
  const unsigned z = attention_rand(seed, idx);
  const float u = (float)(z >> 8) * (1.0f / 16777216.0f);
  return (u >= p_drop) ? p * drop_scale : 0.f;
}

// 一个模板覆盖四个模式；结构与 D=64 版逐句对应，只把 QK 的 4 段 D 维、
// PV 的 8 段 E 维与 OF[9] 补齐。
template <bool HAS_MASK, bool HAS_DROP>
__device__ __forceinline__ void af_attn_w16_d128(
    const __half* __restrict__ Q, const __half* __restrict__ K,
    const __half* __restrict__ V, __half* __restrict__ O,
    int S, int T, float scale, const __half* __restrict__ M, float p_drop,
    float drop_scale, unsigned long long seed, int mask_shared) {
  extern __shared__ __half sm[];
  __half* ksm = sm;                          // 两批 K，各 4096
  __half* vsm = ksm + 8192;                  // 两批 V，各 4096
  __half* psm = vsm + 8192;                  // 每 warp 一个 16x32 的 P
  __half* ones_sm = psm + AF_NWT32 * 512;    // 16x16 全 1，用于 P@ones
  __shared__ float ssums[AF_NWT32 * 16];     // dropout 分母（每 warp 16 行）

  const int warp = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int r0 = lane >> 4;
  const int c0 = lane & 15;
  const int bh = blockIdx.y;
  const int row_base = blockIdx.x * (16 * AF_NWT32) + warp * 16;
  // 行数不是 512 的倍数时，尾部 warp 只读第 0 行、不写输出；块级同步照常。
  const int rb = (row_base < S) ? row_base : 0;

  const __half* Qb = Q + (long long)bh * S * 128;
  __half* Ob = O + (long long)bh * S * 128;
  const __half* Kb0 = K + (long long)bh * T * 128;
  const __half* Vb0 = V + (long long)bh * T * 128;
  __half* pw = psm + warp * 512;

  fragment<matrix_a, 16, 16, 32, __half, row_major> a0, a1, a2, a3, pa0;
  fragment<matrix_b, 16, 16, 32, __half, col_major> b0, b1, b2, b3, vbo;
  fragment<accumulator, 16, 16, 32, float> s0, s1;  // 批 A 分数（keys 0..31）
  fragment<accumulator, 16, 16, 32, float> t0, t1;  // 批 B 分数（keys 32..63）
  fragment<accumulator, 16, 16, 32, float> o0, o1, o2, o3, o4, o5, o6, o7, o8;

  // Q 装载：16 行 × 128 列，借 P 的 512 槽分四趟（每趟 16x32）。
  {
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      for (int idx = lane; idx < 512; idx += 64) {
        int m = idx >> 5, d = idx & 31;
        pw[af_off_row(m, d)] = Qb[(long long)(rb + m) * 128 + h * 32 + d];
      }
      __syncwarp();
      if (h == 0)      load_matrix_sync(a0, pw, 32);
      else if (h == 1) load_matrix_sync(a1, pw, 32);
      else if (h == 2) load_matrix_sync(a2, pw, 32);
      else             load_matrix_sync(a3, pw, 32);
      __syncwarp();
    }
  }

  fragment<accumulator, 16, 16, 32, float>* SF[2] = {&s0, &s1};
  fragment<accumulator, 16, 16, 32, float>* TF[2] = {&t0, &t1};
  fragment<accumulator, 16, 16, 32, float>* OF[9] = {&o0, &o1, &o2, &o3, &o4,
                                                     &o5, &o6, &o7, &o8};

#pragma unroll
  for (int e = 0; e < 9; ++e) fill_fragment(*OF[e], 0.f);

  for (int idx = threadIdx.x; idx < AF_NWT32 * 16; idx += AF_NTH32)
    ssums[idx] = 0.f;
  for (int idx = threadIdx.x; idx < 512; idx += AF_NTH32) {
    int t = idx >> 4, dd = idx & 15;
    ones_sm[af_off_col(t, dd)] = __float2half(1.f);
  }
  __syncthreads();
  load_matrix_sync(vbo, ones_sm, 32);

  const float sc2 = scale * AF_LOG2E;
  const float ml0 = -1e30f;
  float mlr_reg[4] = {ml0, ml0, ml0, ml0};
  float* ssr = ssums + warp * 16;

  const __half* mbase = nullptr;
  long long dbase = 0;
  if (HAS_MASK) {
    const int mr0 = (mask_shared ? (rb % S) : (bh * S + rb)) + r0;
    mbase = M + (long long)mr0 * T;
  }
  if (HAS_DROP) dbase = (long long)(bh * S + rb + r0) * T;

  for (int kt = 0; kt < T; kt += 64) {
    const bool hasB = (kt + 32 < T);
    af_load_kv128(ksm, vsm, Kb0, Vb0, kt >> 5);
    if (hasB) af_load_kv128(ksm + 4096, vsm + 4096, Kb0, Vb0, (kt >> 5) + 1);
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

    if (!HAS_MASK) {
      bool risk = false;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float lim = mlr_reg[q] + 8.f;
        float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
        float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
        risk |= fmaxf(fmaxf(x0, x1), fmaxf(x2, x3)) > lim;
      }
      if (__any_sync(0xffffffffu, risk)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          const float mn = mlr_reg[q];
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          float x2 = TF[0]->x[q] * sc2, x3 = TF[1]->x[q] * sc2;
          float mx = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
          float corr = 1.f;
          if (mx > mn) {
            corr = exp2f(mn - mx);
            mlr_reg[q] = mx;
          }
#pragma unroll
          for (int e = 0; e < 9; ++e) OF[e]->x[q] *= corr;
          if (HAS_DROP && c0 == 0) ssr[ri] *= corr;
        }
      }
    }

    __syncwarp();
    bool risky = false;
    const bool firstq = (kt == 0);
    const bool first = HAS_MASK && !HAS_DROP && firstq;
    if (!first) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int ri = r0 + 4 * q;
        const float mn = mlr_reg[q];
        float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
        if (HAS_MASK) {
          const __half* mr = mbase + (4 * q * T + kt);
          x0 += __half2float(mr[c0]) * AF_LOG2E;
          x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
        }
        if (HAS_MASK) risky |= fmaxf(x0, x1) > mn + 8.f;
        float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
        if (HAS_DROP) {
          const long long base = dbase + 4 * q * T + kt;
          float sv = p0 + p1;
          p0 = af_flash_drop128(p0, p_drop, drop_scale, seed, base + c0);
          p1 = af_flash_drop128(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            sv += __shfl_xor_sync(0xffffffffu, sv, off);
          if (c0 == 0) ssr[ri] += sv;
        }
        const int row = r0 + 4 * q;
        pw[af_off_row(row, c0)] = __float2half(p0);
        pw[af_off_row(row, 16 + c0)] = __float2half(p1);
      }
    }
    if (HAS_MASK) {
      if (firstq || __any_sync(0xffffffffu, risky)) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          float x0 = SF[0]->x[q] * sc2, x1 = SF[1]->x[q] * sc2;
          {
            const __half* mr = mbase + (4 * q * T + kt);
            x0 += __half2float(mr[c0]) * AF_LOG2E;
            x1 += __half2float(mr[16 + c0]) * AF_LOG2E;
          }
          float mn = mlr_reg[q], corr = 1.f;
          float mx = fmaxf(x0, x1);
#pragma unroll
          for (int off = 1; off < 16; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
          const bool fix = mx > mn || first;
          if (fix) {
            corr = exp2f(mn - mx);
            mlr_reg[q] = mx;
            mn = mx;
          }
#pragma unroll
          for (int e = 0; e < 9; ++e) OF[e]->x[q] *= corr;
          float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
          if (HAS_DROP) {
            const long long base = dbase + 4 * q * T + kt;
            float sv = p0 + p1;
            p0 = af_flash_drop128(p0, p_drop, drop_scale, seed, base + c0);
            p1 = af_flash_drop128(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              sv += __shfl_xor_sync(0xffffffffu, sv, off);
            if (c0 == 0 && (fix || firstq)) ssr[ri] = firstq ? sv : ssr[ri] * corr;
          }
          if (fix || firstq) {
            const int row = r0 + 4 * q;
            pw[af_off_row(row, c0)] = __float2half(p0);
            pw[af_off_row(row, 16 + c0)] = __float2half(p1);
          }
        }
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
    if (!HAS_DROP) mma_sync(o8, pa0, vbo, o8);

    if (hasB) {
      __syncwarp();
      risky = false;
      if (!first) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const int ri = r0 + 4 * q;
          const float mn = mlr_reg[q];
          float x0 = TF[0]->x[q] * sc2, x1 = TF[1]->x[q] * sc2;
          if (HAS_MASK) {
            const __half* mr = mbase + (4 * q * T + kt);
            x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
            x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
          }
          if (HAS_MASK) risky |= fmaxf(x0, x1) > mn + 8.f;
          float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
          if (HAS_DROP) {
            const long long base = dbase + 4 * q * T + kt + 32;
            float sv = p0 + p1;
            p0 = af_flash_drop128(p0, p_drop, drop_scale, seed, base + c0);
            p1 = af_flash_drop128(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              sv += __shfl_xor_sync(0xffffffffu, sv, off);
            if (c0 == 0) ssr[ri] += sv;
          }
          const int row = r0 + 4 * q;
          pw[af_off_row(row, c0)] = __float2half(p0);
          pw[af_off_row(row, 16 + c0)] = __float2half(p1);
        }
      }
      if (HAS_MASK) {
        if (firstq || __any_sync(0xffffffffu, risky)) {
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int ri = r0 + 4 * q;
            float x0 = TF[0]->x[q] * sc2, x1 = TF[1]->x[q] * sc2;
            {
              const __half* mr = mbase + (4 * q * T + kt);
              x0 += __half2float(mr[32 + c0]) * AF_LOG2E;
              x1 += __half2float(mr[48 + c0]) * AF_LOG2E;
            }
            float mn = mlr_reg[q], corr = 1.f;
            float mx = fmaxf(x0, x1);
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
              mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const bool fix = mx > mn || first;
            if (fix) {
              corr = exp2f(mn - mx);
              mlr_reg[q] = mx;
              mn = mx;
            }
#pragma unroll
            for (int e = 0; e < 9; ++e) OF[e]->x[q] *= corr;
            float p0 = exp2f(x0 - mn), p1 = exp2f(x1 - mn);
            if (HAS_DROP) {
              const long long base = dbase + 4 * q * T + kt + 32;
              float sv = p0 + p1;
              p0 = af_flash_drop128(p0, p_drop, drop_scale, seed, base + c0);
              p1 = af_flash_drop128(p1, p_drop, drop_scale, seed, base + 16 + c0);
#pragma unroll
              for (int off = 1; off < 16; off <<= 1)
                sv += __shfl_xor_sync(0xffffffffu, sv, off);
              if (c0 == 0 && (fix || firstq)) ssr[ri] = ssr[ri] * corr;
            }
            if (fix || firstq) {
              const int row = r0 + 4 * q;
              pw[af_off_row(row, c0)] = __float2half(p0);
              pw[af_off_row(row, 16 + c0)] = __float2half(p1);
            }
          }
        }
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
      if (!HAS_DROP) mma_sync(o8, pa0, vbo, o8);
    }
    __syncthreads();
  }

  if (row_base < S) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float den = HAS_DROP ? ssr[r0 + 4 * q] : OF[8]->x[q];
      const float inv = den != 0.f ? 1.f / den : 0.f;
      const int row = rb + r0 + 4 * q;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        Ob[row * 128 + e * 16 + c0] = __float2half(OF[e]->x[q] * inv);
    }
  }
}

#define AF_FLASH_ARGS                                                        \
  const __half* __restrict__ Q, const __half* __restrict__ K,                \
      const __half* __restrict__ V, __half* __restrict__ O, int S, int T,    \
      float scale, const __half* __restrict__ M, float p_drop,               \
      float drop_scale, unsigned long long seed, int mask_shared

extern "C" __global__ void __launch_bounds__(AF_NTH32, 1) af_flash_attn_d128_plain(
    AF_FLASH_ARGS) {
  af_attn_w16_d128<false, false>(Q, K, V, O, S, T, scale, M, p_drop,
                                 drop_scale, seed, mask_shared);
}
extern "C" __global__ void __launch_bounds__(AF_NTH32, 1) af_flash_attn_d128_mask(
    AF_FLASH_ARGS) {
  af_attn_w16_d128<true, false>(Q, K, V, O, S, T, scale, M, p_drop,
                                drop_scale, seed, mask_shared);
}
extern "C" __global__ void __launch_bounds__(AF_NTH32, 1) af_flash_attn_d128_drop(
    AF_FLASH_ARGS) {
  af_attn_w16_d128<false, true>(Q, K, V, O, S, T, scale, M, p_drop,
                                drop_scale, seed, mask_shared);
}
extern "C" __global__ void __launch_bounds__(AF_NTH32, 1) af_flash_attn_d128_mask_drop(
    AF_FLASH_ARGS) {
  af_attn_w16_d128<true, true>(Q, K, V, O, S, T, scale, M, p_drop,
                               drop_scale, seed, mask_shared);
}
"""


def flash_attention_source():
    """f16 TCU flash attention 源码（D=E=64，mask/dropout 由启动器区分）。"""
    return _ILUVATAR_FLASH_SRC


def flash_attention128_source():
    """f16 TCU flash attention 源码（D=E=128，块内 512 行 / 2048 线程）。"""
    return _ILUVATAR_FLASH128_SRC


def attention_scores_source(np_reg: int = 0):
    """GEMM 路径的 softmax+mask+dropout 内核源码。

    ``np_reg`` 为 0 时生成两遍版本（先扫 max，再读一遍写 exp）；大于 0 时
    生成"整行驻留寄存器"的特化版本，要求调用方保证 ``T == 64 * np_reg``
    且存储宽度为 16 位。寄存器版把 HBM 读从两遍降到一遍。
    """
    return _ILUVATAR_SCORES_SRC.replace("%NP%", str(int(np_reg)))


def attention_rescale_source():
    """GEMM 路径的输出行缩放内核源码。"""
    return _ILUVATAR_RESCALE_SRC


def attention_source():
    """天数的融合注意力源码（字符串模板，由 ``render`` 按 dtype 展开）。

    CUDA 侧同名函数返回 L2 描述；这里返回手写模板是有意为之：这条内核的
    收益全部来自"子组协作 + 合并读"，用 L2 描述表达不出来。语义仍与共享
    模板逐项对齐，`run_conformance` 负责守住这一点。
    """
    return _ILUVATAR_ATTENTION_SRC


def render(template: str, dtype) -> str:
    """展开模板；f16 的读写与原子加换成本方言的实现。"""
    # 延迟导入：dialect 依赖 cuda.codegen，顶层导入会成环。
    from .dialect import ILUVATAR_DIALECT

    return _cuda.render(
        template, dtype, f16_helpers=_ILUVATAR_F16_HELPERS,
        dialect=ILUVATAR_DIALECT,
    )


def cast_fn(src, dst) -> str:
    """``f32 <-> f16`` 的转换源码（L2 描述与 CUDA 逐字共用）。"""
    # 延迟导入：dialect 顶层就 import 了本模块。
    from .dialect import CAST_DIALECT

    return _cuda.cast_fn(src, dst, dialect=CAST_DIALECT)


def __getattr__(name: str):
    """未定义的常量/函数一律来自 CUDA 侧的模板库（单一事实来源）。"""
    return getattr(_cuda, name)


__all__ = [
    "render",
    "cast_fn",
    "attention_source",
    "attention_scores_source",
    "attention_rescale_source",
]
