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
_ILUVATAR_ATTENTION_SRC = (
    "%PRELUDE%"
    + r"""
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

__device__ __forceinline__ unsigned long long attention_rand(unsigned long long seed,
                                                             long long idx) {
    unsigned long long z =
        seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

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
                const unsigned long long z =
                    attention_rand(seed, (bh * S + s) * T + t);
                const float u =
                    (float)((z >> 40) & 0xFFFFFFULL) * (1.0f / 16777216.0f);
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
)

__all__ = ["render", "cast_fn", "attention_source"]


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
