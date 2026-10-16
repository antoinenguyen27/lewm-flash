// LeWM-MK: the whole LeWM Push-T iCEM solve (30 steps x [sample -> action encoder -> AdaLN table -> 5-step
// autoregressive predictor rollout -> cost -> top-k -> elite mean/std]) as ONE persistent H100 kernel launch.
//
// Layout
//   * Grid = G clusters x CL CTAs (CL = 8 or 16), one CTA per SM, all co-resident (grid barriers are spin barriers).
//   * Each CTA: 2 warpgroups (8 warps, the full 255-register budget). Each streams 24 KB pre-swizzled bf16 weight
//     panels from L2 into its own 2-slot shared-memory ring with cp.async.bulk; its thread 0 refills a slot as soon as
//     the warpgroup releases it, replaying the static panel sequence (PanelIter), so prefetch runs straight through op,
//     layer, pass and barrier boundaries.
//   * Batched envs: every replanning env's candidates share one row space; top-k is per env (one warpgroup each).
//     With many rows each warpgroup takes its own 64-row tile (DP mode); with few rows they split heads/slices (TP).
//   * Predictor rollout: a cluster owns a 64-row chunk (64 candidates x 1 token for passes 0-2 thanks to an exact KV
//     cache, 21 candidates x 3 tokens for passes 3-4). The cluster is tensor-parallel over the CL CTAs: each CTA owns
//     16/CL heads and 2048/CL MLP / pred_proj hidden units. Per half-layer: GEMMs -> bf16 partial in local smem ->
//     cluster barrier -> row owners reduce partials over DSMEM, apply gated residual (fp32) + LayerNorm + AdaLN
//     modulation and broadcast the bf16 next-GEMM operand into every CTA's smem -> cluster barrier.
//   * Action encoder and AdaLN modulation (which depend only on the candidate actions) are computed once per iCEM step
//     for every (candidate, timestep) into a bf16 table; noise is in-kernel Philox; top-k / elite statistics are computed
//     redundantly by every CTA so a single grid barrier publishes the costs.
//
// Numerics mirror the accepted LeWM-Flash BF16 autocast path: bf16 GEMM operands, fp32 accumulation, fp32 residual
// stream and LayerNorm statistics, bf16 AdaLN table / q,k,v / attention output, fp32 costs and iCEM statistics.

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

using namespace kittens;
namespace cg = cooperative_groups;

namespace lewm {

constexpr int D = 192, NH = 16, DH = 64, FF = 2048, NL = 6, HOR = 5, AD = 10, NE = HOR * AD, HID_E = 768;
constexpr int MODW = 6 * D;  // shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp
constexpr int SMAX = 300, ROWS = 64, RS = 2, PANEL_BYTES = 192 * 64 * 2;  // RS = ring slots per warpgroup
constexpr int NWG = 2, NCONS = 128 * NWG, NTHREADS = NCONS;  // 2 warpgroups = 8 warps: the full 255-register budget
constexpr int CPC_SHORT = 64, CPC_LONG = 21;  // candidates per chunk: passes 0-2 (1 token) / 3-4 (3 tokens)
constexpr int MB_CANDS = 12;                  // candidates per action-encoder / AdaLN row block (60 rows)

// Panel ids (host packer in pack.py uses the same layout). P = [K=192, N=64], Q = [K=64, N=192], row-major source.
constexpr int OFF_QKV = 0, OFF_WO = 288, OFF_W1 = 384, OFF_W2 = 576, OFF_P1 = 768, OFF_P2 = 800, OFF_E1 = 832,
              OFF_E2 = 844, OFF_ADA = 856, NPANEL = 964;

using xa_t = st_bf<64, 192>;
using ht_t = st_bf<64, 64>;
using pP_t = st_bf<192, 64>;
using pQ_t = st_bf<64, 192>;
using acc64_t = rt_fl<16, 64>;
using acc192_t = rt_fl<16, 192>;

struct Params {
    const uint8_t *panels;
    const float *b_qkv, *b_o, *b_1, *b_2, *b_p1, *b_p2, *b_e1, *b_e2, *b_ada, *pos, *lnf_g, *lnf_b;
    const float *emb0, *goal, *init_mean, *cand_in;
    float *cand, *emb_hist, *cost, *out, *cost_out, *trace;
    unsigned long long *prof;  // optional [grid][16] per-CTA consumer time breakdown (ns)
    bf16 *actemb, *mod, *kv;
    unsigned *bar_counter;
    int *err;
    float *st_mean, *st_std, *st_cost;  // per-env iCEM state [E][NE], [E][NE], [E] (published by the top-k phase)
    int *st_eidx;                       // [E][8] indices of the previous step's best elites
    float init_std;
    int mode, n_steps, topk, elite_ret, clusters, E;
    int S[32];
    unsigned long long seed;
};

// ------------------------------------------------------------------------------------------------ small helpers
__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
#ifdef LEWM_MK_PROF
#define PNOW() now_ns()
#define PADD(i, t0) do { if (threadIdx.x == 0) s.prof[i] += now_ns() - (t0); } while (0)
#else
#define PNOW() 0ull
#define PADD(i, t0) do { (void)(t0); } while (0)
#endif
#ifdef LEWM_MK_PROF
#define PADD_P(i, t0) do { if (w == 0) s.prof[i] += now_ns() - (t0); } while (0)
#else
#define PADD_P(i, t0) do { (void)(t0); } while (0)
#endif
// No function calls anywhere on the consumer path: a call between WGMMA issue and wait makes ptxas serialize them.
__device__ __forceinline__ void fail(const Params &P, int code) {
    atomicCAS(P.err, 0, code);
    asm volatile("trap;");
}
__device__ __forceinline__ void cons_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
__device__ __forceinline__ void wg_sync(int w) { asm volatile("bar.sync %0, 128;" ::"r"(2 + w) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ float ldcg(const float *p) { return __ldcg(p); }
__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }
// Exact-GELU via Abramowitz & Stegun 7.1.26 erf (|err| <= 1.5e-7, i.e. exact at fp32-for-bf16 precision); libm erff
// costs ~6x more on the latency-bound epilogue path.
__device__ __forceinline__ float erf_fast(float x) {
    const float ax = fabsf(x), t = __fdividef(1.f, 1.f + 0.3275911f * ax);
    const float y = 1.f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t *
                              __expf(-ax * ax);
    return copysignf(y, x);
}
__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erf_fast(v * 0.70710678118654752f)); }
__device__ __forceinline__ float ldbf(const bf16 *p) {
    unsigned short u = __ldcg(reinterpret_cast<const unsigned short *>(p));
    return __uint_as_float(((unsigned)u) << 16);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
template <typename ST> __device__ __forceinline__ uint32_t st_off(int r, int c) {  // byte offset in a 1024-aligned tile
    return ST::idx(0u, int2{r, c});
}
template <typename ST> __device__ __forceinline__ bf16 *st_ptr(void *base, int r, int c) {
    return reinterpret_cast<bf16 *>(reinterpret_cast<char *>(base) + st_off<ST>(r, c));
}
__device__ __forceinline__ uint32_t smem_u32(const void *p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

// Accumulator fragment iteration (TK row layout, wgmma m64nNk16 C fragments): each thread owns float2 pairs.
template <int N, typename F> __device__ __forceinline__ void frag_for(rt_fl<16, N> &acc, F &&f) {
    const int lane = threadIdx.x & 31, w = (threadIdx.x >> 5) & 3;
#pragma unroll
    for (int j = 0; j < N / 16; j++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int r = w * 16 + (lane >> 2) + ((k & 1) ? 8 : 0);
            const int c = j * 16 + 2 * (lane & 3) + ((k & 2) ? 8 : 0);
            f(r, c, acc.tiles[0][j].data[k]);
        }
}
template <int N> __device__ __forceinline__ void frag_zero(rt_fl<16, N> &acc) {
    frag_for(acc, [](int, int, float2 &v) { v.x = 0.f; v.y = 0.f; });
}

// ------------------------------------------------------------------------------------------------ shared memory
// Two row tiles per CTA. Rollout passes run in one of two modes:
//   TP mode (few rows, e.g. one env): both warpgroups work on tile 0 and split this CTA's heads / hidden slices.
//   DP mode (many rows, e.g. batched envs): warpgroup w runs all of this CTA's heads / slices on its own tile w.
// The q/k/v scratch of warpgroup w lives inside part[w] (it is dead before the partial is written there).
struct Smem {
    uint8_t ring[NWG][RS][PANEL_BYTES];  // 1024-aligned (struct is placed at a 1024-aligned address)
    uint8_t xa[NWG][sizeof(xa_t)];       // bf16 [64,192] GEMM A operand per tile (K=192); WG w's A in phases M1/M2
    uint8_t part[NWG][sizeof(xa_t)];     // bf16 [64,192] TP partial per tile (read remotely by row owners) | q,k,v
    uint8_t ht[NWG][sizeof(ht_t)];       // bf16 [64,64] attention output / MLP hidden slice (A operand, K=64)
    float xown[NWG][8][D];               // fp32 residual of the rows this CTA owns, per tile
    union {
        float cbuf[NWG][SMAX];             // one env's costs (top-k), per warpgroup
        float candbuf[NWG][MB_CANDS][NE];  // candidate actions of the current action-encoder block
    } u;
    int idx[NWG][32];
    int slot_tag[NWG][RS];
    unsigned long long prof[16];
    semaphore full[NWG][RS], empty[NWG][RS], cbar[2];
};
static_assert(sizeof(Smem) + 1024 <= 232448, "shared memory budget");

// ------------------------------------------------------------------------------------------------ synchronization
__device__ __forceinline__ void spin_timeout(const Params &P, unsigned long long t0, int code) {
    if (t0 > (1ull << 26)) fail(P, code);  // t0 = spin-iteration count / 256 (~seconds of spinning)
}
__device__ __forceinline__ void grid_sync(const Params &P, Smem &s, unsigned &epoch) {
    const unsigned long long tp = PNOW();
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    cons_sync();
    if (threadIdx.x == 0) {
        epoch++;
        const unsigned target = epoch * gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(P.bar_counter) : "memory");
        unsigned v;
        unsigned long long it = 0;
        while (true) {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(P.bar_counter) : "memory");
            if (v >= target) break;
            if ((++it & 255) == 0) spin_timeout(P, it >> 8, 10);
        }
    }
    cons_sync();
    PADD(2, tp);
}
template <int CL> struct ClusterSync {
    int phase = 0;
    __device__ __forceinline__ void sync(const Params &P, Smem &s) {
        const unsigned long long tp = PNOW();
        cons_sync();
        const int b = phase & 1;
        if (threadIdx.x == 0) {
            // One cluster-scope release fence (cumulative over the CTA's writes ordered by the bar.sync above), then
            // relaxed remote arrives: a .release on every arrive serializes them (~0.4 us each).
            asm volatile("fence.acq_rel.cluster;" ::: "memory");
            const uint32_t local = smem_u32(&s.cbar[b]);
#pragma unroll
            for (int r = 0; r < CL; r++) {
                uint32_t remote;
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(r));
                asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(remote) : "memory");
            }
            const int parity = (phase >> 1) & 1;
            const unsigned long long t0 = PNOW();
            unsigned long long it = 0;
            while (true) {
                uint32_t ok;
                asm volatile(
                    "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                    " selp.u32 %0, 1, 0, p;\n}\n"
                    : "=r"(ok) : "r"(local), "r"(parity) : "memory");
                if (ok) break;
                if ((++it & 255) == 0) spin_timeout(P, it >> 8, 11);
            }
        }
        phase++;
        cons_sync();
        fence_async_smem();
        PADD(1, tp);
    }
};

// ------------------------------------------------------------------------------------------------ weight ring
struct RingP {  // producer side (one thread per warpgroup ring)
    int w, n = 0;
    __device__ __forceinline__ void put(const Params &P, Smem &s, int id) {
        const int slot = n % RS, par = ((n / RS) & 1) ^ 1;
        const unsigned long long t0 = PNOW();
        unsigned long long it = 0;
        while (true) {
            uint32_t ok;
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                : "=r"(ok) : "r"(smem_u32(&s.empty[w][slot])), "r"(par) : "memory");
            if (ok) break;
            if ((++it & 255) == 0) spin_timeout(P, it >> 8, 21);
        }
        PADD_P(11, t0);
        s.slot_tag[w][slot] = id;
        const uint32_t bar = smem_u32(&s.full[w][slot]);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(PANEL_BYTES) : "memory");
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            ::"r"(smem_u32(s.ring[w][slot])), "l"(P.panels + (size_t)id * PANEL_BYTES), "r"(PANEL_BYTES), "r"(bar)
            : "memory");
        n++;
    }
};

// GEMM primitives (one warpgroup). Panels arrive pre-swizzled in the exact st_bf layout.
template <typename Ring>
__device__ __forceinline__ void gemm_P(acc64_t &acc, void *a, Smem &s, const Params &P, Ring &ring, int id) {
    auto &w = *reinterpret_cast<pP_t *>(ring.take(P, s, id));
    const unsigned long long t0 = PNOW();
    warpgroup::mm_AB(acc, *reinterpret_cast<xa_t *>(a), w);
    warpgroup::mma_async_wait();
    PADD(4, t0);
    ring.give(P, s);
}
template <typename Ring>
__device__ __forceinline__ void gemm_Q(acc192_t &acc, void *a, Smem &s, const Params &P, Ring &ring, int id) {
    auto &w = *reinterpret_cast<pQ_t *>(ring.take(P, s, id));
    const unsigned long long t0 = PNOW();
    warpgroup::mma_AB(acc, *reinterpret_cast<ht_t *>(a), w);
    warpgroup::mma_async_wait();
    PADD(4, t0);
    ring.give(P, s);
}
// acc64 (+bias, activation) -> bf16 64x64 tile in smem
template <typename F>
__device__ __forceinline__ void acc_to_tile(acc64_t &acc, void *tile, const float *bias, F &&act, Smem &s) {
    const unsigned long long t0 = PNOW();
    // All bias loads first: they cannot be reordered past the (possibly aliasing) generic shared-memory stores.
    const int lane = threadIdx.x & 31;
    float2 bv[4][2];
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) bv[j][k2] = __ldg(reinterpret_cast<const float2 *>(bias + j * 16 + 2 * (lane & 3) + 8 * k2));
    frag_for(acc, [&](int r, int c, float2 &v) {
        const float2 b2 = bv[c / 16][(c / 8) & 1];
        const float a = act(v.x + b2.x), b = act(v.y + b2.y);
        *reinterpret_cast<bf16_2 *>(st_ptr<ht_t>(tile, r, c)) = __floats2bfloat162_rn(a, b);
    });
    PADD(5, t0);
}

// ------------------------------------------------------------------------------------------------ schedule
struct Pass {
    int cpc, ntok, nunits;
    bool dp;  // two 64-row tiles per cluster unit (one per warpgroup) when 64-row chunks outnumber the clusters
};
__device__ __forceinline__ Pass pass_shape(int S, int p, int G) {
    Pass ps;
    ps.cpc = p < 3 ? CPC_SHORT : CPC_LONG;
    ps.ntok = p < 3 ? 1 : 3;
    const int n64 = (S + ps.cpc - 1) / ps.cpc;
    ps.dp = n64 > G;
    ps.nunits = ps.dp ? (S + 2 * ps.cpc - 1) / (2 * ps.cpc) : n64;
    return ps;
}
__device__ __forceinline__ int n_mblocks(int S) { return (S + MB_CANDS - 1) / MB_CANDS; }

template <int CL> __device__ __forceinline__ void producer(const Params &P, Smem &s, int wg) {
    constexpr int HP = NH / CL, MS = FF / CL / 64;
    RingP ring;
    ring.w = wg;
    const int cta = blockIdx.x, ncta = gridDim.x, g = cta / CL, rank = cta % CL, G = gridDim.x / CL;
    const int worker = cta * NWG + wg, nworkers = ncta * NWG;
    for (int step = 0; step < P.n_steps; step++) {
        const int S = P.S[step] * P.E, nrb = n_mblocks(S);  // all envs' candidates form one row space
        for (int u = worker; u < nrb; u += nworkers)
            for (int j = 0; j < HID_E / 64; j++) {
                ring.put(P, s, OFF_E1 + j);
                ring.put(P, s, OFF_E2 + j);
            }
        for (int u = worker; u < nrb * 12; u += nworkers) {
            const int l = (u % 12) / 2, half = u % 2;
            for (int n = 0; n < 9; n++) ring.put(P, s, OFF_ADA + l * 18 + half * 9 + n);
        }
        for (int p = 0; p < HOR; p++) {
            const Pass ps = pass_shape(S, p, G);
            const int h0 = ps.dp ? 0 : wg, hs = ps.dp ? 1 : NWG;
            for (int q = g; q < ps.nunits; q += G) {
                for (int l = 0; l < NL; l++) {
                    for (int hh = h0; hh < HP; hh += hs) {
                        const int h = rank * HP + hh;
                        for (int part = 0; part < 3; part++) ring.put(P, s, OFF_QKV + (l * NH + h) * 3 + part);
                        ring.put(P, s, OFF_WO + l * NH + h);
                    }
                    for (int ss = h0; ss < MS; ss += hs) {
                        const int sl = rank * MS + ss;
                        ring.put(P, s, OFF_W1 + l * 32 + sl);
                        ring.put(P, s, OFF_W2 + l * 32 + sl);
                    }
                }
                for (int ss = h0; ss < MS; ss += hs) {
                    ring.put(P, s, OFF_P1 + rank * MS + ss);
                    ring.put(P, s, OFF_P2 + rank * MS + ss);
                }
            }
        }
    }
}

// Resumable enumeration of one warpgroup's weight-panel sequence: exactly the order of producer<CL>(), as a state
// machine, so the warpgroup's own thread 0 can refill a ring slot the moment the warpgroup releases it (no producer
// warp: a 9th warp would cap every thread at 168 registers).
template <int CL> struct PanelIter {
    static constexpr int HP = NH / CL, MS = FF / CL / 64;
    int wg, worker, nworkers, g, G, rank;
    int step, ph, u, j, p, q, l, k, S, nrb, h0, hs, nh, ns, nunits;
    __device__ __forceinline__ void start_step(const Params &P) {
        ph = 0; j = 0; u = worker;
        if (step < P.n_steps) { S = P.S[step] * P.E; nrb = n_mblocks(S); }
    }
    __device__ __forceinline__ void start_pass() {
        const Pass ps = pass_shape(S, p, G);
        h0 = ps.dp ? 0 : wg; hs = ps.dp ? 1 : NWG;
        nh = h0 < HP ? (HP - h0 + hs - 1) / hs : 0;
        ns = h0 < MS ? (MS - h0 + hs - 1) / hs : 0;
        nunits = ps.nunits; q = g; l = 0; k = 0;
    }
    __device__ __forceinline__ void init(const Params &P, int w) {
        wg = w; worker = blockIdx.x * NWG + w; nworkers = gridDim.x * NWG;
        g = blockIdx.x / CL; G = gridDim.x / CL; rank = blockIdx.x % CL;
        step = 0; start_step(P);
    }
    __device__ __forceinline__ int next(const Params &P) {
        while (true) {
            if (step >= P.n_steps) return -1;
            if (ph == 0) {
                if (u < nrb) {
                    if (j < 2 * (HID_E / 64)) { const int id = (j & 1) ? OFF_E2 + j / 2 : OFF_E1 + j / 2; j++; return id; }
                    u += nworkers; j = 0; continue;
                }
                ph = 1; u = worker; j = 0; continue;
            }
            if (ph == 1) {
                if (u < nrb * 12) {
                    if (j < 9) { const int id = OFF_ADA + ((u % 12) / 2) * 18 + (u % 2) * 9 + j; j++; return id; }
                    u += nworkers; j = 0; continue;
                }
                ph = 2; p = 0; start_pass(); continue;
            }
            if (p >= HOR) { step++; start_step(P); continue; }
            if (q >= nunits) { p++; if (p < HOR) start_pass(); continue; }
            if (l < NL) {
                if (k < nh * 4) {
                    const int hh = h0 + (k / 4) * hs, h = rank * HP + hh, part = k % 4;
                    k++;
                    return part < 3 ? OFF_QKV + (l * NH + h) * 3 + part : OFF_WO + l * NH + h;
                }
                if (k < nh * 4 + ns * 2) {
                    const int k2 = k - nh * 4, sl = rank * MS + h0 + (k2 / 2) * hs;
                    k++;
                    return (k2 & 1) ? OFF_W2 + l * 32 + sl : OFF_W1 + l * 32 + sl;
                }
                l++; k = 0; continue;
            }
            if (k < ns * 2) {
                const int sl = rank * MS + h0 + (k / 2) * hs;
                const int id = (k & 1) ? OFF_P2 + sl : OFF_P1 + sl;
                k++;
                return id;
            }
            q += G; l = 0; k = 0;
        }
    }
};

__device__ __forceinline__ void issue_panel(const Params &P, Smem &s, int w, int slot, int id) {
    s.slot_tag[w][slot] = id;
    const uint32_t bar = smem_u32(&s.full[w][slot]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(PANEL_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(smem_u32(s.ring[w][slot])), "l"(P.panels + (size_t)id * PANEL_BYTES), "r"(PANEL_BYTES), "r"(bar)
                 : "memory");
}

// Consumer-side ring of one warpgroup: waits for a panel, and after the whole warpgroup is done with it, thread 0
// refills the slot with the next panel of the static sequence.
template <int CL> struct RingT {
    int w, n = 0;
    PanelIter<CL> it;
    __device__ __forceinline__ void start(const Params &P, Smem &s, int w_) {
        w = w_;
        it.init(P, w);
        if ((threadIdx.x & 127) == 0)
            for (int i = 0; i < RS; i++) {
                const int id = it.next(P);
                if (id >= 0) issue_panel(P, s, w, i, id);
            }
    }
    __device__ __forceinline__ void *take(const Params &P, Smem &s, int id) {
        const int slot = n % RS, par = (n / RS) & 1;
        const unsigned long long t0 = PNOW();
        unsigned long long iters = 0;
        while (true) {
            uint32_t ok;
            asm volatile(
                "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
                : "=r"(ok) : "r"(smem_u32(&s.full[w][slot])), "r"(par) : "memory");
            if (ok) break;
            if ((++iters & 255) == 0) spin_timeout(P, iters >> 8, 20);
        }
        PADD(0, t0);
        if (s.slot_tag[w][slot] != id) fail(P, 1000 + (id & 0xffff));
        return s.ring[w][slot];
    }
    __device__ __forceinline__ void give(const Params &P, Smem &s) {
        const int slot = n % RS;
        wg_sync(w);  // every warp of the warpgroup has retired its WGMMAs reading this slot
        if ((threadIdx.x & 127) == 0) {
            const int id = it.next(P);
            if (id >= 0) issue_panel(P, s, w, slot, id);
        }
        n++;
    }
};
template <int CL> using RingC = RingT<CL>;

// ------------------------------------------------------------------------------------------------ consumer phases
__device__ __forceinline__ float philox_normal(unsigned long long seed, int step, int env, int c, int el) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed + 0x9E3779B97F4A7C15ull * (unsigned long long)env, (unsigned long long)((step * SMAX + c) * NE + el), 0, &st);
    return curand_normal(&st);
}

// Phase M1: candidate generation + action encoder for one block of 12 candidates (60 (c,t) rows).
template <typename Ring>
__device__ __forceinline__ void phase_act(const Params &P, Smem &s, Ring &ring, int step, int S, int rb) {
    // S here = E * samples (global candidate rows gi = env * samples + c)
    const int w = ring.w, tid = threadIdx.x & 127, par = step & 1, c0 = rb * MB_CANDS;
    const int spe = P.S[step], spe_prev = step > 0 ? P.S[step - 1] : spe;
    void *A = s.xa[w];
    void *HT = s.ht[w];
    float(&candbuf)[MB_CANDS][NE] = s.u.candbuf[w];
    float *cand = P.cand + (size_t)par * P.E * SMAX * NE;
    const float *cand_prev = P.cand + (size_t)(par ^ 1) * P.E * SMAX * NE;
    for (int i = tid; i < MB_CANDS * NE; i += 128) {
        const int ci = i / NE, el = i % NE, gi = c0 + ci;
        float v = 0.f;
        if (gi < S) {
            const int env = gi / spe, c = gi % spe;
            if (P.mode == 1) v = P.cand_in[(size_t)gi * NE + el];
            else if (c == 0) v = __ldcg(P.st_mean + env * NE + el);
            else if (step > 0 && c <= P.elite_ret)
                v = __ldcg(cand_prev + ((size_t)env * spe_prev + __ldcg(P.st_eidx + env * 8 + c - 1)) * NE + el);
            else v = philox_normal(P.seed, step, env, c, el) * __ldcg(P.st_std + env * NE + el) + __ldcg(P.st_mean + env * NE + el);
            cand[(size_t)gi * NE + el] = v;
        }
        candbuf[ci][el] = v;
    }
    wg_sync(w);
    for (int i = tid; i < 64 * 192 / 2; i += 128) {  // A operand: row (c,t), cols 0..9 = action, rest 0
        const int r = i / 96, col = (i % 96) * 2;
        float a = 0.f, b = 0.f;
        if (r < MB_CANDS * HOR && col < AD) {
            a = candbuf[r / HOR][(r % HOR) * AD + col];
            b = candbuf[r / HOR][(r % HOR) * AD + col + 1];
        }
        *reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(A, r, col)) = __floats2bfloat162_rn(a, b);
    }
    fence_async_smem();
    wg_sync(w);
    acc192_t acc;
    frag_zero(acc);
    acc64_t h;
    for (int j = 0; j < HID_E / 64; j++) {
        gemm_P(h, A, s, P, ring, OFF_E1 + j);
        acc_to_tile(h, HT, P.b_e1 + j * 64, silu, s);
        fence_async_smem();
        wg_sync(w);
        gemm_Q(acc, HT, s, P, ring, OFF_E2 + j);
        wg_sync(w);
    }
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int c_ = c0 + r / HOR;
        if (r < MB_CANDS * HOR && c_ < S) {
            const float a = silu(v.x + __ldg(P.b_e2 + c)), b = silu(v.y + __ldg(P.b_e2 + c + 1));
            *reinterpret_cast<bf16_2 *>(P.actemb + (size_t)(c0 * HOR + r) * D + c) = __floats2bfloat162_rn(a, b);
        }
    });
}

// Phase M2: AdaLN modulation table for one (row block, layer, column half).
template <typename Ring>
__device__ __forceinline__ void phase_mod(const Params &P, Smem &s, Ring &ring, int S, int u) {
    const int w = ring.w, tid = threadIdx.x & 127, rb = u / 12, l = (u % 12) / 2, half = u % 2, r0 = rb * MB_CANDS * HOR;
    const int nrows = min(MB_CANDS * HOR, S * HOR - r0);
    void *A = s.xa[w];
    for (int i = tid; i < 64 * 96; i += 128) {
        const int r = i / 96, col = (i % 96) * 2;
        bf16_2 v = __floats2bfloat162_rn(0.f, 0.f);
        if (r < nrows) {
            const unsigned u32 = __ldcg(reinterpret_cast<const unsigned *>(P.actemb + (size_t)(r0 + r) * D + col));
            v = *reinterpret_cast<const bf16_2 *>(&u32);
        }
        *reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(A, r, col)) = v;
    }
    fence_async_smem();
    wg_sync(w);
    acc64_t acc;
    for (int n = 0; n < 9; n++) {
        const int col0 = (half * 9 + n) * 64;
        gemm_P(acc, A, s, P, ring, OFF_ADA + l * 18 + half * 9 + n);
        frag_for(acc, [&](int r, int c, float2 &v) {
            if (r < nrows) {
                const float a = v.x + __ldg(P.b_ada + l * MODW + col0 + c), b = v.y + __ldg(P.b_ada + l * MODW + col0 + c + 1);
                *reinterpret_cast<bf16_2 *>(P.mod + ((size_t)(r0 + r) * NL + l) * MODW + col0 + c) =
                    __floats2bfloat162_rn(a, b);
            }
        });
        wg_sync(w);
    }
}

// Row-owner helpers: one warp per row; lanes 0..23 own 8 contiguous columns each (one 16-byte vector), lanes 24..31
// idle. Every DSMEM partial read, DSMEM broadcast write and AdaLN-table read is a single 16-byte access.
struct RowCtx {
    int valid, c, tok, posi, t, env;  // global candidate row, emb-history token, window position, timestep, env
};
union V8 {
    uint4 u;
    bf16_2 h[4];
};
__device__ __forceinline__ void unpack8(const uint4 &u, float (&f)[8]) {
    V8 v;
    v.u = u;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        f[2 * k] = __low2float(v.h[k]);
        f[2 * k + 1] = __high2float(v.h[k]);
    }
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
    V8 v;
#pragma unroll
    for (int k = 0; k < 4; k++) v.h[k] = __floats2bfloat162_rn(f[2 * k], f[2 * k + 1]);
    return v.u;
}
__device__ __forceinline__ void ld8g(const bf16 *p, float (&f)[8]) { unpack8(__ldcg(reinterpret_cast<const uint4 *>(p)), f); }
__device__ __forceinline__ void ld8f(const float *p, float (&f)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4 *>(p)), b = __ldg(reinterpret_cast<const float4 *>(p) + 1);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
template <int CL> __device__ __forceinline__ void broadcast_row(Smem &s, int t, int row, const float (&v)[8], bool zero) {
    const int lane = threadIdx.x & 31;
    if (lane >= 24) return;
    cg::cluster_group cl = cg::this_cluster();
    const uint4 u = zero ? make_uint4(0, 0, 0, 0) : pack8(v);
    const uint32_t off = st_off<xa_t>(row, lane * 8);
#pragma unroll
    for (int r = 0; r < CL; r++) {
        char *base = reinterpret_cast<char *>(cl.map_shared_rank(reinterpret_cast<void *>(s.xa[t]), r));
        *reinterpret_cast<uint4 *>(base + off) = u;
    }
}
template <int CL> __device__ __forceinline__ void reduce_row(Smem &s, int t, int row, float (&sum)[8]) {  // += partials
    const int lane = threadIdx.x & 31;
    cg::cluster_group cl = cg::this_cluster();
    const uint32_t off = st_off<xa_t>(row, lane * 8);
    uint4 v[CL];
#pragma unroll
    for (int r = 0; r < CL; r++)
        v[r] = *reinterpret_cast<const uint4 *>(
            reinterpret_cast<const char *>(cl.map_shared_rank(reinterpret_cast<void *>(s.part[t]), r)) + off);
#pragma unroll
    for (int r = 0; r < CL; r++) {
        float f[8];
        unpack8(v[r], f);
#pragma unroll
        for (int k = 0; k < 8; k++) sum[k] += f[k];
    }
}
__device__ __forceinline__ void ln_stats(const float (&x)[8], bool act, float eps, float &mean, float &rstd) {
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) sum += act ? x[k] : 0.f;
    mean = warp_sum(sum) * (1.f / D);
    float sq = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) sq += act ? (x[k] - mean) * (x[k] - mean) : 0.f;
    rstd = rsqrtf(warp_sum(sq) * (1.f / D) + eps);
}
// LN(x, 1e-6) -> modulate(shift, scale) -> LN(., 1e-5) (the affine of the inner LayerNorm is folded into the weights)
__device__ __forceinline__ void lnmod(const float (&x)[8], bool act, const bf16 *modrow, int which, float (&o)[8]) {
    const int lane = threadIdx.x & 31;
    float shift[8], scale[8];
    if (act) {
        ld8g(modrow + which * 3 * D + lane * 8, shift);
        ld8g(modrow + which * 3 * D + D + lane * 8, scale);
    }
    float mean, rstd;
    ln_stats(x, act, 1e-6f, mean, rstd);
    float h[8];
#pragma unroll
    for (int k = 0; k < 8; k++) h[k] = act ? (x[k] - mean) * rstd * (1.f + scale[k]) + shift[k] : 0.f;
    ln_stats(h, act, 1e-5f, mean, rstd);
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = (h[k] - mean) * rstd;
}
template <int CL> __device__ __forceinline__ RowCtx row_ctx(int ri, int p, int cbase, int S, int spe) {
    RowCtx rc;
    if (p < 3) {
        rc.c = cbase + ri;
        rc.valid = ri < CPC_SHORT && rc.c < S;
        rc.tok = p;
        rc.posi = p;
        rc.t = p;
    } else {
        const int ci = ri / 3, j = ri % 3;
        rc.c = cbase + ci;
        rc.valid = ri < CPC_LONG * 3 && rc.c < S;
        rc.tok = p - 2 + j;
        rc.posi = j;
        rc.t = p - 2 + j;
    }
    rc.env = rc.c / spe;
    return rc;
}
__device__ __forceinline__ float dot8(const uint4 &a, const uint4 &b) {
    float fa[8], fb[8];
    unpack8(a, fa);
    unpack8(b, fb);
    float d = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) d += fa[k] * fb[k];
    return d;
}

// Attention for one head over this chunk: two threads per row, each owning 32 head dims (four 16-byte vectors).
// Keys are the row's own token, earlier tokens of the same candidate in this chunk (passes 3-4), or cached k/v of
// earlier tokens from the global KV cache (passes 1-2).
__device__ __forceinline__ void attention_head(const Params &P, Smem &s, int w, const RowCtx &rc, int ri, int hf,
                                               int p, int l, int h) {
    uint8_t *qkv = s.part[w], *ht = s.ht[w];
    const int d0 = hf * 32;
    uint4 o[4];
    if (rc.valid) {
        const size_t kvh = (((size_t)rc.c * 2) * NL + l) * NH + h;  // token 0 base
        const size_t tok_stride = (size_t)NL * NH;
        if (p < 2) {  // cache this token's k, v for the growing-window passes
            bf16 *dst = P.kv + (kvh + p * tok_stride) * 128;
#pragma unroll
            for (int ch = 0; ch < 4; ch++) {
                *reinterpret_cast<uint4 *>(dst + d0 + ch * 8) =
                    *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 8192, ri, d0 + ch * 8));
                *reinterpret_cast<uint4 *>(dst + 64 + d0 + ch * 8) =
                    *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 16384, ri, d0 + ch * 8));
            }
        }
        uint4 q[4];
#pragma unroll
        for (int ch = 0; ch < 4; ch++) q[ch] = *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv, ri, d0 + ch * 8));
        const int j = p < 3 ? 0 : ri % 3;
        const int nk = p < 3 ? p + 1 : j + 1;
        float sc[3];
#pragma unroll
        for (int kk = 0; kk < 3; kk++) {
            float dot = 0.f;
            if (kk < nk) {
                const bool cached = p < 3 && kk < p;
                const int krow = p < 3 ? ri : ri - j + kk;
#pragma unroll
                for (int ch = 0; ch < 4; ch++) {
                    const uint4 kv = cached ? __ldcg(reinterpret_cast<const uint4 *>(P.kv + (kvh + kk * tok_stride) * 128 + d0 + ch * 8))
                                            : *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 8192, krow, d0 + ch * 8));
                    dot += dot8(q[ch], kv);
                }
            }
            dot += __shfl_xor_sync(0xffffffff, dot, 1);
            sc[kk] = kk < nk ? dot * 0.125f : -INFINITY;
        }
        const float m = fmaxf(sc[0], fmaxf(sc[1], sc[2]));
        float w[3], wsum = 0.f;
#pragma unroll
        for (int kk = 0; kk < 3; kk++) {
            w[kk] = kk < nk ? __expf(sc[kk] - m) : 0.f;
            wsum += w[kk];
        }
        const float inv = 1.f / wsum;
#pragma unroll
        for (int ch = 0; ch < 4; ch++) {
            float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 3; kk++) {
                if (kk < nk) {
                    const bool cached = p < 3 && kk < p;
                    const int krow = p < 3 ? ri : ri - j + kk;
                    const uint4 vv = cached ? __ldcg(reinterpret_cast<const uint4 *>(P.kv + (kvh + kk * tok_stride) * 128 + 64 + d0 + ch * 8))
                                            : *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 16384, krow, d0 + ch * 8));
                    float f[8];
                    unpack8(vv, f);
                    const float wk = w[kk] * inv;
#pragma unroll
                    for (int k = 0; k < 8; k++) acc[k] += wk * f[k];
                }
            }
            o[ch] = pack8(acc);
        }
    } else {
#pragma unroll
        for (int kk = 0; kk < 3; kk++) (void)__shfl_xor_sync(0xffffffff, 0.f, 1);  // keep shuffle partners converged
#pragma unroll
        for (int ch = 0; ch < 4; ch++) o[ch] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int ch = 0; ch < 4; ch++) *reinterpret_cast<uint4 *>(st_ptr<ht_t>(ht, ri, d0 + ch * 8)) = o[ch];
}

// Write this CTA's tensor-parallel partial (bf16). DP mode: each warpgroup writes part[w]. TP mode: WG1 stages its
// accumulator in part[1], then WG0 adds its own and writes the combined partial to part[0] (when `both` worked).
__device__ __forceinline__ void store_partial(acc192_t &acc, Smem &s, int w, bool dp, bool both) {
    if (dp) {
        frag_for(acc, [&](int r, int c, float2 &v) {
            *reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(s.part[w], r, c)) = __floats2bfloat162_rn(v.x, v.y);
        });
        return;
    }
    if (both && w == 1)
        frag_for(acc, [&](int r, int c, float2 &v) {
            *reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(s.part[1], r, c)) = __floats2bfloat162_rn(v.x, v.y);
        });
    if (both) cons_sync();
    if (w == 0)
        frag_for(acc, [&](int r, int c, float2 &v) {
            bf16_2 *q = reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(s.part[0], r, c));
            float2 o = make_float2(0.f, 0.f);
            if (both) o = __bfloat1622float2(*reinterpret_cast<const bf16_2 *>(st_ptr<xa_t>(s.part[1], r, c)));
            *q = __floats2bfloat162_rn(v.x + o.x, v.y + o.y);
        });
}

template <int CL>
__device__ __forceinline__ void rollout_chunk(const Params &P, Smem &s, RingC<CL> &ring, ClusterSync<CL> &csync, int step,
                                              int p, int q, int S, const Pass &ps) {
    constexpr int HP = NH / CL, MS = FF / CL / 64, RPO = ROWS / CL;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, w = ring.w, wt = tid & 127;
    const bool act = lane < 24;
    const int col = lane * 8;
    cg::cluster_group cl = cg::this_cluster();
    const int rank = cl.block_rank();
    const bool dp = ps.dp;
    const int t = dp ? w : 0;                               // this warpgroup's row tile
    const int cbase = (dp ? q * 2 + w : q) * ps.cpc, spe = P.S[step];
    const int rr0 = dp ? (warp & 3) : warp, rrs = dp ? 4 : 8;  // owner rows: tile-local warps (DP) / all warps (TP)
    const int h0 = dp ? 0 : w, hs = dp ? 1 : NWG;              // heads / hidden slices of this warpgroup
    const unsigned long long t_roll = PNOW();

    // 1. owners: x = emb + pos ; xa = lnmod(layer 0, msa)
    for (int rr = rr0; rr < RPO; rr += rrs) {
        const int row = rank * RPO + rr;
        const RowCtx rc = row_ctx<CL>(row, p, cbase, S, spe);
        float x[8], o[8], e[8], pe[8];
        const bool live = rc.valid && act;
        if (live) {
            if (rc.tok == 0) ld8f(P.emb0 + rc.env * D + col, e);
            else {
                const float4 *src = reinterpret_cast<const float4 *>(P.emb_hist + ((size_t)rc.c * HOR + rc.tok) * D + col);
                const float4 a = __ldcg(src), b = __ldcg(src + 1);
                e[0] = a.x; e[1] = a.y; e[2] = a.z; e[3] = a.w; e[4] = b.x; e[5] = b.y; e[6] = b.z; e[7] = b.w;
            }
            ld8f(P.pos + rc.posi * D + col, pe);
#pragma unroll
            for (int k = 0; k < 8; k++) {
                x[k] = e[k] + pe[k];
                s.xown[t][rr][col + k] = x[k];
            }
        }
        if (rc.valid) lnmod(x, act, P.mod + ((size_t)(rc.c * HOR + rc.t) * NL + 0) * MODW, 0, o);
        broadcast_row<CL>(s, t, row, o, !rc.valid);
    }
    csync.sync(P, s);

    acc192_t acc;
    acc64_t a64;
    for (int l = 0; l < NL; l++) {
        // ---------------- attention half (warpgroup w owns heads w, w+2, ... of this CTA)
        frag_zero(acc);
        for (int hh = h0; hh < HP; hh += hs) {
            const int h = rank * HP + hh;
            for (int part = 0; part < 3; part++) {
                gemm_P(a64, s.xa[t], s, P, ring, OFF_QKV + (l * NH + h) * 3 + part);
                acc_to_tile(a64, s.part[w] + part * sizeof(ht_t), P.b_qkv + l * 3 * 1024 + part * 1024 + h * 64, [](float v) { return v; }, s);
            }
            wg_sync(w);
            const unsigned long long ta = PNOW();
            {
                const int ri = wt >> 1;
                attention_head(P, s, w, row_ctx<CL>(ri, p, cbase, S, spe), ri, wt & 1, p, l, h);
            }
            fence_async_smem();
            wg_sync(w);
            PADD(3, ta);
            gemm_Q(acc, s.ht[w], s, P, ring, OFF_WO + l * NH + h);
            wg_sync(w);
        }
        store_partial(acc, s, w, dp, HP > 1);
        csync.sync(P, s);
        const unsigned long long to1 = PNOW();
        for (int rr = rr0; rr < RPO; rr += rrs) {  // owners: x += gate_msa * (sum + b_o); xa = lnmod(mlp)
            const int row = rank * RPO + rr;
            const RowCtx rc = row_ctx<CL>(row, p, cbase, S, spe);
            float x[8], o[8];
            if (rc.valid) {
                const bf16 *modrow = P.mod + ((size_t)(rc.c * HOR + rc.t) * NL + l) * MODW;
                if (act) {
                    float sum[8], gate[8];
                    ld8f(P.b_o + l * D + col, sum);
                    ld8g(modrow + 2 * D + col, gate);
                    reduce_row<CL>(s, t, row, sum);
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        x[k] = s.xown[t][rr][col + k] + gate[k] * sum[k];
                        s.xown[t][rr][col + k] = x[k];
                    }
                }
                lnmod(x, act, modrow, 1, o);
            }
            broadcast_row<CL>(s, t, row, o, !rc.valid);
        }
        PADD(12, to1);
        csync.sync(P, s);

        // ---------------- MLP half (warpgroup w owns hidden slices w, w+2, ...)
        frag_zero(acc);
        for (int ss = h0; ss < MS; ss += hs) {
            const int sl = rank * MS + ss;
            gemm_P(a64, s.xa[t], s, P, ring, OFF_W1 + l * 32 + sl);
            acc_to_tile(a64, s.ht[w], P.b_1 + l * FF + sl * 64, gelu, s);
            fence_async_smem();
            wg_sync(w);
            gemm_Q(acc, s.ht[w], s, P, ring, OFF_W2 + l * 32 + sl);
            wg_sync(w);
        }
        store_partial(acc, s, w, dp, MS > 1);
        csync.sync(P, s);
        for (int rr = rr0; rr < RPO; rr += rrs) {  // owners: x += gate_mlp * (sum + b_2); next lnmod / final LN
            const int row = rank * RPO + rr;
            const RowCtx rc = row_ctx<CL>(row, p, cbase, S, spe);
            float x[8], o[8];
            if (rc.valid) {
                const bf16 *modrow = P.mod + ((size_t)(rc.c * HOR + rc.t) * NL + l) * MODW;
                if (act) {
                    float sum[8], gate[8];
                    ld8f(P.b_2 + l * D + col, sum);
                    ld8g(modrow + 5 * D + col, gate);
                    reduce_row<CL>(s, t, row, sum);
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        x[k] = s.xown[t][rr][col + k] + gate[k] * sum[k];
                        s.xown[t][rr][col + k] = x[k];
                    }
                }
                if (l + 1 < NL) {
                    lnmod(x, act, modrow + MODW, 0, o);  // next layer's msa modulation
                } else {
                    float mean, rstd, g[8], b[8];
                    ln_stats(x, act, 1e-5f, mean, rstd);
                    if (act) {
                        ld8f(P.lnf_g + col, g);
                        ld8f(P.lnf_b + col, b);
#pragma unroll
                        for (int k = 0; k < 8; k++) o[k] = (x[k] - mean) * rstd * g[k] + b[k];
                    }
                }
            }
            broadcast_row<CL>(s, t, row, o, !rc.valid);
        }
        csync.sync(P, s);
    }

    // ---------------- pred_proj: Linear -> BN (folded) -> GELU -> Linear
    frag_zero(acc);
    for (int ss = h0; ss < MS; ss += hs) {
        const int sl = rank * MS + ss;
        gemm_P(a64, s.xa[t], s, P, ring, OFF_P1 + sl);
        acc_to_tile(a64, s.ht[w], P.b_p1 + sl * 64, gelu, s);
        fence_async_smem();
        wg_sync(w);
        gemm_Q(acc, s.ht[w], s, P, ring, OFF_P2 + sl);
        wg_sync(w);
    }
    store_partial(acc, s, w, dp, MS > 1);
    csync.sync(P, s);
    for (int rr = rr0; rr < RPO; rr += rrs) {
        const int row = rank * RPO + rr;
        const RowCtx rc = row_ctx<CL>(row, p, cbase, S, spe);
        const bool last = p < 3 || (row % 3) == 2;
        if (!rc.valid || !last) continue;
        float pred[8];
        if (act) {
            ld8f(P.b_p2 + col, pred);
            reduce_row<CL>(s, t, row, pred);
        }
        if (p + 1 < HOR) {
            if (act) {
                float4 *dst = reinterpret_cast<float4 *>(P.emb_hist + ((size_t)rc.c * HOR + p + 1) * D + col);
                dst[0] = make_float4(pred[0], pred[1], pred[2], pred[3]);
                dst[1] = make_float4(pred[4], pred[5], pred[6], pred[7]);
            }
        } else {
            float e = 0.f;
            if (act) {
                float g[8];
                ld8f(P.goal + rc.env * D + col, g);
#pragma unroll
                for (int k = 0; k < 8; k++) e += (pred[k] - g[k]) * (pred[k] - g[k]);
            }
            e = warp_sum(e);
            if (lane == 0) P.cost[(size_t)(step & 1) * P.E * SMAX + rc.c] = e;
        }
    }
    PADD(6, t_roll);
}

// Phase T: per-env top-k (ascending, ties by index) and elite mean / unbiased std, one warpgroup per env; the new
// distribution and the retained-elite indices are published to global memory for the next step's sampling phase.
__device__ __forceinline__ void phase_topk(const Params &P, Smem &s, int w, int step, int env) {
    const int tid = threadIdx.x & 127, par = step & 1, S = P.S[step];
    float *cb = s.u.cbuf[w];
    int *idx = s.idx[w];
    const float *cost = P.cost + (size_t)par * P.E * SMAX + (size_t)env * S;
    for (int i = tid; i < S; i += 128) cb[i] = __ldcg(cost + i);
    wg_sync(w);
    for (int i = tid; i < S; i += 128) {
        const float ci = cb[i];
        int r = 0;
        for (int j = 0; j < S; j++) {
            const float cj = cb[j];
            r += (cj < ci) || (cj == ci && j < i);
        }
        if (r < P.topk) idx[r] = i;
    }
    wg_sync(w);
    const float *cand = P.cand + ((size_t)par * P.E * SMAX + (size_t)env * S) * NE;
    if (tid < NE) {
        float m = 0.f;
        for (int k = 0; k < P.topk; k++) m += __ldcg(cand + idx[k] * NE + tid);
        m /= P.topk;
        float v = 0.f;
        for (int k = 0; k < P.topk; k++) {
            const float d = __ldcg(cand + idx[k] * NE + tid) - m;
            v += d * d;
        }
        P.st_mean[env * NE + tid] = m;
        P.st_std[env * NE + tid] = sqrtf(v / (P.topk - 1));
    }
    if (tid == 64) {
        float c = 0.f;
        for (int k = 0; k < P.topk; k++) c += cb[idx[k]];
        P.st_cost[env] = c / P.topk;
        if (P.trace && env == 0) P.trace[step] = c / P.topk;
    }
    if (tid < 8) P.st_eidx[env * 8 + tid] = idx[min(tid, P.topk - 1)];
    wg_sync(w);
}

template <int CL> __global__ void __launch_bounds__(NTHREADS, 1) solve_kernel(const __grid_constant__ Params P) {
    extern __shared__ __align__(1024) uint8_t smem_raw[];
    Smem &s = *reinterpret_cast<Smem *>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
    const int tid = threadIdx.x;
    if (tid == 0) {
        for (int w = 0; w < NWG; w++)
            for (int i = 0; i < RS; i++) {
                init_semaphore(s.full[w][i], 1, 0);
                init_semaphore(s.empty[w][i], 4, 0);
                s.slot_tag[w][i] = -1;
            }
        init_semaphore(s.cbar[0], CL, 0);
        init_semaphore(s.cbar[1], CL, 0);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (tid < 16) s.prof[tid] = 0;
    __syncthreads();
    const unsigned long long t_kernel = PNOW();
    cg::this_cluster().sync();  // all mbarriers of the cluster initialised before any remote arrive

    RingC<CL> ring;
    ring.start(P, s, tid >> 7);
    const int worker = blockIdx.x * NWG + ring.w, nworkers = gridDim.x * NWG;
    ClusterSync<CL> csync;
    unsigned epoch = 0;
    const int cta = blockIdx.x, ncta = gridDim.x, g = cta / CL, G = gridDim.x / CL;
    if (P.mode >= 2) {  // micro-benchmarks of the synchronization / epilogue primitives (n_steps = 0)
        const int iters = P.topk;
        if (P.mode == 2)
            for (int i = 0; i < iters; i++) csync.sync(P, s);
        else if (P.mode == 3)
            for (int i = 0; i < iters; i++) grid_sync(P, s, epoch);
        else if (P.mode == 4) {
            acc64_t a64;
            frag_for(a64, [&](int r, int c, float2 &v) { v.x = r * 0.01f; v.y = c * 0.01f; });
            for (int i = 0; i < iters; i++) {
                acc_to_tile(a64, s.ht[ring.w], P.b_1, gelu, s);
                wg_sync(ring.w);
            }
        } else if (P.mode == 5) {
            acc64_t a64;
            frag_for(a64, [&](int r, int c, float2 &v) { v.x = r * 0.01f; v.y = c * 0.01f; });
            for (int i = 0; i < iters; i++) {
                acc_to_tile(a64, s.ht[ring.w], P.b_1, [](float v) { return v; }, s);
                wg_sync(ring.w);
            }
        }
        if (cta == 0 && tid == 0) P.out[0] = s.ht[0][0];
        return;
    }
    for (int step = 0; step < P.n_steps; step++) {
        const int S = P.S[step] * P.E, nrb = n_mblocks(S);  // all envs' candidates form one row space
        unsigned long long tq = PNOW();
        for (int u = worker; u < nrb; u += nworkers) phase_act(P, s, ring, step, S, u);
        PADD(7, tq);
        grid_sync(P, s, epoch);
        tq = PNOW();
        for (int u = worker; u < nrb * 12; u += nworkers) phase_mod(P, s, ring, S, u);
        PADD(8, tq);
        grid_sync(P, s, epoch);
        for (int p = 0; p < HOR; p++) {
            const Pass ps = pass_shape(S, p, G);
            for (int q = g; q < ps.nunits; q += G) rollout_chunk<CL>(P, s, ring, csync, step, p, q, S, ps);
            grid_sync(P, s, epoch);
        }
        if (P.mode == 1) {
            if (cta == 0)
                for (int i = tid; i < S; i += NCONS) P.cost_out[i] = __ldcg(P.cost + (size_t)(step & 1) * P.E * SMAX + i);
            continue;
        }
        tq = PNOW();
        for (int env = worker; env < P.E; env += nworkers) phase_topk(P, s, ring.w, step, env);
        PADD(10, tq);
        grid_sync(P, s, epoch);
    }
    if (P.prof && tid == 0) {
        s.prof[15] = PNOW() - t_kernel;
        for (int i = 0; i < 16; i++) P.prof[cta * 16 + i] = s.prof[i];
    }
}

// Pre-swizzle row-major bf16 panels into the st_bf shared-memory layout (one block per panel).
__global__ void swizzle_kernel(const bf16 *src, bf16 *dst, const int *kinds) {
    const int pid = blockIdx.x;
    const bf16 *in = src + (size_t)pid * 192 * 64;
    char *out = reinterpret_cast<char *>(dst + (size_t)pid * 192 * 64);
    const bool isP = kinds[pid] == 0;
    for (int i = threadIdx.x; i < 192 * 64; i += blockDim.x) {
        const int r = isP ? i / 64 : i / 192, c = isP ? i % 64 : i % 192;
        const uint32_t off = isP ? st_off<pP_t>(r, c) : st_off<pQ_t>(r, c);
        *reinterpret_cast<bf16 *>(out + off) = in[i];
    }
}

}  // namespace lewm

// ------------------------------------------------------------------------------------------------ host bindings
static const float *fp(const torch::Tensor &t) { return t.data_ptr<float>(); }

torch::Tensor swizzle_panels(torch::Tensor raw, torch::Tensor kinds) {
    TORCH_CHECK(raw.is_cuda() && raw.scalar_type() == torch::kBFloat16 && raw.size(1) == 192 * 64);
    auto out = torch::empty_like(raw);
    lewm::swizzle_kernel<<<raw.size(0), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16 *>(raw.data_ptr()), reinterpret_cast<bf16 *>(out.data_ptr()), kinds.data_ptr<int>());
    return out;
}

template <int CL> static int max_clusters() {
    auto kern = lewm::solve_kernel<CL>;
    const int smem = sizeof(lewm::Smem) + 1024;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (CL > 8) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(CL * 32);
    cfg.blockDim = dim3(lewm::NTHREADS);
    cfg.dynamicSmemBytes = smem;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    cudaOccupancyMaxActiveClusters(&n, kern, &cfg);
    return n;
}

int query_max_clusters(int cl) { return cl == 16 ? max_clusters<16>() : max_clusters<8>(); }

// args: dict-like positional tensors (see pack.py / runtime.py for the order).
void solve(int cl, int clusters, std::vector<torch::Tensor> w, std::vector<torch::Tensor> buf, std::vector<int64_t> S,
           int64_t mode, int64_t topk, int64_t elite_ret, double init_std, int64_t seed) {
    lewm::Params P = {};
    P.panels = reinterpret_cast<const uint8_t *>(w[0].data_ptr());
    P.b_qkv = fp(w[1]); P.b_o = fp(w[2]); P.b_1 = fp(w[3]); P.b_2 = fp(w[4]); P.b_p1 = fp(w[5]); P.b_p2 = fp(w[6]);
    P.b_e1 = fp(w[7]); P.b_e2 = fp(w[8]); P.b_ada = fp(w[9]); P.pos = fp(w[10]); P.lnf_g = fp(w[11]);
    P.lnf_b = fp(w[12]);
    // buf: emb0[E,192], goal[E,192], init_mean(unused), cand_in, cand, emb_hist, cost, out, cost_out, trace, actemb,
    //      mod, kv, counter, err, prof, st_mean[E,50], st_std[E,50], st_cost[E], st_eidx[E,8]
    P.emb0 = fp(buf[0]); P.goal = fp(buf[1]); P.init_mean = fp(buf[2]); P.cand_in = fp(buf[3]);
    P.E = (int)buf[0].size(0);
    P.st_mean = buf[16].data_ptr<float>(); P.st_std = buf[17].data_ptr<float>(); P.st_cost = buf[18].data_ptr<float>();
    P.st_eidx = buf[19].data_ptr<int>();
    P.cand = buf[4].data_ptr<float>(); P.emb_hist = buf[5].data_ptr<float>(); P.cost = buf[6].data_ptr<float>();
    P.out = buf[7].data_ptr<float>(); P.cost_out = buf[8].data_ptr<float>();
    P.trace = buf[9].numel() ? buf[9].data_ptr<float>() : nullptr;
    P.actemb = reinterpret_cast<bf16 *>(buf[10].data_ptr()); P.mod = reinterpret_cast<bf16 *>(buf[11].data_ptr());
    P.kv = reinterpret_cast<bf16 *>(buf[12].data_ptr());
    P.bar_counter = reinterpret_cast<unsigned *>(buf[13].data_ptr<int>()); P.err = buf[14].data_ptr<int>();
    P.prof = (buf.size() > 15 && buf[15].numel()) ? reinterpret_cast<unsigned long long *>(buf[15].data_ptr<int64_t>()) : nullptr;
    P.init_std = (float)init_std; P.mode = (int)mode; P.topk = (int)topk; P.elite_ret = (int)elite_ret;
    P.clusters = clusters; P.seed = (unsigned long long)seed;
    TORCH_CHECK(S.size() <= 32);
    P.n_steps = (int)S.size();
    for (size_t i = 0; i < S.size(); i++) { TORCH_CHECK(S[i] > 0 && S[i] <= lewm::SMAX); P.S[i] = (int)S[i]; }

    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(P.bar_counter, 0, sizeof(unsigned), stream);
    const int smem = sizeof(lewm::Smem) + 1024;
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cl;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(cl * clusters);
    cfg.blockDim = dim3(lewm::NTHREADS);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaError_t e;
    if (cl == 16) {
        cudaFuncSetAttribute(lewm::solve_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(lewm::solve_kernel<16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        e = cudaLaunchKernelEx(&cfg, lewm::solve_kernel<16>, P);
    } else {
        TORCH_CHECK(cl == 8);
        cudaFuncSetAttribute(lewm::solve_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        e = cudaLaunchKernelEx(&cfg, lewm::solve_kernel<8>, P);
    }
    TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
}

int smem_bytes() { return (int)sizeof(lewm::Smem) + 1024; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swizzle_panels", &swizzle_panels);
    m.def("solve", &solve);
    m.def("query_max_clusters", &query_max_clusters);
    m.def("smem_bytes", &smem_bytes);
}
