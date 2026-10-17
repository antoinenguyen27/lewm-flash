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
              OFF_E2 = 844, OFF_ADA = 856, OFF_EPATCH = 964, OFF_EQKV = 976, OFF_EOUT = 1084, OFF_EFC1 = 1120,
              OFF_EFC2 = 1264, OFF_EPROJ1 = 1408, OFF_EPROJ2 = 1440, NPANEL = 1472;
// Fused ViT-tiny encoder: 224x224 RGB, patch 14 -> 256 patches + CLS, 12 layers, width 192, 3 heads x 64, MLP 768.
constexpr int ENC_L = 12, ENC_TOK = 257, ENC_HEADS = 3, ENC_MLP = 768, IMG = 224, PATCH = 14, GRIDP = 16;
constexpr int ENC_SP = 1 + ENC_L * 5 + 1;  // sub-phases: patch | 12 x (qkv, attention, out, fc1, fc2) | projector

using xa_t = st_bf<64, 192>;
using ht_t = st_bf<64, 64>;
using pP_t = st_bf<192, 64>;
using pQ_t = st_bf<64, 192>;
using acc64_t = rt_fl<16, 64>;
using acc192_t = rt_fl<16, 192>;

struct Params {
    const uint8_t *panels;
    const float *b_qkv, *b_o, *b_1, *b_2, *b_p1, *b_p2, *b_e1, *b_e2, *b_ada, *pos, *lnf_g, *lnf_b;
    float *emb0, *goal;  // [E][192] observation / goal embeddings (written in-kernel when the encoder is fused)
    const float *init_mean, *cand_in;
    // fused encoder (n_img > 0): raw uint8 HWC images, destination per image (e*2 + 0 obs | e*2 + 1 goal)
    const uint8_t *img;
    const int *img_dst;
    int n_img;
    float *ex;                 // [n_img*257][192] fp32 residual stream
    bf16 *eqkv, *eattn, *eh;   // [rows][576], [rows][192], [rows][768]
    const float *e_patch_b, *e_pos, *e_bqkv, *e_bo, *e_b1, *e_b2, *e_pb1, *e_pb2;
    float e_eps;
    float *cand, *emb_hist, *cost, *out, *cost_out, *trace;
    unsigned long long *prof;  // optional [grid][16] per-CTA consumer time breakdown (ns)
    bf16 *actemb, *mod, *kv;
    float *actacc;  // [2 step parities][E*SMAX*5][192] fp32 action-encoder partial sums (pre-SiLU, bias included)
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
template <bool ACCUM = false, typename Ring>
__device__ __forceinline__ void gemm_P(acc64_t &acc, void *a, Smem &s, const Params &P, Ring &ring, int id) {
    auto &w = *reinterpret_cast<pP_t *>(ring.take(P, s, id));
    const unsigned long long t0 = PNOW();
    if constexpr (ACCUM) warpgroup::mma_AB(acc, *reinterpret_cast<xa_t *>(a), w);
    else warpgroup::mm_AB(acc, *reinterpret_cast<xa_t *>(a), w);
    warpgroup::mma_async_wait();
    PADD(4, t0);
    ring.give(P, s);
}
template <bool ACCUM = true, typename Ring>
__device__ __forceinline__ void gemm_Q(acc192_t &acc, void *a, Smem &s, const Params &P, Ring &ring, int id) {
    auto &w = *reinterpret_cast<pQ_t *>(ring.take(P, s, id));
    const unsigned long long t0 = PNOW();
    if constexpr (ACCUM) warpgroup::mma_AB(acc, *reinterpret_cast<ht_t *>(a), w);
    else warpgroup::mm_AB(acc, *reinterpret_cast<ht_t *>(a), w);
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

// Fused-encoder schedule, shared by the consumers and the panel iterator. Sub-phase sp: 0 = patch embed,
// 1 + 5*l + t (t: 0 qkv, 1 attention, 2 out-proj, 3 fc1, 4 fc2), ENC_SP-1 = projector. Units are spread over all
// warpgroups; a grid barrier separates sub-phases.
__device__ __forceinline__ int enc_rtiles(const Params &P) { return (P.n_img * ENC_TOK + 63) / 64; }
__device__ __forceinline__ int enc_units(const Params &P, int sp) {
    const int nrt = enc_rtiles(P);
    if (sp == 0) return nrt * 3;
    if (sp == ENC_SP - 1) return ((P.n_img + 63) / 64) * 8;
    switch ((sp - 1) % 5) {
        case 0: return nrt * 9;
        case 1: return P.n_img * ENC_HEADS * 5;
        case 2: return nrt * 3;
        case 3: return nrt * 12;
        default: return nrt * 3;
    }
}
__device__ __forceinline__ int enc_npanels(int sp) {
    if (sp == 0) return 4;
    if (sp == ENC_SP - 1) return 8;
    const int t = (sp - 1) % 5;
    return t == 1 ? 0 : t == 4 ? 4 : 1;
}
__device__ __forceinline__ int enc_panel(int sp, int u, int j) {
    if (sp == 0) return OFF_EPATCH + j * 3 + u % 3;
    if (sp == ENC_SP - 1) {
        const int sl = (u % 8) * 4 + j / 2;
        return (j & 1) ? OFF_EPROJ2 + sl : OFF_EPROJ1 + sl;
    }
    const int l = (sp - 1) / 5, t = (sp - 1) % 5;
    if (t == 0) return OFF_EQKV + l * 9 + u % 9;
    if (t == 2) return OFF_EOUT + l * 3 + u % 3;
    if (t == 3) return OFF_EFC1 + l * 12 + u % 12;
    return OFF_EFC2 + l * 12 + j * 3 + u % 3;
}

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
    int step, ph, u, j, p, q, l, k, S, nrb, h0, hs, nh, ns, nunits, sp;
    bool split;
    __device__ __forceinline__ void start_step(const Params &P) {
        ph = 0; j = 0; u = worker;
        if (step < P.n_steps) { S = P.S[step] * P.E; nrb = n_mblocks(S); }
    }
    __device__ __forceinline__ void start_pass() {
        const Pass ps = pass_shape(S, p, G);
        h0 = ps.dp ? 0 : wg; hs = ps.dp ? 1 : NWG;
        nh = h0 < HP ? (HP - h0 + hs - 1) / hs : 0;
        split = !ps.dp && HP == 1;  // WG0: q, k, out-proj; WG1: v
        ns = h0 < MS ? (MS - h0 + hs - 1) / hs : 0;
        nunits = ps.nunits; q = g; l = 0; k = 0;
    }
    __device__ __forceinline__ void init(const Params &P, int w) {
        wg = w; worker = blockIdx.x * NWG + w; nworkers = gridDim.x * NWG;
        g = blockIdx.x / CL; G = gridDim.x / CL; rank = blockIdx.x % CL;
        step = 0; start_step(P);
        sp = 0;
        if (P.n_img > 0) { ph = -1; u = worker; j = 0; }  // fused encoder first
    }
    __device__ __forceinline__ int next(const Params &P) {
        while (true) {
            if (ph == -1) {
                if (sp >= ENC_SP) { start_step(P); continue; }
                if (u < enc_units(P, sp)) {
                    if (j < enc_npanels(sp)) { const int id = enc_panel(sp, u, j); j++; return id; }
                    u += nworkers; j = 0; continue;
                }
                sp++; u = worker; j = 0; continue;
            }
            if (step >= P.n_steps) return -1;
            if (ph == 0) {
                if (u < nrb * 4) {
                    if (j < 6) {
                        const int jj = (u % 4) * 3 + j / 2, id = (j & 1) ? OFF_E2 + jj : OFF_E1 + jj;
                        j++;
                        return id;
                    }
                    u += nworkers; j = 0; continue;
                }
                ph = 1; u = worker; j = 0; continue;
            }
            if (ph == 1) {
                if (u < nrb * 18) {
                    if (j < 6) { const int id = OFF_ADA + ((u % 18) / 3) * 18 + (u % 3) * 6 + j; j++; return id; }
                    u += nworkers; j = 0; continue;
                }
                ph = 2; p = 0; start_pass(); continue;
            }
            if (p >= HOR) { step++; start_step(P); continue; }
            if (q >= nunits) { p++; if (p < HOR) start_pass(); continue; }
            if (l < NL) {
                if (split) {
                    const int natt = wg == 0 ? 3 : 1, h = rank;
                    if (k < natt) {
                        const int id = wg == 0 ? (k < 2 ? OFF_QKV + (l * NH + h) * 3 + k : OFF_WO + l * NH + h)
                                               : OFF_QKV + (l * NH + h) * 3 + 2;
                        k++;
                        return id;
                    }
                    if (k < natt + ns * 2) {
                        const int k2 = k - natt, sl = rank * MS + h0 + (k2 / 2) * hs;
                        k++;
                        return (k2 & 1) ? OFF_W2 + l * 32 + sl : OFF_W1 + l * 32 + sl;
                    }
                    l++; k = 0; continue;
                }
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
__device__ __forceinline__ void phase_act(const Params &P, Smem &s, Ring &ring, int step, int S, int rb, int hq) {
    // S here = E * samples (global candidate rows gi = env * samples + c). Unit = (row block, hidden quarter): this
    // warpgroup runs 3 of the encoder's 12 hidden slices and atomically adds its fp32 partial into actacc.
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
            if (hq == 0) cand[(size_t)gi * NE + el] = v;
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
    for (int jj = 0; jj < 3; jj++) {
        const int j = hq * 3 + jj;
        gemm_P(h, A, s, P, ring, OFF_E1 + j);
        acc_to_tile(h, HT, P.b_e1 + j * 64, silu, s);
        fence_async_smem();
        wg_sync(w);
        gemm_Q(acc, HT, s, P, ring, OFF_E2 + j);
        wg_sync(w);
    }
    float *dst = P.actacc + (size_t)par * P.E * SMAX * HOR * D;
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int c_ = c0 + r / HOR;
        if (r < MB_CANDS * HOR && c_ < S) {
            float *o = dst + (size_t)(c0 * HOR + r) * D + c;
            atomicAdd(o, v.x + (hq == 0 ? __ldg(P.b_e2 + c) : 0.f));
            atomicAdd(o + 1, v.y + (hq == 0 ? __ldg(P.b_e2 + c + 1) : 0.f));
        }
    });
}

// Phase M2: AdaLN modulation table for one (row block, layer, column half).
template <typename Ring>
__device__ __forceinline__ void phase_mod(const Params &P, Smem &s, Ring &ring, int step, int S, int u) {
    const int w = ring.w, tid = threadIdx.x & 127, rb = u / 18, l = (u % 18) / 3, third = u % 3, r0 = rb * MB_CANDS * HOR;
    const float *src = P.actacc + (size_t)(step & 1) * P.E * SMAX * HOR * D;
    const int nrows = min(MB_CANDS * HOR, S * HOR - r0);
    void *A = s.xa[w];
    for (int i = tid; i < 64 * 96; i += 128) {
        const int r = i / 96, col = (i % 96) * 2;
        bf16_2 v = __floats2bfloat162_rn(0.f, 0.f);
        if (r < nrows) {
            const float2 a = __ldcg(reinterpret_cast<const float2 *>(src + (size_t)(r0 + r) * D + col));
            v = __floats2bfloat162_rn(silu(a.x), silu(a.y));  // SiLU(act_emb) -> bf16, as the original M1 epilogue
        }
        *reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(A, r, col)) = v;
    }
    fence_async_smem();
    wg_sync(w);
    acc64_t acc;
    for (int n = 0; n < 6; n++) {
        const int col0 = (third * 6 + n) * 64;
        gemm_P(acc, A, s, P, ring, OFF_ADA + l * 18 + third * 6 + n);
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
__device__ __forceinline__ void ld8cg(const float *p, float (&f)[8]) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(p)), b = __ldcg(reinterpret_cast<const float4 *>(p) + 1);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
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
__device__ __forceinline__ void lnmod_v(const float (&x)[8], bool act, const float (&shift)[8], const float (&scale)[8],
                                        float (&o)[8]) {
    float mean, rstd;
    ln_stats(x, act, 1e-6f, mean, rstd);
    float h[8];
#pragma unroll
    for (int k = 0; k < 8; k++) h[k] = act ? (x[k] - mean) * rstd * (1.f + scale[k]) + shift[k] : 0.f;
    ln_stats(h, act, 1e-5f, mean, rstd);
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = (h[k] - mean) * rstd;
}
__device__ __forceinline__ void lnmod(const float (&x)[8], bool act, const bf16 *modrow, int which, float (&o)[8]) {
    const int lane = threadIdx.x & 31;
    float shift[8], scale[8];
    if (act) {
        ld8g(modrow + which * 3 * D + lane * 8, shift);
        ld8g(modrow + which * 3 * D + D + lane * 8, scale);
    }
    lnmod_v(x, act, shift, scale, o);
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
template <int QT>  // QT consecutive lanes share one row, each owning 64/QT head dims (16-byte vectors)
__device__ __forceinline__ void attention_head(const Params &P, uint8_t *qkv, uint8_t *ht, const RowCtx &rc, int ri,
                                               int lq, int p, int l, int h) {
    constexpr int NCH = 8 / QT;  // 8-dim chunks per thread
    const int d0 = lq * (64 / QT);
    uint4 o[NCH];
    if (rc.valid) {
        const size_t kvh = (((size_t)rc.c * 2) * NL + l) * NH + h;  // token 0 base
        const size_t tok_stride = (size_t)NL * NH;
        if (p < 2) {  // cache this token's k, v for the growing-window passes
            bf16 *dst = P.kv + (kvh + p * tok_stride) * 128;
#pragma unroll
            for (int ch = 0; ch < NCH; ch++) {
                *reinterpret_cast<uint4 *>(dst + d0 + ch * 8) =
                    *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 8192, ri, d0 + ch * 8));
                *reinterpret_cast<uint4 *>(dst + 64 + d0 + ch * 8) =
                    *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 16384, ri, d0 + ch * 8));
            }
        }
        uint4 q[NCH];
#pragma unroll
        for (int ch = 0; ch < NCH; ch++) q[ch] = *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv, ri, d0 + ch * 8));
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
                for (int ch = 0; ch < NCH; ch++) {
                    const uint4 kv = cached ? __ldcg(reinterpret_cast<const uint4 *>(P.kv + (kvh + kk * tok_stride) * 128 + d0 + ch * 8))
                                            : *reinterpret_cast<const uint4 *>(st_ptr<ht_t>(qkv + 8192, krow, d0 + ch * 8));
                    dot += dot8(q[ch], kv);
                }
            }
#pragma unroll
            for (int x = 1; x < QT; x <<= 1) dot += __shfl_xor_sync(0xffffffff, dot, x);
            sc[kk] = kk < nk ? dot * 0.125f : -INFINITY;
        }
        const float m = fmaxf(sc[0], fmaxf(sc[1], sc[2]));
        float wt[3], wsum = 0.f;
#pragma unroll
        for (int kk = 0; kk < 3; kk++) {
            wt[kk] = kk < nk ? __expf(sc[kk] - m) : 0.f;
            wsum += wt[kk];
        }
        const float inv = 1.f / wsum;
#pragma unroll
        for (int ch = 0; ch < NCH; ch++) {
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
                    const float wk = wt[kk] * inv;
#pragma unroll
                    for (int k = 0; k < 8; k++) acc[k] += wk * f[k];
                }
            }
            o[ch] = pack8(acc);
        }
    } else {
#pragma unroll
        for (int kk = 0; kk < 3; kk++)
#pragma unroll
            for (int x = 1; x < QT; x <<= 1) (void)__shfl_xor_sync(0xffffffff, 0.f, x);  // keep partners converged
#pragma unroll
        for (int ch = 0; ch < NCH; ch++) o[ch] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int ch = 0; ch < NCH; ch++) *reinterpret_cast<uint4 *>(st_ptr<ht_t>(ht, ri, d0 + ch * 8)) = o[ch];
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
            if (rc.tok == 0) ld8cg(P.emb0 + rc.env * D + col, e);
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
        if (!dp && HP == 1) {  // one head per CTA: split q,k | v GEMMs and the attention rows across both warpgroups
            const int h = rank;
            uint8_t *qkv = s.part[0];
            if (w == 0) {
                for (int part = 0; part < 2; part++) {
                    gemm_P(a64, s.xa[0], s, P, ring, OFF_QKV + (l * NH + h) * 3 + part);
                    acc_to_tile(a64, qkv + part * sizeof(ht_t), P.b_qkv + l * 3 * 1024 + part * 1024 + h * 64,
                                [](float v) { return v; }, s);
                }
            } else {
                gemm_P(a64, s.xa[0], s, P, ring, OFF_QKV + (l * NH + h) * 3 + 2);
                acc_to_tile(a64, qkv + 2 * sizeof(ht_t), P.b_qkv + l * 3 * 1024 + 2 * 1024 + h * 64,
                            [](float v) { return v; }, s);
            }
            cons_sync();
            {
                const int ri = w * 32 + (wt >> 2);
                attention_head<4>(P, qkv, s.ht[0], row_ctx<CL>(ri, p, cbase, S, spe), ri, wt & 3, p, l, h);
            }
            fence_async_smem();
            cons_sync();
            if (w == 0) {
                frag_zero(acc);
                gemm_Q(acc, s.ht[0], s, P, ring, OFF_WO + l * NH + h);
            }
            store_partial(acc, s, w, dp, false);
        } else {
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
                attention_head<2>(P, s.part[w], s.ht[w], row_ctx<CL>(ri, p, cbase, S, spe), ri, wt & 1, p, l, h);
            }
            fence_async_smem();
            wg_sync(w);
            PADD(3, ta);
            gemm_Q(acc, s.ht[w], s, P, ring, OFF_WO + l * NH + h);
            wg_sync(w);
        }
        store_partial(acc, s, w, dp, HP > 1);
        }
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
                ld8cg(P.goal + rc.env * D + col, g);
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

// ------------------------------------------------------------------------------------------------ fused encoder
// The LeWM ViT-tiny observation/goal encoder + projector, run once per plan inside the same launch as the iCEM loop.
// Every unit belongs to one warpgroup: a 64-row tile of the token-major residual stream (rows = image*257 + token)
// times one 64-wide output panel, or one (image, head, 64-query tile) attention block. bf16 operands, fp32
// accumulation and residual stream; LayerNorm affines and the projector BatchNorm are folded into the weights.

// LayerNorm (no affine) of 64 residual rows -> bf16 A operand of warpgroup w. row_of(rr) gives the global row or -1.
template <typename RowFn>
__device__ __forceinline__ void enc_ln_to_A(const Params &P, Smem &s, int w, RowFn &&row_of) {
    const int lane = threadIdx.x & 31, wq = (threadIdx.x >> 5) & 3, col = lane * 8;
    const bool act = lane < 24;
    for (int rr = wq; rr < 64; rr += 4) {
        const int g = row_of(rr);
        uint4 u = make_uint4(0, 0, 0, 0);
        if (g >= 0) {  // warp-uniform
            float x[8], o[8], mean, rstd;
            if (act) ld8cg(P.ex + (size_t)g * D + col, x);
            ln_stats(x, act, P.e_eps, mean, rstd);
#pragma unroll
            for (int k = 0; k < 8; k++) o[k] = (x[k] - mean) * rstd;
            u = pack8(o);
        }
        if (act) *reinterpret_cast<uint4 *>(st_ptr<xa_t>(s.xa[w], rr, col)) = u;
    }
    fence_async_smem();
    wg_sync(w);
}
// 64 rows x 192 bf16 columns of a row-major global matrix (row stride `stride`) -> bf16 A operand of warpgroup w.
__device__ __forceinline__ void enc_bf16_to_A(Smem &s, int w, const bf16 *src, int stride, int g0, int R) {
    const int t = threadIdx.x & 127;
    for (int i = t; i < 64 * 24; i += 128) {
        const int r = i / 24, c = (i % 24) * 8, g = g0 + r;
        uint4 u = make_uint4(0, 0, 0, 0);
        if (g < R) u = __ldcg(reinterpret_cast<const uint4 *>(src + (size_t)g * stride + c));
        *reinterpret_cast<uint4 *>(st_ptr<xa_t>(s.xa[w], r, c)) = u;
    }
    fence_async_smem();
    wg_sync(w);
}

// Patch embedding: im2col straight from raw uint8 HWC pixels with ImageNet normalization (ToDtype(scale) ->
// Normalize; the 224 -> 224 resize is the identity), K = 3*14*14 = 588 padded to 4 x 192; adds bias + position.
template <typename Ring>
__device__ __forceinline__ void enc_patch(const Params &P, Smem &s, Ring &ring, int u) {
    const int w = ring.w, t = threadIdx.x & 127, np = u % 3, g0 = (u / 3) * 64, R = P.n_img * ENC_TOK;
    auto fill = [&](int kc) {
        for (int i = t; i < 64 * 96; i += 128) {
            const int r = i / 96, kk = (i % 96) * 2, g = g0 + r;
            float v[2] = {0.f, 0.f};
            if (g < R && g % ENC_TOK > 0) {
                const int im = g / ENC_TOK, pidx = g % ENC_TOK - 1, py = pidx / GRIDP, px = pidx % GRIDP;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int k = kc * 192 + kk + e;
                    if (k < 3 * PATCH * PATCH) {
                        const int c = k / (PATCH * PATCH), rem = k % (PATCH * PATCH), ky = rem / PATCH, kx = rem % PATCH;
                        const float mean = c == 0 ? 0.485f : c == 1 ? 0.456f : 0.406f;
                        const float istd = c == 0 ? 1.f / 0.229f : c == 1 ? 1.f / 0.224f : 1.f / 0.225f;
                        const uint8_t q = __ldg(P.img + (((size_t)im * IMG + py * PATCH + ky) * IMG + px * PATCH + kx) * 3 + c);
                        v[e] = (q * (1.f / 255.f) - mean) * istd;
                    }
                }
            }
            *reinterpret_cast<bf16_2 *>(st_ptr<xa_t>(s.xa[w], r, kk)) = __floats2bfloat162_rn(v[0], v[1]);
        }
        fence_async_smem();
        wg_sync(w);
    };
    acc64_t acc;
    fill(0);
    gemm_P<false>(acc, s.xa[w], s, P, ring, enc_panel(0, u, 0));
    for (int kc = 1; kc < 4; kc++) {
        fill(kc);
        gemm_P<true>(acc, s.xa[w], s, P, ring, enc_panel(0, u, kc));
    }
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int g = g0 + r, col = np * 64 + c;
        if (g >= R) return;
        const int tok = g % ENC_TOK;
        float2 o = make_float2(__ldg(P.e_pos + tok * D + col), __ldg(P.e_pos + tok * D + col + 1));  // CLS: cls + pos0
        if (tok > 0) {
            o.x += v.x + __ldg(P.e_patch_b + col);
            o.y += v.y + __ldg(P.e_patch_b + col + 1);
        }
        *reinterpret_cast<float2 *>(P.ex + (size_t)g * D + col) = o;
    });
}

template <typename Ring>
__device__ __forceinline__ void enc_qkv(const Params &P, Smem &s, Ring &ring, int l, int u) {
    const int w = ring.w, np = u % 9, g0 = (u / 9) * 64, R = P.n_img * ENC_TOK;
    enc_ln_to_A(P, s, w, [&](int rr) { const int g = g0 + rr; return g < R ? g : -1; });
    acc64_t acc;
    gemm_P(acc, s.xa[w], s, P, ring, enc_panel(1 + 5 * l, u, 0));
    const float *b = P.e_bqkv + l * 576 + np * 64;
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int g = g0 + r;
        if (g < R)
            *reinterpret_cast<bf16_2 *>(P.eqkv + (size_t)g * 576 + np * 64 + c) =
                __floats2bfloat162_rn(v.x + __ldg(b + c), v.y + __ldg(b + c + 1));
    });
}

// One (image, head, 64-query tile): flash attention over 5 key tiles of 64 (257 keys), online softmax in fp32.
__device__ __forceinline__ void enc_attn(const Params &P, Smem &s, int w, int u) {
    const int t = threadIdx.x & 127, im = u / (ENC_HEADS * 5), head = (u % (ENC_HEADS * 5)) / 5, qt = u % 5;
    const size_t base = (size_t)im * ENC_TOK;
    uint8_t *Qs = s.part[w], *Ks = s.part[w] + sizeof(ht_t), *Vs = s.part[w] + 2 * sizeof(ht_t), *Ps = s.ht[w];
    auto load_tile = [&](uint8_t *dst, int tok0, int colbase) {
        for (int i = t; i < 64 * 8; i += 128) {
            const int r = i / 8, c = (i % 8) * 8, tok = tok0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (tok < ENC_TOK) v = __ldcg(reinterpret_cast<const uint4 *>(P.eqkv + (base + tok) * 576 + colbase + c));
            *reinterpret_cast<uint4 *>(st_ptr<ht_t>(dst, r, c)) = v;
        }
    };
    load_tile(Qs, qt * 64, head * 64);
    acc64_t o, sc;
    frag_zero(o);
    float m[2] = {-INFINITY, -INFINITY}, lsum[2] = {0.f, 0.f};
    for (int kt = 0; kt < 5; kt++) {
        load_tile(Ks, kt * 64, 192 + head * 64);
        load_tile(Vs, kt * 64, 384 + head * 64);
        fence_async_smem();
        wg_sync(w);
        warpgroup::mm_ABt(sc, *reinterpret_cast<ht_t *>(Qs), *reinterpret_cast<ht_t *>(Ks));
        warpgroup::mma_async_wait();
        float mx[2] = {-INFINITY, -INFINITY};
        frag_for(sc, [&](int r, int c, float2 &v) {
            const int key = kt * 64 + c, ri = (r & 15) >> 3;
            v.x = key < ENC_TOK ? v.x * 0.125f : -INFINITY;
            v.y = key + 1 < ENC_TOK ? v.y * 0.125f : -INFINITY;
            mx[ri] = fmaxf(mx[ri], fmaxf(v.x, v.y));
        });
        float alpha[2], rs[2] = {0.f, 0.f};
#pragma unroll
        for (int ri = 0; ri < 2; ri++) {
            mx[ri] = fmaxf(mx[ri], __shfl_xor_sync(0xffffffff, mx[ri], 1));
            mx[ri] = fmaxf(mx[ri], __shfl_xor_sync(0xffffffff, mx[ri], 2));
            const float mn = fmaxf(m[ri], mx[ri]);
            alpha[ri] = __expf(m[ri] - mn);
            m[ri] = mn;
        }
        frag_for(sc, [&](int r, int c, float2 &v) {
            const int ri = (r & 15) >> 3;
            const float a = __expf(v.x - m[ri]), b = __expf(v.y - m[ri]);
            rs[ri] += a + b;
            *reinterpret_cast<bf16_2 *>(st_ptr<ht_t>(Ps, r, c)) = __floats2bfloat162_rn(a, b);
        });
#pragma unroll
        for (int ri = 0; ri < 2; ri++) {
            rs[ri] += __shfl_xor_sync(0xffffffff, rs[ri], 1);
            rs[ri] += __shfl_xor_sync(0xffffffff, rs[ri], 2);
            lsum[ri] = lsum[ri] * alpha[ri] + rs[ri];
        }
        frag_for(o, [&](int r, int c, float2 &v) {
            const int ri = (r & 15) >> 3;
            v.x *= alpha[ri];
            v.y *= alpha[ri];
        });
        fence_async_smem();
        wg_sync(w);
        warpgroup::mma_AB(o, *reinterpret_cast<ht_t *>(Ps), *reinterpret_cast<ht_t *>(Vs));
        warpgroup::mma_async_wait();
        wg_sync(w);  // K / V / P tiles are rewritten next iteration
    }
    frag_for(o, [&](int r, int c, float2 &v) {
        const int tok = qt * 64 + r, ri = (r & 15) >> 3;
        if (tok < ENC_TOK) {
            const float inv = 1.f / lsum[ri];
            *reinterpret_cast<bf16_2 *>(P.eattn + (base + tok) * D + head * 64 + c) =
                __floats2bfloat162_rn(v.x * inv, v.y * inv);
        }
    });
}

// residual += acc + bias for a 64x64 output panel of a 192-wide residual stream
__device__ __forceinline__ void enc_residual(const Params &P, acc64_t &acc, const float *b, int g0, int np, int R) {
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int g = g0 + r;
        if (g >= R) return;
        float2 *x = reinterpret_cast<float2 *>(P.ex + (size_t)g * D + np * 64 + c);
        float2 o = __ldcg(x);
        o.x += v.x + __ldg(b + c);
        o.y += v.y + __ldg(b + c + 1);
        *x = o;
    });
}

template <typename Ring>
__device__ __forceinline__ void enc_out(const Params &P, Smem &s, Ring &ring, int l, int u) {
    const int w = ring.w, np = u % 3, g0 = (u / 3) * 64, R = P.n_img * ENC_TOK;
    enc_bf16_to_A(s, w, P.eattn, D, g0, R);
    acc64_t acc;
    gemm_P(acc, s.xa[w], s, P, ring, enc_panel(1 + 5 * l + 2, u, 0));
    enc_residual(P, acc, P.e_bo + l * D + np * 64, g0, np, R);
}

template <typename Ring>
__device__ __forceinline__ void enc_fc1(const Params &P, Smem &s, Ring &ring, int l, int u) {
    const int w = ring.w, np = u % 12, g0 = (u / 12) * 64, R = P.n_img * ENC_TOK;
    enc_ln_to_A(P, s, w, [&](int rr) { const int g = g0 + rr; return g < R ? g : -1; });
    acc64_t acc;
    gemm_P(acc, s.xa[w], s, P, ring, enc_panel(1 + 5 * l + 3, u, 0));
    const float *b = P.e_b1 + l * ENC_MLP + np * 64;
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int g = g0 + r;
        if (g < R)
            *reinterpret_cast<bf16_2 *>(P.eh + (size_t)g * ENC_MLP + np * 64 + c) =
                __floats2bfloat162_rn(gelu(v.x + __ldg(b + c)), gelu(v.y + __ldg(b + c + 1)));
    });
}

template <typename Ring>
__device__ __forceinline__ void enc_fc2(const Params &P, Smem &s, Ring &ring, int l, int u) {
    const int w = ring.w, np = u % 3, g0 = (u / 3) * 64, R = P.n_img * ENC_TOK, sp = 1 + 5 * l + 4;
    acc64_t acc;
    enc_bf16_to_A(s, w, P.eh, ENC_MLP, g0, R);
    gemm_P<false>(acc, s.xa[w], s, P, ring, enc_panel(sp, u, 0));
    for (int kc = 1; kc < 4; kc++) {
        enc_bf16_to_A(s, w, P.eh + kc * 192, ENC_MLP, g0, R);
        gemm_P<true>(acc, s.xa[w], s, P, ring, enc_panel(sp, u, kc));
    }
    enc_residual(P, acc, P.e_b2 + l * D + np * 64, g0, np, R);
}

// Projector on the CLS rows (final LN and BatchNorm folded): 4 of its 32 hidden slices per unit, fp32 partial sums
// atomically accumulated into the (host-zeroed) observation / goal embedding.
template <typename Ring>
__device__ __forceinline__ void enc_proj(const Params &P, Smem &s, Ring &ring, int u) {
    const int w = ring.w, s8 = u % 8, i0 = (u / 8) * 64;
    enc_ln_to_A(P, s, w, [&](int rr) { const int im = i0 + rr; return im < P.n_img ? im * ENC_TOK : -1; });
    acc192_t acc;
    acc64_t a64;
    frag_zero(acc);
    for (int s4 = 0; s4 < 4; s4++) {
        gemm_P(a64, s.xa[w], s, P, ring, enc_panel(ENC_SP - 1, u, 2 * s4));
        acc_to_tile(a64, s.ht[w], P.e_pb1 + (s8 * 4 + s4) * 64, gelu, s);
        fence_async_smem();
        wg_sync(w);
        gemm_Q(acc, s.ht[w], s, P, ring, enc_panel(ENC_SP - 1, u, 2 * s4 + 1));
        wg_sync(w);
    }
    frag_for(acc, [&](int r, int c, float2 &v) {
        const int im = i0 + r;
        if (im >= P.n_img) return;
        const int dst = P.img_dst[im];
        float *out = ((dst & 1) ? P.goal : P.emb0) + (dst >> 1) * D;
        float a = v.x, b = v.y;
        if (s8 == 0) {
            a += __ldg(P.e_pb2 + c);
            b += __ldg(P.e_pb2 + c + 1);
        }
        atomicAdd(out + c, a);
        atomicAdd(out + c + 1, b);
    });
}

template <int CL>
__device__ __forceinline__ void run_encoder(const Params &P, Smem &s, RingC<CL> &ring, unsigned &epoch) {
    const int worker = blockIdx.x * NWG + ring.w, nworkers = gridDim.x * NWG;
    for (int sp = 0; sp < ENC_SP; sp++) {
        const int nu = enc_units(P, sp);
        for (int u = worker; u < nu; u += nworkers) {
            if (sp == 0) enc_patch(P, s, ring, u);
            else if (sp == ENC_SP - 1) enc_proj(P, s, ring, u);
            else {
                const int l = (sp - 1) / 5, t = (sp - 1) % 5;
                if (t == 0) enc_qkv(P, s, ring, l, u);
                else if (t == 1) enc_attn(P, s, ring.w, u);
                else if (t == 2) enc_out(P, s, ring, l, u);
                else if (t == 3) enc_fc1(P, s, ring, l, u);
                else enc_fc2(P, s, ring, l, u);
            }
        }
        grid_sync(P, s, epoch);
    }
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
    if (P.n_img > 0) {
        const unsigned long long te = PNOW();
        run_encoder<CL>(P, s, ring, epoch);  // observation (and new goal) embeddings, then straight into iCEM
        PADD(13, te);
    }
    for (int step = 0; step < P.n_steps; step++) {
        const int S = P.S[step] * P.E, nrb = n_mblocks(S);  // all envs' candidates form one row space
        unsigned long long tq = PNOW();
        for (int u = worker; u < nrb * 4; u += nworkers) phase_act(P, s, ring, step, S, u / 4, u % 4);
        PADD(7, tq);
        grid_sync(P, s, epoch);
        tq = PNOW();
        for (int u = worker; u < nrb * 18; u += nworkers) phase_mod(P, s, ring, step, S, u);
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
        if (step + 1 < P.n_steps) {
            float4 *z = reinterpret_cast<float4 *>(P.actacc + (size_t)((step + 1) & 1) * P.E * SMAX * HOR * D);
            const size_t n4 = (size_t)P.S[step + 1] * P.E * HOR * D / 4;
            for (size_t i = (size_t)blockIdx.x * NCONS + tid; i < n4; i += (size_t)gridDim.x * NCONS)
                z[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
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
    P.emb0 = buf[0].data_ptr<float>(); P.goal = buf[1].data_ptr<float>(); P.init_mean = fp(buf[2]); P.cand_in = fp(buf[3]);
    P.E = (int)buf[0].size(0);
    P.st_mean = buf[16].data_ptr<float>(); P.st_std = buf[17].data_ptr<float>(); P.st_cost = buf[18].data_ptr<float>();
    P.st_eidx = buf[19].data_ptr<int>();
    // fused encoder: buf 20 images uint8 [n_img,224,224,3], 21 destinations int32 [n_img], 22 ex, 23 eqkv, 24 eattn, 25 eh;
    // w 13..20: patch bias, pos (CLS row = cls + pos0), qkv / out / fc1 / fc2 biases, projector biases
    TORCH_CHECK(buf.size() > 20, "buf[20] must be the action-encoder partial-sum buffer");
    P.actacc = buf[20].data_ptr<float>();
    P.n_img = buf.size() > 22 ? (int)buf[21].size(0) : 0;
    if (P.n_img > 0) {
        TORCH_CHECK(w.size() >= 21 && buf.size() >= 27, "fused encoder needs encoder weights and buffers");
        P.img = buf[21].data_ptr<uint8_t>(); P.img_dst = buf[22].data_ptr<int>();
        P.ex = buf[23].data_ptr<float>(); P.eqkv = reinterpret_cast<bf16 *>(buf[24].data_ptr());
        P.eattn = reinterpret_cast<bf16 *>(buf[25].data_ptr()); P.eh = reinterpret_cast<bf16 *>(buf[26].data_ptr());
        P.e_patch_b = fp(w[13]); P.e_pos = fp(w[14]); P.e_bqkv = fp(w[15]); P.e_bo = fp(w[16]); P.e_b1 = fp(w[17]);
        P.e_b2 = fp(w[18]); P.e_pb1 = fp(w[19]); P.e_pb2 = fp(w[20]);
        P.e_eps = 1e-12f;
    }
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
