# LeWM-MK: the whole iCEM solve as one ThunderKittens megakernel

LeWM-MK keeps every planner-level LeWM-Flash change (cached observation/goal embeddings, GPU-resident iCEM loop,
bucketed 300 → 60 sample schedule, top-30 elites, 5 retained elites) and replaces the kernel- and precision-level stack
(`torch.compile` reduce-overhead, TF32, BF16 autocast) with one persistent H100 kernel. One launch runs, for every
replanning env at once, all 30 iCEM steps: in-kernel Philox sampling and elite injection, the action encoder, the
AdaLN modulation table, the 5-step autoregressive predictor rollout with `pred_proj`, the goal cost, and the per-env
top-k and elite mean/std update. The host only sees the final distribution.

Hardware scope: NVIDIA H100 80GB HBM3 (sm_90a, WGMMA, clusters, DSMEM). Not built for L40S.

## Results (H100, same container image for both stacks)

### Planner kernel latency

One env's full 30-step iCEM solve (2,820 model-cost evaluations), CUDA events, warmup excluded.

| Envs solved per call | LeWM-Flash (compile + TF32 + BF16) | LeWM-MK | Speedup |
|---|---:|---:|---:|
| 1 | 82.3 ms | **41.9 ms** (16-CTA clusters) | 2.0x |
| 26 | ~2.14 s (envs run back to back) | **180 ms** | 11.9x |
| 100 | ~8.2 s (envs run back to back) | **651 ms** | 12.6x |

For reference, eager FP32 torch needs 601 ms per single-env solve on the same GPU. The LeWM-Flash rows multiply its
measured 82.3 ms single-env solve by the env count, because the benchmark solver runs replanning envs one after
another (`batch_size=1`).

### Push-T benchmark (3 seeds x 100 episodes, all on H100, same container image)

The no-opt baseline is the stock LeWM / stable-worldmodel CEM planner (300 samples x 30 steps, eager FP32), run with
the exact flags of `benchmark_app.run_lewm_baseline`.

| Metric | No-opt baseline | LeWM-Flash | LeWM-MK | MK vs baseline |
|---|---:|---:|---:|---:|
| Push-T success | 261/300 | 261/300 | 261/300 | same |
| One env's planner solve | ~590-670 ms | 82.3 ms | **41.9 ms** | ~15x |
| Plan call 1 (all 100 envs, warmup-marked) | 60.5-66.9 s | 10.7-17.5 s | **0.77-1.86 s** | ~80x (warm seeds) |
| Plan call 26 (21-27 envs replan) | 12.5-17.1 s | 2.51-3.02 s | **0.28-0.34 s** | ~45x |
| Warmup-excluded planner time | 58.55 s | 18.63 s | 15.44 s | 3.8x |
| Eval wall time per seed | 118-132 s | 53.2-62.6 s | 49.7-50.7 s | 2.5x |

The baseline's single-env solve is its measured CEM call time per env. The planner time row is dominated by the 48
non-planning calls per seed (below), which cost about 105 ms each in both the baseline and LeWM-MK runs.

Runs: `results/runs/lewm_baseline_h100_e100_s0-1-2_20261016t191013z`,
`lewm_flash_h100_e100_s0-1-2_20261016t101037z` and `lewm_mk_h100_e100_s0-1-2_20261016t104430z`.
Two earlier megakernel builds are also saved. `...101037z` (no two-tile mode) scored 262/300 with 0.37-0.40 s on plan
26; `...102353z` (with the producer warp) scored 260/300 with 0.31-0.38 s. The success spread (260-262) is within
binomial noise (about +/-5.6 at 300 episodes).

Only 2 of the 50 plan calls in each seed run the planner: call 1 (100 envs, warmup-marked) and call 26 (26 envs).
The other 48 calls only pop already-planned actions, and each still costs about 105 ms for 100 envs in the baseline
and megakernel runs (74 ms in the LeWM-Flash run). No planner code runs in those calls. That cost sits in the policy's per-call observation
preprocessing, which no planner kernel touches. It now dominates the headline planner time (48 x ~90 ms is about
4-5 s per seed), which is why the headline moves far less than plan calls 1 and 26. It is the next thing to fix, and
it is a policy change, not a kernel change.

### Correctness

Per-candidate costs, identical candidates, relative to the FP32 eager reference (`results/megakernel/parity_final_h100.json`):

| Path | max rel. error (8 problem/S cases) | top-30 overlap with FP32 |
|---|---:|---:|
| LeWM-Flash BF16 autocast | 0.0005-0.0040 | 28-30 / 30 |
| LeWM-MK | 0.0005-0.0043 | 28-30 / 30 |
| LeWM-MK, 6 envs batched in one launch | <= 0.0029 | 29-30 / 30 |

Planner quality: 16 (observation, goal) problems, all solved in one batched megakernel launch, against the torch
iCEM solver, with each returned plan re-scored by the FP32 model. Median plan cost is 189.23 for LeWM-MK and 189.31
for torch; LeWM-MK is better or equal on 7/16. The noise streams differ (in-kernel Philox instead of torch's
generator), so plans differ draw for draw but come from the same distribution.

## Design

```text
grid = G clusters x CL CTAs, 1 CTA/SM, all co-resident (CL=8: 15 clusters, CL=16: 7 clusters on H100)
CTA  = 2 warpgroups (8 warps): WGMMA + elementwise, each with its own self-refilling 2-slot weight ring

per iCEM step:
  M1  candidates (Philox noise * std + mean, slot 0 = mean, slots 1-5 = previous elites) + action encoder
  M2  AdaLN modulation table for every (candidate, timestep, layer): bf16, computed once per step
  passes 0-4  predictor rollout (see below), costs written per candidate
  T   per-env top-k + elite mean / unbiased std, published for the next step   [grid barrier between phases]
```

- **Weights:** stored as pre-swizzled 24 KB bf16 panels in the exact ThunderKittens `st_bf` shared-tile layout, so a
  single `cp.async.bulk` lands each panel ready for WGMMA. The panel sequence is a static function of the sample
  schedule. A resumable iterator (`PanelIter`) replays it, and each warpgroup's thread 0 refills a slot the moment the
  warpgroup releases it. Prefetch therefore runs straight through op, layer, pass and barrier boundaries without a
  producer warp: a 9th warp would cap every thread at 168 registers.
- **Rollout:** each cluster runs a 64-row chunk tensor-parallel. Every CTA owns 16/CL heads and 2048/CL MLP and
  `pred_proj` hidden units. Each half-layer goes: GEMMs, then a bf16 partial in local SMEM, then a cluster barrier,
  then row owners reduce the partials over DSMEM, apply the fp32 gated residual, LayerNorm and AdaLN modulation, and
  broadcast the next GEMM operand into every CTA's SMEM, then another cluster barrier.
- **TP and DP modes:** with few rows (one env), the two warpgroups split a CTA's heads and slices on one tile. With
  many rows (batched envs), each warpgroup takes its own 64-row tile.
- **Exact KV cache:** passes 0-2 grow the history window from position 0, so the K/V of earlier tokens are reused
  exactly. That leaves 9 token-rows per candidate instead of 12.
- **Folding:** the inner LayerNorm affines, `pred_proj`'s BatchNorm and the action encoder's k=1 conv are folded into
  the adjacent GEMMs on the host.
- **Numerics:** follow the BF16 autocast path: bf16 GEMM operands with fp32 accumulation, an fp32 residual stream and
  LayerNorm statistics, bf16 AdaLN/q/k/v/attention, and fp32 costs and iCEM statistics. GELU is exact-erf through
  Abramowitz-Stegun 7.1.26 (|err| <= 1.5e-7).

## What the optimization ladder found (H100, single-env solve)

| Step | Solve | What changed |
|---|---:|---|
| first correct kernel | 76.9 ms | parity on the first run; ptxas serialized WGMMA because of a `printf` failure path |
| vectorized attention / row ops, inline failure path | 55.3 ms | 16-byte DSMEM / shared / AdaLN-table accesses, no calls between WGMMAs |
| relaxed cluster arrives + fast erf | 43.8 ms | cluster barrier 3.6 us -> 0.65 us (a `.release` on every remote arrive serialized them) |
| 2 warpgroups, batched envs, two-tile DP mode | 45-48 ms (1 env), 213 ms (26 envs) | throughput mode for real plan calls |
| no producer warp: each warpgroup refills its own ring | **41.9 ms** (1 env), **180 ms** (26), **651 ms** (100) | 9+ warps cap every thread at 168 registers; 8 warps get 255 |

Measured primitive costs (`results/megakernel/primitives_h100.json`): cluster barrier 0.65 us, grid barrier 1.6 us.
The single-env solve is a 30 x 5 x 6 = 900-layer dependent chain, with about 20 sub-microsecond dependent steps per
layer: qkv GEMMs, epilogues, attention, out-projection, 4 cluster barriers, 2 row-owner passes, MLP. That chain, not
FLOPs or weight bandwidth, is what limits batch-1 latency (the weight ring almost never waits). Getting much below
~40 ms at batch 1 would need fewer cross-CTA hops per layer: for example, resident FP8 weights with one layer per
cluster, or an all-reduce that drops the owner round trip. That is the next design step, not a tuning knob.

## Files

```text
megakernel/lewm_mk.cu   the kernel + torch bindings (ThunderKittens 67845f5f)
megakernel/pack.py      weight folding + panel packing
megakernel/runtime.py   Megakernel wrapper: costs(), solve(), solve_batch()
megakernel/solver.py    MegakernelICEMSolver (benchmark solver `icem_mk`)
megakernel/parity.py    cost parity, batched parity, planner quality, latency
megakernel/profile_mk.py, micro.py, baseline.py   latency breakdowns, primitive micro-benchmarks, torch baselines
modal/megakernel_app.py  H100 entrypoints (build on CPU, check/e2e on H100)
```

## Reproduce

```bash
uv run modal run modal/megakernel_app.py::build                       # CPU container, nvcc sm_90a
uv run modal run modal/megakernel_app.py::check --name parity --args "quality=1,problems=16"
uv run modal run modal/megakernel_app.py::check --name profile_mk --args "cls=8+16,envs=1+26+100"
uv run modal run modal/megakernel_app.py::e2e --variants flash,mk --seeds 0,1,2 --episodes 100
```

The Push-T solver config is `lewm/config/eval/solver/icem_mk.yaml` (variant and solver `icem_mk`, with
`--compile-cost false --model-cost-autocast none --matmul-precision highest`).

## Limitations

- H100 only, Hopper WGMMA path. Up to 300 samples per env and up to 100 envs per launch (by buffer sizing).
- The accepted LeWM-Flash iCEM configuration only: no colored noise, momentum, clipping or best-elite return.
- Noise comes from the kernel's Philox stream, so runs are not draw-for-draw identical to the torch solver.
- The weights must be repacked if the checkpoint changes. `pack.py` folds them from any LeWM Push-T checkpoint with
  this architecture.
