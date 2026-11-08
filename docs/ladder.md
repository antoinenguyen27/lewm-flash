# LeWM on H100: the optimization ladder

This is one-robot (batch-1) inference on one GPU. It starts from the reference LeWM implementation, and each rung
adds exactly one change. Every number below was measured on an **NVIDIA H100 80GB HBM3** (Modal), with all rungs in
the same container image (torch 2.11.0+cu128, CUDA 12.8).

## Metrics

- **Plan latency (primary).** The end-to-end time of a replanning `policy.get_action()` call for one env, measured
  with CUDA events. It covers observation preprocessing, the encoder, the planner solve, and the action back on the
  host. Each rung runs 61 one-env episodes (seeds 1000-1060, the same episodes for every rung); the first seed is
  dropped as warmup, leaving 72-76 plans per rung. Calls that don't replan are reported separately as *step* latency.
- **Accuracy.** Push-T success over 3 seeds x 100 episodes, with a 95% Wilson interval. These episodes run
  batched, which is only an evaluation convenience; each env's planner math is identical either way. Numerically
  changed rungs are also checked for cost parity against FP32.

## Results

| Rung | Change (adds to the rung above) | Kind | Plan p50 | p90 | Step gain | Total vs R0 | Success (95% CI) |
|---|---|---|---:|---:|---:|---:|---:|
| R0 | Reference LeWM: stock CEM 300x30, eager FP32, re-encodes every cost call | — | 622.4 ms | 674.2 | — | 1.0x | 261/300 (82.7-90.3%) |
| R1 | Cached observation/goal embeddings, no CPU syncs in the loop | exact | 377.0 ms | 401.3 | 1.65x | 1.65x | 263/300 (83.5-90.9%) |
| R2 | iCEM (bucketed 300 -> 60 samples, 5 retained elites), still eager FP32 | planner | 410.2 ms | 433.3 | 0.92x | 1.5x | 265/300 (84.2-91.5%) |
| R3 | Megakernel: predictor + whole iCEM loop in one launch | numeric (BF16 class) | 59.3 ms | 61.3 | 6.9x | 10.5x | 262/300 (83.1-90.6%) |
| **R4** | **Fused ViT encoder in the same launch, goal embedding cached per episode, GPU image normalization** | numeric (BF16 class) | **46.5 ms** | **47.0** | 1.28x | **13.4x** | 260/300 (82.4-90.1%) |

Side rungs:

| Rung | What | Plan p50 | p90 | Success |
|---|---|---:|---:|---:|
| R2b | LeWM-Flash: R2 + `torch.compile` + TF32 + BF16 autocast (the best torch path) | 182.1 ms | 204.3 | 261/300 |
| R3-CEM | The megakernel running plain CEM (300 x 30, no retained elites) | 82.1 ms | 83.7 | 261/300 |

Step (non-planning call) p50: about 2 ms for R0-R3, 1.5 ms for R2b, and **0.82 ms for R4**, which skips the per-call
CPU image transform.

All rungs have overlapping success intervals (260-265/300); no change lowered task success beyond noise.

## What each gain is

- **Caching (R1, 1.65x, exact).** The reference re-encodes the observation and goal images on every one of the 30
  cost calls. Encoding them once per plan, and dropping the loop's GPU-to-CPU syncs, removes most of that work.
- **iCEM (R2).** It cuts model evaluations 3.19x (9,000 -> 2,820 per plan), but in eager torch the plan *slows*
  (0.92x): every cost call is launch-bound, so it costs about the same at 60 samples as at 300. iCEM only pays once
  the execution stops being launch-bound. Inside the megakernel it is worth **1.38x** (R3-CEM 82.1 ms -> R3 59.3 ms),
  with no loss in success.
- **Megakernel (R3, 6.9x over eager iCEM; 3.1x over LeWM-Flash).** One persistent kernel launch runs all 30 iCEM
  steps: in-kernel sampling, action encoder, AdaLN modulation, the 5-step predictor rollout with an exact KV cache,
  cost, top-k and the elite update. There is no host round trip and no per-op launch. The best torch path (R2b,
  182 ms) still pays per-step Python, launch and host overhead that one launch removes.
- **Fused encoder (R4, 1.28x).** The ViT-tiny encoder and projector run as the kernel's first phase on raw uint8
  frames, and the goal embedding is reused within an episode. Encoder time falls from 6.9 ms (torch FP32, obs + goal)
  to 1.25-1.34 ms (`results/megakernel` encoder parity run). A plan is now one launch end to end.

Where R4's 46.5 ms goes: about 41.3 ms is the megakernel's predictor/iCEM solve, about 1.3 ms the fused encoder, and
about 3-4 ms host-side policy glue.

## Precision

| Part | R0 / R1 / R2 | R2b (LeWM-Flash) | R3 / R4 (megakernel) |
|---|---|---|---|
| Predictor / action encoder / `pred_proj` matmuls | FP32 | BF16 in, FP32 accumulate | BF16 in, FP32 accumulate |
| Encoder (ViT + projector) | FP32 | TF32 | FP32 (R3) / BF16 in, FP32 accumulate (R4) |
| Residual stream, LayerNorm stats, softmax | FP32 | FP32 | FP32 |
| Costs, top-k, iCEM statistics | FP32 | FP32 | FP32 |

The megakernel is mathematically correct, not bit-exact. Its per-candidate cost error against FP32 (max relative
0.0005-0.0043, top-30 overlap 27-30/30) matches Flash's BF16 path. The fused encoder's embeddings deviate from FP32
by 0.41% (relative L2), against 0.52% for torch BF16 autocast. Its noise comes from an in-kernel Philox stream, so
plans differ draw for draw but follow the same distribution.

## Run-to-run variance

The ladder was measured twice: Phase A on the R3 kernel, and the final pass above. The eager-torch rungs vary with
the container (R0 526.9 -> 622.4 ms, R1 391.4 -> 377.0 ms, R2 410.3 -> 410.2 ms, R2b 196.3 -> 182.1 ms), because
they are bound by CPU launch overhead. The megakernel rungs are stable (R3 58.0 -> 59.3 ms, R3-CEM 81.7 -> 82.1 ms).
Both passes are saved: `results/ladder/ladder_h100_phaseA.json` and `results/ladder/ladder_h100.json`.

## Megakernel tuning notes (single-env solve)

| Change | Solve |
|---|---:|
| First correct kernel (WGMMA serialized by a `printf` failure path) | 76.9 ms |
| 16-byte vector DSMEM / shared / AdaLN-table accesses, no calls between WGMMAs | 55.3 ms |
| Relaxed remote cluster arrives (3.6 -> 0.65 us barrier) + fast exact-erf GELU | 43.8 ms |
| No producer warp (8 warps keep the 255-register budget; a 9th caps all threads at 168) | 41.9 ms |
| With fused encoder support, 16-CTA warpgroup split, parallel action-encoder / AdaLN phases | 41.3 ms |

Tried and reverted, all because register pressure rose and spills grew: peeling the first accumulate of each GEMM
chain, prefetching cached K/V together, and prefetching row-owner operands across the cluster barrier. The remaining
solve is a chain of about 900 dependent layer evaluations (30 steps x 5 passes x 6 layers), each about 20
sub-microsecond steps across 8-16 SMs. Going much below ~40 ms needs fewer cross-SM hand-offs per layer (for
example, resident FP8 weights with one layer per cluster, or a single-barrier all-reduce), which is a redesign.

## Reproduce

```bash
uv run modal run modal/megakernel_app.py::build
uv run modal run modal/megakernel_app.py::ladder --rungs r0,r1,r2,r2b,r3,r3cem,r4 --mode batch1 --n-seeds 61
uv run modal run modal/megakernel_app.py::ladder --rungs r0,r1,r2,r2b,r3,r3cem,r4 --mode quality
uv run python tools/benchmarks/summarize_ladder.py --batch1 r0=results/runs/<run>/metrics.jsonl ... --quality ...
```

The batch-1 sweep is expensive: each of its 61 seeds rebuilds the world, the model and (for R2b) the compile
warmup. One full pass cost about $10-12 of H100 time.

## Runs

- Batch-1 (final): `results/runs/ladder_<rung>_batch1_h100_20261017t092213z`
- Batch-1 (Phase A): `results/runs/ladder_<rung>_batch1_h100_20261017t075029z`
- Quality: R0 `lewm_baseline_h100_e100_s0-1-2_20261016t191013z`; R1 and R2 `ladder_r1|r2_quality_h100_20261017t075029z`;
  R2b `lewm_flash_h100_e100_s0-1-2_20261016t101037z`; R3, R3-CEM and R4 `ladder_r3|r3cem|r4_quality_h100_20261017t092213z`
