# Appendix A. Per-rung results

%APPENDIX A
%TABLE: Batch-1 end-to-end plan latency, final pass (ms). Plans from seeds 1001 to 1060 (seed 1000 dropped as warmup); step calls are the non-replanning calls of the same episodes.
| rung | plans | p50 | 95% CI of p50 | p90 | mean | min | max | step calls | step p50 |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 73 | 622.4 | 605.5 – 626.7 | 673.8 | 622.7 | 555.5 | 758.6 | 1,569 | 2.14 |
| R1 | 73 | 377.0 | 375.8 – 381.0 | 401.0 | 382.2 | 368.6 | 425.7 | 1,598 | 2.02 |
| R2 | 72 | 410.2 | 406.2 – 416.3 | 433.3 | 413.5 | 385.6 | 492.5 | 1,569 | 2.11 |
| R2b | 72 | 182.1 | 177.3 – 185.7 | 204.3 | 177.2 | 112.4 | 259.1 | 1,575 | 1.51 |
| R3 | 75 | 59.3 | 59.1 – 59.4 | 61.1 | 59.0 | 54.3 | 69.7 | 1,671 | 2.35 |
| R3-CEM | 75 | 82.1 | 81.7 – 82.5 | 83.6 | 82.1 | 78.2 | 86.1 | 1,642 | 1.59 |
| R4 | 76 | 46.5 | 46.4 – 46.6 | 47.0 | 51.8 | 45.1 | 457.8 | 1,641 | 0.82 |

%TABLE: Push-T successes per evaluation seed (100 episodes each) and the run that produced them.
| rung | seed 0 | seed 1 | seed 2 | total | run |
|---|---|---|---|---|---|
| R0 | 84 | 89 | 88 | 261 | `lewm_baseline_h100_e100_s0-1-2_20261016t191013z` |
| R1 | 86 | 89 | 88 | 263 | `ladder_r1_quality_h100_20261017t075029z` |
| R2 | 87 | 89 | 89 | 265 | `ladder_r2_quality_h100_20261017t075029z` |
| R2b | 86 | 84 | 91 | 261 | `lewm_flash_h100_e100_s0-1-2_20261016t101037z` |
| R3 | 90 | 86 | 86 | 262 | `ladder_r3_quality_h100_20261017t092213z` |
| R3-CEM | 83 | 86 | 92 | 261 | `ladder_r3cem_quality_h100_20261017t092213z` |
| R4 | 86 | 84 | 90 | 260 | `ladder_r4_quality_h100_20261017t092213z` |

The per-seed totals vary by up to 9 episodes within a rung (R3-CEM: 83 to 92), more than any rung's total differs from the reference's. This is the seed-to-seed variation of the episode sets, and it is why the comparisons in Section 3.6 are paired by episode rather than made on totals.

# Appendix B. Configuration

%APPENDIX B
%TABLE: Planner configuration. R0, R1 and R3-CEM use the CEM column; R2, R2b, R3 and R4 use the iCEM column.
| setting | CEM | bucketed iCEM |
|---|---|---|
| candidates per iteration | 300 | max(⌈300 · 1.25^−i^⌉, 60), rounded up to a multiple of 60: 300, 240, 240, 180, 180, 120, 120, 120, then 60 for 22 iterations |
| iterations | 30 | 30 |
| elites | top 30 | top 30 |
| retained elites | 0 | 5 per iteration |
| model evaluations per plan | 9,000 | 2,820 |
| horizon / receding horizon / action block | 5 / 5 / 5 | 5 / 5 / 5 |
| initial variance scale | 1.0 | 1.0 |
| coloured noise, momentum, clipping, best-elite return | — | none (rejected on Push-T in the LeWM-Flash study) |
| plan output | mean of the final distribution | mean of the final distribution |

%TABLE: Execution settings per rung, as passed to the benchmark harness (`modal/megakernel_app.py`, `LADDER`).
| rung | solver | matmul precision | rollout autocast | compile | other |
|---|---|---|---|---|---|
| R0 | stable-worldmodel `CEMSolver` | highest (FP32) | none | no | as released |
| R1 | `CEMExactSolver` | highest | none | no | cached embeddings, no host sync |
| R2 | `ICEMSolver` | highest | none | no | — |
| R2b | `ICEMSolver` | high (TF32) | BF16 | `torch.compile`, reduce-overhead, warmed at all five population sizes | — |
| R3 | `MegakernelICEMSolver` | highest (encoder in FP32) | none (kernel numerics) | no | 16-CTA clusters, 7 clusters, single-environment launch |
| R4 | `MegakernelICEMSolver`, fused encoder | — | — | no | as R3; uint8 frames to the kernel, goal embedding cached by content hash |
| R3-CEM | `MegakernelICEMSolver` | highest | none | no | CEM schedule, no retained elites |

%TABLE: Megakernel configuration on the H100.
| item | value |
|---|---|
| grid | 7 clusters × 16 CTAs (batch 1) or 15 clusters × 8 CTAs (batched evaluation); one CTA per SM, all co-resident |
| CTA | 256 threads: two warpgroups, each with its own two-slot weight ring; 255 registers per thread |
| shared memory | 231,584 bytes per CTA (the H100 maximum) |
| weight panels | 1,472 × 24 KB BF16 (192 × 64 or 64 × 192), pre-swizzled to the ThunderKittens shared-tile layout; 964 for the predictor, action encoder and projector, 508 for the ViT encoder and its projector |
| row tile | 64 token rows per cluster chunk; two-tile data-parallel mode for batched launches |
| random numbers | Philox 4×32-10 (cuRAND), keyed by plan seed, iteration, environment, candidate and action element |
| limits | up to 300 candidates per environment and 100 environments per launch (buffer sizing) |
| primitives (measured) | cluster barrier 0.65 µs (16-CTA: 0.68 µs); grid barrier 1.6 µs; 64 × 64 epilogue 0.15 µs; empty launch 14 µs |

# Appendix C. Repeatability and the batched regime

%APPENDIX C
%TABLE: The ladder measured twice. Phase A ran on the R3 kernel build before the fused encoder existed; the final pass ran all seven rungs. Medians and p90 in ms.
| rung | Phase A p50 | Phase A p90 | final p50 | final p90 | change in p50 |
|---|---|---|---|---|---|
| R0 | 526.9 | 546.8 | 622.4 | 673.8 | +18.1% |
| R1 | 391.4 | 433.5 | 377.0 | 401.0 | −3.7% |
| R2 | 410.3 | 446.2 | 410.2 | 433.3 | 0.0% |
| R2b | 196.3 | 236.0 | 182.1 | 204.3 | −7.2% |
| R3 | 58.0 | 59.3 | 59.3 | 61.1 | +2.2% |
| R3-CEM | 81.7 | 83.4 | 82.1 | 83.6 | +0.5% |

Taking the Phase A medians instead of the final ones would change the headline ratios but not the conclusions: the reference would be 11.3 times slower than R4 rather than 13.4, caching would be worth 1.35 times rather than 1.65, and iCEM in eager PyTorch would still be slower than CEM.

The quality evaluations ran 100 environments per process, and the megakernel solves every replanning environment of a call in one launch. That regime is not the subject of this paper, but it is where the megakernel's throughput shows, and Table C2 records it for completeness.

%TABLE: The batched evaluation regime (H100, 100-environment Push-T evaluation, 15 × 8-CTA clusters). LeWM-Flash solves replanning environments one after another; its rows for 26 and 100 environments are its single-environment solve times the count.
| quantity | reference (CEM, FP32) | LeWM-Flash | megakernel (R3 build) |
|---|---|---|---|
| solve, 1 environment | about 590–670 ms | 82.3 ms | 44.1 ms |
| solve, 26 environments in one call | — | about 2.14 s | 180 ms |
| solve, 100 environments in one call | — | about 8.2 s | 651 ms |
| plan call 26 in the evaluation (21–27 environments) | 12.5 – 17.1 s | 2.51 – 3.02 s | 0.28 – 0.34 s |
| plan call 1 in the evaluation (100 environments, includes warmup) | 60.5 – 66.9 s | 10.7 – 17.5 s | 0.77 – 1.86 s |
| evaluation wall time per seed (100 episodes) | 118 – 132 s | 53.2 – 62.6 s | 49.7 – 50.7 s |

Once planning is fast, the batched evaluation's wall time is dominated by the 48 non-planning calls per episode, each about 105 ms for 100 environments in the policy's per-call CPU preprocessing, and by the simulator; the fused path of R4 removes most of the former.

# Appendix D. Compute, cost and incident log

%APPENDIX D
All GPU work ran on Modal, on single H100 80GB HBM3 containers for runs and on CPU containers for kernel builds. The study was approved with a budget of $10 for the megakernel alone, extended to $35 when the equal-hardware ladder was added; it used about $31.50 (Table D1).

%TABLE: Approximate spend by stage.
| stage | what ran | spend |
|---|---|---|
| Phase A batch-1 pass | R0–R3 and R3-CEM, 61 seeds each | about $9.50 |
| final batch-1 pass | all seven rungs, 61 seeds each | about $11.60 |
| everything else | kernel builds and development runs, parity checks, micro-benchmarks, the fused encoder, and all 300-episode quality runs | about $10.40 (by difference) |
| total | | about $31.50 |

The batch-1 passes were the expensive part, and the reason is structural rather than computational: the harness starts a fresh process for each of the 61 seeds, which rebuilds the simulator, reloads the model and, for R2b, recompiles and warms the cost function at five shapes, so that most of a pass's GPU time is spent on setup rather than on the 75 plans it measures. A harness that keeps one process across seeds would cut the cost of a pass by an order of magnitude.

Five incidents cost time or money and changed the pipeline. The kernel extension did not rebuild after source changes, because the container mount did not preserve modification times; builds became content-addressed by the hash of the source, with the path of the built library recorded alongside. A `printf` on the failure path of the GEMM loop caused the compiler to serialise every WGMMA (warning C7510), and a run-time choice between `mm` and `mma` forms caused a related warning (C7515); both were removed. A torchvision build mismatched the pinned torch 2.11.0 and was re-pinned from the CUDA 12.8 index; an HDF5 compression plugin had to be imported explicitly before the dataset loaded; and one kernel build failed on a transient DNS error in the build container and was retried. None of these affected a reported result.

# Appendix E. Code and data availability

%APPENDIX E
All code, configuration, per-call latency records, per-episode outcomes, numerical parity records and analysis scripts are in the project repository (`lewm-flash`, branch `antoinenguyen27/lewm-megakernel`). The megakernel is `megakernel/lewm_mk.cu` with its packer (`pack.py`), runtime (`runtime.py`) and solver (`solver.py`); the parity, micro-benchmark and torch-baseline scripts are in the same directory. The ladder's rung definitions and the Modal entry points are in `modal/megakernel_app.py`; the benchmark harness is `tools/benchmarks/lewm_plan_benchmark.py`; the solvers of R1 and R2 are in `lewm/solvers/`. Per-call records are under `results/runs/<run>/metrics.jsonl`, the summaries under `results/ladder/` and the parity and micro-benchmark records under `results/megakernel/`. The script `docs/paper/src/compute_stats.py` regenerates every latency and success statistic in this paper from those records, and `docs/paper/src/make_figs.py` every figure. The model is the released LeWM Push-T checkpoint, evaluated with stable-worldmodel's `swm/PushT-v1` environment and its expert dataset; ThunderKittens is used at commit `67845f5f`. Reproducing one rung's batch-1 latency requires one H100 and about $1.50 to $2 of GPU time with the current harness.
