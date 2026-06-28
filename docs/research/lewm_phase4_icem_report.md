# LeWM Phase 4 iCEM Report

Status: implemented; planner-changing candidate accepted; BF16 model-cost autocast now promoted on top of accepted iCEM.

Date: 2026-06-28

## Context

Phase 4 is planner-changing. The accepted quality control remains
L1+L2+L4+TF32, recorded as `lewm_model_cost_tf32_telem100_l40s` with
`263/300 = 87.67%` Push-T success.

L2 and L4 remain valid controls after the telemetry/capture audit. L2 is exact
CEM cleanup that removes loop-internal GPU-to-CPU materialization. L4 compiles
only the tensor-only `model.get_cost_from_embeddings()` helper, so CEM
telemetry around solver calls does not invalidate the compiled helper evidence.
Do not reinterpret these benchmark timings as whole-planner CUDA graph capture
or clean no-instrumentation serving latency.

Warm-start controls showed that `receding_horizon < horizon` collapses quality
for this Push-T fixed-goal evaluation. The first iCEM branch therefore keeps
`horizon=5`, `receding_horizon=5`, `action_block=5`, policy action-tail
warm-starting enabled, and no cross-MPC solver mean/variance/elite reuse.

## Implementation

Local `ICEMSolver` lives in `lewm/solvers/icem.py` and is exposed through
`lewm/config/eval/solver/icem.yaml`. It supports a CEM-parity mode plus
within-solve iCEM features:

- previous-iteration elite retention,
- optional pink/brown temporally colored noise over block or raw action time,
- light mean/variance momentum,
- sample decay with actual per-step sample counts in telemetry,
- optional bucketed sample decay to avoid excessive runtime shape churn,
- optional return of the best evaluated elite,
- optional action clipping.

The solver deliberately does not shift the CEM posterior or elite set across MPC
replans.

## Runs

| Run | Configuration | Success | Eval Wall | Model-Cost Evals | Post-Warm iCEM CUDA | Quality |
|---|---|---:|---:|---:|---:|---|
| `lewm_icem_parity_rh5_s300_i30_seed0_l40s_v2` | `300x30`, `topk=30`, no iCEM features | `44/50 = 88%` | `41.45 s` | 18,000 | `3341.31 ms` | pass |
| `lewm_icem_rh5_s150_i15_topk15_er5_pinkraw_seed0_l40s` | `150x15`, `topk=15`, elite retention 5, pink raw-time noise | `41/50 = 82%` | `63.30 s` | 4,500 | `1716.34 ms` | fail |
| `lewm_icem_rh5_s300_i30_topk30_er5_decay125_seed0_l40s` | `300x30`, `topk=30`, elite retention 5, sample decay 1.25 | `46/50 = 92%` | `78.12 s` | 5,140 | `1015.17 ms` | pass |
| `lewm_icem_rh5_s300_i30_topk30_er5_decay125_bucket60_seed0_l40s` | previous row plus `sample_count_bucket_size=60`, compiled cost | `46/50 = 92%` | `61.62 s` | 5,640 | `1168.20 ms` | pass |
| `lewm_icem_rh5_s300_i30_topk30_er5_decay125_bucket60_nocompile_seed0_l40s` | previous row, uncompiled cost | `47/50 = 94%` | `41.07 s` | 5,640 | `2692.09 ms` | pass |
| `lewm_icem_bucket60_nocompile_seed0-2_l40s` | 3x100 confirmation of bucket-60 uncompiled candidate | `268/300 = 89.33%` | `274.16 s` | 16,920 | second-call sum `24.02 s` | pass |
| `lewm_icem_bucket60_compile_prewarm_seed0_l40s` | compiled bucket-60 plus `--compile-warmup-sample-counts auto` | `46/50 = 92%` | `34.19 s` | 5,640 | `1176.04 ms` | pass |
| `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s` | 3x100 confirmation of compiled prewarmed bucket-60 candidate | `267/300 = 89.00%` | `253.34 s` | 16,920 | second-call sum `8.47 s` | pass |

The first parity attempt, `lewm_icem_parity_rh5_s300_i30_seed0_l40s`, failed at
`40/50`. It was not used for conclusions because it exposed two implementation
issues that were fixed before the v2 run: the `colored_noise=none` path did not
sample in exact CEM order, and per-step `expanded_infos` construction dropped
the model-cost embedding cache across CEM iterations. After the fix, the local
Modal unit test `unit_phase4_3_icem_v2` passed, including CEM-parity tensor
comparison against `CEMExactSolver`.

## Findings

The CEM-parity iCEM path is now healthy. It passes the seed-0 quality gate,
keeps loop-internal CPU materialization at zero, and records normal encode
counts (`128` total encodes versus the broken parity attempt's pathological
repeat encoding).

The aggressive paper-like reduced run (`150x15`, pink raw-time noise, retention)
is not a valid forward branch. It cuts model-cost evaluations by 4x, but fails
quality at `82%` and does not improve 50-episode wall time.

Sample decay is the first promising mechanism. It keeps 30 CEM update steps,
starts with the full 300 samples, then decays to a `60 = 2 * topk` floor:

```text
[300, 240, 192, 154, 123, 99, ..., 60]
sum = 2570 model-cost evaluations per CEM call
```

That run passed the seed-0 quality gate and reduced total model-cost evaluations
from 18,000 to 5,140. The post-warm measured iCEM CUDA call dropped from
`3341.31 ms` in parity mode to `1015.17 ms`.

The blocker is dynamic-shape compile overhead. The sample-decay run had a
`48.36 s` first measured iCEM call and `78.12 s` total 50-episode wall time.
Torch compile counters rose from 2 unique graphs in parity mode to 3 unique
graphs in the decay run, with additional symbolic-shape matmul signatures. This
means sample decay is not yet a production win under the current compiled
model-cost path, even though the warmed solver work is substantially cheaper.

Step 5b added `sample_count_bucket_size` to reduce sample-shape churn. With
bucket size `60`, the decayed counts become:

```text
[300, 240, 240, 180, 180, 120, 120, 120, 60, ..., 60]
sum = 2820 model-cost evaluations per CEM call
```

Bucketing improved the compiled 50-episode wall time from `78.12 s` to
`61.62 s` and reduced the first measured iCEM call from `48.36 s` to `36.29 s`,
but it did not remove the extra compile graph. Compile counters still showed
3 unique graphs and 2 fxgraph cache misses.

The uncompiled bucketed diagnostic isolated the tradeoff. It avoided compile
shape overhead and was strong on the 50-episode seed-0 diagnostic (`47/50`,
`41.07 s`), but the 3x100 confirmation did not beat the accepted TF32 runtime:
`274.16 s` aggregate wall versus baseline `262.76 s`. Quality passed
(`268/300` versus baseline `263/300`), and model-cost evaluations fell from the
baseline theoretical `54,000` to `16,920`, but uncompiled model-cost calls were
slower enough to erase the sample-count gain.

Step 5c added benchmark-side multi-shape compile warmup. For the bucketed iCEM
configuration, `--compile-warmup-sample-counts auto` derives and prewarms the
unique runtime model-cost sample shapes:

```text
[300, 240, 180, 120, 60]
```

This is intentionally outside planner behavior. It calls compiled
`model.get_cost_from_embeddings()` on zero tensors before evaluation and records
`compile_info.warmup_sample_counts`, `warmup_shape_count`, and per-shape
`warmup_shapes`. The seed-0 diagnostic confirmed the mechanism: evaluation wall
dropped to `34.19 s`, the first measured iCEM call dropped from the previous
compiled bucketed `36.29 s` to `10.32 s`, and the post-warm measured call stayed
near the compiled bucketed value (`1176.04 ms`).

The paired 3x100 confirmation is the promotion evidence. It passed quality at
`267/300 = 89.00%` versus the accepted TF32 baseline `263/300 = 87.67%`. It also
improved aggregate evaluation wall from `262.76 s` to `253.34 s` (`1.037x`) and
reduced model-cost evaluations from the theoretical exact-CEM budget of `54,000`
to a recorded `16,920` (`3.19x`). The measured second iCEM call sum improved
from the accepted TF32 baseline's `14.93 s` to `8.47 s` (`1.76x`).

The wall-clock uplift is much smaller than the model-cost evaluation reduction
because non-CEM work now dominates the 100-episode benchmark. Compile warmup is
also not free: the 3x100 run paid `71.56 s` warmup on seed 0, then only about
`0.13 s` on seeds 1 and 2 after compiler cache reuse. This candidate should
therefore be treated as a warmed-runtime planner-changing branch, not a cold-start
improvement.

## Closeout Diagnostics Before CEM-GD

Task 4.3d added compact per-step convergence telemetry behind
`--icem-step-telemetry summary`. The implementation uses
`ICEMStepSummaryCallback` in `lewm/solvers/icem.py`, installs it from
`tools/benchmarks/lewm_plan_benchmark.py`, and records `icem_step_summary` on
emitted `cem_call` rows. The callback stores compact detached tensor summaries
during the solve and materializes Python scalars at `end_solve`, so it avoids
logging candidates or full cost tensors. This mode is diagnostic-only: it adds
callback work inside the solve and is not suitable for promotion timing or
whole-planner CUDA graph/capture claims.

Validation:

```text
local: uv run python -m py_compile lewm/solvers/icem.py tools/benchmarks/lewm_plan_benchmark.py modal/benchmark_app.py tests/test_lewm_icem.py tests/test_lewm_planner_benchmark_overrides.py
local: uv run python -m unittest tests.test_lewm_planner_benchmark_overrides -v -> OK, 12 tests
local: uv run python -m unittest tests.test_lewm_icem -v -> OK, skipped locally because torch is unavailable
Modal: unit_phase4_3d_icem_step_telemetry -> OK, 9 tests
Modal: unit_phase4_3d_icem_step_telemetry_overrides -> OK, 12 tests
```

The accepted iCEM control with step telemetry,
`lewm_icem_bucket60_steptelemetry_seed0_l40s`, reached `47/50`, `39.54 s`,
and recorded two iCEM calls. The extra telemetry run is diagnostic only; the
non-telemetry seed-0 control remains `lewm_icem_bucket60_compile_prewarm_seed0_l40s`
at `46/50`, `34.19 s`.

For future graph/capture or clean serving-latency checks, run the same candidate
with telemetry disabled:

```text
--profile none
--cem-telemetry none
--compile-diagnostics none
--icem-step-telemetry none
```

Use `--profile cuda-events` and `--cem-telemetry cuda-events` for like-for-like
benchmark comparisons and attribution, not for capture-eligibility claims.

Selected averaged convergence summaries across the two telemetry iCEM calls:

| Step | Samples | Top-k Mean Cost | Best Cost | Mean Delta Norm | Var Mean |
|---:|---:|---:|---:|---:|---:|
| 0 | 300 | 115.34 | 67.35 | 1.737 | 0.962 |
| 5 | 120 | 36.78 | 18.91 | 1.088 | 0.773 |
| 10 | 60 | 34.43 | 14.14 | 0.866 | 0.613 |
| 15 | 60 | 27.50 | 12.05 | 0.727 | 0.502 |
| 20 | 60 | 22.00 | 10.45 | 0.592 | 0.413 |
| 25 | 60 | 17.71 | 8.88 | 0.488 | 0.337 |
| 29 | 60 | 14.82 | 7.94 | 0.405 | 0.286 |

The final third of the solve is still doing work: top-k mean cost improves by
`32.6%` from step 20 to step 29 and by `16.3%` from step 25 to step 29, while
mean movement and variance continue falling. This does not support an aggressive
`15`-step standalone iCEM cut.

Seed-0 zero-cost ablations:

| Run | Changed Flag | Success | Eval Wall | Model-Cost Evals | Post-Warm iCEM CUDA | Decision |
|---|---|---:|---:|---:|---:|---|
| `lewm_icem_bucket60_bestelite_seed0_l40s` | `return_best_elite=true` | `47/50` | `36.33 s` | 5,640 | `942.23 ms` | quality/CUDA positive, wall not promoted |
| `lewm_icem_bucket60_clip_seed0_l40s` | `clip_actions=true` | `41/50` | `37.49 s` | 5,640 | `2148.94 ms` | reject |
| `lewm_icem_bucket60_mom01_seed0_l40s` | `momentum_alpha=0.1` | `47/50` | `41.16 s` | 5,640 | `1212.57 ms` | reject for wall/runtime |
| `lewm_icem_bucket60_pinkraw_seed0_l40s` | `colored_noise=pink`, raw time | `38/50` | `38.63 s` | 5,640 | `3036.56 ms` | reject |
| `lewm_icem_bucket60_pinkblock_seed0_l40s` | `colored_noise=pink`, block time | `15/50` | `38.92 s` | 5,640 | `4947.35 ms` | reject |

`return_best_elite=true` is the only zero-cost toggle worth remembering. It
improved seed-0 success and post-warm solver CUDA, but it did not improve
episode wall versus the non-telemetry seed-0 control, so it is not a replacement
control without a dedicated 3x100 confirmation.

Seed-0 iteration-cut diagnostics with only `cem_steps` changed:

| Run | `cem_steps` | Success | Eval Wall | Model-Cost Evals | Post-Warm iCEM CUDA | Decision |
|---|---:|---:|---:|---:|---:|---|
| `lewm_icem_bucket60_i25_seed0_l40s` | 25 | `46/50` | `34.33 s` | 5,040 | `843.50 ms` | quality pass; solver CUDA win; wall flat |
| `lewm_icem_bucket60_i20_seed0_l40s` | 20 | `44/50` | `32.87 s` | 4,440 | `1096.02 ms` | diagnostic pass at minimum quality bar |

`i25` preserves quality and cuts solver CUDA, but does not improve seed-0 wall
enough to promote. `i20` is the first seed-0 wall-time win and reduces recorded
model-cost evaluations by `21.3%` versus the accepted seed-0 control, but it
lands exactly on the `44/50` minimum quality bar. Because telemetry did not
plateau by step 15 and `i20` has no quality margin, `lewm_icem_bucket60_i15_seed0_l40s`
was intentionally not run.

The paired 3x100 confirmations of `i20` and `i25` resolved the ambiguity:

| Run | Success | Eval Wall | Model-Cost Evals | Post-Warm iCEM CUDA | Decision |
|---|---:|---:|---:|---:|---|
| Exact TF32 control, `lewm_model_cost_tf32_telem100_l40s` | `263/300 = 87.67%` | `262.76 s` | 54,000 theoretical | `14.93 s` | exact/numerical quality baseline |
| Accepted iCEM, `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s` | `267/300 = 89.00%` | `253.34 s` | 16,920 | `8.47 s` | accepted planner baseline |
| `i25`, `lewm_icem_bucket60_i25_seed0-2_l40s` | `259/300 = 86.33%` | `187.80 s` | 15,120 | `8.10 s` | reject as baseline replacement |
| `i20`, `lewm_icem_bucket60_i20_seed0-2_l40s` | `252/300 = 84.00%` | `184.77 s` | 13,320 | `7.32 s` | reject as baseline replacement |

Both reduced-iteration candidates pass the broad Push-T non-regression gate
versus exact TF32. However, the Phase 4.3d adoption rule is stricter for
replacing an already accepted planner branch: the candidate must stay within 2
percentage points of accepted iCEM. `i25` is 2.67 points below accepted iCEM
(`86.33%` versus `89.00%`) and misses the threshold by two successes
(`259/300` versus the required `261/300`). `i20` is 5.0 points below accepted
iCEM (`84.00%` versus `89.00%`). Neither is accepted despite the large wall-time
improvements (`187.80 s` and `184.77 s` versus `253.34 s`).

## Wall Attribution Checkpoint

Task 4.7 added `summarize_wall_attribution()` and generated
`results/lewm_phase4_wall_attribution.md` from the accepted exact TF32 and
accepted iCEM metrics. Plan-latency rows are complete for both runs
(`150/150`), so the attribution is usable.

| Run | Eval Wall | Plan Latency | CEM CPU | Policy Non-CEM | Outside Policy | Model-Cost Evals |
|---|---:|---:|---:|---:|---:|---:|
| Exact TF32, `lewm_model_cost_tf32_telem100_l40s` | `262.76 s` | `101.30 s` (`38.6%`) | `83.15 s` (`31.6%`) | `18.15 s` (`6.9%`) | `161.46 s` (`61.4%`) | 54,000 |
| Accepted iCEM, `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s` | `253.34 s` | `75.51 s` (`29.8%`) | `54.89 s` (`21.7%`) | `20.62 s` (`8.1%`) | `177.83 s` (`70.2%`) | 16,920 |

iCEM reduced CEM CPU wall by `28.26 s` and total plan latency by `25.79 s`,
but aggregate eval wall improved by only `9.42 s` because outside-policy time is
now the dominant bucket. CEM-GD/iCEM-GD can still be useful for recovering
quality at lower iCEM iteration budgets, but the wall-attribution gate says not
to claim large remaining wall-clock headroom from planner-only work without
profiling `world.evaluate`, environment stepping, video, and dataset overhead.

## iCEM-GD First Diagnostic

Task 4.5/4.6 added `CEMGDSolver` and ran
`lewm_icem_gd_bucket60_i25_g1_seed0_l40s`. The solver uses compiled iCEM for the
seed search, captures final evaluated elites, then runs eager action-only GD
from the best elite. The first Modal attempt exposed an autograd boundary bug:
embeddings cloned inside `torch.inference_mode()` remained inference tensors.
The fix moves GD-bound clones outside inference mode and is covered by
`unit_phase4_5_cem_gd_v2`.

The diagnostic matched standalone i25 quality on seed 0 (`46/50`) and accepted
every measured GD refinement with lower model cost
(`gd_cost_delta_mean=-0.277890`, `gd_nan_or_inf_count=0`), but it regressed wall:
`40.04 s` eval wall versus `34.33 s` for standalone i25 and `34.19 s` for the
accepted i30 seed-0 reference. Post-warm solver CUDA was `1,702.50 ms` versus
`843.50 ms` for standalone i25. Do not promote this iCEM-GD path or run its
3x100 promotion. Details are in
`docs/research/lewm_phase4_cem_gd_report.md`.

## Static iCEM Chain Compile Diagnostic

Task 4.8 implemented `ICEMStaticCompileSolver` in
`lewm/solvers/icem_static_compile.py` to test a tensor-only iCEM chain and an
opt-in `torch.compile` wrapper around that chain. Clean timing runs used
`--cem-telemetry none`, `--compile-diagnostics none`,
`--icem-step-telemetry none`, and `--disable-instrumentation`.

Current-code intrusive controls confirmed that historical iCEM telemetry-on
latency should not be mixed with clean whole-planner latency claims: exact-CEM
clean and intrusive p50/p90 were effectively identical, but intrusive iCEM
reported lower CUDA-event p50/p90 than clean iCEM. This does not change the
accepted iCEM quality result or like-for-like telemetry diagnostics.

Static eager iCEM was promising on seed 0: `47/50`, `37.13 s` eval wall, and
manifest plan p50/p90 `27.25/35.37 ms` versus clean accepted iCEM `46/50`,
`39.45 s`, and `43.83/54.69 ms`. Its manifest mean was worse than p50/p90 imply
because the shared plan-26 outlier was larger in the eager static run.

Static compiled iCEM is not promoted. It reached `47/50`, but static-chain
compile warmup took `812.83 s`, generated `4358` kernels, still had a `58.21 s`
first measured plan, and produced `87.74 s` 50-episode eval wall. Keep compile
as an explicit diagnostic, not a default. The only budget-worthy follow-up from
Task 4.8 is a 3x100 clean confirmation of static eager iCEM if we want to test
whether the seed-0 p50/p90 win survives seeds 0-2. Details are in
`docs/research/lewm_phase4_static_icem_compile_report.md`.

## BF16 Model-Cost Autocast Closeout

Task 4.9 added a model-cost-only `bf16` autocast scope inside
`JEPA.get_cost_from_embeddings()` while returning planner costs as `float32`.
This keeps iCEM distribution updates, `topk`, telemetry, action buffers, image
encoding, and action output outside the dtype change.

Modal unit `unit_phase4_9_model_cost_autocast` passed. Both 10-episode BF16
guardrails reached `9/10` and produced valid manifests. Full clean 3x100 results
were:

| Run | Success | Eval Wall | Warmup-Excluded Plan Sum | Plan P50/P90 | Decision |
|---|---:|---:|---:|---:|---|
| accepted iCEM, `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s` | `267/300` | `253.34 s` | `28.37 s` | `144.01/182.50 ms` | prior accepted branch |
| accepted iCEM + BF16, `lewm_icem_bucket60_bf16_seed0-2_l40s` | `265/300` | `210.55 s` | `23.62 s` | `116.30/159.32 ms` | promote |
| static eager, `lewm_icem_static_chain_eager_seed0-2_l40s` | `268/300` | `250.27 s` | `35.57 s` | `115.16/166.53 ms` | diagnostic only |
| static eager + BF16, `lewm_icem_static_chain_eager_bf16_seed0-2_l40s` | `266/300` | `267.92 s` | `40.22 s` | `123.30/164.88 ms` | reject |

The accepted iCEM + BF16 branch passes the replacement quality rule
(`265/300 >= 261/300`) and improves same-regime clean planner latency by about
`1.20x` on warmup-excluded plan sum. Static eager standard has good quality and
normal p50, but it is not promoted because its systematic plan-index-26 spikes
are much larger than the accepted BF16 branch (`20.02 s` total versus
`7.82 s`). Static eager BF16 regresses wall and aggregate plan time.

Details are in `docs/research/lewm_phase4_precision_autocast_report.md`.

## Decision

Promote the compiled, prewarmed, bucketed iCEM configuration with model-cost
BF16 autocast as the current LeWM planner-changing candidate:

```text
solver=icem
samples=300
cem_steps=30
topk=30
elite_retention=5
sample_decay=1.25
sample_count_bucket_size=60
compile_cost=true
compile_warmup_sample_counts=auto
model_cost_autocast=bf16
horizon=5
receding_horizon=5
```

Do not replace exact CEM by default for quality-preserving comparisons. This
branch changes the optimizer distribution and needs task-level gates. Future
MPPI, CEM-GD, or iCEM-GD candidates should beat this iCEM branch, not only the
older exact TF32 baseline, if they are meant to become the best planner-changing
path.

After the Task 4.9 confirmation, the accepted iCEM control is
`lewm_icem_bucket60_bf16_seed0-2_l40s`. The `i25` and `i20` cuts are
useful evidence for low-budget or future iCEM-GD recovery experiments, but they
should not replace the accepted standalone iCEM baseline under the current
quality rule. Do not use `i15`, clipping, or pink-noise iCEM as forward
defaults.

Budget decision after the `i20`/`i25` confirmations: do not run an `i26`-`i29`
standalone interpolation sweep now. The likely upside is a narrow pure-iCEM
tradeoff, and any winner would still need 3x100 confirmation against the
accepted `i30` branch. Reserve at most one `i27` or `i28` 3x100 fallback if
CEM-GD/iCEM-GD fails or if a conservative pure-iCEM Pareto point becomes needed.
After the first iCEM-GD diagnostic, the current planner baseline remains the
accepted standalone iCEM run. Reopen CEM-GD only as a targeted diagnostic to
reduce the small-S GD re-evaluation overhead or test a no-backtracking
improvement-check variant; do not promote the current `i25 + gd_steps=1` path.
After the static-chain compile diagnostic, do not promote whole-chain
`torch.compile`. After the static eager 3x100 confirmation, keep static eager as
a diagnostic/harness only rather than the accepted branch because aggregate plan
stability is worse than accepted iCEM + BF16.
