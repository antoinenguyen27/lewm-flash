# LeWM Phase 4 Precision Autocast Report

Status: implemented; accepted iCEM + model-cost BF16 promoted.

Date: 2026-06-28

## Purpose

This task tested the lowest-complexity precision path after accepted iCEM:
autocast only the repeated LeWM model-cost evaluation to BF16 while leaving the
planner distribution, `topk`, action output, telemetry, image encoding, and
policy preprocessing in their normal dtypes.

The scoped path is:

```text
JEPA.get_cost_from_embeddings()
  -> rollout_from_embeddings()
  -> action_encoder + predictor
  -> cost_from_rollout()
```

`get_cost_from_embeddings()` returns `float32` costs when autocast is active, so
CEM/iCEM ranking and update code sees the same dtype contract as before.

## Implementation

Code changes:

- `lewm/jepa.py`: added `set_model_cost_autocast("none"|"bf16")`, metadata
  accessors, and a CUDA-only `torch.autocast(dtype=torch.bfloat16)` scope around
  tensor-only model-cost rollout.
- `tools/benchmarks/lewm_plan_benchmark.py`: added `--model-cost-autocast`,
  benchmark metadata rows, manifest planner fields, and propagation into
  `compile_info`, `model_info`, `profiler_info`, and `evaluation_result`.
- `modal/benchmark_app.py`: threaded `model_cost_autocast` through both Modal
  benchmark entrypoints.
- Tests cover parser support, benchmark settings metadata, configure-time model
  wiring, and the JEPA cost return dtype contract.

The setting affects both tested branches because accepted iCEM with
`compile_cost=true` and static eager iCEM with `compile_cost=false` both call
`JEPA.get_cost_from_embeddings()`.

## Validation

Local:

```text
uv run python -m py_compile lewm/jepa.py tools/benchmarks/lewm_plan_benchmark.py modal/benchmark_app.py lewm/solvers/icem_static_compile.py -> OK
uv run python -m unittest tests.test_benchmark_utils -v -> OK, 46 tests
uv run python -m unittest tests.test_lewm_planner_benchmark_overrides -v -> OK, 16 tests
uv run python -m unittest tests.test_lewm_icem_static_compile -v -> OK, 5 skipped locally because torch unavailable
```

Modal:

```text
unit_phase4_9_model_cost_autocast -> OK, 5 tests
```

All guardrail and full-sweep manifests validated with:

```text
uv run python tools/benchmarks/run_manifest.py validate --planner-changing
```

## Guardrails

Both guardrails used clean planner timing:

```text
--profile cuda-events
--cem-telemetry none
--compile-diagnostics none
--icem-step-telemetry none
--disable-instrumentation
```

| Run | Configuration | Success | Eval Wall | Notes |
|---|---|---:|---:|---|
| `lewm_icem_bucket60_bf16_guard_seed0_l40s` | accepted iCEM + `compile_cost=true` + BF16 model-cost autocast | `9/10` | `20.02 s` | autocast active; compiled BF16 model-cost path; valid manifest |
| `lewm_icem_static_chain_eager_bf16_guard_seed0_l40s` | static eager iCEM + BF16 model-cost autocast | `9/10` | `9.95 s` | autocast active; valid manifest |

Both completed with finite plan latency rows and no quality collapse, so the
full 3x100 sweeps were run.

## Full 3x100 Results

All rows use clean planner timing. Warmup-excluded plan metrics remove rows
marked `warmup=true`; they still include the systematic plan-index-26
evaluation-cohort spike unless noted.

| Run | Quality | Eval Wall | Warmup-Excluded Plan Sum | Plan P50/P90 | Late P50/P90 | Decision |
|---|---:|---:|---:|---:|---:|---|
| `lewm_icem_bucket60_compile_prewarm_seed0-2_l40s` | `267/300` | `253.34 s` | `28.37 s` | `144.01/182.50 ms` | `147.29/189.63 ms` | prior accepted iCEM |
| `lewm_icem_bucket60_bf16_seed0-2_l40s` | `265/300` | `210.55 s` | `23.62 s` | `116.30/159.32 ms` | `120.45/159.98 ms` | promote |
| `lewm_icem_static_chain_eager_seed0-2_l40s` | `268/300` | `250.27 s` | `35.57 s` | `115.16/166.53 ms` | `115.90/168.29 ms` | diagnostic only |
| `lewm_icem_static_chain_eager_bf16_seed0-2_l40s` | `266/300` | `267.92 s` | `40.22 s` | `123.30/164.88 ms` | `131.61/166.16 ms` | reject |

Per-seed success:

| Run | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| accepted iCEM | `88/100` | `88/100` | `91/100` |
| accepted iCEM + BF16 | `87/100` | `88/100` | `90/100` |
| static eager | `89/100` | `88/100` | `91/100` |
| static eager + BF16 | `89/100` | `86/100` | `91/100` |

## Plan-26 Outlier

All clean 3x100 runs show a systematic plan-index-26 spike in every seed. This
appears to be an evaluation harness/cohort boundary, not ordinary steady-state
planning. It should be reported rather than hidden because it affects aggregate
plan time.

Excluding plan index 26:

| Run | Normal Plan Sum | Normal P50/P90 | Plan-26 Sum |
|---|---:|---:|---:|
| accepted iCEM | `19.50 s` | `143.27/179.51 ms` | `8.88 s` |
| accepted iCEM + BF16 | `15.80 s` | `115.57/156.51 ms` | `7.82 s` |
| static eager | `15.55 s` | `114.72/160.64 ms` | `20.02 s` |
| static eager + BF16 | `16.63 s` | `122.93/163.94 ms` | `23.59 s` |

This explains the decision. Static eager standard has excellent normal p50 and
quality, but its aggregate plan stability is worse because plan-26 spikes are
2.6x the accepted BF16 branch. Static eager BF16 is worse on wall and aggregate
plan time and does not justify promotion.

## Compile Warmup

The accepted compiled model-cost path still prewarms bucketed shapes:

```text
[300, 240, 180, 120, 60]
```

Compared with the prior accepted iCEM run:

| Run | Seed-0 Compile Warmup | Generated Kernels | Seed 1/2 Warmup |
|---|---:|---:|---:|
| accepted iCEM | `71.56 s` | `288` | `0.13/0.13 s` |
| accepted iCEM + BF16 | `61.65 s` | `927` | `0.09/0.09 s` |

BF16 generated more Inductor kernels but did not increase first-seed warmup in
this run. Warmup remains a deployment concern, but the accepted metric here is
warmed serving latency, not cold start.

## Decision

Promote `lewm_icem_bucket60_bf16_seed0-2_l40s` as the new accepted LeWM
planner-changing branch:

```text
L1 + L2 + L4 + TF32/high matmul + accepted iCEM + model-cost BF16 autocast
```

It passes the accepted-iCEM replacement quality rule (`265/300 >= 261/300`) and
improves same-regime clean planner latency:

```text
eval wall: 253.34 s -> 210.55 s = 1.20x
warmup-excluded plan sum: 28.37 s -> 23.62 s = 1.20x
plan p50: 144.01 ms -> 116.30 ms = 1.24x
plan p90: 182.50 ms -> 159.32 ms = 1.15x
```

Keep static eager iCEM implemented as a diagnostic/harness. It is not the new
baseline because its full 3x100 aggregate plan stability is worse, despite good
quality and normal p50.

Next sensible LeWM options:

- Treat accepted iCEM + BF16 as the current best LeWM branch.
- Do not spend further time on static-chain compile unless a new profile changes
  the plan-26/outlier picture.
- If optimizing LeWM further, profile the plan-index-26 boundary and non-CEM
  evaluation overhead separately; otherwise move to V-JEPA transfer.
