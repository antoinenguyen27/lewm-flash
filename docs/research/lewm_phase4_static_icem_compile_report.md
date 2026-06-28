# LeWM Phase 4 Static iCEM Compile Report

Status: implemented; static compiled chain rejected; static eager confirmed at
3x100 but retained as diagnostic only.

Date: 2026-06-28

## Purpose

After the telemetry audit, the only plausible wider compile direction was not
the full `policy.get_action()` loop. The feasible target was a tensor-only iCEM
chain:

```text
cached emb + cached goal_emb + init mean/var + precomputed noise
  -> candidates
  -> LeWM model-cost rollout
  -> top-k / elite gather
  -> mean-var updates
  -> final action sequence
```

This deliberately leaves Python dict preprocessing, active-env selection,
image/goal setup, policy action-buffer logic, telemetry, callbacks, timing, and
CPU action handoff outside the compiled region.

## Implementation

`lewm/solvers/icem_static_compile.py` adds `ICEMStaticCompileSolver` and the
pure tensor `run_static_icem_chain()` helper. The solver keeps accepted iCEM
settings available, precomputes noise outside the compiled function, and calls
`model.get_cost_from_embeddings()` inside the tensor chain. It supports:

- `static_chain_compile=false` for the eager static-chain control.
- `static_chain_compile=true` with `torch.compile(mode="reduce-overhead")`.
- `static_chain_fullgraph=true` only as a diagnostic knob.
- benchmark-side `warmup_static_chain_compile()` metadata without enabling
  compile diagnostics or CEM telemetry.

The default config now keeps `static_chain_compile=false`; the compiled path
must be explicitly requested because the first diagnostic showed unacceptable
compile cost.

## Validation

```text
local: uv run python -m py_compile lewm/solvers/icem_static_compile.py tools/benchmarks/lewm_plan_benchmark.py modal/benchmark_app.py lewm/solvers/__init__.py -> OK
local: uv run python -m unittest tests.test_lewm_planner_benchmark_overrides -v -> OK, 16 tests
local: uv run python -m unittest tests.test_lewm_icem_static_compile -v -> OK, 4 skipped locally because torch unavailable
local: uv run python -m unittest tests.test_benchmark_utils -v -> OK, 44 tests
Modal: unit_phase4_8_icem_static_compile_v2 -> OK, 4 tests
Modal: unit_phase4_9_model_cost_autocast -> OK, 5 tests
manifests: all six Task 4.8 run manifests validated
```

## Runs

All clean runs used:

```text
--profile cuda-events
--cem-telemetry none
--compile-diagnostics none
--icem-step-telemetry none
--disable-instrumentation
```

The intrusive controls used current code with:

```text
--profile cuda-events
--cem-telemetry cuda-events
--compile-diagnostics summary
```

Manifest plan latency excludes the first two plan rows as warmup. CEM telemetry
fields are intentionally zero in clean runs, so the faithful timing signal is
external plan latency plus evaluation time.

| Run | Success | Eval Time | Manifest Plan Mean/P50/P90 | First Plan | Plan 26 | Telemetry | Static Warmup | Kernels |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `lewm_model_cost_tf32_clean_seed0_l40s` | `44/50` | `38.48 s` | `99.41 / 38.81 / 42.00 ms` | `12.53 s` | `2.98 s` | clean | n/a | n/a |
| `lewm_model_cost_tf32_intrusive_current_seed0_l40s` | `46/50` | `39.39 s` | `98.39 / 38.65 / 42.45 ms` | `12.86 s` | `2.93 s` | intrusive | n/a | n/a |
| `lewm_icem_bucket60_clean_seed0_l40s` | `46/50` | `39.45 s` | `69.62 / 43.83 / 54.69 ms` | `11.32 s` | `1.23 s` | clean | n/a | n/a |
| `lewm_icem_bucket60_intrusive_current_seed0_l40s` | `46/50` | `36.65 s` | `60.96 / 35.75 / 39.64 ms` | `11.26 s` | `1.23 s` | intrusive | n/a | n/a |
| `lewm_icem_static_chain_eager_seed0_l40s` | `47/50` | `37.13 s` | `78.79 / 27.25 / 35.37 ms` | `13.70 s` | `2.40 s` | clean | n/a | n/a |
| `lewm_icem_static_chain_compile_seed0_l40s` | `47/50` | `87.74 s` | `58.31 / 38.26 / 43.27 ms` | `58.21 s` | `0.97 s` | clean | `812.83 s` | `4358` |

## Findings

Current-code intrusive exact-CEM and clean exact-CEM are effectively the same
for plan-latency p50/p90. Current-code intrusive iCEM, however, reports lower
CUDA-event p50/p90 than clean iCEM. Therefore historical iCEM telemetry-on
timings should not be mixed into clean whole-planner latency claims. This does
not poison accepted iCEM quality or like-for-like telemetry diagnostics.

Static eager iCEM is the only promising result in this task. It preserved or
improved seed-0 quality (`47/50` versus clean accepted iCEM `46/50`) and reduced
robust plan latency (`27.25 ms` p50 and `35.37 ms` p90 versus `43.83 ms` and
`54.69 ms`). Its manifest mean is worse than the p50/p90 imply because the
shared plan-26 outlier is larger in this run.

Task 4.9 ran the budget-worthy 3x100 static eager confirmation under the same
clean timing flags. Static eager standard passed quality at `268/300` and was
close on eval wall (`250.27 s` versus prior accepted iCEM `253.34 s`), but it
was not a robust latency promotion: warmup-excluded plan sum was `35.57 s`
versus `23.62 s` for accepted iCEM + BF16, and the systematic plan-index-26
spikes summed to `20.02 s` versus `7.82 s`. Static eager + BF16 also passed
quality (`266/300`) but regressed eval wall to `267.92 s` and plan sum to
`40.22 s`.

Static compiled iCEM is not sane as a promotion path now. It compiled one graph,
but the graph was enormous: `40526` captured calls, `7111` extern calls,
`23620` pre-fusion IR nodes, and `4358` generated kernels. Warmup took
`812.83 s`, and the first measured plan still took `58.21 s`. The later
steady-state p50/p90 were not better than static eager. This is not a practical
replacement for accepted iCEM.

## Decision

Do not run a 3x100 promotion for static compiled iCEM.

Keep the implementation because it is useful evidence and a controlled harness,
but leave compile opt-in only. The 3x100 static eager confirmation has now been
run, and it does not replace accepted iCEM + BF16. Further whole-chain
`torch.compile` work is not justified without a new profile that changes the
plan-index-26 or aggregate-plan-stability picture.
