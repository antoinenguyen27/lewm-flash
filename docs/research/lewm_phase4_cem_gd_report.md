# LeWM Phase 4 CEM-GD / iCEM-GD Report

Updated 2026-06-28.

## Implementation

Task 4.5 added `lewm/solvers/cem_gd.py` and
`lewm/config/eval/solver/cem_gd.yaml`. The solver runs a CEM or iCEM seed
search, captures evaluated final elites, then refines only action tensors with
eager Adam. It returns refined actions only when the model cost improves.

The implementation keeps seed-search compilation separate from GD compilation:

```text
--compile-cost       seed CEM/iCEM forward cost helper
--gd-compile-cost    GD objective, default false
```

`install_compile_cost()` now preserves the original eager
`model.get_cost_from_embeddings` before wrapping it with `torch.compile`, so the
first GD path can use eager autograd even when the seed iCEM path is compiled.
The first Modal attempt exposed a real autograd boundary bug: cached seed
embeddings cloned inside `torch.inference_mode()` remained inference tensors and
failed backward. The solver now moves all GD-bound clones outside the inference
context, and `tests.test_lewm_cem_gd` has a regression that uses `goal_emb` in
the differentiable loss.

## Verification

Local:

```text
uv run python -m py_compile lewm/solvers/cem_gd.py tools/benchmarks/lewm_plan_benchmark.py tools/benchmarks/metrics.py tools/benchmarks/summarize_results.py modal/benchmark_app.py tests/test_lewm_cem_gd.py tests/test_lewm_planner_benchmark_overrides.py tests/test_benchmark_utils.py
uv run python -m unittest tests.test_benchmark_utils -v
uv run python -m unittest tests.test_lewm_planner_benchmark_overrides -v
```

Modal torch-backed unit:

```text
unit_phase4_5_cem_gd_v2
tests.test_lewm_cem_gd: 6 tests OK
```

The local uv environment has no torch, so `tests.test_lewm_cem_gd` skips locally
and is verified in Modal.

## First Diagnostic

Run:

```text
lewm_icem_gd_bucket60_i25_g1_seed0_l40s
variant=icem_gd
solver=cem_gd
gd_seed_solver=icem
gd_seed_source=best_elite
gd_refine_topk=1
gd_steps=1
gd_lr=0.01
gd_compile_cost=false
compile_cost=true
compile_warmup_sample_counts=auto
samples=300
cem_steps=25
topk=30
sample_decay=1.25
sample_count_bucket_size=60
elite_retention=5
```

Results:

| Run | Success | Eval Wall | Model-Cost Evals | Post-Warm Solver CUDA | Notes |
|---|---:|---:|---:|---:|---|
| Standalone i25, `lewm_icem_bucket60_i25_seed0_l40s` | `46/50` | `34.33 s` | 5,040 | `843.50 ms` | seed-0 low-budget control |
| Accepted iCEM seed-0, `lewm_icem_bucket60_compile_prewarm_seed0_l40s` | `46/50` | `34.19 s` | 5,640 | `1,176.04 ms` | seed-0 accepted-i30 reference |
| iCEM-GD i25 g1, `lewm_icem_gd_bucket60_i25_g1_seed0_l40s` | `46/50` | `40.04 s` | 5,050 incl. GD | `1,702.50 ms` | not promoted |

GD telemetry on the measured post-warm call:

```text
gd_accept_rate=1.0
gd_cost_delta_mean=-0.277890
gd_nan_or_inf_count=0
gd_forward_time_ms_mean=11.65
gd_backward_time_ms_mean=13.00
gd_model_cost_evals=5
```

Interpretation:

- The action-gradient step is numerically doing something useful: every refined
  measured candidate was accepted, model cost decreased, and no NaN/inf appeared.
- It did not recover any seed-0 task quality because standalone i25 already
  matched accepted i30 at `46/50` on seed 0.
- It materially worsened wall time. The measured post-warm solver call grew from
  `843.50 ms` on standalone i25 to `1,702.50 ms` with iCEM-GD.
- The backward pass itself is not the whole overhead. The implementation also
  does a gradient forward plus four backtracking re-evaluations for each active
  batch. With `batch_size=1` and active-env batching, those small-S model-cost
  calls add substantial Python/model overhead despite adding only five
  candidate-evaluation counts per environment.

## Decision

Do not promote `i25 + gd_steps=1` as implemented. Do not run the 3x100
promotion for this candidate, and do not run the planned `i20_g1` diagnostic
from Task 4.6 yet because the prerequisite was not met: `i25_g1` did not keep
episode wall below accepted standalone iCEM.

The useful next CEM-GD work, if reopened, is not `gd_steps=3`; it is first to
remove or batch the expensive small-S re-evaluation path, or run a no-backtracking
diagnostic that still performs an improvement check. Any such change should be
treated as a new diagnostic, not as a promotion of the current CEM-GD path.
