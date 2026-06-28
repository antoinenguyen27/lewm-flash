# LeWM-Flash

Fast MPC planning for LeWorldModel.

LeWM-Flash is a clean, reproducible implementation artifact for accelerating
LeWorldModel planning on Push-T. Against a no-opt LeWM/stable-worldmodel
baseline, the current best configuration reduces warmup-excluded planner
latency from `94.82 s` to `23.62 s` across 3x100 Push-T evaluation episodes,
with success changing from `262/300` to `265/300`.

| Metric | No-Opt LeWM | LeWM-Flash | Change |
|---|---:|---:|---:|
| Warmup-excluded planner time | `94.82 s` | `23.62 s` | `4.01x faster` |
| Eval wall time | `524.18 s` | `210.55 s` | `2.49x faster` |
| Plan latency p50 | `251.73 ms` | `116.30 ms` | `2.16x faster` |
| Plan latency p90 | `275.29 ms` | `159.32 ms` | `1.73x faster` |
| Push-T success | `262/300` | `265/300` | no regression |

Scope: these are NVIDIA L40S Push-T benchmark results. "Planner time" means
external CUDA-event `policy.get_action()` latency with warmup-marked plan calls
excluded. It is not a claim of full CUDA graph capture, custom kernels, or
universal LeWM speedup. "Flash" is the project name for the optimized planning
stack; it does not imply FlashAttention kernels.

## What Changed

The final LeWM-Flash configuration is:

```text
L1 + L2 + L4 + TF32/high matmul + bucketed iCEM + model-cost BF16 autocast
```

Concretely:

- cache current/goal embeddings for repeated model-cost calls,
- remove loop-internal CEM GPU-to-CPU synchronization,
- compile the tensor-only model-cost helper,
- use TF32-capable matmul for the repeated model-cost GEMMs,
- replace exact CEM with validated bucketed iCEM,
- autocast only `JEPA.get_cost_from_embeddings()` to BF16 and return planner
  costs as `float32`.

Several plausible paths were rejected and are recorded:

- receding-horizon/posterior warm-start,
- rollout preallocation and action-embedding cache,
- whole-chain `torch.compile`,
- first iCEM-GD/CEM-GD path,
- static eager BF16.

## Repository Layout

```text
lewm/                  Upstream-derived LeWM model/eval code plus optimized solvers
tools/benchmarks/      Manifest-backed benchmark and analysis tools
modal/                 Modal L40S benchmark entrypoints
configs/               Human-readable baseline and LeWM-Flash run configs
results/runs/          Selected metrics/manifests for the headline comparison
docs/                  Method, reproducibility, upstream, and experiment notes
scripts/               Small local summary utilities
tests/                 Focused unit tests for benchmark plumbing and solvers
```

## Reproduce The Headline Table

The selected run artifacts are committed under `results/runs/`.

```bash
uv run python scripts/summarize_headline.py
```

Expected output:

```text
No-opt planner sum: 94.82 s
LeWM-Flash planner sum: 23.62 s
Planner speedup: 4.01x
No-opt eval wall: 524.18 s
LeWM-Flash eval wall: 210.55 s
Eval wall speedup: 2.49x
Quality: 262/300 -> 265/300
```

## Run A Small Benchmark Smoke

The full 3x100 sweep is GPU-expensive. A short Modal smoke verifies that the
benchmark stack, checkpoint staging, and optimized planner still run:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant icem \
  --solver icem \
  --episodes 2 \
  --seed 0 \
  --samples 300 \
  --cem-steps 30 \
  --topk 30 \
  --horizon 5 \
  --receding-horizon 5 \
  --warm-start \
  --elite-retention 5 \
  --colored-noise none \
  --colored-noise-time raw \
  --momentum-alpha 0.0 \
  --sample-decay 1.25 \
  --sample-count-bucket-size 60 \
  --compile-cost \
  --compile-warmup-sample-counts auto \
  --matmul-precision high \
  --model-cost-autocast bf16 \
  --profile cuda-events \
  --cem-telemetry none \
  --compile-diagnostics none \
  --icem-step-telemetry none \
  --disable-instrumentation \
  --baseline-run-id lewm_icem_bucket60_bf16_seed0-2_l40s \
  --quality-gate-result diagnostic \
  --run-id lewm_flash_smoke_seed0_l40s
```

See `configs/lewm_flash.yaml` for the full accepted configuration and
`docs/reproducibility.md` for staging and full-sweep commands.

## Tests

Local tests avoid requiring local PyTorch where possible:

```bash
uv run python -m unittest tests.test_benchmark_utils -v
uv run python -m unittest tests.test_lewm_planner_benchmark_overrides -v
uv run python -m unittest tests.test_lewm_icem_static_compile -v
```

The torch-dependent static/iCEM tests are exercised in the Modal runtime.

## Upstream And Citations

This repo builds on LeWorldModel and stable-worldmodel. It includes
upstream-derived LeWM source under the retained LeWM MIT license and records
source provenance in `docs/upstream.md`.

Primary references:

- LeWorldModel paper: https://arxiv.org/abs/2603.19312
- LeWM source: https://github.com/lucas-maes/le-wm
- stable-worldmodel source: https://github.com/galilai-group/stable-worldmodel
- iCEM: https://arxiv.org/abs/2008.06389
- CEM-GD context: https://arxiv.org/abs/2112.07746

## License

MIT. See `LICENSE` and `docs/upstream.md` for upstream attribution.
