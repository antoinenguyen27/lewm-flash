# Reproducibility

## Local Setup

```bash
uv sync
uv run python scripts/summarize_headline.py
```

Local unit tests:

```bash
uv run python -m unittest tests.test_benchmark_utils -v
uv run python -m unittest tests.test_lewm_planner_benchmark_overrides -v
uv run python -m unittest tests.test_lewm_icem_static_compile -v
```

The local environment does not need PyTorch for the benchmark-plumbing tests.
Torch-dependent tests run in the Modal image.

## Modal Runtime

The Modal app installs:

```text
torch
torchvision
stable-worldmodel[train,env]
triton
scipy
numpy
h5py
hdf5plugin
hydra-core
omegaconf
huggingface_hub
```

The app uses two Modal volumes:

```text
jepa-weights
jepa-results
```

`tools/benchmarks/stage_lewm_weights.py` stages the Push-T checkpoint and
dataset from Hugging Face into `jepa-weights`.

## Full Baseline Sweep

Run per seed:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant baseline \
  --solver cem \
  --episodes 100 \
  --seed 0 \
  --profile cuda-events \
  --cem-telemetry cuda-events \
  --run-id lewm_baseline_telem100_seed0_l40s
```

Repeat for seeds `1` and `2`.

## Full LeWM-Flash Sweep

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant icem \
  --solver icem \
  --episodes 100 \
  --seeds 0,1,2 \
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
  --baseline-run-id lewm_icem_bucket60_compile_prewarm_seed0-2_l40s \
  --quality-gate-result diagnostic \
  --run-id lewm_icem_bucket60_bf16_seed0-2_l40s
```

## Validate Manifests

```bash
uv run python tools/benchmarks/run_manifest.py validate \
  --manifest results/runs/lewm_icem_bucket60_bf16_seed0-2_l40s/manifest.json \
  --planner-changing
```
