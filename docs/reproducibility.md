# Reproducibility

## Local Setup

```bash
uv sync
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
uv run modal run modal/benchmark_app.py::run_lewm_baseline \
  --episodes 100 \
  --seed 0 \
  --run-id lewm_baseline_telem100_seed0_l40s
```

Repeat for seeds `1` and `2`.

## Full LeWM-Flash Sweep

```bash
uv run modal run modal/benchmark_app.py::run_lewm_flash \
  --episodes 100 \
  --seeds 0,1,2 \
  --run-id lewm_icem_bucket60_bf16_seed0-2_l40s
```

The wrapper expands to the accepted planner configuration in
`configs/lewm_flash.yaml`. Use the lower-level `run_lewm_benchmark` entrypoint
only for ablations or new solver experiments.

## Validate Manifests

```bash
uv run python tools/benchmarks/run_manifest.py validate \
  --manifest results/runs/lewm_icem_bucket60_bf16_seed0-2_l40s/manifest.json \
  --planner-changing
```
