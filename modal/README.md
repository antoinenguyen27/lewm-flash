# Modal Benchmark Operations

This layer provides the authoritative GPU runner for LeWM-Flash benchmarks.
The Modal image includes `swig` and `build-essential` because
`stable-worldmodel[env]` pulls `box2d-py`, which builds a native extension during
image creation.
It installs `h5py` and `hdf5plugin` explicitly because Push-T evaluation uses
`stable_worldmodel.data.HDF5Dataset`, which is only exported when the HDF5
format dependencies are present.
It also pins `datasets==2.19.2` and `pyarrow<21` because `stable-pretraining`
imports `datasets.config`, while old resolver choices lack that export and
newer PyArrow releases removed `pyarrow.PyExtensionType`.
`transformers<5` is pinned because the LeWM Hugging Face weights use the
Transformers 4 ViT state-dict layout (`encoder.encoder.layer.*`).
Each benchmark manifest records the installed runtime package versions,
including `stable_worldmodel` and `stable_pretraining`, so exact solver
comparisons can be traced back to the package implementation used by Modal.
The app excludes `.git/` from the uploaded source mount to avoid mutable Git
object races during Modal packaging, and injects the local source commit/diff
hash into the remote manifest environment instead.

## Benchmark Resource Contract

GPU benchmark, exactness, and profiling entrypoints request:

```text
gpu="L40S"
cpu=(8.0, 8.0)
```

The CPU tuple requests 8 cores and sets the CPU limit to 8 cores for more
repeatable launch, dispatch, materialization, and profiling behavior. Do not
compare raw timing from this pinned-resource contract against older unpinned
Modal runs except as historical context. Rerun paired controls and candidates
under the same contract before making acceptance claims.

Each benchmark manifest records the requested resource contract under:

```text
modal.gpu
modal.cpu_request
modal.cpu_limit
environment.resources
```

Each `metrics.jsonl` also begins each seed with:

```text
runtime_resource_info
```

That row records observed CPU count/affinity, cgroup CPU quota when available,
torch thread counts, thread environment variables, CUDA device count/name/UUID
when available, and the benchmark resource environment injected by the Modal
wrapper.

## One-Time Setup

```bash
uv sync
uv run modal setup
uv run modal volume create jepa-weights
uv run modal volume create jepa-results
```

## Stage LeWM Push-T Weights

Run the dependency/config preflight before large artifact staging:

```bash
uv run modal run modal/benchmark_app.py::check_lewm_runtime
```

Run a conversion-only smoke before the full dataset download:

```bash
uv run modal run modal/benchmark_app.py::stage_lewm_pusht --skip-dataset
```

```bash
uv run modal run modal/benchmark_app.py::stage_lewm_pusht
```

The staging function downloads the `quentinll/lewm-pusht` model repo into:

```text
/weights/lewm/pusht/
```

Hugging Face and torch caches are pinned to the weights Volume:

```text
/weights/cache/huggingface/
/weights/cache/torch/
```

It also writes the stable-worldmodel-compatible object checkpoint to:

```text
/weights/pusht/lewm_object.ckpt
```

For newer stable-worldmodel loaders that resolve `policy=pusht/lewm` under a
`checkpoints/` cache, it also stages:

```text
/weights/checkpoints/pusht/lewm/weights.pt
/weights/checkpoints/pusht/lewm/config.json
```

It downloads and decompresses the Push-T dataset repo file into:

```text
/weights/lewm/pusht/pusht_expert_train.h5.zst
/weights/datasets/pusht_expert_train.h5
```

The compressed dataset is about 13 GB, so this is intentionally a one-time Volume operation.
Staging logs are persisted at:

```text
/results/runs/stage_lewm_pusht/stdout.txt
/results/runs/stage_lewm_pusht/stderr.txt
```

## Smoke Baseline

```bash
uv run modal run modal/benchmark_app.py::run_lewm_baseline \
  --seed 0 \
  --episodes 2 \
  --run-id smoke_lewm_baseline_seed0
```

Expected artifacts:

```text
/results/runs/smoke_lewm_baseline_seed0/manifest.json
/results/runs/smoke_lewm_baseline_seed0/metrics.jsonl
/results/runs/smoke_lewm_baseline_seed0/stdout.txt
/results/runs/smoke_lewm_baseline_seed0/stderr.txt
```

## Full Baseline

```bash
uv run modal run modal/benchmark_app.py::run_lewm_baseline \
  --seed 0 \
  --episodes 50 \
  --run-id lewm_baseline_seed0_l40s
```

## LeWM-Flash Preset

Use this preset for the accepted optimized planner. It expands to the full
configuration in `configs/lewm_flash.yaml`, including embedding reuse,
GPU-resident CEM iteration bookkeeping, compiled model-cost evaluation,
TF32-capable matmuls, bucketed iCEM, and BF16 autocast scoped only to the
learned model-cost rollout.

Smoke:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_flash \
  --seed 0 \
  --episodes 2 \
  --run-id smoke_lewm_flash_seed0
```

Full 3x100 sweep:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_flash \
  --episodes 100 \
  --seeds 0,1,2 \
  --run-id lewm_icem_bucket60_bf16_seed0-2_l40s
```

Use `run_lewm_benchmark` directly only for ablations, profiling experiments, or
new solver variants that intentionally need the full flag surface.

## Benchmark Telemetry Modes

The benchmark keeps CEM telemetry opt-in so historical commands remain
comparable.

```text
--cem-telemetry none         Do not patch solver timing beyond plan latency.
--cem-telemetry summary      Write per-CEM-call CPU/region records and profiler ranges.
--cem-telemetry cuda-events  Also record CEM-level CUDA elapsed time and CUDA memory snapshots.
```

Telemetry modes are for benchmark attribution. `--profile cuda-events`,
`--cem-telemetry cuda-events`, torch-profiler ranges, compile diagnostics, and
`--icem-step-telemetry summary` are not compatible with claims about capturing
the whole planner loop in a CUDA graph or about clean no-instrumentation serving
latency. For graph/capture or serving-latency checks, run:

```text
--profile none
--cem-telemetry none
--compile-diagnostics none
--icem-step-telemetry none
```

This restriction does not invalidate the accepted L2/L4 controls: L2 removes
loop-internal CPU materialization, and L4 compiles only
`model.get_cost_from_embeddings()` outside the solver telemetry wrapper.

When enabled, `/results/runs/<run_id>/metrics.jsonl` contains:

```text
plan_latency        Per get_action timing, encode deltas, CEM-call deltas.
cem_call            Per solver call shape/config, CPU wall, region CPU, CUDA time/memory when requested.
runtime_resource_info CPU/thread/GPU resource snapshot for benchmark comparability.
profiler_info       Solver class, telemetry mode, profiler range status.
evaluation_result   Task metrics, evaluation wall time, encode totals, plan/CEM call totals.
```

The manifest summarizes these under:

```text
metrics_summary.plan_latency_ms
metrics_summary.cem_calls
metrics_summary.evaluations
environment.resources
```

For L2, the `cem_call` rows also record estimated CPU materialization counts.
Stock CEM should show `loop_cpu_materialization_count_estimate = n_batches *
n_steps`; exact CEM should show `0`.

Reused run IDs are supported, but formal experiments should still prefer fresh
run IDs. At benchmark startup, benchmark-owned artifacts are cleared before new
rows are written:

```text
metrics.jsonl
manifest.json
profiler_summary.json
profiler/
videos/
```

`stdout.txt` and `stderr.txt` are managed by the Modal wrapper and are truncated
by the subprocess log capture layer.

## L1 Cache Embeddings Candidate

Run this only after a matching baseline exists:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant cache_embeddings \
  --seed 0 \
  --episodes 50 \
  --run-id lewm_cache_embeddings_seed0_l40s
```

## L1 Cache Exactness Check

This is a targeted fixed-candidate cost-equivalence check for the `cache_embeddings`
variant. It is not a task-quality benchmark and should run after the L1 paired
benchmark results exist.

```bash
uv run modal run modal/benchmark_app.py::check_lewm_cache_exactness \
  --seed 0 \
  --samples 300 \
  --cem-steps 30 \
  --horizon 5 \
  --topk 30 \
  --run-id lewm_cache_exactness_seed0_l40s
```

Expected artifacts:

```text
/results/runs/lewm_cache_exactness_seed0_l40s/exactness.json
/results/runs/lewm_cache_exactness_seed0_l40s/stdout.txt
/results/runs/lewm_cache_exactness_seed0_l40s/stderr.txt
```

## L2 Exact CEM No CPU Sync

Run the L2 solver through the normal benchmark entrypoint by selecting both the
variant label and the Hydra solver config. This variant inherits the accepted
L1 embedding cache and changes only the CEM solver bookkeeping relative to L1:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant cem_exact_no_cpu_sync \
  --solver cem_exact \
  --seed 0 \
  --episodes 50 \
  --run-id lewm_cem_exact_no_cpu_sync_seed0_l40s
```

Run the solver equivalence unit test in the Modal image:

```bash
uv run modal run modal/benchmark_app.py::run_unit_tests \
  --test-name tests.test_lewm_cem_exact \
  --run-id unit_lewm_cem_exact
```

For a factorial comparison, `cem_exact_no_cpu_sync_only` runs the L2 solver
without the L1 embedding cache. The existing `cem_exact_no_cpu_sync` variant
keeps the accepted L1 cache enabled and therefore represents L1+L2.

## L3 Rollout Cleanup Candidates

The first L3 candidate, `rollout_prealloc`, reduced action-encoder calls but
slowed the predictor/rollout path in the focused torch-profiler pair. Keep its
results for provenance, but do not use it as the base for later phases.

The diagnostic refinement, `rollout_action_cache`, precomputes candidate action
embeddings once while keeping the dynamic embedding rollout. This keeps the
principled part of L3 without the static embedding buffer that made
`rollout_prealloc` slower. It remains experimental only: unit/smoke and
fixed-candidate exactness passed, but the focused profile had a one-episode
quality difference and the clean 3x100 CUDA-event sweep regressed aggregate CEM
CUDA time and evaluation wall time versus accepted L1+L2.

Do not use either L3 candidate as a base for future phases. The forward LeWM
branch is accepted L1+L2 (`cem_exact_no_cpu_sync`). L4 should be evaluated as
L1+L2+L4, not L1+L2+L3b+L4.

Historical unit check:

```bash
uv run modal run modal/benchmark_app.py::run_unit_tests \
  --test-name tests.test_lewm_jepa_cache \
  --run-id unit_lewm_jepa_rollout_action_cache
```

Historical smoke:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant rollout_action_cache \
  --solver cem_exact \
  --seed 0 \
  --episodes 2 \
  --profile cuda-events \
  --cem-telemetry cuda-events \
  --run-id smoke_lewm_rollout_action_cache_seed0
```

Historical focused torch-profiler pair:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_profile \
  --variant cem_exact_no_cpu_sync \
  --solver cem_exact \
  --seed 0 \
  --episodes 10 \
  --plans 100 \
  --cem-telemetry cuda-events \
  --run-id lewm_profile_l1_l2_for_l3b_seed0_l40s

uv run modal run modal/benchmark_app.py::run_lewm_profile \
  --variant rollout_action_cache \
  --solver cem_exact \
  --seed 0 \
  --episodes 10 \
  --plans 100 \
  --cem-telemetry cuda-events \
  --run-id lewm_profile_l3b_rollout_action_cache_seed0_l40s
```

Historical current-source CUDA-event sweep:

```bash
for seed in 0 1 2; do
  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cem_exact_no_cpu_sync \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l1_l2_current_telem100_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant rollout_action_cache \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l3b_current_telem100_seed${seed}_l40s"
done
```

The completed L3b evidence is recorded in:

```text
docs/research/lewm_l3_rollout_experiment_report.md
docs/optimization_ledger.md
```

## L4 Compile Cost

`compile_cost` compiles the tensor-only LeWM model-cost helper with
`torch.compile(mode="reduce-overhead", fullgraph=False)`. It starts from the
accepted L1+L2 branch: embedding cache enabled, `cem_exact` solver, and both L3
rollout candidates disabled.

Unit and smoke:

```bash
uv run modal run modal/benchmark_app.py::run_unit_tests \
  --test-name tests.test_lewm_jepa_cache \
  --run-id unit_lewm_l4_compile_cost_v3

uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant compile_cost \
  --solver cem_exact \
  --seed 0 \
  --episodes 2 \
  --profile cuda-events \
  --cem-telemetry cuda-events \
  --compile-diagnostics summary \
  --compile-warmup-history 1 \
  --run-id smoke_lewm_compile_cost_diag_seed0_v2
```

Fixed-candidate exactness:

```bash
uv run modal run modal/benchmark_app.py::check_lewm_cache_exactness \
  --candidate-mode compile_cost \
  --seed 0 \
  --samples 300 \
  --cem-steps 30 \
  --horizon 5 \
  --topk 30 \
  --run-id lewm_compile_cost_exactness_seed0_l40s_v3
```

The paired confirmation compares accepted L1+L2 against L1+L2+L4 with matching
telemetry and paired seeds. The L4 branch records compile diagnostics on every
CEM call: warmup signature, runtime signature, signature match/mismatch,
first-post-warmup marker, and best-effort PyTorch compiler counter deltas when
available.

```bash
for seed in 0 1 2; do
  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cem_exact_no_cpu_sync \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l1_l2_for_l4_telem100_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant compile_cost \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --compile-diagnostics summary \
    --compile-warmup-history 1 \
    --run-id "lewm_compile_cost_telem100_seed${seed}_l40s"
done
```

L4 is promoted for the warmed-runtime branch. The confirmation result was:
post-warm CEM CUDA `22.34 s -> 18.16 s` (`1.23x`), single post-warm CEM
`7.45 s -> 6.05 s`, and quality `262/300 -> 264/300` versus the paired L1+L2
control. Raw eval wall remains a cold/first-runtime metric and is not the
deployment acceptance metric; serve only after a real full-CEM warmup.

For a focused recompilation debug run, add PyTorch compiler logs to a short L4
profile instead of the broad timing sweep:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_profile \
  --variant compile_cost \
  --solver cem_exact \
  --seed 0 \
  --episodes 10 \
  --plans 120 \
  --cem-telemetry cuda-events \
  --compile-diagnostics summary \
  --compile-warmup-history 1 \
  --torch-compile-logs recompiles \
  --run-id lewm_profile_compile_cost_recompiles_seed0_l40s
```

## Pinned Accepted-Ladder Refresh

After the CPU resource contract changed, refresh the accepted ladder under the
same pinned resources before using it as the control for new work. Compare these
runs only against each other:

```bash
for seed in 0 1 2; do
  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cache_embeddings \
    --solver cem \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_cache_embeddings_pinned_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cem_exact_no_cpu_sync \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l1_l2_pinned_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant compile_cost \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --compile-diagnostics summary \
    --compile-warmup-history 1 \
    --run-id "lewm_compile_cost_pinned_seed${seed}_l40s"
done
```

Include `baseline` only when a fresh no-optimization headline is required.
Use `compile_cost` as the non-TF32 control; use `model_cost_tf32` for the
accepted Phase 3R numerical branch.

## L5 Tensor CEM Diagnostics - Closed

L5 variants are historical-only records. Their active runtime, Modal flags,
exactness modes, and benchmark variants have been removed after closeout. Phase
3R started from `compile_cost` (L1+L2+L4); its promoted numerical warmed-runtime
branch is `model_cost_tf32`.

Historical labels:

```text
tensor_cem_compile                  regional compile over tensorized CEM
tensor_cem_compile_no_inner_compile same diagnostic without separately compiling the inner L4 cost helper
tensor_cem_static                   static workspace/output-edge runtime boundary
```

L5b closeout:

```text
L4 control:       276.09 ms/output, success 46/50
Nested L5b:       279.81 ms/output, success 46/50
No-inner L5b:     277.18 ms/output, success 46/50
Decision:         closed/not promoted
```

The no-inner diagnostic showed nested compile was not the blocker. L5b reduced
visible Python/CEM glue, but the final synchronization absorbed the apparent
savings.

L5c focused closeout:

```text
L4 control post-warm CEM: 862.29 ms
L5c post-warm CEM:       899.54 ms
Focused success:          8/10 for both
CUDA kernel total:        3586.37 ms vs 3584.69 ms
Decision:                 closed/not promoted
```

L5c did not reduce GPU compute. The slowdown was explained by synchronization
relocation plus extra staging around unchanged candidate evaluation:

```text
L4 main wait range:
  cem.cpu_final_cost = 704.57 ms

L5c wait-like ranges:
  cem.static_expand_info = 498.72 ms
  cem.output_edge_cpu    = 241.83 ms
```

Do not run broad L5 acceptance sweeps. If this line of work is deliberately
reopened, reintroduce it as a fresh branch with a new implementation and a
same-stack `compile_cost` control, rather than relying on the removed diagnostic
wiring.

Phase 3R started from:

```text
variant: compile_cost
meaning: accepted L1+L2+L4 warmed-runtime branch
```

Phase 3R closeout:

```text
promoted: model_cost_tf32
rejected: fused_cem_glue, compile_default, compile_fullgraph,
          compile_max_autotune, fused_score
report: docs/research/lewm_phase3r_model_cost_report.md
```

Do not run broad fused-CEM-glue, compile/capture, or fused-score sweeps without
new profiling evidence. Kernel work is justified only if the accepted branch
shows the target is material.

## 100-Episode L0/L1/L2/L1+L2 Telemetry Sweep

Use this for the next L2 confirmation pass. These runs are intentionally a new
telemetry-bearing set, so compare them against each other rather than mixing
their raw latency numbers with the earlier non-CEM-telemetry sweep.

```bash
for seed in 0 1 2; do
  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant baseline \
    --solver cem \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_baseline_telem100_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cache_embeddings \
    --solver cem \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l1_telem100_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cem_exact_no_cpu_sync_only \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l2only_telem100_seed${seed}_l40s"

  uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
    --variant cem_exact_no_cpu_sync \
    --solver cem_exact \
    --seed "$seed" \
    --episodes 100 \
    --profile cuda-events \
    --cem-telemetry cuda-events \
    --run-id "lewm_l1_l2_telem100_seed${seed}_l40s"
done
```

## Torch Profiler Run

Normal benchmark sweeps default to `--profile cuda-events`; torch profiler is
not enabled globally because it perturbs timing and produces large traces. Use
`run_lewm_profile` for deep mechanism checks. That entrypoint always forwards
`--profile torch` and defaults CEM telemetry to `cuda-events`; do not use it for
clean serving-latency or CUDA graph-capture timing claims.

```bash
uv run modal run modal/benchmark_app.py::run_lewm_profile \
  --variant baseline \
  --seed 0 \
  --episodes 50 \
  --plans 100 \
  --cem-telemetry cuda-events \
  --run-id lewm_profile_baseline_seed0_l40s
```

Profiler traces are written under the run directory's `profiler/` folder, with
a compact trace aggregate at:

```text
/results/runs/<run_id>/profiler_summary.json
```

Torch profiler runs also mark both stock CEM and `CEMExactSolver` ranges:

```text
cem.init_action
cem.init_distribution
cem.expand_info
cem.sample_candidates
cem.model_get_cost
cem.topk
cem.elite_stats
cem.cpu_logging
cem.cost_reduce_gpu
cem.cpu_final_cost
cem.cpu_outputs
lewm.encode_initial
lewm.encode_goal
lewm.rollout
```

For a focused L1-vs-L1+L2 profile, run:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_profile \
  --variant cache_embeddings \
  --solver cem \
  --seed 0 \
  --episodes 20 \
  --plans 40 \
  --cem-telemetry cuda-events \
  --run-id lewm_profile_l1_seed0_l40s

uv run modal run modal/benchmark_app.py::run_lewm_profile \
  --variant cem_exact_no_cpu_sync \
  --solver cem_exact \
  --seed 0 \
  --episodes 20 \
  --plans 40 \
  --cem-telemetry cuda-events \
  --run-id lewm_profile_l1_l2_seed0_l40s
```

Add `--torch-profiler-with-stack` only for a short trace when Python stack
attribution is needed; it increases overhead and trace size.

## CEM-GD / iCEM-GD Diagnostics

`cem_gd` is an explicit planner-changing solver. It can seed from either CEM or
iCEM and then refine evaluated elites with eager action-only GD. The first
promotion-relevant diagnostic used iCEM seeding:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant icem_gd \
  --solver cem_gd \
  --episodes 50 \
  --seed 0 \
  --samples 300 \
  --cem-steps 25 \
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
  --gd-seed-solver icem \
  --gd-seed-source best_elite \
  --gd-refine-topk 1 \
  --gd-steps 1 \
  --gd-lr 0.01 \
  --compile-cost \
  --compile-warmup-sample-counts auto \
  --matmul-precision high \
  --cem-telemetry cuda-events \
  --baseline-run-id lewm_model_cost_tf32_telem100_l40s \
  --quality-gate-result diagnostic \
  --run-id lewm_icem_gd_bucket60_i25_g1_seed0_l40s
```

For this command, `return_best_elite=false`, `clip_actions=false`, and
`gd_compile_cost=false` are solver config defaults. Do not treat
`lewm_icem_gd_bucket60_i25_g1_seed0_l40s` as promoted: it lowered model cost but
regressed wall time. See
`docs/research/lewm_phase4_cem_gd_report.md`.

## Static iCEM Chain Diagnostics

`icem_static_compile` is a planner-changing diagnostic solver that moves the
accepted bucketed iCEM optimizer chain into a tensor-only helper. The default is
static eager. Whole-chain `torch.compile` is opt-in because the first diagnostic
compiled one enormous graph and paid an unacceptable warmup cost.

Clean seed-0 eager control:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant icem_static_compile \
  --solver icem_static_compile \
  --episodes 50 \
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
  --no-static-chain-compile \
  --no-compile-cost \
  --matmul-precision high \
  --profile cuda-events \
  --cem-telemetry none \
  --compile-diagnostics none \
  --icem-step-telemetry none \
  --disable-instrumentation \
  --baseline-run-id lewm_icem_bucket60_clean_seed0_l40s \
  --quality-gate-result diagnostic \
  --run-id lewm_icem_static_chain_eager_seed0_l40s
```

Compiled diagnostic, not promoted:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_benchmark \
  --variant icem_static_compile \
  --solver icem_static_compile \
  --episodes 50 \
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
  --static-chain-compile \
  --static-chain-mode reduce-overhead \
  --no-static-chain-fullgraph \
  --no-compile-cost \
  --matmul-precision high \
  --profile cuda-events \
  --cem-telemetry none \
  --compile-diagnostics none \
  --icem-step-telemetry none \
  --disable-instrumentation \
  --baseline-run-id lewm_icem_bucket60_clean_seed0_l40s \
  --quality-gate-result diagnostic \
  --run-id lewm_icem_static_chain_compile_seed0_l40s
```

The compiled run reached `47/50` but took `812.83 s` for static-chain warmup,
generated `4358` kernels, and had a `58.21 s` first measured plan. Keep it as
evidence, not as a promotion path. See
`docs/research/lewm_phase4_static_icem_compile_report.md`.

## BF16 Model-Cost Autocast

`--model-cost-autocast bf16` scopes CUDA BF16 autocast only to
`JEPA.get_cost_from_embeddings()`. The returned planner costs remain
`float32`; CEM/iCEM distribution updates, `topk`, action output, image encoding,
and telemetry stay outside the autocast scope.

Promoted accepted-iCEM + BF16 3x100 command:

```bash
uv run modal run modal/benchmark_app.py::run_lewm_flash \
  --episodes 100 \
  --seeds 0,1,2 \
  --run-id lewm_icem_bucket60_bf16_seed0-2_l40s
```

This run is promoted: `265/300`, eval wall `210.55 s`, and
warmup-excluded plan sum `23.62 s` versus prior accepted iCEM `267/300`,
`253.34 s`, and `28.37 s`. Static eager standard and static eager BF16 were
also run at 3x100, but remain diagnostic. See
`docs/research/lewm_phase4_precision_autocast_report.md`.

## Ledger Acceptance

After copying or mounting the result manifest locally:

```bash
uv run python tools/benchmarks/optimization_ledger.py \
  --ledger docs/optimization_ledger.md \
  --manifest /results/runs/lewm_baseline_seed0_l40s/manifest.json \
  --status accepted
```

Only manifest-backed runs should be accepted into the ledger.

## Nsight Notes

The manifest schema reserves a `profiler_trace` field for Nsight Systems artifacts. Keep Nsight as a targeted deep-profile path after CUDA-event timing and PyTorch profiler traces are reliable. Do not block the `cache_embeddings` experiment on Nsight availability inside Modal.
