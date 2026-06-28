"""Modal entrypoints for JEPA MPC optimization benchmarks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "lewm-flash"
GPU_TYPE = "L40S"
BENCHMARK_CPU_REQUEST = 8.0
BENCHMARK_CPU_LIMIT = 8.0
BENCHMARK_CPU = (BENCHMARK_CPU_REQUEST, BENCHMARK_CPU_LIMIT)
LEWM_FLASH_QUALITY_BASELINE_RUN_ID = "lewm_icem_bucket60_compile_prewarm_seed0-2_l40s"
WORKDIR = "/workspace"
LOCAL_SOURCE_IGNORE = [
    ".git/**",
    ".venv/**",
    "venv/**",
    "wandb/**",
    "outputs/**",
    "runs/**",
    "checkpoints/**",
    "data/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/*.safetensors",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.gif",
    "**/*.ipynb",
    "**/*.npz",
]


weights_volume = modal.Volume.from_name("jepa-weights", create_if_missing=True)
results_volume = modal.Volume.from_name("jepa-results", create_if_missing=True)


def _local_source_env() -> dict[str, str]:
    try:
        from tools.benchmarks.run_manifest import git_metadata

        metadata = git_metadata(Path(__file__).resolve().parents[1])
    except Exception:
        return {}
    if (
        not metadata.get("base_git_commit")
        or metadata.get("base_git_commit") == "unknown"
        or not metadata.get("git_diff_sha256")
        or metadata.get("git_diff_sha256") == "unknown"
    ):
        return {}
    return {
        "JEPA_BASE_GIT_COMMIT": metadata["base_git_commit"],
        "JEPA_GIT_DIFF_SHA256": metadata["git_diff_sha256"],
    }


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "curl",
        "zstd",
        "swig",
        "build-essential",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
    )
    .uv_pip_install(
        "torch",
        "torchvision",
        "stable-worldmodel[train,env]",
        "triton",
        "pytest",
        "scipy",
        "numpy",
        "scikit-learn",
        "pillow",
        "h5py",
        "hdf5plugin",
        "datasets==2.19.2",
        "pyarrow<21",
        "transformers<5",
        "hydra-core",
        "omegaconf",
        "huggingface_hub",
        "hf_xet",
    )
    .env(_local_source_env())
    .add_local_dir(".", remote_path=WORKDIR, ignore=LOCAL_SOURCE_IGNORE)
)


app = modal.App(APP_NAME, image=image)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "STABLEWM_HOME": "/weights",
            "HF_HOME": "/weights/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/weights/cache/huggingface/hub",
            "TORCH_HOME": "/weights/cache/torch",
            "MUJOCO_GL": "egl",
            "JEPA_MODAL_GPU_TYPE": GPU_TYPE,
            "JEPA_MODAL_CPU_REQUEST": str(BENCHMARK_CPU_REQUEST),
            "JEPA_MODAL_CPU_LIMIT": str(BENCHMARK_CPU_LIMIT),
        }
    )
    return env


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=WORKDIR, check=True, env=_subprocess_env())


def _tail(path: Path, lines: int = 80) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<could not read {path}: {type(exc).__name__}: {exc}>"
    return "\n".join(content[-lines:])


def _run_with_logs(cmd: list[str], run_id: str) -> None:
    run_dir = Path("/results") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout = run_dir / "stdout.txt"
    stderr = run_dir / "stderr.txt"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        try:
            subprocess.run(
                cmd,
                cwd=WORKDIR,
                check=True,
                stdout=out,
                stderr=err,
                text=True,
                env=_subprocess_env(),
            )
        except subprocess.CalledProcessError:
            print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
            print(f"stdout: {stdout}", file=sys.stderr)
            print(_tail(stdout), file=sys.stderr)
            print(f"stderr: {stderr}", file=sys.stderr)
            print(_tail(stderr), file=sys.stderr)
            raise


def _append_lewm_planner_args(
    cmd: list[str],
    *,
    samples: int | None = None,
    cem_steps: int | None = None,
    topk: int | None = None,
    horizon: int | None = None,
    receding_horizon: int | None = None,
    action_block: int | None = None,
    warm_start: bool | None = None,
    solver_warm_start: bool | None = None,
    elite_retention: int | None = None,
    colored_noise: str | None = None,
    colored_noise_time: str | None = None,
    momentum_alpha: float | None = None,
    sample_decay: float | None = None,
    sample_count_bucket_size: int | None = None,
    return_best_elite: bool | None = None,
    clip_actions: bool | None = None,
    mppi_temperature: float | None = None,
    gd_seed_solver: str | None = None,
    gd_seed_source: str | None = None,
    gd_refine_topk: int | None = None,
    gd_steps: int | None = None,
    gd_lr: float | None = None,
    gd_compile_cost: bool | None = None,
    static_chain_compile: bool | None = None,
    static_chain_mode: str | None = None,
    static_chain_fullgraph: bool | None = None,
    static_chain_warmup: bool | None = None,
    icem_step_telemetry: str | None = None,
    baseline_run_id: str | None = None,
    quality_gate_result: str | None = None,
) -> None:
    optional_ints = (
        ("--samples", samples),
        ("--cem-steps", cem_steps),
        ("--topk", topk),
        ("--horizon", horizon),
        ("--receding-horizon", receding_horizon),
        ("--action-block", action_block),
        ("--elite-retention", elite_retention),
        ("--gd-refine-topk", gd_refine_topk),
        ("--gd-steps", gd_steps),
    )
    for flag, value in optional_ints:
        if value is not None:
            cmd.extend([flag, str(value)])
    if warm_start is not None:
        cmd.extend(["--warm-start", "true" if warm_start else "false"])
    if solver_warm_start is not None:
        cmd.extend(["--solver-warm-start", "true" if solver_warm_start else "false"])
    if colored_noise is not None:
        cmd.extend(["--colored-noise", colored_noise])
    if colored_noise_time is not None:
        cmd.extend(["--colored-noise-time", colored_noise_time])
    if momentum_alpha is not None:
        cmd.extend(["--momentum-alpha", str(momentum_alpha)])
    if sample_decay is not None:
        cmd.extend(["--sample-decay", str(sample_decay)])
    if sample_count_bucket_size is not None:
        cmd.extend(["--sample-count-bucket-size", str(sample_count_bucket_size)])
    if return_best_elite is not None:
        cmd.extend(["--return-best-elite", "true" if return_best_elite else "false"])
    if clip_actions is not None:
        cmd.extend(["--clip-actions", "true" if clip_actions else "false"])
    if mppi_temperature is not None:
        cmd.extend(["--mppi-temperature", str(mppi_temperature)])
    if gd_seed_solver is not None:
        cmd.extend(["--gd-seed-solver", gd_seed_solver])
    if gd_seed_source is not None:
        cmd.extend(["--gd-seed-source", gd_seed_source])
    if gd_lr is not None:
        cmd.extend(["--gd-lr", str(gd_lr)])
    if gd_compile_cost is not None:
        cmd.extend(["--gd-compile-cost", "true" if gd_compile_cost else "false"])
    if static_chain_compile is not None:
        cmd.extend(["--static-chain-compile", "true" if static_chain_compile else "false"])
    if static_chain_mode is not None:
        cmd.extend(["--static-chain-mode", static_chain_mode])
    if static_chain_fullgraph is not None:
        cmd.extend(["--static-chain-fullgraph", "true" if static_chain_fullgraph else "false"])
    if static_chain_warmup is not None:
        cmd.extend(["--static-chain-warmup", "true" if static_chain_warmup else "false"])
    if icem_step_telemetry is not None:
        cmd.extend(["--icem-step-telemetry", icem_step_telemetry])
    if baseline_run_id is not None:
        cmd.extend(["--baseline-run-id", baseline_run_id])
    if quality_gate_result is not None:
        cmd.extend(["--quality-gate-result", quality_gate_result])


def _seed_suffix(seed: int, seeds: str | None) -> str:
    return f"seeds{seeds.replace(',', '_')}" if seeds else f"seed{seed}"


def _append_seed_args(cmd: list[str], *, seed: int, seeds: str | None) -> None:
    if seeds:
        cmd.extend(["--seeds", seeds])
    else:
        cmd.extend(["--seed", str(seed)])


def _run_benchmark_command(cmd: list[str], run_id: str) -> dict[str, str]:
    try:
        _run_with_logs(cmd, run_id)
    finally:
        results_volume.commit()
    return {"run_id": run_id}


@app.function(
    timeout=20 * 60,
    volumes={"/results": results_volume},
)
def check_lewm_runtime(run_id: str = "check_lewm_runtime") -> dict[str, str]:
    cmd = [
        "python",
        "tools/benchmarks/check_lewm_runtime.py",
        "--out",
        f"/results/runs/{run_id}/runtime_check.json",
    ]
    try:
        _run_with_logs(cmd, run_id)
    finally:
        results_volume.commit()
    return {"run_id": run_id}


@app.function(
    timeout=20 * 60,
    volumes={"/results": results_volume},
)
def run_unit_tests(
    test_name: str = "tests.test_lewm_cem_exact",
    run_id: str | None = None,
) -> dict[str, str]:
    actual_run_id = run_id or f"unit_{test_name.replace('.', '_')}"
    cmd = ["python", "-m", "unittest", test_name]
    try:
        _run_with_logs(cmd, actual_run_id)
    finally:
        results_volume.commit()
    return {"run_id": actual_run_id}


@app.function(
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def stage_lewm_pusht(force: bool = False, skip_dataset: bool = False) -> dict[str, str]:
    stage_run_id = "stage_lewm_pusht"
    cmd = [
        "python",
        "tools/benchmarks/stage_lewm_weights.py",
        "--weights-root",
        "/weights",
        "--task",
        "pusht",
    ]
    if force:
        cmd.append("--force")
    if skip_dataset:
        cmd.append("--skip-dataset")
    hash_paths = [
        "/weights/lewm/pusht/lewm_object.ckpt",
        "/weights/pusht/lewm_object.ckpt",
        "/weights/lewm/pusht/weights.pt",
        "/weights/lewm/pusht/config.json",
        "/weights/checkpoints/pusht/lewm/weights.pt",
        "/weights/checkpoints/pusht/lewm/config.json",
    ]
    if not skip_dataset:
        hash_paths.append("/weights/datasets/pusht_expert_train.h5")
    stage_succeeded = False
    try:
        _run_with_logs(cmd, stage_run_id)
        stage_succeeded = True
        _run(
            [
                "python",
                "tools/benchmarks/hash_artifacts.py",
                "--paths",
                *hash_paths,
                "--out",
                "/results/artifact_hashes/lewm_pusht.json",
            ]
        )
    finally:
        if stage_succeeded:
            weights_volume.commit()
        results_volume.commit()
    return {"status": "staged", "task": "pusht"}


@app.function(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def run_lewm_benchmark(
    variant: str = "baseline",
    seed: int = 0,
    seeds: str | None = None,
    episodes: int = 50,
    run_id: str | None = None,
    solver: str = "cem",
    profile: str = "cuda-events",
    cem_telemetry: str = "none",
    max_recorded_plans: int | None = None,
    max_recorded_cem_calls: int | None = None,
    torch_profiler_with_stack: bool = False,
    compile_cost: bool | None = None,
    compile_diagnostics: str | None = None,
    compile_warmup_history: int | None = None,
    compile_warmup_sample_counts: str | None = None,
    max_compile_signature_records: int | None = None,
    torch_compile_logs: str | None = None,
    matmul_precision: str = "default",
    model_cost_autocast: str = "none",
    disable_instrumentation: bool = False,
    samples: int | None = None,
    cem_steps: int | None = None,
    topk: int | None = None,
    horizon: int | None = None,
    receding_horizon: int | None = None,
    action_block: int | None = None,
    warm_start: bool | None = None,
    solver_warm_start: bool | None = None,
    elite_retention: int | None = None,
    colored_noise: str | None = None,
    colored_noise_time: str | None = None,
    momentum_alpha: float | None = None,
    sample_decay: float | None = None,
    sample_count_bucket_size: int | None = None,
    return_best_elite: bool | None = None,
    clip_actions: bool | None = None,
    mppi_temperature: float | None = None,
    gd_seed_solver: str | None = None,
    gd_seed_source: str | None = None,
    gd_refine_topk: int | None = None,
    gd_steps: int | None = None,
    gd_lr: float | None = None,
    gd_compile_cost: bool | None = None,
    static_chain_compile: bool | None = None,
    static_chain_mode: str | None = None,
    static_chain_fullgraph: bool | None = None,
    static_chain_warmup: bool | None = None,
    icem_step_telemetry: str | None = None,
    baseline_run_id: str | None = None,
    quality_gate_result: str | None = None,
) -> dict[str, str]:
    actual_run_id = run_id or (
        f"lewm_{variant}_seeds{seeds.replace(',', '_')}_l40s"
        if seeds
        else f"lewm_{variant}_seed{seed}_l40s"
    )
    cmd = [
        "python",
        "tools/benchmarks/lewm_plan_benchmark.py",
        "--variant",
        variant,
        "--episodes",
        str(episodes),
        "--run-id",
        actual_run_id,
        "--weights-root",
        "/weights",
        "--results-root",
        "/results",
        "--solver",
        solver,
        "--profile",
        profile,
        "--cem-telemetry",
        cem_telemetry,
        "--matmul-precision",
        matmul_precision,
        "--model-cost-autocast",
        model_cost_autocast,
    ]
    if seeds:
        cmd.extend(["--seeds", seeds])
    else:
        cmd.extend(["--seed", str(seed)])
    if max_recorded_plans is not None:
        cmd.extend(["--max-recorded-plans", str(max_recorded_plans)])
    if max_recorded_cem_calls is not None:
        cmd.extend(["--max-recorded-cem-calls", str(max_recorded_cem_calls)])
    if torch_profiler_with_stack:
        cmd.append("--torch-profiler-with-stack")
    if compile_cost is not None:
        cmd.extend(["--compile-cost", "true" if compile_cost else "false"])
    if compile_diagnostics is not None:
        cmd.extend(["--compile-diagnostics", compile_diagnostics])
    if compile_warmup_history is not None:
        cmd.extend(["--compile-warmup-history", str(compile_warmup_history)])
    if compile_warmup_sample_counts is not None:
        cmd.extend(["--compile-warmup-sample-counts", compile_warmup_sample_counts])
    if max_compile_signature_records is not None:
        cmd.extend(["--max-compile-signature-records", str(max_compile_signature_records)])
    if torch_compile_logs is not None:
        cmd.extend(["--torch-compile-logs", torch_compile_logs])
    if disable_instrumentation:
        cmd.append("--disable-instrumentation")
    _append_lewm_planner_args(
        cmd,
        samples=samples,
        cem_steps=cem_steps,
        topk=topk,
        horizon=horizon,
        receding_horizon=receding_horizon,
        action_block=action_block,
        warm_start=warm_start,
        solver_warm_start=solver_warm_start,
        elite_retention=elite_retention,
        colored_noise=colored_noise,
        colored_noise_time=colored_noise_time,
        momentum_alpha=momentum_alpha,
        sample_decay=sample_decay,
        sample_count_bucket_size=sample_count_bucket_size,
        return_best_elite=return_best_elite,
        clip_actions=clip_actions,
        mppi_temperature=mppi_temperature,
        gd_seed_solver=gd_seed_solver,
        gd_seed_source=gd_seed_source,
        gd_refine_topk=gd_refine_topk,
        gd_steps=gd_steps,
        gd_lr=gd_lr,
        gd_compile_cost=gd_compile_cost,
        static_chain_compile=static_chain_compile,
        static_chain_mode=static_chain_mode,
        static_chain_fullgraph=static_chain_fullgraph,
        static_chain_warmup=static_chain_warmup,
        icem_step_telemetry=icem_step_telemetry,
        baseline_run_id=baseline_run_id,
        quality_gate_result=quality_gate_result,
    )
    return _run_benchmark_command(cmd, actual_run_id)


@app.function(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def run_lewm_baseline(
    seed: int = 0,
    seeds: str | None = None,
    episodes: int = 2,
    run_id: str | None = None,
) -> dict[str, str]:
    """Run the stock LeWM CEM planner with the timing used for comparisons."""
    actual_run_id = run_id or f"lewm_baseline_{_seed_suffix(seed, seeds)}_l40s"
    cmd = [
        "python",
        "tools/benchmarks/lewm_plan_benchmark.py",
        "--variant",
        "baseline",
        "--episodes",
        str(episodes),
        "--run-id",
        actual_run_id,
        "--weights-root",
        "/weights",
        "--results-root",
        "/results",
        "--solver",
        "cem",
        "--profile",
        "cuda-events",
        "--cem-telemetry",
        "cuda-events",
    ]
    _append_seed_args(cmd, seed=seed, seeds=seeds)
    return _run_benchmark_command(cmd, actual_run_id)


@app.function(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def run_lewm_flash(
    seed: int = 0,
    seeds: str | None = None,
    episodes: int = 2,
    run_id: str | None = None,
    baseline_run_id: str = LEWM_FLASH_QUALITY_BASELINE_RUN_ID,
    quality_gate_result: str = "diagnostic",
) -> dict[str, str]:
    """Run the accepted LeWM-Flash planner preset."""
    actual_run_id = run_id or f"lewm_flash_{_seed_suffix(seed, seeds)}_l40s"
    cmd = [
        "python",
        "tools/benchmarks/lewm_plan_benchmark.py",
        "--variant",
        "icem",
        "--episodes",
        str(episodes),
        "--run-id",
        actual_run_id,
        "--weights-root",
        "/weights",
        "--results-root",
        "/results",
        "--solver",
        "icem",
        "--profile",
        "cuda-events",
        "--cem-telemetry",
        "none",
        "--matmul-precision",
        "high",
        "--model-cost-autocast",
        "bf16",
        "--samples",
        "300",
        "--cem-steps",
        "30",
        "--topk",
        "30",
        "--horizon",
        "5",
        "--receding-horizon",
        "5",
        "--warm-start",
        "true",
        "--elite-retention",
        "5",
        "--colored-noise",
        "none",
        "--colored-noise-time",
        "raw",
        "--momentum-alpha",
        "0.0",
        "--sample-decay",
        "1.25",
        "--sample-count-bucket-size",
        "60",
        "--compile-cost",
        "true",
        "--compile-warmup-sample-counts",
        "auto",
        "--compile-diagnostics",
        "none",
        "--icem-step-telemetry",
        "none",
        "--disable-instrumentation",
        "--baseline-run-id",
        baseline_run_id,
        "--quality-gate-result",
        quality_gate_result,
    ]
    _append_seed_args(cmd, seed=seed, seeds=seeds)
    return _run_benchmark_command(cmd, actual_run_id)


@app.function(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def check_lewm_cache_exactness(
    seed: int = 0,
    candidate_mode: str = "embedding_cache",
    batch_size: int = 1,
    samples: int = 300,
    cem_steps: int = 30,
    horizon: int = 5,
    topk: int = 30,
    run_id: str | None = None,
    control_matmul_precision: str = "default",
    candidate_matmul_precision: str = "high",
) -> dict[str, str]:
    if run_id:
        actual_run_id = run_id
    elif candidate_mode == "embedding_cache":
        actual_run_id = f"lewm_cache_exactness_seed{seed}_l40s"
    else:
        actual_run_id = f"lewm_{candidate_mode}_exactness_seed{seed}_l40s"
    cmd = [
        "python",
        "tools/benchmarks/check_lewm_cache_exactness.py",
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--candidate-mode",
        candidate_mode,
        "--samples",
        str(samples),
        "--cem-steps",
        str(cem_steps),
        "--horizon",
        str(horizon),
        "--topk",
        str(topk),
        "--control-matmul-precision",
        control_matmul_precision,
        "--candidate-matmul-precision",
        candidate_matmul_precision,
        "--weights-root",
        "/weights",
        "--out",
        f"/results/runs/{actual_run_id}/exactness.json",
    ]
    try:
        _run_with_logs(cmd, actual_run_id)
    finally:
        results_volume.commit()
    return {"run_id": actual_run_id}


@app.function(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def run_lewm_profile(
    variant: str = "baseline",
    seed: int = 0,
    episodes: int = 50,
    plans: int = 100,
    run_id: str | None = None,
    solver: str = "cem",
    cem_telemetry: str = "cuda-events",
    torch_profiler_with_stack: bool = False,
    compile_cost: bool | None = None,
    compile_diagnostics: str | None = None,
    compile_warmup_history: int | None = None,
    compile_warmup_sample_counts: str | None = None,
    max_compile_signature_records: int | None = None,
    torch_compile_logs: str | None = None,
) -> dict[str, str]:
    actual_run_id = run_id or f"lewm_profile_{variant}_seed{seed}_l40s"
    cmd = [
        "python",
        "tools/benchmarks/profile_lewm_cem.py",
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--plans",
        str(plans),
        "--run-id",
        actual_run_id,
        "--weights-root",
        "/weights",
        "--results-root",
        "/results",
        "--solver",
        solver,
        "--cem-telemetry",
        cem_telemetry,
    ]
    if torch_profiler_with_stack:
        cmd.append("--torch-profiler-with-stack")
    if compile_cost is not None:
        cmd.extend(["--compile-cost", "true" if compile_cost else "false"])
    if compile_diagnostics is not None:
        cmd.extend(["--compile-diagnostics", compile_diagnostics])
    if compile_warmup_history is not None:
        cmd.extend(["--compile-warmup-history", str(compile_warmup_history)])
    if compile_warmup_sample_counts is not None:
        cmd.extend(["--compile-warmup-sample-counts", compile_warmup_sample_counts])
    if max_compile_signature_records is not None:
        cmd.extend(["--max-compile-signature-records", str(max_compile_signature_records)])
    if torch_compile_logs is not None:
        cmd.extend(["--torch-compile-logs", torch_compile_logs])
    try:
        _run_with_logs(cmd, actual_run_id)
    finally:
        results_volume.commit()
    return {"run_id": actual_run_id}


@app.function(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
def run_lewm_model_cost_profile(
    variant: str = "compile_cost",
    seed: int = 0,
    run_id: str | None = None,
    policy: str = "pusht/lewm",
    solver: str = "cem_exact",
    profile: str = "torch",
    batch_size: int = 1,
    samples: int = 300,
    horizon: int = 5,
    history: int = 1,
    warmup: int = 5,
    iters: int = 50,
    matmul_precision: str = "default",
    compile_cost_mode: str = "reduce-overhead",
    compile_cost_fullgraph: bool = False,
    compile_cost_allow_fallback: bool = False,
    model_cost_region_profile: bool = False,
    torch_profiler_with_stack: bool = False,
    torch_compile_logs: str | None = None,
) -> dict[str, str]:
    actual_run_id = run_id or f"lewm_model_cost_{variant}_seed{seed}_l40s"
    cmd = [
        "python",
        "tools/benchmarks/profile_lewm_model_cost.py",
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--run-id",
        actual_run_id,
        "--policy",
        policy,
        "--solver",
        solver,
        "--weights-root",
        "/weights",
        "--results-root",
        "/results",
        "--profile",
        profile,
        "--batch-size",
        str(batch_size),
        "--samples",
        str(samples),
        "--horizon",
        str(horizon),
        "--history",
        str(history),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--matmul-precision",
        matmul_precision,
        "--compile-cost-mode",
        compile_cost_mode,
        "--compile-cost-fullgraph",
        "true" if compile_cost_fullgraph else "false",
    ]
    if compile_cost_allow_fallback:
        cmd.append("--compile-cost-allow-fallback")
    if model_cost_region_profile:
        cmd.append("--model-cost-region-profile")
    if torch_profiler_with_stack:
        cmd.append("--torch-profiler-with-stack")
    if torch_compile_logs is not None:
        cmd.extend(["--torch-compile-logs", torch_compile_logs])
    try:
        _run_with_logs(cmd, actual_run_id)
    finally:
        results_volume.commit()
    return {"run_id": actual_run_id}


@app.cls(
    gpu=GPU_TYPE,
    cpu=BENCHMARK_CPU,
    timeout=60 * 60,
    volumes={"/weights": weights_volume, "/results": results_volume},
)
class LeWMBenchmarkRunner:
    @modal.enter()
    def setup(self) -> None:
        os.environ.update(_subprocess_env())

    @modal.method()
    def run(
        self,
        variant: str = "baseline",
        seed: int = 0,
        episodes: int = 50,
        run_id: str | None = None,
        solver: str = "cem",
        profile: str = "cuda-events",
        cem_telemetry: str = "none",
        max_recorded_plans: int | None = None,
        max_recorded_cem_calls: int | None = None,
        torch_profiler_with_stack: bool = False,
        compile_cost: bool | None = None,
        compile_diagnostics: str | None = None,
        compile_warmup_history: int | None = None,
        compile_warmup_sample_counts: str | None = None,
        max_compile_signature_records: int | None = None,
        torch_compile_logs: str | None = None,
        matmul_precision: str = "default",
        model_cost_autocast: str = "none",
        disable_instrumentation: bool = False,
        samples: int | None = None,
        cem_steps: int | None = None,
        topk: int | None = None,
        horizon: int | None = None,
        receding_horizon: int | None = None,
        action_block: int | None = None,
        warm_start: bool | None = None,
        solver_warm_start: bool | None = None,
        elite_retention: int | None = None,
        colored_noise: str | None = None,
        colored_noise_time: str | None = None,
        momentum_alpha: float | None = None,
        sample_decay: float | None = None,
        sample_count_bucket_size: int | None = None,
        return_best_elite: bool | None = None,
        clip_actions: bool | None = None,
        mppi_temperature: float | None = None,
        gd_seed_solver: str | None = None,
        gd_seed_source: str | None = None,
        gd_refine_topk: int | None = None,
        gd_steps: int | None = None,
        gd_lr: float | None = None,
        gd_compile_cost: bool | None = None,
        static_chain_compile: bool | None = None,
        static_chain_mode: str | None = None,
        static_chain_fullgraph: bool | None = None,
        static_chain_warmup: bool | None = None,
        icem_step_telemetry: str | None = None,
        baseline_run_id: str | None = None,
        quality_gate_result: str | None = None,
    ) -> dict[str, str]:
        actual_run_id = run_id or f"lewm_{variant}_seed{seed}_l40s"
        cmd = [
            "python",
            "tools/benchmarks/lewm_plan_benchmark.py",
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--episodes",
            str(episodes),
            "--run-id",
            actual_run_id,
            "--weights-root",
            "/weights",
            "--results-root",
            "/results",
            "--solver",
            solver,
            "--profile",
            profile,
            "--cem-telemetry",
            cem_telemetry,
            "--matmul-precision",
            matmul_precision,
            "--model-cost-autocast",
            model_cost_autocast,
        ]
        if max_recorded_plans is not None:
            cmd.extend(["--max-recorded-plans", str(max_recorded_plans)])
        if max_recorded_cem_calls is not None:
            cmd.extend(["--max-recorded-cem-calls", str(max_recorded_cem_calls)])
        if torch_profiler_with_stack:
            cmd.append("--torch-profiler-with-stack")
        if compile_cost is not None:
            cmd.extend(["--compile-cost", "true" if compile_cost else "false"])
        if compile_diagnostics is not None:
            cmd.extend(["--compile-diagnostics", compile_diagnostics])
        if compile_warmup_history is not None:
            cmd.extend(["--compile-warmup-history", str(compile_warmup_history)])
        if compile_warmup_sample_counts is not None:
            cmd.extend(["--compile-warmup-sample-counts", compile_warmup_sample_counts])
        if max_compile_signature_records is not None:
            cmd.extend(["--max-compile-signature-records", str(max_compile_signature_records)])
        if torch_compile_logs is not None:
            cmd.extend(["--torch-compile-logs", torch_compile_logs])
        if disable_instrumentation:
            cmd.append("--disable-instrumentation")
        _append_lewm_planner_args(
            cmd,
            samples=samples,
            cem_steps=cem_steps,
            topk=topk,
            horizon=horizon,
            receding_horizon=receding_horizon,
            action_block=action_block,
            warm_start=warm_start,
            solver_warm_start=solver_warm_start,
            elite_retention=elite_retention,
            colored_noise=colored_noise,
            colored_noise_time=colored_noise_time,
            momentum_alpha=momentum_alpha,
            sample_decay=sample_decay,
            sample_count_bucket_size=sample_count_bucket_size,
            return_best_elite=return_best_elite,
            clip_actions=clip_actions,
            mppi_temperature=mppi_temperature,
            gd_seed_solver=gd_seed_solver,
            gd_seed_source=gd_seed_source,
            gd_refine_topk=gd_refine_topk,
            gd_steps=gd_steps,
            gd_lr=gd_lr,
            gd_compile_cost=gd_compile_cost,
            static_chain_compile=static_chain_compile,
            static_chain_mode=static_chain_mode,
            static_chain_fullgraph=static_chain_fullgraph,
            static_chain_warmup=static_chain_warmup,
            icem_step_telemetry=icem_step_telemetry,
            baseline_run_id=baseline_run_id,
            quality_gate_result=quality_gate_result,
        )
        try:
            _run_with_logs(cmd, actual_run_id)
        finally:
            results_volume.commit()
        return {"run_id": actual_run_id}
