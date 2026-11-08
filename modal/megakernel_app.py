"""Modal entrypoints for the LeWM ThunderKittens megakernel track (H100).

    uv run modal run modal/megakernel_app.py::baseline
    uv run modal run modal/megakernel_app.py::build
    uv run modal run modal/megakernel_app.py::check --name parity

`build` compiles the CUDA extension on a CPU container into the results volume (cheap); GPU entrypoints load the
prebuilt .so, so iteration on kernel code does not pay nvcc time on an H100.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "lewm-flash-megakernel"
GPU = "H100"
TK_COMMIT = "67845f5f"  # HazyResearch/ThunderKittens main, 2026-10 (same pin as the π0.5 GEMM calibration)
WORKDIR = "/workspace"

weights_volume = modal.Volume.from_name("jepa-weights", create_if_missing=True)
mk_volume = modal.Volume.from_name("lewm-mk-results", create_if_missing=True)

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.11.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "einops", "ninja", "pybind11")
    .run_commands(
        "git clone https://github.com/HazyResearch/ThunderKittens.git /tk",
        f"cd /tk && git checkout {TK_COMMIT}",
    )
    .env({"THUNDERKITTENS_ROOT": "/tk", "TORCH_CUDA_ARCH_LIST": "9.0a"})
)
image = (
    base_image.add_local_dir("megakernel", remote_path=f"{WORKDIR}/megakernel", ignore=["**/__pycache__/**"])
    .add_local_dir("lewm", remote_path=f"{WORKDIR}/lewm", ignore=["**/__pycache__/**"])
)

# End-to-end Push-T benchmark image: the megakernel image (same torch build the extension links against) plus the
# stable-worldmodel / benchmark stack used by modal/benchmark_app.py.
e2e_image = (
    base_image.apt_install("curl", "zstd", "swig", "libgl1", "libglib2.0-0", "libegl1")
    .pip_install("stable-worldmodel[train,env]", "scipy", "scikit-learn", "pillow", "h5py", "hdf5plugin",
                 "datasets==2.19.2", "pyarrow<21", "transformers<5", "hydra-core", "omegaconf", "huggingface_hub",
                 "hf_xet", "pytest")
    .env({"PYTHONPATH": WORKDIR, "LEWM_MK_BUILD": "/mk/build"})
    .add_local_dir(".", remote_path=WORKDIR, ignore=[".git/**", ".venv/**", "**/__pycache__/**", "**/*.pyc",
                                                      "results/**", "**/*.png", "**/*.mp4"])
)
results_volume = modal.Volume.from_name("jepa-results", create_if_missing=True)

app = modal.App(APP_NAME, image=image)
WEIGHTS = "/weights/lewm/pusht"


def _setup() -> None:
    sys.path.insert(0, WORKDIR)
    os.chdir(WORKDIR)


@app.function(gpu=GPU, timeout=30 * 60, volumes={"/weights": weights_volume, "/mk": mk_volume})
def baseline_remote(run: str) -> dict:
    _setup()
    from megakernel import baseline

    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm", "--format=csv"],
                                  text=True))
    out = baseline.run(WEIGHTS, f"/mk/{run}")
    mk_volume.commit()
    return out


@app.function(cpu=16, memory=32768, timeout=40 * 60, volumes={"/mk": mk_volume})
def build_remote(prof: bool = False) -> str:
    _setup()
    from megakernel import build

    path = build.build("/mk/build", prof=prof)
    mk_volume.commit()
    return path


@app.function(gpu=GPU, timeout=30 * 60, volumes={"/weights": weights_volume, "/mk": mk_volume})
def check_remote(name: str, run: str, args: str = "") -> dict:
    _setup()
    import importlib

    mk_volume.reload()
    mod = importlib.import_module(f"megakernel.{name}")
    out = mod.run(WEIGHTS, f"/mk/{run}", "/mk/build", args)
    mk_volume.commit()
    return out


COMMON_PLANNER = ["--samples", "300", "--cem-steps", "30", "--topk", "30", "--horizon", "5", "--receding-horizon", "5",
                  "--warm-start", "true", "--elite-retention", "5", "--colored-noise", "none", "--colored-noise-time",
                  "raw", "--momentum-alpha", "0.0", "--sample-decay", "1.25", "--sample-count-bucket-size", "60",
                  "--profile", "cuda-events", "--cem-telemetry", "none", "--compile-diagnostics", "none",
                  "--icem-step-telemetry", "none", "--disable-instrumentation", "--quality-gate-result", "diagnostic"]
ICEM_PLANNER = ["--samples", "300", "--cem-steps", "30", "--topk", "30", "--horizon", "5", "--receding-horizon", "5",
                "--warm-start", "true", "--elite-retention", "5", "--colored-noise", "none", "--colored-noise-time", "raw",
                "--momentum-alpha", "0.0", "--sample-decay", "1.25", "--sample-count-bucket-size", "60"]
CLEAN = ["--profile", "cuda-events", "--cem-telemetry", "none", "--compile-diagnostics", "none",
         "--icem-step-telemetry", "none", "--disable-instrumentation", "--quality-gate-result", "diagnostic"]
# The H100 optimization ladder: each rung adds one change to the previous one (see docs/ladder.md).
LADDER = {
    "r0": ["--variant", "baseline", "--solver", "cem", *CLEAN],                       # stock LeWM CEM, eager FP32
    "r1": ["--variant", "cem_exact_no_cpu_sync", "--solver", "cem_exact", *CLEAN],    # + cached embeddings, no CPU sync
    "r2": ["--variant", "icem", "--solver", "icem", "--matmul-precision", "highest", "--model-cost-autocast", "none",
           "--compile-cost", "false", *ICEM_PLANNER, *CLEAN],                            # + iCEM (eager FP32)
    "r2b": ["--variant", "icem", "--solver", "icem", "--matmul-precision", "high", "--model-cost-autocast", "bf16",
            "--compile-cost", "true", "--compile-warmup-sample-counts", "auto", *ICEM_PLANNER, *CLEAN],  # LeWM-Flash
    "r3": ["--variant", "icem_mk", "--solver", "icem_mk", "--matmul-precision", "highest", "--model-cost-autocast",
           "none", "--compile-cost", "false", *ICEM_PLANNER, *CLEAN],                    # + megakernel
    "r4": ["--variant", "icem_mk_fused", "--solver", "icem_mk_fused", "--matmul-precision", "highest",
           "--model-cost-autocast", "none", "--compile-cost", "false", *ICEM_PLANNER, *CLEAN],  # + fused encoder
    "r3cem": ["--variant", "icem_mk", "--solver", "icem_mk", "--matmul-precision", "highest", "--model-cost-autocast",
              "none", "--compile-cost", "false", "--samples", "300", "--cem-steps", "30", "--topk", "30",
              "--horizon", "5", "--receding-horizon", "5", "--warm-start", "true", "--elite-retention", "0",
              "--colored-noise", "none", "--colored-noise-time", "raw", "--momentum-alpha", "0.0",
              "--sample-decay", "1.0", "--sample-count-bucket-size", "0", *CLEAN],     # megakernel running plain CEM
}
VARIANTS = {
    # no-opt reference: stock LeWM / stable-worldmodel CEM, exactly the flags of benchmark_app.run_lewm_baseline
    "baseline": ["--variant", "baseline", "--solver", "cem", "--profile", "cuda-events", "--cem-telemetry", "cuda-events"],
    # accepted LeWM-Flash: torch.compile cost + TF32 + BF16 autocast, iCEM envs solved one after another
    "flash": ["--variant", "icem", "--solver", "icem", "--matmul-precision", "high", "--model-cost-autocast", "bf16",
              "--compile-cost", "true", "--compile-warmup-sample-counts", "auto"],
    # LeWM-MK: same planner, every replanning env's whole iCEM solve in one megakernel launch
    "mk": ["--variant", "icem_mk", "--solver", "icem_mk", "--matmul-precision", "highest", "--model-cost-autocast",
           "none", "--compile-cost", "false"],
}


@app.function(gpu=GPU, cpu=8.0, timeout=60 * 60, image=e2e_image,
              volumes={"/weights": weights_volume, "/results": results_volume, "/mk": mk_volume})
def e2e_remote(variant: str, seeds: str, episodes: int, run_id: str, batch1: bool = False) -> str:
    env = os.environ.copy()
    env.update({"STABLEWM_HOME": "/weights", "HF_HOME": "/weights/cache/huggingface",
                "HUGGINGFACE_HUB_CACHE": "/weights/cache/huggingface/hub", "TORCH_HOME": "/weights/cache/torch",
                "MUJOCO_GL": "egl", "JEPA_MODAL_GPU_TYPE": GPU, "JEPA_MODAL_CPU_REQUEST": "8.0",
                "JEPA_MODAL_CPU_LIMIT": "8.0"})
    if batch1:
        env["LEWM_MK_MAX_ENVS"] = "1"
        env["LEWM_MK_TIMING"] = "1"
    flags = LADDER[variant] if variant in LADDER else [*VARIANTS[variant], *([] if variant == "baseline" else COMMON_PLANNER)]
    cmd = ["python", "tools/benchmarks/lewm_plan_benchmark.py", "--episodes", str(episodes), "--run-id", run_id,
           "--weights-root", "/weights", "--results-root", "/results", "--seeds", seeds, *flags]
    if batch1:
        cmd += ["--warmup-plans", "0"]  # every plan is recorded; the analysis drops the first (warmup) seed
    run_dir = Path("/results/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(run_dir / "stdout.txt", "w") as out, open(run_dir / "stderr.txt", "w") as err:
            rc = subprocess.run(cmd, cwd=WORKDIR, env=env, stdout=out, stderr=err).returncode
    finally:
        results_volume.commit()
    tail = (run_dir / "stderr.txt").read_text(errors="replace").splitlines()[-40:]
    if rc != 0:
        raise RuntimeError(f"benchmark failed rc={rc}:\n" + "\n".join(tail))
    return run_id


@app.function(gpu=GPU, timeout=30 * 60, image=e2e_image,
              volumes={"/weights": weights_volume, "/mk": mk_volume})
def check_e2e_remote(name: str, run: str, args: str = "") -> dict:
    """Checks that need the full pickled JEPA / dataset (stable-worldmodel stack)."""
    import importlib

    sys.path.insert(0, WORKDIR)
    os.chdir(WORKDIR)
    mk_volume.reload()
    mod = importlib.import_module(f"megakernel.{name}")
    out = mod.run("/weights/lewm/pusht", f"/mk/{run}", "/mk/build", args)
    mk_volume.commit()
    return out


@app.local_entrypoint()
def check_e2e(name: str = "enc_parity", args: str = "") -> None:
    import json

    print(json.dumps(check_e2e_remote.remote(name, f"{name}-h100-{_stamp()}", args), indent=1, default=str)[:4000])


@app.local_entrypoint()
def ladder(rungs: str = "r0,r1,r2,r2b,r3,r3cem", mode: str = "batch1", n_seeds: int = 41) -> None:
    """mode=batch1: one env per episode over n_seeds seeds (the first is warmup) -> end-to-end batch-1 plan latency.
    mode=quality: 3 seeds x 100 episodes -> Push-T success."""
    stamp = _stamp()
    batch1 = mode == "batch1"
    seeds = ",".join(str(1000 + i) for i in range(n_seeds)) if batch1 else "0,1,2"
    episodes = 1 if batch1 else 100
    calls = []
    for r in rungs.split(","):
        run_id = f"ladder_{r}_{mode}_h100_{stamp}"
        calls.append(e2e_remote.spawn(r, seeds, episodes, run_id, batch1))
    for c in calls:
        print(c.get())


@app.local_entrypoint()
def e2e(variants: str = "flash,mk", seeds: str = "0,1,2", episodes: int = 100) -> None:
    stamp = _stamp()
    calls = []
    for v in variants.split(","):
        run_id = f"lewm_{v}_h100_e{episodes}_s{seeds.replace(',', '-')}_{stamp}"
        calls.append(e2e_remote.spawn(v, seeds, episodes, run_id))
    for c in calls:
        print(c.get())


def _stamp() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dt%H%M%Sz")


@app.local_entrypoint()
def baseline() -> None:
    print(baseline_remote.remote(f"baseline-h100-{_stamp()}"))


@app.local_entrypoint()
def build(prof: bool = False) -> None:
    print(build_remote.remote(prof))


@app.local_entrypoint()
def check(name: str = "parity", args: str = "") -> None:
    import json

    print(json.dumps(check_remote.remote(name, f"{name}-h100-{_stamp()}", args), indent=1))
