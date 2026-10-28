"""Torch baselines on the same GPU/container as the megakernel.

Times (a) ``get_cost_from_embeddings`` per iCEM bucket and (b) one full single-env iCEM solve, for the eager FP32
path and for the accepted LeWM-Flash kernel/precision stack (torch.compile reduce-overhead + TF32 + BF16 autocast).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ICEM = dict(num_samples=300, n_steps=30, topk=30, var_scale=1.0, elite_retention=5, sample_decay=1.25,
            sample_count_bucket_size=60, colored_noise="none", colored_noise_time="raw", momentum_alpha=0.0)
HORIZON, ACTION_BLOCK, RAW_ACTION_DIM, EMB = 5, 5, 2, 192


class _Box:  # minimal Box-like action space (the solver only reads .shape and the class name)
    def __init__(self, shape):
        self.shape = shape


_Box.__name__ = "Box"


class _PlanCfg:
    horizon = HORIZON
    receding_horizon = HORIZON
    action_block = ACTION_BLOCK


def make_solver(model, seed: int = 0):
    sys.path.insert(0, str(REPO_ROOT))
    from lewm.solvers.icem import ICEMSolver

    solver = ICEMSolver(model, batch_size=1, device="cuda", seed=seed, **ICEM)
    solver.configure(action_space=_Box((1, RAW_ACTION_DIM)), n_envs=1, config=_PlanCfg())
    return solver


def info_for(emb, goal_emb):
    import torch

    return {
        "pixels": torch.zeros(1, 1, 1, device="cuda"),  # only .size(2) (= history 1) is read on the cached path
        "goal": torch.zeros(1, 1, 1, device="cuda"),
        "emb": emb,            # (1, 1, 192) cached observation embedding
        "goal_emb": goal_emb,  # (1, 1, 192) cached goal embedding
    }


def cuda_time(fn, iters: int, warmup: int = 3) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def summarize(ms: list[float]) -> dict:
    s = sorted(ms)
    return {"p50_ms": statistics.median(s), "mean_ms": statistics.fmean(s), "min_ms": s[0],
            "p90_ms": s[int(0.9 * (len(s) - 1))], "n": len(s)}


def run(weights_dir: str, out_dir: str, solve_iters: int = 30) -> dict:
    import contextlib
    import io

    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from megakernel.model_io import load_cost_model

    torch.manual_seed(0)
    results: dict = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    emb = torch.randn(1, 1, EMB, device="cuda")
    goal = torch.randn(1, 1, EMB, device="cuda")
    counts = [300, 240, 180, 120, 60]

    for mode in ("eager_fp32", "flash_compile_bf16"):
        model, _ = load_cost_model(weights_dir)
        model.set_embedding_cache(True)
        if mode == "flash_compile_bf16":
            torch.set_float32_matmul_precision("high")
            model.set_model_cost_autocast("bf16")
            model.get_cost_from_embeddings = torch.compile(model.get_cost_from_embeddings, mode="reduce-overhead",
                                                           fullgraph=False)
        else:
            torch.set_float32_matmul_precision("highest")
        model.set_compile_cost(True)
        per_count = {}
        with torch.inference_mode():
            for s in counts:
                acts = torch.randn(1, s, HORIZON, RAW_ACTION_DIM * ACTION_BLOCK, device="cuda")
                per_count[s] = summarize(cuda_time(lambda: model.get_cost_from_embeddings(emb, goal, acts), 50, 5))
        solver = make_solver(model)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):  # the solver prints its wall time every call
            solve = summarize(cuda_time(lambda: solver.solve(info_for(emb, goal)), solve_iters, 3))
        results[mode] = {"get_cost_ms_by_samples": per_count, "solve_ms": solve,
                         "model_cost_evals_per_solve": solver.model_cost_evals_per_solve()}
        print(mode, json.dumps(results[mode], indent=1), flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "baseline.json").write_text(json.dumps(results, indent=1))
    return results
