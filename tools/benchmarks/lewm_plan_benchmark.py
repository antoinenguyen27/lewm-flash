"""Run LeWM Push-T planning benchmarks with manifest-backed telemetry."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.hash_artifacts import sha256_file
from tools.benchmarks.metrics import (
    append_jsonl,
    json_safe,
    read_jsonl,
    summarize_cem_calls,
    summarize_evaluations,
    summarize_plan_latencies,
)
from tools.benchmarks.run_manifest import (
    build_manifest,
    default_run_id,
    discover_runtime_resources,
    write_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_DIR = REPO_ROOT / "lewm"


SEMANTIC_CLASSES = {
    "baseline": "baseline",
    "cache_embeddings": "exact",
    "cem_exact": "exact",
    "cem_exact_no_cpu_sync_only": "exact",
    "cem_exact_no_cpu_sync": "exact",
    "rollout_prealloc": "exact",
    "rollout_action_cache": "exact",
    "compile_cost": "numerical",
    "model_cost_tf32": "numerical",
    "fused_cem_glue": "numerical",
    "warm_start_cem": "planner",
    "icem": "planner",
    "icem_static_compile": "planner",
    "mppi": "planner",
    "cem_gd": "planner",
    "icem_gd": "planner",
}


CODE_PATHS = {
    "baseline": [],
    "cache_embeddings": ["lewm/jepa.py"],
    "cem_exact": ["lewm/jepa.py", "lewm/solvers/cem_exact.py"],
    "cem_exact_no_cpu_sync_only": ["lewm/solvers/cem_exact.py"],
    "cem_exact_no_cpu_sync": ["lewm/jepa.py", "lewm/solvers/cem_exact.py"],
    "rollout_prealloc": ["lewm/jepa.py"],
    "rollout_action_cache": ["lewm/jepa.py"],
    "compile_cost": ["lewm/jepa.py"],
    "model_cost_tf32": ["lewm/jepa.py", "tools/benchmarks/lewm_plan_benchmark.py"],
    "fused_cem_glue": ["lewm/jepa.py", "lewm/solvers/cem_exact.py"],
    "warm_start_cem": ["lewm/jepa.py", "lewm/solvers/cem_exact.py"],
    "icem": ["lewm/jepa.py", "lewm/solvers/icem.py"],
    "icem_static_compile": [
        "lewm/jepa.py",
        "lewm/solvers/icem_static_compile.py",
    ],
    "mppi": ["lewm/jepa.py", "lewm/solvers/mppi.py"],
    "cem_gd": ["lewm/jepa.py", "lewm/solvers/cem_gd.py"],
    "icem_gd": ["lewm/jepa.py", "lewm/solvers/icem.py", "lewm/solvers/cem_gd.py"],
}


EMBEDDING_CACHE_VARIANTS = {
    "cache_embeddings",
    "cem_exact",
    "cem_exact_no_cpu_sync",
    "rollout_prealloc",
    "rollout_action_cache",
    "compile_cost",
    "model_cost_tf32",
    "fused_cem_glue",
    "warm_start_cem",
    "icem",
    "icem_static_compile",
    "mppi",
    "cem_gd",
    "icem_gd",
}


ROLLOUT_PREALLOC_VARIANTS = {
    "rollout_prealloc",
}


ROLLOUT_ACTION_CACHE_VARIANTS = {
    "rollout_action_cache",
}


RETIRED_VARIANTS = {
    "tensor_cem_compile",
    "tensor_cem_compile_no_inner_compile",
    "tensor_cem_static",
}


COMPILE_COST_VARIANTS = {
    "compile_cost",
    "model_cost_tf32",
}


MATMUL_PRECISION_CHOICES = (
    "default",
    "highest",
    "high",
    "medium",
)


MODEL_COST_AUTOCAST_CHOICES = (
    "none",
    "bf16",
)


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected true or false, got {value!r}")


def planner_sample_counts(
    *,
    samples: Any,
    cem_steps: Any,
    topk: Any = None,
    sample_decay: Any = None,
    sample_count_bucket_size: Any = None,
) -> list[int] | None:
    if samples is None or cem_steps is None:
        return None
    samples_int = int(samples)
    steps_int = int(cem_steps)
    decay = 1.0 if sample_decay is None else float(sample_decay)
    bucket_size = 0 if sample_count_bucket_size is None else int(sample_count_bucket_size)
    if decay == 1.0:
        return [samples_int for _ in range(steps_int)]
    topk_int = int(topk) if topk is not None else 1
    counts = []
    for step in range(steps_int):
        count = max(
            int(math.ceil(samples_int * (decay ** (-step)))),
            min(samples_int, 2 * topk_int),
        )
        if bucket_size > 0:
            count = min(samples_int, int(math.ceil(count / bucket_size)) * bucket_size)
        counts.append(count)
    return counts


def planner_model_cost_evals_per_call(
    *,
    samples: Any,
    cem_steps: Any,
    topk: Any = None,
    sample_decay: Any = None,
    sample_count_bucket_size: Any = None,
) -> int | None:
    counts = planner_sample_counts(
        samples=samples,
        cem_steps=cem_steps,
        topk=topk,
        sample_decay=sample_decay,
        sample_count_bucket_size=sample_count_bucket_size,
    )
    if counts is None:
        return None
    return sum(counts)


def unique_preserving_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def resolve_compile_warmup_sample_counts(value: str | None, cfg: Any) -> list[int]:
    default_samples = int(getattr(cfg.solver, "num_samples"))
    if value is None:
        return [default_samples]
    normalized = value.strip().lower()
    if normalized == "auto":
        counts = planner_sample_counts(
            samples=getattr(cfg.solver, "num_samples", None),
            cem_steps=getattr(cfg.solver, "n_steps", None),
            topk=getattr(cfg.solver, "topk", None),
            sample_decay=getattr(cfg.solver, "sample_decay", None),
            sample_count_bucket_size=getattr(cfg.solver, "sample_count_bucket_size", None),
        )
        if counts is None:
            return [default_samples]
        return unique_preserving_order(counts)

    counts = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        count = int(raw)
        if count <= 0:
            raise ValueError(f"compile warmup sample counts must be positive, got {count}")
        counts.append(count)
    if not counts:
        raise ValueError("compile warmup sample counts must include at least one value")
    return unique_preserving_order(counts)


def effective_compile_diagnostics(args: argparse.Namespace) -> str:
    requested = getattr(args, "compile_diagnostics", None)
    if requested is not None:
        return requested
    return "summary" if compile_cost_requested(args) else "none"


def compile_cost_requested(args: argparse.Namespace) -> bool:
    requested = parse_optional_bool(args.compile_cost)
    if requested is not None:
        return requested
    if args.variant in COMPILE_COST_VARIANTS:
        return True
    return False


def validate_variant_solver_contract(args: argparse.Namespace) -> None:
    if args.variant in RETIRED_VARIANTS:
        raise ValueError(
            f"variant={args.variant} has been retired after the L5 closeout. "
            "Use variant=compile_cost as the accepted L1+L2+L4 control, or "
            "reintroduce this line as a new branch with a fresh implementation."
        )
    if args.variant in COMPILE_COST_VARIANTS and args.solver != "cem_exact":
        raise ValueError(
            f"variant={args.variant} requires "
            "--solver cem_exact. Use an explicit different variant plus "
            "the matching compile flags for diagnostic non-ladder runs."
        )
    if args.variant == "icem" and args.solver != "icem":
        raise ValueError("variant=icem requires --solver icem")
    if args.variant == "icem_static_compile" and args.solver != "icem_static_compile":
        raise ValueError("variant=icem_static_compile requires --solver icem_static_compile")
    if args.variant == "cem_gd" and args.solver != "cem_gd":
        raise ValueError("variant=cem_gd requires --solver cem_gd")
    if args.variant == "icem_gd" and args.solver != "cem_gd":
        raise ValueError("variant=icem_gd requires --solver cem_gd")
    if args.variant == "icem_gd" and getattr(args, "gd_seed_solver", None) != "icem":
        raise ValueError("variant=icem_gd requires --gd-seed-solver icem")
    if args.variant == "cem_gd" and getattr(args, "gd_seed_solver", None) == "icem":
        raise ValueError("Use variant=icem_gd with --gd-seed-solver icem")
    if getattr(args, "icem_step_telemetry", "none") != "none" and args.solver != "icem":
        raise ValueError("--icem-step-telemetry requires --solver icem")
    if (
        effective_compile_diagnostics(args) != "none"
        and getattr(args, "cem_telemetry", "none") == "none"
    ):
        raise ValueError(
            "compile diagnostics require --cem-telemetry summary or "
            "--cem-telemetry cuda-events so runtime cost signatures can be recorded."
        )


def planner_hydra_overrides(args: argparse.Namespace) -> list[str]:
    """Hydra overrides for planner fields that exist in the current configs."""
    overrides: list[str] = []
    mapping = (
        ("samples", "solver.num_samples"),
        ("cem_steps", "solver.n_steps"),
        ("topk", "solver.topk"),
        ("horizon", "plan_config.horizon"),
        ("receding_horizon", "plan_config.receding_horizon"),
        ("action_block", "plan_config.action_block"),
    )
    for attr, hydra_key in mapping:
        value = getattr(args, attr, None)
        if value is not None:
            overrides.append(f"{hydra_key}={value}")
    warm_start = getattr(args, "warm_start", None)
    if warm_start is not None:
        overrides.append(f"plan_config.warm_start={warm_start}")
        if getattr(args, "solver", None) == "cem_exact":
            overrides.append(f"solver.warm_start={warm_start}")
    solver_warm_start = getattr(args, "solver_warm_start", None)
    if solver_warm_start is not None and getattr(args, "solver", None) == "cem_exact":
        overrides.append(f"solver.warm_start={solver_warm_start}")
    if getattr(args, "solver", None) in {"icem", "icem_static_compile", "cem_gd"}:
        icem_mapping = (
            ("elite_retention", "solver.elite_retention"),
            ("colored_noise", "solver.colored_noise"),
            ("colored_noise_time", "solver.colored_noise_time"),
            ("momentum_alpha", "solver.momentum_alpha"),
            ("sample_decay", "solver.sample_decay"),
            ("sample_count_bucket_size", "solver.sample_count_bucket_size"),
            ("return_best_elite", "solver.return_best_elite"),
            ("clip_actions", "solver.clip_actions"),
        )
        for attr, hydra_key in icem_mapping:
            value = getattr(args, attr, None)
            if value is not None:
                overrides.append(f"{hydra_key}={value}")
    if getattr(args, "solver", None) == "icem_static_compile":
        static_mapping = (
            ("static_chain_compile", "solver.static_chain_compile"),
            ("static_chain_mode", "solver.static_chain_mode"),
            ("static_chain_fullgraph", "solver.static_chain_fullgraph"),
            ("static_chain_warmup", "solver.static_chain_warmup"),
        )
        for attr, hydra_key in static_mapping:
            value = getattr(args, attr, None)
            if value is not None:
                overrides.append(f"{hydra_key}={value}")
    if getattr(args, "solver", None) == "cem_gd":
        gd_mapping = (
            ("gd_seed_solver", "solver.gd_seed_solver"),
            ("gd_seed_source", "solver.gd_seed_source"),
            ("gd_refine_topk", "solver.gd_refine_topk"),
            ("gd_steps", "solver.gd_steps"),
            ("gd_lr", "solver.gd_lr"),
            ("gd_compile_cost", "solver.gd_compile_cost"),
        )
        for attr, hydra_key in gd_mapping:
            value = getattr(args, attr, None)
            if value is not None:
                overrides.append(f"{hydra_key}={value}")
    return overrides


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def effective_planner_settings(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    """Planner config resolved from Hydra plus planner-only CLI metadata."""
    solver = _cfg_get(cfg, "solver", {})
    plan_config = _cfg_get(cfg, "plan_config", {})
    samples = _cfg_get(solver, "num_samples")
    cem_steps = _cfg_get(solver, "n_steps")
    topk = _cfg_get(solver, "topk")
    horizon = _cfg_get(plan_config, "horizon")
    receding_horizon = _cfg_get(plan_config, "receding_horizon")
    action_block = _cfg_get(plan_config, "action_block")
    solver_warm_start = _cfg_get(solver, "warm_start")
    elite_retention = _cfg_get(solver, "elite_retention", args.elite_retention)
    colored_noise = _cfg_get(solver, "colored_noise", args.colored_noise)
    colored_noise_time = _cfg_get(solver, "colored_noise_time", args.colored_noise_time)
    momentum_alpha = _cfg_get(solver, "momentum_alpha", args.momentum_alpha)
    sample_decay = _cfg_get(solver, "sample_decay", args.sample_decay)
    sample_count_bucket_size = _cfg_get(
        solver,
        "sample_count_bucket_size",
        args.sample_count_bucket_size,
    )
    return_best_elite = _cfg_get(solver, "return_best_elite")
    clip_actions = _cfg_get(solver, "clip_actions")
    gd_seed_solver = _cfg_get(solver, "gd_seed_solver", args.gd_seed_solver)
    gd_seed_source = _cfg_get(solver, "gd_seed_source", args.gd_seed_source)
    gd_refine_topk = _cfg_get(solver, "gd_refine_topk", args.gd_refine_topk)
    gd_steps = _cfg_get(solver, "gd_steps", args.gd_steps)
    gd_lr = _cfg_get(solver, "gd_lr", args.gd_lr)
    gd_compile_cost = _cfg_get(solver, "gd_compile_cost", args.gd_compile_cost)
    static_chain_compile = _cfg_get(
        solver,
        "static_chain_compile",
        args.static_chain_compile,
    )
    static_chain_mode = _cfg_get(solver, "static_chain_mode", args.static_chain_mode)
    static_chain_fullgraph = _cfg_get(
        solver,
        "static_chain_fullgraph",
        args.static_chain_fullgraph,
    )
    static_chain_warmup = _cfg_get(
        solver,
        "static_chain_warmup",
        args.static_chain_warmup,
    )
    count_sample_decay = sample_decay
    count_bucket_size = sample_count_bucket_size
    if gd_seed_solver == "cem":
        count_sample_decay = 1.0
        count_bucket_size = 0
    model_cost_evals_per_cem_call = planner_model_cost_evals_per_call(
        samples=samples,
        cem_steps=cem_steps,
        topk=topk,
        sample_decay=count_sample_decay,
        sample_count_bucket_size=count_bucket_size,
    )
    return {
        "samples": samples,
        "cem_steps": cem_steps,
        "model_cost_evals_per_cem_call": model_cost_evals_per_cem_call,
        "horizon": horizon,
        "receding_horizon": receding_horizon,
        "action_block": action_block,
        "topk": topk,
        "warm_start": parse_optional_bool(args.warm_start),
        "solver_warm_start": solver_warm_start,
        "elite_retention": elite_retention,
        "colored_noise": colored_noise,
        "colored_noise_time": colored_noise_time,
        "momentum_alpha": momentum_alpha,
        "sample_decay": sample_decay,
        "sample_count_bucket_size": sample_count_bucket_size,
        "return_best_elite": return_best_elite,
        "clip_actions": clip_actions,
        "mppi_temperature": args.mppi_temperature,
        "gd_seed_solver": gd_seed_solver,
        "gd_seed_source": gd_seed_source,
        "gd_refine_topk": gd_refine_topk,
        "gd_steps": gd_steps,
        "gd_lr": gd_lr,
        "gd_compile_cost": gd_compile_cost,
        "static_chain_compile": static_chain_compile,
        "static_chain_mode": static_chain_mode,
        "static_chain_fullgraph": static_chain_fullgraph,
        "static_chain_warmup": static_chain_warmup,
        "model_cost_autocast": args.model_cost_autocast,
        "icem_step_telemetry": args.icem_step_telemetry,
    }


def parse_seed_list(seed_arg: int | None, seeds_arg: str | None) -> list[int]:
    if seeds_arg:
        return [int(s.strip()) for s in seeds_arg.split(",") if s.strip()]
    if seed_arg is not None:
        return [seed_arg]
    return [0]


def strip_yaml_suffix(config_name: str) -> str:
    return config_name[:-5] if config_name.endswith(".yaml") else config_name


def load_lewm_eval_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("lewm_eval_helpers", LEWM_DIR / "eval.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load lewm/eval.py helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lewm_eval_helpers"] = module
    spec.loader.exec_module(module)
    return module


def policy_object_candidates(weights_root: Path, policy: str) -> list[Path]:
    if "/" in policy:
        first, leaf = policy.rsplit("/", 1)
        return [
            weights_root / first / f"{leaf}_object.ckpt",
            weights_root / "lewm" / first / f"{leaf}_object.ckpt",
        ]
    return [weights_root / f"{policy}_object.ckpt"]


def load_policy_model(swm: Any, weights_root: Path, policy: str) -> Any:
    import torch

    for path in policy_object_candidates(weights_root, policy):
        if path.exists() and path.is_file():
            return torch.load(path, map_location="cpu", weights_only=False)
    return swm.wm.utils.load_pretrained(policy)


def configure_model_for_variant(model: Any, variant: str) -> None:
    if hasattr(model, "set_rollout_prealloc"):
        model.set_rollout_prealloc(False)
    if hasattr(model, "set_rollout_action_cache"):
        model.set_rollout_action_cache(False)
    if hasattr(model, "set_compile_cost"):
        model.set_compile_cost(False)
    if variant not in EMBEDDING_CACHE_VARIANTS:
        if hasattr(model, "set_embedding_cache"):
            model.set_embedding_cache(False)
        return
    if not hasattr(model, "set_embedding_cache"):
        raise TypeError(
            f"Variant {variant} requires a model with set_embedding_cache(), got {type(model)}"
        )
    model.set_embedding_cache(True)
    if variant in ROLLOUT_PREALLOC_VARIANTS:
        if not hasattr(model, "set_rollout_prealloc"):
            raise TypeError(
                f"Variant {variant} requires a model with set_rollout_prealloc(), got {type(model)}"
            )
        model.set_rollout_prealloc(True)
    if variant in ROLLOUT_ACTION_CACHE_VARIANTS:
        if not hasattr(model, "set_rollout_action_cache"):
            raise TypeError(
                f"Variant {variant} requires a model with set_rollout_action_cache(), got {type(model)}"
            )
        model.set_rollout_action_cache(True)


def configure_model_cost_autocast(model: Any, mode: str) -> dict[str, Any]:
    if mode == "none":
        if hasattr(model, "set_model_cost_autocast"):
            model.set_model_cost_autocast("none")
    else:
        if not hasattr(model, "set_model_cost_autocast"):
            raise TypeError(
                f"model-cost autocast requires set_model_cost_autocast(), got {type(model)}"
            )
        model.set_model_cost_autocast(mode)

    dtype = (
        model.model_cost_autocast_dtype()
        if hasattr(model, "model_cost_autocast_dtype")
        else None
    )
    enabled = (
        bool(model.model_cost_autocast_enabled())
        if hasattr(model, "model_cost_autocast_enabled")
        else False
    )
    active_mode = (
        model.model_cost_autocast_mode()
        if hasattr(model, "model_cost_autocast_mode")
        else "none"
    )
    return {
        "type": "model_cost_autocast_info",
        "requested": mode,
        "active": active_mode,
        "enabled": enabled,
        "dtype": str(dtype) if dtype is not None else None,
    }


def _first_parameter(model: Any) -> Any:
    try:
        return next(model.parameters())
    except (AttributeError, StopIteration) as exc:
        raise TypeError(f"Could not infer model parameter device/dtype for {type(model)}") from exc


def _infer_compile_embed_dim(model: Any) -> int:
    pos_embedding = getattr(getattr(model, "predictor", None), "pos_embedding", None)
    if pos_embedding is not None and hasattr(pos_embedding, "shape"):
        return int(pos_embedding.shape[-1])
    raise TypeError(
        f"Could not infer predictor embedding dimension for compile_cost model {type(model)}"
    )


def _infer_compile_action_dim(model: Any) -> int:
    patch_embed = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "in_channels"):
        return int(patch_embed.in_channels)
    raise TypeError(f"Could not infer action dimension for compile_cost model {type(model)}")


def install_compile_cost(
    model: Any,
    cfg: Any,
    plan_config: Any,
    *,
    warmup_history: int,
    warmup_batch_size: int = 1,
    warmup_sample_counts: list[int] | None = None,
) -> dict[str, Any]:
    import torch

    if not hasattr(model, "get_cost_from_embeddings"):
        raise TypeError(
            f"compile_cost requires get_cost_from_embeddings(), got model {type(model)}"
        )
    if not hasattr(model, "set_compile_cost"):
        raise TypeError(f"compile_cost requires set_compile_cost(), got model {type(model)}")
    if not hasattr(torch, "compile"):
        raise RuntimeError("compile_cost requires torch.compile")

    if not hasattr(model, "_eager_get_cost_from_embeddings"):
        model._eager_get_cost_from_embeddings = model.get_cost_from_embeddings
    model.get_cost_from_embeddings = torch.compile(
        model.get_cost_from_embeddings,
        mode="reduce-overhead",
        fullgraph=False,
    )
    model.set_compile_cost(True)

    parameter = _first_parameter(model)
    device = parameter.device
    dtype = parameter.dtype
    default_samples = int(getattr(cfg.solver, "num_samples"))
    sample_counts = warmup_sample_counts or [default_samples]
    for sample_count in sample_counts:
        if int(sample_count) <= 0:
            raise ValueError(f"compile_cost warmup sample count must be positive, got {sample_count}")
    sample_counts = [int(sample_count) for sample_count in sample_counts]
    horizon = int(getattr(plan_config, "horizon"))
    history = int(warmup_history)
    if history <= 0:
        raise ValueError(f"compile_cost warmup history must be positive, got {history}")
    requested_batch = int(warmup_batch_size)
    if requested_batch <= 0:
        raise ValueError(f"compile_cost warmup batch size must be positive, got {requested_batch}")
    embed_dim = _infer_compile_embed_dim(model)
    action_dim = _infer_compile_action_dim(model)
    counter_before = torch_compile_counter_snapshot()

    started = time.perf_counter()
    warmup_fallback_count = 0
    warmup_batch = requested_batch
    while True:
        warmup_records = []
        warmup_signature = None
        warmup_cost = None
        try:
            for samples in sample_counts:
                emb = torch.zeros((warmup_batch, history, embed_dim), device=device, dtype=dtype)
                goal_emb = torch.zeros_like(emb)
                action_candidates = torch.zeros(
                    (warmup_batch, samples, horizon, action_dim),
                    device=device,
                    dtype=dtype,
                )
                signature = compile_cost_signature(
                    emb=emb,
                    goal_emb=goal_emb,
                    action_candidates=action_candidates,
                )
                with torch.inference_mode():
                    warmup_cost = model.get_cost_from_embeddings(emb, goal_emb, action_candidates)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if warmup_signature is None:
                    warmup_signature = signature
                warmup_records.append(
                    {
                        "sample_count": samples,
                        "signature": signature,
                        "shape": {
                            "emb": list(emb.shape),
                            "goal_emb": list(goal_emb.shape),
                            "action_candidates": list(action_candidates.shape),
                            "cost": list(warmup_cost.shape),
                        },
                    }
                )
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message and ("cuda" in message or "cublas" in message)
            if not is_oom or warmup_batch <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            warmup_fallback_count += 1
            warmup_batch = max(1, warmup_batch // 2)
    counter_after = torch_compile_counter_snapshot()
    primary_warmup_shape = warmup_records[0]["shape"]

    return {
        "type": "compile_info",
        "compile_cost_requested": True,
        "compile_cost_enabled": True,
        "compile_mode": "reduce-overhead",
        "fullgraph": False,
        "warmup_elapsed_s": time.perf_counter() - started,
        "warmup_history": history,
        "warmup_history_source": "benchmark_cli_runtime_observation_history",
        "warmup_requested_batch_size": requested_batch,
        "warmup_batch_size": warmup_batch,
        "warmup_oom_fallback_count": warmup_fallback_count,
        "warmup_sample_counts": sample_counts,
        "warmup_shape_count": len(sample_counts),
        "warmup_signature": warmup_signature,
        "warmup_signatures": [record["signature"] for record in warmup_records],
        "warmup_counter_delta": diff_numeric_trees(counter_before, counter_after) or {},
        "warmup_counter_snapshot_available": bool(counter_after.get("available")),
        "warmup_counter_sources": list((counter_after.get("sources") or {}).keys()),
        "warmup_counter_errors": counter_after.get("errors") or {},
        "warmup_shape": primary_warmup_shape,
        "warmup_shapes": warmup_records,
    }


def maybe_warmup_static_chain_solver(
    solver: Any,
    model: Any,
    *,
    history: int = 1,
) -> dict[str, Any]:
    warmup = getattr(solver, "warmup_static_chain_compile", None)
    if not callable(warmup):
        return {
            "type": "static_chain_compile_info",
            "static_chain_supported": False,
            "static_chain_compile_enabled": False,
        }

    import torch

    counter_before = torch_compile_counter_snapshot()
    started = time.perf_counter()
    result = warmup(
        embed_dim=_infer_compile_embed_dim(model),
        history=history,
        batch_size=getattr(solver, "batch_size", 1),
    )
    counter_after = torch_compile_counter_snapshot()
    return {
        "type": "static_chain_compile_info",
        "static_chain_supported": True,
        "static_chain_compile_enabled": bool(getattr(result, "enabled", False)),
        "static_chain_compile_mode": getattr(result, "mode", None),
        "static_chain_fullgraph": getattr(result, "fullgraph", None),
        "static_chain_warmup_elapsed_s": getattr(result, "elapsed_s", None),
        "static_chain_warmup_outer_elapsed_s": time.perf_counter() - started,
        "static_chain_sample_counts": getattr(result, "sample_counts", None),
        "static_chain_input_shapes": getattr(result, "input_shapes", None),
        "static_chain_counter_delta": diff_numeric_trees(counter_before, counter_after) or {},
        "static_chain_counter_snapshot_available": bool(counter_after.get("available")),
        "static_chain_counter_sources": list((counter_after.get("sources") or {}).keys()),
        "static_chain_counter_errors": counter_after.get("errors") or {},
        "static_chain_cuda_available": bool(torch.cuda.is_available()),
    }


def dataset_candidates(weights_root: Path, dataset_name: str) -> list[Path]:
    return [
        weights_root / "datasets" / f"{dataset_name}.h5",
        weights_root / "lewm" / f"{dataset_name}.h5",
        weights_root / f"{dataset_name}.h5",
    ]


def first_existing_hash(paths: list[Path]) -> str:
    for path in paths:
        if path.exists() and path.is_file():
            return sha256_file(path)[0]
    return "unknown"


def precomputed_hash(results_root: Path, artifact_path: Path) -> str:
    report = results_root / "artifact_hashes" / "lewm_pusht.json"
    if not report.exists():
        return "unknown"
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    wanted = str(artifact_path)
    for artifact in payload.get("artifacts", []):
        if artifact.get("path") == wanted and artifact.get("sha256"):
            return str(artifact["sha256"])
    return "unknown"


class ModelInstrumentation:
    def __init__(
        self,
        model: Any,
        enabled: bool = True,
        *,
        compile_cost_enabled: bool = False,
    ):
        self.model = model
        self.enabled = enabled
        self.compile_cost_enabled = compile_cost_enabled
        self.encode_counts = {"total": 0, "goal": 0, "initial": 0, "unknown": 0}

    def install(self) -> None:
        if not self.enabled:
            return
        import torch

        def wrap_bound(obj: Any, name: str, label: str | Callable[[tuple[Any, ...], dict[str, Any]], str]) -> None:
            if not hasattr(obj, name):
                return
            original = getattr(obj, name)

            def wrapped(*args: Any, **kwargs: Any) -> Any:
                resolved = label(args, kwargs) if callable(label) else label
                with torch.profiler.record_function(resolved):
                    return original(*args, **kwargs)

            setattr(obj, name, wrapped)

        def encode_label(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
            info = args[0] if args else kwargs.get("info", {})
            bucket = "unknown"
            if isinstance(info, dict):
                bucket = "initial" if "action" in info else "goal"
            self.encode_counts["total"] += 1
            self.encode_counts[bucket] += 1
            return f"lewm.encode_{bucket}"

        wrap_bound(self.model, "encode", encode_label)
        wrap_bound(self.model, "get_cost", "lewm.model_get_cost")
        if self.compile_cost_enabled:
            return
        wrap_bound(self.model, "rollout", "lewm.rollout")
        wrap_bound(self.model, "criterion", "lewm.criterion")
        wrap_bound(self.model, "predict", "lewm.predict_step")
        if hasattr(self.model, "action_encoder"):
            wrap_bound(self.model.action_encoder, "forward", "lewm.action_encoder")

    def snapshot(self) -> dict[str, int]:
        return dict(self.encode_counts)


class LatencyRecorder:
    def __init__(
        self,
        *,
        metrics_path: Path,
        seed: int,
        profile: str,
        warmup_plans: int,
        max_recorded_plans: int | None,
        instrumentation: ModelInstrumentation,
        cem_telemetry: "CEMTelemetry | None" = None,
    ):
        self.metrics_path = metrics_path
        self.seed = seed
        self.profile = profile
        self.warmup_plans = warmup_plans
        self.max_recorded_plans = max_recorded_plans
        self.instrumentation = instrumentation
        self.cem_telemetry = cem_telemetry
        self.plan_index = 0
        self.profiler = None

    def measure(self, fn: Callable[[], Any]) -> Any:
        import torch

        self.plan_index += 1
        before = self.instrumentation.snapshot()
        cem_before = self.cem_telemetry.snapshot() if self.cem_telemetry else {}
        if self.cem_telemetry:
            self.cem_telemetry.current_plan_index = self.plan_index
        use_cuda_events = self.profile == "cuda-events" and torch.cuda.is_available()

        try:
            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                result = fn()
                end.record()
                torch.cuda.synchronize()
                elapsed_ms = start.elapsed_time(end)
                timing_source = "cuda_events"
            else:
                t0 = time.perf_counter()
                result = fn()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                timing_source = "perf_counter"
        finally:
            if self.cem_telemetry:
                self.cem_telemetry.current_plan_index = None

        after = self.instrumentation.snapshot()
        cem_after = self.cem_telemetry.snapshot() if self.cem_telemetry else {}
        if self.max_recorded_plans is None or self.plan_index <= self.max_recorded_plans:
            append_jsonl(
                self.metrics_path,
                {
                    "type": "plan_latency",
                    "seed": self.seed,
                    "plan_index": self.plan_index,
                    "warmup": self.plan_index <= self.warmup_plans,
                    "plan_latency_ms": elapsed_ms,
                    "timing_source": timing_source,
                    "encode_delta": {
                        key: after.get(key, 0) - before.get(key, 0)
                        for key in sorted(after)
                    },
                    "cem_call_delta": cem_after.get("calls", 0) - cem_before.get("calls", 0),
                    "cem_cpu_wall_ms_delta": cem_after.get("cpu_wall_ms", 0.0)
                    - cem_before.get("cpu_wall_ms", 0.0),
                    "cem_cuda_elapsed_ms_delta": cem_after.get("cuda_elapsed_ms", 0.0)
                    - cem_before.get("cuda_elapsed_ms", 0.0),
                    "model_cost_evals_delta": cem_after.get("model_cost_evals", 0)
                    - cem_before.get("model_cost_evals", 0),
                    "model_cost_env_candidate_evals_delta": cem_after.get(
                        "model_cost_env_candidate_evals", 0
                    )
                    - cem_before.get("model_cost_env_candidate_evals", 0),
                },
            )
        if self.profiler is not None:
            self.profiler.step()
        return result


class TimedPolicy:
    def __init__(self, delegate: Any, recorder: LatencyRecorder):
        self.delegate = delegate
        self.recorder = recorder

    def set_env(self, env: Any) -> None:
        self.delegate.set_env(env)

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> Any:
        return self.recorder.measure(lambda: self.delegate.get_action(info_dict, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _shape_list(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except TypeError:
        return None


def _tensor_signature(value: Any) -> dict[str, Any] | None:
    shape = _shape_list(value)
    if shape is None:
        return None
    return {
        "shape": shape,
        "dtype": str(getattr(value, "dtype", "unknown")),
        "device": str(getattr(value, "device", "unknown")),
    }


def compile_cost_signature(
    *,
    emb: Any,
    goal_emb: Any,
    action_candidates: Any,
) -> dict[str, Any]:
    return {
        "emb": _tensor_signature(emb),
        "goal_emb": _tensor_signature(goal_emb),
        "action_candidates": _tensor_signature(action_candidates),
    }


def compile_signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(json_safe(signature), sort_keys=True, separators=(",", ":"))


def compare_compile_signatures(
    expected: dict[str, Any] | None,
    actual: dict[str, Any] | None,
) -> dict[str, Any]:
    if not expected or not actual:
        return {
            "matches": None,
            "mismatch_fields": [],
        }
    mismatches = []
    for name in sorted(set(expected) | set(actual)):
        expected_tensor = expected.get(name) or {}
        actual_tensor = actual.get(name) or {}
        for field in ("shape", "dtype", "device"):
            if expected_tensor.get(field) != actual_tensor.get(field):
                mismatches.append(
                    {
                        "tensor": name,
                        "field": field,
                        "expected": expected_tensor.get(field),
                        "actual": actual_tensor.get(field),
                    }
                )
    return {
        "matches": not mismatches,
        "mismatch_fields": mismatches,
    }


def _numeric_counter_tree(value: Any) -> Any:
    if isinstance(value, (int, float, bool)):
        return int(value) if isinstance(value, bool) else value
    if isinstance(value, dict):
        converted = {
            str(key): _numeric_counter_tree(child)
            for key, child in value.items()
        }
        return {
            key: child
            for key, child in converted.items()
            if child not in ({}, None)
        }
    return None


def torch_compile_counter_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "available": False,
        "sources": {},
        "errors": {},
    }
    try:
        from torch._dynamo.utils import counters  # type: ignore[attr-defined]

        dynamo_counters = _numeric_counter_tree(counters)
        snapshot["sources"]["torch._dynamo.utils.counters"] = dynamo_counters
        snapshot["available"] = True
    except Exception as exc:
        snapshot["errors"]["torch._dynamo.utils.counters"] = f"{type(exc).__name__}: {exc}"

    try:
        import torch._inductor.metrics as inductor_metrics  # type: ignore[attr-defined]

        numeric_metrics = {}
        for name in dir(inductor_metrics):
            if name.startswith("_"):
                continue
            value = getattr(inductor_metrics, name)
            if isinstance(value, (int, float, bool)):
                numeric_metrics[name] = int(value) if isinstance(value, bool) else value
        snapshot["sources"]["torch._inductor.metrics"] = numeric_metrics
        if numeric_metrics:
            snapshot["available"] = True
    except Exception as exc:
        snapshot["errors"]["torch._inductor.metrics"] = f"{type(exc).__name__}: {exc}"

    return snapshot


def diff_numeric_trees(before: Any, after: Any) -> Any:
    if isinstance(before, (int, float)) or isinstance(after, (int, float)):
        before_value = before if isinstance(before, (int, float)) else 0
        after_value = after if isinstance(after, (int, float)) else 0
        delta = after_value - before_value
        return delta if delta else None
    if isinstance(before, dict) or isinstance(after, dict):
        before_dict = before if isinstance(before, dict) else {}
        after_dict = after if isinstance(after, dict) else {}
        delta_dict = {}
        for key in sorted(set(before_dict) | set(after_dict)):
            delta = diff_numeric_trees(before_dict.get(key), after_dict.get(key))
            if delta not in ({}, None):
                delta_dict[key] = delta
        return delta_dict
    return None


def _mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


class CompileDiagnostics:
    def __init__(
        self,
        *,
        metrics_path: Path,
        seed: int,
        enabled: bool,
        expected_signature: dict[str, Any] | None,
        max_signature_records: int,
    ) -> None:
        self.metrics_path = metrics_path
        self.seed = seed
        self.enabled = enabled
        self.expected_signature = expected_signature
        self.max_signature_records = max(0, int(max_signature_records))
        self._last_counter_snapshot: dict[str, Any] | None = None

    def snapshot_counters(self) -> dict[str, Any]:
        if not self.enabled:
            return {"available": False, "sources": {}, "errors": {}}
        return torch_compile_counter_snapshot()

    def emit_event(
        self,
        event: str,
        *,
        phase: str,
        signature: dict[str, Any] | None = None,
        elapsed_s: float | None = None,
        include_counters: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        row: dict[str, Any] = {
            "type": "compile_event",
            "seed": self.seed,
            "event": event,
            "phase": phase,
            "expected_signature": self.expected_signature,
        }
        if signature is not None:
            row["signature"] = signature
            row["signature_matches_expected"] = compare_compile_signatures(
                self.expected_signature,
                signature,
            )
        if elapsed_s is not None:
            row["elapsed_s"] = elapsed_s
        if extra:
            row.update(extra)
        if include_counters:
            snapshot = self.snapshot_counters()
            row["counter_snapshot"] = snapshot
            if self._last_counter_snapshot is not None:
                row["counter_delta_since_previous_event"] = diff_numeric_trees(
                    self._last_counter_snapshot,
                    snapshot,
                ) or {}
            self._last_counter_snapshot = snapshot
        append_jsonl(self.metrics_path, row)
        return row

    def begin_call(self, call: dict[str, Any]) -> None:
        if not self.enabled:
            return
        call["compile_diagnostics_enabled"] = True
        call["compile_expected_signature"] = self.expected_signature
        call["_compile_counter_before"] = self.snapshot_counters()
        call["_compile_signature_keys_seen"] = set()
        call["_compile_signature_observation_count"] = 0

    def observe_cost_inputs(
        self,
        call: dict[str, Any] | None,
        *,
        step: int,
        expanded_infos: dict[str, Any],
        action_candidates: Any,
    ) -> None:
        if not self.enabled or call is None:
            return
        signature = compile_cost_signature(
            emb=expanded_infos.get("emb"),
            goal_emb=expanded_infos.get("goal_emb"),
            action_candidates=action_candidates,
        )
        key = compile_signature_key(signature)
        seen = call.setdefault("_compile_signature_keys_seen", set())
        observation_count = int(call.get("_compile_signature_observation_count", 0)) + 1
        call["_compile_signature_observation_count"] = observation_count

        if "compile_runtime_signature_first" not in call:
            comparison = compare_compile_signatures(self.expected_signature, signature)
            call["compile_runtime_signature_first"] = signature
            call["compile_runtime_signature_first_step"] = step
            call["compile_runtime_signature_key_first"] = key
            call["compile_runtime_signature_matches_expected"] = comparison["matches"]
            call["compile_runtime_signature_mismatch_fields"] = comparison["mismatch_fields"]

        if key not in seen:
            seen.add(key)
            records = call.setdefault("compile_runtime_signature_records", [])
            if len(records) < self.max_signature_records:
                records.append(
                    {
                        "step": step,
                        "signature": signature,
                        "signature_key": key,
                        "matches_expected": compare_compile_signatures(
                            self.expected_signature,
                            signature,
                        ),
                    }
                )

        call["compile_runtime_signature_unique_count"] = len(seen)
        call["compile_runtime_signature_observation_count"] = observation_count

    def end_call(self, call: dict[str, Any]) -> None:
        if not self.enabled:
            return
        after = self.snapshot_counters()
        before = call.get("_compile_counter_before")
        call["compile_counter_snapshot_available"] = bool(after.get("available"))
        call["compile_counter_sources"] = list((after.get("sources") or {}).keys())
        call["compile_counter_errors"] = after.get("errors") or {}
        call["compile_counter_delta"] = diff_numeric_trees(before, after) or {}


class CEMTelemetry:
    """Structured telemetry sink for solver-level CEM calls."""

    def __init__(
        self,
        *,
        metrics_path: Path,
        seed: int,
        mode: str,
        warmup_plans: int,
        max_recorded_calls: int | None,
        compile_diagnostics: CompileDiagnostics | None = None,
    ) -> None:
        self.metrics_path = metrics_path
        self.seed = seed
        self.mode = mode
        self.warmup_plans = warmup_plans
        self.max_recorded_calls = max_recorded_calls
        self.compile_diagnostics = compile_diagnostics
        self.current_plan_index: int | None = None
        self.calls = 0
        self.measured_calls = 0
        self.cpu_wall_ms = 0.0
        self.cuda_elapsed_ms = 0.0
        self.model_cost_evals = 0
        self.model_cost_env_candidate_evals = 0
        self.gd_model_cost_evals = 0
        self.gd_model_cost_env_candidate_evals = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @property
    def use_cuda_events(self) -> bool:
        return self.mode == "cuda-events"

    def snapshot(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "cpu_wall_ms": self.cpu_wall_ms,
            "cuda_elapsed_ms": self.cuda_elapsed_ms,
            "model_cost_evals": self.model_cost_evals,
            "model_cost_env_candidate_evals": self.model_cost_env_candidate_evals,
            "gd_model_cost_evals": self.gd_model_cost_evals,
            "gd_model_cost_env_candidate_evals": self.gd_model_cost_env_candidate_evals,
            "model_cost_evals_including_gd": self.model_cost_evals + self.gd_model_cost_evals,
            "model_cost_env_candidate_evals_including_gd": (
                self.model_cost_env_candidate_evals
                + self.gd_model_cost_env_candidate_evals
            ),
        }

    def begin_call(self, solver: Any, info_dict: dict[str, Any], init_action: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        import numpy as np
        import torch

        self.calls += 1
        plan_index = self.current_plan_index
        is_warmup = bool(plan_index is not None and plan_index <= self.warmup_plans)
        if not is_warmup:
            self.measured_calls += 1
        call: dict[str, Any] = {
            "type": "cem_call",
            "seed": self.seed,
            "plan_index": plan_index,
            "cem_call_index": self.calls,
            "post_warmup_cem_call_index": None if is_warmup else self.measured_calls,
            "first_post_warmup_cem_call": (not is_warmup and self.measured_calls == 1),
            "warmup": is_warmup,
            "runtime_phase": "warmup" if is_warmup else "measured",
            "telemetry_mode": self.mode,
            "solver_class": solver.__class__.__name__,
            "solver_module": solver.__class__.__module__,
            "device": str(getattr(solver, "device", "unknown")),
            "dtype": str(getattr(solver, "dtype", "unknown")),
            "batch_size_config": getattr(solver, "batch_size", None),
            "num_samples": getattr(solver, "num_samples", None),
            "n_steps": getattr(solver, "n_steps", None),
            "topk": getattr(solver, "topk", None),
            "horizon": getattr(solver, "horizon", None),
            "action_dim": getattr(solver, "action_dim", None),
            "total_envs": len(next(iter(info_dict.values()))),
            "n_batches": None,
            "callback_count": len(getattr(solver, "callbacks", []) or []),
            "init_action_shape": _shape_list(init_action),
            "tensor_input_shapes": {},
            "ndarray_input_shapes": {},
            "non_tensor_input_types": {},
            "region_cpu_ms": {},
            "region_count": {},
            "_cpu_start": time.perf_counter(),
            "_emit": self.max_recorded_calls is None or self.calls <= self.max_recorded_calls,
        }
        batch_size = call.get("batch_size_config")
        if isinstance(batch_size, int) and batch_size > 0:
            call["n_batches"] = (call["total_envs"] + batch_size - 1) // batch_size
        num_samples = call.get("num_samples")
        n_steps = call.get("n_steps")
        total_envs = call.get("total_envs")
        sample_counts = None
        sample_counts_fn = getattr(solver, "sample_counts", None)
        if callable(sample_counts_fn):
            try:
                sample_counts = [int(value) for value in sample_counts_fn()]
            except Exception:
                sample_counts = None
        if sample_counts:
            call["sample_counts_per_step"] = sample_counts
            call["sample_count_decay_active"] = len(set(sample_counts)) > 1
            model_cost_evals = sum(sample_counts)
            call["model_cost_evals"] = model_cost_evals
            self.model_cost_evals += model_cost_evals
            if isinstance(total_envs, int):
                env_candidate_evals = total_envs * model_cost_evals
                call["model_cost_env_candidate_evals"] = env_candidate_evals
                self.model_cost_env_candidate_evals += env_candidate_evals
        elif isinstance(num_samples, int) and isinstance(n_steps, int):
            model_cost_evals = num_samples * n_steps
            call["model_cost_evals"] = model_cost_evals
            self.model_cost_evals += model_cost_evals
            if isinstance(total_envs, int):
                env_candidate_evals = total_envs * num_samples * n_steps
                call["model_cost_env_candidate_evals"] = env_candidate_evals
                self.model_cost_env_candidate_evals += env_candidate_evals

        for key, value in info_dict.items():
            if torch.is_tensor(value):
                call["tensor_input_shapes"][key] = _shape_list(value)
            elif isinstance(value, np.ndarray):
                call["ndarray_input_shapes"][key] = _shape_list(value)
            else:
                call["non_tensor_input_types"][key] = type(value).__name__

        if self.use_cuda_events and torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass
            call["cuda_memory_allocated_before_mb"] = _mb(torch.cuda.memory_allocated())
            call["cuda_memory_reserved_before_mb"] = _mb(torch.cuda.memory_reserved())
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call["_cuda_start"] = start
            call["_cuda_end"] = end

        if self.compile_diagnostics is not None:
            self.compile_diagnostics.begin_call(call)

        return call

    @contextlib.contextmanager
    def region(self, call: dict[str, Any] | None, label: str):
        import torch

        t0 = time.perf_counter()
        with torch.profiler.record_function(label):
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if call is not None and call.get("_emit", False):
                    regions = call.setdefault("region_cpu_ms", {})
                    counts = call.setdefault("region_count", {})
                    regions[label] = regions.get(label, 0.0) + elapsed_ms
                    counts[label] = counts.get(label, 0) + 1

    def end_call(
        self,
        call: dict[str, Any] | None,
        *,
        outputs: dict[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if call is None:
            return

        import torch

        cpu_ms = (time.perf_counter() - call["_cpu_start"]) * 1000.0
        call["cpu_wall_ms"] = cpu_ms
        self.cpu_wall_ms += cpu_ms

        if "_cuda_end" in call and torch.cuda.is_available():
            call["_cuda_end"].record()
            torch.cuda.synchronize()
            cuda_ms = call["_cuda_start"].elapsed_time(call["_cuda_end"])
            call["cuda_elapsed_ms"] = cuda_ms
            self.cuda_elapsed_ms += cuda_ms
            call["cuda_memory_allocated_after_mb"] = _mb(torch.cuda.memory_allocated())
            call["cuda_memory_reserved_after_mb"] = _mb(torch.cuda.memory_reserved())
            call["cuda_memory_peak_allocated_mb"] = _mb(torch.cuda.max_memory_allocated())

        if outputs is not None:
            actions = outputs.get("actions")
            costs = outputs.get("costs")
            call["output_actions_shape"] = _shape_list(actions)
            call["output_cost_count"] = len(costs) if isinstance(costs, list) else None
            callbacks = outputs.get("callbacks")
            if isinstance(callbacks, dict) and "icem_step_summary" in callbacks:
                call["icem_step_summary"] = callbacks["icem_step_summary"]
            gd_summary = outputs.get("gd_summary")
            if isinstance(gd_summary, dict):
                call["gd_summary"] = gd_summary
                for key, value in gd_summary.items():
                    call[key] = value
                gd_model_cost_evals = gd_summary.get("gd_model_cost_evals")
                if isinstance(gd_model_cost_evals, (int, float)):
                    gd_model_cost_evals_int = int(gd_model_cost_evals)
                    call["gd_model_cost_evals"] = gd_model_cost_evals_int
                    self.gd_model_cost_evals += gd_model_cost_evals_int
                    total_envs = call.get("total_envs")
                    if isinstance(total_envs, int):
                        gd_env_candidate_evals = total_envs * gd_model_cost_evals_int
                        call["gd_model_cost_env_candidate_evals"] = gd_env_candidate_evals
                        self.gd_model_cost_env_candidate_evals += gd_env_candidate_evals
                    seed_evals = call.get("model_cost_evals")
                    if isinstance(seed_evals, (int, float)):
                        call["model_cost_evals_including_gd"] = (
                            int(seed_evals) + gd_model_cost_evals_int
                        )
                    seed_env_evals = call.get("model_cost_env_candidate_evals")
                    gd_env_evals = call.get("gd_model_cost_env_candidate_evals")
                    if isinstance(seed_env_evals, (int, float)) and isinstance(
                        gd_env_evals, (int, float)
                    ):
                        call["model_cost_env_candidate_evals_including_gd"] = (
                            int(seed_env_evals) + int(gd_env_evals)
                        )
        if error is not None:
            call["error_type"] = type(error).__name__
            call["error_message"] = str(error)

        if self.compile_diagnostics is not None:
            self.compile_diagnostics.end_call(call)

        if call.get("_emit", False):
            append_jsonl(
                self.metrics_path,
                {key: value for key, value in call.items() if not key.startswith("_")},
            )


@contextlib.contextmanager
def maybe_torch_profiler(profile: str, run_dir: Path, *, with_stack: bool = False):
    if profile != "torch":
        yield None
        return
    import torch
    from torch.profiler import ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    trace_dir = run_dir / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=with_stack,
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    ) as prof:
        yield prof


def prepare_run_artifacts(
    run_dir: Path,
    metrics_path: Path,
    manifest_path: Path,
    *,
    profile: str,
) -> None:
    """Remove benchmark-owned artifacts so reused run IDs cannot mix results."""
    for path in (metrics_path, manifest_path, run_dir / "profiler_summary.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    for path in (run_dir / "profiler", run_dir / "videos"):
        if path.exists():
            shutil.rmtree(path)


def install_icem_step_telemetry(solver: Any, mode: str) -> bool:
    """Attach optional iCEM convergence telemetry through the solver callback API."""
    if mode == "none":
        return False
    if mode != "summary":
        raise ValueError(f"Unsupported --icem-step-telemetry mode: {mode}")
    if solver.__class__.__name__ != "ICEMSolver":
        raise ValueError("--icem-step-telemetry summary requires an ICEMSolver instance")

    from lewm.solvers.icem import ICEMStepSummaryCallback

    callback = ICEMStepSummaryCallback()
    callback.configure(
        action_low=getattr(solver, "_action_low", None),
        action_high=getattr(solver, "_action_high", None),
    )
    solver.callbacks.append(callback)
    return True


def install_cem_telemetry(
    solver: Any,
    telemetry: CEMTelemetry,
    *,
    force_ranges: bool = False,
) -> bool:
    """Patch supported CEM solvers with structured telemetry and profiler ranges."""
    solver_class = solver.__class__.__name__
    if solver_class not in {
        "CEMSolver",
        "CEMExactSolver",
        "ICEMSolver",
        "ICEMStaticCompileSolver",
        "CEMGDSolver",
    }:
        return False
    if not telemetry.enabled and not force_ranges:
        return False

    try:
        import time as time_module
        import types

        import numpy as np
        import torch
    except Exception:
        return False

    if solver_class in {"ICEMSolver", "ICEMStaticCompileSolver", "CEMGDSolver"}:
        original_solve = solver.solve

        def solve_with_call_telemetry(
            self: Any,
            info_dict: dict[str, Any],
            init_action: Any = None,
        ) -> dict[str, Any]:
            call = telemetry.begin_call(self, info_dict, init_action)
            if call is not None:
                total_envs = len(next(iter(info_dict.values())))
                n_batches = (total_envs + self.batch_size - 1) // self.batch_size
                call["cpu_sync_in_loop"] = False
                call["loop_cpu_materialization_count_estimate"] = 0
                call["final_cpu_materialization_count_estimate"] = n_batches + 3
            try:
                outputs = original_solve(info_dict, init_action=init_action)
                telemetry.end_call(call, outputs=outputs)
                return outputs
            except BaseException as exc:
                telemetry.end_call(call, error=exc)
                raise

        solver.solve = types.MethodType(solve_with_call_telemetry, solver)
        return True

    if solver_class == "CEMExactSolver":
        from lewm.solvers.cem_exact import prepare_init_action as prepare_init_action_fn

        defer_cpu_logging = True
    else:
        try:
            from stable_worldmodel.solver.utils import prepare_init_action as prepare_init_action_fn
        except Exception:
            return False
        defer_cpu_logging = False

    @torch.inference_mode()
    def solve_with_ranges(self: Any, info_dict: dict[str, Any], init_action: Any = None) -> dict[str, Any]:
        call = telemetry.begin_call(self, info_dict, init_action)

        @contextlib.contextmanager
        def region(label: str):
            if telemetry.enabled or force_ranges:
                with telemetry.region(call, label):
                    yield
            else:
                yield

        start_time = time_module.time()
        outputs = {
            "costs": [],
            "mean": [],
            "var": [],
        }
        total_envs = len(next(iter(info_dict.values())))
        if call is not None:
            n_batches = (total_envs + self.batch_size - 1) // self.batch_size
            call["cpu_sync_in_loop"] = not defer_cpu_logging
            call["loop_cpu_materialization_count_estimate"] = (
                0 if defer_cpu_logging else n_batches * self.n_steps
            )
            call["final_cpu_materialization_count_estimate"] = (
                (n_batches if defer_cpu_logging else 0) + 3
            )

        try:
            with region("cem.init_action"):
                init_action = prepare_init_action_fn(
                    self.model,
                    info_dict,
                    init_action,
                    self.horizon,
                    n_envs=total_envs,
                    action_dim=self.action_dim,
                )

            with region("cem.init_distribution"):
                mean, var = self.init_action_distrib(total_envs, init_action)
                mean = mean.to(self.device)
                var = var.to(self.device)
                if hasattr(self, "_shift_warm_start_distribution"):
                    mean, var = self._shift_warm_start_distribution(mean, var)
                    if call is not None:
                        call["warm_start_distribution"] = getattr(
                            self,
                            "last_warm_start_info",
                            {},
                        )

            with region("cem.reset_callbacks"):
                for cb in self.callbacks:
                    cb.reset()

            for start_idx in range(0, total_envs, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_envs)
                current_bs = end_idx - start_idx
                batch_mean = mean[start_idx:end_idx]
                batch_var = var[start_idx:end_idx]

                with region("cem.expand_info"):
                    expanded_infos = {}
                    for key, value in info_dict.items():
                        value_batch = value[start_idx:end_idx]
                        if torch.is_tensor(value):
                            target_dtype = self.dtype if value_batch.is_floating_point() else None
                            value_batch = (
                                value_batch.to(device=self.device, dtype=target_dtype)
                                .unsqueeze(1)
                                .expand(
                                    current_bs,
                                    self.num_samples,
                                    *value_batch.shape[1:],
                                )
                            )
                        elif isinstance(value, np.ndarray):
                            value_batch = np.repeat(
                                value_batch[:, None, ...], self.num_samples, axis=1
                            )
                        expanded_infos[key] = value_batch

                final_batch_cost = None
                final_batch_cost_gpu = None
                with region("cem.start_batch_callbacks"):
                    for cb in self.callbacks:
                        cb.start_batch()

                for step in range(self.n_steps):
                    with region("cem.sample_candidates"):
                        candidates = torch.randn(
                            current_bs,
                            self.num_samples,
                            self.horizon,
                            self.action_dim,
                            generator=self.torch_gen,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                        candidates[:, 0] = batch_mean

                    with region("cem.model_get_cost"):
                        costs = self.model.get_cost(expanded_infos, candidates)
                    if telemetry.compile_diagnostics is not None:
                        telemetry.compile_diagnostics.observe_cost_inputs(
                            call,
                            step=step,
                            expanded_infos=expanded_infos,
                            action_candidates=candidates,
                        )

                    assert isinstance(costs, torch.Tensor), (
                        f"Expected cost to be a torch.Tensor, got {type(costs)}"
                    )
                    assert (
                        costs.ndim == 2
                        and costs.shape[0] == current_bs
                        and costs.shape[1] == self.num_samples
                    ), (
                        f"Expected cost to be of shape ({current_bs}, {self.num_samples}), "
                        f"got {costs.shape}"
                    )

                    with region("cem.topk"):
                        topk_vals, topk_inds = torch.topk(
                            costs, k=self.topk, dim=1, largest=False
                        )

                    with region("cem.elite_stats"):
                        batch_indices = (
                            torch.arange(current_bs, device=self.device)
                            .unsqueeze(1)
                            .expand(-1, self.topk)
                        )
                        topk_candidates = candidates[batch_indices, topk_inds]
                        prev_mean = batch_mean
                        prev_var = batch_var
                        batch_mean = topk_candidates.mean(dim=1)
                        batch_var = topk_candidates.std(dim=1)

                    if self.callbacks:
                        with region("cem.callbacks"):
                            for cb in self.callbacks:
                                cb(
                                    step=step,
                                    candidates=candidates,
                                    costs=costs,
                                    topk_vals=topk_vals,
                                    topk_inds=topk_inds,
                                    topk_candidates=topk_candidates,
                                    mean=batch_mean,
                                    var=batch_var,
                                    prev_mean=prev_mean,
                                    prev_var=prev_var,
                                )

                    if defer_cpu_logging:
                        with region("cem.cost_reduce_gpu"):
                            final_batch_cost_gpu = topk_vals.mean(dim=1)
                    else:
                        with region("cem.cpu_logging"):
                            final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

                with region("cem.write_back"):
                    mean[start_idx:end_idx] = batch_mean
                    var[start_idx:end_idx] = batch_var

                if defer_cpu_logging:
                    assert final_batch_cost_gpu is not None
                    with region("cem.cpu_final_cost"):
                        outputs["costs"].extend(final_batch_cost_gpu.detach().cpu().tolist())
                else:
                    outputs["costs"].extend(final_batch_cost)

            with region("cem.cpu_outputs"):
                outputs["actions"] = mean.detach().cpu()
                outputs["mean"] = [mean.detach().cpu()]
                outputs["var"] = [var.detach().cpu()]
                if hasattr(self, "last_warm_start_info"):
                    outputs["warm_start"] = self.last_warm_start_info
                if hasattr(self, "_remember_warm_start_distribution"):
                    self._remember_warm_start_distribution(mean, var)

            if self.callbacks:
                with region("cem.end_callbacks"):
                    outputs["callbacks"] = {}
                    for cb in self.callbacks:
                        cb.end_solve()
                        outputs["callbacks"][cb.output_key] = cb.history

            print(f"CEM solve time: {time_module.time() - start_time:.4f} seconds")
            telemetry.end_call(call, outputs=outputs)
            return outputs
        except BaseException as exc:
            telemetry.end_call(call, error=exc)
            raise

    solver.solve = types.MethodType(solve_with_ranges, solver)
    return True


def compose_config(
    config_name: str,
    policy: str,
    solver: str,
    episodes: int,
    seed: int,
    planner_overrides: argparse.Namespace | None = None,
) -> Any:
    import hydra
    from hydra import compose, initialize_config_dir

    overrides = [
        f"policy={policy}",
        f"solver={solver}",
        f"eval.num_eval={episodes}",
        f"seed={seed}",
    ]
    if planner_overrides is not None:
        overrides.extend(planner_hydra_overrides(planner_overrides))
    config_dir = str(LEWM_DIR / "config" / "eval")
    with initialize_config_dir(config_dir=config_dir, version_base=None, job_name="lewm_plan_benchmark"):
        cfg = compose(config_name=strip_yaml_suffix(config_name), overrides=overrides)
    return cfg, hydra


def build_policy_and_world(
    cfg: Any, weights_root: Path, variant: str
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np
    import stable_worldmodel as swm
    from sklearn import preprocessing

    helpers = load_lewm_eval_helpers()
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        "pixels": helpers.img_transform(cfg),
        "goal": helpers.img_transform(cfg),
    }
    expected_dataset = weights_root / "datasets" / f"{cfg.eval.dataset_name}.h5"
    if not expected_dataset.exists():
        legacy_dataset = weights_root / f"{cfg.eval.dataset_name}.h5"
        if legacy_dataset.exists():
            raise FileNotFoundError(
                f"Expected stable-worldmodel dataset at {expected_dataset}; found legacy "
                f"root-level dataset at {legacy_dataset}. Rerun stage_lewm_pusht to migrate it."
            )
        raise FileNotFoundError(
            f"Expected stable-worldmodel dataset at {expected_dataset}. "
            "Run stage_lewm_pusht before running the benchmark."
        )
    dataset = helpers.get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model = load_policy_model(swm, weights_root, cfg.policy)
    configure_model_for_variant(model, variant)
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    return world, dataset, model, {
        "process": process,
        "transform": transform,
        "config": config,
        "ep_indices": ep_indices,
    }


def select_eval_rows(cfg: Any, dataset: Any, ep_indices: Any, seed: int) -> tuple[Any, Any]:
    import numpy as np

    helpers = load_lewm_eval_helpers()
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_len = helpers.get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Not enough valid starting points: need {cfg.eval.num_eval}, have {len(valid_indices)}"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(valid_indices), size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[chosen])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]
    return eval_episodes, eval_start_idx


def run_seed(args: argparse.Namespace, seed: int, run_dir: Path, metrics_path: Path) -> None:
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf

    matmul_precision_before = torch.get_float32_matmul_precision()
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)
    matmul_precision_after = torch.get_float32_matmul_precision()
    cfg, hydra_module = compose_config(
        args.config_name,
        args.policy,
        args.solver,
        args.episodes,
        seed,
        planner_overrides=args,
    )
    planner_settings = effective_planner_settings(args, cfg)
    weights_root = Path(args.weights_root or os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    append_jsonl(
        metrics_path,
        {
            "type": "runtime_resource_info",
            "seed": seed,
            "resources": discover_runtime_resources(),
        },
    )
    append_jsonl(
        metrics_path,
        {
            "type": "matmul_precision_info",
            "seed": seed,
            "requested": args.matmul_precision,
            "before": matmul_precision_before,
            "after": matmul_precision_after,
        },
    )
    world, dataset, model, context = build_policy_and_world(cfg, weights_root, args.variant)
    model_cost_autocast_info = configure_model_cost_autocast(
        model,
        args.model_cost_autocast,
    )
    model_cost_autocast_info["seed"] = seed
    compile_enabled = compile_cost_requested(args)
    compile_diagnostics_mode = effective_compile_diagnostics(args)
    compile_info = {
        "type": "compile_info",
        "seed": seed,
        "compile_cost_requested": compile_enabled,
        "compile_cost_enabled": False,
        "compile_diagnostics_mode": compile_diagnostics_mode,
    }
    if compile_enabled:
        compile_warmup_sample_counts = resolve_compile_warmup_sample_counts(
            args.compile_warmup_sample_counts,
            cfg,
        )
        compile_info = install_compile_cost(
            model,
            cfg,
            context["config"],
            warmup_history=args.compile_warmup_history,
            warmup_batch_size=1,
            warmup_sample_counts=compile_warmup_sample_counts,
        )
        compile_info["seed"] = seed
        compile_info["compile_diagnostics_mode"] = compile_diagnostics_mode
        compile_info["compile_warmup_sample_counts_requested"] = (
            args.compile_warmup_sample_counts
        )
    compile_info["matmul_precision_requested"] = args.matmul_precision
    compile_info["matmul_precision_before"] = matmul_precision_before
    compile_info["matmul_precision_after"] = matmul_precision_after
    compile_info["model_cost_autocast"] = args.model_cost_autocast
    compile_info["model_cost_autocast_enabled"] = model_cost_autocast_info["enabled"]
    compile_info["model_cost_autocast_dtype"] = model_cost_autocast_info["dtype"]
    solver = hydra_module.utils.instantiate(cfg.solver, model=model)
    static_chain_compile_enabled = bool(
        _cfg_get(cfg.solver, "static_chain_compile", False)
    )
    instrumentation = ModelInstrumentation(
        model,
        enabled=not args.disable_instrumentation,
        compile_cost_enabled=compile_enabled or static_chain_compile_enabled,
    )
    instrumentation.install()
    append_jsonl(metrics_path, model_cost_autocast_info)
    append_jsonl(metrics_path, compile_info)
    diagnostic_signature = compile_info.get("warmup_signature")
    compile_diagnostics = CompileDiagnostics(
        metrics_path=metrics_path,
        seed=seed,
        enabled=compile_diagnostics_mode != "none" and compile_enabled,
        expected_signature=diagnostic_signature,
        max_signature_records=args.max_compile_signature_records,
    )
    compile_diagnostics.emit_event(
        "warmup_complete",
        phase="compile_warmup",
        signature=compile_info.get("warmup_signature"),
        elapsed_s=compile_info.get("warmup_elapsed_s"),
        extra={
            "compile_target": "cost_helper",
            "compile_mode": compile_info.get("compile_mode"),
            "fullgraph": compile_info.get("fullgraph"),
        },
    )
    icem_step_telemetry_installed = install_icem_step_telemetry(
        solver,
        args.icem_step_telemetry,
    )
    append_jsonl(
        metrics_path,
        {
            "type": "model_info",
            "seed": seed,
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "policy": args.policy,
            "solver": args.solver,
            "embedding_cache_enabled": bool(
                getattr(model, "embedding_cache_enabled", lambda: False)()
            ),
            "rollout_prealloc_enabled": bool(
                getattr(model, "rollout_prealloc_enabled", lambda: False)()
            ),
            "rollout_action_cache_enabled": bool(
                getattr(model, "rollout_action_cache_enabled", lambda: False)()
            ),
            "compile_cost_enabled": bool(
                getattr(model, "compile_cost_enabled", lambda: False)()
            ),
            "model_cost_autocast": args.model_cost_autocast,
            "model_cost_autocast_enabled": model_cost_autocast_info["enabled"],
            "model_cost_autocast_dtype": model_cost_autocast_info["dtype"],
            "planner": planner_settings,
        },
    )

    cem_telemetry = CEMTelemetry(
        metrics_path=metrics_path,
        seed=seed,
        mode=args.cem_telemetry,
        warmup_plans=args.warmup_plans,
        max_recorded_calls=args.max_recorded_cem_calls,
        compile_diagnostics=compile_diagnostics,
    )
    cem_profiler_ranges = install_cem_telemetry(
        solver,
        cem_telemetry,
        force_ranges=args.profile == "torch",
    )
    append_jsonl(
        metrics_path,
        {
            "type": "profiler_info",
            "seed": seed,
            "cem_profiler_ranges": cem_profiler_ranges,
            "cem_telemetry_mode": args.cem_telemetry,
            "icem_step_telemetry_mode": args.icem_step_telemetry,
            "icem_step_telemetry_installed": icem_step_telemetry_installed,
            "max_recorded_cem_calls": args.max_recorded_cem_calls,
            "torch_profiler_with_stack": args.torch_profiler_with_stack,
            "compile_cost_enabled": compile_enabled,
            "compile_diagnostics_mode": compile_diagnostics_mode,
            "max_compile_signature_records": args.max_compile_signature_records,
            "torch_compile_logs": args.torch_compile_logs,
            "matmul_precision_requested": args.matmul_precision,
            "matmul_precision_before": matmul_precision_before,
            "matmul_precision_after": matmul_precision_after,
            "model_cost_autocast": args.model_cost_autocast,
            "model_cost_autocast_enabled": model_cost_autocast_info["enabled"],
            "model_cost_autocast_dtype": model_cost_autocast_info["dtype"],
            "planner": planner_settings,
        },
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=context["config"],
        process=context["process"],
        transform=context["transform"],
    )
    recorder = LatencyRecorder(
        metrics_path=metrics_path,
        seed=seed,
        profile=args.profile,
        warmup_plans=args.warmup_plans,
        max_recorded_plans=args.max_recorded_plans,
        instrumentation=instrumentation,
        cem_telemetry=cem_telemetry,
    )
    world.set_policy(TimedPolicy(policy, recorder))
    static_chain_info = maybe_warmup_static_chain_solver(
        solver,
        model,
        history=args.compile_warmup_history,
    )
    static_chain_info["seed"] = seed
    append_jsonl(metrics_path, static_chain_info)

    eval_episodes, eval_start_idx = select_eval_rows(
        cfg, dataset, context["ep_indices"], seed
    )
    started = time.perf_counter()
    compile_diagnostics.emit_event(
        "evaluation_start",
        phase="evaluation",
    )
    with maybe_torch_profiler(
        args.profile,
        run_dir,
        with_stack=args.torch_profiler_with_stack,
    ) as profiler:
        recorder.profiler = profiler
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=run_dir / "videos",
        )
        recorder.profiler = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compile_diagnostics.emit_event(
        "evaluation_end",
        phase="evaluation",
        extra={
            "plan_calls_total": recorder.plan_index,
            "cem_calls_total": cem_telemetry.calls,
        },
    )
    append_jsonl(
        metrics_path,
        {
            "type": "evaluation_result",
            "seed": seed,
            "metrics": metrics,
            "evaluation_time_s": time.perf_counter() - started,
            "encode_counts_total": instrumentation.snapshot(),
            "plan_calls_total": recorder.plan_index,
            "cem_calls_total": cem_telemetry.calls,
            "cem_cpu_wall_ms_total": cem_telemetry.cpu_wall_ms,
            "cem_cuda_elapsed_ms_total": cem_telemetry.cuda_elapsed_ms,
            "model_cost_evals_total": cem_telemetry.model_cost_evals,
            "model_cost_env_candidate_evals_total": (
                cem_telemetry.model_cost_env_candidate_evals
            ),
            "gd_model_cost_evals_total": cem_telemetry.gd_model_cost_evals,
            "gd_model_cost_env_candidate_evals_total": (
                cem_telemetry.gd_model_cost_env_candidate_evals
            ),
            "model_cost_evals_total_including_gd": (
                cem_telemetry.model_cost_evals + cem_telemetry.gd_model_cost_evals
            ),
            "model_cost_env_candidate_evals_total_including_gd": (
                cem_telemetry.model_cost_env_candidate_evals
                + cem_telemetry.gd_model_cost_env_candidate_evals
            ),
            "plans_per_episode": recorder.plan_index / args.episodes if args.episodes else None,
            "cem_calls_per_episode": (
                cem_telemetry.calls / args.episodes if args.episodes else None
            ),
            "samples": planner_settings.get("samples"),
            "cem_steps": planner_settings.get("cem_steps"),
            "horizon": planner_settings.get("horizon"),
            "receding_horizon": planner_settings.get("receding_horizon"),
            "action_block": planner_settings.get("action_block"),
            "topk": planner_settings.get("topk"),
            "warm_start": planner_settings.get("warm_start"),
            "solver_warm_start": planner_settings.get("solver_warm_start"),
            "elite_retention": planner_settings.get("elite_retention"),
            "colored_noise": planner_settings.get("colored_noise"),
            "colored_noise_time": planner_settings.get("colored_noise_time"),
            "momentum_alpha": planner_settings.get("momentum_alpha"),
            "sample_decay": planner_settings.get("sample_decay"),
            "sample_count_bucket_size": planner_settings.get("sample_count_bucket_size"),
            "return_best_elite": planner_settings.get("return_best_elite"),
            "clip_actions": planner_settings.get("clip_actions"),
            "mppi_temperature": planner_settings.get("mppi_temperature"),
            "gd_seed_solver": planner_settings.get("gd_seed_solver"),
            "gd_seed_source": planner_settings.get("gd_seed_source"),
            "gd_refine_topk": planner_settings.get("gd_refine_topk"),
            "gd_steps": planner_settings.get("gd_steps"),
            "gd_lr": planner_settings.get("gd_lr"),
            "gd_compile_cost": planner_settings.get("gd_compile_cost"),
            "static_chain_compile": planner_settings.get("static_chain_compile"),
            "static_chain_mode": planner_settings.get("static_chain_mode"),
            "static_chain_fullgraph": planner_settings.get("static_chain_fullgraph"),
            "static_chain_warmup": planner_settings.get("static_chain_warmup"),
            "matmul_precision_requested": args.matmul_precision,
            "matmul_precision_before": matmul_precision_before,
            "matmul_precision_after": matmul_precision_after,
            "model_cost_autocast": args.model_cost_autocast,
            "model_cost_autocast_enabled": model_cost_autocast_info["enabled"],
            "model_cost_autocast_dtype": model_cost_autocast_info["dtype"],
            "planner": planner_settings,
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pusht.yaml")
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--solver", default="cem")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--semantic-class", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--cem-steps", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--receding-horizon", type=int, default=None)
    parser.add_argument("--action-block", type=int, default=None)
    parser.add_argument("--warm-start", choices=["true", "false"], default=None)
    parser.add_argument("--solver-warm-start", choices=["true", "false"], default=None)
    parser.add_argument("--elite-retention", type=int, default=None)
    parser.add_argument("--colored-noise", choices=["none", "pink", "brown"], default=None)
    parser.add_argument("--colored-noise-time", choices=["block", "raw"], default=None)
    parser.add_argument("--momentum-alpha", type=float, default=None)
    parser.add_argument("--sample-decay", type=float, default=None)
    parser.add_argument("--sample-count-bucket-size", type=int, default=None)
    parser.add_argument("--return-best-elite", choices=["true", "false"], default=None)
    parser.add_argument("--clip-actions", choices=["true", "false"], default=None)
    parser.add_argument("--mppi-temperature", type=float, default=None)
    parser.add_argument("--gd-seed-solver", choices=["cem", "icem"], default=None)
    parser.add_argument("--gd-seed-source", choices=["best_elite", "mean"], default=None)
    parser.add_argument("--gd-refine-topk", type=int, choices=[1, 3, 5], default=None)
    parser.add_argument("--gd-steps", type=int, default=None)
    parser.add_argument("--gd-lr", type=float, default=None)
    parser.add_argument("--gd-compile-cost", choices=["true", "false"], default=None)
    parser.add_argument("--static-chain-compile", choices=["true", "false"], default=None)
    parser.add_argument(
        "--static-chain-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default=None,
    )
    parser.add_argument("--static-chain-fullgraph", choices=["true", "false"], default=None)
    parser.add_argument("--static-chain-warmup", choices=["true", "false"], default=None)
    parser.add_argument("--baseline-run-id", default=None)
    parser.add_argument(
        "--quality-gate-result",
        choices=["pass", "fail", "not_applicable", "diagnostic"],
        default=None,
    )
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--profile", choices=["none", "cuda-events", "torch"], default="cuda-events")
    parser.add_argument(
        "--cem-telemetry",
        choices=["none", "summary", "cuda-events"],
        default="none",
        help=(
            "none leaves the solver path unpatched unless --profile torch is active; "
            "summary writes per-CEM-call CPU/region records; cuda-events also records "
            "CEM-level CUDA elapsed time and memory snapshots."
        ),
    )
    parser.add_argument(
        "--icem-step-telemetry",
        choices=["none", "summary"],
        default="none",
        help=(
            "Attach compact per-step iCEM convergence summaries to emitted "
            "cem_call rows. Only valid with --solver icem."
        ),
    )
    parser.add_argument("--warmup-plans", type=int, default=2)
    parser.add_argument("--max-recorded-plans", type=int, default=None)
    parser.add_argument("--max-recorded-cem-calls", type=int, default=None)
    parser.add_argument(
        "--compile-diagnostics",
        choices=["none", "summary"],
        default=None,
        help=(
            "Record compile/capture diagnostics. Defaults to summary for "
            "compile_cost runs and none otherwise."
        ),
    )
    parser.add_argument(
        "--compile-warmup-history",
        type=int,
        default=1,
        help=(
            "Embedding history length used for compile_cost warmup. Push-T runtime "
            "CEM currently starts from one current-observation embedding."
        ),
    )
    parser.add_argument(
        "--compile-warmup-sample-counts",
        default=None,
        help=(
            "Comma-separated sample counts or 'auto' for compile_cost warmup. "
            "Default warms only solver.num_samples. Use 'auto' with bucketed iCEM "
            "to prewarm each unique runtime sample-count shape."
        ),
    )
    parser.add_argument("--max-compile-signature-records", type=int, default=8)
    parser.add_argument(
        "--torch-compile-logs",
        default=None,
        help=(
            "Optional TORCH_LOGS value for PyTorch compiler diagnostics, for "
            "example 'recompiles'. Prefer this for focused debug runs, not broad "
            "timing sweeps."
        ),
    )
    parser.add_argument(
        "--matmul-precision",
        choices=MATMUL_PRECISION_CHOICES,
        default="default",
        help=(
            "Optional torch.set_float32_matmul_precision value. Use high for the "
            "Phase 3R TF32/model-cost numerical candidate."
        ),
    )
    parser.add_argument(
        "--model-cost-autocast",
        choices=MODEL_COST_AUTOCAST_CHOICES,
        default="none",
        help=(
            "Optional autocast scope for JEPA model-cost evaluation only. "
            "BF16 leaves planner updates and action output in their normal dtype."
        ),
    )
    parser.add_argument("--torch-profiler-with-stack", action="store_true")
    parser.add_argument(
        "--compile-cost",
        choices=["true", "false"],
        default=None,
        help=(
            "Compile the tensor-only LeWM model-cost helper. Defaults to true for "
            "variant=compile_cost/model_cost_tf32 and false otherwise."
        ),
    )
    parser.add_argument("--disable-instrumentation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_variant_solver_contract(args)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.torch_compile_logs:
        os.environ["TORCH_LOGS"] = args.torch_compile_logs
    if args.weights_root:
        os.environ["STABLEWM_HOME"] = args.weights_root
    sys.path.insert(0, str(LEWM_DIR))

    seeds = parse_seed_list(args.seed, args.seeds)
    semantic_class = args.semantic_class or SEMANTIC_CLASSES.get(args.variant, "exact")
    run_id = args.run_id or default_run_id("lewm-pusht", args.variant, seeds[0] if len(seeds) == 1 else None)
    results_root = Path(args.results_root)
    run_dir = results_root / "runs" / run_id if results_root.name != run_id else results_root
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.out) if args.out else run_dir / "metrics.jsonl"
    manifest_path = run_dir / "manifest.json"
    prepare_run_artifacts(run_dir, metrics_path, manifest_path, profile=args.profile)

    weights_root = Path(args.weights_root or os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    weights_paths = policy_object_candidates(weights_root, args.policy)
    weights_sha = "unknown"
    for path in weights_paths:
        weights_sha = precomputed_hash(results_root, path)
        if weights_sha != "unknown":
            break
    if weights_sha == "unknown":
        weights_sha = first_existing_hash(weights_paths)
    dataset_sha = "unknown"

    for seed in seeds:
        run_seed(args, seed, run_dir, metrics_path)

    rows = read_jsonl(metrics_path)
    summary = {
        "plan_latency_ms": summarize_plan_latencies(rows),
        "cem_calls": summarize_cem_calls(rows),
        "evaluations": summarize_evaluations(rows),
    }
    profiler_summary_path = None
    if args.profile == "torch":
        try:
            from tools.benchmarks.summarize_torch_trace import summarize_trace_path, write_summary

            trace_summary = summarize_trace_path(run_dir / "profiler")
            summary["torch_profiler"] = trace_summary
            if trace_summary.get("trace_files"):
                profiler_summary_path = run_dir / "profiler_summary.json"
                write_summary(profiler_summary_path, trace_summary)
        except Exception as exc:
            summary["torch_profiler"] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
    try:
        last_cfg, _ = compose_config(
            args.config_name,
            args.policy,
            args.solver,
            args.episodes,
            seeds[0],
            planner_overrides=args,
        )
        dataset_paths = dataset_candidates(weights_root, last_cfg.eval.dataset_name)
        for path in dataset_paths:
            dataset_sha = precomputed_hash(results_root, path)
            if dataset_sha != "unknown":
                break
        if dataset_sha == "unknown":
            dataset_sha = first_existing_hash(dataset_paths)
        planner = {
            **effective_planner_settings(args, last_cfg),
            "profile": args.profile,
            "cem_telemetry": args.cem_telemetry,
            "torch_profiler_with_stack": args.torch_profiler_with_stack,
            "compile_cost": compile_cost_requested(args),
            "compile_diagnostics": effective_compile_diagnostics(args),
            "compile_warmup_history": args.compile_warmup_history,
            "compile_warmup_sample_counts": args.compile_warmup_sample_counts,
            "max_compile_signature_records": args.max_compile_signature_records,
            "torch_compile_logs": args.torch_compile_logs,
            "matmul_precision": args.matmul_precision,
            "model_cost_autocast": args.model_cost_autocast,
        }
    except Exception:
        planner = {}

    manifest = build_manifest(
        run_id=run_id,
        model="lewm-pusht",
        variant=args.variant,
        semantic_class=semantic_class,
        cwd=REPO_ROOT,
        command=sys.argv if argv is None else ["lewm_plan_benchmark.py", *argv],
        weights_sha256=weights_sha,
        dataset_sha256=dataset_sha,
        code_paths=CODE_PATHS.get(args.variant, []),
        planner=planner,
        modal={
            "enabled": results_root.is_absolute()
            and len(results_root.parts) > 1
            and results_root.parts[1] == "results",
            "gpu": os.environ.get("JEPA_MODAL_GPU_TYPE")
            or (
                "L40S"
                if results_root.is_absolute()
                and len(results_root.parts) > 1
                and results_root.parts[1] == "results"
                else None
            ),
            "cpu_request": os.environ.get("JEPA_MODAL_CPU_REQUEST"),
            "cpu_limit": os.environ.get("JEPA_MODAL_CPU_LIMIT"),
            "weights_volume": "jepa-weights" if str(weights_root).startswith("/weights") else None,
            "results_volume": "jepa-results" if str(results_root).startswith("/results") else None,
        },
        artifacts={
            "metrics_jsonl": str(metrics_path),
            "profiler_trace": str(run_dir / "profiler") if args.profile == "torch" else None,
            "profiler_summary": str(profiler_summary_path) if profiler_summary_path else None,
            "stdout": str(run_dir / "stdout.txt"),
            "stderr": str(run_dir / "stderr.txt"),
        },
        metrics_summary=summary,
        quality_gate_result=args.quality_gate_result,
        seed_list=seeds,
        num_eval_episodes=args.episodes,
        baseline_run_id=args.baseline_run_id,
    )
    write_manifest(manifest_path, manifest)
    print(f"wrote {metrics_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
