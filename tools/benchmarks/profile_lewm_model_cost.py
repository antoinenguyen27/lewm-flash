"""Profile the tensor-only LeWM model-cost path at a fixed candidate shape."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.lewm_plan_benchmark import (
    CODE_PATHS,
    LEWM_DIR,
    REPO_ROOT,
    SEMANTIC_CLASSES,
    _first_parameter,
    _infer_compile_action_dim,
    _infer_compile_embed_dim,
    compile_cost_signature,
    configure_model_for_variant,
    dataset_candidates,
    diff_numeric_trees,
    first_existing_hash,
    load_policy_model,
    maybe_torch_profiler,
    policy_object_candidates,
    precomputed_hash,
    torch_compile_counter_snapshot,
)
from tools.benchmarks.metrics import append_jsonl, json_safe, read_jsonl, summarize_numeric
from tools.benchmarks.run_manifest import (
    build_manifest,
    default_run_id,
    discover_runtime_resources,
    write_manifest,
)


MODEL_COST_CODE_PATHS = [
    *CODE_PATHS.get("compile_cost", []),
    "tools/benchmarks/profile_lewm_model_cost.py",
    "tools/benchmarks/lewm_plan_benchmark.py",
]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def validate_profile_shape(
    *,
    batch_size: int,
    samples: int,
    horizon: int,
    action_dim: int,
    require_phase3r_shape: bool,
) -> None:
    shape = [batch_size, samples, horizon, action_dim]
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"profile dimensions must be positive, got {shape}")
    if require_phase3r_shape and shape[1:] != [300, 5, 10]:
        raise ValueError(
            "Phase 3R control requires candidate shape [B, 300, 5, 10], "
            f"got {shape}"
        )


def relative_run_artifact(path: Path) -> str:
    return str(path)


def summarize_model_cost_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row.get("type") == "model_cost_iteration" and not row.get("warmup", False)
    ]
    return {
        "iteration_count": len(measured),
        "elapsed_ms": summarize_numeric(measured, "elapsed_ms"),
        "timing_source": sorted(
            {
                str(row["timing_source"])
                for row in measured
                if row.get("timing_source") is not None
            }
        ),
    }


def summarize_score_reduction_gate(
    torch_profiler_summary: dict[str, Any],
    *,
    model_cost_range: str = "lewm.model_cost_profile.get_cost_from_embeddings",
    score_range: str = "lewm.cost_from_rollout",
    model_cost_threshold: float = 0.10,
) -> dict[str, Any]:
    ranges = torch_profiler_summary.get("user_ranges") or {}
    model_cost = ranges.get(model_cost_range) or {}
    score = ranges.get(score_range) or {}
    model_cost_total_ms = model_cost.get("total_ms")
    score_total_ms = score.get("total_ms")
    ratio = None
    gate_pass = None
    if isinstance(model_cost_total_ms, (int, float)) and model_cost_total_ms > 0:
        if isinstance(score_total_ms, (int, float)):
            ratio = float(score_total_ms) / float(model_cost_total_ms)
            gate_pass = ratio >= model_cost_threshold
    return {
        "model_cost_range": model_cost_range,
        "score_range": score_range,
        "model_cost_total_ms": model_cost_total_ms,
        "score_total_ms": score_total_ms,
        "score_to_model_cost_ratio": ratio,
        "model_cost_threshold": model_cost_threshold,
        "passes_model_cost_gate": gate_pass,
    }


def summarize_model_cost_region_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row.get("type") == "model_cost_iteration" and not row.get("warmup", False)
    ]
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in measured:
        regions = row.get("region_cuda_ms") or {}
        if not isinstance(regions, dict):
            continue
        for name, value in regions.items():
            if isinstance(value, (int, float)):
                key = str(name)
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {
        name: {
            "count": counts[name],
            "total_ms": totals[name],
            "mean_ms": totals[name] / counts[name],
        }
        for name in sorted(totals)
    }


def summarize_score_reduction_cuda_gate(
    rows: list[dict[str, Any]],
    *,
    score_range: str = "lewm.cost_from_rollout",
    model_cost_threshold: float = 0.10,
) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row.get("type") == "model_cost_iteration" and not row.get("warmup", False)
    ]
    total_elapsed_ms = sum(
        float(row["elapsed_ms"])
        for row in measured
        if isinstance(row.get("elapsed_ms"), (int, float))
    )
    score_total_ms = 0.0
    score_count = 0
    for row in measured:
        regions = row.get("region_cuda_ms") or {}
        if not isinstance(regions, dict):
            continue
        value = regions.get(score_range)
        if isinstance(value, (int, float)):
            score_total_ms += float(value)
            score_count += 1
    ratio = None
    gate_pass = None
    if total_elapsed_ms > 0 and score_count:
        ratio = score_total_ms / total_elapsed_ms
        gate_pass = ratio >= model_cost_threshold
    return {
        "timing_source": "cuda_events",
        "score_range": score_range,
        "measured_iterations": len(measured),
        "score_iterations": score_count,
        "model_cost_total_ms": total_elapsed_ms,
        "score_total_ms": score_total_ms if score_count else None,
        "score_to_model_cost_ratio": ratio,
        "model_cost_threshold": model_cost_threshold,
        "passes_model_cost_gate": gate_pass,
    }


def _mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _clean_run_artifacts(run_dir: Path, metrics_path: Path, manifest_path: Path) -> None:
    for path in (
        metrics_path,
        manifest_path,
        run_dir / "model_cost_profile.json",
        run_dir / "profiler_summary.json",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    profiler_dir = run_dir / "profiler"
    if profiler_dir.exists():
        import shutil

        shutil.rmtree(profiler_dir)


def _shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def _model_flag(model: Any, name: str) -> bool:
    attr = getattr(model, name, None)
    if callable(attr):
        return bool(attr())
    return False


def _build_inputs(
    *,
    model: Any,
    batch_size: int,
    samples: int,
    horizon: int,
    history: int,
    action_dim: int | None,
    seed: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    parameter = _first_parameter(model)
    device = parameter.device
    dtype = parameter.dtype
    embed_dim = _infer_compile_embed_dim(model)
    inferred_action_dim = _infer_compile_action_dim(model)
    actual_action_dim = action_dim or inferred_action_dim
    if action_dim is not None and action_dim != inferred_action_dim:
        raise ValueError(
            f"--action-dim {action_dim} does not match model action dimension "
            f"{inferred_action_dim}"
        )
    validate_profile_shape(
        batch_size=batch_size,
        samples=samples,
        horizon=horizon,
        action_dim=actual_action_dim,
        require_phase3r_shape=True,
    )
    if history <= 0:
        raise ValueError(f"--history must be positive, got {history}")
    if history > horizon:
        raise ValueError(f"--history must be <= --horizon, got {history} > {horizon}")

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    emb = torch.randn((batch_size, history, embed_dim), generator=gen, device=device, dtype=dtype)
    goal_emb = torch.randn(emb.shape, generator=gen, device=device, dtype=dtype)
    action_candidates = torch.randn(
        (batch_size, samples, horizon, actual_action_dim),
        generator=gen,
        device=device,
        dtype=dtype,
    )
    info = {
        "embed_dim": embed_dim,
        "action_dim": actual_action_dim,
        "inferred_action_dim": inferred_action_dim,
        "dtype": str(dtype),
        "device": str(device),
        "signature": compile_cost_signature(
            emb=emb,
            goal_emb=goal_emb,
            action_candidates=action_candidates,
        ),
    }
    return emb, goal_emb, action_candidates, info


@contextlib.contextmanager
def _temporary_profiled_method(obj: Any, method_name: str, label: str):
    import torch

    original = getattr(obj, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapped)
    try:
        yield
    finally:
        setattr(obj, method_name, original)


def _profiled_get_cost_from_embeddings(
    model: Any,
    emb: Any,
    goal_emb: Any,
    action_candidates: Any,
) -> Any:
    import torch

    with (
        _temporary_profiled_method(model.action_encoder, "forward", "lewm.action_encoder"),
        _temporary_profiled_method(model, "predict", "lewm.predict_step"),
    ):
        with torch.profiler.record_function("lewm.rollout_from_embeddings"):
            pred_rollout = model.rollout_from_embeddings(emb, action_candidates)
        with torch.profiler.record_function("lewm.cost_from_rollout"):
            return model.cost_from_rollout(pred_rollout, goal_emb)


def _install_compile_cost(
    model: Any,
    *,
    emb: Any,
    goal_emb: Any,
    action_candidates: Any,
    mode: str,
    fullgraph: bool,
    allow_fallback: bool,
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

    original = model.get_cost_from_embeddings
    compile_mode = None if mode == "default" else mode
    compiled = torch.compile(original, mode=compile_mode, fullgraph=fullgraph)
    model.get_cost_from_embeddings = compiled
    model.set_compile_cost(True)

    counter_before = torch_compile_counter_snapshot()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            warmup_cost = model.get_cost_from_embeddings(emb, goal_emb, action_candidates)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fallback_error = None
    except RuntimeError as exc:
        if not allow_fallback:
            raise
        model.get_cost_from_embeddings = original
        model.set_compile_cost(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        with torch.inference_mode():
            warmup_cost = model.get_cost_from_embeddings(emb, goal_emb, action_candidates)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fallback_error = f"{type(exc).__name__}: {exc}"
    counter_after = torch_compile_counter_snapshot()

    return {
        "type": "compile_info",
        "compile_cost_requested": True,
        "compile_cost_enabled": bool(_model_flag(model, "compile_cost_enabled")),
        "compile_mode": mode,
        "fullgraph": fullgraph,
        "allow_fallback": allow_fallback,
        "fallback_used": fallback_error is not None,
        "fallback_error": fallback_error,
        "warmup_elapsed_s": time.perf_counter() - started,
        "warmup_signature": compile_cost_signature(
            emb=emb,
            goal_emb=goal_emb,
            action_candidates=action_candidates,
        ),
        "warmup_counter_delta": diff_numeric_trees(counter_before, counter_after) or {},
        "warmup_counter_snapshot_available": bool(counter_after.get("available")),
        "warmup_counter_sources": list((counter_after.get("sources") or {}).keys()),
        "warmup_counter_errors": counter_after.get("errors") or {},
        "warmup_shape": {
            "emb": _shape(emb),
            "goal_emb": _shape(goal_emb),
            "action_candidates": _shape(action_candidates),
            "cost": _shape(warmup_cost),
        },
    }


def _measure_once(
    model: Any,
    *,
    emb: Any,
    goal_emb: Any,
    action_candidates: Any,
    profile: str,
    model_cost_region_profile: bool,
) -> tuple[Any, float, str, dict[str, float]]:
    import torch

    use_cuda_events = profile in {"cuda-events", "torch"} and torch.cuda.is_available()
    if use_cuda_events:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.profiler.record_function("lewm.model_cost_profile.get_cost_from_embeddings"):
            if model_cost_region_profile:
                rollout_start = torch.cuda.Event(enable_timing=True)
                rollout_end = torch.cuda.Event(enable_timing=True)
                score_start = torch.cuda.Event(enable_timing=True)
                score_end = torch.cuda.Event(enable_timing=True)
                with (
                    _temporary_profiled_method(
                        model.action_encoder,
                        "forward",
                        "lewm.action_encoder",
                    ),
                    _temporary_profiled_method(model, "predict", "lewm.predict_step"),
                ):
                    rollout_start.record()
                    with torch.profiler.record_function("lewm.rollout_from_embeddings"):
                        pred_rollout = model.rollout_from_embeddings(emb, action_candidates)
                    rollout_end.record()
                    score_start.record()
                    with torch.profiler.record_function("lewm.cost_from_rollout"):
                        cost = model.cost_from_rollout(pred_rollout, goal_emb)
                    score_end.record()
            else:
                cost = model.get_cost_from_embeddings(emb, goal_emb, action_candidates)
        end.record()
        torch.cuda.synchronize()
        region_cuda_ms = {}
        if model_cost_region_profile:
            region_cuda_ms = {
                "lewm.rollout_from_embeddings": float(rollout_start.elapsed_time(rollout_end)),
                "lewm.cost_from_rollout": float(score_start.elapsed_time(score_end)),
            }
        return cost, float(start.elapsed_time(end)), "cuda_events", region_cuda_ms

    started = time.perf_counter()
    with torch.profiler.record_function("lewm.model_cost_profile.get_cost_from_embeddings"):
        if model_cost_region_profile:
            cost = _profiled_get_cost_from_embeddings(model, emb, goal_emb, action_candidates)
        else:
            cost = model.get_cost_from_embeddings(emb, goal_emb, action_candidates)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return cost, (time.perf_counter() - started) * 1000.0, "perf_counter", {}


def run_profile(args: argparse.Namespace, run_dir: Path, metrics_path: Path) -> dict[str, Any]:
    import stable_worldmodel as swm
    import torch

    from tools.benchmarks.lewm_plan_benchmark import compose_config

    if args.torch_compile_logs:
        os.environ["TORCH_LOGS"] = args.torch_compile_logs
    if args.weights_root:
        os.environ["STABLEWM_HOME"] = args.weights_root
    sys.path.insert(0, str(LEWM_DIR))

    matmul_precision_before = torch.get_float32_matmul_precision()
    if args.matmul_precision != "default":
        torch.set_float32_matmul_precision(args.matmul_precision)
    matmul_precision_after = torch.get_float32_matmul_precision()

    cfg, _hydra = compose_config(args.config_name, args.policy, args.solver, 0, args.seed)
    weights_root = Path(args.weights_root or os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    model = load_policy_model(swm, weights_root, args.policy)
    configure_model_for_variant(model, args.variant)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    emb, goal_emb, action_candidates, input_info = _build_inputs(
        model=model,
        batch_size=args.batch_size,
        samples=args.samples,
        horizon=args.horizon,
        history=args.history,
        action_dim=args.action_dim,
        seed=args.seed,
    )

    append_jsonl(
        metrics_path,
        {
            "type": "runtime_resource_info",
            "seed": args.seed,
            "resources": discover_runtime_resources(),
        },
    )
    append_jsonl(
        metrics_path,
        {
            "type": "model_info",
            "seed": args.seed,
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "policy": args.policy,
            "solver": args.solver,
            "variant": args.variant,
            "embedding_cache_enabled": _model_flag(model, "embedding_cache_enabled"),
            "rollout_prealloc_enabled": _model_flag(model, "rollout_prealloc_enabled"),
            "rollout_action_cache_enabled": _model_flag(model, "rollout_action_cache_enabled"),
            "compile_cost_enabled_before_install": _model_flag(model, "compile_cost_enabled"),
        },
    )

    compile_info = _install_compile_cost(
        model,
        emb=emb,
        goal_emb=goal_emb,
        action_candidates=action_candidates,
        mode=args.compile_cost_mode,
        fullgraph=args.compile_cost_fullgraph,
        allow_fallback=args.compile_cost_allow_fallback,
    )
    compile_info["seed"] = args.seed
    append_jsonl(metrics_path, compile_info)

    profiler_info = {
        "type": "profiler_info",
        "seed": args.seed,
        "profile": args.profile,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iters,
        "torch_profiler_with_stack": args.torch_profiler_with_stack,
        "model_cost_region_profile": args.model_cost_region_profile,
        "model_cost_region_profile_compiled": False if args.model_cost_region_profile else None,
        "matmul_precision_before": matmul_precision_before,
        "matmul_precision_after": matmul_precision_after,
        "torch_compile_logs": args.torch_compile_logs,
    }
    append_jsonl(metrics_path, profiler_info)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_before = {
            "allocated_mb": _mb(torch.cuda.memory_allocated()),
            "reserved_mb": _mb(torch.cuda.memory_reserved()),
        }
    else:
        memory_before = {"allocated_mb": None, "reserved_mb": None}

    last_cost = None
    with torch.inference_mode():
        for iteration in range(args.warmup):
            with torch.profiler.record_function("lewm.model_cost_profile.warmup_iteration"):
                last_cost, elapsed_ms, timing_source, region_cuda_ms = _measure_once(
                    model,
                    emb=emb,
                    goal_emb=goal_emb,
                    action_candidates=action_candidates,
                    profile=args.profile,
                    model_cost_region_profile=args.model_cost_region_profile,
                )
                row = {
                    "type": "model_cost_iteration",
                    "seed": args.seed,
                    "iteration": iteration + 1,
                    "warmup": True,
                    "elapsed_ms": elapsed_ms,
                    "timing_source": timing_source,
                }
                if region_cuda_ms:
                    row["region_cuda_ms"] = region_cuda_ms
                append_jsonl(metrics_path, row)

        with maybe_torch_profiler(
            args.profile,
            run_dir,
            with_stack=args.torch_profiler_with_stack,
        ) as profiler:
            for iteration in range(args.iters):
                with torch.profiler.record_function("lewm.model_cost_profile.iteration"):
                    last_cost, elapsed_ms, timing_source, region_cuda_ms = _measure_once(
                        model,
                        emb=emb,
                        goal_emb=goal_emb,
                        action_candidates=action_candidates,
                        profile=args.profile,
                        model_cost_region_profile=args.model_cost_region_profile,
                    )
                row = {
                    "type": "model_cost_iteration",
                    "seed": args.seed,
                    "iteration": iteration + 1,
                    "warmup": False,
                    "elapsed_ms": elapsed_ms,
                    "timing_source": timing_source,
                }
                if region_cuda_ms:
                    row["region_cuda_ms"] = region_cuda_ms
                append_jsonl(metrics_path, row)
                if profiler is not None:
                    profiler.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_after = {
            "allocated_mb": _mb(torch.cuda.memory_allocated()),
            "reserved_mb": _mb(torch.cuda.memory_reserved()),
            "peak_allocated_mb": _mb(torch.cuda.max_memory_allocated()),
        }
    else:
        memory_after = {
            "allocated_mb": None,
            "reserved_mb": None,
            "peak_allocated_mb": None,
        }

    if last_cost is None:
        raise RuntimeError("No model-cost iterations were executed")
    checksum = {
        "cost_shape": _shape(last_cost),
        "cost_mean": float(last_cost.detach().float().mean().item()),
        "cost_min": float(last_cost.detach().float().min().item()),
        "cost_max": float(last_cost.detach().float().max().item()),
    }

    rows = read_jsonl(metrics_path)
    summary = {
        "model_cost": summarize_model_cost_rows(rows),
        "memory": {
            "before": memory_before,
            "after": memory_after,
        },
    }
    if args.model_cost_region_profile:
        summary["model_cost_regions"] = summarize_model_cost_region_rows(rows)
        summary["score_reduction_cuda_gate"] = summarize_score_reduction_cuda_gate(rows)
    profile_payload = {
        "run_id": args.run_id,
        "seed": args.seed,
        "policy": args.policy,
        "solver": args.solver,
        "variant": args.variant,
        "semantic_class": args.semantic_class,
        "profile": args.profile,
        "shape": {
            "batch_size": args.batch_size,
            "samples": args.samples,
            "horizon": args.horizon,
            "history": args.history,
            "action_dim": input_info["action_dim"],
            "candidate_shape": _shape(action_candidates),
        },
        "input_signature": input_info["signature"],
        "dtype": input_info["dtype"],
        "device": input_info["device"],
        "compile": compile_info,
        "matmul_precision": {
            "before": matmul_precision_before,
            "after": matmul_precision_after,
            "requested": args.matmul_precision,
        },
        "iterations": {
            "warmup": args.warmup,
            "measured": args.iters,
        },
        "model_cost_region_profile": {
            "enabled": args.model_cost_region_profile,
            "compiled": False if args.model_cost_region_profile else None,
            "note": (
                "diagnostic attribution path; bypasses the compiled cost callable "
                "to preserve internal profiler ranges"
                if args.model_cost_region_profile
                else None
            ),
        },
        "summary": summary,
        "checksum": checksum,
    }
    return json_safe(profile_payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pusht.yaml")
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--solver", default="cem_exact")
    parser.add_argument("--variant", default="compile_cost")
    parser.add_argument("--semantic-class", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--profile", choices=["none", "cuda-events", "torch"], default="torch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--matmul-precision",
        choices=["default", "highest", "high", "medium"],
        default="default",
    )
    parser.add_argument(
        "--compile-cost-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument("--compile-cost-fullgraph", type=parse_bool, default=False)
    parser.add_argument("--compile-cost-allow-fallback", action="store_true")
    parser.add_argument(
        "--model-cost-region-profile",
        action="store_true",
        help=(
            "Use a diagnostic uncompiled wrapper that records rollout, score, "
            "action_encoder, and predict_step profiler ranges."
        ),
    )
    parser.add_argument("--torch-profiler-with-stack", action="store_true")
    parser.add_argument("--torch-compile-logs", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_profile_shape(
        batch_size=args.batch_size,
        samples=args.samples,
        horizon=args.horizon,
        action_dim=args.action_dim or 10,
        require_phase3r_shape=True,
    )
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}")
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    if args.variant != "compile_cost":
        raise ValueError(
            "profile_lewm_model_cost.py is the Phase 3R compile_cost control; "
            f"got --variant {args.variant!r}"
        )
    if args.torch_compile_logs:
        os.environ["TORCH_LOGS"] = args.torch_compile_logs
    os.environ.setdefault("MUJOCO_GL", "egl")

    semantic_class = args.semantic_class or SEMANTIC_CLASSES.get(args.variant, "numerical")
    run_id = args.run_id or default_run_id("lewm-model-cost", args.variant, args.seed)
    args.run_id = run_id
    args.semantic_class = semantic_class

    results_root = Path(args.results_root)
    run_dir = results_root / "runs" / run_id if results_root.name != run_id else results_root
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    profile_path = Path(args.out) if args.out else run_dir / "model_cost_profile.json"
    manifest_path = run_dir / "manifest.json"
    _clean_run_artifacts(run_dir, metrics_path, manifest_path)

    weights_root = Path(args.weights_root or os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    weights_paths = policy_object_candidates(weights_root, args.policy)
    weights_sha = "unknown"
    for path in weights_paths:
        weights_sha = precomputed_hash(results_root, path)
        if weights_sha != "unknown":
            break
    if weights_sha == "unknown":
        weights_sha = first_existing_hash(weights_paths)

    profile_payload = run_profile(args, run_dir, metrics_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(json_safe(profile_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = read_jsonl(metrics_path)
    summary = profile_payload["summary"]
    profiler_summary_path = None
    if args.profile == "torch":
        try:
            from tools.benchmarks.summarize_torch_trace import summarize_trace_path, write_summary

            trace_summary = summarize_trace_path(run_dir / "profiler")
            summary["torch_profiler"] = trace_summary
            if args.model_cost_region_profile:
                summary["score_reduction_gate"] = summarize_score_reduction_gate(trace_summary)
            profile_payload["summary"] = summary
            if trace_summary.get("trace_files"):
                profiler_summary_path = run_dir / "profiler_summary.json"
                write_summary(profiler_summary_path, trace_summary)
        except Exception as exc:
            summary["torch_profiler"] = {"error": f"{type(exc).__name__}: {exc}"}
        profile_path.write_text(
            json.dumps(json_safe(profile_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    dataset_sha = "unknown"
    try:
        from tools.benchmarks.lewm_plan_benchmark import compose_config

        cfg, _ = compose_config(args.config_name, args.policy, args.solver, 0, args.seed)
        for path in dataset_candidates(weights_root, cfg.eval.dataset_name):
            dataset_sha = precomputed_hash(results_root, path)
            if dataset_sha != "unknown":
                break
        if dataset_sha == "unknown":
            dataset_sha = first_existing_hash(dataset_candidates(weights_root, cfg.eval.dataset_name))
    except Exception:
        dataset_sha = "unknown"

    manifest = build_manifest(
        run_id=run_id,
        model="lewm-pusht-model-cost",
        variant=args.variant,
        semantic_class=semantic_class,
        cwd=REPO_ROOT,
        command=sys.argv if argv is None else ["profile_lewm_model_cost.py", *argv],
        weights_sha256=weights_sha,
        dataset_sha256=dataset_sha,
        code_paths=MODEL_COST_CODE_PATHS,
        planner={
            "samples": args.samples,
            "horizon": args.horizon,
            "batch_size": args.batch_size,
            "history": args.history,
            "action_dim": profile_payload["shape"]["action_dim"],
            "candidate_shape": profile_payload["shape"]["candidate_shape"],
            "profile": args.profile,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iters,
            "matmul_precision": args.matmul_precision,
            "compile_cost": True,
            "compile_cost_mode": args.compile_cost_mode,
            "compile_cost_fullgraph": args.compile_cost_fullgraph,
            "compile_cost_allow_fallback": args.compile_cost_allow_fallback,
            "model_cost_region_profile": args.model_cost_region_profile,
            "model_cost_region_profile_compiled": False if args.model_cost_region_profile else None,
            "torch_profiler_with_stack": args.torch_profiler_with_stack,
            "torch_compile_logs": args.torch_compile_logs,
        },
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
            "metrics_jsonl": relative_run_artifact(metrics_path),
            "model_cost_profile": relative_run_artifact(profile_path),
            "profiler_trace": str(run_dir / "profiler") if args.profile == "torch" else None,
            "profiler_summary": str(profiler_summary_path) if profiler_summary_path else None,
            "stdout": str(run_dir / "stdout.txt"),
            "stderr": str(run_dir / "stderr.txt"),
        },
        metrics_summary=summary,
        quality_gate_result="profile_control",
        seed_list=[args.seed],
        num_eval_episodes=0,
    )
    append_jsonl(
        metrics_path,
        {
            "type": "profile_artifacts",
            "seed": args.seed,
            "model_cost_profile": str(profile_path),
            "manifest": str(manifest_path),
            "profile_rows": len(rows),
        },
    )
    write_manifest(manifest_path, manifest)
    print(f"wrote {metrics_path}")
    print(f"wrote {profile_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
