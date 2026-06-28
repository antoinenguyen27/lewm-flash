"""Check that LeWM cost-path candidates preserve fixed-candidate costs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.metrics import json_safe
from tools.benchmarks.lewm_plan_benchmark import REPO_ROOT, load_policy_model


LEWM_DIR = REPO_ROOT / "lewm"


def summarize_exactness_result(
    *,
    baseline_cache_allclose: bool,
    cache_repeat_allclose: bool,
    argmin_equal: bool,
    topk_equal: bool,
    baseline_encode_total: int,
    cache_encode_total: int,
    cache_repeat_encode_delta: int,
) -> bool:
    return (
        baseline_cache_allclose
        and cache_repeat_allclose
        and argmin_equal
        and topk_equal
        and baseline_encode_total > cache_encode_total
        and cache_repeat_encode_delta == 0
    )


def summarize_candidate_exactness_result(
    *,
    control_candidate_allclose: bool,
    candidate_repeat_allclose: bool,
    argmin_equal: bool,
    topk_equal: bool,
) -> bool:
    return (
        control_candidate_allclose
        and candidate_repeat_allclose
        and argmin_equal
        and topk_equal
    )


def summarize_numerical_candidate_result(
    *,
    candidate_repeat_allclose: bool,
    argmin_equal: bool,
    topk_equal: bool,
    ranking_drift_explained_by_near_ties: bool = False,
) -> bool:
    return candidate_repeat_allclose and (
        (argmin_equal and topk_equal) or ranking_drift_explained_by_near_ties
    )


def relative_error(max_abs: float, reference_scale: float) -> float:
    if reference_scale <= 0:
        return 0.0 if max_abs == 0 else float("inf")
    return max_abs / reference_scale


class EncodeCounter:
    def __init__(self, model: Any):
        self.model = model
        self.original = model.encode
        self.counts = {"goal": 0, "initial": 0, "total": 0, "unknown": 0}

    def install(self) -> None:
        def counted(info: dict[str, Any]):
            bucket = "unknown"
            if isinstance(info, dict):
                bucket = "initial" if "action" in info else "goal"
            self.counts["total"] += 1
            self.counts[bucket] += 1
            return self.original(info)

        self.model.encode = counted

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    def restore(self) -> None:
        self.model.encode = self.original


class ActionEncodeCounter:
    def __init__(self, model: Any):
        self.model = model
        self.action_encoder = getattr(model, "action_encoder", None)
        self.original = None
        self.counts = {"total": 0}

    def install(self) -> None:
        if self.action_encoder is None or not hasattr(self.action_encoder, "forward"):
            return
        self.original = self.action_encoder.forward

        def counted(action: Any):
            self.counts["total"] += 1
            return self.original(action)

        self.action_encoder.forward = counted

    def reset(self) -> None:
        self.counts["total"] = 0

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)

    def restore(self) -> None:
        if self.action_encoder is not None and self.original is not None:
            self.action_encoder.forward = self.original


def infer_model_shape(model: Any) -> tuple[int, int]:
    try:
        history_size = int(model.predictor.pos_embedding.shape[1])
    except AttributeError as exc:
        raise TypeError("Could not infer history size from model.predictor.pos_embedding") from exc

    try:
        action_dim = int(model.action_encoder.patch_embed.in_channels)
    except AttributeError as exc:
        raise TypeError("Could not infer action dimension from model.action_encoder.patch_embed") from exc

    return history_size, action_dim


def make_info(
    torch: Any,
    *,
    seed: int,
    batch_size: int,
    samples: int,
    history_size: int,
    action_dim: int,
    image_size: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pixels = torch.randn(
        batch_size,
        1,
        history_size,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    goal = torch.randn(
        batch_size,
        1,
        history_size,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    action = torch.randn(
        batch_size,
        1,
        history_size,
        action_dim,
        generator=generator,
    )
    return {
        "pixels": pixels.expand(batch_size, samples, history_size, 3, image_size, image_size),
        "goal": goal.expand(batch_size, samples, history_size, 3, image_size, image_size),
        "action": action.expand(batch_size, samples, history_size, action_dim),
    }


def make_candidates(
    torch: Any,
    *,
    seed: int,
    cem_steps: int,
    batch_size: int,
    samples: int,
    horizon: int,
    action_dim: int,
    device: str,
) -> Any:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        cem_steps,
        batch_size,
        samples,
        horizon,
        action_dim,
        generator=generator,
        device=device,
    )


def run_cost_sequence(
    model: Any,
    info: dict[str, Any],
    candidates: Any,
    *,
    torch: Any,
) -> Any:
    costs = []
    for step in range(candidates.shape[0]):
        costs.append(model.get_cost(info, candidates[step]).clone())
    return torch.stack(costs)


def cost_stats(torch: Any, baseline: Any, candidate: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    diff = (baseline - candidate).abs()
    max_abs = float(diff.max().item())
    reference_scale = float(torch.maximum(baseline.abs(), candidate.abs()).max().item())
    return {
        "allclose": bool(torch.allclose(baseline, candidate, atol=atol, rtol=rtol)),
        "max_abs": max_abs,
        "max_rel": relative_error(max_abs, reference_scale),
        "reference_scale": reference_scale,
    }


def compare_rankings(torch: Any, baseline: Any, candidate: Any, *, topk: int) -> dict[str, bool]:
    k = min(int(topk), int(baseline.shape[-1]))
    return {
        "argmin_equal": bool(torch.equal(baseline.argmin(dim=-1), candidate.argmin(dim=-1))),
        "topk_equal": bool(
            torch.equal(
                torch.topk(baseline, k=k, dim=-1, largest=False).indices,
                torch.topk(candidate, k=k, dim=-1, largest=False).indices,
            )
        ),
    }


def ranking_drift_diagnostics(
    torch: Any,
    baseline: Any,
    candidate: Any,
    *,
    topk: int,
    tie_tolerance: float,
) -> dict[str, Any]:
    k = min(int(topk), int(baseline.shape[-1]))
    flat_baseline = baseline.reshape(-1, baseline.shape[-1])
    flat_candidate = candidate.reshape(-1, candidate.shape[-1])

    baseline_argmin = flat_baseline.argmin(dim=-1)
    candidate_argmin = flat_candidate.argmin(dim=-1)
    argmin_mismatches = baseline_argmin != candidate_argmin
    argmin_gap_records = []
    for row_idx in torch.nonzero(argmin_mismatches, as_tuple=False).flatten().tolist():
        base_idx = int(baseline_argmin[row_idx].item())
        cand_idx = int(candidate_argmin[row_idx].item())
        baseline_gap = abs(
            float(flat_baseline[row_idx, cand_idx].item())
            - float(flat_baseline[row_idx, base_idx].item())
        )
        candidate_gap = abs(
            float(flat_candidate[row_idx, base_idx].item())
            - float(flat_candidate[row_idx, cand_idx].item())
        )
        argmin_gap_records.append(max(baseline_gap, candidate_gap))

    baseline_topk = torch.topk(flat_baseline, k=k, dim=-1, largest=False)
    candidate_topk = torch.topk(flat_candidate, k=k, dim=-1, largest=False)
    topk_mismatch_rows = 0
    topk_boundary_gap_records = []
    for row_idx in range(flat_baseline.shape[0]):
        baseline_set = set(int(v) for v in baseline_topk.indices[row_idx].tolist())
        candidate_set = set(int(v) for v in candidate_topk.indices[row_idx].tolist())
        if baseline_set == candidate_set:
            continue
        topk_mismatch_rows += 1
        baseline_boundary = float(baseline_topk.values[row_idx, -1].item())
        candidate_boundary = float(candidate_topk.values[row_idx, -1].item())
        for idx in sorted(baseline_set ^ candidate_set):
            baseline_gap = abs(float(flat_baseline[row_idx, idx].item()) - baseline_boundary)
            candidate_gap = abs(float(flat_candidate[row_idx, idx].item()) - candidate_boundary)
            topk_boundary_gap_records.append(max(baseline_gap, candidate_gap))

    max_argmin_gap = max(argmin_gap_records) if argmin_gap_records else 0.0
    max_topk_boundary_gap = max(topk_boundary_gap_records) if topk_boundary_gap_records else 0.0
    return {
        "tie_tolerance": tie_tolerance,
        "argmin_mismatch_count": int(argmin_mismatches.sum().item()),
        "topk_mismatch_rows": int(topk_mismatch_rows),
        "max_argmin_pair_gap": max_argmin_gap,
        "max_topk_boundary_gap": max_topk_boundary_gap,
        "argmin_mismatches_explained_by_near_ties": max_argmin_gap <= tie_tolerance,
        "topk_mismatches_explained_by_near_ties": max_topk_boundary_gap <= tie_tolerance,
        "all_mismatches_explained_by_near_ties": (
            max_argmin_gap <= tie_tolerance and max_topk_boundary_gap <= tie_tolerance
        ),
    }


def disable_rollout_modes(model: Any) -> None:
    if hasattr(model, "set_rollout_prealloc"):
        model.set_rollout_prealloc(False)
    if hasattr(model, "set_rollout_action_cache"):
        model.set_rollout_action_cache(False)


def configure_exactness_mode(model: Any, mode: str) -> None:
    disable_rollout_modes(model)
    if mode not in {
        "compile_cost",
    } and hasattr(model, "set_compile_cost"):
        model.set_compile_cost(False)
    if mode == "baseline":
        model.set_embedding_cache(False)
        return
    if mode in {
        "cache_embeddings",
        "rollout_action_cache_control",
        "compile_cost_control",
    }:
        model.set_embedding_cache(True)
        return
    if mode == "rollout_action_cache":
        if not hasattr(model, "set_rollout_action_cache"):
            raise TypeError(f"Model does not expose set_rollout_action_cache(): {type(model)}")
        model.set_embedding_cache(True)
        model.set_rollout_action_cache(True)
        return
    if mode == "compile_cost":
        import torch

        if not hasattr(model, "get_cost_from_embeddings"):
            raise TypeError(f"Model does not expose get_cost_from_embeddings(): {type(model)}")
        if not hasattr(model, "set_compile_cost"):
            raise TypeError(f"Model does not expose set_compile_cost(): {type(model)}")
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile_cost exactness requires torch.compile")
        model.set_embedding_cache(True)
        if bool(getattr(model, "compile_cost_enabled", lambda: False)()):
            return
        model.get_cost_from_embeddings = torch.compile(
            model.get_cost_from_embeddings,
            mode="reduce-overhead",
            fullgraph=False,
        )
        model.set_compile_cost(True)
        return
    raise ValueError(f"Unknown exactness mode: {mode}")


def apply_matmul_precision(torch: Any, precision: str) -> dict[str, str]:
    before = torch.get_float32_matmul_precision()
    if precision != "default":
        torch.set_float32_matmul_precision(precision)
    after = torch.get_float32_matmul_precision()
    return {
        "requested": precision,
        "before": before,
        "after": after,
    }


def exactness_mode_pair(candidate_mode: str) -> tuple[str, str, str, str]:
    if candidate_mode == "embedding_cache":
        return "baseline", "cache_embeddings", "baseline", "cache_embeddings"
    if candidate_mode == "rollout_action_cache":
        return (
            "rollout_action_cache_control",
            "rollout_action_cache",
            "cem_exact_no_cpu_sync",
            "rollout_action_cache",
        )
    if candidate_mode == "compile_cost":
        return (
            "compile_cost_control",
            "compile_cost",
            "cem_exact_no_cpu_sync",
            "compile_cost",
        )
    if candidate_mode == "model_cost_tf32":
        return (
            "compile_cost",
            "compile_cost",
            "compile_cost",
            "model_cost_tf32",
        )
    raise ValueError(f"Unknown candidate mode: {candidate_mode}")


def run_mode_cost_sequence(
    model: Any,
    *,
    mode: str,
    counter: EncodeCounter,
    action_counter: ActionEncodeCounter,
    torch: Any,
    args: argparse.Namespace,
    candidates: Any,
    history_size: int,
    action_dim: int,
) -> tuple[dict[str, Any], Any, dict[str, int], dict[str, int]]:
    configure_exactness_mode(model, mode)
    counter.reset()
    action_counter.reset()
    info = make_info(
        torch,
        seed=args.seed,
        batch_size=int(getattr(args, "batch_size", 1)),
        samples=args.samples,
        history_size=history_size,
        action_dim=action_dim,
        image_size=args.image_size,
    )
    costs = run_cost_sequence(model, info, candidates, torch=torch)
    return info, costs, counter.snapshot(), action_counter.snapshot()


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    import stable_worldmodel as swm
    import torch

    sys.path.insert(0, str(LEWM_DIR))

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device != "cuda" and args.require_cuda:
        raise RuntimeError("CUDA is required for this Modal exactness check")

    def load_model() -> Any:
        loaded = load_policy_model(swm, Path(args.weights_root), args.policy)
        loaded = loaded.to(device).eval()
        loaded.requires_grad_(False)
        loaded.interpolate_pos_encoding = True
        if not hasattr(loaded, "set_embedding_cache"):
            raise TypeError(f"Model does not expose set_embedding_cache(): {type(loaded)}")
        return loaded

    model = load_model()

    history_size, action_dim = infer_model_shape(model)
    candidates = make_candidates(
        torch,
        seed=args.seed,
        cem_steps=args.cem_steps,
        batch_size=args.batch_size,
        samples=args.samples,
        horizon=args.horizon,
        action_dim=action_dim,
        device=device,
    )
    control_mode, candidate_exactness_mode, control_variant, candidate_variant = exactness_mode_pair(
        args.candidate_mode
    )
    if args.candidate_mode == "model_cost_tf32":
        control_model = model
        candidate_model = load_model()
        control_counter = EncodeCounter(control_model)
        control_action_counter = ActionEncodeCounter(control_model)
        candidate_counter = EncodeCounter(candidate_model)
        candidate_action_counter = ActionEncodeCounter(candidate_model)
        control_counter.install()
        candidate_counter.install()
        try:
            with torch.inference_mode():
                control_precision = apply_matmul_precision(
                    torch,
                    args.control_matmul_precision,
                )
                _, control_costs, control_counts, control_action_counts = run_mode_cost_sequence(
                    control_model,
                    mode=control_mode,
                    counter=control_counter,
                    action_counter=control_action_counter,
                    torch=torch,
                    args=args,
                    candidates=candidates,
                    history_size=history_size,
                    action_dim=action_dim,
                )

                candidate_precision = apply_matmul_precision(
                    torch,
                    args.candidate_matmul_precision,
                )
                (
                    candidate_info,
                    candidate_costs,
                    candidate_counts,
                    candidate_action_counts,
                ) = run_mode_cost_sequence(
                    candidate_model,
                    mode=candidate_exactness_mode,
                    counter=candidate_counter,
                    action_counter=candidate_action_counter,
                    torch=torch,
                    args=args,
                    candidates=candidates,
                    history_size=history_size,
                    action_dim=action_dim,
                )
                candidate_before_repeat = candidate_counter.snapshot()
                candidate_action_before_repeat = candidate_action_counter.snapshot()
                candidate_repeat_costs = run_cost_sequence(
                    candidate_model,
                    candidate_info,
                    candidates,
                    torch=torch,
                )
                candidate_after_repeat = candidate_counter.snapshot()
                candidate_action_after_repeat = candidate_action_counter.snapshot()

            if device == "cuda":
                torch.cuda.synchronize()

            control_candidate = cost_stats(
                torch, control_costs, candidate_costs, atol=args.atol, rtol=args.rtol
            )
            candidate_repeat = cost_stats(
                torch, candidate_costs, candidate_repeat_costs, atol=args.atol, rtol=args.rtol
            )
            rankings = compare_rankings(torch, control_costs, candidate_costs, topk=args.topk)
            tie_tolerance = (
                float(args.ranking_tie_tolerance)
                if args.ranking_tie_tolerance is not None
                else float(control_candidate["max_abs"])
            )
            ranking_drift = ranking_drift_diagnostics(
                torch,
                control_costs,
                candidate_costs,
                topk=args.topk,
                tie_tolerance=tie_tolerance,
            )
            passed = summarize_numerical_candidate_result(
                candidate_repeat_allclose=candidate_repeat["allclose"],
                argmin_equal=rankings["argmin_equal"],
                topk_equal=rankings["topk_equal"],
                ranking_drift_explained_by_near_ties=ranking_drift[
                    "all_mismatches_explained_by_near_ties"
                ],
            )
            return json_safe(
                {
                    "passed": passed,
                    "candidate_mode": args.candidate_mode,
                    "control_variant": control_variant,
                    "candidate_variant": candidate_variant,
                    "seed": args.seed,
                    "policy": args.policy,
                    "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
                    "device": device,
                    "samples": args.samples,
                    "cem_steps": args.cem_steps,
                    "horizon": args.horizon,
                    "history_size": history_size,
                    "action_dim": action_dim,
                    "topk": min(args.topk, args.samples),
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "control_vs_candidate": control_candidate,
                    "candidate_repeat": candidate_repeat,
                    "rankings": rankings,
                    "ranking_drift": ranking_drift,
                    "matmul_precision": {
                        "control": control_precision,
                        "candidate": candidate_precision,
                    },
                    "encode_counts": {
                        "control": control_counts,
                        "candidate_first_pass": candidate_counts,
                        "candidate_after_repeat": candidate_after_repeat,
                        "candidate_repeat_delta": {
                            key: candidate_after_repeat[key] - candidate_before_repeat[key]
                            for key in candidate_after_repeat
                        },
                    },
                    "action_encode_counts": {
                        "enabled": False,
                        "control": control_action_counts,
                        "candidate_first_pass": candidate_action_counts,
                        "candidate_after_repeat": candidate_action_after_repeat,
                        "candidate_repeat_delta": {
                            key: candidate_action_after_repeat[key]
                            - candidate_action_before_repeat[key]
                            for key in candidate_action_after_repeat
                        },
                    },
                }
            )
        finally:
            candidate_action_counter.restore()
            candidate_counter.restore()
            control_action_counter.restore()
            control_counter.restore()

    counter = EncodeCounter(model)
    action_counter = ActionEncodeCounter(model)
    counter.install()
    if args.candidate_mode != "compile_cost":
        action_counter.install()
    try:
        with torch.inference_mode():
            _, control_costs, control_counts, control_action_counts = run_mode_cost_sequence(
                model,
                mode=control_mode,
                counter=counter,
                action_counter=action_counter,
                torch=torch,
                args=args,
                candidates=candidates,
                history_size=history_size,
                action_dim=action_dim,
            )

            (
                candidate_info,
                candidate_costs,
                candidate_counts,
                candidate_action_counts,
            ) = run_mode_cost_sequence(
                model,
                mode=candidate_exactness_mode,
                counter=counter,
                action_counter=action_counter,
                torch=torch,
                args=args,
                candidates=candidates,
                history_size=history_size,
                action_dim=action_dim,
            )

            candidate_before_repeat = counter.snapshot()
            candidate_action_before_repeat = action_counter.snapshot()
            candidate_repeat_costs = run_cost_sequence(
                model, candidate_info, candidates, torch=torch
            )
            candidate_after_repeat = counter.snapshot()
            candidate_action_after_repeat = action_counter.snapshot()

        if device == "cuda":
            torch.cuda.synchronize()

        control_candidate = cost_stats(
            torch, control_costs, candidate_costs, atol=args.atol, rtol=args.rtol
        )
        candidate_repeat = cost_stats(
            torch, candidate_costs, candidate_repeat_costs, atol=args.atol, rtol=args.rtol
        )
        rankings = compare_rankings(torch, control_costs, candidate_costs, topk=args.topk)
        candidate_repeat_encode_delta = (
            candidate_after_repeat["total"] - candidate_before_repeat["total"]
        )
        if args.candidate_mode == "embedding_cache":
            passed = summarize_exactness_result(
                baseline_cache_allclose=control_candidate["allclose"],
                cache_repeat_allclose=candidate_repeat["allclose"],
                argmin_equal=rankings["argmin_equal"],
                topk_equal=rankings["topk_equal"],
                baseline_encode_total=control_counts["total"],
                cache_encode_total=candidate_counts["total"],
                cache_repeat_encode_delta=candidate_repeat_encode_delta,
            )
        else:
            passed = summarize_candidate_exactness_result(
                control_candidate_allclose=control_candidate["allclose"],
                candidate_repeat_allclose=candidate_repeat["allclose"],
                argmin_equal=rankings["argmin_equal"],
                topk_equal=rankings["topk_equal"],
            )
        encode_counts = {
            "control": control_counts,
            "candidate_first_pass": candidate_counts,
            "candidate_after_repeat": candidate_after_repeat,
            "candidate_repeat_delta": {
                key: candidate_after_repeat[key] - candidate_before_repeat[key]
                for key in candidate_after_repeat
            },
        }
        if args.candidate_mode == "embedding_cache":
            encode_counts.update(
                {
                    "baseline": control_counts,
                    "cache_first_pass": candidate_counts,
                    "cache_after_repeat": candidate_after_repeat,
                    "cache_repeat_delta": encode_counts["candidate_repeat_delta"],
                }
            )
        return json_safe(
            {
                "passed": passed,
                "candidate_mode": args.candidate_mode,
                "control_variant": control_variant,
                "candidate_variant": candidate_variant,
                "seed": args.seed,
                "policy": args.policy,
                "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
                "device": device,
                "samples": args.samples,
                "cem_steps": args.cem_steps,
                "horizon": args.horizon,
                "history_size": history_size,
                "action_dim": action_dim,
                "topk": min(args.topk, args.samples),
                "atol": args.atol,
                "rtol": args.rtol,
                "control_vs_candidate": control_candidate,
                "candidate_repeat": candidate_repeat,
                # Backward-compatible aliases for the original embedding-cache check.
                "baseline_vs_cache": control_candidate
                if args.candidate_mode == "embedding_cache"
                else None,
                "cache_repeat": candidate_repeat
                if args.candidate_mode == "embedding_cache"
                else None,
                "rankings": rankings,
                "encode_counts": encode_counts,
                "action_encode_counts": {
                    "enabled": args.candidate_mode != "compile_cost",
                    "control": control_action_counts,
                    "candidate_first_pass": candidate_action_counts,
                    "candidate_after_repeat": candidate_action_after_repeat,
                    "candidate_repeat_delta": {
                        key: candidate_action_after_repeat[key]
                        - candidate_action_before_repeat[key]
                        for key in candidate_action_after_repeat
                    },
                },
            }
        )
    finally:
        action_counter.restore()
        counter.restore()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", default="/weights")
    parser.add_argument("--policy", default="pusht/lewm")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-mode",
        choices=[
            "embedding_cache",
            "rollout_action_cache",
            "compile_cost",
            "model_cost_tf32",
        ],
        default="embedding_cache",
        help=(
            "embedding_cache compares baseline vs L1 cache; rollout_action_cache compares "
            "accepted L1 dynamic rollout vs L3b action-embedding cache; compile_cost "
            "compares accepted L1 dynamic rollout vs compiled tensor-only L4 cost on "
            "fixed candidates; model_cost_tf32 compares compile_cost control vs the "
            "same compiled tensor-only path with torch matmul precision set to high."
        ),
    )
    parser.add_argument(
        "--control-matmul-precision",
        choices=["default", "highest", "high", "medium"],
        default="default",
    )
    parser.add_argument(
        "--candidate-matmul-precision",
        choices=["default", "highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--ranking-tie-tolerance", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-require-cuda", dest="require_cuda", action="store_false")
    parser.set_defaults(require_cuda=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_check(args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
