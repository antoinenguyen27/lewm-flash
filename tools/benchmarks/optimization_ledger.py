"""Append or replace run rows in the optimization ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.metrics import read_jsonl, summarize_plan_latencies
from tools.benchmarks.run_manifest import load_manifest


HEADER = """# Optimization Ledger

| Ladder | Run ID | Model | Variant | Semantic Class | Files Changed | Expected Effect | Actual Latency | Quality Result | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 | pending | LeWM Push-T | baseline | baseline | none | Reference measurement | pending | pending | pending | Original implementation. |
| L1 | pending | LeWM Push-T | cache_embeddings | exact | `lewm/jepa.py` | Remove repeated current/goal encodes | pending | pending | pending | Only valid if local `jepa.JEPA` path is active. |
| L2 | pending | LeWM Push-T | cem_exact_no_cpu_sync | exact | `lewm/solvers/cem_exact.py` | Remove loop-internal GPU-to-CPU sync | pending | pending | pending | Scoring remains inside every CEM iteration. |
| L3a | pending | LeWM Push-T | rollout_prealloc | exact | `lewm/jepa.py` | Remove growing `cat` and repeated action encoding | pending | pending | pending | Rejected unless a rerun contradicts the focused profiler regression. |
| L3b | pending | LeWM Push-T | rollout_action_cache | exact | `lewm/jepa.py` | Precompute action embeddings while keeping dynamic embedding rollout | pending | pending | experimental | Keep isolated for diagnostics only; do not inherit into future phases unless deliberately reopened. |
| L4 | unit_lewm_l4_compile_cost_v3; lewm_compile_cost_exactness_seed0_l40s_v3; smoke_lewm_compile_cost_diag_seed0_v2; lewm_l1_l2_for_l4_telem100_seed0-2_l40s + lewm_compile_cost_telem100_seed0-2_l40s | LeWM Push-T | compile_cost | numerical | `lewm/jepa.py`, `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/check_lewm_cache_exactness.py`, `modal/benchmark_app.py` | Reduce dispatch overhead | Paired warmed-runtime confirmation: post-warm CEM CUDA 22.34 s -> 18.16 s (`1.23x`); single post-warm CEM 7.45 s -> 6.05 s | Quality 264/300 vs L1+L2 262/300; exactness pass; steady-state compile signatures stable | accepted | Build on accepted L1+L2; skip L3b; requires real full-CEM warmup before serving. |
| L6 | lewm_profile_compile_cost_phase3_inner_seed0_l40s | LeWM Push-T | fused_cem_glue | numerical | `lewm/solvers/cem_exact.py`, `tools/benchmarks/lewm_plan_benchmark.py` | Reduce CEM sample/top-k/elite/update glue | Gate failed: glue bucket `20.06 ms`, `3.56%` of post-warm CEM CUDA | Focused profile success `9/10`; no behavior change tested | rejected | Downgraded/no-go before Phase 3R. |
| L6R-a | lewm_compile_cost_tf32_control_telem100_l40s + lewm_model_cost_tf32_telem100_l40s | LeWM Push-T | model_cost_tf32 | numerical | `tools/benchmarks/lewm_plan_benchmark.py`, `tools/benchmarks/profile_lewm_model_cost.py`, `tools/benchmarks/check_lewm_cache_exactness.py`, `modal/benchmark_app.py` | Use TF32-capable matmul kernels in repeated model-cost GEMMs | Post-warm CEM CUDA/output `280.27 ms -> 219.50 ms` (`1.28x`) | Push-T `263/300` vs control `264/300`; fixed-candidate drift explained by near ties | accepted | Promoted Phase 3R warmed-runtime result. |
| L6R-b | lewm_model_cost_default_seed0_l40s; lewm_model_cost_fullgraph_seed0_l40s; lewm_model_cost_max_autotune_seed0_l40s | LeWM Push-T | compile_default; compile_cost_fullgraph; compile_cost_max_autotune | numerical | `tools/benchmarks/profile_lewm_model_cost.py`, `modal/benchmark_app.py` | Test compile/capture settings for tensor-only model cost | `0.983x`; `0.994x`; `1.006x` after about `739 s` warmup | No full Push-T sweep; failed isolated promotion gate | rejected | No further sweep without new evidence. |
| L6R-c | lewm_model_cost_score_regions_tf32_high_seed0_l40s | LeWM Push-T | fused_score | numerical | `tools/benchmarks/profile_lewm_model_cost.py`, `modal/benchmark_app.py` | Fuse terminal latent MSE score only if material | `lewm.cost_from_rollout` `0.692%` of model-cost CUDA | No behavior change tested | rejected | No fused terminal-score kernel. |
| L7 | pending | LeWM Push-T | icem | planner | `lewm/solvers/icem.py` | Reduce samples/iterations | pending | pending | pending | Requires Push-T non-regression. |
| L8 | pending | LeWM Push-T | cem_gd | planner | `lewm/solvers/cem_gd.py` | Use gradients to reduce CEM budget | pending | pending | pending | Requires Push-T non-regression. |
"""


EXPECTED_EFFECTS = {
    "baseline": "Reference measurement",
    "cache_embeddings": "Remove repeated current/goal encodes",
    "cem_exact": "Remove loop-internal GPU-to-CPU sync",
    "cem_exact_no_cpu_sync": "Remove loop-internal GPU-to-CPU sync",
    "rollout_prealloc": "Remove growing cat and repeated action encoding",
    "rollout_action_cache": "Precompute action embeddings while keeping dynamic embedding rollout",
    "compile_cost": "Reduce dispatch overhead",
    "model_cost_tf32": "Use TF32-capable matmul kernels in repeated model-cost GEMMs",
    "fused_cem_glue": "Reduce CEM glue overhead",
    "fused_score": "Fuse terminal latent MSE score only if material",
    "icem": "Reduce samples/iterations",
    "cem_gd": "Use gradients to reduce CEM budget",
}


def ensure_ledger(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER, encoding="utf-8")


def escape_cell(value: Any) -> str:
    text = "pending" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def latency_from_manifest(manifest: dict[str, Any]) -> str:
    summary = manifest.get("metrics_summary", {}).get("plan_latency_ms")
    if not summary and manifest.get("artifacts", {}).get("metrics_jsonl"):
        metrics_path = Path(manifest["artifacts"]["metrics_jsonl"])
        if metrics_path.exists():
            summary = summarize_plan_latencies(read_jsonl(metrics_path))
    if not summary or not summary.get("count"):
        return "pending"
    mean = summary.get("mean")
    p50 = summary.get("p50")
    p90 = summary.get("p90")
    return f"mean {mean:.2f} ms, p50 {p50:.2f} ms, p90 {p90:.2f} ms"


def quality_from_manifest(manifest: dict[str, Any]) -> str:
    gate = manifest.get("quality_gate_result")
    if gate:
        return str(gate)
    summary = manifest.get("metrics_summary", {})
    if "task_score_mean" in summary:
        return f"task_score_mean {summary['task_score_mean']}"
    return "not_applicable" if manifest.get("semantic_class") in {"baseline", "exact", "numerical"} else "pending"


def files_changed(manifest: dict[str, Any]) -> str:
    paths = manifest.get("code_paths") or []
    if not paths:
        return "none"
    return ", ".join(f"`{path}`" for path in paths)


def build_row(manifest: dict[str, Any], status: str, notes: str, ladder: str | None) -> str:
    variant = manifest.get("variant", "unknown")
    cells = [
        ladder or "run",
        manifest.get("run_id", "unknown"),
        manifest.get("model", "unknown"),
        variant,
        manifest.get("semantic_class", "unknown"),
        files_changed(manifest),
        EXPECTED_EFFECTS.get(variant, "Measured experiment"),
        latency_from_manifest(manifest),
        quality_from_manifest(manifest),
        status,
        notes or "manifest-backed run",
    ]
    return "| " + " | ".join(escape_cell(cell) for cell in cells) + " |"


def upsert_row(path: Path, row: str, run_id: str) -> None:
    ensure_ledger(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    target = f"| "
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith(target):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) > 1 and cells[1] == run_id:
                lines[idx] = row
                replaced = True
                break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append(row)
        else:
            lines[-1:] = [row]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--status",
        required=True,
        choices=[
            "pending",
            "running",
            "accepted",
            "rejected",
            "needs_repeat",
            "experimental",
        ],
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--ladder", default=None)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    row = build_row(manifest, args.status, args.notes, args.ladder)
    upsert_row(Path(args.ledger), row, manifest["run_id"])
    print(json.dumps({"ledger_updated": True, "ledger": args.ledger, "run_id": manifest["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
