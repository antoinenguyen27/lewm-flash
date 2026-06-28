"""Small JSONL and metric-summary helpers for benchmark runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-serializable objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, set):
        return [json_safe(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return repr(value)


def append_jsonl(path: Path | str, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize_numeric(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] is not None and isinstance(row[key], (int, float))
    ]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def summarize_plan_latencies(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row.get("type") == "plan_latency" and not row.get("warmup", False)
    ]
    summary = summarize_numeric(measured, "plan_latency_ms")
    if summary.get("count", 0):
        summary["warmup_excluded"] = True
    return summary


def _measured_rows(rows: Iterable[dict[str, Any]], row_type: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("type") == row_type and not row.get("warmup", False)
    ]


def summarize_cem_calls(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-CEM-call telemetry rows, excluding warmup calls."""
    measured = _measured_rows(rows, "cem_call")
    summary: dict[str, Any] = {
        "call_count": len(measured),
        "warmup_excluded": bool(measured),
        "cpu_wall_ms": summarize_numeric(measured, "cpu_wall_ms"),
        "cuda_elapsed_ms": summarize_numeric(measured, "cuda_elapsed_ms"),
        "input_total_envs": summarize_numeric(measured, "total_envs"),
        "output_cost_count": summarize_numeric(measured, "output_cost_count"),
        "loop_cpu_materialization_count_estimate": summarize_numeric(
            measured, "loop_cpu_materialization_count_estimate"
        ),
        "final_cpu_materialization_count_estimate": summarize_numeric(
            measured, "final_cpu_materialization_count_estimate"
        ),
        "cpu_materialization_count_estimate": summarize_numeric(
            measured, "cpu_materialization_count_estimate"
        ),
        "model_cost_evals": summarize_numeric(measured, "model_cost_evals"),
        "model_cost_env_candidate_evals": summarize_numeric(
            measured, "model_cost_env_candidate_evals"
        ),
        "gd_model_cost_evals": summarize_numeric(measured, "gd_model_cost_evals"),
        "gd_model_cost_env_candidate_evals": summarize_numeric(
            measured, "gd_model_cost_env_candidate_evals"
        ),
        "model_cost_evals_including_gd": summarize_numeric(
            measured, "model_cost_evals_including_gd"
        ),
        "model_cost_env_candidate_evals_including_gd": summarize_numeric(
            measured, "model_cost_env_candidate_evals_including_gd"
        ),
        "gd_forward_time_ms_mean": summarize_numeric(measured, "gd_forward_time_ms_mean"),
        "gd_backward_time_ms_mean": summarize_numeric(measured, "gd_backward_time_ms_mean"),
        "gd_grad_norm_mean": summarize_numeric(measured, "gd_grad_norm_mean"),
        "gd_grad_norm_max": summarize_numeric(measured, "gd_grad_norm_max"),
        "gd_cost_delta_mean": summarize_numeric(measured, "gd_cost_delta_mean"),
        "gd_accept_rate": summarize_numeric(measured, "gd_accept_rate"),
        "gd_action_saturation_rate": summarize_numeric(
            measured, "gd_action_saturation_rate"
        ),
        "gd_nan_or_inf_count": summarize_numeric(measured, "gd_nan_or_inf_count"),
        "active_env_count": summarize_numeric(measured, "active_env_count"),
    }

    memory_keys = (
        "cuda_memory_allocated_before_mb",
        "cuda_memory_allocated_after_mb",
        "cuda_memory_reserved_after_mb",
        "cuda_memory_peak_allocated_mb",
    )
    memory = {
        key: summarize_numeric(measured, key)
        for key in memory_keys
        if any(row.get(key) is not None for row in measured)
    }
    if memory:
        summary["memory"] = memory

    compile_rows = [
        row
        for row in measured
        if row.get("compile_diagnostics_enabled")
    ]
    if compile_rows:
        summary["compile_diagnostics"] = {
            "call_count": len(compile_rows),
            "signature_mismatch_count": sum(
                1
                for row in compile_rows
                if row.get("compile_runtime_signature_matches_expected") is False
            ),
            "signature_unknown_count": sum(
                1
                for row in compile_rows
                if row.get("compile_runtime_signature_matches_expected") is None
            ),
            "unique_signature_counts": summarize_numeric(
                compile_rows,
                "compile_runtime_signature_unique_count",
            ),
            "observation_counts": summarize_numeric(
                compile_rows,
                "compile_runtime_signature_observation_count",
            ),
            "counter_snapshot_available_count": sum(
                1 for row in compile_rows if row.get("compile_counter_snapshot_available")
            ),
        }

    region_totals: dict[str, float] = {}
    region_counts: dict[str, int] = {}
    for row in measured:
        for name, value in row.get("region_cpu_ms", {}).items():
            if isinstance(value, (int, float)):
                region_totals[name] = region_totals.get(name, 0.0) + float(value)
        for name, value in row.get("region_count", {}).items():
            if isinstance(value, int):
                region_counts[name] = region_counts.get(name, 0) + value
    if region_totals:
        summary["region_cpu_ms_total"] = region_totals
    if region_counts:
        summary["region_count"] = region_counts
    return summary


def summarize_evaluations(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize evaluation result rows without assuming a fixed task metric schema."""
    measured = [row for row in rows if row.get("type") == "evaluation_result"]
    summary: dict[str, Any] = {
        "evaluation_time_s": summarize_numeric(measured, "evaluation_time_s"),
    }

    encode_totals: dict[str, int] = {}
    plan_calls = 0
    cem_calls = 0
    for row in measured:
        counts = row.get("encode_counts_total", {})
        if isinstance(counts, dict):
            for key, value in counts.items():
                if isinstance(value, int):
                    encode_totals[key] = encode_totals.get(key, 0) + value
        if isinstance(row.get("plan_calls_total"), int):
            plan_calls += row["plan_calls_total"]
        if isinstance(row.get("cem_calls_total"), int):
            cem_calls += row["cem_calls_total"]
    if encode_totals:
        summary["encode_counts_total"] = encode_totals
    if plan_calls:
        summary["plan_calls_total"] = plan_calls
    if cem_calls:
        summary["cem_calls_total"] = cem_calls
    model_cost_evals = sum(
        int(row["model_cost_evals_total"])
        for row in measured
        if isinstance(row.get("model_cost_evals_total"), int)
    )
    if model_cost_evals:
        summary["model_cost_evals_total"] = model_cost_evals
    gd_model_cost_evals = sum(
        int(row["gd_model_cost_evals_total"])
        for row in measured
        if isinstance(row.get("gd_model_cost_evals_total"), int)
    )
    if gd_model_cost_evals:
        summary["gd_model_cost_evals_total"] = gd_model_cost_evals
    model_cost_evals_including_gd = sum(
        int(row["model_cost_evals_total_including_gd"])
        for row in measured
        if isinstance(row.get("model_cost_evals_total_including_gd"), int)
    )
    if not model_cost_evals_including_gd and (model_cost_evals or gd_model_cost_evals):
        model_cost_evals_including_gd = model_cost_evals + gd_model_cost_evals
    if model_cost_evals_including_gd:
        summary["model_cost_evals_total_including_gd"] = model_cost_evals_including_gd
    model_cost_env_candidate_evals = sum(
        int(row["model_cost_env_candidate_evals_total"])
        for row in measured
        if isinstance(row.get("model_cost_env_candidate_evals_total"), int)
    )
    if model_cost_env_candidate_evals:
        summary["model_cost_env_candidate_evals_total"] = model_cost_env_candidate_evals
    gd_model_cost_env_candidate_evals = sum(
        int(row["gd_model_cost_env_candidate_evals_total"])
        for row in measured
        if isinstance(row.get("gd_model_cost_env_candidate_evals_total"), int)
    )
    if gd_model_cost_env_candidate_evals:
        summary["gd_model_cost_env_candidate_evals_total"] = gd_model_cost_env_candidate_evals
    model_cost_env_candidate_evals_including_gd = sum(
        int(row["model_cost_env_candidate_evals_total_including_gd"])
        for row in measured
        if isinstance(row.get("model_cost_env_candidate_evals_total_including_gd"), int)
    )
    if not model_cost_env_candidate_evals_including_gd and (
        model_cost_env_candidate_evals or gd_model_cost_env_candidate_evals
    ):
        model_cost_env_candidate_evals_including_gd = (
            model_cost_env_candidate_evals + gd_model_cost_env_candidate_evals
        )
    if model_cost_env_candidate_evals_including_gd:
        summary["model_cost_env_candidate_evals_total_including_gd"] = (
            model_cost_env_candidate_evals_including_gd
        )
    success_rates = []
    task_scores = []
    for row in measured:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        success_rate = metrics.get("success_rate")
        if isinstance(success_rate, (int, float)):
            success_rates.append(float(success_rate) / 100.0 if success_rate > 1 else float(success_rate))
        for key in ("task_score_mean", "score_mean", "score", "task_score"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                task_scores.append(float(value))
                break
        for key in ("task_scores", "episode_scores", "scores"):
            values = metrics.get(key)
            if isinstance(values, list):
                task_scores.extend(float(v) for v in values if isinstance(v, (int, float)))
                break
    if success_rates:
        summary["success_rate"] = {
            "count": len(success_rates),
            "mean": mean(success_rates),
            "min": min(success_rates),
            "max": max(success_rates),
        }
    if task_scores:
        summary["task_score"] = {
            "count": len(task_scores),
            "mean": mean(task_scores),
            "min": min(task_scores),
            "max": max(task_scores),
        }
    return summary


def _sum_numeric_field(rows: Iterable[dict[str, Any]], key: str) -> float:
    return sum(
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
    )


def _model_cost_evals_from_cem_call(row: dict[str, Any]) -> int:
    value = row.get("model_cost_evals")
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    sample_counts = row.get("sample_counts_per_step")
    if isinstance(sample_counts, list) and all(
        isinstance(item, (int, float)) for item in sample_counts
    ):
        return int(round(sum(float(item) for item in sample_counts)))
    num_samples = row.get("num_samples")
    n_steps = row.get("n_steps")
    if isinstance(num_samples, (int, float)) and isinstance(n_steps, (int, float)):
        return int(round(float(num_samples) * float(n_steps)))
    return 0


def summarize_wall_attribution(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Split evaluation wall into CEM, non-CEM policy, and outside-policy time."""
    row_list = list(rows)
    evaluation_rows = [row for row in row_list if row.get("type") == "evaluation_result"]
    plan_rows = [row for row in row_list if row.get("type") == "plan_latency"]

    evaluation_time_s_total = _sum_numeric_field(evaluation_rows, "evaluation_time_s")
    plan_latency_ms_total = _sum_numeric_field(plan_rows, "plan_latency_ms")
    cem_cpu_wall_ms_total = _sum_numeric_field(evaluation_rows, "cem_cpu_wall_ms_total")
    cem_cuda_elapsed_ms_total = _sum_numeric_field(
        evaluation_rows,
        "cem_cuda_elapsed_ms_total",
    )
    model_cost_evals_total = int(
        round(_sum_numeric_field(evaluation_rows, "model_cost_evals_total_including_gd"))
    )
    if not model_cost_evals_total:
        model_cost_evals_total = int(
            round(_sum_numeric_field(evaluation_rows, "model_cost_evals_total"))
        )
    if not model_cost_evals_total:
        model_cost_evals_total = sum(
            _model_cost_evals_from_cem_call(row)
            for row in row_list
            if row.get("type") == "cem_call"
        )
    plan_calls_total = int(round(_sum_numeric_field(evaluation_rows, "plan_calls_total")))
    cem_calls_total = int(round(_sum_numeric_field(evaluation_rows, "cem_calls_total")))

    evaluation_wall_ms_total = evaluation_time_s_total * 1000.0
    policy_non_cem_ms_total = max(plan_latency_ms_total - cem_cpu_wall_ms_total, 0.0)
    outside_policy_ms_total = max(evaluation_wall_ms_total - plan_latency_ms_total, 0.0)

    def fraction(value: float) -> float | None:
        if evaluation_wall_ms_total <= 0.0:
            return None
        return value / evaluation_wall_ms_total

    plan_latency_row_count = len(plan_rows)
    plan_latency_complete = (
        plan_calls_total == plan_latency_row_count if plan_calls_total else None
    )

    return {
        "evaluation_time_s_total": evaluation_time_s_total,
        "evaluation_wall_ms_total": evaluation_wall_ms_total,
        "plan_latency_ms_total": plan_latency_ms_total,
        "cem_cpu_wall_ms_total": cem_cpu_wall_ms_total,
        "cem_cuda_elapsed_ms_total": cem_cuda_elapsed_ms_total,
        "model_cost_evals_total": model_cost_evals_total,
        "plan_calls_total": plan_calls_total,
        "cem_calls_total": cem_calls_total,
        "plan_latency_row_count": plan_latency_row_count,
        "plan_latency_complete": plan_latency_complete,
        "policy_non_cem_ms_total": policy_non_cem_ms_total,
        "outside_policy_ms_total": outside_policy_ms_total,
        "cem_cpu_fraction_of_eval_wall": fraction(cem_cpu_wall_ms_total),
        "cem_cuda_fraction_of_eval_wall": fraction(cem_cuda_elapsed_ms_total),
        "policy_fraction_of_eval_wall": fraction(plan_latency_ms_total),
        "policy_non_cem_fraction_of_eval_wall": fraction(policy_non_cem_ms_total),
        "outside_policy_fraction_of_eval_wall": fraction(outside_policy_ms_total),
    }
