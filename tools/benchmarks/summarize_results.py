"""Summarize benchmark JSONL results and planner quality gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.metrics import (
    read_jsonl,
    summarize_cem_calls,
    summarize_evaluations,
    summarize_wall_attribution,
)


def _normalized_rate(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    rate = float(value)
    return rate / 100.0 if rate > 1.0 else rate


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [float(item) for item in value if isinstance(item, (int, float))]
    return []


def summarize_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = 0
    success_count = 0
    success_rates = []
    task_scores: list[float] = []
    evaluation_rows = [row for row in rows if row.get("type") == "evaluation_result"]
    for row in evaluation_rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        episode_successes = metrics.get("episode_successes")
        if isinstance(episode_successes, list):
            bool_values = [bool(item) for item in episode_successes]
            successes += sum(1 for item in bool_values if item)
            success_count += len(bool_values)
        else:
            rate = _normalized_rate(metrics.get("success_rate"))
            if rate is not None:
                success_rates.append(rate)

        for key in ("task_scores", "episode_scores", "scores"):
            values = _numeric_values(metrics.get(key))
            if values:
                task_scores.extend(values)
                break
        else:
            for key in ("task_score_mean", "score_mean", "task_score", "score"):
                values = _numeric_values(metrics.get(key))
                if values:
                    task_scores.extend(values)
                    break

    if success_count:
        success_rate = successes / success_count
        num_episodes = success_count
    elif success_rates:
        success_rate = mean(success_rates)
        num_episodes = len(success_rates)
    else:
        success_rate = None
        num_episodes = 0

    eval_summary = summarize_evaluations(rows)
    cem_summary = summarize_cem_calls(rows)
    cem_calls = eval_summary.get("cem_calls_total")
    if cem_calls is None:
        cem_calls = cem_summary.get("call_count", 0)
    model_cost_evals = eval_summary.get("model_cost_evals_total_including_gd")
    if model_cost_evals is None:
        model_cost_evals = eval_summary.get("model_cost_evals_total")
    if model_cost_evals is None:
        cem_eval_summary = cem_summary.get("model_cost_evals", {})
        if isinstance(cem_eval_summary, dict) and cem_eval_summary.get("count"):
            model_cost_evals = int(
                round(float(cem_eval_summary["mean"]) * int(cem_eval_summary["count"]))
            )

    return {
        "success_rate": success_rate,
        "num_episodes": num_episodes,
        "task_score_mean": mean(task_scores) if task_scores else None,
        "task_score_count": len(task_scores),
        "evaluation_time_s": eval_summary.get("evaluation_time_s", {}),
        "cem_calls": cem_calls or 0,
        "model_cost_evals": model_cost_evals or 0,
        "model_cost_env_candidate_evals": eval_summary.get(
            "model_cost_env_candidate_evals_total", 0
        ),
    }


def compare_pusht_non_regression(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = summarize_quality(baseline_rows)
    candidate = summarize_quality(candidate_rows)
    baseline_success = baseline["success_rate"]
    candidate_success = candidate["success_rate"]
    success_pass = (
        baseline_success is not None
        and candidate_success is not None
        and candidate_success >= baseline_success - 0.05
    )

    baseline_score = baseline["task_score_mean"]
    candidate_score = candidate["task_score_mean"]
    task_score_gate_applied = baseline_score is not None and candidate_score is not None
    task_score_pass = (
        candidate_score >= baseline_score - 0.02 if task_score_gate_applied else True
    )
    baseline_score_value = baseline_score if baseline_score is not None else 0.0
    candidate_score_value = candidate_score if candidate_score is not None else 0.0

    return {
        "quality_pass": bool(success_pass and task_score_pass),
        "success_rate_delta": (
            candidate_success - baseline_success
            if baseline_success is not None and candidate_success is not None
            else None
        ),
        "task_score_mean_delta": candidate_score_value - baseline_score_value,
        "baseline_success_rate": baseline_success,
        "candidate_success_rate": candidate_success,
        "baseline_task_score_mean": baseline_score_value,
        "candidate_task_score_mean": candidate_score_value,
        "task_score_gate_applied": task_score_gate_applied,
        "baseline_cem_calls": baseline["cem_calls"],
        "candidate_cem_calls": candidate["cem_calls"],
        "baseline_model_cost_evals": baseline["model_cost_evals"],
        "candidate_model_cost_evals": candidate["model_cost_evals"],
        "baseline_model_cost_env_candidate_evals": baseline[
            "model_cost_env_candidate_evals"
        ],
        "candidate_model_cost_env_candidate_evals": candidate[
            "model_cost_env_candidate_evals"
        ],
        "baseline_num_episodes": baseline["num_episodes"],
        "candidate_num_episodes": candidate["num_episodes"],
        "qualitative_failure_check": "not_evaluated_by_summarizer",
    }


def write_quality_gate(
    *,
    baseline: Path,
    candidate: Path,
    quality_gate: str,
    out: Path | None,
) -> dict[str, Any]:
    if quality_gate != "pusht_non_regression":
        raise ValueError(f"Unsupported quality gate: {quality_gate}")
    result = compare_pusht_non_regression(read_jsonl(baseline), read_jsonl(candidate))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def markdown_summary(inputs: list[Path]) -> str:
    lines = [
        "| File | Success Rate | Episodes | CEM Calls | Model-Cost Evals |",
        "|---|---:|---:|---:|---:|",
    ]
    for path in inputs:
        summary = summarize_quality(read_jsonl(path))
        rate = summary["success_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lines.append(
            "| "
            f"{path} | {rate_text} | {summary['num_episodes']} | "
            f"{summary['cem_calls']} | {summary['model_cost_evals']} |"
        )
    return "\n".join(lines) + "\n"


def _format_seconds_from_ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) / 1000.0:.2f}"


def _format_fraction(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def wall_attribution_markdown(inputs: list[Path]) -> str:
    lines = [
        "# LeWM Phase 4 Wall Attribution",
        "",
        "| File | Eval Wall (s) | Policy / Eval | CEM CPU / Eval | Policy Non-CEM / Eval | Outside Policy / Eval | Model-Cost Evals | Plan Rows Complete |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in inputs:
        summary = summarize_wall_attribution(read_jsonl(path))
        summaries.append((path, summary))
        lines.append(
            "| "
            f"{path} | "
            f"{summary['evaluation_time_s_total']:.2f} | "
            f"{_format_fraction(summary['policy_fraction_of_eval_wall'])} | "
            f"{_format_fraction(summary['cem_cpu_fraction_of_eval_wall'])} | "
            f"{_format_fraction(summary['policy_non_cem_fraction_of_eval_wall'])} | "
            f"{_format_fraction(summary['outside_policy_fraction_of_eval_wall'])} | "
            f"{summary['model_cost_evals_total']} | "
            f"{summary['plan_latency_complete']} |"
        )

    lines.extend(
        [
            "",
            "## Raw Totals",
            "",
            "| File | Plan Latency (s) | CEM CPU (s) | CEM CUDA (s) | Policy Non-CEM (s) | Outside Policy (s) | Plan Calls | CEM Calls |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for path, summary in summaries:
        lines.append(
            "| "
            f"{path} | "
            f"{_format_seconds_from_ms(summary['plan_latency_ms_total'])} | "
            f"{_format_seconds_from_ms(summary['cem_cpu_wall_ms_total'])} | "
            f"{_format_seconds_from_ms(summary['cem_cuda_elapsed_ms_total'])} | "
            f"{_format_seconds_from_ms(summary['policy_non_cem_ms_total'])} | "
            f"{_format_seconds_from_ms(summary['outside_policy_ms_total'])} | "
            f"{summary['plan_calls_total']} | "
            f"{summary['cem_calls_total']} |"
        )

    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument("--quality-gate", choices=["pusht_non_regression"])
    parser.add_argument("--inputs", nargs="+")
    parser.add_argument("--wall-attribution", action="store_true")
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.baseline or args.candidate or args.quality_gate:
        if not (args.baseline and args.candidate and args.quality_gate):
            raise SystemExit("--baseline, --candidate, and --quality-gate must be used together")
        result = write_quality_gate(
            baseline=Path(args.baseline),
            candidate=Path(args.candidate),
            quality_gate=args.quality_gate,
            out=Path(args.out) if args.out else None,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.inputs:
        input_paths = [Path(path) for path in args.inputs]
        summary = (
            wall_attribution_markdown(input_paths)
            if args.wall_attribution
            else markdown_summary(input_paths)
        )
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(summary, encoding="utf-8")
            if args.wall_attribution:
                print(f"wrote {out}")
        else:
            print(summary, end="")
        return 0
    raise SystemExit("Use either --baseline/--candidate/--quality-gate or --inputs")


if __name__ == "__main__":
    raise SystemExit(main())
