from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def load_rows(run_id: str) -> list[dict]:
    path = RUNS / run_id / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty quantile input")
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[f]
    return values[f] * (c - k) + values[c] * (k - f)


def summarize(rows: list[dict]) -> dict[str, float | int]:
    eval_rows = [row for row in rows if row.get("type") == "evaluation_result"]
    plan_rows = [row for row in rows if row.get("type") == "plan_latency"]
    warmup_excluded = [
        float(row["plan_latency_ms"])
        for row in plan_rows
        if not row.get("warmup")
    ]
    successes = sum(
        bool(success)
        for row in eval_rows
        for success in row.get("metrics", {}).get("episode_successes", [])
    )
    episodes = sum(
        len(row.get("metrics", {}).get("episode_successes", []))
        for row in eval_rows
    )
    return {
        "eval_wall_s": sum(float(row.get("evaluation_time_s", 0.0)) for row in eval_rows),
        "planner_sum_s": sum(warmup_excluded) / 1000.0,
        "planner_p50_ms": quantile(warmup_excluded, 0.5),
        "planner_p90_ms": quantile(warmup_excluded, 0.9),
        "successes": successes,
        "episodes": episodes,
    }


def main() -> int:
    baseline_rows = []
    for seed in (0, 1, 2):
        baseline_rows.extend(load_rows(f"lewm_baseline_telem100_seed{seed}_l40s"))
    best_rows = load_rows("lewm_icem_bucket60_bf16_seed0-2_l40s")

    baseline = summarize(baseline_rows)
    best = summarize(best_rows)

    print(f"No-opt planner sum: {baseline['planner_sum_s']:.2f} s")
    print(f"LeWM-Flash planner sum: {best['planner_sum_s']:.2f} s")
    print(f"Planner speedup: {baseline['planner_sum_s'] / best['planner_sum_s']:.2f}x")
    print(f"No-opt eval wall: {baseline['eval_wall_s']:.2f} s")
    print(f"LeWM-Flash eval wall: {best['eval_wall_s']:.2f} s")
    print(f"Eval wall speedup: {baseline['eval_wall_s'] / best['eval_wall_s']:.2f}x")
    print(
        "Quality: "
        f"{baseline['successes']}/{baseline['episodes']} -> "
        f"{best['successes']}/{best['episodes']}"
    )
    print(
        "Plan p50/p90: "
        f"{baseline['planner_p50_ms']:.2f}/{baseline['planner_p90_ms']:.2f} ms -> "
        f"{best['planner_p50_ms']:.2f}/{best['planner_p90_ms']:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
