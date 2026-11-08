"""Summarize the H100 optimization ladder (docs/ladder.md).

batch-1 runs (one env per episode, many seeds, first seed = warmup): end-to-end plan latency per replanning call
(policy.get_action CUDA events: preprocessing + encoder + solve + action to host) and per-step latency of the
non-planning calls. quality runs (3 seeds x 100 episodes): Push-T success with a 95% Wilson interval.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

PLAN_INDICES = {1, 26}  # one plan covers 5 blocks x 5 env steps, so a 1-env episode replans at calls 1 and 26


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def batch1(path: Path) -> dict:
    rows = [r for r in _rows(path) if r["type"] == "plan_latency"]
    seeds = sorted({r["seed"] for r in rows})
    measured = [r for r in rows if r["seed"] != seeds[0]]  # first seed warms up kernels / compile caches
    plan = [r["plan_latency_ms"] for r in measured if r["plan_index"] in PLAN_INDICES]
    step = [r["plan_latency_ms"] for r in measured if r["plan_index"] not in PLAN_INDICES]
    return {"seeds_measured": len(seeds) - 1, "plans": len(plan), "plan_p50_ms": statistics.median(plan),
            "plan_p90_ms": _pct(plan, 0.9), "plan_mean_ms": statistics.fmean(plan),
            "step_p50_ms": statistics.median(step) if step else None}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def quality(path: Path) -> dict:
    ev = [r for r in _rows(path) if r["type"] == "evaluation_result"]
    succ = [sum(bool(x) for x in e["metrics"]["episode_successes"]) for e in ev]
    n = sum(len(e["metrics"]["episode_successes"]) for e in ev)
    lo, hi = wilson(sum(succ), n)
    return {"per_seed": succ, "success": sum(succ), "episodes": n, "rate": sum(succ) / n, "ci95": [lo, hi]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch1", nargs="*", default=[], help="rung=path/to/metrics.jsonl")
    ap.add_argument("--quality", nargs="*", default=[], help="rung=path/to/metrics.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"batch1": {}, "quality": {}}
    for spec in a.batch1:
        k, v = spec.split("=", 1)
        res["batch1"][k] = batch1(Path(v))
    for spec in a.quality:
        k, v = spec.split("=", 1)
        res["quality"][k] = quality(Path(v))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
