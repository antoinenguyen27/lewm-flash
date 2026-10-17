"""Statistics for the paper, from the committed run records (results/runs, results/ladder, results/megakernel).

Latency: batch-1 end-to-end plan latency per rung (61 one-env episodes, first seed dropped as warmup), median and
p90 with a bootstrap 95% interval on the median (10,000 resamples over plans).
Success: 3 seeds x 100 Push-T episodes per rung. Episode start states are a deterministic function of the seed, so
every rung sees the same 300 episodes and comparisons are paired: discordant pairs, exact McNemar test, paired
bootstrap 95% interval on the difference in success rate (10,000 resamples over episodes), Wilson 95% intervals.
Writes data/ladder_stats.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "results" / "runs"
OUT = Path(__file__).resolve().parent / "data" / "ladder_stats.json"
FINAL = "h100_20261017t092213z"
PHASE_A = "h100_20261017t075029z"
PLAN_INDICES = {1, 26}
RUNGS = ["r0", "r1", "r2", "r2b", "r3", "r3cem", "r4"]
QUALITY = {
    "r0": "lewm_baseline_h100_e100_s0-1-2_20261016t191013z",
    "r1": "ladder_r1_quality_h100_20261017t075029z",
    "r2": "ladder_r2_quality_h100_20261017t075029z",
    "r2b": "lewm_flash_h100_e100_s0-1-2_20261016t101037z",
    "r3": f"ladder_r3_quality_{FINAL}",
    "r3cem": f"ladder_r3cem_quality_{FINAL}",
    "r4": f"ladder_r4_quality_{FINAL}",
}
PAIRS = [("r1", "r0"), ("r2", "r1"), ("r2b", "r2"), ("r3", "r2"), ("r3", "r2b"), ("r3cem", "r1"), ("r3", "r3cem"),
         ("r4", "r3"), ("r4", "r0")]
rng = np.random.default_rng(0)


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def latency(run: str) -> dict:
    lat = [r for r in rows(RUNS / run / "metrics.jsonl") if r["type"] == "plan_latency"]
    seeds = sorted({r["seed"] for r in lat})
    m = [r for r in lat if r["seed"] != seeds[0]]
    plan = np.array([r["plan_latency_ms"] for r in m if r["plan_index"] in PLAN_INDICES])
    step = np.array([r["plan_latency_ms"] for r in m if r["plan_index"] not in PLAN_INDICES])
    boots = np.median(rng.choice(plan, size=(10000, plan.size)), axis=1)
    return {"n_plans": int(plan.size), "n_steps": int(step.size), "seeds": len(seeds) - 1,
            "p50": float(np.median(plan)), "p90": float(np.percentile(plan, 90)), "mean": float(plan.mean()),
            "p50_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "min": float(plan.min()), "max": float(plan.max()), "step_p50": float(np.median(step)),
            "plans": [float(x) for x in plan]}


def successes(run: str) -> np.ndarray:
    ev = sorted((r for r in rows(RUNS / run / "metrics.jsonl") if r["type"] == "evaluation_result"),
                key=lambda r: r["seed"])
    assert [e["seed"] for e in ev] == [0, 1, 2], run
    return np.array([bool(x) for e in ev for x in e["metrics"]["episode_successes"]], dtype=int)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    idx = rng.integers(0, a.size, size=(10000, a.size))
    d = (a[idx] - b[idx]).mean(axis=1)
    return {"delta": float(a.mean() - b.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "a_only": only_a, "b_only": only_b, "mcnemar_p": float(p)}


def main() -> None:
    out = {"latency": {}, "latency_phaseA": {}, "success": {}, "paired": {}}
    for r in RUNGS:
        out["latency"][r] = latency(f"ladder_{r}_batch1_{FINAL}")
        if (RUNS / f"ladder_{r}_batch1_{PHASE_A}").exists():
            a = latency(f"ladder_{r}_batch1_{PHASE_A}")
            out["latency_phaseA"][r] = {k: v for k, v in a.items() if k != "plans"}
    succ = {r: successes(QUALITY[r]) for r in RUNGS}
    for r, s in succ.items():
        k, n = int(s.sum()), int(s.size)
        out["success"][r] = {"k": k, "n": n, "rate": k / n, "wilson": list(wilson(k, n)),
                             "per_seed": [int(s[i * 100:(i + 1) * 100].sum()) for i in range(3)], "run": QUALITY[r]}
    for a, b in PAIRS:
        out["paired"][f"{a}_vs_{b}"] = paired(succ[a], succ[b])
    out["paired"].update({f"{r}_vs_r0": paired(succ[r], succ["r0"]) for r in RUNGS if r != "r0"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for r in RUNGS:
        L, S = out["latency"][r], out["success"][r]
        P = out["paired"].get(f"{r}_vs_r0")
        print(f"{r:6s} p50 {L['p50']:7.1f} [{L['p50_ci'][0]:.1f},{L['p50_ci'][1]:.1f}] p90 {L['p90']:7.1f} n={L['n_plans']} "
              f"step {L['step_p50']:.2f} | {S['k']}/{S['n']} {S['per_seed']}"
              + (f" | vs r0 {100*P['delta']:+.1f} [{100*P['ci'][0]:+.1f},{100*P['ci'][1]:+.1f}] "
                 f"{P['a_only']}/{P['b_only']} p={P['mcnemar_p']:.2f}" if P else ""))
    for k, P in out["paired"].items():
        if not k.endswith("_vs_r0"):
            print(f"  {k:12s} {100*P['delta']:+.1f} [{100*P['ci'][0]:+.1f},{100*P['ci'][1]:+.1f}] {P['a_only']}/{P['b_only']} p={P['mcnemar_p']:.2f}")


if __name__ == "__main__":
    main()
