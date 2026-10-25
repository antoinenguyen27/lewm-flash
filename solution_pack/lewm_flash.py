#!/usr/bin/env python3
"""LeWM-Flash solution program: profile, explain and evaluate the latency of world-model robot planning.

This one file is the runnable part of the solution pack. It follows the analytics pipeline of the report:

    prepare    data preparation: extract tidy CSV datasets from the raw benchmark records of the repository
    analyse    preprocessing, compute-profile analysis (pattern discovery), statistical evaluation, tables, figures
    reproduce  re-collect the raw data on a cloud H100 (Modal account required; costs money)

`analyse` is the default, needs only Python 3.10+, numpy and matplotlib, and runs on any computer from the CSVs that
ship in ./data. Everything it computes is written to ./outputs (results.json, tables.md, figs/*.png).

    python lewm_flash.py                 # = analyse
    python lewm_flash.py prepare         # rebuild ./data from ../results (inside the project repository)
    python lewm_flash.py reproduce --seeds 11 --yes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "outputs"
REPO = HERE.parent

# ---------------------------------------------------------------------------------------------------------------------
# Systems and probes. A "system" is a complete planner configuration a user would deploy; a "probe" is a configuration
# run only to isolate one effect for the compute-profile analysis.
SYSTEMS = {
    "reference": "Reference LeWM planner (CEM 300x30, PyTorch eager, FP32)",
    "pytorch_optimised": "Optimised PyTorch planner (iCEM, torch.compile, TF32, BF16)",
    "megakernel": "LeWM-MK: whole plan in one persistent GPU kernel (iCEM, BF16/FP32)",
}
PROBES = {
    "probe_cached_cem": "reference with embeddings encoded once per plan, CEM",
    "probe_cached_icem": "reference with embeddings encoded once per plan, iCEM",
    "probe_mk_cem": "megakernel solve with PyTorch encoder, CEM 300x30",
    "probe_mk_icem": "megakernel solve with PyTorch encoder, iCEM",
}
# raw run directories (results/runs) behind each configuration; batch-1 latency pass and 300-episode quality runs
FINAL = "h100_20261017t092213z"
RAW_LATENCY = {"reference": "r0", "pytorch_optimised": "r2b", "megakernel": "r4", "probe_cached_cem": "r1",
               "probe_cached_icem": "r2", "probe_mk_cem": "r3cem", "probe_mk_icem": "r3"}
RAW_QUALITY = {"reference": "lewm_baseline_h100_e100_s0-1-2_20261016t191013z",
               "pytorch_optimised": "lewm_flash_h100_e100_s0-1-2_20261016t101037z",
               "megakernel": f"ladder_r4_quality_{FINAL}"}
PLAN_CALLS = {1, 26}       # policy calls on which the planner runs (others pop already-planned actions)
CEM_ITERS, CEM_EVALS, ICEM_EVALS = 30, 9000, 2820
LAYERS_PER_PLAN = 30 * 5 * 6  # iterations x predictor passes x transformer layers
COLORS = {"reference": "#52514e", "pytorch_optimised": "#eb6834", "megakernel": "#2a78d6"}
LABELS = {"reference": "Reference", "pytorch_optimised": "Optimised PyTorch", "megakernel": "LeWM-MK (this work)"}


# ---------------------------------------------------------------------------------------------------------------------
# small statistics toolkit (no scipy needed)
def median(x):
    s = sorted(x); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(x, q):
    s = sorted(x); pos = q * (len(s) - 1); lo = math.floor(pos); hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def ols(x, y):
    """Least-squares line y = a + b x, with R^2."""
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x); sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx; a = my - b * mx
    ss_res = sum((yi - a - b * xi) ** 2 for xi, yi in zip(x, y)); ss_tot = sum((yi - my) ** 2 for yi in y)
    return a, b, (1 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def bootstrap(fn, samples, reps=10000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(s) for s in samples]
    out = np.empty(reps)
    for r in range(reps):
        out[r] = fn(*[a[rng.integers(0, a.size, a.size)] for a in arrs])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_bootstrap(a, b, reps=10000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed); a = np.asarray(a); b = np.asarray(b)
    idx = rng.integers(0, a.size, size=(reps, a.size))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def read_csv(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows, fields):
    DATA.mkdir(exist_ok=True)
    with open(DATA / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  wrote data/{name} ({len(rows)} rows)")


# ---------------------------------------------------------------------------------------------------------------------
# 1. data preparation: raw benchmark records -> tidy CSV datasets
def prepare(args):
    runs, mk = REPO / "results" / "runs", REPO / "results" / "megakernel"
    if not runs.exists():
        sys.exit("prepare needs the project repository (results/runs not found); the bundled ./data is already prepared")
    stamp = args.stamp or FINAL

    def jsonl(path):
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    calls = []
    for cfg, rung in RAW_LATENCY.items():
        d = runs / f"ladder_{rung}_batch1_{stamp}"
        if not d.exists():
            print(f"  skip {cfg}: {d.name} not found"); continue
        for r in jsonl(d / "metrics.jsonl"):
            if r["type"] == "plan_latency":
                calls.append({"config": cfg, "role": "system" if cfg in SYSTEMS else "probe", "seed": r["seed"],
                              "call_index": r["plan_index"], "latency_ms": round(r["plan_latency_ms"], 4)})
    write_csv("calls.csv", calls, ["config", "role", "seed", "call_index", "latency_ms"])

    eps = []
    for cfg, run in RAW_QUALITY.items():
        for r in jsonl(runs / run / "metrics.jsonl"):
            if r["type"] == "evaluation_result":
                for i, s in enumerate(r["metrics"]["episode_successes"]):
                    eps.append({"config": cfg, "seed": r["seed"], "episode": i, "success": int(bool(s))})
    write_csv("episodes.csv", eps, ["config", "seed", "episode", "success"])

    tb = json.loads((mk / "torch_baseline_h100.json").read_text())
    kp = json.loads((mk / "parity_final_kernel_cl16_h100.json").read_text())
    scaling = []
    for key, label in [("eager_fp32", "pytorch_eager_fp32"), ("flash_compile_bf16", "pytorch_compiled_bf16")]:
        for n, v in tb[key]["get_cost_ms_by_samples"].items():
            scaling.append({"execution": label, "candidates": int(n), "p50_ms": round(v["p50_ms"], 4), "n": v["n"]})
    for n, v in kp["mk_cost_call_ms"].items():
        scaling.append({"execution": "megakernel", "candidates": int(n), "p50_ms": round(v["p50_ms"], 4), "n": v["n"]})
    write_csv("cost_scaling.csv", sorted(scaling, key=lambda r: (r["execution"], r["candidates"])),
              ["execution", "candidates", "p50_ms", "n"])

    enc = json.loads((mk / "encoder_parity_h100.json").read_text())
    prim = json.loads((mk / "primitives_h100.json").read_text())["cl16"]
    base = jsonl(runs / RAW_QUALITY["reference"] / "metrics.jsonl")
    p26 = [r for r in base if r["type"] == "plan_latency" and r["plan_index"] == 26]
    envs = [round(r["model_cost_env_candidate_evals_delta"] / r["model_cost_evals_delta"]) for r in p26]
    comp = [
        {"component": "encoder_fp32_obs_goal", "value": enc["torch_fp32_encode_2img_ms"]["p50_ms"], "unit": "ms",
         "source": "encoder micro-benchmark, PyTorch FP32, observation + goal"},
        {"component": "encoder_fused_obs_goal", "value": enc["fused_encode_obs+goal_ms"]["p50_ms"], "unit": "ms",
         "source": "encoder micro-benchmark, in-kernel, observation + goal"},
        {"component": "encoder_fused_obs", "value": enc["fused_encode_obs_ms"]["p50_ms"], "unit": "ms",
         "source": "encoder micro-benchmark, in-kernel, observation only (goal cached)"},
        {"component": "solve_megakernel", "value": kp["mk_solve_ms"]["p50_ms"], "unit": "ms",
         "source": "single-robot iCEM solve, final kernel (16-CTA clusters)"},
        {"component": "solve_pytorch_compiled_bf16", "value": tb["flash_compile_bf16"]["solve_ms"]["p50_ms"], "unit": "ms",
         "source": "single-robot iCEM solve, torch.compile + TF32 + BF16"},
        {"component": "solve_pytorch_eager_fp32", "value": tb["eager_fp32"]["solve_ms"]["p50_ms"], "unit": "ms",
         "source": "single-robot iCEM solve, eager FP32 (micro-benchmark harness)"},
        {"component": "cluster_barrier", "value": prim["cluster_sync_us"], "unit": "us", "source": "primitive micro-benchmark"},
        {"component": "grid_barrier", "value": prim["grid_sync_us"], "unit": "us", "source": "primitive micro-benchmark"},
        {"component": "empty_launch", "value": prim["empty_launch_ms"] * 1000, "unit": "us", "source": "primitive micro-benchmark"},
        {"component": "reference_encodes_per_plan_per_robot", "value": sum(r["encode_delta"]["total"] for r in p26) / sum(envs),
         "unit": "count", "source": "reference planner telemetry, 100-robot evaluation, plan call 26"},
        {"component": "reference_solver_share_of_plan_call",
         "value": sum(r["cem_cuda_elapsed_ms_delta"] for r in p26) / sum(r["plan_latency_ms"] for r in p26),
         "unit": "fraction", "source": "reference planner telemetry, 100-robot evaluation, plan call 26"},
    ]
    for c in comp:
        c["value"] = round(c["value"], 5)
    write_csv("components.csv", comp, ["component", "value", "unit", "source"])

    par = []
    for c in kp["cost_parity"]:
        par.append({"check": f"predictor_cost_problem{c['problem']}_S{c['S']}_scale{c['scale']}",
                    "pytorch_bf16_rel_max": round(c["flash_bf16"]["rel_max"], 6), "megakernel_rel_max": round(c["megakernel"]["rel_max"], 6),
                    "pytorch_bf16_top30": c["flash_bf16"]["top30_overlap"], "megakernel_top30": c["megakernel"]["top30_overlap"]})
    for i, c in enumerate(enc["cost_from_fused_embeddings"]):
        par.append({"check": f"cost_from_fused_encoder_case{i}", "pytorch_bf16_rel_max": "", "megakernel_rel_max": round(c["rel_max"], 6),
                    "pytorch_bf16_top30": "", "megakernel_top30": c["top30_overlap"]})
    ev = enc["encoder_vs_fp32"]
    par.append({"check": "encoder_embedding_rel_l2", "pytorch_bf16_rel_max": round(ev["torch_bf16_autocast"]["rel_l2"], 6),
                "megakernel_rel_max": round(ev["megakernel_fused"]["rel_l2"], 6), "pytorch_bf16_top30": "", "megakernel_top30": ""})
    write_csv("parity.csv", par, ["check", "pytorch_bf16_rel_max", "megakernel_rel_max", "pytorch_bf16_top30", "megakernel_top30"])


# ---------------------------------------------------------------------------------------------------------------------
# 2-4. preprocessing, compute-profile analysis, evaluation
def analyse(args):
    import numpy as np
    OUT.mkdir(exist_ok=True); (OUT / "figs").mkdir(exist_ok=True)
    res = {"latency": {}, "profile": {}, "scaling": {}, "evaluation": {}, "parity": {}}

    # 2. preprocessing: drop the warmup seed of every configuration, split planning calls from action-pop calls,
    #    flag (but keep) outliers with the 1.5 IQR rule
    calls = read_csv("calls.csv")
    by_cfg = {}
    for r in calls:
        by_cfg.setdefault(r["config"], []).append((int(r["seed"]), int(r["call_index"]), float(r["latency_ms"])))
    plans, steps = {}, {}
    for cfg, rows in by_cfg.items():
        warm = min(s for s, _, _ in rows)
        kept = [r for r in rows if r[0] != warm]
        plans[cfg] = [ms for _, c, ms in kept if c in PLAN_CALLS]
        steps[cfg] = [ms for _, c, ms in kept if c not in PLAN_CALLS]
        q1, q3 = quantile(plans[cfg], 0.25), quantile(plans[cfg], 0.75)
        out = [x for x in plans[cfg] if x < q1 - 1.5 * (q3 - q1) or x > q3 + 1.5 * (q3 - q1)]
        res["latency"][cfg] = {
            "plans": len(plans[cfg]), "step_calls": len(steps[cfg]), "warmup_seed_dropped": warm,
            "p50": median(plans[cfg]), "p50_ci": bootstrap(np.median, [plans[cfg]]), "p90": quantile(plans[cfg], 0.9),
            "iqr": [q1, q3], "mean": sum(plans[cfg]) / len(plans[cfg]), "min": min(plans[cfg]), "max": max(plans[cfg]),
            "outliers": len(out), "step_p50": median(steps[cfg])}

    comp = {r["component"]: float(r["value"]) for r in read_csv("components.csv")}
    L = res["latency"]

    # 3a. pattern discovery: how does one cost call scale with the number of candidates? fit latency = a + b * N
    scal = {}
    for r in read_csv("cost_scaling.csv"):
        scal.setdefault(r["execution"], []).append((int(r["candidates"]), float(r["p50_ms"])))
    for ex, pts in scal.items():
        pts.sort(); a, b, r2 = ols([p[0] for p in pts], [p[1] for p in pts])
        fixed = a / (a + 300 * b)
        # what the fitted model predicts iCEM (2,820 evaluations in 30 calls) saves over CEM (9,000 in 30 calls)
        pred = (CEM_ITERS * a + b * CEM_EVALS) / (CEM_ITERS * a + b * ICEM_EVALS)
        res["scaling"][ex] = {"points": pts, "intercept_ms": a, "slope_ms_per_100": 100 * b, "r2": r2,
                              "fixed_share_at_300": fixed, "predicted_icem_speedup": pred,
                              "regime": "launch/latency-bound" if fixed > 0.8 else "work-proportional"}
    if all(k in L for k in ("probe_cached_cem", "probe_cached_icem", "probe_mk_cem", "probe_mk_icem")):
        res["scaling"]["observed_icem_speedup"] = {
            "pytorch_eager": L["probe_cached_cem"]["p50"] / L["probe_cached_icem"]["p50"],
            "megakernel": L["probe_mk_cem"]["p50"] / L["probe_mk_icem"]["p50"]}

    # 3b. compute profile of one plan, by stage. Measured components where they exist, the rest by difference.
    enc2 = comp["encoder_fp32_obs_goal"]
    ref, cached = L["reference"], L.get("probe_cached_cem")
    prof_ref = {"CPU preprocessing and policy": ref["step_p50"],
                "image encoding (60 encoder passes)": CEM_ITERS * enc2,
                "predictor rollouts and CEM updates": cached["p50"] - cached["step_p50"] - enc2,
                }
    prof_ref["host synchronisation and other (by difference)"] = ref["p50"] - sum(prof_ref.values())
    mk = L["megakernel"]
    prof_mk = {"fused encoder (observation; goal cached)": comp["encoder_fused_obs"],
               "planner solve (30 iterations, one launch)": comp["solve_megakernel"]}
    prof_mk["CPU, policy and transfers (by difference)"] = mk["p50"] - sum(prof_mk.values())
    rollout_iter = prof_ref["predictor rollouts and CEM updates"] / CEM_ITERS
    sync_budget_ms = (LAYERS_PER_PLAN * 4 * comp["cluster_barrier"] + CEM_ITERS * 8 * comp["grid_barrier"]) / 1000
    res["profile"] = {
        "reference": prof_ref, "megakernel": prof_mk,
        "reference_encodes_per_plan": comp["reference_encodes_per_plan_per_robot"],
        "reference_solver_share_of_plan_call": comp["reference_solver_share_of_plan_call"],
        "reference_ms_per_iteration": rollout_iter,
        "reference_us_per_layer_evaluation": 1000 * prof_ref["predictor rollouts and CEM updates"] / LAYERS_PER_PLAN,
        "megakernel_us_per_layer_evaluation": 1000 * comp["solve_megakernel"] / LAYERS_PER_PLAN,
        "megakernel_sync_budget_ms": sync_budget_ms,
        "megakernel_sync_share_of_solve": sync_budget_ms / comp["solve_megakernel"],
        "solve_ms": {"pytorch_eager_fp32_microbenchmark": comp["solve_pytorch_eager_fp32"],
                     "pytorch_compiled_bf16": comp["solve_pytorch_compiled_bf16"], "megakernel": comp["solve_megakernel"]},
    }

    # 4. evaluation: speed (ratio of medians, bootstrap CI) and accuracy (paired episodes)
    for a, b in [("reference", "megakernel"), ("pytorch_optimised", "megakernel"), ("reference", "pytorch_optimised")]:
        res["evaluation"][f"speedup_{b}_over_{a}"] = {
            "ratio": L[a]["p50"] / L[b]["p50"],
            "ci": bootstrap(lambda x, y: np.median(x) / np.median(y), [plans[a], plans[b]])}
    eps = {}
    for r in read_csv("episodes.csv"):
        eps.setdefault(r["config"], {})[(int(r["seed"]), int(r["episode"]))] = int(r["success"])
    keys = sorted(next(iter(eps.values())))
    succ = {c: [eps[c][k] for k in keys] for c in eps}
    for c, s in succ.items():
        k, n = sum(s), len(s)
        res["evaluation"][f"success_{c}"] = {"k": k, "n": n, "rate": k / n, "wilson": wilson(k, n),
                                             "per_seed": [sum(eps[c][kk] for kk in keys if kk[0] == sd) for sd in (0, 1, 2)]}
    for a, b in [("megakernel", "reference"), ("pytorch_optimised", "reference"), ("megakernel", "pytorch_optimised")]:
        x, y = succ[a], succ[b]
        wa = sum(1 for p, q in zip(x, y) if p and not q); wb = sum(1 for p, q in zip(x, y) if q and not p)
        res["evaluation"][f"paired_{a}_vs_{b}"] = {
            "difference_points": 100 * (sum(x) - sum(y)) / len(x), "ci_points": [100 * v for v in paired_bootstrap(x, y)],
            "wins": wa, "losses": wb, "mcnemar_p": mcnemar_exact(wa, wb)}
    # power: smallest paired difference detectable at 80% power (Connor, 1987) at the observed discordance
    P = res["evaluation"]["paired_megakernel_vs_reference"]; pd = (P["wins"] + P["losses"]) / len(keys)
    n_need = lambda d: (1.96 * math.sqrt(pd) + 0.8416 * math.sqrt(pd - d * d)) ** 2 / d ** 2
    res["evaluation"]["power"] = {"discordance": pd, "detectable_points_at_300": 100 * next(d / 1000 for d in range(5, 300) if n_need(d / 1000) <= len(keys)),
                                  "pairs_needed_for_2_points": round(n_need(0.02))}

    par = read_csv("parity.csv")
    pc = [r for r in par if r["check"].startswith("predictor_cost")]
    res["parity"] = {
        "predictor_cost_rel_max": {"pytorch_bf16": [min(float(r["pytorch_bf16_rel_max"]) for r in pc), max(float(r["pytorch_bf16_rel_max"]) for r in pc)],
                                   "megakernel": [min(float(r["megakernel_rel_max"]) for r in pc), max(float(r["megakernel_rel_max"]) for r in pc)]},
        "top30_overlap": {"pytorch_bf16": [min(int(r["pytorch_bf16_top30"]) for r in pc), max(int(r["pytorch_bf16_top30"]) for r in pc)],
                          "megakernel": [min(int(r["megakernel_top30"]) for r in pc), max(int(r["megakernel_top30"]) for r in pc)]},
        "encoder_rel_l2": {r["check"]: r for r in par if r["check"] == "encoder_embedding_rel_l2"}["encoder_embedding_rel_l2"],
        "cost_from_fused_encoder_rel_max": max(float(r["megakernel_rel_max"]) for r in par if r["check"].startswith("cost_from_fused")),
    }

    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_tables(res)
    if not args.no_figs:
        figures(res, plans, succ)
    summary(res)


def write_tables(res):
    L, E, P = res["latency"], res["evaluation"], res["profile"]
    t = ["# LeWM-Flash results (generated by lewm_flash.py analyse)", "", "## Plan latency, one robot, one H100 (ms)", "",
         "| configuration | plans | p50 [95% CI] | p90 | IQR | step-call p50 |", "|---|---|---|---|---|---|"]
    for c in [*SYSTEMS, *PROBES]:
        if c in L:
            v = L[c]
            t.append(f"| {c} | {v['plans']} | {v['p50']:.1f} [{v['p50_ci'][0]:.1f}, {v['p50_ci'][1]:.1f}] | {v['p90']:.1f} | "
                     f"{v['iqr'][0]:.1f}-{v['iqr'][1]:.1f} | {v['step_p50']:.2f} |")
    t += ["", "## Compute profile of one plan (ms)", ""]
    for c in ("reference", "megakernel"):
        t.append(f"**{c}** ({sum(P[c].values()):.1f} ms)"); t.append("")
        t += [f"- {k}: {v:.1f} ({100 * v / sum(P[c].values()):.0f}%)" for k, v in P[c].items()]; t.append("")
    t += ["## Cost-call scaling: latency = a + b x candidates", "", "| execution | a (ms) | b (ms per 100) | R^2 | fixed share at 300 | predicted iCEM speed-up |",
          "|---|---|---|---|---|---|"]
    for ex, s in res["scaling"].items():
        if "intercept_ms" in s:
            t.append(f"| {ex} | {s['intercept_ms']:.2f} | {s['slope_ms_per_100']:.3f} | {s['r2']:.2f} | {s['fixed_share_at_300']:.2f} | {s['predicted_icem_speedup']:.2f} |")
    if "observed_icem_speedup" in res["scaling"]:
        o = res["scaling"]["observed_icem_speedup"]
        t += ["", f"Observed iCEM speed-up end to end: PyTorch eager {o['pytorch_eager']:.2f}x, megakernel {o['megakernel']:.2f}x."]
    t += ["", "## Evaluation", ""]
    for k, v in E.items():
        if k.startswith("speedup"):
            t.append(f"- {k}: {v['ratio']:.2f}x [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
        elif k.startswith("success"):
            t.append(f"- {k}: {v['k']}/{v['n']} ({100 * v['wilson'][0]:.1f}-{100 * v['wilson'][1]:.1f}%), per seed {v['per_seed']}")
        elif k.startswith("paired"):
            t.append(f"- {k}: {v['difference_points']:+.1f} points [{v['ci_points'][0]:+.1f}, {v['ci_points'][1]:+.1f}], "
                     f"{v['wins']} won / {v['losses']} lost, McNemar p = {v['mcnemar_p']:.2f}")
    pw = E["power"]
    t.append(f"- power: discordance {pw['discordance']:.3f}; detectable at 80% power with 300 pairs: {pw['detectable_points_at_300']:.1f} points; "
             f"pairs for 2 points: {pw['pairs_needed_for_2_points']}")
    (OUT / "tables.md").write_text("\n".join(t) + "\n")


def figures(res, plans, succ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 300,
                         "savefig.bbox": "tight"})
    W, GRID, F = 6.3, "#e6e6e2", OUT / "figs"

    # stage profile, reference vs megakernel
    P = res["profile"]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    pal = {"CPU": "#b7b7b2", "host": "#d9d9d4", "image": "#eb6834", "predictor": "#a8431c", "fused": "#124a8f", "planner": "#2a78d6"}
    for y, c in [(1, "reference"), (0, "megakernel")]:
        x = 0
        for k, v in P[c].items():
            col = next(pal[p] for p in pal if k.startswith(p))
            ax.barh(y, v, left=x, color=col, height=0.55, edgecolor="white", lw=0.8)
            if v > 25: ax.text(x + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white")
            x += v
        ax.text(x + 6, y, f"{x:.1f} ms", va="center", fontsize=8)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["Reference planner", "LeWM-MK"]); ax.set_xlim(0, 700)
    ax.set_xlabel("one plan, one robot, one H100 (ms, median)"); ax.grid(axis="x", color=GRID, lw=0.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=pal["CPU"], label="CPU preprocessing, policy"), Patch(color=pal["image"], label="image encoding (60 passes)"),
                       Patch(color=pal["predictor"], label="predictor rollouts + CEM"), Patch(color=pal["host"], label="host sync, other"),
                       Patch(color=pal["fused"], label="fused encoder"), Patch(color=pal["planner"], label="megakernel solve")],
              frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    fig.savefig(F / "fig_profile.png"); plt.close(fig)

    # cost-call scaling with fitted lines
    fig, ax = plt.subplots(figsize=(W, 2.5))
    sty = {"pytorch_eager_fp32": ("#52514e", "o", "PyTorch eager FP32"), "pytorch_compiled_bf16": ("#a8431c", "s", "PyTorch compiled BF16"),
           "megakernel": ("#2a78d6", "D", "megakernel")}
    xs = np.linspace(50, 310, 50)
    for ex, s in res["scaling"].items():
        if ex not in sty: continue
        col, mk, lab = sty[ex]; pts = np.array(s["points"])
        ax.plot(pts[:, 0], pts[:, 1], mk, color=col, ms=5, mec="white", mew=0.7,
                label=f"{lab}: {s['intercept_ms']:.1f} ms + {s['slope_ms_per_100']:.2f} ms/100 cand. ({100 * s['fixed_share_at_300']:.0f}% fixed)")
        ax.plot(xs, s["intercept_ms"] + s["slope_ms_per_100"] / 100 * xs, color=col, lw=1.1)
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1", "2", "5", "10", "20"]); ax.set_ylim(0.9, 35)
    ax.set_xticks([60, 120, 180, 240, 300]); ax.set_xlabel("candidate action sequences per cost call"); ax.set_ylabel("one cost call (ms)")
    ax.grid(axis="y", color=GRID, lw=0.6, which="both"); ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.6))
    fig.savefig(F / "fig_scaling.png"); plt.close(fig)

    # latency distributions of the three systems
    fig, ax = plt.subplots(figsize=(W, 1.9))
    order = ["reference", "pytorch_optimised", "megakernel"]
    for y, c in zip([2, 1, 0], order):
        v = np.array(plans[c]); j = (np.random.default_rng(1).random(v.size) - 0.5) * 0.35
        ax.scatter(v, y + j, s=5, color=COLORS[c], alpha=0.5, lw=0)
        ax.plot([np.median(v)], [y], "D", color=COLORS[c], ms=7, mec="white", mew=0.8)
        ax.text(1150, y, f"{np.median(v):.1f} ms", va="center", ha="right", fontsize=8)
    ax.set_xscale("log"); ax.set_xlim(30, 1200); ax.set_xticks([30, 50, 100, 200, 500, 1000]); ax.set_xticklabels(["30", "50", "100", "200", "500", "1,000"])
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([LABELS[c] for c in order]); ax.set_xlabel("end-to-end plan latency (ms, log scale)")
    ax.grid(axis="x", color=GRID, lw=0.6, which="both")
    fig.savefig(F / "fig_latency.png"); plt.close(fig)

    # task success with Wilson intervals
    E = res["evaluation"]
    fig, ax = plt.subplots(figsize=(W, 1.7))
    for y, c in zip([2, 1, 0], order):
        s = E[f"success_{c}"]; ax.plot(s["wilson"], [y, y], color=COLORS[c], lw=2.2, solid_capstyle="round")
        ax.plot([s["rate"]], [y], "D", color=COLORS[c], ms=7, mec="white", mew=0.8); ax.text(s["wilson"][1] + 0.004, y, f"{s['k']}/{s['n']}", va="center", fontsize=8)
    ax.axvline(E["success_reference"]["rate"], color="#52514e", ls="--", lw=0.8)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([LABELS[c] for c in order]); ax.set_xlim(0.80, 0.94)
    ax.set_xlabel("Push-T success, 300 paired episodes (Wilson 95% interval)"); ax.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(F / "fig_success.png"); plt.close(fig)
    print(f"  wrote outputs/figs/ (4 figures)")


def summary(res):
    L, E, P = res["latency"], res["evaluation"], res["profile"]
    s = E["speedup_megakernel_over_reference"]; p = E["paired_megakernel_vs_reference"]
    print(f"\nreference {L['reference']['p50']:.1f} ms -> LeWM-MK {L['megakernel']['p50']:.1f} ms per plan: "
          f"{s['ratio']:.1f}x [{s['ci'][0]:.1f}, {s['ci'][1]:.1f}]")
    print(f"vs optimised PyTorch ({L['pytorch_optimised']['p50']:.1f} ms): {E['speedup_megakernel_over_pytorch_optimised']['ratio']:.1f}x")
    print(f"success: {E['success_megakernel']['k']}/300 vs {E['success_reference']['k']}/300, "
          f"{p['difference_points']:+.1f} points [{p['ci_points'][0]:+.1f}, {p['ci_points'][1]:+.1f}], McNemar p = {p['mcnemar_p']:.2f}")
    print(f"reference profile: " + ", ".join(f"{k} {v:.0f} ms" for k, v in P["reference"].items()))
    print("wrote outputs/results.json and outputs/tables.md")


# ---------------------------------------------------------------------------------------------------------------------
# 5. re-collect raw data on a cloud H100 (optional)
def reproduce(args):
    cmd = ["uv", "run", "modal", "run", "modal/megakernel_app.py::ladder", "--rungs", "r0,r2b,r4", "--mode", "batch1",
           "--n-seeds", str(args.seeds)]
    print("This launches three H100 jobs on Modal (reference, optimised PyTorch, LeWM-MK), one robot per episode,\n"
          f"{args.seeds} seeds each (the first is warmup). Expect roughly $0.20 of GPU time per seed per configuration.\n"
          "The kernel must be built first:  uv run modal run modal/megakernel_app.py::build\n\n  " + " ".join(cmd))
    print("\nAfterwards copy the new runs from the 'jepa-results' volume into results/runs/ and run:\n"
          "  modal volume get jepa-results runs/<run_id> results/runs/\n  python solution_pack/lewm_flash.py prepare --stamp <stamp> && python solution_pack/lewm_flash.py")
    if not args.yes:
        print("\n(dry run; add --yes to launch)"); return
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("prepare", help="build ./data from the repository's raw records"); p.add_argument("--stamp", help="run stamp of the latency pass")
    a = sub.add_parser("analyse", help="profile and evaluate (default)"); a.add_argument("--no-figs", action="store_true")
    r = sub.add_parser("reproduce", help="re-collect raw data on a Modal H100"); r.add_argument("--seeds", type=int, default=11); r.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    if args.cmd is None:
        args = ap.parse_args(["analyse"])
    {"prepare": prepare, "analyse": analyse, "reproduce": reproduce}[args.cmd](args)


if __name__ == "__main__":
    main()
