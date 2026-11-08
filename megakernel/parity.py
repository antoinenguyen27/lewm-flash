"""Correctness + latency check for LeWM-MK on one GPU.

1. Model-cost parity: identical candidates through (a) torch FP32 eager (reference), (b) the accepted LeWM-Flash
   BF16-autocast path, (c) the megakernel's single-call cost mode. Reports error vs FP32 and top-30 agreement, with
   (b) as the precision yardstick the megakernel must stay within.
2. Planner quality: full single-env solves from the torch iCEM solver (BF16 autocast) and from the megakernel on the
   same (observation, goal) problems; every returned plan is re-scored by the FP32 model.
3. Latency: megakernel full 30-step solve and single cost calls (CUDA events, warmup excluded).
"""

from __future__ import annotations

import contextlib
import io
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stats(ms):
    s = sorted(ms)
    return {"p50_ms": statistics.median(s), "mean_ms": statistics.fmean(s), "min_ms": s[0],
            "p90_ms": s[int(0.9 * (len(s) - 1))], "max_ms": s[-1], "n": len(s)}


def _time(fn, iters, warmup=3):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return out


def _problems(n, seed=0):
    """Realistic-scale embeddings: observation/goal pairs drawn around the projector's (BatchNorm) unit scale."""
    import torch

    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, 1, 192, generator=g).cuda(), torch.randn(1, 1, 192, generator=g).cuda()) for _ in range(n)]


def run(weights_dir: str, out_dir: str, build_dir: str, args: str = "") -> dict:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from megakernel.baseline import info_for, make_solver
    from megakernel.model_io import load_cost_model
    from megakernel.runtime import Megakernel, icem_sample_counts, load_ext

    opts = dict(a.split("=") for a in args.split(",") if a)
    ext = load_ext(build_dir)
    model, _ = load_cost_model(weights_dir)
    model.set_embedding_cache(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    res: dict = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "smem": ext.smem_bytes(),
                 "max_clusters": {"8": ext.query_max_clusters(8), "16": ext.query_max_clusters(16)}}
    print(json.dumps(res), flush=True)
    cl = int(opts["cl"]) if "cl" in opts else None
    mk = Megakernel(model, ext, cl=cl, clusters=int(opts["clusters"]) if "clusters" in opts else None, max_envs=32)
    res["config"] = {"cl": mk.cl, "clusters": mk.clusters}
    print("config", res["config"], flush=True)

    def ref_cost(emb, goal, cand, bf16):
        model.set_model_cost_autocast("bf16" if bf16 else "none")
        with torch.inference_mode():
            return model.get_cost_from_embeddings(emb, goal, cand.view(1, -1, 5, 10)).view(-1).float()

    # ---------------------------------------------------------------- 1. model-cost parity
    parity = []
    for i, (emb, goal) in enumerate(_problems(4, seed=1)):
        for S, scale in ((300, 1.0), (60, 0.3)):
            g = torch.Generator(device="cuda").manual_seed(100 + i)
            cand = torch.randn(S, 5, 10, device="cuda", generator=g) * scale
            c32 = ref_cost(emb, goal, cand, False)
            c16 = ref_cost(emb, goal, cand, True)
            cmk = mk.costs(emb, goal, cand)
            torch.cuda.synchronize()

            def cmp(c):
                rel = ((c - c32).abs() / c32.abs().clamp_min(1e-6))
                top_ref = set(torch.topk(c32, 30, largest=False).indices.tolist())
                top = set(torch.topk(c, 30, largest=False).indices.tolist())
                return {"rel_max": rel.max().item(), "rel_mean": rel.mean().item(), "top30_overlap": len(top & top_ref),
                        "argmin_match": int(c.argmin().item() == c32.argmin().item())}

            row = {"problem": i, "S": S, "scale": scale, "fp32_cost_mean": c32.mean().item(),
                   "flash_bf16": cmp(c16), "megakernel": cmp(cmk), "mk_finite": bool(torch.isfinite(cmk).all())}
            parity.append(row)
            print(json.dumps(row), flush=True)
    res["cost_parity"] = parity

    # batched cost mode: 6 envs x S candidates in one launch, each env checked against its own FP32 reference
    probs = _problems(6, seed=11)
    embs = torch.cat([e.view(1, 192) for e, _ in probs]); goals = torch.cat([g.view(1, 192) for _, g in probs])
    for S in (60, 300):
        cand = torch.randn(6, S, 5, 10, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))
        cmk = mk.costs(embs, goals, cand)
        worst = 0.0
        overlap = []
        for i, (emb, goal) in enumerate(probs):
            c32 = ref_cost(emb, goal, cand[i], False)
            worst = max(worst, ((cmk[i] - c32).abs() / c32.abs().clamp_min(1e-6)).max().item())
            overlap.append(len(set(torch.topk(cmk[i], 30, largest=False).indices.tolist())
                               & set(torch.topk(c32, 30, largest=False).indices.tolist())))
        row = {"batched_envs": 6, "S": S, "mk_rel_max": worst, "top30_overlap_min": min(overlap)}
        res.setdefault("batched_cost_parity", []).append(row)
        print(json.dumps(row), flush=True)

    # ---------------------------------------------------------------- 3. latency
    emb, goal = _problems(1, seed=7)[0]
    counts = icem_sample_counts()
    res["sample_counts"] = counts
    res["mk_solve_ms"] = _stats(_time(lambda: mk.solve(emb, goal), int(opts.get("iters", 50)), 5))
    print("mk solve", res["mk_solve_ms"], flush=True)
    per_s = {}
    for S in (300, 240, 180, 120, 60):
        cand = torch.randn(S, 5, 10, device="cuda")
        per_s[S] = _stats(_time(lambda: mk.costs(emb, goal, cand), 30, 3))
    res["mk_cost_call_ms"] = per_s
    print("mk cost calls", per_s, flush=True)

    # ---------------------------------------------------------------- 2. planner quality (same problems)
    if opts.get("quality", "1") == "1":
        model.set_model_cost_autocast("bf16")
        model.set_compile_cost(True)
        torch.set_float32_matmul_precision("high")
        solver = make_solver(model, seed=0)
        quality = []
        qprobs = _problems(int(opts.get("problems", 16)), seed=3)
        qe = torch.cat([e.view(1, 192) for e, _ in qprobs]); qg = torch.cat([g.view(1, 192) for _, g in qprobs])
        bm, bs, bc = mk.solve_batch(qe, qg, seed=4242)  # all problems' solves in one launch
        bm, bs, bc = bm.clone(), bs.clone(), bc.clone()
        for i, (emb, goal) in enumerate(qprobs):
            with contextlib.redirect_stdout(io.StringIO()):
                out = solver.solve(info_for(emb, goal))
            plan_torch = out["actions"].cuda().view(1, 5, 10)
            mean, std, fc = bm[i], bs[i], bc[i]
            plan_mk = mean.clone().view(1, 5, 10)
            torch.set_float32_matmul_precision("highest")
            c_t = ref_cost(emb, goal, plan_torch, False).item()
            c_m = ref_cost(emb, goal, plan_mk, False).item()
            torch.set_float32_matmul_precision("high")
            row = {"problem": i, "torch_plan_fp32_cost": c_t, "mk_plan_fp32_cost": c_m,
                   "mk_final_topk_cost": float(fc), "mk_std_mean": std.mean().item(),
                   "torch_std_mean": out["var"][0].mean().item()}
            quality.append(row)
            print(json.dumps(row), flush=True)
        ct = [r["torch_plan_fp32_cost"] for r in quality]
        cm = [r["mk_plan_fp32_cost"] for r in quality]
        res["planner_quality"] = {"rows": quality, "torch_median": statistics.median(ct), "mk_median": statistics.median(cm),
                                  "mk_better_or_equal": sum(m <= t for m, t in zip(cm, ct))}
        print(json.dumps({k: v for k, v in res["planner_quality"].items() if k != "rows"}), flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "parity.json").write_text(json.dumps(res, indent=1))
    return {k: v for k, v in res.items() if k != "planner_quality"} | (
        {"planner_quality": {k: v for k, v in res["planner_quality"].items() if k != "rows"}}
        if "planner_quality" in res else {})
