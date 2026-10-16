"""Per-CTA time breakdown of the megakernel (globaltimer instrumentation inside the kernel)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(weights_dir: str, out_dir: str, build_dir: str, args: str = "") -> dict:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from megakernel.model_io import load_cost_model
    from megakernel.parity import _stats, _time
    from megakernel.runtime import Megakernel, load_ext

    opts = dict(a.split("=") for a in args.split(",") if a)
    ext = load_ext(build_dir, prof=opts.get("prof", "0") == "1")
    model, _ = load_cost_model(weights_dir)
    res = {}
    for cl in [int(c) for c in opts.get("cls", "8").split("+")]:
        mk = Megakernel(model, ext, cl=cl, clusters=int(opts["clusters"]) if "clusters" in opts else None,
                        max_envs=100)
        emb, goal = torch.randn(192, device="cuda"), torch.randn(192, device="cuda")
        r = {"clusters": mk.clusters}
        for S in (60, 300):
            cand = torch.randn(S, 5, 10, device="cuda")
            r[f"cost_{S}_ms"] = _stats(_time(lambda: mk.costs(emb, goal, cand), 10, 2))["p50_ms"]
            if opts.get("prof", "0") == "1":
                mk.enable_profiling(True)
                mk.costs(emb, goal, cand)
                torch.cuda.synchronize()
                r[f"cost_{S}_prof"] = mk.profile_summary()
                mk.enable_profiling(False)
        r["solve_ms"] = _stats(_time(lambda: mk.solve(emb, goal), 10, 2))["p50_ms"]
        if opts.get("prof", "0") == "1":
            mk.enable_profiling(True)
            mk.solve(emb, goal)
            torch.cuda.synchronize()
            r["solve_prof"] = mk.profile_summary()
            mk.enable_profiling(False)
        for E in [int(e) for e in opts.get("envs", "1+26+100").split("+")]:
            eb, gb = torch.randn(E, 192, device="cuda"), torch.randn(E, 192, device="cuda")
            r[f"solve_batch_{E}_ms"] = _stats(_time(lambda: mk.solve_batch(eb, gb), 5, 1))["p50_ms"]
        res[f"cl{cl}"] = r
        print(json.dumps(r, indent=1), flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "profile.json").write_text(json.dumps(res, indent=1))
    return res
