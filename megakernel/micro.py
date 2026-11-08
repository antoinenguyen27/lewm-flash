"""Micro-benchmarks of the megakernel's primitives (cluster sync, grid sync, GEMM epilogues)."""

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

    ext = load_ext(build_dir)
    model, _ = load_cost_model(weights_dir)
    res = {}
    for cl in (8, 16):
        mk = Megakernel(model, ext, cl=cl)
        r = {"clusters": mk.clusters}
        for name, mode in (("cluster_sync", 2), ("grid_sync", 3), ("epilogue_gelu_64x64", 4), ("epilogue_bias_64x64", 5)):
            n = 2000
            base = _stats(_time(lambda: mk.micro(mode, 1), 10, 2))["p50_ms"]
            tot = _stats(_time(lambda: mk.micro(mode, n), 10, 2))["p50_ms"]
            r[name + "_us"] = (tot - base) * 1000.0 / (n - 1)
        r["empty_launch_ms"] = base
        res[f"cl{cl}"] = r
        print(json.dumps(r), flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "micro.json").write_text(json.dumps(res, indent=1))
    return res
