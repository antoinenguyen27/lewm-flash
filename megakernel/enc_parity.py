"""Fused-encoder parity on real Push-T frames (runs in the e2e image: needs the full pickled JEPA and the dataset).

Compares observation/goal embeddings from the megakernel's fused ViT encoder against torch FP32 (the policy's image
transform + JEPA.encode), with torch BF16 autocast as the precision yardstick, then checks that model costs computed
from fused embeddings match costs from FP32 embeddings. Also times encoder-only launches at batch 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _frames(h5_path: str, n: int, seed: int = 0):
    import h5py
    import hdf5plugin  # noqa: F401  (the dataset uses an HDF5 compression filter plugin)
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        found = []
        f.visititems(lambda k, v: found.append(k) if isinstance(v, h5py.Dataset) and v.dtype == np.uint8
                     and v.ndim == 4 and v.shape[-1] == 3 else None)
        if not found:
            raise RuntimeError(f"no uint8 [N,H,W,3] image dataset in {h5_path}")
        ds = f[found[0]]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(ds.shape[0], size=n, replace=False))
        return found[0], np.stack([ds[i] for i in idx])


def run(weights_dir: str, out_dir: str, build_dir: str, args: str = "") -> dict:
    import numpy as np
    import torch
    from torchvision.transforms import v2 as T

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "lewm"))
    from megakernel.parity import _stats, _time
    from megakernel.runtime import Megakernel, load_ext

    opts = dict(a.split("=") for a in args.split(",") if a)
    model = torch.load(Path(weights_dir) / "lewm_object.ckpt", map_location="cuda", weights_only=False).cuda().eval()
    key, frames = _frames("/weights/datasets/pusht_expert_train.h5", int(opts.get("n", 32)))
    res = {"image_key": key, "frames": list(frames.shape)}
    tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), T.Resize(size=224)])
    x = torch.stack([tf(T.functional.to_image(f)) for f in frames]).cuda()  # policy transform (CHW float)

    def ref(bf16: bool):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model.encode({"pixels": x[:, None]})["emb"][:, 0]
        return out.float()

    torch.backends.cuda.matmul.allow_tf32 = False
    e32, e16 = ref(False), ref(True)
    ext = load_ext(build_dir)
    E = len(frames) // 2
    mk = Megakernel(model, ext, cl=8, max_envs=E, encoder=True)
    u8 = torch.from_numpy(frames).cuda()
    emk_obs, emk_goal = mk.encode(u8[:E], u8[E:2 * E], list(range(E)))
    emk = torch.cat([emk_obs.clone(), emk_goal.clone()])

    def cmp(e):
        d = (e - e32[: len(e)])
        return {"rel_l2": (d.norm(dim=1) / e32[: len(e)].norm(dim=1)).max().item(),
                "cos_min": torch.nn.functional.cosine_similarity(e, e32[: len(e)], dim=1).min().item(),
                "max_abs": d.abs().max().item()}

    res["encoder_vs_fp32"] = {"torch_bf16_autocast": cmp(e16[: 2 * E]), "megakernel_fused": cmp(emk)}
    print(json.dumps(res), flush=True)

    # model costs from fused vs FP32 embeddings (same candidates)
    rows = []
    for i in range(min(E, 6)):
        cand = torch.randn(300, 5, 10, device="cuda", generator=torch.Generator(device="cuda").manual_seed(i))
        with torch.inference_mode():
            c32 = model.get_cost_from_embeddings(e32[i].view(1, 1, 192), e32[E + i].view(1, 1, 192),
                                                 cand.view(1, 300, 5, 10)).view(-1)
            cmk = model.get_cost_from_embeddings(emk[i].view(1, 1, 192), emk[E + i].view(1, 1, 192),
                                                 cand.view(1, 300, 5, 10)).view(-1)
        rel = ((cmk - c32).abs() / c32.abs().clamp_min(1e-6))
        rows.append({"rel_max": rel.max().item(), "top30_overlap": len(
            set(torch.topk(cmk, 30, largest=False).indices.tolist()) & set(torch.topk(c32, 30, largest=False).indices.tolist()))})
    res["cost_from_fused_embeddings"] = rows
    print(json.dumps(rows), flush=True)

    # batch-1 latency: torch FP32 encode of (obs, goal) vs fused encoder-only launch (obs+goal, and obs only)
    x2 = x[:2]
    res["torch_fp32_encode_2img_ms"] = _stats(_time(lambda: model.encode({"pixels": x2[:, None]}), 30, 5))
    mk1 = Megakernel(model, ext, cl=8, max_envs=1, encoder=True)
    res["fused_encode_obs+goal_ms"] = _stats(_time(lambda: mk1.encode(u8[:1], u8[1:2], [0]), 30, 5))
    res["fused_encode_obs_ms"] = _stats(_time(lambda: mk1.encode(u8[:1]), 30, 5))
    print(json.dumps({k: v for k, v in res.items() if k.endswith("_ms")}), flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "enc_parity.json").write_text(json.dumps(res, indent=1))
    return res
