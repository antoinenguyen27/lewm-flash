"""Load the LeWM Push-T planning-cost modules (predictor, action encoder, pred_proj) without the ViT encoder.

The megakernel only replaces the repeated model-cost rollout, which never touches the image encoder: observation and
goal embeddings are computed once per plan and cached (LeWM-Flash change L1). Building just these three modules from
``weights.pt`` keeps the kernel harness free of stable-pretraining / transformers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEWM_DIR = REPO_ROOT / "lewm"


def _kwargs(section: dict) -> dict:
    return {k: v for k, v in section.items() if k not in {"_target_", "_partial_"}}


def load_cost_model(weights_dir: str | Path, device: str = "cuda"):
    """Return an eval-mode ``JEPA`` whose encoder/projector are identities (unused by the cost path)."""
    import torch
    from torch import nn

    if str(LEWM_DIR) not in sys.path:
        sys.path.insert(0, str(LEWM_DIR))
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder

    weights_dir = Path(weights_dir)
    raw = json.loads((weights_dir / "config.json").read_text())
    cfg = raw if "predictor" in raw else raw["model"]
    pp = _kwargs(cfg["pred_proj"])
    norm = pp.get("norm_fn")
    norm_target = norm.get("_target_") if isinstance(norm, dict) else norm
    norm_fn = {"torch.nn.BatchNorm1d": nn.BatchNorm1d, "torch.nn.LayerNorm": nn.LayerNorm}[norm_target]
    model = JEPA(
        encoder=nn.Identity(),
        predictor=ARPredictor(**_kwargs(cfg["predictor"])),
        action_encoder=Embedder(**_kwargs(cfg["action_encoder"])),
        projector=nn.Identity(),
        pred_proj=MLP(input_dim=pp["input_dim"], output_dim=pp.get("output_dim"), hidden_dim=pp["hidden_dim"],
                      norm_fn=norm_fn),
    )
    state = torch.load(weights_dir / "weights.pt", map_location="cpu", weights_only=False)
    keep = {k: v for k, v in state.items() if k.split(".")[0] in {"predictor", "action_encoder", "pred_proj"}}
    missing, unexpected = model.load_state_dict(keep, strict=False)
    missing = [k for k in missing if k.split(".")[0] in {"predictor", "action_encoder", "pred_proj"}]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    return model.to(device).eval(), cfg
