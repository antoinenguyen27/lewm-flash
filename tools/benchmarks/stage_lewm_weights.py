"""Stage and convert LeWM Hugging Face weights for stable-worldmodel eval."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEWM_DIR = REPO_ROOT / "lewm"


def load_model_config(src_dir: Path) -> dict:
    import json

    raw = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    if all(key in raw for key in ("encoder", "predictor", "action_encoder")):
        return raw
    if isinstance(raw.get("model"), dict) and all(
        key in raw["model"] for key in ("encoder", "predictor", "action_encoder")
    ):
        return raw["model"]
    raise ValueError(f"Unsupported LeWM config shape in {src_dir / 'config.json'}")


def hydra_kwargs(section: dict) -> dict:
    return {
        key: value
        for key, value in section.items()
        if key not in {"_target_", "_partial_"}
    }


def norm_target_name(norm_cfg: object) -> str | None:
    if norm_cfg is None:
        return None
    if isinstance(norm_cfg, str):
        return norm_cfg
    if isinstance(norm_cfg, dict):
        target = norm_cfg.get("_target_")
        return str(target) if target else None
    raise TypeError(f"Unsupported norm_fn config: {norm_cfg!r}")


def resolve_norm_fn(norm_cfg: object, torch_module):
    target = norm_target_name(norm_cfg)
    if target is None:
        return None
    mapping = {
        "torch.nn.BatchNorm1d": torch_module.nn.BatchNorm1d,
        "torch.nn.LayerNorm": torch_module.nn.LayerNorm,
    }
    if target not in mapping:
        raise ValueError(f"Unsupported norm_fn target: {target}")
    return mapping[target]


def build_model_from_config(cfg: dict):
    sys.path.insert(0, str(LEWM_DIR))

    import torch
    import stable_pretraining as spt

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    encoder_cfg = hydra_kwargs(cfg["encoder"])
    encoder_size = encoder_cfg.pop("size")
    encoder = spt.backbone.utils.vit_hf(
        encoder_size,
        patch_size=encoder_cfg.pop("patch_size"),
        image_size=encoder_cfg.pop("image_size"),
        pretrained=encoder_cfg.pop("pretrained", False),
        use_mask_token=encoder_cfg.pop("use_mask_token", False),
    )
    if encoder_cfg:
        raise ValueError(f"Unsupported encoder config keys: {sorted(encoder_cfg)}")

    def mlp(section: str) -> MLP:
        kwargs = hydra_kwargs(cfg[section])
        kwargs["norm_fn"] = resolve_norm_fn(kwargs.get("norm_fn"), torch)
        return MLP(
            input_dim=kwargs["input_dim"],
            output_dim=kwargs.get("output_dim"),
            hidden_dim=kwargs["hidden_dim"],
            norm_fn=kwargs.get("norm_fn"),
        )

    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**hydra_kwargs(cfg["predictor"])),
        action_encoder=Embedder(**hydra_kwargs(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )


def convert_checkpoint(src_dir: Path, object_out: Path) -> None:
    import torch

    cfg = load_model_config(src_dir)
    model = build_model_from_config(cfg)
    state_dict = torch.load(src_dir / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    object_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, object_out)


def dataset_paths(weights_root: Path, dataset_file: str) -> tuple[Path, Path]:
    dataset_name = dataset_file.removesuffix(".zst")
    canonical = weights_root / "datasets" / dataset_name
    legacy = weights_root / dataset_name
    return canonical, legacy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-repo-id", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-file", default="pusht_expert_train.h5.zst")
    parser.add_argument("--weights-root", default="/weights")
    parser.add_argument("--task", default="pusht")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    weights_root = Path(args.weights_root)
    raw_dir = weights_root / "lewm" / args.task
    object_out = raw_dir / "lewm_object.ckpt"
    stablewm_object_out = weights_root / args.task / "lewm_object.ckpt"
    stablewm_state_dir = weights_root / "checkpoints" / args.task / "lewm"
    compressed_dataset = raw_dir / args.dataset_file
    dataset_out, legacy_dataset_out = dataset_paths(weights_root, args.dataset_file)

    if args.force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    if args.force and not args.skip_dataset and dataset_out.exists():
        dataset_out.unlink()
    if args.force and not args.skip_dataset and legacy_dataset_out.exists():
        legacy_dataset_out.unlink()

    raw_dir.mkdir(parents=True, exist_ok=True)
    if not (raw_dir / "weights.pt").exists() or not (raw_dir / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=args.repo_id, local_dir=raw_dir)

    if args.force or not object_out.exists():
        convert_checkpoint(raw_dir, object_out)

    if not args.skip_dataset:
        dataset_out.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_dataset and not dataset_out.exists():
        if legacy_dataset_out.exists():
            legacy_dataset_out.replace(dataset_out)
        else:
            from huggingface_hub import hf_hub_download

            if not compressed_dataset.exists():
                hf_hub_download(
                    repo_id=args.dataset_repo_id,
                    repo_type="dataset",
                    filename=args.dataset_file,
                    local_dir=raw_dir,
                )
            subprocess.run(
                [
                    "zstd",
                    "-d",
                    "-f",
                    "-o",
                    str(dataset_out),
                    str(compressed_dataset),
                ],
                check=True,
            )
    if not args.skip_dataset and not compressed_dataset.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=args.dataset_repo_id,
            repo_type="dataset",
            filename=args.dataset_file,
            local_dir=raw_dir,
        )

    stablewm_object_out.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not stablewm_object_out.exists():
        shutil.copy2(object_out, stablewm_object_out)
    stablewm_state_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("weights.pt", "config.json"):
        src = raw_dir / filename
        dest = stablewm_state_dir / filename
        if args.force or not dest.exists():
            shutil.copy2(src, dest)

    print(f"staged raw weights: {raw_dir}")
    print(f"staged object checkpoint: {object_out}")
    print(f"stable-worldmodel policy path: {stablewm_object_out}")
    print(f"stable-worldmodel state-dict path: {stablewm_state_dir}")
    if not args.skip_dataset:
        print(f"stable-worldmodel dataset path: {dataset_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
