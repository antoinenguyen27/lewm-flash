"""Check Modal LeWM runtime imports and config instantiation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.metrics import json_safe
from tools.benchmarks.stage_lewm_weights import build_model_from_config, load_model_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-repo-id", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-file", default="pusht_expert_train.h5.zst")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import datasets
    import h5py
    import hdf5plugin  # noqa: F401
    import hydra
    import pyarrow
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    import transformers
    from datasets import config as datasets_config
    from huggingface_hub import HfApi, hf_hub_download

    if int(pyarrow.__version__.split(".", 1)[0]) >= 21:
        raise RuntimeError(f"Expected pyarrow<21, got {pyarrow.__version__}")
    if int(transformers.__version__.split(".", 1)[0]) >= 5:
        raise RuntimeError(f"Expected transformers<5, got {transformers.__version__}")
    if not hasattr(swm.data, "HDF5Dataset"):
        raise RuntimeError(
            "stable_worldmodel.data.HDF5Dataset is unavailable; install HDF5 format dependencies"
        )

    required_executables = ("git", "swig", "zstd")
    executables = {name: shutil.which(name) for name in required_executables}
    missing_executables = [name for name, path in executables.items() if path is None]
    if missing_executables:
        raise RuntimeError(f"Missing required executables: {missing_executables}")

    with tempfile.TemporaryDirectory() as tmp:
        config_path = hf_hub_download(
            repo_id=args.repo_id,
            filename="config.json",
            local_dir=tmp,
        )
        cfg = load_model_config(Path(config_path).parent)
        model = build_model_from_config(cfg)

    dataset_files = HfApi().list_repo_files(args.dataset_repo_id, repo_type="dataset")
    if args.dataset_file not in dataset_files:
        raise RuntimeError(
            f"Dataset file {args.dataset_file!r} not found in {args.dataset_repo_id}"
        )

    payload: dict[str, Any] = {
        "ok": True,
        "dataset_file": args.dataset_file,
        "dataset_repo_id": args.dataset_repo_id,
        "executables": executables,
        "torch": torch.__version__,
        "datasets": datasets.__version__,
        "datasets_config_module": datasets_config.__name__,
        "pyarrow": pyarrow.__version__,
        "hydra": hydra.__version__,
        "h5py": h5py.__version__,
        "stable_pretraining": getattr(spt, "__version__", "unknown"),
        "stable_worldmodel": getattr(swm, "__version__", "unknown"),
        "transformers": transformers.__version__,
        "has_hdf5_dataset": hasattr(swm.data, "HDF5Dataset"),
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "state_dict_keys": len(model.state_dict()),
        "has_predictor": hasattr(model, "predictor"),
        "has_action_encoder": hasattr(model, "action_encoder"),
    }
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
