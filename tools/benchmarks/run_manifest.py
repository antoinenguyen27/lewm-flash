"""Create and validate benchmark run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.metrics import json_safe


REQUIRED_SPEEDUP_FIELDS = (
    "run_id",
    "variant",
    "semantic_class",
    "base_git_commit",
    "git_diff_sha256",
    "weights_sha256",
)


REQUIRED_ARTIFACT_FIELDS = (
    ("artifacts", "metrics_jsonl"),
)


PACKAGE_VERSION_DISTS = {
    "datasets": "datasets",
    "hydra_core": "hydra-core",
    "modal": "modal",
    "omegaconf": "omegaconf",
    "pyarrow": "pyarrow",
    "stable_pretraining": "stable-pretraining",
    "stable_worldmodel": "stable-worldmodel",
    "torchvision": "torchvision",
    "transformers": "transformers",
}

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

BENCHMARK_RESOURCE_ENV_KEYS = (
    "JEPA_MODAL_GPU_TYPE",
    "JEPA_MODAL_CPU_REQUEST",
    "JEPA_MODAL_CPU_LIMIT",
)


def _parse_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def discover_cgroup_cpu_quota(cgroup_root: Path | str = "/sys/fs/cgroup") -> dict[str, Any]:
    """Best-effort CPU quota snapshot from cgroup v2 or v1 files."""
    root = Path(cgroup_root)
    cpu_max = root / "cpu.max"
    if cpu_max.exists():
        try:
            raw = cpu_max.read_text(encoding="utf-8").strip()
        except OSError as exc:
            return {"source": "cgroup_v2", "error": f"{type(exc).__name__}: {exc}"}
        parts = raw.split()
        result: dict[str, Any] = {"source": "cgroup_v2", "raw": raw}
        if len(parts) >= 2:
            quota, period = parts[0], _parse_int(parts[1])
            result["quota"] = quota
            result["period_us"] = period
            if quota != "max":
                quota_us = _parse_int(quota)
                result["quota_us"] = quota_us
                if quota_us is not None and period:
                    result["quota_cores"] = quota_us / period
            else:
                result["quota_cores"] = None
        return result

    quota_path = root / "cpu" / "cpu.cfs_quota_us"
    period_path = root / "cpu" / "cpu.cfs_period_us"
    if quota_path.exists() and period_path.exists():
        try:
            quota_us = _parse_int(quota_path.read_text(encoding="utf-8").strip())
            period_us = _parse_int(period_path.read_text(encoding="utf-8").strip())
        except OSError as exc:
            return {"source": "cgroup_v1", "error": f"{type(exc).__name__}: {exc}"}
        result = {
            "source": "cgroup_v1",
            "quota_us": quota_us,
            "period_us": period_us,
            "quota_cores": None,
        }
        if quota_us is not None and quota_us > 0 and period_us:
            result["quota_cores"] = quota_us / period_us
        return result

    return {"source": None}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def hash_untracked_files(cwd: Path, files: str) -> str:
    digest = hashlib.sha256()
    for rel in sorted(line for line in files.splitlines() if line.strip()):
        path = cwd / rel
        if not path.is_file():
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            digest.update(f"UNREADABLE:{type(exc).__name__}:{exc}".encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def git_metadata(cwd: Path | str) -> dict[str, str]:
    env_commit = os.environ.get("JEPA_BASE_GIT_COMMIT")
    env_diff = os.environ.get("JEPA_GIT_DIFF_SHA256")
    if env_commit and env_diff:
        return {
            "base_git_commit": env_commit,
            "git_diff_sha256": env_diff,
        }

    root = Path(cwd)
    commit = run_git(["rev-parse", "HEAD"], root)
    status = run_git(["status", "--short"], root)
    diff = run_git(["diff", "--binary", "HEAD"], root)
    untracked = run_git(["ls-files", "--others", "--exclude-standard"], root)
    untracked_hash = hash_untracked_files(root, untracked) if untracked != "unknown" else "unknown"
    return {
        "base_git_commit": commit,
        "git_diff_sha256": sha256_text(status + "\n" + diff + "\n" + untracked_hash),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for key, dist_name in PACKAGE_VERSION_DISTS.items():
        try:
            versions[key] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def discover_runtime_resources() -> dict[str, Any]:
    """Record CPU/thread/GPU resource state for benchmark comparability."""
    resources: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "sched_affinity_count": None,
        "cgroup_cpu": discover_cgroup_cpu_quota(),
        "thread_env": {
            key: os.environ[key]
            for key in THREAD_ENV_KEYS
            if key in os.environ
        },
        "benchmark_resource_env": {
            key: os.environ[key]
            for key in BENCHMARK_RESOURCE_ENV_KEYS
            if key in os.environ
        },
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "cuda_device_count": None,
        "cuda_current_device": None,
        "cuda_device_name": None,
        "cuda_device_uuid": None,
    }
    if hasattr(os, "sched_getaffinity"):
        try:
            resources["sched_affinity_count"] = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except Exception as exc:
            resources["sched_affinity_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import torch

        resources["torch_num_threads"] = torch.get_num_threads()
        resources["torch_num_interop_threads"] = torch.get_num_interop_threads()
        resources["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            resources["cuda_current_device"] = int(device)
            resources["cuda_device_name"] = torch.cuda.get_device_name(device)
            resources["cuda_device_uuid"] = getattr(props, "uuid", None)
    except Exception as exc:
        resources["torch_resource_error"] = f"{type(exc).__name__}: {exc}"
    return resources


def discover_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "cuda_available": False,
        "gpu_name": None,
        "packages": package_versions(),
        "resources": discover_runtime_resources(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def default_run_id(model: str, variant: str, seed: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_seed{seed}" if seed is not None else ""
    safe_model = model.replace("/", "_").replace(" ", "_")
    safe_variant = variant.replace("/", "_").replace(" ", "_")
    return f"{safe_model}_{stamp}_{safe_variant}{suffix}"


def build_manifest(
    *,
    run_id: str,
    model: str,
    variant: str,
    semantic_class: str,
    cwd: Path | str,
    command: list[str],
    weights_sha256: str = "unknown",
    dataset_sha256: str = "unknown",
    code_paths: list[str] | None = None,
    planner: dict[str, Any] | None = None,
    modal: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    metrics_summary: dict[str, Any] | None = None,
    quality_gate_result: str | None = None,
    seed_list: list[int] | None = None,
    num_eval_episodes: int | None = None,
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    git = git_metadata(cwd)
    return json_safe(
        {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "model": model,
            "variant": variant,
            "semantic_class": semantic_class,
            "base_git_commit": git["base_git_commit"],
            "git_diff_sha256": git["git_diff_sha256"],
            "weights_sha256": weights_sha256,
            "dataset_sha256": dataset_sha256,
            "code_paths": code_paths or [],
            "command": command,
            "modal": {
                "enabled": False,
                "gpu": None,
                "cpu_request": None,
                "cpu_limit": None,
                "image_label": None,
                "weights_volume": None,
                "results_volume": None,
                **(modal or {}),
            },
            "planner": planner or {},
            "environment": discover_environment(),
            "artifacts": {
                "metrics_jsonl": None,
                "profiler_trace": None,
                "stdout": None,
                "stderr": None,
                **(artifacts or {}),
            },
            "metrics_summary": metrics_summary or {},
            "quality_gate_result": quality_gate_result,
            "seed_list": seed_list or [],
            "num_eval_episodes": num_eval_episodes,
            "baseline_run_id": baseline_run_id,
        }
    )


def write_manifest(path: Path | str, manifest: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)


def load_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def validate_manifest(
    manifest: dict[str, Any],
    *,
    require_speedup_fields: bool = True,
    require_gpu_environment: bool = True,
    planner_changing: bool = False,
) -> list[str]:
    errors: list[str] = []
    if require_speedup_fields:
        for field in REQUIRED_SPEEDUP_FIELDS:
            if not manifest.get(field) or manifest.get(field) == "unknown":
                errors.append(f"missing or unknown field: {field}")
        for field_path in REQUIRED_ARTIFACT_FIELDS:
            if not nested_get(manifest, field_path):
                errors.append(f"missing artifact field: {'.'.join(field_path)}")
        if require_gpu_environment:
            env = manifest.get("environment", {})
            if not env.get("torch"):
                errors.append("missing environment.torch")
            if not env.get("gpu_name"):
                errors.append("missing environment.gpu_name")
    if planner_changing:
        for field in ("baseline_run_id", "quality_gate_result", "seed_list", "num_eval_episodes"):
            if not manifest.get(field):
                errors.append(f"planner run missing field: {field}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate", help="Validate a manifest JSON file")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--allow-local-missing-gpu", action="store_true")
    validate.add_argument("--planner-changing", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "validate":
        manifest = load_manifest(args.manifest)
        errors = validate_manifest(
            manifest,
            require_speedup_fields=True,
            require_gpu_environment=not args.allow_local_missing_gpu,
            planner_changing=args.planner_changing,
        )
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"valid manifest: {args.manifest}")
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
