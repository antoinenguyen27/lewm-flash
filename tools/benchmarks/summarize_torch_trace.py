"""Summarize PyTorch profiler Chrome trace files for benchmark runs."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.metrics import json_safe


def trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = ("*.pt.trace.json", "*.pt.trace.json.gz", "*.trace.json", "*.json")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(pattern))
    return sorted(set(files))


def load_trace(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def duration_ms(event: dict[str, Any]) -> float | None:
    dur = event.get("dur")
    if isinstance(dur, (int, float)):
        return float(dur) / 1000.0
    return None


def update_summary(bucket: dict[str, Any], value_ms: float) -> None:
    bucket["count"] = bucket.get("count", 0) + 1
    bucket["total_ms"] = bucket.get("total_ms", 0.0) + value_ms
    bucket["max_ms"] = max(bucket.get("max_ms", 0.0), value_ms)


def finalize_bucket(bucket: dict[str, Any]) -> None:
    count = bucket.get("count", 0)
    if count:
        bucket["mean_ms"] = bucket["total_ms"] / count


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    user_ranges: dict[str, dict[str, Any]] = {}
    cuda_kernel = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
    cuda_memcpy = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
    cuda_runtime = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
    total_events = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        total_events += 1
        name = str(event.get("name", ""))
        cat = str(event.get("cat", "")).lower()
        ms = duration_ms(event)
        if ms is None:
            continue

        if name.startswith(("cem.", "lewm.")):
            update_summary(user_ranges.setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0}), ms)
        if "kernel" in cat:
            update_summary(cuda_kernel, ms)
        elif "memcpy" in cat or "memcpy" in name.lower() or "memset" in name.lower():
            update_summary(cuda_memcpy, ms)
        elif "cuda" in cat or name.startswith(("cuda", "cu")):
            update_summary(cuda_runtime, ms)

    for bucket in user_ranges.values():
        finalize_bucket(bucket)
    for bucket in (cuda_kernel, cuda_memcpy, cuda_runtime):
        finalize_bucket(bucket)

    return {
        "trace_event_count": total_events,
        "user_ranges": dict(sorted(user_ranges.items())),
        "cuda_kernel": cuda_kernel,
        "cuda_memcpy": cuda_memcpy,
        "cuda_runtime": cuda_runtime,
    }


def summarize_trace_path(path: Path | str) -> dict[str, Any]:
    root = Path(path)
    files = trace_files(root)
    aggregate_events: list[dict[str, Any]] = []
    for trace in files:
        payload = load_trace(trace)
        events = payload.get("traceEvents", [])
        if isinstance(events, list):
            aggregate_events.extend(event for event in events if isinstance(event, dict))
    summary = summarize_events(aggregate_events)
    summary["trace_files"] = [str(path) for path in files]
    return json_safe(summary)


def write_summary(path: Path | str, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    summary = summarize_trace_path(args.trace_dir)
    write_summary(args.out, summary)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
