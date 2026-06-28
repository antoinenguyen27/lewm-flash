"""Hash benchmark artifacts in a stable, manifest-friendly format."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def sha256_directory(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_size = 0
    file_count = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = child.relative_to(path).as_posix()
        child_hash, child_size = sha256_file(child)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(child_hash.encode("ascii"))
        digest.update(b"\0")
        total_size += child_size
        file_count += 1
    return digest.hexdigest(), total_size, file_count


def hash_path(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.is_dir():
        sha, size, file_count = sha256_directory(p)
        return {
            "path": str(p),
            "type": "directory",
            "sha256": sha,
            "size_bytes": size,
            "file_count": file_count,
        }
    if p.is_file():
        sha, size = sha256_file(p)
        return {
            "path": str(p),
            "type": "file",
            "sha256": sha,
            "size_bytes": size,
        }
    raise ValueError(f"Unsupported artifact path: {p}")


def hash_many(paths: list[str]) -> dict[str, Any]:
    artifacts = [hash_path(path) for path in paths]
    return {
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", required=True, help="Files or directories to hash")
    parser.add_argument("--out", required=True, help="JSON output path")
    args = parser.parse_args(argv)

    payload = hash_many(args.paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
