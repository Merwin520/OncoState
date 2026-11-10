"""Reproducibility metadata: environment, hashes, git commit."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "streamlit", "jsonschema", "PyYAML", "Pillow")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    return sha256_bytes(p.read_bytes()) if p.is_file() else None


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


def git_commit(root: str | Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment() -> dict:
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions}


def run_manifest(root: str | Path, seed: int, extra: dict | None = None) -> dict:
    return {"created": now_iso(), "seed": seed, "git_commit": git_commit(root), "environment": environment(), **(extra or {})}
