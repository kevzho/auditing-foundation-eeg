#!/usr/bin/env python3
"""Build a reproducibility manifest for the current artifact set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PACKAGE_DISTS = {
    "mne": "mne",
    "moabb": "moabb",
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
}


DEFAULT_COMMANDS = [
    (
        "python src/preprocess.py --data-dir data/BCICIV_2a_gdf "
        "--out-dir results/preprocessed_mne_check "
        "--labels-dir data/BCICIV_2a_gdf "
        "--npz-dir results/preprocessed_mne_check/npz"
    ),
    "bash reproducibility/00_smoke_test.sh",
    "bash reproducibility/01_readout_from_saved_results.sh",
    "bash reproducibility/02_full_validation_locked_runs.sh",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, dist in PACKAGE_DISTS.items():
        try:
            versions[name] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_info(root: Path) -> dict[str, object]:
    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return completed.stdout.strip()

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--raw-dir", type=Path, default=Path("data") / "BCICIV_2a_gdf")
    parser.add_argument("--npz-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results") / "reproducibility_manifest.json")
    parser.add_argument(
        "--command",
        action="append",
        dest="commands",
        help="Command used in the reproduction path. May be supplied multiple times.",
    )
    return parser.parse_args()


def resolve_from(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    raw_dir = resolve_from(root, args.raw_dir).resolve()
    npz_dir = resolve_from(root, args.npz_dir).resolve()
    results_dir = resolve_from(root, args.results_dir).resolve()
    out = resolve_from(root, args.out).resolve()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "git": git_info(root),
        "commands": args.commands or DEFAULT_COMMANDS,
        "inputs": {
            "raw_gdf": hash_files(raw_dir.glob("*.gdf"), root),
            "bci4_2a_npz": hash_files(npz_dir.glob("bci4_2a_subject*_*.npz"), root),
        },
        "results": {
            "locked_csv": hash_files(results_dir.glob("**/*locked*.csv"), root),
            "subject_metric_csv": hash_files(results_dir.glob("**/*subject_metrics*.csv"), root),
            "summary_csv": hash_files(results_dir.glob("**/*summary*.csv"), root),
        },
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
