#!/usr/bin/env python3
"""Check saved result files for held-out-label leakage flags."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FLAG = "eval_labels_used_for_selection"
TRUE_VALUES = {"1", "true", "yes", "y"}


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in TRUE_VALUES


def find_json_flags(value: Any, path: Path, json_path: str = "$") -> list[tuple[Path, str, Any]]:
    hits: list[tuple[Path, str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{json_path}.{key}"
            if key == FLAG:
                hits.append((path, child_path, child))
            hits.extend(find_json_flags(child, path, child_path))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            hits.extend(find_json_flags(child, path, f"{json_path}[{i}]"))
    return hits


def audit_csv(path: Path) -> list[tuple[Path, str, Any]]:
    hits: list[tuple[Path, str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or FLAG not in reader.fieldnames:
            return hits
        for i, row in enumerate(reader, start=2):
            hits.append((path, f"row {i}", row.get(FLAG)))
    return hits


def audit_json(path: Path) -> list[tuple[Path, str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return find_json_flags(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--allow-true", action="store_true", help="Report true flags without failing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hits: list[tuple[Path, str, Any]] = []
    for path in sorted(args.results_dir.glob("**/*")):
        if path.suffix == ".csv":
            hits.extend(audit_csv(path))
        elif path.suffix == ".json":
            hits.extend(audit_json(path))

    true_hits = [(path, location, value) for path, location, value in hits if is_true(value)]
    print(f"Found {len(hits)} {FLAG} fields under {args.results_dir}.")
    print(f"Fields set true: {len(true_hits)}")
    for path, location, value in true_hits:
        print(f"TRUE {path}:{location} -> {value!r}")

    if true_hits and not args.allow_true:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
