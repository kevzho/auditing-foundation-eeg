"""Download and normalize BCI Competition IV 2a evaluation labels."""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from config import DATA_DIR
from preprocess import _normalize_labels

LABELS_URL = "https://bbci.de/competition/iv/results/ds2a/true_labels.zip"

def _subject_from_name(name: str) -> str | None:
    match = re.search(r"A0?([1-9])E\b", name, flags=re.IGNORECASE)
    if not match:
        return None
    return f"A{int(match.group(1)):02d}E"

def _load_label_member(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return _normalize_labels(np.load(path))
    if path.suffix in {".txt", ".csv"}:
        return _normalize_labels(np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None))
    if path.suffix == ".mat":
        from scipy.io import loadmat

        mat = loadmat(path)
        candidates = []
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value).reshape(-1)
            if arr.size == 288:
                candidates.append(arr)
        if not candidates:
            candidates = [
                np.asarray(value).reshape(-1)
                for key, value in mat.items()
                if not key.startswith("__") and np.asarray(value).size > 0
            ]
        if not candidates:
            raise ValueError(f"No label arrays found in {path}")
        candidates.sort(key=lambda arr: -arr.size)
        return _normalize_labels(candidates[0])
    raise ValueError(f"Unsupported label file type: {path}")

def download_and_extract(labels_dir: Path, url: str = LABELS_URL) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    zip_path = labels_dir / "true_labels.zip"
    tmp_dir = labels_dir / "_true_labels_raw"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[labels] downloading {url}")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    written: set[str] = set()
    for member in sorted(p for p in tmp_dir.rglob("*") if p.is_file()):
        subject = _subject_from_name(member.name)
        if subject is None:
            continue
        labels = _load_label_member(member)
        out = labels_dir / f"{subject}.csv"
        np.savetxt(out, labels, fmt="%d", delimiter=",")
        written.add(out.name)

    if len(written) != 9:
        raise RuntimeError(f"Expected 9 subject label files, wrote {len(written)}: {sorted(written)}")
    print("[labels] wrote " + ", ".join(sorted(written)))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Download BCI IV 2a true labels for A0xE sessions.")
    parser.add_argument("--labels-dir", type=Path, default=DATA_DIR / "true_labels")
    parser.add_argument("--url", default=LABELS_URL)
    args = parser.parse_args()
    download_and_extract(labels_dir=args.labels_dir, url=args.url)

if __name__ == "__main__":
    main()