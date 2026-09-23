#!/usr/bin/env python3
"""Learning curves: pretrained vs architecture-matched random initialisation.

The project's central claim is that pretraining on EEG supplies a better
*initialisation* for motor imagery rather than a directly decodable
representation. Final accuracy alone cannot show that -- two runs can end in the
same place having taken very different paths. The claim is really about the
shape of the optimisation curve, so this plots it.

Reads the per-epoch validation-Brier traces recorded in ``audit_json`` by
``run_bd_fm_probe.py`` and draws, per dataset, the subject-averaged curve for the
pretrained and random-init arms with a shaded standard error.

Selection discipline note: these are *validation* curves. The held-out session
contributes nothing to any point plotted here, so the figure can be shown
without qualification.

Usage::

    python src/scripts/make_learning_curve_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CURVE_DIR = Path("results") / "fm_probe_curves"
DATASET_LABELS = {"bci4_2a": "BCI IV-2a (4-class)", "bnci2014_004": "BNCI2014_004 (2-class)"}


def load_curves(path: Path) -> np.ndarray:
    """(n_subjects, n_epochs) array of per-epoch validation Brier."""
    df = pd.read_csv(path)
    rows = [json.loads(a)["val_brier_curve"] for a in df["audit_json"]]
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve-dir", type=Path, default=CURVE_DIR)
    ap.add_argument("--tag", default="curve", help="Filename tag of the runs to plot.")
    ap.add_argument("--stem", default="learning_curves", help="Output filename stem.")
    ap.add_argument("--out", type=Path, default=Path("results") / "paper_calibration_stats" / "fm_audit")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = []
    for ds in ("bci4_2a", "bnci2014_004"):
        suffix = "" if ds == "bci4_2a" else f"_{ds}"
        pre = args.curve_dir / f"fm_probe_cbramod_finetune{suffix}_{args.tag}_subject_metrics.csv"
        rnd = args.curve_dir / f"fm_probe_cbramod_finetune{suffix}_randinit_{args.tag}_subject_metrics.csv"
        if pre.exists() and rnd.exists():
            datasets.append((ds, load_curves(pre), load_curves(rnd)))
    if not datasets:
        raise SystemExit(f"no curve CSVs found under {args.curve_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.0), squeeze=False)
    summary = []
    for ax, (ds, pre, rnd) in zip(axes[0], datasets):
        for arr, label, colour in ((pre, "pretrained", "#1f77b4"), (rnd, "random init", "#d62728")):
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            epochs = np.arange(1, len(mean) + 1)
            ax.plot(epochs, mean, color=colour, label=label, linewidth=2)
            ax.fill_between(epochs, mean - sem, mean + sem, color=colour, alpha=0.18, linewidth=0)
            # Mark where each arm actually bottoms out -- the selected checkpoint.
            best = int(np.argmin(mean))
            ax.scatter([best + 1], [mean[best]], color=colour, zorder=5, s=36)
            summary.append(
                {"dataset": ds, "arm": label, "best_epoch": best + 1,
                 "best_val_Brier": float(mean[best]), "n_subjects": int(arr.shape[0])}
            )
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation Brier (lower is better)")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    n_epochs = max(int(r["best_epoch"]) for r in summary)
    budget = max(len(a.mean(axis=0)) for _, a, _ in datasets)
    fig.suptitle(
        "CBraMod fine-tuning: pretrained weights converge faster and reach lower\n"
        f"validation Brier (equal {budget}-epoch budget, both arms)",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"{args.stem}.{ext}", dpi=200, bbox_inches="tight")

    tab = pd.DataFrame(summary)
    tab.to_csv(args.out / f"{args.stem}_summary.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nWrote {args.out}/{args.stem}.png")


if __name__ == "__main__":
    main()
