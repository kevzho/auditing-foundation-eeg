"""Verify the frozen-probe reruns, then answer the AUROC question they exist for.

Two jobs, in this order, because the second is worthless if the first fails.

1. **Determinism check.** The frozen arms in ``results/fm_probe`` recorded
   ``deterministic=true`` on cpu. ``run_frozen_probs.sh`` re-ran them with the
   flags transcribed from their own run JSON, so every held-out accuracy and
   Brier must reproduce exactly. This has never actually been tested. A
   mismatch is a finding about the reproducibility claim, not a rounding
   nuisance, so it is reported before anything else and it blocks the rest.

2. **AUROC.** The manuscript title says pretraining is "not a representation",
   which rests on frozen-probe accuracy sitting near chance. Accuracy is
   threshold-dependent and conflates discrimination with a bad operating
   point: the fine-tuned random-init control on BCI IV-2a already sits at
   accuracy 0.335 (chance 0.25) with AUROC 0.599 (chance 0.5). If the frozen
   probes rank trials above chance, "not a representation" overstates what the
   data shows and needs narrowing.

AUROC is computed as macro one-vs-rest for the four-class dataset and directly
for the two-class one. It is deliberately NOT added to ``METRICS`` in
make_fm_audit_report.py: that list is corrected as a single Benjamini-Hochberg
family, so a fifth metric would inflate the family by a quarter and degrade
every p_bh already reported. This is a declared robustness check, run once,
reported whichever way it comes out.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path("/Volumes/kz_extended/CalibMI")
ORIGINAL = ROOT / "results" / "fm_probe"
RERUN = ROOT / "results" / "fm_probe_frozen_probs"
KEYS = ["dataset", "subject", "experiment_key"]
COMPARE = ["heldout_accuracy", "heldout_Brier"]
# Exact reproduction is the claim under test. This tolerance exists only to
# absorb float formatting in the CSV round-trip, not genuine numeric drift.
TOL = 1e-9


def load(directory: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(directory.glob("*_subject_metrics.csv"))]
    if not frames:
        raise SystemExit(f"no subject metrics under {directory}")
    df = pd.concat(frames, ignore_index=True)
    return df[df["experiment_key"].str.contains("frozen")]


def check_determinism() -> bool:
    old, new = load(ORIGINAL), load(RERUN)
    merged = old.merge(new, on=KEYS, suffixes=("_old", "_new"))
    missing = len(new) - len(merged)
    print(f"matched {len(merged)} subject-arm rows"
          f"{f' ({missing} rerun rows had no original)' if missing else ''}")

    clean = True
    for col in COMPARE:
        delta = (merged[f"{col}_new"] - merged[f"{col}_old"]).abs()
        worst = delta.max()
        status = "reproduces" if worst <= TOL else f"DIVERGES (max {worst:.6f})"
        print(f"  {col:<18} {status}")
        if worst > TOL:
            clean = False
            bad = merged.loc[delta > TOL, KEYS + [f"{col}_old", f"{col}_new"]]
            print(bad.to_string(index=False))
    return clean


def auroc_table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(RERUN / "*_probabilities.npz"))):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["metadata_json"]))
        subject = int(re.search(r"_s(\d+)_probabilities\.npz$", path).group(1))
        y = z["y_eval"]
        for field in (f for f in z.files if f.endswith("__eval_proba")):
            proba = z[field]
            rows.append({
                "dataset": meta["dataset"],
                "subject": subject,
                "experiment_key": field.removesuffix("__eval_proba"),
                "AUROC": (roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2
                          else roc_auc_score(y, proba, multi_class="ovr", average="macro")),
                "accuracy": float((proba.argmax(1) == y).mean()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("== 1. determinism: do the reruns reproduce the reported numbers? ==")
    if not check_determinism():
        raise SystemExit(
            "\nRefusing to report AUROC. The reruns did not reproduce the "
            "published frozen-probe numbers, which makes them a different "
            "experiment rather than the same one with probabilities attached. "
            "Diagnose the divergence first."
        )

    print("\n== 2. AUROC on the frozen probes ==")
    auc = auroc_table()
    if auc.empty:
        raise SystemExit("no probability files found; is the rerun still going?")

    summary = (auc.groupby(["dataset", "experiment_key"])[["AUROC", "accuracy"]]
                  .agg(["mean", "count"]))
    summary.columns = ["AUROC", "n", "accuracy", "_n"]
    print(summary[["n", "AUROC", "accuracy"]].round(4).to_string())

    print("\n== 3. pretrained vs random init, paired within subject ==")
    print("Chance AUROC is 0.5 for both datasets regardless of class count.")
    for (dataset, key), grp in auc.groupby(["dataset", "experiment_key"]):
        if "randinit" in key:
            continue
        control = auc[(auc["dataset"] == dataset)
                      & (auc["experiment_key"] == key.replace("_p", "_randinit_p"))]
        if control.empty:
            print(f"  {dataset} {key}: no matched control on disk")
            continue
        merged = grp.merge(control, on="subject", suffixes=("_pre", "_rand"))
        delta = merged["AUROC_pre"] - merged["AUROC_rand"]
        p = wilcoxon(merged["AUROC_pre"], merged["AUROC_rand"]).pvalue
        print(f"  {dataset:<14} {key:<28} n={len(merged)} "
              f"pre={merged['AUROC_pre'].mean():.4f} rand={merged['AUROC_rand'].mean():.4f} "
              f"delta={delta.mean():+.4f} wins={int((delta > 0).sum())}/{len(merged)} p={p:.4f}")

    out = RERUN / "frozen_auroc.csv"
    auc.sort_values(["dataset", "experiment_key", "subject"]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
