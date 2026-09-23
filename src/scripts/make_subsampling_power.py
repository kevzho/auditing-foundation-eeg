"""Detection rate versus cohort size, by resampling the 54-subject cohort.

The audit's small-cohort arms (n=9) and its large-cohort arm (n=54) differ in
dataset, task, class count and electrode count at the same time, so comparing
them conflates cohort size with everything else. This script removes that
confound by drawing sub-cohorts from Lee2019 itself: same subjects, same
recordings, same models, same protocol, only n changes.

For each pretrained-versus-random-initialisation arm it reports, at every
cohort size, the fraction of random sub-cohorts in which the paired test
reaches alpha. It also reports the attainable p-floor at that size, which is a
property of the design rather than of the data: below a certain n no sub-cohort
can reach alpha regardless of how large the effect is.

Outputs results/paper_calibration_stats/fm_audit/subsampling_power.{csv,md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FM_DIR = Path("results") / "fm_probe"
METRIC = "heldout_accuracy"
ALPHA = 0.05

# (label, pretrained stem, random-init stem)
ARMS = [
    ("CBraMod frozen probe", "fm_probe_cbramod_frozen_lee2019_mi",
     "fm_probe_cbramod_frozen_lee2019_mi_randinit"),
    ("CBraMod fine-tuned", "fm_probe_cbramod_finetune_lee2019_mi_sel",
     "fm_probe_cbramod_finetune_lee2019_mi_randinit_sel"),
    ("CBraMod pooled LOSO", "fm_loso_cbramod_lee2019_mi",
     "fm_loso_cbramod_lee2019_mi_randinit"),
    ("LaBraM frozen probe", "fm_probe_frozen_lee2019_mi",
     "fm_probe_frozen_lee2019_mi_randinit"),
    ("LaBraM fine-tuned", "fm_probe_finetune_lee2019_mi_sel",
     "fm_probe_finetune_lee2019_mi_randinit_sel"),
]


def load_arm(stem: str) -> pd.Series:
    path = FM_DIR / f"{stem}_subject_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    leaked = df.get("eval_labels_used_for_selection")
    if leaked is None or bool(leaked.fillna(True).any()):
        raise ValueError(f"{path} is missing the leakage field or has it set")
    return df.set_index("subject")[METRIC].sort_index()


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)


def attainable_p_floor(n_nonzero: int) -> float:
    """Smallest two-sided exact Wilcoxon p reachable with n non-zero pairs."""
    if n_nonzero < 1:
        return 1.0
    return min(1.0, 2.0 ** (1 - n_nonzero))


def sweep(deltas: np.ndarray, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total = deltas.size
    rows = []
    for n in range(3, total + 1):
        if n == total:
            ps = np.array([wilcoxon_p(deltas)])
        else:
            idx = np.array([rng.choice(total, size=n, replace=False) for _ in range(draws)])
            ps = np.array([wilcoxon_p(deltas[i]) for i in idx])
        rows.append({
            "n": n,
            "draws": int(ps.size),
            "detection_rate": float(np.mean(ps < ALPHA)),
            "median_p": float(np.median(ps)),
            "attainable_p_floor": attainable_p_floor(n),
            "floor_permits_alpha": bool(attainable_p_floor(n) < ALPHA),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("results") / "paper_calibration_stats" / "fm_audit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frames, summary = [], []
    for label, pre_stem, rand_stem in ARMS:
        try:
            pre, rand = load_arm(pre_stem), load_arm(rand_stem)
        except FileNotFoundError as exc:
            print(f"skip {label}: missing {exc}")
            continue
        shared = pre.index.intersection(rand.index)
        deltas = (pre.loc[shared] - rand.loc[shared]).to_numpy(dtype=float)
        df = sweep(deltas, args.draws, args.seed)
        df.insert(0, "arm", label)
        frames.append(df)

        full = df[df.n == deltas.size].iloc[0]
        at9 = df[df.n == 9]
        summary.append({
            "arm": label,
            "n_total": int(deltas.size),
            "mean_delta": float(deltas.mean()),
            "wins": f"{int((deltas > 0).sum())}/{deltas.size}",
            "p_full_cohort": float(wilcoxon_p(deltas)),
            "detection_rate_at_9": float(at9.detection_rate.iloc[0]) if len(at9) else float("nan"),
        })

    if not frames:
        print("no arms found; nothing written")
        return

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out / "subsampling_power.csv", index=False)
    summ = pd.DataFrame(summary)

    smallest_ok = int(out[out.floor_permits_alpha].n.min())
    lines = [
        "# Detection rate versus cohort size",
        "",
        f"Generated by `src/scripts/make_subsampling_power.py` "
        f"({args.draws} draws per size, seed {args.seed}, metric `{METRIC}`, "
        f"alpha {ALPHA}). Do not edit by hand; rerun the script.",
        "",
        "Sub-cohorts are drawn from the 54-subject Lee2019 cohort, so dataset, "
        "task, class count, electrode count, model and protocol are all held "
        "fixed and only cohort size varies.",
        "",
        f"The attainable p-floor is `2^(1-n)`. It first falls below alpha={ALPHA} "
        f"at **n={smallest_ok}**: at any smaller cohort size no sub-cohort can "
        "reach significance regardless of effect size. This is a property of the "
        "design, not of these results.",
        "",
        "## Full-cohort effects and the detection rate a 9-subject cohort would see",
        "",
        summ.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Detection rate by cohort size",
        "",
    ]
    for arm, g in out.groupby("arm", sort=False):
        shown = g[g.n.isin([5, 6, 9, 12, 15, 20, 30, 40, int(g.n.max())])]
        lines += [f"### {arm}", "",
                  shown[["n", "detection_rate", "median_p", "attainable_p_floor"]]
                  .to_markdown(index=False, floatfmt=".4f"), ""]
    (args.out / "subsampling_power.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out / 'subsampling_power.csv'}")
    print(f"Wrote {args.out / 'subsampling_power.md'}")


if __name__ == "__main__":
    main()
