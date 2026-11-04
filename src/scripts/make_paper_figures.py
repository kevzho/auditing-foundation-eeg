"""Single-column figures for the IEEE BigData paper.

  fig_matched_input.pdf   per-subject matched-input terms (the data behind Table II)
  fig_learning_curve.pdf  CBraMod pretrained vs random init, equal 240-epoch budget
  fig_reliability.pdf     reliability before and after temperature scaling

The first two need only artifacts in this repository. The reliability figure
needs per-trial probability files, which are not in the repository because of
their size; point --prob-root at a results/ tree that holds them. Without them
the figure is skipped.

    python src/scripts/make_paper_figures.py [--prob-root PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_learning_curve_figure import load_curves

HERE = Path(__file__).resolve().parent
OUT = Path("figures") / "paper"

# IEEE column width; first three slots of the validated categorical palette,
# each paired with a marker so the figures survive grayscale print.
COL_W = 3.45
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def _contrasts_module():
    spec = importlib.util.spec_from_file_location("contrasts", HERE / "make_matched_input_contrasts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


def matched_input() -> None:
    m = _contrasts_module()
    arms = [("atcnet", "ATCNet", BLUE, "o"),
            ("shallow_convnet", "ShallowConvNet", ORANGE, "s"),
            ("eegconformer", "EEG Conformer", AQUA, "^")]
    terms = {}
    for key, *_ in arms:
        narrow, broad = m.ARMS[key]
        terms[key] = (m.load(narrow)["heldout_accuracy"] - m.load(broad)["heldout_accuracy"]).dropna()
    frame = pd.DataFrame(terms)
    rng = np.random.default_rng(m.SEED)

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    xs = np.arange(len(arms))
    # Thin grey lines join each subject across architectures: the paired view.
    for _, row in frame.iterrows():
        ax.plot(xs, row.to_numpy(), color=GRID, linewidth=0.6, zorder=1)
    ax.axhline(0, color=INK_2, linewidth=0.6, zorder=0)
    for x, (key, label, colour, marker) in zip(xs, arms):
        d = frame[key].to_numpy()
        lo, hi = m.boot_ci(d, rng)
        ax.scatter(np.full(d.size, x), d, s=14, marker=marker, facecolor=colour,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(x + 0.22, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2, zorder=4)
        ax.plot(x + 0.22, d.mean(), marker="D", markersize=3.5, color=INK, zorder=5)
        # Round as Table II does (from the 4-dp term, half away from zero).
        shown = Decimal(f"{d.mean():.4f}").quantize(Decimal("0.001"), ROUND_HALF_UP)
        ax.annotate(f"{shown:+}\n{(d > 0).sum()}/{d.size} narrow", (x + 0.30, d.mean()),
                    fontsize=6.5, color=INK_2, va="center", ha="left")
    ax.set_xticks(xs, [a[1] for a in arms])
    ax.set_xlim(-0.35, len(arms) + 0.05)
    ax.set_ylabel("Narrowband $-$ broadband acc.")
    top, bottom = ax.get_ylim()[1], ax.get_ylim()[0]
    ax.text(len(arms), top, "narrowband better $\\uparrow$", fontsize=6.5, color=INK_2, ha="right", va="top")
    ax.text(len(arms), bottom, "broadband better $\\downarrow$", fontsize=6.5, color=INK_2, ha="right", va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    _save(fig, "fig_matched_input")


def learning_curve() -> None:
    budget = Path("results") / "fm_probe_budget"
    pre = load_curves(budget / "fm_probe_cbramod_finetune_budget240_subject_metrics.csv")
    rnd = load_curves(budget / "fm_probe_cbramod_finetune_randinit_budget240_subject_metrics.csv")

    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    ax.axhline(0.75, color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(50, 0.757, "chance (0.750)", fontsize=6.5, color=INK_2, ha="left", va="bottom")
    for arr, label, colour, marker in ((pre, "pretrained", BLUE, "o"), (rnd, "random init.", ORANGE, "s")):
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        ep = np.arange(1, mean.size + 1)
        ax.fill_between(ep, mean - sem, mean + sem, color=colour, alpha=0.18, linewidth=0)
        ax.plot(ep, mean, color=colour, linewidth=1.0)
        best = int(np.argmin(mean))
        ax.plot(best + 1, mean[best], marker=marker, markersize=4, color=colour,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
        text = f"{label}: best {mean[best]:.3f} @ ep. {best + 1}"
        if label == "pretrained":
            ax.annotate(text, (best + 1, mean[best]), xytext=(0, -10), textcoords="offset points",
                        fontsize=6.5, color=INK, ha="center")
        else:  # clear space above the curve, with a leader to the marker
            ax.annotate(text, (best + 1, mean[best]), xytext=(mean.size, 1.05), textcoords="data",
                        fontsize=6.5, color=INK, ha="right", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.5, shrinkB=3))
    ax.set_xlim(0, mean.size + 2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Brier")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    _save(fig, "fig_learning_curve")


def _scale(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15):
    conf, correct = p.max(axis=1), (p.argmax(axis=1) == y).astype(float)
    bins = np.array_split(np.argsort(conf, kind="mergesort"), n_bins)
    c = np.array([conf[b].mean() for b in bins])
    a = np.array([correct[b].mean() for b in bins])
    n = np.array([b.size for b in bins])
    return float(np.sum(n * np.abs(a - c)) / n.sum()), c, a


def _load_arm(root: Path, subdir: str, stem: str, key: str):
    metrics = pd.read_csv(root / subdir / f"{stem}_subject_metrics.csv")
    metrics = metrics[metrics["experiment_key"] == key].set_index("subject")
    raw, scaled, ys, per_subject = [], [], [], []
    for s in range(1, 10):
        d = np.load(root / subdir / f"{stem}_s{s}_probabilities.npz")
        meta = json.loads(str(d["metadata_json"]))
        if meta["validation_discipline"].get("eval_labels_used_for_selection") is not False:
            raise SystemExit(f"{stem} s{s}: eval labels used for selection")
        # Stored probabilities are already temperature-scaled; undoing T recovers
        # the raw ones (checked against heldout_ECE_uncalibrated below).
        p, y, t = d[f"{key}__eval_proba"], d["y_eval"], float(metrics.loc[s, "temperature"])
        raw.append(_scale(p, 1.0 / t)), scaled.append(p), ys.append(y)
        per_subject.append((_ece(raw[-1], y)[0], _ece(p, y)[0]))
    per_subject = np.array(per_subject)
    stored = metrics.loc[range(1, 10), ["heldout_ECE_uncalibrated", "heldout_ECE"]].to_numpy()
    if not np.allclose(per_subject, stored, atol=1e-6):
        raise SystemExit(f"{stem}: recomputed ECE does not match stored subject metrics")
    return np.vstack(raw), np.vstack(scaled), np.concatenate(ys), per_subject.mean(axis=0)


def reliability(root: Path) -> None:
    panels = [("CBraMod, frozen", "fm_probe_frozen_probs", "fm_probe_cbramod_frozen", "cbramod_frozen_p2"),
              ("CBraMod, random init. (CPU rerun)", "fm_probe_cpu",
               "fm_probe_cbramod_finetune_randinit_sel", "cbramod_finetune_randinit_sel")]
    if not all((root / sub / f"{stem}_s1_probabilities.npz").exists() for _, sub, stem, _ in panels):
        print(f"skip fig_reliability: probability files not under {root}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.85), sharey=True)
    for ax, (title, sub, stem, key) in zip(axes, panels):
        raw, scaled, y, (ece_raw, ece_scaled) = _load_arm(root, sub, stem, key)
        ax.plot([0, 1], [0, 1], color=INK_2, linewidth=0.6, linestyle=(0, (3, 2)))
        for p, label, colour, marker, ece in ((raw, "raw", ORANGE, "s", ece_raw),
                                              (scaled, "scaled", BLUE, "o", ece_scaled)):
            _, c, a = _ece(p, y)
            ax.plot(c, a, color=colour, linewidth=1.0, marker=marker, markersize=3,
                    markeredgecolor="white", markeredgewidth=0.4, label=f"{label} (ECE {ece:.3f})")
        ax.set_title(title, color=INK)
        ax.set_xlim(0.2, 1.03), ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Confidence")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.4, borderaxespad=0.2)
        ax.grid(color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_reliability")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prob-root", type=Path, default=Path("results"),
                    help="results/ tree holding per-trial probability files")
    args = ap.parse_args()
    matched_input()
    learning_curve()
    reliability(args.prob_root)


if __name__ == "__main__":
    main()
