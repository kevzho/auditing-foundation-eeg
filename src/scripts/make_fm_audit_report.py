#!/usr/bin/env python3
"""Consolidate every foundation-model audit result into one report.

The numbers in the paper should never be transcribed by hand from a terminal.
This reads whatever is currently on disk and regenerates the full set of tables,
so the report cannot drift from the artifacts and a stale figure is impossible
to keep by accident.

Sections, in the order a reader needs them:

1. Held-out performance per dataset, every method, with its role marked.
2. Foundation model vs supervised, paired within subject, BH-corrected.
3. Pretrained vs architecture-matched random init -- the control that decides
   whether pretraining contributes anything.
4. Leave-one-subject-out: the pooled regime foundation models are built for.
5. Calibration after a validation-fitted temperature.
6. The fine-tuning recipe sweep, ranked on validation.
7. Provenance: platform fingerprints and the files each table came from.

Usage::

    python src/scripts/make_fm_audit_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from make_multiplicity_table import (  # noqa: E402
    attainable_p_floor,
    benjamini_hochberg,
    wilcoxon_p,
)

FM_DIR = Path("results") / "fm_probe"
#: State-of-the-art supervised comparators live here rather than with the
#: foundation-model rows. They matter as *references*: the paper's own finding
#: is that baseline quality decides a benchmark's verdict, so the FM gap has to
#: be reported against the strongest comparator available, not only against the
#: inherited ShallowConvNet.
SOTA_DIR = Path("results") / "supervised_sota"
SWEEP_DIR = Path("results") / "fm_probe_sweep"
COMPARISON_CSV = Path("results") / "validation_only_selection_dataset_comparison.csv"
REFERENCES = ("baseline", "seed_ensemble")
METRICS = ("accuracy", "Brier", "ECE", "NLL")
LOWER_IS_BETTER = {"Brier", "ECE", "NLL"}
CONTROL_MARKER = "randinit"
EPS = 1e-12


def resolve_sources(fm_dirs: list[Path], exclude: tuple[str, ...] = ()) -> dict[str, Path]:
    """Map each metrics filename to the directory that should supply it.

    Later directories win. A configuration rerun on a deterministic backend
    lands in a second directory under the *same* filename, so passing the
    original directory first and the rerun directory second supersedes the old
    artifact without deleting it -- the superseded file stays on disk for the
    platform-sensitivity comparison.
    """
    chosen: dict[str, Path] = {}
    superseded: list[str] = []
    for directory in fm_dirs:
        for path in sorted(directory.glob("*_subject_metrics.csv")):
            if any(pattern in path.name for pattern in exclude):
                continue
            if path.name in chosen:
                superseded.append(f"{path.name}: {chosen[path.name]} -> {directory}")
            chosen[path.name] = directory
    for line in superseded:
        print(f"  superseded {line}")
    return chosen


def backend_of(directory: Path, name: str) -> tuple[str, bool] | None:
    """(device, deterministic) from the run JSON beside a metrics CSV."""
    run_json = directory / name.replace("_subject_metrics.csv", "_run.json")
    if not run_json.exists():
        return None
    try:
        platform = json.loads(run_json.read_text(encoding="utf-8")).get("platform", {})
    except (OSError, json.JSONDecodeError):
        return None
    return (
        str(platform.get("device", "unknown")),
        bool(platform.get("deterministic_algorithms", False)),
    )


def load_fm(
    fm_dirs: list[Path], require_deterministic: bool, exclude: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load every contributing arm; returns (rows, seed spread, backend caveats).

    The caveats are returned rather than only printed because a warning on
    stdout vanishes the moment the script finishes, and the artifact it warns
    about does not. A report generated without ``--require-deterministic``
    previously looked byte-for-byte like the discipline of one generated with
    it; the caveat now travels inside the document, where a reader who never
    saw the terminal will still find it.
    """
    sources = resolve_sources(fm_dirs, exclude)
    if not sources:
        raise SystemExit(f"no results under {', '.join(str(d) for d in fm_dirs)}")

    backends: dict[str, int] = {}
    nondeterministic: list[str] = []
    for name, directory in sources.items():
        backend = backend_of(directory, name)
        if backend is None:
            continue
        device, deterministic = backend
        backends[device] = backends.get(device, 0) + 1
        if not deterministic:
            nondeterministic.append(f"{name} ({device})")

    caveats: list[str] = []
    if len(backends) > 1:
        spread = ", ".join(f"{d} ({n})" for d, n in sorted(backends.items()))
        print(f"  WARNING: contributing runs span backends {sorted(backends)}")
        caveats.append(
            f"Contributing runs span more than one backend: {spread}. Numbers from "
            "different backends are not bitwise comparable."
        )
    if nondeterministic:
        message = (
            f"{len(nondeterministic)} contributing run(s) recorded deterministic=false: "
            + ", ".join(sorted(nondeterministic))
        )
        if require_deterministic:
            raise SystemExit(f"refusing to report: {message}")
        print(f"  WARNING: {message}")
        caveats.append(
            f"**{len(nondeterministic)} contributing run(s) recorded "
            "`deterministic=false`**, so they are not bitwise reproducible and "
            "their variance is bounded by seed replication rather than by the "
            "backend. Affected arms: "
            + ", ".join(f"`{n}`" for n in sorted(nondeterministic))
            + ". See the seed-spread table for the arms where that replication "
            "exists, and the provenance table for the per-file record."
        )

    frames = [pd.read_csv(directory / name) for name, directory in sorted(sources.items())]
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["subject"] = df["subject"].astype(str)
    unsafe = df[df["eval_labels_used_for_selection"].astype(str).str.lower() == "true"]
    if not unsafe.empty:
        raise SystemExit(f"refusing to report: {len(unsafe)} rows used eval labels for selection")
    collapsed, spread = aggregate_seeds(df)
    n_seeds = int(collapsed["n_seeds"].max())
    if n_seeds > 1:
        print(f"  collapsed up to {n_seeds} seeds per subject before testing")
    return collapsed, spread, caveats


SEED_KEYS = ["dataset", "experiment_key", "subject"]


def aggregate_seeds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average replicate seeds within (dataset, experiment_key, subject).

    Replicate seeds produce additional rows for the *same* subject under the
    same method. Feeding those to a paired test unchanged would treat one
    subject as several, inflating n and voiding the pairing -- so seeds are
    collapsed to one row per subject *before* any test runs. With a single
    seed this is an identity operation, which is why it is unconditional
    rather than a flag someone can forget.

    Returns (collapsed frame, per-group seed spread).
    """
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "subject"]
    counts = df.groupby(SEED_KEYS, as_index=False).size().rename(columns={"size": "n_seeds"})

    spread = pd.DataFrame()
    if int(counts["n_seeds"].max()) > 1:
        sd = (
            df.groupby(SEED_KEYS)[["heldout_accuracy", "heldout_Brier"]]
            .std(ddof=1)
            .reset_index()
            .rename(columns={"heldout_accuracy": "sd_accuracy", "heldout_Brier": "sd_Brier"})
        )
        spread = (
            sd.groupby(["dataset", "experiment_key"], as_index=False)[["sd_accuracy", "sd_Brier"]]
            .mean()
            .merge(
                counts.groupby(["dataset", "experiment_key"], as_index=False)["n_seeds"].max(),
                on=["dataset", "experiment_key"],
            )
            .sort_values(["dataset", "sd_accuracy"], ascending=[True, False])
        )

    agg = {c: "mean" for c in numeric}
    for col in df.columns:
        if col not in numeric and col not in SEED_KEYS:
            agg[col] = "first"
    collapsed = df.groupby(SEED_KEYS, as_index=False).agg(agg)
    collapsed = collapsed.merge(counts, on=SEED_KEYS)
    return collapsed, spread


def load_local_supervised() -> pd.DataFrame:
    """Supervised comparators stored alongside the probe results.

    Covers both the ShallowConvNet broadband controls written into the probe
    directory and the SOTA architectures under ``results/supervised_sota``.
    """
    frames = []
    for path in sorted(SOTA_DIR.glob("*_subject_metrics.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["subject"] = df["subject"].astype(str)
    return df


def load_supervised() -> pd.DataFrame:
    if not COMPARISON_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(COMPARISON_CSV)
    df = df[df["experiment_key"].isin(REFERENCES)].copy()
    df["dataset"] = df["dataset"].astype(str).str.lower()
    df["subject"] = df["subject"].astype(str)
    return df


def role_of(key: str) -> str:
    key = str(key)
    if CONTROL_MARKER in key:
        return "random-init control"
    if key.startswith(("labram", "cbramod", "eegpt", "biot")):
        return "foundation model"
    return "supervised"


#: Arms whose training recipe differs from everything this project's own runners
#: produce. ``baseline`` is ``baseline_ce_cropped_shallow_convnet``: multi-scale
#: crops (384 / 512 / 640, stride 128) with aggregation, which is an
#: augmentation scheme as much as a training loop.
#:
#: They stay in the report -- the cropped arm is the strongest supervised result
#: on BCI IV-2a, so excluding it would understate what the foundation models
#: have to beat. But an unlabelled row invites the reader to attribute a
#: recipe difference to the architecture, which is the confound that
#: invalidated the section 8h band term. Label, do not drop.
CROPPED_KEYS = ("baseline", "seed_ensemble")


def recipe_of(key: str) -> str:
    if role_of(key) != "supervised":
        return ""
    return "cropped+aggregated" if str(key) in CROPPED_KEYS else "single-window"


def md_table(df: pd.DataFrame, floats: int = 4) -> str:
    if df.empty:
        return "_(no rows)_\n"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
    header = "| " + " | ".join(str(c) for c in out.columns) + " |"
    rule = "|" + "|".join(["---"] * len(out.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def paired_tests(left: pd.DataFrame, right: pd.DataFrame, label_l: str, label_r: str) -> pd.DataFrame:
    """Paired Wilcoxon of every left-key against every right-key, within subject."""
    rows = []
    for lk, lpart in left.groupby("experiment_key"):
        li = lpart.set_index("subject")
        for rk, rpart in right.groupby("experiment_key"):
            ri = rpart.set_index("subject")
            shared = sorted(set(li.index) & set(ri.index))
            if not shared:
                continue
            for metric in METRICS:
                col = f"heldout_{metric}"
                if col not in lpart or col not in rpart:
                    continue
                a = pd.to_numeric(ri.loc[shared, col], errors="coerce")
                b = pd.to_numeric(li.loc[shared, col], errors="coerce")
                delta = (b - a).dropna().to_numpy(dtype=float)
                if delta.size == 0:
                    continue
                lower = metric in LOWER_IS_BETTER
                nonzero = int(np.count_nonzero(~np.isclose(delta, 0.0)))
                rows.append(
                    {
                        label_l: lk,
                        label_r: rk,
                        "metric": metric,
                        "n": int(delta.size),
                        "mean_left": float(b.mean()),
                        "mean_right": float(a.mean()),
                        "delta": float(delta.mean()),
                        "wins": int((delta < -EPS).sum()) if lower else int((delta > EPS).sum()),
                        "p_raw": wilcoxon_p(delta),
                        "p_floor": attainable_p_floor(nonzero),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh"], _ = benjamini_hochberg(out["p_raw"].tolist(), 0.05)
        out = out.sort_values(["metric", "p_raw"])
    return out


def control_pairs(fm: pd.DataFrame) -> pd.DataFrame:
    """Pair each pretrained key with its own random-init counterpart."""
    keys = set(fm["experiment_key"].unique())
    rows = []
    for key in sorted(k for k in keys if CONTROL_MARKER not in str(k)):
        matches = [
            k for k in keys
            if CONTROL_MARKER in str(k) and str(k).replace(f"_{CONTROL_MARKER}", "") == str(key)
        ]
        if not matches:
            continue
        pre = fm[fm["experiment_key"] == key]
        ctl = fm[fm["experiment_key"] == matches[0]]
        part = paired_tests(pre, ctl, "pretrained", "control")
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # One correction family across all control comparisons for this dataset.
    out["p_bh"], _ = benjamini_hochberg(out["p_raw"].tolist(), 0.05)
    return out.sort_values(["metric", "p_raw"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results") / "paper_calibration_stats" / "fm_audit")
    ap.add_argument(
        "--fm-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory of *_subject_metrics.csv. Repeatable; later directories "
        "supersede earlier ones for identically-named files, so a deterministic "
        "rerun directory can override the original without deleting it.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help="Drop contributing files whose name contains SUBSTRING. Excluded "
        "configurations are listed in the report, so a drop is always recorded "
        "rather than silent.",
    )
    ap.add_argument(
        "--require-deterministic",
        action="store_true",
        help="Refuse to write the report if any contributing run recorded "
        "deterministic=false. Use for anything that will be reported.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fm_dirs = args.fm_dir or [FM_DIR]
    exclude = tuple(args.exclude or ())

    fm, seed_spread, caveats = load_fm(fm_dirs, args.require_deterministic, exclude)
    supervised = load_supervised()
    sota = load_local_supervised()
    parts: list[str] = [
        "# Foundation-model audit: consolidated results",
        "",
        "Generated by `src/scripts/make_fm_audit_report.py` from the CSVs under "
        "`results/fm_probe/`. Do not edit by hand; rerun the script.",
        "",
        "Every configuration is selected on the validation subset only. No row in "
        "any table used held-out labels for selection -- the loader refuses to run "
        "if `eval_labels_used_for_selection` is true anywhere.",
        "",
    ]

    # Stated up front rather than buried in provenance. A caveat a reader
    # reaches only after every table has already persuaded them is a caveat
    # that arrived too late to do its job.
    if caveats:
        parts += ["## Reproducibility caveats\n"]
        parts += [f"- {line}" for line in caveats]
        parts += [
            "",
            "This section appears only when it applies. Its absence means every "
            "contributing run recorded `deterministic=true` on a single backend.",
            "",
        ]

    for dataset, fm_d in fm.groupby("dataset"):
        sup_d = supervised[supervised["dataset"] == dataset] if not supervised.empty else pd.DataFrame()
        sota_d = sota[sota["dataset"] == dataset] if not sota.empty else pd.DataFrame()
        if not sota_d.empty:
            sup_d = pd.concat([sup_d, sota_d], ignore_index=True) if not sup_d.empty else sota_d
        parts.append(f"## Dataset: {dataset}\n")

        # 1. summary
        both = pd.concat([fm_d, sup_d], ignore_index=True) if not sup_d.empty else fm_d
        summary = (
            both.groupby("experiment_key")
            .agg(
                n=("subject", "nunique"),
                acc=("heldout_accuracy", "mean"),
                Brier=("heldout_Brier", "mean"),
                ECE=("heldout_ECE", "mean"),
                NLL=("heldout_NLL", "mean"),
            )
            .reset_index()
        )
        summary.insert(1, "role", summary["experiment_key"].map(role_of))
        summary.insert(2, "recipe", summary["experiment_key"].map(recipe_of))
        summary = summary.sort_values("acc", ascending=False)
        chance = float(both["chance_accuracy"].iloc[0])
        parts += [
            f"### Held-out performance (chance accuracy {chance:.3f})\n",
            md_table(summary),
            "",
        ]

        # 2. fm vs supervised.
        # The broadband convnet rows live alongside the FM results rather than in
        # the supervised comparison CSV, but they are the *most important*
        # reference: they are the only supervised arm fed the same input the
        # foundation models get, so leaving them out would omit the one
        # comparison that separates the models from the filter band.
        fm_only = fm_d[fm_d["experiment_key"].map(role_of) == "foundation model"]
        broadband = fm_d[fm_d["experiment_key"].astype(str).str.startswith("shallow_convnet")]
        references = pd.concat([sup_d, broadband], ignore_index=True) if not sup_d.empty else broadband
        if not references.empty and not fm_only.empty:
            tests = paired_tests(fm_only, references, "fm", "reference")
            n_sig = int((tests["p_bh"] < 0.05).sum()) if not tests.empty else 0
            parts += [
                "### Foundation model vs supervised (paired within subject)\n",
                f"{n_sig} of {len(tests)} tests survive Benjamini-Hochberg at alpha=0.05. "
                f"`delta` is FM minus reference; `wins` counts subjects where the FM is better.\n",
                "References marked `cropped+aggregated` in the table above were trained "
                "with multi-scale crops rather than single windows. That is the right "
                "comparator for *can a foundation model beat supervised training*, "
                "because it is the strongest supervised arm -- but a difference against "
                "it mixes training recipe with everything else, so it is not usable for "
                "attributing a gap to the model. The band-versus-model attribution uses "
                "recipe-matched arms only, in "
                "`make_preprocessing_decomposition.py`.\n",
                md_table(tests[tests["metric"] == "accuracy"]),
                "",
            ]
            tests.to_csv(args.out / f"{dataset}_fm_vs_supervised.csv", index=False)

        # 3. pretrained vs random init
        controls = control_pairs(fm_d)
        if not controls.empty:
            parts += [
                "### Pretrained vs architecture-matched random init\n",
                "The control that separates *pretraining does not transfer* from "
                "*this architecture cannot learn from this much data*.\n",
                md_table(controls[controls["metric"] == "accuracy"]),
                "",
            ]
            controls.to_csv(args.out / f"{dataset}_pretrained_vs_random.csv", index=False)

        # 5. calibration
        calib = (
            fm_d.groupby("experiment_key")
            .agg(
                acc=("heldout_accuracy", "mean"),
                ECE=("heldout_ECE", "mean"),
                ECE_uncal=("heldout_ECE_uncalibrated", "mean"),
                T=("temperature", "mean"),
                at_clamp=("temperature_at_clamp", "sum"),
            )
            .reset_index()
            .sort_values("acc", ascending=False)
        )
        parts += [
            "### Calibration after a validation-fitted temperature\n",
            "`T` is the fitted scalar; `at_clamp` counts subjects pinned at the 20.0 bound, "
            "i.e. logits so overconfident the search saturated.\n",
            md_table(calib),
            "",
        ]
        summary.to_csv(args.out / f"{dataset}_summary.csv", index=False)

    # 4. LOSO gets its own section: different protocol, not comparable row-wise
    loso = fm[fm["selection_split"] == "pooled_training_subjects_validation"]
    if not loso.empty:
        parts += [
            "## Leave-one-subject-out (pooled regime)\n",
            "Eight subjects' training sessions pooled (~8x the data), evaluated on the "
            "ninth subject's held-out session. This is the regime foundation models are "
            "built for, and the fairest test of what pretraining buys.\n",
        ]
        for dataset, part in loso.groupby("dataset"):
            tab = (
                part.groupby("experiment_key")
                .agg(n=("subject", "nunique"), acc=("heldout_accuracy", "mean"),
                     Brier=("heldout_Brier", "mean"), ECE=("heldout_ECE", "mean"))
                .reset_index()
                .sort_values("acc", ascending=False)
            )
            parts += [f"### {dataset}\n", md_table(tab), ""]
            controls = control_pairs(part)
            if not controls.empty:
                parts += [
                    "Pretrained vs random init, pooled:\n",
                    md_table(controls[controls["metric"].isin(["accuracy", "Brier"])]),
                    "",
                ]
                controls.to_csv(args.out / f"{dataset}_loso_pretrained_vs_random.csv", index=False)

    # 6. recipe sweep
    sweeps = sorted(SWEEP_DIR.glob("*_subject_metrics.csv"))
    if sweeps:
        rows = []
        for path in sweeps:
            d = pd.read_csv(path)
            rows.append(
                {
                    "config": d["experiment_key"].iloc[0],
                    "n": len(d),
                    "val_Brier": d["val_Brier"].mean(),
                    "heldout_Brier": d["heldout_Brier"].mean(),
                    "heldout_acc": d["heldout_accuracy"].mean(),
                }
            )
        sweep = pd.DataFrame(rows).sort_values("val_Brier")
        rho = sweep["val_Brier"].corr(sweep["heldout_Brier"], method="spearman")
        parts += [
            "## Fine-tuning recipe sweep\n",
            "Ranked on **validation** Brier -- ranking on held-out would be the very "
            "selection violation this protocol exists to prevent. Spearman correlation "
            f"between the validation and held-out rankings: {rho:.3f}.\n",
            md_table(sweep),
            "",
        ]
        sweep.to_csv(args.out / "recipe_sweep.csv", index=False)

    if not seed_spread.empty:
        parts += [
            "## Seed variability\n",
            "Per-subject standard deviation across replicate seeds, averaged over "
            "subjects. The seed drives both weight initialisation and the "
            "fit/selection split, so this bounds how much of any reported effect "
            "is run-to-run noise. Compare against the effect sizes in the tables "
            "above before treating a small difference as real.\n",
            md_table(seed_spread),
            "",
        ]
        seed_spread.to_csv(args.out / "seed_spread.csv", index=False)

    # 7. provenance
    parts += ["## Provenance\n"]
    prov = []
    for name, directory in sorted(resolve_sources(fm_dirs, exclude).items()):
        path = directory / name.replace("_subject_metrics.csv", "_run.json")
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        platform = meta.get("platform", {})
        prov.append(
            {
                "file": path.name,
                "source_dir": str(directory),
                "device": platform.get("device"),
                "deterministic": platform.get("deterministic_algorithms"),
                "torch": platform.get("torch"),
                "seed": meta.get("seed"),
            }
        )
    devices = sorted({str(r["device"]) for r in prov})
    parts += [
        md_table(pd.DataFrame(prov)),
        "",
        "Only the runs listed above contribute to this report. Where a "
        "configuration exists in more than one source directory, the last "
        "directory on the command line supersedes the earlier ones; the "
        "superseded artifact stays on disk for the platform-sensitivity "
        f"comparison. Contributing backends: {', '.join(devices)}.",
        "",
    ]
    if exclude:
        parts += [
            "Explicitly excluded from this report: "
            + ", ".join(f"`{e}`" for e in exclude)
            + ".",
            "",
        ]

    report = args.out / "REPORT.md"
    report.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {report}")
    print(f"  {len(fm)} foundation-model rows across {fm['dataset'].nunique()} dataset(s)")


if __name__ == "__main__":
    main()
