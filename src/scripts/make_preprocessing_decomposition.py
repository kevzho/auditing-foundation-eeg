#!/usr/bin/env python3
"""Split the foundation-model deficit into a filter-band part and a model part.

The two arms of this project are fed differently by design: classical motor
imagery pipelines want 8-30 Hz, the foundation models want the broadband input
they were pretrained on. A reviewer is entitled to ask how much of the gap is
the models and how much is the filter. Training the *same* supervised
architecture on the *identical* broadband arrays the foundation models see
answers that, because it holds the input fixed and varies only the model:

    total = native_narrowband - foundation_model
    band  = native_narrowband - same_arch_on_broadband     (the filter's share)
    model = same_arch_on_broadband - foundation_model      (the model's share)

Documented in docs section 8h, where this decomposition was first computed by
hand for ShallowConvNet. The lesson recorded there is why this exists as a
script: with an *under-tuned* broadband baseline the same arithmetic gave 54/46
and a non-significant model term, and with a per-subject validation search it
gave 29/71 and a significant one. The split is a property of the baseline as
much as of the foundation models, so it has to be recomputed against every
comparator rather than quoted once.

Both components are tested paired within subject, over the subjects present in
all three arms, and corrected together.

Usage::

    python src/scripts/make_preprocessing_decomposition.py
"""

from __future__ import annotations

import argparse
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
SOTA_DIR = Path("results") / "supervised_sota"
COMPARISON_CSV = Path("results") / "validation_only_selection_dataset_comparison.csv"
#: The narrowband arm for ShallowConvNet predates the band-aware runner and
#: lives in the cross-dataset comparison CSV under the protocol's own name.
#:
#: It is *not* interchangeable with the arms this project's own runner produces.
#: `baseline` is `baseline_ce_cropped_shallow_convnet`: multi-scale crops (384 /
#: 512 / 640, stride 128) with crop aggregation, which is an augmentation scheme
#: as much as a training loop. `run_supervised_on_fm_data.py` trains single
#: windows. Comparing one against the other varies the training recipe and the
#: filter band together, so it is imported under a name that says so and is
#: barred from the decomposition below.
NATIVE_KEY = "baseline"
NATIVE_IMPORT_KEY = "shallow_convnet_narrowband_cropped"
CROSS_PIPELINE_SUFFIX = "_cropped"
NATIVE_ARCH = "shallow_convnet"
CONTROL_MARKER = "randinit"
FM_PREFIXES = ("labram", "cbramod", "eegpt", "biot")
METRIC = "heldout_accuracy"
EPS = 1e-12


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    out["subject"] = out["subject"].astype(str)
    return out


def load_supervised_arms() -> pd.DataFrame:
    """Every supervised arm, tagged with the architecture and band it used."""
    frames: list[pd.DataFrame] = []

    for path in sorted(SOTA_DIR.glob("*_subject_metrics.csv")):
        frames.append(pd.read_csv(path))
    for path in sorted(FM_DIR.glob("*_subject_metrics.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        if str(df["experiment_key"].iloc[0]).startswith(NATIVE_ARCH):
            frames.append(df)

    if COMPARISON_CSV.exists():
        native = pd.read_csv(COMPARISON_CSV)
        native = native[native["experiment_key"] == NATIVE_KEY].copy()
        # Rename into the band-aware scheme so it joins with everything else,
        # keeping the marker that says which training loop produced it.
        native["experiment_key"] = NATIVE_IMPORT_KEY
        frames.append(native)

    if not frames:
        return pd.DataFrame()

    return annotate_arms(pd.concat(frames, ignore_index=True))


def parse_key(key: str) -> dict:
    """Split ``{arch}_{band}[_{variant}][_tuned]`` into its parts.

    Parsed by locating the band token rather than by stripping known suffixes.
    A regex of known endings silently mis-parses the first key that carries an
    unanticipated tag -- `atcnet_narrowband_retune_tuned` would yield an `arch`
    of the whole string, pair with nothing, and vanish from the output without
    an error.
    """
    key = str(key)
    for band in ("narrowband", "broadband"):
        marker = f"_{band}"
        if marker in key:
            arch, _, rest = key.partition(marker)
            parts = [p for p in rest.split("_") if p]
            tuned = parts and parts[-1] == "tuned"
            if tuned:
                parts = parts[:-1]
            cross = CROSS_PIPELINE_SUFFIX.lstrip("_") in parts
            variant = "_".join(p for p in parts if p != CROSS_PIPELINE_SUFFIX.lstrip("_"))
            return {
                "arch": arch,
                "band": band,
                "tuned": bool(tuned),
                "cross_pipeline": bool(cross),
                "variant": variant,
            }
    return {"arch": key, "band": "", "tuned": False, "cross_pipeline": False, "variant": ""}


def annotate_arms(df: pd.DataFrame) -> pd.DataFrame:
    """Split each arm's key into the architecture, band and recipe it encodes."""
    df = _norm(df)
    parsed = pd.DataFrame(
        [parse_key(k) for k in df["experiment_key"]], index=df.index
    )
    return pd.concat([df, parsed], axis=1)


def load_fm_arms() -> pd.DataFrame:
    frames = []
    for path in sorted(FM_DIR.glob("*_subject_metrics.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        key = str(df["experiment_key"].iloc[0])
        if key.startswith(FM_PREFIXES) and CONTROL_MARKER not in key:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return _norm(pd.concat(frames, ignore_index=True))


def pick_arm(part: pd.DataFrame) -> pd.DataFrame:
    """Prefer the validation-tuned variant of an arm when both are on disk.

    Reporting the untuned one would understate the baseline, which is the exact
    failure documented in docs section 8h.

    Arms from a different training loop are dropped outright rather than ranked.
    The decomposition's entire claim is that only the input changed between the
    two supervised arms; a cropped-and-aggregated arm on one side would put the
    training recipe inside the term labelled "filter band".
    """
    if part.empty:
        return part
    part = part[~part["cross_pipeline"]]
    if part.empty:
        return part
    tuned = part[part["tuned"]]
    chosen = tuned if not tuned.empty else part
    key = sorted(chosen["experiment_key"].unique())[0]
    return chosen[chosen["experiment_key"] == key]


def _paired(a: pd.Series, b: pd.Series) -> dict:
    """``a`` minus ``b`` over a shared, aligned subject index."""
    delta = (a - b).dropna().to_numpy(dtype=float)
    nonzero = int(np.count_nonzero(~np.isclose(delta, 0.0)))
    return {
        "delta": float(delta.mean()) if delta.size else float("nan"),
        "wins": int((delta > EPS).sum()),
        "n": int(delta.size),
        "p_raw": wilcoxon_p(delta) if delta.size else float("nan"),
        "p_floor": attainable_p_floor(nonzero),
    }


def decompose(supervised: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sup_d in supervised.groupby("dataset"):
        fm_d = fm[fm["dataset"] == dataset]
        if fm_d.empty:
            continue
        for (arch, variant), arch_d in sup_d.groupby(["arch", "variant"]):
            native = pick_arm(arch_d[arch_d["band"] == "narrowband"])
            broad = pick_arm(arch_d[arch_d["band"] == "broadband"])
            if native.empty or broad.empty:
                continue
            nat_i = native.set_index("subject")[METRIC]
            brd_i = broad.set_index("subject")[METRIC]
            for fm_key, fm_part in fm_d.groupby("experiment_key"):
                fm_i = fm_part.set_index("subject")[METRIC]
                shared = sorted(set(nat_i.index) & set(brd_i.index) & set(fm_i.index))
                if len(shared) < 3:
                    continue
                nat, brd, fmv = nat_i.loc[shared], brd_i.loc[shared], fm_i.loc[shared]
                band = _paired(nat, brd)
                model = _paired(brd, fmv)
                total = _paired(nat, fmv)
                # Shares are only meaningful when the native baseline actually
                # leads the foundation model; otherwise there is no deficit to
                # apportion and the ratios would be noise divided by noise.
                valid = total["delta"] > EPS
                rows.append(
                    {
                        "dataset": dataset,
                        "arch": arch,
                        "variant": variant or "base",
                        "baseline_narrowband": sorted(native["experiment_key"].unique())[0],
                        "baseline_broadband": sorted(broad["experiment_key"].unique())[0],
                        "foundation_model": fm_key,
                        "n": total["n"],
                        "acc_narrowband": float(nat.mean()),
                        "acc_broadband": float(brd.mean()),
                        "acc_fm": float(fmv.mean()),
                        "total_gap": total["delta"],
                        "band_gap": band["delta"],
                        "model_gap": model["delta"],
                        # Negative when the FM-required preprocessing *helps* the
                        # supervised model, which removes the preprocessing
                        # excuse for the foundation model's deficit entirely.
                        "pipeline_effect": "costs" if band["delta"] > EPS else "helps",
                        "band_share": band["delta"] / total["delta"] if valid else float("nan"),
                        "model_share": model["delta"] / total["delta"] if valid else float("nan"),
                        "band_p_raw": band["p_raw"],
                        "model_p_raw": model["p_raw"],
                        "band_wins": f"{band['wins']}/{band['n']}",
                        "model_wins": f"{model['wins']}/{model['n']}",
                        "p_floor": total["p_floor"],
                        "shares_meaningful": bool(valid),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # One correction family: every component test computed by this script.
    pooled = out["band_p_raw"].tolist() + out["model_p_raw"].tolist()
    adjusted, _ = benjamini_hochberg(pooled, 0.05)
    half = len(out)
    out["band_p_bh"] = adjusted[:half]
    out["model_p_bh"] = adjusted[half:]
    return out.sort_values(["dataset", "arch", "model_gap"], ascending=[True, True, False])


def md_table(df: pd.DataFrame, cols: list[str], floats: int = 4) -> str:
    if df.empty:
        return "_(no rows)_\n"
    out = df[cols].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
    header = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in out.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def pipeline_variants(supervised: pd.DataFrame) -> pd.DataFrame:
    """Same architecture, same band, different preprocessing variant.

    The main decomposition pairs a narrowband arm against a broadband one and
    calls the difference `band_gap`, with a standing warning that the two
    pipelines differ in more than the band. This isolates one of those
    differences instead of warning about it: an arm carrying a variant tag
    (`ica`, `retune`) is compared against the plain arm of the same
    architecture, dataset and band.

    Such arms are invisible in the main table by construction -- a variant
    group needs both bands to decompose, and these have one apiece, so
    ``decompose`` withholds them. Withholding the row is right; withholding
    the *result* is not, because the ICA arm is what turns "the pipeline costs
    the baseline 0.037 on BCI IV-2a" into a statement about artifact removal
    rather than about frequency content.
    """
    rows = []
    for (dataset, arch, band), grp in supervised.groupby(["dataset", "arch", "band"]):
        base = grp[grp["variant"].isin(["", "base"])]
        if base.empty:
            continue
        base_arm = pick_arm(base)
        for variant, part in grp[~grp["variant"].isin(["", "base"])].groupby("variant"):
            arm = pick_arm(part)
            merged = base_arm.merge(arm, on="subject", suffixes=("_base", "_var"))
            if merged.empty:
                continue
            rows.append({
                "dataset": dataset,
                "arch": arch,
                "band": band,
                "variant": variant,
                "acc_base": merged["heldout_accuracy_base"].mean(),
                "acc_variant": merged["heldout_accuracy_var"].mean(),
                # _paired supplies delta, wins, n, p_raw and p_floor, so the
                # variant contrast is tested exactly the way every other
                # contrast in this file is.
                **_paired(merged["heldout_accuracy_var"], merged["heldout_accuracy_base"]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results") / "paper_calibration_stats" / "fm_audit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    supervised = load_supervised_arms()
    fm = load_fm_arms()
    if supervised.empty or fm.empty:
        print("No decomposable arms on disk yet.", file=sys.stderr)
        return
    table = decompose(supervised, fm)
    if table.empty:
        print(
            "No architecture has both a narrowband and a broadband arm on the "
            "same dataset yet; nothing to decompose.",
            file=sys.stderr,
        )
        return

    csv_path = args.out / "preprocessing_decomposition.csv"
    table.to_csv(csv_path, index=False)

    cols = [
        "arch", "variant", "foundation_model", "n", "acc_narrowband", "acc_broadband", "acc_fm",
        "total_gap", "band_gap", "band_wins", "band_p_bh", "model_gap", "pipeline_effect",
        "band_share", "model_share", "model_wins", "model_p_bh",
    ]
    parts = [
        "# Preprocessing decomposition of the foundation-model deficit",
        "",
        "Generated by `src/scripts/make_preprocessing_decomposition.py`. Do not "
        "edit by hand; rerun the script.",
        "",
        "`band_gap` is the **preprocessing-pipeline term**, measured by running the "
        "*same* supervised architecture on the broadband arrays the foundation models "
        "are given. It is not a filter-band term: the two pipelines differ in band "
        "(8-30 Hz vs 0.1-75 Hz), filter design (IIR vs FIR), sampling rate (250 vs "
        "200 Hz), notching, and -- for the legacy narrowband arrays -- EOG-guided ICA. "
        "Do not attribute it to the band alone.",
        "",
        "`model_gap` is the remainder, the part attributable to the model, and it is "
        "the clean term: both of its arms see identical broadband input.",
        "",
        "`pipeline_effect` says which way the preprocessing cuts. **`helps` means the "
        "foundation models' own required preprocessing made the supervised baseline "
        "*better*** -- in that case there is no preprocessing excuse for the model "
        "gap at all, and `band_share` is negative with `model_share` above 1 by "
        "construction. Read the raw gaps, not the shares, whenever that happens.",
        "",
        "Shares are blank where the supervised baseline does not lead the foundation "
        "model, because there is then no deficit to apportion.",
        "",
        "Arms are recipe-matched by construction: an arm from a different training "
        "loop is barred, not ranked. The broadband baseline is the validation-tuned "
        "variant wherever one exists -- an under-tuned baseline does not give a "
        "conservative answer, it gave a qualitatively wrong one (docs section 8h).",
        "",
    ]
    for dataset, part in table.groupby("dataset"):
        parts += [f"## Dataset: {dataset}\n", md_table(part, cols), ""]

    variants = pipeline_variants(supervised)
    if not variants.empty:
        parts += [
            "## What the pipeline term is made of\n",
            "The `band_gap` above is named for the band but contains every way the "
            "two pipelines differ. Each row here holds the architecture, dataset and "
            "band fixed and changes exactly one of those ways, so the term can be "
            "attributed instead of caveated. `delta` is the variant minus the plain "
            "arm; positive means the variant is better.",
            "",
            "These arms cannot appear in the tables above: decomposition needs an "
            "arch to have both bands, and a variant arm has one. That is why the "
            "contrast is reported separately rather than dropped.",
            "",
            md_table(
                variants,
                ["dataset", "arch", "band", "variant", "n",
                 "acc_base", "acc_variant", "delta", "wins", "p_raw", "p_floor"],
            ),
            "",
        ]

    md_path = args.out / "preprocessing_decomposition.md"
    md_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
