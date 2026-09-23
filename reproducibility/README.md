# Reproducibility Commands

This folder contains the commands needed to reproduce the paper-facing
artifacts.

Run order:

```bash
bash reproducibility/00_smoke_test.sh
bash reproducibility/01_readout_from_saved_results.sh
```

To verify raw BCI IV-2a provenance without overwriting the locked `data/*.npz`
inputs, regenerate MNE epochs and workflow NPZ files into a scratch results
directory:

```bash
python src/preprocess.py \
  --data-dir data/BCICIV_2a_gdf \
  --out-dir results/preprocessed_mne_check \
  --labels-dir data/BCICIV_2a_gdf \
  --npz-dir results/preprocessed_mne_check/npz
```

Only run the full experiment script when the raw/local datasets are present and
you are ready for long training jobs:

```bash
bash reproducibility/02_full_validation_locked_runs.sh
```

After full runs have saved `probabilities/*.npz` files, reliability diagrams can
be regenerated without retraining:

```bash
bash reproducibility/03_reliability_figures_from_probabilities.sh
```

The readout script is the one to use while writing. It consumes saved CSV/JSON
results and regenerates merged comparisons, paired deltas, bootstrap CIs,
Wilcoxon p-values, and compact paper table snippets.

Generate a manifest with package versions, git state, reproduction commands,
and SHA256 hashes of raw GDF and workflow NPZ files:

```bash
python src/scripts/make_reproducibility_manifest.py \
  --raw-dir data/BCICIV_2a_gdf \
  --npz-dir data \
  --out results/reproducibility_manifest.json
```

Audit retained result artifacts for held-out label leakage flags:

```bash
python src/scripts/audit_leakage_fields.py --results-dir results
```
