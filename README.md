# CalibMI

Code and result artifacts for an audit of pretrained EEG foundation models on
motor imagery, run under a protocol in which every tunable choice is made on
training-session data alone.

**Authors:** Kevin Zhou, Sparsh Roy

## What this is

Two pretrained EEG encoders (LaBraM, CBraMod) are compared against three
supervised decoders (ShallowConvNet, ATCNet, EEG Conformer) across three
adaptation modes and two negative-control families, on BCI Competition IV-2a
and BNCI2014-004.

The selection rule is enforced by the artifact format rather than by author
discipline: every result row carries `eval_labels_used_for_selection`, written
by the runner, and the analysis scripts refuse to load any artifact where that
field is absent or true. A run that violated the protocol cannot enter a table.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-eegnet.txt

./calibmi check     # verify the environment
./calibmi tables    # rebuild every table from saved artifacts, no retraining
```

`./calibmi` with no arguments lists all commands. Everything marked `[offline]`
runs from the artifacts in this repository and needs no EEG data.

## Reproducing from raw data

Raw EEG is not redistributed here. Obtain it from the original sources and
place it as follows:

- **BCI Competition IV-2a** — `data/BCICIV_2a_gdf/A0{1..9}{T,E}.gdf`, plus the
  official evaluation labels (`python src/download_true_labels.py`). The `A0xE`
  files carry unknown cues, so the official label files are required.
- **BNCI2014-004** and **Lee2019/OpenBMI** — downloaded and cached by MOABB.

Then:

```bash
./calibmi smoke     # short code-path check
./calibmi full      # full validation-locked retraining (slow)
```

## Large artifacts

Per-trial probability files (`*_probabilities.npz`, ~460 MB) are not in this
repository. They are archived separately with a DOI; see `CITATION.cff`.
Unpack them into `results/` before running `./calibmi reliability`.

Everything else needed to check the paper's numbers — per-subject metric CSVs,
run records with platform fingerprints, the consolidated report, and the
decomposition tables — is here.

## Layout

```text
calibmi                          command-line entry point
src/experiments/                 experiment drivers
src/scripts/                     analysis, report and figure generation
src/models/                      model definitions and foundation-model adapters
reproducibility/                 the shell scripts the CLI wraps
results/paper_calibration_stats/ generated tables, including fm_audit/REPORT.md
results/fm_probe*/               per-subject metrics and run records
paper/                           manuscript source
figures/                         figures that are not script-generated
```

## Notes on scope

Three seeds were run on the decisive comparison and one elsewhere; frozen-probe
and LaBraM arms are single-seed. Some early runs used a non-deterministic MPS
backend and are marked as such in their run records — every number reported in
the paper comes from the deterministic CPU backend. These limitations are
stated in the report rather than filtered out.

## License

MIT, see `LICENSE`. The EEG datasets are governed by their own terms; this
repository redistributes none of them.
