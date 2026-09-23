import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_calibration_workflow import run_mdrm_t_ea


def _make_split(rng, n_trials, n_channels, n_times, n_classes):
    y = np.arange(n_trials, dtype=np.int64) % n_classes
    X = rng.normal(size=(n_trials, n_channels, n_times)).astype(np.float32)
    for label in range(n_classes):
        X[y == label, label % n_channels, :] += 0.35
    return X, y


def main():
    rng = np.random.default_rng(7)
    n_channels = 6
    n_times = 80
    n_classes = 3
    X_train, y_train = _make_split(rng, 30, n_channels, n_times, n_classes)
    X_val, y_val = _make_split(rng, 15, n_channels, n_times, n_classes)
    X_eval, y_eval = _make_split(rng, 18, n_channels, n_times, n_classes)
    data = {
        "dataset": "synthetic",
        "subject": "smoke",
        "n_classes": n_classes,
        "classes": [str(i) for i in range(n_classes)],
        "X_train_raw": X_train,
        "y_train": y_train,
        "X_val_raw": X_val,
        "y_val": y_val,
        "X_eval_raw": X_eval,
        "y_eval": y_eval,
        "train_split": "synthetic_train",
        "selection_split": "synthetic_validation",
        "heldout_split": "synthetic_eval_final_report_only",
    }
    args = argparse.Namespace(temperature_max=20.0, include_confusion=False)

    result = run_mdrm_t_ea(data, args)
    run = result["run"]

    assert run["experiment"] == "mdrm_t_train_only_euclidean_alignment"
    assert run["selection_audit"]["eval_labels_used_for_selection"] is False
    assert run["metadata"]["euclidean_alignment"]["fit_split"] == "synthetic_train"
    assert run["calibration"]["temperature_fit_source"] == "synthetic_validation"
    assert run["metadata"]["abstention_threshold_source"] == "synthetic_validation"
    assert result["val_proba"].shape == (X_val.shape[0], n_classes)
    assert result["eval_proba"].shape == (X_eval.shape[0], n_classes)
    assert np.allclose(result["val_proba"].sum(axis=1), 1.0)
    assert np.allclose(result["eval_proba"].sum(axis=1), 1.0)

    print("MDRM-T workflow smoke test PASSED")


if __name__ == "__main__":
    main()
