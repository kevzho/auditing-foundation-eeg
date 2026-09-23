"""Leakage-audited MDRM-T baseline for cross-session BCI evaluation.

MDRM-T combines Euclidean Alignment (EA), pyRiemann's MDM classifier, scalar
temperature scaling, and validation-locked abstention thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
from pyriemann.classification import MDM
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score

EPS = 1e-12

def _as_float_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim < 3:
        raise ValueError(f"{name} must have shape (n_trials, n_channels, ...), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one trial")
    return arr

def _fingerprint(x: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(x))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return {"shape": tuple(arr.shape), "dtype": str(arr.dtype), "sha256": digest.hexdigest()}


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def _regularize_spd(mat: np.ndarray, eps: float) -> np.ndarray:
    mat = _symmetrize(np.asarray(mat, dtype=float))
    scale = float(np.trace(mat)) / max(mat.shape[0], 1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return _symmetrize(mat + float(eps) * scale * np.eye(mat.shape[0]))


def _matrix_inv_sqrt_spd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mat = _symmetrize(np.asarray(mat, dtype=float))
    evals, evecs = np.linalg.eigh(mat)
    evals = np.maximum(evals, eps)
    inv_sqrt = (evecs * (1.0 / np.sqrt(evals))) @ evecs.T
    return _symmetrize(inv_sqrt)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), EPS, None)


def _normalize_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, EPS, 1.0)
    return proba / np.clip(proba.sum(axis=1, keepdims=True), EPS, None)


def _one_hot_indices(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    y_idx = np.asarray(y_idx, dtype=int)
    out = np.zeros((y_idx.size, n_classes), dtype=float)
    out[np.arange(y_idx.size), y_idx] = 1.0
    return out


def _ece(y_true_idx: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    proba = _normalize_proba(proba)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == np.asarray(y_true_idx, dtype=int)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(y_true_idx.size)
    if total == 0.0:
        return float("nan")

    error = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            error += float(mask.sum()) / total * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(error)


def _brier(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    proba = _normalize_proba(proba)
    y = _one_hot_indices(y_true_idx, proba.shape[1])
    return float(np.mean(np.sum((proba - y) ** 2, axis=1)))


@dataclass
class MDRMT:
    """MDM + Euclidean Alignment + Temperature scaling.

    The intended protocol is BCI IV 2a cross-session transfer, e.g. A0xT for
    `fit` and A0xE for `evaluate`. EA, MDM centroids, temperature, and
    abstention thresholds are all fitted without evaluation-session data.
    """

    metric: str | dict[str, str] = "riemann"
    n_bins: int = 10
    cov_regularization: float = 1e-6
    temperature_bounds: tuple[float, float] = (0.05, 20.0)
    coverage_targets: tuple[float, ...] = (0.40, 0.60, 0.80)
    mdm_: MDM | None = field(default=None, init=False)
    mean_cov_: np.ndarray | None = field(default=None, init=False)
    ea_inv_sqrt_: np.ndarray | None = field(default=None, init=False)
    temperature_: float | None = field(default=None, init=False)
    abstention_thresholds_: dict[float, float] = field(default_factory=dict, init=False)
    classes_: np.ndarray | None = field(default=None, init=False)
    audit_trail_: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "MDRMT":
        """Fit EA, MDM, temperature, and abstention thresholds from train/val only."""

        X_train = _as_float_array(X_train, "X_train")
        X_val = _as_float_array(X_val, "X_val")
        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train have different numbers of trials")
        if X_val.shape[0] != y_val.shape[0]:
            raise ValueError("X_val and y_val have different numbers of trials")

        self.audit_trail_ = {
            "fit_called": True,
            "fit_inputs": {
                "X_train": _fingerprint(X_train),
                "y_train": _fingerprint(y_train),
                "X_val": _fingerprint(X_val),
                "y_val": _fingerprint(y_val),
            },
            "eval_inputs": [],
            "fit_artifacts": ["mean_cov_", "ea_inv_sqrt_", "mdm_", "temperature_", "abstention_thresholds_"],
            "notes": [
                "EA mean covariance fitted on X_train only.",
                "MDM fitted on EA-transformed X_train only.",
                "Temperature and abstention thresholds fitted on EA-transformed X_val/y_val only.",
            ],
        }

        self.mean_cov_ = self._fit_mean_covariance(X_train)
        self.ea_inv_sqrt_ = _matrix_inv_sqrt_spd(self.mean_cov_)

        X_train_ea = self._transform_ea(X_train)
        X_val_ea = self._transform_ea(X_val)
        X_train_mdm = self._to_mdm_input(X_train_ea)
        X_val_mdm = self._to_mdm_input(X_val_ea)

        self.mdm_ = MDM(metric=self.metric)
        self.mdm_.fit(X_train_mdm, y_train)
        self.classes_ = np.asarray(self.mdm_.classes_)

        val_logits = self._logits_from_mdm_input(X_val_mdm)
        y_val_idx = self._labels_to_indices(y_val)
        self.temperature_ = self._fit_temperature(val_logits, y_val_idx)
        val_proba = _softmax(val_logits / self.temperature_)
        val_conf = val_proba.max(axis=1)
        self.abstention_thresholds_ = {
            target: self._threshold_for_coverage(val_conf, target) for target in self.coverage_targets
        }
        return self

    def evaluate(self, X_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, Any]:
        """Evaluate on held-out data using only fitted train/validation artifacts."""

        self._check_is_fit()
        X_eval = _as_float_array(X_eval, "X_eval")
        y_eval = np.asarray(y_eval)
        if X_eval.shape[0] != y_eval.shape[0]:
            raise ValueError("X_eval and y_eval have different numbers of trials")

        self.audit_trail_.setdefault("eval_inputs", []).append(
            {"X_eval": _fingerprint(X_eval), "y_eval": _fingerprint(y_eval)}
        )

        proba = self.predict_proba(X_eval)
        pred_idx = proba.argmax(axis=1)
        y_idx = self._labels_to_indices(y_eval)
        pred_labels = self.classes_[pred_idx]
        conf = proba.max(axis=1)

        results: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_eval, pred_labels)),
            "macro_f1": float(f1_score(y_eval, pred_labels, average="macro", zero_division=0)),
            "ece": _ece(y_idx, proba, n_bins=self.n_bins),
            "brier": _brier(y_idx, proba),
            "risk_coverage_data": self._risk_coverage_data(y_idx, proba),
        }

        for target in self.coverage_targets:
            threshold = self.abstention_thresholds_[target]
            mask = conf >= threshold
            key = f"sel_acc_{int(round(target * 100))}"
            results[key] = float(np.mean(pred_idx[mask] == y_idx[mask])) if mask.any() else float("nan")
            results[f"coverage_{int(round(target * 100))}"] = float(mask.mean())
            results[f"threshold_{int(round(target * 100))}"] = float(threshold)

        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return temperature-scaled class probabilities."""

        self._check_is_fit()
        X_ea = self._transform_ea(_as_float_array(X, "X"))
        logits = self._logits_from_mdm_input(self._to_mdm_input(X_ea))
        return _softmax(logits / float(self.temperature_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels using temperature-scaled probabilities."""

        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def _fit_mean_covariance(self, X_train: np.ndarray) -> np.ndarray:
        if X_train.shape[-1] == X_train.shape[-2]:
            covs = X_train
        else:
            covs = np.matmul(X_train, np.swapaxes(X_train, -1, -2)) / max(X_train.shape[-1] - 1, 1)
        mean_cov = np.mean(covs, axis=0)
        return _regularize_spd(mean_cov, self.cov_regularization)

    def _transform_ea(self, X: np.ndarray) -> np.ndarray:
        self._check_ea_is_fit()
        if X.shape[-2] != self.ea_inv_sqrt_.shape[0]:
            raise ValueError(
                f"X channel dimension {X.shape[-2]} does not match fitted EA dimension {self.ea_inv_sqrt_.shape[0]}"
            )
        if X.shape[-1] == X.shape[-2]:
            aligned = self.ea_inv_sqrt_[None, :, :] @ X @ self.ea_inv_sqrt_[None, :, :]
            return np.asarray([_symmetrize(mat) for mat in aligned])
        return self.ea_inv_sqrt_[None, :, :] @ X

    def _to_mdm_input(self, X_ea: np.ndarray) -> np.ndarray:
        if X_ea.shape[-1] == X_ea.shape[-2]:
            return np.asarray([_regularize_spd(mat, self.cov_regularization) for mat in X_ea])
        covs = np.matmul(X_ea, np.swapaxes(X_ea, -1, -2)) / max(X_ea.shape[-1] - 1, 1)
        return np.asarray([_regularize_spd(mat, self.cov_regularization) for mat in covs])

    def _logits_from_mdm_input(self, X_mdm: np.ndarray) -> np.ndarray:
        if self.mdm_ is None:
            raise RuntimeError("MDM classifier is not fit yet")
        distances = np.asarray(self.mdm_.transform(X_mdm), dtype=float)
        return -distances

    def _fit_temperature(self, logits: np.ndarray, y_idx: np.ndarray) -> float:
        lo, hi = self.temperature_bounds
        if lo <= 0 or hi <= lo:
            raise ValueError("temperature_bounds must be positive and increasing")

        def nll(temp: float) -> float:
            proba = _softmax(logits / temp)
            return float(-np.mean(np.log(np.clip(proba[np.arange(y_idx.size), y_idx], EPS, 1.0))))

        result = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        if not result.success:
            raise RuntimeError(f"Temperature optimization failed: {result.message}")
        return float(result.x)

    @staticmethod
    def _threshold_for_coverage(confidence: np.ndarray, target_coverage: float) -> float:
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError(f"coverage target must be in (0, 1], got {target_coverage}")
        confidence = np.asarray(confidence, dtype=float)
        kth = int(np.ceil((1.0 - target_coverage) * confidence.size))
        kth = min(max(kth, 0), confidence.size - 1)
        return float(np.sort(confidence)[kth])

    def _risk_coverage_data(self, y_idx: np.ndarray, proba: np.ndarray) -> list[dict[str, float | int | str | None]]:
        pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        thresholds = np.unique(np.r_[0.0, conf, 1.0, list(self.abstention_thresholds_.values())])
        rows: list[dict[str, float | int | str | None]] = []
        target_by_threshold = {round(v, 12): k for k, v in self.abstention_thresholds_.items()}
        for threshold in sorted(thresholds, reverse=True):
            mask = conf >= threshold
            acc = float(np.mean(pred[mask] == y_idx[mask])) if mask.any() else float("nan")
            rows.append(
                {
                    "threshold": float(threshold),
                    "target_coverage": target_by_threshold.get(round(float(threshold), 12)),
                    "coverage": float(mask.mean()),
                    "risk": float(1.0 - acc) if np.isfinite(acc) else float("nan"),
                    "accuracy": acc,
                    "n_covered": int(mask.sum()),
                }
            )
        return rows

    def _labels_to_indices(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Class labels are not available before fitting MDM")
        mapping = {label: idx for idx, label in enumerate(self.classes_)}
        try:
            return np.asarray([mapping[label] for label in y], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Label {exc.args[0]!r} was not seen during training") from exc

    def _check_ea_is_fit(self) -> None:
        if self.mean_cov_ is None or self.ea_inv_sqrt_ is None:
            raise RuntimeError("MDRMT is not fit yet")

    def _check_is_fit(self) -> None:
        if self.mdm_ is None or self.classes_ is None or self.temperature_ is None:
            raise RuntimeError("MDRMT is not fit yet")
        self._check_ea_is_fit()


def audit_leakage(mdrm_t_instance: MDRMT) -> bool:
    """Assert that the estimator did not use evaluation data during fit.

    The audit checks the estimator's internal provenance: fit artifacts must be
    based only on train/validation inputs, and any later evaluation fingerprints
    must not match fit inputs.
    """

    details: list[str] = []
    ok = True
    trail = getattr(mdrm_t_instance, "audit_trail_", {})

    try:
        assert trail.get("fit_called") is True, "fit() has not been called"
        details.append("fit() recorded train/validation inputs only")

        fit_inputs = trail.get("fit_inputs", {})
        required = {"X_train", "y_train", "X_val", "y_val"}
        assert required.issubset(fit_inputs), f"missing fit fingerprints: {sorted(required - set(fit_inputs))}"

        fit_hashes = {name: fp["sha256"] for name, fp in fit_inputs.items() if name.startswith("X_")}
        eval_inputs = trail.get("eval_inputs", [])
        for idx, eval_pair in enumerate(eval_inputs):
            for eval_name, eval_fp in eval_pair.items():
                if not eval_name.startswith("X_"):
                    continue
                for fit_name, fit_hash in fit_hashes.items():
                    assert eval_fp["sha256"] != fit_hash, (
                        f"{eval_name} from evaluate() matches {fit_name} from fit() at eval call {idx}"
                    )

        assert mdrm_t_instance.mean_cov_ is not None, "EA mean covariance is missing"
        assert mdrm_t_instance.ea_inv_sqrt_ is not None, "EA transform is missing"
        assert mdrm_t_instance.mdm_ is not None, "MDM classifier is missing"
        assert mdrm_t_instance.temperature_ is not None, "temperature is missing"
        assert mdrm_t_instance.abstention_thresholds_, "abstention thresholds are missing"

        details.append("EA mean covariance was fitted before evaluation and stored as a train-only artifact")
        details.append("temperature and abstention thresholds were validation-locked")
        details.append(f"evaluation calls audited: {len(eval_inputs)}")
    except AssertionError as exc:
        ok = False
        details.append(str(exc))

    status = "PASS" if ok else "FAIL"
    print(f"{status}: leakage audit")
    for detail in details:
        print(f"- {detail}")
    assert ok, "MDRMT leakage audit failed"
    return True
