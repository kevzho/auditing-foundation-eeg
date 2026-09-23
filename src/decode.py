"""Calibrated LDA/SVM decoders and equal-weight probability voting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from config import CALIBRATION_CV, CALIBRATION_METHOD, CSP_N_COMPONENTS, FBCSP_BANDS, INCLUDE_BANDPOWER, SFREQ
from fbcsp import FBCSPFeatures


@dataclass
class HeldoutPrediction:
    subject: int
    y_true: np.ndarray
    classes: np.ndarray
    proba: dict[str, np.ndarray]


def _feature_step() -> FBCSPFeatures:
    return FBCSPFeatures(
        sfreq=SFREQ,
        bands=FBCSP_BANDS,
        n_components=CSP_N_COMPONENTS,
        include_bandpower=INCLUDE_BANDPOWER,
    )


def build_decoder(name: str) -> CalibratedClassifierCV:
    """Build a calibrated decoder. Only LDA and SVM are on the main path."""

    name = name.upper()
    if name == "LDA":
        base_clf = LinearDiscriminantAnalysis()
    elif name == "SVM":
        base_clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    else:
        raise ValueError(f"{name} is not a main-path decoder. Use only LDA and SVM.")
    base = Pipeline([("feat", _feature_step()), ("clf", base_clf)])
    return CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv=CALIBRATION_CV)


def fit_predict_subject(
    *,
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_names: Sequence[str],
) -> HeldoutPrediction:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    classes = np.arange(len(le.classes_), dtype=int)

    probas: dict[str, np.ndarray] = {}
    for name in model_names:
        pipe = build_decoder(name)
        pipe.fit(X_train, y_train_enc)
        p = pipe.predict_proba(X_test)
        fitted_classes = getattr(pipe, "classes_", classes)
        aligned = np.zeros((X_test.shape[0], len(classes)), dtype=float)
        for j, cls in enumerate(fitted_classes):
            aligned[:, int(cls)] = p[:, j]
        probas[name.upper()] = aligned

    return HeldoutPrediction(subject=subject, y_true=y_test_enc, classes=classes, proba=probas)


def equal_soft_vote(probas: Mapping[str, np.ndarray]) -> np.ndarray:
    if not probas:
        raise ValueError("probas must contain at least one model")
    stacked = np.stack([np.asarray(p, dtype=float) for p in probas.values()], axis=0)
    vote = stacked.mean(axis=0)
    denom = vote.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return vote / denom


def accuracy(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true)
    pred = np.asarray(proba).argmax(axis=1)
    if mask is None:
        mask = np.ones(y_true.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(pred[mask] == y_true[mask]))
