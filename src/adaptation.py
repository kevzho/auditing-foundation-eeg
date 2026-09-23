"""Label-free session adaptation for held-out motor-imagery evaluation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import fractional_matrix_power


def _regularized_epoch_covariances(X: np.ndarray, eps: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected epochs shaped (trials, channels, samples), got {X.shape}")

    centered = X - X.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centered, centered)
    denom = max(X.shape[2] - 1, 1)
    covs /= float(denom)

    eye = np.eye(X.shape[1], dtype=np.float64)
    trace_scale = np.trace(covs, axis1=1, axis2=2) / float(X.shape[1])
    covs += eps * trace_scale[:, None, None] * eye[None, :, :]
    return covs


def euclidean_alignment_matrix(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return the inverse square root of the session mean covariance."""

    covs = _regularized_epoch_covariances(X, eps=eps)
    reference = covs.mean(axis=0)
    reference = 0.5 * (reference + reference.T)
    aligner = fractional_matrix_power(reference, -0.5)
    aligner = np.asarray(np.real_if_close(aligner), dtype=np.float64)
    return 0.5 * (aligner + aligner.T)


def apply_channel_matrix(X: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a channels-by-channels linear transform to every epoch."""

    X = np.asarray(X, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected epochs shaped (trials, channels, samples), got {X.shape}")
    if matrix.shape != (X.shape[1], X.shape[1]):
        raise ValueError(f"Matrix shape {matrix.shape} does not match {X.shape[1]} channels")
    return np.einsum("cd,ndt->nct", matrix, X).astype(np.float32, copy=False)


def adapt_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    method: str = "euclidean",
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Adapt T and E sessions without using labels.

    Euclidean alignment whitens each session by the inverse square root of its
    own mean covariance. For the E session this uses only unlabeled epochs.
    """

    method = method.lower()
    if method in {"none", "off"}:
        return X_train, X_test
    if method not in {"euclidean", "euclidean_alignment"}:
        raise ValueError(f"Unknown session adaptation method: {method}")

    train_aligner = euclidean_alignment_matrix(X_train, eps=eps)
    test_aligner = euclidean_alignment_matrix(X_test, eps=eps)
    return apply_channel_matrix(X_train, train_aligner), apply_channel_matrix(X_test, test_aligner)
