from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


def _as_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_trials, n_channels, n_times); got {X.shape}")
    return X


def _parse_bands(bands: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for lo, hi in bands:
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f <= 0 or hi_f <= 0 or hi_f <= lo_f:
            raise ValueError(f"Invalid band {(lo, hi)}; expected 0 < lo < hi")
        out.append((lo_f, hi_f))
    if not out:
        raise ValueError("bands must not be empty")
    return out


def _bandpass_sos(sfreq: float, l_freq: float, h_freq: float, order: int = 5):
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    if not (0 < low < 1) or not (0 < high < 1) or not (low < high):
        raise ValueError(f"Band {(l_freq, h_freq)} invalid for sfreq={sfreq}")
    return signal.butter(order, [low, high], btype="bandpass", output="sos")


def _apply_sosfiltfilt(X: np.ndarray, sos) -> np.ndarray:
    # Filter along time axis (last axis)
    return signal.sosfiltfilt(sos, X, axis=-1)


def _logvar(X: np.ndarray, axis: int = -1) -> np.ndarray:
    v = np.var(X, axis=axis, ddof=0)
    return np.log(v + 1e-12)


def _get_csp_class():
    fake_home = Path(__file__).resolve().parent / ".mne_home"
    (fake_home / ".mne").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(fake_home))
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
    try:
        from mne.decoding import CSP  # type: ignore

        return CSP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MNE is required for reported FBCSP/CSP runs. Install the pinned "
            "requirements instead of using the removed local CSP fallback."
        ) from exc


@dataclass
class FBCSPFeatures(BaseEstimator, TransformerMixin):
    """
    Filter Bank CSP feature extractor (sklearn Transformer).

    Input: X with shape (n_trials, n_channels, n_times)
    Output: 2D feature matrix (n_trials, n_features)

    For each band:
      - bandpass filter
      - CSP -> projected signals
      - log-variance features over time for each CSP component

    Optionally concatenates bandpower features:
      - per-channel log-variance after bandpass
    """

    sfreq: float = 250.0
    bands: Sequence[Tuple[float, float]] = ((8, 12), (12, 16), (16, 20), (20, 30))
    n_components: int = 4
    include_bandpower: bool = False
    filter_order: int = 5

    _bands: Optional[List[Tuple[float, float]]] = None
    _sos: Optional[List[object]] = None
    _csp_per_band: Optional[List[object]] = None
    _n_channels: Optional[int] = None

    def _make_csp(self):
        """
        Create a CSP instance with numerically-stable defaults.

        Use MNE's CSP with covariance regularization to avoid NaNs/Infs from
        ill-conditioned covariance estimates common in EEG.
        """
        CSP = _get_csp_class()
        return CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = _as_3d(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D and match X.shape[0]")

        self._bands = _parse_bands(self.bands)
        self._sos = [_bandpass_sos(self.sfreq, lo, hi, order=self.filter_order) for lo, hi in self._bands]
        self._n_channels = int(X.shape[1])

        self._csp_per_band = []
        for _ in self._bands:
            self._csp_per_band.append(self._make_csp())

        for i, sos in enumerate(self._sos):
            X_f = _apply_sosfiltfilt(X, sos)
            if not np.isfinite(X_f).all():
                # Filtering should not introduce NaNs, but we guard to keep long runs from crashing.
                X_f = np.nan_to_num(X_f, nan=0.0, posinf=0.0, neginf=0.0)
            self._csp_per_band[i].fit(X_f, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._bands is None or self._sos is None or self._csp_per_band is None or self._n_channels is None:
            raise RuntimeError("FBCSPFeatures must be fit before transform")

        X = _as_3d(X)
        if int(X.shape[1]) != self._n_channels:
            raise ValueError(f"Expected n_channels={self._n_channels}; got {X.shape[1]}")

        feats = []
        for i, sos in enumerate(self._sos):
            X_f = _apply_sosfiltfilt(X, sos)
            # CSP features
            # mne.CSP.transform returns log-variance if log=True, shape (n_trials, n_components)
            csp_feat = self._csp_per_band[i].transform(X_f)
            feats.append(np.asarray(csp_feat, dtype=float))

            if self.include_bandpower:
                # per-channel bandpower (log-variance) features
                bp = _logvar(X_f, axis=-1)  # (n_trials, n_channels)
                feats.append(np.asarray(bp, dtype=float))

        return np.concatenate(feats, axis=1)
