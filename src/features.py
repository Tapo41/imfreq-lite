"""
src/features.py
---------------
FFT + Statistical feature extraction for ImFREQ-Lite.

Usage:
    from src.features import extract_features, window_and_extract
    phi = extract_features(window, K=10)
    Phi, Y_w = window_and_extract(X, Y, W=512, K=10, theta=0.50)

    # For LSTM Autoencoder baseline (raw 3D windows):
    from src.features import build_raw_windows
    X_win, Y_win = build_raw_windows(X, Y, W=512, theta=0.50)

CHANGES vs. original:
    - Added build_raw_windows() helper that returns raw [n_windows, W, C]
      tensors needed by LSTMAutoencoderBaseline.fit_raw().
      The original code had no such helper, forcing the LSTM baseline to
      fall back to reshaping the flat feature matrix (incorrect).
"""

import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis


# ─────────────────────────────────────────────────────────────────────────────
# FFT Features
# ─────────────────────────────────────────────────────────────────────────────

def fft_features(window: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Extract top-K FFT magnitude bins from a single window.

    Parameters
    ----------
    window : np.ndarray, shape [W, C]
        Sensor readings for one window (W samples, C channels).
    K : int
        Number of top spectral magnitude bins to retain per channel.
        DC component (bin 0) is excluded.

    Returns
    -------
    np.ndarray, shape [K * C]
        Flattened top-K spectral magnitudes across all channels.

    Notes
    -----
    Only the one-sided non-redundant half-spectrum is considered:
        bins 1 to W//2  (DC bin 0 excluded — captured by statistical mean).
    Top-K selection is column-wise (per channel independently).
    """
    W, C = window.shape

    # Full FFT → magnitude spectrum (one-sided, excluding DC)
    # Shape: [W//2, C]
    mag = np.abs(fft(window, axis=0))[1 : W // 2 + 1, :]

    # Select top-K bins per channel by magnitude
    top_k_idx = np.argsort(mag, axis=0)[-K:, :]         # [K, C]
    top_k_mag = np.take_along_axis(mag, top_k_idx, axis=0)  # [K, C]

    return top_k_mag.flatten()                            # [K*C]


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Features
# ─────────────────────────────────────────────────────────────────────────────

def statistical_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 5 statistical descriptors per channel from a single window.

    Descriptors: mean (μ), std (σ), skewness (γ₁), excess kurtosis (γ₂), RMS.

    Parameters
    ----------
    window : np.ndarray, shape [W, C]

    Returns
    -------
    np.ndarray, shape [5 * C]
        Flattened statistical features across all channels.
    """
    W, C = window.shape
    feats = []
    for c in range(C):
        col = window[:, c]
        feats.extend([
            np.mean(col),                          # μ
            np.std(col, ddof=1) if W > 1 else 0,  # σ
            float(skew(col)),                      # γ₁  (Fisher)
            float(kurtosis(col, fisher=True)),     # γ₂  (excess)
            np.sqrt(np.mean(col ** 2)),            # RMS
        ])
    return np.array(feats, dtype=np.float32)       # [5*C]


# ─────────────────────────────────────────────────────────────────────────────
# Combined Feature Vector
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(window: np.ndarray, K: int = 10) -> np.ndarray:
    """
    Full ImFREQ-Lite feature vector for one window.

    phi = [FFT_top-K || statistical]  ∈ ℝ^{(K+5)*C}

    For C=3, K=10: phi ∈ ℝ^45.

    Parameters
    ----------
    window : np.ndarray, shape [W, C]
    K : int
        Number of top FFT bins per channel (default 10).

    Returns
    -------
    np.ndarray, shape [(K+5)*C]
    """
    fft_f  = fft_features(window, K=K)           # [K*C]
    stat_f = statistical_features(window)         # [5*C]
    return np.concatenate([fft_f, stat_f])        # [(K+5)*C]


# ─────────────────────────────────────────────────────────────────────────────
# Batch Windowing + Feature Extraction  (flat feature matrix)
# ─────────────────────────────────────────────────────────────────────────────

def window_and_extract(
    X: np.ndarray,
    Y: np.ndarray,
    W: int   = 512,
    K: int   = 10,
    theta: float = 0.50,
    verbose: bool = True,
) -> tuple:
    """
    Segment sensor stream into windows and extract ImFREQ-Lite features.

    Parameters
    ----------
    X : np.ndarray, shape [N, C]
        Raw multivariate sensor stream (N samples, C channels).
    Y : np.ndarray, shape [N]
        Point-level binary anomaly labels.
    W : int
        Window size in samples (default 512).
    K : int
        Number of top FFT bins per channel (default 10).
    theta : float
        Majority-vote window labeling threshold (default 0.50).
        Window labeled anomalous if fraction of anomalous samples > theta.
    verbose : bool
        Print progress summary if True.

    Returns
    -------
    Phi : np.ndarray, shape [n_windows, (K+5)*C]
        Flat feature matrix (used by RF, XGBoost, and all non-LSTM baselines).
    Y_w : np.ndarray, shape [n_windows]
        Window-level binary labels.
    """
    N, C = X.shape
    n_windows = N // W

    feature_dim = (K + 5) * C
    Phi = np.zeros((n_windows, feature_dim), dtype=np.float32)
    Y_w = np.zeros(n_windows, dtype=np.int32)

    for k in range(n_windows):
        x_k = X[k * W : (k + 1) * W, :]           # [W, C]
        y_k = Y[k * W : (k + 1) * W]

        # Majority-vote window label
        Y_w[k] = 1 if np.mean(y_k) > theta else 0

        # Feature extraction
        Phi[k] = extract_features(x_k, K=K)

    if verbose:
        n_anom = int(Y_w.sum())
        print(f"[windowing] W={W}, K={K}, theta={theta}")
        print(f"  Windows : {n_windows}  |  Anomalous : {n_anom} "
              f"({100*n_anom/n_windows:.1f}%)")
        print(f"  Feature dim : {feature_dim}")

    return Phi, Y_w


# ─────────────────────────────────────────────────────────────────────────────
# Raw Window Builder  (for LSTM Autoencoder baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_raw_windows(
    X: np.ndarray,
    Y: np.ndarray,
    W: int   = 512,
    theta: float = 0.50,
    verbose: bool = True,
) -> tuple:
    """
    Segment sensor stream into raw windows without feature extraction.

    Required by LSTMAutoencoderBaseline which needs the actual time-series
    sequences [N, W, C] rather than the flat feature matrix.

    The paper trains the LSTM autoencoder on raw sensor windows (Section IV-C
    and Table IX), not on the pre-extracted FFT+statistical features used by
    the tree-based models.

    Parameters
    ----------
    X : np.ndarray, shape [N, C]
        Raw multivariate sensor stream.
    Y : np.ndarray, shape [N]
        Point-level binary anomaly labels.
    W : int
        Window size in samples (default 512, matching paper).
    theta : float
        Majority-vote window labeling threshold (default 0.50).
    verbose : bool
        Print summary if True.

    Returns
    -------
    X_win : np.ndarray, shape [n_windows, W, C]
        Raw windowed sensor data — 3-D array ready for LSTM input.
    Y_win : np.ndarray, shape [n_windows]
        Window-level binary labels (same labeling rule as window_and_extract).

    Example
    -------
    >>> X_win, Y_win = build_raw_windows(X_train, Y_train, W=512, theta=0.50)
    >>> lstm = LSTMAutoencoderBaseline()
    >>> lstm.fit_raw(X_win, Y_win)
    >>> X_win_test, Y_win_test = build_raw_windows(X_test, Y_test)
    >>> metrics = lstm.evaluate_raw(X_win_test, Y_win_test)
    """
    N, C = X.shape
    n_windows = N // W

    X_win = np.zeros((n_windows, W, C), dtype=np.float32)
    Y_win = np.zeros(n_windows, dtype=np.int32)

    for k in range(n_windows):
        x_k = X[k * W : (k + 1) * W, :]   # [W, C]
        y_k = Y[k * W : (k + 1) * W]

        X_win[k] = x_k.astype(np.float32)
        Y_win[k] = 1 if np.mean(y_k) > theta else 0

    if verbose:
        n_anom = int(Y_win.sum())
        print(f"[build_raw_windows] W={W}, theta={theta}")
        print(f"  Windows : {n_windows}  |  Anomalous : {n_anom} "
              f"({100*n_anom/n_windows:.1f}%)")
        print(f"  Output shape : {X_win.shape}")

    return X_win, Y_win