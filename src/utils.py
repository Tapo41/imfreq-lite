"""
src/utils.py
------------
Data loading, preprocessing, and reproducibility helpers for ImFREQ-Lite.

v3 Changes
----------
- load_nab_yahoo_s5(): channel replication REMOVED. NAB is now evaluated in
  its native univariate mode (C=1, φ ∈ ℝ¹⁵). This eliminates the artificial
  multivariate replication that was flagged as a methodological weakness.
- set_seed(): now also seeds PyTorch (for TCN / Transformer baselines).
- format_results_table(): wider columns to accommodate new baseline names.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

PAPER_SEEDS = [42, 7, 13, 99, 256, 101, 200, 300, 400, 500]


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility (numpy, Python, TF, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_toniot(
    filepath: str = "data/ton_iot_sensor.csv",
    sensor_cols: list = None,
    label_col: str = "label",
    n_samples: int = 48623,
    seed: int = 42,
) -> tuple:
    """
    Load ToN-IoT sensor sub-stream dataset (C=3).

    Paper uses only: temp, humidity, motion_detected.
    Network traffic features are excluded.

    Download: https://research.unsw.edu.au/projects/toniot-datasets

    Returns
    -------
    X : np.ndarray, shape [N, 3]
    Y : np.ndarray, shape [N]
    """
    if sensor_cols is None:
        sensor_cols = ["temp", "humidity", "motion_detected"]

    df = pd.read_csv(filepath)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    X = df[sensor_cols].values.astype(np.float32)
    Y = (df[label_col].values > 0).astype(np.int32)

    col_means = np.nanmean(X, axis=0)
    for c in range(X.shape[1]):
        X[np.isnan(X[:, c]), c] = col_means[c]

    print(f"[ToN-IoT] {len(X)} samples | "
          f"Anomaly: {Y.mean()*100:.1f}% | C={X.shape[1]}")
    return X, Y


def load_skab(
    data_dir: str = "data/skab/",
    cols: list = None,
    label_col: str = "anomaly",
) -> tuple:
    """
    Load SKAB (Skoltech Anomaly Benchmark) — natively multivariate (C=3).

    Columns: accelerometer_x, accelerometer_y, pressure.
    Download: https://www.kaggle.com/datasets/dsv/1693952

    Returns
    -------
    X : np.ndarray, shape [N, 3]
    Y : np.ndarray, shape [N]
    """
    if cols is None:
        cols = ["accelerometer_x", "accelerometer_y", "pressure"]

    all_X, all_Y = [], []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"SKAB directory not found: {data_dir}\n"
            "Download from: https://www.kaggle.com/datasets/dsv/1693952"
        )

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath, sep=";", index_col=0)
                available = [c for c in cols if c in df.columns]
                if not available:
                    continue
                x = df[available].values.astype(np.float32)
                if len(available) < len(cols):
                    pad = np.zeros((len(x), len(cols) - len(available)), dtype=np.float32)
                    x = np.concatenate([x, pad], axis=1)
                y = (df[label_col].values > 0).astype(np.int32) \
                    if label_col in df.columns \
                    else np.zeros(len(x), dtype=np.int32)
                all_X.append(x)
                all_Y.append(y)
            except Exception as e:
                print(f"  Skip {fname}: {e}")

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    print(f"[SKAB] {len(X)} samples | Anomaly: {Y.mean()*100:.1f}% | C={X.shape[1]}")
    return X, Y


def load_nab_yahoo_s5(
    data_dir: str = "data/nab_yahoo_s5/",
) -> tuple:
    """
    Load NAB Yahoo S5 anomaly benchmark in NATIVE UNIVARIATE MODE (C=1).

    v3 Change (breaking): Channel replication (×3) has been REMOVED.
    NAB Yahoo S5 is a genuinely univariate benchmark. Replicating channels
    artificially inflates feature dimensionality and introduces false inter-
    channel correlation, which was identified as a methodological flaw in the
    v2 paper. The pipeline generalises cleanly to C=1, producing φ ∈ ℝ¹⁵
    (K=10 FFT bins + 5 statistical features × 1 channel).

    Download: https://github.com/numenta/NAB/tree/master/data/realYahoo

    Returns
    -------
    X : np.ndarray, shape [N, 1]   ← single channel, no replication
    Y : np.ndarray, shape [N]
    """
    all_X, all_Y = [], []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"NAB Yahoo S5 directory not found: {data_dir}\n"
            "Download from: https://github.com/numenta/NAB"
        )

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath)
                val_col   = [c for c in df.columns if "value"  in c.lower()]
                label_col = [c for c in df.columns
                             if "anomaly" in c.lower() or "label" in c.lower()]
                if val_col and label_col:
                    x = df[val_col[0]].values.astype(np.float32).reshape(-1, 1)
                    y = df[label_col[0]].values.astype(np.int32)
                    all_X.append(x)
                    all_Y.append(y)
            except Exception as e:
                print(f"  Skip {fname}: {e}")

    X = np.concatenate(all_X, axis=0)   # [N, 1]
    Y = np.concatenate(all_Y, axis=0)

    print(f"[NAB Yahoo S5] {len(X)} samples | "
          f"Anomaly: {Y.mean()*100:.1f}% | C=1 (native univariate)")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_iot(
    n_samples: int = 20000,
    n_channels: int = 3,
    anomaly_rate: float = 0.04,
    seed: int = 42,
    fs: float = 1.0,
) -> tuple:
    """
    Generate a synthetic multivariate IoT sensor stream for unit testing.

    Normal: sum of sinusoids + Gaussian noise.
    Anomalies: random amplitude spikes.

    Set n_channels=1 to simulate univariate NAB-style testing.

    Returns
    -------
    X : np.ndarray, shape [n_samples, n_channels]
    Y : np.ndarray, shape [n_samples]
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples) / fs
    X   = np.zeros((n_samples, n_channels), dtype=np.float32)

    for c in range(n_channels):
        freqs = rng.uniform(0.01, 0.1, size=3)
        amps  = rng.uniform(0.5, 2.0, size=3)
        sig   = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amps, freqs))
        X[:, c] = (sig + rng.normal(0, 0.2, size=n_samples)).astype(np.float32)

    Y       = np.zeros(n_samples, dtype=np.int32)
    anom_idx = rng.choice(n_samples, size=int(n_samples * anomaly_rate), replace=False)
    for idx in anom_idx:
        end = min(idx + rng.integers(5, 30), n_samples)
        Y[idx:end] = 1
        X[idx:end, :] += rng.uniform(3, 8, size=(end - idx, n_channels))

    print(f"[synthetic] {n_samples} samples | "
          f"Anomaly: {Y.mean()*100:.1f}% | C={n_channels}")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def normalize(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """StandardScaler fit on train, applied to both. Returns (Xtr, Xte, scaler)."""
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_test_s, scaler


def temporal_train_test_split(X, Y, train_ratio=0.80):
    """Non-shuffled temporal split. Preserves time order for streaming data."""
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], Y[:split], Y[split:]


def stratified_split(Phi, Y_w, train_ratio=0.80, seed=42):
    """Stratified split on window-level features (preserves class ratio)."""
    return train_test_split(Phi, Y_w, train_size=train_ratio,
                            stratify=Y_w, random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Results Table Formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_results_table(results: dict) -> str:
    """Format aggregated results dict into a paper-style text table."""
    header = (
        f"\n{'Method':<35} {'F1':>12} {'PR-AUC':>12} "
        f"{'Recall':>9} {'Prec.':>9} {'Train(s)':>10}\n" + "-" * 95
    )
    rows = [header]
    for name, agg in results.items():
        def fmt(k):
            if k not in agg:
                return "     N/A    "
            return f"{agg[k]['mean']:.3f}±{agg[k]['std']:.3f}"
        row = (f"  {name:<33} {fmt('F1'):>12} {fmt('PR_AUC'):>12} "
               f"{fmt('Recall'):>9} {fmt('Precision'):>9}")
        if "train_time_s" in agg:
            row += f"  {agg['train_time_s']['mean']:.1f}±{agg['train_time_s']['std']:.1f}"
        rows.append(row)
    return "\n".join(rows) + "\n"
