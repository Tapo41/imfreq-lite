"""
src/utils.py
------------
Data loading, preprocessing, reproducibility helpers, and the 10-seed
multi-run experiment loop for ImFREQ-Lite.

CHANGES vs. original:
    - Added multi_run_experiment() — runs the full pipeline (or any
      baseline) across all 10 paper seeds and returns aggregated metrics.
      The original code had no utility for this, making it impossible to
      reproduce the "mean ± std over 10 runs" numbers from the paper
      without writing a custom loop in the notebook.
    - Added run_all_baselines() — convenience wrapper for running all
      baselines in the same 10-seed protocol.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.evaluate import aggregate_runs, print_summary, paired_ttest


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

PAPER_SEEDS = [42, 7, 13, 99, 256, 101, 200, 300, 400, 500]


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Run Experiment Loop  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def multi_run_experiment(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test:  np.ndarray,
    Y_test:  np.ndarray,
    model_cls=None,
    model_kwargs: dict = None,
    seeds: list = None,
    verbose: bool = True,
) -> dict:
    """
    Run ImFREQLite (or any compatible model class) across multiple seeds and
    return aggregated metrics — reproducing the paper's "mean ± std, 10 runs"
    protocol (Section IV-D).

    Parameters
    ----------
    X_train, Y_train : np.ndarray
        Training sensor stream (raw, point-level labels).
    X_test, Y_test : np.ndarray
        Test sensor stream (raw, point-level labels).
    model_cls : class, optional
        Model class with .fit(X, Y) and .evaluate(X, Y) → dict interface.
        Defaults to ImFREQLite.
    model_kwargs : dict, optional
        Constructor keyword arguments passed to model_cls on each run.
        `random_state` is overridden per seed — do not include it here.
    seeds : list of int, optional
        Random seeds to use. Defaults to PAPER_SEEDS (10 seeds).
    verbose : bool
        Print per-run progress and final summary.

    Returns
    -------
    aggregated : dict
        Output of evaluate.aggregate_runs() — maps metric name to
        {"mean": float, "std": float, "values": list}.

    Example
    -------
    >>> from src.utils import multi_run_experiment, PAPER_SEEDS
    >>> from src.pipeline import ImFREQLite
    >>> agg = multi_run_experiment(
    ...     X_train, Y_train, X_test, Y_test,
    ...     model_cls=ImFREQLite,
    ...     model_kwargs=dict(K=10, W=512, theta=0.50, smote_ratio=0.25),
    ...     seeds=PAPER_SEEDS,
    ... )
    """
    # Lazy import to avoid circular dependency
    if model_cls is None:
        from src.pipeline import ImFREQLite
        model_cls = ImFREQLite

    if seeds is None:
        seeds = PAPER_SEEDS

    if model_kwargs is None:
        model_kwargs = {}

    run_metrics = []
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Run {i+1}/{len(seeds)}  |  seed={seed}")
            print(f"{'='*55}")

        set_seed(seed)
        model = model_cls(random_state=seed, **model_kwargs)
        model.fit(X_train, Y_train)
        metrics = model.evaluate(X_test, Y_test)
        run_metrics.append(metrics)

        if verbose:
            print(f"  F1={metrics.get('F1', float('nan')):.4f} | "
                  f"PR-AUC={metrics.get('PR_AUC', float('nan')):.4f} | "
                  f"Train={metrics.get('train_time_s', '?')}s")

    aggregated = aggregate_runs(run_metrics)

    if verbose:
        print_summary(aggregated, title=f"{model_cls.__name__} — {len(seeds)}-Run Summary")

    return aggregated


def run_all_baselines(
    Phi_train: np.ndarray,
    Y_train:   np.ndarray,
    Phi_test:  np.ndarray,
    Y_test:    np.ndarray,
    seeds: list = None,
    verbose: bool = True,
    lstm_raw_train: np.ndarray = None,
    lstm_raw_test:  np.ndarray = None,
) -> dict:
    """
    Run all 7 paper baselines across multiple seeds and return aggregated
    metrics for each.

    Baselines operate on the pre-extracted flat feature matrix Phi (shape
    [n_windows, (K+5)*C]) rather than the raw sensor stream.  Extract
    features once with window_and_extract() and pass the result here.

    The LSTM Autoencoder is a special case: it requires raw [N, W, C] windows.
    Pass these via lstm_raw_train / lstm_raw_test.  If not supplied, the
    LSTM baseline falls back to the flat-feature shim (with a warning).

    Parameters
    ----------
    Phi_train, Y_train : feature matrix + window labels (train).
    Phi_test, Y_test   : feature matrix + window labels (test).
    seeds : list of int, optional. Defaults to PAPER_SEEDS.
    verbose : bool
    lstm_raw_train : np.ndarray, shape [n_windows, W, C], optional
        Raw windows for LSTM training. Build with build_raw_windows().
    lstm_raw_test  : np.ndarray, shape [n_windows, W, C], optional
        Raw windows for LSTM evaluation.

    Returns
    -------
    all_results : dict
        Maps baseline name → aggregated metrics dict.

    Example
    -------
    >>> from src.features import window_and_extract, build_raw_windows
    >>> Phi_tr, Y_tr = window_and_extract(X_train, Y_train)
    >>> Phi_te, Y_te = window_and_extract(X_test, Y_test)
    >>> X_win_tr, _ = build_raw_windows(X_train, Y_train)
    >>> X_win_te, _ = build_raw_windows(X_test, Y_test)
    >>> results = run_all_baselines(
    ...     Phi_tr, Y_tr, Phi_te, Y_te,
    ...     lstm_raw_train=X_win_tr, lstm_raw_test=X_win_te,
    ... )
    """
    from src.baselines import get_all_baselines

    if seeds is None:
        seeds = PAPER_SEEDS

    all_results = {}

    for name, baseline in get_all_baselines().items():
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  Baseline: {name}")
            print(f"{'#'*60}")

        run_metrics = []
        for i, seed in enumerate(seeds):
            if verbose:
                print(f"\n  -- Run {i+1}/{len(seeds)}  seed={seed} --")

            set_seed(seed)

            # Re-instantiate with this seed to reset internal state
            bl = get_all_baselines(random_state=seed)[name]

            # LSTM: use raw windows if provided, else fall back to flat features
            is_lstm = "LSTM" in name
            if is_lstm and lstm_raw_train is not None:
                bl.fit_raw(lstm_raw_train, Y_train)
                metrics = bl.evaluate_raw(lstm_raw_test, Y_test)
            else:
                bl.fit(Phi_train, Y_train)
                metrics = bl.evaluate(Phi_test, Y_test)

            run_metrics.append(metrics)

        agg = aggregate_runs(run_metrics)
        all_results[name] = agg

        if verbose:
            print_summary(agg, title=name)

    return all_results


def significance_vs_imfreq(
    all_results: dict,
    imfreq_results: dict,
    metric: str = "F1",
    alpha: float = 0.05,
) -> None:
    """
    Print a significance table comparing all baselines against ImFREQ-Lite.

    Parameters
    ----------
    all_results    : output of run_all_baselines()
    imfreq_results : output of multi_run_experiment() for ImFREQLite
    metric         : which metric to compare (default "F1")
    alpha          : significance level (default 0.05)
    """
    ref_vals = imfreq_results[metric]["values"]
    ref_mean = imfreq_results[metric]["mean"]

    print(f"\n{'Method':<35} {'Mean':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 65)
    print(f"  {'ImFREQ-Lite (reference)':<33} {ref_mean:>8.4f} "
          f"{'---':>10} {'---':>6}  ← reference")

    for name, agg in all_results.items():
        if metric not in agg:
            continue
        vals = agg[metric]["values"]
        mean = agg[metric]["mean"]

        result = paired_ttest(
            ref_vals, vals,
            label_a="ImFREQ-Lite", label_b=name,
            alpha=alpha,
        )
        sig = "✓" if result["significant"] else "✗"
        print(f"  {name:<33} {mean:>8.4f} {result['p_value']:>10.4f} {sig:>6}")
    print()


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
    Load ToN-IoT sensor sub-stream dataset.

    Paper uses only the sensor columns: temp, humidity, motion_detected.
    Network traffic features are excluded.

    Parameters
    ----------
    filepath : str
        Path to CSV file. Download from:
        https://research.unsw.edu.au/projects/toniot-datasets
    sensor_cols : list or None
        Column names to use as features. If None, uses defaults.
    label_col : str
        Column name for binary anomaly label.
    n_samples : int
        Subsample size (paper uses 48,623).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    X : np.ndarray, shape [n_samples, C]
    Y : np.ndarray, shape [n_samples]
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
        mask = np.isnan(X[:, c])
        X[mask, c] = col_means[c]

    print(f"[ToN-IoT] Loaded {len(X)} samples | "
          f"Anomaly rate: {Y.mean()*100:.1f}% | "
          f"Channels: {X.shape[1]}")
    return X, Y


def load_skab(
    data_dir: str = "data/skab/",
    cols: list = None,
    label_col: str = "anomaly",
) -> tuple:
    """
    Load SKAB (Skoltech Anomaly Benchmark) dataset.

    Download from: https://www.kaggle.com/datasets/dsv/1693952

    Paper uses: accelerometer_x, accelerometer_y, pressure columns.

    Parameters
    ----------
    data_dir : str
        Directory containing SKAB CSV files (all_data/ folder).
    cols : list or None
        Feature column names. If None, uses paper defaults.
    label_col : str
        Binary anomaly label column.

    Returns
    -------
    X : np.ndarray, shape [N, C]
    Y : np.ndarray, shape [N]
    """
    if cols is None:
        cols = ["accelerometer_x", "accelerometer_y", "pressure"]

    all_X, all_Y = [], []

    if os.path.isdir(data_dir):
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
                        pad = np.zeros(
                            (len(x), len(cols) - len(available)), dtype=np.float32
                        )
                        x = np.concatenate([x, pad], axis=1)
                    y = (df[label_col].values > 0).astype(np.int32) \
                        if label_col in df.columns \
                        else np.zeros(len(x), dtype=np.int32)
                    all_X.append(x)
                    all_Y.append(y)
                except Exception as e:
                    print(f"  Skip {fname}: {e}")
    else:
        raise FileNotFoundError(
            f"SKAB data directory not found: {data_dir}\n"
            "Download from: https://www.kaggle.com/datasets/dsv/1693952"
        )

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    print(f"[SKAB] Loaded {len(X)} samples | "
          f"Anomaly rate: {Y.mean()*100:.1f}% | "
          f"Channels: {X.shape[1]}")
    return X, Y


def load_nab_yahoo_s5(
    data_dir: str = "data/nab_yahoo_s5/",
    replicate_channels: int = 3,
) -> tuple:
    """
    Load NAB Yahoo S5 anomaly benchmark.

    Download from:
        https://github.com/numenta/NAB/tree/master/data/realYahoo

    Concatenates all 49 series. For C=3 compatibility with multi-channel
    pipeline, the single univariate channel is replicated 3 times.

    Parameters
    ----------
    data_dir : str
        Directory containing Yahoo S5 CSV files.
    replicate_channels : int
        Number of times to replicate the univariate series (default 3).

    Returns
    -------
    X : np.ndarray, shape [N, replicate_channels]
    Y : np.ndarray, shape [N]
    """
    all_X, all_Y = [], []

    if os.path.isdir(data_dir):
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".csv"):
                fpath = os.path.join(data_dir, fname)
                try:
                    df = pd.read_csv(fpath)
                    val_col   = [c for c in df.columns if "value" in c.lower()]
                    label_col = [c for c in df.columns
                                 if "anomaly" in c.lower() or "label" in c.lower()]
                    if val_col and label_col:
                        x = df[val_col[0]].values.astype(np.float32).reshape(-1, 1)
                        y = df[label_col[0]].values.astype(np.int32)
                        all_X.append(x)
                        all_Y.append(y)
                except Exception as e:
                    print(f"  Skip {fname}: {e}")
    else:
        raise FileNotFoundError(
            f"NAB Yahoo S5 directory not found: {data_dir}\n"
            "Download from: https://github.com/numenta/NAB"
        )

    X_uni = np.concatenate(all_X, axis=0)   # [N, 1]
    Y     = np.concatenate(all_Y, axis=0)

    X = np.concatenate([X_uni] * replicate_channels, axis=1)  # [N, C]

    print(f"[NAB Yahoo S5] Loaded {len(X)} samples | "
          f"Anomaly rate: {Y.mean()*100:.1f}% | "
          f"Channels: {X.shape[1]} (replicated from 1)")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Dataset Generator (for quick testing without downloading)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_iot(
    n_samples: int   = 20000,
    n_channels: int  = 3,
    anomaly_rate: float = 0.04,
    seed: int        = 42,
    fs: float        = 1.0,
) -> tuple:
    """
    Generate a synthetic multivariate IoT sensor stream for testing.

    Normal signal = sum of sinusoids + Gaussian noise.
    Anomalies = random amplitude spikes and frequency disruptions.

    Parameters
    ----------
    n_samples : int
    n_channels : int
    anomaly_rate : float
    seed : int
    fs : float  Sampling frequency (Hz).

    Returns
    -------
    X : np.ndarray, shape [n_samples, n_channels]
    Y : np.ndarray, shape [n_samples]
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(n_samples) / fs

    X = np.zeros((n_samples, n_channels), dtype=np.float32)
    for c in range(n_channels):
        freqs = rng.uniform(0.01, 0.1, size=3)
        amps  = rng.uniform(0.5, 2.0, size=3)
        sig   = sum(a * np.sin(2 * np.pi * f * t) for a, f in zip(amps, freqs))
        sig  += rng.normal(0, 0.2, size=n_samples)
        X[:, c] = sig.astype(np.float32)

    Y = np.zeros(n_samples, dtype=np.int32)
    n_anom   = int(n_samples * anomaly_rate)
    anom_idx = rng.choice(n_samples, size=n_anom, replace=False)

    for idx in anom_idx:
        end = min(idx + rng.integers(5, 30), n_samples)
        Y[idx:end] = 1
        X[idx:end, :] += rng.uniform(3, 8, size=(end - idx, n_channels))

    print(f"[synthetic] {n_samples} samples | "
          f"Anomaly rate: {Y.mean()*100:.1f}% | "
          f"Channels: {n_channels}")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def normalize(
    X_train: np.ndarray,
    X_test:  np.ndarray,
) -> tuple:
    """
    StandardScaler normalization fit on training data, applied to both.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s.astype(np.float32), X_test_s.astype(np.float32), scaler


def temporal_train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.80,
) -> tuple:
    """
    Temporal (non-shuffled) split preserving time order.
    Use for streaming sensor data to avoid look-ahead bias.
    """
    N = len(X)
    split = int(N * train_ratio)
    return X[:split], X[split:], Y[:split], Y[split:]


def stratified_split(
    Phi: np.ndarray,
    Y_w: np.ndarray,
    train_ratio: float = 0.80,
    seed: int = 42,
) -> tuple:
    """
    Stratified split on window-level features (preserves class ratio).
    Use after feature extraction (not on raw time series).
    """
    return train_test_split(
        Phi, Y_w,
        train_size=train_ratio,
        stratify=Y_w,
        random_state=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Results Table Formatter
# ─────────────────────────────────────────────────────────────────────────────

def format_results_table(results: dict) -> str:
    """
    Format a results dict into a clean text table for logging/printing.

    Parameters
    ----------
    results : dict mapping method_name → aggregated metrics dict
              (output of evaluate.aggregate_runs())

    Returns
    -------
    str: formatted table
    """
    header = (
        f"\n{'Method':<35} {'F1':>9} {'PR-AUC':>9} "
        f"{'Recall':>9} {'Prec.':>9} {'Train(s)':>10}\n"
        + "-" * 85
    )
    rows = [header]
    for name, agg in results.items():
        def fmt(k):
            if k not in agg:
                return "  N/A   "
            m, s = agg[k]["mean"], agg[k]["std"]
            return f"{m:.3f}±{s:.3f}"

        row = (
            f"  {name:<33} "
            f"{fmt('F1'):>12} "
            f"{fmt('PR_AUC'):>12} "
            f"{fmt('Recall'):>12} "
            f"{fmt('Precision'):>12} "
        )
        if "train_time_s" in agg:
            m, s = agg["train_time_s"]["mean"], agg["train_time_s"]["std"]
            row += f"  {m:.1f}±{s:.1f}"
        rows.append(row)
    return "\n".join(rows) + "\n"