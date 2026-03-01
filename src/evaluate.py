"""
src/evaluate.py
---------------
Evaluation metrics, multi-run aggregation, and statistical testing.

v3 Changes
----------
- prequential_eval(): sliding-window streaming evaluation with ADWIN drift
  detection. Computes streaming F1, PR-AUC, and per-window pipeline latency.
- significance_table(): now handles 9 baselines.
- All existing functions unchanged.
"""

import time
import numpy as np
from scipy import stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core Metric Function
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> dict:
    """
    Compute all paper-reported evaluation metrics.

    Parameters
    ----------
    y_true : array-like [n]   True binary window labels.
    y_pred : array-like [n]   Predicted binary labels.
    y_prob : array-like [n]   Anomaly probabilities (for PR-AUC / ROC-AUC).

    Returns
    -------
    dict: F1, Precision, Recall, PR_AUC, ROC_AUC, Accuracy,
          n_test, n_anom, anomaly_rate
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "F1"           : f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision"    : precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall"       : recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Accuracy"     : float(np.mean(y_true == y_pred)),
        "n_test"       : int(len(y_true)),
        "n_anom"       : int(y_true.sum()),
        "anomaly_rate" : float(y_true.mean()),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["PR_AUC"]  = average_precision_score(y_true, y_prob, pos_label=1)
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["PR_AUC"]  = float("nan")
            metrics["ROC_AUC"] = float("nan")
    else:
        metrics["PR_AUC"]  = float("nan")
        metrics["ROC_AUC"] = float("nan")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Run Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_runs(run_metrics: list) -> dict:
    """
    Aggregate metrics from multiple independent runs.

    Parameters
    ----------
    run_metrics : list of dicts (each from compute_metrics())

    Returns
    -------
    dict mapping metric → {"mean": ..., "std": ..., "values": [...]}
    """
    keys = [k for k in run_metrics[0].keys()
            if isinstance(run_metrics[0][k], (int, float))]
    aggregated = {}
    for k in keys:
        vals = np.array([rm[k] for rm in run_metrics], dtype=float)
        aggregated[k] = {
            "mean"  : float(np.nanmean(vals)),
            "std"   : float(np.nanstd(vals, ddof=1)),
            "values": vals.tolist(),
        }
    return aggregated


def print_summary(aggregated: dict, title: str = "Results") -> None:
    """Print a clean summary table of aggregated multi-run results."""
    print(f"\n{'='*55}\n  {title}\n{'='*55}")
    for k in ["F1", "PR_AUC", "Recall", "Precision", "ROC_AUC"]:
        if k in aggregated:
            print(f"  {k:<12}  {aggregated[k]['mean']:.4f} ± {aggregated[k]['std']:.4f}")
    if "train_time_s" in aggregated:
        print(f"  {'Train (s)':<12}  "
              f"{aggregated['train_time_s']['mean']:.1f} ± "
              f"{aggregated['train_time_s']['std']:.1f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Significance Testing
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(scores_a, scores_b, alpha=0.05,
                 label_a="Method A", label_b="Method B") -> dict:
    """
    Two-tailed paired t-test between two methods over multiple runs.

    Returns
    -------
    dict: t_stat, p_value, significant, direction, mean_diff
    """
    a, b = np.array(scores_a, dtype=float), np.array(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Both score lists must have the same length.")
    if len(a) < 2:
        raise ValueError("Need at least 2 runs.")
    t_stat, p_value = stats.ttest_rel(a, b)
    result = {
        "t_stat"     : float(t_stat),
        "p_value"    : float(p_value),
        "significant": bool(p_value < alpha),
        "direction"  : f"{label_a} > {label_b}" if np.mean(a) > np.mean(b)
                       else f"{label_b} > {label_a}",
        "mean_diff"  : float(np.mean(a) - np.mean(b)),
    }
    sig = "SIGNIFICANT" if result["significant"] else "not significant"
    print(f"\n[t-test] {label_a} vs {label_b}")
    print(f"  mean({label_a})={np.mean(a):.4f}  mean({label_b})={np.mean(b):.4f}")
    print(f"  t={t_stat:.4f}, p={p_value:.4f} → {sig} at α={alpha}")
    return result


def significance_table(method_scores: dict, reference: str, alpha=0.05) -> None:
    """Print a significance table comparing all methods against a reference."""
    ref = np.array(method_scores[reference])
    print(f"\n{'Method':<35} {'Mean F1':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 65)
    for name, scores in method_scores.items():
        s = np.array(scores)
        if name == reference:
            print(f"  {name:<33} {np.mean(s):>8.4f} {'---':>10} {'---':>6}  ← reference")
            continue
        _, p = stats.ttest_rel(ref, s)
        print(f"  {name:<33} {np.mean(s):>8.4f} {p:>10.4f} {'✓' if p < alpha else '✗':>6}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Prequential (Test-Then-Train) Streaming Evaluation  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

def prequential_eval(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    W: int = 512,
    adwin_delta: float = 0.002,
    verbose: bool = True,
) -> dict:
    """
    Prequential (test-then-train) streaming evaluation with ADWIN drift detection.

    For each completed window the pipeline:
        1. Measures end-to-end pipeline latency (FFT + stat + predict).
        2. Emits a binary prediction.
        3. Computes cumulative streaming F1 and PR-AUC.
        4. Feeds the F1 residual to ADWIN to detect concept drift.

    The model is NOT retrained during streaming (offline-trained model, online
    evaluation). Retraining on drift events is future work.

    Parameters
    ----------
    model       : fitted ImFREQLite instance
    X           : np.ndarray [N, C] full sensor stream
    Y           : np.ndarray [N]    point-level labels
    W           : window size (must match model.W)
    adwin_delta : ADWIN sensitivity parameter (default 0.002)
    verbose     : print progress every 100 windows

    Returns
    -------
    dict:
        streaming_f1        : final streaming F1
        streaming_prauc     : final streaming PR-AUC
        pipeline_latency_ms : per-window latency stats (mean, std, max)
        drift_events        : list of window indices where ADWIN detected drift
        per_window_f1       : list of rolling F1 per window
        per_window_latency  : list of per-window latency in ms
    """
    from src.features import extract_features

    # ADWIN implementation (pure Python, no river dependency needed for basic use)
    # Falls back to river.drift.ADWIN if available for full implementation
    try:
        from river.drift import ADWIN
        adwin = ADWIN(delta=adwin_delta)
        _USE_RIVER = True
    except ImportError:
        adwin = _SimpleADWIN(delta=adwin_delta)
        _USE_RIVER = False
        if verbose:
            print("[prequential] river not installed; using built-in ADWIN approximation.")
            print("  Install river for full ADWIN: pip install river")

    n_windows     = len(X) // W
    all_y_true    = []
    all_y_pred    = []
    all_y_prob    = []
    latencies_ms  = []
    drift_events  = []
    per_win_f1    = []

    for k in range(n_windows):
        window = X[k * W:(k + 1) * W, :]
        y_pts  = Y[k * W:(k + 1) * W]
        y_true = int(np.mean(y_pts) > model.theta)

        # Timed inference
        t0      = time.perf_counter()
        y_hat, p_hat = model.predict_single_window(window)
        lat_ms  = (time.perf_counter() - t0) * 1e3

        all_y_true.append(y_true)
        all_y_pred.append(y_hat)
        all_y_prob.append(p_hat)
        latencies_ms.append(lat_ms)

        # Compute rolling F1 (needs at least 2 samples of each class)
        if len(all_y_true) >= 10:
            try:
                roll_f1 = f1_score(all_y_true, all_y_pred, pos_label=1, zero_division=0)
            except Exception:
                roll_f1 = 0.0
        else:
            roll_f1 = 0.0
        per_win_f1.append(roll_f1)

        # ADWIN drift detection on F1 residuals
        if len(per_win_f1) >= 2:
            residual = abs(per_win_f1[-1] - per_win_f1[-2])
            if _USE_RIVER:
                adwin.update(residual)
                if adwin.drift_detected:
                    drift_events.append(k)
                    if verbose:
                        print(f"  [ADWIN] Drift detected at window {k} "
                              f"(F1 residual={residual:.4f})")
            else:
                if adwin.update(residual):
                    drift_events.append(k)
                    if verbose:
                        print(f"  [ADWIN] Drift detected at window {k} "
                              f"(F1 residual={residual:.4f})")

        if verbose and (k + 1) % 100 == 0:
            print(f"  [streaming] Window {k+1}/{n_windows} | "
                  f"rolling F1={roll_f1:.3f} | "
                  f"latency={lat_ms:.3f} ms")

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    y_prob_arr = np.array(all_y_prob)
    lat_arr    = np.array(latencies_ms)

    try:
        s_f1    = f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)
        s_prauc = average_precision_score(y_true_arr, y_prob_arr, pos_label=1)
    except ValueError:
        s_f1, s_prauc = float("nan"), float("nan")

    result = {
        "streaming_f1"       : float(s_f1),
        "streaming_prauc"    : float(s_prauc),
        "pipeline_latency_ms": {
            "mean": float(lat_arr.mean()),
            "std" : float(lat_arr.std()),
            "max" : float(lat_arr.max()),
        },
        "drift_events"       : drift_events,
        "n_drift_events"     : len(drift_events),
        "per_window_f1"      : per_win_f1,
        "per_window_latency" : latencies_ms,
        "n_windows"          : n_windows,
    }

    if verbose:
        print(f"\n[Streaming Results]")
        print(f"  Streaming F1     : {s_f1:.4f}")
        print(f"  Streaming PR-AUC : {s_prauc:.4f}")
        print(f"  Pipeline latency : {lat_arr.mean():.3f} ± {lat_arr.std():.3f} ms "
              f"(max: {lat_arr.max():.3f} ms)")
        print(f"  Drift events     : {len(drift_events)}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Minimal built-in ADWIN approximation (no river dependency)
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleADWIN:
    """
    Lightweight ADWIN-inspired drift detector.

    Uses a two-window statistical test: splits a growing buffer at its midpoint
    and applies a Kolmogorov-Smirnov test. Flags drift when p < delta.

    This is a simplified approximation. For the full ADWIN algorithm as in
    Bifet & Gavalda (SDM 2007), install: pip install river
    """

    def __init__(self, delta: float = 0.002, min_samples: int = 30):
        self.delta       = delta
        self.min_samples = min_samples
        self._buffer     = []

    def update(self, value: float) -> bool:
        """Add a new observation. Returns True if drift is detected."""
        self._buffer.append(value)
        if len(self._buffer) < self.min_samples:
            return False
        mid  = len(self._buffer) // 2
        win1 = self._buffer[:mid]
        win2 = self._buffer[mid:]
        _, p = stats.ks_2samp(win1, win2)
        if p < self.delta:
            # Reset buffer after drift
            self._buffer = self._buffer[mid:]
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def print_confusion(y_true, y_pred, labels=("Normal", "Anomaly")):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"               Pred {labels[0]}   Pred {labels[1]}")
    print(f"  True {labels[0]}   {cm[0,0]:>10}   {cm[0,1]:>10}")
    print(f"  True {labels[1]}   {cm[1,0]:>10}   {cm[1,1]:>10}\n")
