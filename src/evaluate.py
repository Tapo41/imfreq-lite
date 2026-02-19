"""
src/evaluate.py
---------------
Evaluation metrics for ImFREQ-Lite.

Covers: F1, PR-AUC, Recall, Precision, ROC-AUC + paired t-test.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
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
    y_true : array-like, shape [n]
        True binary window labels.
    y_pred : array-like, shape [n]
        Predicted binary window labels.
    y_prob : array-like, shape [n], optional
        Predicted anomaly probabilities (needed for PR-AUC and ROC-AUC).

    Returns
    -------
    dict with keys:
        F1         — minority-class (anomaly) F1-score
        Precision  — minority-class precision
        Recall     — minority-class recall (sensitivity)
        PR_AUC     — precision-recall AUC (primary metric for imbalanced data)
        ROC_AUC    — receiver operating characteristic AUC
        Accuracy   — overall accuracy (secondary; can be misleading with imbalance)
        n_test     — number of test windows
        n_anom     — number of true anomalous windows
        anomaly_rate — anomaly prevalence in test set
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "F1"        : f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision" : precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall"    : recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Accuracy"  : float(np.mean(y_true == y_pred)),
        "n_test"    : int(len(y_true)),
        "n_anom"    : int(y_true.sum()),
        "anomaly_rate" : float(y_true.mean()),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["PR_AUC"]  = average_precision_score(y_true, y_prob, pos_label=1)
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Only one class present in y_true
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
    run_metrics : list of dicts
        Each dict is the output of compute_metrics() for one run.

    Returns
    -------
    dict mapping metric name → {"mean": ..., "std": ..., "values": [...]}
    """
    keys = [k for k in run_metrics[0].keys()
            if isinstance(run_metrics[0][k], (int, float))]

    aggregated = {}
    for k in keys:
        vals = np.array([rm[k] for rm in run_metrics], dtype=float)
        aggregated[k] = {
            "mean"   : float(np.nanmean(vals)),
            "std"    : float(np.nanstd(vals, ddof=1)),
            "values" : vals.tolist(),
        }
    return aggregated


def print_summary(aggregated: dict, title: str = "Results") -> None:
    """Print a clean summary table of aggregated multi-run results."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    primary = ["F1", "PR_AUC", "Recall", "Precision", "ROC_AUC"]
    for k in primary:
        if k in aggregated:
            m = aggregated[k]["mean"]
            s = aggregated[k]["std"]
            print(f"  {k:<12}  {m:.4f} ± {s:.4f}")
    if "train_time_s" in aggregated:
        m = aggregated["train_time_s"]["mean"]
        s = aggregated["train_time_s"]["std"]
        print(f"  {'Train (s)':<12}  {m:.1f} ± {s:.1f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Significance Testing
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(
    scores_a: list,
    scores_b: list,
    alpha: float = 0.05,
    label_a: str = "Method A",
    label_b: str = "Method B",
) -> dict:
    """
    Two-tailed paired t-test between two methods over multiple runs.

    Parameters
    ----------
    scores_a, scores_b : list of float
        Per-run metric values (e.g., F1 scores) for each method.
    alpha : float
        Significance level (default 0.05).
    label_a, label_b : str
        Display names for the two methods.

    Returns
    -------
    dict with keys: t_stat, p_value, significant, direction
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Both score lists must have the same length (same number of runs).")
    if len(a) < 2:
        raise ValueError("Need at least 2 runs for a paired t-test.")

    t_stat, p_value = stats.ttest_rel(a, b)

    result = {
        "t_stat"      : float(t_stat),
        "p_value"     : float(p_value),
        "significant" : bool(p_value < alpha),
        "direction"   : f"{label_a} > {label_b}" if np.mean(a) > np.mean(b)
                        else f"{label_b} > {label_a}",
        "mean_diff"   : float(np.mean(a) - np.mean(b)),
    }

    sig_str = "SIGNIFICANT" if result["significant"] else "not significant"
    print(f"\n[t-test] {label_a} vs {label_b}")
    print(f"  mean({label_a}) = {np.mean(a):.4f}  |  "
          f"mean({label_b}) = {np.mean(b):.4f}")
    print(f"  t = {t_stat:.4f},  p = {p_value:.4f}  → {sig_str} at α={alpha}")

    return result


def significance_table(
    method_scores: dict,
    reference: str,
    alpha: float = 0.05,
) -> None:
    """
    Print a significance table comparing all methods against a reference.

    Parameters
    ----------
    method_scores : dict mapping method_name → list of per-run F1 scores
    reference : str
        Key in method_scores to compare against.
    alpha : float
        Significance level.
    """
    ref_scores = np.array(method_scores[reference])

    print(f"\n{'Method':<30} {'Mean F1':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 60)
    for name, scores in method_scores.items():
        s = np.array(scores)
        mean_f1 = np.mean(s)
        if name == reference:
            print(f"  {name:<28} {mean_f1:>8.4f} {'---':>10} {'---':>6}  ← reference")
            continue
        _, p_val = stats.ttest_rel(ref_scores, s)
        sig = "✓" if p_val < alpha else "✗"
        print(f"  {name:<28} {mean_f1:>8.4f} {p_val:>10.4f} {sig:>6}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Display
# ─────────────────────────────────────────────────────────────────────────────

def print_confusion(y_true, y_pred, labels=("Normal", "Anomaly")):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"               Pred {labels[0]}   Pred {labels[1]}")
    print(f"  True {labels[0]}   {cm[0,0]:>10}   {cm[0,1]:>10}")
    print(f"  True {labels[1]}   {cm[1,0]:>10}   {cm[1,1]:>10}")
    print()
