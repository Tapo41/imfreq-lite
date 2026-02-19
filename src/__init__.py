"""
ImFREQ-Lite: Lightweight Frequency-Domain Ensemble Framework
for Imbalanced IoT Anomaly Detection.

Package structure:
    src.features   — FFT + statistical feature extraction
    src.pipeline   — Full ImFREQ-Lite pipeline (fit / predict / evaluate)
    src.baselines  — All 7 baseline implementations
    src.evaluate   — Metrics, paired t-test, multi-run aggregation
    src.utils      — Data loading, preprocessing, reproducibility helpers
"""

from src.features  import window_and_extract, extract_features
from src.evaluate  import compute_metrics, aggregate_runs, paired_ttest
from src.utils     import set_seed, PAPER_SEEDS

try:
    from src.pipeline import ImFREQLite
except ImportError:
    pass  # xgboost/sklearn not installed; install requirements.txt

__version__ = "1.0.0"
__all__ = [
    "ImFREQLite",
    "window_and_extract",
    "extract_features",
    "compute_metrics",
    "aggregate_runs",
    "paired_ttest",
    "set_seed",
    "PAPER_SEEDS",
]
