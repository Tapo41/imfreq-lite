"""
ImFREQ-Lite v3: Lightweight Frequency-Domain Ensemble Framework
for Imbalanced IoT Anomaly Detection in Smart City Environments.

Package structure
-----------------
    src.features   — FFT + statistical feature extraction
    src.pipeline   — Full ImFREQ-Lite pipeline (fit / predict / evaluate /
                     infer_latency_breakdown / energy_profile / save / load)
    src.baselines  — All 9 baseline implementations (incl. TCN, Quant. Transformer)
    src.evaluate   — Metrics, paired t-test, multi-run aggregation,
                     prequential streaming evaluation
    src.streaming  — Real-time sliding-window streaming engine + ADWIN drift
    src.arm_deploy — ARM hardware profiler (RPi 4 deployment + energy profiling)
    src.utils      — Data loading (NAB native univariate), preprocessing,
                     reproducibility helpers

v3 Changes Summary
------------------
1. NAB Yahoo S5 evaluated in native univariate mode (C=1) — channel replication removed.
2. TCN and Quantized Transformer baselines added (src.baselines).
3. Inference latency decomposed into FFT / statistical / tree components (src.pipeline).
4. Physical ARM deployment profiling (src.arm_deploy).
5. Energy-per-inference and battery-life projection (src.pipeline, src.arm_deploy).
6. Real-time streaming simulation with ADWIN drift detection (src.streaming, src.evaluate).
7. PyTorch seeding added to set_seed() (src.utils).
"""

from src.features   import window_and_extract, extract_features
from src.evaluate   import (
    compute_metrics, aggregate_runs, print_summary,
    paired_ttest, significance_table, prequential_eval,
)
from src.utils      import set_seed, PAPER_SEEDS

try:
    from src.pipeline   import ImFREQLite
    from src.streaming  import StreamingEngine
    from src.arm_deploy import ARMProfiler, save_model, load_model
except ImportError:
    pass   # optional deps (xgboost, sklearn, torch, psutil) not installed

__version__ = "3.0.0"
__all__ = [
    # Core pipeline
    "ImFREQLite",
    # Features
    "window_and_extract",
    "extract_features",
    # Evaluation
    "compute_metrics",
    "aggregate_runs",
    "print_summary",
    "paired_ttest",
    "significance_table",
    "prequential_eval",
    # Streaming
    "StreamingEngine",
    # ARM deployment
    "ARMProfiler",
    "save_model",
    "load_model",
    # Utils
    "set_seed",
    "PAPER_SEEDS",
]
