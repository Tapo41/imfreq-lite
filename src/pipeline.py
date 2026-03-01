"""
src/pipeline.py
---------------
ImFREQ-Lite full pipeline: data → features → SMOTE → ensemble → predictions.

v3 Changes
----------
- infer_latency_breakdown(): decomposes per-sample inference into
  FFT time, statistical feature time, and tree-ensemble prediction time.
- energy_profile(): computes energy-per-inference (μJ) and projects battery life
  given a measured device power draw (mW). Used for ARM deployment reporting.
- predict_single_window(): low-overhead single-window inference for streaming.
- save() / load(): joblib serialisation for ARM deployment.

Usage:
    from src.pipeline import ImFREQLite
    model = ImFREQLite(K=10, W=512, theta=0.50, smote_ratio=0.25, tau=0.38)
    model.fit(X_train, Y_train)
    metrics = model.evaluate(X_test, Y_test)
    breakdown = model.infer_latency_breakdown(X_test, Y_test, n_trials=1000)
"""

import time
import tracemalloc
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

from src.features import window_and_extract, fft_features, statistical_features
from src.evaluate import compute_metrics


class ImFREQLite:
    """
    ImFREQ-Lite: Lightweight Frequency-Domain Ensemble for IoT Anomaly Detection.

    Five-stage pipeline:
        S1 → Windowing (W=512, majority-vote labeling θ)
        S2 → FFT feature extraction (top-K magnitude bins, DC excluded)
        S3 → Statistical feature fusion (μ, σ, γ₁, γ₂, RMS per channel)
        S4 → Post-windowing SMOTE oversampling
        S5 → RF + XGBoost soft-voting ensemble

    Parameters
    ----------
    W            : Window size in samples (default 512).
    K            : Top FFT magnitude bins per channel (default 10).
    theta        : Window labeling majority-vote threshold (default 0.50).
    smote_ratio  : SMOTE minority-to-majority ratio (default 0.25).
    smote_k      : SMOTE nearest neighbours (default 5).
    tau          : Decision threshold; tune per dataset on validation fold.
    rf_params    : Override Random Forest hyperparameters.
    xgb_params   : Override XGBoost hyperparameters.
    random_state : Global random seed (default 42).
    verbose      : Print stage summaries (default True).
    """

    _RF_DEFAULTS = dict(n_estimators=100, max_depth=15, min_samples_leaf=5,
                        class_weight="balanced", n_jobs=-1)
    _XGB_DEFAULTS = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                         subsample=0.8, scale_pos_weight=4, eval_metric="logloss",
                         use_label_encoder=False, tree_method="hist")

    def __init__(self, W=512, K=10, theta=0.50, smote_ratio=0.25, smote_k=5,
                 tau=0.38, rf_params=None, xgb_params=None,
                 random_state=42, verbose=True):
        self.W            = W
        self.K            = K
        self.theta        = theta
        self.smote_ratio  = smote_ratio
        self.smote_k      = smote_k
        self.tau          = tau
        self.random_state = random_state
        self.verbose      = verbose

        rf_cfg  = {**self._RF_DEFAULTS,  **(rf_params  or {}), "random_state": random_state}
        xgb_cfg = {**self._XGB_DEFAULTS, **(xgb_params or {}), "random_state": random_state}
        self.rf  = RandomForestClassifier(**rf_cfg)
        self.xgb = XGBClassifier(**xgb_cfg)

        self._Phi_train  = None
        self._Y_train    = None
        self.train_time_ = None
        self.ram_mb_     = None

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract(self, X, Y):
        return window_and_extract(X, Y, W=self.W, K=self.K,
                                  theta=self.theta, verbose=self.verbose)

    # ── SMOTE ─────────────────────────────────────────────────────────────────

    def _apply_smote(self, Phi, Y_w):
        n_anom = int(Y_w.sum())
        n_norm = int((Y_w == 0).sum())
        if n_anom == 0:
            raise ValueError("No anomalous windows in training data.")
        target = int(n_norm * self.smote_ratio)
        if target <= n_anom:
            if self.verbose:
                print(f"[SMOTE] Already at target ratio — skipping.")
            return Phi, Y_w
        k = min(self.smote_k, n_anom - 1)
        if k < 1:
            return Phi, Y_w
        smote = SMOTE(sampling_strategy=self.smote_ratio,
                      k_neighbors=k, random_state=self.random_state)
        Phi_res, Y_res = smote.fit_resample(Phi, Y_w)
        if self.verbose:
            print(f"[SMOTE] {n_anom} → {int(Y_res.sum())} anomaly windows "
                  f"(ratio={self.smote_ratio})")
        return Phi_res, Y_res

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit ImFREQ-Lite on a raw sensor stream.

        Parameters
        ----------
        X : np.ndarray, shape [N, C]   (C=1 for univariate NAB, C=3 for ToN-IoT/SKAB)
        Y : np.ndarray, shape [N]      point-level binary labels
        """
        tracemalloc.start()
        t0 = time.perf_counter()

        Phi, Y_w       = self._extract(X, Y)
        Phi_bal, Y_bal = self._apply_smote(Phi, Y_w)

        if self.verbose: print("[ensemble] Fitting RF ...")
        self.rf.fit(Phi_bal, Y_bal)
        if self.verbose: print("[ensemble] Fitting XGBoost ...")
        self.xgb.fit(Phi_bal, Y_bal)

        self.train_time_ = time.perf_counter() - t0
        _, peak          = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.ram_mb_ = peak / 1024 / 1024

        if self.verbose:
            print(f"[fit] Done. Train: {self.train_time_:.1f}s | "
                  f"Peak RAM: {self.ram_mb_:.1f} MB")

        self._Phi_train = Phi
        self._Y_train   = Y_w
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """Return ensemble anomaly probabilities and window-level ground truth."""
        Phi, Y_w = self._extract(X, Y)
        P_rf     = self.rf.predict_proba(Phi)[:, 1]
        P_xgb    = self.xgb.predict_proba(Phi)[:, 1]
        return (P_rf + P_xgb) / 2.0, Y_w

    def predict(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """Return binary predictions, ground truth labels, and probabilities."""
        P_hat, Y_w = self.predict_proba(X, Y)
        return (P_hat >= self.tau).astype(int), Y_w, P_hat

    def predict_single_window(self, window: np.ndarray) -> tuple:
        """
        Low-overhead single-window inference for real-time streaming.

        Parameters
        ----------
        window : np.ndarray, shape [W, C]

        Returns
        -------
        y_hat : int   (0=normal, 1=anomaly)
        p_hat : float (anomaly probability)
        """
        from src.features import extract_features
        phi   = extract_features(window, K=self.K).reshape(1, -1)
        p_rf  = self.rf.predict_proba(phi)[0, 1]
        p_xgb = self.xgb.predict_proba(phi)[0, 1]
        p_hat = (p_rf + p_xgb) / 2.0
        return int(p_hat >= self.tau), float(p_hat)

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Predict on test data and return full metrics dict."""
        t0 = time.perf_counter()
        Y_hat, Y_w, P_hat = self.predict(X, Y)
        infer_time = (time.perf_counter() - t0) / max(len(Y_hat), 1) * 1e6
        metrics = compute_metrics(Y_w, Y_hat, P_hat)
        metrics["train_time_s"]        = round(self.train_time_, 2) if self.train_time_ else None
        metrics["infer_us_per_sample"] = round(infer_time, 2)
        metrics["ram_mb"]              = round(self.ram_mb_, 1) if self.ram_mb_ else None
        return metrics

    # ── Inference Latency Breakdown  (v3 addition) ───────────────────────────

    def infer_latency_breakdown(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_trials: int = 1000,
    ) -> dict:
        """
        Decompose per-sample inference latency into three components:
            1. FFT feature extraction time (μs)
            2. Statistical feature extraction time (μs)
            3. Tree ensemble prediction time (μs)

        Each component is timed independently over n_trials single windows
        after a 100-sample warm-up, then averaged.

        Parameters
        ----------
        X        : raw sensor stream [N, C]
        Y        : point labels [N]
        n_trials : number of timing iterations (default 1000)

        Returns
        -------
        dict with keys:
            fft_us, stat_us, tree_us, total_us  (mean ± std over trials)
        """
        from src.features import fft_features, statistical_features, extract_features

        # Build windows for timing
        n_windows = len(X) // self.W
        windows   = [X[k * self.W:(k + 1) * self.W, :]
                     for k in range(min(n_windows, n_trials + 100))]
        if len(windows) < n_trials + 100:
            # Pad by repeating
            while len(windows) < n_trials + 100:
                windows.extend(windows[:min(n_trials + 100 - len(windows), len(windows))])

        warm_up = windows[:100]
        trial_windows = windows[100: 100 + n_trials]

        # Warm up
        for w in warm_up:
            fft_features(w, K=self.K)
            statistical_features(w)

        # Pre-extract features for tree timing
        Phi_trials = np.array([extract_features(w, K=self.K) for w in trial_windows])

        # Time FFT extraction
        fft_times = []
        for w in trial_windows:
            t0 = time.perf_counter()
            fft_features(w, K=self.K)
            fft_times.append((time.perf_counter() - t0) * 1e6)

        # Time statistical extraction
        stat_times = []
        for w in trial_windows:
            t0 = time.perf_counter()
            statistical_features(w)
            stat_times.append((time.perf_counter() - t0) * 1e6)

        # Time tree ensemble prediction (using pre-extracted features)
        tree_times = []
        for phi in Phi_trials:
            phi_2d = phi.reshape(1, -1)
            t0 = time.perf_counter()
            p_rf  = self.rf.predict_proba(phi_2d)[0, 1]
            p_xgb = self.xgb.predict_proba(phi_2d)[0, 1]
            _ = (p_rf + p_xgb) / 2.0
            tree_times.append((time.perf_counter() - t0) * 1e6)

        fft_arr  = np.array(fft_times)
        stat_arr = np.array(stat_times)
        tree_arr = np.array(tree_times)
        total    = fft_arr + stat_arr + tree_arr

        result = {
            "fft_us"   : {"mean": float(fft_arr.mean()),  "std": float(fft_arr.std())},
            "stat_us"  : {"mean": float(stat_arr.mean()), "std": float(stat_arr.std())},
            "tree_us"  : {"mean": float(tree_arr.mean()), "std": float(tree_arr.std())},
            "total_us" : {"mean": float(total.mean()),    "std": float(total.std())},
            "n_trials" : n_trials,
        }

        print(f"\n[Latency Breakdown] (n_trials={n_trials})")
        print(f"  FFT   : {result['fft_us']['mean']:.2f} ± {result['fft_us']['std']:.2f} μs "
              f"({100*result['fft_us']['mean']/result['total_us']['mean']:.1f}%)")
        print(f"  Stat  : {result['stat_us']['mean']:.2f} ± {result['stat_us']['std']:.2f} μs "
              f"({100*result['stat_us']['mean']/result['total_us']['mean']:.1f}%)")
        print(f"  Tree  : {result['tree_us']['mean']:.2f} ± {result['tree_us']['std']:.2f} μs "
              f"({100*result['tree_us']['mean']/result['total_us']['mean']:.1f}%)")
        print(f"  Total : {result['total_us']['mean']:.2f} ± {result['total_us']['std']:.2f} μs")
        return result

    # ── Energy Profiling  (v3 addition) ──────────────────────────────────────

    def energy_profile(
        self,
        infer_latency_us: float,
        device_power_mw: float,
        battery_capacity_wh: float = 9.25,
        sample_rate_hz: float = 1.0,
    ) -> dict:
        """
        Compute energy-per-inference and project battery life.

        Parameters
        ----------
        infer_latency_us    : measured per-sample inference latency in μs
        device_power_mw     : measured active-inference device power in mW
        battery_capacity_wh : battery capacity in Wh (default 9.25 = 2500mAh @ 3.7V)
        sample_rate_hz      : sensor sampling rate (default 1.0 Hz)

        Returns
        -------
        dict:
            energy_per_infer_uj  : energy per inference (μJ)
            energy_per_sample_nj : energy per sample (nJ) — same for single-sample pipeline
            battery_life_hrs     : projected battery life in hours at continuous operation
        """
        t_s   = infer_latency_us * 1e-6        # latency in seconds
        p_w   = device_power_mw * 1e-3         # power in Watts
        e_j   = p_w * t_s                      # energy per inference in Joules
        e_uj  = e_j * 1e6                      # → μJ
        e_nj  = e_j * 1e9                      # → nJ (same as μJ for single-sample)
        batt_j = battery_capacity_wh * 3600    # battery in Joules
        # Battery life at continuous 1 Hz: each second = 1 inference + idle
        batt_hrs = battery_capacity_wh / (p_w)  # hours at full active draw

        result = {
            "infer_latency_us"   : infer_latency_us,
            "device_power_mw"    : device_power_mw,
            "energy_per_infer_uj": round(e_uj, 3),
            "energy_per_sample_nj": round(e_nj, 3),
            "battery_life_hrs"   : round(batt_hrs, 1),
            "battery_capacity_wh": battery_capacity_wh,
        }

        print(f"\n[Energy Profile]")
        print(f"  Power             : {device_power_mw:.1f} mW")
        print(f"  Inference latency : {infer_latency_us:.2f} μs")
        print(f"  Energy/inference  : {e_uj:.3f} μJ")
        print(f"  Battery life      : {batt_hrs:.1f} hrs "
              f"(2500 mAh @ 3.7V, {sample_rate_hz:.0f} Hz)")
        return result

    # ── Tau Tuning ────────────────────────────────────────────────────────────

    def tune_tau(self, X_val, Y_val, tau_grid=None) -> float:
        """Select tau maximising minority-class F1 on a validation set."""
        from sklearn.metrics import f1_score
        if tau_grid is None:
            tau_grid = np.arange(0.10, 0.90, 0.05).tolist()
        P_hat, Y_w = self.predict_proba(X_val, Y_val)
        best_tau, best_f1 = 0.5, 0.0
        for tau in tau_grid:
            f1 = f1_score(Y_w, (P_hat >= tau).astype(int), pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        if self.verbose:
            print(f"[tau tuning] Best tau={best_tau:.2f}  F1={best_f1:.4f}")
        self.tau = best_tau
        return best_tau

    # ── Serialisation for ARM Deployment ─────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the fitted model to disk using joblib.

        The serialised file contains both RF and XGBoost models plus all
        hyperparameters. Size is typically 3–5 MB for paper configurations.

        Parameters
        ----------
        path : str  e.g. "models/imfreq_lite_toniot.joblib"
        """
        import joblib, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "rf": self.rf, "xgb": self.xgb,
            "W": self.W, "K": self.K, "theta": self.theta,
            "tau": self.tau, "smote_ratio": self.smote_ratio,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path, compress=3)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[save] Model saved → {path}  ({size_mb:.2f} MB)")

    @classmethod
    def load(cls, path: str) -> "ImFREQLite":
        """
        Load a serialised ImFREQ-Lite model from disk.

        Parameters
        ----------
        path : str  path to .joblib file

        Returns
        -------
        ImFREQLite instance ready for inference
        """
        import joblib
        payload = joblib.load(path)
        model = cls(W=payload["W"], K=payload["K"], theta=payload["theta"],
                    tau=payload["tau"], smote_ratio=payload["smote_ratio"],
                    random_state=payload["random_state"], verbose=False)
        model.rf  = payload["rf"]
        model.xgb = payload["xgb"]
        print(f"[load] Model loaded ← {path}")
        return model

    def __repr__(self):
        return (f"ImFREQLite(W={self.W}, K={self.K}, theta={self.theta}, "
                f"tau={self.tau}, smote_ratio={self.smote_ratio})")
