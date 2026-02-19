"""
src/pipeline.py
---------------
ImFREQ-Lite full pipeline: data → features → SMOTE → ensemble → predictions.

Usage:
    from src.pipeline import ImFREQLite
    model = ImFREQLite(K=10, W=512, theta=0.50, smote_ratio=0.25)
    model.fit(X_train, Y_train)          # tau auto-tuned on 10% val fold
    metrics = model.evaluate(X_test, Y_test)

CHANGES vs. original:
    - fit() now automatically carves out a 10% stratified validation fold
      from training data and calls tune_tau() on it, matching paper
      Section IV-D: "10% of the training set held out for threshold tau
      and SMOTE ratio validation."
    - Removed deprecated `use_label_encoder=False` from XGBoost defaults
      (raises warning/error in XGBoost >= 1.6).
    - Added `auto_tune_tau` constructor flag (default True) so users can
      opt out if they want to set tau manually.
"""

import time
import tracemalloc
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

from src.features import window_and_extract
from src.evaluate import compute_metrics


class ImFREQLite:
    """
    ImFREQ-Lite: Lightweight Frequency-Domain Ensemble for IoT Anomaly Detection.

    Five-stage pipeline:
        S1 → Windowing (W=512, majority-vote labeling θ)
        S2 → FFT feature extraction (top-K magnitude bins, DC excluded)
        S3 → Statistical feature fusion (μ, σ, γ₁, γ₂, RMS per channel)
        S4 → Post-windowing SMOTE oversampling (training split only)
        S5 → RF + XGBoost soft-voting ensemble

    Parameters
    ----------
    W : int
        Window size in samples (default 512).
    K : int
        Number of top FFT magnitude bins per channel (default 10).
    theta : float
        Window labeling majority-vote threshold (default 0.50).
    smote_ratio : float
        SMOTE minority-to-majority oversampling ratio (default 0.25).
    smote_k : int
        Number of SMOTE nearest neighbours (default 5).
    tau : float
        Initial decision threshold on anomaly probability (default 0.38).
        Overwritten by tune_tau() if auto_tune_tau=True.
    auto_tune_tau : bool
        If True (default), fit() automatically carves out a 10% stratified
        validation fold from the training data and tunes tau on it,
        matching paper Section IV-D protocol.
    val_ratio : float
        Fraction of training data reserved for tau validation (default 0.10).
    rf_params : dict or None
        Override Random Forest hyperparameters.
    xgb_params : dict or None
        Override XGBoost hyperparameters.
    random_state : int
        Global random seed for reproducibility (default 42).
    verbose : bool
        Print stage summaries during fit/predict (default True).
    """

    # Default hyperparameters (paper settings)
    _RF_DEFAULTS = dict(
        n_estimators    = 100,
        max_depth       = 15,
        min_samples_leaf= 5,
        class_weight    = "balanced",
        n_jobs          = -1,
    )
    # FIX: Removed deprecated `use_label_encoder=False`
    _XGB_DEFAULTS = dict(
        n_estimators    = 100,
        max_depth       = 6,
        learning_rate   = 0.1,
        subsample       = 0.8,
        scale_pos_weight= 4,
        eval_metric     = "logloss",
        tree_method     = "hist",   # fast CPU training
    )

    def __init__(
        self,
        W             = 512,
        K             = 10,
        theta         = 0.50,
        smote_ratio   = 0.25,
        smote_k       = 5,
        tau           = 0.38,
        auto_tune_tau = True,
        val_ratio     = 0.10,
        rf_params     = None,
        xgb_params    = None,
        random_state  = 42,
        verbose       = True,
    ):
        self.W             = W
        self.K             = K
        self.theta         = theta
        self.smote_ratio   = smote_ratio
        self.smote_k       = smote_k
        self.tau           = tau
        self.auto_tune_tau = auto_tune_tau
        self.val_ratio     = val_ratio
        self.random_state  = random_state
        self.verbose       = verbose

        # Build RF (random_state injected here)
        rf_cfg = {**self._RF_DEFAULTS, **(rf_params or {}),
                  "random_state": random_state}
        self.rf  = RandomForestClassifier(**rf_cfg)

        # Build XGBoost (random_state injected here)
        xgb_cfg = {**self._XGB_DEFAULTS, **(xgb_params or {}),
                   "random_state": random_state}
        self.xgb = XGBClassifier(**xgb_cfg)

        # Internal state
        self._Phi_train  = None
        self._Y_train    = None
        self.train_time_ = None
        self.ram_mb_     = None

    # ── Stage 1–3: Feature extraction ────────────────────────────────────────

    def _extract(self, X, Y):
        return window_and_extract(
            X, Y,
            W=self.W, K=self.K, theta=self.theta,
            verbose=self.verbose,
        )

    # ── Stage 4: SMOTE ───────────────────────────────────────────────────────

    def _apply_smote(self, Phi, Y_w):
        n_anom = int(Y_w.sum())
        n_norm = int((Y_w == 0).sum())

        if n_anom == 0:
            raise ValueError(
                "No anomalous windows in training data. "
                "Lower theta or use a dataset with higher anomaly prevalence."
            )

        target_minority = int(n_norm * self.smote_ratio)
        if target_minority <= n_anom:
            if self.verbose:
                print(f"[SMOTE] Already at target ratio — skipping. "
                      f"(anom={n_anom}, target={target_minority})")
            return Phi, Y_w

        k = min(self.smote_k, n_anom - 1)
        if k < 1:
            if self.verbose:
                print("[SMOTE] Too few minority samples — skipping.")
            return Phi, Y_w

        smote = SMOTE(
            sampling_strategy=self.smote_ratio,
            k_neighbors=k,
            random_state=self.random_state,
        )
        Phi_res, Y_res = smote.fit_resample(Phi, Y_w)

        if self.verbose:
            print(f"[SMOTE] {n_anom} → {int(Y_res.sum())} anomaly windows "
                  f"(ratio={self.smote_ratio})")

        return Phi_res, Y_res

    # ── Tau tuning (internal) ─────────────────────────────────────────────────

    def _tune_tau_on_features(
        self,
        Phi_val: np.ndarray,
        Y_val:   np.ndarray,
        tau_grid: list = None,
    ) -> float:
        """Tune tau on a pre-extracted feature matrix (internal use)."""
        if tau_grid is None:
            tau_grid = np.arange(0.10, 0.90, 0.05).tolist()

        P_rf  = self.rf.predict_proba(Phi_val)[:, 1]
        P_xgb = self.xgb.predict_proba(Phi_val)[:, 1]
        P_hat = (P_rf + P_xgb) / 2.0

        best_tau, best_f1 = self.tau, 0.0
        for tau in tau_grid:
            y_pred = (P_hat >= tau).astype(int)
            f1 = f1_score(Y_val, y_pred, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

        if self.verbose:
            print(f"[tau tuning] Best tau={best_tau:.2f}  F1={best_f1:.4f}")

        self.tau = best_tau
        return best_tau

    # ── Stage 5: Ensemble fit ─────────────────────────────────────────────────

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit ImFREQ-Lite on raw sensor stream.

        Follows paper Section IV-D protocol:
            1. Extract windowed features for the full training stream.
            2. Stratified split: 90% sub-train / 10% validation.
            3. Apply SMOTE to sub-train only (no leakage into val).
            4. Fit RF + XGBoost on SMOTE-balanced sub-train features.
            5. If auto_tune_tau=True, tune tau on the 10% val features
               by maximising minority-class F1.

        Parameters
        ----------
        X : np.ndarray, shape [N, C]
            Raw sensor data (N samples, C channels).
        Y : np.ndarray, shape [N]
            Point-level binary labels (0=normal, 1=anomaly).

        Returns
        -------
        self
        """
        tracemalloc.start()
        t0 = time.perf_counter()

        # ── S1–S3: Extract windowed features ──────────────────────────────────
        Phi, Y_w = self._extract(X, Y)

        # ── Stratified 90/10 sub-split for tau validation ──────────────────
        #    Paper: "10% of the training set held out for threshold tau
        #             and SMOTE ratio validation." (Section IV-D)
        if self.auto_tune_tau and len(np.unique(Y_w)) > 1:
            Phi_sub, Phi_val, Y_sub, Y_val = train_test_split(
                Phi, Y_w,
                test_size=self.val_ratio,
                stratify=Y_w,
                random_state=self.random_state,
            )
            if self.verbose:
                print(f"[split] sub-train={len(Y_sub)} | val={len(Y_val)} "
                      f"(val_ratio={self.val_ratio})")
        else:
            # auto_tune_tau=False or only one class → use all data
            Phi_sub, Y_sub = Phi, Y_w
            Phi_val, Y_val = None, None

        # ── S4: SMOTE on sub-train only ────────────────────────────────────
        Phi_bal, Y_bal = self._apply_smote(Phi_sub, Y_sub)

        # ── S5: Fit base classifiers ────────────────────────────────────────
        if self.verbose:
            print("[ensemble] Fitting RF ...")
        self.rf.fit(Phi_bal, Y_bal)

        if self.verbose:
            print("[ensemble] Fitting XGBoost ...")
        self.xgb.fit(Phi_bal, Y_bal)

        # ── Tau tuning on held-out val fold ────────────────────────────────
        if self.auto_tune_tau and Phi_val is not None:
            self._tune_tau_on_features(Phi_val, Y_val)

        self.train_time_ = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.ram_mb_ = peak / 1024 / 1024

        if self.verbose:
            print(f"[fit] Done. tau={self.tau:.2f} | "
                  f"Train time: {self.train_time_:.1f}s | "
                  f"Peak RAM: {self.ram_mb_:.1f} MB")

        self._Phi_train = Phi
        self._Y_train   = Y_w
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """
        Extract features and return ensemble anomaly probabilities.

        Returns
        -------
        P_hat : np.ndarray, shape [n_windows]
            Ensemble anomaly probability per window.
        Y_w : np.ndarray, shape [n_windows]
            Window-level ground truth labels.
        """
        Phi, Y_w = self._extract(X, Y)

        P_rf  = self.rf.predict_proba(Phi)[:, 1]
        P_xgb = self.xgb.predict_proba(Phi)[:, 1]
        P_hat = (P_rf + P_xgb) / 2.0

        return P_hat, Y_w

    def predict(self, X: np.ndarray, Y: np.ndarray) -> tuple:
        """Return binary predictions and ground truth window labels."""
        P_hat, Y_w = self.predict_proba(X, Y)
        Y_hat = (P_hat >= self.tau).astype(int)
        return Y_hat, Y_w, P_hat

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """
        Predict on test data and return full metrics dictionary.

        Returns
        -------
        dict with keys: F1, PR_AUC, Recall, Precision, ROC_AUC,
                        train_time_s, infer_us_per_sample, ram_mb
        """
        t0 = time.perf_counter()
        Y_hat, Y_w, P_hat = self.predict(X, Y)
        infer_time = (time.perf_counter() - t0) / max(len(Y_hat), 1) * 1e6

        metrics = compute_metrics(Y_w, Y_hat, P_hat)
        metrics["train_time_s"]        = round(self.train_time_, 2) if self.train_time_ else None
        metrics["infer_us_per_sample"] = round(infer_time, 2)
        metrics["ram_mb"]              = round(self.ram_mb_, 1) if self.ram_mb_ else None

        return metrics

    # ── Manual tau tuning (on raw sensor data) ────────────────────────────────

    def tune_tau(
        self,
        X_val:    np.ndarray,
        Y_val:    np.ndarray,
        tau_grid: list = None,
    ) -> float:
        """
        Manually tune decision threshold tau on a held-out validation set.

        Use this when auto_tune_tau=False or to override the automatic tuning.

        Parameters
        ----------
        X_val, Y_val : raw sensor stream + point-level labels.
        tau_grid     : list of floats to search (default: 0.05 step 0.10–0.90).

        Returns
        -------
        best_tau : float
        """
        if tau_grid is None:
            tau_grid = np.arange(0.10, 0.90, 0.05).tolist()

        P_hat, Y_w = self.predict_proba(X_val, Y_val)

        best_tau, best_f1 = self.tau, 0.0
        for tau in tau_grid:
            y_pred = (P_hat >= tau).astype(int)
            f1 = f1_score(Y_w, y_pred, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau

        if self.verbose:
            print(f"[tau tuning] Best tau={best_tau:.2f}  F1={best_f1:.4f}")

        self.tau = best_tau
        return best_tau

    def __repr__(self):
        return (
            f"ImFREQLite(W={self.W}, K={self.K}, theta={self.theta}, "
            f"tau={self.tau}, smote_ratio={self.smote_ratio}, "
            f"auto_tune_tau={self.auto_tune_tau})"
        )