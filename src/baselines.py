"""
src/baselines.py
----------------
All 7 baseline implementations used in the ImFREQ-Lite paper.

Each baseline follows the same interface:
    baseline.fit(Phi_train, Y_train)
    baseline.predict(Phi_test)  → (Y_hat, P_hat)

Baselines:
    1. Isolation Forest (IF)
    2. One-Class SVM (OC-SVM)
    3. LightGBM
    4. XGBoost (standalone)
    5. Random Forest (standalone)
    6. LSTM Autoencoder          ← FIX: now operates on raw 3D windows [N, W, C]
    7. Focal-Loss XGBoost

CHANGES vs. original:
    - LSTMAutoencoderBaseline: rewritten to accept raw [N, W, C] windows
      instead of the flat feature matrix. Use fit_raw() / predict_raw().
    - Removed deprecated `use_label_encoder=False` from XGBoost-based models
      (raises warning/error in XGBoost >= 1.6).
    - FocalLossXGBoostBaseline: random_state seed now consistently applied.
"""

import time
import tracemalloc
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.evaluate import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Base wrapper with timing + memory tracking
# ─────────────────────────────────────────────────────────────────────────────

class _BaselineWrapper:
    """Common interface for all baselines."""

    name = "Baseline"

    def fit(self, X, y):
        tracemalloc.start()
        t0 = time.perf_counter()
        self._fit(X, y)
        self.train_time_ = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.ram_mb_ = peak / 1024 / 1024
        print(f"[{self.name}] Train: {self.train_time_:.1f}s | RAM: {self.ram_mb_:.1f} MB")
        return self

    def evaluate(self, X, y_true, tau=0.5):
        t0 = time.perf_counter()
        Y_hat, P_hat = self.predict(X, tau=tau)
        infer_time = (time.perf_counter() - t0) / max(len(Y_hat), 1) * 1e6

        metrics = compute_metrics(y_true, Y_hat, P_hat)
        metrics["train_time_s"]        = round(self.train_time_, 2)
        metrics["infer_us_per_sample"] = round(infer_time, 2)
        metrics["ram_mb"]              = round(self.ram_mb_, 1)
        return metrics

    def tune_tau(self, X_val, y_val, tau_grid=None):
        """Find tau maximising F1 on a validation fold."""
        if tau_grid is None:
            tau_grid = np.arange(0.05, 0.95, 0.05)
        best_tau, best_f1 = 0.5, 0.0
        for tau in tau_grid:
            y_hat, _ = self.predict(X_val, tau=tau)
            f1 = f1_score(y_val, y_hat, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau = f1, tau
        print(f"[{self.name}] Best tau={best_tau:.2f}  val-F1={best_f1:.4f}")
        return best_tau


# ─────────────────────────────────────────────────────────────────────────────
# 1. Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestBaseline(_BaselineWrapper):
    """Unsupervised Isolation Forest (Liu et al., ICDM 2008)."""

    name = "Isolation Forest"

    def __init__(self, n_estimators=100, contamination=0.04, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

    def _fit(self, X, y):
        # Unsupervised: train on all data (ignores y)
        self.model.fit(X)

    def predict(self, X, tau=None):
        # predict: +1=normal, -1=anomaly
        raw_pred = self.model.predict(X)
        Y_hat = (raw_pred == -1).astype(int)

        # Anomaly score: negate so higher = more anomalous
        scores = -self.model.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        P_hat = (scores - s_min) / (s_max - s_min + 1e-9)

        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 2. One-Class SVM
# ─────────────────────────────────────────────────────────────────────────────

class OneClassSVMBaseline(_BaselineWrapper):
    """One-Class SVM (Scholkopf et al., NIPS 2000) trained on normal class."""

    name = "One-Class SVM"

    def __init__(self, kernel="rbf", nu=0.05):
        self.model = OneClassSVM(kernel=kernel, nu=nu)

    def _fit(self, X, y):
        X_normal = X[y == 0]
        print(f"  [OC-SVM] Training on {len(X_normal)} normal windows.")
        self.model.fit(X_normal)

    def predict(self, X, tau=None):
        raw_pred = self.model.predict(X)
        Y_hat = (raw_pred == -1).astype(int)

        scores = -self.model.decision_function(X)
        s_min, s_max = scores.min(), scores.max()
        P_hat = (scores - s_min) / (s_max - s_min + 1e-9)

        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 3. LightGBM
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMBaseline(_BaselineWrapper):
    """LightGBM baseline (Ke et al., NIPS 2017)."""

    name = "LightGBM"

    def __init__(self, n_estimators=100, max_depth=6, random_state=42):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.tau = 0.5

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, tau=None):
        tau = tau if tau is not None else self.tau
        P_hat = self.model.predict_proba(X)[:, 1]
        Y_hat = (P_hat >= tau).astype(int)
        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost (standalone, no ensemble voting)
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostBaseline(_BaselineWrapper):
    """XGBoost standalone baseline (Chen & Guestrin, KDD 2016).

    FIX: Removed deprecated `use_label_encoder=False` parameter.
    """

    name = "XGBoost (standalone)"

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=0.8, scale_pos_weight=4, random_state=42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        self.tau = 0.5

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, tau=None):
        tau = tau if tau is not None else self.tau
        P_hat = self.model.predict_proba(X)[:, 1]
        Y_hat = (P_hat >= tau).astype(int)
        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 5. Random Forest (standalone, no ensemble voting)
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestBaseline(_BaselineWrapper):
    """Random Forest standalone baseline (Breiman, 2001)."""

    name = "Random Forest (standalone)"

    def __init__(self, n_estimators=100, max_depth=15, min_samples_leaf=5,
                 random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.tau = 0.5

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, tau=None):
        tau = tau if tau is not None else self.tau
        P_hat = self.model.predict_proba(X)[:, 1]
        Y_hat = (P_hat >= tau).astype(int)
        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoderBaseline(_BaselineWrapper):
    """
    LSTM Autoencoder baseline (Meidan et al., IEEE Pervasive Comput. 2018).

    Architecture: LSTM(64) → LSTM(32) encoder, RepeatVector,
                  LSTM(32) → LSTM(64) decoder, TimeDistributed Dense.

    Anomaly score = per-window reconstruction MSE.
    Decision threshold = paper_pct-th percentile of training MSE.

    FIX (vs. original code):
    -------------------------
    The original code incorrectly reshaped the flat feature matrix Phi
    (shape [N, (K+5)*C]) and fed it as pseudo-sequences to the LSTM.
    This does NOT reproduce the paper's LSTM autoencoder results.

    The paper trains the LSTM on the **raw sensor windows** X_win of shape
    [N, W, C] where W=512 and C=3.  The corrected interface uses two
    dedicated methods:

        fit_raw(X_win, y)        — trains on raw [N, W, C] windows
        predict_raw(X_win)       — returns (Y_hat, P_hat) from raw windows
        evaluate_raw(X_win, y)   — full metrics from raw windows

    For compatibility with _BaselineWrapper's fit() / evaluate() interface
    (which passes the flat Phi matrix), a thin shim is still provided, but
    callers are strongly encouraged to use fit_raw / predict_raw directly.

    Usage (recommended):
        baseline = LSTMAutoencoderBaseline()
        baseline.fit_raw(X_windows_train, Y_windows_train)
        metrics = baseline.evaluate_raw(X_windows_test, Y_windows_test)

    where X_windows has shape [n_windows, W, C] — build it with:
        from src.features import build_raw_windows
        X_windows, Y_windows = build_raw_windows(X, Y, W=512, theta=0.50)

    Requires TensorFlow >= 2.x.
    """

    name = "LSTM Autoencoder"

    def __init__(
        self,
        lstm_units=(64, 32),
        epochs=20,
        batch_size=64,
        threshold_pct=95,
        random_state=42,
    ):
        self.lstm_units    = lstm_units
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.threshold_pct = threshold_pct
        self.random_state  = random_state
        self.model         = None
        self.threshold_    = None
        self.train_time_   = None
        self.ram_mb_       = None

    # ── Model construction ────────────────────────────────────────────────────

    def _build_model(self, seq_len: int, n_features: int):
        """Build LSTM(64→32) autoencoder as described in paper Section IV-C."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.random.set_seed(self.random_state)
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTMAutoencoderBaseline.\n"
                "Install with:  pip install tensorflow"
            )

        enc_units, lat_units = self.lstm_units  # e.g. (64, 32)

        inp = keras.Input(shape=(seq_len, n_features))

        # Encoder
        x = keras.layers.LSTM(enc_units, return_sequences=True)(inp)
        x = keras.layers.LSTM(lat_units, return_sequences=False)(x)

        # Decoder
        x = keras.layers.RepeatVector(seq_len)(x)
        x = keras.layers.LSTM(lat_units, return_sequences=True)(x)
        x = keras.layers.LSTM(enc_units, return_sequences=True)(x)
        out = keras.layers.TimeDistributed(
            keras.layers.Dense(n_features)
        )(x)

        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        return model

    # ── Correct interface: raw [N, W, C] windows ──────────────────────────────

    def fit_raw(self, X_win: np.ndarray, y: np.ndarray):
        """
        Train the LSTM autoencoder on raw sensor windows.

        Parameters
        ----------
        X_win : np.ndarray, shape [n_windows, W, C]
            Raw windowed sensor data — NOT the flat feature matrix.
        y     : np.ndarray, shape [n_windows]
            Window-level binary labels (0=normal, 1=anomaly).

        Returns
        -------
        self
        """
        import warnings
        warnings.filterwarnings("ignore")

        tracemalloc.start()
        t0 = time.perf_counter()

        n_windows, W, C = X_win.shape
        X_win = X_win.astype(np.float32)

        # Train exclusively on normal windows to learn reconstruction
        X_normal = X_win[y == 0]
        print(
            f"  [LSTM AE] Training on {len(X_normal)} normal windows "
            f"| shape=({W}, {C}) | epochs={self.epochs}"
        )

        self.model = self._build_model(seq_len=W, n_features=C)
        self.model.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )

        # Compute reconstruction-error threshold on ALL training windows
        recon = self.model.predict(X_win, verbose=0)           # [N, W, C]
        mse   = np.mean((X_win - recon) ** 2, axis=(1, 2))    # [N]
        self.threshold_ = float(np.percentile(mse, self.threshold_pct))
        print(
            f"  [LSTM AE] MSE threshold (p{self.threshold_pct}): "
            f"{self.threshold_:.6f}"
        )

        self.train_time_ = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.ram_mb_ = peak / 1024 / 1024

        print(
            f"[{self.name}] Train: {self.train_time_:.1f}s | "
            f"RAM: {self.ram_mb_:.1f} MB"
        )
        return self

    def predict_raw(self, X_win: np.ndarray, tau=None):
        """
        Return (Y_hat, P_hat) from raw [n_windows, W, C] windows.

        Parameters
        ----------
        X_win : np.ndarray, shape [n_windows, W, C]
        tau   : ignored (threshold is reconstruction-error based)

        Returns
        -------
        Y_hat : np.ndarray, shape [n_windows]  binary predictions
        P_hat : np.ndarray, shape [n_windows]  anomaly scores in [0, 1]
        """
        if self.model is None or self.threshold_ is None:
            raise RuntimeError("Call fit_raw() before predict_raw().")

        X_win = X_win.astype(np.float32)
        recon = self.model.predict(X_win, verbose=0)           # [N, W, C]
        mse   = np.mean((X_win - recon) ** 2, axis=(1, 2))    # [N]

        # Normalise MSE to [0, 1] for P_hat
        s_min, s_max = mse.min(), mse.max()
        P_hat = (mse - s_min) / (s_max - s_min + 1e-9)

        Y_hat = (mse >= self.threshold_).astype(int)
        return Y_hat, P_hat

    def evaluate_raw(self, X_win: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Full metrics dict from raw [n_windows, W, C] windows.

        Returns
        -------
        dict with same keys as compute_metrics() + train_time_s, ram_mb,
        infer_us_per_sample.
        """
        t0 = time.perf_counter()
        Y_hat, P_hat = self.predict_raw(X_win)
        infer_time = (time.perf_counter() - t0) / max(len(Y_hat), 1) * 1e6

        metrics = compute_metrics(y_true, Y_hat, P_hat)
        metrics["train_time_s"]        = round(self.train_time_, 2) if self.train_time_ else None
        metrics["infer_us_per_sample"] = round(infer_time, 2)
        metrics["ram_mb"]              = round(self.ram_mb_, 1) if self.ram_mb_ else None
        return metrics

    # ── Compatibility shim: flat-feature interface ────────────────────────────
    # Retained so the class still works when called through the shared
    # _BaselineWrapper.fit() / evaluate() interface (e.g. in ablation loops
    # that pass the flat Phi matrix).  Results from this path DO NOT match
    # the paper; use fit_raw / predict_raw for faithful reproduction.

    def _fit(self, X, y):
        """
        Shim: reshape flat feature matrix → pseudo-sequences and train.

        WARNING: This path does NOT reproduce paper results.
                 Use fit_raw(X_windows, y) for correct LSTM behaviour.
        """
        import warnings
        warnings.warn(
            "LSTMAutoencoderBaseline._fit() received a flat feature matrix. "
            "This does NOT reproduce paper results.  "
            "Call fit_raw(X_windows, y) with raw [N, W, C] windows instead.",
            UserWarning,
            stacklevel=2,
        )
        n_samples, n_feats = X.shape
        # Treat each feature as 1 timestep (not a real sequence)
        X_3d = X.reshape(n_samples, n_feats, 1).astype(np.float32)
        X_normal = X_3d[y == 0]

        self.model = self._build_model(seq_len=n_feats, n_features=1)
        self.model.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )

        recon = self.model.predict(X_3d, verbose=0)
        mse   = np.mean((X_3d - recon) ** 2, axis=(1, 2))
        self.threshold_ = float(np.percentile(mse, self.threshold_pct))

    def predict(self, X, tau=None):
        """Shim: predict from flat feature matrix. See _fit() warning."""
        n_samples, n_feats = X.shape
        X_3d  = X.reshape(n_samples, n_feats, 1).astype(np.float32)
        recon = self.model.predict(X_3d, verbose=0)
        mse   = np.mean((X_3d - recon) ** 2, axis=(1, 2))

        s_min, s_max = mse.min(), mse.max()
        P_hat = (mse - s_min) / (s_max - s_min + 1e-9)
        Y_hat = (mse >= self.threshold_).astype(int)
        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 7. Focal-Loss XGBoost (no SMOTE)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLossXGBoostBaseline(_BaselineWrapper):
    """
    XGBoost with focal loss objective — no SMOTE oversampling.

    Tests whether loss-level imbalance correction can replace oversampling.
    Focal loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Following: Lin et al., IEEE TPAMI 2020.

    FIX: `seed` key now correctly passed to XGBoost params dict.
    """

    name = "Focal-Loss XGBoost"

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        alpha=0.25,
        gamma=2.0,
        random_state=42,
    ):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.alpha         = alpha
        self.gamma         = gamma
        self.random_state  = random_state
        self.model         = None
        self.tau           = 0.5

    def _focal_loss_grad_hess(self, y_pred, dtrain):
        """Custom XGBoost focal loss gradient and hessian."""
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))   # sigmoid

        pt      = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        focal_w = alpha_t * (1 - pt) ** self.gamma

        grad = -focal_w * (y_true - p)
        hess =  focal_w * p * (1 - p)
        return grad, hess

    def _fit(self, X, y):
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "max_depth"    : self.max_depth,
            "learning_rate": self.learning_rate,
            "objective"    : "binary:logistic",
            "eval_metric"  : "logloss",
            "tree_method"  : "hist",
            "seed"         : self.random_state,   # FIX: was missing in some versions
            "nthread"      : -1,
        }
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=self._focal_loss_grad_hess,
            verbose_eval=False,
        )

    def predict(self, X, tau=None):
        import xgboost as xgb
        tau   = tau if tau is not None else self.tau
        dtest = xgb.DMatrix(X)
        P_hat = self.model.predict(dtest)
        Y_hat = (P_hat >= tau).astype(int)
        return Y_hat, P_hat


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_all_baselines(random_state=42) -> dict:
    """
    Return a dict of all 7 baselines with paper hyperparameters.

    Note: LSTMAutoencoderBaseline must be trained via .fit_raw() for correct
    paper-faithful results. See class docstring for details.

    Returns
    -------
    dict mapping name → baseline instance
    """
    return {
        "Isolation Forest"          : IsolationForestBaseline(random_state=random_state),
        "One-Class SVM"             : OneClassSVMBaseline(),
        "LightGBM"                  : LightGBMBaseline(random_state=random_state),
        "XGBoost (standalone)"      : XGBoostBaseline(random_state=random_state),
        "Random Forest (standalone)": RandomForestBaseline(random_state=random_state),
        "LSTM Autoencoder"          : LSTMAutoencoderBaseline(random_state=random_state),
        "Focal-Loss XGBoost"        : FocalLossXGBoostBaseline(random_state=random_state),
    }