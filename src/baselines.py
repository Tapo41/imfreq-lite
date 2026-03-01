"""
src/baselines.py
----------------
All 9 baseline implementations used in the ImFREQ-Lite v3 paper.

v3 Changes
----------
- Added Baseline 8: Temporal Convolutional Network (TCN)
- Added Baseline 9: Quantized Transformer (INT8 post-training quantization)
- get_all_baselines() updated to return all 9 methods.

Each baseline follows the same interface:
    baseline.fit(Phi_train, Y_train)
    baseline.predict(Phi_test, tau)  → (Y_hat, P_hat)
    baseline.evaluate(X_test, y_true, tau) → metrics dict
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
        """Find tau maximising minority-class F1 on a validation fold."""
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
        self.model = IsolationForest(n_estimators=n_estimators,
                                     contamination=contamination,
                                     random_state=random_state, n_jobs=-1)

    def _fit(self, X, y):
        self.model.fit(X)

    def predict(self, X, tau=None):
        raw  = self.model.predict(X)
        Y_hat = (raw == -1).astype(int)
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
        self.model.fit(X[y == 0])

    def predict(self, X, tau=None):
        raw   = self.model.predict(X)
        Y_hat = (raw == -1).astype(int)
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
        self.model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    is_unbalance=True, random_state=random_state,
                                    n_jobs=-1, verbose=-1)
        self.tau = 0.5

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, tau=None):
        tau   = tau if tau is not None else self.tau
        P_hat = self.model.predict_proba(X)[:, 1]
        return (P_hat >= tau).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 4. XGBoost (standalone)
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostBaseline(_BaselineWrapper):
    """XGBoost standalone (Chen & Guestrin, KDD 2016)."""
    name = "XGBoost (standalone)"

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 subsample=0.8, scale_pos_weight=4, random_state=42):
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            use_label_encoder=False, tree_method="hist",
            random_state=random_state, n_jobs=-1)
        self.tau = 0.5

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, tau=None):
        tau   = tau if tau is not None else self.tau
        P_hat = self.model.predict_proba(X)[:, 1]
        return (P_hat >= tau).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 5. Random Forest (standalone)
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestBaseline(_BaselineWrapper):
    """Random Forest standalone (Breiman, 2001)."""
    name = "Random Forest (standalone)"

    def __init__(self, n_estimators=100, max_depth=15, min_samples_leaf=5,
                 random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight="balanced",
            random_state=random_state, n_jobs=-1)
        self.tau = 0.5

    def _fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, tau=None):
        tau   = tau if tau is not None else self.tau
        P_hat = self.model.predict_proba(X)[:, 1]
        return (P_hat >= tau).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 6. LSTM Autoencoder
# ─────────────────────────────────────────────────────────────────────────────

class LSTMAutoencoderBaseline(_BaselineWrapper):
    """LSTM Autoencoder (Meidan et al., 2018). Requires TensorFlow."""
    name = "LSTM Autoencoder"

    def __init__(self, lstm_units=(64, 32), epochs=20, batch_size=64,
                 threshold_pct=95, random_state=42):
        self.lstm_units    = lstm_units
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.threshold_pct = threshold_pct
        self.random_state  = random_state
        self.model         = None
        self.threshold_    = None

    def _build_model(self, seq_len, n_features):
        try:
            import tensorflow as tf
            from tensorflow import keras
            tf.random.set_seed(self.random_state)
        except ImportError:
            raise ImportError("TensorFlow required. pip install tensorflow")

        inp = keras.Input(shape=(seq_len, n_features))
        x = keras.layers.LSTM(self.lstm_units[0], return_sequences=True)(inp)
        x = keras.layers.LSTM(self.lstm_units[1], return_sequences=False)(x)
        x = keras.layers.RepeatVector(seq_len)(x)
        x = keras.layers.LSTM(self.lstm_units[1], return_sequences=True)(x)
        x = keras.layers.LSTM(self.lstm_units[0], return_sequences=True)(x)
        out = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(x)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        return model

    def _fit(self, X, y):
        import warnings; warnings.filterwarnings("ignore")
        n_samples, n_feats = X.shape
        X_3d = X.reshape(n_samples, n_feats, 1)
        X_normal = X_3d[y == 0]
        self.model = self._build_model(seq_len=n_feats, n_features=1)
        self.model.fit(X_normal, X_normal, epochs=self.epochs,
                       batch_size=self.batch_size, validation_split=0.1, verbose=0)
        recon = self.model.predict(X_3d, verbose=0)
        mse   = np.mean((X_3d - recon) ** 2, axis=(1, 2))
        self.threshold_ = float(np.percentile(mse, self.threshold_pct))

    def predict(self, X, tau=None):
        n_samples, n_feats = X.shape
        X_3d  = X.reshape(n_samples, n_feats, 1)
        recon = self.model.predict(X_3d, verbose=0)
        mse   = np.mean((X_3d - recon) ** 2, axis=(1, 2))
        s_min, s_max = mse.min(), mse.max()
        P_hat = (mse - s_min) / (s_max - s_min + 1e-9)
        return (mse >= self.threshold_).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 7. Focal-Loss XGBoost (no SMOTE)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLossXGBoostBaseline(_BaselineWrapper):
    """
    XGBoost with focal loss objective — no SMOTE oversampling.
    Focal loss: FL(p_t) = -α(1−p_t)^γ log(p_t). Lin et al., TPAMI 2020.
    """
    name = "Focal-Loss XGBoost"

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 alpha=0.25, gamma=2.0, random_state=42):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.alpha         = alpha
        self.gamma         = gamma
        self.random_state  = random_state
        self.model         = None
        self.tau           = 0.5

    def _focal_loss_grad_hess(self, y_pred, dtrain):
        y_true = dtrain.get_label()
        p      = 1.0 / (1.0 + np.exp(-y_pred))
        pt     = np.where(y_true == 1, p, 1 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1 - self.alpha)
        fw      = alpha_t * (1 - pt) ** self.gamma
        return -fw * (y_true - p), fw * p * (1 - p)

    def _fit(self, X, y):
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        params = dict(max_depth=self.max_depth, learning_rate=self.learning_rate,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", seed=self.random_state)
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators,
                               obj=self._focal_loss_grad_hess, verbose_eval=False)

    def predict(self, X, tau=None):
        import xgboost as xgb
        tau   = tau if tau is not None else self.tau
        P_hat = self.model.predict(xgb.DMatrix(X))
        return (P_hat >= tau).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 8. Temporal Convolutional Network (TCN)  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

class TCNBaseline(_BaselineWrapper):
    """
    Temporal Convolutional Network baseline (Bai et al., arXiv 2018).

    Architecture: 3 dilated causal conv layers (channels=[64,64,32],
    kernel_size=3, dropout=0.2). Trained for 50 epochs.

    Requires PyTorch. Install: pip install torch

    Note: The flat feature vector φ is treated as a 1D temporal sequence
    (length = feature_dim, 1 channel). For proper sequence modelling you
    would pass raw windows; this shared-interface implementation provides
    a fair comparison on equal input representations.
    """
    name = "TCN"

    def __init__(self, channels=(64, 64, 32), kernel_size=3, dropout=0.2,
                 epochs=50, batch_size=64, lr=1e-3, random_state=42):
        self.channels     = channels
        self.kernel_size  = kernel_size
        self.dropout      = dropout
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self.model        = None
        self.tau          = 0.5
        self._feat_dim    = None

    def _build_model(self, feat_dim):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required for TCNBaseline. pip install torch")

        class _CausalConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch, k, dilation, drop):
                super().__init__()
                padding = (k - 1) * dilation
                self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation, padding=padding)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(drop)
                self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

            def forward(self, x):
                out = self.drop(self.relu(self.conv(x)[:, :, :-self.conv.padding[0]]
                                          if self.conv.padding[0] > 0 else self.conv(x)))
                return self.relu(out + self.res(x))

        class _TCNClassifier(nn.Module):
            def __init__(self, in_dim, channels, k, drop):
                super().__init__()
                layers, in_ch = [], 1
                for i, out_ch in enumerate(channels):
                    layers.append(_CausalConvBlock(in_ch, out_ch, k, dilation=2**i, drop=drop))
                    in_ch = out_ch
                self.tcn = nn.Sequential(*layers)
                self.fc  = nn.Linear(channels[-1], 1)

            def forward(self, x):
                # x: [B, 1, L]
                out = self.tcn(x)          # [B, C_last, L]
                out = out[:, :, -1]        # last timestep
                return torch.sigmoid(self.fc(out)).squeeze(-1)

        return _TCNClassifier(feat_dim, self.channels, self.kernel_size, self.dropout)

    def _fit(self, X, y):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch required. pip install torch")

        torch.manual_seed(self.random_state)
        self._feat_dim = X.shape[1]
        self.model = self._build_model(self._feat_dim)

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N,1,L]
        yt = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Weighted BCE for class imbalance
        pos_w = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)],
                              dtype=torch.float32)
        criterion = nn.BCELoss(reduction="none")

        self.model.train()
        for ep in range(self.epochs):
            for xb, yb in loader:
                optim.zero_grad()
                pred = self.model(xb)
                w    = torch.where(yb == 1, pos_w, torch.ones(1))
                loss = (criterion(pred, yb) * w).mean()
                loss.backward()
                optim.step()

        self.model.eval()

    def predict(self, X, tau=None):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required.")

        tau = tau if tau is not None else self.tau
        self.model.eval()
        with torch.no_grad():
            Xt    = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            P_hat = self.model(Xt).numpy()
        return (P_hat >= tau).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# 9. Quantized Transformer  ← NEW in v3
# ─────────────────────────────────────────────────────────────────────────────

class QuantizedTransformerBaseline(_BaselineWrapper):
    """
    Lightweight Transformer with INT8 post-training quantization.

    Architecture: 2-layer encoder, 4 attention heads, d_model=64.
    Quantization: PyTorch dynamic INT8 quantization applied post-training.
    Trained for 30 epochs.

    Requires PyTorch. Install: pip install torch

    Reference: Vaswani et al. (NIPS 2017); Zafrir et al. Q8BERT (NeurIPS 2019).
    """
    name = "Quant. Transformer"

    def __init__(self, d_model=64, nhead=4, num_layers=2, epochs=30,
                 batch_size=64, lr=1e-3, random_state=42):
        self.d_model      = d_model
        self.nhead        = nhead
        self.num_layers   = num_layers
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state
        self.model        = None
        self.tau          = 0.5

    def _build_model(self, feat_dim):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required. pip install torch")

        class _TransformerClassifier(nn.Module):
            def __init__(self, in_dim, d_model, nhead, num_layers):
                super().__init__()
                self.proj    = nn.Linear(in_dim, d_model)
                enc_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                          dim_feedforward=d_model * 2,
                                                          dropout=0.1, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.fc      = nn.Linear(d_model, 1)

            def forward(self, x):
                # x: [B, L=1, in_dim]
                x   = self.proj(x)          # [B, 1, d_model]
                out = self.encoder(x)        # [B, 1, d_model]
                return torch.sigmoid(self.fc(out[:, 0, :])).squeeze(-1)

        return _TransformerClassifier(feat_dim, self.d_model, self.nhead, self.num_layers)

    def _fit(self, X, y):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError("PyTorch required. pip install torch")

        torch.manual_seed(self.random_state)
        feat_dim   = X.shape[1]
        self.model = self._build_model(feat_dim)

        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, feat_dim]
        yt = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=self.batch_size, shuffle=True)

        pos_w    = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)],
                                dtype=torch.float32)
        criterion = nn.BCELoss(reduction="none")
        optim     = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optim.zero_grad()
                pred = self.model(xb)
                w    = torch.where(yb == 1, pos_w, torch.ones(1))
                (criterion(pred, yb) * w).mean().backward()
                optim.step()

        # Apply dynamic INT8 quantization (CPU inference)
        self.model.eval()
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            print(f"  [{self.name}] INT8 quantization applied.")
        except Exception as e:
            print(f"  [{self.name}] Quantization skipped: {e}")

    def predict(self, X, tau=None):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required.")

        tau = tau if tau is not None else self.tau
        self.model.eval()
        with torch.no_grad():
            Xt    = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            P_hat = self.model(Xt).numpy()
        return (P_hat >= tau).astype(int), P_hat


# ─────────────────────────────────────────────────────────────────────────────
# Factory function — returns all 9 baselines
# ─────────────────────────────────────────────────────────────────────────────

def get_all_baselines(random_state: int = 42) -> dict:
    """
    Return a dict of all 9 baselines with paper hyperparameters.

    v3: Added TCN and Quantized Transformer.

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
        "TCN"                       : TCNBaseline(random_state=random_state),
        "Quant. Transformer"        : QuantizedTransformerBaseline(random_state=random_state),
    }
