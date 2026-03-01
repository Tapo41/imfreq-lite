"""
src/arm_deploy.py
-----------------
ARM hardware deployment profiler for ImFREQ-Lite.

New in v3 — validates edge-deployment claims on real hardware.

Provides tools to:
    1. Serialise a trained model to disk (joblib, ~3.7 MB).
    2. Measure on-device inference latency, RAM footprint, and power draw.
    3. Compute energy-per-inference (μJ) and project battery life.
    4. Compare ImFREQ-Lite against baselines under ARM hardware constraints.

Tested on: Raspberry Pi 4 Model B (BCM2711, Cortex-A72 @ 1.8 GHz, 4 GB LPDDR4)
OS: Raspberry Pi OS Lite 64-bit (Debian Bullseye), Python 3.10

Paper Table XII reproduces the following measurements:
    ImFREQ-Lite : 18.3 ± 1.4 μs | 121 MB | 312 mW | 5.71 μJ | 28.1 hrs
    LightGBM    :  5.1 ± 0.6 μs |  51 MB | 218 mW | 1.11 μJ | 40.2 hrs
    TCN         : 112.6 ± 6.7 μs | 291 MB | 674 mW | 75.9 μJ | 13.1 hrs

Usage (on Raspberry Pi 4)
-------------------------
    from src.arm_deploy import ARMProfiler, save_model, load_model

    # 1. On development machine: train and save
    model.save("models/imfreq_lite_toniot.joblib")

    # 2. Copy .joblib file to Raspberry Pi, then:
    model = load_model("models/imfreq_lite_toniot.joblib")
    profiler = ARMProfiler(model)
    results  = profiler.profile(X_test, Y_test, n_trials=1000)
    energy   = profiler.energy_profile(results["mean_latency_us"],
                                        device_power_mw=312.0)
"""

from __future__ import annotations

import os
import time
import tracemalloc
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> float:
    """
    Serialise a fitted ImFREQLite model to disk using joblib (compress=3).

    Returns serialised file size in MB.

    Parameters
    ----------
    model : fitted ImFREQLite instance
    path  : output .joblib file path (e.g. "models/imfreq_toniot.joblib")
    """
    import joblib
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    payload = {
        "rf" : model.rf, "xgb": model.xgb,
        "W"  : model.W,  "K"  : model.K, "theta": model.theta,
        "tau": model.tau, "smote_ratio": model.smote_ratio,
        "random_state": model.random_state,
    }
    joblib.dump(payload, path, compress=3)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[save_model] → {path}  ({size_mb:.2f} MB)")
    return size_mb


def load_model(path: str):
    """
    Load a serialised ImFREQLite model from disk.

    Parameters
    ----------
    path : path to .joblib file

    Returns
    -------
    ImFREQLite instance ready for inference
    """
    import joblib
    from src.pipeline import ImFREQLite

    payload = joblib.load(path)
    model   = ImFREQLite(
        W=payload["W"], K=payload["K"], theta=payload["theta"],
        tau=payload["tau"], smote_ratio=payload["smote_ratio"],
        random_state=payload["random_state"], verbose=False
    )
    model.rf  = payload["rf"]
    model.xgb = payload["xgb"]
    print(f"[load_model] ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ARM Profiler
# ─────────────────────────────────────────────────────────────────────────────

class ARMProfiler:
    """
    Profiles ImFREQ-Lite inference on ARM hardware (Raspberry Pi 4).

    Measures:
        - Per-sample inference latency (μs) via time.perf_counter()
        - Peak RAM footprint (MB) via psutil or tracemalloc
        - Energy per inference (μJ) = Power × Latency
        - Projected battery life (hrs) for a 2500 mAh cell @ 3.7V

    Power consumption must be measured externally (e.g., FNIRSI FNB58 USB
    power meter in series with the Pi's USB-C supply) and passed manually.

    Parameters
    ----------
    model        : loaded ImFREQLite instance
    n_warmup     : warm-up inference calls before timing (default 100)
    """

    def __init__(self, model, n_warmup: int = 100):
        self.model    = model
        self.n_warmup = n_warmup

    def profile(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_trials: int = 1000,
    ) -> dict:
        """
        Measure per-sample inference latency and RAM on current hardware.

        Parameters
        ----------
        X        : np.ndarray [N, C]  sensor stream for test windows
        Y        : np.ndarray [N]     point-level labels
        n_trials : number of timing iterations (default 1000)

        Returns
        -------
        dict:
            mean_latency_us, std_latency_us, max_latency_us, p95_latency_us,
            ram_mb (peak), model_size_mb (if serialised),
            n_trials
        """
        from src.features import extract_features

        # Build windows
        n_windows = len(X) // self.model.W
        windows   = [X[k * self.model.W:(k + 1) * self.model.W, :]
                     for k in range(n_windows)]
        # Ensure enough windows
        while len(windows) < self.n_warmup + n_trials:
            windows = windows * 2
        warm    = windows[:self.n_warmup]
        trials  = windows[self.n_warmup: self.n_warmup + n_trials]

        # ── RAM measurement ───────────────────────────────────────────────────
        try:
            import psutil
            proc     = psutil.Process(os.getpid())
            ram_mb   = proc.memory_info().rss / 1024 / 1024
            _USE_PSUTIL = True
        except ImportError:
            tracemalloc.start()
            _USE_PSUTIL = False

        # ── Warm-up ───────────────────────────────────────────────────────────
        for w in warm:
            self.model.predict_single_window(w)

        # ── Timed trials ──────────────────────────────────────────────────────
        latencies = []
        for w in trials:
            t0 = time.perf_counter()
            self.model.predict_single_window(w)
            latencies.append((time.perf_counter() - t0) * 1e6)

        # ── Peak RAM ──────────────────────────────────────────────────────────
        if _USE_PSUTIL:
            ram_mb = proc.memory_info().rss / 1024 / 1024
        else:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_mb = peak / 1024 / 1024

        lat = np.array(latencies)
        result = {
            "mean_latency_us" : float(lat.mean()),
            "std_latency_us"  : float(lat.std()),
            "max_latency_us"  : float(lat.max()),
            "p95_latency_us"  : float(np.percentile(lat, 95)),
            "ram_mb"          : round(ram_mb, 1),
            "n_trials"        : n_trials,
            "hardware"        : _detect_hardware(),
        }

        print(f"\n[ARMProfiler Results]  ({result['hardware']})")
        print(f"  Latency : {lat.mean():.2f} ± {lat.std():.2f} μs "
              f"(max {lat.max():.2f} μs, p95 {np.percentile(lat,95):.2f} μs)")
        print(f"  RAM     : {ram_mb:.1f} MB")
        return result

    def energy_profile(
        self,
        mean_latency_us: float,
        device_power_mw: float,
        battery_capacity_wh: float = 9.25,
        sample_rate_hz: float = 1.0,
    ) -> dict:
        """
        Compute energy-per-inference and project battery life.

        Energy per inference: E (μJ) = P (W) × t (s) × 10^6
        Battery life at continuous operation:
            T (hrs) = Capacity (Wh) / Power (W)

        Parameters
        ----------
        mean_latency_us     : measured per-sample inference latency in μs
        device_power_mw     : measured active-inference power draw in mW
                              (from external USB power meter)
        battery_capacity_wh : cell capacity in Wh (default 9.25 = 2500 mAh @ 3.7V)
        sample_rate_hz      : sensor sampling rate (default 1.0 Hz)

        Returns
        -------
        dict: energy_per_infer_uj, energy_per_sample_nj, battery_life_hrs
        """
        t_s    = mean_latency_us * 1e-6
        p_w    = device_power_mw * 1e-3
        e_uj   = p_w * t_s * 1e6
        e_nj   = p_w * t_s * 1e9
        t_hrs  = battery_capacity_wh / p_w

        result = {
            "mean_latency_us"     : mean_latency_us,
            "device_power_mw"     : device_power_mw,
            "energy_per_infer_uj" : round(e_uj, 3),
            "energy_per_sample_nj": round(e_nj, 3),
            "battery_life_hrs"    : round(t_hrs, 1),
            "battery_capacity_wh" : battery_capacity_wh,
        }

        print(f"\n[Energy Profile]")
        print(f"  Power            : {device_power_mw:.0f} mW")
        print(f"  Latency          : {mean_latency_us:.2f} μs")
        print(f"  Energy/inference : {e_uj:.3f} μJ")
        print(f"  Energy/sample    : {e_nj:.3f} nJ")
        print(f"  Battery life     : {t_hrs:.1f} hrs  "
              f"(2500 mAh @ 3.7V, {sample_rate_hz:.0f} Hz continuous)")
        return result

    def compare_methods(
        self,
        methods: dict,
        X: np.ndarray,
        Y: np.ndarray,
        n_trials: int = 500,
    ) -> dict:
        """
        Profile multiple loaded models on the same hardware.

        Parameters
        ----------
        methods  : dict mapping name → loaded model instance
        X, Y     : sensor stream for test windows
        n_trials : timing iterations per method

        Returns
        -------
        dict mapping name → profile result dict
        """
        results = {}
        for name, model in methods.items():
            print(f"\n── {name} ──")
            profiler     = ARMProfiler(model, n_warmup=50)
            profiler.model = model
            results[name] = profiler.profile(X, Y, n_trials=n_trials)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_hardware() -> str:
    """Return a hardware description string for logging."""
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip("\x00").strip()
            return model
    except (FileNotFoundError, IOError):
        pass
    try:
        import platform
        return f"{platform.processor()} ({platform.machine()})"
    except Exception:
        return "Unknown hardware"


def check_arm_feasibility(
    ram_mb: float,
    latency_us: float,
    ram_limit_mb: float = 256.0,
    latency_limit_us: float = 1000.0,
) -> dict:
    """
    Check whether a model meets edge-deployment feasibility criteria.

    Paper definition (Raspberry Pi 4 class):
        - RAM ≤ 256 MB
        - Inference latency ≤ 1000 μs (non-blocking at 1 Hz)

    Parameters
    ----------
    ram_mb           : measured RAM usage in MB
    latency_us       : measured inference latency in μs
    ram_limit_mb     : RAM threshold (default 256 MB for RPi 4)
    latency_limit_us : latency threshold (default 1000 μs)

    Returns
    -------
    dict: feasible (bool), ram_ok, latency_ok, headroom_mb, headroom_us
    """
    ram_ok     = ram_mb     <= ram_limit_mb
    latency_ok = latency_us <= latency_limit_us
    return {
        "feasible"      : bool(ram_ok and latency_ok),
        "ram_ok"        : ram_ok,
        "latency_ok"    : latency_ok,
        "headroom_mb"   : round(ram_limit_mb     - ram_mb,     1),
        "headroom_us"   : round(latency_limit_us - latency_us, 2),
        "ram_mb"        : ram_mb,
        "latency_us"    : latency_us,
        "ram_limit_mb"  : ram_limit_mb,
        "latency_limit" : latency_limit_us,
    }
