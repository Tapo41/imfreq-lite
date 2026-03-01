# ImFREQ-Lite 

> **ImFREQ-Lite: A Lightweight Spectral-Statistical Framework for IoT Anomaly Detection in Smart City Environments**

---

## 📄 Paper Reference

> Basabdutta Konar, Tapojita Kar,  
> *"ImFREQ-Lite: A Lightweight Spectral-Statistical Framework for IoT Anomaly Detection in Smart City Environments"*,  
> Department of Computer Science and Engineering, Institute of Engineering and Management, Kolkata, India.

---

## 🗂️ Repository Structure

```
imfreq-lite/
│
├── notebooks/
│   └── ImFREQ_Lite_Full_Experiment.ipynb   ← Main Colab notebook (run this)
│
├── src/
│   ├── pipeline.py      ← ImFREQ-Lite pipeline + latency breakdown + save/load
│   ├── features.py      ← FFT + statistical feature extraction
│   ├── baselines.py     ← All 9 baselines (incl. TCN, Quant. Transformer)
│   ├── evaluate.py      ← Metrics, t-test, prequential streaming evaluation
│   ├── streaming.py     ← Real-time streaming engine + ADWIN drift detection [NEW]
│   ├── arm_deploy.py    ← ARM hardware profiler + energy calculator [NEW]
│   └── utils.py         ← Data loading (NAB C=1), preprocessing, reproducibility
│
├── data/
│   └── README_data.md   ← Download instructions for ToN-IoT, SKAB, NAB
│
├── results/             ← CSV outputs from notebook
├── figures/             ← Generated plots
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Google Colab — Free Tier)

```bash
# Option A: Open in Colab, then run all cells
# Option B: Local
git clone https://github.com/Tapo41/imfreq-lite.git
cd imfreq-lite
pip install -r requirements.txt
jupyter notebook notebooks/ImFREQ_Lite_Full_Experiment.ipynb
```

---

## 🔬 Method Summary

ImFREQ-Lite processes IoT sensor streams through five stages:

```
IoT Sensor Stream  (C=3 for ToN-IoT/SKAB  |  C=1 for NAB Yahoo S5)
      ↓
[S1] Sliding Window     (W = 512 samples, majority-vote labeling θ = 0.50)
      ↓
[S2] FFT Extraction     (top-K = 10 spectral magnitude bins, DC excluded)
      ↓
[S3] Statistical Fusion (μ, σ, γ₁, γ₂, RMS per channel)
                         φ ∈ ℝ⁴⁵ for C=3  |  φ ∈ ℝ¹⁵ for C=1
      ↓
[S4] SMOTE Oversampling (post-windowing, ratio = 0.25, k_s = 5)
      ↓
[S5] RF + XGBoost Soft-Voting Ensemble   (τ tuned on validation fold)
      ↓
Anomaly Label + Streaming Pipeline (0.62 ms end-to-end on RPi 4)
```

---

## 📊 Key Results (ToN-IoT, 10 Runs)

| Method                | F1 (mean ± std)   | PR-AUC            | Train (s) |
|-----------------------|-------------------|-------------------|-----------|
| Isolation Forest      | 0.591 ± 0.019     | 0.611 ± 0.016     | 11 s      |
| LightGBM              | 0.861 ± 0.009     | 0.876 ± 0.008     | 15 s      |
| LSTM Autoencoder      | 0.797 ± 0.016     | 0.814 ± 0.013     | 3421 s    |
| TCN                   | 0.874 ± 0.011     | 0.889 ± 0.009     | 412 s     |
| Quant. Transformer    | 0.882 ± 0.010     | 0.896 ± 0.008     | 1843 s    |
| **ImFREQ-Lite (Ours)**| **0.891 ± 0.007** | **0.907 ± 0.005** | **57 s**  |

All improvements: p ≤ 0.041 (two-tailed paired t-test, 10 runs).

---

## 🍓 Raspberry Pi 4 Deployment (Table XII)

| Method             | Latency (μs)   | RAM (MB) | Power (mW) | Energy/inf (μJ) | Batt. (hrs) |
|--------------------|----------------|----------|------------|-----------------|-------------|
| ImFREQ-Lite (Ours) | **18.3 ± 1.4** | **121**  | **312**    | **5.71**        | **28.1**    |
| LightGBM           | 5.1 ± 0.6      | 51       | 218        | 1.11            | 40.2        |
| TCN                | 112.6 ± 6.7    | 291      | 674        | 75.9            | 13.1        |
| Quant. Transformer | 74.1 ± 4.3     | 203      | 541        | 40.1            | 16.3        |
| LSTM Autoencoder   | 187.4 ± 9.2    | 412      | 891        | 167.0           | 9.9         |

Battery life: 2500 mAh Li-Ion @ 3.7V (9.25 Wh), 1 sample/s continuous.

---

## 🌊 Streaming Evaluation (Table XIII)

| Dataset    | Offline F1 | Streaming F1 | Latency (ms) | Drift events |
|------------|-----------|--------------|--------------|--------------|
| ToN-IoT    | 0.891     | 0.874 ± 0.009 | 0.62 ± 0.04  | None (p=0.23) |
| SKAB       | 0.879     | 0.861 ± 0.011 | 0.59 ± 0.05  | None (p=0.31) |
| NAB S5(C=1)| 0.869     | 0.847 ± 0.013 | 0.61 ± 0.04  | 3 (mild)     |

Drift detection: ADWIN (δ=0.002) on per-window F1 residuals.

---

## 📦 Datasets

| Dataset       | Channels | Anomaly % | Notes                          |
|---------------|----------|-----------|--------------------------------|
| ToN-IoT       | C=3      | 3.8%      | temp, humidity, motion_detected|
| SKAB          | C=3      | 2.4%      | accel_x, accel_y, pressure     |
| NAB Yahoo S5  | **C=1**  | 2.1%      | **Native univariate — v3 fix** |

See `data/README_data.md` for download instructions.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt

# PyTorch (CPU, for TCN/Transformer baselines):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ADWIN drift detection (optional):
pip install river
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
"# ImFREQ-Lite" 
