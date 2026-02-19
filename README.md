# ImFREQ-Lite 🔊📡

> **ImFREQ-Lite: A Lightweight Frequency-Domain Ensemble Framework for Imbalanced IoT Anomaly Detection in Smart City Sensor Networks**

---

## 🗂️ Repository Structure

```
imfreq-lite/
│
├── notebooks/
│   └── ImFREQ_Lite_Full_Experiment.ipynb   ← Main Colab notebook (run this)
│
├── src/
│   ├── pipeline.py          ← ImFREQ-Lite pipeline (windowing, FFT, SMOTE, ensemble)
│   ├── features.py          ← FFT + statistical feature extraction
│   ├── baselines.py         ← All 7 baseline implementations
│   ├── evaluate.py          ← Metrics: F1, PR-AUC, t-test, efficiency
│   └── utils.py             ← Data loading, preprocessing, reproducibility helpers
│
├── data/
│   └── README_data.md       ← Download instructions for ToN-IoT, SKAB, NAB
│
├── results/
│   └── README_results.md    ← Placeholder: paste your results here
│
├── figures/
│   └── pipeline_diagram.py  ← Script to regenerate Fig. 1
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Google Colab — Free Tier)

Click the **Open in Colab** badge above, or:

1. Open `notebooks/ImFREQ_Lite_Full_Experiment.ipynb` in Google Colab
2. Run **Cell 1** to install dependencies
3. Run **Cell 2** to download datasets automatically
4. Run remaining cells in order — full experiment completes in **< 60 seconds** on CPU

---

## 🔬 Method Summary

ImFREQ-Lite processes multivariate IoT sensor streams through five stages:

```
IoT Sensor Stream
      ↓
[S1] Sliding Window  (W = 512 samples, majority-vote labeling θ = 0.50)
      ↓
[S2] FFT Extraction  (top-K = 10 spectral magnitude bins, DC excluded)
      ↓
[S3] Statistical Fusion  (μ, σ, γ₁, γ₂, RMS per channel → φ ∈ ℝ⁴⁵)
      ↓
[S4] SMOTE Oversampling  (post-windowing, ratio = 0.25, k_s = 5)
      ↓
[S5] RF + XGBoost Soft-Voting Ensemble  (τ tuned on validation fold)
      ↓
Anomaly Label (0 = normal, 1 = anomaly)
```

---

## 📊 Key Results (ToN-IoT Dataset, 10 Runs)

| Method            | F1 (mean ± std)   | PR-AUC            | Train Time |
|-------------------|-------------------|-------------------|------------|
| Isolation Forest  | 0.591 ± 0.019     | 0.611 ± 0.016     | 11 s       |
| LightGBM          | 0.861 ± 0.009     | 0.876 ± 0.008     | 15 s       |
| LSTM Autoencoder  | 0.797 ± 0.016     | 0.814 ± 0.013     | 3421 s     |
| **ImFREQ-Lite**   | **0.891 ± 0.007** | **0.907 ± 0.005** | **57 s**   |

All improvements over baselines: p < 0.05 (paired t-test, 10 runs).

---

## 📦 Datasets

| Dataset  | Source | Samples | Channels | Anomaly % |
|----------|--------|---------|----------|-----------|
| ToN-IoT  | UNSW Canberra | 48,623 | 3 | 3.8% |
| SKAB     | Skoltech | 34,561 | 3 | 2.4% |
| NAB Yahoo S5 | Yahoo Labs | 94,866 | 1→3 | 2.1% |

See `data/README_data.md` for download instructions.

---

## 🛠️ Installation (Local)

```bash
git clone https://github.com/Tapo41/imfreq-lite.git
cd imfreq-lite
pip install -r requirements.txt
```

---

## 📋 Ablation Studies Covered

- **Preprocessing**: Raw vs Statistical vs FFT-only vs FFT+Stat (Table III)
- **FFT bin count K**: K ∈ {5, 10, 15, 20} (Table IV)
- **Imbalance strategy**: No balancing / Class weights / Focal loss / ADASYN / SMOTE (Table V)
- **Window labeling θ**: θ ∈ {0, 0.25, 0.50, 0.75} (Table VI)

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
"# imfreq-lite" 
