# Dataset Download Instructions

This paper uses **three public benchmark datasets**.
All are freely available. Follow the steps below to download and place them
in the correct folder structure before running the notebook.

---

## 1. ToN-IoT (Sensor Sub-Stream)

**Paper citation:** Moustafa & Slay, IEEE MILCOM 2021  
**URL:** https://research.unsw.edu.au/projects/toniot-datasets

### Steps
1. Visit the URL above and fill in the access form (free, academic use).
2. Download **"Sensor_dataset.csv"** from the ToN-IoT Sensor files.
3. Place it at:
   ```
   data/ton_iot_sensor.csv
   ```
4. The file should contain columns including:
   `temp`, `humidity`, `motion_detected`, `label`

### Columns used in paper
| Column | Description |
|---|---|
| `temp` | Temperature sensor reading |
| `humidity` | Relative humidity sensor reading |
| `motion_detected` | Binary motion sensor (0/1) |
| `label` | Ground-truth anomaly label (0=normal, 1=anomaly) |

**Only these 3 sensor columns + label are used.**
Network traffic features are excluded (see paper Section IV-A).

### Anomaly rate
~3.8% of rows are labeled anomalous (attack-induced sensor anomalies).

---

## 2. SKAB — Skoltech Anomaly Benchmark

**Paper citation:** Katser & Kozitsin, Kaggle 2020  
**URL:** https://www.kaggle.com/datasets/dsv/1693952

### Steps
1. Install Kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Download:
   ```bash
   kaggle datasets download -d dsv/1693952
   unzip 1693952.zip -d data/skab/
   ```
3. Your folder should look like:
   ```
   data/skab/
       allbenchmarks/
           0.csv
           1.csv
           ...
       anomaly-free/
           anomaly-free.csv
   ```

### Columns used in paper
| Column | Description |
|---|---|
| `accelerometer_x` | X-axis pump accelerometer |
| `accelerometer_y` | Y-axis pump accelerometer |
| `pressure` | Pump pressure sensor |
| `anomaly` | Ground-truth label (0=normal, 1=anomaly) |

### Anomaly rate
~2.4% across all combined files.

---

## 3. NAB Yahoo S5 Anomaly Benchmark

**Paper citation:** Lavin & Ahmad, IEEE ICMLA 2015  
**URL:** https://github.com/numenta/NAB

### Steps
1. Clone the NAB repository:
   ```bash
   git clone https://github.com/numenta/NAB.git
   ```
2. Copy the Yahoo S5 data files:
   ```bash
   cp -r NAB/data/realYahoo/ data/nab_yahoo_s5/
   ```
3. Your folder should look like:
   ```
   data/nab_yahoo_s5/
       real_1.csv
       real_2.csv
       ...
       real_67.csv
   ```

### Format
Each CSV has columns: `timestamp`, `value`, `anomaly` (or similar).
The 49 series are concatenated and the single value channel is
replicated ×3 for multi-channel compatibility (see paper Section IV-A,
limitation acknowledged in Section V-H).

### Anomaly rate
~2.1% across all 49 concatenated series.

---

## Quick Test (No Download Needed)

To run the notebook **without downloading any data**, use the built-in
synthetic dataset generator:

```python
from src.utils import make_synthetic_iot
X, Y = make_synthetic_iot(n_samples=20000, n_channels=3, anomaly_rate=0.04)
```

This generates a realistic multivariate IoT stream with injected anomalies.
Use it to verify code correctness before running real experiments.

---

## Expected Directory Structure After Setup

```
data/
├── ton_iot_sensor.csv          ← ToN-IoT sensor file
├── skab/
│   ├── allbenchmarks/
│   │   ├── 0.csv
│   │   └── ...
│   └── anomaly-free/
│       └── anomaly-free.csv
├── nab_yahoo_s5/
│   ├── real_1.csv
│   └── ...
└── README_data.md              ← This file
```
