# Experimental Results

After running `notebooks/ImFREQ_Lite_Full_Experiment.ipynb`,
the following CSV files will be automatically saved here:

| File | Contents |
|---|---|
| `table3_preprocessing_ablation.csv` | Table III — P1/P2/P3/P4 comparison |
| `table4_k_ablation.csv` | Table IV — FFT bin count K ∈ {5,10,15,20} |
| `table5_imbalance_ablation.csv` | Table V — SMOTE vs ADASYN vs focal loss |
| `table6_theta_ablation.csv` | Table VI — Window labeling threshold θ |
| `table7_baseline_comparison.csv` | Table VII — All 7 baselines + ImFREQ-Lite |
| `table8_cross_dataset.csv` | Table VIII — ToN-IoT / SKAB / NAB results |
| `table9_efficiency.csv` | Table IX — Training time, RAM, FLOPs |

## Reproducing Paper Results

1. Download datasets (see `data/README_data.md`)
2. Set `USE_SYNTHETIC = False` in Cell 3 of the notebook
3. Run all cells — results auto-save to this folder
4. Replace reported numbers in the paper draft with your actual output

## Hardware Used in Paper

- **Platform:** Google Colab free-tier
- **CPU:** Intel Xeon (shared node, ~25 GB RAM)
- **Python:** 3.10
- **No GPU required**
