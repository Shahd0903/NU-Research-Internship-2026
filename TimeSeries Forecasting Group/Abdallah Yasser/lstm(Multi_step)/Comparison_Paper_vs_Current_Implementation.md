# Model Architecture & Multi-Step Forecasting Comparison: Paper vs. Current Implementation

## 1. Overview & Reference Paper

* **Reference Paper:** *Temporal Convolutional Networks with RNN approach for chaotic time series prediction* (Dudukcu et al., *Applied Soft Computing*, Vol. 133, 2023). DOI: [10.1016/j.asoc.2022.109945](https://doi.org/10.1016/j.asoc.2022.109945)
* **Current Implementation:** Direct Multi-Output Standalone LSTM for ECG Time Series Forecasting.
* **Dataset (Both):** MIT-BIH Arrhythmia Database — 21 patient records (excluding patients 111 & 118), sampling rate $f_s = 360\text{ Hz}$.

---

## 2. Dataset & Data Split Comparison

| Feature | Reference Paper (Dudukcu et al., 2023) | Our Implementation |
| :--- | :--- | :--- |
| **Database** | MIT-BIH Arrhythmia Database | MIT-BIH Arrhythmia Database |
| **No. of Patients** | **21 records** (excl. 111 & 118) | **21 records** (excl. 111 & 118) |
| **Sampling Rate** | **360 Hz** | **360 Hz** |
| **Total Samples / Patient** | **100,000 steps** | **100,000 steps** |
| **Train Split** | **40,000 steps** | **40,000 steps** |
| **Validation Split** | **10,000 steps** | **10,000 steps** |
| **Test Split** | **50,000 steps** | **50,000 steps** |
| **Normalization** | MinMax $[0, 1]$, fitted on training data | MinMax $[0, 1]$ (`MinMaxScaler`), fitted strictly on training data |

---

## 3. Input / Output Steps Comparison

| Feature | Reference Paper (Dudukcu et al., 2023) | Our Implementation |
| :--- | :--- | :--- |
| **Input Steps (Lookback)** | **10 steps** ($\approx 27.8\text{ ms}$ of ECG history) | **10 steps** ($\approx 27.8\text{ ms}$ of ECG history) |
| **Output Steps (Horizon)** | **1 step** ($H = 1$, one-step-ahead, $\approx 2.78\text{ ms}$) | **Multi-Step:** **10, 20, and 50 steps** ($\approx 27.8\text{ ms}$, $55.6\text{ ms}$, $138.9\text{ ms}$) |
| **Task Type** | Single-step-ahead continuous forecasting | Direct multi-step vector output forecasting |

---

## 4. Training Configuration Comparison

| Hyperparameter | Reference Paper (Dudukcu et al., 2023) | Our Implementation |
| :--- | :--- | :--- |
| **Max Epochs** | Not explicitly stated (early stopping used) | **30 epochs** |
| **Batch Size** | **256** | **256** |
| **Optimizer** | **Adam** | **Adam** |
| **Loss Function** | **Mean Squared Error (MSE)** | **Mean Squared Error (MSE)** |
| **Early Stopping** | Validation loss monitoring (patience not specified) | `patience=5`, monitors `val_loss`, `restore_best_weights=True` |
| **Training Scheme** | **Intra-patient** (independent model per patient) | **Intra-patient** (independent model per patient per horizon; **63 models total**) |

---

## 5. Empirical Performance Comparison (ECG — MIT-BIH)

### 5.1 Average Summary

| Model | Prediction Horizon ($H$) | Time Window | RMSE | MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Paper: Standalone LSTM** (Table 7, $H=1$) | **1 step** | $\approx 2.78\text{ ms}$ | **0.0096** | **0.0060** |
| **Paper: Proposed TCN-LSTM** (Table 7, $H=1$) | **1 step** | $\approx 2.78\text{ ms}$ | **0.0082** | **0.0051** |
| **Our Standalone LSTM** ($H=10$) | **10 steps** | $\approx 27.8\text{ ms}$ | **0.0418** | **0.0205** |
| **Our Standalone LSTM** ($H=20$) | **20 steps** | $\approx 55.6\text{ ms}$ | **0.0637** | **0.0310** |
| **Our Standalone LSTM** ($H=50$) | **50 steps** | $\approx 138.9\text{ ms}$ | **0.0842** | **0.0478** |

---

## 6. Critical Analysis & Key Insights

1. **Horizon Expansion ($H = 1 \rightarrow H = 50$) — The Core Difference:**
   * The paper forecasts only **one step ahead** at a time ($H=1$), which is the simplest forecasting task. Multi-step forecasting ($H=10, 20, 50$) is a fundamentally harder problem: errors compound, and uncertainty grows with the horizon.
   * Our standalone LSTM achieves $R^2 = 0.8241$ at $H=10$, degrading to $R^2 = 0.3596$ at $H=50$, showing the direct impact of horizon expansion.
   * Even at the nearest comparable horizon ($H=10$), our LSTM MAE (avg **0.0205**) is ~$3.4\times$ higher than the paper's LSTM at $H=1$ (avg **0.0060**), reflecting the added difficulty of multi-step prediction.

2. **Loss Function Limitation (MSE for Long Horizons):**
   * Both the paper and our baseline use standard **MSE loss**.
   * At $H=50$, the quadratic MSE penalty under high uncertainty causes the model to learn a conservative mean-prediction strategy, **flattening sharp QRS complexes** — which are clinically critical features. This motivates future work using **DILATE Loss** (shape + temporal distortion loss).

