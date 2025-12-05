# EWMA + VIX Volatility Forecasting Report

## 1. Executive Summary

This project implements a robust volatility forecasting model for SMI stocks using **EWMA (Exponentially Weighted Moving Average)** volatility as the target variable, enhanced with the **VIX (CBOE Volatility Index)** as an external macro feature.

**Key Results:**
- **Model:** Amazon Chronos-2 (Zero-Shot)
- **Target:** EWMA Volatility ($\lambda=0.94$)
- **Optimal Features:** `ewma_vol_lag_1` + `vix_lag_1` (K=2)
- **Performance:**
  - **MAE:** `0.00130` (11.3% improvement over baseline)
  - **R²:** `94.3%` (High explanatory power)
  - **Consistency:** Outperforms CTC volatility significantly across all stocks.

---

## 2. Methodology

### 2.1 Target Variable: EWMA Volatility
Instead of using noisy Close-to-Close (CTC) volatility, we use EWMA volatility, which is the industry standard (RiskMetrics). It is calculated recursively:

$$ \sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda) r_t^2 $$

Where:
- $\lambda = 0.94$ (Decay factor)
- $r_t$ = Log return at time $t$

**Advantages:**
- **Smoothness:** Reduces noise and outliers.
- **Responsiveness:** Reacts quickly to market shocks.
- **Predictability:** The recursive nature makes it highly autoregressive and easier to forecast.

### 2.2 External Feature: VIX
The VIX ("Fear Index") measures the market's expectation of 30-day volatility.
- **Impact:** Adding VIX as a feature improves the Mean Absolute Error (MAE) by **11.3%**.
- **Why it works:** While EWMA captures the *internal* stock dynamics, VIX captures the *global* market sentiment, providing a valuable signal during market stress.

---

## 3. Ablation Study Results

We conducted a comprehensive ablation study testing feature sets from K=1 to K=20.

| Rank | Configuration | Features | MAE | R² |
|------|---------------|----------|-----|----|
| 🥇 **1** | **EWMA K=2 + VIX** | `ewma_vol_lag_1`, `vix_lag_1` | **0.00130** | **94.3%** |
| 🥈 2 | EWMA K=1 | `ewma_vol_lag_1` | 0.00147 | 94.1% |
| 🥉 3 | EWMA K=6 | Top-6 Lags | 0.00165 | 91.2% |
| 4 | CTC Top-6 | Top-6 CTC Lags | 0.00223 | 87.1% |

**Conclusion:** The simplest model with just the previous day's EWMA and the VIX is the best. Adding more lags (K>2) leads to overfitting and degrades performance.

---

## 4. Final Model Performance (All Stocks)

Evaluated on 17 SMI stocks (Amrize excluded due to insufficient history).

| Stock | MAE | R² |
|-------|-----|----|
| **Swisscom** | 0.00029 | 97.4% |
| **Sika** | 0.00038 | 97.4% |
| **Novartis** | 0.00044 | 97.4% |
| **Logitech** | 0.00078 | 97.4% |
| ... | ... | ... |
| **AGGREGATE** | **0.00130** | **94.3%** |

The model shows exceptional consistency, with R² > 90% for almost all major stocks.

---

## 5. Project Structure

The `ewma_volatility/` folder contains all necessary files:

- **`inference_ewma.py`**: Main script for running predictions on all stocks.
- **`inference_ewma.ipynb`**: Jupyter Notebook version for interactive analysis.
- **`plots/`**: Contains forecast plots for every stock.
- **`ablation_results/`**: Detailed CSVs and plots from the ablation study.
- **`ewma_vix_all_stocks_metrics.csv`**: Detailed metrics for the final model.

## 6. Usage

To run the inference and generate new plots:

```bash
cd ewma_volatility
python inference_ewma.py
```

This will:
1. Load stock data from `../processed_data/`
2. Load VIX data from `../data_fred/`
3. Run Chronos-2 inference
4. Save plots to `plots/` and metrics to `ewma_vix_all_stocks_metrics.csv`
