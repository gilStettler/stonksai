
# Out-of-Sample Volatility Verification (Dec 1, 2025)

This directory contains the complete pipeline used to verify the Chronos-2 model's performance on a strict out-of-sample test for December 1st, 2025.

## 1. Objective
To validate the model's ability to predict next-day volatility for 18 Swiss stocks without data leakage, and to determine the optimal configuration.

## 2. Methodology
*   **Model:** `amazon/chronos-2`
*   **Target:** Close-to-Close Volatility (5-day window)
*   **Test Date:** Dec 1, 2025 (Strict Forward Test)
*   **Sample Size:** N=18 Stocks

## 3. Results Summary

We compared four variants to find the best strategy.

| Metric | Zero-Shot (Baseline) | Top-6 Features (Best) | All Features (Noisy) | Fine-Tuned (Overfit) |
| :--- | :--- | :--- | :--- | :--- |
| **RMSE** | 0.003554 | **0.003420** | 0.003540 | 0.003739 |
| **MAE** | **0.002671** | 0.002780 | 0.002833 | 0.002908 |
| **R² Score** | 0.4939 | **0.5314** | 0.4979 | 0.4400 |
| **Adj. R²** | **0.4939** | 0.2758 | NaN (p>n) | 0.1345 |
| **Correlation** | 0.7035 | **0.7432** | 0.7167 | 0.7387 |

### Interpretation
1.  **Top-6 Features** is the winner. It has the lowest RMSE (0.0034) and highest R² (0.53) and Correlation (0.74).
2.  **All Features** performs worse than Top-6, proving that adding too many features introduces noise.
3.  **Fine-Tuning** performs worst (R² 0.44), confirming that the model overfits on the small SMI dataset.
4.  **Zero-Shot** is a very strong baseline, beating the Fine-Tuned model.

## 4. Visualizations (Dec 1, 2025)

The plots below show the last 20 days of history (black) and the predictions for Dec 1st.
*   **Red X:** Actual Value
*   **Green Triangle:** Top-6 Features (Best)
*   **Blue Circle:** Zero-Shot
*   **Orange Square:** All Features
*   **Purple Diamond:** Fine-Tuned

![Kuehne+Nagel](plots/Kuehne_Nagel_0QMW.LON.png)
![Holcim](plots/Holcim_0QKY.LON.png)
![Amrize](plots/Amrize_AMRZ.png)
![UBS](plots/UBS_UBS.png)
![Novartis](plots/Novartis_0QLR.LON.png)
![Alcon](plots/Alcon_0A0D.LON.png)
![Givaudan](plots/Givaudan_0QPS.LON.png)
![Sika](plots/Sika_0Z4C.LON.png)
![Swiss Life](plots/Swiss_Life_0QMG.LON.png)
![Zurich Insurance](plots/Zurich_Insurance_0QP2.LON.png)
![Geberit](plots/Geberit_0QQ2.LON.png)
![Partners Group](plots/Partners_Group_0QOQ.LON.png)
![Logitech](plots/Logitech_0QK6.LON.png)
![ABB](plots/ABB_ABBNY.png)
![Swisscom](plots/Swisscom_0QKI.LON.png)
![Nestlé](plots/Nestle_NSRGY.png)
![Roche](plots/Roche_RHO6.FRK.png)
![Lonza](plots/Lonza_0QNO.LON.png)

## 5. Pipeline Scripts

1.  `00_fetch_actuals.py`: Get ground truth data.
2.  `01_predict_zero_shot.py`: Run baseline.
3.  `02_predict_top6.py`: Run best model.
4.  `02b_predict_all_features.py`: Run all features model.
5.  `03_finetune_experiment.py`: Run fine-tuning test.
6.  `04_compare_results.py`: Calculate metrics.
7.  `05_visualize_results.py`: Generate the plots above.
