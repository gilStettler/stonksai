# Chronos-2 Ablation Study Report
**Date:** 2025-11-30  
**Model:** Amazon Chronos-2 (Zero-Shot)  
**Dataset:** 18 SMI Stocks (processed_data)  
**Target:** Daily Volatility (`ctc_vol`)

---

## Executive Summary

This ablation study evaluates the impact of different feature groups on stock volatility forecasting using Amazon's Chronos-2 model. **Technical indicators showed the strongest improvement (+10.86% MAE)**, while macro and peer features provided minimal gains.

---

## Methodology

### Configuration
- **Prediction Length:** 5 days ahead (1 week trading days)
- **Context Window:** 60 days lookback
- **Model:** `amazon/chronos-2` (zero-shot, no fine-tuning)
- **Evaluation Metrics:** MAE, MSE, RMSE
- **Data Preparation:** Business day reindexing with forward fill

### Feature Groups Tested

1. **Baseline (Target Only)**
   - Only historical volatility (`ctc_vol`)
   - No covariates

2. **Technicals** (2 features)
   - `volume`: Trading volume
   - `return`: Daily returns
   - Additional: RSI, ATR, SMA, EMA, BBANDS (if available)

3. **Macro** (4 features)
   - `SP500`: S&P 500 Index
   - `EUROSTOXX_50`: European stock index
   - `VIX`: Volatility index
   - `CHFUSD`, `INFLATION`, `FEDERAL_FUNDS_RATE`

4. **Peers** (4 features)
   - Returns from correlated stocks in SMI
   - `Peer_*` columns

### Stocks Tested (18 Total)
ABB, Alcon, Amrize, Geberit, Givaudan, Holcim, Kuehne+Nagel, Logitech, Lonza, Nestlé, Novartis, Partners Group, Roche, Sika, Swisscom, Swiss Life, UBS, Zurich Insurance

---

## Results

### Overall Performance vs Baseline

| Feature Group | MAE Improvement | RMSE Improvement | # Features |
|--------------|----------------|------------------|------------|
| **Technicals** | **+10.86%** ✅ | **+11.09%** ✅ | 2 |
| Macro | +0.48% | -0.04% | 4 |
| Peers | +0.38% | -0.38% | 4 |

### Statistical Summary

```
                        MAE (mean)   RMSE (mean)   
Baseline                  0.0051       0.0064
Technicals                0.0044       0.0054  ← Best
Macro                     0.0051       0.0064
Peers                     0.0051       0.0064
```

### Top Individual Improvements (Technicals)

1. **Alcon**: -44% MAE (0.0125 → 0.0070)
2. **Kuehne+Nagel**: -29% MAE (0.0040 → 0.0029)
3. **Partners Group**: -25% MAE (0.0049 → 0.0037)

---

## Key Findings

### ✅ What Works
- **Technical indicators significantly improve predictions** (~11% average improvement)
- Volume and return data are highly informative
- Simple features (2) outperform complex macro sets (4)

### ⚠️ What Doesn't Work Well
- **Macro features show minimal impact** despite including VIX
- **Peer correlations provide negligible benefit** 
- More features ≠ better performance

### 🔍 Insights
1. **Local information beats global**: Stock-specific technicals > market indices
2. **Signal-to-noise ratio matters**: 2 strong features > 4 weak features
3. **Model limitation**: Chronos-2 may struggle to extract value from macro time series with different frequencies/scales

---

## Recommendations

### For Production
1. **Use Technical features** (volume, return) for best accuracy
2. **Avoid macro features** unless fine-tuning shows improvement
3. **Skip peer features** - not worth the added complexity

### For Future Research
1. Test fine-tuned models to better leverage macro/peer data
2. Investigate feature engineering (e.g., normalized volume, volatility ratios)
3. Explore alternative covariate combinations
4. Consider model ensemble: Chronos-2 + technical features as baseline

---

## Technical Notes

### Data Preparation
- All data reindexed to business days (`freq='B'`)
- Missing values forward-filled
- Timestamps properly formatted for Chronos-2

### Code
- Script: `ablation_study.py`
- Results: `ablation_results.csv`
- Implementation follows patterns from `chronos-2-test/chronos-2.ipynb`

### Reproducibility
```bash
python ablation_study.py
```
Results will be saved to `ablation_results.csv` with detailed metrics per stock and feature group.

---

## Conclusion

**Technical indicators (volume, return) provide significant value (+11% improvement) for volatility forecasting with Chronos-2. Macro and peer features should be deprioritized in production unless further fine-tuning demonstrates benefits.**

The simplicity and effectiveness of technical features make them the recommended choice for deployment.
