# Covariate Impact Analysis - Ablation Study Results

Generated: 2025-11-28 14:44:40

---

## Summary

This report shows the **marginal impact** of each covariate group on forecast accuracy,
measured as percentage improvement in MAE (Mean Absolute Error).

### Impact Table

| Covariate Group | Baseline MAE | Augmented MAE | Impact | Stocks |
|:----------------|:-------------|:--------------|:-------|:-------|
| VIX | 0.0108 | 0.0108 | ❌ **-0.4%** | 15 |
| FRED (fedfunds, t10y2y) | 0.0108 | 0.0108 | ✅ **+0.5%** | 15 |
| Cross-Stock Correlations | 0.0108 | 0.0108 | ✅ **+0.0%** | 15 |

### Interpretation

- **Total cumulative impact**: +0.1%
- **Best performing covariate**: FRED (fedfunds, t10y2y) (+0.5%)
- **Covariate with negative impact**: VIX (-0.4%)

### Recommendations

⏸️  **Use baseline model**
   - Improvement of 0.1% too small to justify complexity