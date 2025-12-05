"""
Compare EWMA vs Top-6 CTC Predictions for Dec 1, 2025
"""

import pandas as pd
import numpy as np

# Load EWMA results
ewma = pd.read_csv('ewma_validation_dec1.csv')

# Load Top-6 CTC results
ctc = pd.read_csv('../out_of_sample_test/comparison_top-6_features.csv')

# Process CTC data
ctc['Stock'] = ctc['Stock'].apply(lambda x: x.split('_')[0])
ctc['Pred_CTC'] = ctc['Volatility'] * 100
ctc['Actual_CTC'] = ctc['Actual_Vol'] * 100
ctc['Error_CTC'] = abs(ctc['Pred_CTC'] - ctc['Actual_CTC'])
ctc['Error_Pct_CTC'] = ctc['Error_CTC'] / ctc['Actual_CTC'] * 100

# Merge
merged = ewma.merge(ctc[['Stock', 'Pred_CTC', 'Actual_CTC', 'Error_CTC', 'Error_Pct_CTC']], on='Stock', how='inner')

# Compare
print('=' * 90)
print('VERGLEICH: EWMA (K=1) vs Top-6 CTC für 1. Dezember 2025')
print('=' * 90)
print()
print(f"{'Stock':<12} | {'EWMA Err%':>10} | {'CTC Err%':>10} | {'Besser':>10}")
print('-' * 90)

ewma_wins = 0
for _, r in merged.iterrows():
    ewma_err = r['Error_Pct']
    ctc_err = r['Error_Pct_CTC']
    winner = 'EWMA' if ewma_err < ctc_err else 'CTC'
    if ewma_err < ctc_err:
        ewma_wins += 1
    print(f"{r['Stock']:<12} | {ewma_err:>9.2f}% | {ctc_err:>9.2f}% | {winner:>10}")

print('-' * 90)
print()

# Summary
ewma_avg = merged['Error_Pct'].mean()
ctc_avg = merged['Error_Pct_CTC'].mean()

print('ZUSAMMENFASSUNG')
print('=' * 50)
print(f"EWMA durchschnittlicher Fehler:    {ewma_avg:>6.2f}%")
print(f"CTC Top-6 durchschnittlicher Fehler: {ctc_avg:>6.2f}%")
print()
print(f"EWMA gewinnt bei: {ewma_wins}/{len(merged)} Aktien")
print()

if ewma_avg < ctc_avg:
    improvement = (ctc_avg - ewma_avg) / ctc_avg * 100
    print(f"*** EWMA ist {improvement:.1f}% besser als Top-6 CTC! ***")
else:
    improvement = (ewma_avg - ctc_avg) / ewma_avg * 100
    print(f"*** Top-6 CTC ist {improvement:.1f}% besser als EWMA! ***")
