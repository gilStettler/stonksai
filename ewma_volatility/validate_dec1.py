"""
Validate EWMA Predictions vs Actuals (December 1, 2025)
"""

import pandas as pd
import numpy as np
import glob

# Load predictions
pred = pd.read_csv('ewma_predictions_dec1.csv')

# Calculate EWMA for actuals
def calc_ewma(returns, lambda_=0.94):
    n = len(returns)
    ewma_var = np.zeros(n)
    r = returns.fillna(0).values
    ewma_var[0] = r[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t-1] + (1 - lambda_) * (r[t] ** 2)
    return np.sqrt(ewma_var)

results = []
for f in sorted(glob.glob('../out_of_sample_test/actuals_data/*.csv')):
    df = pd.read_csv(f)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['ewma'] = calc_ewma(df['log_return'])
    df = df.dropna()
    
    # Get Dec 1 data
    dec1 = df[df['timestamp'].dt.date == pd.Timestamp('2025-12-01').date()]
    if len(dec1) > 0:
        ticker = f.split('data_')[1].split('_')[0]
        actual = dec1['ewma'].iloc[0] * 100
        results.append({'Stock': ticker, 'Actual_EWMA': round(actual, 4)})

actuals = pd.DataFrame(results)
merged = pred.merge(actuals, on='Stock', how='inner')
merged['Error'] = abs(merged['Pred_EWMA_Dec1'] - merged['Actual_EWMA'])
merged['Error_Pct'] = merged['Error'] / merged['Actual_EWMA'] * 100

print('=' * 75)
print('VERGLEICH: Vorhersage vs Tatsächlich (1. Dezember 2025)')
print('=' * 75)
print()
print(f"Stock      |    Pred    |   Actual   |   Error    |  Error%  ")
print('-' * 75)
for _, r in merged.iterrows():
    print(f"{r['Stock']:<10} | {r['Pred_EWMA_Dec1']:>8.4f}% | {r['Actual_EWMA']:>8.4f}% | {r['Error']:>8.4f}% | {r['Error_Pct']:>6.2f}%")
print('-' * 75)
print(f"AVERAGE    | {merged['Pred_EWMA_Dec1'].mean():>8.4f}% | {merged['Actual_EWMA'].mean():>8.4f}% | {merged['Error'].mean():>8.4f}% | {merged['Error_Pct'].mean():>6.2f}%")
print('=' * 75)
print()
print(f"MAE: {merged['Error'].mean():.4f}%")
print(f"Anzahl Aktien: {len(merged)}")

# Save comparison
merged.to_csv('ewma_validation_dec1.csv', index=False)
print(f"\nGespeichert: ewma_validation_dec1.csv")
