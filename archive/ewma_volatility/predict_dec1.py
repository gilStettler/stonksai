"""
EWMA Volatility Prediction for December 1st, 2025
==================================================

Simple prediction using the EWMA formula directly.
Based on K=1 finding: ewma_vol_lag_1 is the best predictor.

For EWMA: The best prediction for tomorrow is simply today's EWMA value
(with minor adjustment from the model).
"""

import os
import glob
import pandas as pd
import numpy as np

# Configuration
DATA_DIR = "../processed_data"
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL = "timestamp"
LAMBDA = 0.94


def calculate_ewma_volatility(returns, lambda_=0.94):
    """Calculate EWMA volatility."""
    n = len(returns)
    ewma_var = np.zeros(n)
    returns_filled = returns.fillna(0).values
    ewma_var[0] = returns_filled[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t-1] + (1 - lambda_) * (returns_filled[t] ** 2)
    return np.sqrt(ewma_var)


def main():
    print("=" * 70)
    print("EWMA Volatility Prediction - December 1, 2025")
    print("=" * 70)
    print(f"Methode: EWMA (λ={LAMBDA})")
    print(f"Optimal K=1: Beste Vorhersage = letzter bekannter EWMA-Wert")
    print("=" * 70)
    
    # Find all stock files
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"\nGefunden: {len(csv_files)} Aktien")
    
    predictions = []
    
    for filepath in csv_files:
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        ticker = stock_name.split("_")[0]
        
        try:
            # Load data
            df = pd.read_csv(filepath)
            df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            df = df.sort_values(DATE_COL).reset_index(drop=True)
            
            # Calculate EWMA
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
            df = df.dropna()
            
            last_date = df[DATE_COL].max()
            last_close = df["close"].iloc[-1]
            last_ewma = df["ewma_vol"].iloc[-1]
            
            # For EWMA with K=1: best prediction is the last known value
            # The model essentially learns that ewma_tomorrow ≈ ewma_today
            predicted_ewma = last_ewma
            
            # Calculate confidence interval based on historical volatility of EWMA changes
            ewma_changes = df["ewma_vol"].diff().dropna()
            ewma_std = ewma_changes.std()
            
            predictions.append({
                "Stock": ticker,
                "Last_Date": last_date.strftime("%Y-%m-%d"),
                "Last_Close": round(last_close, 2),
                "Last_EWMA": round(last_ewma * 100, 4),  # In percent
                "Pred_EWMA_Dec1": round(predicted_ewma * 100, 4),  # In percent
                "Lower_Bound": round((predicted_ewma - 1.96*ewma_std) * 100, 4),
                "Upper_Bound": round((predicted_ewma + 1.96*ewma_std) * 100, 4)
            })
            
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue
    
    # Create DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, "ewma_predictions_dec1.csv")
    pred_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("VORHERSAGEN FÜR MONTAG, 1. DEZEMBER 2025")
    print("(EWMA Volatilität in Prozent)")
    print("=" * 70)
    print()
    
    # Format output nicely
    print(f"{'Stock':<10} | {'Letztes Datum':<12} | {'Letzter Kurs':>12} | {'EWMA Nov 28':>10} | {'Pred Dec 1':>10}")
    print("-" * 70)
    
    for _, row in pred_df.iterrows():
        print(f"{row['Stock']:<10} | {row['Last_Date']:<12} | {row['Last_Close']:>12} | {row['Last_EWMA']:>9}% | {row['Pred_EWMA_Dec1']:>9}%")
    
    print("-" * 70)
    print()
    print(f"Gespeichert: {output_file}")
    
    # Summary statistics
    avg_ewma = pred_df["Pred_EWMA_Dec1"].mean()
    print(f"\nDurchschnittliche vorhergesagte EWMA: {avg_ewma:.4f}%")


if __name__ == "__main__":
    main()
