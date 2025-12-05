
"""
05_visualize_results.py

Purpose:
    Generates visualization plots for the out-of-sample test.
    It plots the actual volatility history and the predictions from all models
    for Dec 1, 2025.
    
    The plots show:
    - Actual history (black line)
    - Actual target value (red X)
    - Zero-Shot prediction (blue circle)
    - Top-6 Features prediction (green triangle)
    - All Features prediction (orange square)
    - Fine-Tuned prediction (purple diamond)
    
    It displays the prediction value in the legend.

Usage:
    python 05_visualize_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

# Configuration
ACTUALS_DIR = "actuals_data"
PREDS_ZEROSHOT = "predictions_2025_12_01_zeroshot.csv"
PREDS_TOP6 = "predictions_2025_12_01_top6.csv"
PREDS_ALL = "predictions_2025_12_01_all_features.csv"
PREDS_FINETUNED = "predictions_finetuned.csv"
OUTPUT_DIR = "plots"
TARGET_DATE = pd.Timestamp("2025-12-01")

def load_preds(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    return pd.read_csv(filepath)

def plot_stock(stock_name, df_actual, preds_dict):
    plt.figure(figsize=(10, 6))
    
    # Filter history (last 20 days)
    history = df_actual[df_actual["timestamp"] <= TARGET_DATE].tail(20)
    
    # Plot Actual History
    plt.plot(history["timestamp"], history["ctc_vol"], label="Actual History", color="black", linewidth=1.5, alpha=0.7)
    
    # Plot Actual Target
    actual_target = history[history["timestamp"] == TARGET_DATE]
    if not actual_target.empty:
        actual_val = actual_target["ctc_vol"].values[0]
        plt.scatter(actual_target["timestamp"], actual_target["ctc_vol"], color="red", marker="x", s=100, label=f"Actual (Dec 1) ({actual_val:.4f})", zorder=10)
    else:
        actual_val = None

    markers = {
        "Zero-Shot": ("o", "blue"),
        "Top-6 Features": ("^", "green"),
        "All Features": ("s", "orange"),
        "Fine-Tuned": ("D", "purple")
    }
    
    for name, (marker, color) in markers.items():
        if name in preds_dict:
            val = preds_dict[name]
            # Label format: "Model Name (Pred=0.XXXX)"
            label_text = f"{name} (Pred={val:.4f})"
            plt.scatter(TARGET_DATE, val, color=color, marker=marker, s=80, label=label_text, zorder=9)
            
            # Draw dotted line to actual if available
            if actual_val is not None:
                plt.plot([TARGET_DATE, TARGET_DATE], [val, actual_val], color=color, linestyle=":", alpha=0.5)

    plt.title(f"Volatility Prediction: {stock_name}", fontsize=14)
    plt.ylabel("Volatility (ctc_vol)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{stock_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

def main():
    # Load Predictions
    df_zero = load_preds(PREDS_ZEROSHOT)
    df_top6 = load_preds(PREDS_TOP6)
    df_all = load_preds(PREDS_ALL)
    df_fine = load_preds(PREDS_FINETUNED)
    
    # Get list of stocks from actuals
    csv_files = glob.glob(os.path.join(ACTUALS_DIR, "*.csv"))
    
    print(f"Generating plots for {len(csv_files)} stocks...")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        stock_name = filename.replace("data_", "").replace(".csv", "")
        
        # Load Actuals
        try:
            df_actual = pd.read_csv(filepath)
            df_actual["timestamp"] = pd.to_datetime(df_actual["timestamp"])
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Get Predictions for this stock
        preds = {}
        
        # Helper to get value
        def get_val(df, name):
            row = df[df["Stock"] == stock_name]
            if not row.empty:
                return row["Volatility"].values[0]
            return None

        if not df_zero.empty: preds["Zero-Shot"] = get_val(df_zero, stock_name)
        if not df_top6.empty: preds["Top-6 Features"] = get_val(df_top6, stock_name)
        if not df_all.empty: preds["All Features"] = get_val(df_all, stock_name)
        if not df_fine.empty: preds["Fine-Tuned"] = get_val(df_fine, stock_name)
        
        # Plot
        plot_stock(stock_name, df_actual, preds)
        
    print(f"Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
