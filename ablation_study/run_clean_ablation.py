import transformers
try:
    from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
except ImportError:
    pass

# Patch chronos.chronos2 to have T5ForConditionalGeneration
import chronos.chronos2
try:
    chronos.chronos2.T5ForConditionalGeneration = T5ForConditionalGeneration
except NameError:
    from transformers import T5ForConditionalGeneration
    chronos.chronos2.T5ForConditionalGeneration = T5ForConditionalGeneration

print("Applied T5ForConditionalGeneration patch.")


# Core Imports
import os
import glob
import time
import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Configuration
DATA_DIR = "../processed_data"
RESULTS_FILE = "smart_ablation_results.csv"
PLOTS_DIR = "plots"
MODEL_NAME = "amazon/chronos-2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_COL = "ctc_vol"
DATE_COL = "timestamp"
ID_COL = "id"
DEFAULT_ID = "series_1"

# Training parameters
SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80
PREDICTION_LENGTH = 1
FREQ = "B"

print(f"Using device: {DEVICE}")

def get_all_features(df, exclude_cols):
    """Dynamically identify all numeric feature columns.
    CRITICAL: Filters out concurrent features to prevent Data Leakage.
    Only allows features with '_lag_' or '_mean' (which are shifted).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    valid_features = []
    for c in numeric_cols:
        if c in exclude_cols: continue
        
        # Strict filtering for valid forecasting features
        # 1. Explicit lags
        if "_lag_" in c:
            valid_features.append(c)
        # 2. Rolling means (verified to be shifted in generation)
        elif "_mean" in c:
            valid_features.append(c)
            
    return valid_features

def load_and_prep_data(filepath):
    """Load CSV, reindex to business days, and handle missing values."""
    try:
        df = pd.read_csv(filepath)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        
        # Add ID column if missing
        if ID_COL not in df.columns:
            df[ID_COL] = DEFAULT_ID
            
        # Business day reindexing
        pieces = []
        for sid, g in df.groupby(ID_COL):
            g = g.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
            g = g.set_index(DATE_COL)
            
            # Create full business day range
            start_date = g.index.min()
            end_date = g.index.max()
            bdays_idx = pd.date_range(start=start_date, end=end_date, freq=FREQ)
            
            # Reindex and forward fill
            g = g.reindex(bdays_idx)
            g = g.ffill()
            
            g[ID_COL] = sid
            g.index.name = DATE_COL
            pieces.append(g.reset_index())
            
        df_processed = pd.concat(pieces, ignore_index=True)
        
        # Filter for study period
        df_processed = df_processed[df_processed[DATE_COL] >= pd.Timestamp(SPLIT_START_DATE)]
        
        return df_processed.dropna() 
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_metrics(y_true, y_pred):
    """Calculate deterministic metrics."""
    metrics = {}
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(y_true, y_pred)
    return metrics

# Initialize Model
pipeline = Chronos2Pipeline.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
)
print("Model loaded successfully.")

# =========================================================================
# PHASE 1: Run Ablation Study for all stocks
# =========================================================================
print("\n" + "="*70)
print("PHASE 1: Running Ablation Study")
print("="*70)

results = []
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} stock files.")

for filepath in csv_files:
    stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
    print(f"\n{'='*50}")
    print(f"Processing {stock_name}...")
    print(f"{'='*50}")
    
    # Load data
    df = load_and_prep_data(filepath)
    if df is None or len(df) < 50:
        print(f"Skipping {stock_name} due to insufficient data.")
        continue
        
    # Identify ALL features
    exclude = [TARGET_COL, DATE_COL, ID_COL]
    all_features = get_all_features(df, exclude)
    print(f"Found {len(all_features)} potential features.")
    
    # Train/Test Split
    n_total = len(df)
    n_train = int(np.floor(TRAIN_FRACTION * n_total))
    
    context_df = df.iloc[:n_train].copy()
    future_df = df.iloc[n_train:].copy()
    actuals = future_df[TARGET_COL].values
    
    # 1. Baseline (No features)
    print("Testing BASELINE (no features)...")
    start_time = time.time()
    
    baseline_forecast = pipeline.predict_df(
        context_df[[ID_COL, DATE_COL, TARGET_COL]],
        future_df[[ID_COL, DATE_COL]],
        prediction_length=len(future_df),
        quantile_levels=[0.5],
        target=TARGET_COL,
        id_column=ID_COL,
        timestamp_column=DATE_COL
    )
    baseline_preds = baseline_forecast['0.5'].values
    baseline_metrics = calculate_metrics(actuals, baseline_preds)
    
    baseline_res = {
        'stock': stock_name,
        'features': 'BASELINE_NONE',
        'n_features': 0,
        'mae': baseline_metrics['mae'],
        'mse': baseline_metrics['mse'],
        'rmse': baseline_metrics['rmse'],
        'r2': baseline_metrics['r2'],
        'time': time.time() - start_time
    }
    results.append(baseline_res)
    print(f"  Baseline MAE: {baseline_metrics['mae']:.6f}")
    
    # 2. Individual Feature Screening
    print(f"\nScreening {len(all_features)} features individually...")
    feature_performance = []
    
    for i, feat in enumerate(all_features):
        try:
            print(f"  Feature {i+1}/{len(all_features)}: {feat}", end="\r")
            start_time = time.time()
            
            context_cols = [ID_COL, DATE_COL, TARGET_COL, feat]
            future_cols = [ID_COL, DATE_COL, feat]
            
            forecast = pipeline.predict_df(
                context_df[context_cols],
                future_df[future_cols],
                prediction_length=len(future_df),
                quantile_levels=[0.5],
                target=TARGET_COL,
                id_column=ID_COL,
                timestamp_column=DATE_COL
            )
            preds = forecast['0.5'].values
            metrics = calculate_metrics(actuals, preds)
            
            res = {
                'stock': stock_name,
                'features': feat,
                'n_features': 1,
                'mae': metrics['mae'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'time': time.time() - start_time
            }
            results.append(res)
            feature_performance.append((feat, metrics['mae']))
            
        except Exception as e:
            print(f"  Error with feature {feat}: {e}")
            
    print()  # New line after progress
    
    # 3. Top-K Sweep
    feature_performance.sort(key=lambda x: x[1])
    k_values = list(range(1, 21))
    combinations_to_test = [(f"TOP_{k}", k) for k in k_values]
    combinations_to_test.append(("ALL_FEATURES", len(all_features)))
    
    for name, k in combinations_to_test:
        try:
            if k > len(all_features): k = len(all_features)
            
            if name == "ALL_FEATURES":
                selected_feats = all_features
            else:
                selected_feats = [x[0] for x in feature_performance[:k]]
            
            if len(selected_feats) == 0: continue
            
            print(f"Testing combination: {name} ({len(selected_feats)} features)...")
            start_time = time.time()
            
            context_cols = [ID_COL, DATE_COL, TARGET_COL] + selected_feats
            future_cols = [ID_COL, DATE_COL] + selected_feats
            
            forecast = pipeline.predict_df(
                context_df[context_cols],
                future_df[future_cols],
                prediction_length=len(future_df),
                quantile_levels=[0.5],
                target=TARGET_COL,
                id_column=ID_COL,
                timestamp_column=DATE_COL
            )
            preds = forecast['0.5'].values
            metrics = calculate_metrics(actuals, preds)
            
            combo_res = {
                'stock': stock_name,
                'features': f"{name}_COMBINED",
                'n_features': len(selected_feats),
                'mae': metrics['mae'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'time': time.time() - start_time
            }
            results.append(combo_res)
            print(f"  {name} MAE: {metrics['mae']:.6f}")
            
        except Exception as e:
            print(f"  Error with {name}: {e}")

    # Save intermediate results
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
    print(f"Saved results for {stock_name}")

print("\n" + "="*70)
print("PHASE 1 COMPLETE: Ablation Study finished")
print("="*70)

# =========================================================================
# PHASE 2: Analyze results and find global best K
# =========================================================================
print("\n" + "="*70)
print("PHASE 2: Analyzing Results")
print("="*70)

df_results = pd.read_csv(RESULTS_FILE)
df_combined = df_results[df_results['features'].str.contains("_COMBINED")].copy()
df_combined["k"] = df_combined["n_features"].astype(int)

global_metrics = df_combined.groupby("k")[["mae"]].mean().reset_index()
best_global_k = global_metrics.loc[global_metrics["mae"].idxmin(), "k"]
print(f"\nGlobal Optimal K: {int(best_global_k)}")
print(f"Global MAE: {global_metrics.loc[global_metrics['mae'].idxmin(), 'mae']:.6f}")

# =========================================================================
# PHASE 3: Generate plots for all stocks using global best K
# =========================================================================
print("\n" + "="*70)
print(f"PHASE 3: Generating Plots with Top-{int(best_global_k)} Features")
print("="*70)

os.makedirs(PLOTS_DIR, exist_ok=True)

for filepath in csv_files:
    stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
    print(f"Plotting {stock_name}...")
    
    try:
        # Load data
        df = load_and_prep_data(filepath)
        if df is None or len(df) < 50:
            continue
        
        # Get features
        exclude = [TARGET_COL, DATE_COL, ID_COL]
        all_features = get_all_features(df, exclude)
        
        # Split
        n_total = len(df)
        n_train = int(np.floor(TRAIN_FRACTION * n_total))
        context_df = df.iloc[:n_train].copy()
        future_df = df.iloc[n_train:].copy()
        
        # Get top-K features for THIS stock
        stock_results = df_results[df_results['stock'] == stock_name]
        stock_individual = stock_results[~stock_results['features'].str.contains("_COMBINED") & 
                                         (stock_results['features'] != 'BASELINE_NONE')]
        stock_individual = stock_individual.sort_values('mae')
        
        top_k_features = stock_individual.head(int(best_global_k))['features'].tolist()
        
        if len(top_k_features) == 0:
            print(f"  No features found for {stock_name}")
            continue
        
        # Generate forecast
        context_cols = [ID_COL, DATE_COL, TARGET_COL] + top_k_features
        future_cols = [ID_COL, DATE_COL] + top_k_features
        
        forecast = pipeline.predict_df(
            context_df[context_cols],
            future_df[future_cols],
            prediction_length=len(future_df),
            quantile_levels=[0.1, 0.5, 0.9],
            target=TARGET_COL,
            id_column=ID_COL,
            timestamp_column=DATE_COL
        )
        
        # Plot
        plt.figure(figsize=(15, 7))
        plt.plot(context_df[DATE_COL].iloc[-100:], context_df[TARGET_COL].iloc[-100:], color="black", label="History")
        plt.plot(future_df[DATE_COL], future_df[TARGET_COL], color="green", label="Actual")
        plt.plot(forecast[DATE_COL], forecast['0.5'], color="blue", label="Forecast (Median)")
        plt.fill_between(forecast[DATE_COL], forecast['0.1'], forecast['0.9'], color="blue", alpha=0.2, label="80% CI")
        
        plt.title(f"Chronos-2 Forecast: {stock_name} (Top-{int(best_global_k)} Features)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(PLOTS_DIR, f"{stock_name}_rolling_backtest.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved: {save_path}")
        
    except Exception as e:
        print(f"  Error plotting {stock_name}: {e}")

print("\n" + "="*70)
print("ALL PHASES COMPLETE!")
print("="*70)