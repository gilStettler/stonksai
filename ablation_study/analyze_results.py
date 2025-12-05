
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
try:
    df = pd.read_csv("smart_ablation_results.csv")
except FileNotFoundError:
    print("Results file not found. Please run the ablation notebook first.")
    exit()

print("="*100)
print("COMPREHENSIVE TOP-K METRIC ANALYSIS")
print("="*100)

# Filter for TOP_K combinations
k_results = df[df['features'].str.contains('TOP_|ALL_FEATURES')]

# Extract K values for sorting
def extract_k(name):
    if "ALL" in name: return 999
    try:
        return int(name.split('_')[1])
    except:
        return 0

# Aggregate all metrics
k_summary = k_results.groupby('features').agg({
    'mae': 'mean',
    'mse': 'mean',
    'rmse': 'mean',
    'r2': 'mean',
    'adj_r2': 'mean',
    'n_features': 'mean'
}).reset_index()

k_summary['k_sort'] = k_summary['features'].apply(extract_k)
k_summary = k_summary.sort_values('k_sort')

# 1. Print Detailed Table
print("\nGlobal Average Metrics by Feature Count:")
print("-" * 115)
print(f"{'K (Features)':<15} | {'MAE':<10} | {'MSE':<10} | {'RMSE':<10} | {'R2':<10} | {'Adj. R2':<10} | {'Optimal?'}")
print("-" * 115)

best_mae = k_summary['mae'].min()
best_r2 = k_summary['r2'].max()
best_adj_r2 = k_summary['adj_r2'].max()

for _, row in k_summary.iterrows():
    name = row['features'].replace("_COMBINED", "")
    mae = row['mae']
    mse = row['mse']
    rmse = row['rmse']
    r2 = row['r2']
    adj_r2 = row['adj_r2']
    
    # Mark optimal
    optimal_mark = ""
    if mae == best_mae: optimal_mark += "🏆 (Best MAE) "
    if r2 == best_r2: optimal_mark += "🌟 (Best R2) "
    if adj_r2 == best_adj_r2: optimal_mark += "💎 (Best Adj R2)"
    
    print(f"{name:<15} | {mae:.6f}   | {mse:.6f}   | {rmse:.6f}    | {r2:.6f}     | {adj_r2:.6f}     | {optimal_mark}")

print("-" * 115)

# 2. Generate Plots for Each Metric
metrics_to_plot = [
    ('mae', 'Mean Absolute Error (Lower is Better)', 'min'),
    ('mse', 'Mean Squared Error (Lower is Better)', 'min'),
    ('rmse', 'Root Mean Squared Error (Lower is Better)', 'min'),
    ('r2', 'R2 Score (Higher is Better)', 'max'),
    ('adj_r2', 'Adjusted R2 Score (Higher is Better)', 'max')
]

# Change layout to 3x2 to fit 5 plots
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
fig.suptitle('Feature Count (K) vs. Model Performance Metrics', fontsize=16)

# Prepare data for plotting
plot_data = k_summary.copy()
plot_data['Label'] = plot_data['features'].str.replace('_COMBINED', '').str.replace('TOP_', '')
plot_data['Label'] = plot_data['Label'].str.replace('ALL_FEATURES', 'ALL')

for i, (metric, title, goal) in enumerate(metrics_to_plot):
    ax = axes[i//2, i%2]
    
    # Plot line
    sns.lineplot(data=plot_data, x='Label', y=metric, marker='o', ax=ax, sort=False)
    
    # Highlight optimal point
    if goal == 'min':
        best_idx = plot_data[metric].idxmin()
        best_val = plot_data[metric].min()
        offset = (10, 10) # Text above the point (valley)
        va = 'bottom'
    else:
        best_idx = plot_data[metric].idxmax()
        best_val = plot_data[metric].max()
        offset = (10, -30) # Text below the point (peak) to avoid title overlap
        va = 'top'
        
    best_k = plot_data.loc[best_idx, 'Label']
    
    # Annotate best
    ax.plot(best_idx, best_val, 'o', color='red', markersize=10)
    ax.annotate(f'Optimal: {best_k}\n{best_val:.6f}', 
                (best_idx, best_val), 
                xytext=offset, textcoords='offset points',
                va=va,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
    
    ax.set_title(title)
    ax.set_xlabel('Number of Features (K)')
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)

# Remove empty 6th subplot
fig.delaxes(axes[2, 1])

# Adjust layout with more top margin
plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig('metric_comparison_plots.png')
print("\nSaved multi-metric plot to 'metric_comparison_plots.png'")
