"""
Data Analysis Script for AlphaVantage Stock Data
Analyzes data composition and creates comprehensive statistics
"""

import pandas as pd
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Configuration
DATA_DIR = "alphavantage_data"
OUTPUT_DIR = "analysis_results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ticker to Company Name Mapping
TICKER_TO_COMPANY = {
    '0QKI.LON': 'Swisscom AG',
    '0QLR.LON': 'Novartis AG',
    'NSRGY': 'Nestlé SA',
    'RHO6.FRK': 'Roche Holding Ltd',
    'ABBNY': 'ABB Ltd',
    'UBS': 'UBS Group AG',
    '0QP2.LON': 'Zurich Insurance Group AG',
    '0QKY.LON': 'Holcim Ltd',
    '0QNO.LON': 'Lonza Group AG',
    '0QPS.LON': 'Givaudan SA',
    '0A0D.LON': 'Alcon AG',
    '0Z4C.LON': 'Sika AG',
    '0QOQ.LON': 'Partners Group Holding AG',
    '0QMG.LON': 'Swiss Life Holding AG',
    'AMRZ': 'Amrize Ltd',
    '0QQ2.LON': 'Geberit AG',
    '0QMW.LON': 'Kühne & Nagel AG',
    '0QK6.LON': 'Logitech International SA'
}


def load_all_data():
    """Loads all CSV files from the data directory"""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_daily_data.csv"))
    
    all_data = {}
    for file_path in csv_files:
        # Extract ticker symbol from filename
        ticker = os.path.basename(file_path).replace("_daily_data.csv", "")
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter for data from 2020 onwards
            df = df[df['date'] >= '2020-01-01']
            
            if df.empty:
                print(f"⚠ Skipped: {ticker} (no data from 2020 onwards)")
                continue
                
            df = df.sort_values('date')
            all_data[ticker] = df
            print(f"✓ Loaded: {ticker} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error loading {ticker}: {e}")
    
    return all_data

def analyze_data_composition(all_data):
    """Analyzes the data composition of all stocks"""
    
    print("\n" + "="*80)
    print("DATA COMPOSITION - OVERVIEW")
    print("="*80 + "\n")
    
    composition_stats = []
    
    for ticker, df in all_data.items():
        company_name = TICKER_TO_COMPANY.get(ticker, ticker)
        stats = {
            'Ticker': ticker,
            'Company': company_name,
            'Data_Points': len(df),
            'Start_Date': df['date'].min(),
            'End_Date': df['date'].max(),
            'Timespan_Days': (df['date'].max() - df['date'].min()).days,
            'Timespan_Years': round((df['date'].max() - df['date'].min()).days / 365.25, 2),
            'Missing_Values': df.isnull().sum().sum(),
            'Avg_Volume': df['volume'].mean(),
            'Avg_Close': df['close'].mean(),
            'Min_Close': df['close'].min(),
            'Max_Close': df['close'].max(),
            'Volatility_%': round(df['close'].std() / df['close'].mean() * 100, 2) if df['close'].mean() != 0 else 0
        }
        composition_stats.append(stats)
        
        print(f"\n{company_name} ({ticker}):")
        print(f"  Period: {stats['Start_Date'].strftime('%Y-%m-%d')} to {stats['End_Date'].strftime('%Y-%m-%d')}")
        print(f"  Data points: {stats['Data_Points']:,}")
        print(f"  Timespan: {stats['Timespan_Years']} years")
        print(f"  Close price: ${stats['Min_Close']:.2f} - ${stats['Max_Close']:.2f} (Avg ${stats['Avg_Close']:.2f})")
        print(f"  Volatility: {stats['Volatility_%']}%")
        print(f"  Avg volume: {stats['Avg_Volume']:,.0f}")
    
    # Create DataFrame with all statistics
    stats_df = pd.DataFrame(composition_stats)
    
    # Save as CSV
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "data_composition_statistics.csv"), index=False)
    print(f"\n✓ Statistics saved to: {OUTPUT_DIR}/data_composition_statistics.csv")
    
    return stats_df

def create_visualizations(all_data, stats_df):
    """Creates visualizations of the data"""
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # 1. Timespan comparison
    plt.figure(figsize=(14, 6))
    stats_df_sorted = stats_df.sort_values('Timespan_Years', ascending=True)
    plt.barh(stats_df_sorted['Ticker'], stats_df_sorted['Timespan_Years'], color='steelblue')
    plt.xlabel('Timespan (Years)')
    plt.ylabel('Ticker')
    plt.title('Data Timespan per Stock')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "timespan_comparison.png"), dpi=300)
    print("✓ Saved: timespan_comparison.png")
    plt.close()
    
    # 2. Data points comparison
    plt.figure(figsize=(14, 6))
    stats_df_sorted = stats_df.sort_values('Data_Points', ascending=True)
    plt.barh(stats_df_sorted['Ticker'], stats_df_sorted['Data_Points'], color='coral')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Ticker')
    plt.title('Number of Data Points per Stock')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "data_points_comparison.png"), dpi=300)
    print("✓ Saved: data_points_comparison.png")
    plt.close()
    
    # 3. Volatility comparison
    plt.figure(figsize=(14, 6))
    stats_df_sorted = stats_df.sort_values('Volatility_%', ascending=True)
    plt.barh(stats_df_sorted['Ticker'], stats_df_sorted['Volatility_%'], color='green')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Ticker')
    plt.title('Stock Price Volatility')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "volatility_comparison.png"), dpi=300)
    print("✓ Saved: volatility_comparison.png")
    plt.close()
    
    # 4. Price development of all stocks (normalized)
    plt.figure(figsize=(16, 10))
    for ticker, df in all_data.items():
        # Normalize to 100 at start
        normalized = (df['close'] / df['close'].iloc[0]) * 100
        plt.plot(df['date'], normalized, label=ticker, alpha=0.7, linewidth=1.5)
    
    plt.xlabel('Date')
    plt.ylabel('Normalized Price (Start = 100)')
    plt.title('Price Development of All Stocks (Normalized)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_development_normalized.png"), dpi=300, bbox_inches='tight')
    print("✓ Saved: price_development_normalized.png")
    plt.close()
    
    # 5. Volume analysis (Top 5)
    plt.figure(figsize=(14, 6))
    top_volume = stats_df.nlargest(5, 'Avg_Volume')
    plt.bar(top_volume['Ticker'], top_volume['Avg_Volume'], color='purple')
    plt.xlabel('Ticker')
    plt.ylabel('Average Trading Volume')
    plt.title('Top 5 Stocks by Average Trading Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_volume.png"), dpi=300)
    print("✓ Saved: top_volume.png")
    plt.close()
    
def analyze_temporal_coverage(all_data):
    """Analyzes the temporal coverage of the data"""
    
    print("\n" + "="*80)
    print("TEMPORAL COVERAGE")
    print("="*80 + "\n")
    
    # Find earliest and latest date across all stocks
    overall_min = min(df['date'].min() for df in all_data.values())
    overall_max = max(df['date'].max() for df in all_data.values())
    
    print(f"Overall period: {overall_min.strftime('%Y-%m-%d')} to {overall_max.strftime('%Y-%m-%d')}")
    print(f"Total span: {(overall_max - overall_min).days} days ({round((overall_max - overall_min).days / 365.25, 2)} years)\n")
    
    # Visualize timeline
    plt.figure(figsize=(16, 10))
    for i, (ticker, df) in enumerate(all_data.items()):
        company_name = TICKER_TO_COMPANY.get(ticker, ticker)
        start_date = df['date'].min()
        end_date = df['date'].max()
        plt.barh(i, (end_date - start_date).days, left=start_date, height=0.8, 
                 label=ticker, alpha=0.7)
        plt.text(start_date, i, f" {company_name}", va='center', fontsize=7)
    
    plt.xlabel('Date')
    plt.ylabel('Stocks')
    plt.title('Temporal Coverage of Data per Stock')
    plt.yticks([])
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "temporal_coverage.png"), dpi=300, bbox_inches='tight')
    print("✓ Saved: temporal_coverage.png")
    plt.close()

def analyze_correlations(all_data):
    """Analyzes correlations between stocks"""
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80 + "\n")
    
    # Create DataFrame with all close prices
    # Find common time period
    all_dates = set.intersection(*[set(df['date']) for df in all_data.values()])
    
    if len(all_dates) < 2:
        print("⚠ Not enough overlapping data for correlation analysis")
        return
    
    # Create DataFrame
    price_data = {}
    for ticker, df in all_data.items():
        df_filtered = df[df['date'].isin(all_dates)].sort_values('date')
        price_data[ticker] = df_filtered.set_index('date')['close']
    
    correlation_df = pd.DataFrame(price_data)
    
    # Rename columns/index to company names
    correlation_df.columns = [TICKER_TO_COMPANY.get(t, t) for t in correlation_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = correlation_df.corr()
    
    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
    print(f"✓ Correlation matrix saved ({len(all_dates)} common data points)")
    
    # Visualize correlation matrix
    plt.figure(figsize=(14, 12))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                mask=mask)
    plt.title('Correlation Matrix of Stock Prices')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()

def analyze_decomposition(all_data):
    """Analyzes time series decomposition (Trend, Seasonality, Residuals)"""
    
    print("\n" + "="*80)
    print("DECOMPOSITION ANALYSIS")
    print("="*80 + "\n")
    
    # Filter for stocks with enough data (at least 2 years approx for good seasonality)
    # Using period=252 for daily data (approx 1 trading year)
    period = 252
    
    valid_stocks = {}
    for ticker, df in all_data.items():
        if len(df) > period * 2:
            valid_stocks[ticker] = df
    
    if not valid_stocks:
        print("⚠ Not enough data for decomposition analysis (need > 2 years)")
        return

    print(f"Performing decomposition for {len(valid_stocks)} stocks...")
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    
    # Colors for different stocks
    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_stocks)))
    
    for i, (ticker, df) in enumerate(valid_stocks.items()):
        company_name = TICKER_TO_COMPANY.get(ticker, ticker)
        
        # Ensure index is datetime and sorted
        ts = df.set_index('date')['close'].asfreq('B') # Business days
        ts = ts.fillna(method='ffill') # Fill missing business days
        
        try:
            # Multiplicative model usually better for stock prices (variance increases with level)
            # But additive is more robust if there are zeros or negatives (unlikely here)
            # Using additive for simplicity in visualization overlay, or multiplicative?
            # Let's use additive for visualization stability across many stocks
            result = seasonal_decompose(ts, model='additive', period=period)
            
            # Normalize components for comparison? 
            # Or just plot raw values? Raw values might be hard to compare if prices vary wildly.
            # Let's normalize trend and observed to start at 100.
            # Seasonality and resid will be relative.
            
            # Actually, user asked for "all stocks together in one graphic".
            # Normalized is best for Trend/Observed.
            
            start_val = result.trend.dropna().iloc[0]
            if start_val == 0: start_val = 1
            
            norm_trend = (result.trend / start_val) * 100
            
            # Plot Trend
            axes[0].plot(result.trend.index, norm_trend, label=company_name, color=colors[i], alpha=0.8)
            
            # Plot Seasonality (not normalized, but maybe scaled?)
            # Seasonality in additive model is absolute value. 
            # If we want to compare, maybe % of price?
            # Let's stick to raw additive seasonality for now, but it might be messy.
            # Alternative: Decompose Log prices -> Additive becomes Multiplicative equivalent.
            # Let's try decomposing Log prices for better comparability?
            # No, let's stick to simple additive but maybe just plot it.
            # If prices range from 10 to 1000, additive seasonality will be huge for 1000.
            # Let's normalize seasonality by dividing by trend? -> effectively multiplicative
            
            axes[1].plot(result.seasonal.index, result.seasonal, label=company_name, color=colors[i], alpha=0.5)
            
            # Plot Residuals
            axes[2].plot(result.resid.index, result.resid, label=company_name, color=colors[i], alpha=0.5)
            
            # Plot Observed (Normalized)
            norm_observed = (ts / ts.iloc[0]) * 100
            axes[3].plot(ts.index, norm_observed, label=company_name, color=colors[i], alpha=0.8)
            
        except Exception as e:
            print(f"⚠ Could not decompose {ticker}: {e}")
            continue

    axes[0].set_title('Trend Component (Normalized, Start=100)')
    axes[0].grid(True, alpha=0.3)
    # axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    axes[1].set_title('Seasonal Component (Additive)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Residuals')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_title('Observed Data (Normalized, Start=100)')
    axes[3].grid(True, alpha=0.3)
    
    # Add single legend to the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "decomposition_analysis.png"), dpi=300, bbox_inches='tight')
    print("✓ Saved: decomposition_analysis.png")
    plt.close()

def generate_summary_report(stats_df):
    """Creates a comprehensive text report"""
    
    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ALPHAVANTAGE DATA ANALYSIS - SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of stocks analyzed: {len(stats_df)}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total data points: {stats_df['Data_Points'].sum():,}\n")
        f.write(f"Average data points per stock: {stats_df['Data_Points'].mean():,.0f}\n")
        f.write(f"Longest timespan: {stats_df['Timespan_Years'].max()} years ({stats_df.loc[stats_df['Timespan_Years'].idxmax(), 'Ticker']})\n")
        f.write(f"Shortest timespan: {stats_df['Timespan_Years'].min()} years ({stats_df.loc[stats_df['Timespan_Years'].idxmin(), 'Ticker']})\n")
        f.write(f"Highest volatility: {stats_df['Volatility_%'].max()}% ({stats_df.loc[stats_df['Volatility_%'].idxmax(), 'Ticker']})\n")
        f.write(f"Lowest volatility: {stats_df['Volatility_%'].min()}% ({stats_df.loc[stats_df['Volatility_%'].idxmin(), 'Ticker']})\n\n")
        
        f.write("-"*80 + "\n")
        f.write("STOCK OVERVIEW (sorted by timespan)\n")
        f.write("-"*80 + "\n\n")
        
        for _, row in stats_df.sort_values('Timespan_Years', ascending=False).iterrows():
            company = row.get('Company', row['Ticker'])
            f.write(f"{company:35} ({row['Ticker']:12}) | ")
            f.write(f"{row['Timespan_Years']:6.2f} years | ")
            f.write(f"{row['Data_Points']:6,} points | ")
            f.write(f"Vol: {row['Volatility_%']:6.2f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n✓ Summary report saved: {report_path}")

def main():
    """Main function"""
    print("\n" + "="*80)
    print("ALPHAVANTAGE DATA ANALYSIS SCRIPT")
    print("="*80 + "\n")
    
    # 1. Load data
    all_data = load_all_data()
    
    if not all_data:
        print("\n✗ No data found!")
        return
    
    print(f"\n✓ Successfully loaded {len(all_data)} stocks\n")
    
    # 2. Analyze data composition
    stats_df = analyze_data_composition(all_data)
    
    # 3. Analyze temporal coverage
    analyze_temporal_coverage(all_data)
    
    # 4. Analyze correlations
    analyze_correlations(all_data)

    # 5. Analyze decomposition
    analyze_decomposition(all_data)
    
    # 5. Create visualizations
    create_visualizations(all_data, stats_df)
    
    # 6. Generate summary report
    generate_summary_report(stats_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)
    print(f"\nAll results have been saved to the '{OUTPUT_DIR}' folder.")
    print("\nGenerated files:")
    print("  - data_composition_statistics.csv")
    print("  - analysis_report.txt")
    print("  - timespan_comparison.png")
    print("  - data_points_comparison.png")
    print("  - volatility_comparison.png")
    print("  - price_development_normalized.png")
    print("  - top_volume.png")
    print("  - temporal_coverage.png")
    print("  - correlation_matrix.csv")
    print("  - correlation_matrix.png")
    print("  - decomposition_analysis.png\n")

if __name__ == "__main__":
    main()
