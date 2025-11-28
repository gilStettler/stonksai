"""
Covariate Impact Analyzer

Performs ablation analysis to measure the impact of different covariates
on forecasting performance - similar to feature importance in traditional ML.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy import stats


class CovariateImpactAnalyzer:
    """Analyzes the marginal impact of covariates on model performance."""
    
    def __init__(self, results_dir: str = "./daily_results"):
        """
        Initialize analyzer.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.evaluation_dir = self.results_dir / "evaluation"
    
    def load_comparison_results(self) -> Dict:
        """Load comparison results from JSON."""
        comparison_file = self.evaluation_dir / "comparison.json"
        
        if not comparison_file.exists():
            raise FileNotFoundError(f"Comparison results not found: {comparison_file}")
        
        with open(comparison_file, 'r') as f:
            return json.load(f)
    
    def compute_marginal_impact(self, 
                               baseline_mae: float,
                               augmented_mae: float) -> float:
        """
        Compute marginal impact as percentage improvement.
        
        Args:
            baseline_mae: MAE of baseline model
            augmented_mae: MAE of model with additional covariates
        
        Returns:
            Percentage improvement (positive = better)
        """
        return (baseline_mae - augmented_mae) / baseline_mae * 100
    
    def analyze_impact(self, comparison_results: Dict) -> pd.DataFrame:
        """
        Analyze the impact of each covariate group.
        
        Args:
            comparison_results: Results from model comparison
        
        Returns:
            DataFrame with impact metrics
        """
        # Extract average MAE for each version
        versions = {}
        
        for version_name in ['v0_baseline', 'v1_vix', 'v2_vix_fred', 'v3_full']:
            if version_name in comparison_results:
                results = comparison_results[version_name]
                if results:  # Check if results exist
                    mae_values = [r['mae'] for r in results]
                    versions[version_name] = {
                        'mae': np.mean(mae_values),
                        'mae_std': np.std(mae_values),
                        'n_stocks': len(results),
                    }
        
        if 'v0_baseline' not in versions:
            raise ValueError("Baseline results not found")
        
        baseline_mae = versions['v0_baseline']['mae']
        
        # Calculate marginal impacts
        impacts = []
        
        # Impact of VIX
        if 'v1_vix' in versions:
            vix_impact = self.compute_marginal_impact(
                baseline_mae,
                versions['v1_vix']['mae']
            )
            impacts.append({
                'covariate_group': 'VIX',
                'baseline_mae': baseline_mae,
                'augmented_mae': versions['v1_vix']['mae'],
                'impact_pct': vix_impact,
                'mae_std': versions['v1_vix']['mae_std'],
                'n_stocks': versions['v1_vix']['n_stocks'],
            })
        
        # Impact of FRED (on top of VIX)
        if 'v1_vix' in versions and 'v2_vix_fred' in versions:
            fred_impact = self.compute_marginal_impact(
                versions['v1_vix']['mae'],
                versions['v2_vix_fred']['mae']
            )
            impacts.append({
                'covariate_group': 'FRED (fedfunds, t10y2y)',
                'baseline_mae': versions['v1_vix']['mae'],
                'augmented_mae': versions['v2_vix_fred']['mae'],
                'impact_pct': fred_impact,
                'mae_std': versions['v2_vix_fred']['mae_std'],
                'n_stocks': versions['v2_vix_fred']['n_stocks'],
            })
        
        # Impact of Correlations (on top of VIX + FRED)
        if 'v2_vix_fred' in versions and 'v3_full' in versions:
            corr_impact = self.compute_marginal_impact(
                versions['v2_vix_fred']['mae'],
                versions['v3_full']['mae']
            )
            impacts.append({
                'covariate_group': 'Cross-Stock Correlations',
                'baseline_mae': versions['v2_vix_fred']['mae'],
                'augmented_mae': versions['v3_full']['mae'],
                'impact_pct': corr_impact,
                'mae_std': versions['v3_full']['mae_std'],
                'n_stocks': versions['v3_full']['n_stocks'],
            })
        
        return pd.DataFrame(impacts)
    
    def statistical_significance(self, 
                                baseline_scores: List[float],
                                augmented_scores: List[float]) -> Dict:
        """
        Test statistical significance of improvement using paired t-test.
        
        Args:
            baseline_scores: MAE scores from baseline model
            augmented_scores: MAE scores from augmented model
        
        Returns:
            Dict with test results
        """
        # Paired t-test (lower MAE = better, so we expect negative difference)
        t_stat, p_value = stats.ttest_rel(baseline_scores, augmented_scores)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'significance_level': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        }
    
    def visualize_impact(self, impact_df: pd.DataFrame, output_path: str = None):
        """
        Create visualization of covariate impacts.
        
        Args:
            impact_df: DataFrame with impact analysis
            output_path: Path to save figure (default: ./daily_results/impact_analysis/)
        """
        if output_path is None:
            output_path = self.results_dir / "impact_analysis" / "covariate_impact.png"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar plot
        x = np.arange(len(impact_df))
        bars = ax.bar(x, impact_df['impact_pct'], 
                     color=['#2E86AB', '#A23B72', '#F18F01'],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Formatting
        ax.set_xlabel('Covariate Group', fontsize=14, fontweight='bold')
        ax.set_ylabel('Impact (%)', fontsize=14, fontweight='bold')
        ax.set_title('Marginal Impact of Covariates on Forecast Accuracy', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(impact_df['covariate_group'], rotation=15, ha='right')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, impact_df['impact_pct'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:+.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")
        
        return fig
    
    def generate_report(self, impact_df: pd.DataFrame, output_path: str = None):
        """
        Generate markdown report of covariate impact analysis.
        
        Args:
            impact_df: DataFrame with impact analysis
            output_path: Path to save report
        """
        if output_path is None:
            output_path = self.results_dir / "impact_analysis" / "ablation_report.md"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = []
        report.append("# Covariate Impact Analysis - Ablation Study Results\n")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        
        report.append("## Summary\n")
        report.append("This report shows the **marginal impact** of each covariate group on forecast accuracy,")
        report.append("measured as percentage improvement in MAE (Mean Absolute Error).\n")
        
        report.append("### Impact Table\n")
        report.append("| Covariate Group | Baseline MAE | Augmented MAE | Impact | Stocks |")
        report.append("|:----------------|:-------------|:--------------|:-------|:-------|")
        
        for _, row in impact_df.iterrows():
            impact_str = f"{row['impact_pct']:+.1f}%"
            status = "✅" if row['impact_pct'] > 0 else "❌"
            report.append(
                f"| {row['covariate_group']} | "
                f"{row['baseline_mae']:.4f} | "
                f"{row['augmented_mae']:.4f} | "
                f"{status} **{impact_str}** | "
                f"{row['n_stocks']} |"
            )
        
        report.append("\n### Interpretation\n")
        total_impact = impact_df['impact_pct'].sum()
        report.append(f"- **Total cumulative impact**: {total_impact:+.1f}%")
        report.append(f"- **Best performing covariate**: {impact_df.loc[impact_df['impact_pct'].idxmax(), 'covariate_group']} ({impact_df['impact_pct'].max():+.1f}%)")
        
        if (impact_df['impact_pct'] < 0).any():
            worst = impact_df.loc[impact_df['impact_pct'].idxmin()]
            report.append(f"- **Covariate with negative impact**: {worst['covariate_group']} ({worst['impact_pct']:+.1f}%)")
        
        report.append("\n### Recommendations\n")
        
        best_version = None
        if total_impact > 10:
            best_version = "v2_vix_fred" if 'FRED' in impact_df['covariate_group'].values else "v1_vix"
            report.append(f"✅ **Deploy augmented model** ({best_version})")
            report.append(f"   - Cumulative improvement of {total_impact:.1f}% justifies added complexity")
        elif total_impact > 5:
            report.append("⚠️  **Consider deployment**")
            report.append(f"   - Moderate improvement of {total_impact:.1f}%")
            report.append("   - Evaluate trade-off between accuracy and simplicity")
        else:
            report.append("⏸️  **Use baseline model**")
            report.append(f"   - Improvement of {total_impact:.1f}% too small to justify complexity")
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Report saved to: {output_path}")


def main():
    """Run covariate impact analysis."""
    print("\n" + "="*70)
    print("COVARIATE IMPACT ANALYSIS")
    print("="*70)
    
    analyzer = CovariateImpactAnalyzer()
    
    # Load comparison results
    print("\nLoading comparison results...")
    try:
        results = analyzer.load_comparison_results()
        print("✓ Results loaded")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nRun compare_daily_models.py first to generate evaluation results.")
        return
    
    # Analyze impact
    print("\nAnalyzing covariate impact...")
    impact_df = analyzer.analyze_impact(results)
    
    # Display results
    print("\n" + "="*70)
    print("COVARIATE IMPACT RESULTS")
    print("="*70)
    print(impact_df.to_string(index=False))
    print("="*70)
    
    # Generate visualization
    print("\nGenerating visualization...")
    analyzer.visualize_impact(impact_df)
    
    # Generate report
    print("\nGenerating report...")
    analyzer.generate_report(impact_df)
    
    # Save impact data
    impact_path = analyzer.results_dir / "impact_analysis" / "covariate_impact.json"
    impact_path.parent.mkdir(parents=True, exist_ok=True)
    impact_df.to_json(impact_path, orient='records', indent=2)
    print(f"✓ Impact data saved to: {impact_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
