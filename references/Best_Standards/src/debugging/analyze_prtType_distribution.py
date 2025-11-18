# uv run src/debugging/analyze_prtType_distribution.py

"""
Debugging script to investigate prtType distribution over time in processed data.

Purpose:
- Identify why complex trades (prtType >= 102) are missing from 2014-2019
- Investigate the 2014 anomaly where all prtType values are 0
- Track the emergence and evolution of different prtType values over time
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

from src.config import initialize_main
from THIS_IS import PROCESSED_PATH, OUTPUT_PATH

# Configuration
REPORTS_DIR = OUTPUT_PATH / "reports"
FIGURES_DIR = OUTPUT_PATH / "figures"

# Ensure output directories exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def looks_like_ymd(name: str) -> bool:
    """
    Check if a folder name follows the YYYY-MM-DD format.
    
    Args:
        name: Folder name to check
        
    Returns:
        True if folder name matches YYYY-MM-DD format
    """
    parts = name.split("-")
    if len(parts) != 3:
        return False
    try:
        y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return False
    return 1 <= m <= 12 and 1 <= d <= 31 and 1000 <= y <= 3000


def scan_date_prtType_distribution(date_folder: Path, con: duckdb.DuckDBPyConnection) -> Dict:
    """
    Extract prtType value distribution for a single date folder.
    
    Args:
        date_folder: Path to the date folder containing parquet files
        con: DuckDB connection
        
    Returns:
        Dictionary with date and prtType counts
    """
    try:
        date_str = date_folder.name
        parquet_pattern = str(date_folder / "*.parquet")
        
        # Query to get prtType value counts
        query = f"""
            SELECT 
                prtType,
                COUNT(*) as count
            FROM read_parquet('{parquet_pattern}')
            GROUP BY prtType
            ORDER BY prtType
        """
        
        result = con.execute(query).fetchall()
        
        # Convert to dictionary
        prtType_counts = {int(row[0]): int(row[1]) for row in result}
        
        # Calculate totals and flags
        total_count = sum(prtType_counts.values())
        has_complex = any(prt >= 102 for prt in prtType_counts.keys())
        all_zeros = (len(prtType_counts) == 1 and 0 in prtType_counts)
        has_simple = any(73 <= prt < 102 for prt in prtType_counts.keys())
        
        return {
            'date': date_str,
            'prtType_counts': prtType_counts,
            'total_trades': total_count,
            'has_complex_trades': has_complex,
            'all_prtType_zero': all_zeros,
            'has_simple_trades': has_simple,
            'unique_prtTypes': len(prtType_counts),
            'min_prtType': min(prtType_counts.keys()) if prtType_counts else None,
            'max_prtType': max(prtType_counts.keys()) if prtType_counts else None,
        }
        
    except Exception as e:
        return {
            'date': date_folder.name,
            'error': str(e),
            'prtType_counts': {},
            'total_trades': 0,
            'has_complex_trades': False,
            'all_prtType_zero': False,
            'has_simple_trades': False,
            'unique_prtTypes': 0,
            'min_prtType': None,
            'max_prtType': None,
        }


def build_timeseries_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """
    Aggregate individual date results into a time-indexed DataFrame.
    
    Args:
        all_results: List of dictionaries from scan_date_prtType_distribution
        
    Returns:
        DataFrame with dates as index and prtType counts as columns
    """
    # Create base DataFrame with metadata
    df = pd.DataFrame([
        {
            'date': r['date'],
            'total_trades': r['total_trades'],
            'has_complex_trades': r['has_complex_trades'],
            'all_prtType_zero': r['all_prtType_zero'],
            'has_simple_trades': r['has_simple_trades'],
            'unique_prtTypes': r['unique_prtTypes'],
            'min_prtType': r['min_prtType'],
            'max_prtType': r['max_prtType'],
            'has_error': 'error' in r,
        }
        for r in all_results
    ])
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Extract all unique prtType values across all dates
    all_prtTypes = set()
    for r in all_results:
        if 'prtType_counts' in r:
            all_prtTypes.update(r['prtType_counts'].keys())
    
    # Create columns for each prtType value
    for prtType in sorted(all_prtTypes):
        df[f'prtType_{prtType}'] = [
            r['prtType_counts'].get(prtType, 0) 
            for r in all_results
        ]
    
    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and flag various anomalies in the prtType distribution.
    
    Args:
        df: DataFrame from build_timeseries_dataframe
        
    Returns:
        DataFrame with additional anomaly flag columns
    """
    anomaly_df = df.copy()
    
    # Identify date ranges for key anomalies
    anomaly_df['anomaly_type'] = 'normal'
    
    # Flag all-zero dates
    anomaly_df.loc[anomaly_df['all_prtType_zero'], 'anomaly_type'] = 'all_zeros'
    
    # Flag dates with no complex trades but has simple trades
    no_complex_mask = (~anomaly_df['has_complex_trades']) & (anomaly_df['has_simple_trades'])
    anomaly_df.loc[no_complex_mask, 'anomaly_type'] = 'no_complex'
    
    # Flag dates with very low unique prtType diversity
    low_diversity_mask = (anomaly_df['unique_prtTypes'] < 3) & (~anomaly_df['all_prtType_zero'])
    anomaly_df.loc[low_diversity_mask, 'anomaly_type'] = 'low_diversity'
    
    # Flag dates with errors
    anomaly_df.loc[anomaly_df['has_error'], 'anomaly_type'] = 'error'
    
    return anomaly_df


def create_visualizations(df: pd.DataFrame, output_dir: Path, logger):
    """
    Create comprehensive visualizations of prtType distribution over time.
    
    Args:
        df: Anomaly-flagged DataFrame
        output_dir: Directory to save figures
        logger: Logger instance
    """
    logger.info("Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # 1. Overall prtType Distribution Timeline
    logger.info("Creating prtType distribution timeline...")
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    
    # Top plot: Total trades over time
    ax1 = axes[0]
    ax1.plot(df['date'], df['total_trades'], linewidth=1.5, color='navy', alpha=0.7)
    ax1.fill_between(df['date'], 0, df['total_trades'], alpha=0.3, color='navy')
    ax1.set_ylabel('Total Trades', fontsize=12, fontweight='bold')
    ax1.set_title('prtType Distribution Analysis Over Time', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Middle plot: Complex vs Simple trades
    ax2 = axes[1]
    
    # Calculate complex and simple trade counts
    prtType_cols = [col for col in df.columns if col.startswith('prtType_')]
    complex_counts = []
    simple_counts = []
    zero_counts = []
    
    for idx, row in df.iterrows():
        complex_count = sum(row[col] for col in prtType_cols if int(col.split('_')[1]) >= 102)
        simple_count = sum(row[col] for col in prtType_cols if 73 <= int(col.split('_')[1]) < 102)
        zero_count = row.get('prtType_0', 0)
        
        complex_counts.append(complex_count)
        simple_counts.append(simple_count)
        zero_counts.append(zero_count)
    
    df['complex_count'] = complex_counts
    df['simple_count'] = simple_counts
    df['zero_count'] = zero_counts
    
    ax2.fill_between(df['date'], 0, df['complex_count'], label='Complex (≥102)', 
                      alpha=0.6, color='red')
    ax2.fill_between(df['date'], 0, df['simple_count'], label='Simple (73-101)', 
                      alpha=0.6, color='green')
    ax2.fill_between(df['date'], 0, df['zero_count'], label='Zero prtType', 
                      alpha=0.6, color='gray')
    ax2.set_ylabel('Trade Count', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', axis='y')
    
    # Bottom plot: Unique prtType values per day
    ax3 = axes[2]
    colors = ['red' if anomaly != 'normal' else 'steelblue' 
              for anomaly in df['anomaly_type']]
    ax3.scatter(df['date'], df['unique_prtTypes'], c=colors, alpha=0.6, s=20)
    ax3.set_ylabel('Unique prtType Values', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add legend for anomalies
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.6, label='Normal'),
        Patch(facecolor='red', alpha=0.6, label='Anomaly')
    ]
    ax3.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / "prtType_distribution_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved timeline plot to {output_path}")
    
    # 2. Complex Trade Emergence Focus
    logger.info("Creating complex trade emergence plot...")
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Focus on period around complex trade emergence (2019-2020)
    df_filtered = df[(df['date'] >= '2019-01-01') & (df['date'] <= '2020-12-31')].copy()
    
    if not df_filtered.empty:
        ax.plot(df_filtered['date'], df_filtered['complex_count'], 
                linewidth=2, color='red', marker='o', markersize=3, 
                label='Complex Trades (prtType ≥ 102)')
        ax.fill_between(df_filtered['date'], 0, df_filtered['complex_count'], 
                        alpha=0.3, color='red')
        
        # Mark the first appearance of complex trades
        first_complex = df_filtered[df_filtered['complex_count'] > 0]
        if not first_complex.empty:
            first_date = first_complex.iloc[0]['date']
            ax.axvline(x=first_date, color='darkred', linestyle='--', 
                      linewidth=2, label=f'First Complex Trade: {first_date.strftime("%Y-%m-%d")}')
        
        ax.set_ylabel('Complex Trade Count', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_title('Emergence of Complex Trades (2019-2020)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='plain', axis='y')
        
        plt.tight_layout()
        output_path = output_dir / "prtType_complex_emergence.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved complex emergence plot to {output_path}")
    else:
        logger.warning("No data in 2019-2020 range for complex emergence plot")
    
    # 3. 2014 Anomaly Detail
    logger.info("Creating 2014 anomaly plot...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Focus on 2014
    df_2014 = df[df['date'].dt.year == 2014].copy()
    
    if not df_2014.empty:
        # Top: prtType=0 dominance
        ax1 = axes[0]
        ax1.bar(df_2014['date'], df_2014['zero_count'], 
                color='darkgray', alpha=0.7, label='prtType = 0')
        ax1.set_ylabel('Trade Count', fontsize=12, fontweight='bold')
        ax1.set_title('2014 prtType Zero Anomaly Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Bottom: Unique prtType values
        ax2 = axes[1]
        colors_2014 = ['red' if zero else 'green' 
                      for zero in df_2014['all_prtType_zero']]
        ax2.bar(df_2014['date'], df_2014['unique_prtTypes'], 
                color=colors_2014, alpha=0.7)
        ax2.set_ylabel('Unique prtType Values', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='All Zeros'),
            Patch(facecolor='green', alpha=0.7, label='Mixed Values')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        output_path = output_dir / "prtType_2014_anomaly.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved 2014 anomaly plot to {output_path}")
    else:
        logger.warning("No 2014 data available for anomaly plot")
    
    # 4. Heatmap of prtType values over time (sample of dates)
    logger.info("Creating prtType value heatmap...")
    
    # Get top prtType columns by total volume
    prtType_cols = [col for col in df.columns if col.startswith('prtType_')]
    prtType_totals = {col: df[col].sum() for col in prtType_cols}
    top_prtTypes = sorted(prtType_totals.items(), key=lambda x: x[1], reverse=True)[:20]
    top_cols = [col for col, _ in top_prtTypes]
    
    if top_cols:
        # Sort columns by numeric prtType value in descending order (complex trades at top)
        top_cols_sorted = sorted(top_cols, 
                                key=lambda x: int(x.split('_')[1]), 
                                reverse=True)
        
        # Sample dates for readability (every Nth date)
        sample_size = min(100, len(df))
        step = max(1, len(df) // sample_size)
        df_sample = df.iloc[::step].copy()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Use sorted columns for heatmap
        heatmap_data = df_sample[top_cols_sorted].T
        heatmap_data.columns = df_sample['date'].dt.strftime('%Y-%m-%d')
        
        # Use log scale for better visualization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heatmap_data_log = np.log10(heatmap_data + 1)
        
        sns.heatmap(heatmap_data_log, cmap='YlOrRd', ax=ax, 
                   cbar_kws={'label': 'log10(count + 1)'})
        
        # Clean up y-axis labels (remove 'prtType_' prefix)
        y_labels = [col.replace('prtType_', '') for col in top_cols_sorted]
        ax.set_yticklabels(y_labels, rotation=0)
        
        ax.set_xlabel('Date (sampled)', fontsize=12, fontweight='bold')
        ax.set_ylabel('prtType Value (descending)', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 prtType Values Over Time (Log Scale, Ordered)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout()
        output_path = output_dir / "prtType_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved heatmap to {output_path}")


def generate_summary_report(df: pd.DataFrame, logger) -> str:
    """
    Generate a comprehensive text summary of findings.
    
    Args:
        df: Anomaly-flagged DataFrame
        logger: Logger instance
        
    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("prtType DISTRIBUTION ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Date range
    lines.append(f"Analysis Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    lines.append(f"Total Days Analyzed: {len(df)}")
    lines.append("")
    
    # Overall statistics
    lines.append("OVERALL STATISTICS:")
    lines.append(f"  Total Trades (all days): {df['total_trades'].sum():,}")
    lines.append(f"  Average Daily Trades: {df['total_trades'].mean():,.0f}")
    lines.append(f"  Median Daily Trades: {df['total_trades'].median():,.0f}")
    lines.append("")
    
    # Complex trades emergence
    has_complex_df = df[df['has_complex_trades']]
    if not has_complex_df.empty:
        first_complex_date = has_complex_df.iloc[0]['date']
        lines.append("COMPLEX TRADES (prtType >= 102):")
        lines.append(f"  First Appearance: {first_complex_date.strftime('%Y-%m-%d')}")
        lines.append(f"  Days with Complex Trades: {len(has_complex_df)}")
        lines.append(f"  Total Complex Trades: {df['complex_count'].sum():,}")
        
        # Percentage over time
        total_trades_since_emergence = df[df['date'] >= first_complex_date]['total_trades'].sum()
        complex_trades_total = df[df['date'] >= first_complex_date]['complex_count'].sum()
        if total_trades_since_emergence > 0:
            pct = (complex_trades_total / total_trades_since_emergence) * 100
            lines.append(f"  % of All Trades (since emergence): {pct:.2f}%")
    else:
        lines.append("COMPLEX TRADES (prtType >= 102):")
        lines.append("  ⚠️  NO COMPLEX TRADES FOUND IN ENTIRE DATASET")
    lines.append("")
    
    # 2014 Zero Anomaly
    df_2014 = df[df['date'].dt.year == 2014]
    if not df_2014.empty:
        all_zeros_2014 = df_2014[df_2014['all_prtType_zero']]
        lines.append("2014 ZERO ANOMALY:")
        lines.append(f"  Total 2014 Days: {len(df_2014)}")
        lines.append(f"  Days with ONLY prtType=0: {len(all_zeros_2014)}")
        
        if not all_zeros_2014.empty:
            lines.append(f"  First All-Zero Date: {all_zeros_2014.iloc[0]['date'].strftime('%Y-%m-%d')}")
            lines.append(f"  Last All-Zero Date: {all_zeros_2014.iloc[-1]['date'].strftime('%Y-%m-%d')}")
            lines.append(f"  ⚠️  {len(all_zeros_2014)} days in 2014 have exclusively prtType=0")
        else:
            lines.append("  ✓ No all-zero days found in 2014")
    else:
        lines.append("2014 ZERO ANOMALY:")
        lines.append("  No 2014 data available")
    lines.append("")
    
    # Anomaly summary
    anomaly_counts = df['anomaly_type'].value_counts()
    lines.append("ANOMALY DETECTION:")
    for anomaly_type, count in anomaly_counts.items():
        lines.append(f"  {anomaly_type}: {count} days")
    lines.append("")
    
    # prtType diversity
    lines.append("PRTTYPE DIVERSITY:")
    lines.append(f"  Average Unique prtTypes per Day: {df['unique_prtTypes'].mean():.1f}")
    lines.append(f"  Min Unique prtTypes: {df['unique_prtTypes'].min()}")
    lines.append(f"  Max Unique prtTypes: {df['unique_prtTypes'].max()}")
    
    # Get all unique prtTypes across all dates
    prtType_cols = [col for col in df.columns if col.startswith('prtType_')]
    all_prtTypes = [int(col.split('_')[1]) for col in prtType_cols]
    lines.append(f"  Total Unique prtTypes (all dates): {len(all_prtTypes)}")
    lines.append("")
    
    # First appearance of each prtType category
    lines.append("PRTTYPE FIRST APPEARANCES:")
    
    # Find first date for prtType=0
    if 'prtType_0' in df.columns:
        df_with_zero = df[df['prtType_0'] > 0]
        if not df_with_zero.empty:
            lines.append(f"  prtType = 0: {df_with_zero.iloc[0]['date'].strftime('%Y-%m-%d')}")
    
    # Find first simple trade (73-101)
    df_with_simple = df[df['has_simple_trades']]
    if not df_with_simple.empty:
        lines.append(f"  prtType 73-101 (simple): {df_with_simple.iloc[0]['date'].strftime('%Y-%m-%d')}")
    
    # Find first complex trade (>=102)
    if not has_complex_df.empty:
        lines.append(f"  prtType >= 102 (complex): {first_complex_date.strftime('%Y-%m-%d')}")
    
    lines.append("")
    lines.append("=" * 80)
    
    summary = "\n".join(lines)
    
    # Log to console
    logger.info("\n" + summary)
    
    return summary


def generate_reports(df: pd.DataFrame, output_dir: Path, logger):
    """
    Generate CSV and text report files.
    
    Args:
        df: Anomaly-flagged DataFrame
        output_dir: Directory to save reports
        logger: Logger instance
    """
    logger.info("Generating reports...")
    
    # 1. Summary by date
    summary_file = output_dir / "prtType_summary_by_date.csv"
    df.to_csv(summary_file, index=False)
    logger.info(f"Saved daily summary to {summary_file}")
    
    # 2. First appearances
    prtType_cols = [col for col in df.columns if col.startswith('prtType_')]
    first_appearances = []
    
    for col in prtType_cols:
        prtType_value = int(col.split('_')[1])
        df_with_prtType = df[df[col] > 0]
        
        if not df_with_prtType.empty:
            first_date = df_with_prtType.iloc[0]['date']
            first_count = df_with_prtType.iloc[0][col]
            total_count = df[col].sum()
            days_present = len(df_with_prtType)
            
            first_appearances.append({
                'prtType': prtType_value,
                'first_appearance': first_date.strftime('%Y-%m-%d'),
                'first_day_count': first_count,
                'total_count': total_count,
                'days_present': days_present,
                'category': 'zero' if prtType_value == 0 else 
                           ('simple' if 73 <= prtType_value < 102 else 
                            ('complex' if prtType_value >= 102 else 'other'))
            })
    
    first_appearances_df = pd.DataFrame(first_appearances)
    first_appearances_df = first_appearances_df.sort_values('prtType')
    
    first_appearances_file = output_dir / "prtType_first_appearances.csv"
    first_appearances_df.to_csv(first_appearances_file, index=False)
    logger.info(f"Saved first appearances to {first_appearances_file}")
    
    # 3. Anomaly dates
    anomaly_dates = df[df['anomaly_type'] != 'normal']
    anomaly_file = output_dir / "prtType_anomaly_dates.txt"
    
    with open(anomaly_file, 'w') as f:
        f.write("ANOMALOUS DATES\n")
        f.write("=" * 80 + "\n\n")
        
        for anomaly_type in anomaly_dates['anomaly_type'].unique():
            f.write(f"\n{anomaly_type.upper()} ({len(anomaly_dates[anomaly_dates['anomaly_type'] == anomaly_type])} dates):\n")
            f.write("-" * 80 + "\n")
            
            dates = anomaly_dates[anomaly_dates['anomaly_type'] == anomaly_type]['date']
            for date in dates:
                f.write(f"  {date.strftime('%Y-%m-%d')}\n")
    
    logger.info(f"Saved anomaly dates to {anomaly_file}")


def main():
    """Main execution function."""
    logger = initialize_main()
    logger.info("=" * 80)
    logger.info("Starting prtType Distribution Analysis")
    logger.info("=" * 80)
    
    # Validate PROCESSED_PATH exists
    if not PROCESSED_PATH.exists():
        logger.error(f"PROCESSED_PATH does not exist: {PROCESSED_PATH}")
        sys.exit(1)
    
    logger.info(f"Scanning PROCESSED_PATH: {PROCESSED_PATH}")
    
    # Get all date folders
    daily_folders = sorted([
        folder for folder in PROCESSED_PATH.iterdir() 
        if folder.is_dir() and looks_like_ymd(folder.name)
    ])
    
    if not daily_folders:
        logger.error("No valid date folders found")
        sys.exit(1)
    
    logger.info(f"Found {len(daily_folders)} date folders")
    logger.info(f"First date: {daily_folders[0].name}")
    logger.info(f"Last date: {daily_folders[-1].name}")
    logger.info("")
    
    # Initialize DuckDB connection
    con = duckdb.connect(database=':memory:')
    
    # Scan all dates
    logger.info("Scanning prtType distribution for each date...")
    all_results = []
    
    for i, date_folder in enumerate(daily_folders, 1):
        if i % 50 == 0 or i == 1:
            logger.info(f"Processing {i}/{len(daily_folders)}: {date_folder.name}")
        
        result = scan_date_prtType_distribution(date_folder, con)
        all_results.append(result)
    
    logger.info(f"Completed scanning {len(all_results)} dates")
    logger.info("")
    
    # Build time-series DataFrame
    logger.info("Building time-series DataFrame...")
    df = build_timeseries_dataframe(all_results)
    logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    
    # Detect anomalies
    logger.info("Detecting anomalies...")
    df = detect_anomalies(df)
    anomaly_count = len(df[df['anomaly_type'] != 'normal'])
    logger.info(f"Detected {anomaly_count} anomalous dates")
    logger.info("")
    
    # Generate reports
    generate_reports(df, REPORTS_DIR, logger)
    logger.info("")
    
    # Create visualizations
    create_visualizations(df, FIGURES_DIR, logger)
    logger.info("")
    
    # Generate and display summary
    summary = generate_summary_report(df, logger)
    
    # Save summary to file
    summary_file = REPORTS_DIR / "prtType_analysis_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    logger.info(f"Saved summary report to {summary_file}")
    
    # Close DuckDB connection
    con.close()
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Analysis completed successfully!")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  Reports: {REPORTS_DIR}")
    logger.info(f"  Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()

