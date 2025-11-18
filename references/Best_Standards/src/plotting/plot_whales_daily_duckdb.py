# uv run src/plotting/plot_whales_daily_duckdb.py
# uv run src/plotting/plot_whales_daily_duckdb.py --filter-type simple
# uv run src/plotting/plot_whales_daily_duckdb.py --filter-type complex
# uv run src/plotting/plot_whales_daily_duckdb.py --filter-type nonfloor-simple
# uv run src/plotting/plot_whales_daily_duckdb.py --filter-type nonfloor-complex
# uv run src/plotting/plot_whales_daily_duckdb.py --filter-type floor-simple
# uv run src/plotting/plot_whales_daily_duckdb.py --filter-type floor-complex
# uv run src/plotting/plot_whales_daily_duckdb.py --prttype 77

from pathlib import Path
import sys
import argparse
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import initialize_main
from src.config.config_settings import PROCESSED_PATH, plotting
from src.config.logger import get_logger


def build_prttype_filter(filter_type=None, prttype_values=None):
    """
    Build SQL WHERE clause for prtType filtering.
    
    Args:
        filter_type: categorical filter ('simple', 'complex', 'nonfloor-simple', 
                     'nonfloor-complex', 'floor-simple', 'floor-complex')
        prttype_values: specific prtType value(s) as list of integers
    
    Returns:
        SQL WHERE clause string (without WHERE keyword)
    """
    if prttype_values:
        # Specific prtType filtering
        if len(prttype_values) == 1:
            return f"AND prtType = {prttype_values[0]}"
        else:
            values_str = ', '.join(map(str, prttype_values))
            return f"AND prtType IN ({values_str})"
    
    elif filter_type:
        # Categorical filtering
        filter_map = {
            'simple': 'prtType >= 73 AND prtType < 102',
            'complex': 'prtType >= 102',
            'nonfloor-simple': 'prtType >= 73 AND prtType < 101',
            'nonfloor-complex': 'prtType >= 102 AND prtType NOT IN (105, 112, 115)',
            'floor-simple': 'prtType = 101',
            'floor-complex': 'prtType IN (105, 112, 115)'
        }
        
        condition = filter_map.get(filter_type.lower())
        if condition:
            return f"AND {condition}"
    
    # No filter
    return ''


def count_whales_daily_duckdb(filter_type=None, prttype_values=None):
    """
    Count whales (prtSize_agg > 200) per day using DuckDB.
    
    Args:
        filter_type: categorical filter for prtType
        prttype_values: specific prtType value(s) as list
    """
    logger = initialize_main()
    
    # Build filter condition
    prttype_filter = build_prttype_filter(filter_type, prttype_values)
    
    # Generate file name based on filter
    if prttype_values:
        if len(prttype_values) == 1:
            file_suffix = f"prtType_{prttype_values[0]}"
        else:
            file_suffix = "prtType_" + "_".join(map(str, prttype_values))
    elif filter_type:
        file_suffix = filter_type.replace('-', '_')
    else:
        file_suffix = "all"
    
    FILE_NAME = f"{file_suffix}"
    
    # Generate plot title
    if prttype_values:
        if len(prttype_values) == 1:
            title_suffix = f"prtType = {prttype_values[0]}"
        else:
            values_str = ', '.join(map(str, prttype_values))
            title_suffix = f"prtType IN ({values_str})"
    elif filter_type:
        title_suffix = filter_type.replace('-', ' ').title()
    else:
        title_suffix = "All prtTypes"
    
    plot_title = f"Daily Whale Counts - {title_suffix}"
    
    # Resolve paths
    processed_path = PROCESSED_PATH.resolve()
    output_dir = plotting["output_path"] / "whale_counts_daily"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing data from: {processed_path}")
    logger.info(f"Applying filter: {title_suffix}")
    
    # Connect to DuckDB
    con = duckdb.connect()
    
    try:
        query = f"""
        WITH filtered_data AS (
            SELECT 
                timestamp_ny,
                prtSize_agg
            FROM read_parquet('{processed_path}/**/*.parquet', hive_partitioning=0)
            WHERE prtSize_agg > 200
                AND timestamp_ny IS NOT NULL
                AND prtSize_agg IS NOT NULL
                {prttype_filter}
        )
        SELECT 
            DATE(timestamp_ny) AS date,
            COUNT(*) AS whale_count
        FROM filtered_data
        GROUP BY DATE(timestamp_ny)
        ORDER BY date
        """
        
        logger.info("Executing DuckDB query with column pruning...")
        result_df = con.execute(query).fetchdf()
        
        logger.info(f"Found {len(result_df)} days with whale trades")
        logger.info(f"Total whale observations: {result_df['whale_count'].sum():,}")
        
        # Format date column as YYYY-MM-DD
        result_df['date'] = pd.to_datetime(result_df['date']).dt.strftime('%Y-%m-%d')
        
        # Save CSV
        csv_path = output_dir / f"{FILE_NAME}.csv"
        result_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV to: {csv_path}")
        
        # Plot time series
        plot_path = output_dir / f"{FILE_NAME}.png"
        result_df['date_dt'] = pd.to_datetime(result_df['date'])
        
        plt.figure(figsize=(12, 5))
        plt.plot(result_df['date_dt'], result_df['whale_count'], linewidth=1.8)
        plt.title(plot_title)
        plt.xlabel("Date")
        plt.ylabel("Whale Count")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        
        # Format dates
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"Saved plot to: {plot_path}")
        
        return result_df
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count daily whale trades with optional prtType filtering"
    )
    
    # Create mutually exclusive group for filter options
    filter_group = parser.add_mutually_exclusive_group()
    
    filter_group.add_argument(
        '--filter-type',
        choices=['simple', 'complex', 'nonfloor-simple', 'nonfloor-complex', 
                 'floor-simple', 'floor-complex'],
        help='Categorical filter for prtType'
    )
    
    filter_group.add_argument(
        '--prttype',
        nargs='+',
        type=int,
        metavar='N',
        help='Specific prtType value(s) to filter by (e.g., --prttype 77 or --prttype 101 105 112)'
    )
    
    args = parser.parse_args()
    
    # Call function with parsed arguments
    result = count_whales_daily_duckdb(
        filter_type=args.filter_type,
        prttype_values=args.prttype
    )
    
    print(f"\nFirst few rows:")
    print(result.head())
