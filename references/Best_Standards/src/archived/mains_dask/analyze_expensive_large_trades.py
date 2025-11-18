# uv run src/mains/analyze_expensive_large_trades.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

from src.config import config_settings, initialize_main, DaskManager

# Set output path relative to project root
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../_OUTPUT_'))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'analysis_expensive_large_trades.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting analyze_expensive_large_trades.py script.")
    logger.info(f"Output will be saved to: {OUTPUT_PATH}")

    with DaskManager() as dask_manager:
        try:
            logger.info("Loading parquet data with Dask...")
            ddf = dd.read_parquet(
                path=config_settings.PROCESSED_PATH,
                engine=config_settings.parquet["engine"],
                filters=[
                    ('ticker_class', '==', 'Equity'),
                    ('prtType', '>=', 73),
                    ('prtSize_agg', '>=', 1001),
                    ('prtSize_agg', '<=', 10000),
                ],
                columns=[
                    # Identifiers
                    'okey_tk', 'okey_cp', 'okey_xx',
                    # Pricing
                    'prtPrice', 'oBid', 'oAsk', 'spread', 'midpointNBBO', 'uPrc',
                    # Trade metrics
                    'prtSize_agg', 'trade_size_dollar', 'notional_value',
                    # Spreads
                    'quoted_spread', 'relative_spread',
                    # Option characteristics
                    'moneyness', 'leverage', 'prtIv', 'prtDe',
                    # Classifications
                    'trade_type', 'moneyness_class_ratio', 'time_to_expiry', 
                    'moment_of_the_day', 'bid_ask_proximity'
                ],
                split_row_groups='infer',
            )
            logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
            
        except Exception as e:
            logger.exception(f"Error loading parquet: {e}")
            raise

        # Remove rows with missing prtSize_agg
        ddf_nonna = ddf.dropna(subset=['prtSize_agg'])
        logger.info("Removed rows with missing prtSize_agg")

        # Define variables to analyze
        variables_to_analyze = [
            'prtPrice', 'oBid', 'oAsk', 'spread', 'midpointNBBO', 'uPrc',
            'trade_size_dollar', 'notional_value', 'prtSize_agg',
            'quoted_spread', 'relative_spread',
            'moneyness', 'leverage', 'prtIv', 'prtDe'
        ]

        # Define subcategories for analysis
        subcategories = {
            'Contract_Type': {
                'column': 'okey_cp',
                'values': ['Call', 'Put']
            },
            'Trade_Type': {
                'column': 'trade_type', 
                'values': ['simple', 'complex']
            },
            'Moneyness': {
                'column': 'moneyness_class_ratio',
                'values': ['OTM', 'ATM', 'ITM']
            },
            'Time_to_Expiry': {
                'column': 'time_to_expiry',
                'values': ['less than a week', '1-2 weeks', '2-4 weeks', '1-3 months', '3-12 months', 'over a year']
            },
            'Moment_of_Day': {
                'column': 'moment_of_the_day',
                'values': ['morning', 'midday', 'afternoon', 'overnight']
            },
            'Bid_Ask_Proximity': {
                'column': 'bid_ask_proximity',
                'values': ['closer_to_bid', 'same_distance', 'closer_to_ask']
            }
        }

        def compute_comprehensive_stats(filtered_ddf: dd.DataFrame, variable: str) -> Dict[str, Any]:
            """Compute comprehensive statistics for a variable"""
            stats = {}
            
            # Filter out nulls/NaNs and infinite values
            clean_data = filtered_ddf[variable].dropna()
            clean_data = clean_data[(clean_data != np.inf) & (clean_data != -np.inf)]
            
            # Basic statistics
            stats['count'] = clean_data.count()
            stats['mean'] = clean_data.mean()
            stats['median'] = clean_data.median_approximate()
            stats['std'] = clean_data.std()
            stats['min'] = clean_data.min()
            stats['max'] = clean_data.max()
            
            # Percentiles
            percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            for p in percentiles:
                stats[f'p{int(p*100)}'] = clean_data.quantile(p)
            
            return stats

        logger.info("Setting up lazy computations for all categories and variables...")
        
        # Create filtered dataframes for subcategories
        filtered_dfs = {'Overall': {'All': ddf_nonna}}
        
        for category, config in subcategories.items():
            filtered_dfs[category] = {}
            for value in config['values']:
                filtered_dfs[category][value] = ddf_nonna[ddf_nonna[config['column']] == value]

        # Set up all lazy computations
        all_lazy_computations = []
        computation_mapping = []
        
        for category, subcats in filtered_dfs.items():
            for subcategory, filtered_df in subcats.items():
                for variable in variables_to_analyze:
                    logger.info(f"Setting up computations for {category}/{subcategory}/{variable}")
                    stats = compute_comprehensive_stats(filtered_df, variable)
                    
                    for stat_name, computation in stats.items():
                        all_lazy_computations.append(computation)
                        computation_mapping.append((category, subcategory, variable, stat_name))

        logger.info(f"Total lazy computations: {len(all_lazy_computations)}")
        logger.info("Executing all computations with Dask...")
        
        # Execute all computations
        results = dd.compute(*all_lazy_computations)
        
        logger.info("Organizing results into structured format...")
        
        # Organize results into structured format
        results_data = []
        for i, (category, subcategory, variable, stat_name) in enumerate(computation_mapping):
            results_data.append({
                'category': category,
                'subcategory': subcategory,
                'variable': variable,
                'statistic': stat_name,
                'value': results[i]
            })

        # Convert to DataFrame and pivot
        results_df = pd.DataFrame(results_data)
        
        # Pivot to get statistics as columns
        pivot_df = results_df.pivot_table(
            index=['category', 'subcategory', 'variable'],
            columns='statistic',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Reorder columns
        stat_columns = ['count', 'mean', 'median', 'std', 'min', 'max', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']
        available_columns = [col for col in stat_columns if col in pivot_df.columns]
        final_columns = ['category', 'subcategory', 'variable'] + available_columns
        pivot_df = pivot_df[final_columns]
        
        # Round numerical values
        for col in available_columns:
            if col != 'count':  # Don't round count
                pivot_df[col] = pivot_df[col].round(4)
        
        logger.info(f"Writing results to CSV: {OUTPUT_PATH}")
        pivot_df.to_csv(OUTPUT_PATH, index=False)
        
        logger.info(f"Analysis completed successfully!")
        logger.info(f"Total rows in output: {len(pivot_df)}")
        logger.info(f"Categories analyzed: {pivot_df['category'].nunique()}")
        logger.info(f"Variables analyzed: {pivot_df['variable'].nunique()}")
        logger.info("analyze_expensive_large_trades.py script completed successfully.")
