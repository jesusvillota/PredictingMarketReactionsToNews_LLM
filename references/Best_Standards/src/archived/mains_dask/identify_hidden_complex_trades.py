# uv run src/mains/identify_hidden_complex_trades.py
"""
Hidden Complex Trades Detection Script

Identifies "hidden" complex option trades within the simple trades dataset (prtType>=73 & prtType<102)
by detecting groups of trades that appear to be coordinated multi-leg strategies.

Detection Methods:
1. Exact timestamp matching (strictest)
2. Same-size clustering within trading day  
3. Same-size clustering within 5-hour timestamp window
4. Approximate-size clustering on same trading date (within 5% size tolerance)
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import json

from src.config import config_settings, initialize_main, AdaptiveDaskManager
from src.config.config_settings import PROCESSED_PATH

# Import strategy classification functions from existing codebase
from src.mains.create_complex_dataframe import (
    classify_strategy, 
    sign_complex_trade, 
    check_flag_exp_strike
)

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "_OUTPUT_" / "__HIDDEN_COMPLEX_TRADES__"
TRADES_WITH_GROUPS_PATH = OUTPUT_DIR / "trades_with_groups.parquet"
GROUPED_STRATEGIES_PATH = OUTPUT_DIR / "grouped_strategies.parquet"
SUMMARY_STATS_PATH = OUTPUT_DIR / "summary_statistics.csv"


def detect_exact_timestamp_groups(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Detect groups of trades with exact same timestamp but different strikes/expirations.
    
    Groups by: okey_tk, timestamp_ny_round3
    Filters: groups with 2+ unique strikes/expirations
    """
    logger = initialize_main()
    logger.info("Detecting exact timestamp groups...")
    
    # Group by ticker and exact timestamp
    grouping_cols = ['okey_tk', 'timestamp_ny_round3']
    
    # Count unique strikes and expirations per group
    group_stats = ddf.groupby(grouping_cols, observed=True).agg({
        'okey_xx': 'nunique',
        'expiration': 'nunique',
        'okey_cp': 'nunique'
    }).reset_index()
    
    # Rename columns for clarity
    group_stats = group_stats.rename(columns={
        'okey_xx': 'unique_strikes',
        'expiration': 'unique_expirations', 
        'okey_cp': 'unique_call_put'
    })
    
    # Filter for groups with multiple legs (different strikes OR different expirations)
    multi_leg_groups = group_stats[
        (group_stats['unique_call_put'] > 1) |
        (group_stats['unique_strikes'] > 1) | 
        (group_stats['unique_expirations'] > 1)
    ]
    
    # Create group IDs with prefix for exact timestamp method
    multi_leg_groups = multi_leg_groups.assign(
        group_id='0_' + (multi_leg_groups.index + 1).astype(str),
        detection_method='exact_timestamp'
    )
    
    # Merge back to original data
    result = ddf.merge(
        multi_leg_groups[grouping_cols + ['group_id', 'detection_method']],
        on=grouping_cols,
        how='left'
    )
    
    logger.info(f"Found {len(multi_leg_groups)} exact timestamp groups")
    return result


def detect_same_size_same_day_groups(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Detect groups of trades with same size within the same trading day.
    
    Groups by: okey_tk, prtSize_agg, trading_date
    Filters: groups with 2+ unique strikes/expirations
    """
    logger = initialize_main()
    logger.info("Detecting same-size same-day groups...")
    
    # Extract trading date from timestamp
    ddf_with_date = ddf.copy()
    ddf_with_date['trading_date'] = ddf_with_date['timestamp_ny'].dt.date
    
    # Group by ticker, size, and trading date
    grouping_cols = ['okey_tk', 'prtSize_agg', 'trading_date']
    
    # Count unique strikes and expirations per group
    group_stats = ddf_with_date.groupby(grouping_cols, observed=True).agg({
        'okey_xx': 'nunique',
        'expiration': 'nunique',
        'okey_cp': 'nunique'
    }).reset_index()
    
    # Rename columns for clarity
    group_stats = group_stats.rename(columns={
        'okey_xx': 'unique_strikes',
        'expiration': 'unique_expirations',
        'okey_cp': 'unique_call_put'
    })
    
    # Filter for groups with multiple legs
    multi_leg_groups = group_stats[
        (group_stats['unique_call_put'] > 1) |
        (group_stats['unique_strikes'] > 1) | 
        (group_stats['unique_expirations'] > 1)
    ]
    
    # Create group IDs with prefix for same size same day method
    multi_leg_groups = multi_leg_groups.assign(
        group_id='1_' + (multi_leg_groups.index + 1).astype(str),
        detection_method='same_size_same_day'
    )
    
    # Merge back to original data
    result = ddf_with_date.merge(
        multi_leg_groups[grouping_cols + ['group_id', 'detection_method']],
        on=grouping_cols,
        how='left'
    )
    
    # Drop the temporary trading_date column
    result = result.drop(columns=['trading_date'])
    
    logger.info(f"Found {len(multi_leg_groups)} same-size same-day groups")
    return result


def detect_same_size_5hour_window_groups(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Detect groups of trades with same size within a 5-hour timestamp window.
    
    Groups by: okey_tk, prtSize_agg, 5-hour timestamp windows
    Filters: groups with 2+ unique strikes/expirations/call-put combinations
    """
    logger = initialize_main()
    logger.info("Detecting same-size 5-hour window groups...")
    
    # Group by ticker and size first
    ticker_size_groups = ddf.groupby(['okey_tk', 'prtSize_agg'], observed=True)
    
    five_hour_groups = []
    group_id_counter = 1  # Start counter for 5-hour window groups
    
    # Process each ticker-size combination
    for (ticker, size), group in ticker_size_groups:
        if len(group) < 2:
            continue
            
        # Sort by timestamp
        group_sorted = group.sort_values('timestamp_ny')
        
        # Check for trades within 5-hour windows
        timestamps = group_sorted['timestamp_ny'].values
        
        for i, start_time in enumerate(timestamps):
            # Define 5-hour window
            end_time = start_time + pd.Timedelta(hours=5)
            
            # Get trades in this window
            window_trades = group_sorted[
                (group_sorted['timestamp_ny'] >= start_time) & 
                (group_sorted['timestamp_ny'] <= end_time)
            ]
            
            if len(window_trades) >= 2:
                # Check if we have multiple legs (different strikes/expirations/call-put)
                unique_flags = window_trades['okey_cp'].nunique()
                unique_strikes = window_trades['okey_xx'].nunique()
                unique_expirations = window_trades['expiration'].nunique()
                
                if unique_flags > 1 or unique_strikes > 1 or unique_expirations > 1:
                    # Create group assignment with prefix for 5-hour window method
                    window_trades = window_trades.assign(
                        group_id='2_' + str(group_id_counter),
                        detection_method='same_size_5hour_window'
                    )
                    five_hour_groups.append(window_trades)
                    group_id_counter += 1
    
    if five_hour_groups:
        result_df = pd.concat(five_hour_groups, ignore_index=True)
        
        # Merge back to original data
        result = ddf.merge(
            result_df[['okey_tk', 'prtSize_agg', 'timestamp_ny', 'okey_xx', 'expiration', 'okey_cp', 'group_id', 'detection_method']],
            on=['okey_tk', 'prtSize_agg', 'timestamp_ny', 'okey_xx', 'expiration', 'okey_cp'],
            how='left'
        )
        
        logger.info(f"Found {len(five_hour_groups)} same-size 5-hour window groups")
    else:
        result = ddf.copy()
        result['group_id'] = ''
        result['detection_method'] = ''
        logger.info("Found 0 same-size 5-hour window groups")
    
    return result


def detect_approximate_size_same_date_groups(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Detect groups of trades with similar size (within 5% range) on the same trading date.
    
    Groups by: okey_tk, trading_date, with size tolerance of 5%
    Filters: groups with 2+ unique strikes/expirations/call-put combinations
    """
    logger = initialize_main()
    logger.info("Detecting approximate-size same-date groups...")
    
    # Extract trading date from timestamp
    ddf_with_date = ddf.copy()
    ddf_with_date['trading_date'] = ddf_with_date['timestamp_ny'].dt.date
    
    # Group by ticker and trading date first
    ticker_date_groups = ddf_with_date.groupby(['okey_tk', 'trading_date'], observed=True)
    
    approximate_size_groups = []
    group_id_counter = 1  # Start counter for approximate size groups
    
    # Process each ticker-date combination
    for (ticker, trading_date), group in ticker_date_groups:
        if len(group) < 2:
            continue
            
        # Sort by size for easier processing
        group_sorted = group.sort_values('prtSize_agg')
        
        # Check for trades with similar sizes (within 5% range)
        sizes = group_sorted['prtSize_agg'].values
        
        for i, base_size in enumerate(sizes):
            # Define 5% tolerance range
            size_tolerance = base_size * 0.05
            min_size = base_size - size_tolerance
            max_size = base_size + size_tolerance
            
            # Get trades within this size range
            size_range_trades = group_sorted[
                (group_sorted['prtSize_agg'] >= min_size) & 
                (group_sorted['prtSize_agg'] <= max_size)
            ]
            
            if len(size_range_trades) >= 2:
                # Check if we have multiple legs (different strikes/expirations/call-put)
                unique_flags = size_range_trades['okey_cp'].nunique()
                unique_strikes = size_range_trades['okey_xx'].nunique()
                unique_expirations = size_range_trades['expiration'].nunique()
                
                if unique_flags > 1 or unique_strikes > 1 or unique_expirations > 1:
                    # Create group assignment with prefix for approximate size method
                    size_range_trades = size_range_trades.assign(
                        group_id='3_' + str(group_id_counter),
                        detection_method='approximate_size_same_date'
                    )
                    approximate_size_groups.append(size_range_trades)
                    group_id_counter += 1
    
    if approximate_size_groups:
        result_df = pd.concat(approximate_size_groups, ignore_index=True)
        
        # Merge back to original data
        result = ddf_with_date.merge(
            result_df[['okey_tk', 'trading_date', 'okey_xx', 'expiration', 'okey_cp', 'prtSize_agg', 'group_id', 'detection_method']],
            on=['okey_tk', 'trading_date', 'okey_xx', 'expiration', 'okey_cp', 'prtSize_agg'],
            how='left'
        )
        
        # Drop the temporary trading_date column
        result = result.drop(columns=['trading_date'])
        
        logger.info(f"Found {len(approximate_size_groups)} approximate-size same-date groups")
    else:
        result = ddf_with_date.drop(columns=['trading_date'])
        result['group_id'] = ''
        result['detection_method'] = ''
        logger.info("Found 0 approximate-size same-date groups")
    
    return result


def apply_all_detection_methods(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Apply all detection methods and combine results.
    
    Returns DataFrame with:
    - is_hidden_complex (boolean)
    - hidden_complex_group_id (unique identifier)
    - detection_method (which method caught it)
    - n_legs_in_group (number of legs)
    """
    logger = initialize_main()
    logger.info("Applying all detection methods...")
    
    # Apply each detection method
    result_exact = detect_exact_timestamp_groups(ddf)
    result_same_day = detect_same_size_same_day_groups(ddf)
    result_5hour = detect_same_size_5hour_window_groups(ddf)
    result_approximate_size = detect_approximate_size_same_date_groups(ddf)
    
    # Combine results - prioritize exact timestamp matches
    combined_result = ddf.copy()
    
    # Add columns for hidden complex detection
    combined_result['is_hidden_complex'] = False
    combined_result['hidden_complex_group_id'] = ''
    combined_result['detection_method'] = ''
    combined_result['n_legs_in_group'] = 0
    
    # Process exact timestamp groups first (highest priority)
    exact_groups = result_exact[result_exact['group_id'] != '']
    if not exact_groups.empty:
        # Count legs per group
        leg_counts = exact_groups.groupby('group_id').size().reset_index(name='n_legs')
        exact_groups = exact_groups.merge(leg_counts, on='group_id')
        
        # Update combined result
        for idx, row in exact_groups.iterrows():
            mask = (
                (combined_result['okey_tk'] == row['okey_tk']) &
                (combined_result['timestamp_ny_round3'] == row['timestamp_ny_round3'])
            )
            combined_result.loc[mask, 'is_hidden_complex'] = True
            combined_result.loc[mask, 'hidden_complex_group_id'] = row['group_id']
            combined_result.loc[mask, 'detection_method'] = row['detection_method']
            combined_result.loc[mask, 'n_legs_in_group'] = row['n_legs']
    
    # Process same-day groups (medium priority)
    same_day_groups = result_same_day[result_same_day['group_id'] != '']
    if not same_day_groups.empty:
        # Only add if not already classified as hidden complex
        same_day_groups = same_day_groups[~same_day_groups['is_hidden_complex']]
        
        if not same_day_groups.empty:
            leg_counts = same_day_groups.groupby('group_id').size().reset_index(name='n_legs')
            same_day_groups = same_day_groups.merge(leg_counts, on='group_id')
            
            for idx, row in same_day_groups.iterrows():
                mask = (
                    (combined_result['okey_tk'] == row['okey_tk']) &
                    (combined_result['prtSize_agg'] == row['prtSize_agg']) &
                    (combined_result['timestamp_ny'].dt.date == row['trading_date'])
                )
                combined_result.loc[mask, 'is_hidden_complex'] = True
                combined_result.loc[mask, 'hidden_complex_group_id'] = row['group_id']
                combined_result.loc[mask, 'detection_method'] = row['detection_method']
                combined_result.loc[mask, 'n_legs_in_group'] = row['n_legs']
    
    # Process 5-hour window groups (third priority)
    five_hour_groups = result_5hour[result_5hour['group_id'] != '']
    if not five_hour_groups.empty:
        # Only add if not already classified as hidden complex
        five_hour_groups = five_hour_groups[~five_hour_groups['is_hidden_complex']]
        
        if not five_hour_groups.empty:
            leg_counts = five_hour_groups.groupby('group_id').size().reset_index(name='n_legs')
            five_hour_groups = five_hour_groups.merge(leg_counts, on='group_id')
            
            for idx, row in five_hour_groups.iterrows():
                mask = (
                    (combined_result['okey_tk'] == row['okey_tk']) &
                    (combined_result['prtSize_agg'] == row['prtSize_agg']) &
                    (combined_result['timestamp_ny'] == row['timestamp_ny'])
                )
                combined_result.loc[mask, 'is_hidden_complex'] = True
                combined_result.loc[mask, 'hidden_complex_group_id'] = row['group_id']
                combined_result.loc[mask, 'detection_method'] = row['detection_method']
                combined_result.loc[mask, 'n_legs_in_group'] = row['n_legs']
    
    # Process approximate-size groups (lowest priority)
    approximate_size_groups = result_approximate_size[result_approximate_size['group_id'] != '']
    if not approximate_size_groups.empty:
        # Only add if not already classified as hidden complex
        approximate_size_groups = approximate_size_groups[~approximate_size_groups['is_hidden_complex']]
        
        if not approximate_size_groups.empty:
            leg_counts = approximate_size_groups.groupby('group_id').size().reset_index(name='n_legs')
            approximate_size_groups = approximate_size_groups.merge(leg_counts, on='group_id')
            
            for idx, row in approximate_size_groups.iterrows():
                # Use trading date from the row for matching
                trading_date = row['timestamp_ny'].date() if hasattr(row['timestamp_ny'], 'date') else pd.to_datetime(row['timestamp_ny']).date()
                mask = (
                    (combined_result['okey_tk'] == row['okey_tk']) &
                    (combined_result['prtSize_agg'] == row['prtSize_agg']) &
                    (combined_result['timestamp_ny'].dt.date == trading_date)
                )
                combined_result.loc[mask, 'is_hidden_complex'] = True
                combined_result.loc[mask, 'hidden_complex_group_id'] = row['group_id']
                combined_result.loc[mask, 'detection_method'] = row['detection_method']
                combined_result.loc[mask, 'n_legs_in_group'] = row['n_legs']
    
    total_hidden_complex = combined_result['is_hidden_complex'].sum()
    logger.info(f"Total hidden complex trades identified: {total_hidden_complex}")
    
    return combined_result


def classify_hidden_complex_strategy(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify a hidden complex trade group using existing strategy classification logic.
    
    Returns DataFrame with strategy classification details.
    """
    if group_df.empty:
        return pd.DataFrame({
            'hidden_complex_group_id': pd.Series([], dtype='object'),
            'okey_tk': pd.Series([], dtype='object'),
            'n_legs': pd.Series([], dtype='int64'),
            'sign': pd.Series([], dtype='object'),
            'flag': pd.Series([], dtype='object'),
            'strategy_name': pd.Series([], dtype='object'),
            'detection_method': pd.Series([], dtype='object'),
            'avg_timestamp_ny': pd.Series([], dtype='datetime64[ns]'),
            'details': pd.Series([], dtype='object')
        })
    
    # Get group information
    group_id = group_df['hidden_complex_group_id'].iloc[0]
    okey_tk = group_df['okey_tk'].iloc[0]
    detection_method = group_df['detection_method'].iloc[0]
    n_legs = len(group_df)
    avg_timestamp = group_df['timestamp_ny'].mean()
    
    # Classify the strategy using existing logic
    sign, flag, strategy_name = classify_strategy(group_df)
    
    # Create details with leg information
    details = []
    for _, leg in group_df.iterrows():
        leg_info = {
            'okey_cp': leg['okey_cp'],
            'okey_xx': float(leg['okey_xx']),
            'expiration': leg['expiration'].isoformat() if pd.notna(leg['expiration']) else None,
            'prtPrice': float(leg['prtPrice']),
            'midpointNBBO': float(leg['midpointNBBO']),
            'prtSize_agg': float(leg['prtSize_agg']),
            'timestamp_ny': leg['timestamp_ny'].isoformat() if pd.notna(leg['timestamp_ny']) else None
        }
        details.append(leg_info)
    
    return pd.DataFrame({
        'hidden_complex_group_id': [group_id],
        'okey_tk': [okey_tk],
        'n_legs': [n_legs],
        'sign': [sign],
        'flag': [flag],
        'strategy_name': [strategy_name],
        'detection_method': [detection_method],
        'avg_timestamp_ny': [avg_timestamp],
        'details': [json.dumps(details)]
    })


def generate_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for hidden complex trades detection.
    
    Returns DataFrame with:
    - Total hidden complex groups found
    - Breakdown by detection method
    - Breakdown by strategy type
    - Top tickers by hidden complex activity
    - Temporal distribution
    """
    logger = initialize_main()
    logger.info("Generating summary statistics...")
    
    # Filter to only hidden complex trades
    hidden_complex = results_df[results_df['is_hidden_complex'] == True]
    
    if hidden_complex.empty:
        return pd.DataFrame({
            'metric': ['total_groups', 'total_trades'],
            'value': [0, 0]
        })
    
    # Get unique groups
    unique_groups = hidden_complex.groupby('hidden_complex_group_id').first()
    
    summary_data = []
    
    # Basic counts
    summary_data.append({'metric': 'total_groups', 'value': len(unique_groups)})
    summary_data.append({'metric': 'total_trades', 'value': len(hidden_complex)})
    
    # Breakdown by detection method
    method_counts = unique_groups['detection_method'].value_counts()
    for method, count in method_counts.items():
        summary_data.append({'metric': f'groups_by_method_{method}', 'value': count})
    
    # Breakdown by strategy type (if available)
    if 'strategy_name' in unique_groups.columns:
        strategy_counts = unique_groups['strategy_name'].value_counts()
        for strategy, count in strategy_counts.items():
            summary_data.append({'metric': f'groups_by_strategy_{strategy}', 'value': count})
    
    # Top tickers
    ticker_counts = unique_groups['okey_tk'].value_counts().head(10)
    for i, (ticker, count) in enumerate(ticker_counts.items()):
        summary_data.append({'metric': f'top_ticker_{i+1}_{ticker}', 'value': count})
    
    # Temporal distribution (by month)
    if 'timestamp_ny' in hidden_complex.columns:
        hidden_complex['year_month'] = hidden_complex['timestamp_ny'].dt.to_period('M')
        monthly_counts = hidden_complex['year_month'].value_counts().head(12)
        for period, count in monthly_counts.items():
            summary_data.append({'metric': f'trades_in_{period}', 'value': count})
    
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Generated {len(summary_df)} summary statistics")
    
    return summary_df


if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting identify_hidden_complex_trades.py script.")
    logger.info(f"Reading Parquet files from {PROCESSED_PATH}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with AdaptiveDaskManager() as dask_manager:
        try:
            # Load simple trades data
            logger.info("Loading simple trades data...")
            ddf = dd.read_parquet(
                path=PROCESSED_PATH,
                engine=config_settings.parquet["engine"],
                filters=[
                    ('ticker_class', '==', 'Equity'),
                    ('prtType', '>=', 73),
                    ('prtType', '<', 102),
                ],
                columns=[
                    'okey_tk', 'okey_cp', 'okey_xx', 'expiration',
                    'prtPrice', 'midpointNBBO', 'prtSize_agg',
                    'timestamp_ny_round3', 'timestamp_ny', 'prtExch', 'buy_sell_class'
                ],
                split_row_groups='infer',
            )
            
            logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
            
            # Repartition for efficiency
            target_partition_size = "100MB"
            logger.info(f"Repartitioning to target size: {target_partition_size}")
            ddf = ddf.repartition(partition_size=target_partition_size)
            logger.info(f"Repartitioned to {ddf.npartitions} partitions")
            
            # Apply all detection methods
            logger.info("Applying hidden complex trade detection...")
            results_ddf = apply_all_detection_methods(ddf)
            
            # Compute the results
            logger.info("Computing detection results...")
            results_pdf = results_ddf.compute()
            
            logger.info(f"Detection completed. Found {results_pdf['is_hidden_complex'].sum()} hidden complex trades")
            
            # Save trades with group information
            logger.info(f"Saving trades with groups to {TRADES_WITH_GROUPS_PATH}")
            results_pdf.to_parquet(
                TRADES_WITH_GROUPS_PATH,
                engine=config_settings.parquet["engine"],
                compression=config_settings.parquet["compression"],
                index=False
            )
            
            # Process hidden complex groups for strategy classification
            hidden_complex_trades = results_pdf[results_pdf['is_hidden_complex'] == True]
            
            if not hidden_complex_trades.empty:
                logger.info("Classifying hidden complex strategies...")
                
                # Group by hidden_complex_group_id and classify strategies
                grouped_strategies = []
                for group_id, group_df in hidden_complex_trades.groupby('hidden_complex_group_id'):
                    strategy_info = classify_hidden_complex_strategy(group_df)
                    grouped_strategies.append(strategy_info)
                
                if grouped_strategies:
                    strategies_pdf = pd.concat(grouped_strategies, ignore_index=True)
                    
                    # Save grouped strategies
                    logger.info(f"Saving grouped strategies to {GROUPED_STRATEGIES_PATH}")
                    strategies_pdf.to_parquet(
                        GROUPED_STRATEGIES_PATH,
                        engine=config_settings.parquet["engine"],
                        compression=config_settings.parquet["compression"],
                        index=False
                    )
                    
                    # Generate and save summary statistics
                    logger.info("Generating summary statistics...")
                    summary_stats = generate_summary_statistics(results_pdf)
                    
                    logger.info(f"Saving summary statistics to {SUMMARY_STATS_PATH}")
                    summary_stats.to_csv(SUMMARY_STATS_PATH, index=False)
                    
                    logger.info(f"Summary: Found {len(strategies_pdf)} hidden complex groups")
                    logger.info(f"Top detection methods: {strategies_pdf['detection_method'].value_counts().to_dict()}")
                    logger.info(f"Top strategies: {strategies_pdf['strategy_name'].value_counts().to_dict()}")
                else:
                    logger.warning("No grouped strategies to save")
            else:
                logger.warning("No hidden complex trades found")
            
            logger.info("identify_hidden_complex_trades.py script completed successfully.")
            
        except Exception as e:
            logger.exception(f"Error in hidden complex trades detection: {e}")
            raise
