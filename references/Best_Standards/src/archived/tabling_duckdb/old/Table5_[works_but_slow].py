# uv run src/tabling/Table5_[works_but_slow].py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
import pandas as pd
import numpy as np
from typing import Optional

from src.config import config_settings, initialize_main, DaskManager
# from src.config.config_settings import PATHS

PATHS = "_1_PROCESSED_TRADE_DATA_PARQUET_"

# Set output path relative to project root
OUTPUT_DIR = PROJECT_ROOT / "TeX" / "tables"
OUTPUT_PATH = OUTPUT_DIR / "Table_complex_strategies_TEST.tex"


def check_flag_exp_strike(legs: pd.DataFrame) -> tuple[bool, bool, bool]:
    """Check if all legs have same flag, expiration, and strike"""
    same_flag: bool = legs["okey_cp"].nunique() == 1
    same_exp: bool = legs["expdate"].nunique() == 1
    same_strike: bool = legs["okey_xx"].nunique() == 1
    return same_flag, same_exp, same_strike


def sign_complex_trade(legs: pd.DataFrame, 
                       sum_mode: Optional[bool] = None) -> str:
    """Determine if a strategy is Long, Short, or Midpoint based on price comparison"""
    n_legs = legs.shape[0]
    
    if legs is None or n_legs < 2:
        return "Undetermined"

    if n_legs in (3, 4):
        legs: pd.DataFrame = legs.sort_values("okey_xx").reset_index(drop=True)
    
    # Extract prices and midpoints for all legs
    prices: np.ndarray = legs['prtPrice'].values
    midpoints: np.ndarray = legs['midpointNBBO'].values

    # Check for NaN, null, or invalid values
    if np.any(pd.isna(prices)) or np.any(pd.isna(midpoints)):
        return "Undetermined"

    if n_legs == 2:
        if sum_mode:
            netprice: float = abs(prices[0] + prices[1])
            netmid: float = abs(midpoints[0] + midpoints[1])
        else:
            netprice: float = abs(prices[0] - prices[1])
            netmid: float = abs(midpoints[0] - midpoints[1])

    if n_legs == 3:
        netprice: float = abs(prices[0] + prices[2] - 2*prices[1])
        netmid: float = abs(midpoints[0] + midpoints[2] - 2*midpoints[1])
    
    if n_legs == 4: 
        netprice: float = abs(prices[0]-prices[1]) + abs(prices[2]-prices[3])
        netmid: float = abs(midpoints[0] - midpoints[1]) + abs(midpoints[2] - midpoints[3])
    
    if pd.isna(netprice) or pd.isna(netmid) or not np.isfinite(netprice) or not np.isfinite(netmid):
        return "Undetermined"
    
    # Determine sign based on comparison
    if netprice > netmid:
        return "Long"
    elif netprice < netmid:
        return "Short"
    else:
        return "Midpoint"


def classify_strategy(group_df: pd.DataFrame) -> str:
    """Classify a complex strategy based on its legs"""
    
    legs = group_df[[
        "okey_cp", "okey_xx", 
        "expiration", 
        # "okey_yr", "okey_mn", "okey_dy",
        "prtPrice", "midpointNBBO", "prtSize_agg"
    ]].copy()
    
    # legs["expdate"] = pd.to_datetime(legs[["okey_yr", "okey_mn", "okey_dy"]])
    legs["expdate"] = legs["expiration"].dt.normalize()
    legs = legs.drop(columns=["expiration"])
    
    n_legs = len(legs)
    strategy_name = "Other"
    
    # Single Leg
    if n_legs == 1:
        flag: str = legs.iloc[0]["okey_cp"]
        strategy_name = f"Single{flag}"
    
    # 2-leg strategies
    elif n_legs == 2:
        same_flag, same_exp, same_strike = check_flag_exp_strike(legs)
        
        if same_flag:
            flag: str = legs["okey_cp"].iloc[0]
            sign: str = sign_complex_trade(legs, sum_mode=False)
            
            if same_exp and not same_strike:
                strategy_name = f"{sign}{flag}Spread" if sign != "Undetermined" else "Other"
            elif (not same_exp) and same_strike:
                strategy_name = f"{sign}{flag}Calendar" if sign != "Undetermined" else "Other"
            elif (not same_exp) and (not same_strike):
                strategy_name = f"{sign}{flag}Diagonal" if sign != "Undetermined" else "Other"

        elif (not same_flag):
            sign = sign_complex_trade(legs, sum_mode=True)
            
            if sign != "Undetermined":
                if same_exp and same_strike:
                    strategy_name = f"{sign}Straddle"
                elif same_exp and (not same_strike):
                    call_strikes = legs[legs["okey_cp"] == "Call"]["okey_xx"]
                    put_strikes = legs[legs["okey_cp"] == "Put"]["okey_xx"]
                    
                    if len(call_strikes) > 0 and len(put_strikes) > 0:
                        if call_strikes.iloc[0] > put_strikes.iloc[0]:
                            return f"{sign}Strangle"
        
                    # call_strike = legs[legs["okey_cp"] == "Call"]["okey_xx"].iloc[0] if len(legs[legs["okey_cp"] == "Call"]) > 0 else None
                    # put_strike = legs[legs["okey_cp"] == "Put"]["okey_xx"].iloc[0] if len(legs[legs["okey_cp"] == "Put"]) > 0 else None
                    # if pd.notna(call_strike) and pd.notna(put_strike) and (call_strike > put_strike):
                    #     strategy_name = f"{sign}Strangle"
    
    # 3-leg strategies
    elif n_legs == 3:
        same_flag, same_exp, _ = check_flag_exp_strike(legs)
        
        legs = legs.sort_values("okey_xx").reset_index(drop=True)
        sizes: np.ndarray = legs["prtSize_agg"].values
        butterfly_size_condition: bool = (2*sizes[0] == sizes[1] == 2*sizes[2])
        
        if same_flag and same_exp and butterfly_size_condition:
            flag: str = legs.iloc[0]["okey_cp"]
            sign: str = sign_complex_trade(legs)
            strategy_name = f"{sign}{flag}Butterfly" if sign != "Undetermined" else "Other"
    
    # 4-leg strategies
    elif n_legs == 4:
        same_flag, same_exp, same_strike = check_flag_exp_strike(legs)
        
        if same_flag and same_exp:
            flag: str = legs["okey_cp"].iloc[0]
            sign: str = sign_complex_trade(legs)
            strategy_name = f"{sign}{flag}Condor" if sign != "Undetermined" else "Other"
        
        elif not same_flag and same_exp:
            calls = legs[legs["okey_cp"] == "Call"]
            puts = legs[legs["okey_cp"] == "Put"]
            
            if len(calls) == 2 and len(puts) == 2:
                calls = calls.sort_values("okey_xx")
                puts = puts.sort_values("okey_xx")

                width_call = calls.iloc[1]["okey_xx"] - calls.iloc[0]["okey_xx"]
                width_put = puts.iloc[1]["okey_xx"] - puts.iloc[0]["okey_xx"]
                
                if (width_call > 0) and (width_put > 0) and (width_call == width_put):
                    netprice1 = abs(calls.iloc[0]["prtPrice"] - calls.iloc[1]["prtPrice"])
                    netprice2 = abs(puts.iloc[0]["prtPrice"] - puts.iloc[1]["prtPrice"])
                    netprice = netprice1 + netprice2

                    netmid1 = abs(calls.iloc[0]["midpointNBBO"] - calls.iloc[1]["midpointNBBO"])
                    netmid2 = abs(puts.iloc[0]["midpointNBBO"] - puts.iloc[1]["midpointNBBO"])
                    netmid = netmid1 + netmid2

                    sign = "Long" if netprice > netmid else "Short" if netprice < netmid else "Midpoint"
                    strategy_name = f"{sign}IronCondor" if sign != "Undetermined" else "Other"
    
    return strategy_name


# def categorize_strategy_groups(partition: pd.DataFrame) -> pd.DataFrame:
#     """Categorize each strategy group by type and size - VECTORIZED"""
#     if partition.empty:
#         return pd.DataFrame(columns=['category', 'strategy_type', 'count'])
    
#     # Get the size for each group (first value since all rows in group have same size)
#     sizes = partition['size'].values
    
#     # Get strategy type for each group
#     strategy_types = partition['strategy_type'].values
    
#     # Create base dataframe for 'all' category
#     all_cat = pd.DataFrame({
#         'category': 'all',
#         'strategy_type': strategy_types,
#         'count': 1
#     })
    
#     # Define size categories with their conditions
#     size_categories = [
#         ('1_10', (sizes >= 1) & (sizes <= 10)),
#         ('11_200', (sizes > 10) & (sizes <= 200)),
#         ('over_200', (sizes > 200)),
#     ]
    
#     # Create dataframes for each size category using vectorized operations
#     category_dfs = [all_cat]
    
#     for cat_name, mask in size_categories:
#         if mask.any():
#             cat_df = pd.DataFrame({
#                 'category': cat_name,
#                 'strategy_type': strategy_types[mask],
#                 'count': 1
#             })
#             category_dfs.append(cat_df)
    
#     # Concatenate all category dataframes
#     result = pd.concat(category_dfs, ignore_index=True)
    
#     # Optimize dtypes for memory efficiency
#     result['category'] = result['category'].astype('category')
#     result['strategy_type'] = result['strategy_type'].astype('category')
#     result['count'] = result['count'].astype('int64')
    
#     return result


def classify_group_wrapper(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper function for Dask groupby.apply() that classifies a strategy group
    and returns a single-row DataFrame with the strategy type and size.
    """
    if group_df.empty:
        return pd.DataFrame({
            'strategy_type': pd.Series([], dtype='object'),
            'size': pd.Series([], dtype='float64')
        })
    
    strategy_type = classify_strategy(group_df)
    size = float(group_df['prtSize_agg'].iloc[0])  # Ensure float type
    
    # Create DataFrame with explicit column dtypes
    result = pd.DataFrame({
        'strategy_type': pd.Series([strategy_type], dtype='object'),
        'size': pd.Series([size], dtype='float64')
    })
    
    return result


if __name__ == '__main__':
    
    logger = initialize_main()
    logger.info("Starting Table5.py script.")
    logger.info(f"Reading Parquet files from {PATHS}")
    
    # Create temp directory for intermediate results
    TEMP_DIR = PROJECT_ROOT / "_OUTPUT_" / "temp"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_CLASSIFIED = TEMP_DIR / "classified_strategies.parquet"
    
    with DaskManager() as dask_manager:
        
        # Read all complex trades
        logger.info("Loading complex trades...")
        ddf = dd.read_parquet(
            path=PATHS,
            engine=config_settings.parquet["engine"],
            filters=[
                ('ticker_class', '==', 'Equity'),
                ('prtType', '>=', 102),  # Complex trades only
            ],
            columns=[
                'okey_tk',
                'okey_cp',
                'okey_xx',
                'expiration',
                # 'okey_yr', 'okey_mn', 'okey_dy',
                'prtPrice',
                'midpointNBBO',
                'prtSize_agg',
                'prtExch',
                'prtType',
                'timestamp_ny_round3',
            ],
        )
        
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        
        # Repartition to create smaller, more manageable partitions
        # Target: ~100MB per partition
        target_partition_size = "100MB"
        logger.info(f"Repartitioning to target size: {target_partition_size}")
        ddf = ddf.repartition(partition_size=target_partition_size)
        logger.info(f"Repartitioned to {ddf.npartitions} partitions")
        
        # Define grouping columns (same as complex_trades.py)
        grouping_cols: list[str] = ["okey_tk", "prtExch", "prtType", "timestamp_ny_round3"]
        grouping_cols = [col for col in grouping_cols if col in ddf.columns]
        
        logger.info("Grouping complex trades to identify strategies...")
                
        # Define meta with explicit dtypes to ensure consistency
        meta = pd.DataFrame({
            'strategy_type': pd.Series([], dtype='object'),
            'size': pd.Series([], dtype='float64')
        })
        
        classified = ddf.groupby(grouping_cols, observed=True).apply(
            classify_group_wrapper,
            meta=meta
        )

        logger.info("Aggregating results by strategy type and size category...")

        # CRITICAL: Reset index first to convert MultiIndex to regular columns
        logger.info("Resetting index after groupby operation...")
        classified = classified.reset_index(drop=True)

        # Add size category column directly in Dask (no compute yet!)
        def assign_size_category(df):
            """Assign size category to each row - vectorized"""
            # Use pd.cut for efficient categorization
            df = df.copy()
            df['size_category'] = pd.cut(
                df['size'],
                bins=[0, 10, 200, np.inf],
                labels=['1_10', '11_200', 'over_200'],
                right=True  # Include right edge: (0, 10], (10, 200], (200, inf]
            )
            return df

        # Update meta to include the new column
        meta_with_category = meta.copy()
        meta_with_category['size_category'] = pd.Series([], dtype='category')

        # Apply categorization in Dask (still lazy, no compute)
        logger.info("Categorizing strategies by size (in Dask)...")
        classified = classified.map_partitions(assign_size_category, meta=meta_with_category)

        # Aggregate counts in Dask using groupby (still no compute!)
        logger.info("Computing aggregations in Dask...")
        agg_by_category = classified.groupby(['size_category', 'strategy_type'], observed=True).size()
        agg_all = classified.groupby('strategy_type', observed=True).size()

        # NOW compute - but only the aggregated results (much smaller!)
        logger.info("Computing final aggregations (this triggers Dask computation)...")
        counts_by_category = agg_by_category.compute()  # Small Series
        counts_all = agg_all.compute()  # Small Series

        # Convert to DataFrame for easier manipulation
        logger.info("Building results dataframe...")
        results_list = []

        # Add 'all' category
        for strategy_type, count in counts_all.items():
            results_list.append({
                'category': 'all',
                'strategy_type': strategy_type,
                'count': count
            })

        # Add size categories
        for (size_cat, strategy_type), count in counts_by_category.items():
            results_list.append({
                'category': str(size_cat),  # Convert category to string
                'strategy_type': strategy_type,
                'count': count
            })

        result_pdf = pd.DataFrame(results_list)

        logger.info("Building LaTeX table...")
        # Continue with existing code from line 389 onwards...
                
        
       #-----------------------------------
        # # Define bins for pd.cut (matches your categories; include 'all' via a separate column or union)
        # bins = [1, 10, 200, np.inf]  # Edges for 1-10, 11-200, >200
        # labels = ['1_10', '11_200', 'over_200']  # Ignore 'under_1' if no data

        # # Add category column in Dask (vectorized, no shuffle)
        # # classified['size_category'] = dd.cut(classified['size'], bins=bins, labels=labels, right=False)
        # classified['size_category'] = classified['size'].map_partitions(
        #     pd.cut, bins, labels=labels, right=False, meta=pd.Series([], dtype='category')
        # )
        # # For 'all', we'll union later or add a dummy 'all' row per group
        # # Aggregate distributed: counts per strategy + category
        # agg = classified.groupby(['strategy_type', 'size_category']).size().reset_index(name='count')

        # # Handle 'all' by merging with total per strategy
        # totals = classified.groupby('strategy_type').size().reset_index(name='count_all')
        # totals['size_category'] = 'all'
        # totals = totals.rename(columns={'count_all': 'count'})
        # result_ddf = dd.concat([agg, totals]).repartition(partition_size='50MB')  # Smaller for final compute

        # # Now compute only the tiny aggregated result (~few hundred rows)
        # result_pdf = result_ddf.compute().sort_values(['size_category', 'strategy_type'])
        #-----------------------------------
        
        # # Reset index to convert grouping columns back to regular columns
        # logger.info("Resetting index after groupby operation...")
        # classified = classified.reset_index(drop=True)
        
        # # Write intermediate results to disk to avoid memory issues
        # logger.info(f"Writing classified strategies to disk: {TEMP_CLASSIFIED}")
        # classified.to_parquet(
        #     TEMP_CLASSIFIED,
        #     engine=config_settings.parquet["engine"],
        #     write_index=False,
        #     overwrite=True
        # )
        
        # logger.info("Reading back classified strategies...")
        # classified = dd.read_parquet(
        #     TEMP_CLASSIFIED,
        #     engine=config_settings.parquet["engine"]
        # )
        
        # logger.info("Computing strategy dataframe to pandas...")
        # strategy_df = classified.compute()
        
        # logger.info("Aggregating results by strategy type and size category...")
        
        # # Categorize by size
        # results_list = []
        
        # for category_name, size_condition in [
        #     ('all', lambda s: True),
        #     ('1_10', lambda s: (s >= 1) & (s <= 10)),
        #     ('11_200', lambda s: (s > 10) & (s <= 200)),
        #     ('over_200', lambda s: s > 200)
        # ]:
        #     if callable(size_condition):
        #         mask = size_condition(strategy_df['size'])
        #     else:
        #         mask = size_condition
            
        #     cat_df = strategy_df[mask]
        #     counts = cat_df['strategy_type'].value_counts()
            
        #     for strategy_type, count in counts.items():
        #         results_list.append({
        #             'category': category_name,
        #             'strategy_type': strategy_type,
        #             'count': count
        #         })
        
        # result_pdf = pd.DataFrame(results_list)
        
        logger.info("Building LaTeX table...")
        
        # Define all strategy types in the order they should appear
        strategy_types = [
            # 1-Leg Strategies
            'SingleCall', 'SinglePut',
            # 2-Leg Spreads
            'LongCallSpread', 'ShortCallSpread', 'MidpointCallSpread',
            'LongPutSpread', 'ShortPutSpread', 'MidpointPutSpread',
            # 2-Leg Calendars
            'LongCallCalendar', 'ShortCallCalendar', 'MidpointCallCalendar',
            'LongPutCalendar', 'ShortPutCalendar', 'MidpointPutCalendar',
            # 2-Leg Diagonals
            'LongCallDiagonal', 'ShortCallDiagonal', 'MidpointCallDiagonal',
            'LongPutDiagonal', 'ShortPutDiagonal', 'MidpointPutDiagonal',
            # 2-Leg Straddles
            'LongStraddle', 'ShortStraddle', 'MidpointStraddle',
            # 2-Leg Strangles
            'LongStrangle', 'ShortStrangle', 'MidpointStrangle',
            # 3-Leg Butterflies
            'LongCallButterfly', 'ShortCallButterfly', 'MidpointCallButterfly',
            'LongPutButterfly', 'ShortPutButterfly', 'MidpointPutButterfly',
            # 4-Leg Condors
            'LongCallCondor', 'ShortCallCondor', 'MidpointCallCondor',
            'LongPutCondor', 'ShortPutCondor', 'MidpointPutCondor',
            # 4-Leg Iron Condors
            'LongIronCondor', 'ShortIronCondor', 'MidpointIronCondor',
            # Other
            'Other'
        ]
        
        categories = ['all', '1_10', '11_200', 'over_200']
        
        # Build data structure for table
        table_data = {}
        category_totals = {}
        
        for cat in categories:
            table_data[cat] = {}
            category_total = 0
            
            cat_data = result_pdf[result_pdf['category'] == cat]
            
            for strategy in strategy_types:
                strategy_data = cat_data[cat_data['strategy_type'] == strategy]
                if len(strategy_data) > 0:
                    count = int(strategy_data['count'].iloc[0])
                else:
                    count = 0
                
                table_data[cat][strategy] = count
                category_total += count
            
            category_totals[cat] = category_total
        
        # Format functions
        def format_count_pct(count, total):
            """Format count with percentage in parentheses"""
            if count == 0 or total == 0:
                return "0 (0.0\\%)"
            percentage = (count / total) * 100
            return f"{int(count):,} ({percentage:.1f}\\%)"
        
        # Get total sample size
        total_N = category_totals['all']
        
        # Build LaTeX table
        latex_content = r"""\begin{table}[htbp]
% \begin{sidewaystable}[htbp]
    \centering
    \caption{Distribution of Complex Option Strategies by Trade Size Category}
    \subcaption*{
    {\scriptsize
    Distribution of complex option strategies across different trade size categories. Each cell shows the count of trades for that strategy type with the percentage within that size category shown in parentheses. Rows show different strategy types as classified by the algorithm. Columns represent trade size categories based on the number of contracts: All (all trades), 1--10 (size between 1 and 10), 11--200 (size between 11 and 200), and >200 (size greater than 200). Data filtered to include only equity options (ticker\_class == ``Equity'') with prtType $\geq$ 102. Total observations: $N = """ + f'{int(total_N):,}' + r"""$ complex strategies.
    \par}
    \vspace{1em}
    }
    \label{tab:complex_strategies_by_size}
    \scriptsize
    \begin{tabular}{lcccc}
    \toprule
        \textbf{Strategy Type} 
        & \textbf{All} 
        & \textbf{1--10} 
        & \textbf{11--200} 
        & \textbf{>200} \\
    \midrule
"""
        
        # Add rows for each strategy type
        for strategy in strategy_types:
            values = [format_count_pct(table_data[cat][strategy], category_totals[cat]) for cat in categories]
            latex_content += f"    {strategy} & {' & '.join(values)} \\\\\n"
        
        latex_content += r"""    \bottomrule
    \end{tabular}
% \end{sidewaystable}
\end{table}

"""
        
        # Write to file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
        
        # Clean up temporary file
        if TEMP_CLASSIFIED.exists():
            import shutil
            shutil.rmtree(TEMP_CLASSIFIED, ignore_errors=True)
            logger.info(f"Cleaned up temporary file: {TEMP_CLASSIFIED}")
        
        logger.info("Table5.py script completed successfully.")

