# uv run src/tabling/Table5_optimized.py
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

PATHS = "_1_PROCESSED_TRADE_DATA_PARQUET_"
OUTPUT_DIR = PROJECT_ROOT / "TeX" / "tables"
OUTPUT_PATH = OUTPUT_DIR / "Table_complex_strategies_TEST_MULTIROW.tex"

# Define strategy organization once (single source of truth)
LEG_GROUPS = {
    '1 Leg': ['SingleCall', 'SinglePut'],
    '2 Legs': [
        'LongCallSpread', 'ShortCallSpread', 'MidpointCallSpread',
        'LongPutSpread', 'ShortPutSpread', 'MidpointPutSpread',
        'LongCallCalendar', 'ShortCallCalendar', 'MidpointCallCalendar',
        'LongPutCalendar', 'ShortPutCalendar', 'MidpointPutCalendar',
        'LongCallDiagonal', 'ShortCallDiagonal', 'MidpointCallDiagonal',
        'LongPutDiagonal', 'ShortPutDiagonal', 'MidpointPutDiagonal',
        'LongStraddle', 'ShortStraddle', 'MidpointStraddle',
        'LongStrangle', 'ShortStrangle', 'MidpointStrangle'
    ],
    '3 Legs': [
        'LongCallButterfly', 'ShortCallButterfly', 'MidpointCallButterfly',
        'LongPutButterfly', 'ShortPutButterfly', 'MidpointPutButterfly'
    ],
    '4 Legs': [
        'LongCallCondor', 'ShortCallCondor', 'MidpointCallCondor',
        'LongPutCondor', 'ShortPutCondor', 'MidpointPutCondor',
        'LongIronCondor', 'ShortIronCondor', 'MidpointIronCondor'
    ],
    ' ': ['Other']
}

SIZE_CATEGORIES = ['all', '1_10', '11_200', 'over_200']


def check_flag_exp_strike(legs: pd.DataFrame) -> tuple[bool, bool, bool]:
    """Check if all legs have same flag, expiration, and strike"""
    same_flag = legs["okey_cp"].nunique() == 1
    same_exp = legs["expdate"].nunique() == 1
    same_strike = legs["okey_xx"].nunique() == 1
    return same_flag, same_exp, same_strike


def sign_complex_trade(legs: pd.DataFrame, sum_mode: Optional[bool] = None) -> str:
    """Determine if a strategy is Long, Short, or Midpoint based on price comparison"""
    n_legs = legs.shape[0]
    
    if n_legs < 2:
        return "Undetermined"
    
    if n_legs in (3, 4):
        legs = legs.sort_values("okey_xx").reset_index(drop=True)
    
    prices = legs['prtPrice'].values
    midpoints = legs['midpointNBBO'].values

    if np.any(pd.isna(prices)) or np.any(pd.isna(midpoints)):
        return "Undetermined"

    if n_legs == 2:
        if sum_mode:
            netprice = abs(prices[0] + prices[1])
            netmid = abs(midpoints[0] + midpoints[1])
        else:
            netprice = abs(prices[0] - prices[1])
            netmid = abs(midpoints[0] - midpoints[1])
    elif n_legs == 3:
        netprice = abs(prices[0] + prices[2] - 2*prices[1])
        netmid = abs(midpoints[0] + midpoints[2] - 2*midpoints[1])
    elif n_legs == 4:
        netprice = abs(prices[0]-prices[1]) + abs(prices[2]-prices[3])
        netmid = abs(midpoints[0] - midpoints[1]) + abs(midpoints[2] - midpoints[3])
    else:
        return "Undetermined"
    
    if pd.isna(netprice) or pd.isna(netmid) or not np.isfinite(netprice) or not np.isfinite(netmid):
        return "Undetermined"
    
    if netprice > netmid:
        return "Long"
    elif netprice < netmid:
        return "Short"
    else:
        return "Midpoint"


def classify_strategy(group_df: pd.DataFrame) -> str:
    """Classify a complex strategy based on its legs"""
    
    legs = group_df[["okey_cp", "okey_xx", "expiration", 
                     "prtPrice", "midpointNBBO", "prtSize_agg"]].copy()
    
    legs["expdate"] = legs["expiration"].dt.normalize()
    legs = legs.drop(columns=["expiration"])
    
    n_legs = len(legs)
    
    # Single Leg
    if n_legs == 1:
        return f"Single{legs.iloc[0]['okey_cp']}"
    
    # 2-leg strategies
    if n_legs == 2:
        same_flag, same_exp, same_strike = check_flag_exp_strike(legs)
        
        if same_flag:
            flag = legs["okey_cp"].iloc[0]
            sign = sign_complex_trade(legs, sum_mode=False)
            
            if sign == "Undetermined":
                return "Other"
            
            if same_exp and not same_strike:
                return f"{sign}{flag}Spread"
            elif (not same_exp) and same_strike:
                return f"{sign}{flag}Calendar"
            elif (not same_exp) and (not same_strike):
                return f"{sign}{flag}Diagonal"
        else:
            sign = sign_complex_trade(legs, sum_mode=True)
            
            if sign == "Undetermined":
                return "Other"
            
            if same_exp and same_strike:
                return f"{sign}Straddle"
            elif same_exp and (not same_strike):
                call_strikes = legs[legs["okey_cp"] == "Call"]["okey_xx"]
                put_strikes = legs[legs["okey_cp"] == "Put"]["okey_xx"]
                
                if len(call_strikes) > 0 and len(put_strikes) > 0:
                    if call_strikes.iloc[0] > put_strikes.iloc[0]:
                        return f"{sign}Strangle"
    
    # 3-leg strategies
    elif n_legs == 3:
        same_flag, same_exp, _ = check_flag_exp_strike(legs)
        
        legs = legs.sort_values("okey_xx").reset_index(drop=True)
        sizes = legs["prtSize_agg"].values
        
        if same_flag and same_exp and (2*sizes[0] == sizes[1] == 2*sizes[2]):
            sign = sign_complex_trade(legs)
            if sign != "Undetermined":
                return f"{sign}{legs.iloc[0]['okey_cp']}Butterfly"
    
    # 4-leg strategies
    elif n_legs == 4:
        same_flag, same_exp, _ = check_flag_exp_strike(legs)
        
        if same_flag and same_exp:
            sign = sign_complex_trade(legs)
            if sign != "Undetermined":
                return f"{sign}{legs['okey_cp'].iloc[0]}Condor"
        
        elif not same_flag and same_exp:
            calls = legs[legs["okey_cp"] == "Call"]
            puts = legs[legs["okey_cp"] == "Put"]
            
            if len(calls) == 2 and len(puts) == 2:
                calls = calls.sort_values("okey_xx")
                puts = puts.sort_values("okey_xx")

                width_call = calls.iloc[1]["okey_xx"] - calls.iloc[0]["okey_xx"]
                width_put = puts.iloc[1]["okey_xx"] - puts.iloc[0]["okey_xx"]
                
                if width_call > 0 and width_put > 0 and width_call == width_put:
                    netprice = (abs(calls.iloc[0]["prtPrice"] - calls.iloc[1]["prtPrice"]) +
                               abs(puts.iloc[0]["prtPrice"] - puts.iloc[1]["prtPrice"]))
                    netmid = (abs(calls.iloc[0]["midpointNBBO"] - calls.iloc[1]["midpointNBBO"]) +
                             abs(puts.iloc[0]["midpointNBBO"] - puts.iloc[1]["midpointNBBO"]))
                    
                    sign = "Long" if netprice > netmid else "Short" if netprice < netmid else "Midpoint"
                    if sign != "Undetermined":
                        return f"{sign}IronCondor"
    
    return "Other"


def classify_and_categorize_group(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Single function that both classifies strategy AND assigns size category.
    This eliminates the need for a second groupby operation.
    Returns a DataFrame with strategy_type and size_category for aggregation.
    """
    if group_df.empty:
        return pd.DataFrame({
            'strategy_type': pd.Series([], dtype='object'),
            'size_category': pd.Series([], dtype='category')
        })
    
    # Classify the strategy
    strategy_type = classify_strategy(group_df)
    size = float(group_df['prtSize_agg'].iloc[0])
    
    # Assign size category immediately
    if size <= 10:
        size_cat = '1_10'
    elif size <= 200:
        size_cat = '11_200'
    else:
        size_cat = 'over_200'
    
    return pd.DataFrame({
        'strategy_type': [strategy_type],
        'size_category': pd.Categorical([size_cat], categories=['1_10', '11_200', 'over_200'])
    })


if __name__ == '__main__':
    
    logger = initialize_main()
    logger.info("Starting optimized Table5.py script.")
    logger.info(f"Reading Parquet files from {PATHS}")
    
    with DaskManager() as dask_manager:
        
        # Read complex trades
        logger.info("Loading complex trades...")
        ddf = dd.read_parquet(
            path=PATHS,
            engine=config_settings.parquet["engine"],
            filters=[
                ('ticker_class', '==', 'Equity'),
                ('prtType', '>=', 102),
            ],
            columns=[
                'okey_tk', 'okey_cp', 'okey_xx', 'expiration',
                'prtPrice', 'midpointNBBO', 'prtSize_agg',
                'prtExch', 'prtType', 'timestamp_ny_round3',
            ],
        )
        
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        
        # Repartition for memory efficiency
        target_partition_size = "100MB"
        logger.info(f"Repartitioning to target size: {target_partition_size}")
        ddf = ddf.repartition(partition_size=target_partition_size)
        logger.info(f"Repartitioned to {ddf.npartitions} partitions")
        
        # Define grouping columns
        grouping_cols = ["okey_tk", "prtExch", "prtType", "timestamp_ny_round3"]
        grouping_cols = [col for col in grouping_cols if col in ddf.columns]
        
        logger.info("Grouping and classifying strategies...")
        logger.info("(Processing all strategy legs together)")
        
        # Define meta for the output
        meta = pd.DataFrame({
            'strategy_type': pd.Series([], dtype='object'),
            'size_category': pd.Categorical([], categories=['1_10', '11_200', 'over_200'])
        })
        
        # Single groupby operation that does BOTH classification AND categorization
        logger.info("Applying combined classification and categorization...")
        classified = ddf.groupby(grouping_cols, observed=True).apply(
            classify_and_categorize_group,
            meta=meta
        ).reset_index(drop=True)
        
        # Now aggregate - but only compute ONCE at the very end
        logger.info("Building aggregation plan (lazy operations)...")
        
        # Create both aggregations in the lazy execution graph
        agg_by_category = classified.groupby(['size_category', 'strategy_type'], observed=True).size()
        agg_all = classified.groupby('strategy_type', observed=True).size()
        
        # CRITICAL: Use dask.compute() to compute BOTH at once, sharing computation
        logger.info("Computing all aggregations in a single pass...")
        import dask
        counts_by_category, counts_all = dask.compute(agg_by_category, agg_all)
        
        # Convert to unified DataFrame with pivot
        logger.info("Building results table...")
        counts_df = pd.concat([
            counts_all.rename('count').reset_index().assign(category='all'),
            counts_by_category.rename('count').reset_index().rename(columns={'size_category': 'category'})
        ], ignore_index=True)
        
        # Pivot to get strategies as rows, categories as columns
        pivot = counts_df.pivot(index='strategy_type', columns='category', values='count').fillna(0).astype(int)
        pivot = pivot.reindex(columns=SIZE_CATEGORIES, fill_value=0)
        
        # Calculate totals and format helper
        totals = pivot.sum(axis=0)
        
        def fmt(count, total):
            """Format count with percentage"""
            return "0 (0.0\\%)" if count == 0 or total == 0 else f"{int(count):,} ({count/total*100:.1f}\\%)"
        
        # Generate LaTeX table
        logger.info("Building LaTeX table...")
        total_N = totals['all']
        
        latex = [
            r"\begin{table}[htbp]",
            r"    \centering",
            r"    \caption{Distribution of Complex Option Strategies by Trade Size Category}",
            r"    \subcaption*{",
            r"    {\scriptsize",
            r"    Distribution of complex option strategies across different trade size categories. Each cell shows the count of trades for that strategy type with the percentage within that size category shown in parentheses. Rows show different strategy types as classified by the algorithm. Columns represent trade size categories based on the number of contracts: All (all trades), 1--10 (size between 1 and 10), 11--200 (size between 11 and 200), and >200 (size greater than 200). Data filtered to include only equity options (ticker\_class == ``Equity'') with prtType $\geq$ 102. Total observations: $N = " + f"{int(total_N):,}" + r"$ complex strategies.",
            r"    \par}",
            r"    \vspace{1em}",
            r"    }",
            r"    \label{tab:complex_strategies_by_size}",
            r"    \scriptsize",
            r"    \begin{tabular}{llcccc}",
            r"    \toprule",
            r"        \textbf{Legs} & \textbf{Strategy Type} & \textbf{All} & \textbf{1--10} & \textbf{11--200} & \textbf{>200} \\",
            r"    \midrule"
        ]
        
        # Add strategy rows grouped by legs
        for leg_label, strategies in LEG_GROUPS.items():
            for idx, strategy in enumerate(strategies):
                # Get counts for this strategy (default to 0 if not in results)
                counts = pivot.loc[strategy] if strategy in pivot.index else pd.Series(0, index=SIZE_CATEGORIES)
                values = ' & '.join(fmt(counts[cat], totals[cat]) for cat in SIZE_CATEGORIES)
                
                if idx == 0:
                    latex.append(f"    \\multirow{{{len(strategies)}}}{{*}}{{\\textbf{{{leg_label}}}}}")
                    latex.append(f"    & {strategy} & {values} \\\\")
                else:
                    latex.append(f"    & {strategy} & {values} \\\\")
            latex.append(r"    \midrule")
        
        # Add total row
        total_vals = ' & '.join(fmt(totals[cat], totals[cat]) for cat in SIZE_CATEGORIES)
        latex.extend([
            f"    \\textbf{{Total}} &  & {total_vals} \\\\",
            r"    \bottomrule",
            r"    \end{tabular}",
            r"\end{table}"
        ])
        
        latex_content = '\n'.join(latex) + '\n'
        
        # Write to file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
        logger.info("Optimized Table5.py script completed successfully.")