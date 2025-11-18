# uv run src/tabling/Table2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import dask.dataframe as dd
import numpy as np
import pandas as pd
from src.config import config_settings, initialize_main, DaskManager

# Set output path relative to project root
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TeX/tables'))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'Table2.tex')

# Define whale threshold (same as in Figure1.py)
WHALE_THRESHOLD = 270

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting Table2.py script.")

    try:
        logger.info("Loading parquet data with Dask...")
        ddf = dd.read_parquet(
            path=config_settings.PATHS,
            engine=config_settings.parquet["engine"],
            columns=[
                'prtSize_agg', 'okey_cp', 'buy_sell_class', 'trade_type', 
                'prtPrice', 'moneyness', 'leverage', 'quoted_spread', 'relative_spread',
                'moment_of_the_day', 'moneyness_class_ratio', 'bid_ask_proximity', 
                'time_to_expiry', 'trade_size_dollar', 'notional_value'
            ],
            split_row_groups='infer',
        )
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
    except Exception as e:
        logger.exception(f"Error loading parquet: {e}")
        raise

    with DaskManager() as dask_manager:
        logger.info("Computing summary statistics...")
        
        # Define whale classification
        ddf['is_whale'] = ddf['prtSize_agg'] >= WHALE_THRESHOLD
        ddf['is_unitary'] = ddf['prtSize_agg'] == 1
        
        # Compute to pandas for easier statistical calculations
        logger.info("Converting to pandas for statistical computations...")
        df = ddf.compute()
        
        # Define categories for analysis
        categories = {
            'All': df,
            'Unitaries (U)': df[df['is_unitary']],
            'Whales (W)': df[df['is_whale']]
        }
        
        # Initialize results dictionary
        results = {}
        
        # Contract Type - Call/Put percentages
        logger.info("Computing contract type statistics...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                call_pct = (cat_df['okey_cp'] == 'Call').mean() * 100
                put_pct = (cat_df['okey_cp'] == 'Put').mean() * 100
                results[f'{cat_name}_call_pct'] = call_pct
                results[f'{cat_name}_put_pct'] = put_pct
            else:
                results[f'{cat_name}_call_pct'] = np.nan
                results[f'{cat_name}_put_pct'] = np.nan
        
        # Trade Direction - Buy/Sell/Midpoint percentages
        logger.info("Computing trade direction statistics...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                sell_pct = (cat_df['buy_sell_class'] == 'Sell').mean() * 100
                buy_pct = (cat_df['buy_sell_class'] == 'Buy').mean() * 100
                midpoint_pct = (cat_df['buy_sell_class'] == 'Midpoint').mean() * 100
                results[f'{cat_name}_sell_pct'] = sell_pct
                results[f'{cat_name}_buy_pct'] = buy_pct
                results[f'{cat_name}_midpoint_pct'] = midpoint_pct
            else:
                results[f'{cat_name}_sell_pct'] = np.nan
                results[f'{cat_name}_buy_pct'] = np.nan
                results[f'{cat_name}_midpoint_pct'] = np.nan
        
        # Contract Type & Trade Direction combinations
        logger.info("Computing contract type and trade direction combinations...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                call_buy_pct = ((cat_df['okey_cp'] == 'Call') & (cat_df['buy_sell_class'] == 'Buy')).mean() * 100
                call_sell_pct = ((cat_df['okey_cp'] == 'Call') & (cat_df['buy_sell_class'] == 'Sell')).mean() * 100
                put_buy_pct = ((cat_df['okey_cp'] == 'Put') & (cat_df['buy_sell_class'] == 'Buy')).mean() * 100
                put_sell_pct = ((cat_df['okey_cp'] == 'Put') & (cat_df['buy_sell_class'] == 'Sell')).mean() * 100
                call_midpoint_pct = ((cat_df['okey_cp'] == 'Call') & (cat_df['buy_sell_class'] == 'Midpoint')).mean() * 100
                put_midpoint_pct = ((cat_df['okey_cp'] == 'Put') & (cat_df['buy_sell_class'] == 'Midpoint')).mean() * 100
                
                results[f'{cat_name}_call_buy_pct'] = call_buy_pct
                results[f'{cat_name}_call_sell_pct'] = call_sell_pct
                results[f'{cat_name}_put_buy_pct'] = put_buy_pct
                results[f'{cat_name}_put_sell_pct'] = put_sell_pct
                results[f'{cat_name}_call_midpoint_pct'] = call_midpoint_pct
                results[f'{cat_name}_put_midpoint_pct'] = put_midpoint_pct
            else:
                for suffix in ['_call_buy_pct', '_call_sell_pct', '_put_buy_pct', '_put_sell_pct', '_call_midpoint_pct', '_put_midpoint_pct']:
                    results[f'{cat_name}{suffix}'] = np.nan
        
        # Trade Type - Simple/Complex percentages
        logger.info("Computing trade type statistics...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                simple_pct = (cat_df['trade_type'] == 'simple').mean() * 100
                complex_pct = (cat_df['trade_type'] == 'complex').mean() * 100
                results[f'{cat_name}_simple_pct'] = simple_pct
                results[f'{cat_name}_complex_pct'] = complex_pct
            else:
                results[f'{cat_name}_simple_pct'] = np.nan
                results[f'{cat_name}_complex_pct'] = np.nan
        
        # Trade Size - Notional Value, Trade Size (contracts), Trade Size ($)
        logger.info("Computing trade size statistics...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                notional_median = cat_df['notional_value'].median()
                size_contracts_median = cat_df['prtSize_agg'].median()
                size_dollar_median = cat_df['trade_size_dollar'].median()
                
                results[f'{cat_name}_notional_median'] = notional_median
                results[f'{cat_name}_size_contracts_median'] = size_contracts_median
                results[f'{cat_name}_size_dollar_median'] = size_dollar_median
            else:
                results[f'{cat_name}_notional_median'] = np.nan
                results[f'{cat_name}_size_contracts_median'] = np.nan
                results[f'{cat_name}_size_dollar_median'] = np.nan
        
        # Option Characteristics - Option Price, Moneyness, Leverage
        logger.info("Computing option characteristics...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                price_median = cat_df['prtPrice'].median()
                moneyness_median = cat_df['moneyness'].median()
                leverage_median = cat_df['leverage'].median()
                
                results[f'{cat_name}_price_median'] = price_median
                results[f'{cat_name}_moneyness_median'] = moneyness_median
                results[f'{cat_name}_leverage_median'] = leverage_median
            else:
                results[f'{cat_name}_price_median'] = np.nan
                results[f'{cat_name}_moneyness_median'] = np.nan
                results[f'{cat_name}_leverage_median'] = np.nan
        
        # Market Liquidity - Quoted Spread, Relative Spread
        logger.info("Computing market liquidity statistics...")
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                quoted_spread_median = cat_df['quoted_spread'].median() * 100  # Convert to percentage
                relative_spread_median = cat_df['relative_spread'].median() * 100  # Convert to percentage
                
                results[f'{cat_name}_quoted_spread_median'] = quoted_spread_median
                results[f'{cat_name}_relative_spread_median'] = relative_spread_median
            else:
                results[f'{cat_name}_quoted_spread_median'] = np.nan
                results[f'{cat_name}_relative_spread_median'] = np.nan
        
        # Moment of the Day
        logger.info("Computing moment of the day statistics...")
        moment_categories = ['morning', 'midday', 'afternoon', 'overnight']
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                for moment in moment_categories:
                    moment_pct = (cat_df['moment_of_the_day'] == moment).mean() * 100
                    results[f'{cat_name}_{moment}_pct'] = moment_pct
            else:
                for moment in moment_categories:
                    results[f'{cat_name}_{moment}_pct'] = np.nan
        
        # Moneyness
        logger.info("Computing moneyness statistics...")
        moneyness_categories = ['OTM', 'ITM', 'ATM']
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                for moneyness in moneyness_categories:
                    moneyness_pct = (cat_df['moneyness_class_ratio'] == moneyness).mean() * 100
                    results[f'{cat_name}_{moneyness.lower()}_pct'] = moneyness_pct
            else:
                for moneyness in moneyness_categories:
                    results[f'{cat_name}_{moneyness.lower()}_pct'] = np.nan
        
        # Bid-Ask Proximity
        logger.info("Computing bid-ask proximity statistics...")
        proximity_categories = ['closer_to_Bid', 'same_distance', 'closer_to_Ask']
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                for proximity in proximity_categories:
                    proximity_pct = (cat_df['bid_ask_proximity'] == proximity).mean() * 100
                    results[f'{cat_name}_{proximity}_pct'] = proximity_pct
            else:
                for proximity in proximity_categories:
                    results[f'{cat_name}_{proximity}_pct'] = np.nan
        
        # Time to Expiration
        logger.info("Computing time to expiration statistics...")
        expiry_categories = ['less than a week', '1-2 weeks', '2-4 weeks', '1-3 months', '3-12 months', 'over a year']
        for cat_name, cat_df in categories.items():
            if len(cat_df) > 0:
                for expiry in expiry_categories:
                    expiry_pct = (cat_df['time_to_expiry'] == expiry).mean() * 100
                    results[f'{cat_name}_{expiry.replace(" ", "_").replace("-", "_")}_pct'] = expiry_pct
            else:
                for expiry in expiry_categories:
                    results[f'{cat_name}_{expiry.replace(" ", "_").replace("-", "_")}_pct'] = np.nan
        
        logger.info("Writing LaTeX table...")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(OUTPUT_PATH, 'w') as f:
            # Write table header
            f.write(r'\begin{table}[htbp]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(r'\caption{Summary Statistics by Trade Characteristics}' + '\n')
            f.write(r'\subcaption*{' + '\n')
            f.write(r'{\scriptsize' + '\n')
            f.write(r'    Summary statistics for equity option trades from 2014–2025 at millisecond resolution, covering regular and overnight trading sessions. The construction of the columns is purely cross-sectional at the monthly level: for each month (year–month) we pool all trades, rank them by trade size in dollars (option price $\times 100 \times$ contracts), and identify the first (Low, bottom decile) and tenth (High, top decile) trade-size deciles. For every metric in the table we then compute, within that month, the median over: (i) all trades ("All"), (ii) trades in the bottom decile ("Low"), and (iii) trades in the top decile ("High"). The values displayed are time–series averages of these month-by-month medians across the full sample of months. The H–L column is the average (over months) of the monthly High minus Low medians, and the associated $t$-statistic is based on the time–series of those monthly High–Low differences. Thus, each entry summarizes the central (median) within-month cross-sectional level of the variable for a given segment, averaged through time.' + '\n')
            f.write(r'\par}' + '\n')
            f.write(r'\vspace{1em}' + '\n')
            f.write(r'}' + '\n')
            f.write(r'\label{tab:summary_stats}' + '\n')
            f.write(r'\scriptsize' + '\n')
            f.write(r'\begin{tabular}{>{\raggedright\arraybackslash}p{3.0cm}lcccccc}' + '\n')
            f.write(r'\toprule' + '\n')
            f.write(r'	\textbf{Classification}' + '\n')
            f.write(r'    & \textbf{Category}' + '\n')
            f.write(r'    & \textbf{All}' + '\n')
            f.write(r'    & \textbf{Unitaries (U)}' + '\n')
            f.write(r'    & \textbf{Whales (W)}' + '\n')
            f.write(r'    & \textbf{U$-$W}' + '\n')
            f.write(r'    & \textbf{$t$-stat} \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Contract Type
            f.write(r'\multirow{2}{3.0cm}{\textbf{Contract Type}}' + '\n')
            f.write(r' & Call (\%) & ' + f'{results["All_call_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_call_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_pct"] - results["Whales (W)_call_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Put (\%) & ' + f'{results["All_put_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_put_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_pct"] - results["Whales (W)_put_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Trade Direction
            f.write(r'\multirow{2}{3.0cm}{\textbf{Trade Direction}}' + '\n')
            f.write(r' & Sell (\%) & ' + f'{results["All_sell_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_sell_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_sell_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_sell_pct"] - results["Whales (W)_sell_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Buy (\%) & ' + f'{results["All_buy_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_buy_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_buy_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_buy_pct"] - results["Whales (W)_buy_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Midpoint (\%) & ' + f'{results["All_midpoint_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_midpoint_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_midpoint_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_midpoint_pct"] - results["Whales (W)_midpoint_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Contract Type & Trade Direction
            f.write(r'\multirow{4}{3.0cm}{\textbf{Contract Type \& Trade Direction}}' + '\n')
            f.write(r'& Call Buy (\%) & ' + f'{results["All_call_buy_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_buy_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_call_buy_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_buy_pct"] - results["Whales (W)_call_buy_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Call Sell (\%) & ' + f'{results["All_call_sell_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_sell_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_call_sell_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_sell_pct"] - results["Whales (W)_call_sell_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Put Buy (\%) & ' + f'{results["All_put_buy_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_buy_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_put_buy_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_buy_pct"] - results["Whales (W)_put_buy_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Put Sell (\%) & ' + f'{results["All_put_sell_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_sell_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_put_sell_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_sell_pct"] - results["Whales (W)_put_sell_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Call Midpoint (\%) & ' + f'{results["All_call_midpoint_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_midpoint_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_call_midpoint_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_call_midpoint_pct"] - results["Whales (W)_call_midpoint_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Put Midpoint (\%) & ' + f'{results["All_put_midpoint_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_midpoint_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_put_midpoint_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_put_midpoint_pct"] - results["Whales (W)_put_midpoint_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Trade Type
            f.write(r'\multirow{2}{3.0cm}{\textbf{Trade Type}}' + '\n')
            f.write(r' & Simple (\%) & ' + f'{results["All_simple_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_simple_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_simple_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_simple_pct"] - results["Whales (W)_simple_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Complex (\%) & ' + f'{results["All_complex_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_complex_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_complex_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_complex_pct"] - results["Whales (W)_complex_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Trade Size
            f.write(r'\multirow{3}{3.0cm}{\textbf{Trade Size}}' + '\n')
            f.write(r' & Notional Value (\$) & ' + f'{results["All_notional_median"]:,.0f}' + ' & ' + f'{results["Unitaries (U)_notional_median"]:,.0f}' + ' & ' + f'{results["Whales (W)_notional_median"]:,.0f}' + ' & ' + f'{results["Unitaries (U)_notional_median"] - results["Whales (W)_notional_median"]:,.0f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Trade Size (contracts) & ' + f'{results["All_size_contracts_median"]:,.0f}' + ' & ' + f'{results["Unitaries (U)_size_contracts_median"]:,.0f}' + ' & ' + f'{results["Whales (W)_size_contracts_median"]:,.0f}' + ' & ' + f'{results["Unitaries (U)_size_contracts_median"] - results["Whales (W)_size_contracts_median"]:,.0f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Trade Size (\$) & ' + f'{results["All_size_dollar_median"]:,.0f}' + ' & ' + f'{results["Unitaries (U)_size_dollar_median"]:,.0f}' + ' & ' + f'{results["Whales (W)_size_dollar_median"]:,.0f}' + ' & ' + f'{results["Unitaries (U)_size_dollar_median"] - results["Whales (W)_size_dollar_median"]:,.0f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Option Characteristics
            f.write(r'\multirow{3}{3.0cm}{\textbf{Option Characteristics}}' + '\n')
            f.write(r' & Option Price (\$) & ' + f'{results["All_price_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_price_median"]:.2f}' + ' & ' + f'{results["Whales (W)_price_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_price_median"] - results["Whales (W)_price_median"]:.2f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Option Moneyness & ' + f'{results["All_moneyness_median"]:.3f}' + ' & ' + f'{results["Unitaries (U)_moneyness_median"]:.3f}' + ' & ' + f'{results["Whales (W)_moneyness_median"]:.3f}' + ' & ' + f'{results["Unitaries (U)_moneyness_median"] - results["Whales (W)_moneyness_median"]:.3f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Option Leverage & ' + f'{results["All_leverage_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_leverage_median"]:.2f}' + ' & ' + f'{results["Whales (W)_leverage_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_leverage_median"] - results["Whales (W)_leverage_median"]:.2f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Market Liquidity
            f.write(r'\multirow{2}{3.0cm}{\textbf{Market Liquidity}}' + '\n')
            f.write(r' & Quoted Spread (\%) & ' + f'{results["All_quoted_spread_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_quoted_spread_median"]:.2f}' + ' & ' + f'{results["Whales (W)_quoted_spread_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_quoted_spread_median"] - results["Whales (W)_quoted_spread_median"]:.2f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Relative Spread (\%) & ' + f'{results["All_relative_spread_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_relative_spread_median"]:.2f}' + ' & ' + f'{results["Whales (W)_relative_spread_median"]:.2f}' + ' & ' + f'{results["Unitaries (U)_relative_spread_median"] - results["Whales (W)_relative_spread_median"]:.2f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Moment of the Day
            f.write(r'\multirow{4}{3.0cm}{\textbf{Moment of the Day}}' + '\n')
            f.write(r' & 9:30 to 11 & ' + f'{results["All_morning_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_morning_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_morning_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_morning_pct"] - results["Whales (W)_morning_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & 11 to 13 & ' + f'{results["All_midday_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_midday_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_midday_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_midday_pct"] - results["Whales (W)_midday_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & 13 to 16 & ' + f'{results["All_afternoon_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_afternoon_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_afternoon_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_afternoon_pct"] - results["Whales (W)_afternoon_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Overnight & ' + f'{results["All_overnight_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_overnight_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_overnight_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_overnight_pct"] - results["Whales (W)_overnight_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Moneyness
            f.write(r'\multirow{3}{3.0cm}{\textbf{Moneyness}}' + '\n')
            f.write(r' & OTM & ' + f'{results["All_otm_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_otm_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_otm_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_otm_pct"] - results["Whales (W)_otm_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & ITM & ' + f'{results["All_itm_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_itm_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_itm_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_itm_pct"] - results["Whales (W)_itm_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & ATM & ' + f'{results["All_atm_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_atm_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_atm_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_atm_pct"] - results["Whales (W)_atm_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Bid-Ask Proximity
            f.write(r'\multirow{3}{3.0cm}{\textbf{Bid-Ask Proximity}}' + '\n')
            f.write(r' & Closer to Bid & ' + f'{results["All_closer_to_Bid_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_closer_to_Bid_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_closer_to_Bid_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_closer_to_Bid_pct"] - results["Whales (W)_closer_to_Bid_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Closer to Ask & ' + f'{results["All_closer_to_Ask_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_closer_to_Ask_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_closer_to_Ask_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_closer_to_Ask_pct"] - results["Whales (W)_closer_to_Ask_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Same Distance & ' + f'{results["All_same_distance_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_same_distance_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_same_distance_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_same_distance_pct"] - results["Whales (W)_same_distance_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Time to Expiration
            f.write(r'\multirow{6}{3.0cm}{\textbf{Time to Expiration}}' + '\n')
            f.write(r' & Less than a week & ' + f'{results["All_less_than_a_week_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_less_than_a_week_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_less_than_a_week_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_less_than_a_week_pct"] - results["Whales (W)_less_than_a_week_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & 1-2 weeks & ' + f'{results["All_1_2_weeks_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_1_2_weeks_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_1_2_weeks_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_1_2_weeks_pct"] - results["Whales (W)_1_2_weeks_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & 2-4 weeks & ' + f'{results["All_2_4_weeks_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_2_4_weeks_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_2_4_weeks_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_2_4_weeks_pct"] - results["Whales (W)_2_4_weeks_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & 1-3 months & ' + f'{results["All_1_3_months_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_1_3_months_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_1_3_months_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_1_3_months_pct"] - results["Whales (W)_1_3_months_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & 3-12 months & ' + f'{results["All_3_12_months_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_3_12_months_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_3_12_months_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_3_12_months_pct"] - results["Whales (W)_3_12_months_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r' & Over a year & ' + f'{results["All_over_a_year_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_over_a_year_pct"]:.1f}' + ' & ' + f'{results["Whales (W)_over_a_year_pct"]:.1f}' + ' & ' + f'{results["Unitaries (U)_over_a_year_pct"] - results["Whales (W)_over_a_year_pct"]:.1f}' + ' & ' + r'\\' + '\n')
            f.write(r'\bottomrule' + '\n')
            f.write(r'\end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        
        logger.info(f"Table2 generation completed successfully. Output saved to {OUTPUT_PATH}")
