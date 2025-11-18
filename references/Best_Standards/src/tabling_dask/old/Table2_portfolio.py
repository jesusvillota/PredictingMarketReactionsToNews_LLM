# uv run src/tabling/Table2_portfolio.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import dask.dataframe as dd
import numpy as np
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
            # path=config_settings.PATHS,
            path="__REPROCESSED__",
            engine=config_settings.parquet["engine"],
            filters=[
                ('ticker_class', '==', 'Equity'),
                ('prtType', '>=', 73),
            ],
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
        logger.info("Preparing lazy computations for summary statistics...")
        
        # Create boolean masks for each category
        all_mask = ddf['prtSize_agg'].notnull()
        size_1_10_mask = ddf['prtSize_agg'].between(1, 10, inclusive='both')
        size_11_200_mask = (ddf['prtSize_agg'] > 10) & (ddf['prtSize_agg'] <= 200)
        size_over_200_mask = ddf['prtSize_agg'] > 200
        
        # Define lazy computations for all statistics
        lazy_computations = []
        
        # Helper function to create lazy computations for each category
        def add_lazy_stats(mask, prefix):
            """Add lazy computations for a specific category mask"""
            # Contract Type percentages
            for ctype in ['Call', 'Put']:
                lazy_computations.append((ddf['okey_cp'] == ctype)[mask].mean() * 100)
            
            # Trade Direction percentages
            for direction in ['Buy', 'Sell', 'Midpoint']:
                lazy_computations.append((ddf['buy_sell_class'] == direction)[mask].mean() * 100)

            # Contract Type & Trade Direction combinations
            for ctype in ['Call', 'Put']:
                for direction in ['Buy', 'Sell', 'Midpoint']:
                    lazy_computations.append(((ddf['okey_cp'] == ctype) & (ddf['buy_sell_class'] == direction))[mask].mean() * 100)
            # lazy_computations.append(((ddf['okey_cp'] == 'Call') & (ddf['buy_sell_class'] == 'Buy'))[mask].mean() * 100)
            # lazy_computations.append(((ddf['okey_cp'] == 'Call') & (ddf['buy_sell_class'] == 'Sell'))[mask].mean() * 100)
            # lazy_computations.append(((ddf['okey_cp'] == 'Put') & (ddf['buy_sell_class'] == 'Buy'))[mask].mean() * 100)
            # lazy_computations.append(((ddf['okey_cp'] == 'Put') & (ddf['buy_sell_class'] == 'Sell'))[mask].mean() * 100)
            # lazy_computations.append(((ddf['okey_cp'] == 'Call') & (ddf['buy_sell_class'] == 'Midpoint'))[mask].mean() * 100)
            # lazy_computations.append(((ddf['okey_cp'] == 'Put') & (ddf['buy_sell_class'] == 'Midpoint'))[mask].mean() * 100)
            
            # Trade Type percentages
            for trade_type in ['simple', 'complex']:
                lazy_computations.append((ddf['trade_type'] == trade_type)[mask].mean() * 100)

            # Trade Size medians
            for stat in ['notional_value', 'prtSize_agg', 'trade_size_dollar', 'prtPrice', 'moneyness', 'leverage']:
                lazy_computations.append(ddf[stat][mask].median_approximate())
                
            for perc_stat in ['quoted_spread', 'relative_spread']:
                lazy_computations.append(ddf[perc_stat][mask].median_approximate() * 100)  # Convert to percentage
      
            # Moment of the Day percentages
            for moment in ['morning', 'midday', 'afternoon', 'overnight']:
                lazy_computations.append((ddf['moment_of_the_day'] == moment)[mask].mean() * 100)
            
            # Moneyness percentages
            for moneyness in ['OTM', 'ITM', 'ATM']:
                lazy_computations.append((ddf['moneyness_class_ratio'] == moneyness)[mask].mean() * 100)
            
            # Bid-Ask Proximity percentages
            for proximity in ['closer_to_bid', 'same_distance', 'closer_to_ask']:
                lazy_computations.append((ddf['bid_ask_proximity'] == proximity)[mask].mean() * 100)
            
            # Time to Expiration percentages
            for expiry in ['less than a week', '1-2 weeks', '2-4 weeks', '1-3 months', '3-12 months', 'over a year']:
                lazy_computations.append((ddf['time_to_expiry'] == expiry)[mask].mean() * 100)
        
        logger.info("Setting up lazy computations for all categories...")
        
        # Add lazy computations for each category
        add_lazy_stats(all_mask, 'All')
        add_lazy_stats(size_1_10_mask, '1_10')
        add_lazy_stats(size_11_200_mask, '11_200')
        add_lazy_stats(size_over_200_mask, 'over_200')
        
        logger.info("Computing all statistics with Dask...")
        results_list = dd.compute(*lazy_computations)
        logger.info("Computation finished. Organizing results...")
        
        # Organize results into dictionary
        results = {}
        stats_per_category = 37  # Total number of statistics per category (2+3+6+2+6+2+4+3+3+6)
        
        categories = ['All', '1_10', '11_200', 'over_200']
        # Order must match the exact order in add_lazy_stats function
        stat_names = [
            # Contract Type percentages (2)
            'call_pct', 'put_pct', 
            # Trade Direction percentages (3) 
            'buy_pct', 'sell_pct', 'midpoint_pct',
            # Contract Type & Trade Direction combinations (6)
            'call_buy_pct', 'call_sell_pct', 'call_midpoint_pct', 'put_buy_pct', 'put_sell_pct', 'put_midpoint_pct',
            # Trade Type percentages (2)
            'simple_pct', 'complex_pct', 
            # Trade Size medians (6)
            'notional_median', 'size_contracts_median', 'size_dollar_median', 'price_median', 'moneyness_median', 'leverage_median',
            # Percentage medians (2)
            'quoted_spread_median', 'relative_spread_median',
            # Moment of the Day percentages (4)
            'morning_pct', 'midday_pct', 'afternoon_pct', 'overnight_pct',
            # Moneyness percentages (3)
            'otm_pct', 'itm_pct', 'atm_pct',
            # Bid-Ask Proximity percentages (3)
            'closer_to_bid_pct', 'same_distance_pct', 'closer_to_ask_pct',
            # Time to Expiration percentages (6)
            'less_than_a_week_pct', '1_2_weeks_pct', '2_4_weeks_pct', '1_3_months_pct', '3_12_months_pct', 'over_a_year_pct'
        ]
        
        for i, cat in enumerate(categories):
            start_idx = i * stats_per_category
            end_idx = start_idx + stats_per_category
            
            for j, stat_name in enumerate(stat_names):
                results[f'{cat}_{stat_name}'] = results_list[start_idx + j]
        
        logger.info("Writing LaTeX table...")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(OUTPUT_PATH, 'w') as f:
            # Write table header
            f.write(r'\begin{table}[htbp]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(r'\caption{Summary Statistics by Trade Size Categories}' + '\n')
            f.write(r'\subcaption*{' + '\n')
            f.write(r'{\scriptsize' + '\n')
            f.write(r'    Summary statistics for equity option trades from 2014–2025 at millisecond resolution, covering regular and overnight trading sessions. Trades are grouped by trade sizes (number of contracts traded). The columns represent: All (all trades), 1--10 (trades with $1 \leq$ size $\leq$ 10), 11--200 (trades with $10 <$ size $\leq$ 200), and >200 (trades with size $>$ 200). Data filtered to include only equity options (ticker\_class == ``Equity'') with prtType $\geq$ 73. For categorical variables, we compute the overall share (percentage) across all observations in each group for the entire sample period. For numerical variables, we compute the overall median across all observations in each group for the entire sample period.' + '\n')
            f.write(r'\par}' + '\n')
            f.write(r'\vspace{1em}' + '\n')
            f.write(r'}' + '\n')
            f.write(r'\label{tab:summary_stats}' + '\n')
            f.write(r'\scriptsize' + '\n')
            f.write(r'\begin{tabular}{>{\raggedright\arraybackslash}p{3.0cm}lcccc}' + '\n')
            f.write(r'\toprule' + '\n')
            f.write(r'	\textbf{Classification}' + '\n')
            f.write(r'    & \textbf{Category}' + '\n')
            f.write(r'    & \textbf{All}' + '\n')
            f.write(r'    & \textbf{1--10}' + '\n')
            f.write(r'    & \textbf{11--200}' + '\n')
            f.write(r'    & \textbf{>200} \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Contract Type
            f.write(r'\multirow{2}{3.0cm}{\textbf{Contract Type}}' + '\n')
            f.write(r' & Call (\%) & ' + f'{results["All_call_pct"]:.1f}' + ' & ' + f'{results["1_10_call_pct"]:.1f}' + ' & ' + f'{results["11_200_call_pct"]:.1f}' + ' & ' + f'{results["over_200_call_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Put (\%) & ' + f'{results["All_put_pct"]:.1f}' + ' & ' + f'{results["1_10_put_pct"]:.1f}' + ' & ' + f'{results["11_200_put_pct"]:.1f}' + ' & ' + f'{results["over_200_put_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Trade Direction
            f.write(r'\multirow{3}{3.0cm}{\textbf{Trade Direction}}' + '\n')
            f.write(r' & Buy (\%) & ' + f'{results["All_buy_pct"]:.1f}' + ' & ' + f'{results["1_10_buy_pct"]:.1f}' + ' & ' + f'{results["11_200_buy_pct"]:.1f}' + ' & ' + f'{results["over_200_buy_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Sell (\%) & ' + f'{results["All_sell_pct"]:.1f}' + ' & ' + f'{results["1_10_sell_pct"]:.1f}' + ' & ' + f'{results["11_200_sell_pct"]:.1f}' + ' & ' + f'{results["over_200_sell_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Midpoint (\%) & ' + f'{results["All_midpoint_pct"]:.1f}' + ' & ' + f'{results["1_10_midpoint_pct"]:.1f}' + ' & ' + f'{results["11_200_midpoint_pct"]:.1f}' + ' & ' + f'{results["over_200_midpoint_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Contract Type & Trade Direction
            f.write(r'\multirow{6}{3.0cm}{\textbf{Contract Type \& Trade Direction}}' + '\n')
            f.write(r'& Call Buy (\%) & ' + f'{results["All_call_buy_pct"]:.1f}' + ' & ' + f'{results["1_10_call_buy_pct"]:.1f}' + ' & ' + f'{results["11_200_call_buy_pct"]:.1f}' + ' & ' + f'{results["over_200_call_buy_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Call Sell (\%) & ' + f'{results["All_call_sell_pct"]:.1f}' + ' & ' + f'{results["1_10_call_sell_pct"]:.1f}' + ' & ' + f'{results["11_200_call_sell_pct"]:.1f}' + ' & ' + f'{results["over_200_call_sell_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Call Midpoint (\%) & ' + f'{results["All_call_midpoint_pct"]:.1f}' + ' & ' + f'{results["1_10_call_midpoint_pct"]:.1f}' + ' & ' + f'{results["11_200_call_midpoint_pct"]:.1f}' + ' & ' + f'{results["over_200_call_midpoint_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Put Buy (\%) & ' + f'{results["All_put_buy_pct"]:.1f}' + ' & ' + f'{results["1_10_put_buy_pct"]:.1f}' + ' & ' + f'{results["11_200_put_buy_pct"]:.1f}' + ' & ' + f'{results["over_200_put_buy_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Put Sell (\%) & ' + f'{results["All_put_sell_pct"]:.1f}' + ' & ' + f'{results["1_10_put_sell_pct"]:.1f}' + ' & ' + f'{results["11_200_put_sell_pct"]:.1f}' + ' & ' + f'{results["over_200_put_sell_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Put Midpoint (\%) & ' + f'{results["All_put_midpoint_pct"]:.1f}' + ' & ' + f'{results["1_10_put_midpoint_pct"]:.1f}' + ' & ' + f'{results["11_200_put_midpoint_pct"]:.1f}' + ' & ' + f'{results["over_200_put_midpoint_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Trade Type
            f.write(r'\multirow{2}{3.0cm}{\textbf{Trade Type}}' + '\n')
            f.write(r' & Simple (\%) & ' + f'{results["All_simple_pct"]:.1f}' + ' & ' + f'{results["1_10_simple_pct"]:.1f}' + ' & ' + f'{results["11_200_simple_pct"]:.1f}' + ' & ' + f'{results["over_200_simple_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Complex (\%) & ' + f'{results["All_complex_pct"]:.1f}' + ' & ' + f'{results["1_10_complex_pct"]:.1f}' + ' & ' + f'{results["11_200_complex_pct"]:.1f}' + ' & ' + f'{results["over_200_complex_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Trade Size
            f.write(r'\multirow{3}{3.0cm}{\textbf{Trade Size}}' + '\n')
            f.write(r' & Notional Value (\$) & ' + f'{results["All_notional_median"]:,.1f}' + ' & ' + f'{results["1_10_notional_median"]:,.1f}' + ' & ' + f'{results["11_200_notional_median"]:,.1f}' + ' & ' + f'{results["over_200_notional_median"]:,.1f}' + r' \\' + '\n')
            f.write(r' & Trade Size (contracts) & ' + f'{results["All_size_contracts_median"]:,.1f}' + ' & ' + f'{results["1_10_size_contracts_median"]:,.1f}' + ' & ' + f'{results["11_200_size_contracts_median"]:,.1f}' + ' & ' + f'{results["over_200_size_contracts_median"]:,.1f}' + r' \\' + '\n')
            f.write(r' & Trade Size (\$) & ' + f'{results["All_size_dollar_median"]:,.1f}' + ' & ' + f'{results["1_10_size_dollar_median"]:,.1f}' + ' & ' + f'{results["11_200_size_dollar_median"]:,.1f}' + ' & ' + f'{results["over_200_size_dollar_median"]:,.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
        
            # Option Characteristics
            f.write(r'\multirow{3}{3.0cm}{\textbf{Option Characteristics}}' + '\n')
            f.write(r' & Option Price (\$) & ' + f'{results["All_price_median"]:.2f}' + ' & ' + f'{results["1_10_price_median"]:.2f}' + ' & ' + f'{results["11_200_price_median"]:.2f}' + ' & ' + f'{results["over_200_price_median"]:.2f}' + r' \\' + '\n')
            f.write(r' & Option Moneyness & ' + f'{results["All_moneyness_median"]:.3f}' + ' & ' + f'{results["1_10_moneyness_median"]:.3f}' + ' & ' + f'{results["11_200_moneyness_median"]:.3f}' + ' & ' + f'{results["over_200_moneyness_median"]:.3f}' + r' \\' + '\n')
            f.write(r' & Option Leverage & ' + f'{results["All_leverage_median"]:.2f}' + ' & ' + f'{results["1_10_leverage_median"]:.2f}' + ' & ' + f'{results["11_200_leverage_median"]:.2f}' + ' & ' + f'{results["over_200_leverage_median"]:.2f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Market Liquidity
            f.write(r'\multirow{2}{3.0cm}{\textbf{Market Liquidity}}' + '\n')
            f.write(r' & Quoted Spread (\%) & ' + f'{results["All_quoted_spread_median"]:.2f}' + ' & ' + f'{results["1_10_quoted_spread_median"]:.2f}' + ' & ' + f'{results["11_200_quoted_spread_median"]:.2f}' + ' & ' + f'{results["over_200_quoted_spread_median"]:.2f}' + r' \\' + '\n')
            f.write(r' & Relative Spread (\%) & ' + f'{results["All_relative_spread_median"]:.2f}' + ' & ' + f'{results["1_10_relative_spread_median"]:.2f}' + ' & ' + f'{results["11_200_relative_spread_median"]:.2f}' + ' & ' + f'{results["over_200_relative_spread_median"]:.2f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Moment of the Day
            f.write(r'\multirow{4}{3.0cm}{\textbf{Moment of the Day}}' + '\n')
            f.write(r' & 9:30 to 11 & ' + f'{results["All_morning_pct"]:.1f}' + ' & ' + f'{results["1_10_morning_pct"]:.1f}' + ' & ' + f'{results["11_200_morning_pct"]:.1f}' + ' & ' + f'{results["over_200_morning_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & 11 to 13 & ' + f'{results["All_midday_pct"]:.1f}' + ' & ' + f'{results["1_10_midday_pct"]:.1f}' + ' & ' + f'{results["11_200_midday_pct"]:.1f}' + ' & ' + f'{results["over_200_midday_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & 13 to 16 & ' + f'{results["All_afternoon_pct"]:.1f}' + ' & ' + f'{results["1_10_afternoon_pct"]:.1f}' + ' & ' + f'{results["11_200_afternoon_pct"]:.1f}' + ' & ' + f'{results["over_200_afternoon_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Overnight & ' + f'{results["All_overnight_pct"]:.1f}' + ' & ' + f'{results["1_10_overnight_pct"]:.1f}' + ' & ' + f'{results["11_200_overnight_pct"]:.1f}' + ' & ' + f'{results["over_200_overnight_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Moneyness
            f.write(r'\multirow{3}{3.0cm}{\textbf{Moneyness}}' + '\n')
            f.write(r' & OTM & ' + f'{results["All_otm_pct"]:.1f}' + ' & ' + f'{results["1_10_otm_pct"]:.1f}' + ' & ' + f'{results["11_200_otm_pct"]:.1f}' + ' & ' + f'{results["over_200_otm_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & ITM & ' + f'{results["All_itm_pct"]:.1f}' + ' & ' + f'{results["1_10_itm_pct"]:.1f}' + ' & ' + f'{results["11_200_itm_pct"]:.1f}' + ' & ' + f'{results["over_200_itm_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & ATM & ' + f'{results["All_atm_pct"]:.1f}' + ' & ' + f'{results["1_10_atm_pct"]:.1f}' + ' & ' + f'{results["11_200_atm_pct"]:.1f}' + ' & ' + f'{results["over_200_atm_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Bid-Ask Proximity
            f.write(r'\multirow{3}{3.0cm}{\textbf{Bid-Ask Proximity}}' + '\n')
            f.write(r' & Closer to Bid & ' + f'{results["All_closer_to_bid_pct"]:.1f}' + ' & ' + f'{results["1_10_closer_to_bid_pct"]:.1f}' + ' & ' + f'{results["11_200_closer_to_bid_pct"]:.1f}' + ' & ' + f'{results["over_200_closer_to_bid_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Same Distance & ' + f'{results["All_same_distance_pct"]:.1f}' + ' & ' + f'{results["1_10_same_distance_pct"]:.1f}' + ' & ' + f'{results["11_200_same_distance_pct"]:.1f}' + ' & ' + f'{results["over_200_same_distance_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Closer to Ask & ' + f'{results["All_closer_to_ask_pct"]:.1f}' + ' & ' + f'{results["1_10_closer_to_ask_pct"]:.1f}' + ' & ' + f'{results["11_200_closer_to_ask_pct"]:.1f}' + ' & ' + f'{results["over_200_closer_to_ask_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\midrule' + '\n')
            
            # Time to Expiration
            f.write(r'\multirow{6}{3.0cm}{\textbf{Time to Expiration}}' + '\n')
            f.write(r' & Less than a week & ' + f'{results["All_less_than_a_week_pct"]:.1f}' + ' & ' + f'{results["1_10_less_than_a_week_pct"]:.1f}' + ' & ' + f'{results["11_200_less_than_a_week_pct"]:.1f}' + ' & ' + f'{results["over_200_less_than_a_week_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & 1-2 weeks & ' + f'{results["All_1_2_weeks_pct"]:.1f}' + ' & ' + f'{results["1_10_1_2_weeks_pct"]:.1f}' + ' & ' + f'{results["11_200_1_2_weeks_pct"]:.1f}' + ' & ' + f'{results["over_200_1_2_weeks_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & 2-4 weeks & ' + f'{results["All_2_4_weeks_pct"]:.1f}' + ' & ' + f'{results["1_10_2_4_weeks_pct"]:.1f}' + ' & ' + f'{results["11_200_2_4_weeks_pct"]:.1f}' + ' & ' + f'{results["over_200_2_4_weeks_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & 1-3 months & ' + f'{results["All_1_3_months_pct"]:.1f}' + ' & ' + f'{results["1_10_1_3_months_pct"]:.1f}' + ' & ' + f'{results["11_200_1_3_months_pct"]:.1f}' + ' & ' + f'{results["over_200_1_3_months_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & 3-12 months & ' + f'{results["All_3_12_months_pct"]:.1f}' + ' & ' + f'{results["1_10_3_12_months_pct"]:.1f}' + ' & ' + f'{results["11_200_3_12_months_pct"]:.1f}' + ' & ' + f'{results["over_200_3_12_months_pct"]:.1f}' + r' \\' + '\n')
            f.write(r' & Over a year & ' + f'{results["All_over_a_year_pct"]:.1f}' + ' & ' + f'{results["1_10_over_a_year_pct"]:.1f}' + ' & ' + f'{results["11_200_over_a_year_pct"]:.1f}' + ' & ' + f'{results["over_200_over_a_year_pct"]:.1f}' + r' \\' + '\n')
            f.write(r'\bottomrule' + '\n')
            f.write(r'\end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        
        logger.info(f"Table2 generation completed successfully. Output saved to {OUTPUT_PATH}")
