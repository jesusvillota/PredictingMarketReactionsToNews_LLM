# uv run src/tabling_dask/Table2_granular.py
#---------------------------------------------------------------
from pathlib import Path
import sys
import shutil
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
import numpy as np
from src.config import config_settings, get_logger, DaskManager
from typing import Any, Iterable
from src.config.config_settings import PROCESSED_PATH, tables
from src.tabling_dask.common import build_parquet_filters

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _escape_latex_template(template: str, keys: Iterable[str]) -> str:
    """Escape curly braces in a LaTeX template while preserving placeholders."""
    escaped = template.replace('{', '{{').replace('}', '}}')
    for key in keys:
        escaped = escaped.replace(f'{{{{{key}}}}}', f'{{{key}}}')
    return escaped


# # Define whale threshold (same as in Figure1.py)
# WHALE_THRESHOLD = 270


def build_table(parquet_filters: list, output_dir, ddf=None):
    logger = get_logger(__name__)
    logger.info("Starting Table2_granular.build_table() with Dask.")
    required_columns = [
        'prtSize_agg', 'okey_cp', 'trade_type', 
        'prtPrice', 'moneyness', 'leverage', 'quoted_spread', 'relative_spread',
        'moment_of_the_day', 'moneyness_class_ratio', 'bid_ask_proximity', 
        'time_to_expiry', 'trade_size_dollar', 'notional_value'
    ]
    
    if ddf is None:
        with DaskManager() as dask_manager:
            try:
                logger.info("Loading parquet data with Dask...")
                ddf = dd.read_parquet(
                    path=PROCESSED_PATH,
                    engine=config_settings.parquet["engine"],
                    filters=parquet_filters,
                    columns=required_columns,
                    split_row_groups='infer',
                )
                logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
            except Exception as e:
                logger.exception(f"Error loading parquet: {e}")
                raise
    else:
        # Select only required columns from pre-loaded dataframe
        ddf = ddf[required_columns]
        logger.info(f"Using pre-loaded Dask DataFrame with {ddf.npartitions} partitions.")
    
    with DaskManager() as dask_manager:

    
        logger.info("Preparing lazy computations for summary statistics...")
        
        cat_var_map: dict = {
            'okey_cp': ['Call', 'Put'],
            # 'buy_sell_class': ['Buy', 'Sell', 'Midpoint'],
            'trade_type': ['simple', 'complex'],
            'moment_of_the_day': ['morning', 'midday', 'afternoon', 'overnight'],
            'moneyness_class_ratio': ['OTM', 'ITM', 'ATM'],
            'bid_ask_proximity': ['closer_to_bid', 'same_distance', 'closer_to_ask'],
            'time_to_expiry': ['lt_1w', '1w_to_2w', '2w_to_4w', '1m_to_3m', '3m_to_12m', 'gt_1y']
        }
        
        num_var: dict = {
            'pct_var': [
                'quoted_spread', 
                'relative_spread'
            ],
            'non_pct_var': [
                'notional_value', 
                'prtSize_agg', 
                'trade_size_dollar', 
                'prtPrice', 
                'moneyness', 
                'leverage', 
            ]
        }
        
        # Helper function to create lazy computations for each category
        def get_lazy_stats(filtered_ddf: dd.DataFrame) -> dict:
            """Add lazy computations for a specific filtered dataframe"""
            
            lazy_computations: dict = {}
            
            # Categorical percentages
            for key, values in cat_var_map.items():
                for value in values:
                    stat_name = f"{value.lower().replace(' ', '_')}"
                    # filtered_ddf = filtered_ddf[key].dropna()
                    lazy_computations[stat_name] = (filtered_ddf[key] == value).mean() * 100
                    
            # Contract Type & Trade Direction combinations
            # for ctype in ['Call', 'Put']:
            #     for direction in ['Buy', 'Sell', 'Midpoint']:
            #         stat_name = f"{ctype.lower()}_{direction.lower()}"
            #         lazy_computations[stat_name] = ((filtered_ddf['okey_cp'] == ctype) & (filtered_ddf['buy_sell_class'] == direction)).mean() * 100
        
            # Numerical medians
            for key, values in num_var.items():
                for value in values:
                    stat_name = f"{value.lower()}"
                    multiplier = 100 if key == 'pct_var' else 1
                    # Filter out nulls/NaNs and infinite values before median calculation
                    filtered_data = filtered_ddf[value].dropna()
                    # Additional check to filter out infinite values
                    filtered_data = filtered_data[(filtered_data != np.inf) & (filtered_data != -np.inf)]
                    lazy_computations[stat_name] = filtered_data.median_approximate() * multiplier

            return lazy_computations

        logger.info("Setting up lazy computations for all categories...")
        
        ddf_nonna = ddf.dropna(subset=['prtSize_agg'])
        
        # Create filtered dataframes with granular bins
        filtered_dfs: dict[str, dd.DataFrame] = {
            'all': ddf_nonna,
            '1': ddf_nonna[ddf_nonna['prtSize_agg'] == 1],
            '2_10': ddf_nonna[ddf_nonna['prtSize_agg'].between(2, 10, inclusive='both')],
            '11_100': ddf_nonna[ddf_nonna['prtSize_agg'].between(11, 100, inclusive='both')],
            '101_1000': ddf_nonna[ddf_nonna['prtSize_agg'].between(101, 1000, inclusive='both')],
            '1001_10000': ddf_nonna[ddf_nonna['prtSize_agg'].between(1001, 10000, inclusive='both')],
            '10001_100000': ddf_nonna[ddf_nonna['prtSize_agg'].between(10001, 100000, inclusive='both')],
            'over_100000': ddf_nonna[ddf_nonna['prtSize_agg'] > 100000]
        }
        
        logger.info("Computing all statistics with Dask...")

        lazy_dict: dict[str, Any] = {}
        for cat, filtered_df in filtered_dfs.items():
            logger.info(f"Preparing lazy computations for category: {cat}")
            lazy_dict[cat] = get_lazy_stats(filtered_df)
        
        ALL_LAZY_COMPUTATIONS: list = []
        for computations in lazy_dict.values():
            ALL_LAZY_COMPUTATIONS.extend(computations.values())

        results_list = dd.compute(*ALL_LAZY_COMPUTATIONS)
        
        # Nested structure for easier access
        unpack: dict = {}
        result_idx = 0
        for cat, computations in lazy_dict.items():
            unpack[cat] = {}
            for stat_name, _ in computations.items():
                unpack[cat][stat_name] = results_list[result_idx]
                result_idx += 1        
        
        logger.info("Computation finished. Organizing results...")
        
        # Helper function to format numbers
        def format_number(value, is_percentage=False, decimal_places=2):
            """Format numbers for LaTeX table display"""
            if value is None or np.isnan(value):
                return "--"
            if is_percentage:
                return f"{value:.{decimal_places}f}"
            else:
                # For large numbers, use thousands separators
                if value >= 1000:
                    return f"{value:,.{decimal_places}f}"
                else:
                    return f"{value:.{decimal_places}f}"
        
        # Define the order of categories for the table
        categories = ['all', '1', '2_10', '11_100', '101_1000', '1001_10000', '10001_100000', 'over_100000']

        # Compute counts per category for Total row
        logger.info("Computing total counts per granular size bin for Table2_granular...")
        total_counts = {cat: int(filtered_dfs[cat].shape[0].compute()) for cat in categories}
        
        logger.info("Writing LaTeX table...")      
        
        # Resolve output path
        out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "Table2_granular.tex"
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)

        # Build the LaTeX table content
        latex_content = r"""\begin{table}[htbp]
    \centering
    \caption{Summary Statistics by Trade Size (Granular)}
    \subcaption*{
    {\scriptsize
    Summary statistics for equity option trades from 2014–2025 at millisecond resolution, covering regular and overnight trading sessions. Trades are grouped by trade sizes (number of contracts traded). The columns represent: All (all trades), 1 (trades with size = 1), 2--10 (trades with $2 \leq$ size $\leq$ 10), 11--100 (trades with $11 \leq$ size $\leq$ 100), 101--1000 (trades with $101 \leq$ size $\leq$ 1000), 1001--10000 (trades with $1001 \leq$ size $\leq$ 10000), 10001--100000 (trades with $10001 \leq$ size $\leq$ 100000), and >100000 (trades with size $>$ 100000). Data filtered to include only equity options (ticker\_class == ``Equity'') with prtType $\geq$ 73. For categorical variables, we compute the overall share (percentage) across all observations in each group for the entire sample period. For numerical variables, we compute the overall median across all observations in each group for the entire sample period.
    \par}
    \vspace{1em}
    }
    \label{tab:summary_stats_by_size_granular}
    \tiny
    \begin{tabular}{>{\raggedright\arraybackslash}p{2.5cm}lcccccccc}
    \toprule
        \textbf{Classification} 
        & \textbf{Category} 
        & \textbf{All} 
        & \textbf{1} 
        & \textbf{2--10} 
        & \textbf{11--100} 
        & \textbf{101--1K} 
        & \textbf{1K--10K} 
        & \textbf{10K--100K} 
        & \textbf{>100K} \\
    \midrule

    \multirow{2}{2.5cm}{\textbf{Contract Type}} 
    & Call (\%) & """ + ' & '.join([format_number(unpack[cat]['call'], True, 1) for cat in categories]) + r""" \\
    & Put (\%) & """ + ' & '.join([format_number(unpack[cat]['put'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{2}{2.5cm}{\textbf{Trade Type}} 
    & Simple (\%) & """ + ' & '.join([format_number(unpack[cat]['simple'], True, 1) for cat in categories]) + r""" \\
    & Complex (\%) & """ + ' & '.join([format_number(unpack[cat]['complex'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Trade Size}} 
    & Notional Value (\$) & """ + ' & '.join([format_number(unpack[cat]['notional_value'], False, 0) for cat in categories]) + r""" \\
    & Trade Size (contracts) & """ + ' & '.join([format_number(unpack[cat]['prtsize_agg'], False, 0) for cat in categories]) + r""" \\
    & Trade Size (\$) & """ + ' & '.join([format_number(unpack[cat]['trade_size_dollar'], False, 0) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Option Characteristics}} 
    & Option Price (\$) & """ + ' & '.join([format_number(unpack[cat]['prtprice'], False, 2) for cat in categories]) + r""" \\
    & Option Moneyness & """ + ' & '.join([format_number(unpack[cat]['moneyness'], False, 2) for cat in categories]) + r""" \\
    & Option Leverage & """ + ' & '.join([format_number(unpack[cat]['leverage'], False, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{2}{2.5cm}{\textbf{Market Liquidity}} 
    & Quoted Spread (\%) & """ + ' & '.join([format_number(unpack[cat]['quoted_spread'], True, 2) for cat in categories]) + r""" \\
    & Relative Spread (\%) & """ + ' & '.join([format_number(unpack[cat]['relative_spread'], True, 2) for cat in categories]) + r""" \\
    \midrule

    \multirow{4}{2.5cm}{\textbf{Moment of the Day}} 
    & Morning (\%) & """ + ' & '.join([format_number(unpack[cat]['morning'], True, 1) for cat in categories]) + r""" \\
    & Midday (\%) & """ + ' & '.join([format_number(unpack[cat]['midday'], True, 1) for cat in categories]) + r""" \\
    & Afternoon (\%) & """ + ' & '.join([format_number(unpack[cat]['afternoon'], True, 1) for cat in categories]) + r""" \\
    & Overnight (\%) & """ + ' & '.join([format_number(unpack[cat]['overnight'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Moneyness}} 
    & OTM (\%) & """ + ' & '.join([format_number(unpack[cat]['otm'], True, 1) for cat in categories]) + r""" \\
    & ITM (\%) & """ + ' & '.join([format_number(unpack[cat]['itm'], True, 1) for cat in categories]) + r""" \\
    & ATM (\%) & """ + ' & '.join([format_number(unpack[cat]['atm'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Bid-Ask Proximity}} 
    & Closer to Bid (\%) & """ + ' & '.join([format_number(unpack[cat]['closer_to_bid'], True, 1) for cat in categories]) + r""" \\
    & Same Distance (\%) & """ + ' & '.join([format_number(unpack[cat]['same_distance'], True, 1) for cat in categories]) + r""" \\
    & Closer to Ask (\%) & """ + ' & '.join([format_number(unpack[cat]['closer_to_ask'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{6}{2.5cm}{\textbf{Time to Expiration}} 
    & Less than a week (\%) & """ + ' & '.join([format_number(unpack[cat]['lt_1w'], True, 1) for cat in categories]) + r""" \\
    & 1-2 weeks (\%) & """ + ' & '.join([format_number(unpack[cat]['1w_to_2w'], True, 1) for cat in categories]) + r""" \\
    & 2-4 weeks (\%) & """ + ' & '.join([format_number(unpack[cat]['2w_to_4w'], True, 1) for cat in categories]) + r""" \\
    & 1-3 months (\%) & """ + ' & '.join([format_number(unpack[cat]['1m_to_3m'], True, 1) for cat in categories]) + r""" \\
    & 3-12 months (\%) & """ + ' & '.join([format_number(unpack[cat]['3m_to_12m'], True, 1) for cat in categories]) + r""" \\
    & >1 year (\%) & """ + ' & '.join([format_number(unpack[cat]['gt_1y'], True, 1) for cat in categories]) + r""" \\
    \midrule
    \textbf{Total} & & {c_all} & {c_1} & {c_2_10} & {c_11_100} & {c_101_1000} & {c_1001_10000} & {c_10001_100000} & {c_over_100000} \\
    \bottomrule
    \end{tabular}
 \end{table}
"""
        
        placeholders = [
            "c_all",
            "c_1",
            "c_2_10",
            "c_11_100",
            "c_101_1000",
            "c_1001_10000",
            "c_10001_100000",
            "c_over_100000",
        ]
        safe_template = _escape_latex_template(latex_content, placeholders)
        latex_filled = safe_template.format(
            c_all=f"{total_counts['all']:,}",
            c_1=f"{total_counts['1']:,}",
            c_2_10=f"{total_counts['2_10']:,}",
            c_11_100=f"{total_counts['11_100']:,}",
            c_101_1000=f"{total_counts['101_1000']:,}",
            c_1001_10000=f"{total_counts['1001_10000']:,}",
            c_10001_100000=f"{total_counts['10001_100000']:,}",
            c_over_100000=f"{total_counts['over_100000']:,}",
        )

        # Write the LaTeX table to file
        with open(output_path, 'w') as f:
            f.write(latex_filled)
        
        logger.info(f"LaTeX table successfully written to: {output_path}")
        logger.info("Table2_granular.build_table() completed successfully.")


if __name__ == '__main__':
    parquet_filters = build_parquet_filters(
        ticker_type="all", 
        strategy_type="all",
        start=None,
        end=None,
    )
    build_table(parquet_filters, OUTPUT_DIR)