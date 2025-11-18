# uv run src/tabling_duckdb/Table2_granular.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import duckdb
import pandas as pd
from src.config import initialize_main
from src.config.config_settings import PROCESSED_PATH, tables

OUTPUT_DIR = PROJECT_ROOT / tables["duckdb_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_2_granular.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting Table2_granular.py script with DuckDB.")

    try:
        # Connect to DuckDB
        con = duckdb.connect()
        logger.info("DuckDB connection established.")
        
        # Get the parquet path pattern
        parquet_path = str(PROCESSED_PATH / "**/*.parquet")
        logger.info(f"Reading parquet files from: {parquet_path}")
        
        # Build SQL query for aggregations across granular size categories
        logger.info("Building DuckDB query for summary statistics...")
        
        query = f"""
        WITH base_data AS (
            SELECT 
                prtSize_agg,
                okey_cp,
                trade_type,
                moment_of_the_day,
                moneyness_class_ratio,
                bid_ask_proximity,
                time_to_expiry,
                notional_value,
                trade_size_dollar,
                prtPrice,
                moneyness,
                leverage,
                quoted_spread,
                relative_spread,
                CASE 
                    WHEN prtSize_agg = 1 THEN '1'
                    WHEN prtSize_agg BETWEEN 2 AND 10 THEN '2_10'
                    WHEN prtSize_agg BETWEEN 11 AND 100 THEN '11_100'
                    WHEN prtSize_agg BETWEEN 101 AND 1000 THEN '101_1000'
                    WHEN prtSize_agg BETWEEN 1001 AND 10000 THEN '1001_10000'
                    WHEN prtSize_agg BETWEEN 10001 AND 100000 THEN '10001_100000'
                    WHEN prtSize_agg > 100000 THEN 'over_100000'
                END AS size_category
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE ticker_class = 'equity'
                AND prtType >= 73
                AND prtSize_agg IS NOT NULL
        )
        SELECT 
            COALESCE(size_category, 'all') as size_category,
            
            -- Categorical percentages (okey_cp)
            AVG(CASE WHEN okey_cp = 'Call' THEN 100.0 ELSE 0.0 END) as call,
            AVG(CASE WHEN okey_cp = 'Put' THEN 100.0 ELSE 0.0 END) as put,
            
            -- Trade type
            AVG(CASE WHEN trade_type = 'simple' THEN 100.0 ELSE 0.0 END) as simple,
            AVG(CASE WHEN trade_type = 'complex' THEN 100.0 ELSE 0.0 END) as complex,
            
            -- Moment of the day
            AVG(CASE WHEN moment_of_the_day = 'morning' THEN 100.0 ELSE 0.0 END) as morning,
            AVG(CASE WHEN moment_of_the_day = 'midday' THEN 100.0 ELSE 0.0 END) as midday,
            AVG(CASE WHEN moment_of_the_day = 'afternoon' THEN 100.0 ELSE 0.0 END) as afternoon,
            AVG(CASE WHEN moment_of_the_day = 'overnight' THEN 100.0 ELSE 0.0 END) as overnight,
            
            -- Moneyness
            AVG(CASE WHEN moneyness_class_ratio = 'OTM' THEN 100.0 ELSE 0.0 END) as otm,
            AVG(CASE WHEN moneyness_class_ratio = 'ITM' THEN 100.0 ELSE 0.0 END) as itm,
            AVG(CASE WHEN moneyness_class_ratio = 'ATM' THEN 100.0 ELSE 0.0 END) as atm,
            
            -- Bid-ask proximity
            AVG(CASE WHEN bid_ask_proximity = 'closer_to_bid' THEN 100.0 ELSE 0.0 END) as closer_to_bid,
            AVG(CASE WHEN bid_ask_proximity = 'same_distance' THEN 100.0 ELSE 0.0 END) as same_distance,
            AVG(CASE WHEN bid_ask_proximity = 'closer_to_ask' THEN 100.0 ELSE 0.0 END) as closer_to_ask,
            
            -- Time to expiry
            AVG(CASE WHEN time_to_expiry = 'lt_1w' THEN 100.0 ELSE 0.0 END) as lt_1w,
            AVG(CASE WHEN time_to_expiry = '1w_to_2w' THEN 100.0 ELSE 0.0 END) as "1w_to_2w",
            AVG(CASE WHEN time_to_expiry = '2w_to_4w' THEN 100.0 ELSE 0.0 END) as "2w_to_4w",
            AVG(CASE WHEN time_to_expiry = '1m_to_3m' THEN 100.0 ELSE 0.0 END) as "1m_to_3m",
            AVG(CASE WHEN time_to_expiry = '3m_to_12m' THEN 100.0 ELSE 0.0 END) as "3m_to_12m",
            AVG(CASE WHEN time_to_expiry = 'gt_1y' THEN 100.0 ELSE 0.0 END) as gt_1y,
            
            -- Numerical medians
            MEDIAN(notional_value) as notional_value,
            MEDIAN(prtSize_agg) as prtsize_agg,
            MEDIAN(trade_size_dollar) as trade_size_dollar,
            MEDIAN(prtPrice) as prtprice,
            MEDIAN(moneyness) as moneyness,
            MEDIAN(leverage) as leverage,
            MEDIAN(quoted_spread) * 100 as quoted_spread,
            MEDIAN(relative_spread) * 100 as relative_spread
            
        FROM base_data
        GROUP BY ROLLUP(size_category)
        ORDER BY 
            CASE size_category
                WHEN 'all' THEN 1
                WHEN '1' THEN 2
                WHEN '2_10' THEN 3
                WHEN '11_100' THEN 4
                WHEN '101_1000' THEN 5
                WHEN '1001_10000' THEN 6
                WHEN '10001_100000' THEN 7
                WHEN 'over_100000' THEN 8
            END
        """
        
        logger.info("Executing DuckDB query...")
        result_df = con.execute(query).fetchdf()
        
        # Convert to dictionary for easy access
        unpack: dict = {}
        for _, row in result_df.iterrows():
            cat = row['size_category']
            unpack[cat] = row.to_dict()
        
        logger.info("Computation finished. Organizing results...")
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")
        
        # Helper function to format numbers
        def format_number(value, is_percentage=False, decimal_places=2):
            """Format numbers for LaTeX table display"""
            if value is None or pd.isna(value):
                return "--"
            if is_percentage:
                return f"{value:.{decimal_places}f}"
            else:
                # For large numbers, use thousands separators
                if value >= 1000:
                    return f"{value:,.{decimal_places}f}"
                else:
                    return f"{value:.{decimal_places}f}"
        
        # Helper function to safely get values from unpack
        def safe_get(category, key):
            """Safely retrieve value from unpack dict, return None if category or key doesn't exist"""
            if category not in unpack:
                return None
            return unpack[category].get(key, None)
        
        # Define the order of categories for the table
        categories = ['all', '1', '2_10', '11_100', '101_1000', '1001_10000', '10001_100000', 'over_100000']
        
        logger.info("Writing LaTeX table...")      
        
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
    & Call (\%) & """ + ' & '.join([format_number(safe_get(cat, 'call'), True, 1) for cat in categories]) + r""" \\
    & Put (\%) & """ + ' & '.join([format_number(safe_get(cat, 'put'), True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{2}{2.5cm}{\textbf{Trade Type}} 
    & Simple (\%) & """ + ' & '.join([format_number(safe_get(cat, 'simple'), True, 1) for cat in categories]) + r""" \\
    & Complex (\%) & """ + ' & '.join([format_number(safe_get(cat, 'complex'), True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Trade Size}} 
    & Notional Value (\$) & """ + ' & '.join([format_number(safe_get(cat, 'notional_value'), False, 0) for cat in categories]) + r""" \\
    & Trade Size (contracts) & """ + ' & '.join([format_number(safe_get(cat, 'prtsize_agg'), False, 0) for cat in categories]) + r""" \\
    & Trade Size (\$) & """ + ' & '.join([format_number(safe_get(cat, 'trade_size_dollar'), False, 0) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Option Characteristics}} 
    & Option Price (\$) & """ + ' & '.join([format_number(safe_get(cat, 'prtprice'), False, 2) for cat in categories]) + r""" \\
    & Option Moneyness & """ + ' & '.join([format_number(safe_get(cat, 'moneyness'), False, 2) for cat in categories]) + r""" \\
    & Option Leverage & """ + ' & '.join([format_number(safe_get(cat, 'leverage'), False, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{2}{2.5cm}{\textbf{Market Liquidity}} 
    & Quoted Spread (\%) & """ + ' & '.join([format_number(safe_get(cat, 'quoted_spread'), True, 2) for cat in categories]) + r""" \\
    & Relative Spread (\%) & """ + ' & '.join([format_number(safe_get(cat, 'relative_spread'), True, 2) for cat in categories]) + r""" \\
    \midrule

    \multirow{4}{2.5cm}{\textbf{Moment of the Day}} 
    & Morning (\%) & """ + ' & '.join([format_number(safe_get(cat, 'morning'), True, 1) for cat in categories]) + r""" \\
    & Midday (\%) & """ + ' & '.join([format_number(safe_get(cat, 'midday'), True, 1) for cat in categories]) + r""" \\
    & Afternoon (\%) & """ + ' & '.join([format_number(safe_get(cat, 'afternoon'), True, 1) for cat in categories]) + r""" \\
    & Overnight (\%) & """ + ' & '.join([format_number(safe_get(cat, 'overnight'), True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Moneyness}} 
    & OTM (\%) & """ + ' & '.join([format_number(safe_get(cat, 'otm'), True, 1) for cat in categories]) + r""" \\
    & ITM (\%) & """ + ' & '.join([format_number(safe_get(cat, 'itm'), True, 1) for cat in categories]) + r""" \\
    & ATM (\%) & """ + ' & '.join([format_number(safe_get(cat, 'atm'), True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{2.5cm}{\textbf{Bid-Ask Proximity}} 
    & Closer to Bid (\%) & """ + ' & '.join([format_number(safe_get(cat, 'closer_to_bid'), True, 1) for cat in categories]) + r""" \\
    & Same Distance (\%) & """ + ' & '.join([format_number(safe_get(cat, 'same_distance'), True, 1) for cat in categories]) + r""" \\
    & Closer to Ask (\%) & """ + ' & '.join([format_number(safe_get(cat, 'closer_to_ask'), True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{6}{2.5cm}{\textbf{Time to Expiration}} 
    & Less than a week (\%) & """ + ' & '.join([format_number(safe_get(cat, 'lt_1w'), True, 1) for cat in categories]) + r""" \\
    & 1-2 weeks (\%) & """ + ' & '.join([format_number(safe_get(cat, '1w_to_2w'), True, 1) for cat in categories]) + r""" \\
    & 2-4 weeks (\%) & """ + ' & '.join([format_number(safe_get(cat, '2w_to_4w'), True, 1) for cat in categories]) + r""" \\
    & 1-3 months (\%) & """ + ' & '.join([format_number(safe_get(cat, '1m_to_3m'), True, 1) for cat in categories]) + r""" \\
    & 3-12 months (\%) & """ + ' & '.join([format_number(safe_get(cat, '3m_to_12m'), True, 1) for cat in categories]) + r""" \\
    & Over a year (\%) & """ + ' & '.join([format_number(safe_get(cat, 'gt_1y'), True, 1) for cat in categories]) + r""" \\
    \bottomrule
    \end{tabular}
 \end{table}
"""
        
        # Write the LaTeX table to file
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
        logger.info("Table2_granular.py script completed successfully.")
