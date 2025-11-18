# uv run src/tabling_duckdb/Table2_broad.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import duckdb
import pandas as pd
from src.config import initialize_main
from src.config.config_settings import PROCESSED_PATH, tables, TEMP_DIR

OUTPUT_DIR = PROJECT_ROOT / tables["duckdb_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_2_broad.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting Table2_broad.py script with DuckDB.")

    try:
        # Connect to DuckDB
        con = duckdb.connect()
        
        # Configure DuckDB to use disk for temporary storage and set memory limit
        con.execute("SET memory_limit='64GB'")  # Set lower than your RAM to avoid OOM
        con.execute(f"SET temp_directory='{TEMP_DIR}'")  # Use temp directory for spilling
        con.execute("SET preserve_insertion_order=false")  # May help performance

        logger.info(f"DuckDB connection established with memory_limit=64GB and temp_directory={TEMP_DIR}")

        # Get the parquet path pattern
        parquet_path = str(PROCESSED_PATH / "**/*.parquet")
        logger.info(f"Reading parquet files from: {parquet_path}")
        
        # Category name mappings (new names from classify.py)
        cat_var_map: dict = {
            'okey_cp': ['Call', 'Put'],
            'trade_type': ['simple', 'complex'],
            'moment_of_the_day': ['morning', 'midday', 'afternoon', 'overnight'],
            'moneyness_class_ratio': ['OTM', 'ITM', 'ATM'],
            'bid_ask_proximity': ['closer_to_bid', 'same_distance', 'closer_to_ask'],
            'time_to_expiry': ['lt_1w', '1w_to_2w', '2w_to_4w', '1m_to_3m', '3m_to_12m', 'gt_1y']
        }
        
        # Build SQL query for aggregations across size categories
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
                    WHEN prtSize_agg BETWEEN 1 AND 10 THEN '1_10'
                    WHEN prtSize_agg BETWEEN 11 AND 200 THEN '11_200'
                    WHEN prtSize_agg > 200 THEN 'over_200'
                END AS size_category
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE prtType >= 73
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
                WHEN '1_10' THEN 2
                WHEN '11_200' THEN 3
                WHEN 'over_200' THEN 4
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
    
        # Define the order of categories for the table
        categories = ['all', '1_10', '11_200', 'over_200']
        
        logger.info("Writing LaTeX table...")      
        
        # Build the LaTeX table content
        latex_content = r"""\begin{table}[htbp]
    \centering
    \caption{Summary Statistics by Trade Size}
    \subcaption*{
    {\scriptsize
    Summary statistics for equity option trades from 2014–2025 at millisecond resolution, covering regular and overnight trading sessions. Trades are grouped by trade sizes (number of contracts traded). The columns represent: All (all trades), 1--10 (trades with $1 \leq$ size $\leq$ 10), 11--200 (trades with $10 <$ size $\leq$ 200), and >200 (trades with size $>$ 200). Data filtered to include only equity options (ticker\_class == ``Equity'') with prtType $\geq$ 73. For categorical variables, we compute the overall share (percentage) across all observations in each group for the entire sample period. For numerical variables, we compute the overall median across all observations in each group for the entire sample period.
    \par}
    \vspace{1em}
    }
    \label{tab:summary_stats_by_size}
    \scriptsize
    \begin{tabular}{>{\raggedright\arraybackslash}p{3.0cm}lcccc}
    \toprule
        \textbf{Classification} 
        & \textbf{Category} 
        & \textbf{All} 
        & \textbf{1--10} 
        & \textbf{11--200} 
        & \textbf{>200} \\
    \midrule

    \multirow{2}{3.0cm}{\textbf{Contract Type}} 
    & Call (\%) & """ + ' & '.join([format_number(unpack[cat]['call'], True, 1) for cat in categories]) + r""" \\
    & Put (\%) & """ + ' & '.join([format_number(unpack[cat]['put'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{2}{3.0cm}{\textbf{Trade Type}} 
    & Simple (\%) & """ + ' & '.join([format_number(unpack[cat]['simple'], True, 1) for cat in categories]) + r""" \\
    & Complex (\%) & """ + ' & '.join([format_number(unpack[cat]['complex'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{3.0cm}{\textbf{Trade Size}} 
    & Notional Value (\$) & """ + ' & '.join([format_number(unpack[cat]['notional_value'], False, 0) for cat in categories]) + r""" \\
    & Trade Size (contracts) & """ + ' & '.join([format_number(unpack[cat]['prtsize_agg'], False, 0) for cat in categories]) + r""" \\
    & Trade Size (\$) & """ + ' & '.join([format_number(unpack[cat]['trade_size_dollar'], False, 0) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{3.0cm}{\textbf{Option Characteristics}} 
    & Option Price (\$) & """ + ' & '.join([format_number(unpack[cat]['prtprice'], False, 2) for cat in categories]) + r""" \\
    & Option Moneyness & """ + ' & '.join([format_number(unpack[cat]['moneyness'], False, 2) for cat in categories]) + r""" \\
    & Option Leverage & """ + ' & '.join([format_number(unpack[cat]['leverage'], False, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{2}{3.0cm}{\textbf{Market Liquidity}} 
    & Quoted Spread (\%) & """ + ' & '.join([format_number(unpack[cat]['quoted_spread'], True, 2) for cat in categories]) + r""" \\
    & Relative Spread (\%) & """ + ' & '.join([format_number(unpack[cat]['relative_spread'], True, 2) for cat in categories]) + r""" \\
    \midrule

    \multirow{4}{3.0cm}{\textbf{Moment of the Day}} 
    & Morning (\%) & """ + ' & '.join([format_number(unpack[cat]['morning'], True, 1) for cat in categories]) + r""" \\
    & Midday (\%) & """ + ' & '.join([format_number(unpack[cat]['midday'], True, 1) for cat in categories]) + r""" \\
    & Afternoon (\%) & """ + ' & '.join([format_number(unpack[cat]['afternoon'], True, 1) for cat in categories]) + r""" \\
    & Overnight (\%) & """ + ' & '.join([format_number(unpack[cat]['overnight'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{3.0cm}{\textbf{Moneyness}} 
    & OTM (\%) & """ + ' & '.join([format_number(unpack[cat]['otm'], True, 1) for cat in categories]) + r""" \\
    & ITM (\%) & """ + ' & '.join([format_number(unpack[cat]['itm'], True, 1) for cat in categories]) + r""" \\
    & ATM (\%) & """ + ' & '.join([format_number(unpack[cat]['atm'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{3}{3.0cm}{\textbf{Bid-Ask Proximity}} 
    & Closer to Bid (\%) & """ + ' & '.join([format_number(unpack[cat]['closer_to_bid'], True, 1) for cat in categories]) + r""" \\
    & Same Distance (\%) & """ + ' & '.join([format_number(unpack[cat]['same_distance'], True, 1) for cat in categories]) + r""" \\
    & Closer to Ask (\%) & """ + ' & '.join([format_number(unpack[cat]['closer_to_ask'], True, 1) for cat in categories]) + r""" \\
    \midrule

    \multirow{6}{3.0cm}{\textbf{Time to Expiration}} 
    & Less than a week (\%) & """ + ' & '.join([format_number(unpack[cat]['lt_1w'], True, 1) for cat in categories]) + r""" \\
    & 1-2 weeks (\%) & """ + ' & '.join([format_number(unpack[cat]['1w_to_2w'], True, 1) for cat in categories]) + r""" \\
    & 2-4 weeks (\%) & """ + ' & '.join([format_number(unpack[cat]['2w_to_4w'], True, 1) for cat in categories]) + r""" \\
    & 1-3 months (\%) & """ + ' & '.join([format_number(unpack[cat]['1m_to_3m'], True, 1) for cat in categories]) + r""" \\
    & 3-12 months (\%) & """ + ' & '.join([format_number(unpack[cat]['3m_to_12m'], True, 1) for cat in categories]) + r""" \\
    & Over a year (\%) & """ + ' & '.join([format_number(unpack[cat]['gt_1y'], True, 1) for cat in categories]) + r""" \\
    \bottomrule
    \end{tabular}
 \end{table}
"""
        
        # Write the LaTeX table to file
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")
        
    logger.info("Table2_broad.py script completed successfully.")