# uv run src/tabling_duckdb/Table5.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import duckdb
import pandas as pd
from src.config import initialize_main
from src.config.config_settings import tables, TEMP_DIR
from THIS_IS import COMPLEX_TRADES_PATH

OUTPUT_DIR = PROJECT_ROOT / tables["duckdb_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_5_new.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting Table5.py script with DuckDB.")

    try:
        # Connect to DuckDB
        con = duckdb.connect()
        
        # Configure DuckDB to use disk for temporary storage and set memory limit
        con.execute("SET memory_limit='64GB'")
        con.execute(f"SET temp_directory='{TEMP_DIR}'")
        con.execute("SET preserve_insertion_order=false")

        logger.info(f"DuckDB connection established with memory_limit=64GB and temp_directory={TEMP_DIR}")

        # Get the parquet path pattern for complex trades
        complex_parquet_path = str(COMPLEX_TRADES_PATH / "**" / "*.parquet")
        logger.info(f"Reading complex trades parquet files from: {complex_parquet_path}")
        
        # Build SQL query to construct strategy types and assign size categories
        logger.info("Building DuckDB query for complex strategy aggregation...")
        
        query = f"""
        WITH base_data AS (
            SELECT 
                prtSize_agg,
                n_legs,
                sign,
                flag,
                strategy_name,
                -- Map Single sign values: buy->Long, sell->Short, midpoint->Midpoint
                CASE 
                    WHEN n_legs = 1 AND strategy_name = 'Single' THEN
                        CASE sign
                            WHEN 'buy' THEN 'Long'
                            WHEN 'sell' THEN 'Short'
                            WHEN 'midpoint' THEN 'Midpoint'
                            ELSE sign
                        END
                    ELSE sign
                END AS mapped_sign,
                -- Assign size categories
                CASE 
                    WHEN prtSize_agg BETWEEN 1 AND 10 THEN '1_10'
                    WHEN prtSize_agg BETWEEN 11 AND 200 THEN '11_200'
                    WHEN prtSize_agg > 200 THEN 'over_200'
                    ELSE 'other_size'
                END AS size_category
            FROM read_parquet('{complex_parquet_path}', hive_partitioning=0)
            WHERE strategy_name IS NOT NULL
                AND prtSize_agg IS NOT NULL
                AND n_legs IS NOT NULL
        ),
        strategy_details AS (
            SELECT 
                mapped_sign,
                flag,
                strategy_name,
                n_legs,
                size_category,
                -- Construct strategy type based on strategy, sign, and flag
                CASE 
                    -- Explicit Other category: fixed label, no sign/flag
                    WHEN strategy_name = 'Other' THEN 'Other'
                    -- Single strategies: "Sign Flag" (e.g., "Long Call", "Short Put")
                    WHEN n_legs = 1 AND strategy_name = 'Single' THEN
                        mapped_sign || ' ' || flag
                    
                    -- Mixed strategies (Straddle, Strangle, IronCondor): "Sign Strategy" (no flag)
                    WHEN strategy_name IN ('Straddle', 'Strangle') THEN
                        mapped_sign || ' ' || strategy_name
                    WHEN strategy_name = 'IronCondor' THEN
                        mapped_sign || ' ' || strategy_name
                    
                    -- Other strategies: "Sign Strategy Flag" (e.g., "Long Spread Call")
                    ELSE mapped_sign || ' ' || strategy_name || ' ' || flag
                END AS strategy_type,
                -- Construct strategy row identifier for matching with template
                CASE WHEN strategy_name = 'Other' THEN '' ELSE mapped_sign END AS sign_col,
                CASE WHEN strategy_name = 'Other' THEN '' ELSE flag END AS flag_col
            FROM base_data
        )
        SELECT 
            strategy_type,
            sign_col,
            flag_col,
            strategy_name,
            n_legs,
            size_category,
            COUNT(*) as count
        FROM strategy_details
        GROUP BY strategy_type, sign_col, flag_col, strategy_name, n_legs, size_category
        ORDER BY n_legs, strategy_name, sign_col, flag_col, size_category
        """
        
        logger.info("Executing DuckDB query...")
        result_df = con.execute(query).fetchdf()
        
        logger.info(f"Computation finished. Retrieved {len(result_df):,} strategy-size combinations.")
        
        # Get total count
        total_N = con.execute(f"""
            SELECT COUNT(*) as n
            FROM read_parquet('{complex_parquet_path}', hive_partitioning=0)
            WHERE strategy_name IS NOT NULL
                AND prtSize_agg IS NOT NULL
                AND n_legs IS NOT NULL
        """).fetchdf()['n'].iloc[0]
        
        logger.info(f"Total complex strategies: {total_N:,}")
        
        # Compute totals for each size category
        size_totals = con.execute(f"""
            SELECT 
                CASE 
                    WHEN prtSize_agg BETWEEN 1 AND 10 THEN '1_10'
                    WHEN prtSize_agg BETWEEN 11 AND 200 THEN '11_200'
                    WHEN prtSize_agg > 200 THEN 'over_200'
                    ELSE 'other_size'
                END AS size_category,
                COUNT(*) as n
            FROM read_parquet('{complex_parquet_path}', hive_partitioning=0)
            WHERE strategy_name IS NOT NULL
                AND prtSize_agg IS NOT NULL
                AND n_legs IS NOT NULL
            GROUP BY size_category
        """).fetchdf()
        
        totals_dict = dict(zip(size_totals['size_category'], size_totals['n']))
        totals_dict['all'] = total_N
        
        # Aggregate counts by strategy for "all" column
        all_counts = result_df.groupby(['strategy_type', 'sign_col', 'flag_col', 'strategy_name', 'n_legs'])['count'].sum().reset_index()
        all_counts['size_category'] = 'all'
        
        # Combine with size-specific counts
        combined_df = pd.concat([result_df, all_counts], ignore_index=True)
        
        # Create a pivot table for easier access
        pivot_df = combined_df.pivot_table(
            index=['strategy_type', 'sign_col', 'flag_col', 'strategy_name', 'n_legs'],
            columns='size_category',
            values='count',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Helper function to format count with percentage
        def fmt_cell(count: int, total: int) -> str:
            if total == 0:
                return "0 (0.0\\%)"
            pct = (count / total * 100.0)
            return f"{int(count):,} ({pct:.1f}\\%)"
        
        # Create a dictionary to look up counts by strategy signature
        strategy_counts = {}
        for _, row in pivot_df.iterrows():
            key = (row['sign_col'], row['flag_col'], row['strategy_name'])
            strategy_counts[key] = {
                'all': row.get('all', 0),
                '1_10': row.get('1_10', 0),
                '11_200': row.get('11_200', 0),
                'over_200': row.get('over_200', 0)
            }
        
        # Get a set of known strategies to identify "Other"
        known_strategies = set()
        for _, row in pivot_df.iterrows():
            known_strategies.add((row['sign_col'], row['flag_col'], row['strategy_name']))
        
        # No separate read for 'Other' needed; included in aggregated results
        
        logger.info("Writing LaTeX table...")
        
        # Define the exact row order to match the LaTeX template
        def get_counts_for_row(sign, flag, strategy):
            """Helper to get counts for a specific strategy row"""
            key = (sign, flag, strategy)
            return strategy_counts.get(key, {'all': 0, '1_10': 0, '11_200': 0, 'over_200': 0})
        
        # Build LaTeX table content
        latex_rows = []
        
        # Add header
        latex_rows.append(r"\begin{table}[htbp]")
        latex_rows.append(r"    \centering")
        latex_rows.append(r"    \caption{Distribution of Complex Option Strategies by Trade Size Category}")
        latex_rows.append(r"    \subcaption*{")
        latex_rows.append(r"    {\scriptsize")
        latex_rows.append(
            f"    Distribution of complex option strategies across different trade size categories. Each cell shows the count of trades for that strategy type with the percentage within that size category shown in parentheses. Rows show different strategy types as classified by the algorithm. Columns represent trade size categories based on the number of contracts: All (all trades), 1--10 (size between 1 and 10), 11--200 (size between 11 and 200), and >200 (size greater than 200). Data filtered to complex trades with prtType $\\geq$ 102. Total observations: $N = {int(total_N):,}$ complex strategies."
        )
        latex_rows.append(r"    \par}")
        latex_rows.append(r"    \vspace{1em}")
        latex_rows.append(r"    }")
        latex_rows.append(r"    \label{tab:complex_strategies_by_size}")
        latex_rows.append(r"    \scriptsize")
        latex_rows.append(r"    \begin{tabular}{llllcccc}")
        latex_rows.append(r"    \toprule")
        latex_rows.append(r"        \textbf{Legs} & \textbf{Strategy} & \textbf{Sign} & \textbf{Flag} & \textbf{All} & \textbf{1--10} & \textbf{11--200} & \textbf{>200} \\")
        latex_rows.append(r"    \midrule")
        
        # 1-Leg Strategies (6 rows: Long/Short/Midpoint × Call/Put for each sign)
        latex_rows.append(r"    \multirow{6}{*}{\textbf{1 Leg}}")
        # Long Call/Put, Short Call/Put, Midpoint Call/Put
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts_for_row(sign, flag, 'Single')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Single & {sign} & {flag} & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # 2-Leg Strategies (24 rows: Spread/Calendar/Diagonal × Long/Short/Midpoint × Call/Put, Straddle/Strangle × Long/Short/Midpoint)
        latex_rows.append(r"    \multirow{24}{*}{\textbf{2 Legs}}")
        # Spread
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts_for_row(sign, flag, 'Spread')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Spread & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # Calendar
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts_for_row(sign, flag, 'Calendar')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Calendar & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # Diagonal
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts_for_row(sign, flag, 'Diagonal')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Diagonal & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # Straddle (no flag)
        for sign in ['Long', 'Short', 'Midpoint']:
            counts_dict = get_counts_for_row(sign, 'Mixed', 'Straddle')
            counts = [
                fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
            ]
            latex_rows.append(f"    & Straddle & {sign} &  & {' & '.join(counts)} \\\\")
        # Strangle (no flag)
        for sign in ['Long', 'Short', 'Midpoint']:
            counts_dict = get_counts_for_row(sign, 'Mixed', 'Strangle')
            counts = [
                fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
            ]
            latex_rows.append(f"    & Strangle & {sign} &  & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # 3-Leg Strategies - Butterfly (6 rows: Long/Short/Midpoint × Call/Put)
        latex_rows.append(r"    \multirow{6}{*}{\textbf{3 Legs}}")
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts_for_row(sign, flag, 'Butterfly')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Butterfly & {sign} & {flag} & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # 4-Leg Strategies (9 rows: Condor × Long/Short/Midpoint × Call/Put, IronCondor × Long/Short/Midpoint)
        latex_rows.append(r"    \multirow{9}{*}{\textbf{4 Legs}}")
        # Condor
        for sign in ['Long', 'Short', 'Midpoint']:
            for flag in ['Call', 'Put']:
                counts_dict = get_counts_for_row(sign, flag, 'Condor')
                counts = [
                    fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                    fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                    fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                    fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
                ]
                latex_rows.append(f"    & Condor & {sign} & {flag} & {' & '.join(counts)} \\\\")
        # IronCondor (no flag)
        for sign in ['Long', 'Short', 'Midpoint']:
            counts_dict = get_counts_for_row(sign, 'Mixed', 'IronCondor')
            counts = [
                fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
                fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
                fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
                fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
            ]
            latex_rows.append(f"    & IronCondor & {sign} &  & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # Other (from aggregated results; no sign/flag)
        latex_rows.append(r"    \multirow{1}{*}{\textbf{ }}")
        counts_dict = get_counts_for_row('', '', 'Other')
        counts = [
            fmt_cell(counts_dict['all'], totals_dict.get('all', 1)),
            fmt_cell(counts_dict['1_10'], totals_dict.get('1_10', 1)),
            fmt_cell(counts_dict['11_200'], totals_dict.get('11_200', 1)),
            fmt_cell(counts_dict['over_200'], totals_dict.get('over_200', 1))
        ]
        latex_rows.append(f"    & Other &  &  & {' & '.join(counts)} \\\\")
        
        latex_rows.append(r"    \midrule")
        
        # Total row
        total_counts = [
            fmt_cell(totals_dict.get('all', 0), totals_dict.get('all', 1)),
            fmt_cell(totals_dict.get('1_10', 0), totals_dict.get('1_10', 1)),
            fmt_cell(totals_dict.get('11_200', 0), totals_dict.get('11_200', 1)),
            fmt_cell(totals_dict.get('over_200', 0), totals_dict.get('over_200', 1))
        ]
        latex_rows.append(f"    \\textbf{{Total}} &  &  &  & {' & '.join(total_counts)} \\\\")
        latex_rows.append(r"    \bottomrule")
        latex_rows.append(r"    \end{tabular}")
        latex_rows.append(r"\end{table}")
        
        # Write to file
        latex_content = '\n'.join(latex_rows) + '\n'
        
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")
        
    logger.info("Table5.py script completed successfully.")

