# uv run src/tabling_duckdb/Table3.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------
import duckdb
from src.config import initialize_main
from src.config.config_settings import PROCESSED_PATH, tables

OUTPUT_DIR = PROJECT_ROOT / tables["duckdb_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_3.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting Table3.py script with DuckDB.")

    try:
        # Connect to DuckDB
        con = duckdb.connect()
        logger.info("DuckDB connection established.")
        
        # Get the parquet path pattern
        parquet_path = str(PROCESSED_PATH / "**/*.parquet")
        logger.info(f"Reading parquet files from: {parquet_path}")
        
        # Build SQL query for bin-based aggregation
        logger.info("Building DuckDB query for trade size bins...")
        
        query = f"""
        WITH base_data AS (
            SELECT 
                prtSize_agg,
                fragment_count,
                CASE 
                    WHEN prtSize_agg = 1 THEN '1'
                    WHEN prtSize_agg = 2 THEN '2'
                    WHEN prtSize_agg = 3 THEN '3'
                    WHEN prtSize_agg = 4 THEN '4'
                    WHEN prtSize_agg = 5 THEN '5'
                    WHEN prtSize_agg >= 6 AND prtSize_agg < 10 THEN '6--10'
                    WHEN prtSize_agg >= 10 AND prtSize_agg < 20 THEN '10--20'
                    WHEN prtSize_agg >= 20 AND prtSize_agg < 50 THEN '20--50'
                    WHEN prtSize_agg >= 50 AND prtSize_agg < 100 THEN '50--100'
                    WHEN prtSize_agg >= 100 AND prtSize_agg < 200 THEN '100--200'
                    WHEN prtSize_agg >= 200 AND prtSize_agg < 500 THEN '200--500'
                    WHEN prtSize_agg >= 500 AND prtSize_agg < 1000 THEN '500--1,000'
                    WHEN prtSize_agg >= 1000 AND prtSize_agg < 2000 THEN '1,000--2,000'
                    WHEN prtSize_agg >= 2000 AND prtSize_agg < 5000 THEN '2,000--5,000'
                    WHEN prtSize_agg >= 5000 AND prtSize_agg < 10000 THEN '5,000--10,000'
                    WHEN prtSize_agg >= 10000 AND prtSize_agg < 20000 THEN '10,000--20,000'
                    WHEN prtSize_agg >= 20000 AND prtSize_agg < 50000 THEN '20,000--50,000'
                    WHEN prtSize_agg >= 50000 AND prtSize_agg < 100000 THEN '50,000--100,000'
                    WHEN prtSize_agg >= 100000 AND prtSize_agg < 200000 THEN '100,000--200,000'
                    WHEN prtSize_agg >= 200000 THEN '$\\\\geq$ 200,000'
                END AS bin,
                -- Ordering key for proper bin sorting
                CASE 
                    WHEN prtSize_agg = 1 THEN 1
                    WHEN prtSize_agg = 2 THEN 2
                    WHEN prtSize_agg = 3 THEN 3
                    WHEN prtSize_agg = 4 THEN 4
                    WHEN prtSize_agg = 5 THEN 5
                    WHEN prtSize_agg >= 6 AND prtSize_agg < 10 THEN 6
                    WHEN prtSize_agg >= 10 AND prtSize_agg < 20 THEN 7
                    WHEN prtSize_agg >= 20 AND prtSize_agg < 50 THEN 8
                    WHEN prtSize_agg >= 50 AND prtSize_agg < 100 THEN 9
                    WHEN prtSize_agg >= 100 AND prtSize_agg < 200 THEN 10
                    WHEN prtSize_agg >= 200 AND prtSize_agg < 500 THEN 11
                    WHEN prtSize_agg >= 500 AND prtSize_agg < 1000 THEN 12
                    WHEN prtSize_agg >= 1000 AND prtSize_agg < 2000 THEN 13
                    WHEN prtSize_agg >= 2000 AND prtSize_agg < 5000 THEN 14
                    WHEN prtSize_agg >= 5000 AND prtSize_agg < 10000 THEN 15
                    WHEN prtSize_agg >= 10000 AND prtSize_agg < 20000 THEN 16
                    WHEN prtSize_agg >= 20000 AND prtSize_agg < 50000 THEN 17
                    WHEN prtSize_agg >= 50000 AND prtSize_agg < 100000 THEN 18
                    WHEN prtSize_agg >= 100000 AND prtSize_agg < 200000 THEN 19
                    WHEN prtSize_agg >= 200000 THEN 20
                END AS bin_order
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE ticker_class = 'equity'
                AND prtType >= 73
                AND prtSize_agg IS NOT NULL
                AND fragment_count IS NOT NULL
        )
        SELECT 
            bin as label,
            COUNT(*) as count,
            AVG(fragment_count) as frag,
            bin_order
        FROM base_data
        GROUP BY bin, bin_order
        ORDER BY bin_order
        """
        
        logger.info("Executing DuckDB query...")
        result_df = con.execute(query).fetchdf()
        
        # Get total count
        N = result_df['count'].sum()
        
        # Add percentage column
        result_df['pct'] = (result_df['count'] / N * 100)
        
        # Convert to list of dicts
        bin_rows = result_df[['label', 'count', 'pct', 'frag']].to_dict('records')
        
        logger.info(f"Computation finished. Total N = {N:,}; Sum of bin counts = {result_df['count'].sum():,}.")
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")

    logger.info(f"Writing LaTeX table to {OUTPUT_PATH} ...")
    try:
        with open(OUTPUT_PATH, 'w') as f:
            f.write(r'\begin{table}[htbp]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(r'\caption{Trade Size Bins: Percent and Counts of Observations}' + '\n')
            f.write(r'\label{tab:trade_size_bins_obs}' + '\n')
            f.write(r'\small' + '\n')
            f.write(r'\begin{minipage}{0.95\linewidth}' + '\n')
            f.write(r'\vspace{0.35em}' + '\n')
            f.write(r'\footnotesize{\textit{Note:} The table reports counts and percentages of trades by trade size (prtSize\_agg) bins. Exact sizes 1--5 are single-value bins. Ranged bins use left-inclusive, right-exclusive boundaries (e.g., 6--10 means $6 \leq$ size $< 10$) except the final open bin $\geq 200{,}000$. Fragmentation is the mean number of fragments (fragment\_count) for trades in each bin. Data filtered to include only equity options (ticker\_class == Equity) with prtType $\geq$ 73. Total observations: $N = ' + f'{int(N):,}' + r'$.}' + '\n')
            f.write(r'\vspace{0.35em}' + '\n')
            f.write(r'\end{minipage}' + '\n')
            f.write(r'\begin{tabular}{lccc}' + '\n')
            f.write(r'\toprule' + '\n')
            f.write(r'\textbf{Trade Size} & \textbf{\% Obs.} & \textbf{\# Obs.} & \textbf{Fragmentation} \\' + '\n')
            f.write(r'\midrule' + '\n')
            for row in bin_rows:
                pct_fmt = f"{row['pct']:.2f}"
                count_fmt = f"{row['count']:,}"
                frag = row['frag']
                frag_fmt = (f"{frag:.2f}" if frag is not None else '')
                f.write(f"{row['label']} & {pct_fmt} & {count_fmt} & {frag_fmt} " + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            f.write(f"Total & 100.00 & {int(N):,} & - " + r'\\' + '\n')
            f.write(r'\bottomrule' + '\n')
            f.write(r'\end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        logger.info("Finished writing LaTeX file for Table 3.")
    except Exception as e:
        logger.exception(f"Failed to write LaTeX file: {e}")
        raise
