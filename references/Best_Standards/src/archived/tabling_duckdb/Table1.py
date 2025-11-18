# uv run src/tabling_duckdb/Table1.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import duckdb
import numpy as np
from src.config import initialize_main
from src.config.config_settings import PROCESSED_PATH, tables

OUTPUT_DIR = PROJECT_ROOT / tables["duckdb_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_1.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':

    logger = initialize_main()
    logger.info("Starting Table1.py script with DuckDB.")
    
    try:
        # Connect to DuckDB
        con = duckdb.connect()
        logger.info("DuckDB connection established.")
        
        # Get the parquet path pattern
        parquet_path = str(PROCESSED_PATH / "**/*.parquet")
        logger.info(f"Reading parquet files from: {parquet_path}")
        
        percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999, 99.99999]
        qs = [p / 100 for p in percentiles]
        
        # Step 1: Compute quantiles and moments
        logger.info("Computing quantiles and moments with DuckDB...")
        
        quantile_sql = ", ".join([f"quantile(prtSize_agg, {q}) as q{int(p*100 if p >= 1 else p*10000)}" for p, q in zip(percentiles, qs)])
        
        stats_query = f"""
        SELECT 
            COUNT(*) as n,
            -- Trade size moments
            AVG(prtSize_agg) as ts_m1,
            AVG(POW(prtSize_agg, 2)) as ts_m2,
            AVG(POW(prtSize_agg, 3)) as ts_m3,
            AVG(POW(prtSize_agg, 4)) as ts_m4,
            -- Fragmentation moments
            AVG(fragment_count) as fr_m1,
            AVG(POW(fragment_count, 2)) as fr_m2,
            AVG(POW(fragment_count, 3)) as fr_m3,
            AVG(POW(fragment_count, 4)) as fr_m4,
            -- Quantiles
            {quantile_sql}
        FROM read_parquet('{parquet_path}', hive_partitioning=0)
        WHERE ticker_class = 'equity'
            AND prtType >= 73
            AND prtSize_agg IS NOT NULL
            AND fragment_count IS NOT NULL
        """
        
        stats_df = con.execute(stats_query).fetchdf()
        
        # Extract values
        n = stats_df['n'].iloc[0]
        ts_m1, ts_m2, ts_m3, ts_m4 = stats_df['ts_m1'].iloc[0], stats_df['ts_m2'].iloc[0], stats_df['ts_m3'].iloc[0], stats_df['ts_m4'].iloc[0]
        fr_m1, fr_m2, fr_m3, fr_m4 = stats_df['fr_m1'].iloc[0], stats_df['fr_m2'].iloc[0], stats_df['fr_m3'].iloc[0], stats_df['fr_m4'].iloc[0]
        
        # Extract quantile values
        q_values = [stats_df[f'q{int(p*100 if p >= 1 else p*10000)}'].iloc[0] for p in percentiles]
        
        # Compute central moments and statistics
        ts_cm2 = ts_m2 - ts_m1 ** 2
        ts_cm3 = ts_m3 - 3 * ts_m1 * ts_m2 + 2 * ts_m1 ** 3
        ts_cm4 = ts_m4 - 4 * ts_m1 * ts_m3 + 6 * ts_m1 ** 2 * ts_m2 - 3 * ts_m1 ** 4

        fr_cm2 = fr_m2 - fr_m1 ** 2
        fr_cm3 = fr_m3 - 3 * fr_m1 * fr_m2 + 2 * fr_m1 ** 3
        fr_cm4 = fr_m4 - 4 * fr_m1 * fr_m3 + 6 * fr_m1 ** 2 * fr_m2 - 3 * fr_m1 ** 4

        # Statistics (overall)
        ts_mean = ts_m1
        ts_std = np.sqrt(n / (n - 1) * ts_cm2)
        ts_skew = ts_cm3 / ts_cm2 ** 1.5 if ts_cm2 > 0 else np.nan
        ts_kurt = (ts_cm4 / ts_cm2 ** 2) - 3 if ts_cm2 > 0 else np.nan

        fr_mean = fr_m1
        fr_std = np.sqrt(n / (n - 1) * fr_cm2)
        fr_skew = fr_cm3 / fr_cm2 ** 1.5 if fr_cm2 > 0 else np.nan
        fr_kurt = (fr_cm4 / fr_cm2 ** 2) - 3 if fr_cm2 > 0 else np.nan
        
        # Step 2: Compute mean fragmentation for each percentile bin
        logger.info("Computing mean fragmentation for each percentile bin...")
        
        # Build CASE statement for binning
        bin_cases = []
        # First bin: ts <= q_values[0]
        bin_cases.append(f"WHEN prtSize_agg <= {q_values[0]} THEN 0")
        # Intermediate bins
        for i in range(1, len(q_values)):
            bin_cases.append(f"WHEN prtSize_agg > {q_values[i-1]} AND prtSize_agg <= {q_values[i]} THEN {i}")
        # Last bin: ts > q_values[-1]
        bin_cases.append(f"WHEN prtSize_agg > {q_values[-1]} THEN {len(q_values)}")
        
        bin_query = f"""
        SELECT 
            CASE 
                {' '.join(bin_cases)}
            END AS bin_idx,
            AVG(fragment_count) as avg_frag
        FROM read_parquet('{parquet_path}', hive_partitioning=0)
        WHERE ticker_class = 'equity'
            AND prtType >= 73
            AND prtSize_agg IS NOT NULL
            AND fragment_count IS NOT NULL
        GROUP BY bin_idx
        ORDER BY bin_idx
        """
        
        bin_df = con.execute(bin_query).fetchdf()
        
        # Create a dictionary mapping bin_idx to avg_frag, handling missing bins
        bin_dict = dict(zip(bin_df['bin_idx'], bin_df['avg_frag']))
        
        # Ensure we have values for all bins 0 to len(q_values)-1 (26 bins for 26 percentiles)
        # Use np.nan for bins with no observations
        fr_bin_means = [bin_dict.get(i, np.nan) for i in range(len(q_values))]
        
        logger.info(f"Computation finished. Total observations: {n:,}")
        logger.info(f"Bins with data: {len(bin_df)}, Expected bins: {len(q_values)}")
        
    except Exception as e:
        logger.exception(f"Error during DuckDB query execution: {e}")
        raise
    finally:
        con.close()
        logger.info("DuckDB connection closed.")
    
    logger.info(f"Writing results to LaTeX file at {OUTPUT_PATH} ...")
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Distribution of Trade Size and Fragmentation for the Full Sample}' + '\n')
        f.write(r'\label{tab:percentile_moments_trade_fragmentation}' + '\n')
        f.write(r'\scriptsize' + '\n')
        f.write(r'\begin{minipage}{0.95\linewidth}' + '\n')
        f.write(r'\vspace{0.5em}' + '\n')
        # Write the exact footnotesize note from Table1.tex, replacing the placeholder with the actual observation count
        f.write(r'\footnotesize{\textit{Note:} For each percentile of trade size, we report the corresponding value and the mean fragmentation (number of fragments) for trades within that percentile bin. Percentile bins are constructed such that, for each bin, fragmentation is averaged over all trades whose size falls within the bin boundaries. Summary statistics (mean, standard deviation, skewness, kurtosis) are computed for the full sample of trade sizes and fragment counts. Total observations: $N = ' + f'{int(n):,}' + r'$ trades.}' + '\n')
        f.write(r'\vspace{0.5em}' + '\n')
        f.write(r'\end{minipage}' + '\n')
        f.write(r'\footnotesize' + '\n')
        f.write(r'\begin{tabular}{lcc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'    & \textbf{Trade Size} & \textbf{Fragmentation} \\' + '\n')
        f.write(r'\midrule' + '\n')
        for i, p in enumerate(percentiles):
            ts_val = q_values[i]
            fr_val = fr_bin_means[i]
            # Handle NaN values for bins with no observations
            fr_str = f"{fr_val:.2f}" if not np.isnan(fr_val) else "---"
            f.write(f"P{p} & {ts_val:,.0f} & {fr_str} " + r'\\' + '\n')
        f.write(r'\midrule' + '\n')
        f.write(f"Mean & {ts_mean:,.2f} & {fr_mean:,.2f} " + r'\\' + '\n')
        f.write(f"St. Dev & {ts_std:,.2f} & {fr_std:,.2f} " + r'\\' + '\n')
        f.write(f"Skew & {ts_skew:,.2f} & {fr_skew:,.2f} " + r'\\' + '\n')
        f.write(f"Kurt & {ts_kurt:,.2f} & {fr_kurt:,.2f} " + r'\\' + '\n')
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    logger.info("Finished writing LaTeX file.")
        