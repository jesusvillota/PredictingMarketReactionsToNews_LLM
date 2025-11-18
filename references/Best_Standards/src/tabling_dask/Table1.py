# uv run src/tabling_dask/Table1.py
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
from src.config.config_settings import PROCESSED_PATH, tables
from src.tabling_dask.common import build_parquet_filters

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_table(parquet_filters: list, output_dir, ddf=None):
    from src.tabling_dask.common import get_required_columns

    logger = get_logger(__name__)
    
    logger.info("Starting Table1.build_table().")
    required_columns = ['prtSize_agg', 'fragment_count']
    
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

        # logger.info(f"Loaded Dask DataFrame with {len(ddf)} rows.")


        percentiles = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99, 99.9, 99.99, 99.999, 99.9999, 99.99999]
        qs = [p / 100 for p in percentiles]

        # trade size (ts) and fragmentation (fr)
        ddf_nonna = ddf.dropna(subset=['prtSize_agg', 'fragment_count'])
        
        ts = ddf_nonna['prtSize_agg']
        fr = ddf_nonna['fragment_count']

    # with DaskManager() as dask_manager:
        logger.info("Preparing lazy computations for statistics and quantiles...")
        
        ts_quant = ts.quantile(qs).compute()
        q_values = ts_quant.values.tolist()
        
        # Define lazy computations for overall statistics
        overall_lazies = [
            len(ddf_nonna),
            ts.mean(),
            (ts ** 2).mean(),
            (ts ** 3).mean(),
            (ts ** 4).mean(),
            fr.mean(),
            (fr ** 2).mean(),
            (fr ** 3).mean(),
            (fr ** 4).mean(),
        ]
        
        # Define lazy computations for bin means of fragmentation
        bin_lazies = []
        # First bin: ts <= q_values[0] (bottom 1%)
        bin_lazies.append(fr[ts <= q_values[0]].mean())
        # Intermediate bins: q_values[i-1] < ts <= q_values[i] for i=1 to 22
        for i in range(1, len(q_values) - 1):
            bin_lazies.append(fr[(ts > q_values[i - 1]) & (ts <= q_values[i])].mean())
        # Last bin: ts > q_values[-1] (top tail)
        bin_lazies.append(fr[ts > q_values[-1]].mean())

        # Combine all lazy computations
        lazy_computations = overall_lazies + bin_lazies
        logger.info("Computing all statistics and quantiles with Dask...")
        
        results = dd.compute(*lazy_computations)
        logger.info("Computation finished. Unpacking results...")

        # Unpack overall results
        n, ts_m1, ts_m2, ts_m3, ts_m4, fr_m1, fr_m2, fr_m3, fr_m4 = results[:9]
        fr_bin_means = results[9:]  # The 24 bin means

        # Compute central moments and statistics (matching scipy defaults: bias=True)
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
        
        # Note: Quantiles use Dask's default t-digest approximation for efficiency on large data.
        # For exact quantiles (slower, more memory-intensive), replace with:
        # ts_quant = ts.sort_values().reset_index(drop=True).quantile(qs).compute()
        
        # Resolve output path
        out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "Table1.tex"
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        logger.info(f"Writing results to LaTeX file at {output_path} ...")
        
        with open(output_path, 'w') as f:
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
                f.write(f"P{p} & {ts_val:,.0f} & {fr_val:.2f} " + r'\\' + '\n')
            f.write(r'\midrule' + '\n')
            f.write(f"Mean & {ts_mean:,.2f} & {fr_mean:,.2f} " + r'\\' + '\n')
            f.write(f"St. Dev & {ts_std:,.2f} & {fr_std:,.2f} " + r'\\' + '\n')
            f.write(f"Skew & {ts_skew:,.2f} & {fr_skew:,.2f} " + r'\\' + '\n')
            f.write(f"Kurt & {ts_kurt:,.2f} & {fr_kurt:,.2f} " + r'\\' + '\n')
            f.write(r'\bottomrule' + '\n')
            f.write(r'\end{tabular}' + '\n')
            f.write(r'\end{table}' + '\n')
        logger.info("Finished writing LaTeX file.")


if __name__ == '__main__':
    parquet_filters = build_parquet_filters(
        ticker_type="all", 
        strategy_type="all",
        start=None,
        end=None,
    )
    build_table(parquet_filters, OUTPUT_DIR)
        