# uv run src/tabling_dask/Table8.py
#---------------------------------------------------------------
from pathlib import Path
import sys
import shutil
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
from src.config import config_settings, get_logger, DaskManager
from src.config.config_settings import PROCESSED_PATH, tables
from src.tabling_dask.common import fmt_cell, build_parquet_filters, create_standard_size_bins

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_table(parquet_filters: list, output_dir, ddf=None):
    logger = get_logger(__name__)
    logger.info("Starting Table8.build_table().")
    required_columns = ['prtExch', 'prtSize_agg']
    
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

        # Create standard size bins using centralized function
        bins = create_standard_size_bins(ddf, size_col='prtSize_agg')
        ddf_clean = bins['all']  # Keep reference to cleaned dataframe for unique discovery
        
        # Discover unique exchanges in the data
        logger.info("Discovering unique exchanges in the data...")
        unique_prtExch = ddf_clean['prtExch'].unique().compute().tolist()
        unique_prtExch = sorted([str(x) for x in unique_prtExch])
        logger.info(f"Found exchanges: {unique_prtExch}")

        # Prepare lazy computations: counts per exchange in each bin and All
        logger.info("Preparing lazy computations for counts per exchange and size bin...")
        
        lazy_counts = []
        
        # For each exchange, count trades in each size bin and All
        for prtExch in unique_prtExch:
            # All (total for this exchange regardless of size)
            lazy_counts.append(
                bins['all'][bins['all']['prtExch'] == prtExch]['prtExch'].count()
            )
            # Bin 1-10
            lazy_counts.append(
                bins['1_10'][bins['1_10']['prtExch'] == prtExch]['prtExch'].count()
            )
            # Bin 11-200
            lazy_counts.append(
                bins['11_200'][bins['11_200']['prtExch'] == prtExch]['prtExch'].count()
            )
            # Bin >200
            lazy_counts.append(
                bins['over_200'][bins['over_200']['prtExch'] == prtExch]['prtExch'].count()
            )
        
        # Compute totals for each size bin and All (for percentage calculations)
        lazy_counts.append(bins['all']['prtExch'].count())  # total_all
        lazy_counts.append(bins['1_10']['prtExch'].count())  # total_1_10
        lazy_counts.append(bins['11_200']['prtExch'].count())  # total_11_200
        lazy_counts.append(bins['over_200']['prtExch'].count())  # total_over_200

        logger.info("Computing all counts with Dask...")
        results = dd.compute(*lazy_counts)
        logger.info("Computation finished. Unpacking results...")

        # Unpack results
        counts_per_exchange_bin = results[:-4]
        total_all = results[-4]
        total_1_10 = results[-3]
        total_11_200 = results[-2]
        total_over_200 = results[-1]
        
        totals = [total_all, total_1_10, total_11_200, total_over_200]
        logger.info(f"Totals: All={total_all:,}, 1-10={total_1_10:,}, 11-200={total_11_200:,}, >200={total_over_200:,}")

        # Organize counts into a dictionary: {prtExch: [count_all, count_1_10, count_11_200, count_over_200]}
        counts_dict = {}
        for i, prtExch in enumerate(unique_prtExch):
            idx = i * 4
            counts_dict[prtExch] = [
                counts_per_exchange_bin[idx],      # all
                counts_per_exchange_bin[idx + 1],  # bin 1-10
                counts_per_exchange_bin[idx + 2],  # bin 11-200
                counts_per_exchange_bin[idx + 3],  # bin >200
            ]

        out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "Table8.tex"
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        logger.info(f"Writing LaTeX table to {output_path}...")
        
        # Build LaTeX table
        table_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{OPRA Exchanges (\texttt{prtExch}), with placeholders for size bin counts}',
            r'\label{tab:exchanges_bins_complex}',
            r'\scriptsize',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            r'\textbf{Exchange} & \textbf{All} & \textbf{1--10} & \textbf{11--200} & \textbf{$>$200} \\',
            r'\midrule',
        ]

        # Add rows for each exchange
        # Set percentage_only=True to show only percentages (change to False for "count (percentage%)")
        percentage_only = True
        
        for prtExch in unique_prtExch:
            counts = counts_dict[prtExch]
            
            # Format cells using fmt_cell function
            # totals: [total_all, total_1_10, total_11_200, total_over_200]
            cell_all = fmt_cell(counts[0], totals[0], percentage_only=percentage_only)
            cell_1_10 = fmt_cell(counts[1], totals[1], percentage_only=percentage_only)
            cell_11_200 = fmt_cell(counts[2], totals[2], percentage_only=percentage_only)
            cell_over_200 = fmt_cell(counts[3], totals[3], percentage_only=percentage_only)
            
            table_lines.append(f"\\texttt{{{prtExch}}} & {cell_all} & {cell_1_10} & {cell_11_200} & {cell_over_200} \\\\")

        # Add Total row
        table_lines.append(r'\midrule')
        table_lines.append(f"\\textbf{{Total}} & {total_all:,} & {total_1_10:,} & {total_11_200:,} & {total_over_200:,} \\\\")

        table_lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]

        # Write to file
        latex_content = '\n'.join(table_lines)
        with open(output_path, 'w') as f:
            f.write(latex_content)

        logger.info(f"LaTeX table successfully written to: {output_path}")
        logger.info("Table8.build_table() completed successfully.")
if __name__ == '__main__':
    parquet_filters = build_parquet_filters(
        ticker_type="all", 
        strategy_type="all",
        start=None,
        end=None,
    )
    build_table(parquet_filters, OUTPUT_DIR)