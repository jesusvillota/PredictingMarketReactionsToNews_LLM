# uv run src/tabling_dask/Table6.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
from src.config import config_settings, initialize_main, DaskManager
from src.config.config_settings import PROCESSED_PATH, tables

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_6.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting Table_prtTypes_bins.py script.")

    # prtType code to print code name mapping
    prtType_names = {
        73: "Electronic",
        74: "Halt Reopening",
        83: "ISO Order",
        97: "Single Leg Auction Non ISO",
        98: "Single Leg Auction ISO",
        99: "Single Leg Cross Non ISO",
        100: "Single Leg Cross ISO",
        101: "Single Leg Floor Trade"
    }

    def fmt_cell(count: int, total: int, percentage_only: bool = False) -> str:
        """Format count with percentage for LaTeX table cell.
        
        Args:
            count: The count value
            total: The total for percentage calculation
            percentage_only: If True, return only percentage with 3 decimals.
                           If False, return "count (percentage%)" with 1 decimal
        """
        if total == 0:
            return "0.000\\%" if percentage_only else "0 (0.0\\%)"
        pct = (count / total * 100.0)
        if percentage_only:
            return f"{pct:.3f}\\%"
        else:
            return f"{int(count):,} ({pct:.1f}\\%)"

    with DaskManager() as dask_manager:
        try:
            logger.info("Loading parquet data with Dask...")
            ddf = dd.read_parquet(
                path=PROCESSED_PATH,
                engine=config_settings.parquet["engine"],
                filters=[
                    ('ticker_class', '==', 'equity'),
                    ('prtType', '>=', 73),
                    ('prtType', '<', 102),
                ],
                columns=['prtType', 'prtSize_agg'],
                split_row_groups='infer',
            )
            logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
        except Exception as e:
            logger.exception(f"Error loading parquet: {e}")
            raise

        # Drop NA values for prtSize_agg
        ddf_clean = ddf.dropna(subset=['prtSize_agg'])
        
        # Discover unique prtTypes in the data
        logger.info("Discovering unique prtTypes in the data...")
        unique_prtTypes = ddf_clean['prtType'].unique().compute().tolist()
        unique_prtTypes = sorted([int(x) for x in unique_prtTypes])
        logger.info(f"Found prtTypes: {unique_prtTypes}")

        # Define size bin conditions
        mask_1_10 = (ddf_clean['prtSize_agg'] >= 1) & (ddf_clean['prtSize_agg'] <= 10)
        mask_11_200 = (ddf_clean['prtSize_agg'] >= 11) & (ddf_clean['prtSize_agg'] <= 200)
        mask_over_200 = ddf_clean['prtSize_agg'] > 200

        # Create filtered dataframes for each size bin
        ddf_1_10 = ddf_clean[mask_1_10]
        ddf_11_200 = ddf_clean[mask_11_200]
        ddf_over_200 = ddf_clean[mask_over_200]

        # Prepare lazy computations: counts per prtType in each bin
        logger.info("Preparing lazy computations for counts per prtType and size bin...")
        
        lazy_counts = []
        
        # For each prtType, count trades in each size bin
        for prtType in unique_prtTypes:
            # Bin 1-10
            lazy_counts.append(
                ddf_1_10[ddf_1_10['prtType'] == prtType]['prtType'].count()
            )
            # Bin 11-200
            lazy_counts.append(
                ddf_11_200[ddf_11_200['prtType'] == prtType]['prtType'].count()
            )
            # Bin >200
            lazy_counts.append(
                ddf_over_200[ddf_over_200['prtType'] == prtType]['prtType'].count()
            )
        
        # Compute totals for each size bin (for percentage calculations)
        lazy_counts.append(ddf_1_10['prtType'].count())  # total_1_10
        lazy_counts.append(ddf_11_200['prtType'].count())  # total_11_200
        lazy_counts.append(ddf_over_200['prtType'].count())  # total_over_200

        logger.info("Computing all counts with Dask...")
        results = dd.compute(*lazy_counts)
        logger.info("Computation finished. Unpacking results...")

        # Unpack results
        counts_per_prtType_bin = results[:-3]
        total_1_10 = results[-3]
        total_11_200 = results[-2]
        total_over_200 = results[-1]
        
        totals = [total_1_10, total_11_200, total_over_200]
        logger.info(f"Totals: 1-10={total_1_10:,}, 11-200={total_11_200:,}, >200={total_over_200:,}")

        # Organize counts into a dictionary: {prtType: [count_1_10, count_11_200, count_over_200]}
        counts_dict = {}
        for i, prtType in enumerate(unique_prtTypes):
            idx = i * 3
            counts_dict[prtType] = [
                counts_per_prtType_bin[idx],      # bin 1-10
                counts_per_prtType_bin[idx + 1],  # bin 11-200
                counts_per_prtType_bin[idx + 2],  # bin >200
            ]

        logger.info(f"Writing LaTeX table to {OUTPUT_PATH}...")
        
        # Build LaTeX table
        table_lines = [
            r'\begin{table}[htbp]',
            r'\centering',
            r'\caption{OPRA Print Codes (\texttt{prtType}), with placeholders for size bin counts}',
            r'\label{tab:prtTypes_bins}',
            r'\scriptsize',
            r'\begin{tabular}{llccc}',
            r'\toprule',
            r'\textbf{prtType} & \textbf{Print Code} & \textbf{1--10} & \textbf{11--200} & \textbf{$>$200} \\',
            r'\midrule',
        ]

        # Add rows for each prtType
        # Set percentage_only=True to show only percentages (change to False for "count (percentage%)")
        percentage_only = True
        
        for prtType in unique_prtTypes:
            counts = counts_dict[prtType]
            print_code = prtType_names.get(prtType, f"prtType {prtType}")
            
            # Format cells using fmt_cell function
            cell_1_10 = fmt_cell(counts[0], totals[0], percentage_only=percentage_only)
            cell_11_200 = fmt_cell(counts[1], totals[1], percentage_only=percentage_only)
            cell_over_200 = fmt_cell(counts[2], totals[2], percentage_only=percentage_only)
            
            table_lines.append(f"{prtType}  & {print_code}          & {cell_1_10} & {cell_11_200} & {cell_over_200} \\\\")

        table_lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]

        # Write to file
        latex_content = '\n'.join(table_lines)
        with open(OUTPUT_PATH, 'w') as f:
            f.write(latex_content)

        logger.info(f"LaTeX table successfully written to: {OUTPUT_PATH}")
        logger.info("Table_prtTypes_bins.py script completed successfully.")

