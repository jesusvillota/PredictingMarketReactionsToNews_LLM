# uv run src/tabling_dask/Table3.py
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
from src.tabling_dask.common import build_parquet_filters

OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def build_bins(series):
    """Return an ordered list of bin definitions.

    Each element is a dict with:
        label: str (as printed in the table)
        condition: Dask Series boolean mask
    """
    s = series
    bins = []
    # Exact sizes 1–5
    for k in [1, 2, 3, 4, 5]:
        bins.append({
            'label': f"{k}",
            'condition': (s == k)
        })
    # Ranged bins (left-inclusive, right-exclusive) except last open-ended
    ranges = [
        ("6--10", 6, 10),
        ("10--20", 10, 20),
        ("20--50", 20, 50),
        ("50--100", 50, 100),
        ("100--200", 100, 200),
        ("200--500", 200, 500),
        ("500--1,000", 500, 1000),
        ("1,000--2,000", 1000, 2000),
        ("2,000--5,000", 2000, 5000),
        ("5,000--10,000", 5000, 10000),
        ("10,000--20,000", 10000, 20000),
        ("20,000--50,000", 20000, 50000),
        ("50,000--100,000", 50000, 100000),
        ("100,000--200,000", 100000, 200000),
    ]
    for label, left, right in ranges:
        bins.append({
            'label': label,
            'condition': (s >= left) & (s < right)
        })
    # Open-ended final bin
    bins.append({
        'label': r'$\geq$ 200,000',
        'condition': (s >= 200000)
    })
    return bins


def build_table(parquet_filters: list, output_dir, ddf=None):
    logger = get_logger(__name__)
    logger.info("Starting Table3.build_table() with Dask.")
    required_columns = ['prtSize_agg', 'fragment_count']
    
    if ddf is None:
        with DaskManager():
            try:
                logger.info("Loading parquet data with Dask (columns: prtSize_agg, fragment_count)...")
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
    
    with DaskManager():

        # trade size (ts) and fragmentation (fr)
        ddf_nonna = ddf.dropna(subset=['prtSize_agg', 'fragment_count'])
        ts = ddf_nonna['prtSize_agg']
        fr = ddf_nonna['fragment_count']

        bins = build_bins(ts)
        logger.info(f"Prepared {len(bins)} trade size bins for computation.")

        # Collect lazy computations: total count + per-bin counts & fragmentation means
        lazies = []
        # Total observations (using first column for count)
        total_count_lazy = ts.count() # ddf_nonna.count()
        lazies.append(total_count_lazy)

        for b in bins:
            cond = b['condition']
            # Count of observations in bin
            lazies.append(ts[cond].count())
            # Mean fragmentation in bin
            lazies.append(fr[cond].mean())


        logger.info("Computing total count and all bin statistics in a single Dask graph execution...")
        results = dd.compute(*lazies)

    # Unpack results
    N = results[0]
    per_bin = results[1:]

    bin_rows = []  # list of dicts with label, count, pct, frag
    for i, b in enumerate(bins):
        count = per_bin[2*i]
        frag = per_bin[2*i + 1]
        pct = (count / N * 100) if N else 0.0
        bin_rows.append({
            'label': b['label'],
            'count': int(count),
            'pct': pct,
            'frag': frag
        })

    # Sanity check: sum of counts should be <= N (due to boundary assumptions)
    sum_counts = sum(r['count'] for r in bin_rows)
    if sum_counts > N:
        logger.warning(f"Sum of bin counts ({sum_counts}) exceeds total N ({N}). Check bin boundary assumptions.")
    else:
        logger.info(f"Total N = {N:,}; Sum of bin counts = {sum_counts:,}.")

    # Resolve output path
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "Table3.tex"
    if output_path.exists() and output_path.is_dir():
        shutil.rmtree(output_path)
    logger.info(f"Writing LaTeX table to {output_path} ...")
    try:
        with open(output_path, 'w') as f:
            f.write(r'\begin{table}[htbp]' + '\n')
            f.write(r'\centering' + '\n')
            f.write(r'\caption{Trade Size Bins: Percent and Counts of Observations}' + '\n')
            f.write(r'\label{tab:trade_size_bins_obs}' + '\n')
            f.write(r'\small' + '\n')
            f.write(r'\begin{minipage}{0.95\linewidth}' + '\n')
            f.write(r'\vspace{0.35em}' + '\n')
            note_text = (
                r"\footnotesize{\textit{Note:} The table reports counts and percentages of trades by trade size (prtSize\_agg) bins. Exact sizes 1--5 are single-value bins. Ranged bins use left-inclusive, right-exclusive boundaries (e.g., 6--10 means $6 \leq$ size $< 10$) except the final open bin $\geq 200{,}000$. Fragmentation is the mean number of fragments (fragment\_count) for trades in each bin. Data filtered to include only equity options (ticker\_class == Equity) with prtType $\geq$ 73. Total observations: $N = "
                + f"{int(N):,}"
                + r"$.}"
            )
            f.write(note_text + '\n')
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


if __name__ == '__main__':
    parquet_filters = build_parquet_filters(
        ticker_type="all", 
        strategy_type="all",
        start=None,
        end=None,
    )
    build_table(parquet_filters, OUTPUT_DIR)
