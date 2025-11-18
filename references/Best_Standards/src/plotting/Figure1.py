# uv run src/plotting/Figure1.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import dask.dataframe as dd
from src.config import config_settings, initialize_main, DaskManager
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set output path relative to project root
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../TeX/figures'))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'Figure1.pdf')

from src.plotting.config import setup_matplotlib_config
setup_matplotlib_config()

if __name__ == '__main__':

    logger = initialize_main()

    logger.info("Starting Figure1.py script.")
    try:
        logger.info("Loading parquet data with Dask...")
        # Define whale threshold (99th percentile ~= 270 contracts)
        WHALE_THRESHOLD = 270

        # Try to push the filter down to the parquet reader. Some engines (pyarrow) support
        # predicate pushdown via the `filters` argument. If that is unsupported it will
        # simply be ignored and we'll filter after compute as a fallback.
        try:
            ddf = dd.read_parquet(
                path=config_settings.PATHS,
                engine=config_settings.parquet["engine"],
                columns=['timestamp_ny', 'prtSize_agg'],
                split_row_groups='infer',
                filters=[('prtSize_agg', '>=', WHALE_THRESHOLD)],
            )
        except Exception as e:
            logger.warning(f"Predicate pushdown not supported or failed: {e}")
            # Fallback: read all and filter after compute
            ddf = dd.read_parquet(
                path=config_settings.PATHS,
                engine=config_settings.parquet["engine"],
                columns=['timestamp_ny', 'prtSize_agg'],
                split_row_groups='infer',
            )
            ddf = ddf[ddf['prtSize_agg'] >= WHALE_THRESHOLD]

        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
    except Exception as e:
        logger.exception(f"Error loading parquet: {e}")
        # If we cannot load the data, stop execution to avoid using an unbound `ddf`.
        raise
    # Prepare and plot whales
    try:
        logger.info(f"Filtering for whales where prtSize_agg >= {WHALE_THRESHOLD} and converting to pandas...")
        # Compute to pandas; whale observations are rare (top 1%), so this should be manageable in memory
        df = ddf.compute()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp_ny']):
            df['timestamp_ny'] = pd.to_datetime(df['timestamp_ny'])

        if df.empty:
            logger.warning("No whale observations found after filtering. No plot will be produced.")
        else:
            # Keep full datetime for plotting but show ticks at the date/year level
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.scatter(df['timestamp_ny'], df['prtSize_agg'], s=10, alpha=0.6)
            ax.set_xlabel('Date')
            ax.set_ylabel('Trade Size')
            ax.set_title(f'Whales over time (Trade Size $\\geq$ {WHALE_THRESHOLD})')

            # Format x-axis: use yearly ticks for long timelines and rotate labels
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            try:
                logger.info(f"Saving figure to {OUTPUT_PATH}")
                fig.savefig(OUTPUT_PATH, dpi=600)
            except Exception as e:
                logger.warning(f"Could not save tex output ({OUTPUT_PATH}): {e}")                

            plt.close(fig)
            logger.info("Figure1 generation completed successfully.")
            
    except Exception as e:
        logger.exception(f"Error while preparing or plotting whales: {e}")


