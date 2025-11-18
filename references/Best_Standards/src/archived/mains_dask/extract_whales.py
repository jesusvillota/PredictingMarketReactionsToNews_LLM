# uv run src/mains/extract_whales.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
from typing import Any

from src.config import config_settings, initialize_main, DaskManager
from src.config.config_settings import PROCESSED_PATH, DISK
from src.config.utils import DailyFolderFilter

if __name__ == '__main__':

    logger = initialize_main()
    
    # Output directory
    output_dir = DISK / "WHALES"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading Parquet files from {PROCESSED_PATH}")
    logger.info(f"Output directory: {output_dir}")
    
    filter = DailyFolderFilter(PROCESSED_PATH)
    # globs = filter.by_year([2021])
    globs = filter.by_year_month([
        "2019-01", 
        "2019-02", 
        "2019-03",
        "2019-04",
        ])
    
    with DaskManager() as dask_manager:         
        # Load data with proper filters
        ddf = dd.read_parquet(
            path=globs,
            engine=config_settings.parquet["engine"],
            filters=[
                ('prtSize_agg', '>=', 200),
                # ('prtSize_agg', '>=', 100000),
                ('ticker_class', '==', 'Equity'),
                ('prtType', '>=', 73),
                ('prtType', '<', 102),
            ],
            columns=[
                'okey_tk', 
                'okey_xx', 
                'okey_cp',
                'expiration',
                # 'okey_yr',
                # 'okey_mn',
                # 'okey_dy',
                'timestamp_ny',
                'prtPrice',
                'prtSize_agg',
                'prtType',
                'buy_sell_class',
                'moneyness_class_ratio',
                'fragment_count',
                'pnlM1',
                'pnlM10',
            ],
        )
        
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        ddf['expdate'] = ddf['expiration'].dt.strftime('%Y-%m-%d')
        ddf = ddf.drop(columns=['expiration'])
        ddf = ddf.repartition(partition_size="100MB")
        
        ddf.to_parquet(
            path=output_dir,
            compression=config_settings.parquet["compression"],
            engine=config_settings.parquet["engine"],
            # index=False,
        )
        logger.info(f"Saved to {output_dir}")

