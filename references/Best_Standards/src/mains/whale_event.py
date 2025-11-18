# uv run src/mains/simple_attention_sentiment.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
from typing import Any

from src.config import config_settings, initialize_main, AdaptiveDaskManager, DaskManager
from src.config.config_settings import PATHS

if __name__ == '__main__':

    logger = initialize_main()
    
    # Output directory
    # output_dir = PROJECT_ROOT / "_ATTENTION_SENTIMENT_DIRECTION_" / "Simple"
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading Parquet files from {PATHS}")
    logger.info(f"Output directory: {output_dir}")
    
    with DaskManager() as dask_manager: 
    # with AdaptiveDaskManager() as dask_manager:        
        # Load data with proper filters
        ddf = dd.read_parquet(
            path=PATHS,
            engine=config_settings.parquet["engine"],
            filters=[
                ('okey_xx')
                ('ticker_class', '==', 'Equity'),
                ('prtType', '>=', 73),
                ('prtType', '<', 102),
            ],
            columns=[
                'okey_tk', 
                'timestamp_ny', 
                'prtSize_agg',
                'buy_sell_class',
            ],
        )
        
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        