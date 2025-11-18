# src/trades_new/preprocessing_filters.py
import dask.dataframe as dd
from src import get_logger

def apply_filters(ddf: dd.DataFrame) -> dd.DataFrame:
    logger = get_logger(__name__)
    logger.debug("Applying preprocessing filters to trades data")
    initial_partitions = ddf.npartitions
    
    # Chain all filters in one go   
    ddf = ddf[
        (ddf['prtPrice'] > 0)
        & (ddf['prtSize'] > 0)
        & (ddf['uBid'] > 0.1)
        & (ddf['spread'] >= 0)
        & (ddf['BIDabove'] >= 0)
        & (ddf['ASKbelow'] <= 0)
        # & (ddf['tradingSession'] == 'RegularMkt')
    ]
    
    logger.debug(f"Applied filters to trades data. Partitions: {initial_partitions} -> {ddf.npartitions}")
    return ddf