# src/data/get_equities_afterhours.py
from src.config.logger import get_logger
import dask.dataframe as dd
from pathlib import Path
from src.config import config_settings


def temp_reclassify_trading_hours(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    """
    # 1) Delte the existing 'trading_hours_class' column
    if 'trading_hours_class' in ddf.columns:
        ddf = ddf.drop(columns=['trading_hours_class'])
    
    # 2) Recreate the 'trading_hours_class' column with the new logic
    
    def classify_partition(df):
        import numpy as np
        df['trading_hours_class'] = 'Unknown'
        hour = df['timestamp_ny'].dt.hour
        minute = df['timestamp_ny'].dt.minute
        before_market = (hour < 9) | ((hour == 9) & (minute < 30))
        market_hours = ((hour > 9) | ((hour == 9) & (minute >= 30))) & (hour < 16)
        after_market = (hour > 16) | ((hour == 16) & (minute >= 0))
        trading_conditions = [before_market, market_hours, after_market]
        trading_choices = ['BeforeMarket', 'MarketHours', 'AfterMarket']
        df['trading_hours_class'] = np.select(condlist=trading_conditions, 
                                                choicelist=trading_choices, 
                                                default='Unknown')
        return df
    
    meta = ddf._meta.copy()
    meta['trading_hours_class'] = 'Unknown'
    ddf = ddf.map_partitions(classify_partition, meta=meta)
    return ddf
    

def get_equities_afterhours(ddf: dd.DataFrame) -> tuple[dd.DataFrame, dd.DataFrame]:
    """
    Filter the DataFrame to include only equity trades that occurred outside regular market hours.
    """
    logger = get_logger(__name__)
    logger.debug("Starting after-hours equity filtering")
    
    try:
        # Apply the optimal filtering strategy (time first, then equity)
        # Extract hour and minute from timestamp
        hour: dd.Series = ddf['timestamp_ny'].dt.hour
        minute: dd.Series = ddf['timestamp_ny'].dt.minute

        # Equity filter
        equity_mask: dd.Series = (ddf['ticker_class'] == 'Equity')
        equity_ddf: dd.DataFrame = ddf.loc[equity_mask]

        # Before market filter
        before_mask: dd.Series = (hour < 9) | ((hour == 9) & (minute < 30))
        equity_before_ddf: dd.DataFrame = equity_ddf.loc[before_mask]

        # After market filter
        after_mask: dd.Series = (hour > 16) | ((hour == 16) & (minute > 0))
        equity_after_ddf: dd.DataFrame = equity_ddf.loc[after_mask]

        logger.debug("Filtering completed for: (Before market equity) & (After market equity)")
        return equity_before_ddf, equity_after_ddf

    except Exception as e:
        logger.error(f"Error during after-hours equity filtering: {e}", exc_info=True)
        return dd.from_pandas(pd.DataFrame(), npartitions=1)  # Return empty DataFrame on error
    
def save_equities_afterhours(equities_afterhours_ddf: dd.DataFrame, date_str: str) -> bool:
    """
    Save the after-hours equities DataFrame to a parquet file.
    Returns True if successful, False otherwise.
    """
    logger = get_logger(__name__)
    logger.debug(f"Saving after-hours equities for {date_str}")
    
    try:
        # if len(equities_afterhours_ddf) == 0:
        #     logger.info(f"No after-hours equity trades found for {date_str}. No file will be saved.")
        #     return True  # Not an error, just no data to save

        # Define output path
        output_path: Path = config_settings.data_paths["equities_afterhours_path"]
        output_path.mkdir(parents=True, exist_ok=True)
        parquet_file = output_path / f"{date_str}"

        # Save to parquet
        equities_afterhours_ddf.to_parquet(
            path=parquet_file,
            # write_metadata_file=config_settings.parquet["write_metadata_file"],
            # write_index=config_settings.parquet["write_index"],
            compression=config_settings.parquet["compression"],
            engine=config_settings.parquet["engine"],
        )
        
        logger.debug(f"Successfully saved after-hours equities to {parquet_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving after-hours equities for {date_str}: {e}", exc_info=True)
        return False