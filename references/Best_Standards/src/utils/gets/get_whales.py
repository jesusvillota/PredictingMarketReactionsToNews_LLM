# src/data/get_equities_afterhours.py
from src.config.logger import get_logger
import dask.dataframe as dd
from pathlib import Path
from src.config import config_settings
import pandas as pd

def get_whales(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Filter the DataFrame to get only whales (large transactions).
    """
    logger = get_logger(__name__)
    logger.debug("Filtering for whales 🐋 ...")

    try:
        if config_settings.whales["quantile_thresholding"]:
            quantile_threshold = config_settings.whales["quantile_threshold"]
            if config_settings.whales["approximate_quantiles"]:
                size_threshold = ddf["prtSize_agg"].quantile(quantile_threshold, approximate=True).compute()
            else:
                size_threshold = ddf["prtSize_agg"].quantile(quantile_threshold).compute()
        else:
            size_threshold = config_settings.whales["size_threshold"]

        # Filter the whales 🐋 
        ddf_whales = ddf[ddf["prtSize_agg"] >= size_threshold]

        logger.debug("Successfully filtered whales 🐋.")
        return ddf_whales

    except Exception as e:
        logger.error(f"Error filtering whales 🐋 : {e}", exc_info=True)
        return dd.from_pandas(pd.DataFrame(), npartitions=1)
    
def save_whales(ddf: dd.DataFrame, date_str: str) -> bool:
    """
    Save the whales DataFrame to a parquet file.
    """    
    logger = get_logger(__name__)
    
    ddf_whales = get_whales(ddf).persist()
    logger.debug(f"Saving whales 🐋 for {date_str}...")
    
    try:
        # Define output path
        output_path: Path = config_settings.data_paths["whales_path_daily_files_path"]
        output_path.mkdir(parents=True, exist_ok=True)
        parquet_file = output_path / f"{date_str}"

        # Save to parquet
        ddf_whales.to_parquet(
            path=parquet_file,
            # write_metadata_file=config_settings.parquet["write_metadata_file"],
            # write_index=config_settings.parquet["write_index"],
            compression=config_settings.parquet["compression"],
            engine=config_settings.parquet["engine"],
        )

        logger.debug(f"Successfully saved whales 🐋 to {parquet_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving whales 🐋 for {date_str}: {e}", exc_info=True)
        return False