from src.config.logger import get_logger
import dask.dataframe as dd
from .create_new_vars import create_new_vars_1
from .preprocessing_filters import apply_filters
from .classify import classify_trades
from .group_fragmented_trades import group_fragmented_trades
from .compute_deltas import compute_deltas
from .final_subset import final_subset

#------------- On quarantine -------------#
# from src.gets.get_delta_exposure import get_delta_exposure_new
# from src.gets.get_whales import get_whales, save_whales
#-----------------------------------------#

def data_pipeline(ddf, file_name: str) -> dd.DataFrame:
    
    logger = get_logger(__name__)  # Logger for this function

    # Create new variables
    ddf = create_new_vars_1(ddf).persist()
    logger.info(f"Created new variables for {file_name}")

    # Apply filters
    ddf = apply_filters(ddf).persist()
    logger.info(f"Applied filters to {file_name}")

    # Classify trades
    ddf = classify_trades(ddf).persist()
    logger.info(f"Classified trades in {file_name}")

    # Detect fragmented trades
    ddf = group_fragmented_trades(ddf).persist()
    logger.info(f"Detected fragmented trades for {file_name}")

    # # Compute delta exposure and save internally
    # ddf = compute_deltas(ddf).persist()
    # logger.info(f"Computed deltas for {file_name}")
    
    #------------- On quarantine -------------#
    # # Compute delta exposure and save internally
    # get_delta_exposure_new(ddf, file_name)  # This saves a parquet file internally
    # logger.info(f"Computed delta exposure for {file_name}")
    
    # # Filter whales and save internally
    # save_whales(ddf, file_name)
    # logger.info(f"Filtered and saved whales for {file_name}")
    #-----------------------------------------#
    
    # Final subset of columns
    ddf = final_subset(ddf).persist()
    logger.info(f"Selected final subset of columns for {file_name}")
    
    return ddf
