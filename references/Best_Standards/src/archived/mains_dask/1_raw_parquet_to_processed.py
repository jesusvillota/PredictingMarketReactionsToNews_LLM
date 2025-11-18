# uv run src/mains/1_raw_parquet_to_processed.py

from pathlib import Path
import sys
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
import numpy as np
from src.config import config_settings, initialize_main, DaskManager
from dask.delayed import delayed
from dask.base import compute
from src.config.logger import get_logger
from src.config.config_settings import PROCESSED_PATH, RAW_PARQUET_PATH

def log_failed_date(file_name: str):
    with open(PROCESSED_PATH / "_processing_failure_dates_.txt", "a") as f:
        f.write(f"{file_name}\n")

def pipeline(parquet_folder: Path) -> dd.DataFrame:

    logger = get_logger()
    
    try:
        # READ PARQUET FOLDER
        try:
            logger.info(f"Loading parquet data for {parquet_folder.name}...")
            ddf = dd.read_parquet(
                path=parquet_folder,  # Read the entire folder
                engine=config_settings.parquet["engine"],

            )
            logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
        except Exception as e:
            logger.exception(f"Error loading parquet: {e}")
            log_failed_date(parquet_folder.name)
            raise
            
        # CREATE NEW VARIABLES 1
        try: 
            from src.pipeline.create_new_vars import create_new_vars_1
            ddf = create_new_vars_1(ddf).persist()
            logger.info("create_new_vars_1 function applied to Dask DataFrame.")
        except Exception as e:
            logger.exception(f"Error applying create_new_vars_1 function: {e}")
            log_failed_date(parquet_folder.name)
        
        # APPLY FILTERS
        try: 
            from src.pipeline.preprocessing_filters import apply_filters
            ddf = apply_filters(ddf).persist()
            logger.info("Filtering functions applied to Dask DataFrame.")
        except Exception as e:
            logger.exception(f"Error applying filtering functions: {e}")
            log_failed_date(parquet_folder.name)
        
        # CLASSIFY TRADES
        try:
            from src.pipeline.classify import classify_trades
            ddf = classify_trades(ddf).persist()
            logger.info("Classification functions applied to Dask DataFrame.")
        except Exception as e:
            logger.exception(f"Error applying classification functions: {e}")
            log_failed_date(parquet_folder.name)
        
        # GROUP FRAGMENTED TRADES
        try:
            from src.pipeline.group_fragmented_trades import group_fragmented_trades
            ddf = group_fragmented_trades(ddf).persist()
            logger.info("Grouping fragmented trades together applied to Dask DataFrame")
        except Exception as e:
            logger.exception(f"Error applying fragment detection function: {e}")
            log_failed_date(parquet_folder.name)
        
        # CREATE NEW VARIABLES 2
        try: 
            from src.pipeline.create_new_vars import create_new_vars_2
            ddf = create_new_vars_2(ddf).persist()
            logger.info("create_new_vars_2 function applied to Dask DataFrame.")
        except Exception as e:
            logger.exception(f"Error applying create_new_vars_2 function: {e}") 
            log_failed_date(parquet_folder.name)
        
        # SAVE PARQUET - Create output folder with same day name
        day_folder_name = parquet_folder.name  # e.g., "2021-01-29"
        output_folder = PROCESSED_PATH / day_folder_name
        output_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            ddf.to_parquet(
                path=output_folder,  # Save to day-specific folder
                compression=config_settings.parquet["compression"],
                engine=config_settings.parquet["engine"],
            )
            logger.info(f"Saved processed data to {output_folder}.")
            
        except Exception as e:
            logger.exception(f"Error saving reprocessed data: {e}")
            log_failed_date(parquet_folder.name)
            raise
        
        return ddf
    
    except Exception as e:
        logger.error(f"Pipeline failed for {parquet_folder.name}: {e}")
        log_failed_date(parquet_folder.name)
        raise


if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting reprocess.py script.")

    # ------------------------------------------------------------------------------------------------------------
    TARGET_YEARS: list[int] | None = [2020] # [2019] # [2015, 2016, 2017, 2018] # [2014]
    logger.info(f"Starting processing with target years: {TARGET_YEARS if TARGET_YEARS else 'All years'}")

    REPROCESS: bool = True  # If True, reprocess only missing or corrupted dates
    logger.info(f"REPROCESS mode is {'ON' if REPROCESS else 'OFF'}")
    # ------------------------------------------------------------------------------------------------------------

    # All raw parquet folders (each represents a day)
    raw_parquet_folders = sorted([f for f in RAW_PARQUET_PATH.iterdir() if f.is_dir()])
    logger.info(f"Found {len(raw_parquet_folders)} raw parquet folders in {RAW_PARQUET_PATH}")

    # Filter by target years if specified
    if TARGET_YEARS is not None:
        raw_parquet_folders = [f for f in raw_parquet_folders if int(f.name.split('-')[0]) in TARGET_YEARS]
        logger.info(f"Filtered to {len(raw_parquet_folders)} folders for years: {TARGET_YEARS}")

    # Reprocess logic
    if REPROCESS:
        raw_parquet_dates = [f.name for f in raw_parquet_folders]
        processed_dates = sorted([f.name for f in PROCESSED_PATH.iterdir() if f.is_dir()])
        logger.info(f"Found {len(processed_dates)} processed folders in {PROCESSED_PATH}")

        # Check if failure log file exists
        failure_log_file = PROCESSED_PATH / "_processing_failure_dates_.txt"
        failed_processed_dates = []
        if failure_log_file.exists():
            with open(failure_log_file, "r") as f:
                failed_processed_dates = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"Read {len(failed_processed_dates)} failed dates from log")
        else:
            logger.info("No failure log file found")

        if TARGET_YEARS is not None:
            failed_processed_dates = [date for date in failed_processed_dates if int(date.split('-')[0]) in TARGET_YEARS]
            logger.info(f"Filtered to {len(failed_processed_dates)} failed dates for years: {TARGET_YEARS}")
            
            processed_dates = [date for date in processed_dates if int(date.split('-')[0]) in TARGET_YEARS]
            logger.info(f"Filtered to {len(processed_dates)} processed folders for years: {TARGET_YEARS}")
        
        process_dates = (set(raw_parquet_dates) - set(processed_dates)).union(set(failed_processed_dates))
        logger.info(f"Total folders to process after reprocessing logic: {len(process_dates)}")
        
        # Delete existing processed folders that will be reprocessed
        folders_to_delete = []
        for date in process_dates:
            processed_folder = PROCESSED_PATH / date
            if processed_folder.exists() and processed_folder.is_dir():
                folders_to_delete.append(processed_folder)
        
        if folders_to_delete:
            logger.info(f"Deleting {len(folders_to_delete)} existing processed folders that will be reprocessed...")
            for folder in folders_to_delete:
                try:
                    shutil.rmtree(folder)
                    logger.info(f"Deleted existing processed folder: {folder.name}")
                except Exception as e:
                    logger.error(f"Failed to delete folder {folder.name}: {e}")
        else:
            logger.info("No existing processed folders to delete")
        
        process_folders = [RAW_PARQUET_PATH / date for date in process_dates]
        logger.info(f"Prepared {len(process_folders)} folders to process")
    else:
        process_folders = raw_parquet_folders
        logger.info(f"Processing all {len(process_folders)} raw parquet folders")
            
    if not process_folders:
        logger.error(f"No parquet folders found to process for the specified years")
        exit(1)
    logger.info(f"Found {len(process_folders)} parquet folders to process")
    
    # """
    with DaskManager() as dask_manager:

        # DELAYED TASKS
        logger.info("Creating delayed tasks for each parquet folder...")
        delayed_tasks = [delayed(pipeline)(folder) for folder in process_folders]

        # COMPUTE ALL TASKS IN PARALLEL
        logger.info("Computing delayed tasks in parallel...")
        results = compute(*delayed_tasks, scheduler='distributed')
        
        # Count successful vs failed processing
        successful_count = sum(1 for result in results if result is not None)
        failed_count = len(results) - successful_count
        logger.info(f"Processing completed: {successful_count} successful, {failed_count} failed")
        
        logger.info("All parquet folders have been processed.")
    # """