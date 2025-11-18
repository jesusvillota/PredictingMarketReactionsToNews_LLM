# uv run src/debugging/check_missing_prtSize_agg.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
from logging import Logger
import dask.dataframe as dd

from src.config import config_settings, initialize_main
from THIS_IS import PROCESSED_PATH, OUTPUT_PATH


def check_date_for_missing_values(date_folder: Path, logger: Logger) -> bool:
    """
    Check if a date's parquet files contain missing values in target columns.
    
    Args:
        date_folder: Path to the date folder containing parquet files
        logger: Logger instance for logging
    
    Returns:
        True if missing values are found in prtSize_agg or fragment_count, False otherwise
    """
    try:
        # Load all parquet files from the date folder
        parquet_pattern = str(date_folder / "*.parquet")
        
        logger.debug(f"Loading parquet files from {date_folder.name}")
        ddf = dd.read_parquet(parquet_pattern, engine='pyarrow')
        
        # Check if the required columns exist
        if 'prtSize_agg' not in ddf.columns or 'fragment_count' not in ddf.columns:
            logger.warning(f"Date {date_folder.name}: Required columns not found")
            return False
        
        # Check for missing values in both columns
        # We need to compute to get actual boolean values
        has_missing_prtSize = ddf['prtSize_agg'].isnull().any().compute()
        has_missing_fragment = ddf['fragment_count'].isnull().any().compute()
        
        if has_missing_prtSize or has_missing_fragment:
            logger.info(f"Date {date_folder.name}: MISSING VALUES FOUND - "
                       f"prtSize_agg: {has_missing_prtSize}, fragment_count: {has_missing_fragment}")
            return True
        else:
            logger.debug(f"Date {date_folder.name}: No missing values")
            return False
            
    except Exception as e:
        logger.error(f"Error checking date {date_folder.name}: {e}")
        return False


if __name__ == '__main__':
    logger = initialize_main()
    logger.info("Starting missing values check script.")
    
    # Get all date folders from PROCESSED_PATH
    if not PROCESSED_PATH.exists():
        logger.error(f"PROCESSED_PATH does not exist: {PROCESSED_PATH}")
        exit(1)
    
    daily_folders = sorted([dir for dir in PROCESSED_PATH.iterdir() if dir.is_dir()])
    logger.info(f"Found {len(daily_folders)} date folders in {PROCESSED_PATH}")
    
    if not daily_folders:
        logger.error("No date folders found to check")
        exit(1)
    
    logger.info(f"First date: {daily_folders[0].name}")
    logger.info(f"Last date: {daily_folders[-1].name}")
    
    # Check each date for missing values
    dates_with_missing = []
    
    logger.info("Starting to check each date for missing values...")
    for i, date_folder in enumerate(daily_folders, 1):
        logger.info(f"Processing {i}/{len(daily_folders)}: {date_folder.name}")
        
        has_missing = check_date_for_missing_values(date_folder, logger)
        
        if has_missing:
            dates_with_missing.append(date_folder.name)
    
    # Write results to output file
    output_file = OUTPUT_PATH / "_dates_with_missing_prtSize_agg_.txt"
    
    logger.info(f"Writing results to {output_file}")
    with open(output_file, 'w') as f:
        for date in dates_with_missing:
            f.write(f"{date}\n")
    
    # Log summary statistics
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total dates checked: {len(daily_folders)}")
    logger.info(f"Dates with missing values: {len(dates_with_missing)}")
    logger.info(f"Dates without missing values: {len(daily_folders) - len(dates_with_missing)}")
    
    if dates_with_missing:
        logger.info(f"\nDates with missing values:")
        for date in dates_with_missing:
            logger.info(f"  - {date}")
    else:
        logger.info("\nNo dates with missing values found!")
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info("Missing values check completed successfully.")

