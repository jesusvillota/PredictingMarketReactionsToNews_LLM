# uv run src/mains/save_parquet.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
from dask.delayed import delayed
from dask.base import compute as dask_compute

import zipfile
import tempfile

from src.pipeline.extract_date import extract_date_from_single_file
from src.config import config_settings, initialize_main, DaskManager
from src.config.logger import get_logger
from src.config.config_settings import *

# Configuration for processing mode
PROCESS_CORRUPTED_FILES: bool = True  # Set to True to reprocess corrupted files
RAW_PARQUET_PATH = RAW_PARQUET_PATH_CORRUPTED if PROCESS_CORRUPTED_FILES else RAW_PARQUET_PATH
RAW_PARQUET_PATH.mkdir(parents=True, exist_ok=True)

def log_failed_date(file_name: str):
    with open(RAW_PARQUET_PATH / "_complete_failure_dates_.txt", "a") as f:
        f.write(f"{file_name}\n")
        
def log_first_step_failure(file_name: str):
    with open(RAW_PARQUET_PATH / "_first_step_failure_dates_.txt", "a") as f:
        f.write(f"{file_name}\n")

def get_failure_summary() -> dict[str, int]:
    """
    Get a summary of different types of failures.
    
    Returns:
        Dictionary with counts of different failure types
    """
    summary = {
        "first_step_failures": 0,
        "complete_failures": 0
    }
    
    # Count first step failures
    first_step_file = RAW_PARQUET_PATH / "_first_step_failure_dates_.txt"
    if first_step_file.exists():
        with open(first_step_file, "r") as f:
            summary["first_step_failures"] = len(f.readlines())
    
    # Count complete failures
    complete_failure_file = RAW_PARQUET_PATH / "_complete_failure_dates_.txt"
    if complete_failure_file.exists():
        with open(complete_failure_file, "r") as f:
            summary["complete_failures"] = len(f.readlines())
    
    return summary

def filter_zip_files_by_year(zip_files: list[Path], target_years: list[int] | None = None) -> list[Path]:
    """
    Filter zip files to only include those from specified years.
    
    Args:
        zip_files: List of Path objects for zip files
        target_years: List of years to include (e.g., [2023, 2024]). 
                     If None, returns all files.
    
    Returns:
        List of Path objects for zip files from the specified years
    """
    
    logger = get_logger(__name__)  # Logger for this function
    
    if target_years is None:
        return zip_files
    
    filtered_files = []
    skipped_count = 0
    
    for zip_file in zip_files:
        date_obj, success = extract_date_from_single_file(zip_file.name)
        if success and date_obj is not None:
            if date_obj.year in target_years:
                filtered_files.append(zip_file)
            else:
                skipped_count += 1
        else:
            # If we can't extract the date, skip the file
            skipped_count += 1
    
    if logger:
        logger.debug(f"Filtered files: {len(filtered_files)} included, {skipped_count} skipped")
        if target_years:
            logger.debug(f"Target years: {sorted(target_years)}")
    
    return filtered_files

def process_zip_file(zip_file: Path) -> bool:
    """
    Process a single zip file, extracting and processing its text files.
    """
    file_name = zip_file.stem  # Initialize with fallback value
    try:
        logger.debug(f"Processing {zip_file.name}")
        
        # Extract date from filename
        date_obj, success = extract_date_from_single_file(zip_file.name)
        if not success or date_obj is None:
            logger.error(f"Could not extract date from filename: {zip_file.name}")
            file_name = zip_file.stem
        else:
            logger.debug(f"Extracted date {date_obj} from filename {zip_file.name}")
            file_name = date_obj.strftime('%Y-%m-%d')

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            txt_files = [
                name for name in zip_ref.namelist()
                if name.endswith('.txt') and not name.startswith('__MACOSX')
            ]
            if not txt_files:
                logger.error(f"No .txt files found in {zip_file.name}")
                log_failed_date(file_name)
                return False

            # Extract all txt files to a temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                extracted_paths = [
                    zip_ref.extract(txt_file, path=tmpdir) for txt_file in txt_files
                ]

                try:
                    # Try efficient single-step read first
                    logger.debug("Attempting efficient single-step read with dtypes and column selection")
                    ddf = dd.read_csv(
                        extracted_paths,
                        delimiter='\t',
                        usecols=ALL_COLUMNS,
                        dtype=DTYPES,
                        assume_missing=True,
                        na_values=['N/A', 'NA'],
                    )
                    logger.debug("Single-step read successful")
                    logger.debug(f"Loaded files with {ddf.npartitions} partitions")
                    
                except Exception as e:
                    # Fall back to robust two-step read
                    logger.warning(f"Single-step read failed: {e}. Falling back to two-step method...")
                    log_first_step_failure(file_name)  # Track first step failure
                    try:
                        # Step 1: Read all data as object type
                        logger.debug("Reading files with dtype=object")
                        ddf = dd.read_csv(
                            extracted_paths,
                            delimiter='\t',
                            dtype=object,
                            # assume_missing=True,
                            # na_values=['N/A', 'NA'],
                        )
                        logger.debug("Successfully read files with dtype=object")
                        logger.debug(f"Loaded files with {ddf.npartitions} partitions")
                        
                        # Step 2: Select columns and convert dtypes
                        if ALL_COLUMNS:
                            logger.info("Selecting specified columns")
                            ddf = ddf[ALL_COLUMNS]
                        
                        logger.debug("Converting dtypes from object to specified types")
                        ddf = ddf.astype(DTYPES)
                        logger.debug("Successfully converted dtypes using two-step method")
                        
                    except Exception as e2:
                        logger.error(f"Both single-step and two-step methods failed: {e2}", exc_info=True)
                        log_failed_date(file_name)
                        return False

                try: 

                    save_path: Path = RAW_PARQUET_PATH / f"{file_name}"
                    
                    ddf.to_parquet(
                        path=save_path,
                        compression=config_settings.parquet["compression"],
                        engine=config_settings.parquet["engine"],
                        # overwrite=True,
                    )
                    logger.debug(f"Saved processed data to {save_path}")
                    return True
                
                except Exception as e:
                    logger.error(f"Error in processing pipeline for {file_name}: {e}", exc_info=True)
                    log_failed_date(file_name)
                    return False
    
    except Exception as e:
        logger.error(f"Unexpected error processing {file_name}: {e}", exc_info=True)
        log_failed_date(file_name)
        return False


if __name__ == "__main__":

    logger = initialize_main()
    
    TARGET_YEARS: list[int] | None = None # [2023, 2024, 2025] # [2020, 2021, 2022] # [2017, 2018, 2019] # [2014, 2015, 2016] 
    
    if PROCESS_CORRUPTED_FILES:
        logger.info(f"Starting reprocessing of corrupted files. Output will be saved to: {RAW_PARQUET_PATH}")
        logger.info(f"Corrupted files to process: {[f.name for f in RAW_ZIP_PATH_CORRUPTED]}")
    else:
        logger.info(f"Starting normal processing with target years: {TARGET_YEARS if TARGET_YEARS else 'All years'}")
    
    with DaskManager() as dask_manager:

        if PROCESS_CORRUPTED_FILES:
            # Process the specific corrupted files
            _files: list[Path] = RAW_ZIP_PATH_CORRUPTED
            logger.info(f"Processing {len(_files)} corrupted files")
            
            # Check if the corrupted files actually exist
            existing_files = []
            for file_path in _files:
                if file_path.exists():
                    existing_files.append(file_path)
                else:
                    logger.warning(f"Corrupted file not found: {file_path}")
            
            _files = existing_files
            if not _files:
                logger.error("No corrupted files found to process")
                exit(1)
                
        else:
            # Normal processing mode
            if not RAW_ZIP_PATH.exists():
                logger.error(f"Path to zip files does not exist: {RAW_ZIP_PATH}")
                exit(1)
                
            _files: list[Path] = sorted([f for f in RAW_ZIP_PATH.iterdir() if f.suffix == ".zip"])
            
            # Filter files by target years if specified
            if TARGET_YEARS is not None:
                _files = filter_zip_files_by_year(_files, TARGET_YEARS)
                logger.info(f"Filtered to {len(_files)} files for years: {TARGET_YEARS}")
        
        if TEST:
            _files: list[Path] = _files[:3]
        
        # raw_zip_files: list[str] = [str(file) for file in _files]
        raw_zip_files: list[Path] = _files  # Keep the Path objects for delayed tasks

        if len(raw_zip_files) == 0:
            if PROCESS_CORRUPTED_FILES:
                logger.error("No corrupted zip files found to process")
            else:
                logger.error(f"No zip files found in {RAW_ZIP_PATH}")
            exit(1)

        if PROCESS_CORRUPTED_FILES:
            logger.info(f"Found {len(raw_zip_files)} corrupted files to reprocess")
        else:
            logger.info(f"Found {len(raw_zip_files)} zip files to process")

        delayed_tasks = [delayed(process_zip_file)(zip_file) for zip_file in raw_zip_files]

        # Execute tasks in parallel
        results = dask_compute(*delayed_tasks, scheduler='distributed')
                
        # Count successful vs failed processing
        successful_count = sum(1 for result in results if result)
        failed_count = len(results) - successful_count
        logger.info(f"Processing completed: {successful_count} successful, {failed_count} failed")
        
        # Get detailed failure summary
        failure_summary = get_failure_summary()
        logger.info(f"Failure breakdown:")
        logger.info(f"  - First step failures (but recovered): {failure_summary['first_step_failures']}")
        logger.info(f"  - Complete failures: {failure_summary['complete_failures']}")
        logger.info(f"  - Files that failed first step but succeeded overall: {failure_summary['first_step_failures'] - failure_summary['complete_failures']}")
        
        # Log file locations for review
        logger.info(f"Failure logs saved to:")
        logger.info(f"  - First step failures: {RAW_PARQUET_PATH / '_first_step_failure_dates_.txt'}")
        logger.info(f"  - Complete failures: {RAW_PARQUET_PATH / '_complete_failure_dates_.txt'}")