# uv run src/mains_duckdb/1_raw_parquet_to_processed_duckdb.py
# uv run src/mains_duckdb/1_raw_parquet_to_processed_duckdb.py --batched --batch-size 1228800 --output-mode partitioned
# uv run src/mains_duckdb/1_raw_parquet_to_processed_duckdb.py --batched --batch-size 1228800 --output-mode merged

"""
DuckDB-based pipeline to process raw parquet trade data.
Memory-efficient version that achieves the same output as the Dask pipeline.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
import shutil
import argparse
import gc
import psutil

from src.config import config_settings, initialize_main
from src.config.config_settings import PROCESSED_PATH, RAW_PARQUET_PATH, TEMP_DIR
from src.config.logger import get_logger
from src.config.duckdb_manager import DuckDBManager

# ------------------------------------------------------------------------------------------------------------
TARGET_YEARS: list[int] | None = [2021]  # [2019] # [2015, 2016, 2017, 2018] # [2014]
REPROCESS: bool = False  # If True, reprocess only missing or corrupted dates
# ------------------------------------------------------------------------------------------------------------
# PROCESSED_PATH: Path = DISK / "output" / "data" / "processed" / "_1_PROCESSED_TRADE_DATA_PARQUET_DUCKDB_5"

# Import pipeline functions
from src.pipeline_duckdb.create_new_vars import (
    get_create_vars_1_query, 
    get_create_vars_1_step2_query,
    get_create_vars_2_query, 
    get_create_vars_2_step2_query
)
from src.pipeline_duckdb.preprocessing_filters import get_filters_query
from src.pipeline_duckdb.classify import get_classify_query
from src.pipeline_duckdb.group_fragmented_trades import get_group_fragmented_query


def log_failed_date(file_name: str):
    """Log a failed processing date to file"""
    with open(PROCESSED_PATH / "_processing_failure_dates_.txt", "a") as f:
        f.write(f"{file_name}\n")


def log_memory_usage(logger):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")


def write_incremental_results(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    output_dir: Path, 
    batch_num: int,
    output_mode: str = 'partitioned'
) -> bool:
    """Write results incrementally using DuckDB's native COPY TO (much faster). Returns True if file was created."""
    
    # Check if view has data
    count = con.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
    if count == 0:
        return False
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Choose filename based on output mode
    if output_mode == 'partitioned':
        output_file = output_dir / f"partition_{batch_num:03d}.parquet"
    else:  # merged mode
        output_file = output_dir / f"temp_batch_{batch_num:03d}.parquet"
    
    # Fast DuckDB native write - no pandas conversion needed!
    con.execute(f"""
        COPY {view_name} TO '{output_file}' 
        (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 122880)
    """)
    
    return True


def combine_temp_files(con: duckdb.DuckDBPyConnection, output_path: Path, temp_files: list[Path], logger):
    """Combine temporary parquet files into final output using DuckDB (much faster)"""
    if not temp_files:
        return
    
    # Filter to only existing files
    existing_files = [temp_file for temp_file in temp_files if temp_file.exists()]
    
    if not existing_files:
        logger.warning("No temporary files found to combine")
        return
    
    logger.info(f"Combining {len(existing_files)} temporary files...")
    
    # Create a glob pattern for all temp files
    file_pattern = "', '".join(str(f) for f in existing_files)
    
    # Use DuckDB to read and combine all files, then write directly
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet(['{file_pattern}'])
        ) TO '{output_path}' 
        (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 122880)
    """)
    
    # Get record count
    record_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{output_path}')").fetchone()[0]
    
    # Clean up temp files
    for temp_file in existing_files:
        temp_file.unlink(missing_ok=True)
    
    logger.info(f"Successfully combined {record_count:,} records into final output")


def process_day_non_batched(
    con: duckdb.DuckDBPyConnection,
    daily_folder: Path,
    output_folder: Path,
    logger
):
    """Process a single day's data without batching"""
    day_str = daily_folder.name
    
    try:
        # Step 1: Load raw data
        logger.info(f"{day_str}: Loading raw parquet data...")
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_data AS 
            SELECT * FROM read_parquet('{daily_folder}/**/*.parquet')
        """)
        
        # Check if we have data
        count = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        if count == 0:
            logger.warning(f"{day_str}: No data found, skipping")
            return
        logger.info(f"{day_str}: Loaded {count:,} rows")
        
        # Step 2: Create new variables (step 1)
        logger.info(f"{day_str}: Creating derived variables (step 1)...")
        con.execute(get_create_vars_1_query("raw_data", "vars_1_data"))
        con.execute(get_create_vars_1_step2_query("vars_1_data", "vars_1_complete"))
        
        # Step 3: Apply filters
        logger.info(f"{day_str}: Applying preprocessing filters...")
        con.execute(get_filters_query("vars_1_complete", "filtered_data"))
        filtered_count = con.execute("SELECT COUNT(*) FROM filtered_data").fetchone()[0]
        logger.info(f"{day_str}: {filtered_count:,} rows after filtering ({filtered_count/count*100:.1f}%)")
        
        # Step 4: Classify trades
        logger.info(f"{day_str}: Classifying trades...")
        con.execute(get_classify_query("filtered_data", "classified_data"))
        
        # Step 5: Group fragmented trades
        logger.info(f"{day_str}: Grouping fragmented trades...")
        con.execute(get_group_fragmented_query("classified_data", "grouped_data"))
        grouped_count = con.execute("SELECT COUNT(*) FROM grouped_data").fetchone()[0]
        logger.info(f"{day_str}: {grouped_count:,} unique trades after grouping")
        
        # Step 6: Create new variables (step 2)
        logger.info(f"{day_str}: Creating additional derived variables (step 2)...")
        con.execute(get_create_vars_2_query("grouped_data", "vars_2_data"))
        con.execute(get_create_vars_2_step2_query("vars_2_data", "final_data"))
        
        # Step 7: Write output
        logger.info(f"{day_str}: Writing output to {output_folder}...")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        output_file = output_folder / "data.parquet"
        
        # Use DuckDB's native COPY TO - much faster!
        con.execute(f"""
            COPY final_data TO '{output_file}' 
            (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 122880)
        """)
        
        final_count = con.execute("SELECT COUNT(*) FROM final_data").fetchone()[0]
        logger.info(f"{day_str}: Successfully saved {final_count:,} rows to {output_file}")
        
        # Clean up
        gc.collect()
        
    except Exception as e:
        logger.error(f"{day_str}: Error processing data: {e}", exc_info=True)
        log_failed_date(day_str)
        raise


def process_day_batched(
    con: duckdb.DuckDBPyConnection,
    daily_folder: Path,
    output_folder: Path,
    batch_size: int,
    output_mode: str,
    logger
):
    """Process a single day's data in batches"""
    day_str = daily_folder.name
    
    try:
        # Step 1: Load raw data
        logger.info(f"{day_str}: Loading raw parquet data...")
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_data AS 
            SELECT * FROM read_parquet('{daily_folder}/**/*.parquet')
        """)
        
        # Check if we have data
        count = con.execute("SELECT COUNT(*) FROM raw_data").fetchone()[0]
        if count == 0:
            logger.warning(f"{day_str}: No data found, skipping")
            return
        logger.info(f"{day_str}: Loaded {count:,} rows")
        
        # Step 2-6: Apply all transformations to create final_data view
        logger.info(f"{day_str}: Applying pipeline transformations...")
        
        con.execute(get_create_vars_1_query("raw_data", "vars_1_data"))
        con.execute(get_create_vars_1_step2_query("vars_1_data", "vars_1_complete"))
        
        con.execute(get_filters_query("vars_1_complete", "filtered_data"))
        filtered_count = con.execute("SELECT COUNT(*) FROM filtered_data").fetchone()[0]
        logger.info(f"{day_str}: {filtered_count:,} rows after filtering")
        
        con.execute(get_classify_query("filtered_data", "classified_data"))
        con.execute(get_group_fragmented_query("classified_data", "grouped_data"))
        
        grouped_count = con.execute("SELECT COUNT(*) FROM grouped_data").fetchone()[0]
        logger.info(f"{day_str}: {grouped_count:,} unique trades after grouping")
        
        con.execute(get_create_vars_2_query("grouped_data", "vars_2_data"))
        con.execute(get_create_vars_2_step2_query("vars_2_data", "final_data"))
        
        # Step 7: Write output in batches
        logger.info(f"{day_str}: Writing output in batches of {batch_size:,} (mode: {output_mode})...")
        output_folder.mkdir(parents=True, exist_ok=True)
        
        total_rows = con.execute("SELECT COUNT(*) FROM final_data").fetchone()[0]
        total_batches = (total_rows + batch_size - 1) // batch_size
        logger.info(f"{day_str}: Processing {total_rows:,} rows in {total_batches:,} batches")
        
        temp_files = []
        for offset in range(0, total_rows, batch_size):
            batch_num = offset // batch_size
            
            # Log memory usage every 10 batches
            if batch_num % 10 == 0:
                log_memory_usage(logger)
            
            logger.info(f"{day_str}: Writing batch {batch_num + 1}/{total_batches}")
            
            # Create a temporary view for this batch
            con.execute(f"""
                CREATE OR REPLACE VIEW batch_view AS
                SELECT * FROM final_data
                LIMIT {batch_size} OFFSET {offset}
            """)
            
            # Use DuckDB's native write (much faster than pandas)
            file_created = write_incremental_results(con, "batch_view", output_folder, batch_num, output_mode)
            if file_created:
                if output_mode == 'partitioned':
                    temp_files.append(output_folder / f"partition_{batch_num:03d}.parquet")
                else:
                    temp_files.append(output_folder / f"temp_batch_{batch_num:03d}.parquet")
            else:
                break
            
            gc.collect()
        
        # Combine temporary files only in merged mode
        if output_mode == 'merged':
            logger.info(f"{day_str}: Combining batch files into single output...")
            final_output_path = output_folder / "data.parquet"
            combine_temp_files(con, final_output_path, temp_files, logger)
        else:
            logger.info(f"{day_str}: Keeping {len(temp_files)} partition files")
        
        logger.info(f"{day_str}: Successfully processed and saved to {output_folder}")
        
    except Exception as e:
        logger.error(f"{day_str}: Error processing data: {e}", exc_info=True)
        log_failed_date(day_str)
        raise


if __name__ == '__main__':
    logger = initialize_main()
    
    logger.info("Starting 1_raw_parquet_to_processed_duckdb.py script.")
    logger.info(f"Starting processing with target years: {TARGET_YEARS if TARGET_YEARS else 'All years'}")
    logger.info(f"REPROCESS mode is {'ON' if REPROCESS else 'OFF'}")

    # CLI args
    parser = argparse.ArgumentParser(description="Process raw parquet to processed using DuckDB")
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Process each day in batches instead of all at once (default: all at once)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Batch size when --batched is used (default: 100000)"
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=['merged', 'partitioned'],
        default='partitioned',
        help="Output mode: 'merged' combines batches into single file, 'partitioned' keeps separate partition files (default: partitioned)"
    )
    args, unknown = parser.parse_known_args()
    
    logger.info(f"Processing mode: {'Batched' if args.batched else 'Non-batched'}")
    if args.batched:
        logger.info(f"Batch size: {args.batch_size:,}")
        logger.info(f"Output mode: {args.output_mode}")

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
    
    # Configure DuckDB for memory efficiency
    manager = DuckDBManager()
    con = manager.connect()
    try:
        # Already configured via manager.connect()
        
        # Process each day
        successful_count = 0
        failed_count = 0
        
        for daily_folder in process_folders:
            day_str = daily_folder.name
            output_folder = PROCESSED_PATH / day_str
            
            try:
                if args.batched:
                    process_day_batched(con, daily_folder, output_folder, args.batch_size, args.output_mode, logger)
                else:
                    process_day_non_batched(con, daily_folder, output_folder, logger)
                
                successful_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {day_str}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"Processing completed: {successful_count} successful, {failed_count} failed")
        logger.info("All parquet folders have been processed.")
        
    finally:
        con.close()
        logger.info("DuckDB connection closed.")

