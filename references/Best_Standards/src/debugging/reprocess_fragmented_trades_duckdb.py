# uv run src/mains/reprocess_fragmented_trades_duckdb.py
# uv run src/mains/reprocess_fragmented_trades_duckdb.py --batched --batch-size 100000
# uv run src/mains/reprocess_fragmented_trades_duckdb.py --overwrite

"""
Fragmented trades grouping script using DuckDB - groups trades by key characteristics
and aggregates fragment counts and sizes. Memory-efficient version using DuckDB instead of Dask.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
import numpy as np
import argparse
import gc
import psutil
from typing import Optional

from src.config import config_settings, initialize_main
from src.config.config_settings import PROCESSED_PATH, RAM_LIMIT, CPU_LIMIT, DASK_TEMP_DIR, OUTPUT_PATH
from src.config.utils import DailyFolderFilter

#----------------------------------------------------------------------------------------------------------------------#
TARGET_YEAR_MONTHS: list[str] = ["2019-01", "2019-02", "2019-03", "2019-04"]  # Year-month combinations to process
#----------------------------------------------------------------------------------------------------------------------#

# Grouping columns from group_fragmented_trades.py
GROUPING_COLS: list[str] = [
    "okey_tk", 
    "okey_xx", 
    "okey_cp",
    "uBid", 
    "uAsk", 
    "uPrc",
    "prtExch",
    "prtPrice", 
    "prtType", 
    "timestamp_ny_round3",
    "tradingSession"
]


def log_memory_usage(logger):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")


def get_available_grouping_cols(con: duckdb.DuckDBPyConnection, view_name: str) -> list[str]:
    """Check which grouping columns exist in the data"""
    # Get schema from the view
    schema_query = f"DESCRIBE {view_name}"
    schema_df = con.execute(schema_query).df()
    available_cols = schema_df['column_name'].tolist()
    
    # Filter grouping columns to only those available
    available_grouping_cols = [col for col in GROUPING_COLS if col in available_cols]
    return available_grouping_cols


def process_batch_groups(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a batch of grouped data.
    The data already has fragment_count and prtSize_agg from DuckDB query.
    This function can add additional processing if needed.
    """
    return batch_df


def write_incremental_results(
    results_df: pd.DataFrame, 
    output_dir: Path, 
    batch_num: int, 
    overwrite_mode: bool
) -> bool:
    """Write results incrementally to avoid memory accumulation. Returns True if file was created."""
    if results_df.empty:
        return False
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file for this batch
    temp_file = output_dir / f"temp_batch_{batch_num:06d}.parquet"
    
    results_df.to_parquet(
        temp_file,
        engine=config_settings.parquet["engine"],
        compression=config_settings.parquet["compression"],
        index=False
    )
    return True


def combine_temp_files(output_path: Path, temp_files: list[Path], logger, overwrite_mode: bool = False):
    """Combine temporary parquet files into final output"""
    if not temp_files:
        return
    
    # Filter to only existing files
    existing_files = [temp_file for temp_file in temp_files if temp_file.exists()]
    
    if not existing_files:
        logger.warning("No temporary files found to combine")
        return
    
    logger.info(f"Combining {len(existing_files)} temporary files...")
    
    # Read all existing temp files and combine
    combined_df = pd.concat([
        pd.read_parquet(temp_file, engine=config_settings.parquet["engine"])
        for temp_file in existing_files
    ], ignore_index=True)
    
    # Write final output
    combined_df.to_parquet(
        output_path,
        engine=config_settings.parquet["engine"],
        compression=config_settings.parquet["compression"],
        index=False
    )
    
    # Clean up temp files
    for temp_file in existing_files:
        temp_file.unlink(missing_ok=True)
    
    logger.info(f"Successfully combined {len(combined_df)} records into final output")


def process_day_non_batched(
    con: duckdb.DuckDBPyConnection,
    daily_folder: Path,
    output_path: Path,
    logger,
    overwrite_mode: bool = False
):
    """Process a single day's data without batching"""
    day_str = daily_folder.name
    
    # Get available grouping columns
    available_grouping_cols = get_available_grouping_cols(con, 'trades_data')
    
    if not available_grouping_cols:
        logger.warning(f"{day_str}: No grouping columns found in data")
        return
    
    logger.info(f"{day_str}: Using {len(available_grouping_cols)} grouping columns: {available_grouping_cols}")
    
    # Build the partition by clause
    partition_clause = ", ".join(available_grouping_cols)
    
    # Query to get grouped data with aggregations
    # Use window functions to compute aggregations, then deduplicate
    query = f"""
        SELECT *,
               COUNT(*) OVER (PARTITION BY {partition_clause}) as fragment_count,
               SUM(prtSize) OVER (PARTITION BY {partition_clause}) as prtSize_agg
        FROM trades_data
        QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY prtSize) = 1
        ORDER BY {partition_clause}
    """
    
    logger.info(f"{day_str}: Executing grouping query...")
    result_df = con.execute(query).df()
    
    if result_df.empty:
        logger.info(f"{day_str}: No data found")
        return
    
    logger.info(f"{day_str}: Found {len(result_df)} unique groups")
    
    # Write output
    result_df.to_parquet(
        output_path,
        engine=config_settings.parquet["engine"],
        compression=config_settings.parquet["compression"],
        index=False
    )
    
    logger.info(f"{day_str}: Successfully saved to: {output_path}")


def process_day_batched(
    con: duckdb.DuckDBPyConnection,
    daily_folder: Path,
    output_dir: Path,
    final_output_path: Path,
    batch_size: int,
    logger,
    overwrite_mode: bool = False
):
    """Process a single day's data in batches"""
    day_str = daily_folder.name
    
    # Get available grouping columns
    available_grouping_cols = get_available_grouping_cols(con, 'trades_data')
    
    if not available_grouping_cols:
        logger.warning(f"{day_str}: No grouping columns found in data")
        return
    
    logger.info(f"{day_str}: Using {len(available_grouping_cols)} grouping columns")
    
    # Build the partition by clause
    partition_clause = ", ".join(available_grouping_cols)
    
    # First, create a temp table with the grouped data (one row per group)
    logger.info(f"{day_str}: Creating temporary grouped data...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE grouped_data AS
        SELECT *,
               COUNT(*) OVER (PARTITION BY {partition_clause}) as fragment_count,
               SUM(prtSize) OVER (PARTITION BY {partition_clause}) as prtSize_agg
        FROM trades_data
        QUALIFY ROW_NUMBER() OVER (PARTITION BY {partition_clause} ORDER BY prtSize) = 1
    """)
    
    # Count total groups
    total_groups = con.execute("SELECT COUNT(*) FROM grouped_data").fetchone()[0]
    logger.info(f"{day_str}: Found {total_groups:,} unique groups to process in batches")
    
    if total_groups == 0:
        logger.info(f"{day_str}: No groups found")
        return
    
    # Process in batches
    temp_files = []
    processed_groups = 0
    
    total_batches = (total_groups + batch_size - 1) // batch_size
    logger.info(f"{day_str}: Processing in {total_batches:,} batches of {batch_size:,}")
    
    for offset in range(0, total_groups, batch_size):
        batch_num = offset // batch_size + 1
        
        # Log memory usage every 10 batches
        if batch_num % 10 == 1:
            log_memory_usage(logger)
        
        logger.info(f"{day_str}: Processing batch {batch_num}/{total_batches} "
                   f"(groups {offset:,} to {min(offset + batch_size, total_groups):,})")
        
        # Get batch of groups
        batch_query = f"""
            SELECT * FROM grouped_data
            ORDER BY {partition_clause}
            LIMIT {batch_size} OFFSET {offset}
        """
        
        try:
            batch_df = con.execute(batch_query).df()
        except Exception as e:
            logger.error(f"{day_str}: Error executing batch query: {e}")
            break
        
        if batch_df.empty:
            break
        
        # Process the batch (currently just passes through, but can add logic)
        results_df = process_batch_groups(batch_df)
        
        if not results_df.empty:
            # Write results incrementally
            file_created = write_incremental_results(results_df, output_dir, batch_num - 1, overwrite_mode)
            
            if file_created:
                temp_files.append(output_dir / f"temp_batch_{batch_num - 1:06d}.parquet")
        
        processed_groups += len(batch_df)
        
        # Force garbage collection to free memory
        del batch_df, results_df
        gc.collect()
        
        # Log progress every 100 batches or at the end
        if batch_num % 100 == 0 or batch_num == total_batches:
            logger.info(f"{day_str}: Processed {processed_groups:,}/{total_groups:,} groups "
                       f"({processed_groups/total_groups*100:.1f}%)")
    
    # Combine all temporary files into final output
    logger.info(f"{day_str}: Combining results into final output...")
    combine_temp_files(final_output_path, temp_files, logger, overwrite_mode)
    
    logger.info(f"{day_str}: Successfully saved to: {final_output_path}")
    
    # Clean up temp table
    con.execute("DROP TABLE IF EXISTS grouped_data")


if __name__ == '__main__':
    
    logger = initialize_main()
    logger.info("Starting group_fragmented_trades_duckdb.py script.")
    logger.info(f"Reading Parquet files from {PROCESSED_PATH}")
    
    # Configuration for processing mode
    logger.info(f"Starting processing with target year-months: {TARGET_YEAR_MONTHS}")
    
    # CLI args
    parser = argparse.ArgumentParser(description="Group fragmented trades using DuckDB")
    parser.add_argument(
        "--batched",
        action="store_true",
        help="Process each day in batches instead of all at once (default: all at once)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size when --batched is used (default: 10000)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original files instead of creating new folder (default: create new folder)"
    )
    args, unknown = parser.parse_known_args()
    
    logger.info(f"Processing mode: {'Batched' if args.batched else 'Non-batched'}")
    if args.batched:
        logger.info(f"Batch size: {args.batch_size:,}")
    logger.info(f"Output mode: {'Overwrite original files' if args.overwrite else 'Create new folder'}")
    
    # Configure DuckDB for memory efficiency
    con = duckdb.connect()
    try:
        # Set memory limits and temp directory
        con.execute(f"SET memory_limit='{RAM_LIMIT}GB'")
        temp_dir = DASK_TEMP_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{temp_dir}'")
        
        # Configure for better performance on large datasets
        con.execute("SET enable_progress_bar=true")
        con.execute(f"SET threads={CPU_LIMIT}")
        
        con.execute("SET preserve_insertion_order=false")  # Disable to save memory    
        con.execute("SET enable_object_cache=false")  # Disable object cache
        
        # Discover and filter daily folders using DailyFolderFilter
        filter = DailyFolderFilter(PROCESSED_PATH)
        daily_folders = filter.by_year_month(TARGET_YEAR_MONTHS, return_globs=False)
        
        if not daily_folders:
            logger.error(f"No daily folders found for the specified year-months: {TARGET_YEAR_MONTHS}")
            exit(1)
        
        logger.info(f"Found {len(daily_folders)} daily folders to process")
        logger.info(f"First 5 daily folders: {[f.name for f in daily_folders[:5]]}")
        logger.info(f"Last 5 daily folders: {[f.name for f in daily_folders[-5:]]}")
        
        for daily_folder in daily_folders:
            day_str = daily_folder.name
            
            # Determine output path based on mode
            if args.overwrite:
                # Overwrite mode: output to same location as input
                output_dir = daily_folder
                final_output_path = daily_folder / "grouped_trades.parquet"
            else:
                BASE_OUTPUT_DIR = OUTPUT_PATH / "GROUPED_TRADES_DAILY"
                BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                output_dir = BASE_OUTPUT_DIR / day_str
                final_output_path = output_dir / "grouped_trades.parquet"
            
            try:
                # Create/refresh view for this day's data
                logger.info(f"Creating DuckDB view for {day_str}...")
                con.execute(f"""
                    CREATE OR REPLACE VIEW trades_data AS 
                    SELECT *
                    FROM read_parquet('{daily_folder}/**/*.parquet')
                """)
                
                # Check if view has data
                count = con.execute("SELECT COUNT(*) FROM trades_data").fetchone()[0]
                if count == 0:
                    logger.info(f"{day_str}: No data found, skipping")
                    continue
                
                logger.info(f"{day_str}: Loaded {count:,} rows from parquet files")
                
                if args.batched:
                    # Batched processing
                    process_day_batched(
                        con=con,
                        daily_folder=daily_folder,
                        output_dir=output_dir,
                        final_output_path=final_output_path,
                        batch_size=args.batch_size,
                        logger=logger,
                        overwrite_mode=args.overwrite
                    )
                else:
                    # Non-batched processing
                    process_day_non_batched(
                        con=con,
                        daily_folder=daily_folder,
                        output_path=final_output_path,
                        logger=logger,
                        overwrite_mode=args.overwrite
                    )
                
                # Cleanup
                gc.collect()
                
            except Exception as e:
                logger.error(f"{day_str}: Error processing data: {e}", exc_info=True)
                # Continue with next day instead of stopping the whole run
                continue
    
    finally:
        con.close()
        logger.info("Processing complete. DuckDB connection closed.")

