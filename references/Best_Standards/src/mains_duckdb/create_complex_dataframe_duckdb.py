# uv run src/mains_duckdb/create_complex_dataframe_duckdb.py
# uv run src/mains_duckdb/create_complex_dataframe_duckdb.py --batched --batch-size 100000

"""
Complex trades classification script using DuckDB - identifies and classifies complex option strategies
Memory-efficient version that processes 2TB of data without excessive RAM usage
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Optional
import argparse
import tempfile
import gc
import psutil

from src.config import config_settings, initialize_main
from src.config.config_settings import PROCESSED_PATH, TEMP_DIR, COMPLEX_TRADES_PATH
from src.config.duckdb_manager import DuckDBManager
for path in [COMPLEX_TRADES_PATH, TEMP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

#----------------------------------------------------------------------------------------------------------------------#
TARGET_YEARS: list[int] | None = [2021]  # [2020, 2021, 2022, 2023]
#----------------------------------------------------------------------------------------------------------------------#

def check_flag_exp_strike(legs: pd.DataFrame) -> tuple[bool, bool, bool]:
    """Check if all legs have same flag, expiration, and strike"""
    same_flag = legs["okey_cp"].nunique() == 1
    same_exp = legs["expdate"].nunique() == 1
    same_strike = legs["okey_xx"].nunique() == 1
    return same_flag, same_exp, same_strike


def sign_complex_trade(legs: pd.DataFrame, sum_mode: Optional[bool] = None) -> str:
    """Determine if a strategy is Long, Short, or Midpoint based on price comparison"""
    n_legs = legs.shape[0]
    
    if n_legs < 2:
        return "Undetermined"
    
    if n_legs in (3, 4):
        legs = legs.sort_values("okey_xx").reset_index(drop=True)
    
    prices = legs['prtPrice'].values
    midpoints = legs['midpointNBBO'].values

    if np.any(pd.isna(prices)) or np.any(pd.isna(midpoints)):
        return "Undetermined"

    if n_legs == 2:
        if sum_mode:
            netprice = abs(prices[0] + prices[1])
            netmid = abs(midpoints[0] + midpoints[1])
        else:
            netprice = abs(prices[0] - prices[1])
            netmid = abs(midpoints[0] - midpoints[1])
    elif n_legs == 3:
        netprice = abs(prices[0] + prices[2] - 2*prices[1])
        netmid = abs(midpoints[0] + midpoints[2] - 2*midpoints[1])
    elif n_legs == 4:
        netprice = abs(prices[0]-prices[1]) + abs(prices[2]-prices[3])
        netmid = abs(midpoints[0] - midpoints[1]) + abs(midpoints[2] - midpoints[3])
    else:
        return "Undetermined"
    
    if pd.isna(netprice) or pd.isna(netmid) or not np.isfinite(netprice) or not np.isfinite(netmid):
        return "Undetermined"
    
    if netprice > netmid:
        return "Long"
    elif netprice < netmid:
        return "Short"
    else:
        return "Midpoint"


def classify_strategy(group_df: pd.DataFrame) -> tuple[str, str, str]:
    """Classify a complex strategy based on its legs.
    
    Returns:
        tuple[str, str, str]: (sign, flag, strategy_name)
        - sign: "Long", "Short", "Midpoint", "Single", or "Undetermined"
        - flag: "Call", "Put", "Mixed", or "None"
        - strategy_name: "Spread", "Calendar", "Diagonal", "Butterfly", "Condor", "IronCondor", "Straddle", "Strangle", "Single", or "Other"
    """
    
    # legs = group_df[["okey_cp", "okey_xx", "expiration", 
    #                  "prtPrice", "midpointNBBO", "prtSize_agg"]].copy()
    
    legs = group_df[["okey_cp", "okey_xx", "expiration", 
                     "prtPrice", "midpointNBBO", "prtSize_agg", "buy_sell_class"]].copy()
    
    legs["expdate"] = legs["expiration"].dt.normalize()
    legs = legs.drop(columns=["expiration"])
    
    n_legs = len(legs)
    
    # Single Leg
    if n_legs == 1:
        flag = legs.iloc[0]['okey_cp']
        buy_sell_class = legs.iloc[0]['buy_sell_class']
        return buy_sell_class, flag, "Single"
    
    # 2-leg strategies
    if n_legs == 2:
        same_flag, same_exp, same_strike = check_flag_exp_strike(legs)
        
        if same_flag:
            flag = legs["okey_cp"].iloc[0]
            sign = sign_complex_trade(legs, sum_mode=False)
            
            # if sign == "Undetermined":
            #     return "Undetermined", flag, "Other"
            
            if same_exp and not same_strike:
                return sign, flag, "Spread"
            elif (not same_exp) and same_strike:
                return sign, flag, "Calendar"
            elif (not same_exp) and (not same_strike):
                return sign, flag, "Diagonal"
        else:
            sign = sign_complex_trade(legs, sum_mode=True)
            
            # if sign == "Undetermined":
            #     return "Undetermined", "Mixed", "Other"
            
            if same_exp and same_strike:
                return sign, "Mixed", "Straddle"
            elif same_exp and (not same_strike):
                call_strikes = legs[legs["okey_cp"] == "Call"]["okey_xx"]
                put_strikes = legs[legs["okey_cp"] == "Put"]["okey_xx"]
                
                if len(call_strikes) > 0 and len(put_strikes) > 0:
                    if call_strikes.iloc[0] > put_strikes.iloc[0]:
                        return sign, "Mixed", "Strangle"
    
    # 3-leg strategies
    elif n_legs == 3:
        same_flag, same_exp, _ = check_flag_exp_strike(legs)
        
        legs = legs.sort_values("okey_xx").reset_index(drop=True)
        sizes = legs["prtSize_agg"].values
        
        if same_flag and same_exp and (2*sizes[0] == sizes[1] == 2*sizes[2]):
            sign = sign_complex_trade(legs)
            # if sign != "Undetermined":
            flag = legs.iloc[0]['okey_cp']
            return sign, flag, "Butterfly"

    # 4-leg strategies
    elif n_legs == 4:
        same_flag, same_exp, _ = check_flag_exp_strike(legs)
        
        if same_flag and same_exp:
            sign = sign_complex_trade(legs)
            flag = legs['okey_cp'].iloc[0]
            return sign, flag, "Condor"
        
        elif not same_flag and same_exp:
            calls = legs[legs["okey_cp"] == "Call"]
            puts = legs[legs["okey_cp"] == "Put"]
            
            if len(calls) == 2 and len(puts) == 2:
                calls = calls.sort_values("okey_xx")
                puts = puts.sort_values("okey_xx")

                width_call = calls.iloc[1]["okey_xx"] - calls.iloc[0]["okey_xx"]
                width_put = puts.iloc[1]["okey_xx"] - puts.iloc[0]["okey_xx"]
                
                if width_call > 0 and width_put > 0 and width_call == width_put:
                    netprice = (abs(calls.iloc[0]["prtPrice"] - calls.iloc[1]["prtPrice"]) +
                               abs(puts.iloc[0]["prtPrice"] - puts.iloc[1]["prtPrice"]))
                    netmid = (abs(calls.iloc[0]["midpointNBBO"] - calls.iloc[1]["midpointNBBO"]) +
                             abs(puts.iloc[0]["midpointNBBO"] - puts.iloc[1]["midpointNBBO"]))
                    sign = "Long" if netprice > netmid else "Short" if netprice < netmid else "Midpoint"
                    # if sign != "Undetermined":
                    return sign, "Mixed", "IronCondor"
    
    return "Undetermined", "None", "Other"


def process_batch_of_groups(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a batch of complex trade groups to extract strategy information.
    Returns a DataFrame with one row per group containing strategy details.
    """
    if batch_df.empty:
        return pd.DataFrame({
            'okey_tk': pd.Series([], dtype='object'),
            'prtExch': pd.Series([], dtype='object'),
            'prtType': pd.Series([], dtype='int64'),
            'prtSize_agg': pd.Series([], dtype='float64'),
            'timestamp_ny_round3': pd.Series([], dtype='datetime64[ns]'),
            'n_legs': pd.Series([], dtype='int64'),
            'sign': pd.Series([], dtype='object'),
            'flag': pd.Series([], dtype='object'),
            'strategy_name': pd.Series([], dtype='object'),
            'details': pd.Series([], dtype='object')
        })
    
    results = []
    
    for _, group_row in batch_df.iterrows():
        # Parse the legs data from the struct_pack format
        legs_data = group_row['legs']
        
        # Convert list of dictionaries to DataFrame for processing
        legs_df = pd.DataFrame(list(legs_data))
        
        # Get grouping key values
        okey_tk = group_row['okey_tk']
        prtExch = group_row['prtExch']
        prtType = group_row['prtType']
        prtSize_agg = legs_df['prtSize_agg'].max()
        timestamp_ny_round3 = group_row['timestamp_ny_round3']
        
        # Get number of legs
        n_legs = len(legs_df)
        
        # Classify the strategy
        sign, flag, strategy_name = classify_strategy(legs_df)
        
        # Create details list with leg information
        details = []
        for _, leg in legs_df.iterrows():
            leg_info = {
                'okey_cp': leg['okey_cp'],
                'okey_xx': float(leg['okey_xx']),
                'expiration': leg['expiration'],
                'prtPrice': float(leg['prtPrice']),
                'midpointNBBO': float(leg['midpointNBBO']),
                'prtSize_agg': float(leg['prtSize_agg'])
            }
            details.append(leg_info)
        
        results.append({
            'okey_tk': okey_tk,
            'prtExch': prtExch,
            'prtType': prtType,
            'prtSize_agg': prtSize_agg,
            'timestamp_ny_round3': timestamp_ny_round3,
            'n_legs': n_legs,
            'sign': sign,
            'flag': flag,
            'strategy_name': strategy_name,
            'details': details
        })
    
    return pd.DataFrame(results)


def write_incremental_results(results_df: pd.DataFrame, output_dir: Path, batch_num: int) -> bool:
    """Write results incrementally to avoid memory accumulation. Returns True if file was created."""
    if results_df.empty:
        return False
    
    # Ensure output directory exists (per-day)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file for this batch inside the day folder
    temp_file = output_dir / f"temp_batch_{batch_num:06d}.parquet"
    
    results_df.to_parquet(
        temp_file,
        engine=config_settings.parquet["engine"],
        compression=config_settings.parquet["compression"],
        index=False
    )
    return True


def log_memory_usage(logger):
    """Log current memory usage"""
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent:.1f}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")


def combine_temp_files(output_path: Path, temp_files: list[Path], logger):
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


if __name__ == '__main__':
    
    logger = initialize_main()
    logger.info("Starting create_complex_dataframe_duckdb.py script.")
    logger.info(f"Reading Parquet files from {PROCESSED_PATH}")
    
    # Configuration for processing mode
    logger.info(f"Starting processing with target years: {TARGET_YEARS if TARGET_YEARS else 'All years'}")
    
    # CLI args
    parser = argparse.ArgumentParser(description="Create complex dataframe (DuckDB)")
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
    args, unknown = parser.parse_known_args()
    
    # Configure DuckDB for memory efficiency
    manager = DuckDBManager()
    con = manager.connect()
    try:
        # Already configured via manager.connect()
        
        # Discover and sort daily folders (YYYY-MM-DD)
        daily_folders = sorted([dir for dir in PROCESSED_PATH.iterdir() if dir.is_dir()])
        
        # Filter by target years if specified
        if TARGET_YEARS is not None:
            daily_folders = [f for f in daily_folders if int(f.name.split('-')[0]) in TARGET_YEARS]
            logger.info(f"Filtered to {len(daily_folders)} folders for years: {TARGET_YEARS}")
        
        if not daily_folders:
            logger.error(f"No daily folders found for the specified years: {TARGET_YEARS}")
            exit(1)
            
        logger.info(f"First 5 daily folders: {daily_folders[:5]}")
        logger.info(f"Last 5 daily folders: {daily_folders[-5:]}")
        
        for daily_folder in daily_folders:
            day_str = daily_folder.name
            output_dir = COMPLEX_TRADES_PATH / day_str
            output_dir.mkdir(parents=True, exist_ok=True)
            final_output_path = output_dir / "complex_trades.parquet"
            
            try:
                # Create/refresh view for this day's complex trades with filters
                logger.info(f"Creating DuckDB view for complex trades for {day_str}...")
                con.execute(f"""
                    CREATE OR REPLACE VIEW complex_trades AS 
                    SELECT okey_tk, okey_cp, okey_xx, expiration, prtPrice, 
                        midpointNBBO, prtSize_agg, prtExch, prtType, timestamp_ny_round3, buy_sell_class
                    FROM read_parquet('{daily_folder}/**/*.parquet')
                    WHERE ticker_class = 'equity' AND prtType >= 102
                """)
                
                if not args.batched:
                    # Non-batched path: process all groups for the day at once
                    logger.info(f"{day_str}: Processing all complex trades at once (default mode)...")
                    all_query = """
                        SELECT okey_tk, prtExch, prtType, timestamp_ny_round3,
                               LIST(struct_pack(okey_cp := okey_cp, okey_xx := okey_xx, expiration := expiration,
                                                prtPrice := prtPrice, midpointNBBO := midpointNBBO, prtSize_agg := prtSize_agg, buy_sell_class := buy_sell_class)) as legs
                        FROM complex_trades
                        GROUP BY okey_tk, prtExch, prtType, timestamp_ny_round3
                        ORDER BY okey_tk, prtExch, prtType, timestamp_ny_round3
                    """
                    all_df = con.execute(all_query).df()
                    if all_df.empty:
                        logger.info(f"{day_str}: No complex trades found")
                    else:
                        logger.info(f"{day_str}: Found {len(all_df)} complex trades")
                        results_df = process_batch_of_groups(all_df)
                        if not results_df.empty:
                            # Write final per-day output directly
                            results_df.to_parquet(
                                final_output_path,
                                engine=config_settings.parquet["engine"],
                                compression=config_settings.parquet["compression"],
                                index=False
                            )
                            logger.info(f"{day_str}: Complex trades successfully saved to: {final_output_path}")
                        del results_df
                    del all_df
                    gc.collect()
                else:
                    #------------------------------------------------------------------------
                    # Get total number of groups for progress tracking
                    logger.info("Counting total groups...")
                    total_groups = con.execute("""
                        SELECT COUNT(*) as total
                        FROM (
                            SELECT okey_tk, prtExch, prtType, timestamp_ny_round3
                            FROM complex_trades
                            GROUP BY okey_tk, prtExch, prtType, timestamp_ny_round3
                        )
                    """).fetchone()[0]
                    
                    logger.info(f"{day_str}: Found {total_groups:,} complex trade groups to process")
                    #------------------------------------------------------------------------
                    
                    # Process in batches
                    batch_size = args.batch_size
                    temp_files = []
                    processed_groups = 0
                    
                    logger.info(f"{day_str}: Processing {total_groups:,} complex trades in batches of {batch_size:,}...")
                    logger.info(f"{day_str}: Expected {((total_groups + batch_size - 1) // batch_size):,} batches")
                    
                    for offset in range(0, total_groups, batch_size):
                        batch_num = offset // batch_size + 1
                        total_batches = (total_groups + batch_size - 1) // batch_size
                        
                        # Log memory usage every 10 batches
                        if batch_num % 10 == 1:
                            log_memory_usage(logger)
                        
                        logger.info(f"{day_str}: Processing batch {batch_num}/{total_batches} "
                                f"(groups {offset:,} to {min(offset + batch_size, total_groups):,})")
                        
                        # Get batch of groups with their legs
                        batch_query = f"""
                            SELECT okey_tk, prtExch, prtType, timestamp_ny_round3,
                                LIST(struct_pack(okey_cp := okey_cp, okey_xx := okey_xx, expiration := expiration, 
                                                prtPrice := prtPrice, midpointNBBO := midpointNBBO, prtSize_agg := prtSize_agg, buy_sell_class := buy_sell_class)) as legs
                            FROM complex_trades
                            GROUP BY okey_tk, prtExch, prtType, timestamp_ny_round3
                            ORDER BY okey_tk, prtExch, prtType, timestamp_ny_round3
                            LIMIT {batch_size} OFFSET {offset}
                        """
                        try:
                            batch_df = con.execute(batch_query).df()
                        except Exception as e:
                            logger.error(f"{day_str}: Error executing batch query: {e}")
                            break
                        
                        if batch_df.empty:
                            break
                        
                        # Process the batch
                        results_df = process_batch_of_groups(batch_df)
                        
                        if not results_df.empty:
                            # Write results incrementally (per day)
                            file_created = write_incremental_results(results_df, output_dir, batch_num - 1)
                            
                            # Only add to temp_files if the file was actually created
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
                    
                    # Combine all temporary files into final per-day output and cleanup
                    logger.info(f"{day_str}: Combining results into final output...")
                    combine_temp_files(final_output_path, temp_files, logger)
                    
                    logger.info(f"{day_str}: Complex trades successfully saved to: {final_output_path}")
            except Exception as e:
                logger.error(f"{day_str}: Error processing complex trades: {e}", exc_info=True)
                # Continue with next day instead of stopping the whole run
                continue
    finally:
        con.close()
