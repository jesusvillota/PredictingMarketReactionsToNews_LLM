# src/deltas/delta_exposure.py
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from src import get_logger
from .get_equities_afterhours import get_equities_afterhours, save_equities_afterhours

# Path configurations
from src.config import config_settings
daily_files_path = config_settings.data_paths["delta_exposure_daily_files_path"]

def get_delta_exposure_new(ddf: dd.DataFrame, date_str: str) -> None:
    """
    Calculate delta exposure and include the date in the output.
    Returns True if successful (regardless of whether data was found), False if error.
    Only saves files when there's actual data.
    """
    logger = get_logger(__name__)
    logger.debug(f"Starting delta exposure calculation for {date_str}")
    
    try: 

        equities_afterhours_ddf = get_equities_afterhours(ddf)
        success: bool = save_equities_afterhours(equities_afterhours_ddf, date_str)  # Save after-hours equities for reference
        if not success: 
            logger.warning(f"No after-hours equities data to save for {date_str}")
            return False
        
        # THIS SHOULD BE DONE IN PANDAS! (NOT DASK)
        # THIS DATA IS EXPECTED TO BE SMALL AFTER FILTERING!
        try:
            # Single aggregation operation
            result: pd.DataFrame = (equities_afterhours_ddf
                    .groupby('okey_tk', observed=True)
                    .agg({
                        'unsigned_delta': 'sum',
                        'signed_delta': 'sum'
                    })
                    .rename(columns={
                        'unsigned_delta': 'gross_delta_exposure',
                        'signed_delta': 'net_delta_exposure'
                    })
                    .reset_index()
                    .compute())
            
            logger.debug(f"Delta exposure calculation completed for {date_str}")
        except Exception as e:
            logger.error(f"Error computing delta exposure for {file_name}: {e}", exc_info=True)
            return None
        
        # Check if result has data
        if len(result) == 0:
            logger.info(f"No after-hours equity data found for {date_str}")
            return True  # Success, but no data to save
        
        # Add date column and rename ticker column
        result['date'] = date_str
        
        # Reorder columns to desired format
        result = result[['date', 'okey_tk', 'gross_delta_exposure', 'net_delta_exposure']]
        # Only save if we have data
        delta_exposure_daily_files_path = config_settings.data_paths["delta_exposure_daily_files_path"]
        delta_exposure_daily_files_path.mkdir(parents=True, exist_ok=True)
        
        result.to_parquet(
            path=delta_exposure_daily_files_path / f"{date_str}.parquet", 
            # write_metadata_file=config_settings.parquet["write_metadata_file"],
            # write_index=config_settings.parquet["write_index"],
            compression=config_settings.parquet["compression"],
            engine=config_settings.parquet["engine"],
            )
        
        logger.info(f"Saved delta exposure data for {date_str}: {len(result)} records")
        return True
    
    except Exception as e:
        logger.error(f"Error processing the delta exposure for {date_str}: {e}", exc_info=True)
        return False

def aggregate_all_delta_exposures() -> None:
    """
    Efficiently aggregate all daily delta exposure files into a single dataset.
    """
    logger = get_logger(__name__)
    
    if not daily_files_path.exists():
        logger.error("No daily files directory found")
        return
    
    # Find all daily parquet files
    parquet_files = list(daily_files_path.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"No daily parquet files found in {daily_files_path}")
        return

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Read and concatenate all files efficiently
    try:
        
        # Filter out any corrupted files
        valid_files = []
        for file in parquet_files:
            try:
                # Quick validation read
                temp_df = pd.read_parquet(file, engine='pyarrow')
                if len(temp_df) > 0:
                    valid_files.append(file)
                    logger.debug(f"Valid file: {file.name} with {len(temp_df)} records")
                else:
                    logger.warning(f"Empty file found and skipped: {file.name}")
            except Exception as e:
                logger.warning(f"Corrupted file skipped {file.name}: {e}")
        
        if not valid_files:
            logger.error("No valid parquet files found")
            return

        # Use Dask for efficient reading and concatenation of large files
        daily_dfs = [dd.read_parquet(file) for file in valid_files]
        combined_ddf = dd.concat(daily_dfs, ignore_index=True)
        
        # Convert to pandas for final operations and saving
        # combined_df = combined_ddf.compute()
        
        # Sort by date and ticker for better organization
        # combined_df = combined_df.sort_values(['date', 'okey_tk']).reset_index(drop=True)
        
        # Save the aggregated file
        all_days_path = config_settings.data_paths["delta_exposure_path"] / "all_days.parquet"
        combined_ddf.to_parquet(
            all_days_path, 
            # write_metadata_file=config_settings.parquet["write_metadata_file"],
            write_index=config_settings.parquet["write_index"],
            compression=config_settings.parquet["compression"],
            engine=config_settings.parquet["engine"],
            )

        # # Also save as CSV for easy viewing (optional)
        # csv_path = output_path / "all_days.csv"
        # combined_df.to_csv(csv_path, index=False)
        
        logger.info(f"Aggregated delta exposures saved to {all_days_path}")
        logger.info(f"Total records: {len(combined_df)}")
        logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        logger.info(f"Unique tickers: {combined_df['okey_tk'].nunique()}")
        logger.info(f"Days with data: {combined_df['date'].nunique()}")
    
    except Exception as e:
        logger.error(f"Error aggregating delta exposures: {e}", exc_info=True)


def get_processing_summary(daily_files_path: Path = daily_files_path) -> dict:
    """
    Utility function to get a summary of processing results.
    """
    logger = get_logger(__name__)
    
    parquet_files = list(daily_files_path.glob("*.parquet"))
    
    summary = {
        'total_days_with_data': len(parquet_files),
        'files_found': [f.stem for f in parquet_files],
        'date_range': None,
        'total_records': 0
    }
    
    if parquet_files:
        try:
            # Quick scan of all files
            all_dates = []
            total_records = 0
            
            for file in parquet_files:
                try:
                    df = pd.read_parquet(file)
                    all_dates.extend(df['date'].unique())
                    total_records += len(df)
                except Exception as e:
                    logger.warning(f"Error reading {file}: {e}")
            
            if all_dates:
                summary['date_range'] = (min(all_dates), max(all_dates))
                summary['total_records'] = total_records
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
    
    return summary

# def get_delta_exposure_optimal(ddf: dd.DataFrame) -> pd.DataFrame:
#     """
#     Optimally filtered version based on your data characteristics.
#     Filter by time first (most selective), then by ticker_class.
#     """
    
#     # STEP 1: Filter by time first (most selective - reduces to 0.63% of data)
#     hour = ddf['timestamp_ny'].dt.hour
#     minute = ddf['timestamp_ny'].dt.minute
    
#     # After hours filter (highly selective)
#     after_hours_mask = ~(((hour > 9) | ((hour == 9) & (minute >= 30))) & (hour < 16))
#     after_hours_ddf = ddf.loc[after_hours_mask]
    
#     # STEP 2: Filter by equity on the much smaller dataset
#     equity_mask = after_hours_ddf['ticker_class'] == 'Equity'
#     equities_afterhours_ddf = after_hours_ddf.loc[equity_mask]
    
#     # STEP 3: Single aggregation operation
#     result = (equities_afterhours_ddf
#               .groupby('okey_tk', observed=True)
#               .agg({
#                   'unsigned_delta': 'sum',
#                   'signed_delta': 'sum'
#               })
#               .rename(columns={
#                   'unsigned_delta': 'gross_delta_exposure',
#                   'signed_delta': 'net_delta_exposure'
#               })
#               .reset_index()
#               .compute())
    
#     # Save to JSON
#     result.to_json("processed/delta_exposure.json", orient="records", lines=True)
    
#     return None

# def get_delta_exposure(ddf: dd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate the delta exposure for each trade in the Dask DataFrame.
#     Delta exposure is defined as prtSize * delta.
#     """
#     hour = ddf['timestamp_ny'].dt.hour
#     minute = ddf['timestamp_ny'].dt.minute

#     market_hours = ((hour > 9) | ((hour == 9) & (minute >= 30))) & (hour < 16)
#     after_hours = ~market_hours

#     equities = (ddf['ticker_class'] == 'Equity')
#     ddf = ddf.loc[after_hours & equities]

#     unsigned_delta_exposure = ddf['unsigned_delta'].groupby(ddf['okey_tk']).sum().compute()
#     signed_delta_exposure = ddf['signed_delta'].groupby(ddf['okey_tk']).sum().compute()

#     exposures = pd.DataFrame({
#         'okey_tk': unsigned_delta_exposure.index,
#         'unsigned_delta_exposure': unsigned_delta_exposure.values,
#         'signed_delta_exposure': signed_delta_exposure.values
#     })

#     exposures.to_json("processed/delta_exposure.json", orient="records", lines=True)

#     return exposures