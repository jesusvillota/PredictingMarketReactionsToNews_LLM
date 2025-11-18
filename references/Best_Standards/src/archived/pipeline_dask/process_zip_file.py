from src.config.logger import get_logger
import dask.dataframe as dd
import zipfile
import tempfile
from .extract_date import extract_date_from_single_file
from .pipeline import data_pipeline
# Path configurations
from pathlib import Path
from src.config import config_settings
from src.config.config_settings import *

FAILED_DATES_PATH: Path = Path("src/utils/timeline_check/missing_dates") / "before.txt"

def log_failed_date(file_name: str):
    with open(FAILED_DATES_PATH, "a") as f:
        f.write(f"{file_name}\n")

def process_zip_file(zip_file: Path) -> None:
    """
    Process a single zip file, extracting and processing its text files.
    """
    logger = get_logger(__name__)  # Logger for this function
    try:
        logger.info(f"Processing {zip_file.name}")
        
        # Extract date from filename
        date_obj, success = extract_date_from_single_file(zip_file.name)
        if not success or date_obj is None:
            logger.error(f"Could not extract date from filename: {zip_file.name}")
            file_name = zip_file.stem
        else:
            logger.info(f"Extracted date {date_obj} from filename {zip_file.name}")
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

                FILTERED_DTYPES = {col: DTYPES[col] for col in SELECTED_COLUMNS}

                # Read all extracted txt files in parallel with Dask
                ddf = dd.read_csv(
                    extracted_paths,
                    delimiter='\t',
                    usecols=SELECTED_COLUMNS,
                    dtype=FILTERED_DTYPES,
                    assume_missing=True,
                    na_values=['N/A', 'NA', '--', ''],
                )
                logger.info(f"Loaded files with {ddf.npartitions} partitions")

                # # Repartitioning logic
                # if config_settings.compute["partition_size_1"] and config_settings.compute["npartitions_1"]:
                #     logger.warning("Both partition_size_1 and npartitions_1 are set. Using npartitions_1 and ignoring partition_size_1.")
                #     config_settings.compute["partition_size_1"] = None  # Ignore partition_size_1 if both are set
                # if config_settings.compute["partition_size_1"]:
                #     logger.info(f"Repartitioning to partition size {config_settings.compute['partition_size_1']}")
                #     ddf = ddf.repartition(partition_size=config_settings.compute["partition_size_1"])
                # elif config_settings.compute["npartitions_1"]:
                #     logger.info(f"Repartitioning to {config_settings.compute['npartitions_1']} partitions")
                #     ddf = ddf.repartition(npartitions=config_settings.compute["npartitions_1"])

                try: 
                    # Processing pipeline
                    logger.info(f"Starting data processing pipeline for {file_name}")
                    ddf = data_pipeline(ddf, file_name)
                    
                    # Save the processed dd.DataFrame to a parquet file
                    parquet_path = RAW_PARQUET_PATH / f"{file_name}"
                    
                    ddf.to_parquet(
                        path=parquet_path,
                        compression=config_settings.parquet["compression"],
                        engine=config_settings.parquet["engine"],
                        overwrite=True,
                    )
                    logger.info(f"Saved processed data to {parquet_path}")
                    return True
                
                except Exception as e:
                    logger.error(f"Error in processing pipeline for {file_name}: {e}", exc_info=True)
                    log_failed_date(file_name)
                    return False
    
    except Exception as e:
        logger.error(f"Unexpected error processing {file_name}: {e}", exc_info=True)
        log_failed_date(file_name)
        return False
