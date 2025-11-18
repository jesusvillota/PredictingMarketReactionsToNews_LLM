# uv run src/mains/parquet_to_csv.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

import dask.dataframe as dd
from src.config import config_settings
from src.config import initialize_main

if __name__ == '__main__':

    logger = initialize_main()

    logger.info("Starting parquet_to_csv.py script.")

    # Read parquet
    str_path: str = "_ATTENTION_SENTIMENT_DIRECTION_"
    input_path = Path(str_path)
    # input_path: Path = Path("_OUTPUT_") / str_path
    
    ddf = dd.read_parquet(
        input_path,
        engine=config_settings.parquet["engine"],
    )
    logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")

    # Save to CSV
    output_path = input_path
    output_path.mkdir(parents=True, exist_ok=True)
    ddf.compute().to_csv(
        output_path / "output.csv", 
        index=False,
    )
    logger.info(f"Saved to {output_path / 'output.csv'}")