# uv run src/debugging/detect_failed_complex_trades.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import initialize_main
import os

# Use environment variables if set, otherwise use defaults from THIS_IS.py
# This allows for overriding paths during testing.
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", Path.cwd() / "_OUTPUT_"))
COMPLEX_TRADES_DAILY_PATH = Path(os.getenv("COMPLEX_TRADES_DAILY_PATH", Path.cwd() / "COMPLEX_TRADES_DAILY"))

def looks_like_ymd(name: str) -> bool:
    """
    Check if a folder name follows the YYYY-MM-DD format.
    
    Reuses validation logic from DailyFolderFilter._looks_like_ymd()
    """
    parts = name.split("-")
    if len(parts) != 3:
        return False
    try:
        y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return False
    return 1 <= m <= 12 and 1 <= d <= 31 and 1000 <= y <= 3000


if __name__ == '__main__':
    
    logger = initialize_main()
    
    logger.info(f"Scanning COMPLEX_TRADES_DAILY path: {COMPLEX_TRADES_DAILY_PATH}")
    
    # Check if the path exists
    if not COMPLEX_TRADES_DAILY_PATH.exists():
        logger.error(f"COMPLEX_TRADES_DAILY_PATH does not exist: {COMPLEX_TRADES_DAILY_PATH}")
        sys.exit(1)
    
    # Collect all date folders
    all_date_folders = []
    for item in COMPLEX_TRADES_DAILY_PATH.iterdir():
        if item.is_dir() and looks_like_ymd(item.name):
            all_date_folders.append(item)
    
    logger.info(f"Found {len(all_date_folders)} date folders")
    
    # Check which folders are missing complex_trades.parquet
    missing_dates = []
    for folder in sorted(all_date_folders):
        complex_trades_file = folder / "complex_trades.parquet"
        if not complex_trades_file.exists():
            missing_dates.append(folder.name)
            logger.debug(f"Missing complex_trades.parquet: {folder.name}")
    
    # Write results to output file
    output_file = OUTPUT_FOLDER / "missing_complex_trades.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for date in missing_dates:
            f.write(f"{date}\n")
    
    # Log summary
    logger.info(f"Total date folders checked: {len(all_date_folders)}")
    logger.info(f"Folders missing complex_trades.parquet: {len(missing_dates)}")
    logger.info(f"Results written to: {output_file}")
    
    if missing_dates:
        logger.warning(f"Found {len(missing_dates)} dates with failed or incomplete processing")
    else:
        logger.info("All date folders contain complex_trades.parquet")

