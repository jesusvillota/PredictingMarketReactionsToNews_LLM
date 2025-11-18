from src.config.logger import setup_logger
from src import get_logger
import pandas as pd
import numpy as np

from src.config import config_settings
setup_logger(
        name=config_settings.PROJECT_NAME,
        level=config_settings.logging["level"],  # Default
        log_file=None,  # No file yet
        console_output=True
)

logger = get_logger(__name__)


def read_dates_from_file(file_path: str) -> list[str]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Remove newline characters
        lines = [line.strip() for line in lines]
        logger.info(f"Read dates from {file_path}")
    return lines

dates_official = read_dates_from_file("FILE_MANAGER/timeline_official.txt")
# Print a small sample of the dates
logger.info(f"Sample official dates: {dates_official[:5]}")

dates_processed = read_dates_from_file("FILE_MANAGER/timeline_processed.txt")
# Print a small sample of the dates
logger.info(f"Sample processed dates: {dates_processed[:5]}")

missing_dates = set(dates_official) - set(dates_processed)
logger.info(f"Dates in processed but missing from official: {missing_dates}")

# Save missing_dates to a file
with open("FILE_MANAGER/missing_dates.txt", "w") as f:
    for date in missing_dates:
        f.write(f"{date}\n")
logger.info(f"Saved missing dates to FILE_MANAGER/missing_dates.txt")