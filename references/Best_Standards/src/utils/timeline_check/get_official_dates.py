# Here I have a path to a set of folders with name: tbloptionprintsethist_EQT_v2.00_YYYY_MM_DD
# I want to extract the date part YYYY_MM_DD and save it into a text file, one date per line

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


import os
import re

# Update the path to use forward slashes for cross-platform compatibility
path = r"D:/tbloptionprintsethist_EQT"

# Regex pattern to match the filename and extract the date part (YYYY-MM-DD)
pattern = re.compile(r"tbloptionprintsethist_EQT_v2\.00_(\d{4}-\d{2}-\d{2})\.zip")

dates = []


# List all items in the directory
logger.info(f"Scanning directory: {path}")
for filename in os.listdir(path):
    logger.debug(f"Checking file: {filename}")
    match = pattern.match(filename)
    if match:
        date_found = match.group(1)
        logger.info(f"Matched date: {date_found}")
        dates.append(date_found)
logger.info(f"Total dates found: {len(dates)}")

# Save the dates to a text file, one per line
output_file = "timeline_official.txt"
logger.info(f"Writing dates to file: {output_file}")
with open(output_file, "w") as f:
    for date in dates:
        f.write(date + "\n")
logger.info(f"Finished writing {len(dates)} dates to {output_file}")