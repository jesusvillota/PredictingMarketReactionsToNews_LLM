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

# FOLDERS WHERE THE PROCESSED DAILY FILES ARE STORED IN MY CEMFI COMPUTER
path_1 = r"C:\Users\j-vill36\Desktop\_OUTPUT_\data\daily_files"
path_2 = r"D:\jesus_data_daily_files"

# Additional path for afterhours data
# path_3 = r"C:\Users\j-vill36\Desktop\whales\_OUTPUT_\data\equities_afterhours\daily_files"

# TEST PATH
path_test = "test_folders"

# Each of these paths contains multiple folders with the following naming convention: YYYY_MM_DD.parquet
# I want to remove the ".parquet" extension from the folder names (as they are not files but directories)
# Then, I want to list all the folder names into a txt file, one per line

def list_cleaned_folder_names(paths, output_txt):
        all_folders = []
        for path in paths:
                if not os.path.exists(path):
                        logger.warning(f"Path does not exist: {path}")
                        continue
                for name in os.listdir(path):
                        full_path = os.path.join(path, name)
                        if os.path.isdir(full_path):
                                clean_name = name
                                if clean_name.endswith('.parquet'):
                                        clean_name = clean_name[:-8]
                                        new_full_path = os.path.join(path, clean_name)
                                        try:
                                                os.rename(full_path, new_full_path)
                                                logger.info(f"Renamed {full_path} -> {new_full_path}")
                                                all_folders.append(clean_name)
                                        except Exception as e:
                                                logger.error(f"Failed to rename {full_path} to {new_full_path}: {e}")
                                                all_folders.append(name)  # fallback to original name
                                else:
                                        all_folders.append(clean_name)
        with open(output_txt, 'w') as f:
                for folder in all_folders:
                        f.write(folder + '\n')
        logger.info(f"Wrote {len(all_folders)} folder names to {output_txt}")

if __name__ == "__main__":
    list_cleaned_folder_names([path_1, path_2], 'timeline_processed.txt')
    # list_cleaned_folder_names([path_test], 'test_folder_names.txt')