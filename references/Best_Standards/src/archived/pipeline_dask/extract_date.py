from src import get_logger
import re
from datetime import datetime
import os
from pathlib import Path

def extract_date_from_single_file(filename: str) -> tuple[datetime, bool]:
    """
    Extract date from a single filename with pattern: tbloptionprintsethist_EQT_v2.00_YYYY-MM-DD.zip
    
    Args:
        filename: String filename to extract date from
    
    Returns:
        tuple: (datetime object, bool) - (date_obj, success)
               Returns (None, False) if date extraction fails
    """
    logger = get_logger(__name__)
    
    # Regex pattern to extract YYYY-MM-DD from filename
    pattern = r'tbloptionprintsethist_EQT_v2\.00_(\d{4}-\d{2}-\d{2})\.zip'
    
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        try:
            date_obj: datetime = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj, True
        except ValueError:
            logger.warning(f"Invalid date format in file: {filename}")
            return None, False
    else:
        return None, False


def extract_dates_from_filenames(file_list: list[str] = None, directory_path: Path = None) -> tuple[list[datetime], list[str]]:
    """
    Extract dates from multiple filenames with pattern: tbloptionprintsethist_EQT_v2.00_YYYY-MM-DD.zip
    
    Args:
        file_list: List of filenames (optional)
        directory_path: Path to directory containing files (optional)
    
    Returns:
        tuple: (List of datetime objects, List of valid filenames)
    """
    logger = get_logger(__name__)
    
    if file_list is None and directory_path is None:
        raise ValueError("Either file_list or directory_path must be provided")
    
    if directory_path:
        file_list = [f for f in os.listdir(directory_path) if f.endswith('.zip')]
    
    # Ensure file_list is not None at this point
    if file_list is None:
        file_list = []
    
    dates = []
    valid_files = []
    
    for filename in file_list:
        date_obj, success = extract_date_from_single_file(filename)
        if success:
            dates.append(date_obj)
            valid_files.append(filename)
    
    # Log statistics if dates were found
    if dates:
        import pandas as pd
        df = pd.DataFrame({'date': dates})
        df['year'] = df['date'].dt.year
        logger.info(f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}")
        logger.info(f"Time span: {(max(dates) - min(dates)).days} days")
        logger.info(f"Average files per year: {len(dates) / len(set(df['year'])):.1f}")
    else:
        logger.warning("No valid dates found.")

    return dates, valid_files
