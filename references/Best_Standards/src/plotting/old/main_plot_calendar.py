from pathlib import Path
from logging import Logger
from src import get_logger
logger: Logger = get_logger(__name__)

# zip_dir: Path = Path("tbloptionprintsethist_EQT")
zip_dir: Path = Path("/Users/jesusvillotamiranda/Desktop/LOCAL_DATASETS/SPIDERROCKdata/zip_files")
# zip_dir = Path("/Volumes/LaCie/tbloptionprintsethist_EQT")

if not zip_dir.exists():
    logger.error(f"Directory {str(zip_dir)} does not exist")
    # return

# Find all zip files in the directory
zip_files = list(zip_dir.glob('*.zip'))

if not zip_files:
    logger.warning(f"No zip files found in {str(zip_dir)}")
    # return

logger.info(f"Found {len(zip_files)} zip files to process")

from src.data.extract_date import extract_dates_from_filenames
dates, valid_files = extract_dates_from_filenames(directory_path=zip_dir)


from src.plotting.plotting_2 import visualize_file_dates, create_github_calendar, plot_github_like_calendar
visualize_file_dates(dates)
create_github_calendar(dates)
plot_github_like_calendar(dates)