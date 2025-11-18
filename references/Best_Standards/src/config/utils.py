from pathlib import Path
from typing import Callable, Iterable, Sequence, Tuple, Union, List
import os
import shutil
from .logger import get_logger


class DailyFolderFilter:
    """
    Build parquet file globs from a directory containing daily subfolders
    named as YYYY-MM-DD.

    All methods return a list of glob strings, each pointing to
    "<daily_folder>/*.parquet" for every matched day.
    """

    def __init__(self, processed_path: Union[str, Path]) -> None:
        self.processed_path: Path = Path(processed_path)

    def by_year(self, years: Iterable[int], return_globs: bool = True) -> Union[List[str], List[Path]]:
        """
        Filter folders by year only (all months and days in those years).
        
        Args:
            years: Years to filter by
            return_globs: If True, return glob patterns like "path/*.parquet". 
                         If False, return raw folder paths as Path objects.
        """
        years_set = set(years)
        folders = self._filter(lambda y, m, d: y in years_set)
        return self._to_parquet_globs(folders) if return_globs else folders

    def by_month(self, months: Iterable[int], return_globs: bool = True) -> Union[List[str], List[Path]]:
        """
        Filter folders by month only (all years and days in those months).
        
        Args:
            months: Months to filter by (1-12)
            return_globs: If True, return glob patterns like "path/*.parquet". 
                         If False, return raw folder paths as Path objects.
        """
        months_set = set(months)
        folders = self._filter(lambda y, m, d: m in months_set)
        return self._to_parquet_globs(folders) if return_globs else folders

    def by_day(self, days: Iterable[int], return_globs: bool = True) -> Union[List[str], List[Path]]:
        """
        Filter folders by day of month only (all years and months with those days).
        
        Args:
            days: Days to filter by (1-31)
            return_globs: If True, return glob patterns like "path/*.parquet". 
                         If False, return raw folder paths as Path objects.
        """
        days_set = set(days)
        folders = self._filter(lambda y, m, d: d in days_set)
        return self._to_parquet_globs(folders) if return_globs else folders

    def by_year_month(self, year_months: Iterable[Union[str, Tuple[int, int]]], return_globs: bool = True) -> Union[List[str], List[Path]]:
        """
        Filter folders by specific year-month combinations (all days in those months).
        
        Args:
            year_months: Year-month combinations as strings "YYYY-MM" or tuples (year, month)
            return_globs: If True, return glob patterns like "path/*.parquet". 
                         If False, return raw folder paths as Path objects.
        """
        pairs: set[Tuple[int, int]] = set(self._normalize_year_month_iterable(year_months))
        folders = self._filter(lambda y, m, d: (y, m) in pairs)
        return self._to_parquet_globs(folders) if return_globs else folders

    def by_year_month_date(self, dates: Iterable[Union[str, Tuple[int, int, int]]], return_globs: bool = True) -> Union[List[str], List[Path]]:
        """
        Filter folders by exact dates (year-month-day combinations).
        
        Args:
            dates: Dates as strings "YYYY-MM-DD" or tuples (year, month, day)
            return_globs: If True, return glob patterns like "path/*.parquet". 
                         If False, return raw folder paths as Path objects.
        """
        triples: set[Tuple[int, int, int]] = set(self._normalize_year_month_date_iterable(dates))
        folders = self._filter(lambda y, m, d: (y, m, d) in triples)
        return self._to_parquet_globs(folders) if return_globs else folders

    def _to_parquet_globs(self, folders: Sequence[Path]) -> List[str]:
        return [str(folder / "*.parquet") for folder in folders]

    def _filter(self, predicate: Callable[[int, int, int], bool]) -> List[Path]:
        matched: List[Path] = []
        for folder in self._iter_daily_folders():
            y, m, d = self._parse_ymd(folder)
            if predicate(y, m, d):
                matched.append(folder)
        return matched

    def _iter_daily_folders(self) -> List[Path]:
        if not self.processed_path.exists():
            return []
        # Only include subdirectories that look like YYYY-MM-DD
        return sorted(
            [p for p in self.processed_path.iterdir() if p.is_dir() and self._looks_like_ymd(p.name)]
        )

    @staticmethod
    def _looks_like_ymd(name: str) -> bool:
        parts = name.split("-")
        if len(parts) != 3:
            return False
        try:
            y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return False
        return 1 <= m <= 12 and 1 <= d <= 31 and 1000 <= y <= 3000

    @staticmethod
    def _parse_ymd(folder: Path) -> Tuple[int, int, int]:
        y_str, m_str, d_str = folder.name.split("-")
        return int(y_str), int(m_str), int(d_str)

    @staticmethod
    def _normalize_year_month_iterable(items: Iterable[Union[str, Tuple[int, int]]]) -> List[Tuple[int, int]]:
        normalized: List[Tuple[int, int]] = []
        for it in items:
            if isinstance(it, tuple):
                y, m = it
            else:
                y_str, m_str = it.split("-")
                y, m = int(y_str), int(m_str)
            normalized.append((y, m))
        return normalized

    @staticmethod
    def _normalize_year_month_date_iterable(items: Iterable[Union[str, Tuple[int, int, int]]]) -> List[Tuple[int, int, int]]:
        normalized: List[Tuple[int, int, int]] = []
        for it in items:
            if isinstance(it, tuple):
                y, m, d = it
            else:
                y_str, m_str, d_str = it.split("-")
                y, m, d = int(y_str), int(m_str), int(d_str)
            normalized.append((y, m, d))
        return normalized


#############################################################################################
################################## IMPORTANT NOTE: ##########################################
#############################################################################################
#----------> globs are only read by 'dask' ('pandas' does not support globs) !!! <----------#
#############################################################################################
#############################################################################################

# Example usage
# from src.config.utils import DailyFolderFilter
# from src.config.config_settings import PROCESSED_PATH

# filter = DailyFolderFilter(PROCESSED_PATH)

# # All Januaries and Marches across all years (as globs)
# globs = filter.by_month([1,2,3])
# # Returns: [".../2021-01-29/*.parquet", ".../2021-02-01/*.parquet", ...]

# # All Januaries and Marches across all years (as folder paths)
# folders = filter.by_month([1,2,3], return_globs=False)
# # Returns: [Path(".../2021-01-29"), Path(".../2021-02-01"), ...]

# # All 1st days of every month and year
# globs = filter.by_day([29])

# # All days in 2021
# globs = filter.by_year([2021])

# # All days in Jan 2021 and Mar 2021 specifically (as globs)
# globs = filter.by_year_month(["2021-01", "2021-03"])

# # All days in Jan 2021 and Mar 2021 specifically (as folder paths)
# folders = filter.by_year_month(["2021-01", "2021-03"], return_globs=False)

# # Exact dates only
# globs = filter.by_year_month_date(["2021-01-01", "2021-01-02"])


def delete_pycache(root_dir: str = "src") -> None:
    """
    Recursively delete all __pycache__ directories from the given root directory (default: src).

    Args:
        root_dir: Root directory to search for __pycache__ directories
    """
    logger = get_logger(__name__)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(pycache_path)
            logger.info(f"Deleted: {pycache_path}")