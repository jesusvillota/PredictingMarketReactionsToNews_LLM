"""
Centralized DuckDB connection/configuration manager.

Usage:
    from src.config.duckdb_manager import DuckDBManager

    manager = DuckDBManager()
    con = manager.connect()  # returns a configured duckdb connection
"""

from pathlib import Path
import duckdb
from .config_settings import RAM_LIMIT, CPU_LIMIT, TEMP_DIR
from .logger import get_logger


class DuckDBManager:
    """
    Manage DuckDB connection and common configuration flags.
    """

    def __init__(
        self,
        ram_limit_gb: int = RAM_LIMIT,
        cpu_limit: int = CPU_LIMIT,
        temp_directory: Path = TEMP_DIR,
        enable_progress_bar: bool = True,
        preserve_insertion_order: bool = False,
        enable_object_cache: bool = False,
    ) -> None:
        self.ram_limit_gb = ram_limit_gb
        self.cpu_limit = cpu_limit
        self.temp_directory = Path(temp_directory)
        self.enable_progress_bar = enable_progress_bar
        self.preserve_insertion_order = preserve_insertion_order
        self.enable_object_cache = enable_object_cache
        self._logger = get_logger(__name__)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Create a DuckDB connection and apply standard configuration.
        """
        self.temp_directory.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        self.configure(con)
        return con

    def configure(self, con: duckdb.DuckDBPyConnection) -> None:
        """
        Apply standard settings to an existing connection.
        """
        con.execute(f"SET memory_limit='{self.ram_limit_gb}GB'")
        con.execute(f"SET temp_directory='{self.temp_directory}'")
        con.execute(f"SET threads={self.cpu_limit}")
        con.execute(f"SET enable_progress_bar={'true' if self.enable_progress_bar else 'false'}")
        con.execute(f"SET preserve_insertion_order={'true' if self.preserve_insertion_order else 'false'}")
        con.execute(f"SET enable_object_cache={'true' if self.enable_object_cache else 'false'}")
        self._logger.info(
            f"Configured DuckDB (RAM={self.ram_limit_gb}GB, threads={self.cpu_limit}, temp='{self.temp_directory}')"
        )


