"""Path management for the news market analysis project."""

import os
from pathlib import Path
from typing import Dict, Optional

from .settings import Settings, get_settings


class PathManager:
    """Manages all directory paths for the project.

    This class centralizes path management and ensures all directories
    exist before use.

    Attributes:
        base_path: Base directory for the project
        settings: Settings instance containing directory configuration
        paths: Dictionary mapping logical names to absolute paths
    """

    def __init__(
        self, base_path: Optional[Path] = None, settings: Optional[Settings] = None
    ) -> None:
        """Initialize PathManager.

        Args:
            base_path: Optional base directory. If not provided, uses settings base_path.
            settings: Optional Settings instance. If not provided, uses global settings.
        """
        self.settings = settings if settings else get_settings()
        self.base_path = base_path if base_path else self.settings.base_path
        self.paths = self._initialize_paths()

    def _initialize_paths(self) -> Dict[str, Path]:
        """Initialize all paths from configuration.

        Returns:
            Dictionary mapping logical names to absolute Path objects
        """
        directories = self.settings.get_directories()
        paths = {}

        for key, rel_path in directories.items():
            paths[key] = self.base_path / rel_path

        return paths

    def get(self, key: str) -> Path:
        """Get a path by its logical name.

        Args:
            key: Logical name of the path (e.g., 'raw_data', 'processed_data')

        Returns:
            Absolute Path object

        Raises:
            KeyError: If key doesn't exist in configuration
        """
        if key not in self.paths:
            raise KeyError(f"Path '{key}' not found in configuration")
        return self.paths[key]

    def get_raw_data_path(self) -> Path:
        """Get path to raw data directory."""
        return self.get("raw_data")

    def get_processed_data_path(self) -> Path:
        """Get path to processed data directory."""
        return self.get("processed_data")

    def get_output_path(self, output_type: str) -> Path:
        """Get path to specific output directory.

        Args:
            output_type: Type of output ('kmeans', 'llama', 'descriptives')

        Returns:
            Absolute Path to output directory
        """
        key = f"output_{output_type}"
        return self.get(key)

    def create_directories(self, verbose: bool = False) -> None:
        """Create all configured directories if they don't exist.

        Args:
            verbose: If True, print messages about directory creation
        """
        for key, path in self.paths.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                if verbose:
                    print(f"Created directory: {path}")

    def exists(self, key: str) -> bool:
        """Check if a directory exists.

        Args:
            key: Logical name of the path

        Returns:
            True if directory exists, False otherwise
        """
        try:
            return self.get(key).exists()
        except KeyError:
            return False


# Global path manager instance
_path_manager: Optional[PathManager] = None


def get_path_manager(base_path: Optional[Path] = None) -> PathManager:
    """Get the global PathManager instance.

    Args:
        base_path: Optional base path. Only used on first call.

    Returns:
        Global PathManager instance
    """
    global _path_manager
    if _path_manager is None:
        _path_manager = PathManager(base_path)
    return _path_manager


def reset_path_manager() -> None:
    """Reset the global path manager instance. Useful for testing."""
    global _path_manager
    _path_manager = None
