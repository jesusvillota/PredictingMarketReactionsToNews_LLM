"""Path management utilities."""

import os
from pathlib import Path
from typing import Dict
from . import config_settings


class PathManager:
    """Manages project directory paths."""
    
    def __init__(self, base_path: Path = None):
        """
        Initialize path manager.
        
        Args:
            base_path: Base path of the project. If None, uses project root.
        """
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)
        
        self.config = config_settings.directories
        self.paths = self._initialize_paths()
    
    def _initialize_paths(self) -> Dict[str, Path]:
        """Initialize all paths from configuration."""
        paths = {}
        for key, directory in self.config.items():
            paths[key] = self.base_path / directory
        return paths
    
    def get_path(self, key: str) -> Path:
        """
        Get path for a given key.
        
        Args:
            key: Configuration key for the path
        
        Returns:
            Path object
        """
        return self.paths.get(key)
    
    def create_directories(self):
        """Create all directories specified in configuration."""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def get_raw_data_path(self) -> Path:
        """Get path to raw data directory."""
        return self.get_path("raw_data")
    
    def get_processed_data_path(self) -> Path:
        """Get path to processed data directory."""
        return self.get_path("processed_data")
    
    def get_output_path(self, output_type: str = None) -> Path:
        """
        Get path to output directory.
        
        Args:
            output_type: Type of output ('descriptives', 'kmeans', 'llama'). 
                        If None, returns base output directory.
        
        Returns:
            Path to output directory
        """
        if output_type == "descriptives":
            return self.get_path("output_data_description")
        elif output_type == "kmeans":
            return self.get_path("output_kmeans")
        elif output_type == "llama":
            return self.get_path("output_llama")
        else:
            # Return parent output directory
            return self.base_path / "output"


def get_paths(base_path: Path = None) -> PathManager:
    """
    Get a PathManager instance.
    
    Args:
        base_path: Base path of the project. If None, uses project root.
    
    Returns:
        PathManager instance
    """
    return PathManager(base_path)

