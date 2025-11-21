"""Configuration settings for the news market analysis project."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class Settings:
    """Central configuration management for the project.

    This class loads configuration from config.yaml and provides
    type-safe access to configuration values.

    Attributes:
        config_path: Path to the config.yaml file
        base_path: Base directory for the project
        config: Dictionary containing all configuration values
    """

    def __init__(
        self, config_path: Optional[Path] = None, base_path: Optional[Path] = None
    ) -> None:
        """Initialize settings.

        Args:
            config_path: Optional path to config.yaml. If not provided, will search
                        standard locations.
            base_path: Optional base path for the project. If not provided, will use
                      the parent directory of config_path or current working directory.

        Raises:
            ConfigurationError: If config file cannot be found or loaded
        """
        self.config_path = config_path if config_path else self._find_config_file()
        self.base_path = base_path if base_path else self.config_path.parent
        self.config = self._load_config()

    def _find_config_file(self) -> Path:
        """Find config.yaml in standard locations.

        Searches in the following order:
        1. Current working directory
        2. Master_Thesis-Replication_Package-main subdirectory
        3. Parent directories up to 3 levels

        Returns:
            Path to config.yaml

        Raises:
            ConfigurationError: If config.yaml cannot be found
        """
        search_paths = [
            Path.cwd() / "config.yaml",
            Path.cwd() / "Master_Thesis-Replication_Package-main" / "config.yaml",
        ]

        # Also search parent directories
        current = Path.cwd()
        for _ in range(3):
            current = current.parent
            search_paths.append(current / "config.yaml")
            search_paths.append(
                current / "Master_Thesis-Replication_Package-main" / "config.yaml"
            )

        for path in search_paths:
            if path.exists():
                return path

        raise ConfigurationError(
            f"config.yaml not found. Searched locations:\n"
            + "\n".join(f"  - {p}" for p in search_paths)
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Returns:
            Dictionary containing configuration

        Raises:
            ConfigurationError: If config file cannot be loaded or is invalid
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ConfigurationError("Config file must contain a dictionary")
            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Supports nested keys using dot notation (e.g., 'directories.raw_data').

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> settings.get('directories.raw_data')
            'data/raw'
            >>> settings.get('nonexistent', 'default_value')
            'default_value'
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_directories(self) -> Dict[str, str]:
        """Get all directory configurations.

        Returns:
            Dictionary mapping directory names to paths
        """
        return self.get("directories", {})


# Global settings instance (lazy-loaded)
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[Path] = None) -> Settings:
    """Get the global settings instance.

    Args:
        config_path: Optional path to config.yaml. Only used on first call.

    Returns:
        Global Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings(config_path)
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance. Useful for testing."""
    global _settings
    _settings = None
