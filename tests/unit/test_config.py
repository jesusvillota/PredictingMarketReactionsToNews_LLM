"""Tests for configuration module."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml

from PMRTN.config import (
    ConfigurationError,
    PathManager,
    Settings,
    get_path_manager,
    get_settings,
    reset_path_manager,
    reset_settings,
)


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary config file for testing."""
    config_data = {
        "directories": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "output_kmeans": "output/kmeans",
            "output_llama": "output/llama",
            "output_data_description": "output/descriptives",
        }
    }

    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    yield config_path

    # Cleanup
    reset_settings()
    reset_path_manager()


class TestSettings:
    """Tests for Settings class."""

    def test_settings_load_valid_config(self, temp_config_file: Path) -> None:
        """Test loading a valid configuration file."""
        settings = Settings(config_path=temp_config_file)

        assert settings.config is not None
        assert "directories" in settings.config
        assert settings.get("directories.raw_data") == "data/raw"

    def test_settings_missing_config(self) -> None:
        """Test error when config file doesn't exist."""
        with pytest.raises(ConfigurationError):
            Settings(config_path=Path("/nonexistent/config.yaml"))

    def test_settings_get_with_default(self, temp_config_file: Path) -> None:
        """Test getting config value with default."""
        settings = Settings(config_path=temp_config_file)

        assert settings.get("nonexistent.key", "default") == "default"
        assert settings.get("directories.raw_data", "default") == "data/raw"

    def test_settings_get_nested_keys(self, temp_config_file: Path) -> None:
        """Test accessing nested configuration keys."""
        settings = Settings(config_path=temp_config_file)

        # Test nested access
        assert settings.get("directories.raw_data") == "data/raw"
        assert settings.get("directories.processed_data") == "data/processed"

        # Test non-existent nested key
        assert settings.get("directories.nonexistent") is None
        assert settings.get("directories.nonexistent", "default") == "default"

    def test_settings_get_directories(self, temp_config_file: Path) -> None:
        """Test getting all directories."""
        settings = Settings(config_path=temp_config_file)

        dirs = settings.get_directories()
        assert isinstance(dirs, dict)
        assert "raw_data" in dirs
        assert "processed_data" in dirs

    def test_settings_invalid_yaml(self, temp_dir: Path) -> None:
        """Test error with invalid YAML."""
        invalid_config = temp_dir / "invalid.yaml"
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            Settings(config_path=invalid_config)

    def test_get_settings_singleton(self, temp_config_file: Path) -> None:
        """Test that get_settings returns same instance."""
        reset_settings()

        settings1 = get_settings(temp_config_file)
        settings2 = get_settings()

        assert settings1 is settings2

    def test_reset_settings(self, temp_config_file: Path) -> None:
        """Test resetting settings singleton."""
        settings1 = get_settings(temp_config_file)
        reset_settings()
        settings2 = get_settings(temp_config_file)

        assert settings1 is not settings2


class TestPathManager:
    """Tests for PathManager class."""

    def test_path_manager_initialization(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test PathManager initialization."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        assert pm.base_path == temp_dir
        assert pm.settings is settings
        assert len(pm.paths) > 0
        assert pm.get("raw_data") == temp_dir / "data/raw"

    def test_path_manager_get(self, temp_config_file: Path, temp_dir: Path) -> None:
        """Test getting paths by key."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        raw_path = pm.get("raw_data")
        assert raw_path == temp_dir / "data/raw"

        processed_path = pm.get("processed_data")
        assert processed_path == temp_dir / "data/processed"

    def test_path_manager_get_invalid_key(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting invalid path key."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        with pytest.raises(KeyError, match="Path 'nonexistent' not found"):
            pm.get("nonexistent")

    def test_path_manager_get_raw_data_path(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting raw data path."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        raw_path = pm.get_raw_data_path()
        assert raw_path == temp_dir / "data/raw"

    def test_path_manager_get_processed_data_path(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting processed data path."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        processed_path = pm.get_processed_data_path()
        assert processed_path == temp_dir / "data/processed"

    def test_path_manager_get_output_path(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting output paths."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        kmeans_path = pm.get_output_path("kmeans")
        assert kmeans_path == temp_dir / "output/kmeans"

        llama_path = pm.get_output_path("llama")
        assert llama_path == temp_dir / "output/llama"

    def test_path_manager_create_directories(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test directory creation."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        # Directories shouldn't exist yet
        assert not pm.exists("raw_data")
        assert not pm.exists("processed_data")

        # Create directories
        pm.create_directories()

        # Now they should exist
        assert pm.exists("raw_data")
        assert pm.exists("processed_data")
        assert pm.get("raw_data").exists()

    def test_path_manager_exists(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test checking if directories exist."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        # Initially doesn't exist
        assert not pm.exists("raw_data")

        # Create it
        pm.get("raw_data").mkdir(parents=True)

        # Now it exists
        assert pm.exists("raw_data")

        # Non-existent key
        assert not pm.exists("nonexistent_key")

    def test_get_path_manager_singleton(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test that get_path_manager returns same instance."""
        reset_settings()
        reset_path_manager()

        # Initialize settings first
        get_settings(temp_config_file)

        pm1 = get_path_manager(temp_dir)
        pm2 = get_path_manager()

        assert pm1 is pm2

    def test_reset_path_manager(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test resetting path manager singleton."""
        reset_settings()
        reset_path_manager()

        get_settings(temp_config_file)

        pm1 = get_path_manager(temp_dir)
        reset_path_manager()
        pm2 = get_path_manager(temp_dir)

        assert pm1 is not pm2
