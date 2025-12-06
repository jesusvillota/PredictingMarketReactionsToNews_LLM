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


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_configuration_error_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test error")

    def test_configuration_error_message_preserved(self) -> None:
        """Test exception message is preserved."""
        error_msg = "Configuration file not found"
        try:
            raise ConfigurationError(error_msg)
        except ConfigurationError as e:
            assert str(e) == error_msg


class TestSettings:
    """Tests for Settings class."""

    def test_settings_load_valid_config(self, temp_config_file: Path) -> None:
        """Test loading a valid configuration file."""
        settings = Settings(config_path=temp_config_file)

        assert settings.config is not None
        assert "directories" in settings.config
        assert settings.get("directories.raw_data") == "data/raw"

    def test_settings_init_with_base_path(self, temp_config_file: Path, temp_dir: Path) -> None:
        """Test initialization with base_path provided."""
        base_path = temp_dir / "custom_base"
        settings = Settings(config_path=temp_config_file, base_path=base_path)
        
        assert settings.base_path == base_path

    def test_settings_init_without_config_path(self, temp_dir: Path) -> None:
        """Test initialization without config_path (auto-discovery)."""
        # Create config in current directory
        config_path = Path.cwd() / "config.yaml"
        config_data = {"directories": {"raw_data": "data/raw"}}
        
        try:
            with open(config_path, "w") as f:
                yaml.dump(config_data, f)
            
            settings = Settings()
            assert settings.config_path == config_path
        finally:
            # Cleanup
            if config_path.exists():
                config_path.unlink()

    def test_settings_missing_config(self) -> None:
        """Test error when config file doesn't exist."""
        with pytest.raises(ConfigurationError):
            Settings(config_path=Path("/nonexistent/config.yaml"))

    def test_settings_invalid_yaml_format(self, temp_dir: Path) -> None:
        """Test error with invalid YAML format."""
        invalid_config = temp_dir / "invalid.yaml"
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            Settings(config_path=invalid_config)

    def test_settings_config_not_dict(self, temp_dir: Path) -> None:
        """Test error when config file is not a dictionary."""
        invalid_config = temp_dir / "not_dict.yaml"
        with open(invalid_config, "w") as f:
            f.write("- item1\n- item2\n")

        with pytest.raises(ConfigurationError, match="must contain a dictionary"):
            Settings(config_path=invalid_config)

    def test_settings_empty_config_file(self, temp_dir: Path) -> None:
        """Test handling of empty config file."""
        empty_config = temp_dir / "empty.yaml"
        empty_config.write_text("")
        
        # Empty YAML should result in None, which should raise error
        with pytest.raises(ConfigurationError):
            Settings(config_path=empty_config)

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

    def test_settings_get_empty_key(self, temp_config_file: Path) -> None:
        """Test getting value with empty key string."""
        settings = Settings(config_path=temp_config_file)
        
        # Empty key should return default
        assert settings.get("", "default") == "default"

    def test_settings_get_multiple_levels_nesting(self, temp_dir: Path) -> None:
        """Test accessing keys with multiple levels of nesting."""
        config_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep_value"
                    }
                }
            }
        }
        config_path = temp_dir / "nested_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        settings = Settings(config_path=config_path)
        assert settings.get("level1.level2.level3.value") == "deep_value"
        assert settings.get("level1.level2.level3.nonexistent") is None

    def test_settings_get_directories(self, temp_config_file: Path) -> None:
        """Test getting all directories."""
        settings = Settings(config_path=temp_config_file)

        dirs = settings.get_directories()
        assert isinstance(dirs, dict)
        assert "raw_data" in dirs
        assert "processed_data" in dirs

    def test_settings_get_directories_missing_key(self, temp_dir: Path) -> None:
        """Test get_directories when 'directories' key is missing."""
        config_data = {"other_key": "value"}
        config_path = temp_dir / "no_dirs_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        settings = Settings(config_path=config_path)
        dirs = settings.get_directories()
        assert dirs == {}

    def test_get_settings_singleton(self, temp_config_file: Path) -> None:
        """Test that get_settings returns same instance."""
        reset_settings()

        settings1 = get_settings(temp_config_file)
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_with_config_path_on_first_call(self, temp_config_file: Path) -> None:
        """Test get_settings with config_path on first call."""
        reset_settings()
        
        settings = get_settings(temp_config_file)
        assert settings.config_path == temp_config_file

    def test_reset_settings(self, temp_config_file: Path) -> None:
        """Test resetting settings singleton."""
        settings1 = get_settings(temp_config_file)
        reset_settings()
        settings2 = get_settings(temp_config_file)

        assert settings1 is not settings2

    def test_reset_settings_followed_by_get_settings(self, temp_config_file: Path) -> None:
        """Test reset_settings followed by get_settings creates new instance."""
        reset_settings()
        
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

    def test_path_manager_init_without_arguments(
        self, temp_config_file: Path
    ) -> None:
        """Test initialization without arguments (uses global settings)."""
        reset_settings()
        get_settings(temp_config_file)
        
        pm = PathManager()
        assert pm.settings is not None
        assert pm.base_path is not None

    def test_path_manager_init_with_settings_none(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test initialization with settings=None (should use global)."""
        reset_settings()
        get_settings(temp_config_file)
        
        pm = PathManager(base_path=temp_dir, settings=None)
        assert pm.settings is not None
        assert pm.base_path == temp_dir

    def test_path_manager_initialize_paths(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test _initialize_paths creates paths dictionary from settings."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        assert isinstance(pm.paths, dict)
        assert len(pm.paths) > 0
        # Check that paths are absolute
        for path in pm.paths.values():
            assert path.is_absolute()

    def test_path_manager_initialize_paths_empty_directories(
        self, temp_dir: Path
    ) -> None:
        """Test _initialize_paths with no directories in settings."""
        config_data = {}  # No directories key
        config_path = temp_dir / "empty_dirs_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        settings = Settings(config_path=config_path)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        assert pm.paths == {}

    def test_path_manager_get(self, temp_config_file: Path, temp_dir: Path) -> None:
        """Test getting paths by key."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)

        raw_path = pm.get("raw_data")
        assert raw_path == temp_dir / "data/raw"
        assert isinstance(raw_path, Path)
        assert raw_path.is_absolute()

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

    def test_path_manager_get_empty_key(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting path with empty key string."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        with pytest.raises(KeyError):
            pm.get("")

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
        
        descriptives_path = pm.get_output_path("data_description")
        assert descriptives_path == temp_dir / "output/descriptives"

    def test_path_manager_get_output_path_invalid(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting output path with invalid output_type."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        with pytest.raises(KeyError):
            pm.get_output_path("invalid_type")

    def test_path_manager_get_output_path_empty(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test getting output path with empty output_type."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        with pytest.raises(KeyError):
            pm.get_output_path("")

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

    def test_path_manager_create_directories_already_exist(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test directory creation when directories already exist."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        # Create directories first time
        pm.create_directories()
        assert pm.exists("raw_data")
        
        # Create again (should not error)
        pm.create_directories()
        assert pm.exists("raw_data")

    def test_path_manager_create_directories_no_directories(
        self, temp_dir: Path
    ) -> None:
        """Test create_directories when no directories configured."""
        config_data = {}
        config_path = temp_dir / "no_dirs_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        settings = Settings(config_path=config_path)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        # Should not raise error
        pm.create_directories()

    def test_path_manager_create_directories_verbose(
        self, temp_config_file: Path, temp_dir: Path, capsys
    ) -> None:
        """Test directory creation with verbose=True prints messages."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        pm.create_directories(verbose=True)
        
        captured = capsys.readouterr()
        assert "Created directory" in captured.out

    def test_path_manager_create_directories_not_verbose(
        self, temp_config_file: Path, temp_dir: Path, capsys
    ) -> None:
        """Test directory creation with verbose=False doesn't print."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        pm.create_directories(verbose=False)
        
        captured = capsys.readouterr()
        assert "Created directory" not in captured.out

    def test_path_manager_create_directories_nested(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test directory creation creates parent directories as needed."""
        # Modify config to have nested path
        config_data = {
            "directories": {
                "nested": "level1/level2/level3"
            }
        }
        config_path = temp_dir / "nested_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        settings = Settings(config_path=config_path)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        pm.create_directories()
        
        nested_path = pm.get("nested")
        assert nested_path.exists()
        assert nested_path.is_dir()

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

    def test_path_manager_exists_returns_false_for_invalid_key(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test exists returns False for invalid key (should not raise)."""
        settings = Settings(config_path=temp_config_file)
        pm = PathManager(base_path=temp_dir, settings=settings)
        
        # Should return False, not raise error
        assert pm.exists("nonexistent_key") is False

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

    def test_get_path_manager_multiple_calls(
        self, temp_config_file: Path, temp_dir: Path
    ) -> None:
        """Test multiple calls to get_path_manager return same instance."""
        reset_settings()
        reset_path_manager()
        
        get_settings(temp_config_file)
        
        pm1 = get_path_manager(temp_dir)
        pm2 = get_path_manager()
        pm3 = get_path_manager()
        
        assert pm1 is pm2 is pm3

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
