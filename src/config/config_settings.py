"""Configuration settings loaded from config.yaml."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict

PROJECT_NAME: str = "predicting-market-reactions-news"

# Get project root directory (parent of src/)
_project_root = Path(__file__).parent.parent.parent
_config_path = _project_root / "config.yaml"


def load_config(config_path: Path = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default location.
    
    Returns:
        Dictionary containing configuration settings.
    """
    if config_path is None:
        config_path = _config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


# Load configuration
_config = load_config()

# Extract configuration sections
directories: Dict[str, str] = _config.get("directories", {})
logging_config: Dict[str, Any] = _config.get("logging", {})
embedding_config: Dict[str, Any] = _config.get("embedding", {})
clustering_config: Dict[str, Any] = _config.get("clustering", {})
data_config: Dict[str, Any] = _config.get("data", {})

# Logging settings
logging: Dict[str, Any] = {
    "level": logging_config.get("level", "INFO"),
    "console_output": logging_config.get("console_output", True),
    "log_file": Path(logging_config.get("log_file", "logs/predicting-market-reactions.log"))
}

# Ensure log file path is relative to project root
if not logging["log_file"].is_absolute():
    logging["log_file"] = _project_root / logging["log_file"]

