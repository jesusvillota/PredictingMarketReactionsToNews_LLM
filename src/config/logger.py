"""Logging setup with colored console output."""

import logging
import sys
from pathlib import Path
from colorama import init, Fore, Style
from typing import Optional
from . import config_settings

project_name: str = config_settings.PROJECT_NAME

init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }
    
    def __init__(self, *args, use_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record):
        if self.use_color:
            import copy
            record_copy = copy.copy(record)
            color = self.COLORS.get(record_copy.levelname, "")
            record_copy.levelname = color + record_copy.levelname + Style.RESET_ALL
            record_copy.msg = color + str(record_copy.msg) + Style.RESET_ALL
            return super().format(record_copy)
        return super().format(record)


def setup_logger(
    name: str = config_settings.PROJECT_NAME,
    level: str = None,
    log_file: Optional[Path] = None,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output. Call once in main script.
    
    Args:
        name: Logger name (default: project name)
        level: Logging level (e.g., 'DEBUG', 'INFO'). If None, uses config default.
        log_file: Path to log file. If None, uses config default.
        console_output: Whether to log to console
    
    Returns:
        Configured logger
    """
    if level is None:
        level = config_settings.logging["level"]
    if log_file is None:
        log_file = config_settings.logging["log_file"]
    
    logger = logging.getLogger(name)
    logger.handlers.clear()  # Always start fresh to avoid duplicates

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with colors (only if tty)
    if console_output:
        import io
        utf8_stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace', 
            line_buffering=True
        )
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_formatter = ColorFormatter(
            "[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=sys.stdout.isatty(),
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler without colors
    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            log_path = project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_formatter = logging.Formatter(
            "[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger. Assumes setup_logger was called for the project.
    
    Args:
        name: Logger name (e.g., 'data' for 'predicting-market-reactions-news.data'). 
              If None, returns root logger.
    
    Returns:
        Logger instance
    """
    root = logging.getLogger(project_name)
    if name is None:
        return root
    logger = logging.getLogger(f"{project_name}.{name}")
    if not root.handlers:
        logging.getLogger().warning(f"Logger {logger.name} accessed before setup")
    return logger

