# src/config/logger.py
import logging
import sys
from pathlib import Path
from colorama import init, Fore, Style  # Removed Back if not used
from typing import Optional
from . import config_settings
from dask.distributed import WorkerPlugin  # Added WorkerPlugin

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
            # Create a copy of the record to avoid modifying the original
            import copy
            record_copy = copy.copy(record)
            color = self.COLORS.get(record_copy.levelname, "")
            record_copy.levelname = color + record_copy.levelname + Style.RESET_ALL
            record_copy.msg = color + str(record_copy.msg) + Style.RESET_ALL
            return super().format(record_copy)
        return super().format(record)

def setup_logger(
    name: str = config_settings.PROJECT_NAME,
    level: str = config_settings.logging["level"],
    log_file: Optional[Path] = config_settings.logging["log_file"],
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output. Call once in main script.
    
    Args:
        name: Logger name (default: 'whales')
        level: Logging level (e.g., 'DEBUG', 'INFO')
        log_file: Path to log file (relative to project root or absolute)
        console_output: Whether to log to console
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.handlers.clear()  # Always start fresh to avoid duplicates

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with colors (only if tty)
    if console_output:
        # Create a UTF-8 encoded stream wrapper for Windows compatibility
        import io
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))  # Set handler level
        console_formatter = ColorFormatter(
            "[%(asctime)s] - [%(processName)s] - [%(name)s] - [%(levelname)s] - [%(message)s]",  # Added %(processName)s
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=sys.stdout.isatty(),
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler without colors
    if log_file:
        # Resolve relative path to project root
        log_path = Path(log_file)
        if not log_path.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            log_path = project_root / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))  # Set handler level
        file_formatter = logging.Formatter(
            "[%(asctime)s] - [%(processName)s] - [%(name)s] - [%(levelname)s] - [%(message)s]",  # Added %(processName)s
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger. Assumes setup_logger was called for 'whales'.
    
    Args:
        name: Logger name (e.g., 'config' for 'whales.config'). If None, returns 'whales'.
    
    Returns:
        Logger instance
    """
    root = logging.getLogger(project_name)
    logger = root if name is None else logging.getLogger(f"whales.{name}")
    if name is not None and not root.handlers:
        logging.getLogger().warning(f"Logger {logger.name} accessed before setup")
    return logger


# Dask WorkerPlugin for logging setup in workers (console only)
class LoggingPlugin(WorkerPlugin):
    def __init__(self):
        self.level = config_settings.logging["level"]
        self.log_file = config_settings.logging["log_file"]
        self.name = "LoggingPlugin"

    def setup(self, worker):
        setup_logger(
            name=config_settings.PROJECT_NAME,
            level=self.level,
            log_file=self.log_file,  # Set to None to avoid file logging in workers if desired
            console_output=True
        )
        from src import get_logger
        logger = get_logger(__name__)
        logger.info(f"LoggingPlugin setup completed for worker {worker.id}")
