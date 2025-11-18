"""
Main initialization and setup utilities for all main scripts.
"""
# from src.config.config import get_config
from .logger import setup_logger, get_logger
from .utils import delete_pycache
from . import config_settings


def initialize_main():
    """
    Initialize common setup for all main scripts.
    
    This function handles:
    - Configuration loading and directory creation
    - Logger setup
    - Cache cleanup
    - Resource limit validation
    
    Returns:
        tuple: (config, logger)
    """
    # Load config
    # config = get_config()
    # config.create_output_dirs()  # Main process only - creates directories including logs
    
    # Ensure log directory exists
    config_settings.logging["log_file"].parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logger AFTER directories are created to avoid file handle issues
    setup_logger(
        name=config_settings.PROJECT_NAME,
        level=config_settings.logging["level"],
        log_file=config_settings.logging["log_file"],
        console_output=config_settings.logging["console_output"]
    )
    
    # Main logger (after setup)
    logger = get_logger(__name__)
    
    # Clean up __pycache__ directories before starting
    delete_pycache()
    
    # Validate resource limits
    _validate_resource_limits(logger)
    
    # return config, logger
    return logger


def _validate_resource_limits(logger):
    """
    Validate CPU and RAM limits from configuration.
    
    Args:
        logger: Logger instance for reporting
        
    Raises:
        ValueError: If RAM_LIMIT is less than CPU_LIMIT
    """
    RAM_LIMIT = config_settings.RAM_LIMIT
    CPU_LIMIT = config_settings.CPU_LIMIT
    logger.info(f"CPU_LIMIT: {CPU_LIMIT}, RAM_LIMIT: {RAM_LIMIT}GB")
    
    if RAM_LIMIT < CPU_LIMIT:
        logger.error(f"CPU_LIMIT: {CPU_LIMIT} is less than RAM_LIMIT: {RAM_LIMIT}GB. Set RAM_LIMIT >= CPU_LIMIT")
        raise ValueError("RAM_LIMIT must be greater than or equal to CPU_LIMIT")