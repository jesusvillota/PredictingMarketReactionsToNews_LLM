"""
Options Pedro - SPX Options Trading Analysis Package
"""

from .config.logger import get_logger

# # Lazy import config to avoid immediate initialization
# def get_config():
#     from .config.config import get_config as _get_config
#     return _get_config()

__version__ = "0.1.0"
__author__ = "Jesus Villota Miranda"
__description__ = "SPX Options Trading Analysis Package"
__all__ = ["get_logger"]