"""
Utilities package for common main script initialization and helper functions.

This package provides:
- initialize_main: Complete setup for main scripts (config, logging, cleanup, dask)
- delete_pycache: Cache cleanup utility
- DaskManager: Dask configuration and client lifecycle management
"""

from .setup import initialize_main
from .logger import get_logger
from .utils import delete_pycache
from .dask_manager import DaskManager
from .dask_manager_adaptive import AdaptiveDaskManager

__all__ = ['initialize_main', 'get_logger', 'delete_pycache', 'DaskManager', 'AdaptiveDaskManager']