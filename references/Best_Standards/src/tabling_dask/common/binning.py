"""Binning utilities for creating standard size categories.

Provides centralized functions for creating the standard 4-size-bin pattern
used across multiple table scripts.
"""

from __future__ import annotations

from typing import Dict
import dask.dataframe as dd
import pandas as pd


def create_standard_size_bins(
    ddf: dd.DataFrame, 
    size_col: str = 'prtSize_agg'
) -> Dict[str, dd.DataFrame]:
    """Create standard 4-size-bin categories: all, 1-10, 11-200, >200.
    
    Parameters:
    -----------
    ddf : dd.DataFrame
        Input Dask DataFrame
    size_col : str, default='prtSize_agg'
        Name of the column containing size values
        
    Returns:
    --------
    Dict[str, dd.DataFrame]
        Dictionary with keys: 'all', '1_10', '11_200', 'over_200'
        Each value is a filtered Dask DataFrame
        
    Bin boundaries:
    - 'all': all data (after dropna on size_col)
    - '1_10': size between 1 and 10 (inclusive on both ends)
    - '11_200': size between 11 and 200 (inclusive on both ends)
    - 'over_200': size > 200
    """
    # Drop NA values for size column
    ddf_clean = ddf.dropna(subset=[size_col])
    
    return {
        'all': ddf_clean,
        '1_10': ddf_clean[ddf_clean[size_col].between(1, 10, inclusive='both')],
        '11_200': ddf_clean[ddf_clean[size_col].between(11, 200, inclusive='both')],
        'over_200': ddf_clean[ddf_clean[size_col] > 200]
    }

