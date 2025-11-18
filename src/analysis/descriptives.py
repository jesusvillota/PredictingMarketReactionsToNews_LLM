"""Generate descriptive statistics and visualizations."""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import get_paths, get_logger

logger = get_logger("analysis.descriptives")


def generate_descriptives(
    processed_data_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> dict:
    """
    Generate descriptive statistics for the dataset.
    
    Args:
        processed_data_path: Path to processed data directory. If None, uses config default.
        output_path: Path to output directory. If None, uses config default.
    
    Returns:
        Dictionary with descriptive statistics
    """
    if processed_data_path is None:
        path_manager = get_paths()
        processed_data_path = path_manager.get_processed_data_path()
    
    if output_path is None:
        path_manager = get_paths()
        output_path = path_manager.get_output_path("descriptives")
    
    logger.info("Generating descriptive statistics")
    
    # Load data
    D = pd.read_csv(processed_data_path / 'D.csv')
    
    # Basic statistics
    stats = {
        'total_articles': len(D),
        'articles_with_tickers': len(D[D['tickers'].apply(lambda x: len(eval(x)) > 0)]),
        'date_range': {
            'start': D['publ_datetime'].min(),
            'end': D['publ_datetime'].max()
        }
    }
    
    logger.info(f"Total articles: {stats['total_articles']}")
    logger.info(f"Articles with tickers: {stats['articles_with_tickers']}")
    
    # Save statistics (can be extended to save plots, tables, etc.)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return stats

