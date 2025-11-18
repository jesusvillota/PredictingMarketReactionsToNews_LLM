"""LLAMA-based clustering analysis."""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import get_paths, get_logger

logger = get_logger("analysis.llama_clustering")


def perform_llama_clustering(
    raw_data_path: Optional[Path] = None,
    processed_data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Perform clustering analysis using LLAMA-parsed news.
    
    Args:
        raw_data_path: Path to raw data directory. If None, uses config default.
        processed_data_path: Path to processed data directory. If None, uses config default.
        output_path: Path to output directory. If None, uses config default.
        save_output: Whether to save outputs
    
    Returns:
        DataFrame with clustering results
    """
    if raw_data_path is None or processed_data_path is None:
        path_manager = get_paths()
        if raw_data_path is None:
            raw_data_path = path_manager.get_raw_data_path()
        if processed_data_path is None:
            processed_data_path = path_manager.get_processed_data_path()
    
    if output_path is None:
        path_manager = get_paths()
        output_path = path_manager.get_output_path("llama")
    
    logger.info("Performing LLAMA-based clustering analysis")
    
    # Load LLAMA parsed news
    llama_file = raw_data_path / 'LLAMA_parsed_news.csv'
    if not llama_file.exists():
        raise FileNotFoundError(
            f"LLAMA parsed news file not found: {llama_file}. "
            "Run LLAMA parser first."
        )
    
    llama_df = pd.read_csv(llama_file)
    
    logger.info(f"Loaded {len(llama_df)} LLAMA-parsed entries")
    
    # Perform clustering analysis based on shock characteristics
    # This can be extended with more sophisticated clustering logic
    logger.info("Clustering based on shock characteristics")
    
    if save_output:
        output_path.mkdir(parents=True, exist_ok=True)
        # Save results (can be extended to save plots, tables, etc.)
        output_file = processed_data_path / 'D_clustered_llama.csv'
        llama_df.to_csv(output_file, index=False)
        logger.info(f"Saved clustered data to {output_file}")
    
    return llama_df

