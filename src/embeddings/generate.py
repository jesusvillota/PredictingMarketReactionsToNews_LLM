"""Generate embeddings for articles using sentence transformers."""

import pandas as pd
import ast
from pathlib import Path
from typing import Optional, Dict
import torch
from sentence_transformers import SentenceTransformer

from src.config import get_paths, get_logger, config_settings
from src.data.utils import create_trading_calendar_adjustments

logger = get_logger("embeddings.generate")


# Model dictionary - loaded lazily
_model_dict: Optional[Dict[str, SentenceTransformer]] = None


def get_model_dict() -> Dict[str, SentenceTransformer]:
    """
    Get dictionary of available embedding models.
    
    Returns:
        Dictionary mapping model names to SentenceTransformer instances
    """
    global _model_dict
    
    if _model_dict is None:
        logger.info("Loading embedding models")
        _model_dict = {
            'paraphrase-MiniLM-L6-v2': SentenceTransformer('paraphrase-MiniLM-L6-v2'),
            'paraphrase-multilingual-MiniLM-L12-v2': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
            'distiluse-base-multilingual-cased-v1': SentenceTransformer('distiluse-base-multilingual-cased-v1'),
        }
        logger.info(f"Loaded {len(_model_dict)} embedding models")
    
    return _model_dict


def get_embedding(article: str, model_name: str) -> list:
    """
    Get embedding for a single article using the specified model.
    
    Args:
        article: Article text
        model_name: Name of the model to use
    
    Returns:
        List of embedding values
    """
    model_dict = get_model_dict()
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_dict.keys())}")
    
    model = model_dict[model_name]
    return model.encode(article).tolist()


def generate_embeddings(
    processed_data_path: Optional[Path] = None,
    model_name: Optional[str] = None,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Generate embeddings for all articles.
    
    Args:
        processed_data_path: Path to processed data directory. If None, uses config default.
        model_name: Name of the embedding model to use. If None, uses config default.
        save_output: Whether to save output to CSV
    
    Returns:
        DataFrame with embeddings added
    """
    if processed_data_path is None:
        path_manager = get_paths()
        processed_data_path = path_manager.get_processed_data_path()
    
    if model_name is None:
        model_name = config_settings.embedding_config.get(
            "model", 
            "distiluse-base-multilingual-cased-v1"
        )
    
    logger.info(f"Generating embeddings using model: {model_name}")
    
    # Load data
    D = pd.read_csv(processed_data_path / 'D.csv')
    D['publ_datetime'] = pd.to_datetime(D['publ_datetime'])
    D['tickers'] = D['tickers'].apply(lambda x: ast.literal_eval(x))
    
    # Load return data to create trading calendar adjustments
    R = pd.read_csv(processed_data_path / 'R_KMeans.csv', index_col=0, parse_dates=True)
    trading_days = [dt.date() if isinstance(dt, pd.Timestamp) else dt for dt in R.index]
    
    # Create trading calendar adjustments
    adj = create_trading_calendar_adjustments(R)
    
    # Add date_affect column
    D['date_affect'] = D['publ_datetime'].apply(adj.impute_date_affect)
    
    # Generate embeddings
    logger.info("Calculating embeddings for all articles...")
    D['embeddings'] = D['articles'].apply(lambda x: get_embedding(x, model_name))
    logger.info("Done generating embeddings")
    
    # Save output
    if save_output:
        output_file = processed_data_path / 'D_embeddings.csv'
        D.to_csv(output_file, index=False)
        logger.info(f"Saved embeddings to {output_file}")
    
    return D

