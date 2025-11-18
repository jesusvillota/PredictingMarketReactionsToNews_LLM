"""Load and process article data from raw sources."""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import get_paths, get_logger
from src.utils.text_processing import (
    convert_to_datetime,
    extract_tickers_from_article,
    clean_article_text
)


logger = get_logger("data.load_articles")


def load_raw_articles(raw_data_path: Path, filename: str = "ibex_sample.pqt.gziq") -> pd.DataFrame:
    """
    Load raw article data from parquet file.
    
    Args:
        raw_data_path: Path to raw data directory
        filename: Name of the parquet file
    
    Returns:
        DataFrame with raw article data
    """
    filepath = raw_data_path / filename
    logger.info(f"Loading raw articles from {filepath}")
    
    df_full = pd.read_parquet(filepath)
    
    # Convert publication_datetime from EPOCH to YYYY-MM-DD
    df_full['publication_datetime'] = pd.to_datetime(
        df_full['publication_datetime'], 
        unit='ms'
    ).dt.strftime('%Y-%m-%d')
    
    # Sort by publication_date
    df_full = df_full.sort_values('publication_date')
    
    logger.info(f"Loaded {len(df_full)} articles")
    return df_full


def filter_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter articles to keep only those related to firms.
    
    Args:
        df: DataFrame with article data
    
    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering articles related to firms")
    
    df_filtered = df[
        (df['company_codes_about'] != '') & 
        (df['title'] != 'España: Agenda política y económica -Semana')
    ].copy()
    
    logger.info(f"Filtered to {len(df_filtered)} articles")
    return df_filtered


def merge_article_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge title, snippet, and body fields into a single document.
    
    Args:
        df: DataFrame with title, snippet, body columns
    
    Returns:
        DataFrame with merged articles and publication datetime
    """
    logger.info("Merging article fields")
    
    columns_to_keep = ['title', 'snippet', 'body']
    df_subset = df[columns_to_keep].copy()
    df_subset.fillna('', inplace=True)
    
    # Merge title, snippet, and body
    documents = (
        df_subset['title'] + '. ' + 
        df_subset['snippet'] + '. ' + 
        df_subset['body']
    )
    
    publ_datetime = df['publication_datetime'].copy().tolist()
    
    result = pd.DataFrame({
        'publ_datetime': publ_datetime,
        'articles': documents
    })
    
    return result


def process_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process articles: clean text and extract tickers.
    
    Args:
        df: DataFrame with articles
    
    Returns:
        DataFrame with cleaned articles and extracted tickers
    """
    logger.info("Processing articles: cleaning text and extracting tickers")
    
    docs_filtered = df.copy()
    
    # Clean article text
    docs_filtered['articles'] = docs_filtered['articles'].apply(clean_article_text)
    
    # Extract tickers
    docs_filtered['tickers'] = docs_filtered['articles'].apply(extract_tickers_from_article)
    
    # Filter articles with at least one ticker
    docs_n_tickers = docs_filtered[
        docs_filtered['tickers'].apply(len) > 0
    ].copy()
    
    logger.info(f"Found {len(docs_n_tickers)} articles with tickers")
    
    return docs_n_tickers


def load_and_process_articles(
    raw_data_path: Optional[Path] = None,
    processed_data_path: Optional[Path] = None,
    filename: str = "ibex_sample.pqt.gziq",
    save_output: bool = True
) -> pd.DataFrame:
    """
    Main function to load and process articles from raw data.
    
    Args:
        raw_data_path: Path to raw data directory. If None, uses config default.
        processed_data_path: Path to processed data directory. If None, uses config default.
        filename: Name of the raw data file
        save_output: Whether to save processed data to CSV
    
    Returns:
        Processed DataFrame with articles and tickers
    """
    if raw_data_path is None or processed_data_path is None:
        path_manager = get_paths()
        if raw_data_path is None:
            raw_data_path = path_manager.get_raw_data_path()
        if processed_data_path is None:
            processed_data_path = path_manager.get_processed_data_path()
    
    # Load raw data
    df_full = load_raw_articles(raw_data_path, filename)
    
    # Filter columns
    columns_to_keep = [
        'publication_date', 'title', 'snippet', 'body', 
        'word_count', 'company_codes_about', 'company_codes_about_ticker_exchange'
    ]
    df = df_full[columns_to_keep].copy()
    
    # Convert publication_date
    df['publication_datetime'] = df['publication_date'].apply(convert_to_datetime)
    df.drop(columns=['publication_date'], inplace=True)
    df['publication_date'] = df_full['publication_datetime']
    
    # Filter articles
    df_filtered = filter_articles(df)
    
    # Merge article fields
    documents = merge_article_fields(df_filtered)
    
    # Process articles
    docs_n_tickers = process_articles(documents)
    
    # Save to CSV
    if save_output:
        output_file = processed_data_path / 'D.csv'
        docs_n_tickers.to_csv(output_file, index=False)
        logger.info(f"Saved processed articles to {output_file}")
    
    return docs_n_tickers

