"""Data loading utilities for news and market data.

This module provides functions to load various data files used in the
news market analysis pipeline, including raw articles, processed data,
embeddings, and returns data.
"""

import ast
from pathlib import Path
from typing import Optional, Union

import pandas as pd


def load_raw_articles(
    data_path: Union[str, Path],
    filter_agenda: bool = True
) -> pd.DataFrame:
    """Load raw articles data from parquet file.
    
    Args:
        data_path: Path to the raw data file (ibex_sample.pqt.gziq)
        filter_agenda: Whether to filter out agenda articles (default: True)
    
    Returns:
        DataFrame with columns: publication_date, publication_datetime, title,
        snippet, body, word_count, company_codes_about, company_codes_about_ticker_exchange
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the data file is empty or has unexpected format
    
    Examples:
        >>> df = load_raw_articles('data/raw/ibex_sample.pqt.gziq')
        >>> print(df.shape)
        (1234, 8)
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Read the parquet file
    df = pd.read_parquet(data_path)
    
    if df.empty:
        raise ValueError(f"Data file is empty: {data_path}")
    
    # Convert publication_datetime from EPOCH to datetime
    df['publication_datetime'] = pd.to_datetime(df['publication_datetime'], unit='ms')
    
    # Create a formatted date string column
    df['publication_date_str'] = df['publication_datetime'].dt.strftime('%Y-%m-%d')
    
    # Sort by publication_datetime
    df = df.sort_values('publication_datetime').reset_index(drop=True)
    
    # Filter out agenda articles if requested
    if filter_agenda:
        df = filter_articles(df, filter_agenda=True)
    
    return df


def filter_articles(
    df: pd.DataFrame,
    filter_agenda: bool = True
) -> pd.DataFrame:
    """Filter articles based on content criteria.
    
    Args:
        df: DataFrame with article data
        filter_agenda: Whether to filter out political and economic agenda articles
    
    Returns:
        Filtered DataFrame
    
    Examples:
        >>> df_filtered = filter_articles(df, filter_agenda=True)
    """
    filtered = df.copy()
    
    if filter_agenda:
        # Filter out articles that are not referred to firms
        # Disregard news about political and economic agenda
        filtered = filtered[
            (filtered['company_codes_about'] != '') & 
            (filtered['title'] != 'España: Agenda política y económica -Semana')
        ].copy()
    
    return filtered


def load_processed_articles(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load processed articles with tickers from CSV.
    
    Args:
        data_path: Path to the processed data CSV file (D.csv)
    
    Returns:
        DataFrame with columns: publ_datetime, articles, tickers
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
    
    Examples:
        >>> df = load_processed_articles('data/processed/D.csv')
        >>> print(df.columns.tolist())
        ['publ_datetime', 'articles', 'tickers']
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert tickers from string representation to list
    if 'tickers' in df.columns:
        df['tickers'] = df['tickers'].apply(ast.literal_eval)
    
    # Convert datetime column
    if 'publ_datetime' in df.columns:
        df['publ_datetime'] = pd.to_datetime(df['publ_datetime'])
    
    required_columns = ['publ_datetime', 'articles', 'tickers']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df


def load_embeddings(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load articles with embeddings from CSV.
    
    Args:
        data_path: Path to the embeddings CSV file (D_embeddings.csv)
    
    Returns:
        DataFrame with columns: publ_datetime, articles, tickers, embeddings
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If required columns are missing
    
    Examples:
        >>> df = load_embeddings('data/processed/D_embeddings.csv')
        >>> print(df.shape)
        (1234, 4)
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert tickers from string representation to list
    if 'tickers' in df.columns:
        df['tickers'] = df['tickers'].apply(ast.literal_eval)
    
    # Convert datetime column
    if 'publ_datetime' in df.columns:
        df['publ_datetime'] = pd.to_datetime(df['publ_datetime'])
    
    # Convert embeddings from string representation to list (if needed)
    if 'embeddings' in df.columns and df['embeddings'].dtype == 'object':
        df['embeddings'] = df['embeddings'].apply(ast.literal_eval)
    
    return df


def load_returns_data(
    data_path: Union[str, Path],
    model: str = 'KMeans'
) -> pd.DataFrame:
    """Load returns data for specified model.
    
    Args:
        data_path: Path to the returns CSV file (R_KMeans.csv or R_LLAMA.csv)
        model: Model type ('KMeans' or 'LLAMA')
    
    Returns:
        DataFrame with returns data for the specified model
    
    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If model type is invalid
    
    Examples:
        >>> df = load_returns_data('data/processed/R_KMeans.csv', model='KMeans')
    """
    if model not in ['KMeans', 'LLAMA']:
        raise ValueError(f"Invalid model type: {model}. Must be 'KMeans' or 'LLAMA'")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    return df


def save_processed_data(
    df: pd.DataFrame,
    path: Union[str, Path],
    index: bool = False
) -> None:
    """Save processed data to CSV.
    
    Args:
        df: DataFrame to save
        path: Output file path
        index: Whether to include DataFrame index in output (default: False)
    
    Examples:
        >>> save_processed_data(df, 'data/processed/D.csv')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
