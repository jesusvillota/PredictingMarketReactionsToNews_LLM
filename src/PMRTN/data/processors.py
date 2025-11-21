"""Data processing utilities for text and articles.

This module provides functions for processing and cleaning article text,
extracting information, and preparing data for analysis.
"""

import re
from datetime import datetime
from typing import List, Optional

import pandas as pd


def eliminate_text_after_word(text: str, word: str) -> str:
    """Remove all text after the first occurrence of a specific word.
    
    Args:
        text: Input text string
        word: Word after which to eliminate all text
    
    Returns:
        Text up to (but not including) the first occurrence of word,
        or original text if word is not found
    
    Examples:
        >>> eliminate_text_after_word("Hello world. Goodbye.", "Goodbye")
        'Hello world. '
        >>> eliminate_text_after_word("No match here", "missing")
        'No match here'
    """
    index = text.find(word)
    if index != -1:
        return text[:index]
    else:
        return text


def extract_datetime(text: str) -> Optional[str]:
    """Extract datetime string from text using regex pattern.
    
    Searches for datetime in format: 'DD-MM-YY HHMMGMT'
    
    Args:
        text: Text containing datetime string
    
    Returns:
        Extracted datetime string if found, None otherwise
    
    Examples:
        >>> extract_datetime("Published on 01-05-23 1430GMT by author")
        '01-05-23 1430GMT'
        >>> extract_datetime("No datetime here")
        None
    """
    datetime_pattern = r'\d{2}-\d{2}-\d{2} \d{4}GMT'
    datetime_match = re.search(datetime_pattern, text)
    if datetime_match:
        return datetime_match.group(0)
    else:
        return None


def convert_to_datetime(timestamp_ms: float) -> datetime:
    """Convert millisecond EPOCH timestamp to datetime.
    
    Args:
        timestamp_ms: Timestamp in milliseconds since EPOCH
    
    Returns:
        datetime object
    
    Examples:
        >>> dt = convert_to_datetime(1609459200000)  # 2021-01-01 00:00:00
        >>> dt.year
        2021
    """
    timestamp_seconds = timestamp_ms / 1000
    return datetime.fromtimestamp(timestamp_seconds)


def extract_tickers_from_article(article: str) -> List[str]:
    """Extract stock ticker symbols from article text.
    
    Searches for patterns like (TICKER.MC) where TICKER is any uppercase
    combination representing Spanish stock tickers on the Madrid exchange.
    
    Args:
        article: Article text
    
    Returns:
        List of unique ticker symbols found in the article
    
    Examples:
        >>> extract_tickers_from_article("Telefónica (TEF.MC) and Santander (SAN.MC) rose")
        ['TEF.MC', 'SAN.MC']
        >>> extract_tickers_from_article("No tickers mentioned here")
        []
    """
    pattern = r'\(([A-Z]+\.MC)\)'
    matches = re.findall(pattern, article)
    return list(set(matches))


def merge_article_components(
    df: pd.DataFrame,
    title_col: str = 'title',
    snippet_col: str = 'snippet',
    body_col: str = 'body',
    output_col: str = 'articles'
) -> pd.DataFrame:
    """Merge title, snippet, and body into single article text.
    
    Args:
        df: DataFrame containing article components
        title_col: Name of title column
        snippet_col: Name of snippet column
        body_col: Name of body column
        output_col: Name of output merged column
    
    Returns:
        DataFrame with new merged articles column
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'title': ['Title 1'],
        ...     'snippet': ['Snippet 1'],
        ...     'body': ['Body text 1']
        ... })
        >>> result = merge_article_components(df)
        >>> 'articles' in result.columns
        True
    """
    df = df.copy()
    
    # Handle empty DataFrame
    if len(df) == 0:
        df[output_col] = pd.Series([], dtype=str)
        return df
    
    # Fill NaN values with empty strings
    df[title_col] = df[title_col].fillna('')
    df[snippet_col] = df[snippet_col].fillna('')
    df[body_col] = df[body_col].fillna('')
    
    # Merge components with period separators
    df[output_col] = df[title_col] + '. ' + df[snippet_col] + '. ' + df[body_col]
    
    return df


def clean_article_text(article: str) -> str:
    """Clean article text by removing unwanted content.
    
    This function performs multiple cleaning operations:
    1. Removes text after certain phrases (e.g., "-Escriba a", "Editado por")
    2. Removes specific expressions (e.g., "MARKET TALK: ", city names)
    3. Removes email patterns and author attributions
    
    Args:
        article: Raw article text
    
    Returns:
        Cleaned article text
    
    Examples:
        >>> clean_article_text("News content -Escriba a author@email.com")
        'News content '
    """
    cleaned = article
    
    # Words after which to eliminate all text
    eliminate_after_words = [
        "-Escriba a",
        "Editado por",
        "(END)",
        "Versión española de",
        "Escriba a",
        "Traductores:"
    ]
    
    for word in eliminate_after_words:
        cleaned = eliminate_text_after_word(cleaned, word)
    
    # Specific expressions to remove
    expressions_to_remove = [
        'MARKET TALK: ',
        'MADRID',
        'BARCELONA',
        'LONDRES',
        'MÉXICO',
        'ROMA',
        'BRUSELAS',
        'FRÁNCFORT',
        'SÍDNEY',
        'PARÍS',
        'RÍO DE JANEIRO',
        '(EFE Dow Jones)--',
        '(EFE Dow Jones).-',
        '(EFE Dow Jones)',
        '(MORE TO FOLLOW)',
        'Dow Jones Newswires',
        'GMT',
        'gmt',
        'Gmt',
        '(rodrigo.demiguelroncal@dowjones.com )',
        '(Reenfoca titular y añade detalles a lo largo del texto)',
        'Rodríguez',
        '--Giulia Petroni contribuyó a esta nota',
        '--Mauro Orrù contribuyó a esta nota',
        '-Ben Otto contribuyó a esta nota'
    ]
    
    for expression in expressions_to_remove:
        cleaned = cleaned.replace(expression, '')
    
    # Patterns to remove using regex
    patterns = [
        r'\([^)]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\)',  # (name@example.domain)
        r'\(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b\)',  # (name@example)
        r'\nRedactada por [A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+',  # Redactada por Name Surname
        r'Redactada por [A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]\sy\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]',  # Redactada por Name1 Surname1 y Name2 Surname2
        r'\([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}; @\w+\)',  # (email; @twitter)
        r'Por\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+'  # Por Name1 Name2 Name3
    ]
    
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned


def process_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw articles: merge components, clean text, extract tickers.
    
    This is the main processing pipeline that:
    1. Merges title, snippet, and body into a single article
    2. Cleans the article text
    3. Extracts ticker symbols
    4. Filters articles without tickers
    
    Args:
        df: DataFrame with raw article data
    
    Returns:
        Processed DataFrame with publ_datetime, articles, and tickers columns
    
    Examples:
        >>> df_raw = load_raw_articles('data/raw/ibex_sample.pqt.gziq')
        >>> df_processed = process_articles(df_raw)
        >>> 'tickers' in df_processed.columns
        True
    """
    # Select necessary columns and merge components
    df_merged = merge_article_components(
        df,
        title_col='title',
        snippet_col='snippet',
        body_col='body',
        output_col='articles'
    )
    
    # Keep only datetime and articles
    df_processed = pd.DataFrame({
        'publ_datetime': df_merged['publication_datetime'],
        'articles': df_merged['articles']
    })
    
    # Clean article text
    df_processed['articles'] = df_processed['articles'].apply(clean_article_text)
    
    # Extract tickers
    df_processed['tickers'] = df_processed['articles'].apply(extract_tickers_from_article)
    
    # Filter articles that have at least one ticker
    df_processed = df_processed[df_processed['tickers'].apply(len) > 0].copy()
    
    # Reset index
    df_processed = df_processed.reset_index(drop=True)
    
    return df_processed
