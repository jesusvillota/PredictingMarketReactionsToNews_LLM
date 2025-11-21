"""Data loading, processing, and validation utilities."""

from .loaders import (
    filter_articles,
    load_embeddings,
    load_processed_articles,
    load_raw_articles,
    load_returns_data,
    save_processed_data,
)
from .processors import (
    clean_article_text,
    convert_to_datetime,
    eliminate_text_after_word,
    extract_datetime,
    extract_tickers_from_article,
    merge_article_components,
    process_articles,
)
from .validators import (
    DataValidationError,
    check_data_quality,
    validate_article_dataframe,
    validate_embeddings_dataframe,
    validate_returns_dataframe,
    validate_tickers_list,
)

__all__ = [
    # Loaders
    "load_raw_articles",
    "filter_articles",
    "load_processed_articles",
    "load_embeddings",
    "load_returns_data",
    "save_processed_data",
    # Processors
    "eliminate_text_after_word",
    "extract_datetime",
    "convert_to_datetime",
    "extract_tickers_from_article",
    "merge_article_components",
    "clean_article_text",
    "process_articles",
    # Validators
    "DataValidationError",
    "validate_article_dataframe",
    "validate_embeddings_dataframe",
    "validate_returns_dataframe",
    "validate_tickers_list",
    "check_data_quality",
]
