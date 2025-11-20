"""Data validation utilities.

This module provides functions to validate DataFrames at various stages
of the processing pipeline, ensuring data integrity and expected structure.
"""

from typing import List, Optional

import pandas as pd


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_article_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> None:
    """Validate article DataFrame structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names. If None, uses default set.
    
    Raises:
        DataValidationError: If validation fails
    
    Examples:
        >>> df = pd.DataFrame({'publ_datetime': [datetime.now()], 'articles': ['text']})
        >>> validate_article_dataframe(df, required_columns=['publ_datetime', 'articles'])
    """
    if df.empty:
        raise DataValidationError("DataFrame is empty")
    
    if required_columns is None:
        required_columns = ['publ_datetime', 'articles']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Validate datetime column
    if 'publ_datetime' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['publ_datetime']):
            # Try to convert
            try:
                df['publ_datetime'] = pd.to_datetime(df['publ_datetime'])
            except Exception as e:
                raise DataValidationError(
                    f"Column 'publ_datetime' cannot be converted to datetime: {e}"
                )
    
    # Validate articles column has text
    if 'articles' in df.columns:
        if df['articles'].isna().all():
            raise DataValidationError("All articles are NaN")
        
        # Check if at least some articles have content
        non_empty = df['articles'].str.len() > 0
        if not non_empty.any():
            raise DataValidationError("All articles are empty strings")


def validate_embeddings_dataframe(df: pd.DataFrame) -> None:
    """Validate embeddings DataFrame structure.
    
    Args:
        df: DataFrame with embeddings to validate
    
    Raises:
        DataValidationError: If validation fails
    
    Examples:
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'publ_datetime': [datetime.now()],
        ...     'articles': ['text'],
        ...     'embeddings': [[0.1, 0.2, 0.3]]
        ... })
        >>> validate_embeddings_dataframe(df)
    """
    required_columns = ['publ_datetime', 'articles', 'embeddings']
    
    if df.empty:
        raise DataValidationError("DataFrame is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Validate embeddings are not all NaN
    if df['embeddings'].isna().all():
        raise DataValidationError("All embeddings are NaN")
    
    # Check that embeddings have consistent dimensions
    non_null_embeddings = df['embeddings'].dropna()
    if len(non_null_embeddings) > 0:
        first_embedding = non_null_embeddings.iloc[0]
        if isinstance(first_embedding, (list, tuple)):
            expected_dim = len(first_embedding)
            
            for idx, emb in non_null_embeddings.items():
                if isinstance(emb, (list, tuple)) and len(emb) != expected_dim:
                    raise DataValidationError(
                        f"Inconsistent embedding dimensions at index {idx}: "
                        f"expected {expected_dim}, got {len(emb)}"
                    )


def validate_returns_dataframe(df: pd.DataFrame) -> None:
    """Validate returns DataFrame structure.
    
    Args:
        df: DataFrame with returns data to validate
    
    Raises:
        DataValidationError: If validation fails
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'publ_datetime': [datetime.now()],
        ...     'ticker': ['TEF.MC'],
        ...     'return': [0.05]
        ... })
        >>> validate_returns_dataframe(df)
    """
    if df.empty:
        raise DataValidationError("DataFrame is empty")
    
    # Check for at least some expected columns related to returns
    expected_column_patterns = ['return', 'ticker', 'date', 'datetime']
    has_expected = any(
        any(pattern in col.lower() for pattern in expected_column_patterns)
        for col in df.columns
    )
    
    if not has_expected:
        raise DataValidationError(
            f"DataFrame doesn't appear to contain returns data. "
            f"Expected columns matching patterns: {expected_column_patterns}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Check for numeric data (returns should have numeric values)
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        raise DataValidationError(
            "No numeric columns found in returns DataFrame"
        )


def validate_tickers_list(tickers: List[str]) -> None:
    """Validate a list of ticker symbols.
    
    Args:
        tickers: List of ticker symbols to validate
    
    Raises:
        DataValidationError: If validation fails
    
    Examples:
        >>> validate_tickers_list(['TEF.MC', 'SAN.MC'])
        >>> validate_tickers_list(['INVALID'])  # doctest: +SKIP
        DataValidationError: ...
    """
    if not tickers:
        raise DataValidationError("Ticker list is empty")
    
    # Check format: should end with .MC for Spanish stocks
    spanish_pattern = r'[A-Z]+\.MC$'
    import re
    
    invalid_tickers = [t for t in tickers if not re.match(spanish_pattern, t)]
    if invalid_tickers:
        raise DataValidationError(
            f"Invalid ticker format (expected TICKER.MC): {invalid_tickers}"
        )


def check_data_quality(df: pd.DataFrame, name: str = "DataFrame") -> dict:
    """Check data quality and return statistics.
    
    This function doesn't raise errors but returns information about
    data quality issues that might need attention.
    
    Args:
        df: DataFrame to check
        name: Name of the DataFrame for reporting
    
    Returns:
        Dictionary with data quality statistics
    
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, None], 'b': [4, 5, 6]})
        >>> stats = check_data_quality(df, "test_data")
        >>> stats['total_rows']
        3
        >>> stats['columns_with_nulls']
        ['a']
    """
    stats = {
        'name': name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'null_counts': df.isna().sum().to_dict(),
        'columns_with_nulls': [col for col in df.columns if df[col].isna().any()],
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return stats
