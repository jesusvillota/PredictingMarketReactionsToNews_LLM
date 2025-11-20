"""Tests for data validation utilities."""

from datetime import datetime

import pandas as pd
import pytest

from news_market_analysis.data.validators import (
    DataValidationError,
    check_data_quality,
    validate_article_dataframe,
    validate_embeddings_dataframe,
    validate_returns_dataframe,
    validate_tickers_list,
)


class TestValidateArticleDataframe:
    """Tests for validate_article_dataframe function."""

    def test_valid_dataframe(self):
        """Test validation passes for valid DataFrame."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'articles': ['Article 1', 'Article 2']
        })
        # Should not raise an exception
        validate_article_dataframe(df)

    def test_empty_dataframe(self):
        """Test error for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validate_article_dataframe(df)

    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            'wrong_column': ['data']
        })
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_article_dataframe(df, required_columns=['publ_datetime', 'articles'])

    def test_non_datetime_column(self):
        """Test conversion of non-datetime to datetime."""
        df = pd.DataFrame({
            'publ_datetime': ['2023-01-01', '2023-01-02'],
            'articles': ['Article 1', 'Article 2']
        })
        # Should convert to datetime without error
        validate_article_dataframe(df)
        assert pd.api.types.is_datetime64_any_dtype(df['publ_datetime'])

    def test_invalid_datetime_column(self):
        """Test error when datetime column cannot be converted."""
        df = pd.DataFrame({
            'publ_datetime': ['not a date', 'invalid'],
            'articles': ['Article 1', 'Article 2']
        })
        with pytest.raises(DataValidationError, match="cannot be converted to datetime"):
            validate_article_dataframe(df)

    def test_all_articles_nan(self):
        """Test error when all articles are NaN."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01']),
            'articles': [None]
        })
        with pytest.raises(DataValidationError, match="All articles are NaN"):
            validate_article_dataframe(df)

    def test_all_articles_empty(self):
        """Test error when all articles are empty strings."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01']),
            'articles': ['']
        })
        with pytest.raises(DataValidationError, match="All articles are empty"):
            validate_article_dataframe(df)

    def test_custom_required_columns(self):
        """Test validation with custom required columns."""
        df = pd.DataFrame({
            'custom_date': pd.to_datetime(['2023-01-01']),
            'custom_text': ['Some text']
        })
        # Should not raise with correct custom columns
        validate_article_dataframe(df, required_columns=['custom_date', 'custom_text'])


class TestValidateEmbeddingsDataframe:
    """Tests for validate_embeddings_dataframe function."""

    def test_valid_dataframe(self):
        """Test validation passes for valid embeddings DataFrame."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01']),
            'articles': ['Article text'],
            'embeddings': [[0.1, 0.2, 0.3]]
        })
        # Should not raise
        validate_embeddings_dataframe(df)

    def test_empty_dataframe(self):
        """Test error for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validate_embeddings_dataframe(df)

    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01']),
            'articles': ['Article']
            # Missing embeddings column
        })
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_embeddings_dataframe(df)

    def test_all_embeddings_nan(self):
        """Test error when all embeddings are NaN."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01']),
            'articles': ['Article'],
            'embeddings': [None]
        })
        with pytest.raises(DataValidationError, match="All embeddings are NaN"):
            validate_embeddings_dataframe(df)

    def test_inconsistent_embedding_dimensions(self):
        """Test error when embedding dimensions are inconsistent."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'articles': ['Article 1', 'Article 2'],
            'embeddings': [[0.1, 0.2, 0.3], [0.1, 0.2]]  # Different dimensions
        })
        with pytest.raises(DataValidationError, match="Inconsistent embedding dimensions"):
            validate_embeddings_dataframe(df)

    def test_consistent_dimensions_with_nan(self):
        """Test that NaN embeddings don't cause dimension check to fail."""
        df = pd.DataFrame({
            'publ_datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'articles': ['Article 1', 'Article 2', 'Article 3'],
            'embeddings': [[0.1, 0.2, 0.3], None, [0.4, 0.5, 0.6]]
        })
        # Should not raise - NaN values are skipped in dimension check
        validate_embeddings_dataframe(df)


class TestValidateReturnsDataframe:
    """Tests for validate_returns_dataframe function."""

    def test_valid_dataframe(self):
        """Test validation passes for valid returns DataFrame."""
        df = pd.DataFrame({
            'ticker': ['TEF.MC', 'SAN.MC'],
            'return': [0.05, -0.02],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02'])
        })
        # Should not raise
        validate_returns_dataframe(df)

    def test_empty_dataframe(self):
        """Test error for empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validate_returns_dataframe(df)

    def test_no_expected_columns(self):
        """Test error when DataFrame doesn't have expected column patterns."""
        df = pd.DataFrame({
            'random_col1': [1, 2],
            'random_col2': ['a', 'b']
        })
        with pytest.raises(DataValidationError, match="doesn't appear to contain returns data"):
            validate_returns_dataframe(df)

    def test_no_numeric_columns(self):
        """Test error when there are no numeric columns."""
        df = pd.DataFrame({
            'ticker': ['TEF.MC', 'SAN.MC'],
            'date': ['2023-01-01', '2023-01-02']
        })
        with pytest.raises(DataValidationError, match="No numeric columns found"):
            validate_returns_dataframe(df)

    def test_various_column_patterns(self):
        """Test that various expected column patterns are recognized."""
        # Test with 'return' column
        df1 = pd.DataFrame({
            'return': [0.05, -0.02]
        })
        validate_returns_dataframe(df1)
        
        # Test with 'ticker' column
        df2 = pd.DataFrame({
            'ticker': ['TEF.MC'],
            'value': [1.5]
        })
        validate_returns_dataframe(df2)
        
        # Test with 'datetime' column
        df3 = pd.DataFrame({
            'datetime': pd.to_datetime(['2023-01-01']),
            'price': [100.0]
        })
        validate_returns_dataframe(df3)


class TestValidateTickersList:
    """Tests for validate_tickers_list function."""

    def test_valid_tickers(self):
        """Test validation passes for valid tickers."""
        tickers = ['TEF.MC', 'SAN.MC', 'BBVA.MC']
        # Should not raise
        validate_tickers_list(tickers)

    def test_empty_list(self):
        """Test error for empty ticker list."""
        with pytest.raises(DataValidationError, match="Ticker list is empty"):
            validate_tickers_list([])

    def test_invalid_format(self):
        """Test error for tickers with invalid format."""
        tickers = ['TEF.MC', 'INVALID', 'SAN.MC']
        with pytest.raises(DataValidationError, match="Invalid ticker format"):
            validate_tickers_list(tickers)

    def test_lowercase_ticker(self):
        """Test error for lowercase tickers."""
        tickers = ['tef.mc', 'SAN.MC']
        with pytest.raises(DataValidationError, match="Invalid ticker format"):
            validate_tickers_list(tickers)

    def test_missing_mc_suffix(self):
        """Test error for tickers without .MC suffix."""
        tickers = ['TEF', 'SAN.MC']
        with pytest.raises(DataValidationError, match="Invalid ticker format"):
            validate_tickers_list(tickers)

    def test_single_valid_ticker(self):
        """Test with a single valid ticker."""
        tickers = ['TEF.MC']
        validate_tickers_list(tickers)


class TestCheckDataQuality:
    """Tests for check_data_quality function."""

    def test_basic_statistics(self):
        """Test that basic statistics are returned."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        stats = check_data_quality(df, "test_df")
        
        assert stats['name'] == 'test_df'
        assert stats['total_rows'] == 3
        assert stats['total_columns'] == 2
        assert 'col1' in stats['columns']
        assert 'col2' in stats['columns']

    def test_null_detection(self):
        """Test detection of null values."""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', 'c']
        })
        stats = check_data_quality(df)
        
        assert 'col1' in stats['columns_with_nulls']
        assert 'col2' not in stats['columns_with_nulls']
        assert stats['null_counts']['col1'] == 1
        assert stats['null_counts']['col2'] == 0

    def test_duplicate_detection(self):
        """Test detection of duplicate rows."""
        df = pd.DataFrame({
            'col1': [1, 2, 1],
            'col2': ['a', 'b', 'a']
        })
        stats = check_data_quality(df)
        
        assert stats['duplicate_rows'] == 1

    def test_memory_usage(self):
        """Test that memory usage is calculated."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        stats = check_data_quality(df)
        
        assert 'memory_usage_mb' in stats
        assert stats['memory_usage_mb'] > 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        stats = check_data_quality(df)
        
        assert stats['total_rows'] == 0
        assert stats['total_columns'] == 0
        assert stats['duplicate_rows'] == 0

    def test_all_nulls(self):
        """Test with DataFrame containing all nulls."""
        df = pd.DataFrame({
            'col1': [None, None, None]
        })
        stats = check_data_quality(df)
        
        assert 'col1' in stats['columns_with_nulls']
        assert stats['null_counts']['col1'] == 3
