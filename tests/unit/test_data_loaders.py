"""Tests for data loading utilities."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from PMRTN.data.loaders import (
    filter_articles,
    load_embeddings,
    load_processed_articles,
    load_raw_articles,
    load_returns_data,
    save_processed_data,
)


@pytest.fixture
def temp_parquet_file():
    """Create a temporary parquet file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df = pd.DataFrame({
            'publication_datetime': [1609459200000, 1609545600000],  # Millisecond timestamps
            'title': ['Article 1', 'España: Agenda política y económica -Semana'],
            'snippet': ['Snippet 1', 'Snippet 2'],
            'body': ['Body 1', 'Body 2'],
            'word_count': [100, 150],
            'company_codes_about': ['TELEFONICA', ''],
            'company_codes_about_ticker_exchange': ['TEF.MC', '']
        })
        df.to_parquet(f.name)
        yield Path(f.name)
        Path(f.name).unlink()


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        df = pd.DataFrame({
            'publ_datetime': ['2023-01-01 10:00:00', '2023-01-02 11:00:00'],
            'articles': ['Article text 1', 'Article text 2'],
            'tickers': ["['TEF.MC']", "['SAN.MC', 'BBVA.MC']"]
        })
        df.to_csv(f.name, index=False)
        yield Path(f.name)
        Path(f.name).unlink()


@pytest.fixture
def temp_embeddings_file():
    """Create a temporary embeddings CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        df = pd.DataFrame({
            'publ_datetime': ['2023-01-01 10:00:00'],
            'articles': ['Article text'],
            'tickers': ["['TEF.MC']"],
            'embeddings': ["[0.1, 0.2, 0.3]"]
        })
        df.to_csv(f.name, index=False)
        yield Path(f.name)
        Path(f.name).unlink()


class TestLoadRawArticles:
    """Tests for load_raw_articles function."""

    def test_load_valid_file(self, temp_parquet_file):
        """Test loading a valid parquet file."""
        df = load_raw_articles(temp_parquet_file, filter_agenda=False)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'publication_datetime' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['publication_datetime'])

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_raw_articles('nonexistent.parquet')

    def test_filter_agenda_articles(self, temp_parquet_file):
        """Test filtering of agenda articles."""
        df = load_raw_articles(temp_parquet_file, filter_agenda=True)
        
        # Should filter out the agenda article
        assert len(df) == 1
        assert 'España: Agenda política' not in df['title'].values

    def test_datetime_conversion(self, temp_parquet_file):
        """Test that timestamps are correctly converted."""
        df = load_raw_articles(temp_parquet_file, filter_agenda=False)
        
        # Check that datetime is properly converted
        first_date = df['publication_datetime'].iloc[0]
        assert first_date.year == 2021
        assert first_date.month == 1
        assert first_date.day == 1

    def test_sorting(self, temp_parquet_file):
        """Test that data is sorted by publication_datetime."""
        df = load_raw_articles(temp_parquet_file, filter_agenda=False)
        
        # Check that data is sorted
        assert df['publication_datetime'].is_monotonic_increasing


class TestFilterArticles:
    """Tests for filter_articles function."""

    def test_filter_empty_company_codes(self):
        """Test filtering articles with empty company codes."""
        df = pd.DataFrame({
            'company_codes_about': ['TELEFONICA', '', 'SANTANDER'],
            'title': ['Article 1', 'Article 2', 'Article 3']
        })
        result = filter_articles(df, filter_agenda=True)
        
        assert len(result) == 2
        assert '' not in result['company_codes_about'].values

    def test_filter_agenda_title(self):
        """Test filtering agenda articles by title."""
        df = pd.DataFrame({
            'company_codes_about': ['TELEFONICA', 'SANTANDER'],
            'title': ['Regular Article', 'España: Agenda política y económica -Semana']
        })
        result = filter_articles(df, filter_agenda=True)
        
        assert len(result) == 1
        assert 'España: Agenda política' not in result['title'].values

    def test_no_filtering(self):
        """Test when filtering is disabled."""
        df = pd.DataFrame({
            'company_codes_about': ['TELEFONICA', ''],
            'title': ['Article 1', 'España: Agenda política y económica -Semana']
        })
        result = filter_articles(df, filter_agenda=False)
        
        assert len(result) == 2


class TestLoadProcessedArticles:
    """Tests for load_processed_articles function."""

    def test_load_valid_file(self, temp_csv_file):
        """Test loading a valid processed articles CSV."""
        df = load_processed_articles(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'publ_datetime' in df.columns
        assert 'articles' in df.columns
        assert 'tickers' in df.columns

    def test_tickers_converted_to_list(self, temp_csv_file):
        """Test that ticker strings are converted to lists."""
        df = load_processed_articles(temp_csv_file)
        
        assert isinstance(df['tickers'].iloc[0], list)
        assert df['tickers'].iloc[0] == ['TEF.MC']
        assert df['tickers'].iloc[1] == ['SAN.MC', 'BBVA.MC']

    def test_datetime_conversion(self, temp_csv_file):
        """Test that datetime strings are converted properly."""
        df = load_processed_articles(temp_csv_file)
        
        assert pd.api.types.is_datetime64_any_dtype(df['publ_datetime'])

    def test_missing_required_columns(self):
        """Test error when required columns are missing."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df = pd.DataFrame({
                'wrong_column': ['data']
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_processed_articles(temp_path)
        finally:
            temp_path.unlink()

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_processed_articles('nonexistent.csv')


class TestLoadEmbeddings:
    """Tests for load_embeddings function."""

    def test_load_valid_file(self, temp_embeddings_file):
        """Test loading a valid embeddings file."""
        df = load_embeddings(temp_embeddings_file)
        
        assert isinstance(df, pd.DataFrame)
        assert 'embeddings' in df.columns

    def test_embeddings_converted_to_list(self, temp_embeddings_file):
        """Test that embedding strings are converted to lists."""
        df = load_embeddings(temp_embeddings_file)
        
        embeddings = df['embeddings'].iloc[0]
        assert isinstance(embeddings, list)
        assert embeddings == [0.1, 0.2, 0.3]

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_embeddings('nonexistent.csv')


class TestLoadReturnsData:
    """Tests for load_returns_data function."""

    def test_load_kmeans_data(self):
        """Test loading KMeans returns data."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df = pd.DataFrame({
                'ticker': ['TEF.MC', 'SAN.MC'],
                'return': [0.05, -0.02]
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            result = load_returns_data(temp_path, model='KMeans')
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
        finally:
            temp_path.unlink()

    def test_load_llama_data(self):
        """Test loading LLAMA returns data."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df = pd.DataFrame({
                'ticker': ['TEF.MC'],
                'return': [0.03]
            })
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            result = load_returns_data(temp_path, model='LLAMA')
            assert isinstance(result, pd.DataFrame)
        finally:
            temp_path.unlink()

    def test_invalid_model_type(self):
        """Test error with invalid model type."""
        with pytest.raises(ValueError, match="Invalid model type"):
            load_returns_data('dummy.csv', model='InvalidModel')

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_returns_data('nonexistent.csv', model='KMeans')


class TestSaveProcessedData:
    """Tests for save_processed_data function."""

    def test_save_dataframe(self):
        """Test saving a DataFrame to CSV."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'output.csv'
            save_processed_data(df, output_path)
            
            assert output_path.exists()
            
            # Load and verify
            loaded_df = pd.read_csv(output_path)
            assert len(loaded_df) == 3
            assert list(loaded_df.columns) == ['col1', 'col2']

    def test_save_with_index(self):
        """Test saving with index included."""
        df = pd.DataFrame({
            'col1': [1, 2, 3]
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'output.csv'
            save_processed_data(df, output_path, index=True)
            
            loaded_df = pd.read_csv(output_path)
            assert 'Unnamed: 0' in loaded_df.columns or loaded_df.index.name is not None

    def test_creates_directories(self):
        """Test that parent directories are created if they don't exist."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'subdir' / 'output.csv'
            save_processed_data(df, output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()
