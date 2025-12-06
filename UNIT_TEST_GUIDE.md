# Unit Test Guide for PMRTN Project

## Overview

This guide provides comprehensive instructions for creating unit tests for all functionalities in the `src/PMRTN/` source code. The goal is to ensure that all code runs correctly and produces expected behavior through systematic unit testing.

**Purpose**: Create a complete test suite that validates every function and class in the codebase, ensuring reliability, correctness, and maintainability.

**Testing Framework**: pytest

**Test Location**: `tests/unit/`

---

## Table of Contents

1. [Introduction](#introduction)
2. [Testing Patterns](#testing-patterns)
3. [Module-by-Module Testing Guide](#module-by-module-testing-guide)
   - [Config Module](#config-module)
   - [Data Module](#data-module)
   - [Embeddings Module](#embeddings-module)
   - [Models Module](#models-module)
   - [Analysis Module](#analysis-module)
   - [Visualization Module](#visualization-module)
   - [Utils Module](#utils-module)
   - [CLI Module](#cli-module)
4. [Coverage Checklist](#coverage-checklist)

---

## Introduction

### Testing Philosophy

- **Completeness**: Every public function and class should have unit tests
- **Isolation**: Tests should be independent and not rely on external state
- **Clarity**: Test names should clearly describe what is being tested
- **Coverage**: Aim for high code coverage, especially for critical paths
- **Maintainability**: Tests should be easy to understand and update

### Test Structure

Follow the existing pattern in `tests/unit/`:

```python
"""Tests for [module name]."""

import pytest
from PMRTN.[module] import [functions/classes]

class Test[FunctionName]:
    """Tests for [function name] function."""
    
    def test_[specific_scenario](self, fixture):
        """Test [description of what is tested]."""
        # Arrange
        # Act
        # Assert
```

### Fixtures

Common fixtures are defined in `tests/conftest.py`:
- `temp_dir`: Temporary directory for file operations
- `sample_config_data`: Sample configuration dictionary

Create additional fixtures as needed in test files or `conftest.py`.

---

## Testing Patterns

### 1. Happy Path Tests

Test normal operation with valid inputs:

```python
def test_function_with_valid_input(self):
    """Test function works correctly with valid input."""
    result = function(valid_input)
    assert result == expected_output
```

### 2. Edge Case Tests

Test boundary conditions and edge cases:
- Empty inputs (empty lists, empty strings, empty DataFrames)
- None values
- Boundary values (min/max, zero, negative)
- Single element collections

```python
def test_function_with_empty_input(self):
    """Test function handles empty input correctly."""
    with pytest.raises(ValueError):
        function([])
```

### 3. Error Handling Tests

Test that appropriate exceptions are raised for invalid inputs:

```python
def test_function_raises_error_on_invalid_input(self):
    """Test function raises appropriate error for invalid input."""
    with pytest.raises(ValueError, match="expected message"):
        function(invalid_input)
```

### 4. Type Validation Tests

Test that functions handle incorrect types correctly:

```python
def test_function_raises_type_error(self):
    """Test function raises TypeError for wrong type."""
    with pytest.raises(TypeError):
        function(wrong_type_input)
```

### 5. Mocking External Dependencies

For functions that call external APIs or services, use mocks:

```python
from unittest.mock import patch, MagicMock

@patch('PMRTN.module.external_service')
def test_function_with_mocked_service(self, mock_service):
    """Test function with mocked external service."""
    mock_service.return_value = expected_value
    result = function()
    assert result == expected_output
    mock_service.assert_called_once()
```

### 6. DataFrame Tests

For functions that work with pandas DataFrames:

```python
def test_function_with_dataframe(self):
    """Test function processes DataFrame correctly."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    result = function(df)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_length
    assert 'expected_column' in result.columns
    pd.testing.assert_frame_equal(result, expected_df)
```

### 7. File I/O Tests

For functions that read/write files:

```python
def test_function_reads_file(self, temp_dir):
    """Test function reads file correctly."""
    test_file = temp_dir / "test.csv"
    # Create test file
    test_file.write_text("test content")
    
    result = function(test_file)
    assert result == expected_output

def test_function_writes_file(self, temp_dir):
    """Test function writes file correctly."""
    output_file = temp_dir / "output.csv"
    function(data, output_file)
    
    assert output_file.exists()
    # Verify file contents
```

---

## Module-by-Module Testing Guide

## Config Module

### File: `config/settings.py`

#### Class: `ConfigurationError`
- **Purpose**: Custom exception for configuration errors
- **Tests Needed**:
  - Test exception can be raised and caught
  - Test exception message is preserved

#### Class: `Settings`
- **Purpose**: Central configuration management
- **Methods to Test**:

##### `__init__(config_path, base_path)`
- **Happy Path**:
  - Initialize with valid config file path
  - Initialize with base_path provided
  - Initialize without config_path (auto-discovery)
- **Edge Cases**:
  - Config file not found (should raise ConfigurationError)
  - Invalid YAML format (should raise ConfigurationError)
  - Config file is not a dictionary (should raise ConfigurationError)
  - Empty config file
- **Error Cases**:
  - File permission errors
  - Invalid path type

##### `_find_config_file()`
- **Happy Path**:
  - Finds config in current directory
  - Finds config in Master_Thesis-Replication_Package-main subdirectory
  - Finds config in parent directories
- **Edge Cases**:
  - Config not found in any location (should raise ConfigurationError)
  - Multiple config files exist (should return first found)
- **Error Cases**:
  - Permission denied when searching

##### `_load_config()`
- **Happy Path**:
  - Loads valid YAML config
  - Returns dictionary
- **Edge Cases**:
  - Empty YAML file
  - YAML with only comments
- **Error Cases**:
  - Invalid YAML syntax (should raise ConfigurationError)
  - File is not a dictionary (should raise ConfigurationError)
  - File read errors

##### `get(key, default)`
- **Happy Path**:
  - Get top-level key
  - Get nested key with dot notation (e.g., 'directories.raw_data')
  - Get key with default value when missing
- **Edge Cases**:
  - Key doesn't exist (should return default)
  - Nested key path doesn't exist (should return default)
  - Empty key string
  - Key with multiple levels of nesting
- **Error Cases**:
  - Key is None

##### `get_directories()`
- **Happy Path**:
  - Returns dictionary of directories
  - Returns empty dict if 'directories' key missing
- **Edge Cases**:
  - Config has no 'directories' key (should return {})
  - 'directories' is not a dictionary

#### Functions: `get_settings()`, `reset_settings()`
- **Tests Needed**:
  - `get_settings()` returns singleton instance
  - `get_settings()` with config_path on first call
  - `reset_settings()` clears global instance
  - Multiple calls to `get_settings()` return same instance
  - `reset_settings()` followed by `get_settings()` creates new instance

### File: `config/paths.py`

#### Class: `PathManager`
- **Purpose**: Manages all directory paths for the project
- **Methods to Test**:

##### `__init__(base_path, settings)`
- **Happy Path**:
  - Initialize with base_path
  - Initialize with settings instance
  - Initialize without arguments (uses global settings)
- **Edge Cases**:
  - base_path is None
  - settings is None (should use global)
- **Error Cases**:
  - Invalid base_path type

##### `_initialize_paths()`
- **Happy Path**:
  - Creates paths dictionary from settings
  - Converts relative paths to absolute
- **Edge Cases**:
  - No directories in settings (should return empty dict)
  - Empty directories dict in settings

##### `get(key)`
- **Happy Path**:
  - Returns Path object for valid key
  - Returns absolute path
- **Edge Cases**:
  - Key doesn't exist (should raise KeyError)
  - Empty key string
- **Error Cases**:
  - Key is None

##### `get_raw_data_path()`, `get_processed_data_path()`
- **Happy Path**:
  - Returns correct path for each method
- **Edge Cases**:
  - Key missing in config (should raise KeyError)

##### `get_output_path(output_type)`
- **Happy Path**:
  - Returns path for 'kmeans'
  - Returns path for 'llama'
  - Returns path for 'descriptives'
- **Edge Cases**:
  - Invalid output_type (should raise KeyError)
  - Empty output_type string

##### `create_directories(verbose)`
- **Happy Path**:
  - Creates all directories that don't exist
  - Skips directories that already exist
  - Creates parent directories as needed
- **Edge Cases**:
  - All directories already exist (should do nothing)
  - No directories configured
  - verbose=True prints messages
  - verbose=False doesn't print
- **Error Cases**:
  - Permission denied

##### `exists(key)`
- **Happy Path**:
  - Returns True for existing directory
  - Returns False for non-existing directory
  - Returns False for invalid key (should not raise)

#### Functions: `get_path_manager()`, `reset_path_manager()`
- **Tests Needed**:
  - `get_path_manager()` returns singleton
  - `reset_path_manager()` clears global instance
  - Multiple calls return same instance

---

## Data Module

### File: `data/loaders.py`

#### Function: `load_raw_articles(data_path, filter_agenda)`
- **Happy Path**:
  - Loads valid parquet file
  - Converts EPOCH timestamps to datetime
  - Sorts by publication_datetime
  - Creates publication_date_str column
  - Filters agenda articles when filter_agenda=True
- **Edge Cases**:
  - Empty parquet file (should raise ValueError)
  - File with no datetime column
  - File with all agenda articles (filtered result is empty)
  - filter_agenda=False doesn't filter
- **Error Cases**:
  - File doesn't exist (should raise FileNotFoundError)
  - Invalid parquet format
  - Permission denied

#### Function: `filter_articles(df, filter_agenda)`
- **Happy Path**:
  - Filters out articles with empty company_codes_about
  - Filters out agenda title articles
  - Returns copy of DataFrame
- **Edge Cases**:
  - Empty DataFrame (should return empty DataFrame)
  - All articles filtered out
  - filter_agenda=False doesn't filter
  - DataFrame with missing columns
- **Error Cases**:
  - df is not a DataFrame (should raise TypeError)

#### Function: `load_processed_articles(data_path)`
- **Happy Path**:
  - Loads CSV file
  - Converts tickers from string to list using ast.literal_eval
  - Converts publ_datetime to datetime
  - Validates required columns exist
- **Edge Cases**:
  - CSV with no tickers column
  - CSV with tickers as empty strings
  - CSV with invalid ticker format (should handle gracefully)
  - Missing required columns (should raise ValueError)
- **Error Cases**:
  - File doesn't exist (should raise FileNotFoundError)
  - Invalid CSV format
  - Invalid ticker string format (ast.literal_eval fails)

#### Function: `load_embeddings(data_path)`
- **Happy Path**:
  - Loads CSV with embeddings
  - Converts tickers to lists
  - Converts embeddings from string to list if needed
  - Converts datetime column
- **Edge Cases**:
  - Embeddings already as lists (no conversion needed)
  - Embeddings as strings (converts to lists)
  - Missing embeddings column
  - Empty embeddings DataFrame
- **Error Cases**:
  - File doesn't exist (should raise FileNotFoundError)
  - Invalid embedding format

#### Function: `load_returns_data(data_path, model)`
- **Happy Path**:
  - Loads returns CSV for 'KMeans' model
  - Loads returns CSV for 'LLAMA' model
- **Edge Cases**:
  - Invalid model type (should raise ValueError)
  - Empty returns file
- **Error Cases**:
  - File doesn't exist (should raise FileNotFoundError)
  - Invalid CSV format

#### Function: `save_processed_data(df, path, index)`
- **Happy Path**:
  - Saves DataFrame to CSV
  - Creates parent directories if needed
  - Saves without index when index=False
  - Saves with index when index=True
- **Edge Cases**:
  - Empty DataFrame
  - DataFrame with special characters in data
  - Path with nested directories (creates them)
- **Error Cases**:
  - Permission denied
  - Invalid path type
  - Disk full

### File: `data/processors.py`

#### Function: `eliminate_text_after_word(text, word)`
- **Happy Path**:
  - Removes text after first occurrence of word
  - Returns text up to (but not including) word
- **Edge Cases**:
  - Word not found (returns original text)
  - Word appears multiple times (removes after first)
  - Empty text
  - Empty word
  - Word at start of text
  - Word at end of text

#### Function: `extract_datetime(text)`
- **Happy Path**:
  - Extracts datetime in format 'DD-MM-YY HHMMGMT'
  - Returns matched string
- **Edge Cases**:
  - No datetime found (returns None)
  - Multiple datetimes (returns first)
  - Invalid format (returns None)
  - Empty text (returns None)

#### Function: `convert_to_datetime(timestamp_ms)`
- **Happy Path**:
  - Converts millisecond timestamp to datetime
  - Handles positive timestamps
- **Edge Cases**:
  - Zero timestamp
  - Negative timestamp (if supported)
  - Very large timestamp
  - Timestamp with decimal milliseconds
- **Error Cases**:
  - Invalid timestamp type

#### Function: `extract_tickers_from_article(article)`
- **Happy Path**:
  - Extracts tickers in format (TICKER.MC)
  - Returns list of unique tickers
  - Handles multiple tickers
- **Edge Cases**:
  - No tickers found (returns empty list)
  - Duplicate tickers (returns unique list)
  - Ticker without .MC suffix (not matched)
  - Lowercase tickers (not matched)
  - Empty article (returns empty list)

#### Function: `merge_article_components(df, title_col, snippet_col, body_col, output_col)`
- **Happy Path**:
  - Merges title, snippet, body with period separators
  - Creates new column with merged text
  - Handles custom column names
- **Edge Cases**:
  - Empty DataFrame (creates empty output column)
  - NaN values in components (filled with empty strings)
  - All components empty
  - Missing columns (should raise KeyError)
  - Custom output column name

#### Function: `clean_article_text(article)`
- **Happy Path**:
  - Removes text after elimination words
  - Removes specific expressions
  - Removes email patterns
  - Removes author attribution patterns
- **Edge Cases**:
  - Empty article (returns empty string)
  - Article with no patterns to remove
  - Article with multiple patterns
  - Article with all elimination words
- **Error Cases**:
  - article is not a string (should raise TypeError)

#### Function: `process_articles(df)`
- **Happy Path**:
  - Merges components
  - Cleans text
  - Extracts tickers
  - Filters articles without tickers
  - Returns DataFrame with publ_datetime, articles, tickers
- **Edge Cases**:
  - Empty DataFrame (returns empty DataFrame)
  - All articles filtered out (no tickers)
  - Articles with multiple tickers
  - Missing required columns (should raise KeyError)
- **Error Cases**:
  - df is not a DataFrame

### File: `data/validators.py`

#### Class: `DataValidationError`
- **Tests Needed**:
  - Exception can be raised and caught
  - Exception message preserved

#### Function: `validate_article_dataframe(df, required_columns)`
- **Happy Path**:
  - Validates DataFrame with all required columns
  - Validates datetime column is datetime type
  - Converts datetime column if possible
  - Validates articles column has content
- **Edge Cases**:
  - Empty DataFrame (should raise DataValidationError)
  - Missing required columns (should raise DataValidationError)
  - All articles are NaN (should raise DataValidationError)
  - All articles are empty strings (should raise DataValidationError)
  - Custom required_columns list
  - required_columns=None uses defaults
- **Error Cases**:
  - df is not a DataFrame
  - Datetime column cannot be converted

#### Function: `validate_embeddings_dataframe(df)`
- **Happy Path**:
  - Validates required columns exist
  - Validates embeddings are not all NaN
  - Validates embedding dimensions are consistent
- **Edge Cases**:
  - Empty DataFrame (should raise DataValidationError)
  - Missing columns (should raise DataValidationError)
  - All embeddings NaN (should raise DataValidationError)
  - Inconsistent embedding dimensions (should raise DataValidationError)
  - Embeddings as lists
  - Embeddings as numpy arrays

#### Function: `validate_returns_dataframe(df)`
- **Happy Path**:
  - Validates DataFrame has returns-related columns
  - Validates numeric columns exist
- **Edge Cases**:
  - Empty DataFrame (should raise DataValidationError)
  - No expected column patterns (should raise DataValidationError)
  - No numeric columns (should raise DataValidationError)
  - DataFrame with only text columns

#### Function: `validate_tickers_list(tickers)`
- **Happy Path**:
  - Validates list of valid ticker formats (TICKER.MC)
  - All tickers valid
- **Edge Cases**:
  - Empty list (should raise DataValidationError)
  - Invalid format tickers (should raise DataValidationError)
  - Mixed valid/invalid tickers
  - Tickers without .MC suffix

#### Function: `check_data_quality(df, name)`
- **Happy Path**:
  - Returns dictionary with quality statistics
  - Calculates null counts
  - Identifies columns with nulls
  - Counts duplicate rows
  - Calculates memory usage
- **Edge Cases**:
  - Empty DataFrame
  - DataFrame with no nulls
  - DataFrame with all nulls
  - DataFrame with duplicates
  - Custom name parameter

---

## Embeddings Module

### File: `embeddings/generators.py`

#### Class: `EmbeddingGeneratorError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `get_model(model_name)`
- **Happy Path**:
  - Loads model from AVAILABLE_MODELS
  - Returns SentenceTransformer instance
  - Caches model after first load
- **Edge Cases**:
  - Model already in cache (returns cached instance)
  - Invalid model name (should raise EmbeddingGeneratorError)
  - Model name not in AVAILABLE_MODELS
- **Error Cases**:
  - Model download fails (network error)
  - Model file corrupted

#### Function: `clear_model_cache()`
- **Happy Path**:
  - Clears model cache
  - Subsequent get_model() reloads model
- **Edge Cases**:
  - Cache already empty (no error)
  - Multiple models in cache (all cleared)

#### Function: `get_embedding(article, model_name)`
- **Happy Path**:
  - Generates embedding for single article
  - Returns list of floats
  - Uses default model if not specified
- **Edge Cases**:
  - Empty article (should raise EmbeddingGeneratorError)
  - Whitespace-only article (should raise EmbeddingGeneratorError)
  - Very long article
  - Custom model_name
- **Error Cases**:
  - Model fails to encode (should raise EmbeddingGeneratorError)
  - article is not a string

#### Function: `generate_embeddings(texts, model_name, show_progress, batch_size)`
- **Happy Path**:
  - Generates embeddings for list of texts
  - Generates embeddings for pandas Series
  - Returns list of embedding lists
  - Handles batch processing
- **Edge Cases**:
  - Empty list (should raise EmbeddingGeneratorError)
  - Empty Series (should raise EmbeddingGeneratorError)
  - Single text
  - Large batch (tests batch_size parameter)
  - show_progress=True/False
  - Text with empty string (should raise EmbeddingGeneratorError)
  - Text that is not string (should raise EmbeddingGeneratorError)
- **Error Cases**:
  - Model encoding fails

#### Function: `add_embeddings_to_dataframe(df, text_column, embedding_column, model_name, show_progress, batch_size)`
- **Happy Path**:
  - Adds embeddings column to DataFrame
  - Returns copy of DataFrame
  - Uses specified text column
  - Creates new embedding column
- **Edge Cases**:
  - text_column doesn't exist (should raise EmbeddingGeneratorError)
  - Empty DataFrame
  - Custom column names
  - embedding_column already exists (overwrites)
- **Error Cases**:
  - df is not a DataFrame

#### Function: `get_embedding_dimension(model_name)`
- **Happy Path**:
  - Returns correct dimension for model
  - Tests with default model
  - Tests with different models
- **Edge Cases**:
  - Model not available (should raise EmbeddingGeneratorError)
- **Error Cases**:
  - Model fails to load

---

## Models Module

### File: `models/kmeans.py`

#### Class: `ClusteringError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Class: `NewsClusteringModel`
- **Purpose**: KMeans clustering wrapper for news embeddings
- **Methods to Test**:

##### `__init__(n_clusters, random_state, max_iter, n_init, init, **kwargs)`
- **Happy Path**:
  - Initializes with default parameters
  - Initializes with custom parameters
  - Creates KMeans model instance
  - Initializes scaler
- **Edge Cases**:
  - n_clusters < 2 (should raise ValueError)
  - n_clusters = 2 (minimum valid)
  - Very large n_clusters
  - random_state for reproducibility
- **Error Cases**:
  - Invalid init method

##### `fit(embeddings, scale)`
- **Happy Path**:
  - Fits model on numpy array embeddings
  - Fits model on list of lists
  - Scales embeddings when scale=True
  - Doesn't scale when scale=False
  - Sets is_fitted=True
  - Stores cluster centers, labels, inertia, n_iter
- **Edge Cases**:
  - embeddings as list (converts to array)
  - scale=True applies StandardScaler
  - scale=False uses raw embeddings
  - Number of samples < n_clusters (should raise ClusteringError)
  - 1D array (should raise ClusteringError)
  - Empty embeddings (should raise ClusteringError)
- **Error Cases**:
  - Invalid embeddings shape
  - Fitting fails (should raise ClusteringError)

##### `predict(embeddings, scale)`
- **Happy Path**:
  - Predicts clusters for new embeddings
  - Returns cluster labels
  - Uses fitted scaler when scale=True
- **Edge Cases**:
  - Model not fitted (should raise ClusteringError)
  - embeddings as list (converts to array)
  - Single embedding (1D array)
  - Multiple embeddings (2D array)
- **Error Cases**:
  - Invalid embeddings format

##### `fit_predict(embeddings, scale)`
- **Happy Path**:
  - Fits and predicts in one call
  - Returns cluster labels
- **Edge Cases**:
  - Same as fit() and predict() combined

##### `evaluate(embeddings, scale, metric)`
- **Happy Path**:
  - Calculates silhouette score
  - Calculates Davies-Bouldin score
  - Returns dictionary of metrics
- **Edge Cases**:
  - Model not fitted (should raise ClusteringError)
  - Invalid metric name (should raise ValueError)
  - Single cluster (evaluation may fail)
- **Error Cases**:
  - Evaluation calculation fails

##### `get_cluster_centers(scale)`
- **Happy Path**:
  - Returns cluster centers
  - Returns scaled centers when scale=True
  - Returns original centers when scale=False
- **Edge Cases**:
  - Model not fitted (should raise ClusteringError)

##### `get_labels()`
- **Happy Path**:
  - Returns training labels
- **Edge Cases**:
  - Model not fitted (should raise ClusteringError)

#### Function: `find_optimal_k(embeddings, method, k_range, scale, random_state, verbose, save_this_plot)`
- **Happy Path**:
  - Finds optimal k using silhouette score
  - Tests range of k values
  - Returns optimal k and scores dictionary
- **Edge Cases**:
  - k_range with single value
  - k_range with all values
  - All k values give same score
  - verbose=True/False
  - save_this_plot parameter (may not be testable without file system)
- **Error Cases**:
  - Invalid method
  - Invalid k_range

#### Function: `cluster_train_val_test(method, e_train_scaled, e_val_scaled, e_test_scaled, k_opt, random_state, **kwargs)`
- **Happy Path**:
  - Clusters train, val, test sets
  - Returns dictionary with labels for each set
  - Uses same model for all sets
- **Edge Cases**:
  - Different sized sets
  - k_opt = 2 (minimum)
  - Empty validation set
  - Empty test set
- **Error Cases**:
  - Invalid method
  - Mismatched embedding dimensions

### File: `models/llama.py`

#### Class: `LLAMAParserError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Class: `FirmShock`
- **Purpose**: Represents a shock affecting a firm
- **Methods to Test**:

##### `__init__(firm, ticker, shock_type, shock_magnitude, shock_direction)`
- **Happy Path**:
  - Creates FirmShock with all valid values
  - Validates shock_type against VALID_SHOCK_TYPES
  - Validates shock_magnitude against VALID_MAGNITUDES
  - Validates shock_direction against VALID_DIRECTIONS
- **Edge Cases**:
  - Empty strings for optional fields (should be allowed)
  - Invalid shock_type (should raise ValueError)
  - Invalid shock_magnitude (should raise ValueError)
  - Invalid shock_direction (should raise ValueError)
  - All valid shock types
  - All valid magnitudes
  - All valid directions

##### `to_dict()`
- **Happy Path**:
  - Returns dictionary with all attributes
  - Dictionary has correct keys

##### `__repr__()`
- **Happy Path**:
  - Returns string representation
  - Contains all attributes

#### Class: `LLAMANewsParser`
- **Purpose**: LLAMA-based parser for Spanish business news
- **Methods to Test**:

##### `__init__(api_key, model, max_retries, retry_delay)`
- **Happy Path**:
  - Initializes with valid API key
  - Uses default model
  - Uses custom model
  - Sets retry parameters
- **Edge Cases**:
  - Empty API key (may be allowed or raise error)
  - Invalid model name (should raise LLAMAParserError)
  - Groq library not installed (should raise LLAMAParserError)
- **Error Cases**:
  - API key is None

##### `parse_article(article_text)`
- **Happy Path**:
  - Parses article and returns FirmShock list
  - Handles valid API response
  - Extracts firm, ticker, shock attributes
- **Edge Cases**:
  - Empty article text
  - Article with no firms
  - Article with multiple firms
  - Article with no shocks
  - API returns empty response
  - Invalid API response format
- **Error Cases**:
  - API call fails (should raise LLAMAParserError)
  - Rate limit exceeded (should retry)
  - Network error (should retry)
  - Max retries exceeded

##### `parse_articles_batch(articles, show_progress)`
- **Happy Path**:
  - Parses multiple articles
  - Returns list of FirmShock lists
  - Handles progress display
- **Edge Cases**:
  - Empty articles list
  - Single article
  - Large batch
  - Some articles fail (should continue)
  - show_progress=True/False
- **Error Cases**:
  - All articles fail

#### Function: `create_parser(api_key, model)`
- **Happy Path**:
  - Creates LLAMANewsParser instance
  - Uses provided API key
  - Uses default or custom model
- **Edge Cases**:
  - Default model
  - Custom model

---

## Analysis Module

### File: `analysis/statistics.py`

#### Class: `StatisticsError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `split_data(df, split1, split2, split2_type, seed, verbose)`
- **Happy Path**:
  - Splits data into train/val/test
  - Sequential split (split2_type='sequential')
  - Random split (split2_type='random')
  - Returns dictionary with D, D_train, D_val, D_test
  - Adds 'split' column to original DataFrame
- **Edge Cases**:
  - split1 = 1.0 (no test set)
  - split2 = 1.0 (no validation set)
  - split1 = 0.5, split2 = 0.5 (equal splits)
  - Empty DataFrame (should handle gracefully)
  - Single row DataFrame
  - verbose=True prints information
  - verbose=False doesn't print
- **Error Cases**:
  - split1 <= 0 or > 1 (should raise ValueError)
  - split2 <= 0 or > 1 (should raise ValueError)
  - Invalid split2_type (should raise ValueError)
  - df is not a DataFrame

#### Function: `get_e_data(df_train, df_val, df_test, embeddings_col)`
- **Happy Path**:
  - Extracts embeddings from DataFrames
  - Converts to numpy arrays
  - Fits scaler on train, applies to all
  - Returns dictionary with scaled and unscaled embeddings
- **Edge Cases**:
  - Empty train set (should handle or raise error)
  - Empty val/test sets
  - Custom embeddings_col name
  - Embeddings as lists
  - Embeddings as numpy arrays
- **Error Cases**:
  - embeddings_col not found (should raise KeyError)
  - Invalid embedding format (should raise ValueError)
  - Inconsistent embedding dimensions

#### Function: `compute_statistics_for_l_values(...)`
- **Happy Path**:
  - Computes statistics for multiple L values
  - Returns nested dictionary structure
  - Handles multiple algorithms
  - Handles multiple splits
- **Edge Cases**:
  - Empty l_values list
  - Single L value
  - Large L values
  - Empty articles_df
  - Empty ts_dict
- **Error Cases**:
  - Invalid parameters
  - Missing required columns

#### Function: `compute_statistics_for_theta_values(...)`
- **Happy Path**:
  - Computes statistics for multiple theta values
  - Similar structure to L values function
- **Edge Cases**:
  - Empty theta_values list
  - Single theta value
  - Theta values out of valid range

### File: `analysis/portfolio.py`

#### Class: `PortfolioError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `initialize_portfolio(articles_df, trading_days, l_value)`
- **Happy Path**:
  - Creates return DataFrames for each split
  - Creates trading day lists for each split
  - Extends timeline by L days
  - Returns r_P_dict and trading_days_dict
- **Edge Cases**:
  - Empty articles_df
  - No articles in a split
  - l_value = 0
  - Large l_value
  - Trading days extend beyond available data
- **Error Cases**:
  - Missing required columns
  - Invalid l_value type

#### Function: `calculate_portfolio_returns(...)`
- **Happy Path**:
  - Calculates gross returns
  - Calculates net returns (with trading costs)
  - Tracks trading signal evolution
  - Calculates turnover
  - Returns comprehensive dictionary
- **Edge Cases**:
  - trading_cost_bps = 0 (no costs)
  - trading_cost_bps = 100 (1% costs)
  - l_value = 1 (minimum holding period)
  - No trading signals
  - All positions long
  - All positions short
  - Mixed long/short positions
  - verbose=True/False
- **Error Cases**:
  - Missing required columns
  - Invalid ts_dict
  - Trading days not in chronological order

#### Function: `calculate_trading_intensity_statistics(...)`
- **Happy Path**:
  - Calculates average positions
  - Calculates turnover statistics
  - Calculates trading costs
  - Calculates active days
  - Returns statistics dictionary
- **Edge Cases**:
  - Empty trading signal evolution
  - No turnover
  - High turnover
  - Single split
  - All splits

#### Function: `calculate_portfolio_statistics(r_P)`
- **Happy Path**:
  - Calculates all portfolio metrics
  - Returns dictionary with statistics
  - Handles gross and net returns
- **Edge Cases**:
  - Empty returns series
  - All positive returns
  - All negative returns
  - Zero returns
  - Returns with extreme values

### File: `analysis/backtesting.py`

#### Class: `BacktestingError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `calculate_trading_strategy_data(ticker, date_affect, returns_df, successful_tickers, l_max, market_model_window, market_model_buffer)`
- **Happy Path**:
  - Fits market model on pre-event data
  - Calculates abnormal returns
  - Calculates cumulative abnormal returns
  - Calculates performance metrics (μ, σ, SR)
  - Returns DataFrame with AR, CAR, μ, σ, SR columns
- **Edge Cases**:
  - Ticker not in successful_tickers (returns None)
  - date_affect not in returns_df index (should raise ValueError)
  - Insufficient data for market model (should raise IndexError)
  - Holding period extends beyond data (should raise IndexError)
  - l_max = 1 (minimum)
  - l_max = 260 (maximum, ~1 year)
  - market_model_window = 50 (minimum reasonable)
  - market_model_buffer = 0 (no buffer)
- **Error Cases**:
  - Missing required columns
  - Market model fitting fails
  - Invalid date_affect type

#### Function: `process_article_ticker_pair(...)`
- **Happy Path**:
  - Processes single (article, ticker) pair
  - Calls calculate_trading_strategy_data
  - Handles None returns gracefully
- **Edge Cases**:
  - Ticker not in successful_tickers
  - date_affect not in returns_df
  - Processing fails (returns None)

#### Function: `calculate_average_metrics_by_group(...)`
- **Happy Path**:
  - Groups by specified column
  - Calculates average metrics
  - Returns DataFrame with group averages
- **Edge Cases**:
  - Empty ts_dict
  - Single group
  - Multiple groups
  - Groups with no data
  - Missing l_value in some DataFrames

### File: `analysis/cluster_selection.py`

#### Class: `ClusterSelectionError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `calculate_cluster_sharpe_ratios(articles_df, ts_dict, l_value)`
- **Happy Path**:
  - Calculates average SR for each (split, cluster)
  - Returns dictionary mapping (split, cluster) to avg SR
  - Handles missing data gracefully
- **Edge Cases**:
  - Empty articles_df
  - Empty ts_dict
  - Some articles missing from ts_dict
  - NaN SR values (skipped)
  - Single cluster
  - Multiple clusters
- **Error Cases**:
  - Missing required columns
  - Invalid l_value

#### Function: `rank_clusters_by_sharpe(avg_sr_dict)`
- **Happy Path**:
  - Ranks clusters by SR within each split
  - Returns dictionary with ranked lists
  - Sorts by SR descending
- **Edge Cases**:
  - Empty avg_sr_dict
  - Single cluster per split
  - Tied SR values
  - NaN SR values (handled in sorting)

#### Function: `select_clusters_greedy(ranking_dict, avg_sr_dict, theta, train_split, val_split)`
- **Happy Path**:
  - Selects top theta clusters from validation
  - Separates positive and negative SR clusters
  - Returns long_clusters and short_clusters lists
- **Edge Cases**:
  - theta = 1 (minimum)
  - theta = total clusters (all selected)
  - theta > total clusters (all selected)
  - No positive SR clusters
  - No negative SR clusters
  - All clusters have zero SR
  - Empty ranking_dict
- **Error Cases**:
  - Invalid theta value
  - Missing splits

#### Function: `select_clusters_rank_stable(ranking_dict, avg_sr_dict, theta, train_split, val_split)`
- **Happy Path**:
  - Selects clusters with stable rankings
  - Minimizes rank difference between train/val
  - Returns long_clusters and short_clusters
- **Edge Cases**:
  - Perfect rank correlation (all clusters stable)
  - No rank correlation (no stable clusters)
  - theta = 1
  - theta = total clusters
  - Empty ranking_dict
- **Error Cases**:
  - Invalid parameters

#### Function: `assign_trading_rules(...)`
- **Happy Path**:
  - Assigns +1 to long clusters
  - Assigns -1 to short clusters
  - Assigns 0 to unselected clusters
  - Returns trading rules dictionary
- **Edge Cases**:
  - Empty cluster lists
  - All clusters selected
  - No clusters selected
  - Overlapping long/short lists (should handle)

#### Function: `calculate_spearman_correlation(ranking_dict, train_split, val_split)`
- **Happy Path**:
  - Calculates Spearman correlation between train/val rankings
  - Returns correlation coefficient
  - Handles tied ranks
- **Edge Cases**:
  - Perfect correlation (1.0)
  - No correlation (0.0)
  - Negative correlation
  - Empty rankings
  - Single cluster (correlation undefined)

#### Function: `separate_clusters_by_sr_sign(avg_sr_dict, split)`
- **Happy Path**:
  - Separates clusters into positive and negative SR
  - Returns two lists
- **Edge Cases**:
  - All positive SR
  - All negative SR
  - All zero SR
  - Empty avg_sr_dict
  - Split not in dictionary

### File: `analysis/trading_calendar.py`

#### Class: `TradingCalendarAdjustments`
- **Purpose**: Handles trading calendar adjustments
- **Methods to Test** (if class has methods):
  - All methods should be tested following the same patterns
  - Test date adjustments
  - Test holiday handling
  - Test weekend handling

---

## Visualization Module

### File: `visualization/plotting.py`

#### Class: `PlottingError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `plot_cluster_distribution(...)`
- **Happy Path**:
  - Creates bar plot of cluster distribution
  - Saves plot to file if path provided
  - Returns figure object
- **Edge Cases**:
  - Empty cluster data
  - Single cluster
  - Many clusters
  - save_path=None (doesn't save)
  - Custom figure size
- **Error Cases**:
  - Invalid data format
  - Invalid save path

#### Function: `plot_cluster_distributions_by_split(...)`
- **Happy Path**:
  - Creates subplots for each split
  - Shows cluster distributions
  - Returns figure object
- **Edge Cases**:
  - Empty splits
  - Single split
  - Missing data for some splits

#### Function: `plot_average_cars_by_cluster(...)`
- **Happy Path**:
  - Plots average CARs by cluster
  - Handles multiple L values
  - Returns figure object
- **Edge Cases**:
  - Empty cluster data
  - Single cluster
  - Single L value
  - Multiple L values

#### Function: `plot_cumulative_returns(...)`
- **Happy Path**:
  - Plots cumulative returns over time
  - Handles multiple splits
  - Returns figure object
- **Edge Cases**:
  - Empty returns data
  - Single split
  - Multiple splits
  - Returns with negative values
  - Returns with extreme values

#### Function: `configure_matplotlib_style()`
- **Happy Path**:
  - Configures matplotlib style
  - Sets style parameters
- **Edge Cases**:
  - Called multiple times (should be idempotent)

#### Function: `reset_matplotlib_style()`
- **Happy Path**:
  - Resets matplotlib to default style
- **Edge Cases**:
  - Called when no custom style set

#### Function: `plot_time_series_with_ma(...)`
- **Happy Path**:
  - Plots time series with moving average
  - Handles different window sizes
  - Returns figure object
- **Edge Cases**:
  - Empty time series
  - window_size = 1
  - window_size = length of series
  - window_size > length of series

#### Function: `plot_histogram_with_density(...)`
- **Happy Path**:
  - Creates histogram with density curve
  - Handles different bin counts
  - Returns figure object
- **Edge Cases**:
  - Empty data
  - Single value
  - Bimodal distribution
  - Skewed distribution

#### Function: `generate_wordcloud(...)`
- **Happy Path**:
  - Generates word cloud from text
  - Saves to file if path provided
  - Returns word cloud object
- **Edge Cases**:
  - Empty text
  - Single word
  - Very long text
  - Custom parameters

#### Function: `plot_silhouette_scores(...)`
- **Happy Path**:
  - Plots silhouette scores for different k values
  - Highlights optimal k
  - Returns figure object
- **Edge Cases**:
  - Empty scores dictionary
  - Single k value
  - All scores equal
  - Optimal k at boundary

### File: `visualization/tables.py`

#### Class: `TableGenerationError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `generate_cluster_mapping_table(...)`
- **Happy Path**:
  - Generates LaTeX table for cluster mapping
  - Includes trading rules
  - Saves to file if path provided
  - Returns table string
- **Edge Cases**:
  - Empty cluster data
  - Single cluster
  - Many clusters
  - Missing trading rules

#### Function: `generate_portfolio_statistics_table(...)`
- **Happy Path**:
  - Generates LaTeX table for portfolio statistics
  - Handles gross and net returns
  - Formats numbers correctly
  - Returns table string
- **Edge Cases**:
  - Empty statistics
  - Single split
  - Multiple splits
  - Missing statistics keys

#### Function: `generate_trading_intensity_table(...)`
- **Happy Path**:
  - Generates LaTeX table for trading intensity
  - Includes turnover, costs, active days
  - Returns table string
- **Edge Cases**:
  - Empty data
  - Zero turnover
  - High turnover

#### Function: `generate_llama_shock_mapping_table(...)`
- **Happy Path**:
  - Generates LaTeX table for LLAMA shock mapping
  - Shows shock type, magnitude, direction
  - Returns table string
- **Edge Cases**:
  - Empty shock data
  - All shock types
  - Missing shock attributes

---

## Utils Module

### File: `utils/financial.py`

#### Class: `FinancialUtilsError`
- **Tests Needed**:
  - Exception can be raised and caught

#### Function: `calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)`
- **Happy Path**:
  - Calculates annualized Sharpe ratio
  - Handles different risk-free rates
  - Handles different periods per year
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Zero standard deviation with zero mean (returns 0.0)
  - Zero standard deviation with non-zero mean (should raise FinancialUtilsError)
  - All positive returns
  - All negative returns
  - risk_free_rate > 0
  - periods_per_year = 252 (daily), 52 (weekly), 12 (monthly)

#### Function: `calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)`
- **Happy Path**:
  - Calculates Sortino ratio using downside deviation
  - Similar edge cases to Sharpe ratio
- **Edge Cases**:
  - No negative returns (returns inf if positive mean, 0.0 if zero mean)
  - Zero downside deviation (should raise FinancialUtilsError)
  - All positive returns

#### Function: `calculate_calmar_ratio(returns, periods_per_year)`
- **Happy Path**:
  - Calculates Calmar ratio
  - Handles different periods per year
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Zero max drawdown (returns inf if positive return, 0.0 if zero return)
  - No drawdowns

#### Function: `calculate_max_drawdown(returns)`
- **Happy Path**:
  - Calculates maximum drawdown
  - Returns negative value
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - All positive returns (drawdown = 0)
  - All negative returns
  - Single return value

#### Function: `calculate_cumulative_return(returns)`
- **Happy Path**:
  - Calculates cumulative returns
  - Returns Series
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Single return
  - All positive returns
  - All negative returns
  - Zero returns

#### Function: `calculate_annualized_return(returns, periods_per_year)`
- **Happy Path**:
  - Calculates annualized return
  - Handles different periods per year
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Zero mean return
  - Negative mean return
  - Very high return

#### Function: `calculate_annualized_volatility(returns, periods_per_year)`
- **Happy Path**:
  - Calculates annualized volatility
  - Handles different periods per year
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Zero volatility (constant returns)
  - High volatility

#### Function: `calculate_portfolio_statistics(returns, risk_free_rate, periods_per_year)`
- **Happy Path**:
  - Calculates all portfolio statistics
  - Returns comprehensive dictionary
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - All statistics calculated correctly
  - Missing risk_free_rate (uses default 0.0)

#### Function: `calculate_turnover(current_positions, previous_positions)`
- **Happy Path**:
  - Calculates portfolio turnover
  - Returns value between 0.0 and 2.0
- **Edge Cases**:
  - Empty previous_positions (returns 0.0)
  - Empty current_positions (returns 0.0)
  - No change in positions (returns 0.0)
  - Complete replacement (returns 2.0)
  - Invalid input types (should raise FinancialUtilsError)

#### Function: `calculate_trading_costs(positions, trading_cost_bps)`
- **Happy Path**:
  - Calculates trading costs
  - Handles different cost rates
- **Edge Cases**:
  - trading_cost_bps = 0 (no costs)
  - trading_cost_bps = 100 (1%)
  - Empty positions
  - Negative positions (shorts)

#### Function: `calculate_excess_returns(returns, risk_free_rate)`
- **Happy Path**:
  - Calculates excess returns
  - Returns Series
- **Edge Cases**:
  - risk_free_rate = 0 (returns unchanged)
  - risk_free_rate > 0
  - Empty returns

#### Function: `calculate_information_ratio(returns, benchmark_returns)`
- **Happy Path**:
  - Calculates information ratio
  - Compares to benchmark
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Mismatched lengths (should handle or raise error)
  - Zero tracking error (should raise FinancialUtilsError)
  - Perfect correlation

#### Function: `calculate_beta(returns, market_returns)`
- **Happy Path**:
  - Calculates beta coefficient
  - Uses CAPM regression
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - Mismatched lengths
  - Zero market variance (should raise FinancialUtilsError)
  - Beta = 1 (market beta)
  - Beta > 1 (high volatility)
  - Beta < 1 (low volatility)

#### Function: `calculate_var(returns, confidence_level)`
- **Happy Path**:
  - Calculates Value at Risk
  - Handles different confidence levels
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - confidence_level = 0.95, 0.99
  - All positive returns (VaR = 0 or negative)
  - All negative returns

#### Function: `calculate_cvar(returns, confidence_level)`
- **Happy Path**:
  - Calculates Conditional VaR (Expected Shortfall)
  - Similar to VaR but average of tail
- **Edge Cases**:
  - Empty returns (should raise FinancialUtilsError)
  - All positive returns
  - All negative returns

#### Function: `load_risk_free_rate(data_path)`
- **Happy Path**:
  - Loads risk-free rate data from file
  - Returns DataFrame
- **Edge Cases**:
  - File doesn't exist (should raise FileNotFoundError)
  - Empty file
  - Invalid format

#### Function: `download_market_index(ticker, start_date, end_date)`
- **Happy Path**:
  - Downloads market index data using yfinance
  - Returns DataFrame with returns
- **Edge Cases**:
  - Invalid ticker (should raise FinancialUtilsError)
  - Date range with no data
  - Network error (should handle gracefully)
- **Note**: Should use mocking for yfinance calls

#### Function: `_fetch_single_ticker_data(ticker, start_date, end_date)`
- **Happy Path**:
  - Fetches data for single ticker
  - Handles errors gracefully
- **Edge Cases**:
  - Invalid ticker (returns None)
  - No data available (returns None)
  - Network timeout
- **Note**: Should use mocking

#### Function: `download_stock_returns(tickers, start_date, end_date, risk_free_rate_path, n_jobs)`
- **Happy Path**:
  - Downloads returns for multiple tickers
  - Uses parallel processing
  - Calculates excess returns
  - Returns DataFrame
- **Edge Cases**:
  - Empty tickers list
  - Single ticker
  - Some tickers fail (continues with others)
  - All tickers fail
  - n_jobs = 1 (sequential)
  - n_jobs > 1 (parallel)
- **Note**: Should use mocking for yfinance

### File: `utils/text_processing.py`

All functions in this module should be tested with:
- **Happy Path**: Normal operation with valid input
- **Edge Cases**: Empty strings, None values, boundary conditions
- **Error Cases**: Invalid types, unexpected formats

#### Key Functions to Test:
- `normalize_whitespace()`: Multiple spaces, newlines, tabs
- `remove_urls()`: Various URL formats
- `remove_email_addresses()`: Various email formats
- `truncate_text()`: Text shorter/longer than max_length
- `extract_sentences()`: Various sentence patterns
- `count_words()`: Empty text, single word, many words
- `extract_numbers()`: Integers, floats, negative numbers
- `contains_keywords()`: Case sensitive/insensitive
- `remove_special_characters()`: Various special characters
- `capitalize_first_letter()`: Empty, single char, already capitalized
- `remove_repeated_punctuation()`: Multiple punctuation marks
- `extract_quoted_text()`: Single quotes, double quotes, nested quotes
- `calculate_text_statistics()`: All statistics calculated correctly
- `vocabulary_filter()`: Complex function with multiple parameters
  - Test min_word_count threshold
  - Test max_word_count_threshold
  - Test verbose output
  - Test empty text
  - Test filtering statistics

---

## CLI Module

### File: `cli/main.py`

#### Function: `cli(ctx, config, verbose)`
- **Happy Path**:
  - Initializes CLI
  - Loads config if provided
  - Sets verbose mode
- **Edge Cases**:
  - config=None (uses default)
  - verbose=True/False

#### Function: `version(ctx)`
- **Happy Path**:
  - Prints version information
- **Edge Cases**:
  - Version format correct

#### Function: `info(ctx)`
- **Happy Path**:
  - Prints project information
  - Shows paths
- **Edge Cases**:
  - All paths displayed correctly

### File: `cli/data_commands.py`

#### Function: `load_articles(...)`
- **Happy Path**:
  - Loads articles via CLI
  - Saves processed data
- **Edge Cases**:
  - File not found
  - Invalid config
- **Note**: Test CLI command execution, use Click's CliRunner

#### Function: `describe_data(...)`
- **Happy Path**:
  - Generates descriptive statistics
  - Creates visualizations
- **Edge Cases**:
  - No data available
  - Missing output directory

#### Function: `download_returns(...)`
- **Happy Path**:
  - Downloads returns data
  - Saves to file
- **Edge Cases**:
  - Network errors
  - Invalid tickers
- **Note**: Mock yfinance calls

#### Function: `fetch_tickers(...)`
- **Happy Path**:
  - Extracts tickers from articles
  - Saves ticker list
- **Edge Cases**:
  - No tickers found
  - Invalid article format

#### Function: `generate_embeddings(...)`
- **Happy Path**:
  - Generates embeddings via CLI
  - Saves to file
- **Edge Cases**:
  - Model download fails
  - Out of memory
- **Note**: Mock model loading if possible

### File: `cli/clustering_commands.py`

#### Function: `kmeans_clustering(...)`
- **Happy Path**:
  - Runs KMeans clustering pipeline
  - Generates outputs
- **Edge Cases**:
  - Invalid parameters
  - Missing input data
- **Note**: Test full pipeline execution

### File: `cli/llama_commands.py`

#### Function: `llama_parse(...)`
- **Happy Path**:
  - Parses articles using LLAMA
  - Saves results
- **Edge Cases**:
  - API key missing
  - API rate limits
  - Network errors
- **Note**: Mock Groq API calls

#### Function: `llama_clustering(...)`
- **Happy Path**:
  - Runs LLAMA clustering pipeline
  - Generates outputs
- **Edge Cases**:
  - Missing parsed data
  - Invalid parameters

### File: `cli/pipeline_commands.py`

#### Function: `run_all(...)`
- **Happy Path**:
  - Runs complete pipeline
  - Executes all steps
- **Edge Cases**:
  - Skip certain steps
  - Partial failures
  - Missing dependencies

**Note for CLI Testing**: Use pytest with Click's `CliRunner`:

```python
from click.testing import CliRunner
from PMRTN.cli.main import cli

def test_cli_command():
    runner = CliRunner()
    result = runner.invoke(cli, ['command', '--option', 'value'])
    assert result.exit_code == 0
    assert 'expected output' in result.output
```

---

## Coverage Checklist

### Module Coverage Status

| Module | File | Functions/Classes | Tests Created | Status |
|--------|------|-------------------|---------------|--------|
| config | settings.py | Settings, get_settings, reset_settings | [ ] | Not Started |
| config | paths.py | PathManager, get_path_manager, reset_path_manager | [ ] | Not Started |
| data | loaders.py | load_raw_articles, filter_articles, load_processed_articles, load_embeddings, load_returns_data, save_processed_data | [ ] | Not Started |
| data | processors.py | eliminate_text_after_word, extract_datetime, convert_to_datetime, extract_tickers_from_article, merge_article_components, clean_article_text, process_articles | [ ] | Not Started |
| data | validators.py | DataValidationError, validate_article_dataframe, validate_embeddings_dataframe, validate_returns_dataframe, validate_tickers_list, check_data_quality | [ ] | Not Started |
| embeddings | generators.py | EmbeddingGeneratorError, get_model, clear_model_cache, get_embedding, generate_embeddings, add_embeddings_to_dataframe, get_embedding_dimension | [ ] | Not Started |
| models | kmeans.py | ClusteringError, NewsClusteringModel, find_optimal_k, cluster_train_val_test | [ ] | Not Started |
| models | llama.py | LLAMAParserError, FirmShock, LLAMANewsParser, create_parser | [ ] | Not Started |
| analysis | statistics.py | StatisticsError, split_data, get_e_data, compute_statistics_for_l_values, compute_statistics_for_theta_values | [ ] | Not Started |
| analysis | portfolio.py | PortfolioError, initialize_portfolio, calculate_portfolio_returns, calculate_trading_intensity_statistics, calculate_portfolio_statistics | [ ] | Not Started |
| analysis | backtesting.py | BacktestingError, calculate_trading_strategy_data, process_article_ticker_pair, calculate_average_metrics_by_group | [ ] | Not Started |
| analysis | cluster_selection.py | ClusterSelectionError, calculate_cluster_sharpe_ratios, rank_clusters_by_sharpe, select_clusters_greedy, select_clusters_rank_stable, assign_trading_rules, calculate_spearman_correlation, separate_clusters_by_sr_sign | [ ] | Not Started |
| analysis | trading_calendar.py | TradingCalendarAdjustments | [ ] | Not Started |
| visualization | plotting.py | PlottingError, plot_cluster_distribution, plot_cluster_distributions_by_split, plot_average_cars_by_cluster, plot_cumulative_returns, configure_matplotlib_style, reset_matplotlib_style, plot_time_series_with_ma, plot_histogram_with_density, generate_wordcloud, plot_silhouette_scores | [ ] | Not Started |
| visualization | tables.py | TableGenerationError, generate_cluster_mapping_table, generate_portfolio_statistics_table, generate_trading_intensity_table, generate_llama_shock_mapping_table | [ ] | Not Started |
| utils | financial.py | FinancialUtilsError, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio, calculate_max_drawdown, calculate_cumulative_return, calculate_annualized_return, calculate_annualized_volatility, calculate_portfolio_statistics, calculate_turnover, calculate_trading_costs, calculate_excess_returns, calculate_information_ratio, calculate_beta, calculate_var, calculate_cvar, load_risk_free_rate, download_market_index, _fetch_single_ticker_data, download_stock_returns | [ ] | Not Started |
| utils | text_processing.py | normalize_whitespace, remove_urls, remove_email_addresses, truncate_text, extract_sentences, count_words, extract_numbers, contains_keywords, remove_special_characters, capitalize_first_letter, remove_repeated_punctuation, extract_quoted_text, calculate_text_statistics, vocabulary_filter | [ ] | Not Started |
| cli | main.py | cli, version, info | [ ] | Not Started |
| cli | data_commands.py | load_articles, describe_data, download_returns, fetch_tickers, generate_embeddings | [ ] | Not Started |
| cli | clustering_commands.py | kmeans_clustering | [ ] | Not Started |
| cli | llama_commands.py | llama_parse, llama_clustering | [ ] | Not Started |
| cli | pipeline_commands.py | run_all | [ ] | Not Started |

### Test Status Legend
- **Not Started**: No tests created yet
- **In Progress**: Tests partially created
- **Complete**: All tests created and passing

### Progress Tracking

Use this checklist to track your progress as you create tests:

1. [ ] Config module tests
2. [ ] Data module tests
3. [ ] Embeddings module tests
4. [ ] Models module tests
5. [ ] Analysis module tests
6. [ ] Visualization module tests
7. [ ] Utils module tests
8. [ ] CLI module tests

---

## Additional Notes

### Testing Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Test Naming**: Use descriptive names that explain what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **One Assertion Per Test**: Focus each test on a single behavior
5. **Test Edge Cases**: Don't just test happy paths
6. **Mock External Dependencies**: Use mocks for APIs, file I/O, and external services
7. **Use Fixtures**: Reuse common test data through fixtures
8. **Clean Up**: Ensure tests don't leave side effects (use temp directories, reset state)

### Common Pitfalls to Avoid

1. **Testing Implementation Details**: Test behavior, not implementation
2. **Over-Mocking**: Don't mock everything; only mock external dependencies
3. **Brittle Tests**: Avoid tests that break when implementation changes but behavior is correct
4. **Missing Edge Cases**: Don't forget to test empty inputs, None values, boundary conditions
5. **Ignoring Error Cases**: Test that appropriate errors are raised for invalid inputs

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_config.py

# Run with coverage
pytest --cov=PMRTN --cov-report=html

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/test_config.py::TestSettings::test_settings_load_valid_config
```

---

## Conclusion

This guide provides a comprehensive roadmap for creating unit tests for the entire PMRTN codebase. Follow the module-by-module breakdown, create tests for each function and class, and use the coverage checklist to track progress.

Remember: The goal is not just to achieve 100% coverage, but to ensure that all code behaves correctly and handles edge cases appropriately. Quality over quantity!

Good luck with your testing!


