"""Tests for statistics module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from news_market_analysis.analysis import split_data, get_e_data


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    # Create 100 rows with embeddings
    n_samples = 100
    embedding_dim = 10
    
    embeddings = [np.random.rand(embedding_dim).tolist() for _ in range(n_samples)]
    
    df = pd.DataFrame({
        'id': range(n_samples),
        'text': [f'Article {i}' for i in range(n_samples)],
        'embeddings': embeddings,
        'value': np.random.randn(n_samples)
    })
    
    return df


def test_split_data_default_parameters(sample_dataframe):
    """Test split_data with default parameters."""
    result = split_data(sample_dataframe)
    
    # Check all keys are present
    assert 'D' in result
    assert 'D_train' in result
    assert 'D_val' in result
    assert 'D_test' in result
    
    # Check sizes (80% for train+val, 20% for test; then 80% train, 20% val)
    # With 100 samples: 64 train, 16 val, 20 test
    assert len(result['D_train']) == 64
    assert len(result['D_val']) == 16
    assert len(result['D_test']) == 20
    
    # Check that 'split' column was added
    assert 'split' in result['D'].columns
    
    # Check split labels
    assert (result['D'].loc[result['D_train'].index, 'split'] == 'Train').all()
    assert (result['D'].loc[result['D_val'].index, 'split'] == 'Validation').all()
    assert (result['D'].loc[result['D_test'].index, 'split'] == 'Test').all()


def test_split_data_custom_split1(sample_dataframe):
    """Test split_data with custom split1 parameter."""
    result = split_data(sample_dataframe, split1=0.9, split2=0.8)
    
    # With 100 samples: 72 train, 18 val, 10 test
    assert len(result['D_train']) == 72
    assert len(result['D_val']) == 18
    assert len(result['D_test']) == 10


def test_split_data_custom_split2(sample_dataframe):
    """Test split_data with custom split2 parameter."""
    result = split_data(sample_dataframe, split1=0.8, split2=0.75)
    
    # With 100 samples: 60 train, 20 val, 20 test
    assert len(result['D_train']) == 60
    assert len(result['D_val']) == 20
    assert len(result['D_test']) == 20


def test_split_data_sequential_type(sample_dataframe):
    """Test split_data with sequential split type."""
    result = split_data(sample_dataframe, split2_type='sequential')
    
    # Check that indices are sequential
    train_indices = result['D_train'].index.tolist()
    val_indices = result['D_val'].index.tolist()
    test_indices = result['D_test'].index.tolist()
    
    # Train should be first, then val, then test
    assert train_indices == list(range(0, 64))
    assert val_indices == list(range(64, 80))
    assert test_indices == list(range(80, 100))


def test_split_data_random_type(sample_dataframe):
    """Test split_data with random split type."""
    result = split_data(sample_dataframe, split2_type='random', seed=42)
    
    train_indices = result['D_train'].index.tolist()
    val_indices = result['D_val'].index.tolist()
    test_indices = result['D_test'].index.tolist()
    
    # Train and val should not be sequential
    # Test should still be the last 20 rows
    assert test_indices == list(range(80, 100))
    
    # Train and val should be shuffled but from the first 80 rows
    all_train_val = sorted(train_indices + val_indices)
    assert all_train_val == list(range(0, 80))
    
    # Check that they're different from sequential split
    assert train_indices != list(range(0, 64))


def test_split_data_random_reproducible(sample_dataframe):
    """Test that random split is reproducible with same seed."""
    result1 = split_data(sample_dataframe, split2_type='random', seed=42)
    result2 = split_data(sample_dataframe, split2_type='random', seed=42)
    
    # Should get same indices
    assert result1['D_train'].index.tolist() == result2['D_train'].index.tolist()
    assert result1['D_val'].index.tolist() == result2['D_val'].index.tolist()


def test_split_data_random_different_seeds(sample_dataframe):
    """Test that random split gives different results with different seeds."""
    result1 = split_data(sample_dataframe, split2_type='random', seed=42)
    result2 = split_data(sample_dataframe, split2_type='random', seed=123)
    
    # Should get different indices
    assert result1['D_train'].index.tolist() != result2['D_train'].index.tolist()


def test_split_data_verbose_output(sample_dataframe, capsys):
    """Test that verbose parameter prints split information."""
    split_data(sample_dataframe, verbose=True)
    
    captured = capsys.readouterr()
    assert 'SPLIT:' in captured.out
    assert 'Train' in captured.out
    assert 'Validation' in captured.out
    assert 'Test' in captured.out


def test_split_data_invalid_split1():
    """Test split_data with invalid split1 values."""
    df = pd.DataFrame({'values': range(10)})
    
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        split_data(df, split1=0.0)
    
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        split_data(df, split1=1.5)
    
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        split_data(df, split1=-0.1)


def test_split_data_invalid_split2():
    """Test split_data with invalid split2 values."""
    df = pd.DataFrame({'values': range(10)})
    
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        split_data(df, split2=0.0)
    
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        split_data(df, split2=1.5)


def test_split_data_invalid_split2_type():
    """Test split_data with invalid split2_type."""
    df = pd.DataFrame({'values': range(10)})
    
    with pytest.raises(ValueError, match="must be either 'sequential' or 'random'"):
        split_data(df, split2_type='invalid')


def test_split_data_no_overlap(sample_dataframe):
    """Test that train, val, and test sets don't overlap."""
    result = split_data(sample_dataframe)
    
    train_indices = set(result['D_train'].index)
    val_indices = set(result['D_val'].index)
    test_indices = set(result['D_test'].index)
    
    # No overlaps
    assert len(train_indices & val_indices) == 0
    assert len(train_indices & test_indices) == 0
    assert len(val_indices & test_indices) == 0
    
    # All indices covered
    all_indices = train_indices | val_indices | test_indices
    assert len(all_indices) == len(sample_dataframe)


def test_get_e_data_basic(sample_dataframe):
    """Test get_e_data with basic split."""
    splits = split_data(sample_dataframe)
    result = get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])
    
    # Check all keys are present
    assert 'e_train' in result
    assert 'e_val' in result
    assert 'e_test' in result
    assert 'e_train_scaled' in result
    assert 'e_val_scaled' in result
    assert 'e_test_scaled' in result
    assert 'scaler' in result
    
    # Check types
    assert isinstance(result['e_train'], np.ndarray)
    assert isinstance(result['e_val'], np.ndarray)
    assert isinstance(result['e_test'], np.ndarray)
    assert isinstance(result['scaler'], StandardScaler)


def test_get_e_data_shapes(sample_dataframe):
    """Test that get_e_data returns correct shapes."""
    splits = split_data(sample_dataframe)
    result = get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])
    
    # Check shapes match the number of samples and embedding dimension
    assert result['e_train'].shape == (64, 10)
    assert result['e_val'].shape == (16, 10)
    assert result['e_test'].shape == (20, 10)
    assert result['e_train_scaled'].shape == (64, 10)
    assert result['e_val_scaled'].shape == (16, 10)
    assert result['e_test_scaled'].shape == (20, 10)


def test_get_e_data_scaling(sample_dataframe):
    """Test that embeddings are properly scaled."""
    splits = split_data(sample_dataframe)
    result = get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])
    
    # Scaled training data should have mean ≈ 0 and std ≈ 1
    assert np.allclose(result['e_train_scaled'].mean(axis=0), 0, atol=1e-7)
    assert np.allclose(result['e_train_scaled'].std(axis=0), 1, atol=1e-7)
    
    # Validation and test should be scaled using training statistics
    # They won't have mean=0 and std=1, but should be transformed using same scaler
    # Check that scaler was fit on training data
    expected_train_scaled = result['scaler'].transform(result['e_train'])
    assert np.allclose(result['e_train_scaled'], expected_train_scaled)


def test_get_e_data_custom_column_name():
    """Test get_e_data with custom embeddings column name."""
    # Create DataFrame with custom column name
    n_samples = 30
    df = pd.DataFrame({
        'id': range(n_samples),
        'custom_emb': [np.random.rand(5).tolist() for _ in range(n_samples)]
    })
    
    splits = split_data(df, split1=0.8, split2=0.75)
    result = get_e_data(
        splits['D_train'],
        splits['D_val'],
        splits['D_test'],
        embeddings_col='custom_emb'
    )
    
    # Should work with custom column name
    assert result['e_train'].shape[1] == 5
    assert result['e_val'].shape[1] == 5
    assert result['e_test'].shape[1] == 5


def test_get_e_data_missing_column():
    """Test get_e_data raises error when embeddings column is missing."""
    df = pd.DataFrame({
        'id': range(10),
        'value': range(10)
    })
    
    splits = split_data(df)
    
    with pytest.raises(KeyError, match="Column 'embeddings' not found"):
        get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])


def test_get_e_data_invalid_embeddings():
    """Test get_e_data raises error when embeddings are invalid."""
    df = pd.DataFrame({
        'id': range(10),
        'embeddings': ['invalid'] * 10  # String instead of list/array
    })
    
    splits = split_data(df)
    
    # The error is raised by sklearn when trying to scale invalid data
    with pytest.raises(ValueError, match="could not convert string to float"):
        get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])


def test_get_e_data_scaler_consistency():
    """Test that the same scaler is used for all sets."""
    n_samples = 50
    df = pd.DataFrame({
        'embeddings': [np.random.rand(8).tolist() for _ in range(n_samples)]
    })
    
    splits = split_data(df)
    result = get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])
    
    # Manually scale validation and test using the returned scaler
    manual_val_scaled = result['scaler'].transform(result['e_val'])
    manual_test_scaled = result['scaler'].transform(result['e_test'])
    
    # Should match the scaled versions in result
    assert np.allclose(result['e_val_scaled'], manual_val_scaled)
    assert np.allclose(result['e_test_scaled'], manual_test_scaled)


def test_get_e_data_preserves_order():
    """Test that get_e_data preserves the order of samples."""
    n_samples = 40
    embeddings = [np.array([float(i)] * 5) for i in range(n_samples)]
    df = pd.DataFrame({
        'id': range(n_samples),
        'embeddings': [e.tolist() for e in embeddings]
    })
    
    splits = split_data(df)
    result = get_e_data(splits['D_train'], splits['D_val'], splits['D_test'])
    
    # Check that first embedding in train corresponds to first row in D_train
    first_train_id = splits['D_train'].iloc[0]['id']
    first_train_emb = embeddings[first_train_id]
    
    # The raw embedding should match (before scaling)
    assert np.allclose(result['e_train'][0], first_train_emb)
