"""Statistical analysis utilities for news market analysis.

This module provides functions for data splitting, embedding scaling, and statistical
calculations used throughout the analysis pipeline.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_data(
    df: pd.DataFrame,
    split1: float = 0.8,
    split2: float = 0.8,
    split2_type: str = 'sequential',
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """Split dataset into training, validation, and test sets.
    
    This function performs a two-stage split:
    1. First split: separate out test set (using split1)
    2. Second split: divide remaining data into train and validation (using split2)
    
    The second split can be either sequential (chronological) or random.
    
    Args:
        df: Input DataFrame containing the data to split
        split1: Proportion of data for training+validation (default 0.8)
        split2: Proportion of split1 data for training (default 0.8)
        split2_type: Type of train/val split - 'sequential' or 'random' (default 'sequential')
        seed: Random seed for reproducibility when split2_type='random' (default 42)
        verbose: Whether to print split information (default False)
        
    Returns:
        Dictionary containing:
            - 'D': Original DataFrame with 'split' column added
            - 'D_train': Training set DataFrame
            - 'D_val': Validation set DataFrame
            - 'D_test': Test set DataFrame
            
    Raises:
        ValueError: If split1 or split2 not in (0, 1] or split2_type is invalid
        
    Example:
        >>> data = pd.DataFrame({'values': range(100)})
        >>> splits = split_data(data, split1=0.8, split2=0.75)
        >>> # Result: 60% train, 20% validation, 20% test
    """
    if not (0 < split1 <= 1) or not (0 < split2 <= 1):
        raise ValueError("`split1` and `split2` must be between 0 and 1.")
    
    if split2_type not in ['sequential', 'random']:
        raise ValueError("`split2_type` must be either 'sequential' or 'random'.")

    n_split1 = int(split1 * df.shape[0])
    n_split2 = int(split2 * n_split1)

    # Create the test set (last portion of data)
    df_test = df.iloc[n_split1:]

    if split2_type == 'sequential':
        # Sequential split: first portion for training, middle for validation
        df_train = df.iloc[:n_split2]
        df_val = df.iloc[n_split2:n_split1]

    elif split2_type == 'random':
        # Random split: sample from first portion
        df_split2 = df.iloc[:n_split1]
        df_train = df_split2.sample(n=n_split2, random_state=seed)
        df_val = df_split2.drop(df_train.index)

    # Add a new column to indicate the split each row belongs to
    df_new = df.copy()
    df_new.loc[df_train.index, 'split'] = 'Train'
    df_new.loc[df_val.index, 'split'] = 'Validation'
    df_new.loc[df_test.index, 'split'] = 'Test'

    split_data_dict = {
        'D': df_new,
        'D_train': df_train,
        'D_val': df_val,
        'D_test': df_test,
    }

    if verbose:
        train_percentage = split1 * split2 * 100
        val_percentage = split1 * (1 - split2) * 100
        test_percentage = (1 - split1) * 100
        print(
            f"SPLIT: [ Train ({train_percentage:.2f}%) | "
            f"Validation ({val_percentage:.2f}%) | "
            f"Test ({test_percentage:.2f}%) ] ---- "
            f"Train-Validation split: {split2_type}"
        )

    return split_data_dict


def get_e_data(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    embeddings_col: str = 'embeddings'
) -> Dict[str, np.ndarray]:
    """Extract and scale embeddings from DataFrames.
    
    This function:
    1. Extracts embedding vectors from the specified column
    2. Converts them to numpy arrays
    3. Fits a StandardScaler on training embeddings
    4. Applies the scaler to all three sets
    
    Args:
        df_train: Training set DataFrame
        df_val: Validation set DataFrame
        df_test: Test set DataFrame
        embeddings_col: Name of column containing embeddings (default 'embeddings')
        
    Returns:
        Dictionary containing:
            - 'e_train': Raw training embeddings
            - 'e_val': Raw validation embeddings
            - 'e_test': Raw test embeddings
            - 'e_train_scaled': Scaled training embeddings
            - 'e_val_scaled': Scaled validation embeddings
            - 'e_test_scaled': Scaled test embeddings
            - 'scaler': Fitted StandardScaler instance
            
    Raises:
        KeyError: If embeddings_col not found in DataFrames
        ValueError: If embeddings cannot be converted to arrays
        
    Example:
        >>> e_data = get_e_data(df_train, df_val, df_test)
        >>> X_train = e_data['e_train_scaled']
        >>> X_val = e_data['e_val_scaled']
    """
    # Extracting and converting embeddings to numpy arrays
    try:
        e_train = np.array(df_train[embeddings_col].tolist())
        e_val = np.array(df_val[embeddings_col].tolist())
        e_test = np.array(df_test[embeddings_col].tolist())
    except KeyError as e:
        raise KeyError(f"Column '{embeddings_col}' not found in DataFrame: {e}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error converting embeddings to numpy array: {e}")

    # Scaling the embeddings using StandardScaler
    scaler = StandardScaler()
    e_train_scaled = scaler.fit_transform(e_train)
    e_val_scaled = scaler.transform(e_val)
    e_test_scaled = scaler.transform(e_test)

    e_data = {
        'e_train': e_train,
        'e_val': e_val,
        'e_test': e_test,
        'e_train_scaled': e_train_scaled,
        'e_val_scaled': e_val_scaled,
        'e_test_scaled': e_test_scaled,
        'scaler': scaler
    }
    
    return e_data
