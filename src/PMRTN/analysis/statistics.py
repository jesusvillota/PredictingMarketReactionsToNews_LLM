"""Statistical analysis utilities for news market analysis.

This module provides functions for data splitting, embedding scaling, and statistical
calculations used throughout the analysis pipeline.
"""

from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import functions from other modules that we'll need
from PMRTN.analysis.portfolio import (
    initialize_portfolio,
    calculate_portfolio_returns,
    calculate_portfolio_statistics
)
from PMRTN.analysis.cluster_selection import (
    calculate_cluster_sharpe_ratios,
    rank_clusters_by_sharpe,
    select_clusters_greedy,
    select_clusters_rank_stable
)


class StatisticsError(Exception):
    """Raised when statistical computation operations fail."""
    pass


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


def compute_statistics_for_l_values(
    articles_df: pd.DataFrame,
    ts_dict: Dict[int, pd.DataFrame],
    trading_days: List[pd.Timestamp],
    l_values: List[int],
    trading_rule_col: str = 'TR',
    trading_cost_bps: float = 10.0,
    algorithms: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    theta: float = 0.2,
    verbose: bool = False
) -> Dict[int, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute portfolio statistics for multiple L (holding period) values.
    
    This function iterates over multiple L values and computes comprehensive portfolio
    statistics for each combination of L, algorithm, and data split. It's designed for
    parameter sweep analysis to find the optimal holding period.
    
    Args:
        articles_df: DataFrame with columns: 'split', 'cluster', 'date_affect', and article indices.
                    Must have trading rule column specified by trading_rule_col.
        ts_dict: Dictionary mapping article indices to trading strategy DataFrames.
                Each DataFrame must have columns for different L values with returns and Sharpe ratios.
        trading_days: List of trading days in chronological order.
        l_values: List of L (holding period) values to compute statistics for.
        trading_rule_col: Name of column in articles_df containing trading rules (default 'TR').
        trading_cost_bps: Trading costs in basis points (default 10.0).
        algorithms: List of cluster selection algorithms to use. Options: ['Greedy', 'Stable'].
                   If None, uses ['Greedy', 'Stable'].
        splits: List of data splits to compute statistics for. Options: ['All', 'Train', 'Validation', 'Test'].
               If None, uses ['All', 'Train', 'Validation', 'Test'].
        theta: Cluster selection parameter (fraction of clusters to select, default 0.2).
        verbose: Whether to print progress information (default False).
    
    Returns:
        Nested dictionary with structure:
        {
            L_value: {
                algorithm: {
                    split: {
                        'cumulative_return_gross': float,
                        'cumulative_return_net': float,
                        'annualized_return_gross': float,
                        'annualized_return_net': float,
                        'volatility_gross': float,
                        'volatility_net': float,
                        'sharpe_ratio_gross': float,
                        'sharpe_ratio_net': float,
                        'max_drawdown_gross': float,
                        'max_drawdown_net': float,
                        'calmar_ratio_gross': float,
                        'calmar_ratio_net': float,
                    }
                }
            }
        }
    
    Raises:
        StatisticsError: If required columns are missing or computation fails.
        ValueError: If parameters are invalid.
    
    Example:
        >>> l_values = [5, 10, 15, 20]
        >>> stats = compute_statistics_for_l_values(
        ...     articles_df, ts_dict, trading_days, l_values,
        ...     algorithms=['Greedy'], splits=['Test'], verbose=True
        ... )
        >>> # Access statistics: stats[10]['Greedy']['Test']['sharpe_ratio_net']
    
    Notes:
        - Reuses portfolio construction for efficiency (constructs once per L, algorithm)
        - Each L value requires re-running cluster selection and portfolio construction
        - Statistics are computed for both gross and net returns
        - Net returns include trading costs specified by trading_cost_bps
    """
    # Validate inputs
    if trading_rule_col not in articles_df.columns:
        raise StatisticsError(f"Trading rule column '{trading_rule_col}' not found in articles_df")
    
    if not l_values or len(l_values) == 0:
        raise ValueError("l_values cannot be empty")
    
    if algorithms is None:
        algorithms = ['Greedy', 'Stable']
    
    if splits is None:
        splits = ['All', 'Train', 'Validation', 'Test']
    
    # Validate algorithms
    valid_algorithms = ['Greedy', 'Stable']
    for algo in algorithms:
        if algo not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{algo}'. Must be one of {valid_algorithms}")
    
    # Validate splits
    valid_splits = ['All', 'Train', 'Validation', 'Test']
    for split in splits:
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
    
    # Initialize results dictionary
    results: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {}
    
    # Iterate over L values
    for l_idx, l_value in enumerate(l_values):
        if verbose:
            print(f"\nComputing statistics for L={l_value} ({l_idx + 1}/{len(l_values)})...")
        
        results[l_value] = {}
        
        # Iterate over algorithms
        for algorithm in algorithms:
            if verbose:
                print(f"  Algorithm: {algorithm}")
            
            results[l_value][algorithm] = {}
            
            # Calculate Sharpe ratios and rankings for this L value
            try:
                avg_sr_dict = calculate_cluster_sharpe_ratios(
                    articles_df=articles_df,
                    ts_dict=ts_dict,
                    l_value=l_value
                )
                ranking_dict = rank_clusters_by_sharpe(avg_sr_dict)
            except Exception as e:
                raise StatisticsError(f"Failed to calculate SRs/rankings for L={l_value}: {e}")
            
            # Calculate number of clusters to select (theta as integer)
            unique_clusters = articles_df['cluster'].nunique()
            theta_int = max(1, int(theta * unique_clusters))
            
            # Select clusters based on algorithm
            try:
                if algorithm == 'Greedy':
                    long_clusters, short_clusters = select_clusters_greedy(
                        ranking_dict=ranking_dict,
                        avg_sr_dict=avg_sr_dict,
                        theta=theta_int
                    )
                elif algorithm == 'Stable':
                    long_clusters, short_clusters = select_clusters_rank_stable(
                        ranking_dict=ranking_dict,
                        avg_sr_dict=avg_sr_dict,
                        theta=theta_int
                    )
                else:
                    raise StatisticsError(f"Unknown algorithm: {algorithm}")
                
                # Combine long and short clusters
                selected_clusters = list(set(long_clusters + short_clusters))
            except Exception as e:
                raise StatisticsError(f"Cluster selection failed for L={l_value}, {algorithm}: {e}")
            
            # Filter articles to selected clusters
            articles_filtered = cast(pd.DataFrame, articles_df[
                articles_df['cluster'].isin(selected_clusters)
            ].copy())
            
            if len(articles_filtered) == 0:
                if verbose:
                    print(f"    Warning: No articles selected for {algorithm} at L={l_value}")
                # Fill with NaN statistics
                for split in splits:
                    results[l_value][algorithm][split] = {
                        'cumulative_return_gross': np.nan,
                        'cumulative_return_net': np.nan,
                        'annualized_return_gross': np.nan,
                        'annualized_return_net': np.nan,
                        'volatility_gross': np.nan,
                        'volatility_net': np.nan,
                        'sharpe_ratio_gross': np.nan,
                        'sharpe_ratio_net': np.nan,
                        'max_drawdown_gross': np.nan,
                        'max_drawdown_net': np.nan,
                        'calmar_ratio_gross': np.nan,
                        'calmar_ratio_net': np.nan,
                    }
                continue

            # Calculate portfolio returns
            try:
                portfolio_results = calculate_portfolio_returns(
                    articles_df=articles_filtered,
                    trading_days=trading_days,
                    ts_dict=ts_dict,
                    l_value=l_value,
                    trading_rule_col=trading_rule_col,
                    trading_cost_bps=trading_cost_bps,
                    verbose=False
                )
            except Exception as e:
                raise StatisticsError(
                    f"Portfolio returns calculation failed for L={l_value}, {algorithm}: {e}"
                )

            # Compute statistics for each split
            for split in splits:
                if split not in portfolio_results:
                    if verbose:
                        print(f"    Warning: Split '{split}' not found in portfolio_results")
                    continue
                
                returns_df = portfolio_results[split]
                
                try:
                    stats = calculate_portfolio_statistics(
                        returns_df=returns_df,
                        risk_free_rate=0.0,
                        trading_days_per_year=252
                    )
                    results[l_value][algorithm][split] = stats
                    
                    if verbose:
                        sharpe_net = stats.get('sharpe_ratio_net', np.nan)
                        print(f"    {split}: Sharpe Ratio (net) = {sharpe_net:.4f}")
                        
                except Exception as e:
                    raise StatisticsError(
                        f"Statistics calculation failed for L={l_value}, {algorithm}, {split}: {e}"
                    )
    
    if verbose:
        print(f"\nCompleted statistics computation for {len(l_values)} L values.")
    
    return results


def compute_statistics_for_theta_values(
    articles_df: pd.DataFrame,
    ts_dict: Dict[int, pd.DataFrame],
    trading_days: List[pd.Timestamp],
    l_value: int,
    theta_values: List[float],
    trading_rule_col: str = 'TR',
    trading_cost_bps: float = 10.0,
    algorithms: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict[float, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute portfolio statistics for multiple θ (cluster selection parameter) values.
    
    This function iterates over multiple θ values and computes comprehensive portfolio
    statistics for each combination of θ, algorithm, and data split. It's designed for
    parameter sweep analysis to find the optimal cluster selection threshold.
    
    Args:
        articles_df: DataFrame with columns: 'split', 'cluster', 'date_affect', and article indices.
                    Must have trading rule column specified by trading_rule_col.
        ts_dict: Dictionary mapping article indices to trading strategy DataFrames.
                Each DataFrame must have columns for different L values with returns and Sharpe ratios.
        trading_days: List of trading days in chronological order.
        l_value: Fixed L (holding period) value to use for all computations.
        theta_values: List of θ values (fraction of clusters to select) to test.
                     Each value should be in (0, 1].
        trading_rule_col: Name of column in articles_df containing trading rules (default 'TR').
        trading_cost_bps: Trading costs in basis points (default 10.0).
        algorithms: List of cluster selection algorithms to use. Options: ['Greedy', 'Stable'].
                   If None, uses ['Greedy', 'Stable'].
        splits: List of data splits to compute statistics for. Options: ['All', 'Train', 'Validation', 'Test'].
               If None, uses ['All', 'Train', 'Validation', 'Test'].
        verbose: Whether to print progress information (default False).
    
    Returns:
        Nested dictionary with structure:
        {
            theta_value: {
                algorithm: {
                    split: {
                        'cumulative_return_gross': float,
                        'cumulative_return_net': float,
                        'annualized_return_gross': float,
                        'annualized_return_net': float,
                        'volatility_gross': float,
                        'volatility_net': float,
                        'sharpe_ratio_gross': float,
                        'sharpe_ratio_net': float,
                        'max_drawdown_gross': float,
                        'max_drawdown_net': float,
                        'calmar_ratio_gross': float,
                        'calmar_ratio_net': float,
                    }
                }
            }
        }
    
    Raises:
        StatisticsError: If required columns are missing or computation fails.
        ValueError: If parameters are invalid.
    
    Example:
        >>> theta_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> stats = compute_statistics_for_theta_values(
        ...     articles_df, ts_dict, trading_days, l_value=10,
        ...     theta_values=theta_values, algorithms=['Greedy'], verbose=True
        ... )
        >>> # Access statistics: stats[0.2]['Greedy']['Test']['sharpe_ratio_net']
    
    Notes:
        - θ determines the fraction of top-performing clusters to trade
        - For θ=0.2, the top 20% of clusters (by validation Sharpe ratio) are selected
        - Each θ value requires re-running cluster selection and portfolio construction
        - Statistics are computed for both gross and net returns
        - Net returns include trading costs specified by trading_cost_bps
    """
    # Validate inputs
    if trading_rule_col not in articles_df.columns:
        raise StatisticsError(f"Trading rule column '{trading_rule_col}' not found in articles_df")
    
    if not theta_values or len(theta_values) == 0:
        raise ValueError("theta_values cannot be empty")
    
    # Validate theta values are in valid range
    for theta in theta_values:
        if not (0 < theta <= 1):
            raise ValueError(f"theta value {theta} is out of range (0, 1]")
    
    if algorithms is None:
        algorithms = ['Greedy', 'Stable']
    
    if splits is None:
        splits = ['All', 'Train', 'Validation', 'Test']
    
    # Validate algorithms
    valid_algorithms = ['Greedy', 'Stable']
    for algo in algorithms:
        if algo not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{algo}'. Must be one of {valid_algorithms}")
    
    # Validate splits
    valid_splits = ['All', 'Train', 'Validation', 'Test']
    for split in splits:
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
    
    # Initialize results dictionary
    results: Dict[float, Dict[str, Dict[str, Dict[str, float]]]] = {}
    
    # Iterate over theta values
    for theta_idx, theta in enumerate(theta_values):
        if verbose:
            print(f"\nComputing statistics for θ={theta} ({theta_idx + 1}/{len(theta_values)})...")
        
        results[theta] = {}
        
        # Iterate over algorithms
        for algorithm in algorithms:
            if verbose:
                print(f"  Algorithm: {algorithm}")
            
            results[theta][algorithm] = {}
            
            # Calculate Sharpe ratios and rankings for this theta value
            try:
                avg_sr_dict = calculate_cluster_sharpe_ratios(
                    articles_df=articles_df,
                    ts_dict=ts_dict,
                    l_value=l_value
                )
                ranking_dict = rank_clusters_by_sharpe(avg_sr_dict)
            except Exception as e:
                raise StatisticsError(f"Failed to calculate SRs/rankings for θ={theta}: {e}")
            
            # Calculate number of clusters to select (theta as integer)
            unique_clusters = articles_df['cluster'].nunique()
            theta_int = max(1, int(theta * unique_clusters))
            
            # Select clusters based on algorithm
            try:
                if algorithm == 'Greedy':
                    long_clusters, short_clusters = select_clusters_greedy(
                        ranking_dict=ranking_dict,
                        avg_sr_dict=avg_sr_dict,
                        theta=theta_int
                    )
                elif algorithm == 'Stable':
                    long_clusters, short_clusters = select_clusters_rank_stable(
                        ranking_dict=ranking_dict,
                        avg_sr_dict=avg_sr_dict,
                        theta=theta_int
                    )
                else:
                    raise StatisticsError(f"Unknown algorithm: {algorithm}")
                
                # Combine long and short clusters
                selected_clusters = list(set(long_clusters + short_clusters))
            except Exception as e:
                raise StatisticsError(
                    f"Cluster selection failed for θ={theta}, {algorithm}: {e}"
                )
            
            # Filter articles to selected clusters
            articles_filtered = cast(pd.DataFrame, articles_df[
                articles_df['cluster'].isin(selected_clusters)
            ].copy())
            
            if len(articles_filtered) == 0:
                if verbose:
                    print(f"    Warning: No articles selected for {algorithm} at θ={theta}")
                # Fill with NaN statistics
                for split in splits:
                    results[theta][algorithm][split] = {
                        'cumulative_return_gross': np.nan,
                        'cumulative_return_net': np.nan,
                        'annualized_return_gross': np.nan,
                        'annualized_return_net': np.nan,
                        'volatility_gross': np.nan,
                        'volatility_net': np.nan,
                        'sharpe_ratio_gross': np.nan,
                        'sharpe_ratio_net': np.nan,
                        'max_drawdown_gross': np.nan,
                        'max_drawdown_net': np.nan,
                        'calmar_ratio_gross': np.nan,
                        'calmar_ratio_net': np.nan,
                    }
                continue

            # Calculate portfolio returns
            try:
                portfolio_results = calculate_portfolio_returns(
                    articles_df=articles_filtered,
                    trading_days=trading_days,
                    ts_dict=ts_dict,
                    l_value=l_value,
                    trading_rule_col=trading_rule_col,
                    trading_cost_bps=trading_cost_bps,
                    verbose=False
                )
            except Exception as e:
                raise StatisticsError(
                    f"Portfolio returns calculation failed for θ={theta}, {algorithm}: {e}"
                )

            # Compute statistics for each split
            for split in splits:
                if split not in portfolio_results:
                    if verbose:
                        print(f"    Warning: Split '{split}' not found in portfolio_results")
                    continue
                
                returns_df = portfolio_results[split]
                
                try:
                    stats = calculate_portfolio_statistics(
                        returns_df=returns_df,
                        risk_free_rate=0.0,
                        trading_days_per_year=252
                    )
                    results[theta][algorithm][split] = stats
                    
                    if verbose:
                        sharpe_net = stats.get('sharpe_ratio_net', np.nan)
                        print(f"    {split}: Sharpe Ratio (net) = {sharpe_net:.4f}")
                        
                except Exception as e:
                    raise StatisticsError(
                        f"Statistics calculation failed for θ={theta}, {algorithm}, {split}: {e}"
                    )
    
    if verbose:
        print(f"\nCompleted statistics computation for {len(theta_values)} θ values.")
    
    return results
