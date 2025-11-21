"""Cluster selection algorithms for trading strategy construction.

This module provides algorithms for selecting clusters to trade based on
their historical performance (Sharpe ratios). Two main algorithms are implemented:
1. Greedy: Selects top θ clusters by validation Sharpe ratio
2. Rank-Stable: Selects clusters with stable rankings across train/validation splits
"""

from collections import OrderedDict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats


class ClusterSelectionError(Exception):
    """Raised when cluster selection operations fail."""
    pass


def calculate_cluster_sharpe_ratios(
    articles_df: pd.DataFrame,
    ts_dict: Dict[int, pd.DataFrame],
    l_value: int
) -> Dict[Tuple[str, int], float]:
    """Calculate average Sharpe ratios for each (split, cluster) combination.
    
    Parameters
    ----------
    articles_df : pd.DataFrame
        DataFrame with columns: 'split', 'cluster', and article indices.
    ts_dict : Dict[int, pd.DataFrame]
        Dictionary mapping article indices to trading strategy DataFrames.
    l_value : int
        Holding period L to extract Sharpe ratios for.
    
    Returns
    -------
    Dict[Tuple[str, int], float]
        Dictionary mapping (split, cluster) to average Sharpe ratio.
        Example: {('Train', 0): 0.85, ('Validation', 0): 0.92}
    
    Notes
    -----
    The average Sharpe ratio for group g is:
        SR̄_g = (1 / |𝓑_g|) * Σ SR_L^(i,j) for (i,j) in 𝓑_g
    where 𝓑_g := {(i,j) | (i,j) in 𝓑 ∧ cluster(i,j) = g}
    """
    # Accumulate SR values by (split, cluster)
    sr_accumulator: Dict[Tuple[str, int], Dict[str, float]] = {}
    
    for idx, row in articles_df.iterrows():
        ts_data = ts_dict.get(idx)
        
        # Skip if no data or insufficient length
        if ts_data is None or not isinstance(ts_data, pd.DataFrame):
            continue
        if len(ts_data) <= l_value:
            continue
        
        split = row['split']
        cluster = row['cluster']
        sr_l = ts_data.loc[l_value, 'SR']
        
        # Skip NaN values
        if np.isnan(sr_l):
            continue
        
        # Initialize group if needed
        key = (split, cluster)
        if key not in sr_accumulator:
            sr_accumulator[key] = {'sr_sum': 0.0, 'count': 0}
        
        # Accumulate
        sr_accumulator[key]['sr_sum'] += sr_l
        sr_accumulator[key]['count'] += 1
    
    # Calculate averages
    avg_sr_dict = {}
    for key, values in sr_accumulator.items():
        if values['count'] > 0:
            avg_sr_dict[key] = values['sr_sum'] / values['count']
        else:
            avg_sr_dict[key] = np.nan
    
    # Sort by cluster number for consistent ordering
    avg_sr_dict = OrderedDict(sorted(avg_sr_dict.items(), key=lambda x: x[0][1]))
    
    return avg_sr_dict


def rank_clusters_by_sharpe(
    avg_sr_dict: Dict[Tuple[str, int], float]
) -> Dict[str, List[Tuple[int, float]]]:
    """Rank clusters by Sharpe ratio within each split.
    
    Parameters
    ----------
    avg_sr_dict : Dict[Tuple[str, int], float]
        Dictionary mapping (split, cluster) to average Sharpe ratio.
    
    Returns
    -------
    Dict[str, List[Tuple[int, float]]]
        Dictionary mapping split to ranked list of (cluster, avg_sr) tuples.
        Lists are sorted by SR in descending order.
        Example: {'Train': [(5, 1.2), (3, 0.8), ...]}
    
    Notes
    -----
    The rank of cluster g in split s is:
        ℜ_g^s = Σ_{h∈𝓖} 𝟙(SR̄_h^s ≥ SR̄_g^s)
    """
    # Group by split
    split_dict: Dict[str, List[Tuple[int, float]]] = {}
    
    for (split, cluster), avg_sr in avg_sr_dict.items():
        if split not in split_dict:
            split_dict[split] = []
        split_dict[split].append((cluster, avg_sr))
    
    # Sort each split by SR (descending)
    ranking_dict = {}
    for split, cluster_sr_list in split_dict.items():
        ranking_dict[split] = sorted(cluster_sr_list, key=lambda x: x[1], reverse=True)
    
    return ranking_dict


def select_clusters_greedy(
    ranking_dict: Dict[str, List[Tuple[int, float]]],
    avg_sr_dict: Dict[Tuple[str, int], float],
    theta: int,
    train_split: str = 'Train',
    val_split: str = 'Validation'
) -> Tuple[List[int], List[int]]:
    """Select clusters using the Greedy algorithm.
    
    The Greedy algorithm:
    1. Separates clusters by positive vs. negative SR in train/val
    2. Selects top θ clusters from validation set (by absolute SR)
    3. Goes long on positive SR clusters, short on negative SR clusters
    
    Parameters
    ----------
    ranking_dict : Dict[str, List[Tuple[int, float]]]
        Dictionary mapping split to ranked cluster list.
    avg_sr_dict : Dict[Tuple[str, int], float]
        Dictionary mapping (split, cluster) to average Sharpe ratio.
    theta : int
        Number of clusters to trade (in each direction if available).
    train_split : str, default='Train'
        Name of training split.
    val_split : str, default='Validation'
        Name of validation split.
    
    Returns
    -------
    Tuple[List[int], List[int]]
        (long_clusters, short_clusters)
        - long_clusters: Clusters to go long (positive SR)
        - short_clusters: Clusters to go short (negative SR)
    
    Notes
    -----
    Long-traded clusters: 𝓖_θ^+ := {g ∈ 𝓖 | 1 ≤ ℜ_g^val ≤ θ^+}
    Short-traded clusters: 𝓖_θ^- := {g ∈ 𝓖 | k* - θ^- < ℜ_g^val ≤ k*}
    """
    # Get validation ranking
    if val_split not in ranking_dict:
        raise ClusterSelectionError(f"Split '{val_split}' not found in ranking_dict")
    
    val_ranking = ranking_dict[val_split]
    
    # Separate by sign of SR
    positive_clusters = [cluster for cluster, sr in val_ranking if sr > 0]
    negative_clusters = [cluster for cluster, sr in val_ranking if sr < 0]
    
    # Select top θ from each group
    long_clusters = positive_clusters[:min(theta, len(positive_clusters))]
    
    # For short: sort by SR ascending (most negative first), then take top θ
    negative_sorted = sorted(
        [(c, sr) for c, sr in val_ranking if sr < 0],
        key=lambda x: x[1]
    )
    short_clusters = [c for c, sr in negative_sorted[:min(theta, len(negative_clusters))]]
    
    return long_clusters, short_clusters


def select_clusters_rank_stable(
    ranking_dict: Dict[str, List[Tuple[int, float]]],
    avg_sr_dict: Dict[Tuple[str, int], float],
    theta: int,
    train_split: str = 'Train',
    val_split: str = 'Validation',
    max_rank_diff: int = 3
) -> Tuple[List[int], List[int]]:
    """Select clusters using the Rank-Stable algorithm.
    
    The Rank-Stable algorithm:
    1. Calculates rank differences between train and validation
    2. Selects 2*θ clusters with smallest rank differences
    3. Goes long/short only if SR is positive/negative in BOTH splits
    
    Parameters
    ----------
    ranking_dict : Dict[str, List[Tuple[int, float]]]
        Dictionary mapping split to ranked cluster list.
    avg_sr_dict : Dict[Tuple[str, int], float]
        Dictionary mapping (split, cluster) to average Sharpe ratio.
    theta : int
        Target number of clusters to trade (in each direction).
    train_split : str, default='Train'
        Name of training split.
    val_split : str, default='Validation'
        Name of validation split.
    max_rank_diff : int, default=3
        Maximum allowed rank difference (not currently enforced, for future use).
    
    Returns
    -------
    Tuple[List[int], List[int]]
        (long_clusters, short_clusters)
        - long_clusters: Clusters with stable positive SR
        - short_clusters: Clusters with stable negative SR
    
    Notes
    -----
    Rank stability: |ℜ_g^train - ℜ_g^val| should be minimized
    Only trade if sign of SR is consistent across both splits.
    """
    # Validate splits exist
    if train_split not in ranking_dict:
        raise ClusterSelectionError(f"Split '{train_split}' not found in ranking_dict")
    if val_split not in ranking_dict:
        raise ClusterSelectionError(f"Split '{val_split}' not found in ranking_dict")
    
    # Create rank dictionaries
    ranks = {
        split: {cluster: rank for rank, (cluster, _) in enumerate(ranked_list, start=1)}
        for split, ranked_list in ranking_dict.items()
    }
    
    # Get common clusters
    common_clusters = set(ranks[train_split].keys()) & set(ranks[val_split].keys())
    
    # Calculate rank differences
    rank_differences = {
        cluster: abs(ranks[train_split][cluster] - ranks[val_split][cluster])
        for cluster in common_clusters
    }
    
    # Sort by rank difference
    sorted_by_stability = sorted(rank_differences.items(), key=lambda x: x[1])
    
    # Select top 2*θ most stable clusters
    most_stable = [cluster for cluster, _ in sorted_by_stability[:2 * theta]]
    
    # Filter by sign consistency
    long_clusters = [
        cluster for cluster in most_stable
        if avg_sr_dict.get((train_split, cluster), 0) > 0
        and avg_sr_dict.get((val_split, cluster), 0) > 0
    ]
    
    short_clusters = [
        cluster for cluster in most_stable
        if avg_sr_dict.get((train_split, cluster), 0) < 0
        and avg_sr_dict.get((val_split, cluster), 0) < 0
    ]
    
    return long_clusters, short_clusters


def assign_trading_rules(
    articles_df: pd.DataFrame,
    long_clusters: List[int],
    short_clusters: List[int],
    rule_column: str = 'TR'
) -> pd.DataFrame:
    """Assign trading rules to articles based on cluster membership.
    
    Parameters
    ----------
    articles_df : pd.DataFrame
        DataFrame with 'cluster' column.
    long_clusters : List[int]
        List of cluster IDs to go long (+1).
    short_clusters : List[int]
        List of cluster IDs to go short (-1).
    rule_column : str, default='TR'
        Name of column to create/update with trading rules.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added/updated trading rule column.
        Values: +1 (long), -1 (short), 0 (no trade)
    
    Notes
    -----
    Trading rule definition:
        TR_L,θ⟨(i,j), d⟩ := {
            +1  if (i,j) ∈ 𝓑_g ∧ g ∈ 𝓖_θ^+ ∧ d ∈ ℋ^i
             0  if (i,j) ∈ 𝓑_g ∧ g ∉ 𝓖_θ   ∨ d ∉ ℋ^i
            -1  if (i,j) ∈ 𝓑_g ∧ g ∈ 𝓖_θ^- ∧ d ∈ ℋ^i
        }
    """
    # Create a copy to avoid modifying original
    df = articles_df.copy()
    
    # Initialize all rules to 0
    df[rule_column] = 0
    
    # Assign +1 to long clusters
    df.loc[df['cluster'].isin(long_clusters), rule_column] = 1
    
    # Assign -1 to short clusters
    df.loc[df['cluster'].isin(short_clusters), rule_column] = -1
    
    return df


def calculate_spearman_correlation(
    ranking_dict: Dict[str, List[Tuple[int, float]]],
    split1: str,
    split2: str
) -> Tuple[float, List[int]]:
    """Calculate Spearman rank correlation between two splits.
    
    Parameters
    ----------
    ranking_dict : Dict[str, List[Tuple[int, float]]]
        Dictionary mapping split to ranked cluster list.
    split1 : str
        Name of first split (e.g., 'Train').
    split2 : str
        Name of second split (e.g., 'Validation').
    
    Returns
    -------
    Tuple[float, List[int]]
        (correlation, common_clusters)
        - correlation: Spearman rank correlation coefficient
        - common_clusters: List of clusters present in both splits
    """
    # Create rank dictionaries
    ranks = {
        split: {cluster: rank for rank, (cluster, _) in enumerate(ranked_list, start=1)}
        for split, ranked_list in ranking_dict.items()
    }
    
    # Get common clusters
    common_clusters = sorted(set(ranks[split1].keys()) & set(ranks[split2].keys()))
    
    if not common_clusters:
        return np.nan, []
    
    # Extract ranks for common clusters
    ranks1 = [ranks[split1][c] for c in common_clusters]
    ranks2 = [ranks[split2][c] for c in common_clusters]
    
    # Calculate Spearman correlation
    correlation, _ = scipy.stats.spearmanr(ranks1, ranks2)
    
    return correlation, common_clusters


def separate_clusters_by_sr_sign(
    ranking_dict: Dict[str, List[Tuple[int, float]]],
    splits: List[str] = None
) -> Dict[str, Dict[str, Set[int]]]:
    """Separate clusters by sign of Sharpe ratio for each split.
    
    Parameters
    ----------
    ranking_dict : Dict[str, List[Tuple[int, float]]]
        Dictionary mapping split to ranked cluster list.
    splits : List[str], optional
        List of splits to process. If None, processes all splits.
    
    Returns
    -------
    Dict[str, Dict[str, Set[int]]]
        Nested dictionary: {split: {'positive': {clusters}, 'negative': {clusters}}}
        Example: {'Train': {'positive': {0, 5}, 'negative': {3, 7}}}
    """
    if splits is None:
        splits = list(ranking_dict.keys())
    
    result = {}
    
    for split in splits:
        if split not in ranking_dict:
            continue
        
        positive_clusters = set()
        negative_clusters = set()
        
        for cluster, avg_sr in ranking_dict[split]:
            if avg_sr > 0:
                positive_clusters.add(cluster)
            elif avg_sr < 0:
                negative_clusters.add(cluster)
        
        result[split] = {
            'positive': positive_clusters,
            'negative': negative_clusters
        }
    
    return result
