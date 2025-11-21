"""Portfolio construction and backtesting utilities.

This module provides functions for constructing and backtesting trading portfolios
based on cluster trading signals, including return calculation, turnover tracking,
and performance statistics computation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PortfolioError(Exception):
    """Raised when portfolio operations fail."""
    pass


def initialize_portfolio(
    articles_df: pd.DataFrame,
    trading_days: List[pd.Timestamp],
    l_value: int
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[pd.Timestamp]]]:
    """Initialize portfolio DataFrames for tracking returns by split.
    
    Parameters
    ----------
    articles_df : pd.DataFrame
        DataFrame with 'split', 'date_affect' columns.
    trading_days : List[pd.Timestamp]
        Complete list of trading days in chronological order.
    l_value : int
        Holding period L in trading days. Used to extend timeline beyond last article.
    
    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, List[pd.Timestamp]]]
        (r_P_dict, trading_days_dict)
        - r_P_dict: Dict mapping split names to return DataFrames
        - trading_days_dict: Dict mapping split names to their trading day lists
    
    Notes
    -----
    Creates separate DataFrames for 'All', 'Train', 'Validation', 'Test' splits.
    Timeline construction ensures proper alignment with article publication dates.
    Timeline is extended by L days after the last article to account for holding period.
    """
    # Get split DataFrames
    splits_data = {
        'Train': articles_df[articles_df['split'] == 'Train'],
        'Validation': articles_df[articles_df['split'] == 'Validation'],
        'Test': articles_df[articles_df['split'] == 'Test']
    }
    
    # Get first and last days for each split
    indices = {}
    for split_name, split_data in splits_data.items():
        if len(split_data) == 0:
            continue
        first_day = split_data['date_affect'].min()
        last_day = split_data['date_affect'].max()
        
        indices[f'first_day_{split_name}_index'] = trading_days.index(first_day)
        indices[f'last_day_{split_name}_index'] = trading_days.index(last_day)
    
    # Extract individual indices
    first_train_idx = indices.get('first_day_Train_index', 0)
    last_train_idx = indices.get('last_day_Train_index', len(trading_days) - 1)
    first_val_idx = indices.get('first_day_Validation_index', last_train_idx)
    last_val_idx = indices.get('last_day_Validation_index', len(trading_days) - 1)
    first_test_idx = indices.get('first_day_Test_index', last_val_idx)
    last_test_idx = indices.get('last_day_Test_index', len(trading_days) - 1)
    
    # Extend the end indices by L days to account for holding period
    # This ensures we can track portfolio performance for L days after the last article
    last_test_idx_extended = min(last_test_idx + l_value, len(trading_days) - 1)
    last_train_idx_extended = min(last_train_idx + l_value, len(trading_days) - 1)
    last_val_idx_extended = min(last_val_idx + l_value, len(trading_days) - 1)
    
    # Create trading day timelines
    trading_days_all = trading_days[first_train_idx:last_test_idx_extended + 1]
    trading_days_train = trading_days[first_train_idx:last_train_idx_extended + 1]
    trading_days_val = trading_days[last_train_idx:last_val_idx_extended + 1]
    trading_days_test = trading_days[last_val_idx:last_test_idx_extended + 1]
    
    # Initialize return DataFrames
    r_p_all = pd.DataFrame({'returns': 0.0}, index=trading_days_all)
    r_p_train = pd.DataFrame({'returns': 0.0}, index=trading_days_train)
    r_p_val = pd.DataFrame({'returns': 0.0}, index=trading_days_val)
    r_p_test = pd.DataFrame({'returns': 0.0}, index=trading_days_test)
    
    r_p_dict = {
        'All': r_p_all,
        'Train': r_p_train,
        'Validation': r_p_val,
        'Test': r_p_test
    }
    
    trading_days_dict = {
        'All': trading_days_all,
        'Train': trading_days_train,
        'Validation': trading_days_val,
        'Test': trading_days_test
    }
    
    return r_p_dict, trading_days_dict


def calculate_portfolio_returns(
    articles_df: pd.DataFrame,
    trading_days: List[pd.Timestamp],
    l_value: int,
    ts_dict: Dict[int, pd.DataFrame],
    trading_rule_col: str = 'TR',
    trading_cost_bps: float = 10.0,
    verbose: bool = False
) -> Dict:
    """Calculate portfolio returns with trading costs and turnover tracking.
    
    This function implements the portfolio construction and backtesting:
    1. For each trading day, identify active positions (within L days of publication)
    2. Calculate weighted portfolio returns based on abnormal returns
    3. Track turnover and apply trading costs
    4. Compute both gross and net returns
    
    Parameters
    ----------
    articles_df : pd.DataFrame
        DataFrame with columns: 'tickers', 'date_affect', 'split', 'cluster', and trading_rule_col.
    trading_days : List[pd.Timestamp]
        Complete list of trading days in chronological order.
    l_value : int
        Holding period L in trading days.
    ts_dict : Dict[int, pd.DataFrame]
        Dictionary mapping article indices to trading strategy DataFrames.
    trading_rule_col : str, default='TR'
        Column name containing trading rules (+1=long, -1=short, 0=no trade).
    trading_cost_bps : float, default=10.0
        Trading costs in basis points (10 bps = 0.1%).
    verbose : bool, default=False
        If True, print detailed progress information.
    
    Returns
    -------
    Dict
        Dictionary containing:
        - 'All', 'Train', 'Validation', 'Test': DataFrames with gross_returns and net_returns
        - 'trading_signal_evolution': Dict of DataFrames tracking number of positions
        - 'turnover': DataFrame tracking daily turnover
        - 'turnover_stats': Dict with average turnover by split
    
    Notes
    -----
    Portfolio return on day d:
        r_d^𝓟 = (1 / |𝓟_d|) * Σ_{(i,j)∈𝓟_d} TR_L,θ⟨(i,j), d⟩ · AR_d^(i,j)
    
    where 𝓟_d := {(i,j) ∈ 𝓑 | d ∈ 𝓗^i ∧ TR_L,θ⟨(i,j), d⟩ ≠ 0}
    and 𝓗^i := {d ∈ 𝔇̃ | d̃_0^i ≤ d ≤ d̃_0^i + L}
    
    Turnover:
        τ_d = Σ|w_d^i - w_{d-1}^i| / Σ|w_d^i|
    where w_d^i is the weight of position i on day d.
    """
    # Initialize portfolio
    r_p_dict, trading_days_dict = initialize_portfolio(articles_df, trading_days, l_value)
    trading_days_all = trading_days_dict['All']
    
    # Initialize tracking DataFrames
    trading_signal_evolution = {
        split: pd.DataFrame({'total_trading_signal': 0}, index=trading_days_dict[split])
        for split in ['All', 'Train', 'Validation', 'Test']
    }
    
    # Create a union of all trading days from all splits to ensure turnover_tracking covers everything
    all_split_days = set(trading_days_all)
    for split in ['Train', 'Validation', 'Test']:
        all_split_days.update(trading_days_dict[split])
    all_split_days_sorted = sorted(list(all_split_days))
    
    turnover_tracking = pd.DataFrame({'turnover': 0.0}, index=all_split_days_sorted)
    previous_positions: Dict[str, float] = {}
    
    # Initialize return columns
    for split in ['All', 'Train', 'Validation', 'Test']:
        r_p_dict[split]['gross_returns'] = 0.0
        r_p_dict[split]['net_returns'] = 0.0
        if 'returns' in r_p_dict[split].columns:
            r_p_dict[split] = r_p_dict[split].drop(columns=['returns'])
    
    # Process each trading day
    for day in trading_days_all:
        day_idx = trading_days_all.index(day)
        
        if verbose:
            print(f'\n{"=" * 100}')
            print(f'Day: {day} | Index: {day_idx}')
            print(f'{"=" * 100}')
        
        # Find L trading days before current day
        if day_idx < l_value:
            l_days_before_idx = 0
        else:
            l_days_before_idx = day_idx - l_value
        l_days_before = trading_days[trading_days.index(day) - (day_idx - l_days_before_idx)]
        
        # Select active portfolio positions
        # Position is active if: article date in [day-L, day] AND trading rule != 0
        portfolio = articles_df[
            (articles_df['date_affect'] >= l_days_before) &
            (articles_df['date_affect'] <= day) &
            (articles_df[trading_rule_col] != 0)
        ]
        
        total_trading_signal = 0.0
        total_weighted_return = 0.0
        current_positions: Dict[str, float] = {}
        
        # Process each position
        for idx, row in portfolio.iterrows():
            ticker = row['tickers']
            split = row['split']
            date_affect = row['date_affect']
            trading_rule = row[trading_rule_col]
            
            # Track current position
            current_positions[ticker] = trading_rule
            
            # Calculate days since publication
            date_affect_idx = trading_days_all.index(date_affect)
            days_since_publication = day_idx - date_affect_idx
            
            # Get trading strategy data
            ts_data = ts_dict.get(idx)
            if ts_data is None or not isinstance(ts_data, pd.DataFrame):
                if verbose:
                    print(f"  ✕ No data for {ticker} (idx={idx})")
                continue
            
            if days_since_publication >= len(ts_data):
                if verbose:
                    print(f"  ✕ Insufficient data for {ticker} (need {days_since_publication} days)")
                continue
            
            # Get abnormal return
            ar = ts_data.loc[days_since_publication, 'AR']
            
            if np.isnan(ar):
                if verbose:
                    print(f"  ✕ NaN AR for {ticker}")
                continue
            
            # Accumulate weighted returns
            position_return = ar * trading_rule
            total_weighted_return += position_return
            total_trading_signal += abs(trading_rule)
            
            if verbose:
                print(f"  ✓ {ticker}: AR={ar:.4f}, TR={trading_rule:+d}, "
                      f"Contrib={position_return:.4f}")
        
        # Calculate turnover
        daily_turnover = 0.0
        if previous_positions:
            position_changes = 0.0
            total_position_size = sum(abs(pos) for pos in current_positions.values())
            
            all_tickers = set(current_positions.keys()) | set(previous_positions.keys())
            for ticker in all_tickers:
                curr_pos = current_positions.get(ticker, 0.0)
                prev_pos = previous_positions.get(ticker, 0.0)
                position_changes += abs(curr_pos - prev_pos)
            
            if total_position_size > 0:
                daily_turnover = position_changes / total_position_size
        
        turnover_tracking.loc[day, 'turnover'] = daily_turnover
        previous_positions = current_positions.copy()
        
        # Store trading signal
        trading_signal_evolution['All'].loc[day, 'total_trading_signal'] = total_trading_signal
        for split_name in ['Train', 'Validation', 'Test']:
            if day in trading_days_dict[split_name]:
                trading_signal_evolution[split_name].loc[day, 'total_trading_signal'] = (
                    total_trading_signal
                )
        
        # Calculate returns
        gross_return = (
            total_weighted_return / total_trading_signal if total_trading_signal > 0 else 0.0
        )
        trading_costs = daily_turnover * (trading_cost_bps / 10000.0)
        net_return = gross_return - trading_costs
        
        if verbose:
            print(f"\n  Portfolio Summary:")
            print(f"    Positions: {total_trading_signal:.0f}")
            print(f"    Gross Return: {gross_return:.6f}")
            print(f"    Turnover: {daily_turnover:.4f}")
            print(f"    Costs: {trading_costs:.6f}")
            print(f"    Net Return: {net_return:.6f}")
        
        # Store returns
        r_p_dict['All'].loc[day, 'gross_returns'] = gross_return
        r_p_dict['All'].loc[day, 'net_returns'] = net_return
        
        for split_name in ['Train', 'Validation', 'Test']:
            if day in trading_days_dict[split_name]:
                r_p_dict[split_name].loc[day, 'gross_returns'] = gross_return
                r_p_dict[split_name].loc[day, 'net_returns'] = net_return
    
    # Calculate turnover statistics by split
    turnover_stats = {}
    for split_name in ['All', 'Train', 'Validation', 'Test']:
        split_days = trading_days_dict[split_name]
        split_turnover = turnover_tracking.loc[split_days, 'turnover']
        turnover_stats[split_name] = float(split_turnover.mean())
    
    # Return all results
    return {
        **r_p_dict,
        'trading_signal_evolution': trading_signal_evolution,
        'turnover': turnover_tracking,
        'turnover_stats': turnover_stats
    }


def calculate_trading_intensity_statistics(
    trading_signal_evolution: Dict[str, pd.DataFrame],
    turnover_stats: Dict[str, float]
) -> pd.DataFrame:
    """Calculate comprehensive trading intensity statistics.
    
    Parameters
    ----------
    trading_signal_evolution : Dict[str, pd.DataFrame]
        Dictionary mapping split to DataFrames with 'total_trading_signal' column.
    turnover_stats : Dict[str, float]
        Dictionary mapping split to average turnover.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with trading intensity metrics by split.
        Columns: 'Split', 'Avg_Positions', 'Std_Positions', 'Max_Positions',
                'Min_Positions', 'Turnover_Pct', 'Active_Days_Pct'
    
    Notes
    -----
    Metrics include:
    - Position statistics (mean, std, max, min)
    - Turnover as percentage
    - Percentage of days with active trading
    """
    results = []
    
    for split_name in ['All', 'Train', 'Validation', 'Test']:
        positions = trading_signal_evolution[split_name]['total_trading_signal']
        
        # Calculate position statistics
        avg_positions = float(positions.mean())
        std_positions = float(positions.std())
        max_positions = float(positions.max())
        min_positions = float(positions.min())
        
        # Calculate active trading days
        active_days_pct = float((positions > 0).mean() * 100)
        
        # Get turnover
        turnover_pct = turnover_stats.get(split_name, 0.0) * 100
        
        results.append({
            'Split': split_name,
            'Avg_Positions': avg_positions,
            'Std_Positions': std_positions,
            'Max_Positions': max_positions,
            'Min_Positions': min_positions,
            'Turnover_Pct': turnover_pct,
            'Active_Days_Pct': active_days_pct
        })
    
    return pd.DataFrame(results)


def calculate_portfolio_statistics(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252
) -> Dict[str, float]:
    """Calculate comprehensive portfolio performance statistics.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with 'gross_returns' and 'net_returns' columns.
    risk_free_rate : float, default=0.0
        Risk-free rate for Sharpe ratio calculation (annualized).
    trading_days_per_year : int, default=252
        Number of trading days per year for annualization.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing performance metrics for both gross and net returns:
        - 'cumulative_return_gross/net': Total cumulative return
        - 'annualized_return_gross/net': Annualized return
        - 'volatility_gross/net': Annualized volatility
        - 'sharpe_ratio_gross/net': Annualized Sharpe ratio
        - 'max_drawdown_gross/net': Maximum drawdown
        - 'calmar_ratio_gross/net': Calmar ratio (return / max drawdown)
    """
    stats = {}
    
    for return_type in ['gross_returns', 'net_returns']:
        returns = returns_df[return_type]
        suffix = return_type.replace('_returns', '')
        
        # Cumulative return
        cumulative_return = (1 + returns).prod() - 1
        stats[f'cumulative_return_{suffix}'] = float(cumulative_return)
        
        # Annualized return
        n_days = len(returns)
        if n_days > 0:
            annualized_return = (1 + cumulative_return) ** (trading_days_per_year / n_days) - 1
        else:
            annualized_return = 0.0
        stats[f'annualized_return_{suffix}'] = float(annualized_return)
        
        # Volatility
        volatility = float(returns.std() * np.sqrt(trading_days_per_year))
        stats[f'volatility_{suffix}'] = volatility
        
        # Sharpe ratio
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = np.nan
        stats[f'sharpe_ratio_{suffix}'] = float(sharpe_ratio)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(drawdown.min())
        stats[f'max_drawdown_{suffix}'] = max_drawdown
        
        # Calmar ratio
        if max_drawdown < 0:
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = np.nan
        stats[f'calmar_ratio_{suffix}'] = float(calmar_ratio)
    
    return stats
