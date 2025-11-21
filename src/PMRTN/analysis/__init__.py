"""Analysis utilities for trading calendar, backtesting, and statistics."""

from .backtesting import (
    BacktestingError,
    calculate_average_metrics_by_group,
    calculate_trading_strategy_data,
    process_article_ticker_pair,
)
from .cluster_selection import (
    ClusterSelectionError,
    assign_trading_rules,
    calculate_cluster_sharpe_ratios,
    calculate_spearman_correlation,
    rank_clusters_by_sharpe,
    select_clusters_greedy,
    select_clusters_rank_stable,
    separate_clusters_by_sr_sign,
)
from .portfolio import (
    PortfolioError,
    calculate_portfolio_returns,
    calculate_portfolio_statistics,
    calculate_trading_intensity_statistics,
    initialize_portfolio,
)
from .statistics import (
    get_e_data,
    split_data,
)
from .trading_calendar import TradingCalendarAdjustments

__all__ = [
    # Backtesting
    "BacktestingError",
    "calculate_average_metrics_by_group",
    "calculate_trading_strategy_data",
    "process_article_ticker_pair",
    # Cluster Selection
    "ClusterSelectionError",
    "assign_trading_rules",
    "calculate_cluster_sharpe_ratios",
    "calculate_spearman_correlation",
    "rank_clusters_by_sharpe",
    "select_clusters_greedy",
    "select_clusters_rank_stable",
    "separate_clusters_by_sr_sign",
    # Portfolio
    "PortfolioError",
    "calculate_portfolio_returns",
    "calculate_portfolio_statistics",
    "calculate_trading_intensity_statistics",
    "initialize_portfolio",
    # Statistics & Trading Calendar
    "TradingCalendarAdjustments",
    "split_data",
    "get_e_data",
]
