"""Visualization utilities for news market analysis.

This module provides plotting and table generation functions
for visualizing clustering results, portfolio performance, and analysis outputs.
"""

from .plotting import (
    PlottingError,
    configure_matplotlib_style,
    plot_average_cars_by_cluster,
    plot_cluster_distribution,
    plot_cluster_distributions_by_split,
    plot_cumulative_returns,
    reset_matplotlib_style,
)
from .tables import (
    TableGenerationError,
    generate_cluster_mapping_table,
    generate_llama_shock_mapping_table,
    generate_portfolio_statistics_table,
    generate_trading_intensity_table,
)

__all__ = [
    # Plotting
    'PlottingError',
    'plot_cluster_distribution',
    'plot_cluster_distributions_by_split',
    'plot_average_cars_by_cluster',
    'plot_cumulative_returns',
    'configure_matplotlib_style',
    'reset_matplotlib_style',
    # Tables
    'TableGenerationError',
    'generate_cluster_mapping_table',
    'generate_portfolio_statistics_table',
    'generate_trading_intensity_table',
    'generate_llama_shock_mapping_table',
]

__version__ = "0.1.0"
