"""Plotting utilities for visualization of clustering and portfolio results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Configure matplotlib for LaTeX-style output
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Default font sizes
DEFAULT_TITLE_FONTSIZE = 16
DEFAULT_LABEL_FONTSIZE = 14
DEFAULT_TICK_FONTSIZE = 12
DEFAULT_TITLE_PAD = 20
DEFAULT_LABEL_PAD = 10


class PlottingError(Exception):
    """Raised when plotting operations fail."""
    pass


def plot_cluster_distribution(
    df: pd.DataFrame,
    cluster_column: str = 'cluster',
    split_column: Optional[str] = None,
    split_value: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_title: bool = True,
    plot_density: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
    title_pad: int = DEFAULT_TITLE_PAD,
    label_pad: int = DEFAULT_LABEL_PAD,
    figsize: Tuple[int, int] = (12, 6),
    show_plot: bool = True
) -> plt.Figure:
    """Plot distribution of articles per cluster.

    Args:
        df: DataFrame containing cluster assignments.
        cluster_column: Name of the column containing cluster labels.
        split_column: Optional column name to filter by data split.
        split_value: Value of split to filter (e.g., 'Train', 'Test').
        output_path: Optional path to save the plot.
        show_title: Whether to show plot title.
        plot_density: Whether to overlay kernel density estimate.
        title_fontsize: Font size for title.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        title_pad: Padding for title.
        label_pad: Padding for axis labels.
        figsize: Figure size as (width, height).
        show_plot: Whether to display the plot.

    Returns:
        Matplotlib Figure object.

    Raises:
        PlottingError: If required columns are missing or data is invalid.
    """
    # Validate inputs
    if cluster_column not in df.columns:
        raise PlottingError(f"Cluster column '{cluster_column}' not found in DataFrame")
    
    if split_column and split_column not in df.columns:
        raise PlottingError(f"Split column '{split_column}' not found in DataFrame")
    
    # Filter data if split is specified
    if split_column and split_value:
        plot_df = df[df[split_column] == split_value].copy()
        title_suffix = f"Split: {split_value}"
    else:
        plot_df = df.copy()
        title_suffix = "All data"
    
    if len(plot_df) == 0:
        raise PlottingError("No data to plot after filtering")
    
    # Calculate cluster counts
    cluster_counts = plot_df[cluster_column].value_counts().sort_index()
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Bar plot
    cluster_counts.plot(
        kind='bar',
        ax=ax1,
        alpha=0.6,
        color='blue',
        edgecolor='black'
    )
    
    # Configure primary axis
    ax1.set_xlabel('Cluster', fontsize=label_fontsize, labelpad=label_pad)
    ax1.set_ylabel('Number of Articles', fontsize=label_fontsize, labelpad=label_pad)
    
    if show_title:
        ax1.set_title(
            f'Distribution of Articles per Cluster $~\\mid~$ {title_suffix}',
            fontsize=title_fontsize,
            pad=title_pad
        )
    
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_facecolor('#f5f5f5')
    ax1.spines['top'].set_visible(False)
    
    # Add density plot if requested
    if plot_density:
        ax2 = ax1.twinx()
        sns.kdeplot(plot_df[cluster_column], ax=ax2, color='red', linewidth=2)
        ax2.set_ylabel('Density', fontsize=label_fontsize, labelpad=label_pad)
        ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax2.spines['top'].set_visible(False)
    else:
        ax1.spines['right'].set_visible(False)
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_cluster_distributions_by_split(
    df: pd.DataFrame,
    cluster_column: str = 'cluster',
    split_column: str = 'split',
    output_dir: Optional[Union[str, Path]] = None,
    filename_prefix: str = 'Cluster_Distribution',
    **kwargs
) -> Dict[str, plt.Figure]:
    """Plot cluster distributions for each data split.

    Args:
        df: DataFrame containing cluster assignments and splits.
        cluster_column: Name of the column containing cluster labels.
        split_column: Name of the column containing split labels.
        output_dir: Optional directory to save plots.
        filename_prefix: Prefix for output filenames.
        **kwargs: Additional arguments passed to plot_cluster_distribution.

    Returns:
        Dictionary mapping split names to Figure objects.

    Raises:
        PlottingError: If required columns are missing.
    """
    if split_column not in df.columns:
        raise PlottingError(f"Split column '{split_column}' not found in DataFrame")
    
    figures = {}
    splits = df[split_column].unique()
    
    for split in splits:
        # Construct output path if directory provided
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{filename_prefix}_{split}.pdf"
        
        # Create plot for this split
        fig = plot_cluster_distribution(
            df=df,
            cluster_column=cluster_column,
            split_column=split_column,
            split_value=split,
            output_path=output_path,
            **kwargs
        )
        
        figures[split] = fig
    
    return figures


def plot_average_cars_by_cluster(
    car_data: Dict[Tuple[str, int], np.ndarray],
    split: str,
    max_points: int = 100,
    output_path: Optional[Union[str, Path]] = None,
    show_title: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
    title_pad: int = DEFAULT_TITLE_PAD,
    label_pad: int = DEFAULT_LABEL_PAD,
    figsize: Tuple[int, int] = (12, 6),
    show_plot: bool = True
) -> plt.Figure:
    """Plot time series of average CARs for each cluster.

    Args:
        car_data: Dictionary mapping (split, cluster) to CAR arrays.
        split: Data split to plot ('Train', 'Validation', 'Test').
        max_points: Maximum number of points to plot.
        output_path: Optional path to save the plot.
        show_title: Whether to show plot title.
        title_fontsize: Font size for title.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        title_pad: Padding for title.
        label_pad: Padding for axis labels.
        figsize: Figure size as (width, height).
        show_plot: Whether to display the plot.

    Returns:
        Matplotlib Figure object.

    Raises:
        PlottingError: If no data found for the specified split.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Track if we plotted anything
    plotted = False
    
    # Plot CAR for each cluster in the specified split
    for (data_split, cluster), avg_car in car_data.items():
        if data_split == split and len(avg_car) > 0:
            # Limit points to plot
            avg_car_subset = avg_car[:max_points]
            ax.plot(avg_car_subset, label=f'Cluster {cluster}')
            plotted = True
    
    if not plotted:
        raise PlottingError(f"No CAR data found for split '{split}'")
    
    # Configure plot
    if show_title:
        ax.set_title(
            f'Average CARs for each cluster ({split})',
            fontsize=title_fontsize,
            pad=title_pad
        )
    
    ax.set_xlabel('Trading Days', fontsize=label_fontsize, labelpad=label_pad)
    ax.set_ylabel('Average CAR', fontsize=label_fontsize, labelpad=label_pad)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=7)
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_cumulative_returns(
    returns_dict: Dict[str, pd.Series],
    output_path: Optional[Union[str, Path]] = None,
    title: str = 'Cumulative Portfolio Returns',
    show_title: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
    title_pad: int = DEFAULT_TITLE_PAD,
    label_pad: int = DEFAULT_LABEL_PAD,
    figsize: Tuple[int, int] = (12, 6),
    show_plot: bool = True
) -> plt.Figure:
    """Plot cumulative returns for different portfolios or splits.

    Args:
        returns_dict: Dictionary mapping labels to return series.
        output_path: Optional path to save the plot.
        title: Plot title.
        show_title: Whether to show plot title.
        title_fontsize: Font size for title.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        title_pad: Padding for title.
        label_pad: Padding for axis labels.
        figsize: Figure size as (width, height).
        show_plot: Whether to display the plot.

    Returns:
        Matplotlib Figure object.

    Raises:
        PlottingError: If returns_dict is empty or invalid.
    """
    if not returns_dict:
        raise PlottingError("No returns data provided")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative returns for each series
    for label, returns in returns_dict.items():
        if isinstance(returns, pd.Series):
            cum_returns = (1 + returns).cumprod()
            ax.plot(cum_returns, label=label, linewidth=2)
        else:
            raise PlottingError(f"Returns for '{label}' must be a pandas Series")
    
    # Configure plot
    if show_title:
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    
    ax.set_xlabel('Date', fontsize=label_fontsize, labelpad=label_pad)
    ax.set_ylabel('Cumulative Return', fontsize=label_fontsize, labelpad=label_pad)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=tick_fontsize)
    
    # Format x-axis dates if index is datetime
    first_series = next(iter(returns_dict.values()))
    if isinstance(first_series.index, pd.DatetimeIndex):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig


def configure_matplotlib_style(
    use_latex: bool = True,
    font_family: str = 'serif',
    latex_preamble: str = r'\usepackage{amsmath}'
) -> None:
    """Configure matplotlib style for publication-quality plots.

    Args:
        use_latex: Whether to use LaTeX for text rendering.
        font_family: Font family to use.
        latex_preamble: LaTeX preamble for additional packages.
    """
    plt.rc('text', usetex=use_latex)
    plt.rc('font', family=font_family)
    if use_latex:
        plt.rc('text.latex', preamble=latex_preamble)


def reset_matplotlib_style() -> None:
    """Reset matplotlib to default style."""
    plt.rcdefaults()
