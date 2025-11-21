"""Plotting utilities for visualization of clustering and portfolio results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud


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


def plot_time_series_with_ma(
    series: Union[pd.Series, Dict[str, int]],
    ma_window: int = 7,
    title: str = "Time Series",
    xlabel: str = "Date",
    ylabel: str = "Value",
    output_path: Optional[Union[str, Path]] = None,
    save_output: bool = False,
    series_color: str = 'blue',
    ma_color: str = 'red',
    series_label: str = 'Original',
    ma_label: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_plot: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
    title_pad: int = DEFAULT_TITLE_PAD,
    label_pad: int = DEFAULT_LABEL_PAD,
    date_format: str = '%Y-%m',
    rotation: int = 45,
) -> plt.Figure:
    """Plot time series with moving average overlay.
    
    Creates a publication-quality time series plot with a moving average line.
    Supports custom date formatting for the x-axis and matches the styling
    from script 1 (LaTeX, serif font, gray background).
    
    Args:
        series: Time series data as pandas Series (with DatetimeIndex) or
                dictionary mapping dates to values.
        ma_window: Window size for moving average (default 7).
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        output_path: Optional path to save the plot as PDF.
        save_output: Whether to save the plot to file.
        series_color: Color for original series line (default 'blue').
        ma_color: Color for moving average line (default 'red').
        series_label: Label for original series in legend.
        ma_label: Label for moving average in legend. If None, uses
                  f"{ma_window}-day Moving Average".
        figsize: Figure size as (width, height).
        show_plot: Whether to display the plot.
        title_fontsize: Font size for title.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        title_pad: Padding for title.
        label_pad: Padding for axis labels.
        date_format: Format string for date labels (default '%Y-%m').
        rotation: Rotation angle for date labels (default 45).
    
    Returns:
        Matplotlib Figure object.
    
    Raises:
        PlottingError: If series is empty, invalid, or not time-indexed.
    
    Examples:
        >>> # Example 1: Articles per day
        >>> dates = pd.date_range('2020-01-01', periods=100)
        >>> articles_per_day = pd.Series(np.random.randint(1, 20, 100), index=dates)
        >>> fig = plot_time_series_with_ma(
        ...     articles_per_day,
        ...     ma_window=7,
        ...     title="Articles per Day",
        ...     ylabel="Number of Articles",
        ...     save_output=True,
        ...     output_path="articles_per_day.pdf"
        ... )
        
        >>> # Example 2: Custom date formatting
        >>> fig = plot_time_series_with_ma(
        ...     articles_per_day,
        ...     ma_window=30,
        ...     date_format='%b %Y',
        ...     rotation=30
        ... )
    """
    # Convert dictionary to Series if needed
    if isinstance(series, dict):
        series = pd.Series(series)
        # Try to convert index to datetime if possible
        try:
            series.index = pd.to_datetime(series.index)
        except (ValueError, TypeError):
            pass
    
    if not isinstance(series, pd.Series):
        raise PlottingError("Series must be a pandas Series or dictionary")
    
    if len(series) == 0:
        raise PlottingError("Series is empty")
    
    if series.isna().all():
        raise PlottingError("All series values are NaN")
    
    # Check if index is datetime
    if not isinstance(series.index, pd.DatetimeIndex):
        raise PlottingError(
            "Series must have a DatetimeIndex. "
            "Use pd.to_datetime() to convert index to datetime."
        )
    
    # Calculate moving average
    ma_series = series.rolling(window=ma_window, center=False).mean()
    
    # Set default MA label if not provided
    if ma_label is None:
        ma_label = f"{ma_window}-day Moving Average"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color
    ax.set_facecolor('#f5f5f5')
    
    # Plot original series
    ax.plot(
        series.index,
        series.values,
        color=series_color,
        alpha=0.7,
        linewidth=1.5,
        label=series_label
    )
    
    # Plot moving average
    ax.plot(
        ma_series.index,
        ma_series.values,
        color=ma_color,
        linewidth=2.5,
        label=ma_label
    )
    
    # Set labels
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=label_pad)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=label_pad)
    
    # Set title
    if title and (not save_output or show_plot):
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')
    
    # Configure spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(fontsize=label_fontsize - 2, loc='best')
    
    # Add grid
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_output and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from title if path is a directory
        if output_path.is_dir():
            filename = f"{title.replace(' ', '_')}.pdf"
            output_path = output_path / filename
        
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Display if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_histogram_with_density(
    data: Union[pd.Series, np.ndarray, List[float]],
    title: str = "Distribution",
    xlabel: str = "",
    ylabel_left: str = "Frequency",
    ylabel_right: str = "Density",
    xlim: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
    save_output: bool = False,
    bins: int = 30,
    hist_color: str = 'skyblue',
    density_color: str = 'orange',
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = 22,
    tick_fontsize: int = 24,
    title_pad: int = DEFAULT_TITLE_PAD,
    label_pad: int = DEFAULT_LABEL_PAD,
) -> plt.Figure:
    """Plot histogram with overlaid density curve.
    
    Creates a publication-quality histogram with frequency on the left y-axis
    and a kernel density estimate overlaid with density on the right y-axis.
    Matches the styling from script 1 (LaTeX, serif font, gray background).
    
    Args:
        data: Data to plot (Series, array, or list of numbers).
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel_left: Label for left y-axis (frequency).
        ylabel_right: Label for right y-axis (density).
        xlim: Optional tuple of (xmin, xmax) for x-axis limits.
        output_path: Optional path to save the plot as PDF.
        save_output: Whether to save the plot to file.
        bins: Number of bins for histogram (default 30).
        hist_color: Color for histogram bars (default 'skyblue').
        density_color: Color for density curve (default 'orange').
        figsize: Figure size as (width, height).
        show_plot: Whether to display the plot.
        title_fontsize: Font size for title.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        title_pad: Padding for title.
        label_pad: Padding for axis labels.
    
    Returns:
        Matplotlib Figure object.
    
    Raises:
        PlottingError: If data is empty or invalid.
    
    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> fig = plot_histogram_with_density(
        ...     data,
        ...     title="Word Count Distribution",
        ...     xlabel="Number of Words",
        ...     xlim=(0, 1000),
        ...     save_output=True,
        ...     output_path="word_count_dist.pdf"
        ... )
    """
    # Convert data to pandas Series for consistent handling
    if isinstance(data, (list, np.ndarray)):
        data = pd.Series(data)
    
    if len(data) == 0:
        raise PlottingError("Data is empty")
    
    if data.isna().all():
        raise PlottingError("All data values are NaN")
    
    # Remove NaN values
    data_clean = data.dropna()
    
    if len(data_clean) == 0:
        raise PlottingError("No valid data after removing NaN values")
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Set background color
    ax1.set_facecolor('#f5f5f5')
    
    # Plot histogram on primary axis
    counts, bins, patches = ax1.hist(
        data_clean,
        bins=bins,
        color=hist_color,
        edgecolor='black',
        alpha=0.7,
        label='Histogram'
    )
    
    # Create secondary y-axis for density
    ax2 = ax1.twinx()
    
    # Plot density on secondary axis
    data_clean.plot(
        kind='density',
        color=density_color,
        ax=ax2,
        label='Density',
        linewidth=2
    )
    
    # Set labels
    ax1.set_xlabel(xlabel, fontsize=label_fontsize if xlabel else 30, labelpad=label_pad)
    ax1.set_ylabel(ylabel_left, fontsize=label_fontsize, labelpad=label_pad)
    ax2.set_ylabel(ylabel_right, fontsize=label_fontsize, labelpad=label_pad)
    
    # Set title (only if not saving or if explicitly showing title)
    if title and (not save_output or show_plot):
        plt.title(title, fontsize=title_fontsize, pad=title_pad)
    
    # Set x-axis limits if specified
    if xlim is not None:
        ax1.set_xlim(*xlim)
    
    # Set tick label sizes
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    
    # Configure spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    # Create legend combining both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=label_fontsize)
    
    # Add grid
    ax1.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.3)
    
    # Save if requested
    if save_output and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from title if path is a directory
        if output_path.is_dir():
            filename = f"{title.replace(' ', '_')}.pdf"
            output_path = output_path / filename
        
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Display if requested
    if show_plot:
        plt.show()
    
    return fig


def generate_wordcloud(
    text: Union[str, List[str]],
    title: str = "Word Cloud",
    mask: Optional[Union[np.ndarray, str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    save_output: bool = False,
    colormap: Optional[str] = None,
    max_words: int = 150,
    background_color: str = 'white',
    width: int = 800,
    height: int = 400,
    stopwords: Optional[set] = None,
    relative_scaling: float = 0.5,
    min_font_size: int = 10,
    figsize: Optional[Tuple[int, int]] = None,
    show_plot: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    title_pad: int = DEFAULT_TITLE_PAD,
) -> plt.Figure:
    """Generate a word cloud from text data.
    
    Creates a publication-quality word cloud visualization with optional custom
    masks, colormaps, and Spanish stopword filtering. Matches the styling from
    script 1 (LaTeX, serif font).
    
    Args:
        text: Text data as a string or list of strings to generate word cloud from.
        title: Plot title.
        mask: Optional mask for word cloud shape. Can be:
              - NumPy array (e.g., from an image)
              - Path to an image file
              - None for rectangular shape
        output_path: Optional path to save the plot as PDF.
        save_output: Whether to save the plot to file.
        colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'Blues').
                  If None, uses default word cloud colors.
        max_words: Maximum number of words to include (default 150).
        background_color: Background color for word cloud (default 'white').
        width: Width of word cloud in pixels (default 800).
        height: Height of word cloud in pixels (default 400).
        stopwords: Set of stopwords to exclude. If None, uses Spanish stopwords.
        relative_scaling: Importance of word frequency vs rank (0-1, default 0.5).
        min_font_size: Minimum font size for words (default 10).
        figsize: Figure size as (width, height). If None, auto-calculated from mask.
        show_plot: Whether to display the plot.
        title_fontsize: Font size for title.
        title_pad: Padding for title.
    
    Returns:
        Matplotlib Figure object.
    
    Raises:
        PlottingError: If text is empty or invalid, or mask cannot be loaded.
    
    Examples:
        >>> # Example 1: Basic word cloud
        >>> text = "España economía mercado valores acciones bolsa..."
        >>> fig = generate_wordcloud(
        ...     text,
        ...     title="Most Frequent Words",
        ...     max_words=100,
        ...     save_output=True,
        ...     output_path="wordcloud.pdf"
        ... )
        
        >>> # Example 2: With custom mask (e.g., Spain contour)
        >>> fig = generate_wordcloud(
        ...     text,
        ...     title="Word Cloud - Spain Shape",
        ...     mask="spain_mask.png",
        ...     colormap="Blues",
        ...     max_words=150
        ... )
        
        >>> # Example 3: From list of documents
        >>> documents = ["doc1 text...", "doc2 text...", "doc3 text..."]
        >>> fig = generate_wordcloud(
        ...     documents,
        ...     title="Document Corpus Word Cloud"
        ... )
    """
    # Convert list of strings to single string
    if isinstance(text, list):
        text = ' '.join(text)
    
    if not isinstance(text, str):
        raise PlottingError("Text must be a string or list of strings")
    
    if len(text.strip()) == 0:
        raise PlottingError("Text is empty")
    
    # Load mask if provided
    mask_array = None
    if mask is not None:
        if isinstance(mask, (str, Path)):
            try:
                mask_image = Image.open(mask)
                mask_array = np.array(mask_image)
            except Exception as e:
                raise PlottingError(f"Failed to load mask image: {e}")
        elif isinstance(mask, np.ndarray):
            mask_array = mask
        else:
            raise PlottingError("Mask must be a path string, Path object, or numpy array")
    
    # Spanish stopwords if none provided
    if stopwords is None:
        stopwords = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 'haber',
            'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo',
            'pero', 'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro', 'ese',
            'la', 'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
            'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno', 'mismo',
            'yo', 'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'primero',
            'desde', 'grande', 'eso', 'ni', 'nos', 'llegar', 'pasar', 'tiempo', 'ella',
            'sí', 'día', 'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
            'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde', 'ahora', 'parte',
            'después', 'vida', 'quedar', 'siempre', 'creer', 'hablar', 'llevar', 'dejar',
            'nada', 'cada', 'seguir', 'menos', 'nuevo', 'encontrar', 'algo', 'solo',
            'decir', 'aunque', 'aquel', 'esa', 'vez', 'nunca', 'caso', 'tal', 'otro',
            'cómo', 'país', 'sea', 'sido', 'ha', 'han', 'son', 'es', 'las', 'los',
            'del', 'al', 'una', 'unos', 'unas', 'estos', 'estas', 'aquellos', 'aquellas',
            'sus', 'mis', 'tus', 'fueron', 'fue', 'eran', 'era', 'siendo', 'será', 'serán',
            'durante', 'mediante', 'ante', 'bajo', 'contra', 'hacia', 'según', 'tras',
        }
    
    # Create word cloud
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            mask=mask_array,
            stopwords=stopwords,
            relative_scaling=relative_scaling,
            min_font_size=min_font_size,
            colormap=colormap if colormap else None,
            contour_width=0,
            contour_color='steelblue' if mask_array is not None else None,
        ).generate(text)
    except Exception as e:
        raise PlottingError(f"Failed to generate word cloud: {e}")
    
    # Determine figure size
    if figsize is None:
        if mask_array is not None:
            # Scale figure size based on mask dimensions
            aspect_ratio = mask_array.shape[1] / mask_array.shape[0]
            figsize = (12, 12 / aspect_ratio)
        else:
            # Default figure size
            figsize = (12, 6)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display word cloud
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Set title
    if title and (not save_output or show_plot):
        plt.title(title, fontsize=title_fontsize, pad=title_pad)
    
    # Tight layout
    plt.tight_layout(pad=0)
    
    # Save if requested
    if save_output and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from title if path is a directory
        if output_path.is_dir():
            filename = f"{title.replace(' ', '_')}.pdf"
            output_path = output_path / filename
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Word cloud saved to: {output_path}")
    
    # Display if requested
    if show_plot:
        plt.show()
    
    return fig


def plot_silhouette_scores(
    k_range: Union[List[int], range, np.ndarray],
    scores: Union[List[float], np.ndarray],
    optimal_k: Optional[int] = None,
    title: str = "Silhouette Score vs Number of Clusters",
    xlabel: str = "Number of Clusters (k)",
    ylabel: str = "Average Silhouette Score",
    output_path: Optional[Union[str, Path]] = None,
    save_output: bool = False,
    line_color: str = 'blue',
    marker: str = 'o',
    optimal_k_color: str = 'red',
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = True,
    title_fontsize: int = DEFAULT_TITLE_FONTSIZE,
    label_fontsize: int = DEFAULT_LABEL_FONTSIZE,
    tick_fontsize: int = DEFAULT_TICK_FONTSIZE,
    title_pad: int = DEFAULT_TITLE_PAD,
    label_pad: int = DEFAULT_LABEL_PAD,
) -> plt.Figure:
    """Plot silhouette scores versus number of clusters.
    
    Creates a publication-quality line plot showing how silhouette scores
    vary with the number of clusters, with optional marking of the optimal k.
    Matches the styling from script 4 (LaTeX, serif font, gray background).
    
    Args:
        k_range: Range or list of k values (number of clusters).
        scores: Corresponding silhouette scores for each k value.
        optimal_k: Optional k value to mark as optimal with vertical line.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        output_path: Optional path to save the plot as PDF.
        save_output: Whether to save the plot to file.
        line_color: Color for the score line (default 'blue').
        marker: Marker style for data points (default 'o').
        optimal_k_color: Color for optimal k marker line (default 'red').
        figsize: Figure size as (width, height).
        show_plot: Whether to display the plot.
        title_fontsize: Font size for title.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        title_pad: Padding for title.
        label_pad: Padding for axis labels.
    
    Returns:
        Matplotlib Figure object.
    
    Raises:
        PlottingError: If k_range and scores have different lengths or are empty.
    
    Examples:
        >>> # Example 1: Basic silhouette score plot
        >>> k_range = range(2, 11)
        >>> scores = [0.45, 0.52, 0.58, 0.55, 0.51, 0.48, 0.44, 0.42, 0.40]
        >>> fig = plot_silhouette_scores(
        ...     k_range,
        ...     scores,
        ...     optimal_k=4,
        ...     save_output=True,
        ...     output_path="silhouette_scores.pdf"
        ... )
        
        >>> # Example 2: Without optimal k marking
        >>> fig = plot_silhouette_scores(
        ...     k_range,
        ...     scores,
        ...     title="K-Means Clustering Quality"
        ... )
    """
    # Convert to arrays for consistent handling
    k_array = np.array(list(k_range))
    scores_array = np.array(scores)
    
    # Validate inputs
    if len(k_array) == 0:
        raise PlottingError("k_range is empty")
    
    if len(scores_array) == 0:
        raise PlottingError("scores array is empty")
    
    if len(k_array) != len(scores_array):
        raise PlottingError(
            f"k_range and scores must have the same length. "
            f"Got k_range={len(k_array)}, scores={len(scores_array)}"
        )
    
    # Check for NaN values
    if np.any(np.isnan(scores_array)):
        raise PlottingError("scores contains NaN values")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set background color
    ax.set_facecolor('#f5f5f5')
    
    # Plot silhouette scores
    ax.plot(
        k_array,
        scores_array,
        color=line_color,
        marker=marker,
        linewidth=2,
        markersize=8,
        label='Silhouette Score'
    )
    
    # Mark optimal k with vertical line
    if optimal_k is not None:
        if optimal_k in k_array:
            ax.axvline(
                x=optimal_k,
                color=optimal_k_color,
                linestyle='--',
                linewidth=2,
                label=f'Optimal k = {optimal_k}'
            )
        else:
            print(f"Warning: optimal_k={optimal_k} not in k_range")
    
    # Set labels
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=label_pad)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=label_pad)
    
    # Set title
    if title and (not save_output or show_plot):
        ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Configure spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(fontsize=label_fontsize - 2, loc='best')
    
    # Add grid
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    
    # Set x-axis to show integer ticks only
    ax.set_xticks(k_array)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_output and output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from title if path is a directory
        if output_path.is_dir():
            filename = f"{title.replace(' ', '_')}.pdf"
            output_path = output_path / filename
        
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Display if requested
    if show_plot:
        plt.show()
    
    return fig
