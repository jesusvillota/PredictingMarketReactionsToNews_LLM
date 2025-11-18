import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from src import get_logger
from pathlib import Path
from src.config import config_settings

# Initialize matplotlib settings on import
from src.plotting.config import setup_matplotlib_config
setup_matplotlib_config()

# Settings
save_plot: bool = config_settings.plotting['save_plot']
output_path: Path = config_settings.plotting['output_path']
output_path.mkdir(parents=True, exist_ok=True)
dpi: int = config_settings.plotting.get('dpi', 600)
show_title: bool = config_settings.plotting.get('show_title', True)

# Fonts
# label_size: int = config_settings.plotting["fonts"]['label_size']
# tick_size: int = config_settings.plotting["fonts"]['tick_size']
# legend_size: int = config_settings.plotting["fonts"]['legend_size']
# # Padding
# title_pad = config_settings.plotting['padding']['title_pad']
# label_pad = config_settings.plotting['padding']['label_pad']


def time_histogram(pdf: pd.DataFrame, 
                   start_time: str,
                   end_time: str,
                   freq_min: int = 15,
                   ) -> None:
    
    from datetime import datetime, timedelta
    logger = get_logger(__name__)
    logger.debug(f"Creating time histogram from {start_time} to {end_time}")

    pdf = pdf.copy()
    start_dt = datetime.strptime(start_time, "%H:%M")
    end_dt = datetime.strptime(end_time, "%H:%M")
    start_sec = start_dt.hour * 3600 + start_dt.minute * 60
    end_sec = end_dt.hour * 3600 + end_dt.minute * 60

    pdf["time"] = pd.to_datetime(pdf["timestamp_ny"]).dt.time
    pdf['time_seconds'] = pdf['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)
    mask = (pdf['time_seconds'] >= start_sec) & (pdf['time_seconds'] <= end_sec)
    subset = pdf.loc[mask, 'time_seconds']

    logger.debug(f"Plotting histogram with {len(subset)} data points")

    # Bin edges for freq_min intervals
    bin_edges = np.arange(start_sec, end_sec + freq_min * 60, freq_min * 60)
    counts, _ = np.histogram(subset, bins=bin_edges)

    # Bin centers for plotting
    bin_centers = bin_edges[:-1] + (freq_min * 60) / 2

    plt.figure()
    plt.bar(bin_centers, counts, width=freq_min * 60 * 0.8, color='skyblue', edgecolor='black')
    plt.ylabel('Frequency')
    if show_title:
        plt.title(f'Histogram of trades ({start_time} to {end_time}): number of trades per {freq_min}-min interval')

    # Set xticks every freq_min minutes
    xtick_secs = bin_edges
    xtick_labels = [f"{int(s//3600):02d}:{int((s%3600)//60):02d}" for s in xtick_secs]
    plt.xticks(ticks=xtick_secs, labels=xtick_labels, rotation=45)

    plt.tight_layout()

    if save_plot:
        filename = f"hist_[{start_time.replace(':', '')}_{end_time.replace(':', '')}]_[freq_{freq_min}min].png"
        plt.savefig(output_path / filename, dpi=dpi, bbox_inches='tight')
        logger.debug(f"💾 Plot saved to: {output_path / filename}")

def analyze_trade_sizes(pdf, 
                        start_hour=9,
                        end_hour=19,
                        interval=15,
                        ):
    
    logger = get_logger(__name__)
    logger.debug(f"Analyzing trade sizes from hour {start_hour} to {end_hour} with {interval}-min intervals")
    pdf["hour"] = pd.to_datetime(pdf["timestamp_ny"]).dt.hour
    pdf = pdf[(pdf['hour'] >= start_hour) & (pdf['hour'] < end_hour)].copy()
    pdf['minute'] = pd.to_datetime(pdf['timestamp_ny']).dt.minute
    pdf['time_interval'] = (pdf['hour'] * 60 + pdf['minute']) // interval * interval

    # Convert back to hours and minutes for labeling
    pdf['time_interval_label'] = pdf['time_interval'].apply(
        lambda x: f"{int(x//60):02d}:{int(x%60):02d}"
    )

    groups=np.arange(0, 1.1, 0.1)
    # Calculate stats for intervals
    interval_stats = pdf.groupby(f'time_interval')['prtSize_agg'].agg(['mean', 'median', 'std', 'count']).reset_index()

    # Calculate percentiles for intervals
    interval_quartiles = pdf.groupby(f'time_interval')['prtSize_agg'].quantile(groups).unstack().reset_index()
    group_columns = [groups[i] for i in range(len(groups))]
    interval_quartiles.columns = [f'time_interval', *group_columns]

    # Create time labels for plotting
    interval_quartiles['time_label'] = interval_quartiles[f'time_interval'].apply(
        lambda x: f"{int(x//60):02d}:{int(x%60):02d}"
    )
    interval_stats['time_label'] = interval_stats[f'time_interval'].apply(
        lambda x: f"{int(x//60):02d}:{int(x%60):02d}"
    )

    logger.debug(f"Creating plot for {len(interval_stats)} time intervals")
    # Create figure with proper spacing - make upper plot taller
    fig, (ax1, ax2) = plt.subplots(figsize=(18, 12), nrows=2, ncols=1, height_ratios=[1, 1])

    # Plot groups for intervals
    lines = []
    for i in range(len(groups)):
        line = ax1.plot(interval_quartiles['time_interval'], interval_quartiles[groups[i]], 
                       marker='o', linewidth=2, 
                       color=plt.cm.get_cmap('viridis')(i/len(groups)))[0]
        lines.append(line)

    ax1.set_xlabel(f'Time ({interval}-min intervals, NY Time)')
    ax1.set_ylabel('Trade Size')
    if show_title:
        ax1.set_title(f'Trade Size Percentiles by {interval}-Minute Intervals')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Set x-ticks for the filtered time range only
    hour_ticks = [i*60 for i in range(start_hour, end_hour)]  # Only filtered hours
    hour_labels = [f"{i:02d}:00" for i in range(start_hour, end_hour)]
    ax1.set_xticks(ticks=hour_ticks)
    ax1.set_xticklabels(labels=hour_labels, rotation=45)

    # Plot trade count for intervals
    ax2.bar(interval_stats['time_interval'], interval_stats['count'], alpha=0.7, color='green', width=10)
    ax2.set_xlabel(f'Time ({interval}-min intervals, NY Time)')
    ax2.set_ylabel('Number of Trades')
    if show_title:
        ax2.set_title(f'Trade Count by {interval}-Minute Intervals')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(ticks=hour_ticks)
    ax2.set_xticklabels(labels=hour_labels, rotation=45)

    # Create colorbar for the entire figure (shared)
    norm = mcolors.Normalize(vmin=groups.min()*100, vmax=groups.max()*100)
    sm = cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))
    sm.set_array([])

    plt.subplots_adjust(right=0.85, hspace=0.4)

    cbar_ax = fig.add_axes((0.87, 0.15, 0.02, 0.7))  # (left, bottom, width, height)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'Percentiles (\%)')

    tick_values = [groups.min()*100, groups.max()*100]
    if (groups.max() - groups.min()) > 0.2:
        mid_values = [25, 50, 75]
        tick_values = sorted(set(tick_values + [v for v in mid_values if groups.min()*100 <= v <= groups.max()*100]))
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{v:.1f}' for v in tick_values])

    # Save plot if configured
    if save_plot:
        filename = f"trade_size_percentiles_[{start_hour:02d}00_{end_hour:02d}00]_[interval_{interval}min].png"
        plt.savefig(output_path / filename, dpi=dpi, bbox_inches='tight')
        logger.debug(f"💾 Plot saved to: {output_path / filename}")

    # plt.show()

    # Display some statistics about the intervals
    logger.debug(f"Number of {interval}-minute intervals with data: {len(interval_stats)}")
    logger.debug(f"Average trades per {interval}-min interval: {interval_stats['count'].mean():.1f}")
    logger.debug(f"Peak trading {interval}-min interval: {interval_stats.loc[interval_stats['count'].idxmax(), 'time_label']} with {interval_stats['count'].max()} trades")
    

def plot_trade_size_percentile_heatmap(pdf, interval=15, start_hour=9, end_hour=20, 
                                       quantile_bin_pct=1, figsize=None, title=None,
                                       normalize='by_slot'):
    
    logger = get_logger(__name__)
    logger.debug(f"Creating heatmap for trade size percentiles with {interval}-min intervals")
    # interval=15
    # start_hour=9
    # end_hour=20
    # quantile_bin_pct=1
    # figsize=None
    # title=None
    # normalize='by_slot'

    # Compute GLOBAL percentile ranks once on full dataset (0..100)
    if 'size_pct_global' not in pdf.columns:
        pdf = pdf.copy()
        pdf['size_pct_global'] = pdf['prtSize_agg'].rank(pct=True) * 100

    # Filter hours and build time slots
    pdf["hour"] = pd.to_datetime(pdf["timestamp_ny"]).dt.hour
    pdf = pdf[(pdf['hour'] >= start_hour) & (pdf['hour'] < end_hour)].copy()
    pdf['minute'] = pd.to_datetime(pdf['timestamp_ny']).dt.minute
    pdf['time_slot'] = (pdf['hour'] * 60 + pdf['minute']) // interval * interval

    # Bin into percentile bands (0..100], include upper bound
    # Create bins that will result in labels from 0 to 100 (inclusive)
    bins = np.arange(0, 102, quantile_bin_pct)  # 0, 1, 2, ..., 100
    # Use bin edges as labels to show 0, 1, 2, ..., 99 (last bin covers 99-100)
    labels = bins[:-1]  # Use left edge of each bin as label: 0, 1, 2, ..., 99
    pdf['size_pct_bin'] = pd.cut(pdf['size_pct_global'], bins=bins.tolist(), labels=labels.tolist(), include_lowest=True, right=True)

    # Aggregate: for each time_slot x percentile bin, compute share within the time slot
    ct = (pdf.groupby(['time_slot', 'size_pct_bin'], observed=False)
                .size()
                .reset_index(name='count'))

    # Choose normalization method
    if normalize == 'by_slot':
        # Original behavior: Normalize per time_slot to get percentages within each slot
        ct['slot_total'] = ct.groupby('time_slot')['count'].transform('sum')
        ct['pct_in_slot'] = 100 * ct['count'] / ct['slot_total']
        value_col = 'pct_in_slot'
        cbar_label = 'Percent of trades in slot'
    elif normalize == 'global':
        # New behavior: Global percentage across all time slots for each percentile
        total_trades = ct['count'].sum()
        ct['global_pct'] = 100 * ct['count'] / total_trades
        value_col = 'global_pct'
        cbar_label = 'Global percent of all trades'
    else:
        raise ValueError("normalize must be either 'by_slot' or 'global'")

    # Pivot to heatmap format
    heat = ct.pivot(index='size_pct_bin', columns='time_slot', values=value_col)

    # Build ordered axes
    time_ticks = list(range(start_hour*60, end_hour*60, 60))
    time_labels = [f"{h:02d}:00" for h in range(start_hour, end_hour)]

    # Set figure size
    if figsize is None:
        figsize = (max(10, (end_hour-start_hour)*1.2), 10)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)


    # FIXED: Sort index in descending order to have 0 at bottom, 100 at top
    heatmap = sns.heatmap(heat.sort_index(ascending=False), cmap='YlGnBu', 
                            cbar_kws={'label': cbar_label},
                            robust=True, ax=ax)

    # Set proper y-axis ticks to show 0 to 100 with granular spacing
    # Use every 5 percentiles for more detailed display
    step = 5  # Show every <step>th percentile for more granular view

    # Calculate tick positions (every <step> rows in the heatmap)
    y_tick_positions = np.arange(0, len(heat.index), step)

    # Calculate corresponding percentile labels (reversed because heatmap is sorted descending)
    y_tick_labels = [str(int(100 - i * quantile_bin_pct)) for i in y_tick_positions]

    # Always include the 0th percentile at the bottom
    # if len(heat.index) - 1 not in y_tick_positions:
    #     y_tick_positions = np.append(y_tick_positions, len(heat.index) - 1)
    #     y_tick_labels.append('0')

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)  # Smaller font for more labels

    # Add right y-axis with actual prtSize_agg values for the percentiles
    ax2 = ax.twinx()

    # Calculate actual trade size values for the percentiles shown on left axis
    percentile_values = [float(label) for label in y_tick_labels]
    # percentile_values = [float(label) for label in y_tick_labels if label != '0']
    # if '0' in y_tick_labels:
    #     percentile_values.append(0.0)

    # Get actual prtSize_agg values for these percentiles
    actual_sizes = []
    for pct in percentile_values:
        if pct == 0:
            actual_sizes.append(pdf['prtSize_agg'].min())
        elif pct == 100:
            actual_sizes.append(pdf['prtSize_agg'].max())
        else:
            actual_sizes.append(pdf['prtSize_agg'].quantile(pct/100))

    # Set right y-axis ticks and labels
    ax2.set_ylim(ax.get_ylim())  # Match the main axis limits
    ax2.set_yticks(y_tick_positions)

    # Format the actual size values for display
    size_labels = []
    for size in actual_sizes:
        if size >= 1000000:
            size_labels.append(f'{size/1000000:.1f}M')
        elif size >= 1000:
            size_labels.append(f'{size/1000:.1f}K')
        else:
            size_labels.append(f'{size:.0f}')

    ax2.set_yticklabels(size_labels)
    ax2.set_ylabel('Trade size (prtSize_agg)')
    # Position the right y-axis label closer to the ticks
    ax2.yaxis.set_label_coords(1.05, 0.5)

    # Fix: Align the bottom of the heatmap with the 0 percentile line
    # ax.set_ylim(len(heat.index)-0.5, -0.0)
    # ax2.set_ylim(len(heat.index)-0.5, -0.0)  # Keep both axes aligned

    # X ticks at each hour
    ax.set_xlabel(f'Time ({interval}-min slots, NY)')
    ax.set_ylabel(r'Trade size percentile (global, \%)')

    if title is None:
        if normalize == 'by_slot':
            title = 'Empirical Distribution of Trade Size Percentiles by Time Slot'
        else:  # normalize == 'global'
            title = 'Global Distribution of Trade Size Percentiles Across All Time Slots'
    if show_title:
        ax.set_title(title)
    # Read heatmap labels and ticks
    # Convert time_slot (in minutes) to hour labels on x-axis
    col_slots = heat.columns.values.astype(float)

    # Create proper x-axis labeling
    # Get all unique time slots and sort them
    sorted_slots = sorted(col_slots)

    # Find positions for hour markers
    hour_positions = []
    hour_labels_to_show = []

    if len(col_slots):
        for h in range(start_hour, end_hour):
            target_time = h * 60  # Convert hour to minutes
            # Find the index of the closest time slot
            if target_time in sorted_slots:
                idx = list(sorted_slots).index(target_time)
                hour_positions.append(idx + 0.5)  # Center the tick
                hour_labels_to_show.append(f"{h:02d}:00")

    if hour_positions:
        ax.set_xticks(hour_positions)
        ax.set_xticklabels(hour_labels_to_show, rotation=45, ha='right')

    # Optional overlay: mean/median percentile per slot
    slot_stats = pdf.groupby('time_slot')['size_pct_global'].agg(['mean', 'median']).reset_index()
    # Map slot to x-position using the sorted slots
    sorted_slots = sorted(heat.columns.values)
    slot_to_pos = {slot: i + 1 for i, slot in enumerate(sorted_slots)}
    slot_stats['x'] = slot_stats['time_slot'].map(slot_to_pos)

    # Plot lines - need to adjust y positions for reversed axis
    if slot_stats['x'].notna().any():
        # Since we reversed the y-axis, we need to convert percentiles to y-positions
        # The heatmap has percentiles from high to low (top to bottom)
        y_positions_mean = len(heat.index) - (slot_stats['mean'].values / 100 * len(heat.index))
        y_positions_median = len(heat.index) - (slot_stats['median'].values / 100 * len(heat.index))
        ax.plot(slot_stats['x'].values, y_positions_mean, color='black', linewidth=2, label='Mean %tile')
        ax.plot(slot_stats['x'].values, y_positions_median, color='red', linewidth=2, linestyle='--', label='Median %tile')
    # Place legend closer to xlabel for better visual grouping
    ax.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=2)

    # No need for plt.tight_layout() or plt.subplots_adjust() with constrained_layout=True

    # Save heatmap if configured
    if save_plot:
        filename = f"trade_size_percentile_heatmap_[{start_hour:02d}00_{end_hour:02d}00]_[interval_{interval}min]_[normalize_{normalize}].png"
        fig.savefig(output_path / filename, dpi=dpi, bbox_inches='tight')
        logger.debug(f"💾 Plot saved to: {output_path / filename}")


def scatter_with_trend(
    pdf, x, y, xlabel=None, ylabel=None, title=None, show_trend=True, 
    color = None,
    degree = 1,
    cmap = "viridis",
    ):
    
    logger = get_logger(__name__)
    logger.debug(f"Creating scatter plot: {y} vs {x}")
    

    sns.set_style("whitegrid")
    plt.figure()

    scatter = plt.scatter(
        pdf[x], pdf[y],
        alpha=0.6,
        c=pdf[color] if color else None,
        cmap=cmap if color else None,
        s=50,
        edgecolor='k',
        linewidth=0.5
    )

    if color:
        cbar = plt.colorbar(scatter)
        cbar.set_label(color, fontsize=12)

    if show_trend:
        z = np.polyfit(pdf[x], pdf[y], degree)
        p = np.poly1d(z)
        x_sorted = np.sort(pdf[x].unique())
        y_smooth = p(x_sorted)
        plt.plot(
            x_sorted, y_smooth,
            linestyle='--',
            color='red',
            linewidth=2,
            label=f'Polynomial fit (degree {degree})'
        )

    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    # Add a red dashed vertical line at x=1
    plt.axvline(x=1, color='red', linestyle='--', linewidth=1.5, label='x=1')
    if show_title:
        plt.title(title if title else f'{y} vs {x}')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()
    if save_plot:
        filename = f"scatter_{y}_vs_{x}.png"
        plt.savefig(output_path / filename, dpi=dpi, bbox_inches='tight')
        logger.debug(f"💾 Plot saved to: {output_path / filename}")

def scatter_with_trend_matplotlib(
    pdf, x, y, xlabel=None, ylabel=None, title=None, show_trend=True,
    color=None, degree=1, cmap="viridis",
):
    """
    Pure Matplotlib version of scatter_with_trend, using module-level config settings.
    """
    logger = get_logger(__name__)
    logger.debug(f"Creating matplotlib scatter plot: {y} vs {x}")

    # Create figure and axis with configured size and style
    fig, ax = plt.subplots()

    # Scatter plot
    if color:
        values = pdf[color]
        norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
        cmap_obj = plt.cm.get_cmap(cmap)
        sc = ax.scatter(
            pdf[x], pdf[y], c=values, cmap=cmap_obj, norm=norm,
            alpha=0.6, s=50, edgecolors='k', linewidth=0.5
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color, fontsize=12)
    else:
        ax.scatter(
            pdf[x], pdf[y], alpha=0.6, s=50,
            edgecolors='k', linewidth=0.5
        )

    # Trend line
    if show_trend:
        z = np.polyfit(pdf[x], pdf[y], degree)
        p = np.poly1d(z)
        x_sorted = np.sort(pdf[x].unique())
        y_smooth = p(x_sorted)
        ax.plot(
            x_sorted, y_smooth,
            linestyle='--', color='red', linewidth=2,
            label=f'Polynomial fit (degree {degree})'
        )

    # Vertical reference line at x=1
    ax.axvline(1, color='red', linestyle='--', linewidth=1.5, label='x=1')

    # Labels and title
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if show_title:
        ax.set_title(title or f'{y} vs {x}')

    # Grid, legend, and layout
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    fig.tight_layout()

    # Save plot if enabled
    if save_plot:
        filename = f"scatter_{y}_vs_{x}_mpl.png"
        fig.savefig(output_path / filename, dpi=dpi, bbox_inches='tight')
        logger.debug(f"💾 Plot saved to: {output_path / filename}")
