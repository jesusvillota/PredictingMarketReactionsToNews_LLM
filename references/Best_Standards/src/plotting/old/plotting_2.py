import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import calendar
from pathlib import Path

from src.config import config_settings
save_plot: bool = config_settings.plotting["save_plot"]
output_path: Path = Path(config_settings.plotting["output_path"])
output_path.mkdir(parents=True, exist_ok=True)

############################################################################################################################################

def visualize_file_dates(dates):
    """Create visualizations of file dates: monthly and yearly distributions"""
    
    if not dates:
        print("No valid dates found!")
        return
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Create figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # 1. Monthly distribution
    monthly_counts = df['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bars = axes[0].bar(range(1, 13), [monthly_counts.get(i, 0) for i in range(1, 13)], 
                      color='skyblue', alpha=0.8)
    axes[0].set_title('Distribution by Month', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Number of Files')
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names)
    axes[0].grid(True, alpha=0.3)
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
    
    # 2. Yearly distribution
    yearly_counts = df['year'].value_counts().sort_index()
    bars = axes[1].bar(yearly_counts.index, yearly_counts.values, 
                      color='lightcoral', alpha=0.8)
    axes[1].set_title('Distribution by Year', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Number of Files')
    axes[1].set_xticks(yearly_counts.index)
    axes[1].set_xticklabels(yearly_counts.index, rotation=45)
    axes[1].grid(True, alpha=0.3)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_plot:
        fig.savefig(output_path / 'file_dates_distribution.png', bbox_inches='tight', dpi=600)
    plt.show()
    
############################################################################################################################################

def create_github_calendar(dates, figsize_per_year=(15, 3), colors=None):
    """
    Create GitHub-style calendar visualization showing data availability by year.
    
    Args:
        dates: List of datetime objects
        figsize_per_year: Tuple of (width, height) for each year's calendar
        colors: Dict with 'no_data' and 'has_data' colors
    
    Returns:
        matplotlib figure
    """
    if not dates:
        print("No dates provided!")
        return None
    
    # Default colors (GitHub-like)
    if colors is None:
        colors = {
            'no_data': '#ebedf0',    # Light gray for days without data
            'has_data': '#216e39',   # Green for days with data
            'grid': '#d0d7de'        # Grid lines
        }
    
    # Convert dates to set for O(1) lookup
    date_set = set(date.date() for date in dates)
    
    # Get unique years and sort them
    years = sorted(set(date.year for date in dates))
    n_years = len(years)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_years, 1, 
                            figsize=(figsize_per_year[0], figsize_per_year[1] * n_years),
                            facecolor='white')
    
    # Handle case where there's only one year
    if n_years == 1:
        axes = [axes]
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        
        # Create yearly calendar for this year
        create_single_year_calendar(ax, year, date_set, colors)
        
        # Set title for each year
        ax.set_title(f'{year}', fontsize=16, fontweight='bold', pad=50)
    
    plt.tight_layout()
    if save_plot:
        fig.savefig(output_path / 'github_calendar.png', bbox_inches='tight', dpi=600)
    # return fig

def create_single_year_calendar(ax, year, date_set, colors):
    """Create a single year's GitHub-style calendar"""
    
    # Start from first day of the year
    start_date = datetime(year, 1, 1).date()
    
    # Find the first Monday on or before January 1st
    days_from_monday = start_date.weekday()  # 0 = Monday, 6 = Sunday
    first_monday = start_date - timedelta(days=days_from_monday)
    
    # Create calendar grid
    square_size = 10
    gap = 2
    
    # Calculate dimensions
    weeks_in_year = 53  # Maximum weeks we might need
    days_in_week = 7
    
    width = weeks_in_year * (square_size + gap) - gap
    height = days_in_week * (square_size + gap) - gap
    
    # Set up the axes
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Month labels
    month_positions = []
    current_date = first_monday
    week = 0
    
    # Track which month we're in for labels
    current_month = None
    
    # Draw the calendar
    while current_date.year <= year:
        # Calculate position
        week_offset = week * (square_size + gap)
        day_offset = current_date.weekday() * (square_size + gap)
        
        x = week_offset
        y = height - day_offset - square_size  # Flip y-axis so Monday is at top
        
        # Choose color based on whether we have data for this date
        if current_date.year == year and current_date in date_set:
            color = colors['has_data']
        else:
            color = colors['no_data']
        
        # Only draw squares for the target year
        if current_date.year == year:
            # Draw square
            square = patches.Rectangle((x, y), square_size, square_size,
                                     linewidth=0.5, edgecolor=colors['grid'],
                                     facecolor=color)
            ax.add_patch(square)
        
        # Track month positions for labels
        if current_date.year == year and current_date.day == 1:
            month_positions.append((week_offset, current_date.month))
        
        # Move to next day
        current_date += timedelta(days=1)
        
        # Move to next week on Sunday
        if current_date.weekday() == 0:  # Monday
            week += 1
    
    # Add month labels
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for x_pos, month_num in month_positions:
        ax.text(x_pos, height + 15, month_names[month_num - 1], 
               fontsize=10, ha='left', va='bottom')

    # Add day labels (Mon, Tues, Wed, Thurs, Fri, Sat)
    day_labels = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
    for i, label in enumerate(day_labels):
        if label:  # Only show Mon, Tues, Wed, Thurs, Fri, Sat, Sun
            y_pos = height - i * (square_size + gap) - square_size/2
            ax.text(-15, y_pos, label, fontsize=8, ha='right', va='center')


############################################################################################################################################

def plot_github_like_calendar(dates):
    """
    Plot a GitHub-like yearly calendar: one subplot per year.
    Each day is a square; colored if present in `dates`.

    Args:
        dates: Iterable of datetime-like (datetime.date or pandas.Timestamp)
    """
    if dates is None or len(dates) == 0:
        print("No dates to plot.")
        return

    # Normalize to date objects and build a set for fast membership checks
    date_list = [pd.to_datetime(d).date() for d in dates]
    date_set = set(date_list)
    years = sorted({d.year for d in date_list})

    # Layout: one row if <= 3 years, else multi-row grid up to 3 columns
    n = len(years)
    ncols = 1 if n == 1 else (2 if n == 2 else min(3, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(6*ncols, 1.6*nrows), 
        squeeze=False
        )
    cmap_present = plt.cm.Greens
    color_present = cmap_present(0.8)
    color_absent = '#ebedf0'  # GitHub light gray background for absent days
    edge_color = 'white'

    for idx, year in enumerate(years):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_title(str(year), fontsize=14, fontweight='bold', pad=12)
        ax.set_aspect('equal')
        ax.axis('off')

        # Determine first Sunday of the year grid and last day of the year
        first_day = datetime(year, 1, 1).date()
        last_day = datetime(year, 12, 31).date()

        # GitHub-like grid starts on Sunday columns; compute the Sunday on/before Jan 1
        start_sunday = first_day - timedelta(days=(first_day.weekday() + 1) % 7)

        # Compute number of weeks to cover the whole year up to the last week that includes Dec 31
        # Find the Saturday of the week containing Dec 31, then add one day to include that week in range
        end_saturday = last_day + timedelta(days=(5 - last_day.weekday()) % 7)
        total_days = (end_saturday - start_sunday).days + 1
        n_weeks = int(np.ceil(total_days / 7))

        # Day-of-week labels (rows). Use Sunday-first to match the grid rows (r=0 is Sunday)
        dow_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

        # Draw weekday labels at left
        for r, lbl in enumerate(dow_labels):
            ax.text(-0.6, r + 0.5, lbl, va='center', ha='right', fontsize=9, color='#6a737d')

        # Draw month labels above columns when month changes
        month_positions = {}
        # Build mapping of each day in grid
        for w in range(n_weeks):
            for r in range(7):
                current_day = start_sunday + timedelta(days=w*7 + r)
                # Only shade cells belonging to this year's grid rows; keep off-year days as background
                in_year = (current_day.year == year)
                color = color_present if (in_year and current_day in date_set) else color_absent
                rect = plt.Rectangle((w, r), 1, 1, facecolor=color, edgecolor=edge_color, linewidth=0.5)
                ax.add_patch(rect)

                # Record month position if this is a Monday of a month inside the year
                if in_year and current_day.weekday() == 0 and current_day.day <= 7:
                    month_positions.setdefault(current_day.month, w)

        # Place month labels roughly once per month
        month_names = calendar.month_abbr
        for m, w in month_positions.items():
            ax.text(w + 0.5, -0.4, month_names[m], ha='center', va='center', fontsize=10, color='#24292e')

        # Set limits to fit the grid
        ax.set_xlim(-1.1, n_weeks + 0.5)
        ax.set_ylim(7, -1.2)

    # Hide any unused subplots
    for j in range(idx + 1, nrows * ncols):
        ax = axes[j // ncols][j % ncols]
        ax.axis('off')

    plt.tight_layout()
    if save_plot:
        fig.savefig(output_path / 'github_like_calendar.png', bbox_inches='tight', dpi=600)
    plt.show()