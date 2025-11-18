"""Visualization utilities for plotting and figures."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def setup_plotting_style():
    """Set up matplotlib and seaborn plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def save_figure(fig, output_path: Path, filename: str, dpi: int = 300, 
                bbox_inches: str = 'tight'):
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        output_path: Directory to save the figure
        filename: Name of the file (with or without extension)
        dpi: Resolution in dots per inch
        bbox_inches: Bounding box setting for tight layout
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith(('.png', '.pdf', '.svg', '.jpg')):
        filename = f"{filename}.png"
    
    filepath = output_path / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

