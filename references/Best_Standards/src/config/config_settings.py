# src/config/config_settings.py

PROJECT_NAME: str = "whales"
import os
from pathlib import Path
from typing import Any

from .data_schemas import *
from THIS_IS import *

#-------------------------------------------------------------------------------------#
# TEST: bool = TEST                            # Options: True, False (whether to run in test mode)
# RUN_MISSING_DATES: bool = RUN_MISSING_DATES  # Options: True, False (whether to re-process missing dates)
#-------------------------------------------------------------------------------------#

if MACHINE_ID == "cluster": 
    # Inherited from main.slurm
    CPU_LIMIT: int = int(os.environ.get("CPU_LIMIT", "5"))
    RAM_LIMIT: int = int(os.environ.get("RAM_LIMIT", "60"))
    DASK_TEMP_DIR: Path = Path(os.environ.get("DASK_TEMP_DIR", "/tmp"))
    # Hardcoded INPUT data path
    PROCESSED_PATH: Path = Path("PROCESSED_TRADE_DATA_PARQUET")
    PATHS: Path = Path("PROCESSED_TRADE_DATA_PARQUET")

parquet: dict[str, Any] = {
    "engine": "pyarrow",  # Parquet engine: 'pyarrow' or 'fastparquet'
    "compression": "snappy",  # Compression algorithm: 'snappy', 'gzip', 'brotli', etc.
    "write_index": False, # Whether to write the DataFrame index
    # "write_metadata_file": False  # Whether to write the _metadata file
}

# Logging settings
logging: dict[str, Any] = {
    "level": "DEBUG",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "console_output": True,
    "log_file": Path(OUTPUT_PATH / f"logs/{PROJECT_NAME}.log")  # Set to None to disable file logging
}

# Tables settings
tables: dict[str, Any] = {
    "save_table": True,
    # "duckdb_path": Path("TeX/tables/duckdb/test"),
    "dask_path": Path("TeX/tables/")
}

# Plotting settings
plotting: dict[str, Any] = {
    # Plot saving settings
    "save_plot": True,  # True or False
    "output_path": Path("TeX/figures"),
    "dpi": 600,  # Dots per inch for saved figures
    "show_title": False,

    # LaTeX and font settings
    "tex_settings": {
        "text.usetex": True,
        "font.family": "serif",  # "courier", "sans-serif", "fantasy", "monospace", "cursive", "serif"
        "font.serif": ["Computer Modern Roman"],
        "axes.formatter.use_mathtext": True,
        "mathtext.fontset": "cm",
        "text.latex.preamble": r"\usepackage{amsmath}"  # Raw string for LaTeX
    },

    # Font sizes for different plot elements
    "fontsize": {
        "title_size": 22,    # Size for plot titles
        "label_size": 20,    # Size for axis labels
        "tick_size": 14,     # Size for axis ticks
        "legend_size": 18,    # Size for legend text
    },

    # Padding settings for plot elements
    "padding": {
        "title_pad": 15,     # Padding for plot title
        "label_pad": 15,       # Padding for axis labels
    },

    # Figure dimensions
    "figsize": {
        "width": 15,
        "height": 10,
    },

    # Color scheme for different plot elements
    "colors": {
        "background": "#f5f5f5"     # Plot background color
    },

    # Transparency settings
    "alpha": {
        "grid": 0.3,      # Transparency for grid lines
        "fill": 0.3       # Transparency for filled areas
    }
}