from src.config.logger import setup_logger
from src import get_logger
import pandas as pd
import numpy as np

from src.config import config_settings
setup_logger(
        name=config_settings.PROJECT_NAME,
        level=config_settings.logging["level"],  # Default
        log_file=None,  # No file yet
        console_output=True
)

logger = get_logger(__name__)

path = "/Users/jesusvillotamiranda/Desktop/LOCAL_DATASETS/WHALES/OLD/merged_0.95_new.parquet"
logger.info(f"Loading data from {path}")
pdf = pd.read_parquet(path)
logger.info(f"Successfully loaded {len(pdf)} records")

from src.plotting.plotting_1 import *

# logger.info("Creating time histograms for different periods")
# freq_min = 10
# time_histogram(pdf, start_time="00:00", end_time="09:29", freq_min=freq_min)
# time_histogram(pdf, start_time="09:30", end_time="17:15", freq_min=freq_min)
# time_histogram(pdf, start_time="17:15", end_time="23:59", freq_min=freq_min)


# logger.info("Analyzing trade sizes")
# analyze_trade_sizes(pdf=pdf, 
#                     start_hour=9,
#                     end_hour=20,
#                     interval=15)


# Example usage with current parameters - Original behavior (by slot)
# logger.info("Creating percentage plots within each time slot")
# plot_trade_size_percentile_heatmap(
#     pdf,
#     interval=15,
#     start_hour=9,
#     end_hour=20,
#     quantile_bin_pct=1,
#     normalize='by_slot'  # Original behavior
# )

# logger.info("Creating global percentage plots across all time slots")
# # Example usage with global normalization
# plot_trade_size_percentile_heatmap(
#     pdf, 
#     interval=15, 
#     start_hour=9, 
#     end_hour=20, 
#     quantile_bin_pct=1,
#     normalize='global'  # New behavior
# )

logger.info("Creating scatter plot with trend")
# scatter_with_trend(pdf, "fragment_count", "prtSize_agg", show_trend=False)

scatter_with_trend_matplotlib(pdf, "fragment_count", "prtSize_agg", show_trend=True)