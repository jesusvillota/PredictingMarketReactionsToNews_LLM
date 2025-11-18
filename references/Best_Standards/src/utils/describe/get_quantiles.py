# This script takes daily files and filters them by prtSize_agg quantile / or by a fixed size value

# 1. Identify large trades (top 5% or 1% by prtSize_agg)
volume_threshold = ddf['prtSize_agg'].quantile(percentile_threshold)
large_trades = ddf[ddf['prtSize_agg'] >= volume_threshold]