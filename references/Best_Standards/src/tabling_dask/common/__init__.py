"""Common utilities for tabling_dask.

Modules:
- formatting: table cell formatting and total row helpers
- filters: parquet filter builders for ticker/strategy types
- table_metadata: table column requirements registry
- binning: standard size bin creation utilities
"""

from .formatting import fmt_cell, format_count, format_percentage, append_total_row  # re-export
from .filters import build_parquet_filters  # re-export
from .table_metadata import TABLE_COLUMNS, get_required_columns, get_column_union  # re-export
from .binning import create_standard_size_bins  # re-export


