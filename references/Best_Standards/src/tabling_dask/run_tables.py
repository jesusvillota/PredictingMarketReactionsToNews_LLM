"""
Central runner script for generating LaTeX tables from processed trade data.

Usage Examples:
---------------

uv run src/tabling_dask/run_tables.py --tables all --ticker-type equity --strategy-type simple
uv run src/tabling_dask/run_tables.py --tables all --ticker-type equity --strategy-type complex
uv run src/tabling_dask/run_tables.py --tables all --ticker-type equity --strategy-type all
uv run src/tabling_dask/run_tables.py --tables all --ticker-type all --strategy-type simple
uv run src/tabling_dask/run_tables.py --tables all --ticker-type all --strategy-type complex
uv run src/tabling_dask/run_tables.py --tables all --ticker-type all --strategy-type all

# Generate specific tables (Table1, Table4, Table8) with simple strategies:
uv run src/tabling_dask/run_tables.py --tables Table1,Table4,Table8 --ticker-type all --strategy-type simple

# Generate a single table (Table2_broad) for equity options, all strategies, custom output directory:
uv run src/tabling_dask/run_tables.py --tables Table2_broad --ticker-type equity --strategy-type all --output-dir custom/path

# Generate multiple tables with equity filter and simple strategies:
uv run src/tabling_dask/run_tables.py --tables Table6,Table7,Table8 --ticker-type equity --strategy-type simple
"""

from __future__ import annotations

#---------------------------------------------------------------
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.config import initialize_main, DaskManager
from src.config.config_settings import tables as tables_settings, PROCESSED_PATH
from src.config import config_settings
from src.tabling_dask.common import build_parquet_filters, get_column_union
import dask.dataframe as dd

# Type for build_table functions: (parquet_filters: list, output_dir: Path, ddf: Optional[dd.DataFrame]) -> None
TableFunc = Callable[[list, Path, Optional[dd.DataFrame]], object]


def _build_filter_subdir(ticker_type: str, strategy_type: str) -> str:
    """Build subdirectory name based on filter combination.
    
    Examples:
        equity + simple -> "equity_ticker-simple_strat"
        all + complex -> "all_ticker-complex_strat"
        equity + all -> "equity_ticker-all_strat"
    """
    return f"{ticker_type}_ticker-{strategy_type}_strat"


def _resolve_output_dir(override: str | None, ticker_type: str, strategy_type: str) -> Path:
    """Resolve base output directory and append filter-based subdirectory."""
    if override:
        base_dir = Path(override)
    else:
        base_dir = Path(tables_settings["dask_path"])  # default from config
    
    # Append filter-based subdirectory
    filter_subdir = _build_filter_subdir(ticker_type, strategy_type)
    output_dir = base_dir / filter_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_registry() -> Dict[str, TableFunc]:
    """Import and register build_table functions from table modules.

    The key is the module base name (e.g., "Table6").
    """
    # Local imports to avoid heavy imports when just parsing CLI
    from src.tabling_dask import (
        Table1,
        Table2_broad,
        Table2_granular,
        Table3,
        # Table4,
        # Table5,
        Table6,
        Table7,
        Table8,
    )

    registry: Dict[str, TableFunc] = {
        "Table1": getattr(Table1, "build_table", None),
        "Table2_broad": getattr(Table2_broad, "build_table", None),
        "Table2_granular": getattr(Table2_granular, "build_table", None),
        "Table3": getattr(Table3, "build_table", None),
        # "Table4": getattr(Table4, "build_table", None),
        # "Table5": getattr(Table5, "build_table", None),
        "Table6": getattr(Table6, "build_table", None),
        "Table7": getattr(Table7, "build_table", None),
        "Table8": getattr(Table8, "build_table", None),
    }
    # Drop missing ones (before refactor some may be None)
    return {k: v for k, v in registry.items() if callable(v)}


def main(
    tables: List[str] | None = None,
    ticker_type: str = "all",
    strategy_type: str = "all",
    output_dir: str | None = None,
) -> None:
    logger = initialize_main()

    selected_tables = tables or ["all"]
    registry = _load_registry()

    if selected_tables == ["all"] or (len(selected_tables) == 1 and selected_tables[0].lower() == "all"):
        to_run = list(registry.keys())
    else:
        to_run = selected_tables

    out_dir = _resolve_output_dir(output_dir, ticker_type, strategy_type)
    parquet_filters = build_parquet_filters(ticker_type, strategy_type)

    logger.info(
        f"Running tables: {to_run} | ticker_type={ticker_type} | strategy_type={strategy_type} | output_dir={out_dir}"
    )

    # Pre-load data if running multiple tables (excluding Table4 & Table5 which uses different source)
    pre_loaded_ddf = None
    tables_to_preload = [t for t in to_run if t not in {"Table4", "Table5"}]
    
    if len(tables_to_preload) > 1:
        # Multiple tables selected (excluding Table4 & Table5) - pre-load data
        union_columns = get_column_union(tables_to_preload)
        if union_columns:
            logger.info(f"Pre-loading data with columns: {union_columns}")
            logger.info(f"This will be shared across {len(tables_to_preload)} tables for efficiency.")
            try:
                with DaskManager():
                    pre_loaded_ddf = dd.read_parquet(
                        path=PROCESSED_PATH,
                        engine=config_settings.parquet["engine"],
                        filters=parquet_filters,
                        columns=union_columns,
                        split_row_groups='infer',
                    )
                    logger.info(f"Pre-loaded Dask DataFrame with {pre_loaded_ddf.npartitions} partitions.")
            except Exception as e:
                logger.exception(f"Error pre-loading parquet: {e}")
                logger.warning("Falling back to individual table loading.")
                pre_loaded_ddf = None

    # Build each table
    for name in to_run:
        func = registry.get(name)
        if not func:
            logger.warning(f"Table {name} not found or not refactored with build_table yet. Skipping.")
            continue
        logger.info(f"Building {name} ...")
        
        # Table4 and Table5 always load independently (use COMPLEX_TRADES_PATH)
        if name in {"Table4", "Table5"}:
            func([], out_dir)  # Empty filters for tables that don't use PROCESSED_PATH
        # Other tables use pre-loaded ddf if available
        else:
            func(parquet_filters, out_dir, ddf=pre_loaded_ddf)
        
        logger.info(f"Completed {name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tabling_dask tables")
    parser.add_argument(
        "--tables",
        type=str,
        default="all",
        help="Comma-separated list of table module names (e.g., Table1,Table4) or 'all'",
    )
    parser.add_argument(
        "--ticker-type",
        type=str,
        choices=["equity", "all"],
        default="all",
        help="Filter by ticker class",
    )
    parser.add_argument(
        "--strategy-type",
        type=str,
        choices=["simple", "complex", "all"],
        default="all",
        help="Filter by strategy type via prtType",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override",
    )

    args = parser.parse_args()
    table_list = [s.strip() for s in args.tables.split(",")] if args.tables else ["all"]
    main(
        tables=table_list,
        ticker_type=args.ticker_type,
        strategy_type=args.strategy_type,
        output_dir=args.output_dir,
    )