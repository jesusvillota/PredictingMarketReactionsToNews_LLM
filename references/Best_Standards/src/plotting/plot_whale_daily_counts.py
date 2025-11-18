# uv run src/plotting/plot_whale_daily_counts.py

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import polars as pl

import THIS_IS  # project-level configuration

def resolve_input_path(cli_input: Optional[str]) -> Path:
    """Resolve the input path for whale parquet partitions.

    Prefers CLI override; falls back to THIS_IS.WHALES if available.
    """
    if cli_input:
        return Path(cli_input).expanduser().resolve()
    if THIS_IS is not None and hasattr(THIS_IS, "WHALES"):
        return Path(getattr(THIS_IS, "WHALES")).expanduser().resolve()
    raise ValueError("Input path not provided and THIS_IS.WHALES is unavailable.")


def resolve_output_dir(cli_output_dir: Optional[str]) -> Path:
    """Resolve the output directory for CSV and PNG outputs."""
    if cli_output_dir:
        out = Path(cli_output_dir).expanduser().resolve()
    elif THIS_IS is not None and hasattr(THIS_IS, "OUTPUT_FOLDER"):
        out = Path(getattr(THIS_IS, "OUTPUT_FOLDER")).expanduser().resolve()
    else:
        out = Path.cwd() / "_OUTPUT_"
    out.mkdir(parents=True, exist_ok=True)
    return out


def compute_daily_counts(
    input_dir: Path,
    pattern: str = "**/*.parquet",
    timestamp_col: str = "timestamp_ny",
) -> pl.DataFrame:
    """Compute daily counts of whales from parquet partitions using Polars lazy.

    Only the aggregated result (date, count) is collected into memory.
    """
    # Verify there are files to read to fail fast with a friendly message
    import glob

    matched_files = glob.glob(os.path.join(str(input_dir), pattern), recursive=True)
    if not matched_files:
        raise FileNotFoundError(
            f"No parquet files matched under '{input_dir}' with pattern '{pattern}'."
        )

    lf = pl.scan_parquet(os.path.join(str(input_dir), pattern))

    # Validate the timestamp column exists by forcing a minimal projection
    try:
        _ = lf.select(pl.col(timestamp_col)).head(1).collect()
    except Exception as exc:  # pragma: no cover
        raise KeyError(
            f"Required column '{timestamp_col}' not found or unreadable in input parquet."
        ) from exc

    # Assume parquet stores proper temporal type; aggregate by date
    daily_lazy = (
        lf.select(pl.col(timestamp_col).dt.date().alias("date"))
        .group_by("date")
        .len()
        .sort("date")
    )

    # Prefer new streaming engine; gracefully fall back if unavailable
    try:
        daily_df = daily_lazy.collect(engine="streaming")  # polars >= 1.25
    except TypeError:
        daily_df = daily_lazy.collect()
    # Normalize output column names
    daily_df = daily_df.rename({"len": "count"})
    return daily_df


def save_csv(daily_df: pl.DataFrame, output_dir: Path, filename: str = "whales_daily_counts.csv") -> Path:
    out_csv = output_dir / filename
    daily_df.write_csv(out_csv)
    return out_csv


def save_plot(
    daily_df: pl.DataFrame,
    output_dir: Path,
    filename: str = "whales_daily_counts.png",
) -> Path:
    out_png = output_dir / filename

    # Convert to Python lists for plotting
    dates = daily_df["date"].to_list()
    counts = daily_df["count"].to_list()

    plt.figure(figsize=(12, 5))
    plt.plot(dates, counts, linewidth=1.8)
    plt.title("Daily Whale Counts")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    # Improve date tick formatting without heavy imports
    try:
        import matplotlib.dates as mdates

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    except Exception:
        pass

    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot the number of whale observations per day from parquet partitions."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory containing parquet partitions (defaults to THIS_IS.WHALES)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write CSV and PNG (defaults to THIS_IS.OUTPUT_FOLDER or ./_OUTPUT_)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.parquet",
        help="Glob pattern to find parquet files recursively",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default="timestamp_ny",
        help="Name of the timestamp column to aggregate by (daily)",
    )

    args = parser.parse_args()

    input_dir = resolve_input_path(args.input)
    output_dir = resolve_output_dir(args.output_dir)

    daily_df = compute_daily_counts(
        input_dir=input_dir, pattern=args.pattern, timestamp_col=args.timestamp_col
    )

    csv_path = save_csv(daily_df, output_dir)
    png_path = save_plot(daily_df, output_dir)

    print(f"Saved daily counts CSV to: {csv_path}")
    print(f"Saved daily counts plot to: {png_path}")


if __name__ == "__main__":
    main()


