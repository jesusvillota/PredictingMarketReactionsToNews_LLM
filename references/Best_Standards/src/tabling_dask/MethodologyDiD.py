"""Standalone script to run staggered floor-closure DiD regressions.

Usage examples:
---------------

Run with the default +/-30 business-day window around each closure:

    uv run src/tabling_dask/MethodologyDiD.py

Tighten the window to +/-7 business days:

    uv run src/tabling_dask/MethodologyDiD.py --window-days 7

Optionally override output locations:

    uv run src/tabling_dask/MethodologyDiD.py --window-days 45 \
        --table-output TeX/tables/dask/test --fig-output TeX/figures

The script loads processed trade-level data from ``PROCESSED_PATH`` (see
``src/config/config_settings.py``), filters to simple equity trades executed on
open-outcry venues that faced COVID-related floor suspensions, and estimates
both a baseline difference-in-differences regression and an event-study
specification that allows treatment effects to vary for whale trades (>=200
contracts).

Outputs:
    * LaTeX regression table saved alongside other Dask-generated tables.
    * Event-study plot with 95% confidence bands saved under the configured
      figures directory.

Future integration: this module is written as a standalone utility so it can be
invoked manually while the workflow is being validated. Once stable, the
``run_tables.py`` registry can import and orchestrate this script similar to
other tabling modules.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
from dataclasses import dataclass
from datetime import date, timedelta
import argparse
import math
from pathlib import Path
import sys
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import dask.dataframe as dd

try:
    from linearmodels.iv.model import AbsorbingLS  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback import
    try:
        from linearmodels.iv.absorbing import AbsorbingLS  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - final fallback
        raise ImportError(
            "linearmodels >= 5.0 with AbsorbingLS is required for MethodologyDiD"
        ) from exc

from src.config import config_settings, initialize_main, DaskManager, get_logger
from src.config.config_settings import PROCESSED_PATH, tables
from src.tabling_dask.common import build_parquet_filters


# ---------------------------------------------------------------------------
# Hard-coded floor closure timeline based on notes/covid_floor_restrictions.md
# Dates are inclusive. Secondary closure for PHLX in late 2020 approximates the
# November shutdown noted in public filings; adjust as needed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosurePeriod:
    exchange: str
    start: date
    end: date
    label: str


CLOSURE_PERIODS: Sequence[ClosurePeriod] = (
    ClosurePeriod("CBOE", date(2020, 3, 16), date(2020, 6, 15), "covid_2020"),
    ClosurePeriod("PHLX", date(2020, 3, 17), date(2020, 6, 3), "covid_2020"),
    # ClosurePeriod("PHLX", date(2020, 11, 2), date(2020, 12, 31), "covid_wave2"),
    # ClosurePeriod("AMEX", date(2020, 3, 23), date(2020, 5, 26), "covid_2020"),
    ClosurePeriod("BOX", date(2020, 3, 20), date(2020, 5, 4), "covid_2020"),
    ClosurePeriod("NYSE", date(2020, 3, 23), date(2020, 5, 26), "covid_2020"),
)


FLOOR_EXCHANGES: List[str] = sorted({period.exchange for period in CLOSURE_PERIODS})


TIME_TO_EXPIRY_ORDER = {
    "lt_1w": 1,
    "1w_to_2w": 2,
    "2w_to_4w": 3,
    "1m_to_3m": 4,
    "3m_to_12m": 5,
    "gt_1y": 6,
}


DEFAULT_WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run floor-closure DiD and event-study regressions."
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Number of business days before and after each closure to retain in the sample.",
    )
    parser.add_argument(
        "--table-output",
        type=str,
        default=None,
        help="Optional override for LaTeX table output directory (defaults to tables['dask_path']).",
    )
    parser.add_argument(
        "--fig-output",
        type=str,
        default=None,
        help="Optional override for figure output directory (defaults to plotting['output_path']).",
    )
    parser.add_argument(
        "--persist-intermediate",
        action="store_true",
        help="Persist the filtered dataset to disk (debug aid).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _window_bounds(window_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_start = min(period.start for period in CLOSURE_PERIODS)
    max_end = max(period.end for period in CLOSURE_PERIODS)
    pre_span = pd.tseries.offsets.BusinessDay(window_days)
    post_span = pd.tseries.offsets.BusinessDay(window_days)
    window_start = pd.Timestamp(min_start) - pre_span
    window_end = pd.Timestamp(max_end) + post_span
    return window_start.normalize(), window_end.normalize()


def load_trades(window_days: int) -> pd.DataFrame:
    logger = get_logger(__name__)
    required_columns = [
        "timestamp_ny",
        "prtExch",
        "prtSize_agg",
        "quoted_spread",
        "trade_type",
        "ticker_class",
        "prtDe",
        "prtIv",
        "moneyness",
        "time_to_expiry",
        "buy_sell_class",
        "okey_cp",
        "okey_tk",
    ]

    start_bound, end_bound = _window_bounds(window_days)
    logger.debug(
        "Date window bounds (business days): %s to %s", start_bound.date(), end_bound.date()
    )

    filters = build_parquet_filters(
        ticker_type="equity",
        strategy_type="simple",
        start=start_bound,
        end=end_bound,
    )
    logger.info("Loading processed trades for exchanges with floor closures ...")
    logger.debug("Parquet filters (with date bounds): %s", filters)

    with DaskManager():
        try:
            ddf = dd.read_parquet(
                path=PROCESSED_PATH,
                engine=config_settings.parquet["engine"],
                columns=required_columns,
                filters=filters,
                split_row_groups="infer",
            )
        except Exception:  # pragma: no cover - logging adds context
            logger.exception("Failed to read processed parquet data")
            raise

        # Apply exchange filter lazily before materialising to pandas
        ddf = ddf[ddf["prtExch"].isin(FLOOR_EXCHANGES)]
        ddf["timestamp_ny"] = dd.to_datetime(ddf["timestamp_ny"], utc=False, errors="coerce")

        pdf = ddf.compute()

    pdf = pdf.dropna(subset=["timestamp_ny", "quoted_spread", "prtSize_agg"])

    # Enforce simple equity trades redundantly for safety
    # pdf = pdf.loc[
    #     (pdf["trade_type"] == "simple") & (pdf["ticker_class"].str.lower() == "equity")
    # ].copy()

    pdf["trade_date"] = pdf["timestamp_ny"].dt.tz_localize(None).dt.normalize()
    pdf["trade_month"] = pdf["trade_date"].dt.to_period("M").astype(str)

    logger.info("Loaded %s trades after filtering", f"{len(pdf):,}")
    return pdf


# ---------------------------------------------------------------------------
# Event-panel construction
# ---------------------------------------------------------------------------


def build_event_panel(trades: pd.DataFrame, window_days: int) -> pd.DataFrame:
    logger = get_logger(__name__)
    panels: List[pd.DataFrame] = []

    for idx, period in enumerate(CLOSURE_PERIODS, start=1):
        logger.info(
            "Processing closure window: %s (%s to %s)",
            period.exchange,
            period.start,
            period.end,
        )

        pre_window = pd.tseries.offsets.BusinessDay(window_days)
        post_window = pd.tseries.offsets.BusinessDay(window_days)

        window_start = pd.Timestamp(period.start) - pre_window
        window_end = pd.Timestamp(period.end) + post_window

        event_slice = trades.loc[
            (trades["prtExch"] == period.exchange)
            & (trades["trade_date"] >= window_start)
            & (trades["trade_date"] <= window_end)
        ].copy()

        if event_slice.empty:
            logger.warning(
                "No observations found for %s around %s (window +/- %d business days)",
                period.exchange,
                period.start,
                window_days,
            )
            continue

        trade_dates = event_slice["trade_date"].values.astype("datetime64[D]")
        start_np = np.datetime64(period.start)
        end_np = np.datetime64(period.end)

        after_start = trade_dates >= start_np
        event_slice.loc[:, "event_time"] = np.where(
            after_start,
            np.busday_count(start_np, trade_dates),
            -np.busday_count(trade_dates, start_np),
        )
        event_slice.loc[:, "is_closed"] = (
            (trade_dates >= start_np) & (trade_dates <= end_np)
        )
        event_slice.loc[:, "event_id"] = f"{period.exchange}_{period.label}_{idx}"
        event_slice.loc[:, "closure_start"] = period.start
        event_slice.loc[:, "closure_end"] = period.end
        panels.append(event_slice)

    if not panels:
        raise ValueError("No data available for any closure window; adjust filters or window size.")

    combined = pd.concat(panels, axis=0, ignore_index=True)
    combined.sort_values(["event_id", "trade_date", "timestamp_ny"], inplace=True)

    return combined


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


def make_regression_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["whale"] = (df["prtSize_agg"] >= 200).astype(int)
    df["closed_whale"] = df["is_closed"].astype(int) * df["whale"]

    # Direction controls relative to midpoint
    df["is_buy"] = (df["buy_sell_class"] == "buy").astype(int)
    df["is_sell"] = (df["buy_sell_class"] == "sell").astype(int)

    # Option type control (Call baseline)
    df["is_put"] = (df["okey_cp"].str.lower() == "put").astype(int)

    # Time-to-expiry ordinal score (fallback to median if missing)
    df["tte_bucket"] = df["time_to_expiry"].map(TIME_TO_EXPIRY_ORDER)
    if df["tte_bucket"].isna().any():
        median_tte = df["tte_bucket"].dropna().median()
        df["tte_bucket"] = df["tte_bucket"].fillna(median_tte)

    # Delta/moneyness/IV sanitisation
    df["abs_delta"] = df["prtDe"].abs()
    df["moneyness"] = df["moneyness"].replace([-np.inf, np.inf], np.nan)
    df["prtIv"] = df["prtIv"].replace([-np.inf, np.inf], np.nan)
    df["moneyness"] = df["moneyness"].fillna(df["moneyness"].median())
    df["prtIv"] = df["prtIv"].fillna(df["prtIv"].median())

    return df


# ---------------------------------------------------------------------------
# Regression routines
# ---------------------------------------------------------------------------


def run_baseline_regression(df: pd.DataFrame) -> AbsorbingLS:
    y = df["quoted_spread"].astype(np.float32)
    X = df[
        [
            "is_closed",
            "whale",
            "closed_whale",
            "abs_delta",
            "moneyness",
            "prtIv",
            "tte_bucket",
            "is_put",
            "is_buy",
            "is_sell",
        ]
    ].astype(np.float32)
    # Convert absorb columns to categorical to avoid string-to-float conversion errors
    absorb = df[["prtExch", "trade_month", "okey_tk"]].copy()
    for col in absorb.columns:
        absorb[col] = absorb[col].astype("category")
    
    clusters = df["prtExch"].astype(str) + "_" + df["trade_date"].astype(str)

    model = AbsorbingLS(y, X, absorb=absorb)
    return model.fit(cov_type="clustered", clusters=clusters)


def run_event_study(df: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, AbsorbingLS]:
    # Limit to relevant event times to avoid spurious bins
    max_closure_span = max(
        np.busday_count(np.datetime64(period.start), np.datetime64(period.end)) + 1
        for period in CLOSURE_PERIODS
    )
    max_event_horizon = max_closure_span + window_days

    filtered = df.loc[
        (df["event_time"] >= -window_days)
        & (df["event_time"] <= max_event_horizon)
    ].copy()

    negative_times = sorted(filtered.loc[filtered["event_time"] < 0, "event_time"].unique())
    if negative_times:
        base_period = negative_times[-1]
    else:
        base_period = 0

    event_dummies = pd.get_dummies(filtered["event_time"], prefix="event")
    base_col = f"event_{base_period}"
    if base_col in event_dummies:
        event_dummies.drop(columns=[base_col], inplace=True)

    # Retain only event bins with at least one observation
    nonzero_cols = [col for col in event_dummies.columns if event_dummies[col].sum() > 0]
    event_dummies = event_dummies[nonzero_cols]
    if event_dummies.empty:
        raise ValueError("Event-study design matrix is empty; widen the window or inspect data availability.")

    # Convert event dummies to sparse format to save memory, then back to dense for compatibility
    # Create interaction terms efficiently using vectorized operations
    interaction_dummies = event_dummies.multiply(filtered["whale"].values, axis=0)
    interaction_dummies.columns = [f"{col}_whale" for col in event_dummies.columns]
    
    # Build the full design matrix using concat (much faster than repeated column assignment)
    control_cols = ["whale", "abs_delta", "moneyness", "prtIv", "tte_bucket", "is_put", "is_buy", "is_sell"]
    X = pd.concat(
        [
            event_dummies.astype(np.float32),  # Use float32 to save memory
            interaction_dummies.astype(np.float32),
            filtered[control_cols].astype(np.float32),
        ],
        axis=1,
    )

    # Convert absorb columns to categorical to avoid string-to-float conversion errors
    absorb = filtered[["prtExch", "trade_month", "okey_tk"]].copy()
    for col in absorb.columns:
        absorb[col] = absorb[col].astype("category")
    
    clusters = filtered["prtExch"].astype(str) + "_" + filtered["trade_date"].astype(str)

    model = AbsorbingLS(filtered["quoted_spread"].astype(np.float32), X, absorb=absorb)
    results = model.fit(cov_type="clustered", clusters=clusters)

    # Extract coefficient names from the design matrix
    exog_cols = list(event_dummies.columns)
    
    event_summary = []
    for col in exog_cols:
        k = int(col.split("_")[-1])
        if col not in results.params.index:
            continue

        whale_col = f"{col}_whale"
        base_effect = results.params[col]
        base_se = results.std_errors[col]
        whale_increment = results.params.get(whale_col, 0.0)
        whale_se = results.std_errors.get(whale_col, np.nan)

        if np.isnan(base_se):
            continue

        total_whale = base_effect + whale_increment
        cov_term = 0.0
        if whale_col in results.cov.index and col in results.cov.columns:
            cov_term = results.cov.loc[col, whale_col]
        elif whale_col in results.cov.columns and col in results.cov.index:
            cov_term = results.cov.loc[whale_col, col]

        if not np.isnan(whale_se):
            total_se = math.sqrt(base_se**2 + whale_se**2 + 2 * cov_term)
        else:
            total_se = np.nan

        event_summary.append(
            {
                "k": k,
                "beta": base_effect,
                "beta_se": base_se,
                "beta_whale_inc": whale_increment,
                "beta_whale_inc_se": whale_se,
                "beta_whale_total": total_whale,
                "beta_whale_total_se": total_se,
            }
        )

    event_df = pd.DataFrame(event_summary).sort_values("k")
    event_df["ci_lower"] = event_df["beta"] - 1.96 * event_df["beta_se"]
    event_df["ci_upper"] = event_df["beta"] + 1.96 * event_df["beta_se"]
    event_df["ci_lower_whale"] = event_df["beta_whale_total"] - 1.96 * event_df["beta_whale_total_se"]
    event_df["ci_upper_whale"] = event_df["beta_whale_total"] + 1.96 * event_df["beta_whale_total_se"]

    return event_df, results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _format_coef(value: float, se: float) -> str:
    stars = ""
    if abs(value / se) >= 2.58:
        stars = "^{***}"
    elif abs(value / se) >= 1.96:
        stars = "^{**}"
    elif abs(value / se) >= 1.65:
        stars = "^{*}"
    return f"{value:0.4f}{stars}\\ ({se:0.4f})"


def write_latex_table(
    baseline_res,
    df: pd.DataFrame,
    window_days: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "MethodologyDiD.tex"

    coef_rows = [
        ("Closure (non-whale)", "is_closed"),
        ("Whale", "whale"),
        ("Closure $\times$ Whale", "closed_whale"),
    ]

    latex_lines = [
        "\\begin{table}[htbp]",
        "    \\centering",
        "    \\caption{Floor Closure Difference-in-Differences}",
        "    \\label{tab:methodology_did}",
        "    \\scriptsize",
        "    \\begin{tabular}{lc}",
        "    \\toprule",
        "        & Quoted Spread \\",
        "    \\midrule",
    ]

    for label, param in coef_rows:
        coef = baseline_res.params[param]
        se = baseline_res.std_errors[param]
        latex_lines.append(f"        {label} & {_format_coef(coef, se)} \\")

    latex_lines.extend(
        [
            "    \\midrule",
            f"        Observations & {len(df):,} \\",
            f"        Exchanges & {df['prtExch'].nunique()} \\",
            f"        Underlyings & {df['okey_tk'].nunique()} \\",
            f"        Window (business days) & $\\pm$ {window_days} \\",
            "    \\bottomrule",
            "    \\end{tabular}",
            "    \\vspace{0.5em}",
            "    {\\footnotesize Clustered standard errors by exchange-date in parentheses.}",
        ]
    )

    latex_lines.append("\\end{table}")

    with table_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(latex_lines))

    return table_path


def write_event_plot(event_df: pd.DataFrame, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    plotting_cfg = config_settings.plotting
    if plotting_cfg.get("tex_settings"):
        plt.rcParams.update(plotting_cfg["tex_settings"])

    colors = plotting_cfg.get("colors", {"background": "white"})
    alpha = plotting_cfg.get("alpha", {"fill": 0.2})
    figsize = (
        plotting_cfg.get("figsize", {}).get("width", 12),
        plotting_cfg.get("figsize", {}).get("height", 6),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(colors.get("background", "white"))

    ax.plot(event_df["k"], event_df["beta"], label="Non-whale", color="#1f77b4")
    ax.fill_between(
        event_df["k"],
        event_df["ci_lower"],
        event_df["ci_upper"],
        color="#1f77b4",
        alpha=alpha.get("fill", 0.2),
    )

    ax.plot(event_df["k"], event_df["beta_whale_total"], label="Whale", color="#d62728")
    ax.fill_between(
        event_df["k"],
        event_df["ci_lower_whale"],
        event_df["ci_upper_whale"],
        color="#d62728",
        alpha=alpha.get("fill", 0.2),
    )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Event Time (Business Days)")
    ax.set_ylabel("Effect on Quoted Spread")
    ax.set_title("Floor Closure Event Study")
    ax.legend()
    ax.grid(True, alpha=alpha.get("grid", 0.3))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "methodology_did_eventstudy.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=plotting_cfg.get("dpi", 300))
    plt.close(fig)

    return fig_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logger = initialize_main()
    logger.info("Running MethodologyDiD with window +/- %d business days", args.window_days)

    table_output_dir = (
        Path(args.table_output)
        if args.table_output
        else PROJECT_ROOT / tables["dask_path"]
    )
    fig_output_dir = (
        Path(args.fig_output)
        if args.fig_output
        else PROJECT_ROOT / config_settings.plotting["output_path"] / "methodology"
    )

    trades = load_trades(args.window_days)
    event_panel = build_event_panel(trades, args.window_days)
    enriched = make_regression_features(event_panel)

    if args.persist_intermediate:
        debug_path = PROJECT_ROOT / "debug" / "methodology_did_sample.parquet"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(debug_path, index=False)
        logger.info("Persisted filtered sample to %s", debug_path)

    baseline_res = run_baseline_regression(enriched)
    logger.info("Baseline regression completed")

    event_df, _ = run_event_study(enriched, args.window_days)
    logger.info("Event study regression completed")

    table_path = write_latex_table(baseline_res, enriched, args.window_days, table_output_dir)
    logger.info("LaTeX table written to %s", table_path)

    fig_path = write_event_plot(event_df, fig_output_dir)
    logger.info("Event-study figure written to %s", fig_path)


if __name__ == "__main__":
    main()


