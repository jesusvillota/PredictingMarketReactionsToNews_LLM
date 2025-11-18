"""Floor-closure DiD regressions with an aggregation-first workflow.

This refactored script aggregates trades to the exchange × symbol × day × whale
level before estimating the staggered difference-in-differences and event-study
specifications described in
``TeX/article/txt_3_methodology.tex``.  Aggregation keeps the design matrices
compact enough to run on a single workstation (the earlier trade-level
approach attempted to materialise ~40M rows, exhausting memory).

Usage examples
--------------

Default +/-30 business-day window:

    uv run src/tabling_dask/DiD.py

Tighter +/-7 business-day window:

    uv run src/tabling_dask/DiD.py --window-days 7

Override output locations:

    uv run src/tabling_dask/DiD.py --window-days 45 --table-output TeX/tables/dask/test --fig-output TeX/figures
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import dask.dataframe as dd

from src.tabling_dask.common import build_parquet_filters

try:
    from linearmodels.iv.model import AbsorbingLS  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - fallback import
    try:
        from linearmodels.iv.absorbing import AbsorbingLS  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - final fallback
        raise ImportError(
            "linearmodels >= 5.0 with AbsorbingLS is required for MethodologyDiD"
        ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config_settings, initialize_main, DaskManager, get_logger
from src.config.config_settings import PROCESSED_PATH, tables
from src.tabling_dask.common import build_parquet_filters


@dataclass(frozen=True)
class ClosurePeriod:
    exchange: str
    start: date
    end: date
    label: str


CLOSURE_PERIODS: Sequence[ClosurePeriod] = (
    ClosurePeriod("CBOE", date(2020, 3, 16), date(2020, 6, 15), "covid_2020"),
    ClosurePeriod("PHLX", date(2020, 3, 17), date(2020, 6, 3), "covid_2020"),
    ClosurePeriod("BOX", date(2020, 3, 20), date(2020, 5, 4), "covid_2020"),
    ClosurePeriod("NYSE", date(2020, 3, 23), date(2020, 5, 26), "covid_2020"), 
)

FLOOR_EXCHANGES = sorted({period.exchange for period in CLOSURE_PERIODS})

TIME_TO_EXPIRY_ORDER = {
    "lt_1w": 1,
    "1w_to_2w": 2,
    "2w_to_4w": 3,
    "1m_to_3m": 4,
    "3m_to_12m": 5,
    "gt_1y": 6,
}

DEFAULT_WINDOW_DAYS = 30


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
        help="Persist the aggregated regression sample to disk (debug aid).",
    )
    return parser.parse_args()


def _window_bounds(window_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_start = min(period.start for period in CLOSURE_PERIODS)
    max_end = max(period.end for period in CLOSURE_PERIODS)
    pre_span = pd.tseries.offsets.BusinessDay(window_days)
    post_span = pd.tseries.offsets.BusinessDay(window_days)
    window_start = (pd.Timestamp(min_start) - pre_span).normalize()
    window_end = (pd.Timestamp(max_end) + post_span).normalize()
    return window_start, window_end


def load_trades(window_days: int) -> pd.DataFrame:
    logger = get_logger(__name__)
    required_columns = [
        "timestamp_ny",
        "prtExch",
        "prtSize_agg",
        "quoted_spread",
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
        "Date window bounds (business days): %s to %s",
        start_bound.date(),
        end_bound.date(),
    )

    filters = build_parquet_filters(
        ticker_type="equity",
        strategy_type="simple",
        start=start_bound,
        end=end_bound,
    )
    logger.info(
        "Loading processed trades for floor exchanges and aggregating before computation..."
    )
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

        ddf = ddf[ddf["prtExch"].isin(FLOOR_EXCHANGES)]
        ddf["timestamp_ny"] = dd.to_datetime(
            ddf["timestamp_ny"], utc=False, errors="coerce"
        )
        ddf = ddf.dropna(subset=["timestamp_ny", "quoted_spread", "prtSize_agg"])

        ddf = ddf.assign(
            trade_date=ddf["timestamp_ny"].dt.tz_localize(None).dt.normalize(),
        )
        ddf = ddf[
            (ddf["trade_date"] >= start_bound)
            & (ddf["trade_date"] <= end_bound)
        ]

        ddf = ddf.assign(
            whale=(ddf["prtSize_agg"] >= 200).astype("int8"),
            abs_delta=ddf["prtDe"].abs(),
            is_buy=(ddf["buy_sell_class"] == "buy").astype("float32"),
            is_sell=(ddf["buy_sell_class"] == "sell").astype("float32"),
        )

        ddf["okey_cp"] = ddf["okey_cp"].fillna("")
        ddf = ddf.assign(
            is_put=(ddf["okey_cp"].str.lower() == "put").astype("float32"),
            moneyness=ddf["moneyness"].replace([np.inf, -np.inf], np.nan),
            prtIv=ddf["prtIv"].replace([np.inf, -np.inf], np.nan),
            tte_bucket=ddf["time_to_expiry"].replace(TIME_TO_EXPIRY_ORDER).astype(
                "float32"
            ),
        )

        group_cols = ["prtExch", "okey_tk", "trade_date", "whale"]
        grouped = ddf.groupby(group_cols)

        aggregated = grouped.agg(
            quoted_spread=("quoted_spread", "mean"),
            abs_delta=("abs_delta", "mean"),
            moneyness=("moneyness", "mean"),
            prtIv=("prtIv", "mean"),
            tte_bucket=("tte_bucket", "mean"),
            is_put=("is_put", "mean"),
            is_buy=("is_buy", "mean"),
            is_sell=("is_sell", "mean"),
            total_contracts=("prtSize_agg", "sum"),
            trade_count=("quoted_spread", "count"),
        )

        aggregated = aggregated.reset_index()
        pdf = aggregated.compute()

    pdf["trade_date"] = pd.to_datetime(pdf["trade_date"])
    pdf["trade_month"] = pdf["trade_date"].dt.to_period("M").astype(str)
    pdf["whale"] = pdf["whale"].astype(int)
    pdf["trade_count"] = pdf["trade_count"].astype(np.int32)
    pdf["total_contracts"] = pdf["total_contracts"].astype(np.float32)

    numeric_cols = [
        "quoted_spread",
        "abs_delta",
        "moneyness",
        "prtIv",
        "tte_bucket",
        "is_put",
        "is_buy",
        "is_sell",
    ]
    for col in numeric_cols:
        pdf[col] = pdf[col].astype(np.float32)

    pdf["weight"] = pdf["trade_count"].clip(lower=1).astype(np.float64)

    total_trades = int(pdf["trade_count"].sum())
    logger.info(
        "Aggregated to %s grouped observations covering %s trades.",
        f"{len(pdf):,}",
        f"{total_trades:,}",
    )
    return pdf


def attach_event_metadata(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["is_closed"] = 0
    enriched["event_time"] = np.nan
    enriched["event_id"] = pd.Series(index=enriched.index, dtype="object")
    enriched["closure_start"] = pd.NaT
    enriched["closure_end"] = pd.NaT

    for idx, period in enumerate(CLOSURE_PERIODS, start=1):
        mask = enriched["prtExch"] == period.exchange
        if not mask.any():
            continue

        trade_dates = enriched.loc[mask, "trade_date"].values.astype("datetime64[D]")
        start_np = np.datetime64(period.start)
        end_np = np.datetime64(period.end)

        is_closed = (trade_dates >= start_np) & (trade_dates <= end_np)
        event_time = np.where(
            trade_dates >= start_np,
            np.busday_count(start_np, trade_dates),
            -np.busday_count(trade_dates, start_np),
        )

        enriched.loc[mask, "is_closed"] = is_closed.astype(int)
        enriched.loc[mask, "event_time"] = event_time.astype(np.int32)
        enriched.loc[mask, "event_id"] = f"{period.exchange}_{period.label}_{idx}"
        enriched.loc[mask, "closure_start"] = pd.Timestamp(period.start)
        enriched.loc[mask, "closure_end"] = pd.Timestamp(period.end)

    if enriched["event_time"].isna().any():
        raise ValueError(
            "Missing event-time assignments—verify closure periods or date filters."
        )

    enriched["event_time"] = enriched["event_time"].astype(np.int32)
    enriched["is_closed"] = enriched["is_closed"].astype(int)
    return enriched


def prepare_regression_sample(window_days: int) -> pd.DataFrame:
    aggregated = load_trades(window_days)
    if aggregated.empty:
        raise ValueError(
            "No trades remained after filtering. Try widening the window or verifying data availability."
        )
    enriched = attach_event_metadata(aggregated)

    enriched["closed_whale"] = enriched["is_closed"] * enriched["whale"]
    enriched["trade_month"] = enriched["trade_month"].astype(str)
    enriched["trade_date"] = pd.to_datetime(enriched["trade_date"])
    enriched["weight"] = enriched["weight"].astype(np.float64)

    filler_cols = ["quoted_spread", "abs_delta", "moneyness", "prtIv", "tte_bucket"]
    for col in filler_cols:
        if enriched[col].isna().any():
            enriched[col] = enriched[col].fillna(enriched[col].median())

    indicator_cols = ["is_put", "is_buy", "is_sell"]
    for col in indicator_cols:
        if enriched[col].isna().any():
            enriched[col] = enriched[col].fillna(0.0)

    return enriched


def run_baseline_regression(df: pd.DataFrame) -> AbsorbingLS:
    y = df["quoted_spread"].astype(np.float64)
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
    ].astype(np.float64)

    absorb = df[["prtExch", "trade_month", "okey_tk"]].copy()
    for col in absorb.columns:
        absorb[col] = absorb[col].astype("category")

    clusters = df["prtExch"].astype(str) + "_" + df["trade_date"].astype(str)
    weights = df["weight"].to_numpy()

    model = AbsorbingLS(y, X, absorb=absorb, weights=weights)
    return model.fit(cov_type="clustered", clusters=clusters)


def run_event_study(
    df: pd.DataFrame, window_days: int
) -> tuple[pd.DataFrame, AbsorbingLS]:
    max_closure_span = max(
        np.busday_count(np.datetime64(period.start), np.datetime64(period.end)) + 1
        for period in CLOSURE_PERIODS
    )
    max_event_horizon = max_closure_span + window_days

    filtered = df.loc[
        (df["event_time"] >= -window_days) & (df["event_time"] <= max_event_horizon)
    ].copy()

    negative_times = sorted(filtered.loc[filtered["event_time"] < 0, "event_time"].unique())
    base_period = negative_times[-1] if negative_times else 0

    event_dummies = pd.get_dummies(filtered["event_time"], prefix="event", dtype=np.float64)
    base_col = f"event_{base_period}"
    if base_col in event_dummies:
        event_dummies.drop(columns=[base_col], inplace=True)

    nonzero_cols = [col for col in event_dummies.columns if event_dummies[col].sum() > 0]
    event_dummies = event_dummies[nonzero_cols]
    if event_dummies.empty:
        raise ValueError(
            "Event-study design matrix is empty; widen the window or inspect data availability."
        )

    interaction_dummies = event_dummies.multiply(filtered["whale"].values, axis=0)
    interaction_dummies.columns = [f"{col}_whale" for col in event_dummies.columns]

    control_cols = [
        "whale",
        "abs_delta",
        "moneyness",
        "prtIv",
        "tte_bucket",
        "is_put",
        "is_buy",
        "is_sell",
    ]
    X = pd.concat(
        [
            event_dummies,
            interaction_dummies,
            filtered[control_cols].astype(np.float64),
        ],
        axis=1,
    )

    absorb = filtered[["prtExch", "trade_month", "okey_tk"]].copy()
    for col in absorb.columns:
        absorb[col] = absorb[col].astype("category")

    clusters = filtered["prtExch"].astype(str) + "_" + filtered["trade_date"].astype(str)
    weights = filtered["weight"].to_numpy()

    model = AbsorbingLS(
        filtered["quoted_spread"].astype(np.float64),
        X,
        absorb=absorb,
        weights=weights,
    )
    results = model.fit(cov_type="clustered", clusters=clusters)

    event_summary: list[dict[str, float]] = []
    for col in event_dummies.columns:
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
        ("Closure $\\times$ Whale", "closed_whale"),
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
            f"        Grouped observations & {len(df):,} \\",
            f"        Trades (sum of weights) & {int(df['trade_count'].sum()):,} \\",
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

    try:
        sample = prepare_regression_sample(args.window_days)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    total_trades = int(sample["trade_count"].sum())
    logger.info(
        "Prepared regression sample with %s grouped rows spanning %s trades.",
        f"{len(sample):,}",
        f"{total_trades:,}",
    )

    if args.persist_intermediate:
        debug_path = PROJECT_ROOT / "debug" / "methodology_did_sample.parquet"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(debug_path, index=False)
        logger.info("Persisted aggregated sample to %s", debug_path)

    baseline_res = run_baseline_regression(sample)
    logger.info("Baseline regression completed")

    event_df, _ = run_event_study(sample, args.window_days)
    logger.info("Event study regression completed")

    table_path = write_latex_table(baseline_res, sample, args.window_days, table_output_dir)
    logger.info("LaTeX table written to %s", table_path)

    fig_path = write_event_plot(event_df, fig_output_dir)
    logger.info("Event-study figure written to %s", fig_path)


if __name__ == "__main__":
    main()


