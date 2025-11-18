# uv run src/tabling_dask/Table5.py
#---------------------------------------------------------------
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
#---------------------------------------------------------------

import dask.dataframe as dd
import numpy as np
from typing import Dict, Callable, Any

from src.config import initialize_main, DaskManager
from src.config.config_settings import tables

from src.config.config_settings import COMPLEX_TRADES_PATH


OUTPUT_DIR = PROJECT_ROOT / tables["dask_path"]
OUTPUT_PATH = OUTPUT_DIR / "Table_complex_strategies.tex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_size_bins(ddf: dd.DataFrame) -> dd.DataFrame:
    """Create a categorical size bin column matching table columns."""
    s = ddf["prtSize_agg"]
    conditions = [
        (s >= 1) & (s <= 10),
        (s >= 11) & (s <= 200),
        (s > 200),
    ]
    choices = ["1_10", "11_200", "over_200"]
    size_bin = dd.map_partitions(
        lambda part: np.select(
            [cond.loc[part.index] for cond in conditions],
            choices,
            default="all",
        ),
        meta=("size_bin", "object"),
    )
    ddf = ddf.assign(size_bin=size_bin)
    return ddf


def fmt_cell(count: int, total: int) -> str:
    pct = 0.0 if not total else (count / total * 100.0)
    return f"{int(count):,} ({pct:.1f}\\%)"


def main() -> None:
    logger = initialize_main()
    logger.info("Starting Table5.py script with Dask.")

    with DaskManager() as _dm:
        logger.info("Loading complex_trades parquet with Dask...")
        # Read all per-day complex outputs
        ddf = dd.read_parquet(
            path=str(COMPLEX_TRADES_PATH / "**" / "complex_trades.parquet"),
            engine="pyarrow",
            columns=[
                "prtSize_agg",
                "n_legs",
                "sign",
                "flag",
                "strategy_name",
            ],
            split_row_groups="infer",
        )
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")

        ddf = ddf.dropna(subset=["prtSize_agg", "n_legs", "strategy_name"])  # sign/flag can be NA
        ddf = build_size_bins(ddf)

        # Pre-compute totals per column (size bin)
        logger.info("Computing column totals per size bin...")
        totals = ddf.groupby("size_bin", observed=True).size().compute()
        # Ensure 'all' column total equals full count
        totals["all"] = int(ddf.shape[0].compute())

        # Known strategy sets by legs
        def is_single(df):
            return (df["n_legs"] == 1) & (df["strategy_name"] == "Single")

        def is_two_callput(df):
            return (df["n_legs"] == 2) & (df["strategy_name"].isin(["Spread", "Calendar", "Diagonal"])) & (df["flag"].isin(["Call", "Put"]))

        def is_two_mixed_pair(df):
            return (df["n_legs"] == 2) & (df["strategy_name"].isin(["Straddle", "Strangle"])) & (df["flag"] == "Mixed")

        def is_butterfly(df):
            return (df["n_legs"] == 3) & (df["strategy_name"] == "Butterfly") & (df["flag"].isin(["Call", "Put"]))

        def is_condor(df):
            return (df["n_legs"] == 4) & (df["strategy_name"] == "Condor") & (df["flag"].isin(["Call", "Put"]))

        def is_iron_condor(df):
            return (df["n_legs"] == 4) & (df["strategy_name"] == "IronCondor") & (df["flag"] == "Mixed")

        # Row specification in order: (legs_label, strategy, sign, flag, predicate)
        Row = tuple[str, str, str, str, Callable[[dd.DataFrame], Any]]

        rows: list[Row] = []

        # 1 Leg
        for sign in ["Long", "Short", "Midpoint"]:
            for flag in ["Call", "Put"]:
                rows.append(("1 Leg", "Single", sign, flag, lambda df, s=sign, f=flag: is_single(df) & (df["sign"] == s) & (df["flag"] == f)))

        # 2 Legs - same-flag families
        for strategy in ["Spread", "Calendar", "Diagonal"]:
            for sign in ["Long", "Short", "Midpoint"]:
                for flag in ["Call", "Put"]:
                    rows.append(("2 Legs", strategy, sign, flag, lambda df, st=strategy, s=sign, f=flag: (df["n_legs"] == 2) & (df["strategy_name"] == st) & (df["sign"] == s) & (df["flag"] == f)))
        # 2 Legs - mixed pairs (Straddle/Strangle)
        for strategy in ["Straddle", "Strangle"]:
            for sign in ["Long", "Short", "Midpoint"]:
                rows.append(("2 Legs", strategy, sign, "", lambda df, st=strategy, s=sign: is_two_mixed_pair(df) & (df["strategy_name"] == st) & (df["sign"] == s)))

        # 3 Legs - Butterfly
        for sign in ["Long", "Short", "Midpoint"]:
            for flag in ["Call", "Put"]:
                rows.append(("3 Legs", "Butterfly", sign, flag, lambda df, s=sign, f=flag: is_butterfly(df) & (df["sign"] == s) & (df["flag"] == f)))

        # 4 Legs - Condor and IronCondor
        for sign in ["Long", "Short", "Midpoint"]:
            for flag in ["Call", "Put"]:
                rows.append(("4 Legs", "Condor", sign, flag, lambda df, s=sign, f=flag: is_condor(df) & (df["sign"] == s) & (df["flag"] == f)))
        for sign in ["Long", "Short", "Midpoint"]:
            rows.append(("4 Legs", "IronCondor", sign, "", lambda df, s=sign: is_iron_condor(df) & (df["sign"] == s)))

        # OTHER bucket: include Undetermined sign OR unknown strategies
        known_strategies = {
            (1, "Single"),
            (2, "Spread"), (2, "Calendar"), (2, "Diagonal"), (2, "Straddle"), (2, "Strangle"),
            (3, "Butterfly"),
            (4, "Condor"), (4, "IronCondor"),
        }

        def is_other(df):
            known = df[["n_legs", "strategy_name"]].map_partitions(
                lambda part: part.apply(lambda r: (int(r["n_legs"]), r["strategy_name"]) in known_strategies, axis=1),
                meta=(None, "bool"),
            )
            undet = (df["sign"].fillna("Undetermined") == "Undetermined")
            return (~known) | undet

        rows.append(("", "Other", "", "", is_other))

        # Compute counts per row per bin lazily
        logger.info("Preparing lazy computations for all table cells...")
        size_bins = ["all", "1_10", "11_200", "over_200"]

        lazy_counts: Dict[tuple, Any] = {}
        for (_, _, _, _, predicate) in rows:
            mask = predicate(ddf)
            for b in size_bins:
                lazy_counts[(predicate, b)] = ddf[(ddf["size_bin"] == b) & mask].shape[0]

        # Compute all counts in one go
        results = dd.compute(*lazy_counts.values())
        key_list = list(lazy_counts.keys())
        counts_map: Dict[tuple, int] = {key_list[i]: int(results[i]) for i in range(len(key_list))}

        # Build LaTeX table
        logger.info("Building LaTeX table content...")
        header = r"""\begin{table}[htbp]
    \centering
    \caption{Distribution of Complex Option Strategies by Trade Size Category}
    \subcaption*{
    {\scriptsize
    \par}
    \vspace{1em}
    }
    \label{tab:complex_strategies_by_size}
    \scriptsize
    \begin{tabular}{llllcccc}
    \toprule
        \textbf{Legs}
        & \textbf{Strategy}
        & \textbf{Sign}
        & \textbf{Flag}
        & \textbf{All}
        & \textbf{1--10}
        & \textbf{11--200}
        & \textbf{>200} \\
    \midrule
"""

        # Helper to render a row
        def render_row(legs: str, strategy: str, sign: str, flag: str, pred: Callable[[dd.DataFrame], Any]) -> str:
            cells = []
            for b in size_bins:
                total = int(totals.get(b, 0))
                count = counts_map.get((pred, b), 0)
                cells.append(fmt_cell(count, total))
            leg_txt = legs if legs else ""
            flag_txt = flag if flag else ""
            return f"    {leg_txt} & {strategy} & {sign} & {flag_txt} & " + " & ".join(cells) + r" \\" + "\n"

        content = header

        # Emit in block sections with midrules to mirror template
        def emit_block(start_idx: int, end_idx: int, add_midrule: bool = True):
            nonlocal content
            for i in range(start_idx, end_idx):
                legs, strategy, sign, flag, pred = rows[i]
                content += render_row(legs, strategy, sign, flag, pred)
            if add_midrule:
                content += "    \\midrule\n"

        idx = 0
        # 1-Leg: 3 signs x 2 flags = 6 rows
        emit_block(idx, idx + 6)
        idx += 6

        # 2-Leg (Spread, Calendar, Diagonal): 3 strategies x 3 signs x 2 flags = 18 rows
        emit_block(idx, idx + 18)
        idx += 18

        # 2-Leg (Straddle, Strangle): 2 strategies x 3 signs = 6 rows
        emit_block(idx, idx + 6)
        idx += 6

        # 3-Leg (Butterfly): 3 signs x 2 flags = 6 rows
        emit_block(idx, idx + 6)
        idx += 6

        # 4-Leg (Condor): 3 signs x 2 flags = 6 rows
        emit_block(idx, idx + 6)
        idx += 6

        # 4-Leg (IronCondor): 3 signs = 3 rows
        emit_block(idx, idx + 3)
        idx += 3

        # Other: 1 row
        emit_block(idx, idx + 1, add_midrule=False)

        content += r"""    \midrule
    \textbf{Total} &  &  &  & {0} & {1} & {2} & {3} \\
    \bottomrule
    \end{tabular}
\end{table}
""".format(
            int(totals.get("all", 0)),
            int(totals.get("1_10", 0)),
            int(totals.get("11_200", 0)),
            int(totals.get("over_200", 0)),
        )

        logger.info(f"Writing LaTeX table to {OUTPUT_PATH} ...")
        with open(OUTPUT_PATH, "w") as f:
            f.write(content)

        logger.info("LaTeX table successfully written. Table5.py completed.")


if __name__ == "__main__":
    main()


