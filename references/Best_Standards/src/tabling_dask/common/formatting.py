from __future__ import annotations

from typing import Iterable, Optional
import pandas as pd


def fmt_cell(count: int, total: int, percentage_only: bool = False) -> str:
    r"""Format count with percentage for LaTeX table cell.

    When percentage_only=True → return only percentage with 3 decimals (e.g., "12.345\%").
    Otherwise → "count (percentage%)" with 1 decimal (e.g., "1,234 (12.3\%)").
    """
    if total == 0:
        return "0.000\\%" if percentage_only else "0 (0.0\\%)"
    pct = (count / total * 100.0)
    if percentage_only:
        return f"{pct:.3f}\\%"
    return f"{int(count):,} ({pct:.1f}\\%)"


def format_count(count: int | float) -> str:
    r"""Format count without percentage for LaTeX table cell.
    
    Returns formatted count with thousands separators (e.g., "1,234").
    Handles zero and NaN values.
    """
    if count == 0 or pd.isna(count):
        return "0"
    return f"{int(count):,}"


def format_percentage(count: int | float, total: int | float) -> str:
    r"""Format percentage only for LaTeX table cell.
    
    Returns percentage in parentheses with 1 decimal (e.g., "(12.3\%)").
    Handles zero and NaN values.
    """
    if count == 0 or total == 0 or pd.isna(count):
        return "(0.0\\%)"
    percentage = (count / total) * 100
    return f"({percentage:.1f}\\%)"


def append_total_row(
    table_lines: list[str],
    totals: Iterable[int],
    leading_cells: Optional[Iterable[str]] = None,
) -> None:
    """Append a LaTeX total row to an existing list of table lines.

    Parameters
    - table_lines: mutable list of LaTeX lines to which the total row is appended
    - totals: list/iterable of integer totals matching the table's numeric columns
    - leading_cells: optional cells before the numeric totals (e.g., ["\\textbf{Total}", "", ""])
    """
    leading = [] if leading_cells is None else list(leading_cells)
    total_cells = [f"{int(v):,}" for v in totals]
    row = " & ".join([*leading, *total_cells])
    table_lines.append(r"\midrule")
    table_lines.append(row + r" \\")


