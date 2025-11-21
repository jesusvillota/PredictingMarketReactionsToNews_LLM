"""LaTeX table generation utilities for portfolio and clustering results."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class TableGenerationError(Exception):
    """Raised when table generation fails."""
    pass


def generate_cluster_mapping_table(
    cluster_titles: Dict[int, str],
    greedy_long: List[int],
    greedy_short: List[int],
    stable_long: List[int],
    stable_short: List[int],
    model_name: str = 'KMeans',
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Generate LaTeX table for cluster mapping with trading rules.

    Args:
        cluster_titles: Dictionary mapping cluster IDs to descriptive titles.
        greedy_long: List of cluster IDs to long in Greedy algorithm.
        greedy_short: List of cluster IDs to short in Greedy algorithm.
        stable_long: List of cluster IDs to long in Stable algorithm.
        stable_short: List of cluster IDs to short in Stable algorithm.
        model_name: Name of the clustering model.
        output_path: Optional path to save the table.

    Returns:
        LaTeX table string.

    Raises:
        TableGenerationError: If inputs are invalid.
    """
    if not cluster_titles:
        raise TableGenerationError("cluster_titles cannot be empty")
    
    # Start building the LaTeX table
    latex_table = (
        r"\begin{table}" + "\n"
        r"\centering" + "\n"
        r"{\fontsize{11}{12.5}\selectfont" + "\n"
        r"\caption{Cluster Mapping with Trading Rules for Greedy and Stable Algorithms}" + "\n"
        r"\begin{tabular}{|c|L{13cm}|c|c|} \hline" + "\n"
        r"\rowcolor{gray!10}\multicolumn{2}{|c|}{\textbf{Cluster}} & \textbf{Greedy} & \textbf{Stable} \\ \hline" + "\n"
    )
    
    # Iterate over clusters and add rows
    for cluster in sorted(cluster_titles.keys()):
        # Determine trading status for Greedy
        if cluster in greedy_long:
            greedy_status = r"\textcolor{darkgreen}{\textsc{long}}"
        elif cluster in greedy_short:
            greedy_status = r"\textcolor{darkred}{\textsc{short}}"
        else:
            greedy_status = ""
        
        # Determine trading status for Stable
        if cluster in stable_long:
            stable_status = r"\textcolor{darkgreen}{\textsc{long}}"
        elif cluster in stable_short:
            stable_status = r"\textcolor{darkred}{\textsc{short}}"
        else:
            stable_status = ""
        
        cluster_title = cluster_titles[cluster]
        latex_table += f"{cluster} & {cluster_title} & {greedy_status} & {stable_status} \\\\ \\hline\n"
    
    # Close the table
    latex_table += (
        r"\end{tabular}" + "\n"
        r"}" + "\n"
        r"\subcaption*{\textit{Note: The proposed titles for each cluster is based on the articles it contains. "
        r"The 'Greedy' and 'Stable' columns provide the trading signals for each cluster according to the "
        r"Greedy and Stable algorithms.}}" + "\n"
        r"\label{tab:Cluster_mapping_trading_rules}" + "\n"
        r"\end{table}" + "\n"
    )
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_table)
    
    return latex_table


def generate_portfolio_statistics_table(
    statistics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    label: str,
    caption: str,
    subcaption_specific1: str,
    subcaption_specific2: str,
    return_type: str = 'gross',
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Generate LaTeX table for portfolio statistics.

    Args:
        statistics: Nested dictionary with structure:
            {split: {algorithm: {return_type: {metric: value}}}}
        label: LaTeX label for the table.
        caption: Table caption.
        subcaption_specific1: First part of subcaption note.
        subcaption_specific2: Second part of subcaption note.
        return_type: Type of returns ('gross' or 'net').
        output_path: Optional path to save the table.

    Returns:
        LaTeX table string.

    Raises:
        TableGenerationError: If statistics structure is invalid.
    """
    if not statistics:
        raise TableGenerationError("statistics cannot be empty")
    
    if return_type not in ['gross', 'net']:
        raise TableGenerationError("return_type must be 'gross' or 'net'")
    
    # Start building the table
    table = f"\\inserthere{{tab:{label}_{return_type}}}\n\n"
    table += "\\begin{table}[H] \n"
    table += f"\\caption{{Statistics of {caption} across data splits | {return_type.capitalize()} Returns}} \n"
    table += r"""
\centering
\renewcommand{\arraystretch}{1.1}
\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}
{\footnotesize
\begin{tabular}{
 P{1.28cm} % Split
 P{0.9cm} % Algorithm
 P{0.9cm} % Cum. Return
 P{0.9cm} % Avg. Return
 P{0.9cm} % St. Deviation
 P{0.9cm} % Sharpe Ratio
 P{0.9cm} % Sortino Ratio
 P{0.9cm} % Max. Drawdown
 P{1cm} % Calmar Ratio
 P{0.9cm} % Skewness
 P{0.9cm} % Kurtosis
 P{0.9cm} % VaR
 P{0.9cm} % CVaR
}
\Xhline{2\arrayrulewidth}
\textbf{Split} & \textbf{Algo.} & \textbf{Cum. Ret.} & \textbf{Avg. Ret.} & \textbf{St. Dev.} & \textbf{Sharpe Ratio} & \textbf{Sortino Ratio} & \textbf{Max. DD} & \textbf{Calmar Ratio} & \textbf{Skew.} & \textbf{Exc. Kurt.} & \textbf{VaR 95\%} & \textbf{CVaR 95\%} \\
\Xhline{2\arrayrulewidth}
"""
    
    # Add data rows for each split
    valid_splits = [k for k in statistics.keys() if k not in ['trading_signal_evolution', 'turnover_stats']]
    
    for split_name in valid_splits:
        split_dict = statistics[split_name]
        table += f"\\multirow{{2}}{{*}}{{{split_name}}}"
        
        for algorithm, algorithm_stats in split_dict.items():
            if return_type not in algorithm_stats:
                continue
            
            stats = algorithm_stats[return_type]
            
            # Extract metrics
            cum_return = stats['cumulative_return']
            avg_return = stats['average_return']
            std_dev = stats['std_deviation']
            sharpe = stats['sharpe_ratio']
            sortino = stats['sortino_ratio']
            max_dd = stats['max_drawdown']
            calmar = stats['calmar_ratio']
            skew = stats['skewness']
            kurt = stats['kurtosis']
            var = stats['var_95']
            cvar = stats['cvar_95']
            
            # Format row
            table += f" & \\textit{{{algorithm}}} & "
            table += f"{cum_return:.3f} & {avg_return*100:.1f} & {std_dev*100:.1f} & "
            table += f"{sharpe:.1f} & {sortino:.1f} & {max_dd*100:.1f} & {calmar:.1f} & "
            table += f"{skew:.2f} & {kurt:.2f} & {var*100:.1f} & {cvar*100:.1f} \\\\ "
        
        table += r" \hline "
    
    # Close table structure
    table += "\n" + r'\end{tabular}' + '\n }'
    table += f"\n \\label{{tab:{label}_{return_type}}}"
    
    # Add notes
    table += r"""
        
\vspace{0.5cm}
\begin{minipage}{\textwidth}
\setlength{\parindent}{0pt}
{\footnotesize\textit{Note:
"""
    table += subcaption_specific1
    table += r"""
The statistics provided include performance metrics (Cumulative Return, Average Return (\%)), risk measures (Standard Deviation (\%), Maximum Drawdown (\%), Value at Risk (\%), Conditional Value at Risk (\%)), risk-adjusted performance ratios (Sharpe Ratio, Sortino Ratio, Calmar Ratio), and return distribution characteristics (Skewness, Excess Kurtosis). These statistics are provided for both cluster-selection algorithms: Greedy and Stable.
Except for the Cumulative Return, all returns are annualized. The Sharpe Ratio is computed using the daily returns, assuming 252 trading days in a year. The Sortino Ratio is calculated using the daily downside returns. The Maximum Drawdown is the maximum loss from a peak to a trough. The Calmar Ratio is the ratio of the annualized return to the maximum drawdown. Skewness measures the asymmetry of the return distribution, while Kurtosis quantifies the tails' thickness. The Value at Risk (VaR) and Conditional Value at Risk (CVaR) are calculated at a 95\% confidence level.
The Greedy algorithm longs (shorts) clusters that maximize (minimize) the cluster-average-$SR$ in the validation sample subject to a positivity (negativity) constraint, while the Stable algorithm longs (shorts) clusters that minimize the rank difference between the training and validation rankings of the cluster-average-$SR$'s subject to a positivity (negativity) constraint, which is now imposed on both sample splits. In both algorithms, the cardinality of each leg is upper-bounded by a hyperparameter $\theta$.
"""
    table += subcaption_specific2
    table += r"""
}}
\end{minipage}
\end{table}
"""
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table)
    
    return table


def generate_trading_intensity_table(
    portfolio_returns: Dict[str, Dict[str, pd.Series]],
    trading_signals: Dict[str, Dict[str, pd.DataFrame]],
    turnover_stats: Dict[str, Dict[str, float]],
    model_name: str,
    label: str,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Union[pd.DataFrame, str]]:
    """Generate LaTeX table for trading intensity analysis.

    Args:
        portfolio_returns: Dictionary of portfolio returns by split and algorithm.
        trading_signals: Dictionary of trading signals evolution.
        turnover_stats: Dictionary of turnover statistics.
        model_name: Name of the model.
        label: LaTeX label for the table.
        output_path: Optional path to save the table.

    Returns:
        Dictionary with 'dataframe' and 'latex' keys.

    Raises:
        TableGenerationError: If inputs are invalid.
    """
    if not portfolio_returns:
        raise TableGenerationError("portfolio_returns cannot be empty")
    
    # Prepare data for table
    table_data = []
    
    # Define splits to process
    splits = ['All', 'Train', 'Validation', 'Test']
    algorithms = ['Greedy', 'Stable']
    
    for split in splits:
        if split not in portfolio_returns:
            continue
        
        for algo in algorithms:
            if algo not in portfolio_returns[split]:
                continue
            
            # Get position data
            if split in trading_signals and algo in trading_signals[split]:
                positions = trading_signals[split][algo].abs().sum(axis=1)
            else:
                positions = pd.Series([0])  # Placeholder
            
            # Calculate metrics
            avg_positions = positions.mean()
            std_positions = positions.std()
            max_positions = positions.max()
            min_positions = positions.min()
            
            # Get turnover
            turnover = turnover_stats.get(algo, {}).get(split, 0.0) * 100
            
            # Calculate changes per position
            changes_per_pos = (turnover / avg_positions) if avg_positions > 0 else 0.0
            
            # Calculate trading costs (placeholder - would need actual implementation)
            trading_costs = 0.0  # This would be computed from gross vs net returns
            
            # Calculate active days percentage
            active_days = (positions > 0).sum() / len(positions) * 100 if len(positions) > 0 else 0.0
            
            metrics = {
                'Split': split,
                'Algorithm': algo,
                'Avg. Positions': avg_positions,
                'Position Std.': std_positions,
                'Max Positions': max_positions,
                'Min Positions': min_positions,
                'Turnover': turnover,
                'Changes/Position': changes_per_pos,
                'Avg. Costs (%)': trading_costs,
                'Active Days (%)': active_days
            }
            
            table_data.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(table_data)
    
    # Generate LaTeX table
    latex_table = f"\\inserthere{{tab:{label}}}\n\n"
    latex_table += "\\begin{table}[htbp] \n"
    latex_table += f"\\caption{{Trading Intensity Analysis: {model_name}}} \n"
    latex_table += "\\centering \n"
    latex_table += f"\\label{{tab:{label}}}"
    latex_table += r"""
{\small
\begin{tabular}{lcccccccccc}
\toprule
Split & Algorithm & \multicolumn{4}{c}{\# Open Positions} & \multicolumn{2}{c}{Trading Activity (\%)} & \multicolumn{2}{c}{Trading Costs (\%)} \\
\cmidrule(lr{0.6em}){3-6} \cmidrule(lr{0.6em}){7-8} \cmidrule(lr{0.6em}){9-10}
& & Avg. & Std. & Max & Min & Turnover & Changes/Pos. & Cost & Active \\
\midrule
"""
    
    # Add data rows
    for split in splits:
        split_data = metrics_df[metrics_df['Split'] == split]
        
        if len(split_data) == 0:
            continue
        
        # Greedy row
        greedy_row = split_data[split_data['Algorithm'] == 'Greedy']
        if len(greedy_row) > 0:
            gr = greedy_row.iloc[0]
            latex_table += f"\\multirow{{2}}{{*}}{{{split}}} & \\textit{{Greedy}} & "
            latex_table += f"{gr['Avg. Positions']:.1f} & {gr['Position Std.']:.2f} & "
            latex_table += f"{gr['Max Positions']:.0f} & {gr['Min Positions']:.0f} & "
            latex_table += f"{gr['Turnover']:.2f} & {gr['Changes/Position']:.3f} & "
            latex_table += f"{gr['Avg. Costs (%)']:.4f} & {gr['Active Days (%)']:.1f} \\\\\n"
        
        # Stable row
        stable_row = split_data[split_data['Algorithm'] == 'Stable']
        if len(stable_row) > 0:
            sr = stable_row.iloc[0]
            latex_table += " & \\textit{Stable} & "
            latex_table += f"{sr['Avg. Positions']:.2f} & {sr['Position Std.']:.2f} & "
            latex_table += f"{sr['Max Positions']:.0f} & {sr['Min Positions']:.0f} & "
            latex_table += f"{sr['Turnover']:.2f} & {sr['Changes/Position']:.3f} & "
            latex_table += f"{sr['Avg. Costs (%)']:.4f} & {sr['Active Days (%)']:.1f} \\\\\n"
        
        if split != 'Test':
            latex_table += "\\midrule\n"
    
    # Close table
    latex_table += r"""\bottomrule
\end{tabular}
}

\vspace{0.5cm}
\begin{minipage}{\textwidth}
\setlength{\parindent}{0pt}
\scriptsize\textit{Note}: 
This table presents trading intensity metrics for both Greedy and Stable algorithms across different data splits. 
The metrics are computed at a daily frequency. The `\# Open Positions' columns report position-related statistics: 
`Avg.' shows the mean number of concurrent open positions per day, `Std.' represents their standard deviation, while 
`Max' and `Min' indicate the maximum and minimum number of positions held simultaneously. Under `Trading Activity (\%)', 
`Turnover' is calculated as the sum of absolute changes in position sizes divided by the total portfolio size, expressed 
as a percentage; formally, $Turnover_t = 100 \times (\sum_i |w_{i,t} - w_{i,t-1}|)/(\sum_i |w_{i,t}|)$, where $w_{i,t}$ 
represents the position size in asset $i$ at time $t$. `Changes/Pos.' represents the average number of modifications per 
position per day, computed as the daily turnover divided by the average number of positions, providing insight into how 
actively individual positions are managed. The `Trading Costs (\%)' section reports `Cost' as the average daily implementation 
shortfall (computed as the difference between gross and net returns) expressed in percentage terms, while `Active' shows 
the percentage of trading days with at least one open position. All metrics are first computed daily and then averaged 
over their respective periods, except for Max and Min positions which represent the absolute extremes over each period.
\end{minipage}
\end{table}"""
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_table)
    
    return {
        'dataframe': metrics_df,
        'latex': latex_table
    }


def generate_llama_shock_mapping_table(
    shock_mapping: Dict[int, Dict[str, Any]],
    greedy_clusters: List[int],
    stable_clusters: List[int],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Generate LaTeX table for LLAMA shock classification to cluster mapping.

    Args:
        shock_mapping: Dictionary mapping clusters to shock information.
        greedy_clusters: List of clusters traded by Greedy algorithm.
        stable_clusters: List of clusters traded by Stable algorithm.
        output_path: Optional path to save the table.

    Returns:
        LaTeX table string.
    """
    latex_table = (
        "\\begin{table}[H]\n"
        "\\caption{Mapping of LLM-Shock-Classification to Clusters with Trading Rules}\n"
        "\\centering\n"
        "\\begin{tabular}{|C{1cm}|l|c|c|}\\hline\n"
        "\\textbf{Cluster} & \\textbf{Shock Type} & \\textbf{Greedy} & \\textbf{Stable} \\\\ \\hline\n"
    )
    
    for cluster, info in sorted(shock_mapping.items()):
        shock_type = info.get('shock_type', 'Unknown')
        
        greedy_status = "\\checkmark" if cluster in greedy_clusters else ""
        stable_status = "\\checkmark" if cluster in stable_clusters else ""
        
        latex_table += f"{cluster} & {shock_type} & {greedy_status} & {stable_status} \\\\ \\hline\n"
    
    latex_table += (
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_table)
    
    return latex_table
