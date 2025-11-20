"""CLI commands for clustering analysis."""

import logging
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from news_market_analysis.config import get_settings, get_path_manager
from news_market_analysis.data import load_embeddings
from news_market_analysis.models import NewsClusteringModel, find_optimal_k
from news_market_analysis.analysis import split_data
from news_market_analysis.visualization import (
    configure_matplotlib_style,
    plot_cluster_distribution,
    plot_cluster_distributions_by_split,
)

logger = logging.getLogger(__name__)


@click.command('kmeans-clustering')
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to configuration file',
)
@click.option(
    '--input',
    type=click.Path(exists=True, path_type=Path),
    help='Input path for embeddings CSV',
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for clustering results',
)
@click.option(
    '--k-min',
    type=int,
    default=2,
    help='Minimum number of clusters to test',
)
@click.option(
    '--k-max',
    type=int,
    default=10,
    help='Maximum number of clusters to test',
)
@click.option(
    '--k',
    type=int,
    help='Fixed number of clusters (skips optimal k search)',
)
@click.option(
    '--method',
    type=click.Choice(['silhouette', 'davies_bouldin', 'inertia']),
    default='silhouette',
    help='Method for finding optimal k',
)
@click.option(
    '--split-ratio',
    type=float,
    default=0.6,
    help='Train/validation split ratio',
)
@click.option(
    '--test-ratio',
    type=float,
    default=0.2,
    help='Test set ratio',
)
@click.option(
    '--random-seed',
    type=int,
    default=42,
    help='Random seed for reproducibility',
)
@click.pass_context
def kmeans_clustering(
    ctx: click.Context,
    config: Optional[Path],
    input: Optional[Path],
    output_dir: Optional[Path],
    k_min: int,
    k_max: int,
    k: Optional[int],
    method: str,
    split_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> None:
    """Perform KMeans clustering analysis.
    
    This command loads article embeddings, performs KMeans clustering,
    and generates cluster assignments and visualizations. It can either
    use a fixed number of clusters or search for the optimal k.
    
    Equivalent to script: 4_kmeans_clustering.py
    """
    logger.info("Starting KMeans clustering analysis")
    
    # Get configuration
    config_path = config or ctx.obj.get('config_path')
    settings = get_settings(config_path)
    path_manager = get_path_manager()
    
    # Determine paths
    input_path = input or (path_manager.processed_data / 'articles_with_embeddings.csv')
    output_path = output_dir or path_manager.output_kmeans
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading embeddings from: {input_path}")
    df = load_embeddings(input_path)
    
    logger.info(f"Loaded {len(df)} articles with embeddings")
    
    # Split data into train/validation/test
    click.echo("Splitting data into train/validation/test sets...")
    splits = split_data(
        df,
        train_ratio=split_ratio,
        test_ratio=test_ratio,
        val_ratio=1 - split_ratio - test_ratio,
        random_state=random_seed,
        sequential=False,
        verbose=True,
    )
    
    df_train = splits['train']
    df_val = splits['validation']
    df_test = splits['test']
    
    click.echo(f"  Train: {len(df_train)} articles")
    click.echo(f"  Validation: {len(df_val)} articles")
    click.echo(f"  Test: {len(df_test)} articles")
    
    # Extract embeddings
    import ast
    import numpy as np
    
    train_embeddings = np.array([ast.literal_eval(e) for e in df_train['embedding']])
    
    # Find or use optimal k
    if k is None:
        click.echo(f"\nSearching for optimal k (range: {k_min}-{k_max}, method: {method})...")
        optimal_k, scores = find_optimal_k(
            train_embeddings,
            k_range=range(k_min, k_max + 1),
            method=method,
            random_state=random_seed,
        )
        click.echo(f"  Optimal k: {optimal_k}")
        
        # Save scores
        scores_df = pd.DataFrame({
            'k': list(scores.keys()),
            'score': list(scores.values()),
        })
        scores_path = output_path / f'optimal_k_scores_{method}.csv'
        scores_df.to_csv(scores_path, index=False)
        logger.info(f"Saved optimal k scores to: {scores_path}")
    else:
        optimal_k = k
        click.echo(f"\nUsing fixed k: {optimal_k}")
    
    # Train clustering model
    click.echo(f"\nTraining KMeans model with k={optimal_k}...")
    model = NewsClusteringModel(n_clusters=optimal_k, random_state=random_seed)
    model.fit(train_embeddings)
    
    # Add cluster labels to all splits
    df_train_clustered = model.add_clusters_to_dataframe(df_train, train_embeddings)
    
    val_embeddings = np.array([ast.literal_eval(e) for e in df_val['embedding']])
    df_val_clustered = model.add_clusters_to_dataframe(df_val, val_embeddings)
    
    test_embeddings = np.array([ast.literal_eval(e) for e in df_test['embedding']])
    df_test_clustered = model.add_clusters_to_dataframe(df_test, test_embeddings)
    
    # Calculate evaluation metrics
    silhouette = model.calculate_silhouette_score(train_embeddings)
    davies_bouldin = model.calculate_davies_bouldin_score(train_embeddings)
    
    click.echo(f"\nClustering quality metrics (train set):")
    click.echo(f"  Silhouette score: {silhouette:.4f}")
    click.echo(f"  Davies-Bouldin score: {davies_bouldin:.4f}")
    
    # Save clustered data
    train_output = output_path / 'train_clustered.csv'
    val_output = output_path / 'val_clustered.csv'
    test_output = output_path / 'test_clustered.csv'
    
    df_train_clustered.to_csv(train_output, index=False)
    df_val_clustered.to_csv(val_output, index=False)
    df_test_clustered.to_csv(test_output, index=False)
    
    logger.info(f"Saved clustered data to: {output_path}")
    
    # Generate visualizations
    click.echo("\nGenerating visualizations...")
    configure_matplotlib_style()
    
    # Combine all splits for full distribution
    df_all = pd.concat([df_train_clustered, df_val_clustered, df_test_clustered])
    df_all['split'] = (
        ['train'] * len(df_train_clustered) +
        ['validation'] * len(df_val_clustered) +
        ['test'] * len(df_test_clustered)
    )
    
    # Plot overall distribution
    dist_plot_path = output_path / 'cluster_distribution.pdf'
    plot_cluster_distribution(df_all, save_path=dist_plot_path)
    
    # Plot by split
    plot_cluster_distributions_by_split(df_all, output_dir=output_path)
    
    click.echo(f"\n✓ KMeans clustering complete")
    click.echo(f"  Number of clusters: {optimal_k}")
    click.echo(f"  Output directory: {output_path}")
    click.echo(f"  Clustered data: train_clustered.csv, val_clustered.csv, test_clustered.csv")
    click.echo(f"  Visualizations: cluster_distribution*.pdf")
