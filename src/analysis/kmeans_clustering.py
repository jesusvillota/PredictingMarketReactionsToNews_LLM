"""KMeans clustering analysis on article embeddings."""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Optional
from sklearn.cluster import KMeans

from src.config import get_paths, get_logger, config_settings

logger = get_logger("analysis.kmeans_clustering")


def perform_kmeans_clustering(
    processed_data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    n_clusters: Optional[int] = None,
    save_output: bool = True
) -> tuple:
    """
    Perform KMeans clustering on article embeddings.
    
    Args:
        processed_data_path: Path to processed data directory. If None, uses config default.
        output_path: Path to output directory. If None, uses config default.
        n_clusters: Number of clusters. If None, uses config default.
        save_output: Whether to save outputs
    
    Returns:
        Tuple of (clustered_data, kmeans_model)
    """
    if processed_data_path is None:
        path_manager = get_paths()
        processed_data_path = path_manager.get_processed_data_path()
    
    if output_path is None:
        path_manager = get_paths()
        output_path = path_manager.get_output_path("kmeans")
    
    if n_clusters is None:
        n_clusters = config_settings.clustering_config.get("kmeans", {}).get("n_clusters", 10)
    
    logger.info(f"Performing KMeans clustering with {n_clusters} clusters")
    
    # Load data with embeddings
    D = pd.read_csv(processed_data_path / 'D_embeddings.csv')
    D['embeddings'] = D['embeddings'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Convert embeddings to numpy array
    embeddings = np.array(D['embeddings'].tolist())
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    D['cluster'] = kmeans.fit_predict(embeddings)
    
    logger.info(f"Clustering complete. Cluster distribution:")
    logger.info(D['cluster'].value_counts().sort_index())
    
    if save_output:
        output_path.mkdir(parents=True, exist_ok=True)
        # Save clustered data (can be extended to save plots, tables, etc.)
        output_file = processed_data_path / 'D_clustered_kmeans.csv'
        D.to_csv(output_file, index=False)
        logger.info(f"Saved clustered data to {output_file}")
    
    return D, kmeans

