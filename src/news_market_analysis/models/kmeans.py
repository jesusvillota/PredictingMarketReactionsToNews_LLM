"""KMeans clustering model for news article embeddings.

This module provides a wrapper around scikit-learn's KMeans implementation
with utilities for training, prediction, and evaluation of clustering models
on news article embeddings.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, pairwise_distances_argmin_min, silhouette_score
from sklearn.preprocessing import StandardScaler


class ClusteringError(Exception):
    """Raised when clustering operations fail."""

    pass


class NewsClusteringModel:
    """KMeans clustering model for news article embeddings.

    This class wraps scikit-learn's KMeans with additional functionality
    for news article clustering, including model training, prediction,
    cluster assignment, and evaluation metrics.

    Attributes:
        n_clusters: Number of clusters to form.
        random_state: Random seed for reproducibility.
        model: The underlying KMeans model.
        scaler: StandardScaler for embedding normalization.
        is_fitted: Whether the model has been trained.
        cluster_centers: Cluster centroids in scaled space.
        labels: Cluster labels for training data.
        inertia: Sum of squared distances to nearest cluster center.
        n_iter: Number of iterations run by KMeans.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
        init: str = "k-means++",
        **kwargs,
    ) -> None:
        """Initialize the NewsClusteringModel.

        Args:
            n_clusters: Number of clusters to form.
            random_state: Random seed for reproducibility.
            max_iter: Maximum number of iterations.
            n_init: Number of times KMeans is run with different seeds.
            init: Method for initialization ('k-means++', 'random', or ndarray).
            **kwargs: Additional arguments passed to KMeans.

        Raises:
            ValueError: If n_clusters is less than 2.
        """
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init

        # Initialize the KMeans model
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            init=init,
            **kwargs,
        )

        # Initialize scaler for embeddings
        self.scaler = StandardScaler()

        # Tracking state
        self.is_fitted = False
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        scale: bool = True,
    ) -> "NewsClusteringModel":
        """Fit the clustering model on training embeddings.

        Args:
            embeddings: Training embeddings, shape (n_samples, n_features).
            scale: Whether to apply StandardScaler to embeddings.

        Returns:
            Self for method chaining.

        Raises:
            ClusteringError: If embeddings are invalid or fitting fails.
        """
        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if embeddings.ndim != 2:
            raise ClusteringError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )

        if embeddings.shape[0] < self.n_clusters:
            raise ClusteringError(
                f"Number of samples ({embeddings.shape[0]}) must be >= "
                f"n_clusters ({self.n_clusters})"
            )

        try:
            # Scale embeddings if requested
            if scale:
                embeddings_scaled = self.scaler.fit_transform(embeddings)
            else:
                embeddings_scaled = embeddings

            # Fit the model
            self.model.fit(embeddings_scaled)

            # Store results
            self.cluster_centers_ = self.model.cluster_centers_
            self.labels_ = self.model.labels_
            self.inertia_ = self.model.inertia_
            self.n_iter_ = self.model.n_iter_
            self.is_fitted = True

            return self

        except Exception as e:
            raise ClusteringError(f"Failed to fit clustering model: {str(e)}")

    def predict(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        scale: bool = True,
    ) -> np.ndarray:
        """Predict cluster labels for new embeddings.

        Args:
            embeddings: Embeddings to predict, shape (n_samples, n_features).
            scale: Whether to apply the fitted scaler to embeddings.

        Returns:
            Cluster labels for each sample, shape (n_samples,).

        Raises:
            ClusteringError: If model not fitted or prediction fails.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted before prediction")

        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if embeddings.ndim != 2:
            raise ClusteringError(
                f"Embeddings must be 2D array, got shape {embeddings.shape}"
            )

        try:
            # Scale embeddings if requested
            if scale:
                embeddings_scaled = self.scaler.transform(embeddings)
            else:
                embeddings_scaled = embeddings

            # Predict labels
            labels = self.model.predict(embeddings_scaled)
            return labels

        except Exception as e:
            raise ClusteringError(f"Failed to predict cluster labels: {str(e)}")

    def fit_predict(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        scale: bool = True,
    ) -> np.ndarray:
        """Fit the model and predict cluster labels in one step.

        Args:
            embeddings: Training embeddings, shape (n_samples, n_features).
            scale: Whether to apply StandardScaler to embeddings.

        Returns:
            Cluster labels for training data, shape (n_samples,).
        """
        self.fit(embeddings, scale=scale)
        return self.labels_

    def get_cluster_centers(self, unscale: bool = False) -> np.ndarray:
        """Get the cluster centers.

        Args:
            unscale: Whether to inverse-transform centers back to original scale.

        Returns:
            Cluster centers, shape (n_clusters, n_features).

        Raises:
            ClusteringError: If model not fitted.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted first")

        centers = self.cluster_centers_

        if unscale:
            centers = self.scaler.inverse_transform(centers)

        return centers

    def calculate_silhouette_score(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        labels: Optional[np.ndarray] = None,
        scale: bool = True,
    ) -> float:
        """Calculate silhouette score for clustering quality.

        The silhouette score ranges from -1 to 1, where:
        - 1 indicates well-separated clusters
        - 0 indicates overlapping clusters
        - -1 indicates incorrect clustering

        Args:
            embeddings: Embeddings to evaluate, shape (n_samples, n_features).
            labels: Cluster labels. If None, use training labels.
            scale: Whether to scale embeddings.

        Returns:
            Silhouette score.

        Raises:
            ClusteringError: If model not fitted or calculation fails.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted first")

        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # Use training labels if not provided
        if labels is None:
            labels = self.labels_

        # Scale embeddings if requested
        if scale:
            embeddings_scaled = self.scaler.transform(embeddings)
        else:
            embeddings_scaled = embeddings

        try:
            score = silhouette_score(embeddings_scaled, labels)
            return float(score)
        except Exception as e:
            raise ClusteringError(f"Failed to calculate silhouette score: {str(e)}")

    def calculate_davies_bouldin_score(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        labels: Optional[np.ndarray] = None,
        scale: bool = True,
    ) -> float:
        """Calculate Davies-Bouldin index for clustering quality.

        Lower values indicate better clustering. The score is defined as the
        average similarity measure of each cluster with its most similar cluster,
        where similarity is the ratio of within-cluster to between-cluster distances.

        Args:
            embeddings: Embeddings to evaluate, shape (n_samples, n_features).
            labels: Cluster labels. If None, use training labels.
            scale: Whether to scale embeddings.

        Returns:
            Davies-Bouldin index.

        Raises:
            ClusteringError: If model not fitted or calculation fails.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted first")

        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # Use training labels if not provided
        if labels is None:
            labels = self.labels_

        # Scale embeddings if requested
        if scale:
            embeddings_scaled = self.scaler.transform(embeddings)
        else:
            embeddings_scaled = embeddings

        try:
            score = davies_bouldin_score(embeddings_scaled, labels)
            return float(score)
        except Exception as e:
            raise ClusteringError(f"Failed to calculate Davies-Bouldin score: {str(e)}")

    def get_cluster_distribution(
        self, labels: Optional[np.ndarray] = None
    ) -> Dict[int, int]:
        """Get the distribution of samples across clusters.

        Args:
            labels: Cluster labels. If None, use training labels.

        Returns:
            Dictionary mapping cluster ID to sample count.

        Raises:
            ClusteringError: If model not fitted.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted first")

        if labels is None:
            labels = self.labels_

        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_distances_to_centers(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        scale: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate distances from each sample to nearest cluster center.

        Args:
            embeddings: Embeddings to evaluate, shape (n_samples, n_features).
            scale: Whether to scale embeddings.

        Returns:
            Tuple of (cluster_labels, distances) where:
                - cluster_labels: Nearest cluster for each sample, shape (n_samples,)
                - distances: Distance to nearest cluster, shape (n_samples,)

        Raises:
            ClusteringError: If model not fitted.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted first")

        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # Scale embeddings if requested
        if scale:
            embeddings_scaled = self.scaler.transform(embeddings)
        else:
            embeddings_scaled = embeddings

        # Calculate distances
        labels, distances = pairwise_distances_argmin_min(
            embeddings_scaled, self.cluster_centers_
        )

        return labels, distances

    def add_clusters_to_dataframe(
        self,
        df: pd.DataFrame,
        embeddings_col: str = "embeddings",
        cluster_col: str = "cluster",
        scale: bool = True,
    ) -> pd.DataFrame:
        """Add cluster labels to a DataFrame with embeddings.

        Args:
            df: DataFrame containing embeddings.
            embeddings_col: Name of column with embeddings.
            cluster_col: Name for new cluster label column.
            scale: Whether to scale embeddings.

        Returns:
            DataFrame with added cluster column.

        Raises:
            ClusteringError: If model not fitted or operation fails.
        """
        if not self.is_fitted:
            raise ClusteringError("Model must be fitted first")

        if embeddings_col not in df.columns:
            raise ClusteringError(f"Column '{embeddings_col}' not found in DataFrame")

        df = df.copy()

        try:
            # Extract embeddings as numpy array
            embeddings = np.array(df[embeddings_col].tolist())

            # Predict clusters
            labels = self.predict(embeddings, scale=scale)

            # Add to DataFrame
            df[cluster_col] = labels

            return df

        except Exception as e:
            raise ClusteringError(f"Failed to add clusters to DataFrame: {str(e)}")

    def get_model_info(self) -> Dict[str, Union[int, float, bool]]:
        """Get information about the model configuration and state.

        Returns:
            Dictionary with model information.
        """
        info = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "init": self.init,
            "is_fitted": self.is_fitted,
        }

        if self.is_fitted:
            info.update(
                {
                    "inertia": self.inertia_,
                    "n_iter": self.n_iter_,
                    "n_features": self.cluster_centers_.shape[1],
                }
            )

        return info


def find_optimal_k(
    embeddings: Union[np.ndarray, List[List[float]]],
    k_range: range = range(2, 11),
    method: str = "silhouette",
    scale: bool = True,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[int, Dict[int, float]]:
    """Find optimal number of clusters using elbow method or silhouette analysis.

    Args:
        embeddings: Training embeddings, shape (n_samples, n_features).
        k_range: Range of cluster numbers to try.
        method: Evaluation method - 'silhouette', 'davies_bouldin', or 'inertia'.
        scale: Whether to scale embeddings.
        random_state: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        Tuple of (optimal_k, scores_dict) where:
            - optimal_k: Optimal number of clusters
            - scores_dict: Dictionary mapping k to evaluation score

    Raises:
        ValueError: If method is invalid.
        ClusteringError: If optimization fails.
    """
    if method not in ["silhouette", "davies_bouldin", "inertia"]:
        raise ValueError(
            f"Invalid method '{method}'. Must be 'silhouette', 'davies_bouldin', or 'inertia'"
        )

    # Convert to numpy array if needed
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)

    # Scale embeddings if requested
    if scale:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
    else:
        embeddings_scaled = embeddings

    scores = {}

    try:
        for k in k_range:
            if verbose:
                print(f"Evaluating k={k}...")

            # Fit model
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(embeddings_scaled)

            # Calculate score based on method
            if method == "silhouette":
                score = silhouette_score(embeddings_scaled, labels)
                scores[k] = score
            elif method == "davies_bouldin":
                score = davies_bouldin_score(embeddings_scaled, labels)
                scores[k] = score
            elif method == "inertia":
                score = model.inertia_
                scores[k] = score

        # Find optimal k
        if method == "silhouette":
            # Higher is better
            optimal_k = max(scores, key=scores.get)
        elif method in ["davies_bouldin", "inertia"]:
            # Lower is better
            optimal_k = min(scores, key=scores.get)

        if verbose:
            print(f"\nOptimal k: {optimal_k}")
            print(f"Score: {scores[optimal_k]:.4f}")

        return optimal_k, scores

    except Exception as e:
        raise ClusteringError(f"Failed to find optimal k: {str(e)}")


def cluster_train_val_test(
    embeddings_train: Union[np.ndarray, List[List[float]]],
    embeddings_val: Union[np.ndarray, List[List[float]]],
    embeddings_test: Union[np.ndarray, List[List[float]]],
    n_clusters: int = 8,
    scale: bool = True,
    random_state: int = 42,
) -> Tuple[NewsClusteringModel, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster train/val/test splits using the same model.

    This function fits a clustering model on training data and predicts
    labels for validation and test sets.

    Args:
        embeddings_train: Training embeddings.
        embeddings_val: Validation embeddings.
        embeddings_test: Test embeddings.
        n_clusters: Number of clusters.
        scale: Whether to scale embeddings.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (model, labels_train, labels_val, labels_test).

    Raises:
        ClusteringError: If clustering fails.
    """
    try:
        # Initialize and fit model on training data
        model = NewsClusteringModel(n_clusters=n_clusters, random_state=random_state)
        labels_train = model.fit_predict(embeddings_train, scale=scale)

        # Predict on validation and test
        labels_val = model.predict(embeddings_val, scale=scale)
        labels_test = model.predict(embeddings_test, scale=scale)

        return model, labels_train, labels_val, labels_test

    except Exception as e:
        raise ClusteringError(f"Failed to cluster train/val/test splits: {str(e)}")
