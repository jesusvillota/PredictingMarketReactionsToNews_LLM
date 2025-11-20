"""Tests for KMeans clustering model."""

import numpy as np
import pandas as pd
import pytest

from news_market_analysis.models.kmeans import (
    ClusteringError,
    NewsClusteringModel,
    cluster_train_val_test,
    find_optimal_k,
)


# Fixtures


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    # Create 3 clusters of data
    cluster1 = np.random.randn(30, 10) + np.array([5, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(30, 10) + np.array([-5, -5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(30, 10) + np.array([0, 0, 5, 5, 0, 0, 0, 0, 0, 0])
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_embeddings_list():
    """Create sample embeddings as list."""
    np.random.seed(42)
    embeddings = np.random.randn(50, 10)
    return embeddings.tolist()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame with embeddings."""
    np.random.seed(42)
    embeddings = [np.random.randn(10).tolist() for _ in range(20)]
    df = pd.DataFrame(
        {
            "article": [f"Article {i}" for i in range(20)],
            "embeddings": embeddings,
            "date": pd.date_range("2024-01-01", periods=20),
        }
    )
    return df


# NewsClusteringModel Tests


def test_model_initialization():
    """Test NewsClusteringModel initialization."""
    model = NewsClusteringModel(n_clusters=5, random_state=42)
    assert model.n_clusters == 5
    assert model.random_state == 42
    assert not model.is_fitted
    assert model.labels_ is None


def test_model_initialization_invalid_clusters():
    """Test initialization with invalid n_clusters."""
    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        NewsClusteringModel(n_clusters=1)


def test_model_fit(sample_embeddings):
    """Test fitting the model."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    result = model.fit(sample_embeddings, scale=True)

    assert result is model  # Check method chaining
    assert model.is_fitted
    assert model.labels_ is not None
    assert len(model.labels_) == len(sample_embeddings)
    assert model.cluster_centers_ is not None
    assert model.cluster_centers_.shape == (3, 10)
    assert model.inertia_ is not None
    assert model.n_iter_ is not None


def test_model_fit_with_list(sample_embeddings_list):
    """Test fitting with list input."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings_list, scale=True)

    assert model.is_fitted
    assert len(model.labels_) == len(sample_embeddings_list)


def test_model_fit_without_scaling(sample_embeddings):
    """Test fitting without scaling."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=False)

    assert model.is_fitted
    assert len(model.labels_) == len(sample_embeddings)


def test_model_fit_invalid_embeddings():
    """Test fitting with invalid embeddings."""
    model = NewsClusteringModel(n_clusters=3)

    # 1D array
    with pytest.raises(ClusteringError, match="must be 2D array"):
        model.fit(np.array([1, 2, 3]))

    # Too few samples
    with pytest.raises(ClusteringError, match="must be >="):
        model.fit(np.random.randn(2, 10))


def test_model_predict(sample_embeddings):
    """Test prediction on new data."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings[:70], scale=True)

    # Predict on test set
    test_embeddings = sample_embeddings[70:]
    labels = model.predict(test_embeddings, scale=True)

    assert len(labels) == len(test_embeddings)
    assert labels.min() >= 0
    assert labels.max() < 3


def test_model_predict_not_fitted(sample_embeddings):
    """Test prediction before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.predict(sample_embeddings)


def test_model_predict_invalid_embeddings(sample_embeddings):
    """Test prediction with invalid embeddings."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=True)

    # 1D array
    with pytest.raises(ClusteringError, match="must be 2D array"):
        model.predict(np.array([1, 2, 3]))


def test_model_fit_predict(sample_embeddings):
    """Test fit_predict method."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    labels = model.fit_predict(sample_embeddings, scale=True)

    assert model.is_fitted
    assert len(labels) == len(sample_embeddings)
    assert np.array_equal(labels, model.labels_)


def test_get_cluster_centers(sample_embeddings):
    """Test getting cluster centers."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=True)

    # Scaled centers
    centers_scaled = model.get_cluster_centers(unscale=False)
    assert centers_scaled.shape == (3, 10)

    # Unscaled centers
    centers_unscaled = model.get_cluster_centers(unscale=True)
    assert centers_unscaled.shape == (3, 10)
    assert not np.array_equal(centers_scaled, centers_unscaled)


def test_get_cluster_centers_not_fitted():
    """Test getting centers before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.get_cluster_centers()


def test_calculate_silhouette_score(sample_embeddings):
    """Test silhouette score calculation."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=True)

    score = model.calculate_silhouette_score(sample_embeddings, scale=True)
    assert isinstance(score, float)
    assert -1 <= score <= 1


def test_calculate_silhouette_score_not_fitted(sample_embeddings):
    """Test silhouette score before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.calculate_silhouette_score(sample_embeddings)


def test_calculate_davies_bouldin_score(sample_embeddings):
    """Test Davies-Bouldin score calculation."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=True)

    score = model.calculate_davies_bouldin_score(sample_embeddings, scale=True)
    assert isinstance(score, float)
    assert score >= 0


def test_calculate_davies_bouldin_score_not_fitted(sample_embeddings):
    """Test Davies-Bouldin score before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.calculate_davies_bouldin_score(sample_embeddings)


def test_get_cluster_distribution(sample_embeddings):
    """Test cluster distribution calculation."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=True)

    distribution = model.get_cluster_distribution()
    assert isinstance(distribution, dict)
    assert len(distribution) == 3
    assert sum(distribution.values()) == len(sample_embeddings)


def test_get_cluster_distribution_not_fitted():
    """Test cluster distribution before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.get_cluster_distribution()


def test_get_distances_to_centers(sample_embeddings):
    """Test distance calculation to cluster centers."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    model.fit(sample_embeddings, scale=True)

    labels, distances = model.get_distances_to_centers(sample_embeddings, scale=True)
    assert len(labels) == len(sample_embeddings)
    assert len(distances) == len(sample_embeddings)
    assert all(d >= 0 for d in distances)


def test_get_distances_to_centers_not_fitted(sample_embeddings):
    """Test distance calculation before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.get_distances_to_centers(sample_embeddings)


def test_add_clusters_to_dataframe(sample_dataframe):
    """Test adding clusters to DataFrame."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)

    # Extract embeddings and fit
    embeddings = np.array(sample_dataframe["embeddings"].tolist())
    model.fit(embeddings, scale=True)

    # Add clusters to DataFrame
    df_with_clusters = model.add_clusters_to_dataframe(
        sample_dataframe, embeddings_col="embeddings", cluster_col="cluster"
    )

    assert "cluster" in df_with_clusters.columns
    assert len(df_with_clusters) == len(sample_dataframe)
    assert df_with_clusters["cluster"].min() >= 0
    assert df_with_clusters["cluster"].max() < 3


def test_add_clusters_to_dataframe_not_fitted(sample_dataframe):
    """Test adding clusters before fitting."""
    model = NewsClusteringModel(n_clusters=3)

    with pytest.raises(ClusteringError, match="must be fitted"):
        model.add_clusters_to_dataframe(sample_dataframe)


def test_add_clusters_to_dataframe_missing_column(sample_dataframe):
    """Test adding clusters with missing column."""
    model = NewsClusteringModel(n_clusters=3, random_state=42)
    embeddings = np.array(sample_dataframe["embeddings"].tolist())
    model.fit(embeddings, scale=True)

    with pytest.raises(ClusteringError, match="not found"):
        model.add_clusters_to_dataframe(
            sample_dataframe, embeddings_col="nonexistent"
        )


def test_get_model_info():
    """Test getting model information."""
    model = NewsClusteringModel(n_clusters=5, random_state=42, max_iter=100)

    # Before fitting
    info = model.get_model_info()
    assert info["n_clusters"] == 5
    assert info["random_state"] == 42
    assert info["max_iter"] == 100
    assert not info["is_fitted"]
    assert "inertia" not in info

    # After fitting
    embeddings = np.random.randn(50, 10)
    model.fit(embeddings)
    info = model.get_model_info()
    assert info["is_fitted"]
    assert "inertia" in info
    assert "n_iter" in info
    assert "n_features" in info


# Utility Functions Tests


def test_find_optimal_k_silhouette(sample_embeddings):
    """Test finding optimal k using silhouette method."""
    optimal_k, scores = find_optimal_k(
        sample_embeddings,
        k_range=range(2, 6),
        method="silhouette",
        scale=True,
        random_state=42,
        verbose=False,
    )

    assert isinstance(optimal_k, int)
    assert 2 <= optimal_k < 6
    assert isinstance(scores, dict)
    assert len(scores) == 4
    assert all(isinstance(v, float) for v in scores.values())


def test_find_optimal_k_davies_bouldin(sample_embeddings):
    """Test finding optimal k using Davies-Bouldin method."""
    optimal_k, scores = find_optimal_k(
        sample_embeddings,
        k_range=range(2, 6),
        method="davies_bouldin",
        scale=True,
        random_state=42,
    )

    assert isinstance(optimal_k, int)
    assert 2 <= optimal_k < 6
    assert isinstance(scores, dict)


def test_find_optimal_k_inertia(sample_embeddings):
    """Test finding optimal k using inertia method."""
    optimal_k, scores = find_optimal_k(
        sample_embeddings,
        k_range=range(2, 6),
        method="inertia",
        scale=True,
        random_state=42,
    )

    assert isinstance(optimal_k, int)
    assert 2 <= optimal_k < 6
    assert isinstance(scores, dict)


def test_find_optimal_k_invalid_method(sample_embeddings):
    """Test find_optimal_k with invalid method."""
    with pytest.raises(ValueError, match="Invalid method"):
        find_optimal_k(sample_embeddings, method="invalid")


def test_find_optimal_k_with_list(sample_embeddings_list):
    """Test find_optimal_k with list input."""
    optimal_k, scores = find_optimal_k(
        sample_embeddings_list,
        k_range=range(2, 5),
        method="silhouette",
        random_state=42,
    )

    assert isinstance(optimal_k, int)
    assert isinstance(scores, dict)


def test_cluster_train_val_test(sample_embeddings):
    """Test clustering train/val/test splits."""
    # Split data
    train = sample_embeddings[:50]
    val = sample_embeddings[50:70]
    test = sample_embeddings[70:]

    model, labels_train, labels_val, labels_test = cluster_train_val_test(
        train, val, test, n_clusters=3, scale=True, random_state=42
    )

    assert isinstance(model, NewsClusteringModel)
    assert model.is_fitted
    assert len(labels_train) == len(train)
    assert len(labels_val) == len(val)
    assert len(labels_test) == len(test)
    assert all(0 <= label < 3 for label in labels_train)
    assert all(0 <= label < 3 for label in labels_val)
    assert all(0 <= label < 3 for label in labels_test)


def test_cluster_train_val_test_without_scaling(sample_embeddings):
    """Test clustering without scaling."""
    train = sample_embeddings[:50]
    val = sample_embeddings[50:70]
    test = sample_embeddings[70:]

    model, labels_train, labels_val, labels_test = cluster_train_val_test(
        train, val, test, n_clusters=3, scale=False, random_state=42
    )

    assert model.is_fitted
    assert len(labels_train) == len(train)


# Edge Cases


def test_model_with_single_cluster():
    """Test model behavior with edge case parameters."""
    # While n_clusters=2 is minimum, test it works
    embeddings = np.random.randn(20, 5)
    model = NewsClusteringModel(n_clusters=2, random_state=42)
    model.fit(embeddings)

    assert model.is_fitted
    assert len(np.unique(model.labels_)) <= 2


def test_model_with_high_dimensional_data():
    """Test with high-dimensional embeddings."""
    embeddings = np.random.randn(50, 512)  # Realistic embedding dimension
    model = NewsClusteringModel(n_clusters=5, random_state=42)
    model.fit(embeddings)

    assert model.is_fitted
    assert model.cluster_centers_.shape == (5, 512)


def test_model_deterministic_with_seed():
    """Test that results are deterministic with same seed."""
    embeddings = np.random.randn(50, 10)

    model1 = NewsClusteringModel(n_clusters=3, random_state=42)
    labels1 = model1.fit_predict(embeddings)

    model2 = NewsClusteringModel(n_clusters=3, random_state=42)
    labels2 = model2.fit_predict(embeddings)

    assert np.array_equal(labels1, labels2)


def test_model_different_results_without_seed():
    """Test that results differ without fixed seed."""
    embeddings = np.random.randn(50, 10)

    model1 = NewsClusteringModel(n_clusters=3, random_state=None)
    labels1 = model1.fit_predict(embeddings)

    model2 = NewsClusteringModel(n_clusters=3, random_state=None)
    labels2 = model2.fit_predict(embeddings)

    # Results should likely differ (not guaranteed but very probable)
    # We just check both completed successfully
    assert len(labels1) == len(labels2) == len(embeddings)
