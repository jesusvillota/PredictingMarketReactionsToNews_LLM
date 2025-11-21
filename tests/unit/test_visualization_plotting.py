"""Tests for visualization plotting module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from PMRTN.visualization.plotting import (
    PlottingError,
    configure_matplotlib_style,
    plot_average_cars_by_cluster,
    plot_cluster_distribution,
    plot_cluster_distributions_by_split,
    plot_cumulative_returns,
    reset_matplotlib_style,
)


@pytest.fixture
def sample_cluster_df():
    """Create sample DataFrame with cluster assignments."""
    np.random.seed(42)
    return pd.DataFrame({
        'cluster': np.random.randint(0, 5, 100),
        'split': np.random.choice(['Train', 'Validation', 'Test'], 100),
        'article': [f'Article {i}' for i in range(100)]
    })


@pytest.fixture
def sample_car_data():
    """Create sample CAR data."""
    np.random.seed(42)
    car_data = {}
    for split in ['Train', 'Validation', 'Test']:
        for cluster in range(3):
            car_data[(split, cluster)] = np.random.randn(50).cumsum()
    return car_data


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return {
        'Strategy A': pd.Series(np.random.randn(100) * 0.01, index=dates),
        'Strategy B': pd.Series(np.random.randn(100) * 0.01, index=dates)
    }


class TestPlotClusterDistribution:
    """Tests for plot_cluster_distribution function."""
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_basic_plot(self, mock_show, sample_cluster_df):
        """Test basic cluster distribution plot."""
        fig = plot_cluster_distribution(
            df=sample_cluster_df,
            show_plot=True
        )
        
        assert fig is not None
        mock_show.assert_called_once()
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_plot_with_split_filter(self, mock_show, sample_cluster_df):
        """Test plot with split filtering."""
        fig = plot_cluster_distribution(
            df=sample_cluster_df,
            split_column='split',
            split_value='Train',
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_plot_without_density(self, mock_show, sample_cluster_df):
        """Test plot without density overlay."""
        fig = plot_cluster_distribution(
            df=sample_cluster_df,
            plot_density=False,
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_plot_without_title(self, mock_show, sample_cluster_df):
        """Test plot without title."""
        fig = plot_cluster_distribution(
            df=sample_cluster_df,
            show_title=False,
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_save_plot(self, mock_show, sample_cluster_df):
        """Test saving plot to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_plot.pdf'
            
            fig = plot_cluster_distribution(
                df=sample_cluster_df,
                output_path=output_path,
                show_plot=False
            )
            
            assert fig is not None
            assert output_path.exists()
    
    def test_missing_cluster_column(self, sample_cluster_df):
        """Test error when cluster column is missing."""
        with pytest.raises(PlottingError, match="Cluster column.*not found"):
            plot_cluster_distribution(
                df=sample_cluster_df,
                cluster_column='nonexistent',
                show_plot=False
            )
    
    def test_missing_split_column(self, sample_cluster_df):
        """Test error when split column is missing."""
        with pytest.raises(PlottingError, match="Split column.*not found"):
            plot_cluster_distribution(
                df=sample_cluster_df,
                split_column='nonexistent',
                split_value='Train',
                show_plot=False
            )
    
    def test_empty_after_filtering(self, sample_cluster_df):
        """Test error when filtering results in empty data."""
        with pytest.raises(PlottingError, match="No data to plot"):
            plot_cluster_distribution(
                df=sample_cluster_df,
                split_column='split',
                split_value='NonexistentSplit',
                show_plot=False
            )
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_custom_figsize(self, mock_show, sample_cluster_df):
        """Test plot with custom figure size."""
        fig = plot_cluster_distribution(
            df=sample_cluster_df,
            figsize=(10, 5),
            show_plot=False
        )
        
        assert fig is not None


class TestPlotClusterDistributionsBySplit:
    """Tests for plot_cluster_distributions_by_split function."""
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_plot_all_splits(self, mock_show, sample_cluster_df):
        """Test plotting for all splits."""
        figures = plot_cluster_distributions_by_split(
            df=sample_cluster_df,
            show_plot=False
        )
        
        assert len(figures) == 3  # Train, Validation, Test
        assert 'Train' in figures
        assert 'Validation' in figures
        assert 'Test' in figures
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_save_all_splits(self, mock_show, sample_cluster_df):
        """Test saving plots for all splits."""
        with tempfile.TemporaryDirectory() as tmpdir:
            figures = plot_cluster_distributions_by_split(
                df=sample_cluster_df,
                output_dir=tmpdir,
                show_plot=False
            )
            
            assert len(figures) == 3
            # Check files were created
            output_dir = Path(tmpdir)
            assert (output_dir / 'Cluster_Distribution_Train.pdf').exists()
            assert (output_dir / 'Cluster_Distribution_Validation.pdf').exists()
            assert (output_dir / 'Cluster_Distribution_Test.pdf').exists()
    
    def test_missing_split_column(self, sample_cluster_df):
        """Test error when split column is missing."""
        with pytest.raises(PlottingError, match="Split column.*not found"):
            plot_cluster_distributions_by_split(
                df=sample_cluster_df,
                split_column='nonexistent',
                show_plot=False
            )


class TestPlotAverageCARsByCluster:
    """Tests for plot_average_cars_by_cluster function."""
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_basic_car_plot(self, mock_show, sample_car_data):
        """Test basic CAR plot."""
        fig = plot_average_cars_by_cluster(
            car_data=sample_car_data,
            split='Train',
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_car_plot_with_limit(self, mock_show, sample_car_data):
        """Test CAR plot with point limit."""
        fig = plot_average_cars_by_cluster(
            car_data=sample_car_data,
            split='Train',
            max_points=30,
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_save_car_plot(self, mock_show, sample_car_data):
        """Test saving CAR plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'car_plot.pdf'
            
            fig = plot_average_cars_by_cluster(
                car_data=sample_car_data,
                split='Train',
                output_path=output_path,
                show_plot=False
            )
            
            assert fig is not None
            assert output_path.exists()
    
    def test_no_data_for_split(self, sample_car_data):
        """Test error when no data exists for split."""
        with pytest.raises(PlottingError, match="No CAR data found"):
            plot_average_cars_by_cluster(
                car_data=sample_car_data,
                split='NonexistentSplit',
                show_plot=False
            )
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_empty_car_arrays_ignored(self, mock_show):
        """Test that empty CAR arrays are ignored."""
        car_data = {
            ('Train', 0): np.array([1, 2, 3]),
            ('Train', 1): np.array([]),  # Empty array
            ('Train', 2): np.array([4, 5, 6])
        }
        
        fig = plot_average_cars_by_cluster(
            car_data=car_data,
            split='Train',
            show_plot=False
        )
        
        assert fig is not None


class TestPlotCumulativeReturns:
    """Tests for plot_cumulative_returns function."""
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_basic_returns_plot(self, mock_show, sample_returns):
        """Test basic cumulative returns plot."""
        fig = plot_cumulative_returns(
            returns_dict=sample_returns,
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_custom_title(self, mock_show, sample_returns):
        """Test plot with custom title."""
        fig = plot_cumulative_returns(
            returns_dict=sample_returns,
            title='Custom Portfolio Returns',
            show_plot=False
        )
        
        assert fig is not None
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_save_returns_plot(self, mock_show, sample_returns):
        """Test saving returns plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'returns.pdf'
            
            fig = plot_cumulative_returns(
                returns_dict=sample_returns,
                output_path=output_path,
                show_plot=False
            )
            
            assert fig is not None
            assert output_path.exists()
    
    def test_empty_returns_dict(self):
        """Test error with empty returns dictionary."""
        with pytest.raises(PlottingError, match="No returns data provided"):
            plot_cumulative_returns(
                returns_dict={},
                show_plot=False
            )
    
    def test_invalid_returns_type(self):
        """Test error with invalid returns type."""
        with pytest.raises(PlottingError, match="must be a pandas Series"):
            plot_cumulative_returns(
                returns_dict={'Strategy': [1, 2, 3]},  # List instead of Series
                show_plot=False
            )
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_non_datetime_index(self, mock_show):
        """Test plot with non-datetime index."""
        returns_dict = {
            'Strategy': pd.Series(np.random.randn(100) * 0.01)
        }
        
        fig = plot_cumulative_returns(
            returns_dict=returns_dict,
            show_plot=False
        )
        
        assert fig is not None


class TestMatplotlibConfiguration:
    """Tests for matplotlib configuration functions."""
    
    def test_configure_matplotlib_style(self):
        """Test configuring matplotlib style."""
        configure_matplotlib_style(
            use_latex=True,
            font_family='serif'
        )
        
        # Just verify it doesn't raise an error
        # Actual configuration is difficult to test
    
    def test_configure_without_latex(self):
        """Test configuring without LaTeX."""
        configure_matplotlib_style(use_latex=False)
    
    def test_reset_matplotlib_style(self):
        """Test resetting matplotlib style."""
        configure_matplotlib_style()
        reset_matplotlib_style()
        
        # Just verify it doesn't raise an error


class TestPlottingIntegration:
    """Integration tests for plotting functions."""
    
    @patch('PMRTN.visualization.plotting.plt.show')
    def test_full_workflow(self, mock_show, sample_cluster_df, sample_car_data, sample_returns):
        """Test complete plotting workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Plot cluster distributions
            figures1 = plot_cluster_distributions_by_split(
                df=sample_cluster_df,
                output_dir=output_dir / 'clusters',
                show_plot=False
            )
            assert len(figures1) == 3
            
            # Plot CARs
            fig2 = plot_average_cars_by_cluster(
                car_data=sample_car_data,
                split='Train',
                output_path=output_dir / 'cars.pdf',
                show_plot=False
            )
            assert fig2 is not None
            
            # Plot returns
            fig3 = plot_cumulative_returns(
                returns_dict=sample_returns,
                output_path=output_dir / 'returns.pdf',
                show_plot=False
            )
            assert fig3 is not None
