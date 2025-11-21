"""Unit tests for histogram with density plotting function."""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from PMRTN.visualization.plotting import (
    PlottingError,
    plot_histogram_with_density,
)


class TestPlotHistogramWithDensity:
    """Tests for plot_histogram_with_density function."""

    def test_basic_plotting_with_series(self):
        """Test basic histogram plotting with pandas Series."""
        data = pd.Series(np.random.normal(100, 15, 1000))
        
        fig = plot_histogram_with_density(
            data,
            title="Test Distribution",
            xlabel="Value",
            ylabel_left="Frequency",
            ylabel_right="Density",
            show_plot=False
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotting_with_numpy_array(self):
        """Test histogram plotting with numpy array."""
        data = np.random.normal(50, 10, 500)
        
        fig = plot_histogram_with_density(
            data,
            title="NumPy Array Test",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)

    def test_plotting_with_list(self):
        """Test histogram plotting with Python list."""
        data = list(range(1, 101))
        
        fig = plot_histogram_with_density(
            data,
            title="List Test",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)

    def test_with_xlim(self):
        """Test setting x-axis limits."""
        data = pd.Series(np.random.normal(100, 15, 1000))
        
        fig = plot_histogram_with_density(
            data,
            xlim=(50, 150),
            show_plot=False
        )
        
        ax = fig.axes[0]
        xlim = ax.get_xlim()
        assert xlim[0] == pytest.approx(50, abs=1)
        assert xlim[1] == pytest.approx(150, abs=1)
        plt.close(fig)

    def test_with_custom_colors(self):
        """Test with custom histogram and density colors."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        fig = plot_histogram_with_density(
            data,
            hist_color='red',
            density_color='blue',
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)

    def test_with_custom_bins(self):
        """Test with custom number of bins."""
        data = pd.Series(np.random.normal(100, 15, 1000))
        
        fig = plot_histogram_with_density(
            data,
            bins=50,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self):
        """Test saving plot to PDF file."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_histogram.pdf"
            
            fig = plot_histogram_with_density(
                data,
                title="Save Test",
                output_path=output_path,
                save_output=True,
                show_plot=False
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)

    def test_save_to_directory_generates_filename(self):
        """Test saving to directory generates filename from title."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            fig = plot_histogram_with_density(
                data,
                title="My Test Plot",
                output_path=output_dir,
                save_output=True,
                show_plot=False
            )
            
            expected_file = output_dir / "My_Test_Plot.pdf"
            assert expected_file.exists()
            plt.close(fig)

    def test_empty_data_raises_error(self):
        """Test that empty data raises PlottingError."""
        with pytest.raises(PlottingError, match="Data is empty"):
            plot_histogram_with_density(
                pd.Series([]),
                show_plot=False
            )

    def test_all_nan_data_raises_error(self):
        """Test that all-NaN data raises PlottingError."""
        data = pd.Series([np.nan, np.nan, np.nan])
        
        with pytest.raises(PlottingError, match="All data values are NaN"):
            plot_histogram_with_density(
                data,
                show_plot=False
            )

    def test_data_with_some_nans(self):
        """Test that data with some NaN values works (NaNs removed)."""
        data = pd.Series([1, 2, np.nan, 3, 4, np.nan, 5])
        
        fig = plot_histogram_with_density(
            data,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)

    def test_only_nan_after_cleaning_raises_error(self):
        """Test error when only NaN values remain after cleaning."""
        data = pd.Series([np.nan] * 10)
        
        with pytest.raises(PlottingError):
            plot_histogram_with_density(
                data,
                show_plot=False
            )

    def test_custom_figure_size(self):
        """Test with custom figure size."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        fig = plot_histogram_with_density(
            data,
            figsize=(15, 8),
            show_plot=False
        )
        
        # Check figure size
        size = fig.get_size_inches()
        assert size[0] == pytest.approx(15, abs=0.1)
        assert size[1] == pytest.approx(8, abs=0.1)
        plt.close(fig)

    def test_has_two_y_axes(self):
        """Test that plot has two y-axes (frequency and density)."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        fig = plot_histogram_with_density(
            data,
            show_plot=False
        )
        
        # Should have 2 axes (primary and secondary y-axis)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_legend_present(self):
        """Test that legend is present."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        fig = plot_histogram_with_density(
            data,
            show_plot=False
        )
        
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_grid_is_visible(self):
        """Test that grid is visible."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        fig = plot_histogram_with_density(
            data,
            show_plot=False
        )
        
        ax = fig.axes[0]
        assert ax.get_xgridlines()[0].get_visible()
        plt.close(fig)

    def test_background_color(self):
        """Test that background has custom color."""
        data = pd.Series(np.random.normal(100, 15, 100))
        
        fig = plot_histogram_with_density(
            data,
            show_plot=False
        )
        
        ax = fig.axes[0]
        facecolor = ax.get_facecolor()
        # Should be light gray (#f5f5f5)
        assert facecolor[0] > 0.9  # Red channel
        assert facecolor[1] > 0.9  # Green channel
        assert facecolor[2] > 0.9  # Blue channel
        plt.close(fig)

    def test_matches_original_script_style(self):
        """Test that function produces output matching original script style."""
        # Simulate word count data from articles
        data = pd.Series(np.random.lognormal(5, 1, 1000))  # Log-normal for word counts
        
        fig = plot_histogram_with_density(
            data,
            title="Word Count Distribution",
            xlabel="Number of Words",
            ylabel_left="Frequency",
            ylabel_right="Density",
            xlim=(0, 500),
            bins=30,
            hist_color='skyblue',
            density_color='orange',
            show_plot=False
        )
        
        # Verify key components exist
        assert fig is not None
        assert len(fig.axes) == 2  # Primary and secondary axes
        
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        
        # Check that histogram exists on ax1
        assert len(ax1.patches) > 0  # Histogram bars exist
        
        # Check that density line exists on ax2
        assert len(ax2.lines) > 0  # Density curve exists
        
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
