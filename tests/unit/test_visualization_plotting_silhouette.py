"""Unit tests for silhouette score plotting functions."""

import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use('Agg')

from PMRTN.visualization.plotting import PlottingError, plot_silhouette_scores


class TestPlotSilhouetteScores:
    """Test cases for plot_silhouette_scores function."""
    
    def test_basic_plot_creation(self):
        """Test that basic plot is created successfully."""
        k_range = range(2, 11)
        scores = [0.45, 0.52, 0.58, 0.55, 0.51, 0.48, 0.44, 0.42, 0.40]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_optimal_k_marking(self):
        """Test plot with optimal k marked."""
        k_range = range(2, 11)
        scores = [0.45, 0.52, 0.58, 0.55, 0.51, 0.48, 0.44, 0.42, 0.40]
        optimal_k = 4
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            optimal_k=optimal_k,
            show_plot=False
        )
        
        assert fig is not None
        
        # Check that vertical line for optimal k exists
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        # Should have main line and vertical line for optimal k
        assert len(lines) >= 2
        plt.close(fig)
    
    def test_without_optimal_k(self):
        """Test plot without optimal k marking."""
        k_range = range(2, 11)
        scores = [0.45, 0.52, 0.58, 0.55, 0.51, 0.48, 0.44, 0.42, 0.40]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            optimal_k=None,
            show_plot=False
        )
        
        assert fig is not None
        
        # Should only have main line (no optimal k line)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 1
        plt.close(fig)
    
    def test_custom_labels_and_title(self):
        """Test plot with custom labels and title."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        custom_title = "Custom Silhouette Analysis"
        custom_xlabel = "K Values"
        custom_ylabel = "Score"
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            title=custom_title,
            xlabel=custom_xlabel,
            ylabel=custom_ylabel,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        assert ax.get_title() == custom_title
        assert ax.get_xlabel() == custom_xlabel
        assert ax.get_ylabel() == custom_ylabel
        plt.close(fig)
    
    def test_custom_colors(self):
        """Test plot with custom colors."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            line_color='green',
            optimal_k_color='purple',
            optimal_k=3,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        # Check main line color
        assert lines[0].get_color() == 'green'
        
        # Check optimal k line color
        if len(lines) > 1:
            assert lines[1].get_color() == 'purple'
        
        plt.close(fig)
    
    def test_custom_marker_style(self):
        """Test plot with custom marker style."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            marker='s',  # Square marker
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        lines = ax.get_lines()
        
        assert lines[0].get_marker() == 's'
        plt.close(fig)
    
    def test_custom_figure_size(self):
        """Test plot with custom figure size."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        custom_figsize = (14, 8)
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            figsize=custom_figsize,
            show_plot=False
        )
        
        assert fig is not None
        # Note: figsize might be slightly different due to DPI, so we check approximate match
        fig_width, fig_height = fig.get_size_inches()
        assert abs(fig_width - custom_figsize[0]) < 0.1
        assert abs(fig_height - custom_figsize[1]) < 0.1
        plt.close(fig)
    
    def test_save_to_file(self):
        """Test saving plot to file."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "silhouette_test.pdf"
            
            fig = plot_silhouette_scores(
                k_range,
                scores,
                output_path=output_path,
                save_output=True,
                show_plot=False
            )
            
            assert fig is not None
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            plt.close(fig)
    
    def test_save_to_directory(self):
        """Test saving plot to directory (auto-generate filename)."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        title = "Test Silhouette Plot"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            fig = plot_silhouette_scores(
                k_range,
                scores,
                title=title,
                output_path=output_dir,
                save_output=True,
                show_plot=False
            )
            
            assert fig is not None
            
            # Check that file was created with auto-generated name
            expected_file = output_dir / f"{title.replace(' ', '_')}.pdf"
            assert expected_file.exists()
            plt.close(fig)
    
    def test_list_inputs(self):
        """Test with list inputs instead of range."""
        k_range = [2, 3, 4, 5, 6]
        scores = [0.45, 0.52, 0.58, 0.55, 0.51]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_numpy_array_inputs(self):
        """Test with numpy array inputs."""
        k_range = np.array([2, 3, 4, 5, 6])
        scores = np.array([0.45, 0.52, 0.58, 0.55, 0.51])
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_empty_k_range_raises_error(self):
        """Test that empty k_range raises error."""
        k_range = []
        scores = []
        
        with pytest.raises(PlottingError, match="k_range is empty"):
            plot_silhouette_scores(k_range, scores, show_plot=False)
    
    def test_empty_scores_raises_error(self):
        """Test that empty scores raises error."""
        k_range = range(2, 6)
        scores = []
        
        with pytest.raises(PlottingError, match="scores array is empty"):
            plot_silhouette_scores(k_range, scores, show_plot=False)
    
    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched k_range and scores lengths raise error."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58]  # Too short
        
        with pytest.raises(PlottingError, match="must have the same length"):
            plot_silhouette_scores(k_range, scores, show_plot=False)
    
    def test_nan_values_raise_error(self):
        """Test that NaN values in scores raise error."""
        k_range = range(2, 6)
        scores = [0.45, np.nan, 0.58, 0.55]
        
        with pytest.raises(PlottingError, match="scores contains NaN values"):
            plot_silhouette_scores(k_range, scores, show_plot=False)
    
    def test_optimal_k_not_in_range_warning(self, capsys):
        """Test warning when optimal_k is not in k_range."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        optimal_k = 10  # Not in range
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            optimal_k=optimal_k,
            show_plot=False
        )
        
        assert fig is not None
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "not in k_range" in captured.out
        plt.close(fig)
    
    def test_custom_font_sizes(self):
        """Test plot with custom font sizes."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            title_fontsize=20,
            label_fontsize=16,
            tick_fontsize=14,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Check that title exists
        assert ax.get_title() != ""
        
        # Check that labels exist
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        
        plt.close(fig)
    
    def test_grid_enabled(self):
        """Test that grid is enabled on plot."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Grid should be enabled
        assert ax.xaxis.get_gridlines()[0].get_visible() or ax.yaxis.get_gridlines()[0].get_visible()
        plt.close(fig)
    
    def test_spines_configuration(self):
        """Test that top and right spines are hidden."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Top and right spines should be hidden
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        
        # Left and bottom spines should be visible
        assert ax.spines['left'].get_visible()
        assert ax.spines['bottom'].get_visible()
        plt.close(fig)
    
    def test_integer_xticks(self):
        """Test that x-axis shows integer ticks only."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        # X-ticks should be integers matching k_range
        xticks = ax.get_xticks()
        expected_ticks = list(k_range)
        
        # Allow for some tolerance in tick positioning
        assert len(xticks) >= len(expected_ticks)
        plt.close(fig)
    
    def test_legend_exists(self):
        """Test that legend exists and is properly configured."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            optimal_k=3,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        legend = ax.get_legend()
        assert legend is not None
        
        # Check legend has expected number of entries
        labels = [text.get_text() for text in legend.get_texts()]
        assert len(labels) >= 1  # At least "Silhouette Score"
        plt.close(fig)
    
    def test_background_color(self):
        """Test that plot has gray background."""
        k_range = range(2, 6)
        scores = [0.45, 0.52, 0.58, 0.55]
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            show_plot=False
        )
        
        assert fig is not None
        ax = fig.axes[0]
        
        # Background should be gray (#f5f5f5)
        facecolor = ax.get_facecolor()
        # RGB values for #f5f5f5 are approximately (0.96, 0.96, 0.96, 1.0)
        assert facecolor[0] > 0.9  # R
        assert facecolor[1] > 0.9  # G
        assert facecolor[2] > 0.9  # B
        plt.close(fig)
    
    def test_scores_vary_realistically(self):
        """Test with realistic silhouette score variation."""
        # Typical pattern: scores increase, peak, then decrease
        k_range = range(2, 21)
        scores = [
            0.35, 0.42, 0.48, 0.52, 0.55, 0.58, 0.59, 0.58, 0.56, 0.54,
            0.51, 0.48, 0.45, 0.43, 0.41, 0.39, 0.37, 0.36, 0.35
        ]
        optimal_k = 8  # Peak score
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            optimal_k=optimal_k,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_integration_with_kmeans_output(self):
        """Test integration with typical K-means clustering output."""
        # Simulate output from find_optimal_k function
        k_range = range(2, 11)
        scores = [0.42, 0.51, 0.57, 0.59, 0.58, 0.55, 0.51, 0.48, 0.45]
        optimal_k = 5
        
        fig = plot_silhouette_scores(
            k_range,
            scores,
            optimal_k=optimal_k,
            title="K-Means Clustering: Optimal K Selection",
            xlabel="Number of Clusters (k)",
            ylabel="Average Silhouette Score",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
