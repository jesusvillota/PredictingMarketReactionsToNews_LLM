"""Tests for time series plotting functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from PMRTN.visualization.plotting import plot_time_series_with_ma, PlottingError


@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randint(1, 20, 100)
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_time_series_with_trend():
    """Create time series with trend."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    trend = np.linspace(5, 15, 100)
    noise = np.random.normal(0, 2, 100)
    values = trend + noise
    return pd.Series(values, index=dates)


class TestPlotTimeSeriesWithMA:
    """Tests for plot_time_series_with_ma function."""
    
    def test_basic_plot(self, sample_time_series):
        """Test basic time series plotting."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            show_plot=False
        )
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_trend(self, sample_time_series_with_trend):
        """Test plotting time series with trend."""
        fig = plot_time_series_with_ma(
            sample_time_series_with_trend,
            ma_window=10,
            title="Test Time Series with Trend",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_custom_ma_window(self, sample_time_series):
        """Test different moving average windows."""
        for window in [5, 10, 20]:
            fig = plot_time_series_with_ma(
                sample_time_series,
                ma_window=window,
                show_plot=False
            )
            assert fig is not None
            plt.close(fig)
    
    def test_custom_colors(self, sample_time_series):
        """Test custom color settings."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            series_color='green',
            ma_color='orange',
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_custom_labels(self, sample_time_series):
        """Test custom labels."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            title="Custom Title",
            xlabel="Custom X Label",
            ylabel="Custom Y Label",
            series_label="Original Data",
            ma_label="Custom MA",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_date_formatting(self, sample_time_series):
        """Test different date format strings."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            date_format='%d/%m/%Y',
            rotation=90,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_save_output(self, sample_time_series, tmp_path):
        """Test saving plot to file."""
        output_path = tmp_path / "test_timeseries.pdf"
        
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            output_path=output_path,
            save_output=True,
            show_plot=False
        )
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)
    
    def test_dict_input(self):
        """Test with dictionary input."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        data_dict = {date: np.random.randint(1, 20) for date in dates}
        
        fig = plot_time_series_with_ma(
            data_dict,
            ma_window=5,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_empty_series_raises_error(self):
        """Test that empty series raises PlottingError."""
        empty_series = pd.Series([], dtype=float)
        
        with pytest.raises(PlottingError, match="Series is empty"):
            plot_time_series_with_ma(empty_series, show_plot=False)
    
    def test_all_nan_raises_error(self):
        """Test that all-NaN series raises PlottingError."""
        dates = pd.date_range('2020-01-01', periods=10)
        nan_series = pd.Series([np.nan] * 10, index=dates)
        
        with pytest.raises(PlottingError, match="All series values are NaN"):
            plot_time_series_with_ma(nan_series, show_plot=False)
    
    def test_non_datetime_index_raises_error(self):
        """Test that non-datetime index raises PlottingError."""
        series = pd.Series([1, 2, 3, 4, 5])
        
        with pytest.raises(PlottingError, match="must have a DatetimeIndex"):
            plot_time_series_with_ma(series, show_plot=False)
    
    def test_invalid_type_raises_error(self):
        """Test that invalid input type raises PlottingError."""
        with pytest.raises(PlottingError, match="must be a pandas Series"):
            plot_time_series_with_ma([1, 2, 3], show_plot=False)
    
    def test_custom_figsize(self, sample_time_series):
        """Test custom figure size."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            figsize=(15, 8),
            show_plot=False
        )
        
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 8
        plt.close(fig)
    
    def test_custom_font_sizes(self, sample_time_series):
        """Test custom font sizes."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            title_fontsize=20,
            label_fontsize=16,
            tick_fontsize=14,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_articles_per_day_example(self):
        """Test realistic example: articles per day."""
        # Simulate articles per day data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        # More articles on weekdays, fewer on weekends
        articles = []
        for date in dates:
            if date.dayofweek < 5:  # Weekday
                articles.append(np.random.randint(5, 15))
            else:  # Weekend
                articles.append(np.random.randint(0, 5))
        
        series = pd.Series(articles, index=dates)
        
        fig = plot_time_series_with_ma(
            series,
            ma_window=7,
            title="Articles per Day",
            ylabel="Number of Articles",
            date_format='%b %Y',
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_monthly_aggregation_example(self):
        """Test with monthly aggregated data."""
        dates = pd.date_range('2018-01-01', '2022-12-31', freq='MS')
        values = np.random.randint(50, 200, len(dates))
        series = pd.Series(values, index=dates)
        
        fig = plot_time_series_with_ma(
            series,
            ma_window=3,  # 3-month moving average
            title="Monthly Article Count",
            ylabel="Articles",
            date_format='%Y-%m',
            ma_label="3-month MA",
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_nan_values(self):
        """Test handling of NaN values in series."""
        dates = pd.date_range('2020-01-01', periods=50)
        values = np.random.randint(1, 20, 50).astype(float)
        # Add some NaN values
        values[10:15] = np.nan
        values[30:33] = np.nan
        
        series = pd.Series(values, index=dates)
        
        fig = plot_time_series_with_ma(
            series,
            ma_window=5,
            show_plot=False
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_save_to_directory(self, sample_time_series, tmp_path):
        """Test saving with directory path (should use title for filename)."""
        fig = plot_time_series_with_ma(
            sample_time_series,
            ma_window=7,
            title="Test Plot",
            output_path=tmp_path,
            save_output=True,
            show_plot=False
        )
        
        expected_path = tmp_path / "Test_Plot.pdf"
        assert expected_path.exists()
        plt.close(fig)
