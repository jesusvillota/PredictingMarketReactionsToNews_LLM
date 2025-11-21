"""Tests for visualization tables module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from PMRTN.visualization.tables import (
    TableGenerationError,
    generate_cluster_mapping_table,
    generate_llama_shock_mapping_table,
    generate_portfolio_statistics_table,
    generate_trading_intensity_table,
)


@pytest.fixture
def sample_cluster_titles():
    """Create sample cluster titles."""
    return {
        0: "Technology and Innovation",
        1: "Financial Services",
        2: "Energy and Utilities",
        3: "Healthcare and Pharmaceuticals",
        4: "Real Estate"
    }


@pytest.fixture
def sample_portfolio_statistics():
    """Create sample portfolio statistics."""
    return {
        'All': {
            'Greedy': {
                'gross': {
                    'cumulative_return': 0.15,
                    'average_return': 0.05,
                    'std_deviation': 0.10,
                    'sharpe_ratio': 0.50,
                    'sortino_ratio': 0.65,
                    'max_drawdown': -0.08,
                    'calmar_ratio': 0.625,
                    'skewness': -0.2,
                    'kurtosis': 1.5,
                    'var_95': -0.02,
                    'cvar_95': -0.03
                },
                'net': {
                    'cumulative_return': 0.12,
                    'average_return': 0.04,
                    'std_deviation': 0.10,
                    'sharpe_ratio': 0.40,
                    'sortino_ratio': 0.55,
                    'max_drawdown': -0.08,
                    'calmar_ratio': 0.50,
                    'skewness': -0.2,
                    'kurtosis': 1.5,
                    'var_95': -0.02,
                    'cvar_95': -0.03
                }
            },
            'Stable': {
                'gross': {
                    'cumulative_return': 0.13,
                    'average_return': 0.045,
                    'std_deviation': 0.09,
                    'sharpe_ratio': 0.50,
                    'sortino_ratio': 0.62,
                    'max_drawdown': -0.07,
                    'calmar_ratio': 0.643,
                    'skewness': -0.15,
                    'kurtosis': 1.2,
                    'var_95': -0.018,
                    'cvar_95': -0.028
                },
                'net': {
                    'cumulative_return': 0.10,
                    'average_return': 0.035,
                    'std_deviation': 0.09,
                    'sharpe_ratio': 0.39,
                    'sortino_ratio': 0.52,
                    'max_drawdown': -0.07,
                    'calmar_ratio': 0.50,
                    'skewness': -0.15,
                    'kurtosis': 1.2,
                    'var_95': -0.018,
                    'cvar_95': -0.028
                }
            }
        },
        'Train': {
            'Greedy': {
                'gross': {
                    'cumulative_return': 0.18,
                    'average_return': 0.06,
                    'std_deviation': 0.11,
                    'sharpe_ratio': 0.55,
                    'sortino_ratio': 0.70,
                    'max_drawdown': -0.09,
                    'calmar_ratio': 0.67,
                    'skewness': -0.25,
                    'kurtosis': 1.6,
                    'var_95': -0.022,
                    'cvar_95': -0.032
                },
                'net': {
                    'cumulative_return': 0.15,
                    'average_return': 0.05,
                    'std_deviation': 0.11,
                    'sharpe_ratio': 0.45,
                    'sortino_ratio': 0.60,
                    'max_drawdown': -0.09,
                    'calmar_ratio': 0.56,
                    'skewness': -0.25,
                    'kurtosis': 1.6,
                    'var_95': -0.022,
                    'cvar_95': -0.032
                }
            },
            'Stable': {
                'gross': {
                    'cumulative_return': 0.16,
                    'average_return': 0.055,
                    'std_deviation': 0.10,
                    'sharpe_ratio': 0.55,
                    'sortino_ratio': 0.68,
                    'max_drawdown': -0.08,
                    'calmar_ratio': 0.69,
                    'skewness': -0.20,
                    'kurtosis': 1.3,
                    'var_95': -0.020,
                    'cvar_95': -0.030
                },
                'net': {
                    'cumulative_return': 0.13,
                    'average_return': 0.045,
                    'std_deviation': 0.10,
                    'sharpe_ratio': 0.45,
                    'sortino_ratio': 0.58,
                    'max_drawdown': -0.08,
                    'calmar_ratio': 0.56,
                    'skewness': -0.20,
                    'kurtosis': 1.3,
                    'var_95': -0.020,
                    'cvar_95': -0.030
                }
            }
        }
    }


@pytest.fixture
def sample_portfolio_returns():
    """Create sample portfolio returns."""
    import numpy as np
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return {
        'All': {
            'Greedy': pd.Series(np.random.randn(100) * 0.01, index=dates),
            'Stable': pd.Series(np.random.randn(100) * 0.01, index=dates)
        },
        'Train': {
            'Greedy': pd.Series(np.random.randn(60) * 0.01, index=dates[:60]),
            'Stable': pd.Series(np.random.randn(60) * 0.01, index=dates[:60])
        }
    }


@pytest.fixture
def sample_trading_signals():
    """Create sample trading signals."""
    import numpy as np
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return {
        'All': {
            'Greedy': pd.DataFrame(
                np.random.randint(-1, 2, (100, 5)),
                index=dates,
                columns=[f'Ticker_{i}' for i in range(5)]
            ),
            'Stable': pd.DataFrame(
                np.random.randint(-1, 2, (100, 5)),
                index=dates,
                columns=[f'Ticker_{i}' for i in range(5)]
            )
        }
    }


@pytest.fixture
def sample_turnover_stats():
    """Create sample turnover statistics."""
    return {
        'Greedy': {
            'All': 0.15,
            'Train': 0.18
        },
        'Stable': {
            'All': 0.12,
            'Train': 0.14
        }
    }


class TestGenerateClusterMappingTable:
    """Tests for generate_cluster_mapping_table function."""
    
    def test_basic_table_generation(self, sample_cluster_titles):
        """Test basic table generation."""
        latex = generate_cluster_mapping_table(
            cluster_titles=sample_cluster_titles,
            greedy_long=[0, 1],
            greedy_short=[3, 4],
            stable_long=[0, 2],
            stable_short=[3]
        )
        
        assert '\\begin{table}' in latex
        assert '\\end{table}' in latex
        assert 'Technology and Innovation' in latex
        assert 'Financial Services' in latex
    
    def test_table_with_no_signals(self, sample_cluster_titles):
        """Test table with no trading signals."""
        latex = generate_cluster_mapping_table(
            cluster_titles=sample_cluster_titles,
            greedy_long=[],
            greedy_short=[],
            stable_long=[],
            stable_short=[]
        )
        
        assert '\\begin{table}' in latex
        # Should not contain any trading signals
        assert latex.count('\\textcolor{darkgreen}') == 0
        assert latex.count('\\textcolor{darkred}') == 0
    
    def test_save_to_file(self, sample_cluster_titles):
        """Test saving table to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'cluster_table.tex'
            
            latex = generate_cluster_mapping_table(
                cluster_titles=sample_cluster_titles,
                greedy_long=[0, 1],
                greedy_short=[3, 4],
                stable_long=[0, 2],
                stable_short=[3],
                output_path=output_path
            )
            
            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
            assert content == latex
    
    def test_empty_cluster_titles(self):
        """Test error with empty cluster titles."""
        with pytest.raises(TableGenerationError, match="cluster_titles cannot be empty"):
            generate_cluster_mapping_table(
                cluster_titles={},
                greedy_long=[],
                greedy_short=[],
                stable_long=[],
                stable_short=[]
            )
    
    def test_custom_model_name(self, sample_cluster_titles):
        """Test table with custom model name."""
        latex = generate_cluster_mapping_table(
            cluster_titles=sample_cluster_titles,
            greedy_long=[0],
            greedy_short=[1],
            stable_long=[0],
            stable_short=[1],
            model_name='LLAMA'
        )
        
        assert '\\begin{table}' in latex


class TestGeneratePortfolioStatisticsTable:
    """Tests for generate_portfolio_statistics_table function."""
    
    def test_gross_returns_table(self, sample_portfolio_statistics):
        """Test table generation for gross returns."""
        latex = generate_portfolio_statistics_table(
            statistics=sample_portfolio_statistics,
            label='test_portfolio',
            caption='Test Portfolio',
            subcaption_specific1='Test subcaption 1.',
            subcaption_specific2='Test subcaption 2.',
            return_type='gross'
        )
        
        assert '\\begin{table}' in latex
        assert 'Gross Returns' in latex
        assert 'Sharpe Ratio' in latex
        assert 'All' in latex
        assert 'Greedy' in latex
        assert 'Stable' in latex
    
    def test_net_returns_table(self, sample_portfolio_statistics):
        """Test table generation for net returns."""
        latex = generate_portfolio_statistics_table(
            statistics=sample_portfolio_statistics,
            label='test_portfolio',
            caption='Test Portfolio',
            subcaption_specific1='Test subcaption 1.',
            subcaption_specific2='Test subcaption 2.',
            return_type='net'
        )
        
        assert 'Net Returns' in latex
    
    def test_save_to_file(self, sample_portfolio_statistics):
        """Test saving table to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'portfolio_stats.tex'
            
            latex = generate_portfolio_statistics_table(
                statistics=sample_portfolio_statistics,
                label='test_portfolio',
                caption='Test Portfolio',
                subcaption_specific1='Test subcaption 1.',
                subcaption_specific2='Test subcaption 2.',
                output_path=output_path
            )
            
            assert output_path.exists()
    
    def test_empty_statistics(self):
        """Test error with empty statistics."""
        with pytest.raises(TableGenerationError, match="statistics cannot be empty"):
            generate_portfolio_statistics_table(
                statistics={},
                label='test',
                caption='Test',
                subcaption_specific1='Test',
                subcaption_specific2='Test'
            )
    
    def test_invalid_return_type(self, sample_portfolio_statistics):
        """Test error with invalid return type."""
        with pytest.raises(TableGenerationError, match="return_type must be"):
            generate_portfolio_statistics_table(
                statistics=sample_portfolio_statistics,
                label='test',
                caption='Test',
                subcaption_specific1='Test',
                subcaption_specific2='Test',
                return_type='invalid'
            )


class TestGenerateTradingIntensityTable:
    """Tests for generate_trading_intensity_table function."""
    
    def test_basic_table_generation(self, sample_portfolio_returns, sample_trading_signals, sample_turnover_stats):
        """Test basic trading intensity table generation."""
        result = generate_trading_intensity_table(
            portfolio_returns=sample_portfolio_returns,
            trading_signals=sample_trading_signals,
            turnover_stats=sample_turnover_stats,
            model_name='KMeans',
            label='test_intensity'
        )
        
        assert 'dataframe' in result
        assert 'latex' in result
        assert isinstance(result['dataframe'], pd.DataFrame)
        assert isinstance(result['latex'], str)
        assert '\\begin{table}' in result['latex']
    
    def test_dataframe_structure(self, sample_portfolio_returns, sample_trading_signals, sample_turnover_stats):
        """Test structure of returned DataFrame."""
        result = generate_trading_intensity_table(
            portfolio_returns=sample_portfolio_returns,
            trading_signals=sample_trading_signals,
            turnover_stats=sample_turnover_stats,
            model_name='KMeans',
            label='test_intensity'
        )
        
        df = result['dataframe']
        assert 'Split' in df.columns
        assert 'Algorithm' in df.columns
        assert 'Avg. Positions' in df.columns
        assert 'Turnover' in df.columns
    
    def test_save_to_file(self, sample_portfolio_returns, sample_trading_signals, sample_turnover_stats):
        """Test saving table to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'intensity_table.tex'
            
            result = generate_trading_intensity_table(
                portfolio_returns=sample_portfolio_returns,
                trading_signals=sample_trading_signals,
                turnover_stats=sample_turnover_stats,
                model_name='KMeans',
                label='test_intensity',
                output_path=output_path
            )
            
            assert output_path.exists()
    
    def test_empty_portfolio_returns(self):
        """Test error with empty portfolio returns."""
        with pytest.raises(TableGenerationError, match="portfolio_returns cannot be empty"):
            generate_trading_intensity_table(
                portfolio_returns={},
                trading_signals={},
                turnover_stats={},
                model_name='KMeans',
                label='test'
            )


class TestGenerateLLAMAShockMappingTable:
    """Tests for generate_llama_shock_mapping_table function."""
    
    def test_basic_table_generation(self):
        """Test basic shock mapping table generation."""
        shock_mapping = {
            0: {'shock_type': 'Positive Earnings'},
            1: {'shock_type': 'Negative Regulatory'},
            2: {'shock_type': 'Neutral Market'}
        }
        
        latex = generate_llama_shock_mapping_table(
            shock_mapping=shock_mapping,
            greedy_clusters=[0, 2],
            stable_clusters=[0, 1]
        )
        
        assert '\\begin{table}' in latex
        assert 'Positive Earnings' in latex
        assert 'Negative Regulatory' in latex
    
    def test_save_to_file(self):
        """Test saving shock mapping table to file."""
        shock_mapping = {
            0: {'shock_type': 'Test Shock'}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'shock_table.tex'
            
            latex = generate_llama_shock_mapping_table(
                shock_mapping=shock_mapping,
                greedy_clusters=[0],
                stable_clusters=[],
                output_path=output_path
            )
            
            assert output_path.exists()
    
    def test_empty_mapping(self):
        """Test with empty shock mapping."""
        latex = generate_llama_shock_mapping_table(
            shock_mapping={},
            greedy_clusters=[],
            stable_clusters=[]
        )
        
        # Should generate table structure even if empty
        assert '\\begin{table}' in latex


class TestTableIntegration:
    """Integration tests for table generation."""
    
    def test_generate_all_tables(
        self,
        sample_cluster_titles,
        sample_portfolio_statistics,
        sample_portfolio_returns,
        sample_trading_signals,
        sample_turnover_stats
    ):
        """Test generating all table types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Cluster mapping table
            latex1 = generate_cluster_mapping_table(
                cluster_titles=sample_cluster_titles,
                greedy_long=[0, 1],
                greedy_short=[3, 4],
                stable_long=[0, 2],
                stable_short=[3],
                output_path=output_dir / 'clusters.tex'
            )
            assert latex1 is not None
            
            # Portfolio statistics table
            latex2 = generate_portfolio_statistics_table(
                statistics=sample_portfolio_statistics,
                label='portfolio',
                caption='Test Portfolio',
                subcaption_specific1='Test 1',
                subcaption_specific2='Test 2',
                output_path=output_dir / 'portfolio.tex'
            )
            assert latex2 is not None
            
            # Trading intensity table
            result = generate_trading_intensity_table(
                portfolio_returns=sample_portfolio_returns,
                trading_signals=sample_trading_signals,
                turnover_stats=sample_turnover_stats,
                model_name='KMeans',
                label='intensity',
                output_path=output_dir / 'intensity.tex'
            )
            assert result['latex'] is not None
            
            # Verify all files were created
            assert (output_dir / 'clusters.tex').exists()
            assert (output_dir / 'portfolio.tex').exists()
            assert (output_dir / 'intensity.tex').exists()
