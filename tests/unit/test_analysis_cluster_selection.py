"""Tests for cluster selection module."""

import numpy as np
import pandas as pd
import pytest

from PMRTN.analysis.cluster_selection import (
    ClusterSelectionError,
    assign_trading_rules,
    calculate_cluster_sharpe_ratios,
    calculate_spearman_correlation,
    rank_clusters_by_sharpe,
    select_clusters_greedy,
    select_clusters_rank_stable,
    separate_clusters_by_sr_sign,
)


class TestClusterSelectionError:
    """Tests for ClusterSelectionError exception."""

    def test_cluster_selection_error_can_be_raised(self) -> None:
        """Test exception can be raised and caught."""
        with pytest.raises(ClusterSelectionError):
            raise ClusterSelectionError("Test error")

    def test_cluster_selection_error_message_preserved(self) -> None:
        """Test exception message is preserved."""
        error_msg = "Cluster selection failed"
        try:
            raise ClusterSelectionError(error_msg)
        except ClusterSelectionError as e:
            assert str(e) == error_msg


class TestCalculateClusterSharpeRatios:
    """Tests for calculate_cluster_sharpe_ratios function."""

    @pytest.fixture
    def sample_articles_df(self) -> pd.DataFrame:
        """Create sample articles DataFrame."""
        return pd.DataFrame({
            'split': ['Train', 'Train', 'Validation', 'Validation', 'Test'],
            'cluster': [0, 1, 0, 1, 0],
            'tickers': ['TEF.MC'] * 5
        }, index=[0, 1, 2, 3, 4])

    @pytest.fixture
    def sample_ts_dict(self) -> dict:
        """Create sample trading strategy dictionary."""
        ts_dict = {}
        for i in range(5):
            df = pd.DataFrame({
                'AR': np.random.randn(10) * 0.01,
                'SR': np.random.randn(10) * 0.5 + 0.5
            })
            ts_dict[i] = df
        return ts_dict

    def test_calculate_cluster_sharpe_ratios_happy_path(
        self, sample_articles_df: pd.DataFrame, sample_ts_dict: dict
    ) -> None:
        """Test calculating average SR for each (split, cluster)."""
        avg_sr_dict = calculate_cluster_sharpe_ratios(
            sample_articles_df, sample_ts_dict, l_value=5
        )

        assert isinstance(avg_sr_dict, dict)
        # Check that keys are tuples of (split, cluster)
        for key in avg_sr_dict.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)  # split
            assert isinstance(key[1], int)  # cluster

    def test_calculate_cluster_sharpe_ratios_empty_articles_df(
        self, sample_ts_dict: dict
    ) -> None:
        """Test with empty articles DataFrame."""
        empty_df = pd.DataFrame(columns=['split', 'cluster'])
        avg_sr_dict = calculate_cluster_sharpe_ratios(empty_df, sample_ts_dict, l_value=5)

        assert isinstance(avg_sr_dict, dict)
        assert len(avg_sr_dict) == 0

    def test_calculate_cluster_sharpe_ratios_empty_ts_dict(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with empty ts_dict."""
        empty_ts_dict = {}
        avg_sr_dict = calculate_cluster_sharpe_ratios(
            sample_articles_df, empty_ts_dict, l_value=5
        )

        assert isinstance(avg_sr_dict, dict)
        assert len(avg_sr_dict) == 0

    def test_calculate_cluster_sharpe_ratios_missing_articles(
        self, sample_articles_df: pd.DataFrame, sample_ts_dict: dict
    ) -> None:
        """Test when some articles are missing from ts_dict."""
        # Remove some entries from ts_dict
        partial_ts_dict = {k: v for k, v in list(sample_ts_dict.items())[:3]}
        
        avg_sr_dict = calculate_cluster_sharpe_ratios(
            sample_articles_df, partial_ts_dict, l_value=5
        )

        # Should handle gracefully
        assert isinstance(avg_sr_dict, dict)

    def test_calculate_cluster_sharpe_ratios_nan_sr_values(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with NaN SR values (should be skipped)."""
        ts_dict = {}
        for i in range(5):
            df = pd.DataFrame({
                'AR': np.random.randn(10) * 0.01,
                'SR': [np.nan] * 10  # All NaN
            })
            ts_dict[i] = df

        avg_sr_dict = calculate_cluster_sharpe_ratios(
            sample_articles_df, ts_dict, l_value=5
        )

        # NaN values should be skipped
        assert isinstance(avg_sr_dict, dict)

    def test_calculate_cluster_sharpe_ratios_single_cluster(
        self, sample_ts_dict: dict
    ) -> None:
        """Test with single cluster."""
        df = pd.DataFrame({
            'split': ['Train'] * 3,
            'cluster': [0] * 3,
            'tickers': ['TEF.MC'] * 3
        }, index=[0, 1, 2])

        avg_sr_dict = calculate_cluster_sharpe_ratios(df, sample_ts_dict, l_value=5)
        assert isinstance(avg_sr_dict, dict)

    def test_calculate_cluster_sharpe_ratios_multiple_clusters(
        self, sample_ts_dict: dict
    ) -> None:
        """Test with multiple clusters."""
        df = pd.DataFrame({
            'split': ['Train'] * 6,
            'cluster': [0, 1, 2, 0, 1, 2],
            'tickers': ['TEF.MC'] * 6
        }, index=list(range(6)))

        avg_sr_dict = calculate_cluster_sharpe_ratios(df, sample_ts_dict, l_value=5)
        assert isinstance(avg_sr_dict, dict)


class TestRankClustersBySharpe:
    """Tests for rank_clusters_by_sharpe function."""

    @pytest.fixture
    def sample_avg_sr_dict(self) -> dict:
        """Create sample average SR dictionary."""
        return {
            ('Train', 0): 0.8,
            ('Train', 1): 1.2,
            ('Train', 2): 0.5,
            ('Validation', 0): 0.9,
            ('Validation', 1): 1.1,
            ('Validation', 2): 0.6,
        }

    def test_rank_clusters_by_sharpe_happy_path(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test ranking clusters by SR within each split."""
        ranking_dict = rank_clusters_by_sharpe(sample_avg_sr_dict)

        assert isinstance(ranking_dict, dict)
        assert 'Train' in ranking_dict
        assert 'Validation' in ranking_dict

        # Check that rankings are sorted by SR descending
        train_ranking = ranking_dict['Train']
        assert train_ranking[0][1] >= train_ranking[1][1]  # First has higher SR
        assert train_ranking[1][1] >= train_ranking[2][1]  # Second has higher SR than third

    def test_rank_clusters_by_sharpe_empty_dict(self) -> None:
        """Test with empty avg_sr_dict."""
        empty_dict = {}
        ranking_dict = rank_clusters_by_sharpe(empty_dict)

        assert isinstance(ranking_dict, dict)
        assert len(ranking_dict) == 0

    def test_rank_clusters_by_sharpe_single_cluster_per_split(self) -> None:
        """Test with single cluster per split."""
        single_dict = {
            ('Train', 0): 0.8,
            ('Validation', 0): 0.9
        }
        ranking_dict = rank_clusters_by_sharpe(single_dict)

        assert len(ranking_dict['Train']) == 1
        assert len(ranking_dict['Validation']) == 1

    def test_rank_clusters_by_sharpe_tied_sr_values(self) -> None:
        """Test with tied SR values."""
        tied_dict = {
            ('Train', 0): 1.0,
            ('Train', 1): 1.0,
            ('Train', 2): 0.5
        }
        ranking_dict = rank_clusters_by_sharpe(tied_dict)

        # Should handle ties gracefully
        assert len(ranking_dict['Train']) == 3

    def test_rank_clusters_by_sharpe_nan_sr_values(self) -> None:
        """Test with NaN SR values."""
        nan_dict = {
            ('Train', 0): 0.8,
            ('Train', 1): np.nan,
            ('Train', 2): 0.5
        }
        ranking_dict = rank_clusters_by_sharpe(nan_dict)

        # Should handle NaN in sorting
        assert isinstance(ranking_dict, dict)


class TestSelectClustersGreedy:
    """Tests for select_clusters_greedy function."""

    @pytest.fixture
    def sample_ranking_dict(self) -> dict:
        """Create sample ranking dictionary."""
        return {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5), (3, -0.3), (4, -0.8)],
            'Validation': [(1, 1.1), (0, 0.9), (2, 0.6), (3, -0.4), (4, -0.7)]
        }

    @pytest.fixture
    def sample_avg_sr_dict(self) -> dict:
        """Create sample average SR dictionary."""
        return {
            ('Train', 0): 0.8,
            ('Train', 1): 1.2,
            ('Validation', 0): 0.9,
            ('Validation', 1): 1.1,
        }

    def test_select_clusters_greedy_happy_path(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test greedy cluster selection."""
        long_clusters, short_clusters = select_clusters_greedy(
            sample_ranking_dict, sample_avg_sr_dict, theta=2
        )

        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)
        # Should select top theta clusters
        assert len(long_clusters) <= 2
        assert len(short_clusters) <= 2

    def test_select_clusters_greedy_theta_one(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with theta = 1 (minimum)."""
        long_clusters, short_clusters = select_clusters_greedy(
            sample_ranking_dict, sample_avg_sr_dict, theta=1
        )

        assert len(long_clusters) <= 1
        assert len(short_clusters) <= 1

    def test_select_clusters_greedy_theta_equals_total(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with theta = total clusters (all selected)."""
        long_clusters, short_clusters = select_clusters_greedy(
            sample_ranking_dict, sample_avg_sr_dict, theta=10
        )

        # Should select all available clusters
        assert len(long_clusters) <= len([c for c, sr in sample_ranking_dict['Validation'] if sr > 0])
        assert len(short_clusters) <= len([c for c, sr in sample_ranking_dict['Validation'] if sr < 0])

    def test_select_clusters_greedy_theta_greater_than_total(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with theta > total clusters (all selected)."""
        long_clusters, short_clusters = select_clusters_greedy(
            sample_ranking_dict, sample_avg_sr_dict, theta=100
        )

        # Should select all available
        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)

    def test_select_clusters_greedy_no_positive_sr(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test with no positive SR clusters."""
        negative_ranking = {
            'Validation': [(0, -0.3), (1, -0.5), (2, -0.8)]
        }
        negative_sr_dict = {
            ('Validation', 0): -0.3,
            ('Validation', 1): -0.5,
            ('Validation', 2): -0.8
        }

        long_clusters, short_clusters = select_clusters_greedy(
            negative_ranking, negative_sr_dict, theta=2
        )

        assert len(long_clusters) == 0
        assert len(short_clusters) > 0

    def test_select_clusters_greedy_no_negative_sr(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test with no negative SR clusters."""
        positive_ranking = {
            'Validation': [(0, 0.3), (1, 0.5), (2, 0.8)]
        }
        positive_sr_dict = {
            ('Validation', 0): 0.3,
            ('Validation', 1): 0.5,
            ('Validation', 2): 0.8
        }

        long_clusters, short_clusters = select_clusters_greedy(
            positive_ranking, positive_sr_dict, theta=2
        )

        assert len(long_clusters) > 0
        assert len(short_clusters) == 0

    def test_select_clusters_greedy_all_zero_sr(
        self
    ) -> None:
        """Test with all clusters having zero SR."""
        zero_ranking = {
            'Validation': [(0, 0.0), (1, 0.0), (2, 0.0)]
        }
        zero_sr_dict = {
            ('Validation', 0): 0.0,
            ('Validation', 1): 0.0,
            ('Validation', 2): 0.0
        }

        long_clusters, short_clusters = select_clusters_greedy(
            zero_ranking, zero_sr_dict, theta=2
        )

        # Zero SR should not be selected
        assert len(long_clusters) == 0
        assert len(short_clusters) == 0

    def test_select_clusters_greedy_empty_ranking_dict(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test with empty ranking_dict."""
        empty_ranking = {}
        
        with pytest.raises(ClusterSelectionError):
            select_clusters_greedy(empty_ranking, sample_avg_sr_dict, theta=2)

    def test_select_clusters_greedy_missing_split(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with missing validation split."""
        incomplete_ranking = {'Train': sample_ranking_dict['Train']}
        
        with pytest.raises(ClusterSelectionError):
            select_clusters_greedy(incomplete_ranking, sample_avg_sr_dict, theta=2)


class TestSelectClustersRankStable:
    """Tests for select_clusters_rank_stable function."""

    @pytest.fixture
    def sample_ranking_dict(self) -> dict:
        """Create sample ranking dictionary."""
        return {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5), (3, -0.3)],
            'Validation': [(1, 1.1), (0, 0.9), (2, 0.6), (3, -0.4)]
        }

    @pytest.fixture
    def sample_avg_sr_dict(self) -> dict:
        """Create sample average SR dictionary."""
        return {
            ('Train', 0): 0.8,
            ('Train', 1): 1.2,
            ('Train', 2): 0.5,
            ('Train', 3): -0.3,
            ('Validation', 0): 0.9,
            ('Validation', 1): 1.1,
            ('Validation', 2): 0.6,
            ('Validation', 3): -0.4
        }

    def test_select_clusters_rank_stable_happy_path(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test rank-stable cluster selection."""
        long_clusters, short_clusters = select_clusters_rank_stable(
            sample_ranking_dict, sample_avg_sr_dict, theta=2
        )

        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)

    def test_select_clusters_rank_stable_perfect_correlation(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test with perfect rank correlation (all clusters stable)."""
        perfect_ranking = {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5)],
            'Validation': [(1, 1.1), (0, 0.9), (2, 0.6)]
        }

        long_clusters, short_clusters = select_clusters_rank_stable(
            perfect_ranking, sample_avg_sr_dict, theta=2
        )

        # Should select clusters with consistent rankings
        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)

    def test_select_clusters_rank_stable_no_correlation(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test with no rank correlation (no stable clusters)."""
        no_corr_ranking = {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5)],
            'Validation': [(2, 0.6), (1, 1.1), (0, 0.9)]  # Reversed order
        }

        long_clusters, short_clusters = select_clusters_rank_stable(
            no_corr_ranking, sample_avg_sr_dict, theta=2
        )

        # May still select some clusters
        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)

    def test_select_clusters_rank_stable_theta_one(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with theta = 1."""
        long_clusters, short_clusters = select_clusters_rank_stable(
            sample_ranking_dict, sample_avg_sr_dict, theta=1
        )

        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)

    def test_select_clusters_rank_stable_theta_equals_total(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with theta = total clusters."""
        long_clusters, short_clusters = select_clusters_rank_stable(
            sample_ranking_dict, sample_avg_sr_dict, theta=10
        )

        assert isinstance(long_clusters, list)
        assert isinstance(short_clusters, list)

    def test_select_clusters_rank_stable_empty_ranking_dict(
        self, sample_avg_sr_dict: dict
    ) -> None:
        """Test with empty ranking_dict."""
        empty_ranking = {}
        
        with pytest.raises(ClusterSelectionError):
            select_clusters_rank_stable(empty_ranking, sample_avg_sr_dict, theta=2)

    def test_select_clusters_rank_stable_missing_train_split(
        self, sample_ranking_dict: dict, sample_avg_sr_dict: dict
    ) -> None:
        """Test with missing train split."""
        incomplete_ranking = {'Validation': sample_ranking_dict['Validation']}
        
        with pytest.raises(ClusterSelectionError):
            select_clusters_rank_stable(incomplete_ranking, sample_avg_sr_dict, theta=2)


class TestAssignTradingRules:
    """Tests for assign_trading_rules function."""

    @pytest.fixture
    def sample_articles_df(self) -> pd.DataFrame:
        """Create sample articles DataFrame."""
        return pd.DataFrame({
            'cluster': [0, 1, 2, 3, 4],
            'tickers': ['TEF.MC'] * 5
        })

    def test_assign_trading_rules_happy_path(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test assigning trading rules."""
        long_clusters = [0, 1]
        short_clusters = [2, 3]

        result = assign_trading_rules(
            sample_articles_df, long_clusters, short_clusters
        )

        assert isinstance(result, pd.DataFrame)
        assert 'TR' in result.columns
        assert (result.loc[result['cluster'].isin(long_clusters), 'TR'] == 1).all()
        assert (result.loc[result['cluster'].isin(short_clusters), 'TR'] == -1).all()
        assert (result.loc[~result['cluster'].isin(long_clusters + short_clusters), 'TR'] == 0).all()

    def test_assign_trading_rules_empty_cluster_lists(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with empty cluster lists."""
        result = assign_trading_rules(
            sample_articles_df, [], []
        )

        assert (result['TR'] == 0).all()

    def test_assign_trading_rules_all_clusters_selected(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with all clusters selected."""
        all_clusters = [0, 1, 2, 3, 4]
        long_clusters = all_clusters[:3]
        short_clusters = all_clusters[3:]

        result = assign_trading_rules(
            sample_articles_df, long_clusters, short_clusters
        )

        assert (result['TR'] != 0).all()

    def test_assign_trading_rules_no_clusters_selected(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with no clusters selected."""
        result = assign_trading_rules(
            sample_articles_df, [], []
        )

        assert (result['TR'] == 0).all()

    def test_assign_trading_rules_overlapping_lists(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with overlapping long/short lists (should handle)."""
        long_clusters = [0, 1, 2]  # Overlap with short
        short_clusters = [2, 3, 4]  # Cluster 2 in both

        result = assign_trading_rules(
            sample_articles_df, long_clusters, short_clusters
        )

        # Short should overwrite long (last assignment wins)
        assert (result.loc[result['cluster'] == 2, 'TR'] == -1).all()

    def test_assign_trading_rules_custom_column_name(
        self, sample_articles_df: pd.DataFrame
    ) -> None:
        """Test with custom rule column name."""
        result = assign_trading_rules(
            sample_articles_df, [0], [1], rule_column='custom_TR'
        )

        assert 'custom_TR' in result.columns
        assert 'TR' not in result.columns


class TestCalculateSpearmanCorrelation:
    """Tests for calculate_spearman_correlation function."""

    @pytest.fixture
    def sample_ranking_dict(self) -> dict:
        """Create sample ranking dictionary."""
        return {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5)],
            'Validation': [(1, 1.1), (0, 0.9), (2, 0.6)]
        }

    def test_calculate_spearman_correlation_happy_path(
        self, sample_ranking_dict: dict
    ) -> None:
        """Test calculating Spearman correlation."""
        correlation, common_clusters = calculate_spearman_correlation(
            sample_ranking_dict, 'Train', 'Validation'
        )

        assert isinstance(correlation, (float, type(np.nan)))
        assert isinstance(common_clusters, list)
        assert len(common_clusters) > 0

    def test_calculate_spearman_correlation_perfect_correlation(
        self
    ) -> None:
        """Test with perfect correlation (1.0)."""
        perfect_ranking = {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5)],
            'Validation': [(1, 1.1), (0, 0.9), (2, 0.6)]  # Same order
        }

        correlation, common_clusters = calculate_spearman_correlation(
            perfect_ranking, 'Train', 'Validation'
        )

        assert correlation == 1.0

    def test_calculate_spearman_correlation_no_correlation(
        self
    ) -> None:
        """Test with no correlation (0.0 or negative)."""
        no_corr_ranking = {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5)],
            'Validation': [(2, 0.6), (1, 1.1), (0, 0.9)]  # Reversed
        }

        correlation, common_clusters = calculate_spearman_correlation(
            no_corr_ranking, 'Train', 'Validation'
        )

        assert correlation < 0  # Negative correlation

    def test_calculate_spearman_correlation_empty_rankings(
        self
    ) -> None:
        """Test with empty rankings."""
        empty_ranking = {
            'Train': [],
            'Validation': []
        }

        correlation, common_clusters = calculate_spearman_correlation(
            empty_ranking, 'Train', 'Validation'
        )

        assert np.isnan(correlation)
        assert len(common_clusters) == 0

    def test_calculate_spearman_correlation_single_cluster(
        self
    ) -> None:
        """Test with single cluster (correlation undefined)."""
        single_ranking = {
            'Train': [(0, 0.8)],
            'Validation': [(0, 0.9)]
        }

        correlation, common_clusters = calculate_spearman_correlation(
            single_ranking, 'Train', 'Validation'
        )

        # Single element correlation may be NaN or 1.0
        assert isinstance(correlation, (float, type(np.nan)))


class TestSeparateClustersBySrSign:
    """Tests for separate_clusters_by_sr_sign function."""

    @pytest.fixture
    def sample_ranking_dict(self) -> dict:
        """Create sample ranking dictionary."""
        return {
            'Train': [(1, 1.2), (0, 0.8), (2, -0.5), (3, -0.8)],
            'Validation': [(1, 1.1), (0, 0.9), (2, -0.6), (3, -0.7)]
        }

    def test_separate_clusters_by_sr_sign_happy_path(
        self, sample_ranking_dict: dict
    ) -> None:
        """Test separating clusters by SR sign."""
        result = separate_clusters_by_sr_sign(sample_ranking_dict)

        assert isinstance(result, dict)
        assert 'Train' in result
        assert 'Validation' in result
        assert 'positive' in result['Train']
        assert 'negative' in result['Train']
        assert isinstance(result['Train']['positive'], set)
        assert isinstance(result['Train']['negative'], set)

    def test_separate_clusters_by_sr_sign_all_positive(
        self
    ) -> None:
        """Test with all positive SR."""
        positive_ranking = {
            'Train': [(1, 1.2), (0, 0.8), (2, 0.5)]
        }

        result = separate_clusters_by_sr_sign(positive_ranking)

        assert len(result['Train']['positive']) == 3
        assert len(result['Train']['negative']) == 0

    def test_separate_clusters_by_sr_sign_all_negative(
        self
    ) -> None:
        """Test with all negative SR."""
        negative_ranking = {
            'Train': [(1, -0.3), (0, -0.5), (2, -0.8)]
        }

        result = separate_clusters_by_sr_sign(negative_ranking)

        assert len(result['Train']['positive']) == 0
        assert len(result['Train']['negative']) == 3

    def test_separate_clusters_by_sr_sign_all_zero(
        self
    ) -> None:
        """Test with all zero SR."""
        zero_ranking = {
            'Train': [(1, 0.0), (0, 0.0), (2, 0.0)]
        }

        result = separate_clusters_by_sr_sign(zero_ranking)

        assert len(result['Train']['positive']) == 0
        assert len(result['Train']['negative']) == 0

    def test_separate_clusters_by_sr_sign_empty_dict(
        self
    ) -> None:
        """Test with empty ranking_dict."""
        empty_ranking = {}
        result = separate_clusters_by_sr_sign(empty_ranking)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_separate_clusters_by_sr_sign_split_not_in_dict(
        self, sample_ranking_dict: dict
    ) -> None:
        """Test with split not in dictionary."""
        result = separate_clusters_by_sr_sign(sample_ranking_dict, splits=['Nonexistent'])

        assert isinstance(result, dict)
        assert 'Nonexistent' not in result

    def test_separate_clusters_by_sr_sign_custom_splits(
        self, sample_ranking_dict: dict
    ) -> None:
        """Test with custom splits list."""
        result = separate_clusters_by_sr_sign(sample_ranking_dict, splits=['Train'])

        assert 'Train' in result
        assert 'Validation' not in result


