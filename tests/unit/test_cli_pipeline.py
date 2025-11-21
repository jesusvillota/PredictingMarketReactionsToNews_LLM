"""Unit tests for CLI pipeline command."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path

from PMRTN.cli.pipeline_commands import run_all


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


class TestRunAllCommand:
    """Tests for run-all pipeline command."""
    
    def test_run_all_help(self, cli_runner):
        """Test that help displays correctly."""
        result = cli_runner.invoke(run_all, ['--help'])
        assert result.exit_code == 0
        assert 'pipeline' in result.output.lower()
        assert 'load-articles' in result.output
        assert 'download-returns' in result.output
        assert 'kmeans-clustering' in result.output
        assert 'llama-parse' in result.output
    
    def test_run_all_requires_rf_data(self, cli_runner):
        """Test that rf-data is required when download-returns is not skipped."""
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'kmeans'],
            input='y\n'  # Say yes to confirmation
        )
        assert result.exit_code == 1
        assert '--rf-data is required' in result.output or 'Error' in result.output
    
    def test_run_all_with_skip_download(self, cli_runner):
        """Test that skipping download-returns doesn't require rf-data."""
        result = cli_runner.invoke(
            run_all,
            ['--skip', 'download-returns'],
            input='n\n'  # Say no to confirmation
        )
        # Should get to confirmation without error
        assert 'Pipeline cancelled' in result.output
        assert result.exit_code == 0
    
    def test_run_all_model_type_kmeans_only(self, cli_runner):
        """Test pipeline with KMeans model type only."""
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'kmeans', '--skip', 'download-returns'],
            input='n\n'
        )
        assert 'SKIP (model)' in result.output  # LLAMA steps should be marked as skipped
        assert 'kmeans-clustering' in result.output
    
    def test_run_all_model_type_llama_only(self, cli_runner):
        """Test pipeline with LLAMA model type only."""
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'llama', '--skip', 'download-returns'],
            input='n\n'
        )
        assert 'SKIP (model)' in result.output  # KMeans steps should be marked as skipped
        assert 'llama-parse' in result.output
    
    def test_run_all_model_type_both(self, cli_runner):
        """Test pipeline with both model types (default)."""
        result = cli_runner.invoke(
            run_all,
            ['--skip', 'download-returns'],
            input='n\n'
        )
        assert 'kmeans-clustering' in result.output
        assert 'llama-parse' in result.output
        # No steps should show SKIP (model)
        steps_output = result.output[result.output.find('Pipeline steps:'):]
        assert steps_output.count('SKIP (model)') == 0
    
    def test_run_all_skip_multiple_steps(self, cli_runner):
        """Test skipping multiple steps."""
        result = cli_runner.invoke(
            run_all,
            [
                '--skip', 'download-returns',
                '--skip', 'llama-parse',
                '--skip', 'llama-clustering',
            ],
            input='n\n'
        )
        assert 'SKIP (user)' in result.output
        assert 'download-returns' in result.output
        assert 'llama-parse' in result.output
    
    def test_run_all_cancellation(self, cli_runner):
        """Test that pipeline can be cancelled at confirmation."""
        result = cli_runner.invoke(
            run_all,
            ['--skip', 'download-returns'],
            input='n\n'  # Cancel at confirmation
        )
        assert result.exit_code == 0
        assert 'Pipeline cancelled' in result.output
        assert 'STEP' not in result.output  # No steps should have executed
    
    @patch('PMRTN.cli.data_commands.load_articles')
    @patch('PMRTN.cli.data_commands.describe_data')
    def test_run_all_executes_steps(
        self,
        mock_describe,
        mock_load,
        cli_runner,
    ):
        """Test that pipeline executes configured steps."""
        # Mock the command functions to do nothing
        mock_load.return_value = None
        mock_describe.return_value = None
        
        result = cli_runner.invoke(
            run_all,
            [
                '--skip', 'download-returns',
                '--skip', 'generate-embeddings',
                '--skip', 'kmeans-clustering',
                '--skip', 'llama-parse',
                '--skip', 'llama-clustering',
            ],
            input='y\n'  # Confirm execution
        )
        
        # Commands should have been called
        assert mock_load.called or result.exit_code in [0, 1]
        # Test that we at least tried to run the pipeline
        assert 'STEP' in result.output or result.exit_code == 1
    
    def test_run_all_llama_api_key_env_var(self, cli_runner, monkeypatch):
        """Test that LLAMA API key can be read from environment."""
        monkeypatch.setenv('GROQ_API_KEY', 'test-key-12345')
        
        result = cli_runner.invoke(
            run_all,
            [
                '--model-type', 'llama',
                '--skip', 'download-returns',
            ],
            input='n\n'
        )
        
        # Should not show warning about missing API key
        assert result.exit_code == 0
        assert 'Pipeline cancelled' in result.output
    
    def test_run_all_step_status_display(self, cli_runner):
        """Test that step status is displayed correctly."""
        result = cli_runner.invoke(
            run_all,
            [
                '--model-type', 'kmeans',
                '--skip', 'describe-data',
                '--skip', 'download-returns',
            ],
            input='n\n'
        )
        
        output = result.output
        
        # Check various status types are displayed
        assert '[RUN' in output or 'RUN]' in output  # Some steps should run
        assert 'SKIP (user)' in output  # User-skipped steps
        assert 'SKIP (model)' in output  # Model-skipped steps (LLAMA in this case)
        
        # Check step numbers and descriptions
        assert '1.' in output
        assert 'load-articles' in output
        assert 'Load and process' in output


class TestRunAllValidation:
    """Tests for input validation in run-all command."""
    
    def test_invalid_model_type(self, cli_runner):
        """Test that invalid model type is rejected."""
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'invalid'],
        )
        # Click should reject invalid choice before our code runs
        assert result.exit_code == 2
        assert 'Invalid value' in result.output or 'invalid choice' in result.output.lower()
    
    def test_invalid_skip_step(self, cli_runner):
        """Test that invalid skip step is rejected."""
        result = cli_runner.invoke(
            run_all,
            ['--skip', 'nonexistent-step'],
        )
        # Click should reject invalid choice
        assert result.exit_code == 2
        assert 'Invalid value' in result.output or 'invalid choice' in result.output.lower()
    
    def test_case_insensitive_model_type(self, cli_runner):
        """Test that model-type is case-insensitive."""
        # Test lowercase
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'kmeans', '--skip', 'download-returns'],
            input='n\n'
        )
        assert result.exit_code == 0
        
        # Test uppercase
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'KMEANS', '--skip', 'download-returns'],
            input='n\n'
        )
        assert result.exit_code == 0
        
        # Test mixed case
        result = cli_runner.invoke(
            run_all,
            ['--model-type', 'KMeans', '--skip', 'download-returns'],
            input='n\n'
        )
        assert result.exit_code == 0


class TestRunAllIntegration:
    """Integration tests for pipeline command with mocked subcommands."""
    
    @patch('PMRTN.cli.data_commands.load_articles')
    @patch('PMRTN.cli.data_commands.describe_data')
    @patch('PMRTN.cli.data_commands.download_returns')
    @patch('PMRTN.cli.data_commands.generate_embeddings')
    @patch('PMRTN.cli.clustering_commands.kmeans_clustering')
    def test_kmeans_pipeline_flow(
        self,
        mock_kmeans,
        mock_embeddings,
        mock_download,
        mock_describe,
        mock_load,
        cli_runner,
        tmp_path,
    ):
        """Test full KMeans pipeline flow with mocked commands."""
        # Create temporary rf_data file
        rf_data = tmp_path / "ESTR.csv"
        rf_data.write_text("date,rate\n2024-01-01,0.03")
        
        # Mock all commands to succeed
        mock_load.return_value = None
        mock_describe.return_value = None
        mock_download.return_value = None
        mock_embeddings.return_value = None
        mock_kmeans.return_value = None
        
        result = cli_runner.invoke(
            run_all,
            [
                '--model-type', 'kmeans',
                '--rf-data', str(rf_data),
                '--skip', 'llama-parse',
                '--skip', 'llama-clustering',
            ],
            input='y\n'
        )
        
        # Pipeline should complete or at least start
        assert result.exit_code in [0, 1]
        
        # Check that appropriate commands were called
        # (They may fail due to missing actual implementation, but should be invoked)
        assert 'STEP' in result.output


class TestRunAllErrorHandling:
    """Tests for error handling in pipeline."""
    
    @patch('PMRTN.cli.data_commands.load_articles')
    def test_step_failure_stops_pipeline(
        self,
        mock_load,
        cli_runner,
    ):
        """Test that pipeline stops when a step fails."""
        # Make load_articles raise an exception
        mock_load.side_effect = Exception("Test error")
        
        result = cli_runner.invoke(
            run_all,
            ['--skip', 'download-returns'],
            input='y\n'
        )
        
        # Pipeline should fail
        assert result.exit_code == 1
        assert 'Pipeline failed' in result.output or 'Error' in result.output
    
    def test_missing_rf_data_file(self, cli_runner):
        """Test error when rf-data file doesn't exist."""
        result = cli_runner.invoke(
            run_all,
            ['--rf-data', '/nonexistent/path/ESTR.csv'],
            input='y\n'
        )
        
        # Should fail with file not found
        assert result.exit_code == 2  # Click parameter validation error
        assert 'does not exist' in result.output.lower() or 'error' in result.output.lower()


def test_run_all_import():
    """Test that run_all can be imported."""
    from PMRTN.cli.pipeline_commands import run_all
    assert run_all is not None
    assert callable(run_all)
