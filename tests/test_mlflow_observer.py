"""Tests for MLflow Observer Plugin.

Tests the optional MLflow observer that integrates with CallbackManager
to provide automatic observability without requiring MLflow to be installed.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType


class TestMLflowObserver:
    """Test MLflow observer plugin functionality."""

    def test_observer_graceful_degradation_without_mlflow(self):
        """Test that observer gracefully handles missing MLflow package."""
        # Temporarily hide mlflow import
        with patch.dict('sys.modules', {'mlflow': None}):
            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Should not be available
            assert not observer.is_available()

            # Should not raise errors when handling events
            event = ExecutionEvent(
                event_type=ExecutionEventType.MODEL_CALL_START,
                metadata={"model_name": "gpt-4"}
            )
            observer.handle_event(event)  # Should do nothing, not crash

    def test_observer_disabled_without_env_var(self):
        """Test that observer is disabled when environment variable not set."""
        # Ensure env var is not set
        os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

        from promptchain.observability import MLflowObserver

        observer = MLflowObserver()

        # Should not be available without env var
        assert not observer.is_available()

    def test_observer_enabled_with_env_var(self):
        """Test that observer activates when environment variable is set."""
        # Mock MLflow to avoid actual server connection
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Mock MLflow functions
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"
            mock_mlflow.set_experiment.return_value = None

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Should be available with env var and mocked MLflow
            assert observer.is_available()

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_chain_start_creates_run(self):
        """Test that CHAIN_START event creates MLflow run."""
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value = mock_run

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Emit chain start event
            event = ExecutionEvent(
                event_type=ExecutionEventType.CHAIN_START,
                metadata={
                    "chain_id": "test-chain",
                    "model": "gpt-4"
                }
            )

            observer.handle_event(event)

            # Verify MLflow run was started
            mock_mlflow.start_run.assert_called_once()
            assert observer._active_run is not None

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_model_call_logs_tokens(self):
        """Test that MODEL_CALL_END logs token metrics."""
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value = mock_run

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Create active run
            chain_event = ExecutionEvent(
                event_type=ExecutionEventType.CHAIN_START,
                metadata={"chain_id": "test"}
            )
            observer.handle_event(chain_event)

            # Emit model call end event with token usage
            model_event = ExecutionEvent(
                event_type=ExecutionEventType.MODEL_CALL_END,
                metadata={
                    "call_id": "call-123",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150
                    }
                }
            )

            observer.handle_event(model_event)

            # Verify token metrics were logged
            mock_mlflow.log_metrics.assert_called()
            call_args = mock_mlflow.log_metrics.call_args[0][0]
            assert "tokens.prompt.call-123" in call_args
            assert call_args["tokens.prompt.call-123"] == 100
            assert call_args["tokens.completion.call-123"] == 50
            assert call_args["tokens.total.call-123"] == 150

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_tool_call_tracking(self):
        """Test that TOOL_CALL events are tracked."""
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value = mock_run

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Create active run
            chain_event = ExecutionEvent(
                event_type=ExecutionEventType.CHAIN_START,
                metadata={"chain_id": "test"}
            )
            observer.handle_event(chain_event)

            # Emit tool call start
            tool_start = ExecutionEvent(
                event_type=ExecutionEventType.TOOL_CALL_START,
                metadata={
                    "call_id": "tool-123",
                    "tool_name": "search_web",
                    "arguments": {"query": "AI trends"}
                }
            )
            observer.handle_event(tool_start)

            # Verify tool parameter logged
            mock_mlflow.log_param.assert_called()

            # Emit tool call end
            tool_end = ExecutionEvent(
                event_type=ExecutionEventType.TOOL_CALL_END,
                metadata={
                    "call_id": "tool-123",
                    "result": "Found 10 results"
                }
            )
            observer.handle_event(tool_end)

            # Verify duration metric logged
            mock_mlflow.log_metric.assert_called()

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_error_handling_doesnt_crash(self):
        """Test that errors in event handling don't crash the application."""
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"

            # Make log_param raise an error
            mock_mlflow.log_param.side_effect = Exception("MLflow error")

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Create active run
            chain_event = ExecutionEvent(
                event_type=ExecutionEventType.CHAIN_START,
                metadata={"chain_id": "test"}
            )

            # Should not raise exception
            observer.handle_event(chain_event)

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_shutdown_cleanup(self):
        """Test that shutdown properly cleans up resources."""
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value = mock_run

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Create active run
            chain_event = ExecutionEvent(
                event_type=ExecutionEventType.CHAIN_START,
                metadata={"chain_id": "test"}
            )
            observer.handle_event(chain_event)

            # Shutdown
            observer.shutdown()

            # Verify run was ended
            mock_mlflow.end_run.assert_called()
            assert observer._active_run is None
            assert not observer._initialized
            assert not observer._enabled

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_nested_step_runs(self):
        """Test that STEP_START creates nested runs."""
        with patch('promptchain.observability.mlflow_observer.mlflow') as mock_mlflow:
            # Setup mocks
            mock_mlflow.get_experiment_by_name.return_value = None
            mock_mlflow.create_experiment.return_value = "exp-123"
            mock_run = MagicMock()
            mock_run.info.run_id = "run-123"
            mock_mlflow.start_run.return_value = mock_run

            os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

            from promptchain.observability import MLflowObserver

            observer = MLflowObserver()

            # Create active run
            chain_event = ExecutionEvent(
                event_type=ExecutionEventType.CHAIN_START,
                metadata={"chain_id": "test"}
            )
            observer.handle_event(chain_event)

            # Emit step start
            step_event = ExecutionEvent(
                event_type=ExecutionEventType.STEP_START,
                step_number=0,
                step_instruction="Analyze input",
                metadata={"step_index": 0}
            )
            observer.handle_event(step_event)

            # Verify nested run was created
            calls = mock_mlflow.start_run.call_args_list
            assert len(calls) == 2  # Parent chain run + nested step run
            # Second call should be nested
            assert calls[1][1].get("nested") == True

            # Cleanup
            os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    def test_import_without_mlflow(self):
        """Test that importing MLflowObserver works without MLflow installed."""
        # This test verifies the import doesn't crash
        from promptchain.observability import MLflowObserver

        # Can create instance
        observer = MLflowObserver()

        # Just not available
        assert not observer.is_available()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
