"""
MLflow Observer Plugin for CallbackManager

Optional observability plugin that listens to CallbackManager events and logs
them to MLflow. This provides automatic MLflow integration without requiring
decorators or code changes.

USAGE:
    1. Install MLflow: pip install mlflow
    2. Set environment variable: PROMPTCHAIN_MLFLOW_ENABLED=true
    3. Start MLflow server: mlflow ui --port 5000
    4. Observer auto-activates when conditions are met

ARCHITECTURE:
    - Plugin design: Works without MLflow installed
    - Event-driven: Listens to CallbackManager events (same as ObservePanel)
    - Zero configuration: Auto-detects MLflow availability
    - Graceful degradation: Silently disabled if unavailable
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from ..utils.execution_events import ExecutionEvent, ExecutionEventType

logger = logging.getLogger(__name__)

# Check MLflow availability
_MLFLOW_AVAILABLE = False
try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    logger.debug("MLflow not installed - observer disabled")


class MLflowObserver:
    """
    Optional observer that logs CallbackManager events to MLflow.

    This is a PLUGIN - it requires:
    1. MLflow installed: pip install mlflow
    2. Environment variable: PROMPTCHAIN_MLFLOW_ENABLED=true
    3. MLflow server running (optional): mlflow ui --port 5000

    Features:
    - Automatic tracking of LLM calls with token metrics
    - Tool execution tracking with timing
    - Chain execution flow visualization
    - Nested run support for multi-step workflows
    - Graceful fallback if MLflow unavailable

    Example:
        >>> observer = MLflowObserver(experiment_name="my-app")
        >>> if observer.is_available():
        ...     chain.register_callback(observer.handle_event)
        ...     # Events now logged to MLflow automatically
    """

    def __init__(
        self,
        experiment_name: str = "promptchain-cli",
        tracking_uri: str = "http://localhost:5000",
        enabled_env_var: str = "PROMPTCHAIN_MLFLOW_ENABLED",
        auto_log_artifacts: bool = True,
    ):
        """Initialize MLflow observer with configuration.

        Args:
            experiment_name: MLflow experiment name for tracking runs
            tracking_uri: MLflow tracking server URI (can be local directory or HTTP)
            enabled_env_var: Environment variable to check for activation
            auto_log_artifacts: Automatically log prompts/responses as artifacts
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.auto_log_artifacts = auto_log_artifacts
        self._initialized = False
        self._active_run: Optional[Any] = None
        self._step_runs: Dict[str, Any] = {}  # Track nested runs for steps
        self._run_start_times: Dict[str, float] = (
            {}
        )  # Track timing for duration metrics

        # Check if observer should be enabled
        self._enabled = _MLFLOW_AVAILABLE and os.environ.get(
            enabled_env_var, "false"
        ).lower() in ("true", "1", "yes")

        if not _MLFLOW_AVAILABLE:
            logger.debug("MLflow not available - install with: pip install mlflow")
            return

        if not self._enabled:
            logger.debug(
                f"MLflow observer disabled - set {enabled_env_var}=true to enable"
            )
            return

        # Initialize MLflow
        self._initialize_mlflow()

    def _initialize_mlflow(self) -> None:
        """Initialize MLflow tracking configuration."""
        if not self._enabled:
            return

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI: {self.tracking_uri}")

            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(
                        f"Created MLflow experiment: {self.experiment_name} "
                        f"(ID: {experiment_id})"
                    )
                else:
                    logger.info(
                        f"Using existing MLflow experiment: {self.experiment_name}"
                    )

                mlflow.set_experiment(self.experiment_name)
                self._initialized = True
                logger.info("✓ MLflow observer initialized successfully")

            except Exception as e:
                logger.warning(f"Failed to set MLflow experiment: {e}")
                self._enabled = False

        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self._enabled = False

    def is_available(self) -> bool:
        """Check if MLflow observer is available and enabled.

        Returns:
            True if MLflow is installed, enabled, and initialized
        """
        return self._enabled and self._initialized

    def handle_event(self, event: ExecutionEvent) -> None:
        """Handle callback event and log to MLflow.

        This is the main entry point for CallbackManager integration.
        Routes events to appropriate handlers based on event type.

        Args:
            event: ExecutionEvent from CallbackManager
        """
        if not self.is_available():
            return

        try:
            # Route to specific handlers based on event type
            if event.event_type == ExecutionEventType.CHAIN_START:
                self._handle_chain_start(event)
            elif event.event_type == ExecutionEventType.CHAIN_END:
                self._handle_chain_end(event)
            elif event.event_type == ExecutionEventType.CHAIN_ERROR:
                self._handle_chain_error(event)

            elif event.event_type == ExecutionEventType.MODEL_CALL_START:
                self._handle_model_call_start(event)
            elif event.event_type == ExecutionEventType.MODEL_CALL_END:
                self._handle_model_call_end(event)
            elif event.event_type == ExecutionEventType.MODEL_CALL_ERROR:
                self._handle_model_call_error(event)

            elif event.event_type == ExecutionEventType.TOOL_CALL_START:
                self._handle_tool_call_start(event)
            elif event.event_type == ExecutionEventType.TOOL_CALL_END:
                self._handle_tool_call_end(event)
            elif event.event_type == ExecutionEventType.TOOL_CALL_ERROR:
                self._handle_tool_call_error(event)

            elif event.event_type == ExecutionEventType.STEP_START:
                self._handle_step_start(event)
            elif event.event_type == ExecutionEventType.STEP_END:
                self._handle_step_end(event)

        except Exception as e:
            # Don't let MLflow errors break the application
            logger.warning(f"MLflow event handling error: {e}")

    def _handle_chain_start(self, event: ExecutionEvent) -> None:
        """Handle chain execution start - creates parent MLflow run."""
        if self._active_run is not None:
            logger.warning(
                "Chain start event while run already active - ending previous"
            )
            self._end_active_run(status="KILLED")

        # Start new MLflow run for chain
        run_name = event.metadata.get("chain_id", "chain-execution")
        self._active_run = mlflow.start_run(run_name=run_name)

        # Log chain parameters
        mlflow.log_param("event_type", "chain")
        if event.model_name:
            mlflow.log_param("model", event.model_name)

        # Log chain metadata as tags
        for key, value in event.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.set_tag(f"chain.{key}", str(value))

        logger.debug(f"Started MLflow run: {self._active_run.info.run_id}")

    def _handle_chain_end(self, event: ExecutionEvent) -> None:
        """Handle chain execution end - closes parent MLflow run."""
        # Log final metrics
        if "execution_time_ms" in event.metadata:
            mlflow.log_metric("chain.duration_ms", event.metadata["execution_time_ms"])

        if "total_tokens" in event.metadata:
            mlflow.log_metric("chain.total_tokens", event.metadata["total_tokens"])

        self._end_active_run(status="FINISHED")

    def _handle_chain_error(self, event: ExecutionEvent) -> None:
        """Handle chain execution error."""
        # Log error details
        if "error" in event.metadata:
            mlflow.set_tag("error", str(event.metadata["error"]))

        self._end_active_run(status="FAILED")

    def _handle_model_call_start(self, event: ExecutionEvent) -> None:
        """Handle LLM call start - track request timing."""
        call_id = event.metadata.get("call_id", f"model-{event.timestamp.timestamp()}")
        self._run_start_times[call_id] = datetime.now()

        # Log model parameters if active run
        if self._active_run:
            model = event.metadata.get("model_name", "unknown")
            mlflow.log_param(f"model.{call_id}", model)

            # Log prompt as artifact if enabled
            if self.auto_log_artifacts and "messages" in event.metadata:
                messages = event.metadata["messages"]
                prompt_text = "\n\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
                )
                mlflow.log_text(prompt_text, f"prompts/{call_id}.txt")

    def _handle_model_call_end(self, event: ExecutionEvent) -> None:
        """Handle LLM call end - track tokens and timing."""
        if not self._active_run:
            return

        call_id = event.metadata.get("call_id", f"model-{event.timestamp.timestamp()}")

        # Calculate duration
        if call_id in self._run_start_times:
            duration_ms = (
                datetime.now() - self._run_start_times[call_id]
            ).total_seconds() * 1000
            mlflow.log_metric(f"model.duration_ms.{call_id}", duration_ms)
            del self._run_start_times[call_id]

        # Log token usage
        usage = event.metadata.get("usage", {})
        if usage:
            mlflow.log_metrics(
                {
                    f"tokens.prompt.{call_id}": usage.get("prompt_tokens", 0),
                    f"tokens.completion.{call_id}": usage.get("completion_tokens", 0),
                    f"tokens.total.{call_id}": usage.get("total_tokens", 0),
                }
            )

        # Log response as artifact if enabled
        if self.auto_log_artifacts and "response" in event.metadata:
            response = event.metadata["response"]
            mlflow.log_text(str(response), f"responses/{call_id}.txt")

    def _handle_model_call_error(self, event: ExecutionEvent) -> None:
        """Handle LLM call error."""
        if self._active_run:
            error_msg = event.metadata.get("error", "Unknown error")
            mlflow.set_tag("model.error", str(error_msg))

    def _handle_tool_call_start(self, event: ExecutionEvent) -> None:
        """Handle tool call start - track timing."""
        call_id = event.metadata.get("call_id", f"tool-{event.timestamp.timestamp()}")
        self._run_start_times[call_id] = datetime.now()

        if self._active_run:
            tool_name = event.metadata.get("tool_name", "unknown")
            mlflow.log_param(f"tool.{call_id}", tool_name)

            # Log tool arguments as artifact if enabled
            if self.auto_log_artifacts and "arguments" in event.metadata:
                args = event.metadata["arguments"]
                mlflow.log_dict(args, f"tool_args/{call_id}.json")

    def _handle_tool_call_end(self, event: ExecutionEvent) -> None:
        """Handle tool call end - track timing and results."""
        if not self._active_run:
            return

        call_id = event.metadata.get("call_id", f"tool-{event.timestamp.timestamp()}")

        # Calculate duration
        if call_id in self._run_start_times:
            duration_ms = (
                datetime.now() - self._run_start_times[call_id]
            ).total_seconds() * 1000
            mlflow.log_metric(f"tool.duration_ms.{call_id}", duration_ms)
            del self._run_start_times[call_id]

        # Log tool result as artifact if enabled
        if self.auto_log_artifacts and "result" in event.metadata:
            result = event.metadata["result"]
            mlflow.log_text(str(result), f"tool_results/{call_id}.txt")

    def _handle_tool_call_error(self, event: ExecutionEvent) -> None:
        """Handle tool call error."""
        if self._active_run:
            error_msg = event.metadata.get("error", "Unknown error")
            tool_name = event.metadata.get("tool_name", "unknown")
            mlflow.set_tag(f"tool.error.{tool_name}", str(error_msg))

    def _handle_step_start(self, event: ExecutionEvent) -> None:
        """Handle chain step start - create nested run for step."""
        if not self._active_run:
            return

        step_num = event.step_number
        if step_num is None:
            return

        # Create nested run for this step
        step_run = mlflow.start_run(run_name=f"step-{step_num}", nested=True)
        self._step_runs[step_num] = step_run

        # Log step parameters
        mlflow.log_param("step_number", step_num)
        if event.step_instruction:
            mlflow.log_param("instruction", event.step_instruction[:100])

    def _handle_step_end(self, event: ExecutionEvent) -> None:
        """Handle chain step end - close nested run."""
        step_num = event.step_number
        if step_num is None or step_num not in self._step_runs:
            return

        # Log step duration if available
        if "execution_time_ms" in event.metadata:
            mlflow.log_metric("duration_ms", event.metadata["execution_time_ms"])

        # End nested step run
        mlflow.end_run(status="FINISHED")
        del self._step_runs[step_num]

    def _end_active_run(self, status: str = "FINISHED") -> None:
        """End the currently active MLflow run.

        Args:
            status: Run status - "FINISHED", "FAILED", or "KILLED"
        """
        if self._active_run is None:
            return

        try:
            # Close any open nested runs first
            for step_num in list(self._step_runs.keys()):
                mlflow.end_run(status="KILLED")
                del self._step_runs[step_num]

            # Close parent run
            mlflow.end_run(status=status)
            logger.debug(f"Ended MLflow run with status: {status}")

        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")

        finally:
            self._active_run = None
            self._run_start_times.clear()

    def shutdown(self) -> None:
        """Cleanup MLflow resources.

        Call this when shutting down the observer to ensure proper cleanup.
        """
        if not self.is_available():
            return

        try:
            # End any active runs
            self._end_active_run(status="KILLED")
            logger.info("MLflow observer shutdown complete")

        except Exception as e:
            logger.warning(f"Error during MLflow observer shutdown: {e}")

        finally:
            self._initialized = False
            self._enabled = False

    @contextmanager
    def temporary_run(self, run_name: str = "temporary"):
        """Context manager for temporary MLflow runs.

        Useful for one-off tracking without affecting the main run lifecycle.

        Args:
            run_name: Name for the temporary run

        Example:
            >>> with observer.temporary_run("test-run"):
            ...     mlflow.log_metric("test_metric", 42)
        """
        if not self.is_available():
            yield None
            return

        run = mlflow.start_run(run_name=run_name)
        try:
            yield run
            mlflow.end_run(status="FINISHED")
        except Exception as e:
            logger.error(f"Error in temporary MLflow run: {e}")
            mlflow.end_run(status="FAILED")
            raise
