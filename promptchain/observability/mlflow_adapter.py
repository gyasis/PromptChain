"""
MLflow API Wrapper with Graceful Degradation

Provides a safe interface to MLflow tracking that gracefully handles:
- MLflow package not being installed
- MLflow server being unavailable
- Connection errors and timeouts

All functions fail silently with warning logs rather than raising exceptions.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

if TYPE_CHECKING:
    from mlflow.entities import Run

# Attempt to import mlflow
MLFLOW_AVAILABLE = False
mlflow: Any = None

try:
    import mlflow  # type: ignore[no-redef]

    MLFLOW_AVAILABLE = True
except ImportError:
    logging.debug("MLflow not installed - tracking will be disabled")


logger = logging.getLogger(__name__)


def safe_import_mlflow() -> bool:
    """
    Check if MLflow is available and can be imported.

    Returns:
        bool: True if MLflow is available, False otherwise
    """
    return MLFLOW_AVAILABLE


def is_available() -> bool:
    """
    Alias for safe_import_mlflow() for backward compatibility.

    Returns:
        bool: True if MLflow is available, False otherwise
    """
    return MLFLOW_AVAILABLE


def initialize_mlflow(
    tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None
) -> bool:
    """
    Initialize MLflow with tracking URI and experiment.

    Args:
        tracking_uri: MLflow tracking server URI (default: ./mlruns)
        experiment_name: Name of the experiment (default: PromptChain)

    Returns:
        bool: True if initialization successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available - tracking disabled")
        return False

    try:
        # Import config here to avoid circular dependency
        from promptchain.observability.config import (get_experiment_name,
                                                      get_tracking_uri)

        uri = tracking_uri or get_tracking_uri()
        exp_name = experiment_name or get_experiment_name()

        mlflow.set_tracking_uri(uri)
        logger.info(f"MLflow tracking URI set to: {uri}")

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(exp_name)
                logger.info(
                    f"Created MLflow experiment: {exp_name} (ID: {experiment_id})"
                )
            else:
                logger.info(f"Using existing MLflow experiment: {exp_name}")
            mlflow.set_experiment(exp_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
            return False

        return True

    except Exception as e:
        logger.warning(f"MLflow initialization failed: {e}")
        return False


def set_experiment(experiment_name: str) -> bool:
    """
    Set the active MLflow experiment, creating it if it doesn't exist.

    Args:
        experiment_name: Name of the experiment

    Returns:
        bool: True if successful, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(
                f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})"
            )

        mlflow.set_experiment(experiment_name)
        return True

    except Exception as e:
        logger.warning(f"Failed to set MLflow experiment '{experiment_name}': {e}")
        return False


def start_run(run_name: Optional[str] = None, nested: bool = False) -> Optional[Run]:
    """
    Start a new MLflow run.

    Args:
        run_name: Name for the run
        nested: Whether this is a nested run (child of current active run)

    Returns:
        MLflow Run object if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        run = mlflow.start_run(run_name=run_name, nested=nested)
        logger.debug(
            f"Started MLflow run: {run.info.run_id}" + (f" (nested)" if nested else "")
        )
        return run

    except Exception as e:
        logger.warning(f"Failed to start MLflow run: {e}")
        return None


def end_run(status: str = "FINISHED") -> None:
    """
    End the current MLflow run.

    Args:
        status: Run status - "FINISHED", "FAILED", or "KILLED"
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.end_run(status=status)
        logger.debug(f"Ended MLflow run with status: {status}")

    except Exception as e:
        logger.warning(f"Failed to end MLflow run: {e}")


def log_param(key: str, value: Any) -> None:
    """
    Log a single parameter to the current MLflow run.

    Args:
        key: Parameter name
        value: Parameter value (will be converted to string)
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_param(key, value)

    except Exception as e:
        logger.warning(f"Failed to log MLflow param '{key}': {e}")


def log_params(params: Dict[str, Any]) -> None:
    """
    Log multiple parameters to the current MLflow run.

    Args:
        params: Dictionary of parameter names and values
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_params(params)

    except Exception as e:
        logger.warning(f"Failed to log MLflow params: {e}")


def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    """
    Log a single metric to the current MLflow run.

    Args:
        key: Metric name
        value: Metric value (must be numeric)
        step: Optional step number for time-series metrics
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_metric(key, value, step=step)

    except Exception as e:
        logger.warning(f"Failed to log MLflow metric '{key}': {e}")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log multiple metrics to the current MLflow run.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number for time-series metrics
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_metrics(metrics, step=step)

    except Exception as e:
        logger.warning(f"Failed to log MLflow metrics: {e}")


def set_tag(key: str, value: str) -> None:
    """
    Set a single tag on the current MLflow run.

    Args:
        key: Tag name
        value: Tag value
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_tag(key, value)

    except Exception as e:
        logger.warning(f"Failed to set MLflow tag '{key}': {e}")


def set_tags(tags: Dict[str, str]) -> None:
    """
    Set multiple tags on the current MLflow run.

    Args:
        tags: Dictionary of tag names and values
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_tags(tags)

    except Exception as e:
        logger.warning(f"Failed to set MLflow tags: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file as an artifact.

    Args:
        local_path: Path to the local file
        artifact_path: Optional subdirectory in artifact storage
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_artifact(local_path, artifact_path)

    except Exception as e:
        logger.warning(f"Failed to log MLflow artifact '{local_path}': {e}")


def log_text(text: str, artifact_file: str) -> None:
    """
    Log text content as an artifact file.

    Args:
        text: Text content to log
        artifact_file: Name of the artifact file
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_text(text, artifact_file)

    except Exception as e:
        logger.warning(f"Failed to log MLflow text artifact '{artifact_file}': {e}")


def log_dict(dictionary: Dict[str, Any], artifact_file: str) -> None:
    """
    Log a dictionary as a JSON artifact file.

    Args:
        dictionary: Dictionary to log
        artifact_file: Name of the artifact file
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.log_dict(dictionary, artifact_file)

    except Exception as e:
        logger.warning(f"Failed to log MLflow dict artifact '{artifact_file}': {e}")


@contextmanager
def run_context(
    run_name: Optional[str] = None, nested: bool = False
) -> Generator[Optional[Run], None, None]:
    """
    Context manager for MLflow runs.

    Automatically starts and ends a run, ensuring proper cleanup.

    Args:
        run_name: Name for the run
        nested: Whether this is a nested run

    Example:
        with run_context("my_experiment"):
            log_param("model", "gpt-4")
            log_metric("accuracy", 0.95)
    """
    run = start_run(run_name=run_name, nested=nested)
    try:
        yield run
        end_run(status="FINISHED")
    except Exception as e:
        logger.error(f"Error in MLflow run context: {e}")
        end_run(status="FAILED")
        raise
