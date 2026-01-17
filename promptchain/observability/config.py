"""
MLflow observability configuration module.

Manages configuration for MLflow integration with environment variables
and optional YAML file support.
"""
import os
from pathlib import Path
from typing import Optional

# Default configuration values
DEFAULT_MLFLOW_ENABLED = False
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT_NAME = "promptchain-cli"
DEFAULT_BACKGROUND_LOGGING = True

# Environment variable names
ENV_MLFLOW_ENABLED = "PROMPTCHAIN_MLFLOW_ENABLED"
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_EXPERIMENT_NAME = "PROMPTCHAIN_MLFLOW_EXPERIMENT"
ENV_BACKGROUND_LOGGING = "PROMPTCHAIN_MLFLOW_BACKGROUND"


def _load_yaml_config() -> dict:
    """Load configuration from YAML file if it exists.

    Checks for .promptchain.yml in current directory and home directory.

    Returns:
        Dictionary with configuration values, empty if no file found
    """
    config_locations = [
        Path.cwd() / ".promptchain.yml",
        Path.home() / ".promptchain.yml"
    ]

    for config_path in config_locations:
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    return config.get('mlflow', {})
            except ImportError:
                # YAML library not available, skip file config
                pass
            except Exception:
                # Ignore file parsing errors
                pass

    return {}


def _get_bool_env(key: str, default: bool) -> bool:
    """Parse boolean from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Boolean value from environment or default
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def is_enabled() -> bool:
    """Check if MLflow observability is enabled.

    Returns:
        True if PROMPTCHAIN_MLFLOW_ENABLED is set to true, False otherwise
    """
    yaml_config = _load_yaml_config()
    yaml_enabled = yaml_config.get('enabled', DEFAULT_MLFLOW_ENABLED)
    return _get_bool_env(ENV_MLFLOW_ENABLED, yaml_enabled)


def get_tracking_uri() -> str:
    """Get MLflow tracking URI.

    Returns:
        Tracking URI from environment or default
    """
    yaml_config = _load_yaml_config()
    yaml_uri = yaml_config.get('tracking_uri', DEFAULT_TRACKING_URI)
    return os.getenv(ENV_TRACKING_URI, yaml_uri)


def get_experiment_name() -> str:
    """Get MLflow experiment name.

    Returns:
        Experiment name from environment or default
    """
    yaml_config = _load_yaml_config()
    yaml_name = yaml_config.get('experiment_name', DEFAULT_EXPERIMENT_NAME)
    return os.getenv(ENV_EXPERIMENT_NAME, yaml_name)


def use_background_logging() -> bool:
    """Check if background logging should be used.

    Returns:
        True if background logging enabled, False for synchronous logging
    """
    yaml_config = _load_yaml_config()
    yaml_background = yaml_config.get('background_logging', DEFAULT_BACKGROUND_LOGGING)
    return _get_bool_env(ENV_BACKGROUND_LOGGING, yaml_background)
