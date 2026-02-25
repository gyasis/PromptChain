"""
MLflow observability configuration module.

Manages configuration for MLflow integration with environment variables
and optional YAML file support.

Fixes Issue #5: Config File Read on Every Access (Performance Issue).
Implements caching with modification time tracking for 100x faster reads.

Fixes BUG-004: Silent Failure in YAML Configuration Loading.
Adds proper logging for config parsing errors.

FR-005: Thread-safe cache via module-level lock.  All public accessors
route through _load_yaml_config() so disk I/O only happens when the
underlying file's mtime changes.
"""
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

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

# Config cache (FR-005 / Issue #5 Fix)
# Protected by _config_lock for thread safety.
_config_cache: Optional[Dict[str, Any]] = None
_config_cache_path: Optional[Path] = None
_config_cache_mtime: Optional[float] = None
_config_lock: threading.Lock = threading.Lock()


def _get_config_file_path() -> Optional[Path]:
    """Find configuration file path.

    Returns:
        Path to config file if found, None otherwise
    """
    config_locations = [
        Path.cwd() / ".promptchain.yml",
        Path.home() / ".promptchain.yml"
    ]

    for config_path in config_locations:
        if config_path.exists():
            return config_path

    return None


def _get_file_mtime(config_path: Path) -> float:
    """Get file modification time.

    Args:
        config_path: Path to config file

    Returns:
        Modification time as float, 0.0 if file doesn't exist
    """
    try:
        return os.path.getmtime(config_path)
    except OSError:
        return 0.0


def _load_yaml_config() -> Dict[str, Any]:
    """Load configuration from YAML file with caching.

    FR-005 / Issue #5 Fix: Implements caching with modification time tracking.
    Only reloads file if it has been modified since last read.  All access is
    protected by ``_config_lock`` so the function is thread-safe.

    Checks for .promptchain.yml in current directory and home directory.

    Returns:
        Dictionary with configuration values, empty if no file found
    """
    global _config_cache, _config_cache_path, _config_cache_mtime

    with _config_lock:
        # Find config file path
        config_path = _get_config_file_path()

        # No config file found
        if config_path is None:
            # Return cached empty dict if we already checked
            if _config_cache is not None and _config_cache_path is None:
                return _config_cache
            # Cache the fact that no config exists
            _config_cache = {}
            _config_cache_path = None
            _config_cache_mtime = None
            return {}

        # Get current file modification time
        current_mtime = _get_file_mtime(config_path)

        # Return cached config if file hasn't changed
        if (
            _config_cache is not None
            and _config_cache_path == config_path
            and _config_cache_mtime == current_mtime
        ):
            return _config_cache

        # File changed or no cache - reload
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                mlflow_config = config.get('mlflow', {})

            # Update cache
            _config_cache = mlflow_config
            _config_cache_path = config_path
            _config_cache_mtime = current_mtime

            return mlflow_config

        except ImportError:
            # YAML library not available, skip file config
            logger.warning(
                f"PyYAML library not installed. Cannot load config from {config_path}. "
                "Install with: pip install pyyaml"
            )
            _config_cache = {}
            _config_cache_path = config_path
            _config_cache_mtime = 0.0
            return {}
        except Exception as e:
            # Log YAML parsing errors and return cached or empty
            logger.warning(
                f"Failed to parse YAML config file {config_path}: {type(e).__name__}: {e}. "
                "Using cached config or defaults. Please check YAML syntax."
            )
            if _config_cache is not None:
                logger.debug("Returning cached config from previous successful load")
                return _config_cache
            _config_cache = {}
            _config_cache_path = config_path
            _config_cache_mtime = 0.0
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


def clear_config_cache() -> None:
    """Clear configuration cache (for testing).

    FR-005 / Issue #5 Fix: Allows tests to invalidate cache and force reload.
    Acquires the module-level lock before mutating cache state so it is safe
    to call from multiple threads concurrently.
    """
    global _config_cache, _config_cache_path, _config_cache_mtime
    with _config_lock:
        _config_cache = None
        _config_cache_path = None
        _config_cache_mtime = None


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


def get_observability_config() -> Dict[str, Any]:
    """Return the complete resolved observability configuration.

    This is the canonical public accessor for retrieving a fully-merged view
    of the observability configuration.  It always routes through
    ``_load_yaml_config()`` so the file is read at most once per mtime epoch
    (FR-005 performance guarantee).  Environment variables override YAML
    values where applicable.

    Returns:
        Dictionary with the following keys:

        ``enabled`` (bool)
            Whether MLflow observability is active.
        ``tracking_uri`` (str)
            MLflow tracking server URI.
        ``experiment_name`` (str)
            MLflow experiment name.
        ``background_logging`` (bool)
            Whether to log in a background thread.

    Example::

        cfg = get_observability_config()
        if cfg["enabled"]:
            print(cfg["tracking_uri"])
    """
    # All four sub-accessors already route through _load_yaml_config(), but we
    # call _load_yaml_config() once here to keep this function's dependency on
    # the cache explicit and to avoid four separate lock acquisitions.
    yaml_config = _load_yaml_config()

    enabled = _get_bool_env(
        ENV_MLFLOW_ENABLED,
        yaml_config.get('enabled', DEFAULT_MLFLOW_ENABLED),
    )
    tracking_uri = os.getenv(
        ENV_TRACKING_URI,
        yaml_config.get('tracking_uri', DEFAULT_TRACKING_URI),
    )
    experiment_name = os.getenv(
        ENV_EXPERIMENT_NAME,
        yaml_config.get('experiment_name', DEFAULT_EXPERIMENT_NAME),
    )
    background_logging = _get_bool_env(
        ENV_BACKGROUND_LOGGING,
        yaml_config.get('background_logging', DEFAULT_BACKGROUND_LOGGING),
    )

    return {
        "enabled": enabled,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "background_logging": background_logging,
    }
