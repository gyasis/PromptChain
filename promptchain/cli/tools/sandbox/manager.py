"""
Sandbox environment manager (singleton pattern).

Manages all active sandbox environments across the CLI session,
handling lifecycle, resource limits, and environment discovery.
"""

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    BaseSandbox,
    EnvironmentConfig,
    EnvironmentType,
    SandboxResult,
    ProvisionError,
    ExecutionError,
    CleanupError,
)
from .uv_sandbox import UVSandbox
from .docker_sandbox import DockerSandbox


class SandboxManager:
    """Singleton manager for all sandbox environments.

    Manages the lifecycle of all active sandbox environments,
    enforces resource limits, and provides cleanup mechanisms.

    Features:
    - Singleton pattern: One manager per CLI session
    - Resource limits: Max 10 environments per session
    - Automatic cleanup: Oldest environments removed when limit reached
    - Thread-safe: Protected with locks for concurrent access

    Example:
        >>> manager = SandboxManager()
        >>>
        >>> # Create UV environment
        >>> config = EnvironmentConfig(
        ...     env_type=EnvironmentType.UV,
        ...     packages=["pandas", "numpy"]
        ... )
        >>> env_id = manager.create_environment(EnvironmentType.UV, config)
        >>>
        >>> # Execute code
        >>> result = manager.execute_in_environment(
        ...     env_id,
        ...     "import pandas as pd; print(pd.__version__)"
        ... )
        >>>
        >>> # List environments
        >>> envs = manager.list_environments()
        >>>
        >>> # Cleanup
        >>> manager.cleanup_environment(env_id)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize manager (only once due to singleton)."""
        if self._initialized:
            return

        self.environments: Dict[str, BaseSandbox] = {}
        self.max_environments: int = 10
        self._initialized = True

    def create_environment(
        self,
        env_type: EnvironmentType,
        config: EnvironmentConfig
    ) -> str:
        """Create and provision new sandbox environment.

        Args:
            env_type: Type of environment to create (UV or DOCKER)
            config: Environment configuration

        Returns:
            str: Environment ID

        Raises:
            ProvisionError: If environment creation fails
            ValueError: If environment type is not supported
        """
        with self._lock:
            # Check if we're at capacity
            if len(self.environments) >= self.max_environments:
                self._cleanup_oldest()

            # Create appropriate sandbox instance
            if env_type == EnvironmentType.UV:
                sandbox = UVSandbox(config)
            elif env_type == EnvironmentType.DOCKER:
                sandbox = DockerSandbox(config)
            else:
                raise ValueError(
                    f"Unsupported environment type: {env_type}. "
                    f"Supported: {EnvironmentType.UV}, {EnvironmentType.DOCKER}"
                )

            # Provision the environment
            env_id = sandbox.provision()

            # Store in manager
            self.environments[env_id] = sandbox

            return env_id

    def execute_in_environment(
        self,
        env_id: str,
        code: str,
        timeout: Optional[int] = None
    ) -> SandboxResult:
        """Execute code in a specific environment.

        Args:
            env_id: Environment identifier
            code: Python code to execute
            timeout: Optional execution timeout (uses env default if None)

        Returns:
            SandboxResult: Execution result

        Raises:
            ValueError: If environment doesn't exist
            ExecutionError: If execution fails
        """
        sandbox = self._get_environment(env_id)
        return sandbox.execute(code, timeout=timeout)

    def list_environments(self) -> List[Dict]:
        """List all active environments with status.

        Returns:
            List of environment status dictionaries
        """
        with self._lock:
            envs = []
            for env_id, sandbox in self.environments.items():
                envs.append(sandbox.get_status())
            return envs

    def cleanup_environment(self, env_id: str) -> None:
        """Cleanup and remove a specific environment.

        Args:
            env_id: Environment identifier

        Raises:
            ValueError: If environment doesn't exist
            CleanupError: If cleanup fails
        """
        with self._lock:
            if env_id not in self.environments:
                raise ValueError(f"Environment not found: {env_id}")

            sandbox = self.environments[env_id]

            try:
                sandbox.cleanup()
            finally:
                # Remove from manager even if cleanup fails
                del self.environments[env_id]

    def cleanup_all(self) -> None:
        """Cleanup all environments.

        Used for graceful shutdown or reset.
        """
        with self._lock:
            env_ids = list(self.environments.keys())

        for env_id in env_ids:
            try:
                self.cleanup_environment(env_id)
            except Exception:
                pass  # Continue cleaning up other environments

    def get_environment_status(self, env_id: str) -> Dict:
        """Get status of a specific environment.

        Args:
            env_id: Environment identifier

        Returns:
            Dict with environment status

        Raises:
            ValueError: If environment doesn't exist
        """
        sandbox = self._get_environment(env_id)
        return sandbox.get_status()

    def _get_environment(self, env_id: str) -> BaseSandbox:
        """Get environment by ID (thread-safe).

        Args:
            env_id: Environment identifier

        Returns:
            BaseSandbox: Sandbox instance

        Raises:
            ValueError: If environment doesn't exist
        """
        with self._lock:
            if env_id not in self.environments:
                raise ValueError(
                    f"Environment not found: {env_id}. "
                    f"Available: {list(self.environments.keys())}"
                )
            return self.environments[env_id]

    def _cleanup_oldest(self) -> None:
        """Cleanup the oldest (least recently used) environment.

        Called when max_environments limit is reached.
        """
        if not self.environments:
            return

        # Find oldest environment (by last_used timestamp)
        oldest_env_id = min(
            self.environments.keys(),
            key=lambda eid: self.environments[eid].last_used or 0
        )

        # Cleanup oldest environment
        try:
            self.cleanup_environment(oldest_env_id)
        except Exception:
            pass  # Ignore cleanup errors

    def __repr__(self) -> str:
        """String representation of manager."""
        return (
            f"SandboxManager("
            f"active_environments={len(self.environments)}, "
            f"max_environments={self.max_environments})"
        )
