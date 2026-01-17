"""
Base abstractions for sandbox environment management.

This module provides the foundation for agentic environment provisioning,
enabling AI agents to dynamically create and manage their own execution
environments based on task requirements.

Components:
- EnvironmentType: Enum of available sandbox types
- SandboxResult: Execution result data structure
- EnvironmentConfig: Environment configuration
- BaseSandbox: Abstract base class for all sandbox implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


class EnvironmentType(Enum):
    """Types of sandbox environments available for code execution.

    Attributes:
        UV: UV-based Python virtual environment (fast, Python-level isolation)
        DOCKER: Docker container (secure, full OS-level isolation, GPU support)
        DIRECT: No isolation (uses existing file_operations tools directly)
    """
    UV = "uv"
    DOCKER = "docker"
    DIRECT = "direct"

    def __str__(self) -> str:
        return self.value

    @property
    def typical_startup_ms(self) -> int:
        """Typical environment startup time in milliseconds."""
        return {
            EnvironmentType.UV: 75,        # 50-100ms
            EnvironmentType.DOCKER: 750,   # 500-1000ms
            EnvironmentType.DIRECT: 1      # <10ms
        }[self]

    @property
    def isolation_level(self) -> str:
        """Description of isolation level provided."""
        return {
            EnvironmentType.UV: "Python environment (process-level)",
            EnvironmentType.DOCKER: "Full container (OS-level)",
            EnvironmentType.DIRECT: "None (trusted tools only)"
        }[self]


@dataclass
class SandboxResult:
    """Result of code execution in a sandbox environment.

    Captures all aspects of execution including success/failure, outputs,
    timing, and resource usage.

    Attributes:
        success: Whether code executed successfully (exit code 0)
        return_value: Last expression value or explicit return (if captured)
        stdout: Captured standard output
        stderr: Captured standard error
        execution_time: Wall-clock execution time in seconds
        error: Error message if execution failed
        timeout: Whether execution exceeded timeout limit
        memory_used_mb: Peak memory usage in megabytes
        exit_code: Process exit code (0 = success)
        metadata: Additional execution metadata
    """
    success: bool
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None
    timeout: bool = False
    memory_used_mb: float = 0.0
    exit_code: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result consistency."""
        # Ensure success matches exit_code
        if self.success and self.exit_code != 0:
            self.success = False

        # Set error message if failed but no error provided
        if not self.success and not self.error and self.stderr:
            self.error = self.stderr[:500]  # Truncate long errors

    @property
    def formatted_output(self) -> str:
        """Formatted string representation of execution result.

        Returns human-readable summary suitable for agent responses.
        """
        if not self.success:
            return (
                f"❌ Execution Failed ({self.execution_time:.2f}s)\n\n"
                f"Error: {self.error}\n\n"
                f"Stderr:\n{self.stderr[:1000]}"
            )

        parts = [f"✅ Execution Successful ({self.execution_time:.2f}s)"]

        if self.stdout:
            parts.append(f"\nOutput:\n{self.stdout}")

        if self.return_value is not None:
            parts.append(f"\nReturn Value:\n{self.return_value}")

        if self.memory_used_mb > 0:
            parts.append(f"\nMemory Used: {self.memory_used_mb:.1f} MB")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "error": self.error,
            "timeout": self.timeout,
            "memory_used_mb": self.memory_used_mb,
            "exit_code": self.exit_code,
            "metadata": self.metadata
        }


@dataclass
class EnvironmentConfig:
    """Configuration for a sandbox environment.

    Contains all parameters needed to provision and configure an
    execution environment.

    Attributes:
        env_id: Unique identifier for this environment (auto-generated if None)
        env_type: Type of environment (UV, DOCKER, DIRECT)
        python_version: Python version to use (e.g., "3.12", "3.11")
        packages: List of packages to install (e.g., ["pandas", "numpy"])
        memory_limit: Memory limit string (e.g., "512MB", "2GB")
        timeout: Default execution timeout in seconds
        gpu: Whether to enable GPU access (Docker only)
        network_enabled: Whether to allow network access
        working_dir: Working directory for code execution
        env_vars: Environment variables to set

        # UV-specific
        uv_path: Path to UV virtual environment (auto-created if None)

        # Docker-specific
        base_image: Docker base image (e.g., "python:3.12-slim")
        container_name: Docker container name (auto-generated if None)
        docker_volumes: Volume mounts for Docker container
    """
    env_id: Optional[str] = None
    env_type: EnvironmentType = EnvironmentType.UV
    python_version: str = "3.12"
    packages: List[str] = field(default_factory=list)
    memory_limit: str = "512MB"
    timeout: int = 300  # 5 minutes default
    gpu: bool = False
    network_enabled: bool = False
    working_dir: Optional[Path] = None
    env_vars: Dict[str, str] = field(default_factory=dict)

    # UV-specific
    uv_path: Optional[Path] = None

    # Docker-specific
    base_image: Optional[str] = None
    container_name: Optional[str] = None
    docker_volumes: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure env_type is EnvironmentType enum
        if isinstance(self.env_type, str):
            self.env_type = EnvironmentType(self.env_type)

        # Convert string paths to Path objects
        if self.working_dir and isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)

        if self.uv_path and isinstance(self.uv_path, str):
            self.uv_path = Path(self.uv_path)

        # Validate GPU requirement
        if self.gpu and self.env_type != EnvironmentType.DOCKER:
            raise ValueError("GPU support only available with Docker environments")

        # Set default base_image for Docker if not specified
        if self.env_type == EnvironmentType.DOCKER and not self.base_image:
            if self.gpu:
                self.base_image = "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
            else:
                self.base_image = f"python:{self.python_version}-slim"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "env_id": self.env_id,
            "env_type": str(self.env_type),
            "python_version": self.python_version,
            "packages": self.packages,
            "memory_limit": self.memory_limit,
            "timeout": self.timeout,
            "gpu": self.gpu,
            "network_enabled": self.network_enabled,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "env_vars": self.env_vars,
            "uv_path": str(self.uv_path) if self.uv_path else None,
            "base_image": self.base_image,
            "container_name": self.container_name,
            "docker_volumes": self.docker_volumes
        }


class BaseSandbox(ABC):
    """Abstract base class for all sandbox implementations.

    Defines the interface that all sandbox types (UV, Docker, etc.)
    must implement for consistent environment management.

    Lifecycle:
        1. __init__: Create sandbox instance with configuration
        2. provision(): Provision the environment (create venv/container)
        3. execute(): Run code in the environment (multiple times)
        4. cleanup(): Destroy the environment and free resources

    Attributes:
        config: Environment configuration
        env_id: Unique identifier for this environment
        created_at: Timestamp when environment was created
        last_used: Timestamp of last code execution
    """

    def __init__(self, config: EnvironmentConfig):
        """Initialize sandbox with configuration.

        Args:
            config: Environment configuration
        """
        self.config = config
        self.env_id: Optional[str] = None
        self.created_at: Optional[float] = None
        self.last_used: Optional[float] = None
        self._is_provisioned: bool = False

    @abstractmethod
    def provision(self) -> str:
        """Provision the environment.

        Creates and initializes the execution environment. This may involve:
        - Creating a virtual environment (UV)
        - Starting a container (Docker)
        - Installing packages
        - Setting up working directory

        Returns:
            str: Environment ID (unique identifier)

        Raises:
            ProvisionError: If environment creation fails
            TimeoutError: If provisioning exceeds timeout
        """
        pass

    @abstractmethod
    def execute(
        self,
        code: str,
        timeout: Optional[int] = None,
        capture_output: bool = True
    ) -> SandboxResult:
        """Execute Python code in the environment.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (uses config default if None)
            capture_output: Whether to capture stdout/stderr

        Returns:
            SandboxResult: Execution result with outputs and metadata

        Raises:
            ExecutionError: If execution fails
            TimeoutError: If execution exceeds timeout
            SecurityError: If code violates security policy
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Destroy the environment and free resources.

        Performs complete cleanup including:
        - Removing virtual environments
        - Stopping and removing containers
        - Deleting temporary files
        - Freeing system resources

        Raises:
            CleanupError: If cleanup fails
        """
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if environment is still available and functional.

        Returns:
            bool: True if environment is ready for code execution
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get current environment status and metadata.

        Returns:
            Dict with status information including:
            - env_id: Environment identifier
            - env_type: Environment type
            - is_alive: Whether environment is functional
            - created_at: Creation timestamp
            - last_used: Last execution timestamp
            - uptime: Seconds since creation
            - idle_time: Seconds since last execution
        """
        now = time.time()

        return {
            "env_id": self.env_id,
            "env_type": str(self.config.env_type),
            "is_alive": self.is_alive(),
            "created_at": self.created_at,
            "last_used": self.last_used,
            "uptime": now - self.created_at if self.created_at else 0,
            "idle_time": now - self.last_used if self.last_used else 0,
            "config": self.config.to_dict()
        }

    def _update_timestamps(self, operation: str = "execute") -> None:
        """Update environment timestamps.

        Args:
            operation: Type of operation ("provision", "execute")
        """
        now = time.time()

        if operation == "provision":
            self.created_at = now
            self.last_used = now
        elif operation == "execute":
            self.last_used = now

    def __repr__(self) -> str:
        """String representation of sandbox."""
        return (
            f"{self.__class__.__name__}("
            f"env_id={self.env_id!r}, "
            f"env_type={self.config.env_type}, "
            f"is_alive={self.is_alive()})"
        )


# Custom exceptions for sandbox operations

class SandboxError(Exception):
    """Base exception for all sandbox-related errors."""
    pass


class ProvisionError(SandboxError):
    """Error during environment provisioning."""
    pass


class ExecutionError(SandboxError):
    """Error during code execution."""
    pass


class CleanupError(SandboxError):
    """Error during environment cleanup."""
    pass


class SecurityError(SandboxError):
    """Security policy violation detected."""
    pass
