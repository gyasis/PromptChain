"""
Sandbox environment management for agentic code execution.

This package provides dynamic environment provisioning for AI agents,
enabling them to create and manage their own execution environments
based on task requirements.

Components:
- base: Base abstractions (EnvironmentType, SandboxResult, EnvironmentConfig, BaseSandbox)
- uv_sandbox: UV-based Python virtual environment sandbox
- docker_sandbox: Docker container-based sandbox
- manager: Singleton environment manager
- agent_tools: Agent-facing tool functions for environment provisioning

Usage:
    from promptchain.cli.tools.sandbox import (
        EnvironmentType,
        EnvironmentConfig,
        SandboxManager
    )

    # Create environment manager
    manager = SandboxManager()

    # Provision UV environment
    config = EnvironmentConfig(
        env_type=EnvironmentType.UV,
        packages=["pandas", "numpy"]
    )
    env_id = manager.create_environment(EnvironmentType.UV, config)

    # Execute code
    result = manager.execute_in_environment(
        env_id,
        "import pandas as pd; print(pd.__version__)"
    )

    # Cleanup
    manager.cleanup_environment(env_id)
"""

from .base import (
    EnvironmentType,
    SandboxResult,
    EnvironmentConfig,
    BaseSandbox,
    SandboxError,
    ProvisionError,
    ExecutionError,
    CleanupError,
    SecurityError,
)

from .uv_sandbox import UVSandbox
from .docker_sandbox import DockerSandbox
from .manager import SandboxManager

# Import agent-facing tools
from .agent_tools import (
    provision_uv_environment,
    provision_docker_environment,
    execute_in_environment,
    list_environments,
    cleanup_environment,
)

# Import and auto-register CLI tools
# This triggers @registry.register() decorators when package is imported
try:
    from . import registration  # noqa: F401
except ImportError:
    # Graceful fallback if CLI tools dependencies are not available
    pass

__all__ = [
    # Core abstractions
    "EnvironmentType",
    "SandboxResult",
    "EnvironmentConfig",
    "BaseSandbox",

    # Sandbox implementations
    "UVSandbox",
    "DockerSandbox",

    # Manager
    "SandboxManager",

    # Agent-facing tools (5 functions)
    "provision_uv_environment",
    "provision_docker_environment",
    "execute_in_environment",
    "list_environments",
    "cleanup_environment",

    # Exceptions
    "SandboxError",
    "ProvisionError",
    "ExecutionError",
    "CleanupError",
    "SecurityError",
]

# Version will be set by implementations
__version__ = "0.1.0"
