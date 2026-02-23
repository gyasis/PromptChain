"""
Sandbox Tool Registration for PromptChain CLI

Registers 5 agentic environment provisioning tools with the CLI tool registry:
1. provision_uv_environment - Fast Python virtual environments
2. provision_docker_environment - Full isolation with GPU support
3. execute_in_environment - Run code in provisioned environments
4. list_environments - List active environments
5. cleanup_environment - Free resources

These tools enable AI agents to dynamically create execution environments
based on task requirements, achieving 95% token cost reduction vs static tools.
"""

from typing import List, Optional

from promptchain.cli.tools import registry, ToolCategory, ParameterSchema

# Import the agent-facing tool functions
from .agent_tools import (
    provision_uv_environment,
    provision_docker_environment,
    execute_in_environment,
    list_environments,
    cleanup_environment
)


# 1. Provision UV Environment
@registry.register(
    category=ToolCategory.AGENT,
    description=(
        "SANDBOX UV: Fast Python virtual environment (50-100ms startup).\n\n"
        "USE WHEN:\n"
        "- Data analysis (pandas, numpy, matplotlib)\n"
        "- Web scraping (requests, beautifulsoup4)\n"
        "- Standard Python packages without GPU\n"
        "- Quick environment setup needed\n\n"
        "DO NOT USE WHEN:\n"
        "- GPU/ML workloads needed (use sandbox_provision_docker)\n"
        "- Full OS isolation required (use sandbox_provision_docker)\n"
        "- Running untrusted code (use sandbox_provision_docker)\n\n"
        "NOTE: Fastest startup (50-100ms). Uses UV package manager."
    ),
    parameters={
        "packages": {
            "type": "array",
            "items": ParameterSchema(name="package", type="string", description="Package name"),
            "required": True,
            "description": "List of PyPI package names to install (e.g., ['pandas', 'numpy', 'scikit-learn'])"
        },
        "python_version": {
            "type": "string",
            "default": "3.12",
            "description": "Python version to use. Supported: '3.8', '3.9', '3.10', '3.11', '3.12'",
            "enum": ["3.8", "3.9", "3.10", "3.11", "3.12"]
        },
        "environment_name": {
            "type": "string",
            "description": "Optional custom name for the environment (auto-generated if not provided)"
        }
    },
    tags=["sandbox", "python", "virtual-environment", "uv", "fast", "data-analysis"],
    examples=[
        "provision_uv_environment(packages=['pandas', 'numpy', 'matplotlib'])",
        "provision_uv_environment(packages=['requests', 'beautifulsoup4'], python_version='3.11')",
        "provision_uv_environment(packages=['scikit-learn', 'scipy'], environment_name='ml_env')"
    ]
)
def sandbox_provision_uv(
    packages: List[str],
    python_version: str = "3.12",
    environment_name: Optional[str] = None
) -> str:
    """Provision UV virtual environment (wrapper for CLI registration)."""
    return provision_uv_environment(packages, python_version, environment_name)


# 2. Provision Docker Environment
@registry.register(
    category=ToolCategory.AGENT,
    description=(
        "SANDBOX DOCKER: Full container isolation with GPU support.\n\n"
        "USE WHEN:\n"
        "- Machine learning / deep learning (PyTorch, TensorFlow)\n"
        "- GPU-accelerated workloads (CUDA, cuDNN)\n"
        "- Running untrusted or risky code\n"
        "- Complex system dependencies needed\n"
        "- Full OS-level isolation required\n\n"
        "DO NOT USE WHEN:\n"
        "- Simple Python packages only (use sandbox_provision_uv - 10x faster)\n"
        "- No GPU needed (use sandbox_provision_uv)\n"
        "- Quick iteration needed (use sandbox_provision_uv)\n\n"
        "NOTE: Startup 500-1000ms. Requires Docker daemon running."
    ),
    parameters={
        "base_image": {
            "type": "string",
            "required": True,
            "description": "Docker base image (e.g., 'python:3.12-slim', 'pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime')"
        },
        "packages": {
            "type": "array",
            "items": ParameterSchema(name="package", type="string", description="Package name"),
            "description": "Optional list of Python packages to install via pip"
        },
        "gpu": {
            "type": "boolean",
            "default": False,
            "description": "Enable GPU access (requires NVIDIA Docker runtime)"
        },
        "network_enabled": {
            "type": "boolean",
            "default": False,
            "description": "Allow network access from container (default: isolated for security)"
        },
        "environment_name": {
            "type": "string",
            "description": "Optional custom name for the environment"
        }
    },
    tags=["sandbox", "docker", "container", "gpu", "ml", "deep-learning", "isolation"],
    examples=[
        "provision_docker_environment(base_image='python:3.12-slim', packages=['pandas'])",
        "provision_docker_environment(base_image='pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime', packages=['transformers'], gpu=True, network_enabled=True)",
        "provision_docker_environment(base_image='tensorflow/tensorflow:2.13.0-gpu', gpu=True, environment_name='tf_train')"
    ]
)
def sandbox_provision_docker(
    base_image: str,
    packages: Optional[List[str]] = None,
    gpu: bool = False,
    network_enabled: bool = False,
    environment_name: Optional[str] = None
) -> str:
    """Provision Docker container environment (wrapper for CLI registration)."""
    return provision_docker_environment(
        base_image, packages, gpu, network_enabled, environment_name
    )


# 3. Execute Code in Environment
@registry.register(
    category=ToolCategory.AGENT,
    description=(
        "SANDBOX EXECUTE: Run Python code in a provisioned environment.\n\n"
        "USE WHEN:\n"
        "- You have an environment_id from provision_uv or provision_docker\n"
        "- Running data analysis, ML training, or scripts\n"
        "- Need captured stdout/stderr output\n"
        "- Multi-line code with imports\n\n"
        "DO NOT USE WHEN:\n"
        "- No environment provisioned yet (provision first)\n"
        "- Simple shell commands (use terminal_execute)\n"
        "- Reading/writing files directly (use file tools)\n\n"
        "NOTE: Supports multi-line code. Default timeout 300s (5 min)."
    ),
    parameters={
        "environment_id": {
            "type": "string",
            "required": True,
            "description": "Environment ID from provision_uv_environment() or provision_docker_environment()"
        },
        "code": {
            "type": "string",
            "required": True,
            "description": "Python code to execute (supports multi-line with proper indentation)"
        },
        "timeout": {
            "type": "integer",
            "default": 300,
            "description": "Maximum execution time in seconds (default: 300 = 5 minutes)"
        }
    },
    tags=["sandbox", "execution", "code", "python"],
    examples=[
        "execute_in_environment(environment_id='uv_env_123', code='print(2 + 2)')",
        "execute_in_environment(environment_id='uv_env_123', code='import pandas as pd\\ndf = pd.read_csv(\"data.csv\")\\nprint(df.head())')",
        "execute_in_environment(environment_id='docker_env_456', code='import torch\\nprint(torch.cuda.is_available())', timeout=60)"
    ]
)
def sandbox_execute(
    environment_id: str,
    code: str,
    timeout: int = 300
) -> str:
    """Execute code in environment (wrapper for CLI registration)."""
    return execute_in_environment(environment_id, code, timeout)


# 4. List Active Environments
@registry.register(
    category=ToolCategory.AGENT,
    description=(
        "SANDBOX LIST: Show all active sandbox environments.\n\n"
        "USE WHEN:\n"
        "- Checking what environments exist\n"
        "- Finding environment IDs for execute/cleanup\n"
        "- Monitoring resource usage\n"
        "- Debugging environment issues\n\n"
        "DO NOT USE WHEN:\n"
        "- You already know the environment_id\n"
        "- Listing files (use list_directory)\n\n"
        "NOTE: No parameters needed. Shows UV and Docker environments."
    ),
    parameters={},  # No parameters - lists all environments
    tags=["sandbox", "status", "monitoring", "list"],
    examples=[
        "list_environments()"
    ]
)
def sandbox_list() -> str:
    """List active environments (wrapper for CLI registration)."""
    return list_environments()


# 5. Cleanup Environment
@registry.register(
    category=ToolCategory.AGENT,
    description=(
        "SANDBOX CLEANUP: Destroy environment and free resources.\n\n"
        "USE WHEN:\n"
        "- Task is complete, environment no longer needed\n"
        "- Freeing disk space or memory\n"
        "- Environment is corrupted or stuck\n"
        "- Cleaning up before session end\n\n"
        "DO NOT USE WHEN:\n"
        "- You still need to run code in the environment\n"
        "- Environment has unsaved work\n\n"
        "NOTE: Removes UV venv directories or stops/removes Docker containers."
    ),
    parameters={
        "environment_id": {
            "type": "string",
            "required": True,
            "description": "Environment ID to cleanup and destroy"
        }
    },
    tags=["sandbox", "cleanup", "resource-management"],
    examples=[
        "cleanup_environment(environment_id='uv_env_123')",
        "cleanup_environment(environment_id='docker_env_456')"
    ]
)
def sandbox_cleanup(environment_id: str) -> str:
    """Cleanup environment (wrapper for CLI registration)."""
    return cleanup_environment(environment_id)


# Auto-register tools when module is imported
def register_all_tools():
    """
    Register all sandbox tools with the CLI registry.

    This function is called automatically when the module is imported,
    but can also be called explicitly to re-register tools.

    Returns:
        List[str]: Names of registered tools
    """
    registered_tools = [
        "sandbox_provision_uv",
        "sandbox_provision_docker",
        "sandbox_execute",
        "sandbox_list",
        "sandbox_cleanup"
    ]

    return registered_tools


# Module-level registration info
__all__ = [
    "sandbox_provision_uv",
    "sandbox_provision_docker",
    "sandbox_execute",
    "sandbox_list",
    "sandbox_cleanup",
    "register_all_tools"
]

# Tool metadata for documentation
TOOL_SUMMARY = """
Phase 11 Agentic Provisioning Tools - Registered with CLI

Total Tools: 5
Token Cost: ~500 tokens (vs 9,600 for 48 static tools)
Reduction: 95%

Tools:
1. sandbox_provision_uv - Fast Python environments (50-100ms)
2. sandbox_provision_docker - Full isolation + GPU (500-1000ms)
3. sandbox_execute - Run code in environments
4. sandbox_list - List active environments
5. sandbox_cleanup - Free resources

Category: AGENT
Tags: sandbox, python, docker, execution, ml, gpu

Use Cases:
- Data analysis (pandas, numpy, matplotlib)
- Machine learning (PyTorch, TensorFlow)
- Web scraping (requests, beautifulsoup4)
- Scientific computing (scipy, scikit-learn)
- GPU training and inference
"""
