"""
Agent-facing tools for dynamic environment provisioning.

Provides 5 high-level functions that AI agents can use to:
1. Provision UV environments (fast, Python-level isolation)
2. Provision Docker environments (secure, full isolation, GPU support)
3. Execute code in provisioned environments
4. List active environments
5. Cleanup environments

These tools replace 48 static tool schemas (9,600 tokens) with
5 environment management tools (500 tokens) while providing
unlimited flexibility for package installation and execution.
"""

from typing import List, Optional

from .base import EnvironmentConfig, EnvironmentType
from .manager import SandboxManager


def provision_uv_environment(
    packages: List[str],
    python_version: str = "3.12",
    environment_name: Optional[str] = None
) -> str:
    """Provision a UV virtual environment with specified packages.

    **When to use:**
    - Data analysis (pandas, numpy, matplotlib, seaborn)
    - Scientific computing (scipy, scikit-learn, statsmodels)
    - Standard Python packages (requests, beautifulsoup4, etc.)
    - Fast iteration needed (50-100ms startup)

    **Performance:**
    - Provision: 50-100ms
    - Execution overhead: <200ms
    - Isolation: Python environment level (process-level)

    **Examples:**
        # Data analysis
        env_id = provision_uv_environment(
            packages=["pandas", "numpy", "matplotlib"],
            python_version="3.12"
        )

        # Web scraping
        env_id = provision_uv_environment(
            packages=["requests", "beautifulsoup4", "lxml"],
            environment_name="web_scraper"
        )

    Args:
        packages: List of PyPI package names to install
                 (e.g., ["pandas", "numpy", "scikit-learn"])
        python_version: Python version to use (default: "3.12")
                       Supported: "3.8", "3.9", "3.10", "3.11", "3.12"
        environment_name: Optional custom name for the environment
                         (auto-generated if not specified)

    Returns:
        str: Environment ID for use with execute_in_environment()
             Format: "✅ UV environment created: {env_id}\nPackages: {packages}"

    Raises:
        ProvisionError: If environment creation or package installation fails

    Performance:
        Typical: 75ms provision + 30s package installation (depends on packages)
        Best case: 50ms (no packages)
        Worst case: 2 minutes (large packages like torch, tensorflow)
    """
    # Create environment configuration
    config = EnvironmentConfig(
        env_id=environment_name,
        env_type=EnvironmentType.UV,
        python_version=python_version,
        packages=packages
    )

    # Get manager and create environment
    manager = SandboxManager()
    env_id = manager.create_environment(EnvironmentType.UV, config)

    # Format success message
    packages_str = ", ".join(packages) if packages else "none"
    return (
        f"✅ UV environment created: {env_id}\n"
        f"Python: {python_version}\n"
        f"Packages: {packages_str}\n"
        f"Use execute_in_environment('{env_id}', code) to run code in this environment."
    )


def provision_docker_environment(
    base_image: str,
    packages: Optional[List[str]] = None,
    gpu: bool = False,
    network_enabled: bool = False,
    environment_name: Optional[str] = None
) -> str:
    """Provision a Docker container environment.

    **When to use:**
    - Machine Learning / Deep Learning (PyTorch, TensorFlow, JAX)
    - GPU workloads (training models, inference at scale)
    - Complex dependencies (system packages, compiled libraries)
    - Untrusted code execution (full isolation)
    - Network-isolated analysis

    **Performance:**
    - Provision: 500-1000ms
    - Execution overhead: <200ms
    - Isolation: Full container (OS-level)

    **Common base images:**
        # Standard Python
        "python:3.12-slim"

        # PyTorch with GPU
        "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"

        # TensorFlow with GPU
        "tensorflow/tensorflow:2.13.0-gpu"

        # Jupyter environment
        "jupyter/scipy-notebook:latest"

    **Examples:**
        # PyTorch with GPU for training
        env_id = provision_docker_environment(
            base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
            packages=["transformers", "datasets", "accelerate"],
            gpu=True,
            network_enabled=True  # For downloading models
        )

        # TensorFlow for inference (no GPU)
        env_id = provision_docker_environment(
            base_image="tensorflow/tensorflow:2.13.0",
            packages=["pillow", "opencv-python"],
            gpu=False
        )

        # Isolated data analysis (no network)
        env_id = provision_docker_environment(
            base_image="python:3.12-slim",
            packages=["pandas", "scikit-learn"],
            network_enabled=False
        )

    Args:
        base_image: Docker image to use as base
                   Examples: "python:3.12-slim", "pytorch/pytorch:latest"
        packages: Optional list of Python packages to install via pip
                 (creates custom image with packages pre-installed)
        gpu: Enable GPU access (requires NVIDIA Docker runtime)
             Auto-selects PyTorch GPU image if base_image not specified
        network_enabled: Allow network access from container
                        Default: False (isolated for security)
        environment_name: Optional custom name for the environment

    Returns:
        str: Environment ID for use with execute_in_environment()
             Format: "✅ Docker environment created: {env_id}"

    Raises:
        ProvisionError: If Docker is not available or container creation fails

    Performance:
        Typical: 750ms provision + 2-5 minutes image build (if packages)
        Best case: 500ms (no packages, image cached)
        Worst case: 10 minutes (large packages + GPU image download)

    Security:
        - Read-only root filesystem
        - No network access by default
        - Memory and CPU limits enforced
        - Isolated /tmp directory
    """
    # Create environment configuration
    config = EnvironmentConfig(
        env_id=environment_name,
        env_type=EnvironmentType.DOCKER,
        base_image=base_image,
        packages=packages or [],
        gpu=gpu,
        network_enabled=network_enabled
    )

    # Get manager and create environment
    manager = SandboxManager()
    env_id = manager.create_environment(EnvironmentType.DOCKER, config)

    # Format success message
    gpu_status = "✅ GPU enabled" if gpu else "❌ GPU disabled"
    network_status = "✅ Network enabled" if network_enabled else "❌ Network isolated"
    packages_str = ", ".join(packages) if packages else "none"

    return (
        f"✅ Docker environment created: {env_id}\n"
        f"Base image: {base_image}\n"
        f"Packages: {packages_str}\n"
        f"{gpu_status} | {network_status}\n"
        f"Use execute_in_environment('{env_id}', code) to run code in this environment."
    )


def execute_in_environment(
    environment_id: str,
    code: str,
    timeout: int = 300
) -> str:
    """Execute Python code in a provisioned environment.

    **Usage:**
        # After provisioning environment
        env_id = provision_uv_environment(["pandas", "numpy"])

        # Execute code
        result = execute_in_environment(
            environment_id=env_id,
            code='''
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100)
})

print(data.describe())
            '''
        )

    **Multi-line code:**
        Always use triple-quoted strings for multi-line code to preserve
        formatting and indentation.

    **Return values:**
        Code execution captures:
        - stdout: print() statements
        - stderr: error messages
        - return value: last expression or explicit return
        - execution time: wall-clock time in seconds

    Args:
        environment_id: Environment ID from provision_*_environment()
        code: Python code to execute (supports multi-line)
        timeout: Maximum execution time in seconds (default: 300 = 5 minutes)

    Returns:
        str: Formatted execution result
             Success: "✅ Execution successful (time)\n\nOutput:\n{stdout}"
             Failure: "❌ Execution failed\n\nError: {error}"

    Raises:
        ValueError: If environment_id doesn't exist
        ExecutionError: If code execution fails
        TimeoutError: If execution exceeds timeout

    Examples:
        # Simple calculation
        result = execute_in_environment(
            env_id,
            "print(2 + 2)"
        )
        # Output: "✅ Execution successful (0.05s)\n\nOutput:\n4"

        # Data analysis
        result = execute_in_environment(
            env_id,
            '''
import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
            '''
        )

        # With timeout
        result = execute_in_environment(
            env_id,
            "import time; time.sleep(600)",  # Will timeout
            timeout=10
        )
        # Output: "❌ Execution failed\n\nError: Execution exceeded timeout of 10s"
    """
    # Get manager
    manager = SandboxManager()

    # Execute code
    result = manager.execute_in_environment(environment_id, code, timeout=timeout)

    # Return formatted result
    return result.formatted_output


def list_environments() -> str:
    """List all active sandbox environments.

    **Usage:**
        envs = list_environments()
        print(envs)

    Returns:
        str: Formatted list of all active environments with status

        Format:
            📦 Active Environments (2)

            1. uv_env_a1b2c3d4 (UV)
               - Python: 3.12
               - Packages: pandas, numpy, matplotlib
               - Status: ✅ Alive
               - Uptime: 5m 23s
               - Idle: 2m 15s

            2. docker_env_e5f6g7h8 (Docker)
               - Base image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
               - GPU: ✅ Enabled
               - Status: ✅ Alive
               - Uptime: 15m 42s
               - Idle: 5s

    Examples:
        # List all environments
        print(list_environments())

        # Check if specific environment exists
        envs = list_environments()
        if "my_env" in envs:
            print("Environment found!")
    """
    # Get manager
    manager = SandboxManager()

    # Get all environments
    envs = manager.list_environments()

    if not envs:
        return "📦 No active environments"

    # Format output
    lines = [f"📦 Active Environments ({len(envs)})\n"]

    for i, env in enumerate(envs, 1):
        env_type = env.get("env_type", "unknown").upper()
        env_id = env.get("env_id", "unknown")
        is_alive = "✅ Alive" if env.get("is_alive") else "❌ Dead"
        uptime = _format_duration(env.get("uptime", 0))
        idle_time = _format_duration(env.get("idle_time", 0))

        lines.append(f"\n{i}. {env_id} ({env_type})")

        # Add type-specific details
        config = env.get("config", {})
        if env_type == "UV":
            python = config.get("python_version", "unknown")
            packages = ", ".join(config.get("packages", [])) or "none"
            lines.append(f"   - Python: {python}")
            lines.append(f"   - Packages: {packages}")
        elif env_type == "DOCKER":
            base_image = config.get("base_image", "unknown")
            gpu = "✅ Enabled" if config.get("gpu") else "❌ Disabled"
            lines.append(f"   - Base image: {base_image}")
            lines.append(f"   - GPU: {gpu}")

        lines.append(f"   - Status: {is_alive}")
        lines.append(f"   - Uptime: {uptime}")
        lines.append(f"   - Idle: {idle_time}")

    return "\n".join(lines)


def cleanup_environment(environment_id: str) -> str:
    """Destroy a sandbox environment and free resources.

    **When to use:**
    - After completing task (free resources)
    - Environment no longer needed
    - Before creating new environment (if at max capacity)

    **What gets cleaned up:**
    - UV: Deletes virtual environment directory (~50-500MB)
    - Docker: Stops and removes container, optionally removes custom image

    **Auto-cleanup:**
    - Manager automatically cleans oldest environment when limit (10) reached
    - Environments persist until explicitly cleaned or limit reached

    Args:
        environment_id: Environment ID to cleanup

    Returns:
        str: Confirmation message
             "✅ Environment {env_id} destroyed and resources freed"

    Raises:
        ValueError: If environment_id doesn't exist
        CleanupError: If cleanup fails (environment still usable until fixed)

    Examples:
        # After task completion
        env_id = provision_uv_environment(["pandas"])
        result = execute_in_environment(env_id, "...")
        cleanup_environment(env_id)  # Free resources

        # Error handling
        try:
            cleanup_environment("nonexistent_env")
        except ValueError as e:
            print(f"Environment not found: {e}")
    """
    # Get manager
    manager = SandboxManager()

    # Cleanup environment
    manager.cleanup_environment(environment_id)

    return f"✅ Environment {environment_id} destroyed and resources freed"


# Helper functions

def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration.

    Args:
        seconds: Duration in seconds

    Returns:
        str: Formatted duration (e.g., "5m 23s", "1h 15m")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
