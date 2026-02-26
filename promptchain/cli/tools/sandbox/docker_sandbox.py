"""
Docker-based container sandbox for full isolation.

Provides full OS-level isolation using Docker containers with support
for GPU access, network isolation, and complex dependencies.

Characteristics:
- Slower startup: 500-1000ms
- Full OS-level isolation
- GPU support (NVIDIA)
- Network isolation
- Best for: ML/DL, untrusted code, GPU workloads, complex dependencies

Performance Targets:
- Provision: <1000ms
- Execute: <200ms overhead (after container running)
- Cleanup: <500ms
"""

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .base import (BaseSandbox, CleanupError, EnvironmentConfig,
                   EnvironmentType, ExecutionError, ProvisionError,
                   SandboxResult)


class DockerSandbox(BaseSandbox):
    """Docker container-based sandbox for full isolation.

    Uses Docker to create isolated containers with optional GPU access,
    network isolation, and custom package installations.

    Example:
        >>> config = EnvironmentConfig(
        ...     env_type=EnvironmentType.DOCKER,
        ...     base_image="python:3.12-slim",
        ...     packages=["torch", "transformers"],
        ...     gpu=True,
        ...     network_enabled=False
        ... )
        >>> sandbox = DockerSandbox(config)
        >>> env_id = sandbox.provision()
        >>> result = sandbox.execute("import torch; print(torch.cuda.is_available())")
        >>> print(result.formatted_output)
        >>> sandbox.cleanup()
    """

    def __init__(self, config: EnvironmentConfig):
        """Initialize Docker sandbox with configuration.

        Args:
            config: Environment configuration (must have env_type=DOCKER)

        Raises:
            ValueError: If env_type is not DOCKER
        """
        super().__init__(config)

        if config.env_type != EnvironmentType.DOCKER:
            raise ValueError(
                f"DockerSandbox requires env_type=DOCKER, got {config.env_type}"
            )

        self.container_name: Optional[str] = None
        self.image_name: Optional[str] = None
        self._container_running: bool = False

    def provision(self) -> str:
        """Provision Docker container environment.

        Creates a Docker container with specified configuration, builds
        custom image if packages specified, and starts the container.

        Returns:
            str: Environment ID (container name)

        Raises:
            ProvisionError: If container creation or image building fails
        """
        start_time = time.time()

        try:
            # Generate environment ID and container name
            self.env_id = self.config.env_id or f"docker_env_{uuid.uuid4().hex[:8]}"
            self.container_name = self.config.container_name or self.env_id

            # Check if Docker is available
            self._ensure_docker_available()

            # Determine base image (auto-select if not specified)
            base_image = self.config.base_image or self._auto_select_base_image()

            # Build custom image if packages specified
            if self.config.packages:
                self.image_name = self._build_custom_image(
                    base_image, self.config.packages
                )
            else:
                self.image_name = base_image

            # Create container
            self._create_container()

            # Start container
            self._start_container()

            # Mark as provisioned
            self._is_provisioned = True
            self._container_running = True
            self._update_timestamps("provision")

            provision_time = time.time() - start_time

            return self.env_id

        except Exception as e:
            # Cleanup partial resources
            try:
                self._cleanup_partial()
            except Exception:
                pass  # Ignore cleanup errors during error handling

            raise ProvisionError(f"Failed to provision Docker environment: {e}") from e

    def execute(
        self, code: str, timeout: Optional[int] = None, capture_output: bool = True
    ) -> SandboxResult:
        """Execute Python code in Docker container.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (uses config default if None)
            capture_output: Whether to capture stdout/stderr

        Returns:
            SandboxResult: Execution result with outputs and metadata

        Raises:
            ExecutionError: If execution fails
            SecurityError: If code violates security policy
        """
        if not self._is_provisioned or not self._container_running:
            raise ExecutionError(
                "Container not running. Call provision() first or container was stopped."
            )

        # Use config timeout if not specified
        if timeout is None:
            timeout = self.config.timeout

        start_time = time.time()

        try:
            # Validate code security
            self._validate_code_security(code)

            # Write code to temporary file
            code_file_path = self._write_code_to_temp_file(code)

            try:
                # Copy code file into container
                container_code_path = f"/tmp/code_{uuid.uuid4().hex[:8]}.py"
                self._copy_to_container(code_file_path, container_code_path)

                # Execute code in container
                exec_cmd: List[str] = [
                    x
                    for x in [
                        "docker",
                        "exec",
                        self.container_name,
                        "python",
                        container_code_path,
                    ]
                    if x is not None
                ]

                result = subprocess.run(
                    exec_cmd, capture_output=capture_output, text=True, timeout=timeout
                )

                execution_time = time.time() - start_time

                # Update last used timestamp
                self._update_timestamps("execute")

                return SandboxResult(
                    success=(result.returncode == 0),
                    stdout=result.stdout if capture_output else "",
                    stderr=result.stderr if capture_output else "",
                    execution_time=execution_time,
                    exit_code=result.returncode,
                    error=result.stderr if result.returncode != 0 else None,
                    metadata={
                        "env_id": self.env_id,
                        "env_type": "docker",
                        "container_name": self.container_name,
                        "image": self.image_name,
                        "gpu": self.config.gpu,
                    },
                )

            finally:
                # Clean up temporary files
                try:
                    os.unlink(code_file_path)
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return SandboxResult(
                success=False,
                error=f"Execution exceeded timeout of {timeout}s",
                execution_time=execution_time,
                timeout=True,
                metadata={"env_id": self.env_id},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            raise ExecutionError(f"Execution failed: {e}") from e

    def cleanup(self) -> None:
        """Destroy Docker container and free resources.

        Stops and removes the container, and removes custom images if created.

        Raises:
            CleanupError: If cleanup fails
        """
        try:
            # Stop container if running
            if self._container_running:
                self._stop_container()

            # Remove container
            if self.container_name:
                self._remove_container()

            # Remove custom image if created (not base images)
            if self.image_name and self.config.packages:
                self._remove_custom_image()

            self._is_provisioned = False
            self._container_running = False
            self.container_name = None
            self.image_name = None

        except Exception as e:
            raise CleanupError(f"Failed to cleanup Docker environment: {e}") from e

    def is_alive(self) -> bool:
        """Check if Docker container is running and functional.

        Returns:
            bool: True if container is running and responsive
        """
        if not self._is_provisioned or not self.container_name:
            return False

        try:
            # Check container status
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    self.container_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return False

            # Container is running if output is "true"
            return result.stdout.strip() == "true"

        except Exception:
            return False

    # Private helper methods

    def _ensure_docker_available(self) -> None:
        """Ensure Docker is installed and running.

        Raises:
            ProvisionError: If Docker is not available
        """
        try:
            result = subprocess.run(
                ["docker", "version"], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                raise ProvisionError(
                    "Docker is not running. Please start Docker daemon."
                )
        except FileNotFoundError:
            raise ProvisionError("Docker is not installed. Please install Docker.")

    def _auto_select_base_image(self) -> str:
        """Auto-select appropriate base image based on configuration.

        Returns:
            str: Docker image name
        """
        if self.config.gpu:
            # GPU workloads: Use PyTorch image with CUDA
            return "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime"
        else:
            # Standard workloads: Use slim Python image
            return f"python:{self.config.python_version}-slim"

    def _build_custom_image(self, base_image: str, packages: List[str]) -> str:
        """Build custom Docker image with packages installed.

        Args:
            base_image: Base image to build from
            packages: List of Python packages to install

        Returns:
            str: Custom image name

        Raises:
            ProvisionError: If image build fails
        """
        try:
            # Generate custom image name
            image_name = f"promptchain_sandbox_{self.env_id}"

            # Create Dockerfile
            dockerfile_content = f"""FROM {base_image}

# Install packages
RUN pip install --no-cache-dir {' '.join(packages)}

# Set working directory
WORKDIR /workspace
"""

            # Write Dockerfile to temporary directory
            with tempfile.TemporaryDirectory() as build_dir:
                dockerfile_path = Path(build_dir) / "Dockerfile"
                dockerfile_path.write_text(dockerfile_content)

                # Build image
                build_cmd = [
                    "docker",
                    "build",
                    "-t",
                    image_name,
                    "-f",
                    str(dockerfile_path),
                    build_dir,
                ]

                result = subprocess.run(
                    build_cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minutes for image build
                )

                if result.returncode != 0:
                    raise ProvisionError(f"Docker image build failed: {result.stderr}")

            return image_name

        except subprocess.TimeoutExpired:
            raise ProvisionError("Docker image build exceeded 10 minute timeout")

    def _create_container(self) -> None:
        """Create Docker container (not started yet).

        Raises:
            ProvisionError: If container creation fails
        """
        try:
            # Build create command
            create_cmd: List[str] = [
                x
                for x in [
                    "docker",
                    "create",
                    "--name",
                    self.container_name,
                    "--memory",
                    self.config.memory_limit,
                    "--cpus",
                    "2.0",  # Limit to 2 CPUs
                    "--read-only",  # Read-only root filesystem for security
                    "--tmpfs",
                    "/tmp:rw,size=100m",  # Writable tmp directory
                ]
                if x is not None
            ]

            # Add network configuration
            if not self.config.network_enabled:
                create_cmd.extend(["--network", "none"])
            else:
                create_cmd.extend(["--network", "bridge"])

            # Add GPU support if requested
            if self.config.gpu:
                create_cmd.extend(["--gpus", "all"])

            # Add volume mounts
            for host_path, container_path in self.config.docker_volumes.items():
                create_cmd.extend(["-v", f"{host_path}:{container_path}"])

            # Add environment variables
            for key, value in self.config.env_vars.items():
                create_cmd.extend(["-e", f"{key}={value}"])

            # Add image name and command (keep container running)
            create_cmd.extend(
                [x for x in [self.image_name] if x is not None]
                + ["tail", "-f", "/dev/null"]
            )

            # Create container
            result = subprocess.run(
                create_cmd, capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                raise ProvisionError(
                    f"Docker container creation failed: {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise ProvisionError("Docker container creation exceeded 30s timeout")

    def _start_container(self) -> None:
        """Start the Docker container.

        Raises:
            ProvisionError: If container start fails
        """
        try:
            container_name: str = self.container_name or ""
            result = subprocess.run(
                ["docker", "start", container_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise ProvisionError(f"Docker container start failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise ProvisionError("Docker container start exceeded 10s timeout")

    def _stop_container(self) -> None:
        """Stop the Docker container."""
        if not self.container_name:
            return
        try:
            subprocess.run(
                ["docker", "stop", self.container_name], capture_output=True, timeout=30
            )
        except Exception:
            pass  # Ignore errors during stop

    def _remove_container(self) -> None:
        """Remove the Docker container."""
        if not self.container_name:
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass  # Ignore errors during removal

    def _remove_custom_image(self) -> None:
        """Remove custom Docker image."""
        if not self.image_name:
            return
        try:
            subprocess.run(
                ["docker", "rmi", "-f", self.image_name],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass  # Ignore errors during image removal

    def _cleanup_partial(self) -> None:
        """Clean up partial resources after provision failure."""
        self._stop_container()
        self._remove_container()
        self._remove_custom_image()

    def _write_code_to_temp_file(self, code: str) -> str:
        """Write code to temporary file.

        Args:
            code: Python code

        Returns:
            str: Path to temporary file
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.config.working_dir
        ) as code_file:
            code_file.write(code)
            return code_file.name

    def _copy_to_container(self, host_path: str, container_path: str) -> None:
        """Copy file from host to container.

        Args:
            host_path: Path on host
            container_path: Path in container

        Raises:
            ExecutionError: If copy fails
        """
        try:
            container_name: str = self.container_name or ""
            result = subprocess.run(
                ["docker", "cp", host_path, f"{container_name}:{container_path}"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise ExecutionError(
                    f"Failed to copy file to container: {result.stderr}"
                )

        except subprocess.TimeoutExpired:
            raise ExecutionError("File copy to container exceeded 10s timeout")

    def _validate_code_security(self, code: str) -> None:
        """Validate code against security policy.

        Reuses existing SafetyValidator from T117 if available,
        otherwise performs basic validation.

        Args:
            code: Python code to validate

        Raises:
            SecurityError: If code violates security policy
        """
        try:
            # Try to use existing SafetyValidator
            from promptchain.cli.tools.safety import SafetyValidator

            project_root = self.config.working_dir or Path(".")
            validator = SafetyValidator(project_root=project_root)
            validator.validate_command(["python", "-c", code])
        except ImportError:
            # Fallback to basic validation if SafetyValidator not available
            self._basic_security_validation(code)

    def _basic_security_validation(self, code: str) -> None:
        """Basic security validation (fallback).

        Args:
            code: Python code to validate

        Raises:
            SecurityError: If code contains forbidden patterns
        """
        from .base import SecurityError

        forbidden_patterns = [
            "eval(",
            "exec(",
            "compile(",
            "__import__",
        ]

        for pattern in forbidden_patterns:
            if pattern in code:
                raise SecurityError(f"Forbidden pattern detected: {pattern}")
