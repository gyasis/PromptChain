"""
UV-based Python virtual environment sandbox.

Provides fast, lightweight Python environment isolation using UV
(ultra-fast Python package installer and resolver).

Characteristics:
- Fast startup: 50-100ms
- Python-level isolation (process-level)
- Excellent package management via UV
- Best for: Data analysis, scientific computing, standard Python workloads

Performance Targets:
- Provision: <100ms
- Execute: <200ms overhead
- Cleanup: <500ms
"""

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional

from .base import (BaseSandbox, CleanupError, EnvironmentConfig,
                   EnvironmentType, ExecutionError, ProvisionError,
                   SandboxResult)


class UVSandbox(BaseSandbox):
    """UV-based Python virtual environment sandbox.

    Uses UV to create isolated Python environments with rapid package
    installation and excellent dependency resolution.

    Example:
        >>> config = EnvironmentConfig(
        ...     env_type=EnvironmentType.UV,
        ...     packages=["pandas", "numpy", "matplotlib"],
        ...     python_version="3.12"
        ... )
        >>> sandbox = UVSandbox(config)
        >>> env_id = sandbox.provision()
        >>> result = sandbox.execute("import pandas as pd; print(pd.__version__)")
        >>> print(result.formatted_output)
        >>> sandbox.cleanup()
    """

    def __init__(self, config: EnvironmentConfig):
        """Initialize UV sandbox with configuration.

        Args:
            config: Environment configuration (must have env_type=UV)

        Raises:
            ValueError: If env_type is not UV
        """
        super().__init__(config)

        if config.env_type != EnvironmentType.UV:
            raise ValueError(f"UVSandbox requires env_type=UV, got {config.env_type}")

        self.env_path: Optional[Path] = None
        self.python_path: Optional[Path] = None

    def provision(self) -> str:
        """Provision UV virtual environment.

        Creates a UV-based virtual environment with specified Python version
        and installs required packages.

        Returns:
            str: Environment ID (unique identifier)

        Raises:
            ProvisionError: If environment creation or package installation fails
        """
        start_time = time.time()

        try:
            # Generate environment ID
            self.env_id = self.config.env_id or f"uv_env_{uuid.uuid4().hex[:8]}"

            # Determine environment path
            if self.config.uv_path:
                self.env_path = self.config.uv_path
            else:
                # Use .promptchain/sandboxes directory
                sandboxes_dir = Path.home() / ".promptchain" / "sandboxes"
                sandboxes_dir.mkdir(parents=True, exist_ok=True)
                self.env_path = sandboxes_dir / self.env_id

            # Check if UV is installed
            self._ensure_uv_installed()

            # Create UV virtual environment
            self._create_venv()

            # Set python path IMMEDIATELY after venv creation (required for package installation)
            self.python_path = self.env_path / "bin" / "python"

            # Install packages if specified
            if self.config.packages:
                self._install_packages(self.config.packages)

            # Mark as provisioned
            self._is_provisioned = True
            self._update_timestamps("provision")

            provision_time = time.time() - start_time

            return self.env_id

        except Exception as e:
            raise ProvisionError(f"Failed to provision UV environment: {e}") from e

    def execute(
        self, code: str, timeout: Optional[int] = None, capture_output: bool = True
    ) -> SandboxResult:
        """Execute Python code in UV environment.

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
        if not self._is_provisioned:
            raise ExecutionError("Environment not provisioned. Call provision() first.")

        # Use config timeout if not specified
        if timeout is None:
            timeout = self.config.timeout

        start_time = time.time()

        try:
            # Validate code security (reuse existing SafetyValidator)
            self._validate_code_security(code)

            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, dir=self.config.working_dir
            ) as code_file:
                code_file.write(code)
                code_file_path = code_file.name

            try:
                # Execute code with UV Python
                result = subprocess.run(
                    [str(self.python_path), code_file_path],
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout,
                    cwd=self.config.working_dir,
                    env=self._get_execution_env(),
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
                        "env_type": "uv",
                        "python_version": self.config.python_version,
                    },
                )

            finally:
                # Clean up temporary code file
                try:
                    os.unlink(code_file_path)
                except Exception:
                    pass  # Ignore cleanup errors

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
        """Destroy UV environment and free resources.

        Removes the virtual environment directory and all its contents.

        Raises:
            CleanupError: If cleanup fails
        """
        try:
            if self.env_path and self.env_path.exists():
                # Remove entire environment directory
                import shutil

                shutil.rmtree(self.env_path)

            self._is_provisioned = False
            self.env_path = None
            self.python_path = None

        except Exception as e:
            raise CleanupError(f"Failed to cleanup UV environment: {e}") from e

    def is_alive(self) -> bool:
        """Check if UV environment is functional.

        Returns:
            bool: True if environment exists and Python executable is available
        """
        if not self._is_provisioned:
            return False

        if not self.env_path or not self.env_path.exists():
            return False

        if not self.python_path or not self.python_path.exists():
            return False

        # Verify Python executable works
        try:
            result = subprocess.run(
                [str(self.python_path), "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    # Private helper methods

    def _ensure_uv_installed(self) -> None:
        """Ensure UV is installed on the system.

        Raises:
            ProvisionError: If UV is not available
        """
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, timeout=5)
            if result.returncode != 0:
                raise ProvisionError(
                    "UV is not installed. Install with: pip install uv"
                )
        except FileNotFoundError:
            raise ProvisionError("UV is not installed. Install with: pip install uv")

    def _create_venv(self) -> None:
        """Create UV virtual environment.

        Raises:
            ProvisionError: If venv creation fails
        """
        try:
            # UV venv command with Python version
            cmd = [
                "uv",
                "venv",
                "--python",
                self.config.python_version,
                str(self.env_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                raise ProvisionError(f"UV venv creation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise ProvisionError("UV venv creation exceeded 30s timeout")

    def _install_packages(self, packages: List[str]) -> None:
        """Install packages in UV environment.

        Args:
            packages: List of package names to install

        Raises:
            ProvisionError: If package installation fails
        """
        if not packages:
            return

        try:
            # Get pip path in UV environment
            if not self.env_path:
                raise ProvisionError("UV environment path is not set")
            pip_path = self.env_path / "bin" / "pip"

            # Use UV pip for ultra-fast installation
            cmd = ["uv", "pip", "install", "--python", str(self.python_path), *packages]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for package installation
            )

            if result.returncode != 0:
                raise ProvisionError(f"Package installation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise ProvisionError(
                f"Package installation exceeded 5 minute timeout for: {packages}"
            )

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
            "os.system",
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
        ]

        for pattern in forbidden_patterns:
            if pattern in code:
                raise SecurityError(f"Forbidden pattern detected: {pattern}")

    def _get_execution_env(self) -> dict:
        """Get environment variables for code execution.

        Returns:
            dict: Environment variables
        """
        env = os.environ.copy()

        # Add user-specified env vars
        env.update(self.config.env_vars)

        # Set Python-specific variables
        env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time output

        return env
