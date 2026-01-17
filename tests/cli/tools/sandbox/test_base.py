"""
Unit tests for sandbox base abstractions.

Tests for:
- EnvironmentType enum
- SandboxResult dataclass
- EnvironmentConfig dataclass
- BaseSandbox abstract class
- Custom exceptions
"""

import pytest
from promptchain.cli.tools.sandbox.base import (
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


class TestEnvironmentType:
    """Tests for EnvironmentType enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert EnvironmentType.UV.value == "uv"
        assert EnvironmentType.DOCKER.value == "docker"
        assert EnvironmentType.DIRECT.value == "direct"

    def test_enum_str(self):
        """Test enum string representation."""
        assert str(EnvironmentType.UV) == "uv"
        assert str(EnvironmentType.DOCKER) == "docker"
        assert str(EnvironmentType.DIRECT) == "direct"

    def test_typical_startup_ms(self):
        """Test typical startup time properties."""
        assert EnvironmentType.UV.typical_startup_ms == 75
        assert EnvironmentType.DOCKER.typical_startup_ms == 750
        assert EnvironmentType.DIRECT.typical_startup_ms == 1

    def test_isolation_level(self):
        """Test isolation level descriptions."""
        assert "Python environment" in EnvironmentType.UV.isolation_level
        assert "Full container" in EnvironmentType.DOCKER.isolation_level
        assert "None" in EnvironmentType.DIRECT.isolation_level


class TestSandboxResult:
    """Tests for SandboxResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = SandboxResult(
            success=True,
            stdout="Hello, World!",
            execution_time=0.123,
            exit_code=0
        )

        assert result.success is True
        assert result.stdout == "Hello, World!"
        assert result.execution_time == 0.123
        assert result.error is None
        assert result.timeout is False

    def test_failed_result(self):
        """Test failed execution result."""
        result = SandboxResult(
            success=False,
            stderr="Error occurred",
            error="Execution failed",
            exit_code=1
        )

        assert result.success is False
        assert result.error == "Execution failed"
        assert result.exit_code == 1

    def test_timeout_result(self):
        """Test timeout result."""
        result = SandboxResult(
            success=False,
            error="Timeout",
            timeout=True,
            execution_time=10.0
        )

        assert result.timeout is True
        assert result.success is False

    def test_formatted_output_success(self):
        """Test formatted output for successful execution."""
        result = SandboxResult(
            success=True,
            stdout="Output text",
            return_value="42",
            execution_time=0.5
        )

        output = result.formatted_output
        assert "✅" in output
        assert "Execution Successful" in output
        assert "0.5" in output
        assert "Output text" in output

    def test_formatted_output_failure(self):
        """Test formatted output for failed execution."""
        result = SandboxResult(
            success=False,
            error="Division by zero",
            stderr="Traceback...",
            execution_time=0.1
        )

        output = result.formatted_output
        assert "❌" in output
        assert "Execution Failed" in output
        assert "Division by zero" in output

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SandboxResult(
            success=True,
            stdout="output",
            execution_time=1.0
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["success"] is True
        assert data["stdout"] == "output"
        assert data["execution_time"] == 1.0

    def test_post_init_validation(self):
        """Test __post_init__ validation."""
        # Success should be False if exit_code != 0
        result = SandboxResult(
            success=True,
            exit_code=1  # Non-zero exit code
        )
        assert result.success is False

        # Error message from stderr if no error provided
        result = SandboxResult(
            success=False,
            stderr="Error details here"
        )
        assert result.error is not None


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EnvironmentConfig()

        assert config.env_type == EnvironmentType.UV
        assert config.python_version == "3.12"
        assert config.packages == []
        assert config.memory_limit == "512MB"
        assert config.timeout == 300
        assert config.gpu is False
        assert config.network_enabled is False

    def test_uv_config(self):
        """Test UV-specific configuration."""
        config = EnvironmentConfig(
            env_type=EnvironmentType.UV,
            python_version="3.11",
            packages=["pandas", "numpy"]
        )

        assert config.env_type == EnvironmentType.UV
        assert config.python_version == "3.11"
        assert config.packages == ["pandas", "numpy"]

    def test_docker_config(self):
        """Test Docker-specific configuration."""
        config = EnvironmentConfig(
            env_type=EnvironmentType.DOCKER,
            base_image="python:3.12-slim",
            gpu=False
        )

        assert config.env_type == EnvironmentType.DOCKER
        assert config.base_image == "python:3.12-slim"

    def test_docker_gpu_config(self):
        """Test Docker GPU configuration."""
        config = EnvironmentConfig(
            env_type=EnvironmentType.DOCKER,
            gpu=True
        )

        assert config.gpu is True
        # Should auto-select PyTorch GPU image
        assert "pytorch" in config.base_image.lower()
        assert "cuda" in config.base_image.lower()

    def test_gpu_validation(self):
        """Test GPU requires Docker environment."""
        with pytest.raises(ValueError, match="GPU support only available"):
            EnvironmentConfig(
                env_type=EnvironmentType.UV,
                gpu=True  # Should fail - UV doesn't support GPU
            )

    def test_string_to_enum_conversion(self):
        """Test string env_type converts to enum."""
        config = EnvironmentConfig(env_type="docker")
        assert config.env_type == EnvironmentType.DOCKER
        assert isinstance(config.env_type, EnvironmentType)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = EnvironmentConfig(
            env_type=EnvironmentType.UV,
            packages=["pandas"]
        )

        data = config.to_dict()
        assert isinstance(data, dict)
        assert data["env_type"] == "uv"
        assert data["packages"] == ["pandas"]


class TestBaseSandbox:
    """Tests for BaseSandbox abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test BaseSandbox cannot be instantiated directly."""
        config = EnvironmentConfig()

        with pytest.raises(TypeError):
            BaseSandbox(config)

    def test_abstract_methods(self):
        """Test all required abstract methods are defined."""
        # Create a minimal concrete implementation
        class TestSandbox(BaseSandbox):
            def provision(self) -> str:
                return "test_env"

            def execute(self, code: str, timeout=None, capture_output=True):
                return SandboxResult(success=True)

            def cleanup(self) -> None:
                pass

            def is_alive(self) -> bool:
                return True

        config = EnvironmentConfig()
        sandbox = TestSandbox(config)

        assert sandbox.provision() == "test_env"
        assert sandbox.execute("test").success is True
        assert sandbox.is_alive() is True
        sandbox.cleanup()  # Should not raise

    def test_get_status(self):
        """Test get_status method."""
        class TestSandbox(BaseSandbox):
            def provision(self):
                self.env_id = "test"
                self._update_timestamps("provision")
                return self.env_id

            def execute(self, code, timeout=None, capture_output=True):
                return SandboxResult(success=True)

            def cleanup(self):
                pass

            def is_alive(self):
                return True

        config = EnvironmentConfig()
        sandbox = TestSandbox(config)
        sandbox.provision()

        status = sandbox.get_status()
        assert status["env_id"] == "test"
        assert status["env_type"] == "uv"
        assert status["is_alive"] is True
        assert "uptime" in status
        assert "idle_time" in status

    def test_repr(self):
        """Test string representation."""
        class TestSandbox(BaseSandbox):
            def provision(self):
                self.env_id = "test_env"
                return self.env_id

            def execute(self, code, timeout=None, capture_output=True):
                return SandboxResult(success=True)

            def cleanup(self):
                pass

            def is_alive(self):
                return False

        config = EnvironmentConfig()
        sandbox = TestSandbox(config)
        sandbox.provision()

        repr_str = repr(sandbox)
        assert "TestSandbox" in repr_str
        assert "test_env" in repr_str
        assert "is_alive=False" in repr_str


class TestCustomExceptions:
    """Tests for custom exception hierarchy."""

    def test_exception_hierarchy(self):
        """Test exception inheritance."""
        assert issubclass(ProvisionError, SandboxError)
        assert issubclass(ExecutionError, SandboxError)
        assert issubclass(CleanupError, SandboxError)
        assert issubclass(SecurityError, SandboxError)
        assert issubclass(SandboxError, Exception)

    def test_raise_provision_error(self):
        """Test ProvisionError can be raised and caught."""
        with pytest.raises(ProvisionError):
            raise ProvisionError("Failed to provision")

    def test_raise_execution_error(self):
        """Test ExecutionError can be raised and caught."""
        with pytest.raises(ExecutionError):
            raise ExecutionError("Failed to execute")

    def test_raise_cleanup_error(self):
        """Test CleanupError can be raised and caught."""
        with pytest.raises(CleanupError):
            raise CleanupError("Failed to cleanup")

    def test_raise_security_error(self):
        """Test SecurityError can be raised and caught."""
        with pytest.raises(SecurityError):
            raise SecurityError("Security violation")

    def test_catch_as_sandbox_error(self):
        """Test all exceptions can be caught as SandboxError."""
        with pytest.raises(SandboxError):
            raise ProvisionError("Test")

        with pytest.raises(SandboxError):
            raise ExecutionError("Test")

        with pytest.raises(SandboxError):
            raise CleanupError("Test")

        with pytest.raises(SandboxError):
            raise SecurityError("Test")
