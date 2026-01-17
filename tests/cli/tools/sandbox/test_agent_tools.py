"""
Unit tests for agent-facing sandbox provisioning tools.

Tests for:
- provision_uv_environment
- provision_docker_environment
- execute_in_environment
- list_environments
- cleanup_environment
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from promptchain.cli.tools.sandbox.base import (
    EnvironmentType,
    EnvironmentConfig,
    SandboxResult,
)
from promptchain.cli.tools.sandbox.manager import SandboxManager
from promptchain.cli.tools.sandbox.agent_tools import (
    provision_uv_environment,
    provision_docker_environment,
    execute_in_environment,
    list_environments,
    cleanup_environment,
)


class TestProvisionUVEnvironment:
    """Tests for provision_uv_environment function."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_uv_basic(self, mock_create):
        """Test basic UV environment provisioning."""
        mock_create.return_value = "uv_env_test123"

        result = provision_uv_environment(packages=["pandas", "numpy"])

        # Verify manager was called correctly
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[0][0] == EnvironmentType.UV
        config = call_args[0][1]
        assert isinstance(config, EnvironmentConfig)
        assert config.packages == ["pandas", "numpy"]
        assert config.python_version == "3.12"  # Default

        # Verify response format
        assert "✅" in result
        assert "uv_env_test123" in result
        assert "pandas" in result
        assert "numpy" in result

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_uv_custom_python_version(self, mock_create):
        """Test UV provisioning with custom Python version."""
        mock_create.return_value = "uv_env_custom"

        result = provision_uv_environment(
            packages=["requests"],
            python_version="3.11"
        )

        config = mock_create.call_args[0][1]
        assert config.python_version == "3.11"

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_uv_named_environment(self, mock_create):
        """Test UV provisioning with custom environment name."""
        mock_create.return_value = "my_data_env"

        result = provision_uv_environment(
            packages=["pandas"],
            environment_name="my_data_env"
        )

        config = mock_create.call_args[0][1]
        assert config.env_id == "my_data_env"

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_uv_no_packages(self, mock_create):
        """Test UV provisioning with no packages."""
        mock_create.return_value = "uv_env_empty"

        result = provision_uv_environment(packages=[])

        config = mock_create.call_args[0][1]
        assert config.packages == []


class TestProvisionDockerEnvironment:
    """Tests for provision_docker_environment function."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_docker_basic(self, mock_create):
        """Test basic Docker environment provisioning."""
        mock_create.return_value = "docker_env_test123"

        result = provision_docker_environment(
            base_image="python:3.12-slim",
            packages=["pandas"]
        )

        # Verify manager was called correctly
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[0][0] == EnvironmentType.DOCKER
        config = call_args[0][1]
        assert config.base_image == "python:3.12-slim"
        assert config.packages == ["pandas"]
        assert config.gpu is False
        assert config.network_enabled is False

        # Verify response format
        assert "✅" in result
        assert "docker_env_test123" in result

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_docker_with_gpu(self, mock_create):
        """Test Docker provisioning with GPU support."""
        mock_create.return_value = "docker_env_gpu"

        result = provision_docker_environment(
            base_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
            packages=["transformers"],
            gpu=True
        )

        config = mock_create.call_args[0][1]
        assert config.gpu is True
        assert "gpu" in result.lower() or "GPU" in result

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_docker_with_network(self, mock_create):
        """Test Docker provisioning with network enabled."""
        mock_create.return_value = "docker_env_network"

        result = provision_docker_environment(
            base_image="python:3.12",
            network_enabled=True
        )

        config = mock_create.call_args[0][1]
        assert config.network_enabled is True

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_docker_named_environment(self, mock_create):
        """Test Docker provisioning with custom environment name."""
        mock_create.return_value = "my_ml_env"

        result = provision_docker_environment(
            base_image="pytorch/pytorch:latest",
            environment_name="my_ml_env"
        )

        config = mock_create.call_args[0][1]
        assert config.env_id == "my_ml_env"

    @patch.object(SandboxManager, 'create_environment')
    def test_provision_docker_no_packages(self, mock_create):
        """Test Docker provisioning with no additional packages."""
        mock_create.return_value = "docker_env_base"

        result = provision_docker_environment(base_image="python:3.12")

        config = mock_create.call_args[0][1]
        assert config.packages == []


class TestExecuteInEnvironment:
    """Tests for execute_in_environment function."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None

    @patch.object(SandboxManager, 'execute_in_environment')
    def test_execute_success(self, mock_execute):
        """Test successful code execution."""
        mock_execute.return_value = SandboxResult(
            success=True,
            stdout="Hello, World!",
            execution_time=0.5
        )

        result = execute_in_environment(
            environment_id="test_env",
            code="print('Hello, World!')"
        )

        # Verify manager was called correctly
        mock_execute.assert_called_once_with(
            "test_env",
            "print('Hello, World!')",
            timeout=300  # Default timeout
        )

        # Verify response contains formatted output
        assert "Hello, World!" in result or "✅" in result

    @patch.object(SandboxManager, 'execute_in_environment')
    def test_execute_with_custom_timeout(self, mock_execute):
        """Test execution with custom timeout."""
        mock_execute.return_value = SandboxResult(success=True)

        execute_in_environment(
            environment_id="test_env",
            code="import time; time.sleep(5)",
            timeout=60
        )

        mock_execute.assert_called_once_with(
            "test_env",
            "import time; time.sleep(5)",
            timeout=60
        )

    @patch.object(SandboxManager, 'execute_in_environment')
    def test_execute_failure(self, mock_execute):
        """Test failed code execution."""
        mock_execute.return_value = SandboxResult(
            success=False,
            error="ZeroDivisionError",
            stderr="Traceback...",
            execution_time=0.1
        )

        result = execute_in_environment(
            environment_id="test_env",
            code="1/0"
        )

        # Verify response contains error information
        assert "❌" in result or "error" in result.lower()

    @patch.object(SandboxManager, 'execute_in_environment')
    def test_execute_nonexistent_environment(self, mock_execute):
        """Test execution in nonexistent environment raises error."""
        mock_execute.side_effect = ValueError("Environment not found: nonexistent")

        with pytest.raises(ValueError, match="Environment not found"):
            execute_in_environment(
                environment_id="nonexistent",
                code="print('test')"
            )


class TestListEnvironments:
    """Tests for list_environments function."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None

    @patch.object(SandboxManager, 'list_environments')
    def test_list_empty(self, mock_list):
        """Test listing when no environments exist."""
        mock_list.return_value = []

        result = list_environments()

        mock_list.assert_called_once()
        assert "0" in result or "No" in result or "empty" in result.lower()

    @patch.object(SandboxManager, 'list_environments')
    def test_list_single_environment(self, mock_list):
        """Test listing single environment."""
        mock_list.return_value = [
            {
                "env_id": "uv_env_123",
                "env_type": "uv",
                "is_alive": True,
                "uptime": 60.0,
                "idle_time": 10.0
            }
        ]

        result = list_environments()

        # Verify response contains environment information
        assert "uv_env_123" in result
        assert "uv" in result.lower() or "UV" in result

    @patch.object(SandboxManager, 'list_environments')
    def test_list_multiple_environments(self, mock_list):
        """Test listing multiple environments."""
        mock_list.return_value = [
            {
                "env_id": "uv_env_1",
                "env_type": "uv",
                "is_alive": True,
                "uptime": 120.0,
                "idle_time": 20.0
            },
            {
                "env_id": "docker_env_2",
                "env_type": "docker",
                "is_alive": True,
                "uptime": 60.0,
                "idle_time": 5.0
            }
        ]

        result = list_environments()

        # Verify both environments are listed
        assert "uv_env_1" in result
        assert "docker_env_2" in result
        assert "2" in result  # Count of environments


class TestCleanupEnvironment:
    """Tests for cleanup_environment function."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None

    @patch.object(SandboxManager, 'cleanup_environment')
    def test_cleanup_success(self, mock_cleanup):
        """Test successful environment cleanup."""
        mock_cleanup.return_value = None

        result = cleanup_environment("test_env")

        mock_cleanup.assert_called_once_with("test_env")

        # Verify response indicates success
        assert "✅" in result
        assert "test_env" in result
        assert "destroyed" in result.lower() or "cleaned" in result.lower()

    @patch.object(SandboxManager, 'cleanup_environment')
    def test_cleanup_nonexistent_environment(self, mock_cleanup):
        """Test cleanup of nonexistent environment raises error."""
        mock_cleanup.side_effect = ValueError("Environment not found: nonexistent")

        with pytest.raises(ValueError, match="Environment not found"):
            cleanup_environment("nonexistent")


class TestIntegrationWorkflow:
    """Integration tests for typical agent workflows."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None

    @patch.object(SandboxManager, 'create_environment')
    @patch.object(SandboxManager, 'execute_in_environment')
    @patch.object(SandboxManager, 'cleanup_environment')
    def test_complete_uv_workflow(self, mock_cleanup, mock_execute, mock_create):
        """Test complete workflow: provision → execute → cleanup."""
        # Setup mocks
        mock_create.return_value = "uv_env_workflow"
        mock_execute.return_value = SandboxResult(
            success=True,
            stdout="Analysis complete",
            execution_time=1.5
        )

        # 1. Provision environment
        env_id_result = provision_uv_environment(packages=["pandas", "numpy"])
        assert "uv_env_workflow" in env_id_result

        # 2. Execute code
        exec_result = execute_in_environment(
            environment_id="uv_env_workflow",
            code="import pandas as pd; print('Analysis complete')"
        )
        assert "Analysis complete" in exec_result or "✅" in exec_result

        # 3. Cleanup
        cleanup_result = cleanup_environment("uv_env_workflow")
        assert "✅" in cleanup_result

        # Verify all operations were called
        mock_create.assert_called_once()
        mock_execute.assert_called_once()
        mock_cleanup.assert_called_once()

    @patch.object(SandboxManager, 'create_environment')
    @patch.object(SandboxManager, 'execute_in_environment')
    @patch.object(SandboxManager, 'cleanup_environment')
    def test_complete_docker_workflow(self, mock_cleanup, mock_execute, mock_create):
        """Test complete workflow with Docker: provision → execute → cleanup."""
        # Setup mocks
        mock_create.return_value = "docker_env_workflow"
        mock_execute.return_value = SandboxResult(
            success=True,
            stdout="Model trained",
            execution_time=5.0
        )

        # 1. Provision Docker environment with GPU
        env_id_result = provision_docker_environment(
            base_image="pytorch/pytorch:latest",
            packages=["transformers"],
            gpu=True
        )
        assert "docker_env_workflow" in env_id_result

        # 2. Execute ML code
        exec_result = execute_in_environment(
            environment_id="docker_env_workflow",
            code="import torch; print('Model trained')",
            timeout=600
        )
        assert "Model trained" in exec_result or "✅" in exec_result

        # 3. Cleanup
        cleanup_result = cleanup_environment("docker_env_workflow")
        assert "✅" in cleanup_result

        # Verify all operations were called
        mock_create.assert_called_once()
        mock_execute.assert_called_once()
        mock_cleanup.assert_called_once()
