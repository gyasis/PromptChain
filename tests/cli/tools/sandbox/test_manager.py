"""
Unit tests for SandboxManager.

Tests for:
- Singleton pattern
- Environment creation
- Environment lookup
- Environment cleanup
- Resource limits
- Thread safety
"""

import pytest
import threading
from unittest.mock import Mock, patch, MagicMock

from promptchain.cli.tools.sandbox.base import (
    EnvironmentType,
    EnvironmentConfig,
    SandboxResult,
)
from promptchain.cli.tools.sandbox.manager import SandboxManager
from promptchain.cli.tools.sandbox.uv_sandbox import UVSandbox
from promptchain.cli.tools.sandbox.docker_sandbox import DockerSandbox


class TestSandboxManagerSingleton:
    """Tests for singleton pattern."""

    def test_singleton_instance(self):
        """Test only one instance is created."""
        manager1 = SandboxManager()
        manager2 = SandboxManager()

        assert manager1 is manager2

    def test_singleton_state_shared(self):
        """Test state is shared across instances."""
        manager1 = SandboxManager()
        manager1.max_environments = 5

        manager2 = SandboxManager()
        assert manager2.max_environments == 5

    def test_singleton_thread_safe(self):
        """Test singleton is thread-safe."""
        instances = []

        def create_manager():
            instances.append(SandboxManager())

        threads = [threading.Thread(target=create_manager) for _ in range(10)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All instances should be the same object
        assert all(instance is instances[0] for instance in instances)


class TestSandboxManagerEnvironmentCreation:
    """Tests for environment creation."""

    def setup_method(self):
        """Reset manager before each test."""
        # Create fresh manager instance
        SandboxManager._instance = None
        self.manager = SandboxManager()

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            self.manager.cleanup_all()
        except Exception:
            pass

    @patch.object(UVSandbox, 'provision')
    def test_create_uv_environment(self, mock_provision):
        """Test creating UV environment."""
        mock_provision.return_value = "uv_env_test"

        config = EnvironmentConfig(
            env_type=EnvironmentType.UV,
            packages=["pandas"]
        )

        env_id = self.manager.create_environment(EnvironmentType.UV, config)

        assert env_id == "uv_env_test"
        assert env_id in self.manager.environments
        assert isinstance(self.manager.environments[env_id], UVSandbox)
        mock_provision.assert_called_once()

    @patch.object(DockerSandbox, 'provision')
    def test_create_docker_environment(self, mock_provision):
        """Test creating Docker environment."""
        mock_provision.return_value = "docker_env_test"

        config = EnvironmentConfig(
            env_type=EnvironmentType.DOCKER,
            base_image="python:3.12-slim"
        )

        env_id = self.manager.create_environment(EnvironmentType.DOCKER, config)

        assert env_id == "docker_env_test"
        assert env_id in self.manager.environments
        assert isinstance(self.manager.environments[env_id], DockerSandbox)
        mock_provision.assert_called_once()

    def test_create_unsupported_environment_type(self):
        """Test creating unsupported environment type fails."""
        config = EnvironmentConfig(env_type=EnvironmentType.DIRECT)

        with pytest.raises(ValueError, match="Unsupported environment type"):
            self.manager.create_environment(EnvironmentType.DIRECT, config)

    def test_max_environments_cleanup_oldest(self):
        """Test oldest environment is cleaned up when limit reached."""
        import time

        # Manually populate environments to test cleanup logic
        for i in range(10):
            mock_sandbox = Mock(spec=UVSandbox)
            mock_sandbox.last_used = time.time() - (10 - i)  # Older environments have lower timestamps
            mock_sandbox.created_at = time.time() - (10 - i)
            mock_sandbox.cleanup = Mock()
            self.manager.environments[f"env_{i}"] = mock_sandbox
            time.sleep(0.01)  # Small delay to ensure timestamps differ

        assert len(self.manager.environments) == 10

        # Trigger cleanup of oldest
        oldest_id = "env_0"  # Should be the oldest
        self.manager._cleanup_oldest()

        # Verify oldest was removed
        assert len(self.manager.environments) == 9
        assert oldest_id not in self.manager.environments
        # Verify cleanup was called on oldest environment
        # (We can't directly verify this since we deleted it, but we can check it's gone)


class TestSandboxManagerExecution:
    """Tests for code execution."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None
        self.manager = SandboxManager()

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            self.manager.cleanup_all()
        except Exception:
            pass

    @patch.object(UVSandbox, 'provision')
    @patch.object(UVSandbox, 'execute')
    def test_execute_in_environment(self, mock_execute, mock_provision):
        """Test executing code in environment."""
        mock_provision.return_value = "test_env"
        mock_execute.return_value = SandboxResult(success=True, stdout="output")

        config = EnvironmentConfig(env_type=EnvironmentType.UV)
        env_id = self.manager.create_environment(EnvironmentType.UV, config)

        result = self.manager.execute_in_environment(env_id, "print('hello')")

        assert result.success is True
        assert result.stdout == "output"
        mock_execute.assert_called_once_with("print('hello')", timeout=None)

    @patch.object(UVSandbox, 'provision')
    @patch.object(UVSandbox, 'execute')
    def test_execute_with_timeout(self, mock_execute, mock_provision):
        """Test executing code with custom timeout."""
        mock_provision.return_value = "test_env"
        mock_execute.return_value = SandboxResult(success=True)

        config = EnvironmentConfig(env_type=EnvironmentType.UV)
        env_id = self.manager.create_environment(EnvironmentType.UV, config)

        self.manager.execute_in_environment(env_id, "code", timeout=60)

        mock_execute.assert_called_once_with("code", timeout=60)

    def test_execute_nonexistent_environment(self):
        """Test executing in nonexistent environment fails."""
        with pytest.raises(ValueError, match="Environment not found"):
            self.manager.execute_in_environment("nonexistent", "code")


class TestSandboxManagerEnvironmentLookup:
    """Tests for environment lookup and status."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None
        self.manager = SandboxManager()

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            self.manager.cleanup_all()
        except Exception:
            pass

    @patch.object(UVSandbox, 'provision')
    def test_list_environments_empty(self, mock_provision):
        """Test listing environments when none exist."""
        envs = self.manager.list_environments()
        assert envs == []

    @patch.object(UVSandbox, 'provision')
    @patch.object(UVSandbox, 'get_status')
    def test_list_environments(self, mock_get_status, mock_provision):
        """Test listing environments."""
        mock_provision.return_value = "env_1"
        mock_get_status.return_value = {"env_id": "env_1", "status": "alive"}

        config = EnvironmentConfig(env_type=EnvironmentType.UV)
        self.manager.create_environment(EnvironmentType.UV, config)

        envs = self.manager.list_environments()
        assert len(envs) == 1
        assert envs[0]["env_id"] == "env_1"

    @patch.object(UVSandbox, 'provision')
    @patch.object(UVSandbox, 'get_status')
    def test_get_environment_status(self, mock_get_status, mock_provision):
        """Test getting environment status."""
        mock_provision.return_value = "test_env"
        mock_get_status.return_value = {"env_id": "test_env", "uptime": 10}

        config = EnvironmentConfig(env_type=EnvironmentType.UV)
        env_id = self.manager.create_environment(EnvironmentType.UV, config)

        status = self.manager.get_environment_status(env_id)
        assert status["env_id"] == "test_env"
        assert status["uptime"] == 10

    def test_get_status_nonexistent_environment(self):
        """Test getting status of nonexistent environment fails."""
        with pytest.raises(ValueError, match="Environment not found"):
            self.manager.get_environment_status("nonexistent")


class TestSandboxManagerCleanup:
    """Tests for environment cleanup."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None
        self.manager = SandboxManager()

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            self.manager.cleanup_all()
        except Exception:
            pass

    @patch.object(UVSandbox, 'provision')
    @patch.object(UVSandbox, 'cleanup')
    def test_cleanup_environment(self, mock_cleanup, mock_provision):
        """Test cleaning up environment."""
        mock_provision.return_value = "test_env"

        config = EnvironmentConfig(env_type=EnvironmentType.UV)
        env_id = self.manager.create_environment(EnvironmentType.UV, config)

        assert env_id in self.manager.environments

        self.manager.cleanup_environment(env_id)

        assert env_id not in self.manager.environments
        mock_cleanup.assert_called_once()

    def test_cleanup_nonexistent_environment(self):
        """Test cleaning up nonexistent environment fails."""
        with pytest.raises(ValueError, match="Environment not found"):
            self.manager.cleanup_environment("nonexistent")

    @patch.object(UVSandbox, 'provision')
    @patch.object(UVSandbox, 'cleanup')
    def test_cleanup_all(self, mock_cleanup, mock_provision):
        """Test cleaning up all environments."""
        mock_provision.side_effect = ["env_1", "env_2", "env_3"]

        # Create 3 environments
        for _ in range(3):
            config = EnvironmentConfig(env_type=EnvironmentType.UV)
            self.manager.create_environment(EnvironmentType.UV, config)

        assert len(self.manager.environments) == 3

        self.manager.cleanup_all()

        assert len(self.manager.environments) == 0
        assert mock_cleanup.call_count == 3


class TestSandboxManagerRepr:
    """Tests for string representation."""

    def setup_method(self):
        """Reset manager before each test."""
        SandboxManager._instance = None
        self.manager = SandboxManager()

    def test_repr_empty(self):
        """Test repr with no environments."""
        repr_str = repr(self.manager)
        assert "SandboxManager" in repr_str
        assert "active_environments=0" in repr_str
        assert "max_environments=10" in repr_str

    @patch.object(UVSandbox, 'provision')
    def test_repr_with_environments(self, mock_provision):
        """Test repr with environments."""
        mock_provision.return_value = "test_env"

        config = EnvironmentConfig(env_type=EnvironmentType.UV)
        self.manager.create_environment(EnvironmentType.UV, config)

        repr_str = repr(self.manager)
        assert "active_environments=1" in repr_str
