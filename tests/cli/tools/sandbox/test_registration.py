"""
Unit tests for sandbox tool registration with CLI registry.

Tests that all 5 agentic provisioning tools are properly registered
with the PromptChain CLI tool registry and can be discovered/executed.
"""

import pytest
from unittest.mock import Mock, patch

from promptchain.cli.tools import registry, ToolCategory


class TestSandboxToolRegistration:
    """Tests for sandbox tool registration in CLI registry."""

    def test_all_sandbox_tools_registered(self):
        """Test that all 5 sandbox tools are registered."""
        expected_tools = [
            "sandbox_provision_uv",
            "sandbox_provision_docker",
            "sandbox_execute",
            "sandbox_list",
            "sandbox_cleanup"
        ]

        all_tools = registry.list_tools()

        for tool_name in expected_tools:
            assert tool_name in all_tools, f"Tool '{tool_name}' not found in registry"

    def test_sandbox_tools_in_agent_category(self):
        """Test that sandbox tools are in the AGENT category."""
        agent_tools = registry.get_by_category(ToolCategory.AGENT)
        agent_tool_names = [t.name for t in agent_tools]

        sandbox_tools = [
            "sandbox_provision_uv",
            "sandbox_provision_docker",
            "sandbox_execute",
            "sandbox_list",
            "sandbox_cleanup"
        ]

        for tool_name in sandbox_tools:
            assert tool_name in agent_tool_names, \
                f"Sandbox tool '{tool_name}' not in AGENT category"

    def test_sandbox_provision_uv_metadata(self):
        """Test provision_uv_environment tool metadata."""
        tool = registry.get("sandbox_provision_uv")

        assert tool is not None
        assert tool.name == "sandbox_provision_uv"
        assert tool.category == ToolCategory.AGENT
        assert "UV" in tool.description or "uv" in tool.description.lower()
        assert "packages" in tool.parameters
        assert "python_version" in tool.parameters
        assert "environment_name" in tool.parameters

        # Check parameters
        packages_param = tool.parameters["packages"]
        assert packages_param.type == "array"
        assert packages_param.required is True

        python_param = tool.parameters["python_version"]
        assert python_param.type == "string"
        assert python_param.default == "3.12"

        # Check tags
        assert "sandbox" in tool.tags
        assert "python" in tool.tags or "uv" in tool.tags

    def test_sandbox_provision_docker_metadata(self):
        """Test provision_docker_environment tool metadata."""
        tool = registry.get("sandbox_provision_docker")

        assert tool is not None
        assert tool.name == "sandbox_provision_docker"
        assert tool.category == ToolCategory.AGENT
        assert "Docker" in tool.description or "docker" in tool.description.lower()

        # Check parameters
        assert "base_image" in tool.parameters
        assert "packages" in tool.parameters
        assert "gpu" in tool.parameters
        assert "network_enabled" in tool.parameters

        base_image_param = tool.parameters["base_image"]
        assert base_image_param.required is True

        gpu_param = tool.parameters["gpu"]
        assert gpu_param.type == "boolean"
        assert gpu_param.default is False

    def test_sandbox_execute_metadata(self):
        """Test execute_in_environment tool metadata."""
        tool = registry.get("sandbox_execute")

        assert tool is not None
        assert tool.name == "sandbox_execute"
        assert "execute" in tool.description.lower()

        # Check required parameters
        assert "environment_id" in tool.parameters
        assert "code" in tool.parameters
        assert "timeout" in tool.parameters

        env_id_param = tool.parameters["environment_id"]
        assert env_id_param.required is True

        code_param = tool.parameters["code"]
        assert code_param.required is True

        timeout_param = tool.parameters["timeout"]
        assert timeout_param.default == 300

    def test_sandbox_list_metadata(self):
        """Test list_environments tool metadata."""
        tool = registry.get("sandbox_list")

        assert tool is not None
        assert tool.name == "sandbox_list"
        assert "list" in tool.description.lower()

        # Should have no required parameters
        assert len(tool.parameters) == 0 or \
               len(tool.get_required_parameters()) == 0

    def test_sandbox_cleanup_metadata(self):
        """Test cleanup_environment tool metadata."""
        tool = registry.get("sandbox_cleanup")

        assert tool is not None
        assert tool.name == "sandbox_cleanup"
        assert "cleanup" in tool.description.lower() or \
               "destroy" in tool.description.lower()

        # Check parameters
        assert "environment_id" in tool.parameters
        env_id_param = tool.parameters["environment_id"]
        assert env_id_param.required is True

    def test_openai_schema_generation(self):
        """Test that sandbox tools generate valid OpenAI schemas."""
        sandbox_tools = [
            "sandbox_provision_uv",
            "sandbox_provision_docker",
            "sandbox_execute",
            "sandbox_list",
            "sandbox_cleanup"
        ]

        for tool_name in sandbox_tools:
            tool = registry.get(tool_name)
            schema = tool.to_openai_schema()

            # Validate OpenAI schema structure
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

            function = schema["function"]
            assert function["name"] == tool_name
            assert function["parameters"]["type"] == "object"
            assert "properties" in function["parameters"]
            assert "required" in function["parameters"]

    @patch('promptchain.cli.tools.sandbox.registration.provision_uv_environment')
    def test_sandbox_provision_uv_execution(self, mock_provision):
        """Test that sandbox_provision_uv can be executed via registry."""
        mock_provision.return_value = "✅ UV environment created: test_env"

        result = registry.execute(
            "sandbox_provision_uv",
            packages=["pandas", "numpy"],
            python_version="3.12"
        )

        mock_provision.assert_called_once_with(
            ["pandas", "numpy"],
            "3.12",
            None
        )
        assert "✅" in result
        assert "test_env" in result

    @patch('promptchain.cli.tools.sandbox.registration.provision_docker_environment')
    def test_sandbox_provision_docker_execution(self, mock_provision):
        """Test that sandbox_provision_docker can be executed via registry."""
        mock_provision.return_value = "✅ Docker environment created: docker_env"

        result = registry.execute(
            "sandbox_provision_docker",
            base_image="python:3.12-slim",
            packages=["pandas"],
            gpu=False,
            network_enabled=False
        )

        mock_provision.assert_called_once()
        assert "✅" in result

    @patch('promptchain.cli.tools.sandbox.registration.execute_in_environment')
    def test_sandbox_execute_execution(self, mock_execute):
        """Test that sandbox_execute can be executed via registry."""
        mock_execute.return_value = "✅ Execution successful (0.5s)\n\nOutput:\n42"

        result = registry.execute(
            "sandbox_execute",
            environment_id="test_env",
            code="print(42)",
            timeout=300
        )

        mock_execute.assert_called_once_with("test_env", "print(42)", 300)
        assert "✅" in result
        assert "42" in result

    @patch('promptchain.cli.tools.sandbox.registration.list_environments')
    def test_sandbox_list_execution(self, mock_list):
        """Test that sandbox_list can be executed via registry."""
        mock_list.return_value = "📦 Active Environments (2)"

        result = registry.execute("sandbox_list")

        mock_list.assert_called_once()
        assert "📦" in result

    @patch('promptchain.cli.tools.sandbox.registration.cleanup_environment')
    def test_sandbox_cleanup_execution(self, mock_cleanup):
        """Test that sandbox_cleanup can be executed via registry."""
        mock_cleanup.return_value = "✅ Environment test_env destroyed"

        result = registry.execute(
            "sandbox_cleanup",
            environment_id="test_env"
        )

        mock_cleanup.assert_called_once_with("test_env")
        assert "✅" in result

    def test_sandbox_tools_have_examples(self):
        """Test that all sandbox tools have usage examples."""
        sandbox_tools = [
            "sandbox_provision_uv",
            "sandbox_provision_docker",
            "sandbox_execute"
        ]

        for tool_name in sandbox_tools:
            tool = registry.get(tool_name)
            assert len(tool.examples) > 0, \
                f"Tool '{tool_name}' has no examples"

    def test_sandbox_tools_have_tags(self):
        """Test that all sandbox tools are properly tagged."""
        sandbox_tools = [
            "sandbox_provision_uv",
            "sandbox_provision_docker",
            "sandbox_execute",
            "sandbox_list",
            "sandbox_cleanup"
        ]

        for tool_name in sandbox_tools:
            tool = registry.get(tool_name)
            assert "sandbox" in tool.tags, \
                f"Tool '{tool_name}' missing 'sandbox' tag"
            assert len(tool.tags) > 1, \
                f"Tool '{tool_name}' should have multiple tags"

    def test_registry_can_filter_by_sandbox_tag(self):
        """Test that sandbox tools can be found by tag."""
        sandbox_tools = registry.get_by_tags(["sandbox"])

        sandbox_tool_names = [t.name for t in sandbox_tools]

        expected_tools = [
            "sandbox_provision_uv",
            "sandbox_provision_docker",
            "sandbox_execute",
            "sandbox_list",
            "sandbox_cleanup"
        ]

        for tool_name in expected_tools:
            assert tool_name in sandbox_tool_names, \
                f"Tool '{tool_name}' not found by 'sandbox' tag"

    def test_parameter_validation_for_provision_uv(self):
        """Test parameter validation for sandbox_provision_uv."""
        tool = registry.get("sandbox_provision_uv")

        # Valid parameters
        tool.validate_parameters({
            "packages": ["pandas", "numpy"],
            "python_version": "3.12"
        })

        # Invalid: missing required 'packages'
        with pytest.raises(Exception):  # ToolValidationError
            tool.validate_parameters({"python_version": "3.12"})

        # Invalid: wrong type for packages
        with pytest.raises(Exception):
            tool.validate_parameters({"packages": "pandas"})  # Should be array

    def test_parameter_validation_for_provision_docker(self):
        """Test parameter validation for sandbox_provision_docker."""
        tool = registry.get("sandbox_provision_docker")

        # Valid parameters
        tool.validate_parameters({
            "base_image": "python:3.12-slim",
            "gpu": True
        })

        # Invalid: missing required 'base_image'
        with pytest.raises(Exception):
            tool.validate_parameters({"gpu": True})

    def test_parameter_validation_for_execute(self):
        """Test parameter validation for sandbox_execute."""
        tool = registry.get("sandbox_execute")

        # Valid parameters
        tool.validate_parameters({
            "environment_id": "test_env",
            "code": "print('hello')",
            "timeout": 60
        })

        # Invalid: missing required parameters
        with pytest.raises(Exception):
            tool.validate_parameters({"code": "print('hello')"})  # Missing environment_id

        with pytest.raises(Exception):
            tool.validate_parameters({"environment_id": "test_env"})  # Missing code
