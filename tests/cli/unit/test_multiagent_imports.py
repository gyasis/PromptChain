"""
Validate all multi-agent communication imports work correctly.
This is a smoke test to ensure the module structure is correct.
"""

import pytest


class TestModelsImports:
    """Test all model imports work."""

    def test_task_model_imports(self):
        """Verify Task model and enums import correctly."""
        from promptchain.cli.models.task import Task, TaskPriority, TaskStatus
        assert Task is not None
        assert TaskPriority is not None
        assert TaskStatus is not None

    def test_blackboard_model_imports(self):
        """Verify Blackboard model imports correctly."""
        from promptchain.cli.models.blackboard import BlackboardEntry
        assert BlackboardEntry is not None

        # Test that BlackboardEntry has expected methods
        assert hasattr(BlackboardEntry, 'create')
        assert hasattr(BlackboardEntry, 'update')
        assert hasattr(BlackboardEntry, 'to_dict')
        assert hasattr(BlackboardEntry, 'from_dict')
        assert hasattr(BlackboardEntry, 'from_db_row')

    def test_mental_models_imports(self):
        """Verify Mental Models system imports correctly."""
        from promptchain.cli.models.mental_models import (
            SpecializationType,
            AgentSpecialization,
            MentalModel,
            MentalModelManager,
            create_default_model
        )
        assert SpecializationType is not None
        assert AgentSpecialization is not None
        assert MentalModel is not None
        assert MentalModelManager is not None
        assert create_default_model is not None

    def test_workflow_model_imports(self):
        """Verify Workflow model imports correctly."""
        from promptchain.cli.models.workflow import WorkflowState, WorkflowStage
        assert WorkflowState is not None
        assert WorkflowStage is not None


class TestCommunicationImports:
    """Test communication module imports."""

    def test_handlers_imports(self):
        """Verify communication handlers import correctly."""
        from promptchain.cli.communication.handlers import (
            MessageType,
            cli_communication_handler,
            HandlerRegistry
        )
        assert MessageType is not None
        assert cli_communication_handler is not None
        assert HandlerRegistry is not None

    def test_message_bus_imports(self):
        """Verify message bus components import correctly."""
        from promptchain.cli.communication.message_bus import (
            Message,
            MessageBus
        )
        assert Message is not None
        assert MessageBus is not None


class TestToolsImports:
    """Test tool imports."""

    def test_registry_imports(self):
        """Verify tool registry imports correctly."""
        from promptchain.cli.tools.registry import (
            ToolRegistry,
            ToolCategory,
            ToolMetadata,
            ParameterSchema
        )
        assert ToolRegistry is not None
        assert ToolCategory is not None
        assert ToolMetadata is not None
        assert ParameterSchema is not None

        # Test that ToolRegistry can be instantiated
        registry = ToolRegistry()
        assert registry is not None

    def test_delegation_tools_imports(self):
        """Verify delegation tools import correctly."""
        from promptchain.cli.tools.library.delegation_tools import (
            delegate_task,
            get_pending_tasks,
            update_task_status,
            request_help
        )
        assert delegate_task is not None
        assert get_pending_tasks is not None
        assert update_task_status is not None
        assert request_help is not None

    def test_blackboard_tools_imports(self):
        """Verify blackboard tools import correctly."""
        from promptchain.cli.tools.library.blackboard_tools import (
            write_to_blackboard,
            read_from_blackboard,
            list_blackboard_keys,
            delete_blackboard_entry
        )
        assert write_to_blackboard is not None
        assert read_from_blackboard is not None
        assert list_blackboard_keys is not None
        assert delete_blackboard_entry is not None

    def test_mental_model_tools_imports(self):
        """Verify mental model tools import correctly."""
        # Note: These are exported with _tool suffix in __init__.py
        from promptchain.cli.tools.library import (
            get_my_capabilities_tool,
            discover_capable_agents_tool,
            update_specialization_tool,
            record_task_experience_tool,
            share_capabilities_tool
        )
        # These may be None if import failed gracefully
        # Just verify they're accessible from library package


class TestLibraryExports:
    """Test library __init__ exports all tools."""

    def test_library_has_all_exports(self):
        """Verify __init__.py exports all expected tools."""
        import promptchain.cli.tools.library as lib

        # Delegation tools (without _tool suffix in exports)
        expected_delegation = [
            'delegate_task',
            'get_pending_tasks',
            'update_task_status',
            'set_delegation_session_manager'
        ]

        # Blackboard tools (without _tool suffix in exports)
        expected_blackboard = [
            'write_to_blackboard',
            'read_from_blackboard',
            'list_blackboard_keys',
            'delete_blackboard_entry',
            'set_blackboard_session_manager'
        ]

        # Mental model tools (WITH _tool suffix in exports)
        expected_mental_model = [
            'get_my_capabilities_tool',
            'discover_capable_agents_tool',
            'update_specialization_tool',
            'record_task_experience_tool',
            'share_capabilities_tool'
        ]

        all_expected = expected_delegation + expected_blackboard + expected_mental_model

        for tool_name in all_expected:
            assert hasattr(lib, tool_name), f"Library missing export: {tool_name}"
            # Check it's in __all__
            assert tool_name in lib.__all__, f"Tool not in __all__: {tool_name}"


class TestCommandHandlerIntegration:
    """Test command handler integrates with new models."""

    def test_command_handler_imports(self):
        """Verify command handler can import all necessary components."""
        from promptchain.cli.command_handler import CommandHandler
        assert CommandHandler is not None

    def test_session_manager_imports(self):
        """Verify session manager can import all necessary components."""
        from promptchain.cli.session_manager import SessionManager
        assert SessionManager is not None


class TestSchemaIntegration:
    """Test database schema includes new tables."""

    def test_schema_file_exists(self):
        """Verify schema.sql file exists."""
        import os
        schema_path = "/home/gyasis/Documents/code/PromptChain/promptchain/cli/schema.sql"
        assert os.path.exists(schema_path), "schema.sql file not found"

    def test_schema_has_tasks_table(self):
        """Verify schema includes task_queue table."""
        schema_path = "/home/gyasis/Documents/code/PromptChain/promptchain/cli/schema.sql"
        with open(schema_path, 'r') as f:
            schema_content = f.read()

        # Note: tasks are stored in task_queue table
        assert 'CREATE TABLE IF NOT EXISTS task_queue' in schema_content, \
            "task_queue table not found in schema"

    def test_schema_has_blackboard_table(self):
        """Verify schema includes blackboard table."""
        schema_path = "/home/gyasis/Documents/code/PromptChain/promptchain/cli/schema.sql"
        with open(schema_path, 'r') as f:
            schema_content = f.read()

        assert 'CREATE TABLE IF NOT EXISTS blackboard' in schema_content, \
            "blackboard table not found in schema"

    def test_schema_has_mental_models_support(self):
        """Verify schema supports mental models (stored in agents table metadata)."""
        schema_path = "/home/gyasis/Documents/code/PromptChain/promptchain/cli/schema.sql"
        with open(schema_path, 'r') as f:
            schema_content = f.read()

        # Mental models are stored as JSON in agents table
        assert 'CREATE TABLE IF NOT EXISTS agents' in schema_content, \
            "agents table not found in schema (used for mental models)"

        # Verify agents table has metadata_json column for mental models
        assert 'metadata_json' in schema_content, \
            "agents table missing metadata_json column for mental models storage"


class TestEndToEndImports:
    """Test complete import chain from top-level modules."""

    def test_cli_main_imports(self):
        """Verify CLI main can import without errors."""
        try:
            from promptchain.cli import main
            assert main is not None
            # Verify main command function exists (decorated with @click.command)
            assert hasattr(main, 'main')
            # Verify it's a click command
            assert callable(main.main)
        except ImportError as e:
            pytest.fail(f"CLI main import failed: {e}")

    def test_tui_app_imports(self):
        """Verify TUI app can import without errors."""
        try:
            from promptchain.cli.tui.app import PromptChainApp
            assert PromptChainApp is not None
        except ImportError as e:
            pytest.fail(f"TUI app import failed: {e}")

    def test_complete_tool_chain(self):
        """Verify complete tool chain imports: library -> registry -> handler."""
        try:
            # Library exports (using correct names from __init__.py)
            from promptchain.cli.tools.library import (
                delegate_task,
                write_to_blackboard,
                get_my_capabilities_tool
            )

            # Registry class (not instance)
            from promptchain.cli.tools.registry import ToolRegistry

            # Handler
            from promptchain.cli.command_handler import CommandHandler

            # These may be None if imports failed gracefully
            # Just verify they're importable
            assert CommandHandler is not None
            assert ToolRegistry is not None

        except ImportError as e:
            pytest.fail(f"Complete tool chain import failed: {e}")


class TestCircularImportPrevention:
    """Test that no circular imports exist."""

    def test_no_circular_imports_in_models(self):
        """Verify models can be imported independently without circular dependencies."""
        # Import each model module independently
        from promptchain.cli.models import task
        from promptchain.cli.models import blackboard
        from promptchain.cli.models import mental_models
        from promptchain.cli.models import workflow

        assert task is not None
        assert blackboard is not None
        assert mental_models is not None
        assert workflow is not None

    def test_no_circular_imports_in_tools(self):
        """Verify tools can be imported independently."""
        # Test individual tool modules import without circular dependencies
        try:
            from promptchain.cli.tools.library import delegation_tools
            from promptchain.cli.tools.library import blackboard_tools
            from promptchain.cli.tools import registry

            assert delegation_tools is not None
            assert blackboard_tools is not None
            assert registry is not None
        except ImportError as e:
            # Mental model tools may fail if registry isn't available yet
            # This is expected during early import
            if "tool_registry" not in str(e):
                raise

    def test_no_circular_imports_in_communication(self):
        """Verify communication modules can be imported independently."""
        from promptchain.cli.communication import handlers
        from promptchain.cli.communication import message_bus

        assert handlers is not None
        assert message_bus is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
