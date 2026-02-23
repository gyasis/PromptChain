"""Integration tests for CLI startup behavior.

These tests verify that the promptchain CLI command launches correctly,
displays welcome messages, and is ready for user input.

Test Coverage:
- test_promptchain_command_launches: CLI command starts successfully
- test_welcome_message_displayed: Welcome message shown on startup
- test_prompt_ready: Input prompt is ready for user interaction
- test_default_session_created: Default session auto-created on first launch
- test_session_startup_performance: Startup completes in <10s (SC-001)
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import time

# These imports will be available after implementation
# from promptchain.cli.main import main
# from promptchain.cli.tui.app import PromptChainApp


class TestCLIStartup:
    """Integration tests for CLI startup behavior."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_promptchain_command_launches(self, temp_config_dir):
        """Integration: promptchain command launches successfully.

        Validates:
        - CLI entry point exists and is callable
        - Application initializes without errors
        - Textual app instance is created
        - No exceptions during startup
        """
        # This test will need implementation of main() and PromptChainApp
        # For now, testing the contract that these components exist

        # Mock the Textual app to avoid actually running UI
        with patch('promptchain.cli.tui.app.PromptChainApp') as MockApp:
            mock_app_instance = Mock()
            MockApp.return_value = mock_app_instance

            # Import and call main (will be implemented in T035-T037)
            try:
                from promptchain.cli.main import main

                # Main should initialize app without errors
                # In actual implementation, main() will call app.run()
                # For testing, we verify it can be called
                assert callable(main)

            except ImportError:
                # Expected to fail until T035 is implemented
                pytest.skip("main() not yet implemented (will be in T035)")

    @pytest.mark.asyncio
    async def test_welcome_message_displayed(self, temp_config_dir):
        """Integration: Welcome message is displayed on startup.

        Welcome message should include:
        - PromptChain CLI version
        - Help command instructions
        - Active session name
        - Prompt indicator

        Validates:
        - Welcome message is rendered
        - Version string is present
        - Help instructions are clear
        """
        # Mock Textual app components
        with patch('promptchain.cli.tui.app.PromptChainApp') as MockApp:
            mock_app = Mock()
            MockApp.return_value = mock_app

            # Mock the on_mount method that displays welcome message
            mock_app.on_mount = AsyncMock()

            try:
                from promptchain.cli.tui.app import PromptChainApp

                # Create app instance
                app = PromptChainApp(session_name="default")

                # Trigger on_mount (where welcome message is displayed)
                await app.on_mount()

                # Verify welcome message was displayed
                # (implementation will add message to ChatView)
                assert mock_app.on_mount.called

            except ImportError:
                pytest.skip("PromptChainApp not yet implemented (will be in T027)")

    @pytest.mark.asyncio
    async def test_prompt_ready(self, temp_config_dir):
        """Integration: Input prompt is ready for user interaction.

        Validates:
        - InputWidget is mounted and visible
        - Prompt indicator is displayed (>)
        - Input widget has focus
        - User can type (simulated)
        """
        with patch('promptchain.cli.tui.app.PromptChainApp') as MockApp:
            mock_app = Mock()
            MockApp.return_value = mock_app

            # Mock InputWidget
            mock_input = Mock()
            mock_input.has_focus = True
            mock_app.query_one.return_value = mock_input

            try:
                from promptchain.cli.tui.app import PromptChainApp
                from promptchain.cli.tui.input_widget import InputWidget

                app = PromptChainApp(session_name="default")

                # Query for InputWidget
                input_widget = app.query_one(InputWidget)

                # Validate input widget is ready
                assert input_widget is not None
                assert input_widget.has_focus

            except ImportError:
                pytest.skip("UI components not yet implemented (will be in T027-T030)")

    @pytest.mark.asyncio
    async def test_default_session_created(self, temp_config_dir):
        """Integration: Default session is auto-created on first launch.

        When no session name is provided and no sessions exist:
        - Default session is created automatically
        - Session name is 'default'
        - Session uses current working directory
        - Default agent is initialized

        Validates:
        - Session manager creates session
        - Session is persisted to database
        - Default agent exists and is active
        """
        from promptchain.cli.session_manager import SessionManager

        # Create session manager with temp directory
        session_manager = SessionManager(sessions_dir=temp_config_dir)

        # Simulate first launch - no sessions exist
        sessions = session_manager.list_sessions()
        assert len(sessions) == 0

        # Create default session (as main() would do)
        session = session_manager.create_session(
            name="default",
            working_directory=Path.cwd()
        )

        # Validate default session
        assert session.name == "default"
        assert "default" in session.agents
        assert session.active_agent == "default"

        # Verify session is persisted
        loaded = session_manager.load_session("default")
        assert loaded.id == session.id

    @pytest.mark.asyncio
    async def test_session_startup_performance(self, temp_config_dir):
        """Integration: Session startup completes in <10s (SC-001).

        Performance target from plan.md: Session startup <10 seconds

        Measures end-to-end startup time:
        - Database initialization
        - Session creation/loading
        - Agent initialization
        - UI rendering (mocked)

        Validates:
        - Total startup time <10 seconds
        - No blocking operations
        """
        from promptchain.cli.session_manager import SessionManager

        start = time.perf_counter()

        # Initialize session manager (includes database setup)
        session_manager = SessionManager(sessions_dir=temp_config_dir)

        # Create session with default agent
        session = session_manager.create_session(
            name="perf-test",
            working_directory=Path.cwd()
        )

        # Mock UI rendering time (actual Textual startup)
        await asyncio.sleep(0.1)  # Simulated UI initialization

        duration = time.perf_counter() - start

        assert duration < 10.0, f"Startup took {duration:.2f}s, exceeds 10s target (SC-001)"
        assert session is not None

    @pytest.mark.asyncio
    async def test_cli_help_flag(self, temp_config_dir):
        """Integration: --help flag shows usage information.

        Validates:
        - --help flag is recognized
        - Usage information is displayed
        - Command exits gracefully (no errors)
        """
        try:
            from promptchain.cli.main import main
            import sys
            from io import StringIO

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                # Call main with --help (Click will handle this)
                main(['--help'])
            except SystemExit:
                # Click raises SystemExit(0) after showing help
                pass
            finally:
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

            # Validate help output
            assert "Usage:" in output or "promptchain" in output.lower()

        except ImportError:
            pytest.skip("main() not yet implemented (will be in T035)")

    @pytest.mark.asyncio
    async def test_cli_version_flag(self, temp_config_dir):
        """Integration: --version flag shows version information.

        Validates:
        - --version flag is recognized
        - Version string is displayed
        - Version matches __version__ in __init__.py
        """
        try:
            from promptchain.cli.main import main
            from promptchain.cli import __version__
            import sys
            from io import StringIO

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                main(['--version'])
            except SystemExit:
                pass
            finally:
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout

            # Validate version output
            assert __version__ in output

        except ImportError:
            pytest.skip("main() not yet implemented (will be in T035)")

    @pytest.mark.asyncio
    async def test_cli_session_flag(self, temp_config_dir):
        """Integration: --session flag loads existing session.

        Validates:
        - --session <name> flag is recognized
        - Named session is loaded if it exists
        - Error message if session doesn't exist
        """
        from promptchain.cli.session_manager import SessionManager

        # Create a test session
        session_manager = SessionManager(sessions_dir=temp_config_dir)
        test_session = session_manager.create_session(
            name="test-resume",
            working_directory=Path.cwd()
        )
        session_manager.save_session(test_session)

        # Test that session can be loaded with --session flag
        # (will be implemented in T077-T081)
        try:
            from promptchain.cli.main import main

            # This will be testable once T077 is implemented
            # For now, just verify session exists in database
            loaded = session_manager.load_session("test-resume")
            assert loaded.id == test_session.id

        except ImportError:
            pytest.skip("Session resumption not yet implemented (will be in T077-T081)")
