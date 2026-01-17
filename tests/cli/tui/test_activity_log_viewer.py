#!/usr/bin/env python3
"""
Test script for ActivityLogViewer widget (Phase 5).

Tests the TUI widget for interactive activity log viewing:
- Widget initialization
- Activity loading and display
- Search and filter functionality
- Statistics display
- Auto-refresh mechanism
- Keyboard shortcuts
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from promptchain import PromptChain
from promptchain.cli.activity_searcher import ActivitySearcher
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.tui.activity_log_viewer import ActivityLogItem, ActivityLogViewer
from promptchain.utils.agent_chain import AgentChain


class TestActivityLogItem:
    """Test ActivityLogItem component."""

    def test_activity_log_item_creation(self):
        """Test 1: ActivityLogItem can be created with activity data."""
        activity = {
            'timestamp': '2025-11-20T10:30:00',
            'agent_name': 'test_agent',
            'activity_type': 'agent_output',
            'content': {'message': 'Test response'}
        }

        item = ActivityLogItem(activity)

        assert item.activity == activity
        # Note: item.expanded is a reactive object, not directly comparable
        # We just check it exists
        assert hasattr(item, 'expanded')
        print("✓ ActivityLogItem created successfully")

    def test_activity_log_item_with_missing_fields(self):
        """Test 2: ActivityLogItem handles missing fields gracefully."""
        activity = {
            'timestamp': '2025-11-20T10:30:00'
            # Missing agent_name, activity_type, content
        }

        item = ActivityLogItem(activity)

        assert item.activity == activity
        # Should not crash with missing fields
        print("✓ ActivityLogItem handles missing fields")


class TestActivityLogViewerInitialization:
    """Test ActivityLogViewer initialization."""

    @pytest.fixture
    def temp_session_dir(self):
        """Create temporary session directory."""
        tmpdir = tempfile.mkdtemp()
        sessions_dir = Path(tmpdir) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Create activity log directories
        session_dir = sessions_dir / "test-session-id"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "activity_logs").mkdir(parents=True, exist_ok=True)

        yield sessions_dir, session_dir

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_viewer_initialization(self, temp_session_dir):
        """Test 3: ActivityLogViewer can be initialized."""
        sessions_dir, session_dir = temp_session_dir

        viewer = ActivityLogViewer(
            session_name="test-session",
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        assert viewer.session_name == "test-session"
        assert viewer.log_dir == session_dir / "activity_logs"
        assert viewer.db_path == session_dir / "activities.db"
        assert viewer.searcher is not None
        assert isinstance(viewer.searcher, ActivitySearcher)
        assert viewer.current_pattern == ""
        assert viewer.current_agent_filter == ""
        assert viewer.current_type_filter == ""
        assert viewer.activities == []
        assert viewer.total_activities == 0
        assert viewer.auto_refresh_enabled == False
        print("✓ ActivityLogViewer initialized correctly")


class TestActivityLogViewerWithData:
    """Test ActivityLogViewer with real activity data."""

    @pytest_asyncio.fixture
    async def session_with_activities(self):
        """Create session and log test activities."""
        tmpdir = tempfile.mkdtemp()
        sessions_dir = Path(tmpdir) / "sessions"

        session_manager = SessionManager(sessions_dir=sessions_dir)
        session = session_manager.create_session(
            name="test-viewer-session",
            working_directory=Path.cwd()
        )

        # Create agents and log activities
        agent1 = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Process: {input}"],
            verbose=False
        )

        agent2 = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Analyze: {input}"],
            verbose=False
        )

        agent_chain = AgentChain(
            agents={"processor": agent1, "analyzer": agent2},
            agent_descriptions={
                "processor": "Processes input",
                "analyzer": "Analyzes data"
            },
            execution_mode="pipeline",
            activity_logger=session.activity_logger,
            verbose=False
        )

        # Execute to generate activities
        await agent_chain.process_input("Test input 1")
        await agent_chain.process_input("Test input 2")

        yield session, session_manager, sessions_dir

        # Cleanup
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_load_activities(self, session_with_activities):
        """Test 4: ActivityLogViewer can load activities."""
        session, session_manager, sessions_dir = session_with_activities

        session_dir = sessions_dir / session.id
        viewer = ActivityLogViewer(
            session_name=session.name,
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Load activities
        viewer.load_activities(pattern=".*", limit=50)

        assert len(viewer.activities) > 0, "Expected to load activities"
        assert viewer.total_activities > 0, "Expected total_activities count"
        print(f"✓ Loaded {len(viewer.activities)} activities")

    @pytest.mark.asyncio
    async def test_search_activities(self, session_with_activities):
        """Test 5: ActivityLogViewer can search activities by pattern."""
        session, session_manager, sessions_dir = session_with_activities

        session_dir = sessions_dir / session.id
        viewer = ActivityLogViewer(
            session_name=session.name,
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Search for specific pattern
        viewer.current_pattern = "Test input"
        viewer.load_activities(pattern="Test input", limit=50)

        assert len(viewer.activities) > 0, "Expected to find activities matching pattern"
        print(f"✓ Found {len(viewer.activities)} activities matching 'Test input'")

    @pytest.mark.asyncio
    async def test_filter_by_agent(self, session_with_activities):
        """Test 6: ActivityLogViewer can filter activities by agent."""
        session, session_manager, sessions_dir = session_with_activities

        session_dir = sessions_dir / session.id
        viewer = ActivityLogViewer(
            session_name=session.name,
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Filter by agent
        viewer.current_agent_filter = "processor"
        viewer.load_activities(pattern=".*", limit=50)

        # All results should be from processor agent
        if len(viewer.activities) > 0:
            for activity in viewer.activities:
                # Some activities might not have agent_name (system activities)
                if 'agent_name' in activity and activity['agent_name']:
                    assert activity['agent_name'] == "processor", \
                        f"Expected processor, got {activity.get('agent_name')}"
        print(f"✓ Filtered to {len(viewer.activities)} processor activities")

    @pytest.mark.asyncio
    async def test_filter_by_type(self, session_with_activities):
        """Test 7: ActivityLogViewer can filter activities by type."""
        session, session_manager, sessions_dir = session_with_activities

        session_dir = sessions_dir / session.id
        viewer = ActivityLogViewer(
            session_name=session.name,
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Filter by type
        viewer.current_type_filter = "user_input"
        viewer.load_activities(pattern=".*", limit=50)

        # All results should be user_input type
        if len(viewer.activities) > 0:
            for activity in viewer.activities:
                assert activity.get('activity_type') == "user_input", \
                    f"Expected user_input, got {activity.get('activity_type')}"
        print(f"✓ Filtered to {len(viewer.activities)} user_input activities")

    @pytest.mark.asyncio
    async def test_statistics_retrieval(self, session_with_activities):
        """Test 8: ActivityLogViewer can retrieve statistics."""
        session, session_manager, sessions_dir = session_with_activities

        session_dir = sessions_dir / session.id
        viewer = ActivityLogViewer(
            session_name=session.name,
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Get statistics
        stats = viewer.searcher.get_statistics()

        assert 'total_activities' in stats
        assert 'total_chains' in stats
        assert 'activities_by_type' in stats
        assert 'activities_by_agent' in stats
        assert stats['total_activities'] > 0
        print(f"✓ Retrieved statistics: {stats['total_activities']} total activities")


class TestAutoRefresh:
    """Test auto-refresh mechanism."""

    @pytest.fixture
    def mock_viewer(self):
        """Create mock viewer for testing."""
        tmpdir = tempfile.mkdtemp()
        sessions_dir = Path(tmpdir) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_dir = sessions_dir / "test-id"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "activity_logs").mkdir(parents=True, exist_ok=True)

        viewer = ActivityLogViewer(
            session_name="test-session",
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        yield viewer

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_enable_auto_refresh(self, mock_viewer):
        """Test 9: Can enable auto-refresh."""
        viewer = mock_viewer

        assert viewer.auto_refresh_enabled == False
        assert viewer.refresh_task is None

        viewer.enable_auto_refresh(interval=1.0)

        assert viewer.auto_refresh_enabled == True
        assert viewer.refresh_task is not None
        print("✓ Auto-refresh enabled")

        # Cleanup
        viewer.disable_auto_refresh()

    @pytest.mark.asyncio
    async def test_disable_auto_refresh(self, mock_viewer):
        """Test 10: Can disable auto-refresh."""
        viewer = mock_viewer

        # Enable first
        viewer.enable_auto_refresh(interval=1.0)
        assert viewer.auto_refresh_enabled == True

        # Disable
        viewer.disable_auto_refresh()

        assert viewer.auto_refresh_enabled == False
        print("✓ Auto-refresh disabled")

    @pytest.mark.asyncio
    async def test_auto_refresh_loop(self, mock_viewer):
        """Test 11: Auto-refresh loop executes periodically."""
        viewer = mock_viewer

        # Track load_activities calls
        load_count = 0
        original_load = viewer.load_activities

        def mock_load(*args, **kwargs):
            nonlocal load_count
            load_count += 1
            # Don't actually load (no data available)

        viewer.load_activities = mock_load

        # Enable auto-refresh
        viewer.enable_auto_refresh(interval=0.1)  # Fast interval for testing

        # Wait for a few refresh cycles
        await asyncio.sleep(0.35)

        # Disable auto-refresh
        viewer.disable_auto_refresh()

        assert load_count >= 2, f"Expected at least 2 refreshes, got {load_count}"
        print(f"✓ Auto-refresh loop executed {load_count} times")


class TestViewerUpdates:
    """Test viewer update methods."""

    @pytest.fixture
    def viewer_with_mock_searcher(self):
        """Create viewer with mocked searcher."""
        tmpdir = tempfile.mkdtemp()
        sessions_dir = Path(tmpdir) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_dir = sessions_dir / "test-id"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "activity_logs").mkdir(parents=True, exist_ok=True)

        viewer = ActivityLogViewer(
            session_name="test-session",
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Mock searcher methods
        viewer.searcher.grep_logs = Mock(return_value=[
            {
                'timestamp': '2025-11-20T10:30:00',
                'agent_name': 'test_agent',
                'activity_type': 'agent_output',
                'content': {'message': 'Test 1'}
            },
            {
                'timestamp': '2025-11-20T10:31:00',
                'agent_name': 'test_agent',
                'activity_type': 'user_input',
                'content': {'input': 'Test 2'}
            }
        ])

        viewer.searcher.get_statistics = Mock(return_value={
            'total_activities': 2,
            'total_chains': 1,
            'active_chains': 0,
            'total_errors': 0,
            'avg_chain_depth': 2.0,
            'activities_by_type': {'agent_output': 1, 'user_input': 1},
            'activities_by_agent': {'test_agent': 2}
        })

        yield viewer

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_update_stats_display(self, viewer_with_mock_searcher):
        """Test 12: Can update stats display."""
        viewer = viewer_with_mock_searcher

        # Load activities to populate stats
        viewer.load_activities()

        assert len(viewer.activities) == 2
        assert viewer.total_activities == 2
        print("✓ Stats display updated correctly")

    def test_update_with_filters(self, viewer_with_mock_searcher):
        """Test 13: Stats display shows active filters."""
        viewer = viewer_with_mock_searcher

        viewer.current_pattern = "test"
        viewer.current_agent_filter = "test_agent"
        viewer.current_type_filter = "agent_output"

        # Pass pattern explicitly to load_activities
        viewer.load_activities(pattern="test")

        # Verify filters were passed to searcher
        viewer.searcher.grep_logs.assert_called_with(
            pattern="test",
            agent_name="test_agent",
            activity_type="agent_output",
            max_results=50
        )
        print("✓ Filters applied correctly")


def main():
    """Run all tests."""
    print("[bold magenta]ActivityLogViewer Tests (Phase 5)[/bold magenta]")
    print("=" * 60)

    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_",  # Run all test methods
    ])

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("[bold green]✓ All ActivityLogViewer Tests Passed![/bold green]")
        print("\n[dim]ActivityLogViewer widget is working correctly![/dim]")
    else:
        print("\n" + "=" * 60)
        print(f"[bold red]✗ Some tests failed (exit code: {exit_code})[/bold red]")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
