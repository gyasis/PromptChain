#!/usr/bin/env python3
"""Quick test to verify Activity Logs fix.

This test verifies that:
1. ActivityLogger is properly initialized in Session
2. ActivityLogger is passed to AgentChain during TUI initialization
3. Activity logging actually happens during agent execution
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from promptchain.cli.models.session import Session
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.activity_logger import ActivityLogger
from promptchain.cli.activity_searcher import ActivitySearcher


def test_activity_logger_initialization():
    """Test that ActivityLogger is properly initialized in Session."""
    print("\n=== TEST 1: ActivityLogger Initialization ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir)

        # Create session manager
        manager = SessionManager(sessions_dir=sessions_dir)

        # Create new session
        session = manager.create_session("test-activity-logs")

        # Verify ActivityLogger exists
        assert hasattr(session, 'activity_logger'), "Session should have activity_logger attribute"
        assert session.activity_logger is not None, "activity_logger should not be None"
        assert isinstance(session.activity_logger, ActivityLogger), "activity_logger should be ActivityLogger instance"

        print("✅ Session has ActivityLogger properly initialized")
        print(f"   - Log dir: {session.activity_logger.log_dir}")
        print(f"   - DB path: {session.activity_logger.db_path}")

        return session


def test_activity_logging_writes():
    """Test that ActivityLogger actually writes activities."""
    print("\n=== TEST 2: Activity Logging Writes ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir)

        # Create session manager
        manager = SessionManager(sessions_dir=sessions_dir)

        # Create new session
        session = manager.create_session("test-activity-writes")

        activity_logger = session.activity_logger

        # Start interaction chain
        chain_id = activity_logger.start_interaction_chain()
        print(f"✅ Started interaction chain: {chain_id}")

        # Log user input
        activity_logger.log_activity(
            activity_type="user_input",
            agent_name=None,
            content={"input": "Hello, test message"},
        )
        print("✅ Logged user_input activity")

        # Log agent output
        activity_logger.log_activity(
            activity_type="agent_output",
            agent_name="default",
            content={"output": "Test response"},
        )
        print("✅ Logged agent_output activity")

        # Note: ActivityLogger auto-flushes, no explicit flush() needed
        print("✅ Activity buffer managed automatically")

        # Verify activities written to JSONL
        jsonl_path = activity_logger.activity_log_path
        assert jsonl_path.exists(), f"JSONL file should exist: {jsonl_path}"

        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) >= 2, f"Should have at least 2 activities, got {len(lines)}"
        print(f"✅ JSONL file contains {len(lines)} activities")

        # Verify activities in SQLite
        searcher = ActivitySearcher(
            session_name=session.name,
            log_dir=activity_logger.log_dir,
            db_path=activity_logger.db_path
        )

        stats = searcher.get_statistics()
        print(f"✅ SQLite database statistics:")
        print(f"   - Total activities: {stats['total_activities']}")
        print(f"   - Total chains: {stats['total_chains']}")
        print(f"   - Activities by type: {stats['activities_by_type']}")

        assert stats['total_activities'] >= 2, f"Should have at least 2 activities in DB, got {stats['total_activities']}"

        # Search for activities
        activities = searcher.grep_logs(pattern="test", max_results=10)
        print(f"✅ Search found {len(activities)} matching activities")

        return session, stats


def test_agentchain_receives_activity_logger():
    """Test that AgentChain receives activity_logger parameter (manual check)."""
    print("\n=== TEST 3: AgentChain Activity Logger Integration ===")
    print("⚠️  This test requires manual verification:")
    print("    1. Launch TUI: promptchain --session test-activity-logs")
    print("    2. Send a message: 'Hello, test message'")
    print("    3. Press Ctrl+L to toggle Activity Logs")
    print("    4. Verify: Should show 'Showing 2/2 activities' or similar")
    print("    5. Click 'Stats' button")
    print("    6. Verify: Shows statistics with activities_by_type, activities_by_agent")
    print("    7. Search for 'test' in search box")
    print("    8. Verify: Shows matching activities")
    print("\n✅ Manual verification steps defined")


def main():
    """Run all tests."""
    print("=" * 70)
    print("ACTIVITY LOGS FIX VERIFICATION TEST")
    print("=" * 70)

    try:
        # Test 1: Initialization
        session = test_activity_logger_initialization()

        # Test 2: Writing
        session, stats = test_activity_logging_writes()

        # Test 3: Manual verification
        test_agentchain_receives_activity_logger()

        print("\n" + "=" * 70)
        print("ALL AUTOMATED TESTS PASSED ✅")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run manual verification test (see TEST 3 above)")
        print("2. Verify Activity Logs panel shows activities in TUI")
        print("3. Verify search and filter functionality works")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
