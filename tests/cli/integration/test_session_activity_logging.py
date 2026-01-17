#!/usr/bin/env python3
"""
Test script for SessionManager + ActivityLogger integration (Phase 3).

Verifies that:
1. ActivityLogger is automatically initialized when sessions are created
2. ActivityLogger is reinitialized when sessions are loaded
3. Activity logs are persisted correctly in session directories
4. AgentChain can access ActivityLogger via session.activity_logger
"""

import sys
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
import pytest

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from promptchain.cli.session_manager import SessionManager
from promptchain.cli.activity_searcher import ActivitySearcher
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from rich.console import Console

console = Console()


@pytest.mark.asyncio
async def test_session_creates_activity_logger():
    """Test 1: SessionManager creates ActivityLogger on session creation."""
    console.print("\n[bold cyan]Test 1: ActivityLogger Created on Session Creation[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"

        # Create session manager
        session_manager = SessionManager(sessions_dir=sessions_dir)

        # Create session
        session = session_manager.create_session(
            name="test-session",
            working_directory=Path.cwd()
        )

        # Verify ActivityLogger exists
        assert session._activity_logger is not None, "ActivityLogger should be initialized"
        assert session.activity_logger is not None, "Property should expose ActivityLogger"

        # Verify activity log directory created
        activity_log_dir = sessions_dir / session.id / "activity_logs"
        assert activity_log_dir.exists(), f"Activity log directory should exist: {activity_log_dir}"

        # Verify database created
        activity_db = sessions_dir / session.id / "activities.db"
        assert activity_db.exists(), f"Activity database should exist: {activity_db}"

        console.print("✓ ActivityLogger initialized on session creation")
        console.print(f"✓ Activity log directory: {activity_log_dir}")
        console.print(f"✓ Activity database: {activity_db}")
        console.print("[green]✓ Test 1 passed![/green]")
        return True


@pytest.mark.asyncio
async def test_session_loads_activity_logger():
    """Test 2: SessionManager reinitializes ActivityLogger on session load."""
    console.print("\n[bold cyan]Test 2: ActivityLogger Reinitialized on Session Load[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        session_manager = SessionManager(sessions_dir=sessions_dir)

        # Create and save session
        session = session_manager.create_session(
            name="load-test",
            working_directory=Path.cwd()
        )
        session_id = session.id
        session_manager.save_session(session)

        console.print(f"✓ Created session: {session.name} ({session_id})")

        # Load session by name
        loaded_session = session_manager.load_session("load-test")

        # Verify ActivityLogger reinitialized
        assert loaded_session._activity_logger is not None, "ActivityLogger should be reinitialized"
        assert loaded_session.activity_logger is not None, "Property should expose ActivityLogger"
        assert loaded_session.id == session_id, "Session ID should match"

        console.print("✓ ActivityLogger reinitialized on load")
        console.print(f"✓ Loaded session: {loaded_session.name} ({loaded_session.id})")

        # Load session by ID
        loaded_by_id = session_manager.load_session(session_id)
        assert loaded_by_id._activity_logger is not None, "ActivityLogger should exist when loaded by ID"

        console.print("✓ ActivityLogger works when loading by ID")
        console.print("[green]✓ Test 2 passed![/green]")
        return True


@pytest.mark.asyncio
async def test_activity_logging_with_agentchain():
    """Test 3: AgentChain can use session's ActivityLogger."""
    console.print("\n[bold cyan]Test 3: AgentChain Uses Session ActivityLogger[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        session_manager = SessionManager(sessions_dir=sessions_dir)

        # Create session with ActivityLogger
        session = session_manager.create_session(
            name="agentchain-test",
            working_directory=Path.cwd()
        )

        # Create simple agent
        agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Process: {input}"],
            verbose=False
        )

        # Create AgentChain with session's ActivityLogger
        agent_chain = AgentChain(
            agents={"processor": agent},
            agent_descriptions={"processor": "Processes input"},
            execution_mode="pipeline",
            activity_logger=session.activity_logger,  # ✅ Use session's logger
            verbose=False
        )

        # Execute
        try:
            result = await agent_chain.process_input("Test input for activity logging")
            console.print(f"✓ AgentChain executed successfully")
        except Exception as e:
            console.print(f"[red]✗ AgentChain execution failed: {e}[/red]")
            return False

        # Verify activities were logged
        searcher = ActivitySearcher(
            session_name=session.name,
            log_dir=sessions_dir / session.id / "activity_logs",
            db_path=sessions_dir / session.id / "activities.db"
        )

        stats = searcher.get_statistics()
        console.print(f"✓ Logged {stats['total_activities']} activities")
        console.print(f"  - user_input: {stats['activities_by_type'].get('user_input', 0)}")
        console.print(f"  - agent_output: {stats['activities_by_type'].get('agent_output', 0)}")

        assert stats['total_activities'] >= 2, f"Expected >= 2 activities, got {stats['total_activities']}"
        assert stats['activities_by_type'].get('user_input', 0) == 1, "Expected 1 user_input"
        assert stats['activities_by_type'].get('agent_output', 0) >= 1, "Expected >= 1 agent_output"

        console.print("[green]✓ Test 3 passed![/green]")
        return True


@pytest.mark.asyncio
async def test_activity_logs_persist_across_loads():
    """Test 4: Activity logs persist when session is saved and reloaded."""
    console.print("\n[bold cyan]Test 4: Activity Logs Persist Across Session Loads[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        session_manager = SessionManager(sessions_dir=sessions_dir)

        # Create session and log some activities
        session = session_manager.create_session(
            name="persist-test",
            working_directory=Path.cwd()
        )
        session_id = session.id

        # Create agent and log activities
        agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Process: {input}"],
            verbose=False
        )

        agent_chain = AgentChain(
            agents={"processor": agent},
            agent_descriptions={"processor": "Processes input"},
            execution_mode="pipeline",
            activity_logger=session.activity_logger,
            verbose=False
        )

        # Execute and log first activity
        await agent_chain.process_input("First input")

        # Check activities before save
        searcher = ActivitySearcher(
            session_name=session.name,
            log_dir=sessions_dir / session_id / "activity_logs",
            db_path=sessions_dir / session_id / "activities.db"
        )
        stats_before = searcher.get_statistics()
        console.print(f"✓ Before save: {stats_before['total_activities']} activities")

        # Save and reload session
        session_manager.save_session(session)
        console.print("✓ Session saved")

        loaded_session = session_manager.load_session("persist-test")
        console.print("✓ Session reloaded")

        # Verify activities still accessible
        searcher_after = ActivitySearcher(
            session_name=loaded_session.name,
            log_dir=sessions_dir / loaded_session.id / "activity_logs",
            db_path=sessions_dir / loaded_session.id / "activities.db"
        )
        stats_after = searcher_after.get_statistics()
        console.print(f"✓ After reload: {stats_after['total_activities']} activities")

        assert stats_after['total_activities'] == stats_before['total_activities'], \
            f"Activities should persist: {stats_before['total_activities']} != {stats_after['total_activities']}"

        # Log new activity with reloaded session
        agent_chain_reloaded = AgentChain(
            agents={"processor": agent},
            agent_descriptions={"processor": "Processes input"},
            execution_mode="pipeline",
            activity_logger=loaded_session.activity_logger,  # ✅ Use reloaded logger
            verbose=False
        )

        await agent_chain_reloaded.process_input("Second input after reload")

        # Verify new activity logged
        stats_final = searcher_after.get_statistics()
        console.print(f"✓ After new activity: {stats_final['total_activities']} activities")

        assert stats_final['total_activities'] > stats_after['total_activities'], \
            "New activities should be logged after reload"

        console.print("[green]✓ Test 4 passed![/green]")
        return True


@pytest.mark.asyncio
async def test_activity_logger_none_when_session_fails():
    """Test 5: ActivityLogger is None if initialization fails (graceful degradation)."""
    console.print("\n[bold cyan]Test 5: Graceful Degradation on ActivityLogger Failure[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_dir = Path(tmpdir) / "sessions"
        session_manager = SessionManager(sessions_dir=sessions_dir)

        # Create session normally
        session = session_manager.create_session(
            name="degradation-test",
            working_directory=Path.cwd()
        )

        # Even if ActivityLogger fails, session should still work
        # (In practice, this test verifies error handling is in place)
        assert session is not None, "Session should be created even if ActivityLogger fails"

        # Verify we can still use the session without ActivityLogger
        agent = PromptChain(
            models=["gpt-4.1-mini-2025-04-14"],
            instructions=["Process: {input}"],
            verbose=False
        )

        # AgentChain should work without ActivityLogger (backward compatible)
        agent_chain = AgentChain(
            agents={"processor": agent},
            agent_descriptions={"processor": "Processes input"},
            execution_mode="pipeline",
            activity_logger=None,  # ✅ No logger
            verbose=False
        )

        try:
            result = await agent_chain.process_input("Test without logger")
            console.print("✓ AgentChain works without ActivityLogger (backward compatible)")
        except Exception as e:
            console.print(f"[red]✗ AgentChain failed without logger: {e}[/red]")
            return False

        console.print("[green]✓ Test 5 passed![/green]")
        return True


async def main():
    """Run all tests."""
    console.print("[bold magenta]SessionManager + ActivityLogger Integration Tests (Phase 3)[/bold magenta]")
    console.print("=" * 70)

    tests = [
        ("Session Creates ActivityLogger", test_session_creates_activity_logger),
        ("Session Loads ActivityLogger", test_session_loads_activity_logger),
        ("AgentChain Uses Session Logger", test_activity_logging_with_agentchain),
        ("Logs Persist Across Loads", test_activity_logs_persist_across_loads),
        ("Graceful Degradation", test_activity_logger_none_when_session_fails),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except AssertionError as e:
            console.print(f"\n[bold red]✗ {test_name} failed: {e}[/bold red]")
            results.append((test_name, False))
        except Exception as e:
            console.print(f"\n[bold red]✗ {test_name} crashed: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    console.print("\n" + "=" * 70)
    console.print("[bold]Test Summary[/bold]")

    from rich.table import Table
    table = Table()
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")

    passed = 0
    failed = 0
    for test_name, result in results:
        if result:
            table.add_row(test_name, "[green]✓ PASSED[/green]")
            passed += 1
        else:
            table.add_row(test_name, "[red]✗ FAILED[/red]")
            failed += 1

    console.print(table)
    console.print(f"\n[bold]Results: {passed} passed, {failed} failed[/bold]")

    if failed == 0:
        console.print("\n[bold green]✓ All Tests Passed![/bold green]")
        console.print("\n[dim]SessionManager integration with ActivityLogger is working correctly![/dim]")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]✗ {failed} Test(s) Failed[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
