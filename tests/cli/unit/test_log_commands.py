#!/usr/bin/env python3
"""
Test script for /log commands (Phase 4).

Tests all activity log slash commands:
- /log search <pattern>
- /log agent <agent_name>
- /log errors
- /log stats
- /log chain <chain_id>
"""

import sys
import asyncio
import tempfile
from pathlib import Path
import pytest

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from promptchain.cli.session_manager import SessionManager
from promptchain.cli.command_handler import CommandHandler
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from rich.console import Console

console = Console()


async def setup_session_with_activities():
    """Create a session and log some test activities."""
    tmpdir = tempfile.mkdtemp()
    sessions_dir = Path(tmpdir) / "sessions"

    # Create session manager and session
    session_manager = SessionManager(sessions_dir=sessions_dir)
    session = session_manager.create_session(
        name="log-test",
        working_directory=Path.cwd()
    )

    # Create agents
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

    # Log some activities using AgentChain
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

    # Create command handler
    command_handler = CommandHandler(session_manager)

    return session, session_manager, command_handler, sessions_dir


@pytest.mark.asyncio
async def test_log_search_command():
    """Test 1: /log search command."""
    console.print("\n[bold cyan]Test 1: /log search Command[/bold cyan]")

    session, session_manager, handler, sessions_dir = await setup_session_with_activities()

    try:
        # Test basic search
        result = handler.handle_log_search(
            session=session,
            pattern="Test input",
            limit=10
        )

        assert result.success, f"Search failed: {result.error}"
        assert result.data['count'] > 0, "Expected to find activities"

        console.print(f"✓ Found {result.data['count']} activities")

        # Test search with agent filter
        result_filtered = handler.handle_log_search(
            session=session,
            pattern=".*",
            agent="processor",
            limit=10
        )

        assert result_filtered.success, f"Filtered search failed: {result_filtered.error}"
        console.print(f"✓ Filtered search found {result_filtered.data['count']} processor activities")

        # Test search with type filter
        result_type = handler.handle_log_search(
            session=session,
            pattern=".*",
            type="user_input",
            limit=10
        )

        assert result_type.success, f"Type search failed: {result_type.error}"
        console.print(f"✓ Type search found {result_type.data['count']} user_input activities")

        console.print("[green]✓ Test 1 passed![/green]")
        return True

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_agent_command():
    """Test 2: /log agent command."""
    console.print("\n[bold cyan]Test 2: /log agent Command[/bold cyan]")

    session, session_manager, handler, sessions_dir = await setup_session_with_activities()

    try:
        # Test agent activities
        result = handler.handle_log_agent(
            session=session,
            agent_name="processor",
            limit=20
        )

        assert result.success, f"Agent command failed: {result.error}"
        assert result.data['count'] > 0, "Expected to find processor activities"
        assert result.data['agent'] == "processor", "Agent name mismatch"

        console.print(f"✓ Found {result.data['count']} activities for 'processor' agent")

        # Test non-existent agent
        result_none = handler.handle_log_agent(
            session=session,
            agent_name="nonexistent",
            limit=20
        )

        assert result_none.success, "Should succeed even with no results"
        assert result_none.data['count'] == 0, "Expected 0 activities for nonexistent agent"

        console.print("✓ Correctly handled non-existent agent")
        console.print("[green]✓ Test 2 passed![/green]")
        return True

    finally:
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_errors_command():
    """Test 3: /log errors command."""
    console.print("\n[bold cyan]Test 3: /log errors Command[/bold cyan]")

    session, session_manager, handler, sessions_dir = await setup_session_with_activities()

    try:
        # Test errors (should be 0 for successful executions)
        result = handler.handle_log_errors(
            session=session,
            limit=10
        )

        assert result.success, f"Errors command failed: {result.error}"
        console.print(f"✓ Found {result.data['count']} errors (expected 0 for successful runs)")

        # Verify message for no errors
        if result.data['count'] == 0:
            assert "🎉" in result.message, "Expected celebration message for no errors"
            console.print("✓ Correct message for no errors")

        console.print("[green]✓ Test 3 passed![/green]")
        return True

    finally:
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_stats_command():
    """Test 4: /log stats command."""
    console.print("\n[bold cyan]Test 4: /log stats Command[/bold cyan]")

    session, session_manager, handler, sessions_dir = await setup_session_with_activities()

    try:
        # Test statistics
        result = handler.handle_log_stats(session=session)

        assert result.success, f"Stats command failed: {result.error}"
        assert 'total_activities' in result.data, "Missing total_activities"
        assert 'total_chains' in result.data, "Missing total_chains"
        assert 'activities_by_type' in result.data, "Missing activities_by_type"
        assert 'activities_by_agent' in result.data, "Missing activities_by_agent"

        stats = result.data
        console.print(f"✓ Total Activities: {stats['total_activities']}")
        console.print(f"✓ Total Chains: {stats['total_chains']}")
        console.print(f"✓ Total Errors: {stats['total_errors']}")

        # Verify stats make sense
        assert stats['total_activities'] > 0, "Expected some activities"
        assert stats['total_chains'] >= 2, "Expected at least 2 chains (2 process_input calls)"

        console.print("[green]✓ Test 4 passed![/green]")
        return True

    finally:
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_chain_command():
    """Test 5: /log chain command."""
    console.print("\n[bold cyan]Test 5: /log chain Command[/bold cyan]")

    session, session_manager, handler, sessions_dir = await setup_session_with_activities()

    try:
        # First, get stats to find a chain_id
        from promptchain.cli.activity_searcher import ActivitySearcher

        session_dir = sessions_dir / session.id
        searcher = ActivitySearcher(
            session_name=session.name,
            log_dir=session_dir / "activity_logs",
            db_path=session_dir / "activities.db"
        )

        # Get all chains
        chains_query = searcher.sql_query(
            "SELECT chain_id FROM interaction_chains WHERE session_name = ? LIMIT 1",
            (session.name,)
        )

        if not chains_query:
            console.print("[yellow]⚠ No chains found, skipping chain test[/yellow]")
            return True

        chain_id = chains_query[0]['chain_id']
        console.print(f"✓ Found chain ID: {chain_id[:20]}...")

        # Test chain retrieval
        result = handler.handle_log_chain(
            session=session,
            chain_id=chain_id
        )

        assert result.success, f"Chain command failed: {result.error}"
        assert 'chain_id' in result.data or 'status' in result.data, "Missing chain data"

        console.print(f"✓ Retrieved chain with {result.data.get('total_activities', 0)} activities")

        # Test non-existent chain
        result_none = handler.handle_log_chain(
            session=session,
            chain_id="nonexistent_chain_id"
        )

        assert not result_none.success, "Should fail for non-existent chain"
        console.print("✓ Correctly handled non-existent chain")

        console.print("[green]✓ Test 5 passed![/green]")
        return True

    finally:
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_log_commands_without_activity_logger():
    """Test 6: /log commands fail gracefully without ActivityLogger."""
    console.print("\n[bold cyan]Test 6: Graceful Failure Without ActivityLogger[/bold cyan]")

    tmpdir = tempfile.mkdtemp()
    sessions_dir = Path(tmpdir) / "sessions"

    try:
        # Create session WITHOUT logging any activities
        session_manager = SessionManager(sessions_dir=sessions_dir)
        session = session_manager.create_session(
            name="no-logger-test",
            working_directory=Path.cwd()
        )

        # Artificially remove ActivityLogger to test error handling
        session._activity_logger = None

        command_handler = CommandHandler(session_manager)

        # Test all commands - should fail gracefully
        commands = [
            ("search", lambda: command_handler.handle_log_search(session, "test")),
            ("agent", lambda: command_handler.handle_log_agent(session, "test")),
            ("errors", lambda: command_handler.handle_log_errors(session)),
            ("stats", lambda: command_handler.handle_log_stats(session)),
            ("chain", lambda: command_handler.handle_log_chain(session, "test")),
        ]

        for name, cmd in commands:
            result = cmd()
            assert not result.success, f"Command '{name}' should fail without ActivityLogger"
            assert "Activity logging not enabled" in result.message, \
                f"Command '{name}' should have correct error message"
            console.print(f"✓ /{name} failed gracefully: {result.message}")

        console.print("[green]✓ Test 6 passed![/green]")
        return True

    finally:
        import shutil
        shutil.rmtree(sessions_dir, ignore_errors=True)


async def main():
    """Run all tests."""
    console.print("[bold magenta]/log Command Tests (Phase 4)[/bold magenta]")
    console.print("=" * 60)

    tests = [
        ("log search", test_log_search_command),
        ("log agent", test_log_agent_command),
        ("log errors", test_log_errors_command),
        ("log stats", test_log_stats_command),
        ("log chain", test_log_chain_command),
        ("graceful failure", test_log_commands_without_activity_logger),
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
    console.print("\n" + "=" * 60)
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
        console.print("\n[dim]/log commands are working correctly![/dim]")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]✗ {failed} Test(s) Failed[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
