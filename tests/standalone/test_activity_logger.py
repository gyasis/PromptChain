#!/usr/bin/env python3
"""
Test script for ActivityLogger and ActivitySearcher functionality.

This script demonstrates and tests the comprehensive agent activity logging
system including JSONL persistence, SQLite indexing, and search capabilities.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from promptchain.cli.activity_logger import ActivityLogger
from promptchain.cli.activity_searcher import ActivitySearcher
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

console = Console()


def test_basic_logging():
    """Test 1: Basic activity logging functionality."""
    console.print("\n[bold cyan]Test 1: Basic Activity Logging[/bold cyan]")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Initialize logger
        logger = ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        # Start interaction chain
        chain_id = logger.start_interaction_chain()
        console.print(f"✓ Started interaction chain: {chain_id[:20]}...")

        # Log user input
        activity_id = logger.log_activity(
            activity_type="user_input",
            agent_name=None,
            content={"input": "What is machine learning?"},
            tags=["root", "question"]
        )
        console.print(f"✓ Logged user input: {activity_id[:20]}...")

        # Log agent processing
        activity_id = logger.log_activity(
            activity_type="agent_input",
            agent_name="researcher",
            agent_model="gpt-4",
            content={
                "refined_query": "Provide comprehensive overview of machine learning"
            },
            tags=["research"]
        )
        console.print(f"✓ Logged agent input: {activity_id[:20]}...")

        # Increase depth (nested reasoning)
        logger.increase_depth()

        # Log reasoning step
        activity_id = logger.log_activity(
            activity_type="reasoning_step",
            agent_name="researcher",
            content={
                "step": "Gather background information",
                "reasoning": "First, need to understand the fundamental concepts"
            },
            metadata={"step_number": 1},
            tags=["reasoning"]
        )
        console.print(f"✓ Logged reasoning step: {activity_id[:20]}...")

        # Log tool call
        activity_id = logger.log_activity(
            activity_type="tool_call",
            agent_name="researcher",
            content={
                "tool_name": "search_documents",
                "parameters": {"query": "machine learning basics"}
            },
            tags=["tool_use"]
        )
        console.print(f"✓ Logged tool call: {activity_id[:20]}...")

        # Log tool result
        activity_id = logger.log_activity(
            activity_type="tool_result",
            agent_name="researcher",
            content={
                "tool_name": "search_documents",
                "result": "Found 10 relevant documents"
            },
            metadata={"documents_found": 10}
        )
        console.print(f"✓ Logged tool result: {activity_id[:20]}...")

        # Decrease depth
        logger.decrease_depth()

        # Log final agent output
        activity_id = logger.log_activity(
            activity_type="agent_output",
            agent_name="researcher",
            content={
                "output": "Machine learning is a subset of artificial intelligence..."
            },
            metadata={"tokens_used": 150},
            tags=["final_response"]
        )
        console.print(f"✓ Logged agent output: {activity_id[:20]}...")

        # End interaction chain
        logger.end_interaction_chain(status="completed")
        console.print("✓ Ended interaction chain")

        # Verify files created
        assert log_dir.exists(), "Log directory not created"
        assert (log_dir / "activities.jsonl").exists(), "JSONL file not created"
        assert db_path.exists(), "Database not created"

        console.print("\n[green]✓ Basic logging test passed![/green]")


def test_chain_retrieval():
    """Test 2: Retrieve interaction chain."""
    console.print("\n[bold cyan]Test 2: Chain Retrieval[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Create logger and log activities
        logger = ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        chain_id = logger.start_interaction_chain()

        # Log multiple activities
        for i in range(5):
            logger.log_activity(
                activity_type="agent_output",
                agent_name=f"agent_{i}",
                content={"output": f"Response {i}"},
                tags=[f"step_{i}"]
            )

        logger.end_interaction_chain()

        # Retrieve chain
        activities = logger.get_chain_activities(chain_id, include_content=True)

        console.print(f"✓ Retrieved {len(activities)} activities from chain")

        # Display activities
        table = Table(title="Chain Activities")
        table.add_column("Activity Type", style="cyan")
        table.add_column("Agent", style="magenta")
        table.add_column("Depth", style="yellow")

        for activity in activities:
            table.add_row(
                activity["activity_type"],
                activity.get("agent_name") or "System",
                str(activity["depth_level"])
            )

        console.print(table)

        assert len(activities) == 5, f"Expected 5 activities, got {len(activities)}"
        console.print("\n[green]✓ Chain retrieval test passed![/green]")


def test_agent_activity_retrieval():
    """Test 3: Retrieve agent-specific activities."""
    console.print("\n[bold cyan]Test 3: Agent Activity Retrieval[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        logger = ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        # Log activities for multiple agents
        agents = ["researcher", "coder", "analyst", "researcher", "coder"]
        for agent in agents:
            logger.start_interaction_chain()
            logger.log_activity(
                activity_type="agent_output",
                agent_name=agent,
                content={"output": f"Response from {agent}"}
            )
            logger.end_interaction_chain()

        # Retrieve researcher activities
        researcher_activities = logger.get_agent_activities(
            "researcher",
            limit=10,
            include_content=True
        )

        console.print(f"✓ Retrieved {len(researcher_activities)} activities for 'researcher'")

        assert len(researcher_activities) == 2, \
            f"Expected 2 researcher activities, got {len(researcher_activities)}"

        console.print("[green]✓ Agent activity retrieval test passed![/green]")


def test_search_functionality():
    """Test 4: Activity search with ActivitySearcher."""
    console.print("\n[bold cyan]Test 4: Search Functionality[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Create logger and log activities
        logger = ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        # Log activities with specific keywords
        logger.start_interaction_chain()

        logger.log_activity(
            activity_type="user_input",
            agent_name=None,
            content={"input": "Explain database indexing"}
        )

        logger.log_activity(
            activity_type="agent_output",
            agent_name="analyst",
            content={"output": "Database indexing improves query performance..."}
        )

        logger.log_activity(
            activity_type="error",
            agent_name="analyst",
            content={"error": "Connection timeout to database"}
        )

        logger.end_interaction_chain()

        # Create searcher
        searcher = ActivitySearcher(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        # Test grep search
        results = searcher.grep_logs(
            pattern="database",
            max_results=10
        )

        console.print(f"✓ Grep search for 'database' returned {len(results)} results")

        # Test filtered search
        error_results = searcher.grep_logs(
            pattern=".*",
            activity_type="error",
            max_results=10
        )

        console.print(f"✓ Filtered search for errors returned {len(error_results)} results")

        assert len(results) >= 2, f"Expected at least 2 results, got {len(results)}"
        assert len(error_results) == 1, f"Expected 1 error, got {len(error_results)}"

        console.print("[green]✓ Search functionality test passed![/green]")


def test_sql_queries():
    """Test 5: SQL query interface."""
    console.print("\n[bold cyan]Test 5: SQL Queries[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Create logger with multiple agents
        logger = ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        # Log activities for different agents
        for agent in ["researcher", "coder", "analyst"]:
            for i in range(3):
                logger.start_interaction_chain()
                logger.log_activity(
                    activity_type="agent_output",
                    agent_name=agent,
                    agent_model="gpt-4",
                    content={"output": f"Response {i} from {agent}"}
                )
                logger.end_interaction_chain()

        # Create searcher
        searcher = ActivitySearcher(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        # Test SQL: Count by agent
        results = searcher.sql_query("""
            SELECT agent_name, COUNT(*) as count
            FROM agent_activities
            WHERE session_name = ?
            GROUP BY agent_name
        """, ("test-session",))

        console.print("✓ SQL query: Activities by agent")

        table = Table()
        table.add_column("Agent", style="cyan")
        table.add_column("Count", style="magenta")

        for row in results:
            table.add_row(row["agent_name"], str(row["count"]))

        console.print(table)

        assert len(results) == 3, f"Expected 3 agents, got {len(results)}"

        # Test SQL: Get chains
        chains = searcher.sql_query("""
            SELECT chain_id, status, total_activities
            FROM interaction_chains
            WHERE session_name = ?
        """, ("test-session",))

        console.print(f"\n✓ Found {len(chains)} interaction chains")

        assert len(chains) == 9, f"Expected 9 chains, got {len(chains)}"

        console.print("[green]✓ SQL query test passed![/green]")


def test_statistics():
    """Test 6: Activity statistics."""
    console.print("\n[bold cyan]Test 6: Statistics[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Create logger with various activities
        logger = ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        # Create complex interaction
        logger.start_interaction_chain()
        logger.log_activity("user_input", None, {"input": "Test"})
        logger.log_activity("agent_output", "researcher", {"output": "Response"})
        logger.increase_depth()
        logger.log_activity("reasoning_step", "researcher", {"step": "Think"})
        logger.log_activity("tool_call", "researcher", {"tool": "search"})
        logger.decrease_depth()
        logger.log_activity("error", "researcher", {"error": "Test error"})
        logger.end_interaction_chain()

        # Get statistics
        searcher = ActivitySearcher(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        stats = searcher.get_statistics()

        # Display statistics
        console.print(Panel.fit(
            f"""
[cyan]Total Activities:[/cyan] {stats['total_activities']}
[cyan]Total Chains:[/cyan] {stats['total_chains']}
[cyan]Active Chains:[/cyan] {stats['active_chains']}
[cyan]Average Chain Depth:[/cyan] {stats['avg_chain_depth']}
[cyan]Total Errors:[/cyan] {stats['total_errors']}

[yellow]Activities by Type:[/yellow]
{chr(10).join(f"  {k}: {v}" for k, v in stats['activities_by_type'].items())}

[yellow]Activities by Agent:[/yellow]
{chr(10).join(f"  {k}: {v}" for k, v in stats['activities_by_agent'].items())}
            """.strip(),
            title="[bold]Activity Statistics[/bold]",
            border_style="green"
        ))

        assert stats['total_activities'] == 5, \
            f"Expected 5 activities, got {stats['total_activities']}"
        assert stats['total_errors'] == 1, \
            f"Expected 1 error, got {stats['total_errors']}"

        console.print("\n[green]✓ Statistics test passed![/green]")


def test_context_manager():
    """Test 7: Context manager auto-close."""
    console.print("\n[bold cyan]Test 7: Context Manager[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Test context manager
        with ActivityLogger(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        ) as logger:
            chain_id = logger.start_interaction_chain()
            logger.log_activity("user_input", None, {"input": "Test"})
            # Don't explicitly end chain - context manager should do it

        # Verify chain was closed
        searcher = ActivitySearcher(
            session_name="test-session",
            log_dir=log_dir,
            db_path=db_path
        )

        chains = searcher.sql_query("""
            SELECT status FROM interaction_chains WHERE chain_id = ?
        """, (chain_id,))

        assert len(chains) == 1, "Chain not found"
        assert chains[0]["status"] == "completed", \
            f"Expected status 'completed', got '{chains[0]['status']}'"

        console.print("✓ Context manager auto-closed chain")
        console.print("[green]✓ Context manager test passed![/green]")


def main():
    """Run all tests."""
    console.print("[bold magenta]ActivityLogger & ActivitySearcher Test Suite[/bold magenta]")
    console.print("=" * 60)

    try:
        test_basic_logging()
        test_chain_retrieval()
        test_agent_activity_retrieval()
        test_search_functionality()
        test_sql_queries()
        test_statistics()
        test_context_manager()

        console.print("\n" + "=" * 60)
        console.print("[bold green]✓ All Tests Passed![/bold green]")
        console.print("\n[dim]Activity logging system is working correctly![/dim]")

    except AssertionError as e:
        console.print(f"\n[bold red]✗ Test failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]✗ Test suite failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
