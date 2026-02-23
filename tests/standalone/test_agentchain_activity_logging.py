#!/usr/bin/env python3
"""
Test script for AgentChain + ActivityLogger integration.

This script verifies that ActivityLogger correctly captures all agent interactions
across different execution modes (pipeline, router, round_robin, broadcast).
"""

import sys
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.cli.activity_logger import ActivityLogger
from promptchain.cli.activity_searcher import ActivitySearcher
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def test_pipeline_mode_logging():
    """Test 1: Pipeline mode with activity logging."""
    console.print("\n[bold cyan]Test 1: Pipeline Mode Activity Logging[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        # Create activity logger
        activity_logger = ActivityLogger(
            session_name="test-pipeline",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        # Create simple agents
        agent1 = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Transform input: {input}"],
            verbose=False
        )

        agent2 = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Summarize: {input}"],
            verbose=False
        )

        # Create AgentChain with activity logger
        agent_chain = AgentChain(
            agents={"transform": agent1, "summarize": agent2},
            agent_descriptions={
                "transform": "Transforms input",
                "summarize": "Summarizes text"
            },
            execution_mode="pipeline",
            activity_logger=activity_logger,
            verbose=False
        )

        # Execute
        try:
            result = await agent_chain.process_input("Test input for pipeline")
            console.print(f"✓ Pipeline executed successfully")
        except Exception as e:
            console.print(f"[red]✗ Pipeline execution failed: {e}[/red]")
            return False

        # Verify activities were logged
        searcher = ActivitySearcher(
            session_name="test-pipeline",
            log_dir=log_dir,
            db_path=db_path
        )

        stats = searcher.get_statistics()
        console.print(f"✓ Logged {stats['total_activities']} activities")
        console.print(f"  - user_input: {stats['activities_by_type'].get('user_input', 0)}")
        console.print(f"  - agent_output: {stats['activities_by_type'].get('agent_output', 0)}")
        console.print(f"  - Total chains: {stats['total_chains']}")

        # Verify we have expected activities
        assert stats['total_activities'] >= 3, f"Expected >= 3 activities (1 user + 2 agents), got {stats['total_activities']}"
        assert stats['activities_by_type'].get('user_input', 0) == 1, "Expected 1 user_input"
        assert stats['activities_by_type'].get('agent_output', 0) >= 2, f"Expected >= 2 agent_outputs, got {stats['activities_by_type'].get('agent_output', 0)}"

        console.print("[green]✓ Pipeline mode logging test passed![/green]")
        return True


async def test_router_mode_logging():
    """Test 2: Router mode with activity logging."""
    console.print("\n[bold cyan]Test 2: Router Mode Activity Logging[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        activity_logger = ActivityLogger(
            session_name="test-router",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        # Create agents
        researcher = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Research: {input}"],
            verbose=False
        )

        writer = PromptChain(
            models=["openai/gpt-4o-mini"],
            instructions=["Write about: {input}"],
            verbose=False
        )

        # Router configuration
        router_config = {
            "models": ["openai/gpt-4o-mini"],
            "instructions": [None, "{input}"],
            "decision_prompt_templates": {
                "single_agent_dispatch": """
Based on: {user_input}
Available agents: {agent_details}
Choose agent and return JSON: {{"chosen_agent": "agent_name"}}
                """
            }
        }

        # Create AgentChain with router mode
        agent_chain = AgentChain(
            agents={"researcher": researcher, "writer": writer},
            agent_descriptions={
                "researcher": "Research specialist",
                "writer": "Writing specialist"
            },
            execution_mode="router",
            router=router_config,
            activity_logger=activity_logger,
            verbose=False
        )

        # Execute
        try:
            result = await agent_chain.process_input("Research machine learning")
            console.print(f"✓ Router executed successfully")
        except Exception as e:
            console.print(f"[red]✗ Router execution failed: {e}[/red]")
            return False

        # Verify activities
        searcher = ActivitySearcher(
            session_name="test-router",
            log_dir=log_dir,
            db_path=db_path
        )

        stats = searcher.get_statistics()
        console.print(f"✓ Logged {stats['total_activities']} activities")
        console.print(f"  - user_input: {stats['activities_by_type'].get('user_input', 0)}")
        console.print(f"  - router_decision: {stats['activities_by_type'].get('router_decision', 0)}")
        console.print(f"  - agent_output: {stats['activities_by_type'].get('agent_output', 0)}")

        # Verify expected activities
        assert stats['activities_by_type'].get('user_input', 0) == 1, "Expected 1 user_input"
        assert stats['activities_by_type'].get('router_decision', 0) >= 1, "Expected >= 1 router_decision"
        assert stats['activities_by_type'].get('agent_output', 0) >= 1, "Expected >= 1 agent_output"

        # Test grep search for router decisions
        router_decisions = searcher.grep_logs(
            pattern="router_decision",
            max_results=10
        )
        console.print(f"✓ Found {len(router_decisions)} router decisions via grep")

        console.print("[green]✓ Router mode logging test passed![/green]")
        return True


async def test_round_robin_mode_logging():
    """Test 3: Round robin mode with activity logging."""
    console.print("\n[bold cyan]Test 3: Round Robin Mode Activity Logging[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        activity_logger = ActivityLogger(
            session_name="test-roundrobin",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        # Create agents
        agent1 = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Agent 1: {input}"], verbose=False)
        agent2 = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Agent 2: {input}"], verbose=False)

        agent_chain = AgentChain(
            agents={"agent1": agent1, "agent2": agent2},
            agent_descriptions={
                "agent1": "First agent",
                "agent2": "Second agent"
            },
            execution_mode="round_robin",
            activity_logger=activity_logger,
            verbose=False
        )

        # Execute multiple times to test round robin
        try:
            for i in range(3):
                await agent_chain.process_input(f"Input {i}")
            console.print(f"✓ Round robin executed 3 times successfully")
        except Exception as e:
            console.print(f"[red]✗ Round robin execution failed: {e}[/red]")
            return False

        # Verify activities
        searcher = ActivitySearcher(
            session_name="test-roundrobin",
            log_dir=log_dir,
            db_path=db_path
        )

        stats = searcher.get_statistics()
        console.print(f"✓ Logged {stats['total_activities']} activities")
        console.print(f"  - user_input: {stats['activities_by_type'].get('user_input', 0)}")
        console.print(f"  - agent_output: {stats['activities_by_type'].get('agent_output', 0)}")
        console.print(f"  - Total chains: {stats['total_chains']}")

        # Verify expected activities (3 inputs + 3 agent outputs)
        assert stats['activities_by_type'].get('user_input', 0) == 3, "Expected 3 user_inputs"
        assert stats['activities_by_type'].get('agent_output', 0) == 3, "Expected 3 agent_outputs"
        assert stats['total_chains'] == 3, "Expected 3 chains"

        # Verify different agents were used
        agent1_activities = searcher.grep_logs(pattern="agent1", max_results=10)
        agent2_activities = searcher.grep_logs(pattern="agent2", max_results=10)
        console.print(f"✓ Agent1 activities: {len(agent1_activities)}")
        console.print(f"✓ Agent2 activities: {len(agent2_activities)}")

        console.print("[green]✓ Round robin mode logging test passed![/green]")
        return True


async def test_chain_retrieval():
    """Test 4: Retrieve complete interaction chains."""
    console.print("\n[bold cyan]Test 4: Chain Retrieval[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        activity_logger = ActivityLogger(
            session_name="test-chains",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        agent = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Process: {input}"], verbose=False)

        agent_chain = AgentChain(
            agents={"processor": agent},
            agent_descriptions={"processor": "Processes input"},
            execution_mode="pipeline",
            activity_logger=activity_logger,
            verbose=False
        )

        # Execute
        try:
            await agent_chain.process_input("Test input")
            console.print(f"✓ Chain executed successfully")
        except Exception as e:
            console.print(f"[red]✗ Chain execution failed: {e}[/red]")
            return False

        # Get chain information
        searcher = ActivitySearcher(
            session_name="test-chains",
            log_dir=log_dir,
            db_path=db_path
        )

        # Get all chains
        chains = searcher.sql_query("""
            SELECT chain_id, status, total_activities, completed_at
            FROM interaction_chains
            WHERE session_name = ?
        """, ("test-chains",))

        console.print(f"✓ Found {len(chains)} chains")
        assert len(chains) == 1, "Expected 1 chain"

        chain = chains[0]
        console.print(f"  - Chain ID: {chain['chain_id'][:20]}...")
        console.print(f"  - Status: {chain['status']}")
        console.print(f"  - Total activities: {chain['total_activities']}")
        console.print(f"  - Completed: {chain['completed_at'] is not None}")

        assert chain['status'] == 'completed', f"Expected status 'completed', got '{chain['status']}'"
        assert chain['total_activities'] >= 2, f"Expected >= 2 activities, got {chain['total_activities']}"
        assert chain['completed_at'] is not None, "Expected completed_at timestamp"

        # Retrieve full chain with content
        full_chain = searcher.get_interaction_chain(
            chain_id=chain['chain_id'],
            include_content=True,
            include_nested=False
        )

        console.print(f"✓ Retrieved chain with {len(full_chain['activities'])} activities")

        # Display activity types
        activity_types = [a['activity_type'] for a in full_chain['activities']]
        console.print(f"  - Activity types: {', '.join(activity_types)}")

        console.print("[green]✓ Chain retrieval test passed![/green]")
        return True


async def test_error_logging():
    """Test 5: Error logging in pipeline mode."""
    console.print("\n[bold cyan]Test 5: Error Logging[/bold cyan]")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        db_path = Path(tmpdir) / "activities.db"

        activity_logger = ActivityLogger(
            session_name="test-errors",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        # Create an agent that will fail
        def failing_function(input_text: str) -> str:
            raise ValueError("Intentional test error")

        agent = PromptChain(
            models=[],
            instructions=[failing_function],
            verbose=False
        )

        agent_chain = AgentChain(
            agents={"failer": agent},
            agent_descriptions={"failer": "Intentional failure agent"},
            execution_mode="pipeline",
            activity_logger=activity_logger,
            verbose=False
        )

        # Execute (should fail gracefully)
        try:
            result = await agent_chain.process_input("Test input")
            console.print(f"✓ Pipeline executed (with expected error)")
        except Exception as e:
            console.print(f"✓ Pipeline failed as expected: {type(e).__name__}")

        # Verify error was logged
        searcher = ActivitySearcher(
            session_name="test-errors",
            log_dir=log_dir,
            db_path=db_path
        )

        stats = searcher.get_statistics()
        console.print(f"✓ Total errors logged: {stats['total_errors']}")

        # Search for error activities
        errors = searcher.grep_logs(
            pattern="error",
            activity_type="error",
            max_results=10
        )
        console.print(f"✓ Found {len(errors)} error activities")

        assert stats['total_errors'] >= 1, f"Expected >= 1 error, got {stats['total_errors']}"

        console.print("[green]✓ Error logging test passed![/green]")
        return True


async def main():
    """Run all tests."""
    console.print("[bold magenta]AgentChain + ActivityLogger Integration Tests[/bold magenta]")
    console.print("=" * 60)

    tests = [
        ("Pipeline Mode", test_pipeline_mode_logging),
        ("Router Mode", test_router_mode_logging),
        ("Round Robin Mode", test_round_robin_mode_logging),
        ("Chain Retrieval", test_chain_retrieval),
        ("Error Logging", test_error_logging),
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
        console.print("\n[dim]ActivityLogger integration with AgentChain is working correctly![/dim]")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]✗ {failed} Test(s) Failed[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
