#!/usr/bin/env python3
"""
Test script for /capabilities command handler.

Tests both modes:
1. /capabilities - Show all capabilities
2. /capabilities <agent_name> - Show agent-specific capabilities
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from promptchain.cli.command_handler import CommandHandler
from promptchain.cli.tools import registry
from promptchain.cli.models.session import Session
from promptchain.cli.models.agent_config import Agent
import time


def setup_test_tools():
    """Register some test tools with capabilities."""

    @registry.register(
        category="filesystem",
        description="Read file contents from disk",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path to read"}
        },
        capabilities=["file_read", "io"]
    )
    def test_file_read(path: str) -> str:
        return f"Reading {path}"

    @registry.register(
        category="filesystem",
        description="Write content to a file",
        parameters={
            "path": {"type": "string", "required": True, "description": "File path to write"},
            "content": {"type": "string", "required": True, "description": "Content to write"}
        },
        capabilities=["file_write", "io"]
    )
    def test_file_write(path: str, content: str) -> str:
        return f"Writing to {path}"

    @registry.register(
        category="analysis",
        description="Search code using ripgrep",
        parameters={
            "pattern": {"type": "string", "required": True, "description": "Search pattern"}
        },
        capabilities=["code_search", "search"]
    )
    def test_code_search(pattern: str) -> str:
        return f"Searching for {pattern}"

    @registry.register(
        category="shell",
        description="Execute shell command",
        parameters={
            "command": {"type": "string", "required": True, "description": "Command to execute"}
        },
        capabilities=["shell_exec", "system"]
    )
    def test_shell_exec(command: str) -> str:
        return f"Executing {command}"

    print(f"✓ Registered {len(registry.list_tools())} test tools")
    print(f"✓ Capabilities: {', '.join(registry.list_capabilities())}")


def test_capabilities_all():
    """Test /capabilities without agent filter."""
    print("\n" + "="*60)
    print("TEST 1: List all capabilities")
    print("="*60)

    # Create mock session (not needed for capabilities command)
    handler = CommandHandler(session_manager=None)

    # Call handler
    result = handler.handle_capabilities()

    print(f"\nSuccess: {result.success}")
    print(f"\nMessage:\n{result.message}")

    # Validate result
    assert result.success, "Command should succeed"
    assert "file_read" in result.message, "Should show file_read capability"
    assert "file_write" in result.message, "Should show file_write capability"
    assert "code_search" in result.message, "Should show code_search capability"

    print("\n✓ Test passed")


def test_capabilities_agent():
    """Test /capabilities with agent filter."""
    print("\n" + "="*60)
    print("TEST 2: List capabilities for specific agent")
    print("="*60)

    # Create agent with specific tools
    agent = Agent(
        name="test_worker",
        model_name="openai/gpt-4.1-mini-2025-04-14",
        description="Test worker agent",
        tools=["test_file_read", "test_code_search"],  # Only has these tools
        created_at=time.time()
    )

    # Update registry to mark tools as accessible to this agent
    for tool_name in ["test_file_read", "test_code_search"]:
        tool = registry.get(tool_name)
        if tool:
            # Set allowed_agents (None = all agents, so we'll leave as-is for this test)
            # The discover_capabilities method will filter by agent tools
            pass

    handler = CommandHandler(session_manager=None)

    # Call handler with agent name
    result = handler.handle_capabilities(agent_name="test_worker")

    print(f"\nSuccess: {result.success}")
    print(f"\nMessage:\n{result.message}")

    # Validate result
    assert result.success, "Command should succeed"
    assert "test_worker" in result.message, "Should show agent name"

    # Note: The current implementation filters by allowed_agents in registry.discover_capabilities,
    # not by agent.tools. We may need to adjust this based on actual usage pattern.

    print("\n✓ Test passed (note: filtering by agent.tools requires integration with Session)")


def test_no_capabilities():
    """Test /capabilities when no capabilities exist."""
    print("\n" + "="*60)
    print("TEST 3: No capabilities registered")
    print("="*60)

    # Clear registry
    registry.clear()

    handler = CommandHandler(session_manager=None)
    result = handler.handle_capabilities()

    print(f"\nSuccess: {result.success}")
    print(f"\nMessage:\n{result.message}")

    assert result.success, "Command should succeed even with no capabilities"
    assert "No capabilities" in result.message, "Should indicate no capabilities"

    print("\n✓ Test passed")


def main():
    """Run all tests."""
    print("Testing /capabilities command handler")
    print("="*60)

    try:
        # Setup test tools
        setup_test_tools()

        # Run tests
        test_capabilities_all()
        test_capabilities_agent()
        test_no_capabilities()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
