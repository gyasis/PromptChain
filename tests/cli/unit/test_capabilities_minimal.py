#!/usr/bin/env python3
"""
Minimal test for /capabilities command handler logic.

Tests the handler implementation without full dependency imports.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


# Mock CommandResult
@dataclass
class CommandResult:
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Mock ToolMetadata for testing
@dataclass
class MockToolMetadata:
    name: str
    description: str
    capabilities: List[str]


# Mock ToolRegistry
class MockToolRegistry:
    def __init__(self):
        self.tools = []

    def add_tool(self, name: str, description: str, capabilities: List[str]):
        self.tools.append(MockToolMetadata(name, description, capabilities))

    def list_capabilities(self) -> List[str]:
        caps = set()
        for tool in self.tools:
            caps.update(tool.capabilities)
        return sorted(list(caps))

    def get_by_capability(self, capability: str) -> List[MockToolMetadata]:
        return [t for t in self.tools if capability in t.capabilities]

    def discover_capabilities(self, agent_name: Optional[str] = None) -> List[MockToolMetadata]:
        # For testing, just return all tools
        # In real implementation, this would filter by allowed_agents
        return self.tools


# Create mock registry
registry = MockToolRegistry()


def handle_capabilities(agent_name: Optional[str] = None) -> CommandResult:
    """
    Handle /capabilities command - list tools by capability.

    This is the actual implementation from command_handler.py.
    """
    try:
        # Get all capabilities
        all_capabilities = registry.list_capabilities()

        if not all_capabilities:
            return CommandResult(
                success=True,
                message="No capabilities registered yet. Register tools with capabilities to see them here.",
                data={"capabilities": [], "count": 0}
            )

        # If agent_name provided, filter by agent access
        if agent_name:
            # Get tools available to this agent
            agent_tools = registry.discover_capabilities(agent_name=agent_name)

            if not agent_tools:
                return CommandResult(
                    success=True,
                    message=f"No tools available to agent '{agent_name}'.\n"
                            f"Use /agent update {agent_name} --add-tools <tool_name> to add tools.",
                    data={"agent_name": agent_name, "tools": [], "count": 0}
                )

            # Format agent-specific tool list
            lines = [f"Tools available to agent '{agent_name}':", ""]

            # Group by capability
            capability_groups = {}
            for tool in agent_tools:
                for cap in tool.capabilities:
                    if cap not in capability_groups:
                        capability_groups[cap] = []
                    capability_groups[cap].append(tool)

            # Format each capability group
            for capability in sorted(capability_groups.keys()):
                tools = capability_groups[capability]
                lines.append(f"Capability: {capability} ({len(tools)} tools)")
                for tool in tools:
                    lines.append(f"  - {tool.name}: {tool.description[:60]}...")
                lines.append("")

            # Add ungrouped tools (tools with no capabilities)
            ungrouped = [t for t in agent_tools if not t.capabilities]
            if ungrouped:
                lines.append(f"Other Tools ({len(ungrouped)} tools):")
                for tool in ungrouped:
                    lines.append(f"  - {tool.name}: {tool.description[:60]}...")
                lines.append("")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "agent_name": agent_name,
                    "tools": [t.name for t in agent_tools],
                    "count": len(agent_tools),
                    "capabilities": list(capability_groups.keys())
                }
            )

        # No agent specified - show all capabilities
        lines = [f"Available Capabilities ({len(all_capabilities)} total):", ""]

        # Get tools for each capability
        for capability in sorted(all_capabilities):
            tools = registry.get_by_capability(capability)

            # Format capability line with tool count and names
            tool_names = [t.name for t in tools]
            if len(tool_names) <= 3:
                tool_list = ", ".join(tool_names)
            else:
                tool_list = ", ".join(tool_names[:3]) + f", ... ({len(tool_names) - 3} more)"

            lines.append(f"  {capability} ({len(tools)} tools): {tool_list}")

        lines.append("")
        lines.append("Usage:")
        lines.append("  /capabilities <agent_name>  - Show capabilities for specific agent")
        lines.append("  /tools list                 - Show all tools")

        message = "\n".join(lines)

        return CommandResult(
            success=True,
            message=message,
            data={
                "capabilities": all_capabilities,
                "count": len(all_capabilities)
            }
        )

    except Exception as e:
        return CommandResult(
            success=False,
            message=f"Failed to list capabilities: {str(e)}",
            error=str(e)
        )


def test_capabilities_all():
    """Test /capabilities without agent filter."""
    print("\n" + "="*60)
    print("TEST 1: List all capabilities")
    print("="*60)

    result = handle_capabilities()

    print(f"\nSuccess: {result.success}")
    print(f"\nMessage:\n{result.message}")

    # Validate result
    assert result.success, "Command should succeed"
    assert "file_read" in result.message, "Should show file_read capability"
    assert "file_write" in result.message, "Should show file_write capability"
    assert "code_search" in result.message, "Should show code_search capability"
    assert "7 total" in result.message, "Should show 7 capabilities (including io, search, shell_exec, system)"

    print("\n✓ Test passed")


def test_capabilities_agent():
    """Test /capabilities with agent filter."""
    print("\n" + "="*60)
    print("TEST 2: List capabilities for specific agent")
    print("="*60)

    result = handle_capabilities(agent_name="test_worker")

    print(f"\nSuccess: {result.success}")
    print(f"\nMessage:\n{result.message}")

    # Validate result
    assert result.success, "Command should succeed"
    assert "test_worker" in result.message, "Should show agent name"
    assert "file_read" in result.message, "Should show capabilities"

    print("\n✓ Test passed")


def test_no_capabilities():
    """Test /capabilities when no capabilities exist."""
    print("\n" + "="*60)
    print("TEST 3: No capabilities registered")
    print("="*60)

    # Clear registry
    registry.tools = []

    result = handle_capabilities()

    print(f"\nSuccess: {result.success}")
    print(f"\nMessage:\n{result.message}")

    assert result.success, "Command should succeed even with no capabilities"
    assert "No capabilities" in result.message, "Should indicate no capabilities"

    print("\n✓ Test passed")


def main():
    """Run all tests."""
    print("Testing /capabilities command handler logic")
    print("="*60)

    try:
        # Setup test tools
        registry.add_tool("file_read", "Read file contents from disk", ["file_read", "io"])
        registry.add_tool("file_write", "Write content to a file", ["file_write", "io"])
        registry.add_tool("code_search", "Search code using ripgrep", ["code_search", "search"])
        registry.add_tool("shell_exec", "Execute shell command", ["shell_exec", "system"])

        print(f"✓ Created {len(registry.tools)} test tools")
        print(f"✓ Capabilities: {', '.join(registry.list_capabilities())}")

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
