"""CLI Tools - Safety validation and execution utilities for PromptChain CLI.

Provides secure execution wrappers for file operations, shell commands, and other
potentially dangerous operations with comprehensive safety validation.

Also provides an extensible tool registration and discovery system for CLI commands
and agent interactions.
"""

from .executor import ToolExecutionError, ToolExecutor
from .filesystem_tools import (filesystem_registry, find_paths, get_cwd,
                               is_within_working_dir, path_info, resolve_path)
from .models import ExecutionMetrics, ToolExecutionContext, ToolResult
from .registry import (ParameterSchema, ToolCategory, ToolMetadata,
                       ToolNotFoundError, ToolRegistrationError, ToolRegistry,
                       ToolValidationError)
from .safety import SafetyValidator, SecurityError

# Create global registry instance
registry = ToolRegistry()

# Register filesystem tools into the global registry
for tool_name in filesystem_registry.list_tools():
    tool_meta = filesystem_registry.get(tool_name)
    if tool_meta and tool_name not in registry.list_tools():
        # Re-register the tool in the global registry
        registry._tools[tool_name] = tool_meta
        registry._category_index[tool_meta.category].add(tool_name)  # type: ignore[index]
        for tag in tool_meta.tags:
            if tag not in registry._tag_index:
                registry._tag_index[tag] = set()
            registry._tag_index[tag].add(tool_name)

__all__ = [
    "SafetyValidator",
    "SecurityError",
    "ToolRegistry",
    "ToolMetadata",
    "ToolCategory",
    "ParameterSchema",
    "ToolRegistrationError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ToolExecutor",
    "ToolExecutionError",
    "ToolResult",
    "ExecutionMetrics",
    "ToolExecutionContext",
    "registry",
    # Filesystem tools
    "filesystem_registry",
    "resolve_path",
    "find_paths",
    "get_cwd",
    "path_info",
    "is_within_working_dir",
]
