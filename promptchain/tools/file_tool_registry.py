"""
File Tool Registry

Automatic registration helper for file operation tools with PromptChain agents.
Makes file operations a standard part of any agent that uses TerminalTool.

This ensures consistent tool availability across all agents without manual
registration in every script.
"""

from typing import Any, Dict, List, Optional

from promptchain.tools.efficient_file_edit import (insert_after_pattern,
                                                   insert_at_line,
                                                   insert_before_pattern,
                                                   replace_lines)
from promptchain.tools.file_operations import (create_directory, file_append,
                                               file_delete, file_edit,
                                               file_read, file_write,
                                               list_directory, read_file_range)

# Complete tool function list
ALL_FILE_TOOLS = [
    file_read,
    file_write,
    file_edit,
    file_append,
    file_delete,
    list_directory,
    create_directory,
    read_file_range,
    insert_at_line,
    replace_lines,
    insert_after_pattern,
    insert_before_pattern,
]

# Tool schemas for LLM (JSON format for OpenAI-style tool calling)
FILE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file, creating it if it doesn't exist",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Edit a file by replacing old_text with new_text using sed",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Text to find and replace",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace with",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_append",
            "description": "Append content to the end of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to append to",
                    },
                    "content": {"type": "string", "description": "Content to append"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_delete",
            "description": "Delete a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to delete",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to directory to list (defaults to current directory)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Create a directory and all parent directories",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to directory to create",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "Read a specific range of lines from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (inclusive)",
                    },
                },
                "required": ["path", "start_line", "end_line"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_at_line",
            "description": "Insert content at a specific line number (efficient for large files - no full read required). Content is inserted BEFORE the specified line. Preserves proper indentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Line number to insert at (1-indexed, content goes BEFORE this line)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert (can be multi-line, preserves tabs/spaces)",
                    },
                },
                "required": ["path", "line_number", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": "Replace a range of lines with new content (efficient for large files - no full read required). Preserves proper indentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to replace (1-indexed, inclusive)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to replace (1-indexed, inclusive)",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New content to replace the lines with (can be multi-line)",
                    },
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_after_pattern",
            "description": "Insert content after a regex pattern match (efficient for large files). Preserves proper indentation. Use for adding code after function definitions, imports, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (e.g., '^def main', '^import os')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert after the pattern (can be multi-line)",
                    },
                    "first_match": {
                        "type": "boolean",
                        "description": "If True, only insert after first match; if False, after all matches (default: True)",
                    },
                },
                "required": ["path", "pattern", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insert_before_pattern",
            "description": "Insert content before a regex pattern match (efficient for large files). Preserves proper indentation. Use for adding imports, comments, or code before specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (e.g., '^if __name__', '^class MyClass')",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert before the pattern (can be multi-line)",
                    },
                    "first_match": {
                        "type": "boolean",
                        "description": "If True, only insert before first match; if False, before all matches (default: True)",
                    },
                },
                "required": ["path", "pattern", "content"],
            },
        },
    },
]


def register_file_tools(agent, include_efficient: bool = True) -> None:
    """
    Register all file operation tools with a PromptChain agent.

    This is the standard way to add file operations to any agent. Call this
    once after creating your agent to get all file operation capabilities.

    Args:
        agent: PromptChain instance to register tools with
        include_efficient: Whether to include efficient editing tools (default: True)
                          Set to False if you only want basic file operations

    Example:
        >>> from promptchain import PromptChain
        >>> from promptchain.tools.file_tool_registry import register_file_tools
        >>>
        >>> agent = PromptChain(models=["openai/gpt-4"], instructions=["..."])
        >>> register_file_tools(agent)  # Now has all 12 file operation tools
    """
    # Determine which tools to register
    if include_efficient:
        tools_to_register = ALL_FILE_TOOLS
        schemas_to_register = FILE_TOOL_SCHEMAS
    else:
        # Only basic tools (first 8)
        tools_to_register = ALL_FILE_TOOLS[:8]
        schemas_to_register = FILE_TOOL_SCHEMAS[:8]

    # Register tool functions
    for tool_func in tools_to_register:
        agent.register_tool_function(tool_func)

    # Add tool schemas for LLM
    agent.add_tools(schemas_to_register)


def get_file_tool_prompt_guidance(include_efficient: bool = True) -> str:
    """
    Get prompt text to add to agent objectives for file operation guidance.

    This provides best practices and decision trees for using file operations
    correctly. Include this in your agent's objective or system prompt.

    Args:
        include_efficient: Whether to include efficient editing tool guidance

    Returns:
        str: Formatted prompt text for agent objective

    Example:
        >>> guidance = get_file_tool_prompt_guidance()
        >>> agent_objective = f'''
        ... You are a coding agent with file operation tools.
        ...
        ... {guidance}
        ... '''
    """
    basic_guidance = """
CRITICAL: File Editing Best Practices

FOR SMALL FILES or simple edits:
1. ALWAYS use file_read() FIRST to see current content
2. Determine edit type:
   - REPLACE specific text → file_edit(path, old_text, new_text)
   - ADD to END → file_append(path, content)
   - COMPLEX changes → file_read() + modify + file_write()
3. NEVER use file_append() for middle insertions (data loss risk!)
4. NEVER use file_write() without reading first (overwrites everything!)

Decision Tree:
User wants to modify existing file
  ↓ Does file exist?
    ├─ NO → file_write()
    └─ YES → What type?
        ├─ REPLACE text → file_edit()
        ├─ ADD to END → file_append()
        └─ INSERT/RESTRUCTURE → file_read() + file_write()
"""

    efficient_guidance = """
FOR LARGE FILES (use efficient tools - NO full file read):
1. INSERT at specific line → insert_at_line(path, line_number, content)
2. REPLACE line range → replace_lines(path, start, end, new_content)
3. INSERT after pattern → insert_after_pattern(path, "^def main", content)
4. INSERT before pattern → insert_before_pattern(path, "^if __name__", content)

Examples - Efficient Tools (PREFERRED for large files):
- "Add function at line 50" → insert_at_line(path, 50, function_code)
- "Replace lines 100-110" → replace_lines(path, 100, 110, new_code)
- "Add comment after main" → insert_after_pattern(path, "^def main", comment)
- "Add import before class" → insert_before_pattern(path, "^class", import_line)

Benefits:
- No memory overhead from reading entire file
- Fast operations on multi-thousand line files
- Preserves indentation and formatting
- Pattern-based insertion for precise placement
"""

    if include_efficient:
        return basic_guidance + "\n" + efficient_guidance
    else:
        return basic_guidance


# --- Auto-registration for common patterns ---


def create_agent_with_file_tools(
    models: List[str], instructions: List[Any], include_efficient: bool = True, **kwargs
):
    """
    Create a PromptChain agent with file tools automatically registered.

    This is a convenience factory function that creates an agent and registers
    all file operation tools in one step.

    Args:
        models: List of model names (e.g., ["openai/gpt-4"])
        instructions: List of instructions for the agent
        include_efficient: Whether to include efficient editing tools
        **kwargs: Additional arguments passed to PromptChain constructor

    Returns:
        PromptChain: Agent with file tools pre-registered

    Example:
        >>> from promptchain.tools.file_tool_registry import create_agent_with_file_tools
        >>>
        >>> coding_agent = create_agent_with_file_tools(
        ...     models=["openai/gpt-4"],
        ...     instructions=["Write Python code: {input}"],
        ...     verbose=True
        ... )
        >>> # Agent already has all 12 file operation tools registered
    """
    from promptchain import PromptChain

    # Create agent
    agent = PromptChain(models=models, instructions=instructions, **kwargs)  # type: ignore[arg-type]

    # Register file tools
    register_file_tools(agent, include_efficient=include_efficient)

    return agent


if __name__ == "__main__":
    # Example usage demonstration
    print("File Tool Registry - Example Usage\n")
    print("=" * 60)

    print("\n1. Manual Registration:")
    print("   from promptchain import PromptChain")
    print("   from promptchain.tools.file_tool_registry import register_file_tools")
    print("")
    print("   agent = PromptChain(models=['openai/gpt-4'], instructions=[...])")
    print("   register_file_tools(agent)  # Adds all 12 file tools")

    print("\n2. Factory Function:")
    print(
        "   from promptchain.tools.file_tool_registry import create_agent_with_file_tools"
    )
    print("")
    print("   agent = create_agent_with_file_tools(")
    print("       models=['openai/gpt-4'],")
    print("       instructions=['Code generation: {input}']")
    print("   )")

    print("\n3. Get Prompt Guidance:")
    print(
        "   from promptchain.tools.file_tool_registry import get_file_tool_prompt_guidance"
    )
    print("")
    print("   guidance = get_file_tool_prompt_guidance()")
    print("   agent_objective = f'''")
    print("   You are a coding agent.")
    print("")
    print("   {guidance}")
    print("   '''")

    print("\n" + "=" * 60)
    print("Available Tools (12 total):")
    print("=" * 60)

    for i, tool in enumerate(ALL_FILE_TOOLS, 1):
        print(f"{i:2d}. {tool.__name__}()")

    print("\n" + "=" * 60)
    print("Tool Categories:")
    print("=" * 60)
    print("Basic Operations (8):   file_read, file_write, file_edit,")
    print("                        file_append, file_delete, list_directory,")
    print("                        create_directory, read_file_range")
    print("")
    print("Efficient Editing (4):  insert_at_line, replace_lines,")
    print("                        insert_after_pattern, insert_before_pattern")
