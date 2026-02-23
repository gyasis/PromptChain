#!/usr/bin/env python3
"""
Agentic Chat Team with Gemini MCP Integration
==============================================

A 5-agent conversational system with:
- Research, Analysis, Terminal, Documentation, and Synthesis agents
- Gemini MCP server integration for information searches
- Complete logging and history management
- Interactive chat sessions with persistent context

Usage:
    python agentic_team_chat.py                    # Default: verbose logging
    python agentic_team_chat.py --quiet            # Suppress all logging
    python agentic_team_chat.py --no-logging       # Disable file logging
    python agentic_team_chat.py --log-level INFO   # Set specific log level
"""

import asyncio
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.logging_utils import RunLogger
from promptchain.utils.orchestrator_supervisor import OrchestratorSupervisor  # ✅ Import from library
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Import visual output system
from visual_output import ChatVisualizer
from promptchain.tools.terminal.terminal_tool import TerminalTool

# Load environment variables
load_dotenv()


def setup_gemini_mcp_config():
    """Configure Gemini MCP server integration"""
    return [{
        "id": "gemini",
        "type": "stdio",
        "command": "/home/gyasis/Documents/code/claude_code-gemini-mcp/.venv/bin/python",
        "args": ["/home/gyasis/Documents/code/claude_code-gemini-mcp/server.py"],
        "cwd": "/home/gyasis/Documents/code/claude_code-gemini-mcp",
        "env": {}
    }]


def create_research_agent(mcp_config, verbose=False):
    """
    Research Agent: Expert at finding information and conducting research
    Uses Gemini MCP for advanced search capabilities with AgenticStepProcessor
    """
    # Create agentic step processor with Gemini MCP access
    research_step = AgenticStepProcessor(
        objective="""You are the Research Agent with access to Gemini search capabilities.

OBJECTIVE: Conduct comprehensive research using available Gemini search tools, verify information, and provide detailed findings with sources.

Your expertise:
- Deep web research using Gemini tools (gemini_research, ask_gemini)
- Fact verification from multiple sources
- Comprehensive information gathering
- Trend analysis and pattern recognition
- Source citation and validation

Research methodology:
1. Use gemini_research for broad topics requiring Google Search grounding
2. Use ask_gemini for direct questions and clarifications
3. Verify critical facts from multiple angles
4. Organize findings logically with proper citations
5. Highlight confidence levels and uncertainties

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: MARKDOWN FORMATTING GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE HEADERS, NOT NUMBERED LISTS:
✅ GOOD:
## Candle Library Overview
**Candle** is a minimalist ML framework for Rust...

### Key Features
Candle provides **pure Rust implementation** with no Python dependencies...

### Basic Example
```rust
use candle_core::{Tensor, Device};
```

❌ BAD (DON'T DO THIS):
1. Candle Library Overview
   • Candle is a minimalist ML framework
   • Key features:
      1. Pure Rust implementation
      2. No Python dependencies

FORMATTING RULES:
- Use ## for main sections (h2)
- Use ### for subsections (h3)
- Use **bold** for emphasis and key terms
- Use bullet points SPARINGLY - only for lists of 3+ related items
- Write in paragraphs with proper flow
- Use code blocks for examples
- Use tables for comparisons
- Avoid nested numbered lists

STRUCTURE YOUR RESPONSE:
## [Main Topic]
Brief intro paragraph with **key terms bold**.

### [Subtopic 1]
Paragraph explaining this aspect...

### [Subtopic 2]
Another paragraph...

### Quick Comparison (if relevant)
| Feature | Option A | Option B |
|---------|----------|----------|
| ...     | ...      | ...      |

### Summary
Final thoughts paragraph...

Always provide: findings, sources, confidence assessment, and recommendations for further research if needed.""",
        max_internal_steps=8,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[research_step],
        mcp_servers=mcp_config,  # Only research agent has Gemini access
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_analysis_agent(verbose=False):
    """
    Analysis Agent: Expert at analyzing data and extracting insights
    """
    analysis_step = AgenticStepProcessor(
        objective="""You are the Analysis Agent, an expert at analytical reasoning and insight extraction.

OBJECTIVE: Perform deep analytical reasoning to extract insights, identify patterns, and provide evidence-based conclusions.

Your expertise:
- Critical analysis of complex data and information
- Pattern recognition and trend identification
- Logical reasoning and inference
- Hypothesis formation and testing
- Risk assessment and opportunity evaluation
- Causal relationship analysis

Analytical methodology:
1. Decompose complex information into analyzable components
2. Identify key patterns, correlations, and anomalies
3. Apply logical reasoning to draw evidence-based conclusions
4. Assess confidence levels and identify uncertainties
5. Highlight actionable insights and strategic implications
6. Consider alternative explanations and counterarguments

Always provide: key insights, supporting evidence, confidence levels, and strategic recommendations.""",
        max_internal_steps=6,
        model_name="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5 with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[analysis_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_terminal_agent(scripts_dir: Path = None, verbose=False):
    """
    Terminal Agent: Expert at executing terminal commands and system operations
    Has access to TerminalTool for safe command execution
    Has access to file operation tools for direct file management
    Can execute scripts written by the Coding Agent
    """
    # Import file operations wrapper
    from promptchain.tools.file_operations import (
        file_read, file_write, file_edit, file_append, file_delete,
        list_directory, create_directory, read_file_range
    )

    # Create terminal tool with security guardrails
    terminal = TerminalTool(
        timeout=120,
        require_permission=False,  # Let agent make decisions autonomously
        verbose=verbose,  # Show terminal output when verbose
        use_persistent_session=True,
        session_name="agentic_terminal_session",
        visual_debug=True  # ✅ ALWAYS enable visual terminal emulator (better UX, not debug noise)
    )

    scripts_note = f"\n\nScripts workspace: {scripts_dir}\nYou can execute scripts written by the Coding Agent from this directory." if scripts_dir else ""

    terminal_step = AgenticStepProcessor(
        objective=f"""You are the Terminal Agent, an expert at system operations and command-line execution.

OBJECTIVE: Execute terminal commands and scripts safely and efficiently to accomplish system tasks and gather information.

Your expertise:
- Safe terminal command execution
- System administration and diagnostics
- File system operations and navigation
- Process management and monitoring
- Package management and installation
- Environment configuration
- Log analysis and debugging
- Executing scripts from the Coding Agent{scripts_note}

Available tools:
- execute_terminal_command(command: str) -> str: Execute terminal commands with security guardrails
- file_read(path: str) -> str: Read file contents
- file_write(path: str, content: str) -> str: Write/create file
- file_edit(path: str, old_text: str, new_text: str) -> str: Edit file by replacing text
- file_append(path: str, content: str) -> str: Append to file
- file_delete(path: str) -> str: Delete file
- list_directory(path: str) -> str: List directory contents
- create_directory(path: str) -> str: Create directory
- read_file_range(path: str, start_line: int, end_line: int) -> str: Read specific lines

Tool usage guidelines:
- Use execute_terminal_command() to run scripts created by Coding agent
- Use execute_terminal_command() for system commands (git, npm, apt, etc.)
- Use file operation tools for direct file management tasks:
  * create_directory() for "create a backups directory"
  * list_directory() for "list all log files" or "show files in folder"
  * file_delete() for "delete temp files" or "remove old logs"
  * file_read() for "check what's in error.log" or "read config file"
  * file_write() for quick file creation (use Coding agent for complex code)
- For writing complex scripts/code, defer to Coding agent
- You handle execution, file navigation, and direct file operations

CRITICAL: File Editing Best Practices
When user asks to edit/modify an EXISTING file:
1. ALWAYS use file_read() FIRST to see current content
2. Determine the edit type:
   a) REPLACE specific text → Use file_edit(path, old_text, new_text)
   b) ADD to END only → Use file_append(path, content)
   c) RESTRUCTURE → Use file_read() + file_write() with complete new content
3. NEVER use file_write() without reading first (destroys existing content)
4. For complex edits, consider deferring to Coding agent

Examples:
- "Add my API key to .env file" → file_append() (add to end)
- "Change port 8000 to 9000 in config" → file_edit(old_text, new_text)
- "Delete the DEBUG line from config" → file_read() + file_write() (complex)

Command execution methodology:
1. Analyze what needs to be accomplished
2. Plan safe, efficient command sequences
3. Execute commands or scripts one at a time
4. Verify results and handle errors
5. Provide clear output interpretation
6. Suggest follow-up actions if needed

Script execution:
- Scripts from Coding Agent are in: {scripts_dir if scripts_dir else 'scripts directory'}
- Execute with: bash <script>.sh or python <script>.py
- Review script contents before execution if needed

Security guidelines:
- Avoid destructive operations without explicit user request
- Use safe flags (e.g., -i for interactive confirmations)
- Verify paths before operations
- Check command availability before execution
- Handle errors gracefully

Always provide: command explanations, execution results, output interpretation, and recommendations.""",
        max_internal_steps=7,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    # Create agent with terminal tool registered
    agent = PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[terminal_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )

    # Create wrapper function with correct name for registration
    def execute_terminal_command(command: str) -> str:
        """Execute terminal command - wrapper for TerminalTool"""
        return terminal(command)

    # Register all tools (terminal + file operations)
    agent.register_tool_function(execute_terminal_command)
    agent.register_tool_function(file_read)
    agent.register_tool_function(file_write)
    agent.register_tool_function(file_edit)
    agent.register_tool_function(file_append)
    agent.register_tool_function(file_delete)
    agent.register_tool_function(list_directory)
    agent.register_tool_function(create_directory)
    agent.register_tool_function(read_file_range)

    # Add tool schemas for LLM
    agent.add_tools([
        {
            "type": "function",
            "function": {
                "name": "execute_terminal_command",
                "description": "Execute a terminal command safely with security guardrails. Returns command output (stdout/stderr). Use for system operations, file management, running programs, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The terminal command to execute (e.g., 'ls -la', 'python script.py', 'git status')"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file, creating it if it doesn't exist",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_edit",
                "description": "Edit a file by replacing old_text with new_text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Text to find and replace"},
                        "new_text": {"type": "string", "description": "Text to replace with"}
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_append",
                "description": "Append content to the end of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to append to"},
                        "content": {"type": "string", "description": "Content to append"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_delete",
                "description": "Delete a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to delete"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to directory to list (defaults to current directory)"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_directory",
                "description": "Create a directory and all parent directories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to directory to create"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file_range",
                "description": "Read a specific range of lines from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                        "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                        "end_line": {"type": "integer", "description": "Last line to read (inclusive)"}
                    },
                    "required": ["path", "start_line", "end_line"]
                }
            }
        }
        {
            "type": "function",
            "function": {
                "name": "insert_at_line",
                "description": "Insert content at a specific line number (efficient for large files - no full read required). Content is inserted BEFORE the specified line. Preserves proper indentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "line_number": {"type": "integer", "description": "Line number to insert at (1-indexed, content goes BEFORE this line)"},
                        "content": {"type": "string", "description": "Content to insert (can be multi-line, preserves tabs/spaces)"}
                    },
                    "required": ["path", "line_number", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "replace_lines",
                "description": "Replace a range of lines with new content (efficient for large files - no full read required). Preserves proper indentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "start_line": {"type": "integer", "description": "First line to replace (1-indexed, inclusive)"},
                        "end_line": {"type": "integer", "description": "Last line to replace (1-indexed, inclusive)"},
                        "new_content": {"type": "string", "description": "New content to replace the lines with (can be multi-line)"}
                    },
                    "required": ["path", "start_line", "end_line", "new_content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert_after_pattern",
                "description": "Insert content after a regex pattern match (efficient for large files). Preserves proper indentation. Use for adding code after function definitions, imports, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "pattern": {"type": "string", "description": "Regex pattern to search for (e.g., '^def main', '^import os')"},
                        "content": {"type": "string", "description": "Content to insert after the pattern (can be multi-line)"},
                        "first_match": {"type": "boolean", "description": "If True, only insert after first match; if False, after all matches (default: True)"}
                    },
                    "required": ["path", "pattern", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert_before_pattern",
                "description": "Insert content before a regex pattern match (efficient for large files). Preserves proper indentation. Use for adding imports, comments, or code before specific sections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "pattern": {"type": "string", "description": "Regex pattern to search for (e.g., '^if __name__', '^class MyClass')"},
                        "content": {"type": "string", "description": "Content to insert before the pattern (can be multi-line)"},
                        "first_match": {"type": "boolean", "description": "If True, only insert before first match; if False, before all matches (default: True)"}
                    },
                    "required": ["path", "pattern", "content"]
                }
            }
        },
    ])

    return agent


def create_documentation_agent(verbose=False):
    """
    Documentation Agent: Expert at creating clear, comprehensive documentation
    """
    documentation_step = AgenticStepProcessor(
        objective="""You are the Documentation Agent, an expert technical writer and educator.

OBJECTIVE: Create clear, comprehensive, and user-friendly documentation that explains concepts effectively.

Your expertise:
- Clear technical writing and explanation
- User guide and tutorial creation
- Complex concept simplification
- Information architecture and organization
- Example creation and illustration
- Troubleshooting and FAQ development

Documentation methodology:
1. Identify target audience and their knowledge level
2. Structure information hierarchically (overview → details)
3. Use clear, concise, jargon-free language
4. Provide concrete, practical examples
5. Anticipate common questions and pain points
6. Include troubleshooting guides and best practices
7. Add visual aids descriptions when beneficial

Always provide: clear explanations, practical examples, common pitfalls, and quick-start guides.""",
        max_internal_steps=5,
        model_name="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5 with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[documentation_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_synthesis_agent(verbose=False):
    """
    Synthesis Agent: Expert at combining insights and creating cohesive final outputs
    """
    synthesis_step = AgenticStepProcessor(
        objective="""You are the Synthesis Agent, an expert at integrating diverse information into unified strategic outputs.

OBJECTIVE: Synthesize insights from multiple sources into cohesive, actionable recommendations with strategic clarity.

Your expertise:
- Multi-source insight integration
- Narrative construction and storytelling
- Contradiction resolution and harmonization
- Strategic recommendation development
- Holistic thinking and systems analysis
- Actionable roadmap creation

Synthesis methodology:
1. Gather and review all available information comprehensively
2. Identify common themes, patterns, and unique insights
3. Resolve contradictions using logical analysis
4. Integrate perspectives into a unified framework
5. Develop clear, prioritized, actionable recommendations
6. Create strategic roadmaps with next steps
7. Highlight synergies and opportunities

Always provide: integrated insights, strategic recommendations, prioritized action items, and implementation considerations.""",
        max_internal_steps=6,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    return PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[synthesis_step],
        verbose=verbose,  # Configurable verbosity
        store_steps=True
    )


def create_coding_agent(scripts_dir: Path, verbose=False):
    """
    Coding Agent: Expert at writing scripts and code for terminal execution
    Writes scripts to shared workspace for terminal agent to execute
    Has access to file operation tools (file_read, file_write, file_edit, etc.)
    """
    # Import file operations wrapper
    from promptchain.tools.file_operations import (
        file_read, file_write, file_edit, file_append, file_delete,
        list_directory, create_directory, read_file_range
    )
    # Import efficient editing tools
    from promptchain.tools.efficient_file_edit import (
        insert_at_line, replace_lines, insert_after_pattern, insert_before_pattern
    )

    # Create write_script tool
    def write_script(filename: str, code: str) -> str:
        """Write a script to the scripts workspace"""
        try:
            # Ensure filename is safe (no path traversal)
            safe_filename = Path(filename).name
            script_path = scripts_dir / safe_filename

            # Write script
            with open(script_path, 'w') as f:
                f.write(code)

            # Make executable if it's a shell script
            if safe_filename.endswith(('.sh', '.bash', '.py')):
                import os
                os.chmod(script_path, 0o755)

            # Provide clear execution command
            if safe_filename.endswith('.py'):
                exec_cmd = f"python {script_path}"
            elif safe_filename.endswith(('.sh', '.bash')):
                exec_cmd = f"bash {script_path}"
            else:
                exec_cmd = f"{script_path}"

            return f"✅ Script written to: {script_path}\n✅ Terminal agent can execute it with: {exec_cmd}"
        except Exception as e:
            return f"❌ Error writing script: {e}"

    coding_step = AgenticStepProcessor(
        objective=f"""You are the Coding Agent, an expert at writing clean, efficient scripts and code.

OBJECTIVE: Write scripts, automation code, and utilities that the Terminal agent can execute.

Your expertise:
- Python scripting and automation
- Bash/shell scripting
- Data processing scripts
- System administration scripts
- API integration scripts
- File manipulation utilities

Scripts workspace: {scripts_dir}

Available tools:
- write_script(filename: str, code: str) -> str: Write a script to the workspace
- file_read(path: str) -> str: Read file contents
- file_write(path: str, content: str) -> str: Write/create file
- file_edit(path: str, old_text: str, new_text: str) -> str: Edit file by replacing text
- file_append(path: str, content: str) -> str: Append to file
- file_delete(path: str) -> str: Delete file
- list_directory(path: str) -> str: List directory contents
- create_directory(path: str) -> str: Create directory
- read_file_range(path: str, start_line: int, end_line: int) -> str: Read specific lines

EFFICIENT EDITING TOOLS (for large files - avoid reading entire file):
- insert_at_line(path: str, line_number: int, content: str) -> str: Insert at specific line
- replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str: Replace line range
- insert_after_pattern(path: str, pattern: str, content: str, first_match: bool) -> str: Insert after regex match
- insert_before_pattern(path: str, pattern: str, content: str, first_match: bool) -> str: Insert before regex match

Tool usage guidelines:
- Use write_script() for executable scripts (.py, .sh) that Terminal agent will run
- Use file_write() for config files, data files, documentation, non-executable files
- Use file_read() to analyze existing code/files before creating new scripts
- Use file_edit() to update configuration files or modify existing code
- Use list_directory() to understand project structure before creating files
- Terminal agent handles direct file operations ("create a directory", "delete temp files")
- You focus on creating CODE and SCRIPTS that accomplish tasks

CRITICAL: File Editing Best Practices
When user asks to "edit", "update", "add to", or "modify" an EXISTING file:

FOR LARGE FILES (use efficient tools - NO full file read):
1. INSERT at specific line → insert_at_line(path, line_number, content)
2. REPLACE line range → replace_lines(path, start, end, new_content)
3. INSERT after pattern → insert_after_pattern(path, "^def main", content)
4. INSERT before pattern → insert_before_pattern(path, "^if __name__", content)

FOR SMALL FILES or simple edits:
1. ALWAYS use file_read() FIRST to see current content
2. Determine the edit type:
   a) REPLACE specific text → Use file_edit(path, old_text, new_text)
   b) ADD section to END → Use file_append(path, content)
   c) INSERT/RESTRUCTURE → Use file_read() + file_write() with full new content
3. NEVER use file_append() for edits that need to preserve/modify existing content
4. NEVER use file_write() without reading first (destroys existing content)

Examples - Efficient Tools (PREFERRED for large files):
- "Add function at line 50" → insert_at_line(path, 50, function_code)
- "Replace lines 100-110" → replace_lines(path, 100, 110, new_code)
- "Add comment after main function" → insert_after_pattern(path, "^def main", comment)
- "Add import before if __name__" → insert_before_pattern(path, "^if __name__", import_line)

Examples - Traditional Tools:
- "Append today's log entry" → file_append() (add to end)
- "Change DEBUG=False to DEBUG=True" → file_edit(old_text, new_text)
- "Update the API key in config" → file_edit(old_text, new_text)

Coding methodology:
1. Understand the task requirements clearly
2. Choose appropriate language (Python, Bash, etc.)
3. Write clean, well-commented code
4. Include error handling and validation
5. Make scripts executable and robust
6. Provide clear usage instructions
7. Follow best practices and security guidelines

Security guidelines:
- Validate all inputs
- Avoid hardcoded credentials
- Use safe file operations
- Include proper error handling
- Add helpful comments and documentation

Always provide: script filename, code with comments, usage instructions, and any dependencies needed.""",
        max_internal_steps=6,
        model_name="openai/gpt-4.1-mini",  # Latest model with 1M token context
        history_mode="progressive"  # Better multi-hop reasoning with full history
    )

    # Create agent with write_script tool
    agent = PromptChain(
        models=[],  # Empty - AgenticStepProcessor has its own model_name
        instructions=[coding_step],
        verbose=verbose,
        store_steps=True
    )

    # Register all tools (write_script + file operations + efficient editing)
    agent.register_tool_function(write_script)
    agent.register_tool_function(file_read)
    agent.register_tool_function(file_write)
    agent.register_tool_function(file_edit)
    agent.register_tool_function(file_append)
    agent.register_tool_function(file_delete)
    agent.register_tool_function(list_directory)
    agent.register_tool_function(create_directory)
    agent.register_tool_function(read_file_range)
    # Efficient editing tools
    agent.register_tool_function(insert_at_line)
    agent.register_tool_function(replace_lines)
    agent.register_tool_function(insert_after_pattern)
    agent.register_tool_function(insert_before_pattern)

    # Add tool schemas for LLM
    agent.add_tools([
        {
            "type": "function",
            "function": {
                "name": "write_script",
                "description": "Write a script to the scripts workspace for terminal agent execution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Script filename (e.g., process_data.py, backup.sh)"},
                        "code": {"type": "string", "description": "Complete script code with proper syntax"}
                    },
                    "required": ["filename", "code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": "Read the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_write",
                "description": "Write content to a file, creating it if it doesn't exist",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write"},
                        "content": {"type": "string", "description": "Content to write to the file"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_edit",
                "description": "Edit a file by replacing old_text with new_text using sed",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "old_text": {"type": "string", "description": "Text to find and replace"},
                        "new_text": {"type": "string", "description": "Text to replace with"}
                    },
                    "required": ["path", "old_text", "new_text"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_append",
                "description": "Append content to the end of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to append to"},
                        "content": {"type": "string", "description": "Content to append"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "file_delete",
                "description": "Delete a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to delete"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to directory to list (defaults to current directory)"}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_directory",
                "description": "Create a directory and all parent directories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to directory to create"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file_range",
                "description": "Read a specific range of lines from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                        "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                        "end_line": {"type": "integer", "description": "Last line to read (inclusive)"}
                    },
                    "required": ["path", "start_line", "end_line"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert_at_line",
                "description": "Insert content at a specific line number (efficient for large files - no full read required). Content is inserted BEFORE the specified line. Preserves proper indentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "line_number": {"type": "integer", "description": "Line number to insert at (1-indexed, content goes BEFORE this line)"},
                        "content": {"type": "string", "description": "Content to insert (can be multi-line, preserves tabs/spaces)"}
                    },
                    "required": ["path", "line_number", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "replace_lines",
                "description": "Replace a range of lines with new content (efficient for large files - no full read required). Preserves proper indentation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "start_line": {"type": "integer", "description": "First line to replace (1-indexed, inclusive)"},
                        "end_line": {"type": "integer", "description": "Last line to replace (1-indexed, inclusive)"},
                        "new_content": {"type": "string", "description": "New content to replace the lines with (can be multi-line)"}
                    },
                    "required": ["path", "start_line", "end_line", "new_content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert_after_pattern",
                "description": "Insert content after a regex pattern match (efficient for large files). Preserves proper indentation. Use for adding code after function definitions, imports, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "pattern": {"type": "string", "description": "Regex pattern to search for (e.g., '^def main', '^import os')"},
                        "content": {"type": "string", "description": "Content to insert after the pattern (can be multi-line)"},
                        "first_match": {"type": "boolean", "description": "If True, only insert after first match; if False, after all matches (default: True)"}
                    },
                    "required": ["path", "pattern", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "insert_before_pattern",
                "description": "Insert content before a regex pattern match (efficient for large files). Preserves proper indentation. Use for adding imports, comments, or code before specific sections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to edit"},
                        "pattern": {"type": "string", "description": "Regex pattern to search for (e.g., '^if __name__', '^class MyClass')"},
                        "content": {"type": "string", "description": "Content to insert before the pattern (can be multi-line)"},
                        "first_match": {"type": "boolean", "description": "If True, only insert before first match; if False, before all matches (default: True)"}
                    },
                    "required": ["path", "pattern", "content"]
                }
            }
        }
    ])

    return agent


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Agentic Chat Team - 6-Agent Collaborative System with Coding + Terminal Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                        # Run with default verbose logging
  %(prog)s --clean                # Clean output (only agent responses), full logs to files
  %(prog)s --quiet                # Suppress all console output
  %(prog)s --dev                  # Development mode with full observability
  %(prog)s --no-logging           # Disable file logging
  %(prog)s --log-level ERROR      # Only show errors
  %(prog)s --quiet --no-logging   # Minimal mode (no logging at all)
        """
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all console logging and verbose output'
    )

    parser.add_argument(
        '--dev',
        action='store_true',
        help='Dev mode: quiet terminal + full debug logs to file (recommended for development)'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean output mode: only show agent responses in terminal, still save full logs to files'
    )

    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable file logging completely'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level for terminal output (default: INFO)'
    )

    parser.add_argument(
        '--no-history-management',
        action='store_true',
        help='Disable automatic history truncation'
    )

    parser.add_argument(
        '--max-history-tokens',
        type=int,
        default=180000,  # 90% of 200k standard context window for optimal model usage
        help='Maximum tokens for history (default: 180000 - 90%% of context limit)'
    )

    parser.add_argument(
        '--session-name',
        type=str,
        default=None,
        help='Custom session name for logs and cache'
    )

    return parser.parse_args()


def create_router_config():
    """Configure intelligent autonomous routing logic with multi-agent planning capability"""
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")

    return {
        "models": ["openai/gpt-4.1-mini"],  # Latest model with 1M token context for better routing decisions
        "instructions": [None, "{input}"],
        "decision_prompt_templates": {
            "static_plan": f"""You are the Master Orchestrator, responsible for creating multi-agent execution plans.

CURRENT DATE: {current_date}
KNOWLEDGE CUTOFF: Most LLM training data is from early 2024 or earlier

USER REQUEST: {{user_input}}

CONVERSATION HISTORY:
{{history}}

AVAILABLE AGENTS & TOOLS:
{{agent_details}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT TOOL CAPABILITIES (CRITICAL - USE CORRECT TOOLS FOR EACH TASK):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RESEARCH AGENT ✅ [HAS TOOLS]
   Tools: gemini_research, ask_gemini (Gemini MCP with Google Search grounding)
   When to use: ANY question about current events, recent tech, latest updates, post-2024 info
   Capabilities: Real-time web search, fact verification, current information retrieval
   Example: "What is Candle library?" → MUST use Research (current docs/examples from web)

2. ANALYSIS AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge + provided data)
   When to use: Analyzing data/text already provided, pattern recognition, logical reasoning
   Limitations: Cannot get external info, stuck with training data cutoff

3. CODING AGENT ✅ [HAS TOOLS]
   Tools: write_script (creates Python/Bash files in workspace)
   When to use: Need to CREATE scripts/code files
   Capabilities: Writes executable scripts to disk

4. TERMINAL AGENT ✅ [HAS TOOLS]
   Tools: execute_terminal_command (runs shell commands)
   When to use: Need to EXECUTE commands or scripts
   Capabilities: Run scripts, check system, file operations

5. DOCUMENTATION AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge)
   When to use: Explaining concepts from training data, writing tutorials
   Limitations: CANNOT verify current info, knowledge cutoff applies

6. SYNTHESIS AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge + provided context)
   When to use: Combining/synthesizing information already gathered
   Limitations: Cannot get new info, only works with what's provided

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL ROUTING RULES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚨 KNOWLEDGE BOUNDARY DETECTION:
- If query is about NEW tech/libraries/events (post-2024) → MUST START with RESEARCH
- If query asks "what is X?" for unknown/recent tech → MUST START with RESEARCH
- If query needs CURRENT information → MUST START with RESEARCH
- If query can be answered from training data alone → Can use agents without tools
- If unsure whether info is current → SAFER to use RESEARCH first

🚨 TOOL REQUIREMENT DETECTION:
- Need web search/current info? → RESEARCH (has Gemini MCP)
- Need to write code files? → CODING (has write_script)
- Need to execute commands? → TERMINAL (has execute_terminal_command)
- Just explaining known concepts? → DOCUMENTATION or ANALYSIS (no tools needed)

🚨 MULTI-STEP PLANNING:
When task needs multiple capabilities, create a PLAN (sequence of agents):
- "What is X library? Show examples" → ["research", "documentation"]
  (research gets current info, documentation formats it clearly)
- "Research X and analyze findings" → ["research", "analysis"]
  (research gathers data, analysis processes it)
- "Create backup script and run it" → ["coding", "terminal"]
  (coding writes script, terminal executes it)
- "Find latest AI trends and create strategy" → ["research", "synthesis"]
  (research gets current trends, synthesis creates strategy)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF CORRECT PLANS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What is Rust's Candle library?"
✅ CORRECT: {{"plan": ["research", "documentation"], "initial_input": "Research Rust Candle library - current features, examples, and documentation"}}
❌ WRONG: {{"plan": ["documentation"], ...}} ← Documentation has NO tools, will hallucinate!

Query: "Analyze this dataset: [data]"
✅ CORRECT: {{"plan": ["analysis"], "initial_input": "Analyze the provided dataset..."}}
❌ No research needed - data already provided

Query: "Create a log cleanup script and run it"
✅ CORRECT: {{"plan": ["coding", "terminal"], "initial_input": "Create log cleanup script, then execute it"}}

Query: "What are latest GPT-5 features and how to use them?"
✅ CORRECT: {{"plan": ["research", "documentation"], "initial_input": "Research GPT-5 latest features (current as of {current_date}), then create usage guide"}}

Query: "Explain how neural networks work"
✅ CORRECT: {{"plan": ["documentation"], "initial_input": "Explain neural networks fundamentals"}}
❌ No research needed - well-established concept in training data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RETURN JSON ONLY (no other text):
{{
    "plan": ["agent1", "agent2", ...],  // Ordered list of agents to execute sequentially
    "initial_input": "refined task description for first agent",
    "reasoning": "why this plan and these specific agents with their tools"
}}

If only ONE agent needed, plan can have single agent: {{"plan": ["agent_name"], ...}}
""",
            "single_agent_dispatch": """You are the Orchestrator, responsible for routing tasks to the most capable agent.

USER REQUEST: {user_input}

CONVERSATION HISTORY:
{history}

AVAILABLE AGENTS:
{agent_details}

AGENT CAPABILITIES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. RESEARCH AGENT
   - Has exclusive access to Gemini MCP tools (gemini_research, ask_gemini)
   - Expert at: web research, fact-finding, information gathering, source verification
   - Best for: "find information about X", "research Y", "what are the latest trends in Z"
   - Uses: AgenticStepProcessor with up to 8 reasoning steps

2. ANALYSIS AGENT
   - Expert at: data analysis, pattern recognition, critical thinking, insight extraction
   - Best for: "analyze this data", "what patterns exist", "evaluate X vs Y"
   - Uses: AgenticStepProcessor with up to 6 reasoning steps

3. CODING AGENT
   - Has write_script tool to create Python/Bash scripts saved to scripts workspace
   - Expert at: writing scripts, automation code, utilities, data processing scripts
   - Best for: "write a script to X", "create a Python program for Y", "automate Z task"
   - Scripts saved to workspace for Terminal agent execution
   - Uses: AgenticStepProcessor with up to 6 reasoning steps

4. TERMINAL AGENT
   - Has access to TerminalTool for safe command execution (execute_terminal_command)
   - Can execute scripts from Coding Agent's workspace
   - Expert at: terminal commands, script execution, file operations, system administration
   - Best for: "execute this script", "run commands", "check system status", "install package"
   - Uses: AgenticStepProcessor with up to 7 reasoning steps

5. DOCUMENTATION AGENT
   - Expert at: technical writing, tutorials, concept explanation, user guides
   - Best for: "explain how X works", "create guide for Y", "document Z"
   - Uses: AgenticStepProcessor with up to 5 reasoning steps

6. SYNTHESIS AGENT
   - Expert at: integrating insights, strategic planning, actionable recommendations
   - Best for: "combine insights", "create strategy", "final recommendations"
   - Uses: AgenticStepProcessor with up to 6 reasoning steps

ROUTING LOGIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- If request needs web search or fact-finding → RESEARCH (has Gemini tools)
- If request needs data analysis or pattern recognition → ANALYSIS
- If request needs writing a script or code → CODING (has write_script tool)
- If request needs executing commands or scripts → TERMINAL (has TerminalTool)
- If request needs terminal commands or system operations → TERMINAL (has TerminalTool)
- If request needs explanation or documentation → DOCUMENTATION
- If request needs synthesis or strategic planning → SYNTHESIS

Consider conversation history for context-aware routing.
Each agent uses AgenticStepProcessor for multi-step reasoning autonomy.

RETURN JSON ONLY:
{{"chosen_agent": "agent_name", "refined_query": "optimally refined query for selected agent", "reasoning": "brief explanation of routing choice"}}
"""
        }
    }


def create_agentic_orchestrator(agent_descriptions: dict, log_event_callback=None, dev_print_callback=None):
    """
    Create AgenticStepProcessor-based orchestrator - returns async wrapper function for AgentChain

    Args:
        agent_descriptions: Dict of agent names -> descriptions
        log_event_callback: Optional callback for logging orchestrator decisions
        dev_print_callback: Optional callback for --dev mode terminal output
    """
    from datetime import datetime
    from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

    # Format agent descriptions for the objective
    agents_formatted = "\n".join([f"- {name}: {desc}" for name, desc in agent_descriptions.items()])

    # Storage for agentic step metadata (captured via events)
    orchestrator_metadata = {
        "total_steps": 0,
        "tools_called": 0,
        "execution_time_ms": 0
    }

    # Create the orchestrator PromptChain once (closure captures this)
    orchestrator_step = AgenticStepProcessor(
        objective=f"""You are the Master Orchestrator. Use multi-step reasoning to create the best execution plan.

CURRENT DATE: {{current_date}}
KNOWLEDGE CUTOFF: Most LLM training data is from early 2024 or earlier

AVAILABLE AGENTS:
{agents_formatted}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT TOOL CAPABILITIES (CRITICAL - USE MULTI-HOP REASONING TO DETERMINE NEEDS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. RESEARCH AGENT ✅ [HAS TOOLS]
   Tools: gemini_research, ask_gemini (Gemini MCP with Google Search grounding)
   When to use: ANY question about current events, recent tech, latest updates, post-2024 info
   Capabilities: Real-time web search, fact verification, current information retrieval
   Example: "What is Candle library?" → MUST use Research (current docs/examples from web)

2. ANALYSIS AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge + provided data)
   When to use: Analyzing data/text already provided, pattern recognition, logical reasoning
   Limitations: Cannot get external info, stuck with training data cutoff

3. CODING AGENT ✅ [HAS TOOLS]
   Tools: write_script (creates Python/Bash files in workspace)
   When to use: Need to CREATE scripts/code files
   Capabilities: Writes executable scripts to disk

4. TERMINAL AGENT ✅ [HAS TOOLS]
   Tools: execute_terminal_command (runs shell commands)
   When to use: Need to EXECUTE commands or scripts
   Capabilities: Run scripts, check system, file operations

5. DOCUMENTATION AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge)
   When to use: Explaining concepts from training data, writing tutorials
   Limitations: CANNOT verify current info, knowledge cutoff applies

6. SYNTHESIS AGENT ❌ [NO TOOLS - INTERNAL KNOWLEDGE ONLY]
   Tools: None (uses only training knowledge + provided context)
   When to use: Combining/synthesizing information already gathered
   Limitations: Cannot get new info, only works with what's provided

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 MULTI-HOP REASONING PROCESS (5-STEP ANALYSIS):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: Analyze task complexity
- Is this a simple query needing one agent? Or complex needing multiple?
- What capabilities are required? (research, analysis, coding, execution, documentation, synthesis)

STEP 2: Distinguish KNOWLEDGE vs EXECUTION
⚠️ CRITICAL: Even if you have system context (like current_date), the user may want VERIFICATION via command execution!

Knowledge-based queries (NO TOOLS NEEDED):
- "What year is it?" → Can answer from system context
- "Explain neural networks" → Training knowledge sufficient
- "Tell me about X concept" → Documentation/analysis sufficient

Execution-based queries (TOOLS REQUIRED):
- "Check the date" → TERMINAL needed (must run 'date' command)
- "Verify system time" → TERMINAL needed (execute actual check)
- "Run date command" → TERMINAL needed (explicit command request)
- "Get current timestamp" → TERMINAL needed (system execution)

🔴 IF QUERY CONTAINS EXECUTION VERBS, USE TOOLS:
- "check", "verify", "run", "execute", "test", "show me actual"
- These indicate user wants REAL SYSTEM OUTPUT, not just your knowledge

STEP 3: Check knowledge boundaries
- Is the topic NEW/UNKNOWN (post-2024)? → Need RESEARCH
- Is the topic WELL-KNOWN (in training)? → Can use agents without tools
- Uncertain? → SAFER to use RESEARCH first

STEP 4: Check tool requirements
- Need web search/current info? → RESEARCH (has Gemini MCP)
- Need to write code files? → CODING (has write_script)
- Need to execute commands/verify system? → TERMINAL (has execute_terminal_command)
- Just explaining known concepts? → DOCUMENTATION or ANALYSIS (no tools needed)

STEP 5: Design execution plan
- If multiple capabilities needed → Create SEQUENTIAL PLAN
- If single capability sufficient → Create SINGLE-AGENT PLAN
- Order matters! Research before documentation, coding before terminal, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MULTI-AGENT PLANNING PATTERNS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Pattern 1: Research → Documentation
- "What is X library? Show examples" → ["research", "documentation"]
- Research gets current info, documentation formats it clearly

Pattern 2: Research → Analysis → Synthesis
- "Research X and create strategy" → ["research", "analysis", "synthesis"]
- Research gathers data, analysis processes it, synthesis creates strategy

Pattern 3: Coding → Terminal
- "Create backup script and run it" → ["coding", "terminal"]
- Coding writes script, terminal executes it

Pattern 4: Research → Coding → Terminal
- "Find latest X library and create demo" → ["research", "coding", "terminal"]
- Research finds current library, coding creates demo, terminal runs it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF CORRECT MULTI-HOP REASONING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Query: "What year is it?"
Step 1: Simple task - needs one agent
Step 2: KNOWLEDGE-based - can answer from system context (current_date available)
Step 3: Well-known information in context
Step 4: No tools needed
Step 5: Plan = ["documentation"]
✅ CORRECT: {{"plan": ["documentation"], "initial_input": "Confirm current year", "reasoning": "Simple knowledge query, system context available, no execution needed"}}

Query: "Check the date on the system"
Step 1: Simple task - needs one agent
Step 2: EXECUTION-based - user wants ACTUAL system output (verb: "check")
Step 3: Well-known but requires verification
Step 4: TERMINAL needed (execute_terminal_command to run 'date')
Step 5: Plan = ["terminal"]
✅ CORRECT: {{"plan": ["terminal"], "initial_input": "Execute date command to verify system date", "reasoning": "User wants command execution verification, not just knowledge answer"}}

Query: "Write tutorial about Zig sound manipulation"
Step 1: Complex task - needs research + documentation
Step 2: KNOWLEDGE-based but topic is unknown
Step 3: "Zig sound" is post-2024 tech → Need RESEARCH first
Step 4: Research has gemini_research, documentation for writing
Step 5: Plan = ["research", "documentation"]
✅ CORRECT: {{"plan": ["research", "documentation"], "initial_input": "Research Zig audio libraries", "reasoning": "Unknown tech needs web research, then tutorial writing"}}

Query: "Explain neural networks"
Step 1: Simple task - needs one agent
Step 2: KNOWLEDGE-based, well-known concept
Step 3: Training knowledge sufficient
Step 4: No tools needed for known concepts
Step 5: Plan = ["documentation"]
✅ CORRECT: {{"plan": ["documentation"], "initial_input": "Explain neural networks", "reasoning": "Well-known concept, no research needed"}}

Query: "Create log cleanup script and test it"
Step 1: Complex task - needs coding + execution
Step 2: EXECUTION-based - needs file creation + command execution
Step 3: Well-known task, no research needed
Step 4: Coding has write_script, terminal has execute_terminal_command
Step 5: Plan = ["coding", "terminal"]
✅ CORRECT: {{"plan": ["coding", "terminal"], "initial_input": "Create log cleanup script", "reasoning": "Need to write script then execute it"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USE YOUR INTERNAL REASONING STEPS TO ANALYZE THE QUERY, THEN OUTPUT:

RETURN JSON ONLY (no other text):
{{
    "plan": ["agent1", "agent2", ...],  // Ordered list of agents (can be single agent ["agent_name"])
    "initial_input": "refined task description for first agent",
    "reasoning": "multi-step reasoning summary of why this plan"
}}
""",
        max_internal_steps=8,  # ✅ MORE STEPS for complex multi-agent planning
        model_name="openai/gpt-4.1-mini",
        history_mode="progressive"  # Context accumulation - CRITICAL for multi-hop reasoning!
    )

    orchestrator_chain = PromptChain(
        models=[],  # AgenticStepProcessor has its own model
        instructions=[orchestrator_step],
        verbose=False,
        store_steps=True
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EVENT CALLBACK: Capture agentic step metadata (v0.4.1d+)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def orchestrator_event_callback(event: ExecutionEvent):
        """Capture orchestrator execution metadata via events"""
        # Capture agentic step completion
        if event.event_type == ExecutionEventType.AGENTIC_STEP_END:
            orchestrator_metadata["total_steps"] = event.metadata.get("total_steps", 0)
            orchestrator_metadata["tools_called"] = event.metadata.get("total_tools_called", 0)
            orchestrator_metadata["execution_time_ms"] = event.metadata.get("execution_time_ms", 0)

            # Log agentic step details
            if log_event_callback:
                log_event_callback("orchestrator_agentic_step", {
                    "total_steps": event.metadata.get("total_steps", 0),
                    "tools_called": event.metadata.get("total_tools_called", 0),
                    "execution_time_ms": event.metadata.get("execution_time_ms", 0),
                    "objective_achieved": event.metadata.get("objective_achieved", False),
                    "max_steps_reached": event.metadata.get("max_steps_reached", False)
                }, level="DEBUG")

    # Register event callback (v0.4.1d)
    orchestrator_chain.register_callback(orchestrator_event_callback)

    # Return async wrapper function that AgentChain expects
    async def agentic_router_wrapper(
        user_input: str,
        history: list,  # AgentChain passes 'history', not 'conversation_history'
        agent_descriptions: dict
    ) -> str:
        """Async wrapper for AgenticStepProcessor orchestrator with decision logging"""
        # Inject current date dynamically
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Log the orchestrator input
        if log_event_callback:
            log_event_callback("orchestrator_input", {
                "user_query": user_input,
                "history_size": len(history),
                "current_date": current_date
            }, level="DEBUG")

        # Use the orchestrator chain with progressive history
        result = await orchestrator_chain.process_prompt_async(user_input)

        # Parse and log the orchestrator decision
        try:
            import json
            decision = json.loads(result)
            chosen_agent = decision.get("chosen_agent", "unknown")
            reasoning = decision.get("reasoning", "no reasoning provided")

            # Get step metadata from event callback (v0.4.1d+)
            step_count = orchestrator_metadata.get("total_steps", 0)
            tools_called = orchestrator_metadata.get("tools_called", 0)
            exec_time_ms = orchestrator_metadata.get("execution_time_ms", 0)

            # Log the complete orchestrator decision with metadata
            if log_event_callback:
                log_event_callback("orchestrator_decision", {
                    "chosen_agent": chosen_agent,
                    "reasoning": reasoning,
                    "raw_output": result,
                    "internal_steps": step_count,
                    "tools_called": tools_called,
                    "execution_time_ms": exec_time_ms,
                    "user_query": user_input
                }, level="INFO")

            logging.info(f"🎯 ORCHESTRATOR DECISION: Agent={chosen_agent} | Steps={step_count} | "
                       f"Tools={tools_called} | Time={exec_time_ms:.2f}ms | Reasoning: {reasoning}")

            # --DEV MODE: Show orchestrator decision in terminal
            if dev_print_callback:
                dev_print_callback(f"\n🎯 Orchestrator Decision:", "")
                dev_print_callback(f"   Agent chosen: {chosen_agent}", "")
                dev_print_callback(f"   Reasoning: {reasoning}", "")
                if step_count > 0:
                    dev_print_callback(f"   Internal steps: {step_count}", "")
                dev_print_callback("", "")  # Blank line

        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse orchestrator output as JSON: {result}")
            if log_event_callback:
                log_event_callback("orchestrator_parse_error", {
                    "error": str(e),
                    "raw_output": result
                }, level="WARNING")

        return result  # Returns JSON string with chosen_agent

    return agentic_router_wrapper


async def main():
    """Main function to run the agentic chat team"""

    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging with proper date-based organization FIRST (needed for file handler)
    today = datetime.now().strftime('%Y-%m-%d')

    # Get the directory where this script is located, then create logs relative to it
    script_dir = Path(__file__).parent
    log_dir = script_dir / "logs" / today
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine session name early (needed for log file)
    session_name = args.session_name or f"session_{datetime.now().strftime('%H%M%S')}"
    session_log_path = log_dir / f"{session_name}.log"  # .log extension for detailed debug logs
    session_jsonl_path = log_dir / f"{session_name}.jsonl"  # .jsonl for structured events

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LOGGING ARCHITECTURE: Dual-Mode Observability (v0.4.2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NORMAL MODE (default):
    #   - Terminal: Clean output (agent responses only, no observability spam)
    #   - Files: EVERYTHING (complete audit trail - DEBUG level)
    #
    # --dev MODE:
    #   - Terminal: Full observability (orange history, cyan agentic steps, etc.)
    #   - Files: EVERYTHING (same as normal mode)
    #
    # --clean MODE:
    #   - Terminal: ONLY agent responses (no system messages, no logs at all)
    #   - Files: EVERYTHING (complete audit trail - DEBUG level)
    #
    # --quiet MODE:
    #   - Terminal: Errors/warnings only
    #   - Files: EVERYTHING (same as normal mode)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ✅ NEW (v0.4.2): ObservabilityFilter for clean terminal in normal mode
    class ObservabilityFilter(logging.Filter):
        """
        Filter to block observability logs from terminal while preserving them in files.

        Blocks logs containing observability markers like:
        - [HISTORY INJECTED] - Inter-agent history injection
        - [AGENTIC STEP] - AgenticStepProcessor internal reasoning
        - [INTERNAL CONVERSATION HISTORY] - Agentic internal conversations
        - orchestrator_reasoning_step - Multi-hop reasoning steps
        - plan_agent_start - Multi-agent plan execution

        Usage: Applied to terminal handler in NORMAL mode only (not --dev or --quiet)
        """

        OBSERVABILITY_MARKERS = [
            # History injection patterns
            '[HISTORY INJECTED]',           # History injection to agents
            '[HISTORY START]',              # History block markers
            '[HISTORY END]',

            # Agentic step patterns
            '[AGENTIC STEP]',               # AgenticStepProcessor iterations
            '[INTERNAL CONVERSATION HISTORY', # Agentic internal conversations
            '[AgenticStepProcessor]',       # Agentic processor logs
            '[Agentic Callback]',           # Agentic callback events

            # Orchestrator patterns
            'orchestrator_reasoning_step',  # Multi-hop reasoning
            '[orchestrator_decision]',      # Orchestrator decision logs
            'Executing tool: output_reasoning_step',  # Orchestrator tool calls
            'Tool output_reasoning_step',   # Orchestrator tool results

            # Plan execution patterns
            'plan_agent_start',             # Multi-agent plan steps
            'plan_agent_complete',

            # Tool execution patterns (catch all tool logs except user-visible ones)
            'Executing tool:',              # Generic tool execution
            'Tool write_script',            # File writing operations
            'Tool read_file',               # File reading operations

            # RunLog and event patterns
            '[RunLog] Event:',              # RunLogger event logs
            'reasoning_steps',              # Metadata about reasoning

            # Internal history and thought patterns
            '💭 Latest Thought:',
            '🔧 Tools Called',
            '📊 Internal History:',

            # Separator lines
            '═' * 40,                       # Double-line separators
            '=' * 80,                       # Single-line separators (80 chars)
            '=' * 60,                       # Single-line separators (60 chars)
            '────────────────',             # Dash separators
        ]

        def filter(self, record):
            """Return False to BLOCK log from terminal, True to allow"""
            msg = record.getMessage()

            # Block if ANY observability marker present
            for marker in self.OBSERVABILITY_MARKERS:
                if marker in msg:
                    return False  # Block from terminal

            # Allow user-facing logs (agent responses, errors, warnings)
            return True

    # Clear any existing handlers
    logging.root.handlers = []

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # 1. File Handler - ALWAYS DEBUG level (captures everything)
    if not args.no_logging:
        file_handler = logging.FileHandler(session_log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Capture EVERYTHING to file
        file_handler.setFormatter(detailed_formatter)
        logging.root.addHandler(file_handler)

    # 2. Terminal Handler - Controlled by flags with observability filtering
    terminal_handler = logging.StreamHandler()

    if args.dev:
        # ✅ DEV MODE: Show EVERYTHING in terminal (full observability)
        terminal_handler.setLevel(logging.INFO)
        # NO filter - show all orange/cyan observability logs
        logging.info("🔧 DEV MODE: Full observability enabled in terminal")

    elif args.clean:
        # ✅ CLEAN MODE: Only agent responses (block ALL system logs, observability, etc.)
        terminal_handler.setLevel(logging.CRITICAL + 1)  # Block everything except direct output
        logging.debug("✨ CLEAN MODE: Minimal terminal output (only agent responses), full logs to files")

    elif args.quiet:
        # ✅ QUIET MODE: Only errors/warnings
        terminal_handler.setLevel(logging.ERROR)

    else:
        # ✅ NORMAL MODE: Clean terminal (filter out observability spam)
        terminal_handler.setLevel(logging.INFO)
        terminal_handler.addFilter(ObservabilityFilter())  # Block [HISTORY INJECTED], [AGENTIC STEP], etc.
        logging.debug("✨ NORMAL MODE: Clean terminal output (observability logs filtered to files only)")

    terminal_handler.setFormatter(simple_formatter)
    logging.root.addHandler(terminal_handler)

    # Set root logger to DEBUG (handlers control what's actually shown/saved)
    logging.root.setLevel(logging.DEBUG)

    # Configure specific loggers
    logging.getLogger("promptchain").setLevel(logging.DEBUG)  # Capture all promptchain events

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SUPPRESS NOISY THIRD-PARTY LOGGERS (--dev mode clean output)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    logging.getLogger("httpx").setLevel(logging.ERROR)  # Suppress HTTP request logs
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)  # Suppress "completion() model=" logs
    logging.getLogger("litellm").setLevel(logging.ERROR)  # Suppress lowercase variant
    logging.getLogger("google.generativeai").setLevel(logging.ERROR)  # Suppress Gemini AFC logs
    logging.getLogger("google").setLevel(logging.ERROR)  # Suppress all Google SDK logs
    logging.getLogger("mcp").setLevel(logging.ERROR)  # Suppress MCP server logs
    logging.getLogger("__main__").setLevel(logging.ERROR)  # Suppress MCP server __main__ logs

    # Suppress rich console handler logs from MCP servers
    for logger_name in ["google.ai.generativelanguage", "google.api_core", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Always filter out the "No models defined" warning (expected behavior with AgenticStepProcessor)
    import warnings
    warnings.filterwarnings("ignore", message=".*No models defined in PromptChain.*")

    # Configure output based on quiet/dev/clean flags
    verbose = not (args.quiet or args.dev or args.clean)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # --DEV MODE HELPER: Direct stdout printing for backend visibility
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def dev_print(message: str, prefix: str = ""):
        """Print to stdout in --dev mode (bypasses logging system)"""
        if args.dev:
            print(f"{prefix}{message}")

    # Log the logging configuration (v0.4.2 - dual-mode observability)
    logging.debug(f"━━━ DUAL-MODE LOGGING CONFIGURATION (v0.4.2) ━━━")
    if args.dev:
        logging.debug(f"Mode: DEV (full observability in terminal + files)")
    elif args.clean:
        logging.debug(f"Mode: CLEAN (only agent responses in terminal, everything in files)")
    elif args.quiet:
        logging.debug(f"Mode: QUIET (errors only in terminal, everything in files)")
    else:
        logging.debug(f"Mode: NORMAL (clean terminal with ObservabilityFilter, everything in files)")
    logging.debug(f"Terminal Level: {logging.getLevelName(terminal_handler.level)}")
    logging.debug(f"Terminal Filter: {'ObservabilityFilter (blocks [HISTORY INJECTED], [AGENTIC STEP], etc.)' if not args.dev and not args.quiet and not args.clean else 'None' if not args.clean else 'CRITICAL+1 (blocks everything except direct output)'}")
    logging.debug(f"File Level: DEBUG (all events)" if not args.no_logging else "File Logging: DISABLED")
    logging.debug(f"Session Log: {session_log_path}")
    logging.debug(f"Event Log: {session_jsonl_path}")
    logging.debug(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Initialize visual output system
    viz = ChatVisualizer()

    if verbose:
        viz.render_header(
            "AGENTIC CHAT TEAM",
            "6-Agent Collaborative System with Multi-Hop Reasoning"
        )
        viz.render_system_message("Initializing agents and systems...", "info")

    # Cache organized by date (log_dir already created earlier)
    cache_dir = script_dir / "logs" / "cache" / today
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Scripts directory for coding agent (shared workspace for generated scripts)
    scripts_dir = script_dir / "logs" / "scripts" / today
    scripts_dir.mkdir(parents=True, exist_ok=True)

    if verbose and not args.no_logging:
        print(f"✅ Debug Log: {session_log_path}")
        print(f"✅ Event Log: {session_jsonl_path}")
        print(f"✅ Scripts workspace: {scripts_dir}")

    # Initialize history manager with configurable settings
    history_manager = ExecutionHistoryManager(
        max_tokens=args.max_history_tokens,
        max_entries=100,
        truncation_strategy="oldest_first"
    )

    if verbose:
        print(f"✅ History manager: {args.max_history_tokens} max tokens")

    # Setup MCP configuration for Gemini
    mcp_config = setup_gemini_mcp_config()

    # Create the 6 specialized agents with AgenticStepProcessor
    # NOTE: All agents are created with verbose=False for clean terminal output
    # All detailed logging goes to the session log file
    if verbose:
        print("🔧 Creating 6 specialized agents with agentic reasoning capabilities...\n")

    agents = {
        "research": create_research_agent(mcp_config, verbose=False),  # Only agent with Gemini MCP access
        "analysis": create_analysis_agent(verbose=False),
        "coding": create_coding_agent(scripts_dir, verbose=False),  # Coding agent with script writing capability
        "terminal": create_terminal_agent(scripts_dir, verbose=False),  # ✅ Terminal agent with visual emulator ALWAYS enabled
        "documentation": create_documentation_agent(verbose=False),
        "synthesis": create_synthesis_agent(verbose=False)
    }

    agent_descriptions = {
        "research": "Expert researcher with exclusive Gemini MCP access (gemini_research, ask_gemini) for web search and fact-finding. Uses AgenticStepProcessor with 8-step reasoning.",
        "analysis": "Expert analyst for data analysis, pattern recognition, and insight extraction. Uses AgenticStepProcessor with 6-step reasoning.",
        "coding": f"Expert coder who writes Python/Bash scripts and saves them to {scripts_dir} for Terminal agent execution. Has comprehensive file operation tools: write_script, file_read, file_write, file_edit, file_append, file_delete, list_directory, create_directory, read_file_range. Uses AgenticStepProcessor with 6-step reasoning.",
        "terminal": f"Expert system operator with TerminalTool access for executing commands and scripts from {scripts_dir}. Also has file operation tools (file_read, file_write, list_directory, etc.) for direct file management. Uses AgenticStepProcessor with 7-step reasoning.",
        "documentation": "Expert technical writer for clear guides, tutorials, and documentation. Uses AgenticStepProcessor with 5-step reasoning.",
        "synthesis": "Expert synthesizer for integrating insights and strategic planning. Uses AgenticStepProcessor with 6-step reasoning."
    }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENHANCED EVENT LOGGING SYSTEM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Structured events (JSONL) for orchestrator decisions, tool calls, agent actions
    # NO TRUNCATION - Capture complete details for debugging
    # Uses PromptChain Event System (v0.4.1d+) - No more regex parsing!
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def log_event(event_type: str, data: dict, level: str = "INFO"):
        """
        Enhanced event logging - captures FULL details without truncation

        Args:
            event_type: Type of event (orchestrator_decision, tool_call, agent_response, etc.)
            data: Complete event data (not truncated)
            level: Log level (INFO, DEBUG, WARNING, ERROR)
        """
        if not args.no_logging:
            try:
                import json
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "level": level,
                    "event": event_type,
                    "session": session_name,
                    **data
                }
                # Write to JSONL with pretty formatting for readability (NO truncation)
                with open(session_jsonl_path, 'a') as f:
                    f.write(json.dumps(log_entry, indent=None, ensure_ascii=False) + "\n")

                # Also log to Python logging system for .log file
                log_msg = f"[{event_type}] " + " | ".join([f"{k}={v}" for k, v in data.items()])
                getattr(logging, level.lower(), logging.info)(log_msg)

            except Exception as e:
                logging.error(f"Event logging failed: {e}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EVENT CALLBACKS: Structured Tool Call Logging (v0.4.1d+)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NO MORE REGEX PARSING! Using PromptChain event system for structured data
    # Events: TOOL_CALL_START, TOOL_CALL_END, MCP_TOOL_DISCOVERED, etc.
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

    def agent_event_callback(event: ExecutionEvent):
        """
        Unified event callback for all agents
        Captures tool calls, model calls, and agentic steps with structured data
        """
        # Tool call start - capture arguments
        if event.event_type == ExecutionEventType.TOOL_CALL_START:
            tool_name = event.metadata.get("tool_name", "unknown")
            tool_args = event.metadata.get("tool_args", {})
            is_mcp = event.metadata.get("is_mcp_tool", False)

            # Display in terminal
            tool_type = "mcp" if is_mcp else "local"
            args_str = str(tool_args) if tool_args else ""

            # --DEV MODE: Show tool calls directly in stdout
            dev_print(f"🔧 Tool Call: {tool_name} ({tool_type})", "")
            if tool_args:
                # Show first 150 chars of args
                args_preview = args_str[:150] + "..." if len(args_str) > 150 else args_str
                dev_print(f"   Args: {args_preview}", "")

            # Normal mode: use viz
            if not args.dev:
                viz.render_tool_call(tool_name, tool_type, args_str[:100])

            # Log to file with FULL structured data (no truncation)
            log_event("tool_call_start", {
                "tool_name": tool_name,
                "tool_type": tool_type,
                "tool_args": tool_args,  # ✅ Structured dict, not string!
                "is_mcp_tool": is_mcp
            }, level="INFO")

        # Tool call end - capture results
        elif event.event_type == ExecutionEventType.TOOL_CALL_END:
            tool_name = event.metadata.get("tool_name", "unknown")
            tool_result = event.metadata.get("result", "")
            execution_time = event.metadata.get("execution_time_ms", 0)
            success = event.metadata.get("success", True)

            # Log to file with complete results (NO TRUNCATION - user wants EVERYTHING)
            log_event("tool_call_end", {
                "tool_name": tool_name,
                "result_length": len(str(tool_result)),
                "result_preview": str(tool_result),  # ✅ FULL OUTPUT - no truncation
                "execution_time_ms": execution_time,
                "success": success
            }, level="INFO")

        # Agentic internal steps - track reasoning (v0.4.2 - enhanced with full activity visibility + internal history)
        elif event.event_type == ExecutionEventType.AGENTIC_INTERNAL_STEP:
            iteration = event.metadata.get("iteration", 0)
            max_iterations = event.metadata.get("max_iterations", 0)
            assistant_thought = event.metadata.get("assistant_thought", "")
            tool_calls = event.metadata.get("tool_calls", [])
            tools_count = event.metadata.get("tools_called_count", 0)
            exec_time = event.metadata.get("execution_time_ms", 0)
            tokens = event.metadata.get("tokens_used", 0)
            has_answer = event.metadata.get("has_final_answer", False)
            error = event.metadata.get("error")

            # ✅ NEW (v0.4.2): Internal conversation history tracking
            internal_history = event.metadata.get("internal_history", [])
            internal_history_length = event.metadata.get("internal_history_length", 0)
            internal_history_tokens = event.metadata.get("internal_history_tokens", 0)
            llm_history_sent = event.metadata.get("llm_history_sent", [])

            # ✅ LIBRARY-LEVEL OBSERVABILITY: Always display (not just --dev mode)
            # Use cyan color for agentic internal reasoning differentiation
            cyan_start = "\033[36m"
            cyan_end = "\033[0m"

            logging.info(f"{cyan_start}[AGENTIC STEP] iteration={iteration}/{max_iterations} | time={exec_time:.0f}ms | tokens={tokens}{cyan_end}")
            logging.info(f"{cyan_start}📊 Internal History: {internal_history_length} messages, {internal_history_tokens} tokens{cyan_end}")

            # ✅ Show internal conversation history (ALWAYS, not just --dev)
            if internal_history:
                logging.info(f"{cyan_start}{'='*80}{cyan_end}")
                logging.info(f"{cyan_start}[INTERNAL CONVERSATION HISTORY - Step {iteration}]{cyan_end}")
                for idx, msg in enumerate(internal_history):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    # Log full content at INFO level (no truncation)
                    logging.info(f"{cyan_start}[{idx+1}] {role}:{cyan_end}")
                    if content:
                        # Split content by lines for better readability
                        for line in str(content).split('\n'):
                            logging.info(f"{cyan_start}    {line}{cyan_end}")
                    else:
                        logging.info(f"{cyan_start}    [No content]{cyan_end}")
                logging.info(f"{cyan_start}{'='*80}{cyan_end}")

            # Show what the agent thought/decided
            if assistant_thought and assistant_thought != "[No output]":
                logging.info(f"{cyan_start}💭 Latest Thought: {assistant_thought}{cyan_end}")

            # Show tool calls with results
            if tool_calls:
                logging.info(f"{cyan_start}🔧 Tools Called ({tools_count}):{cyan_end}")
                for tc in tool_calls:
                    tool_name = tc.get("name", "unknown")
                    tool_result = str(tc.get("result", ""))  # ✅ FULL result
                    logging.info(f"{cyan_start}   • {tool_name}:{cyan_end}")
                    # Log full tool result (no truncation)
                    for line in tool_result.split('\n'):
                        logging.info(f"{cyan_start}      {line}{cyan_end}")

            # Show if this produced the final answer
            if has_answer:
                logging.info(f"{cyan_start}✅ Produced final answer{cyan_end}")

            # Show errors
            if error:
                logging.error(f"{cyan_start}❌ Error: {error}{cyan_end}")

            # Log agentic reasoning to file (FULL internal history, NO truncation)
            log_event("agentic_internal_step", {
                "iteration": iteration,
                "max_iterations": max_iterations,
                "assistant_thought": assistant_thought,  # ✅ FULL thought
                "tools_called_count": tools_count,
                "execution_time_ms": exec_time,
                "tokens_used": tokens,
                "has_final_answer": has_answer,
                "error": error,
                # ✅ NEW: Full internal conversation history for debugging
                "internal_history": internal_history,  # Complete conversation
                "internal_history_length": internal_history_length,
                "internal_history_tokens": internal_history_tokens,
                "llm_history_sent": llm_history_sent  # What LLM actually received
            }, level="DEBUG")

        # Model calls - track LLM usage
        elif event.event_type == ExecutionEventType.MODEL_CALL_END:
            model_name = event.metadata.get("model", event.model_name)
            tokens_used = event.metadata.get("tokens_used", 0)
            exec_time = event.metadata.get("execution_time_ms", 0)

            log_event("model_call", {
                "model": model_name,
                "tokens_used": tokens_used,
                "execution_time_ms": exec_time
            }, level="DEBUG")

    # Register event callback for all agents (v0.4.1d)
    for agent_name, agent in agents.items():
        agent.register_callback(agent_event_callback)
        logging.debug(f"✅ Registered event callback for {agent_name} agent")

    # Create the orchestrator supervisor with oversight capabilities
    orchestrator_supervisor = OrchestratorSupervisor(
        agent_descriptions=agent_descriptions,
        log_event_callback=log_event,
        dev_print_callback=dev_print,  # ✅ Show orchestrator's internal reasoning steps in terminal (full output, no truncation)
        max_reasoning_steps=8,  # Multi-hop reasoning for complex decisions
        model_name="openai/gpt-4.1-mini"
    )

    # Create the agent chain with router mode
    if verbose:
        print("🚀 Creating AgentChain with OrchestratorSupervisor (multi-hop reasoning)...\n")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PER-AGENT HISTORY CONFIGURATION (v0.4.2)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Optimize token usage by configuring history per agent based on their needs:
    # - Terminal/Coding agents: Minimal/no history (save 30-60% tokens)
    # - Research/Analysis agents: Full history for context
    # - Documentation/Synthesis agents: Full history for comprehensive understanding
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    agent_history_configs = {
        # Terminal agent: No history needed - executes commands based on current input
        # Saves ~3000-5000 tokens per call (40-60% reduction)
        "terminal": {
            "enabled": False,  # Completely disable history
        },

        # Coding agent: Full history (kitchen_sink mode) - needs context for comprehensive coding
        "coding": {
            "enabled": True,
            "max_tokens": 80000,  # ✅ MUCH HIGHER for kitchen_sink mode
            "max_entries": 500,  # ✅ NO LIMIT on entries
            "truncation_strategy": "oldest_first",
            # ✅ NO include_types filter = kitchen_sink mode (includes EVERYTHING)
        },

        # Research agent: Full history (kitchen_sink mode) - needs ALL context for comprehensive research
        "research": {
            "enabled": True,
            "max_tokens": 80000,  # ✅ MUCH HIGHER for kitchen_sink mode (1M context model)
            "max_entries": 500,  # ✅ NO LIMIT on entries
            "truncation_strategy": "oldest_first",
            # ✅ NO include_types filter = kitchen_sink mode (includes EVERYTHING)
        },

        # Analysis agent: Full history (kitchen_sink mode) - needs ALL conversation context for analysis
        "analysis": {
            "enabled": True,
            "max_tokens": 80000,  # ✅ kitchen_sink mode
            "max_entries": 500,
            "truncation_strategy": "oldest_first",
        },

        # Documentation agent: Full history (kitchen_sink mode) - needs complete context for docs
        "documentation": {
            "enabled": True,
            "max_tokens": 80000,  # ✅ kitchen_sink mode
            "max_entries": 500,
            "truncation_strategy": "oldest_first",
        },

        # Synthesis agent: Full history (kitchen_sink mode) - synthesizes across entire conversation
        "synthesis": {
            "enabled": True,
            "max_tokens": 80000,  # ✅ kitchen_sink mode
            "max_entries": 500,
            "truncation_strategy": "oldest_first",
        }
    }

    if verbose:
        print("✅ Per-agent history configurations (KITCHEN_SINK MODE - FULL VISIBILITY):")
        print(f"   - Terminal: History disabled (command execution doesn't need context)")
        print(f"   - Coding: KITCHEN_SINK (80K tokens, 500 entries, ALL message types)")
        print(f"   - Research: KITCHEN_SINK (80K tokens, 500 entries, ALL message types)")
        print(f"   - Analysis: KITCHEN_SINK (80K tokens, 500 entries, ALL message types)")
        print(f"   - Documentation: KITCHEN_SINK (80K tokens, 500 entries, ALL message types)")
        print(f"   - Synthesis: KITCHEN_SINK (80K tokens, 500 entries, ALL message types)")
        print(f"   🔍 NO filters on message types = see EVERYTHING in logs\n")

    agent_chain = AgentChain(
        agents=agents,
        agent_descriptions=agent_descriptions,
        execution_mode="router",
        router=orchestrator_supervisor.supervise_and_route,  # ✅ Use supervisor's routing method
        router_strategy="static_plan",  # ✅ MULTI-AGENT ORCHESTRATION: Enable multi-agent planning with oversight
        cache_config={
            "name": session_name,
            "path": str(cache_dir)
        },
        verbose=False,  # Clean terminal - all logs go to file
        auto_include_history=True,  # ✅ Global setting (can be overridden per-agent)
        agent_history_configs=agent_history_configs  # ✅ NEW (v0.4.2): Per-agent history optimization
    )

    # ✅ Register orchestration callback for multi-agent plan progress visibility
    def orchestration_event_callback(event_type: str, data: dict):
        """Handle orchestration events for multi-agent plan execution"""
        if event_type == "plan_agent_start":
            step_num = data.get("step_number", 0)
            total_steps = data.get("total_steps", 0)
            agent_name = data.get("agent_name", "unknown")
            plan = data.get("plan", [])

            # Display plan progress
            dev_print(f"📋 Multi-Agent Plan: {' → '.join(plan)}", "")
            dev_print(f"▶️  Step {step_num}/{total_steps}: {agent_name.upper()} Agent", "")

            # Log to file
            log_event("plan_agent_start", data, level="INFO")

        elif event_type == "plan_agent_complete":
            step_num = data.get("step_number", 0)
            total_steps = data.get("total_steps", 0)
            agent_name = data.get("agent_name", "unknown")
            output_len = data.get("output_length", 0)
            output_content = data.get("output_content", "")
            plan = data.get("plan", [])

            # Display completion
            dev_print(f"✅ Step {step_num}/{total_steps} Complete: {agent_name} ({output_len} chars output)", "")

            # ✅ NEW (v0.4.2): Show actual output content in --dev mode for full visibility
            if args.dev and output_content:
                dev_print(f"", "")
                dev_print(f"   📤 Output from {agent_name}:", "")
                dev_print(f"   ───────────────────────────────────────────────────────────────", "")
                # Show output with indentation
                for line in output_content.split('\n'):
                    dev_print(f"   {line}", "")
                dev_print(f"   ───────────────────────────────────────────────────────────────", "")

            # Show data flow to next agent if not the last step
            if step_num < total_steps:
                next_agent = plan[step_num] if step_num < len(plan) else "unknown"
                dev_print(f"   ↪️  Passing output to {next_agent} agent...", "")
                dev_print(f"", "")  # Blank line for readability

            # Log to file
            log_event("plan_agent_complete", data, level="INFO")

    agent_chain.register_orchestration_callback(orchestration_event_callback)
    logging.debug("✅ Registered orchestration callback for multi-agent plan tracking")

    # Log initialization
    log_event("system_initialized", {
        "agents": list(agents.keys()),
        "mcp_servers": ["gemini"],
        "cache_location": str(cache_dir),
        "max_history_tokens": args.max_history_tokens
    })

    viz.render_system_message("System initialized successfully!", "success")

    # Display team roster with visual styling
    viz.render_team_roster(agents)

    # Display available commands
    viz.render_commands_help()

    # Track session stats
    stats = {
        "total_queries": 0,
        "agent_usage": {name: 0 for name in agents.keys()},
        "start_time": datetime.now(),
        "history_truncations": 0,
        "total_tokens_processed": 0
    }

    # Helper function for automatic history management
    def manage_history_automatically():
        """Automatically manage and truncate history if needed"""
        try:
            # Check current history size
            current_size = history_manager.current_token_count
            max_size = history_manager.max_tokens

            if current_size > max_size * 0.9:  # 90% threshold
                print(f"\n⚠️  History approaching limit ({current_size}/{max_size} tokens)")
                print("    Automatically truncating oldest entries...")

                # Truncate using built-in strategy
                history_manager.truncate_to_limit()
                stats["history_truncations"] += 1

                new_size = history_manager.current_token_count
                print(f"    ✅ History truncated: {current_size} → {new_size} tokens\n")

                # Log the truncation event
                log_event("history_truncated", {
                    "old_size": current_size,
                    "new_size": new_size,
                    "truncation_count": stats["history_truncations"]
                })

            return current_size
        except Exception as e:
            log_event("history_management_error", {"error": str(e)})
            return 0

    # Helper function to display history summary
    def show_history_summary():
        """Display a summary of current history"""
        try:
            total_entries = history_manager.history_size
            total_tokens = history_manager.current_token_count
            max_tokens = history_manager.max_tokens

            print("\n📊 HISTORY SUMMARY:")
            print("=" * 80)
            print(f"Total Entries: {total_entries}")
            print(f"Total Tokens: {total_tokens} / {max_tokens} ({(total_tokens/max_tokens*100):.1f}% full)")
            print(f"Truncations: {stats['history_truncations']}")
            print(f"Queries Processed: {stats['total_queries']}")
            print("=" * 80)

            # Show entry type breakdown
            entry_types = {}
            for entry in history_manager.history:
                entry_type = entry.get('type', 'unknown')
                entry_types[entry_type] = entry_types.get(entry_type, 0) + 1

            print("\nEntry Breakdown:")
            for entry_type, count in sorted(entry_types.items()):
                print(f"  - {entry_type}: {count}")
            print("=" * 80)

        except Exception as e:
            print(f"⚠️  Error displaying history: {e}")

    try:
        # Interactive chat loop
        while True:
            # Use Claude Code-style input box
            user_input = viz.get_input_with_box().strip()

            if not user_input:
                continue

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # SLASH COMMANDS (v0.4.2)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if user_input.startswith('/'):
                command_parts = user_input[1:].strip().split()
                command = command_parts[0].lower() if command_parts else ''

                # /exit, /quit, /q - Exit the program
                if command in ['exit', 'quit', 'q']:
                    viz.render_system_message("👋 Exiting Agentic Team Chat...", "info")
                    log_event("slash_command", {
                        "command": "exit",
                        "session_duration": str(datetime.now() - stats['start_time'])
                    })
                    break

                # Unknown command
                else:
                    viz.render_system_message(f"❌ Unknown command: /{command}", "error")
                    viz.render_system_message("Available commands: /exit, /quit, /q", "info")
                    continue
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if user_input.lower() in ['exit', 'quit']:
                viz.render_system_message("Ending session...", "info")
                break

            if user_input.lower() == 'help':
                viz.render_commands_help()
                continue

            if user_input.lower() == 'clear':
                viz.clear_screen()
                viz.render_header(
                    "AGENTIC CHAT TEAM",
                    "6-Agent Collaborative System with Multi-Hop Reasoning"
                )
                continue

            if user_input.lower() == 'history':
                print("\n📜 CONVERSATION HISTORY:")
                print("=" * 80)
                formatted_history = history_manager.get_formatted_history(
                    format_style='chat',
                    max_tokens=4000
                )
                print(formatted_history)
                print("=" * 80)

                # Also show summary
                show_history_summary()
                continue

            if user_input.lower() == 'history-summary':
                show_history_summary()
                continue

            if user_input.lower() == 'stats':
                duration = datetime.now() - stats["start_time"]
                current_tokens = history_manager.current_token_count
                max_tokens = history_manager.max_tokens

                viz.render_stats({
                    "Session Duration": str(duration).split('.')[0],
                    "Total Queries": stats['total_queries'],
                    "History Truncations": stats['history_truncations'],
                    "History Size": f"{current_tokens} / {max_tokens} tokens",
                    "History Usage": f"{(current_tokens/max_tokens*100):.1f}%",
                    "Total Entries": history_manager.history_size
                })

                print("\nAgent Usage:")
                for agent, count in stats['agent_usage'].items():
                    print(f"  - {agent}: {count} queries")

                # Logging info
                print(f"\nLogs Directory: {log_dir}")
                print("=" * 80)
                continue

            if user_input.lower() == 'save':
                summary_path = log_dir / f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(summary_path, 'w') as f:
                    f.write("AGENTIC TEAM SESSION SUMMARY\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Session Duration: {datetime.now() - stats['start_time']}\n")
                    f.write(f"Total Queries: {stats['total_queries']}\n\n")
                    f.write("CONVERSATION HISTORY:\n")
                    f.write("=" * 80 + "\n")
                    f.write(history_manager.get_formatted_history(format_style='chat'))
                print(f"\n💾 Session saved to: {summary_path}")
                continue

            # Process user query
            stats['total_queries'] += 1

            # Automatic history management - check before processing
            current_history_size = manage_history_automatically()

            # Add user input to history
            history_manager.add_entry("user_input", user_input, source="user")

            # Comprehensive logging
            log_event("user_query", {
                "query": user_input,
                "history_size_before": current_history_size,
                "history_entries": history_manager.history_size
            })

            try:
                # Let the agent chain handle routing and execution
                # Clean terminal: just show processing indicator
                if verbose:
                    print("\n⏳ Processing...\n")

                # Execute agent chain with full metadata (v0.4.1b)
                result = await agent_chain.process_input(user_input, return_metadata=True)

                # Track which agent was used
                history_manager.add_entry("agent_output", result.response, source="agent_chain")

                # --DEV MODE: Show which agent is responding
                dev_print(f"📤 Agent Responding: {result.agent_name}", "")
                dev_print("─" * 80, "")

                # Render response with rich markdown formatting
                viz.render_agent_response(result.agent_name, result.response, show_banner=False)

                # Log comprehensive execution metadata (NO TRUNCATION)
                # Note: router_steps now contains orchestrator reasoning steps when OrchestratorSupervisor is used
                # (number of output_reasoning_step tool calls during multi-hop reasoning)
                log_event("agent_execution_complete", {
                    "agent_name": result.agent_name,
                    "response": result.response,  # Full response, not truncated
                    "response_length": len(result.response),
                    "response_word_count": len(result.response.split()),
                    "execution_time_ms": result.execution_time_ms,
                    "router_decision": result.router_decision,
                    "router_steps": result.router_steps,  # ✅ Now properly tracked from OrchestratorSupervisor
                    "tools_called": len(result.tools_called),  # TODO: Capture agent tool calls (not orchestrator tools)
                    "tool_details": result.tools_called,  # Full structured data
                    "total_tokens": result.total_tokens,  # TODO: Aggregate from agent LLM calls
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cache_hit": result.cache_hit,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "history_size_after": history_manager.current_token_count
                }, level="INFO")

                # Log summary to Python logger (appears in .log file)
                logging.info(f"✅ Agent execution completed | Agent: {result.agent_name} | "
                           f"Time: {result.execution_time_ms:.2f}ms | Tools: {len(result.tools_called)} | "
                           f"Tokens: {result.total_tokens if result.total_tokens else 'N/A'}")

            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                viz.render_system_message(error_msg, "error")
                history_manager.add_entry("error", error_msg, source="system")

                # Log error details to file
                log_event("error", {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": user_input
                })

    finally:
        # Save final session summary
        if verbose:
            print("\n💾 Saving final session summary...")

        # Final log entry with complete session summary (all in one file)
        log_event("session_ended", {
            "total_queries": stats['total_queries'],
            "history_truncations": stats['history_truncations'],
            "duration": str(datetime.now() - stats['start_time']),
            "final_history_size": history_manager.current_token_count,
            "agent_usage": stats['agent_usage'],
            "log_file": str(session_log_path)
        })

        if verbose:
            print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    asyncio.run(main())
