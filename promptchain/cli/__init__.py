"""PromptChain CLI - Interactive terminal interface for PromptChain.

This package provides a Textual-based terminal UI for interactive conversations
with AI agents, supporting multi-agent management, session persistence, file
references, and shell command execution.
"""

# Import sandbox tools to trigger CLI tool registration
# This makes the 5 agentic provisioning tools available to agents
try:
    from .tools.sandbox import registration  # noqa: F401
except ImportError:
    # Graceful fallback if dependencies are not available
    pass

# Import library tools to trigger CLI tool registration
# This makes 14 core library tools available to agents (file ops, search, terminal)
try:
    from .tools.library import \
        registration as _library_registration  # noqa: F401
except ImportError:
    # Graceful fallback if dependencies are not available
    pass

# NOTE (v0.6.1): The TUI subpackage is NOT eagerly imported here. Importing it
# pulls in `textual`, which is an optional dependency declared under
# `extras_require["tui"]`. Library consumers (`pip install promptchain`) must
# be able to `import promptchain` and use programmatic APIs (PromptChain,
# PubSubBus, etc.) without textual installed. TUI users should:
#   - install `pip install "promptchain[tui]"`
#   - explicitly `from promptchain.cli.tui.app import PromptChainApp`
#   - or use the `promptchain` console script entry point.
from .command_handler import CommandHandler, CommandResult, ParsedCommand
from .file_reference_parser import FileReference, FileReferenceParser
from .models import Agent, Message, Session
from .session_manager import SessionManager
from .shell_executor import ShellExecutor, ShellResult

__all__ = [
    # Session management
    "SessionManager",
    "Session",
    # Command handling
    "CommandHandler",
    "CommandResult",
    "ParsedCommand",
    # File operations
    "FileReference",
    "FileReferenceParser",
    # Shell execution
    "ShellExecutor",
    "ShellResult",
    # Data models
    "Agent",
    "Message",
]

__version__ = "0.5.0"
