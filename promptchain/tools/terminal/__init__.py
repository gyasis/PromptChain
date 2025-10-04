"""
Terminal Tool Package for PromptChain

A secure, modular terminal execution tool that works with all PromptChain patterns:
- Simple function injection
- Direct callable usage
- AgenticStepProcessor integration  
- Multi-agent systems
- Persistent terminal sessions with command persistence

Usage:
    from promptchain.tools.terminal import TerminalTool
    
    # Basic usage (non-persistent)
    terminal = TerminalTool(use_persistent_session=False)
    result = terminal("ls -la")
    
    # Persistent sessions (commands maintain state)
    terminal = TerminalTool(use_persistent_session=True, session_name="dev")
    terminal("export NODE_ENV=development")
    terminal("echo $NODE_ENV")  # Persists the environment variable
"""

from .terminal_tool import TerminalTool, create_terminal_tool, SecurityError
from .security import SecurityGuardrails
from .environment import EnvironmentManager
from .path_resolver import PathResolver
from .session_manager import SessionManager, PersistentTerminalSession

__version__ = "2.0.0"
__author__ = "PromptChain Team"

__all__ = [
    'TerminalTool',
    'create_terminal_tool', 
    'SecurityGuardrails',
    'EnvironmentManager',
    'PathResolver',
    'SessionManager',
    'PersistentTerminalSession',
    'SecurityError'
]