"""
Visual Terminal Formatter for PromptChain Terminal Tool

Provides terminal-like visual formatting for debugging LLM interactions
with terminal commands. Can be used as an optional feature.
"""

import time
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live


class VisualTerminalFormatter:
    """
    Formats terminal input/output to look like a real terminal.
    
    This is useful for debugging LLM interactions with the terminal tool,
    providing a visual representation of what commands are being executed
    and their outputs.
    """
    
    def __init__(self, style: str = "dark", max_history: int = 50):
        """
        Initialize the visual formatter.
        
        Args:
            style: Visual style ("dark" or "light")
            max_history: Maximum number of commands to keep in history
        """
        self.console = Console()
        self.history: List[Dict[str, Any]] = []
        self.style = style
        self.max_history = max_history
        
    def add_command(self, command: str, output: str, working_dir: str = "/tmp", 
                   user: str = "user", host: str = "system", error: bool = False):
        """
        Add a command and its output to the visual terminal history.
        
        Args:
            command: The command that was executed
            output: The output from the command
            working_dir: Current working directory
            user: Username for display
            host: Hostname for display
            error: Whether this was an error/failed command
        """
        entry = {
            "command": command,
            "output": output.rstrip() if output else "",
            "working_dir": working_dir,
            "user": user,
            "host": host,
            "error": error,
            "timestamp": time.time()
        }
        
        self.history.append(entry)
        
        # Keep only max_history entries
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def format_as_terminal(self, max_entries: int = 10, show_timestamps: bool = False) -> Panel:
        """
        Format recent commands as a terminal-like display.
        
        Args:
            max_entries: Maximum number of recent entries to show
            show_timestamps: Whether to show timestamps
            
        Returns:
            Rich Panel with terminal-like formatting
        """
        terminal_content = Text()
        
        # Get recent entries
        recent = self.history[-max_entries:] if self.history else []
        
        if not recent:
            terminal_content.append("No commands executed yet...", style="dim")
        
        for i, entry in enumerate(recent):
            # Add timestamp if requested
            if show_timestamps:
                timestamp = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                terminal_content.append(f"[{timestamp}] ", style="dim")
            
            # Create prompt line
            prompt = Text()
            
            # User@host
            user_style = "red bold" if entry['error'] else "green bold"
            prompt.append(f"{entry['user']}@{entry['host']}", style=user_style)
            prompt.append(":", style="white")
            
            # Working directory
            prompt.append(f"{entry['working_dir']}", style="blue bold")
            
            # Prompt symbol
            prompt_symbol = "$ " if not entry['error'] else "✗ "
            prompt.append(prompt_symbol, style="white bold")
            
            # Command
            cmd_style = "white" if not entry['error'] else "red"
            prompt.append(f"{entry['command']}\n", style=cmd_style)
            
            terminal_content.append(prompt)
            
            # Add output (if any)
            if entry['output'].strip():
                output_style = "white" if not entry['error'] else "red"
                output_text = Text(entry['output'] + "\n", style=output_style)
                terminal_content.append(output_text)
            
            # Add spacing between commands (except last one)
            if i < len(recent) - 1:
                terminal_content.append("\n")
        
        # Choose title based on style
        title_style = "cyan" if not any(e['error'] for e in recent) else "red"
        title = "🖥️  Terminal Session"
        
        return Panel(
            terminal_content,
            title=title,
            border_style=title_style,
            padding=(1, 2)
        )
    
    def print_terminal(self, max_entries: int = 10, show_timestamps: bool = False):
        """Print the terminal display to console."""
        self.console.print(self.format_as_terminal(max_entries, show_timestamps))
    
    def print_last_command(self):
        """Print only the last command and output."""
        if not self.history:
            return
            
        last = self.history[-1]
        
        # Create prompt
        prompt = Text()
        user_style = "red bold" if last['error'] else "green bold"
        prompt.append(f"{last['user']}@{last['host']}", style=user_style)
        prompt.append(":", style="white")
        prompt.append(f"{last['working_dir']}", style="blue bold")
        prompt_symbol = "$ " if not last['error'] else "✗ "
        prompt.append(prompt_symbol, style="white bold")
        cmd_style = "white" if not last['error'] else "red"
        prompt.append(f"{last['command']}", style=cmd_style)
        
        self.console.print(prompt)
        
        # Print output if any
        if last['output'].strip():
            output_style = "white" if not last['error'] else "red"
            self.console.print(last['output'], style=output_style)
    
    def get_live_display(self, max_entries: int = 10):
        """
        Get a Rich Live display for real-time updates.
        
        Usage:
            with formatter.get_live_display() as live:
                # Execute commands
                # Live display updates automatically
        """
        return Live(self.format_as_terminal(max_entries), refresh_per_second=2)
    
    def clear_history(self):
        """Clear the command history."""
        self.history.clear()
    
    def get_history_summary(self) -> Dict[str, Any]:
        """Get a summary of the command history."""
        if not self.history:
            return {"total_commands": 0, "errors": 0, "success": 0}
        
        total = len(self.history)
        errors = sum(1 for entry in self.history if entry['error'])
        success = total - errors
        
        return {
            "total_commands": total,
            "errors": errors,
            "success": success,
            "error_rate": errors / total if total > 0 else 0,
            "recent_commands": [entry['command'] for entry in self.history[-5:]]
        }


class LiveTerminalDisplay:
    """
    A live updating terminal display for real-time command monitoring.
    
    This is useful for watching LLM agents execute commands in real-time.
    """
    
    def __init__(self, formatter: VisualTerminalFormatter, max_entries: int = 10):
        self.formatter = formatter
        self.max_entries = max_entries
        self.live = None
        
    def __enter__(self):
        self.live = Live(
            self.formatter.format_as_terminal(self.max_entries),
            refresh_per_second=2
        )
        self.live.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self):
        """Update the live display with current formatter state."""
        if self.live:
            self.live.update(self.formatter.format_as_terminal(self.max_entries))


# Convenience functions for quick usage

def create_visual_formatter(style: str = "dark") -> VisualTerminalFormatter:
    """Create a visual terminal formatter with default settings."""
    return VisualTerminalFormatter(style=style)


def format_command_output(command: str, output: str, working_dir: str = "/tmp", 
                         error: bool = False) -> str:
    """
    Quickly format a single command/output as terminal-like text.
    
    Args:
        command: The command
        output: The output 
        working_dir: Working directory
        error: Whether this was an error
        
    Returns:
        Formatted string that looks like terminal output
    """
    user_host = "user@system"
    prompt_symbol = "$ " if not error else "✗ "
    
    result = f"{user_host}:{working_dir}{prompt_symbol}{command}\n"
    if output.strip():
        result += output.rstrip() + "\n"
    
    return result