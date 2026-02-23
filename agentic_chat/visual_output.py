#!/usr/bin/env python3
"""
Visual Output System for Agentic Chat
Renders markdown beautifully with colors, tables, and Claude Code-style interface
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich import box
from rich.theme import Theme
import re
from typing import Optional

# Custom theme for better markdown rendering
custom_theme = Theme({
    "markdown.h1": "bold bright_cyan",
    "markdown.h1.border": "bright_cyan",
    "markdown.h2": "bold bright_yellow",
    "markdown.h3": "bold bright_green",
    "markdown.h4": "bold bright_magenta",
    "markdown.h5": "bold cyan",
    "markdown.h6": "bold yellow",
    "markdown.code": "bright_white on #1e1e1e",
    "markdown.code_block": "bright_white on #1e1e1e",
    "markdown.link": "bright_blue underline",
    "markdown.item.bullet": "bright_cyan",
})

console = Console(theme=custom_theme)

class ChatVisualizer:
    """Claude Code-inspired visual output system"""

    def __init__(self):
        self.console = Console(theme=custom_theme)

    def render_agent_response(self, agent_name: str, content: str, show_banner: bool = True):
        """Render agent response with rich markdown formatting"""

        if show_banner:
            # Agent banner with color coding
            agent_colors = {
                "research": "cyan",
                "analysis": "yellow",
                "coding": "green",
                "terminal": "magenta",
                "documentation": "blue",
                "synthesis": "red"
            }

            color = agent_colors.get(agent_name.lower(), "white")

            # Create agent header
            agent_emoji = {
                "research": "🔍",
                "analysis": "📊",
                "coding": "💡",
                "terminal": "💻",
                "documentation": "📝",
                "synthesis": "🎯"
            }

            emoji = agent_emoji.get(agent_name.lower(), "🤖")
            header = Text(f"{emoji} {agent_name.upper()}", style=f"bold {color}")

            self.console.print(Panel(header, box=box.ROUNDED, border_style=color))

        # Preprocess content for better bullet spacing
        # Add extra newline after bullet points for better readability
        lines = content.split('\n')
        improved_lines = []
        in_list = False

        for i, line in enumerate(lines):
            improved_lines.append(line)

            # Detect if we're in a bullet list
            is_bullet = line.strip().startswith(('- ', '• ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. '))

            # Add spacing after bullet items (but not if next line is also a bullet)
            if is_bullet and i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                next_is_bullet = next_line.startswith(('- ', '• ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. '))

                # Add extra spacing between top-level bullets
                if not next_is_bullet and not next_line.startswith(' '):
                    improved_lines.append('')

        improved_content = '\n'.join(improved_lines)

        # Render markdown with rich formatting and custom theme
        md = Markdown(improved_content)
        self.console.print(md)
        self.console.print()  # Add spacing

    def render_user_input(self, user_text: str):
        """Render user input with Claude Code style"""
        user_panel = Panel(
            Text(user_text, style="bold white"),
            title="💬 You",
            title_align="left",
            border_style="bright_blue",
            box=box.DOUBLE
        )
        self.console.print(user_panel)
        self.console.print()

    def get_input_with_box(self, prompt_text: str = "💬 You") -> str:
        """Get user input with Claude Code-style ASCII box"""

        # Create the input box visual
        self.console.print("─" * 80, style="bright_blue")

        # Get input
        user_input = Prompt.ask(
            f"[bold bright_blue]{prompt_text}[/bold bright_blue]",
            console=self.console
        )

        self.console.print("─" * 80, style="bright_blue")
        self.console.print()

        return user_input

    def render_system_message(self, message: str, message_type: str = "info"):
        """Render system messages with appropriate styling"""

        styles = {
            "info": ("ℹ️", "cyan"),
            "success": ("✅", "green"),
            "warning": ("⚠️", "yellow"),
            "error": ("❌", "red"),
            "thinking": ("🤔", "magenta")
        }

        emoji, color = styles.get(message_type, ("ℹ️", "white"))

        self.console.print(f"[{color}]{emoji} {message}[/{color}]")
        self.console.print()

    def render_header(self, title: str, subtitle: str = ""):
        """Render application header"""

        header_text = Text()
        header_text.append("🤖 ", style="bold cyan")
        header_text.append(title, style="bold white")

        if subtitle:
            header_text.append("\n")
            header_text.append(subtitle, style="dim white")

        panel = Panel(
            header_text,
            box=box.DOUBLE_EDGE,
            border_style="bright_cyan",
            padding=(1, 2)
        )

        self.console.print(panel)
        self.console.print()

    def render_team_roster(self, agents: dict):
        """Render team member roster in a table"""

        table = Table(
            title="🤝 Team Members",
            box=box.ROUNDED,
            title_style="bold cyan",
            border_style="cyan"
        )

        table.add_column("Agent", style="bold yellow", no_wrap=True)
        table.add_column("Role", style="white")
        table.add_column("Capabilities", style="dim cyan")

        agent_info = {
            "research": ("🔍 Research", "Web research specialist", "Gemini MCP, Google Search"),
            "analysis": ("📊 Analysis", "Data analysis expert", "Pattern recognition"),
            "coding": ("💡 Coding", "Script writer", "Python/Bash scripts"),
            "terminal": ("💻 Terminal", "Command executor", "Shell commands"),
            "documentation": ("📝 Documentation", "Technical writer", "Tutorials & guides"),
            "synthesis": ("🎯 Synthesis", "Strategic planner", "Insight integration")
        }

        for agent_key, (name, role, caps) in agent_info.items():
            if agent_key in agents:
                table.add_row(name, role, caps)

        self.console.print(table)
        self.console.print()

    def render_stats(self, stats: dict):
        """Render session statistics"""

        stats_table = Table(
            title="📊 Session Statistics",
            box=box.SIMPLE,
            show_header=False,
            border_style="dim cyan"
        )

        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")

        for key, value in stats.items():
            stats_table.add_row(key, str(value))

        self.console.print(stats_table)
        self.console.print()

    def render_commands_help(self):
        """Render available commands"""

        help_table = Table(
            title="💡 Available Commands",
            box=box.ROUNDED,
            border_style="yellow"
        )

        help_table.add_column("Command", style="bold green")
        help_table.add_column("Description", style="white")

        commands = [
            ("history", "Show conversation history"),
            ("stats", "Show session statistics"),
            ("save", "Save session summary"),
            ("clear", "Clear the screen"),
            ("help", "Show this help message"),
            ("exit / quit", "End the session")
        ]

        for cmd, desc in commands:
            help_table.add_row(cmd, desc)

        self.console.print(help_table)
        self.console.print()

    def clear_screen(self):
        """Clear the console"""
        self.console.clear()

    def render_thinking_indicator(self, agent_name: str):
        """Show thinking indicator while agent processes"""

        agent_emoji = {
            "research": "🔍",
            "analysis": "📊",
            "coding": "💡",
            "terminal": "💻",
            "documentation": "📝",
            "synthesis": "🎯"
        }

        emoji = agent_emoji.get(agent_name.lower(), "🤔")
        self.console.print(f"[dim cyan]{emoji} {agent_name} is thinking...[/dim cyan]")

    def render_tool_call(self, tool_name: str, tool_type: str = "unknown", args: str = ""):
        """Display tool call in a clean, visible way"""

        # Determine tool icon and color based on type
        if tool_type == "mcp" or "mcp__" in tool_name:
            icon = "🔌"
            color = "bright_magenta"
            type_label = "MCP"
        elif tool_name in ["write_script"]:
            icon = "📝"
            color = "bright_green"
            type_label = "Local"
        elif tool_name in ["execute_terminal_command"]:
            icon = "⚡"
            color = "bright_yellow"
            type_label = "Local"
        else:
            icon = "🔧"
            color = "cyan"
            type_label = "Tool"

        # Clean up tool name (remove mcp__ prefix for display)
        display_name = tool_name.replace("mcp__", "").replace("_", " ").title()

        # Format args if provided
        args_display = f": {args[:60]}..." if args and len(args) > 60 else f": {args}" if args else ""

        # Create compact tool call display
        tool_text = Text()
        tool_text.append(f"{icon} ", style=color)
        tool_text.append(f"[{type_label}] ", style=f"dim {color}")
        tool_text.append(display_name, style=f"bold {color}")
        tool_text.append(args_display, style=f"dim white")

        self.console.print(tool_text)


# Example usage and testing
if __name__ == "__main__":
    viz = ChatVisualizer()

    # Test header
    viz.render_header(
        "AGENTIC CHAT TEAM",
        "6-Agent Collaborative System with Multi-Hop Reasoning"
    )

    # Test team roster
    agents = {"research": {}, "analysis": {}, "coding": {}, "terminal": {}, "documentation": {}, "synthesis": {}}
    viz.render_team_roster(agents)

    # Test commands
    viz.render_commands_help()

    # Test user input
    viz.render_user_input("What is Rust's Candle library?")

    # Test thinking indicator
    viz.render_thinking_indicator("research")

    # Test agent response with markdown
    candle_response = """# Candle Library - Overview

Candle is a **minimalist machine learning framework** for Rust developed by Hugging Face.

## Key Features

1. **Pure Rust** - No Python dependencies
2. **Lightweight** - ~5-10 MB vs PyTorch's 500+ MB
3. **Fast** - Optimized for performance
4. **WASM Support** - Run ML in browsers

## Quick Example

```rust
use candle_core::{Tensor, Device};

let tensor = Tensor::new(&[[1f32, 2., 3.]], &Device::Cpu)?;
let sum = tensor.sum_all()?;
```

## Comparison Table

| Feature | Candle | PyTorch |
|---------|--------|---------|
| Language | Rust | Python |
| Size | 5-10 MB | 500+ MB |
| WASM | ✅ Yes | ❌ No |

## When to Use

- ✅ Production ML services in Rust
- ✅ Edge/IoT deployments
- ✅ Serverless functions
- ❌ Research/experimentation (use PyTorch)
"""

    viz.render_agent_response("research", candle_response)

    # Test system messages
    viz.render_system_message("Agent completed task successfully!", "success")
    viz.render_system_message("Warning: High token usage detected", "warning")

    # Test stats
    stats = {
        "Messages": 15,
        "Agents Used": "research, documentation",
        "Total Tokens": "~45,000",
        "Session Time": "5 minutes"
    }
    viz.render_stats(stats)

    # Test input box
    print("\n" + "="*80)
    print("Testing Claude Code-style input box:")
    # user_input = viz.get_input_with_box()
    # print(f"You entered: {user_input}")
