"""Output formatting utilities for better visual organization.

Provides:
- Animated status indicators (spinning characters)
- Formatted message blocks with spacing and indentation
- Visual hierarchy for nested operations
"""

from typing import Optional


class AnimatedIndicator:
    """Animated status indicators for processing states.

    Uses rotating characters to show activity:
    - Squares: ◰ ◳ ◲ ◱
    - Circles: ◐ ◓ ◑ ◒
    - Dots: ⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏
    - Arrows: ← ↖ ↑ ↗ → ↘ ↓ ↙
    """

    SQUARES = ["◰", "◳", "◲", "◱"]
    CIRCLES = ["◐", "◓", "◑", "◒"]
    DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]

    def __init__(self, style: str = "squares"):
        """Initialize with animation style.

        Args:
            style: Animation style (squares, circles, dots, arrows)
        """
        self.style = style
        self.frames = {
            "squares": self.SQUARES,
            "circles": self.CIRCLES,
            "dots": self.DOTS,
            "arrows": self.ARROWS,
        }.get(style, self.SQUARES)
        self.frame_index = 0

    def next_frame(self) -> str:
        """Get next animation frame.

        Returns:
            Next character in animation sequence
        """
        frame = self.frames[self.frame_index]
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        return frame

    def format_message(self, message: str, color: str = "#666666") -> str:
        """Format message with current animation frame.

        Args:
            message: Message to display
            color: Color for the indicator

        Returns:
            Formatted string with indicator
        """
        indicator = self.next_frame()
        return f"[{color}]{indicator}[/{color}] {message}"


class OutputFormatter:
    """Format output with better spacing and visual hierarchy."""

    @staticmethod
    def format_user_message(content: str) -> str:
        """Format user message with spacing.

        Args:
            content: User message content

        Returns:
            Formatted message with newlines
        """
        return f"\n{content}\n"

    @staticmethod
    def format_assistant_message(
        content: str, agent_name: Optional[str] = None, model_name: Optional[str] = None
    ) -> str:
        """Format assistant message with agent info and spacing.

        Args:
            content: Assistant response content
            agent_name: Name of responding agent
            model_name: Model used for response

        Returns:
            Formatted message with header and spacing
        """
        header_parts = []
        if agent_name:
            header_parts.append(f"[bold green]{agent_name}[/bold green]")
        if model_name:
            header_parts.append(f"[dim]({model_name})[/dim]")

        if header_parts:
            header = " ".join(header_parts) + ":\n"
        else:
            header = ""

        return f"\n{header}{content}\n"

    @staticmethod
    def format_system_message(content: str, indent: int = 0) -> str:
        """Format system message with optional indentation.

        Args:
            content: System message content
            indent: Number of spaces to indent

        Returns:
            Formatted system message
        """
        indent_str = " " * indent
        lines = content.split("\n")
        indented_lines = [f"{indent_str}{line}" if line else "" for line in lines]
        return f"\n{''.join(indented_lines)}\n"

    @staticmethod
    def format_shell_output(
        stdout: str,
        stderr: str,
        return_code: int,
        execution_time: float,
        timed_out: bool = False,
        error_message: Optional[str] = None,
    ) -> str:
        """Format shell command output with visual hierarchy.

        Args:
            stdout: Standard output
            stderr: Standard error
            return_code: Exit code
            execution_time: Duration in seconds
            timed_out: Whether command timed out
            error_message: Optional error message

        Returns:
            Formatted shell output with indentation
        """
        output_parts = []

        # Add stdout with indentation
        if stdout:
            output_parts.append("\n[bold green]Output:[/bold green]")
            for line in stdout.rstrip().split("\n"):
                output_parts.append(f"  {line}")

        # Add stderr with indentation
        if stderr:
            output_parts.append("\n[bold red]Errors:[/bold red]")
            for line in stderr.rstrip().split("\n"):
                output_parts.append(f"  {line}")

        # Add execution status
        output_parts.append("")  # Blank line before status

        if timed_out:
            output_parts.append("[bold yellow]⚠ Command timed out[/bold yellow]")
            if error_message:
                output_parts.append(f"  {error_message}")

        status_color = "green" if return_code == 0 else "red"
        status_symbol = "✓" if return_code == 0 else "✗"
        status_line = (
            f"[{status_color}]{status_symbol} Exit code: {return_code} | "
            f"Duration: {execution_time:.2f}s[/{status_color}]"
        )
        output_parts.append(status_line)
        output_parts.append("")  # Blank line after

        return "\n".join(output_parts)

    @staticmethod
    def format_processing_status(
        message: str,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        indent: int = 2,
    ) -> str:
        """Format processing status with step information.

        Args:
            message: Status message
            step: Current step number
            total_steps: Total number of steps
            indent: Indentation level

        Returns:
            Formatted processing status
        """
        indent_str = " " * indent

        if step is not None and total_steps is not None:
            prefix = f"[dim][{step}/{total_steps}][/dim] "
        else:
            prefix = ""

        return f"{indent_str}{prefix}{message}"

    @staticmethod
    def format_section_header(title: str, color: str = "#666666") -> str:
        """Format section header with visual separator.

        Args:
            title: Section title
            color: Color for the separator

        Returns:
            Formatted section header
        """
        separator = "─" * 40
        return f"\n[{color}]{separator}[/{color}]\n[bold]{title}[/bold]\n[{color}]{separator}[/{color}]\n"

    @staticmethod
    def format_nested_item(content: str, level: int = 1, bullet: str = "•") -> str:
        """Format nested list item with indentation.

        Args:
            content: Item content
            level: Nesting level (1, 2, 3, etc.)
            bullet: Bullet character

        Returns:
            Formatted nested item
        """
        indent = "  " * (level - 1)
        return f"{indent}{bullet} {content}"
