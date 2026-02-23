"""TokenBar widget for displaying real-time token usage."""

from textual.reactive import reactive
from textual.widgets import Static


class TokenBar(Static):
    """Static toolbar showing real-time token usage.

    Displays API token consumption and context window usage in format:
    API: current_tokens / max_history_tokens

    Features:
    - Real-time updates during LLM calls
    - Shows both prompt and completion token breakdown
    - Visual indicator when approaching limits
    """

    # API token tracking (from LLM responses)
    api_prompt_tokens: reactive[int] = reactive(0)
    api_completion_tokens: reactive[int] = reactive(0)

    # Context/history tracking
    history_tokens: reactive[int] = reactive(0)
    max_history_tokens: reactive[int] = reactive(64000)

    DEFAULT_CSS = """
    TokenBar {
        dock: bottom;
        height: 1;
        background: #1a1a1a;
        color: #ffffff;
        text-align: center;
        padding: 0 1;
    }
    """

    def render(self) -> str:
        """Render the token bar with usage statistics."""
        api_total = self.api_prompt_tokens + self.api_completion_tokens

        # Calculate percentage for visual indicator
        if self.max_history_tokens > 0:
            usage_pct = (self.history_tokens / self.max_history_tokens) * 100
        else:
            usage_pct = 0

        # Build display parts
        parts = []

        # API tokens: total (prompt + completion) - WHITE for visibility
        if api_total > 0:
            parts.append(
                f"[white bold]API: {api_total}[/white bold] [white]({self.api_prompt_tokens}+{self.api_completion_tokens})[/white]"
            )
        else:
            parts.append("[white]API: 0[/white]")

        # Context tokens: current / max with visual indicator
        if usage_pct >= 90:
            # Critical - red/bold
            ctx_display = f"[red bold]{self.history_tokens}[/red bold][white]/{self.max_history_tokens}[/white]"
            indicator = "[red bold]![/red bold]"
        elif usage_pct >= 70:
            # Warning - yellow
            ctx_display = f"[yellow bold]{self.history_tokens}[/yellow bold][white]/{self.max_history_tokens}[/white]"
            indicator = "[yellow bold]![/yellow bold]"
        elif usage_pct >= 50:
            # Notice - white
            ctx_display = f"[white bold]{self.history_tokens}[/white bold][white]/{self.max_history_tokens}[/white]"
            indicator = "[white]o[/white]"
        else:
            # Normal - white
            ctx_display = f"[white]{self.history_tokens}/{self.max_history_tokens}[/white]"
            indicator = "[white].[/white]"

        parts.append(f"{indicator} [white]Context:[/white] {ctx_display}")

        # Add percentage - white
        parts.append(f"[white]({usage_pct:.0f}%)[/white]")

        return " | ".join(parts)

    def update_tokens(
        self,
        api_prompt_tokens: int = None,
        api_completion_tokens: int = None,
        history_tokens: int = None,
        max_history_tokens: int = None,
    ) -> None:
        """Update token values.

        Args:
            api_prompt_tokens: Cumulative prompt tokens from LLM API
            api_completion_tokens: Cumulative completion tokens from LLM API
            history_tokens: Current tokens in conversation history
            max_history_tokens: Maximum tokens before truncation
        """
        if api_prompt_tokens is not None:
            self.api_prompt_tokens = api_prompt_tokens
        if api_completion_tokens is not None:
            self.api_completion_tokens = api_completion_tokens
        if history_tokens is not None:
            self.history_tokens = history_tokens
        if max_history_tokens is not None:
            self.max_history_tokens = max_history_tokens
