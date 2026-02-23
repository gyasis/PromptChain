"""StatusBar widget for displaying session info."""

from typing import Optional, List, Dict

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Status bar showing session name, active agent, and other info.

    Features:
    - Session name display
    - Active agent name
    - Model name
    - Message count
    - Session state indicator
    - Router mode indicator (T039)
    - Agent switching indicator (T039)
    - MCP server status display (T069)
    - Token usage tracking with visual indicators (T079)
    """

    session_name: reactive[str] = reactive("default")
    active_agent: reactive[str] = reactive("default")
    model_name: reactive[str] = reactive("")
    message_count: reactive[int] = reactive(0)
    session_state: reactive[str] = reactive("Active")
    router_mode: reactive[bool] = reactive(False)  # T039: Router mode indicator
    last_agent_switch: reactive[str] = reactive("")  # T039: Track agent switches
    selected_agent: reactive[str] = reactive("")  # T041: Router-selected agent name
    mcp_servers: reactive[List[Dict[str, str]]] = reactive([])  # T069: MCP server status
    token_count: reactive[int] = reactive(0)  # T079: Current history token count
    max_tokens: reactive[int] = reactive(0)  # T079: Maximum history token limit
    api_prompt_tokens: reactive[int] = reactive(0)  # API: Cumulative prompt tokens from LLM
    api_completion_tokens: reactive[int] = reactive(0)  # API: Cumulative completion tokens from LLM
    workflow_objective: reactive[str] = reactive("")  # T092: Current workflow objective
    workflow_progress: reactive[float] = reactive(0.0)  # T092: Workflow progress percentage

    def render(self) -> str:
        """Render the status bar content (T039: Enhanced with router mode indicator)."""
        # Session state indicator (grayscale)
        if self.session_state == "Active":
            state_indicator = "[bold]*[/bold]"
        elif self.session_state == "Paused":
            state_indicator = "[dim]*[/dim]"
        else:
            state_indicator = "[italic]*[/italic]"

        # Build status line
        parts = [
            f"{state_indicator} Session: [bold]{self.session_name}[/bold]",
        ]

        # Agent display with router mode indicator (T039, T041)
        if self.router_mode:
            # Router mode: Show selected agent with router icon (T041)
            agent_name = self.selected_agent if self.selected_agent else self.active_agent
            agent_display = f"[bold]\u2699[/bold] Agent: {agent_name}"
            parts.append(agent_display)
        else:
            # Single-agent mode: Show agent normally
            parts.append(f"Agent: [bold]{self.active_agent}[/bold]")

        # Show agent switch indicator if recent switch occurred (T039)
        if self.last_agent_switch:
            parts.append(f"[dim]\u2192 {self.last_agent_switch}[/dim]")

        if self.model_name:
            parts.append(f"Model: [dim]{self.model_name}[/dim]")

        parts.append(f"Messages: {self.message_count}")

        # History token usage display (T079) - RED when at/over limit
        if self.max_tokens > 0:
            percentage = (self.token_count / self.max_tokens) * 100 if self.max_tokens > 0 else 0
            at_limit = self.token_count >= self.max_tokens

            # RED when at/over limit, otherwise grayscale indicator
            if at_limit:
                # At/over limit: RED token count
                token_display = f"[bold]![/bold] Ctx: [red bold]{self.token_count}[/red bold]/{self.max_tokens}"
            elif percentage < 60:
                token_display = f"[dim].[/dim] Ctx: {self.token_count}/{self.max_tokens}"
            elif percentage < 85:
                token_display = f"o Ctx: {self.token_count}/{self.max_tokens}"
            else:
                token_display = f"[bold]![/bold] Ctx: {self.token_count}/{self.max_tokens}"
            parts.append(token_display)

        # API token usage display (cumulative prompt + completion tokens from LLM)
        api_total = self.api_prompt_tokens + self.api_completion_tokens
        if api_total > 0:
            parts.append(f"API: {api_total} ({self.api_prompt_tokens}+{self.api_completion_tokens})")

        # MCP server status display (T069) - grayscale
        if self.mcp_servers:
            mcp_status_parts = []
            for server in self.mcp_servers:
                server_id = server.get("id", "unknown")
                state = server.get("state", "disconnected")

                # Grayscale indicators
                if state == "connected":
                    indicator = "[bold]+[/bold]"
                elif state == "error":
                    indicator = "[bold]x[/bold]"
                else:  # disconnected
                    indicator = "[dim]o[/dim]"

                mcp_status_parts.append(f"{indicator}{server_id}")

            if mcp_status_parts:
                parts.append(f"MCP: {' '.join(mcp_status_parts)}")

        # Workflow progress display (T092) - grayscale
        if self.workflow_objective:
            # Truncate long objectives for status bar
            max_obj_length = 40
            objective_display = self.workflow_objective
            if len(objective_display) > max_obj_length:
                objective_display = objective_display[:max_obj_length - 3] + "..."

            # Grayscale progress indicator
            if self.workflow_progress >= 100:
                style = "bold"
                icon = "+"
            elif self.workflow_progress >= 50:
                style = ""
                icon = "o"
            else:
                style = "dim"
                icon = "-"

            if style:
                workflow_display = f"[{style}]{icon}[/{style}] Workflow: {objective_display} [{style}]{self.workflow_progress:.0f}%[/{style}]"
            else:
                workflow_display = f"{icon} Workflow: {objective_display} {self.workflow_progress:.0f}%"
            parts.append(workflow_display)

        return " | ".join(parts)

    def update_session_info(
        self,
        session_name: Optional[str] = None,
        active_agent: Optional[str] = None,
        model_name: Optional[str] = None,
        message_count: Optional[int] = None,
        session_state: Optional[str] = None,
        router_mode: Optional[bool] = None,  # T039: Router mode indicator
        last_agent_switch: Optional[str] = None,  # T039: Agent switch tracking
        selected_agent: Optional[str] = None,  # T041: Router-selected agent
        mcp_servers: Optional[List[Dict[str, str]]] = None,  # T069: MCP server status
        token_count: Optional[int] = None,  # T079: Current history token count
        max_tokens: Optional[int] = None,  # T079: Maximum history token limit
        api_prompt_tokens: Optional[int] = None,  # API: Cumulative prompt tokens
        api_completion_tokens: Optional[int] = None,  # API: Cumulative completion tokens
        workflow_objective: Optional[str] = None,  # T092: Workflow objective
        workflow_progress: Optional[float] = None,  # T092: Workflow progress percentage
    ):
        """Update status bar information.

        Args:
            session_name: Session name
            active_agent: Active agent name
            model_name: Current model name
            message_count: Number of messages
            session_state: Session state (Active/Paused/Archived)
            router_mode: Whether router mode is active (T039)
            last_agent_switch: Name of agent that was switched to (T039)
            selected_agent: Router-selected agent name (T041)
            mcp_servers: List of MCP server status dicts with 'id' and 'state' keys (T069)
            token_count: Current token count in history (T079)
            max_tokens: Maximum token limit for history (T079)
            api_prompt_tokens: Cumulative prompt tokens from LLM API
            api_completion_tokens: Cumulative completion tokens from LLM API
            workflow_objective: Current workflow objective (T092)
            workflow_progress: Current workflow progress percentage 0-100 (T092)
        """
        if session_name is not None:
            self.session_name = session_name
        if active_agent is not None:
            self.active_agent = active_agent
        if model_name is not None:
            self.model_name = model_name
        if message_count is not None:
            self.message_count = message_count
        if session_state is not None:
            self.session_state = session_state
        if router_mode is not None:
            self.router_mode = router_mode
        if last_agent_switch is not None:
            self.last_agent_switch = last_agent_switch
        if selected_agent is not None:
            self.selected_agent = selected_agent
        if mcp_servers is not None:
            self.mcp_servers = mcp_servers
        if token_count is not None:
            self.token_count = token_count
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_prompt_tokens is not None:
            self.api_prompt_tokens = api_prompt_tokens
        if api_completion_tokens is not None:
            self.api_completion_tokens = api_completion_tokens
        if workflow_objective is not None:
            self.workflow_objective = workflow_objective
        if workflow_progress is not None:
            self.workflow_progress = workflow_progress
