"""Command handler for PromptChain CLI slash commands.

This module handles parsing, validation, and routing of slash commands
to appropriate handler methods.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParsedCommand:
    """Represents a parsed slash command.

    Attributes:
        name: Command name (e.g., "agent", "session")
        subcommand: Subcommand name (e.g., "create", "list")
        args: Command arguments as key-value pairs
    """

    name: str
    subcommand: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.args is None:
            self.args = {}


@dataclass
class CommandResult:
    """Result of command execution.

    Attributes:
        success: Whether command succeeded
        message: User-friendly message
        data: Optional result data
        error: Optional error message
    """

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __str__(self) -> str:
        """User-friendly string representation."""
        if self.success:
            return self.message
        else:
            return f"Error: {self.error}"


# Command registry for autocomplete with descriptions
COMMAND_REGISTRY: Dict[str, Dict[str, str]] = {
    # Basic commands
    "/exit": {"description": "Exit the CLI", "usage": "/exit"},
    "/help": {"description": "Show help information", "usage": "/help"},
    "/clear": {"description": "Clear the chat display", "usage": "/clear"},

    # Session commands
    "/session": {"description": "Show current session info", "usage": "/session"},
    "/session list": {"description": "List all saved sessions", "usage": "/session list"},
    "/session save": {"description": "Save current session", "usage": "/session save [name]"},
    "/session delete": {"description": "Delete a saved session", "usage": "/session delete <name>"},

    # Agent commands
    "/agent": {"description": "Show current agent info", "usage": "/agent"},
    "/agent list": {"description": "List all agents in session", "usage": "/agent list"},
    "/agent create": {"description": "Create a new agent", "usage": "/agent create <name> --model=<model>"},
    "/agent delete": {"description": "Delete an agent", "usage": "/agent delete <name>"},
    "/agent use": {"description": "Switch to an agent", "usage": "/agent use <name>"},
    "/agent update": {"description": "Update agent settings", "usage": "/agent update <name> --model=<model>"},
    "/agent templates": {"description": "List agent templates", "usage": "/agent templates"},
    "/agent from-template": {"description": "Create agent from template", "usage": "/agent from-template <template>"},

    # Tools commands
    "/tools": {"description": "List available tools", "usage": "/tools"},
    "/tools list": {"description": "List all registered tools", "usage": "/tools list"},
    "/tools add": {"description": "Add MCP server tools", "usage": "/tools add <server_id>"},
    "/tools remove": {"description": "Remove MCP server tools", "usage": "/tools remove <server_id>"},
    "/capabilities": {"description": "List tool capabilities", "usage": "/capabilities [agent_name]"},

    # Log commands
    "/log": {"description": "Search activity logs", "usage": "/log <query>"},
    "/log search": {"description": "Search logs by keyword", "usage": "/log search <query>"},
    "/log agent": {"description": "Show agent activity log", "usage": "/log agent <name>"},
    "/log errors": {"description": "Show recent errors", "usage": "/log errors [limit]"},
    "/log stats": {"description": "Show session statistics", "usage": "/log stats"},
    "/log chain": {"description": "Show chain execution log", "usage": "/log chain <chain_id>"},

    # Workflow commands (T067)
    "/workflow": {"description": "Show workflow status", "usage": "/workflow [show|stage|tasks]"},
    "/workflow show": {"description": "Display current workflow status", "usage": "/workflow show"},
    "/workflow stage": {"description": "Show current stage", "usage": "/workflow stage"},
    "/workflow tasks": {"description": "Show completed task count", "usage": "/workflow tasks"},

    # Config commands
    "/config": {"description": "Show session config", "usage": "/config"},
    "/config show": {"description": "Display all settings", "usage": "/config show"},
    "/config export": {"description": "Export config to file", "usage": "/config export <filename>"},
    "/config import": {"description": "Import config from file", "usage": "/config import <filename>"},

    # History commands
    "/history": {"description": "Show conversation history", "usage": "/history"},
    "/history stats": {"description": "Show history statistics", "usage": "/history stats"},

    # Security commands
    "/security": {"description": "Show/set security mode", "usage": "/security [strict|trusted|default]"},

    # Task commands
    "/tasks": {"description": "List pending/in-progress tasks", "usage": "/tasks [agent_name]"},

    # Blackboard commands
    "/blackboard": {"description": "List or show blackboard entries", "usage": "/blackboard [key]"},

    # Mental Model commands
    "/mentalmodel": {"description": "Show agent's mental model", "usage": "/mentalmodel"},

    # MCP server commands
    "/mcp": {"description": "List MCP server status", "usage": "/mcp"},
    "/mcp list": {"description": "List all MCP servers", "usage": "/mcp list"},
    "/mcp add": {"description": "Add new MCP server", "usage": "/mcp add <id> --type=stdio --command=<cmd>"},
    "/mcp connect": {"description": "Connect to MCP server", "usage": "/mcp connect <server_id>"},
    "/mcp disconnect": {"description": "Disconnect MCP server", "usage": "/mcp disconnect <server_id>"},
    "/mcp remove": {"description": "Remove MCP server", "usage": "/mcp remove <server_id>"},

    # Pattern commands (004a - TUI Pattern Integration)
    "/branch": {"description": "Generate branching hypotheses", "usage": '/branch "query" [--count=N] [--mode=local|global|hybrid]'},
    "/expand": {"description": "Expand query variations", "usage": '/expand "query" [--strategies=semantic,synonym] [--max=N]'},
    "/multihop": {"description": "Multi-hop retrieval", "usage": '/multihop "query" [--max-hops=N] [--mode=hybrid]'},
    "/hybrid": {"description": "Hybrid search fusion", "usage": '/hybrid "query" [--fusion=rrf|linear|borda] [--top-k=N]'},
    "/sharded": {"description": "Sharded retrieval", "usage": '/sharded "query" --shards=shard1,shard2 [--aggregation=rrf]'},
    "/speculate": {"description": "Speculative execution", "usage": '/speculate "context" [--min-confidence=0.7] [--prefetch=N]'},
    "/patterns": {"description": "Show pattern commands help", "usage": "/patterns"},
}


def get_command_suggestions(prefix: str) -> List[Dict[str, str]]:
    """Get command suggestions matching a prefix.

    Args:
        prefix: The prefix to match (e.g., "/a" or "/agent c")

    Returns:
        List of matching commands with descriptions, sorted by command name
    """
    prefix = prefix.lower()
    matches = []

    for cmd, info in COMMAND_REGISTRY.items():
        if cmd.lower().startswith(prefix):
            matches.append({
                "command": cmd,
                "description": info["description"],
                "usage": info["usage"]
            })

    # Sort by command name
    return sorted(matches, key=lambda x: x["command"])


class CommandHandler:
    """Handles slash command parsing and execution.

    Responsibilities:
    - Parse command syntax (/command subcommand args)
    - Validate command arguments
    - Route to appropriate handler methods
    - Return structured results or errors
    """

    def __init__(self, session_manager):
        """Initialize command handler.

        Args:
            session_manager: SessionManager instance for session operations
        """
        self.session_manager = session_manager

    def parse_command(self, command_text: str) -> Optional[ParsedCommand]:
        """Parse a slash command into structured format.

        Args:
            command_text: Command string (e.g., "/exit", "/agent create")

        Returns:
            ParsedCommand object or None if invalid

        Examples:
            >>> handler.parse_command("/exit")
            ParsedCommand(name="exit", subcommand=None, args={})
            >>> handler.parse_command("/agent create myagent")
            ParsedCommand(name="agent", subcommand="create", args={"name": "myagent"})
        """
        if not command_text or not command_text.startswith("/"):
            return None

        # Remove leading slash and split by whitespace
        parts = command_text[1:].split()

        if not parts:
            return None

        # First part is always the command name
        name = parts[0]

        # Check if there's a subcommand
        subcommand = None
        args = {}

        if len(parts) > 1:
            # Second part could be subcommand or argument
            # For now, treat as subcommand if it doesn't contain "="
            if "=" not in parts[1]:
                subcommand = parts[1]
                arg_parts = parts[2:]
            else:
                arg_parts = parts[1:]

            # Parse remaining parts as key=value arguments
            for part in arg_parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    # Remove leading dashes from key (--model becomes model)
                    key = key.lstrip("-")
                    args[key] = value

        return ParsedCommand(name=name, subcommand=subcommand, args=args)

    async def handle_exit(self, session) -> CommandResult:
        """Handle /exit command - save session and trigger shutdown.

        Args:
            session: Current Session object

        Returns:
            CommandResult indicating success with goodbye message
        """
        try:
            # Save session before exiting
            self.session_manager.save_session(session)

            return CommandResult(
                success=True,
                message=f"Session '{session.name}' saved. Goodbye!",
                data={"session_id": session.id},
            )

        except Exception as e:
            return CommandResult(
                success=False, message="Failed to save session on exit", error=str(e)
            )

    # === Agent Management Commands (T051-T056, T099-T100) ===

    def handle_agent_create_from_template(
        self, session, template_name: str, agent_name: str,
        model_override: Optional[str] = None,
        description_override: Optional[str] = None
    ) -> CommandResult:
        """Handle /agent create-from-template command - create agent from template (T099).

        Args:
            session: Current Session object
            template_name: Template name (researcher, coder, analyst, terminal)
            agent_name: Name for the new agent
            model_override: Optional custom model to override template default
            description_override: Optional custom description to override template default

        Returns:
            CommandResult indicating success or error
        """
        from .utils.agent_templates import create_from_template, AGENT_TEMPLATES

        try:
            # Validate template_name
            if template_name not in AGENT_TEMPLATES:
                available = ", ".join(AGENT_TEMPLATES.keys())
                return CommandResult(
                    success=False,
                    message=f"Template '{template_name}' not found. Available templates: {available}",
                    error=f"Invalid template: {template_name}"
                )

            # Check if agent name already exists
            if agent_name in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Agent '{agent_name}' already exists in this session",
                    error=f"Duplicate agent name: {agent_name}"
                )

            # Create agent from template
            agent = create_from_template(
                template_name=template_name,
                agent_name=agent_name,
                model_override=model_override,
                description_override=description_override
            )

            # Add to session
            session.agents[agent_name] = agent

            # Persist to database
            self.session_manager.save_session(session)

            # Get template info for response
            template = AGENT_TEMPLATES[template_name]

            return CommandResult(
                success=True,
                message=f"Created '{agent_name}' from '{template.display_name}' template\n"
                        f"Model: {agent.model_name}\n"
                        f"Description: {agent.description}\n"
                        f"Tools: {', '.join(agent.tools) if agent.tools else 'none'}\n"
                        f"History: {'enabled' if agent.history_config.enabled else 'disabled'}",
                data={
                    "agent_name": agent_name,
                    "template": template_name,
                    "model": agent.model_name,
                    "tools_count": len(agent.tools),
                    "history_enabled": agent.history_config.enabled
                }
            )

        except (ValueError, AssertionError) as e:
            return CommandResult(
                success=False,
                message=f"Failed to create agent from template: {str(e)}",
                error=str(e)
            )

    def handle_agent_list_templates(self) -> CommandResult:
        """Handle /agent list-templates command - list available agent templates (T100).

        Returns:
            CommandResult with formatted template list showing capabilities and use cases
        """
        from .utils.agent_templates import AGENT_TEMPLATES

        try:
            # Format template list
            lines = [f"Available Agent Templates ({len(AGENT_TEMPLATES)} total):", ""]

            for template_name, template in AGENT_TEMPLATES.items():
                # Get category and complexity from metadata
                category = template.metadata.get("category", "general")
                complexity = template.metadata.get("complexity", "medium")
                token_usage = template.metadata.get("token_usage", "moderate")

                # Get history configuration details
                history_status = "enabled" if template.history_config.enabled else "disabled"
                if template.history_config.enabled:
                    history_details = f"{template.history_config.max_tokens} tokens"
                else:
                    history_details = "no history (max token efficiency)"

                # Format template section
                lines.append(f"  {template.display_name} ({template_name})")
                lines.append(f"    Description: {template.description}")
                lines.append(f"    Model: {template.model}")
                lines.append(f"    Category: {category}")
                lines.append(f"    Complexity: {complexity}")
                lines.append(f"    Token Usage: {token_usage}")
                lines.append(f"    Tools: {', '.join(template.tools) if template.tools else 'none'}")
                lines.append(f"    History: {history_status} ({history_details})")
                lines.append("")

            # Add usage instructions
            lines.append("Usage:")
            lines.append("  /agent create-from-template <template_name> <agent_name>")
            lines.append("  /agent create-from-template <template_name> <agent_name> --model <custom_model>")
            lines.append("")
            lines.append("Example:")
            lines.append("  /agent create-from-template researcher my-researcher")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "templates": list(AGENT_TEMPLATES.keys()),
                    "count": len(AGENT_TEMPLATES)
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list templates: {str(e)}",
                error=str(e)
            )

    def handle_agent_create(
        self, session, name: str, model: str, description: Optional[str] = None
    ) -> CommandResult:
        """Handle /agent create command - create new agent (T051).

        Args:
            session: Current Session object
            name: Agent name (alphanumeric + dashes/underscores)
            model: LiteLLM model string (provider/model-name)
            description: Optional agent description

        Returns:
            CommandResult indicating success or validation error
        """
        import time

        from .models.agent_config import Agent

        try:
            # Check if agent already exists
            if name in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Agent '{name}' already exists",
                    error=f"Duplicate agent name: {name}",
                )

            # Create agent (will validate name and model format)
            agent = Agent(
                name=name,
                model_name=model,
                description=description or f"{name} agent",
                created_at=time.time(),
            )

            # Add to session
            session.agents[name] = agent

            # Persist to SQLite (will be implemented in T053)
            self.session_manager.save_session(session)

            return CommandResult(
                success=True,
                message=f"Created agent '{name}' with model {model}",
                data={"agent_name": name, "model": model},
            )

        except (ValueError, AssertionError) as e:
            return CommandResult(
                success=False, message=f"Failed to create agent: {str(e)}", error=str(e)
            )

    def handle_agent_list(self, session) -> CommandResult:
        """Handle /agent list command - list all agents (T054).

        Args:
            session: Current Session object

        Returns:
            CommandResult with formatted agent list
        """
        if not session.agents:
            return CommandResult(
                success=True,
                message="No agents created yet. Use /agent create to add one.",
                data={"agents": []},
            )

        # Format agent list
        lines = ["Available agents:"]
        for name, agent in session.agents.items():
            # Mark active agent
            marker = " (active)" if name == session.active_agent else ""

            # Format usage stats
            usage = f"used {agent.usage_count} times" if agent.usage_count > 0 else "never used"

            lines.append(f"  - {name}{marker}: {agent.model_name} - {agent.description} ({usage})")

        message = "\n".join(lines)

        return CommandResult(
            success=True,
            message=message,
            data={
                "agents": list(session.agents.keys()),
                "active": session.active_agent,
            },
        )

    def handle_agent_delete(self, session, name: str) -> CommandResult:
        """Handle /agent delete command - delete agent (T055).

        Args:
            session: Current Session object
            name: Agent name to delete

        Returns:
            CommandResult indicating success or error
        """
        try:
            # Check if agent exists
            if name not in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Agent '{name}' not found",
                    error=f"Agent not found: {name}",
                )

            # Prevent deleting active agent
            if name == session.active_agent:
                return CommandResult(
                    success=False,
                    message=f"Cannot delete active agent '{name}'. Switch to another agent first.",
                    error=f"Cannot delete active agent: {name}",
                )

            # Remove from session
            del session.agents[name]

            # Persist to SQLite (will remove from agents table in T056)
            self.session_manager.save_session(session)

            return CommandResult(
                success=True,
                message=f"Deleted agent '{name}'",
                data={"agent_name": name},
            )

        except Exception as e:
            return CommandResult(
                success=False, message=f"Failed to delete agent: {str(e)}", error=str(e)
            )

    def handle_agent_update(
        self,
        session,
        name: str,
        model: Optional[str] = None,
        description: Optional[str] = None,
        add_tools: Optional[List[str]] = None,
        remove_tools: Optional[List[str]] = None
    ) -> CommandResult:
        """Handle /agent update command - modify agent properties (T102).

        Args:
            session: Current Session object
            name: Agent name to update
            model: Optional new model name
            description: Optional new description
            add_tools: Optional list of tools to add
            remove_tools: Optional list of tools to remove

        Returns:
            CommandResult indicating success or error

        Example:
            /agent update my-researcher --model claude-3-opus-20240229
            /agent update my-coder --description "Python specialist"
            /agent update my-terminal --add-tools web_search,mcp_filesystem_read
        """
        try:
            # Check if agent exists
            if name not in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Agent '{name}' not found. Use /agent list to see available agents.",
                    error=f"Agent not found: {name}"
                )

            agent = session.agents[name]
            changes = []

            # Update model
            if model is not None:
                old_model = agent.model_name
                agent.model_name = model
                changes.append(f"Model: {old_model} → {model}")

            # Update description
            if description is not None:
                old_desc = agent.description
                agent.description = description
                changes.append(f"Description updated")

            # Add tools
            if add_tools:
                for tool in add_tools:
                    if tool not in agent.tools:
                        agent.tools.append(tool)
                        changes.append(f"Added tool: {tool}")
                    else:
                        changes.append(f"Tool already exists: {tool} (skipped)")

            # Remove tools
            if remove_tools:
                for tool in remove_tools:
                    if tool in agent.tools:
                        agent.tools.remove(tool)
                        changes.append(f"Removed tool: {tool}")
                    else:
                        changes.append(f"Tool not found: {tool} (skipped)")

            # Check if any changes were made
            if not changes:
                return CommandResult(
                    success=False,
                    message=f"No updates specified for agent '{name}'. Use --model, --description, --add-tools, or --remove-tools.",
                    error="No updates provided"
                )

            # Persist changes
            self.session_manager.save_session(session)

            # Format response
            message = f"Updated agent '{name}':\n" + "\n".join(f"  • {change}" for change in changes)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "agent_name": name,
                    "changes": changes,
                    "model": agent.model_name,
                    "tools": agent.tools
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to update agent: {str(e)}",
                error=str(e)
            )

    def handle_agent_use(self, session, name: str) -> CommandResult:
        """Handle /agent use command - switch active agent (T057).

        Args:
            session: Current Session object
            name: Agent name to switch to

        Returns:
            CommandResult indicating success or error
        """
        import time

        try:
            # Check if agent exists
            if name not in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Agent '{name}' not found. Use /agent list to see available agents.",
                    error=f"Agent not found: {name}",
                )

            # Update usage statistics (T060)
            agent = session.agents[name]
            agent.usage_count += 1
            agent.last_used = time.time()

            # Switch active agent
            session.active_agent = name

            # Persist changes
            self.session_manager.save_session(session)

            return CommandResult(
                success=True,
                message=f"Switched to agent '{name}' ({agent.model_name})",
                data={"agent_name": name, "model": agent.model_name},
            )

        except Exception as e:
            return CommandResult(
                success=False, message=f"Failed to switch agent: {str(e)}", error=str(e)
            )

    def handle_session_list(self, current_session_id: Optional[str] = None) -> CommandResult:
        """Handle /session list command - list all saved sessions (T074).

        Args:
            current_session_id: ID of currently active session (optional)

        Returns:
            CommandResult with list of sessions
        """
        import sqlite3

        try:
            conn = sqlite3.connect(self.session_manager.db_path)
            cursor = conn.execute(
                """
                SELECT id, name, created_at, last_accessed, active_agent
                FROM sessions
                ORDER BY last_accessed DESC
                """
            )
            rows = cursor.fetchall()

            if not rows:
                return CommandResult(
                    success=True,
                    message="No sessions found. Create a new session to get started.",
                    data={"sessions": []},
                )

            # Build session list with metadata
            sessions = []
            for row in rows:
                session_id, name, created_at, last_accessed, active_agent = row

                # Count agents for this session
                agent_cursor = conn.execute(
                    "SELECT COUNT(*) FROM agents WHERE session_id = ?", (session_id,)
                )
                agent_count = agent_cursor.fetchone()[0]

                # Count messages (from JSONL file)
                messages_file = self.session_manager.sessions_dir / session_id / "messages.jsonl"
                message_count = 0
                if messages_file.exists():
                    with open(messages_file, "r") as f:
                        message_count = sum(1 for line in f if line.strip())

                # Format last_accessed as human-readable
                import datetime

                last_accessed_dt = datetime.datetime.fromtimestamp(last_accessed)
                time_ago = self._format_time_ago(last_accessed)

                session_info = {
                    "id": session_id,
                    "name": name,
                    "last_accessed": last_accessed,
                    "last_accessed_formatted": last_accessed_dt.strftime("%Y-%m-%d %H:%M"),
                    "time_ago": time_ago,
                    "agent_count": agent_count,
                    "message_count": message_count,
                    "active_agent": active_agent,
                    "is_current": session_id == current_session_id,
                }
                sessions.append(session_info)

            conn.close()

            # Format message
            lines = ["Available sessions (most recent first):"]
            for s in sessions:
                current_marker = " (current)" if s["is_current"] else ""
                lines.append(
                    f"  - {s['name']}{current_marker}: "
                    f"{s['message_count']} messages, "
                    f"{s['agent_count']} agents, "
                    f"last accessed {s['time_ago']}"
                )

            return CommandResult(
                success=True, message="\n".join(lines), data={"sessions": sessions}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list sessions: {str(e)}",
                error=str(e),
            )

    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable time ago.

        Args:
            timestamp: Unix timestamp

        Returns:
            str: Human-readable time ago (e.g., "2 hours ago")
        """
        import datetime

        now = datetime.datetime.now().timestamp()
        diff = now - timestamp

        if diff < 60:
            return "just now"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff < 604800:
            days = int(diff / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            weeks = int(diff / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"

    def handle_session_delete(
        self, session_id: str, is_active: bool = False, confirmed: bool = False
    ) -> CommandResult:
        """Handle /session delete command - delete a session (T082).

        Args:
            session_id: ID of session to delete
            is_active: Whether this is the currently active session
            confirmed: Whether deletion is confirmed

        Returns:
            CommandResult with deletion status
        """
        import shutil
        import sqlite3

        try:
            # Check if session is active
            if is_active:
                return CommandResult(
                    success=False,
                    message="Cannot delete active session. Please switch to another session or close this session first.",
                    error="Cannot delete active session",
                )

            # Check if confirmation is needed
            if not confirmed:
                return CommandResult(
                    success=False,
                    message=f"Please confirm deletion of session '{session_id}'. This action cannot be undone.",
                    error="Confirmation required",
                )

            # Check if session exists
            conn = sqlite3.connect(self.session_manager.db_path)
            try:
                # Enable foreign keys for cascade delete
                conn.execute("PRAGMA foreign_keys = ON")

                cursor = conn.execute("SELECT name FROM sessions WHERE id = ?", (session_id,))
                row = cursor.fetchone()

                if not row:
                    return CommandResult(
                        success=False,
                        message=f"Session '{session_id}' not found",
                        error="Session not found",
                    )

                session_name = row[0]

                # Delete from database (cascade will delete agents)
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                conn.commit()

            finally:
                conn.close()

            # Delete session directory and files
            session_dir = self.session_manager.sessions_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)

            return CommandResult(
                success=True,
                message=f"Session '{session_name}' deleted successfully",
                data={"session_id": session_id, "session_name": session_name},
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to delete session: {str(e)}",
                error=str(e),
            )

    # === Activity Log Commands (Phase 4) ===

    def handle_log_search(
        self, session, pattern: str, agent: Optional[str] = None, type: Optional[str] = None, limit: int = 10
    ) -> CommandResult:
        """Handle /log search command - search activity logs (Phase 4).

        Args:
            session: Current Session object
            pattern: Search pattern (regex)
            agent: Optional agent name filter
            type: Optional activity type filter
            limit: Maximum results to return (default: 10)

        Returns:
            CommandResult with search results
        """
        try:
            from .activity_searcher import ActivitySearcher

            # Check if ActivityLogger exists
            if session.activity_logger is None:
                return CommandResult(
                    success=False,
                    message="Activity logging not enabled for this session",
                    error="ActivityLogger not initialized",
                )

            # Get log directory and database path
            session_dir = self.session_manager.sessions_dir / session.id
            log_dir = session_dir / "activity_logs"
            db_path = session_dir / "activities.db"

            # Create searcher
            searcher = ActivitySearcher(
                session_name=session.name,
                log_dir=log_dir,
                db_path=db_path
            )

            # Search activities
            results = searcher.grep_logs(
                pattern=pattern,
                agent_name=agent,
                activity_type=type,
                max_results=limit
            )

            if not results:
                return CommandResult(
                    success=True,
                    message=f"No activities found matching pattern '{pattern}'",
                    data={"results": [], "count": 0}
                )

            # Format results
            lines = [f"Found {len(results)} activities matching '{pattern}':"]
            for i, activity in enumerate(results, 1):
                timestamp = activity.get('timestamp', 'unknown')
                agent_name = activity.get('agent_name', 'system')
                activity_type = activity.get('activity_type', 'unknown')

                # Get preview of content
                content = activity.get('content', {})
                if isinstance(content, dict):
                    preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                else:
                    preview = str(content)[:100] + "..." if len(str(content)) > 100 else str(content)

                lines.append(f"  {i}. [{timestamp}] {activity_type} - {agent_name}")
                lines.append(f"     {preview}")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={"results": results, "count": len(results)}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to search activity logs: {str(e)}",
                error=str(e)
            )

    def handle_log_agent(self, session, agent_name: str, limit: int = 20) -> CommandResult:
        """Handle /log agent command - get agent-specific activities (Phase 4).

        Args:
            session: Current Session object
            agent_name: Agent name to filter by
            limit: Maximum results to return (default: 20)

        Returns:
            CommandResult with agent activities
        """
        try:
            from .activity_searcher import ActivitySearcher

            if session.activity_logger is None:
                return CommandResult(
                    success=False,
                    message="Activity logging not enabled for this session",
                    error="ActivityLogger not initialized",
                )

            session_dir = self.session_manager.sessions_dir / session.id
            searcher = ActivitySearcher(
                session_name=session.name,
                log_dir=session_dir / "activity_logs",
                db_path=session_dir / "activities.db"
            )

            # Get agent activities
            results = searcher.grep_logs(
                pattern=".*",
                agent_name=agent_name,
                max_results=limit
            )

            if not results:
                return CommandResult(
                    success=True,
                    message=f"No activities found for agent '{agent_name}'",
                    data={"results": [], "count": 0, "agent": agent_name}
                )

            # Format results
            lines = [f"Activities for agent '{agent_name}' ({len(results)} total):"]
            for i, activity in enumerate(results, 1):
                timestamp = activity.get('timestamp', 'unknown')
                activity_type = activity.get('activity_type', 'unknown')

                content = activity.get('content', {})
                if isinstance(content, dict):
                    preview = str(content)[:80] + "..." if len(str(content)) > 80 else str(content)
                else:
                    preview = str(content)[:80] + "..." if len(str(content)) > 80 else str(content)

                lines.append(f"  {i}. [{timestamp}] {activity_type}: {preview}")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={"results": results, "count": len(results), "agent": agent_name}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get agent activities: {str(e)}",
                error=str(e)
            )

    def handle_log_errors(self, session, limit: int = 10) -> CommandResult:
        """Handle /log errors command - get recent errors (Phase 4).

        Args:
            session: Current Session object
            limit: Maximum errors to return (default: 10)

        Returns:
            CommandResult with recent errors
        """
        try:
            from .activity_searcher import ActivitySearcher

            if session.activity_logger is None:
                return CommandResult(
                    success=False,
                    message="Activity logging not enabled for this session",
                    error="ActivityLogger not initialized",
                )

            session_dir = self.session_manager.sessions_dir / session.id
            searcher = ActivitySearcher(
                session_name=session.name,
                log_dir=session_dir / "activity_logs",
                db_path=session_dir / "activities.db"
            )

            # Get error activities
            results = searcher.grep_logs(
                pattern="error",
                activity_type="error",
                max_results=limit
            )

            if not results:
                return CommandResult(
                    success=True,
                    message="No errors found in activity logs 🎉",
                    data={"results": [], "count": 0}
                )

            # Format results
            lines = [f"Found {len(results)} errors:"]
            for i, activity in enumerate(results, 1):
                timestamp = activity.get('timestamp', 'unknown')
                agent_name = activity.get('agent_name', 'unknown')
                content = activity.get('content', {})

                error_msg = content.get('error', 'Unknown error') if isinstance(content, dict) else str(content)
                error_type = content.get('error_type', 'Exception') if isinstance(content, dict) else 'Exception'

                lines.append(f"  {i}. [{timestamp}] {agent_name}")
                lines.append(f"     {error_type}: {error_msg}")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={"results": results, "count": len(results)}
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get error logs: {str(e)}",
                error=str(e)
            )

    def handle_log_stats(self, session) -> CommandResult:
        """Handle /log stats command - get activity statistics (Phase 4).

        Args:
            session: Current Session object

        Returns:
            CommandResult with statistics
        """
        try:
            from .activity_searcher import ActivitySearcher

            if session.activity_logger is None:
                return CommandResult(
                    success=False,
                    message="Activity logging not enabled for this session",
                    error="ActivityLogger not initialized",
                )

            session_dir = self.session_manager.sessions_dir / session.id
            searcher = ActivitySearcher(
                session_name=session.name,
                log_dir=session_dir / "activity_logs",
                db_path=session_dir / "activities.db"
            )

            # Get statistics
            stats = searcher.get_statistics()

            # Format statistics
            lines = [
                f"Activity Log Statistics for '{session.name}':",
                f"",
                f"Total Activities: {stats['total_activities']}",
                f"Total Chains: {stats['total_chains']}",
                f"Active Chains: {stats['active_chains']}",
                f"Average Chain Depth: {stats['avg_chain_depth']:.1f}",
                f"Total Errors: {stats['total_errors']}",
                f"",
                f"Activities by Type:",
            ]

            for activity_type, count in stats['activities_by_type'].items():
                lines.append(f"  - {activity_type}: {count}")

            lines.append(f"")
            lines.append(f"Activities by Agent:")

            for agent_name, count in stats['activities_by_agent'].items():
                lines.append(f"  - {agent_name or 'system'}: {count}")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data=stats
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get statistics: {str(e)}",
                error=str(e)
            )

    def handle_log_chain(self, session, chain_id: str) -> CommandResult:
        """Handle /log chain command - get full interaction chain (Phase 4).

        Args:
            session: Current Session object
            chain_id: Chain ID to retrieve

        Returns:
            CommandResult with chain details
        """
        try:
            from .activity_searcher import ActivitySearcher

            if session.activity_logger is None:
                return CommandResult(
                    success=False,
                    message="Activity logging not enabled for this session",
                    error="ActivityLogger not initialized",
                )

            session_dir = self.session_manager.sessions_dir / session.id
            searcher = ActivitySearcher(
                session_name=session.name,
                log_dir=session_dir / "activity_logs",
                db_path=session_dir / "activities.db"
            )

            # Get chain
            chain = searcher.get_interaction_chain(
                chain_id=chain_id,
                include_content=True,
                include_nested=True
            )

            if not chain:
                return CommandResult(
                    success=False,
                    message=f"Chain '{chain_id}' not found",
                    error="Chain not found"
                )

            # Format chain
            lines = [
                f"Interaction Chain: {chain_id}",
                f"Status: {chain['status']}",
                f"Total Activities: {chain['total_activities']}",
                f"Max Depth: {chain['max_depth_level']}",
                f"Started: {chain['started_at']}",
                f"Completed: {chain['completed_at'] or 'in progress'}",
                f"",
                f"Activities:",
            ]

            for i, activity in enumerate(chain['activities'], 1):
                activity_type = activity.get('activity_type', 'unknown')
                agent_name = activity.get('agent_name', 'system')
                depth = activity.get('depth_level', 0)
                indent = "  " * (depth + 1)

                content = activity.get('content', {})
                if isinstance(content, dict):
                    preview = str(content)[:60] + "..." if len(str(content)) > 60 else str(content)
                else:
                    preview = str(content)[:60] + "..." if len(str(content)) > 60 else str(content)

                lines.append(f"{indent}{i}. {activity_type} - {agent_name}: {preview}")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data=chain
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get chain: {str(e)}",
                error=str(e)
            )

    # === MCP Tool Management Commands (T065-T067) ===

    async def handle_tools_list(self, session) -> CommandResult:
        """Handle /tools list command - list all available MCP tools (T065).

        Args:
            session: Current Session object

        Returns:
            CommandResult with list of MCP tools and their registration status
        """
        try:
            # Get connected servers
            connected_servers = [
                server for server in session.mcp_servers
                if server.state == "connected"
            ]

            if not connected_servers:
                return CommandResult(
                    success=True,
                    message="No MCP servers connected. Use /mcp connect <server_id> to connect a server.",
                    data={"tools": [], "server_count": 0}
                )

            # Collect all tools from connected servers
            all_tools = []
            for server in connected_servers:
                for tool_name in server.discovered_tools:
                    # Find which agents have this tool registered
                    registered_agents = [
                        agent_name for agent_name, agent in session.agents.items()
                        if tool_name in agent.tools
                    ]

                    tool_info = {
                        "name": tool_name,
                        "server_id": server.id,
                        "registered_agents": registered_agents
                    }
                    all_tools.append(tool_info)

            # Format user-friendly message
            if not all_tools:
                return CommandResult(
                    success=True,
                    message=f"Connected servers have no tools available ({len(connected_servers)} servers connected).",
                    data={"tools": [], "server_count": len(connected_servers)}
                )

            # Build formatted output
            lines = [f"Available MCP Tools ({len(all_tools)} total from {len(connected_servers)} servers):"]
            lines.append("")

            # Group tools by server
            tools_by_server = {}
            for tool in all_tools:
                server_id = tool["server_id"]
                if server_id not in tools_by_server:
                    tools_by_server[server_id] = []
                tools_by_server[server_id].append(tool)

            # Format each server's tools
            for server_id, tools in tools_by_server.items():
                lines.append(f"Server: {server_id} ({len(tools)} tools)")
                for tool in tools:
                    # Show registration status
                    if tool["registered_agents"]:
                        agents_str = ", ".join(tool["registered_agents"])
                        lines.append(f"  - {tool['name']} (registered with: {agents_str})")
                    else:
                        lines.append(f"  - {tool['name']} (not registered)")
                lines.append("")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "tools": all_tools,
                    "server_count": len(connected_servers)
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list tools: {str(e)}",
                error=str(e)
            )

    async def handle_tools_add(self, session, server_id: str) -> CommandResult:
        """Handle /tools add command - register MCP tools with current agent (T066).

        Args:
            session: Current Session object
            server_id: Server identifier whose tools to register

        Returns:
            CommandResult with registration status
        """
        try:
            # Check current agent exists
            if not hasattr(session, 'current_agent') or not session.current_agent:
                return CommandResult(
                    success=False,
                    message="No current agent set. Use /agent use <name> to select an agent first.",
                    error="No current agent"
                )

            # Check agent exists in session
            if session.current_agent not in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Current agent '{session.current_agent}' not found in session.",
                    error="Agent not found"
                )

            # Find server by ID
            server = None
            for s in session.mcp_servers:
                if s.id == server_id:
                    server = s
                    break

            if not server:
                return CommandResult(
                    success=False,
                    message=f"MCP server '{server_id}' not found in session.",
                    error="Server not found"
                )

            # Check server is connected
            if server.state != "connected":
                return CommandResult(
                    success=False,
                    message=f"Server '{server_id}' is not connected. Use /mcp connect {server_id} first.",
                    error="Server not connected"
                )

            # Register tools using MCPManager
            from promptchain.cli.utils.mcp_manager import MCPManager
            mcp_manager = MCPManager(session)

            success = await mcp_manager.register_tools_with_agent(
                server_id=server_id,
                agent_name=session.current_agent
            )

            if not success:
                return CommandResult(
                    success=False,
                    message=f"Failed to register tools from '{server_id}' with agent '{session.current_agent}'.",
                    error="Registration failed"
                )

            # Get tool count
            tool_count = len(server.discovered_tools)

            return CommandResult(
                success=True,
                message=f"Successfully registered {tool_count} tools from '{server_id}' with agent '{session.current_agent}'.",
                data={
                    "server_id": server_id,
                    "agent_name": session.current_agent,
                    "tool_count": tool_count,
                    "tools": server.discovered_tools
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to add tools: {str(e)}",
                error=str(e)
            )

    async def handle_tools_remove(self, session, server_id: str) -> CommandResult:
        """Handle /tools remove command - unregister MCP tools from current agent (T067).

        Args:
            session: Current Session object
            server_id: Server identifier whose tools to unregister

        Returns:
            CommandResult with unregistration status
        """
        try:
            # Check current agent exists
            if not hasattr(session, 'current_agent') or not session.current_agent:
                return CommandResult(
                    success=False,
                    message="No current agent set. Use /agent use <name> to select an agent first.",
                    error="No current agent"
                )

            # Check agent exists in session
            if session.current_agent not in session.agents:
                return CommandResult(
                    success=False,
                    message=f"Current agent '{session.current_agent}' not found in session.",
                    error="Agent not found"
                )

            # Find server by ID
            server = None
            for s in session.mcp_servers:
                if s.id == server_id:
                    server = s
                    break

            if not server:
                return CommandResult(
                    success=False,
                    message=f"MCP server '{server_id}' not found in session.",
                    error="Server not found"
                )

            # Unregister tools using MCPManager
            from promptchain.cli.utils.mcp_manager import MCPManager
            mcp_manager = MCPManager(session)

            success = await mcp_manager.unregister_tools_from_agent(
                server_id=server_id,
                agent_name=session.current_agent
            )

            if not success:
                return CommandResult(
                    success=False,
                    message=f"Failed to unregister tools from '{server_id}' with agent '{session.current_agent}'.",
                    error="Unregistration failed"
                )

            # Get tool count (number of tools that were in the server's discovered tools)
            tool_count = len(server.discovered_tools)

            return CommandResult(
                success=True,
                message=f"Successfully unregistered {tool_count} tools from '{server_id}' with agent '{session.current_agent}'.",
                data={
                    "server_id": server_id,
                    "agent_name": session.current_agent,
                    "tool_count": tool_count,
                    "tools": server.discovered_tools
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to remove tools: {str(e)}",
                error=str(e)
            )

    # === History Management Commands (T081) ===

    def handle_history_stats(self, agent_chain) -> CommandResult:
        """Handle /history stats command - display detailed token usage statistics (T081).

        Args:
            agent_chain: AgentChain instance with history managers

        Returns:
            CommandResult with comprehensive token usage statistics
        """
        try:
            # Check if AgentChain has history managers
            if not hasattr(agent_chain, '_history_managers'):
                return CommandResult(
                    success=False,
                    message="History management not enabled for this session",
                    error="No history managers found"
                )

            history_managers = agent_chain._history_managers

            if not history_managers:
                return CommandResult(
                    success=True,
                    message="No agents with history tracking in this session",
                    data={"agents": {}}
                )

            # Collect statistics from all agents
            total_tokens = 0
            total_entries = 0
            agent_stats = {}

            for agent_name, history_manager in history_managers.items():
                if history_manager is None:
                    # Agent has history disabled
                    agent_stats[agent_name] = {
                        "enabled": False,
                        "tokens": 0,
                        "entries": 0,
                        "max_tokens": 0,
                        "max_entries": 0,
                        "utilization_pct": 0.0,
                        "entry_types": {},
                        "truncation_strategy": "N/A"
                    }
                else:
                    # Get statistics from ExecutionHistoryManager
                    stats = history_manager.get_statistics()
                    agent_stats[agent_name] = {
                        "enabled": True,
                        "tokens": stats["total_tokens"],
                        "entries": stats["total_entries"],
                        "max_tokens": stats["max_tokens"],
                        "max_entries": stats["max_entries"],
                        "utilization_pct": stats["utilization_pct"],
                        "entry_types": stats["entry_types"],
                        "truncation_strategy": stats["truncation_strategy"]
                    }
                    total_tokens += stats["total_tokens"]
                    total_entries += stats["total_entries"]

            # Calculate token savings
            enabled_agents = [name for name, stats in agent_stats.items() if stats["enabled"]]
            disabled_agents = [name for name, stats in agent_stats.items() if not stats["enabled"]]

            # Estimate baseline (if all agents had max token limit)
            baseline_tokens = 0
            for agent_name, stats in agent_stats.items():
                if stats["enabled"] and stats["max_tokens"] > 0:
                    baseline_tokens += stats["max_tokens"]

            # Calculate savings percentage
            if baseline_tokens > 0:
                savings_pct = ((baseline_tokens - total_tokens) / baseline_tokens) * 100
            else:
                savings_pct = 0.0

            # Format output
            lines = [
                "=== Token Usage Statistics ===",
                "",
                "Overview:",
                f"  Total Tokens: {total_tokens:,}",
                f"  Total Entries: {total_entries:,}",
                f"  Agents with History: {len(enabled_agents)}",
                f"  Agents without History: {len(disabled_agents)}",
            ]

            if baseline_tokens > 0:
                lines.append(f"  Baseline Tokens: {baseline_tokens:,}")
                lines.append(f"  Token Savings: {savings_pct:.1f}%")

            lines.append("")
            lines.append("Per-Agent Statistics:")

            for agent_name, stats in sorted(agent_stats.items()):
                lines.append(f"  {agent_name}:")
                if stats["enabled"]:
                    lines.append(f"    Status: Enabled")
                    lines.append(f"    Tokens: {stats['tokens']:,} / {stats['max_tokens']:,} ({stats['utilization_pct']:.1f}%)")
                    lines.append(f"    Entries: {stats['entries']} / {stats['max_entries']}")
                    lines.append(f"    Truncation Strategy: {stats['truncation_strategy']}")

                    # Show entry type breakdown
                    if stats["entry_types"]:
                        lines.append(f"    Entry Types:")
                        for entry_type, count in stats["entry_types"].items():
                            lines.append(f"      - {entry_type}: {count}")
                else:
                    lines.append(f"    Status: Disabled (token savings enabled)")
                lines.append("")

            # Add token savings explanation
            if disabled_agents:
                lines.append("Token Savings:")
                lines.append(f"  Agents with disabled history: {', '.join(disabled_agents)}")
                lines.append(f"  These agents use no tokens for history (100% savings per agent)")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "total_tokens": total_tokens,
                    "total_entries": total_entries,
                    "baseline_tokens": baseline_tokens,
                    "savings_pct": savings_pct,
                    "agent_stats": agent_stats,
                    "enabled_agents": enabled_agents,
                    "disabled_agents": disabled_agents
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get history statistics: {str(e)}",
                error=str(e)
            )

    # === Workflow Management Commands (T086-T090) ===

    async def handle_workflow_create(self, session, objective: str) -> CommandResult:
        """Create new workflow with LLM-generated steps (T086).

        Args:
            session: Current Session object
            objective: Workflow objective (e.g., "Implement user authentication")

        Returns:
            CommandResult with workflow ID and initial status

        Process:
            1. Use LLM to generate steps from objective
            2. Create WorkflowState with objective and generated steps
            3. Save via session_manager.save_workflow(session, workflow)
            4. Return success message with workflow ID
        """
        from datetime import datetime
        from litellm import completion
        from .models.workflow import WorkflowState, WorkflowStep

        try:
            # 1. Generate steps using LLM (similar to AgenticStepProcessor pattern)
            generation_prompt = f"""Break down this objective into 5-7 actionable, specific steps:

Objective: {objective}

Return ONLY a JSON array of step descriptions (strings), with no additional text or markdown formatting.
Example format: ["Step 1 description", "Step 2 description", ...]

Each step should be:
- Specific and actionable
- Focused on a single task
- In logical dependency order
- Clear enough to guide implementation"""

            # Get active agent's model
            active_agent = session.agents.get(session.active_agent)
            model_name = active_agent.model_name if active_agent else session.default_model

            # Call LLM to generate steps
            messages = [{"role": "user", "content": generation_prompt}]
            response = completion(model=model_name, messages=messages, temperature=0.3)

            # Extract response content
            llm_response = response.choices[0].message.content.strip()

            # Parse JSON response
            import json
            import re

            # Clean markdown code blocks if present
            cleaned_response = re.sub(r'```json\s*|\s*```', '', llm_response).strip()

            try:
                step_descriptions = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Fallback: Try to extract array from text
                array_match = re.search(r'\[.*\]', cleaned_response, re.DOTALL)
                if array_match:
                    step_descriptions = json.loads(array_match.group(0))
                else:
                    raise ValueError("LLM did not return valid JSON array")

            # Validate step descriptions
            if not isinstance(step_descriptions, list) or len(step_descriptions) == 0:
                raise ValueError("LLM returned empty or invalid step list")

            if len(step_descriptions) > 10:
                step_descriptions = step_descriptions[:10]  # Cap at 10 steps

            # 2. Create WorkflowStep objects (all status="pending")
            steps = [WorkflowStep(description=desc, status="pending") for desc in step_descriptions]

            # 3. Create WorkflowState
            workflow = WorkflowState(
                objective=objective,
                steps=steps,
                current_step_index=0,
                created_at=datetime.now().timestamp()
            )

            # 4. Persist to database
            self.session_manager.save_workflow(session.id, workflow)

            # 5. Format status output
            status_lines = [
                f"Workflow: {workflow.objective}",
                f"Progress: {workflow.progress_percentage:.0f}% ({sum(1 for s in workflow.steps if s.status == 'completed')}/{len(workflow.steps)} steps completed)",
                ""
            ]

            for i, step in enumerate(workflow.steps, 1):
                # Status icons
                status_icon = {
                    "pending": "⬜",
                    "in_progress": "🔄",
                    "completed": "✅",
                    "failed": "❌"
                }.get(step.status, "⬜")

                status_lines.append(f"{status_icon} Step {i}: {step.description} ({step.status})")

            status_message = "\n".join(status_lines)

            # 6. Return CommandResult
            return CommandResult(
                success=True,
                message=f"Workflow created: {len(steps)} steps\n\n{status_message}",
                data={
                    "objective": objective,
                    "step_count": len(steps),
                    "steps": [s.description for s in steps]
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to create workflow: {str(e)}",
                error=str(e)
            )

    async def handle_workflow_resume(self, session) -> CommandResult:
        """Resume workflow by reminding agent of current state (T089).

        Args:
            session: Current Session object

        Returns:
            CommandResult with resume message or error

        Process:
            1. Load current workflow from session
            2. If no workflow or workflow completed, return appropriate message
            3. If workflow active, inject reminder message into conversation
        """
        try:
            # Load workflow from database
            workflow = self.session_manager.load_workflow(session.id)

            if not workflow:
                return CommandResult(
                    success=False,
                    message="No workflow to resume. Use /workflow create <objective> to start one.",
                    error="No workflow found"
                )

            if workflow.is_completed:
                return CommandResult(
                    success=True,
                    message=f"Workflow already complete! All {len(workflow.steps)} steps finished.",
                    data={"workflow": workflow.to_dict()}
                )

            # Build resume message
            current_step = workflow.current_step
            completed_steps = [s.description for s in workflow.steps if s.status == "completed"]

            resume_message = f"""Resuming workflow: {workflow.objective}

Progress: {workflow.progress_percentage:.0f}% ({workflow.current_step_index}/{len(workflow.steps)} steps)

Current step: {current_step.description}

Completed steps:
"""
            for desc in completed_steps:
                resume_message += f"- ✅ {desc}\n"

            resume_message += f"\nLet's continue with: {current_step.description}"

            # Inject as system message into session
            session.add_message(role="system", content=resume_message)

            return CommandResult(
                success=True,
                message=resume_message,
                data={
                    "workflow": workflow.to_dict(),
                    "current_step": current_step.to_dict(),
                    "completed_count": len(completed_steps)
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to resume workflow: {str(e)}",
                error=str(e)
            )

    async def handle_workflow_status(self, session) -> CommandResult:
        """Display current workflow status (T088).

        Args:
            session: Current Session object

        Returns:
            CommandResult with workflow status display

        Display format:
            - If no workflow: "No active workflow"
            - If workflow exists:
                - Objective
                - Progress percentage
                - Current step
                - List of all steps with status indicators:
                    ✅ completed
                    🔄 in_progress
                    ⏳ pending
                    ❌ failed
        """
        try:
            # Load workflow from database
            workflow = self.session_manager.load_workflow(session.id)

            # Check if workflow exists
            if not workflow:
                return CommandResult(
                    success=True,
                    message="No active workflow. Use /workflow create <objective> to start one.",
                    data={"workflow_exists": False}
                )

            # Format status message
            status_lines = [
                f"Workflow: {workflow.objective}",
                f"Progress: {workflow.progress_percentage:.0f}%",
                f"Current Step: {workflow.current_step_index + 1}/{len(workflow.steps)}",
                "",
                "Steps:"
            ]

            # Add each step with status indicator
            for i, step in enumerate(workflow.steps, 1):
                # Map status to indicator
                indicator = {
                    "completed": "✅",
                    "in_progress": "🔄",
                    "pending": "⏳",
                    "failed": "❌"
                }.get(step.status, "⏳")  # Default to pending if unknown status

                # Format step line
                status_lines.append(f"  {i}. {indicator} {step.description}")

            message = "\n".join(status_lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "workflow_exists": True,
                    "objective": workflow.objective,
                    "progress_percentage": workflow.progress_percentage,
                    "current_step_index": workflow.current_step_index,
                    "total_steps": len(workflow.steps),
                    "completed_steps": sum(1 for s in workflow.steps if s.status == "completed"),
                    "in_progress_steps": sum(1 for s in workflow.steps if s.status == "in_progress"),
                    "pending_steps": sum(1 for s in workflow.steps if s.status == "pending"),
                    "failed_steps": sum(1 for s in workflow.steps if s.status == "failed")
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get workflow status: {str(e)}",
                error=str(e)
            )

    async def handle_workflow_list(self) -> CommandResult:
        """List all workflows across sessions (T093).

        Returns:
            CommandResult with formatted workflow list showing:
            - Session name
            - Workflow objective
            - Progress percentage
            - Status (active/complete) with visual indicator

        Format:
            - ✅ for complete workflows
            - 🔄 for active workflows
            - Shows: [session_name] objective (progress%)
        """
        try:
            # Get all workflows from database
            workflows = self.session_manager.list_all_workflows()

            if not workflows:
                return CommandResult(
                    success=True,
                    message="No workflows found. Use /workflow create <objective> to start one.",
                    data={"workflows": [], "count": 0}
                )

            # Format list
            list_lines = ["All Workflows:", ""]
            for wf in workflows:
                # Status icons
                status_icon = "✅" if wf["status"] == "complete" else "🔄"

                # Format workflow line
                session_name = wf["session_name"]
                objective = wf["objective"]
                progress = wf["progress"]
                completed = wf["completed_count"]
                total = wf["step_count"]
                list_lines.append(
                    f"{status_icon} [{session_name}] {objective} "
                    f"({progress:.0f}% - {completed}/{total} steps)"
                )

            return CommandResult(
                success=True,
                message="\n".join(list_lines),
                data={
                    "workflows": workflows,
                    "count": len(workflows)
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list workflows: {str(e)}",
                error=str(e)
            )

    # === Configuration Management Commands (T110-T111) ===

    def handle_config_export(self, session, filename: str) -> CommandResult:
        """Handle /config export command - export CLI configuration to YAML/JSON (T111).

        Args:
            session: Current Session object
            filename: Output filename (.yml, .yaml, or .json)

        Returns:
            CommandResult with export status and file path

        Exports:
            - Session metadata (name, working directory, active agent)
            - All agent configurations (model, tools, history settings)
            - MCP server configurations
            - Orchestration settings (execution mode, router config)
            - History configuration (global and per-agent overrides)

        Security:
            - Validates filename to prevent path traversal
            - No sensitive data exported (API keys, passwords)
            - Uses safe YAML/JSON serialization
        """
        import json
        import yaml
        from pathlib import Path

        try:
            # Validate filename (no path traversal)
            file_path = Path(filename)

            # Check for path traversal attempts
            if ".." in file_path.parts or str(file_path).startswith("/"):
                if not Path(filename).is_absolute():
                    return CommandResult(
                        success=False,
                        message=f"Invalid filename: path traversal detected in '{filename}'",
                        error="Path traversal attempt"
                    )

            # Determine format from extension
            suffix = file_path.suffix.lower()
            if suffix not in [".yml", ".yaml", ".json"]:
                return CommandResult(
                    success=False,
                    message=f"Invalid file extension '{suffix}'. Use .yml, .yaml, or .json",
                    error="Invalid file extension"
                )

            format_type = "yaml" if suffix in [".yml", ".yaml"] else "json"

            # Build configuration dictionary
            config_data = {
                "version": "1.0",
                "session": {
                    "name": session.name,
                    "working_directory": str(session.working_directory),
                    "active_agent": session.active_agent or "default",
                    "default_model": session.default_model,
                    "auto_save_enabled": session.auto_save_enabled,
                },
                "agents": {},
                "mcp_servers": [],
                "history_configuration": {},
                "orchestration": {}
            }

            # Export agents
            for agent_name, agent in session.agents.items():
                agent_config = {
                    "model": agent.model_name,
                    "description": agent.description,
                    "tools": agent.tools,
                }

                # Add template reference if available in metadata
                if agent.metadata and "template" in agent.metadata:
                    agent_config["template"] = agent.metadata["template"]

                # Add history configuration
                if agent.history_config:
                    agent_config["history_config"] = {
                        "enabled": agent.history_config.enabled,
                        "max_tokens": agent.history_config.max_tokens,
                        "max_entries": agent.history_config.max_entries,
                        "truncation_strategy": agent.history_config.truncation_strategy,
                    }

                config_data["agents"][agent_name] = agent_config

            # Export MCP servers
            for server in session.mcp_servers:
                server_config = {
                    "id": server.id,
                    "type": server.type,
                    "auto_connect": server.auto_connect,
                }

                if server.type == "stdio":
                    server_config["command"] = server.command
                    if server.args:
                        server_config["args"] = server.args
                elif server.type == "http":
                    server_config["url"] = server.url

                config_data["mcp_servers"].append(server_config)

            # Export orchestration settings
            if session.orchestration_config:
                orc = session.orchestration_config
                config_data["orchestration"] = {
                    "execution_mode": orc.execution_mode,
                    "auto_include_history": orc.auto_include_history,
                }

                if orc.router_config:
                    router_data = {}
                    if hasattr(orc.router_config, "model"):
                        router_data["model"] = orc.router_config.model
                        router_data["timeout_seconds"] = orc.router_config.timeout_seconds
                    elif isinstance(orc.router_config, dict):
                        router_data = orc.router_config

                    config_data["orchestration"]["router_config"] = router_data

            # Export global history configuration
            if hasattr(session, "_history_manager") and session._history_manager:
                hm = session._history_manager
                config_data["history_configuration"]["global"] = {
                    "max_tokens": hm.max_tokens,
                    "max_entries": hm.max_entries,
                    "truncation_strategy": hm.truncation_strategy,
                }

            # Export per-agent history overrides
            agent_history_overrides = {}
            for agent_name, agent in session.agents.items():
                if agent.history_config:
                    agent_history_overrides[agent_name] = {
                        "enabled": agent.history_config.enabled,
                        "max_tokens": agent.history_config.max_tokens,
                        "max_entries": agent.history_config.max_entries,
                        "truncation_strategy": agent.history_config.truncation_strategy,
                    }

            if agent_history_overrides:
                config_data["history_configuration"]["per_agent"] = agent_history_overrides

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(file_path, "w") as f:
                if format_type == "yaml":
                    yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(config_data, f, indent=2)

            return CommandResult(
                success=True,
                message=f"Configuration exported to {file_path.absolute()}\n"
                        f"Format: {format_type.upper()}\n"
                        f"Agents: {len(config_data['agents'])}\n"
                        f"MCP Servers: {len(config_data['mcp_servers'])}",
                data={
                    "file_path": str(file_path.absolute()),
                    "format": format_type,
                    "agent_count": len(config_data["agents"]),
                    "mcp_server_count": len(config_data["mcp_servers"]),
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to export configuration: {str(e)}",
                error=str(e)
            )

    def handle_config_import(self, session, filename: str) -> CommandResult:
        """Handle /config import command - import CLI configuration from YAML/JSON (T111).

        Args:
            session: Current Session object
            filename: Input filename (.yml, .yaml, or .json)

        Returns:
            CommandResult with import status

        Imports:
            - Agent configurations (creates/updates agents)
            - MCP server configurations (adds servers)
            - History settings (updates per-agent configs)
            - Orchestration settings (if present)

        Security:
            - Validates file exists and is readable
            - Validates configuration structure
            - Sanitizes all imported values
        """
        import json
        import yaml
        from pathlib import Path

        try:
            # Validate file path
            file_path = Path(filename)

            # Determine format from extension (validate before checking existence)
            suffix = file_path.suffix.lower()
            if suffix not in [".yml", ".yaml", ".json"]:
                return CommandResult(
                    success=False,
                    message=f"Invalid file extension '{suffix}'. Use .yml, .yaml, or .json",
                    error="Invalid file extension"
                )

            if not file_path.exists():
                return CommandResult(
                    success=False,
                    message=f"Configuration file not found: {filename}",
                    error="File not found"
                )

            format_type = "yaml" if suffix in [".yml", ".yaml"] else "json"

            # Load configuration
            with open(file_path, "r") as f:
                if format_type == "yaml":
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Validate configuration structure
            if not isinstance(config_data, dict):
                return CommandResult(
                    success=False,
                    message="Invalid configuration format: expected dictionary",
                    error="Invalid format"
                )

            if "version" not in config_data:
                return CommandResult(
                    success=False,
                    message="Invalid configuration: missing 'version' field",
                    error="Missing version"
                )

            changes = []

            # Import agents
            if "agents" in config_data and isinstance(config_data["agents"], dict):
                from promptchain.cli.models.agent_config import Agent, HistoryConfig
                import time

                for agent_name, agent_config in config_data["agents"].items():
                    # Skip if agent exists (don't overwrite)
                    if agent_name in session.agents:
                        changes.append(f"Skipped existing agent: {agent_name}")
                        continue

                    # Create history config if present
                    history_config = None
                    if "history_config" in agent_config:
                        hc = agent_config["history_config"]
                        history_config = HistoryConfig(
                            enabled=hc.get("enabled", True),
                            max_tokens=hc.get("max_tokens", 4000),
                            max_entries=hc.get("max_entries", 20),
                            truncation_strategy=hc.get("truncation_strategy", "oldest_first"),
                        )

                    # Create agent
                    agent = Agent(
                        name=agent_name,
                        model_name=agent_config.get("model", "openai/gpt-4.1-mini-2025-04-14"),
                        description=agent_config.get("description", f"Imported agent: {agent_name}"),
                        tools=agent_config.get("tools", []),
                        history_config=history_config,
                        created_at=time.time(),
                    )

                    # Add template metadata if present
                    if "template" in agent_config:
                        agent.metadata["template"] = agent_config["template"]

                    session.agents[agent_name] = agent
                    changes.append(f"Imported agent: {agent_name} ({agent.model_name})")

            # Import MCP servers
            if "mcp_servers" in config_data and isinstance(config_data["mcp_servers"], list):
                from promptchain.cli.models.mcp_config import MCPServerConfig

                for server_config in config_data["mcp_servers"]:
                    # Skip if server with same ID exists
                    if any(s.id == server_config["id"] for s in session.mcp_servers):
                        changes.append(f"Skipped existing MCP server: {server_config['id']}")
                        continue

                    # Create server
                    server = MCPServerConfig(
                        id=server_config["id"],
                        type=server_config.get("type", "stdio"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                        url=server_config.get("url"),
                        auto_connect=server_config.get("auto_connect", False),
                    )

                    session.mcp_servers.append(server)
                    changes.append(f"Imported MCP server: {server.id} ({server.type})")

            # Import orchestration settings (optional)
            if "orchestration" in config_data:
                from promptchain.cli.models.orchestration_config import OrchestrationConfig

                orc_data = config_data["orchestration"]
                if session.orchestration_config is None:
                    session.orchestration_config = OrchestrationConfig()

                if "execution_mode" in orc_data:
                    session.orchestration_config.execution_mode = orc_data["execution_mode"]
                    changes.append(f"Updated execution mode: {orc_data['execution_mode']}")

                if "auto_include_history" in orc_data:
                    session.orchestration_config.auto_include_history = orc_data["auto_include_history"]
                    changes.append(f"Updated auto-include history: {orc_data['auto_include_history']}")

            # Persist changes
            self.session_manager.save_session(session)

            if not changes:
                return CommandResult(
                    success=True,
                    message=f"No changes imported from {filename}\n(All items already exist or config is empty)",
                    data={"changes": []}
                )

            # Format success message
            message = f"Configuration imported from {filename}\n\nChanges:\n"
            message += "\n".join(f"  • {change}" for change in changes)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "file_path": str(file_path.absolute()),
                    "format": format_type,
                    "changes": changes,
                    "change_count": len(changes),
                }
            )

        except yaml.YAMLError as e:
            return CommandResult(
                success=False,
                message=f"Failed to parse YAML configuration: {str(e)}",
                error=str(e)
            )
        except json.JSONDecodeError as e:
            return CommandResult(
                success=False,
                message=f"Failed to parse JSON configuration: {str(e)}",
                error=str(e)
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to import configuration: {str(e)}",
                error=str(e)
            )

    def handle_config_show(self, session) -> CommandResult:
        """Handle /config show command - display current CLI configuration (T110).

        Args:
            session: Current Session object

        Returns:
            CommandResult with comprehensive configuration display including:
            - Session information
            - MCP server status and tools
            - Agent templates
            - History settings (global and per-agent)
            - Working directory

        Example Output:
            ╭─── Configuration ───╮
            │ Session: my-session │
            │ Active Agent: coder │
            ╰─────────────────────╯

            MCP Servers (2 total):
            ┌────────────┬──────┬─────────┬────────────┐
            │ Server ID  │ Type │ Status  │ Tools      │
            ├────────────┼──────┼─────────┼────────────┤
            │ filesystem │ stdio│ ✓ Conn  │ read,write │
            │ browser    │ stdio│ ○ Disc  │ -          │
            └────────────┴──────┴─────────┴────────────┘

            [... more sections ...]
        """
        from .utils.agent_templates import AGENT_TEMPLATES

        try:
            # Build configuration sections
            lines = []

            # === Section 1: Session Info ===
            lines.append("=== Session Configuration ===")
            lines.append("")
            lines.append(f"Session Name: {session.name}")
            lines.append(f"Session ID: {session.id}")
            lines.append(f"Active Agent: {session.active_agent or 'default'}")
            lines.append(f"Default Model: {session.default_model}")
            lines.append(f"Working Directory: {session.working_directory}")
            lines.append(f"Total Agents: {len(session.agents)}")
            lines.append(f"Auto-Save: {'enabled' if session.auto_save_enabled else 'disabled'}")
            if session.auto_save_enabled:
                lines.append(f"  - Message Interval: {getattr(session, 'autosave_message_interval', 5)} messages")
                lines.append(f"  - Time Interval: {getattr(session, 'autosave_time_interval', 120)} seconds")
            lines.append("")

            # === Section 2: MCP Servers ===
            lines.append("=== MCP Servers ===")
            lines.append("")

            if not session.mcp_servers:
                lines.append("No MCP servers configured")
            else:
                lines.append(f"Total Servers: {len(session.mcp_servers)}")
                connected_count = sum(1 for s in session.mcp_servers if s.state == "connected")
                lines.append(f"Connected: {connected_count}/{len(session.mcp_servers)}")
                lines.append("")

                # Server details
                for server in session.mcp_servers:
                    status_icon = {"connected": "✓", "disconnected": "○", "error": "✗"}.get(server.state, "?")
                    lines.append(f"{status_icon} {server.id}")
                    lines.append(f"  Type: {server.type}")
                    lines.append(f"  Status: {server.state}")

                    if server.type == "stdio":
                        lines.append(f"  Command: {server.command}")
                        if server.args:
                            lines.append(f"  Args: {' '.join(server.args)}")
                    elif server.type == "http":
                        lines.append(f"  URL: {server.url}")

                    if server.state == "connected":
                        tool_count = len(server.discovered_tools)
                        lines.append(f"  Tools: {tool_count} discovered")
                        if tool_count > 0:
                            # Show first 5 tools, truncate if more
                            tools_preview = server.discovered_tools[:5]
                            lines.append(f"    - {', '.join(tools_preview)}")
                            if tool_count > 5:
                                lines.append(f"    - ... and {tool_count - 5} more")
                    elif server.state == "error":
                        lines.append(f"  Error: {server.error_message}")

                    lines.append("")

            # === Section 3: Agent Templates ===
            lines.append("=== Available Agent Templates ===")
            lines.append("")
            lines.append(f"Total Templates: {len(AGENT_TEMPLATES)}")
            lines.append("")

            for template_name, template in AGENT_TEMPLATES.items():
                # Template metadata
                category = template.metadata.get("category", "general") if template.metadata else "general"
                complexity = template.metadata.get("complexity", "medium") if template.metadata else "medium"

                lines.append(f"{template.display_name} ({template_name})")
                lines.append(f"  Description: {template.description[:80]}...")
                lines.append(f"  Model: {template.model}")
                lines.append(f"  Category: {category}")
                lines.append(f"  Complexity: {complexity}")
                lines.append(f"  Tools: {len(template.tools)} configured")
                lines.append(f"  History: {'enabled' if template.history_config.enabled else 'disabled'}")
                if template.history_config.enabled:
                    lines.append(f"    - Max Tokens: {template.history_config.max_tokens}")
                    lines.append(f"    - Max Entries: {template.history_config.max_entries}")
                    lines.append(f"    - Strategy: {template.history_config.truncation_strategy}")
                lines.append("")

            # === Section 4: History Settings ===
            lines.append("=== History Configuration ===")
            lines.append("")

            # Global settings (from session's history_manager if initialized)
            if hasattr(session, '_history_manager') and session._history_manager is not None:
                hm = session._history_manager
                lines.append("Global History Manager:")
                lines.append(f"  Status: Initialized")
                lines.append(f"  Max Tokens: {hm.max_tokens}")
                lines.append(f"  Max Entries: {hm.max_entries}")
                lines.append(f"  Truncation Strategy: {hm.truncation_strategy}")
                lines.append(f"  Current Entries: {len(hm.entries)}")
                stats = hm.get_statistics()
                lines.append(f"  Current Tokens: {stats['total_tokens']}")
                lines.append("")
            else:
                lines.append("Global History Manager: Not initialized (lazy init on first use)")
                lines.append("")

            # Per-agent history configuration
            lines.append("Per-Agent History Overrides:")
            if session.agents:
                for agent_name, agent in session.agents.items():
                    if hasattr(agent, 'history_config') and agent.history_config:
                        hc = agent.history_config
                        lines.append(f"  {agent_name}:")
                        lines.append(f"    Enabled: {hc.enabled}")
                        if hc.enabled:
                            lines.append(f"    Max Tokens: {hc.max_tokens}")
                            lines.append(f"    Max Entries: {hc.max_entries}")
                            lines.append(f"    Truncation: {hc.truncation_strategy}")
                        else:
                            lines.append(f"    Token Savings: ~60% (history disabled)")
                    else:
                        lines.append(f"  {agent_name}: Using global defaults")
                lines.append("")
            else:
                lines.append("  No agents configured yet")
                lines.append("")

            # === Section 5: Agent Summary ===
            lines.append("=== Configured Agents ===")
            lines.append("")

            if not session.agents:
                lines.append("No agents created yet")
            else:
                lines.append(f"Total Agents: {len(session.agents)}")
                lines.append("")

                for agent_name, agent in session.agents.items():
                    active_marker = " (active)" if agent_name == session.active_agent else ""
                    lines.append(f"{agent_name}{active_marker}")
                    lines.append(f"  Model: {agent.model_name}")
                    lines.append(f"  Description: {agent.description}")
                    lines.append(f"  Tools: {len(agent.tools)} registered")
                    if agent.tools:
                        tools_preview = agent.tools[:3]
                        lines.append(f"    - {', '.join(tools_preview)}")
                        if len(agent.tools) > 3:
                            lines.append(f"    - ... and {len(agent.tools) - 3} more")
                    lines.append(f"  Usage Count: {agent.usage_count}")
                    lines.append("")

            # === Section 6: Orchestration Config (if present) ===
            if hasattr(session, 'orchestration_config') and session.orchestration_config:
                lines.append("=== Orchestration Configuration ===")
                lines.append("")
                orc = session.orchestration_config
                lines.append(f"Execution Mode: {orc.execution_mode}")
                lines.append(f"Auto-Include History: {orc.auto_include_history}")
                if orc.router_config:
                    # Check if router_config is a RouterConfig object or dict
                    if hasattr(orc.router_config, 'model'):
                        lines.append(f"Router Model: {orc.router_config.model}")
                    elif isinstance(orc.router_config, dict):
                        models = orc.router_config.get('models', ['not configured'])
                        model = models[0] if isinstance(models, list) else models
                        lines.append(f"Router Model: {model}")
                lines.append("")

            # Build final message
            message = "\n".join(lines)

            # Return result with structured data
            return CommandResult(
                success=True,
                message=message,
                data={
                    "session": {
                        "name": session.name,
                        "id": session.id,
                        "active_agent": session.active_agent,
                        "agent_count": len(session.agents),
                        "working_directory": str(session.working_directory),
                    },
                    "mcp_servers": {
                        "total": len(session.mcp_servers),
                        "connected": sum(1 for s in session.mcp_servers if s.state == "connected"),
                        "servers": [s.to_dict() for s in session.mcp_servers],
                    },
                    "templates": {
                        "available": list(AGENT_TEMPLATES.keys()),
                        "count": len(AGENT_TEMPLATES),
                    },
                    "agents": {
                        "count": len(session.agents),
                        "names": list(session.agents.keys()),
                    },
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to display configuration: {str(e)}",
                error=str(e)
            )

    # === Tool Capabilities Commands ===

    def handle_capabilities(self, agent_name: Optional[str] = None) -> CommandResult:
        """Handle /capabilities command - list tools by capability.

        Args:
            agent_name: Optional agent name to filter tools (shows only tools accessible to that agent)

        Returns:
            CommandResult with formatted capability list

        Usage:
            /capabilities              - List all capabilities and their tools
            /capabilities <agent_name> - List capabilities available to specific agent

        Example Output:
            Available Capabilities:
              file_read (4 tools): file_read, read_file_range, list_directory, ...
              file_write (6 tools): file_write, file_edit, file_append, ...
              code_search (1 tool): ripgrep_search

            When agent name provided:
            Tools available to agent 'worker':
              file_read: Read complete file contents
              file_write: Write/create files
              ...
        """
        try:
            from promptchain.cli.tools import registry

            # Get all capabilities
            all_capabilities = registry.list_capabilities()

            if not all_capabilities:
                return CommandResult(
                    success=True,
                    message="No capabilities registered yet. Register tools with capabilities to see them here.",
                    data={"capabilities": [], "count": 0}
                )

            # If agent_name provided, filter by agent access
            if agent_name:
                # Get tools available to this agent
                agent_tools = registry.discover_capabilities(agent_name=agent_name)

                if not agent_tools:
                    return CommandResult(
                        success=True,
                        message=f"No tools available to agent '{agent_name}'.\n"
                                f"Use /agent update {agent_name} --add-tools <tool_name> to add tools.",
                        data={"agent_name": agent_name, "tools": [], "count": 0}
                    )

                # Format agent-specific tool list
                lines = [f"Tools available to agent '{agent_name}':", ""]

                # Group by capability
                capability_groups = {}
                for tool in agent_tools:
                    for cap in tool.capabilities:
                        if cap not in capability_groups:
                            capability_groups[cap] = []
                        capability_groups[cap].append(tool)

                # Format each capability group
                for capability in sorted(capability_groups.keys()):
                    tools = capability_groups[capability]
                    lines.append(f"Capability: {capability} ({len(tools)} tools)")
                    for tool in tools:
                        lines.append(f"  - {tool.name}: {tool.description[:60]}...")
                    lines.append("")

                # Add ungrouped tools (tools with no capabilities)
                ungrouped = [t for t in agent_tools if not t.capabilities]
                if ungrouped:
                    lines.append(f"Other Tools ({len(ungrouped)} tools):")
                    for tool in ungrouped:
                        lines.append(f"  - {tool.name}: {tool.description[:60]}...")
                    lines.append("")

                message = "\n".join(lines)

                return CommandResult(
                    success=True,
                    message=message,
                    data={
                        "agent_name": agent_name,
                        "tools": [t.name for t in agent_tools],
                        "count": len(agent_tools),
                        "capabilities": list(capability_groups.keys())
                    }
                )

            # No agent specified - show all capabilities
            lines = [f"Available Capabilities ({len(all_capabilities)} total):", ""]

            # Get tools for each capability
            for capability in sorted(all_capabilities):
                tools = registry.get_by_capability(capability)

                # Format capability line with tool count and names
                tool_names = [t.name for t in tools]
                if len(tool_names) <= 3:
                    tool_list = ", ".join(tool_names)
                else:
                    tool_list = ", ".join(tool_names[:3]) + f", ... ({len(tool_names) - 3} more)"

                lines.append(f"  {capability} ({len(tools)} tools): {tool_list}")

            lines.append("")
            lines.append("Usage:")
            lines.append("  /capabilities <agent_name>  - Show capabilities for specific agent")
            lines.append("  /tools list                 - Show all tools")

            message = "\n".join(lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "capabilities": all_capabilities,
                    "count": len(all_capabilities)
                }
            )

        except ImportError:
            return CommandResult(
                success=False,
                message="Tool registry not available. Ensure promptchain.cli.tools.registry is properly configured.",
                error="Registry import failed"
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list capabilities: {str(e)}",
                error=str(e)
            )

    # === Security Mode Commands ===

    def handle_security(self, session, mode: Optional[str] = None) -> CommandResult:
        """Handle /security command - manage session security mode.

        Args:
            session: Current Session object
            mode: Optional security mode to set (strict, trusted, default)

        Returns:
            CommandResult with security status or mode change confirmation

        Usage:
            /security          - Show current security mode and status
            /security strict   - Enable strict mode (all warnings, require confirmation)
            /security trusted  - Enable trusted mode (no warnings)
            /security default  - Enable default mode (warn once per path)
        """
        from .security_context import (
            SecurityMode,
            get_security_context,
            set_security_context,
            SecurityContext
        )

        try:
            # Get or create security context
            security_ctx = get_security_context()

            # If no mode specified, show current status
            if mode is None:
                status = security_ctx.get_status()
                return CommandResult(
                    success=True,
                    message=f"Current Security Status:\n\n{status}\n\nUsage:\n  /security strict   - Require confirmation for outside paths\n  /security trusted  - Allow all paths without warnings\n  /security default  - Warn once per path (default)",
                    data={
                        "mode": security_ctx.mode.value,
                        "working_directory": security_ctx.working_directory,
                        "approved_paths": list(security_ctx.approved_paths),
                        "denied_paths": list(security_ctx.denied_paths),
                    }
                )

            # Validate and set mode
            mode_lower = mode.lower().strip()

            if mode_lower == "strict":
                new_mode = SecurityMode.STRICT
            elif mode_lower == "trusted":
                new_mode = SecurityMode.TRUSTED
            elif mode_lower == "default":
                new_mode = SecurityMode.DEFAULT
            else:
                return CommandResult(
                    success=False,
                    message=f"Invalid security mode: '{mode}'\n\nValid modes:\n  strict  - Require confirmation for outside paths\n  trusted - Allow all paths without warnings\n  default - Warn once per path",
                    error=f"Invalid security mode: {mode}"
                )

            # Set the new mode
            confirmation_message = security_ctx.set_mode(new_mode)

            # Update session with security context (for persistence)
            if hasattr(session, 'security_context_data'):
                session.security_context_data = security_ctx.to_dict()

            return CommandResult(
                success=True,
                message=confirmation_message,
                data={
                    "mode": new_mode.value,
                    "previous_mode": security_ctx.mode.value if security_ctx.mode != new_mode else None,
                    "working_directory": security_ctx.working_directory,
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to manage security mode: {str(e)}",
                error=str(e)
            )

    # === Task Management Commands ===

    def handle_tasks(self, session, agent_name: Optional[str] = None) -> CommandResult:
        """Handle /tasks command - list pending/in-progress tasks.

        Args:
            session: Current Session object
            agent_name: Optional agent name to filter tasks

        Returns:
            CommandResult with task list

        Usage:
            /tasks              - List tasks for current agent
            /tasks agent_name   - List tasks for specific agent
        """
        try:
            # Determine which agent to filter by
            target_agent = agent_name or session.current_agent

            # List tasks with filtering
            tasks = self.session_manager.list_tasks(
                session_id=session.session_id,
                status=None,  # Get all statuses
                target_agent=target_agent,
                limit=100
            )

            if not tasks:
                return CommandResult(
                    success=True,
                    message=f"No tasks found for agent '{target_agent}'",
                    data={"tasks": [], "count": 0}
                )

            # Filter for pending and in_progress tasks
            active_tasks = [
                task for task in tasks
                if task.status in ["pending", "in_progress"]
            ]

            if not active_tasks:
                return CommandResult(
                    success=True,
                    message=f"No pending or in-progress tasks for agent '{target_agent}'",
                    data={"tasks": [], "count": 0}
                )

            # Format task list
            task_lines = [f"Tasks for agent '{target_agent}':\n"]
            for i, task in enumerate(active_tasks, 1):
                priority_symbol = "🔴" if task.priority == "high" else "🟡" if task.priority == "medium" else "🟢"
                status_symbol = "▶️" if task.status == "in_progress" else "⏸️"
                task_lines.append(
                    f"{i}. {priority_symbol} {status_symbol} [{task.task_id[:8]}] {task.description}"
                )
                if task.context:
                    task_lines.append(f"   Context: {task.context}")

            message = "\n".join(task_lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "tasks": [
                        {
                            "task_id": task.task_id,
                            "description": task.description,
                            "priority": task.priority,
                            "status": task.status,
                            "source_agent": task.source_agent,
                            "target_agent": task.target_agent,
                            "context": task.context
                        }
                        for task in active_tasks
                    ],
                    "count": len(active_tasks),
                    "agent": target_agent
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list tasks: {str(e)}",
                error=str(e)
            )

    # === Blackboard Commands ===

    def handle_blackboard(self, session, key: Optional[str] = None) -> CommandResult:
        """Handle /blackboard command - list or show blackboard entries.

        Args:
            session: Current Session object
            key: Optional key to show specific entry

        Returns:
            CommandResult with blackboard data

        Usage:
            /blackboard      - List all blackboard keys
            /blackboard key  - Show specific blackboard entry
        """
        try:
            # If key specified, show specific entry
            if key:
                entry = self.session_manager.read_blackboard(
                    session_id=session.session_id,
                    key=key
                )

                if not entry:
                    return CommandResult(
                        success=False,
                        message=f"Blackboard entry not found: '{key}'",
                        error="Entry not found"
                    )

                # Format entry details
                from datetime import datetime
                written_at = datetime.fromtimestamp(entry.written_at).strftime("%Y-%m-%d %H:%M:%S")

                message = f"Blackboard Entry: {key}\n\n"
                message += f"Written by: {entry.written_by}\n"
                message += f"Written at: {written_at}\n"
                message += f"Version: {entry.version}\n\n"
                message += f"Value:\n{entry.value}"

                return CommandResult(
                    success=True,
                    message=message,
                    data={
                        "key": key,
                        "value": entry.value,
                        "written_by": entry.written_by,
                        "written_at": written_at,
                        "version": entry.version
                    }
                )

            # Otherwise, list all keys
            keys = self.session_manager.list_blackboard_keys(
                session_id=session.session_id
            )

            if not keys:
                return CommandResult(
                    success=True,
                    message="No blackboard entries found",
                    data={"keys": [], "count": 0}
                )

            # Format key list
            message = f"Blackboard Entries ({len(keys)}):\n\n"
            for i, k in enumerate(keys, 1):
                message += f"{i}. {k}\n"

            message += "\nUse '/blackboard <key>' to view entry details"

            return CommandResult(
                success=True,
                message=message,
                data={
                    "keys": keys,
                    "count": len(keys)
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to access blackboard: {str(e)}",
                error=str(e)
            )

    # === Mental Model Commands ===

    def handle_mentalmodel(self, session) -> CommandResult:
        """Handle /mentalmodel command - show agent's mental model.

        Args:
            session: Current Session object

        Returns:
            CommandResult with mental model visualization

        Usage:
            /mentalmodel  - Display current agent's mental model
        """
        try:
            from .models import MentalModelManager, create_default_model, SpecializationType

            # Get current agent name
            agent_name = session.active_agent or "default"

            # Get or create mental model manager
            if not hasattr(self.session_manager, '_mental_model_manager'):
                self.session_manager._mental_model_manager = MentalModelManager()

            model = self.session_manager._mental_model_manager.get_or_create(agent_name)

            # If no specializations, create default
            if not model.specializations:
                model = create_default_model(agent_name, [SpecializationType.GENERAL])
                self.session_manager._mental_model_manager.update(model)

            # Build output message
            message_lines = [f"Mental Model for agent: {agent_name}", ""]

            # Specializations
            if model.specializations:
                message_lines.append("Specializations:")
                for spec in model.specializations:
                    message_lines.append(f"  - {spec.specialization.value} (proficiency: {spec.proficiency:.2f})")
                    if spec.related_capabilities:
                        caps_str = ", ".join(spec.related_capabilities)
                        message_lines.append(f"    Capabilities: {caps_str}")
                message_lines.append("")
            else:
                message_lines.append("Specializations: None")
                message_lines.append("")

            # Known Agents
            if model.known_agents:
                message_lines.append("Known Agents:")
                for agent, capabilities in model.known_agents.items():
                    caps_str = ", ".join(capabilities) if capabilities else "none"
                    message_lines.append(f"  - {agent}: {caps_str}")
                message_lines.append("")
            else:
                message_lines.append("Known Agents: None")
                message_lines.append("")

            # Recent Tasks (last 5)
            if model.task_history:
                message_lines.append("Recent Tasks (last 5):")
                recent_tasks = model.task_history[-5:]
                for task in recent_tasks:
                    status_icon = "✓" if task.get("success", False) else "✗"
                    task_type = task.get("task_type", "unknown")
                    status = "success" if task.get("success", False) else "failed"
                    message_lines.append(f"  {status_icon} {task_type} - {status}")
            else:
                message_lines.append("Recent Tasks: None")

            message = "\n".join(message_lines)

            return CommandResult(
                success=True,
                message=message,
                data={
                    "agent_name": agent_name,
                    "specializations": [
                        {
                            "type": s.specialization.value,
                            "proficiency": s.proficiency,
                            "capabilities": s.related_capabilities,
                            "experience": s.experience_count
                        }
                        for s in model.specializations
                    ],
                    "known_agents": model.known_agents,
                    "recent_tasks": model.task_history[-5:] if model.task_history else []
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to display mental model: {str(e)}",
                error=str(e)
            )

    # === Workflow Commands (T067) ===

    def handle_workflow(
        self, session, subcommand: Optional[str] = None
    ) -> CommandResult:
        """Handle /workflow command - show current workflow state.

        Args:
            session: Current Session object
            subcommand: Optional subcommand (show/stage/tasks)

        Returns:
            CommandResult with workflow state information

        Usage:
            /workflow        - Display current workflow status (default: show)
            /workflow show   - Display current workflow status
            /workflow stage  - Show current stage only
            /workflow tasks  - Show completed task count
        """
        try:
            # Get current workflow from session
            workflow = self.session_manager.get_multi_agent_workflow(session.id)

            if not workflow:
                return CommandResult(
                    success=True,
                    message="No active workflow in this session",
                    data={"workflow": None}
                )

            # Normalize subcommand (default to "show")
            subcommand = (subcommand or "show").lower()

            # Handle subcommands
            if subcommand in ["show", ""]:
                # Full workflow status display
                from datetime import datetime

                # Calculate progress percentage
                total_tasks = len(workflow.completed_tasks)
                if workflow.current_task:
                    total_tasks += 1  # Include current task in total

                # Determine progress percentage based on stage
                stage_progress = {
                    "planning": 10,
                    "execution": 50,
                    "review": 75,
                    "complete": 100
                }
                progress = stage_progress.get(workflow.stage.value, 0)

                # Format timestamps
                started = datetime.fromtimestamp(workflow.started_at).strftime("%Y-%m-%d %H:%M:%S")
                updated = datetime.fromtimestamp(workflow.updated_at).strftime("%Y-%m-%d %H:%M:%S")

                # Build status message
                message_lines = [
                    f"Current Workflow: {workflow.workflow_id}",
                    f"Stage: {workflow.stage.value}",
                    f"Agents Involved: {', '.join(workflow.agents_involved) if workflow.agents_involved else 'None'}",
                    f"Completed Tasks: {len(workflow.completed_tasks)}",
                    f"Current Task: {workflow.current_task or 'None'}",
                    f"Progress: {progress}% complete",
                    f"Started: {started}",
                    f"Updated: {updated}"
                ]

                message = "\n".join(message_lines)

                return CommandResult(
                    success=True,
                    message=message,
                    data={
                        "workflow_id": workflow.workflow_id,
                        "stage": workflow.stage.value,
                        "agents_involved": workflow.agents_involved,
                        "completed_tasks": len(workflow.completed_tasks),
                        "current_task": workflow.current_task,
                        "progress": progress,
                        "started_at": workflow.started_at,
                        "updated_at": workflow.updated_at
                    }
                )

            elif subcommand == "stage":
                # Show current stage only
                return CommandResult(
                    success=True,
                    message=f"Current Stage: {workflow.stage.value}",
                    data={
                        "workflow_id": workflow.workflow_id,
                        "stage": workflow.stage.value
                    }
                )

            elif subcommand == "tasks":
                # Show completed task count
                completed = len(workflow.completed_tasks)
                current = workflow.current_task or "None"

                message = f"Completed Tasks: {completed}\nCurrent Task: {current}"

                return CommandResult(
                    success=True,
                    message=message,
                    data={
                        "workflow_id": workflow.workflow_id,
                        "completed_tasks": completed,
                        "current_task": current,
                        "task_ids": workflow.completed_tasks
                    }
                )

            else:
                # Invalid subcommand
                return CommandResult(
                    success=False,
                    message=f"Invalid subcommand: '{subcommand}'. Use: show, stage, or tasks",
                    error=f"Invalid subcommand: {subcommand}"
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to get workflow status: {str(e)}",
                error=str(e)
            )

