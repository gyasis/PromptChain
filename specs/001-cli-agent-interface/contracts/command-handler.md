# Contract: CommandHandler

**Component**: `promptchain.cli.command_handler.CommandHandler`
**Purpose**: Parse, validate, and route slash commands to appropriate handlers
**Integration Points**: SessionManager, AgentChain, Session state

---

## Public API Contract

### Class Definition

```python
class CommandHandler:
    """
    Handles slash command parsing and execution.

    Responsibilities:
    - Parse command syntax (/command subcommand args)
    - Validate command arguments
    - Route to appropriate handler methods
    - Return structured results or errors
    """

    def __init__(self, session_manager: SessionManager):
        """
        Initialize command handler.

        Args:
            session_manager: Session manager for session operations
        """
```

---

## Supported Commands (MVP)

### /agent Commands

#### /agent create

**Syntax**: `/agent create <name> --model <model> [--description <desc>]`

**Contract**:
```python
async def handle_agent_create(
    self,
    session: Session,
    name: str,
    model: str,
    description: str = ""
) -> CommandResult:
    """
    Create new agent in session.

    Args:
        session: Active session
        name: Agent name (1-32 chars, alphanumeric+dashes)
        model: LiteLLM model string (e.g., "openai/gpt-4")
        description: Optional agent description (max 256 chars)

    Returns:
        CommandResult: Success with agent details or error

    Validates:
        - Agent name uniqueness within session
        - Agent name format
        - Model string format
        - Description length

    Side Effects:
        - Agent added to session.agents
        - Agent entry created in database
        - PromptChain instance initialized for agent
    """
```

**Example**:
```python
result = await handler.handle_agent_create(
    session,
    name="coding",
    model="openai/gpt-4",
    description="Coding specialist"
)
# result.success = True
# result.message = "Agent 'coding' created with model openai/gpt-4"
```

---

#### /agent list

**Syntax**: `/agent list`

**Contract**:
```python
async def handle_agent_list(self, session: Session) -> CommandResult:
    """
    List all agents in session.

    Args:
        session: Active session

    Returns:
        CommandResult: List of agents with details

    Output Format:
        Available agents:
        - default (openai/gpt-4) - Default agent [ACTIVE]
        - coding (openai/gpt-4) - Coding specialist (used 5 times)
        - research (anthropic/claude-3-opus-20240229) - Research agent (used 2 times)
    """
```

**Example**:
```python
result = await handler.handle_agent_list(session)
# result.data = [
#   {"name": "default", "model": "openai/gpt-4", "is_active": True, "usage_count": 0},
#   {"name": "coding", "model": "openai/gpt-4", "usage_count": 5},
# ]
```

---

#### /agent use

**Syntax**: `/agent use <name>`

**Contract**:
```python
async def handle_agent_use(
    self,
    session: Session,
    name: str
) -> CommandResult:
    """
    Switch to different agent.

    Args:
        session: Active session
        name: Agent name to activate

    Returns:
        CommandResult: Success with agent name or error

    Validates:
        - Agent exists in session

    Side Effects:
        - session.active_agent updated
        - Auto-save triggered (before switch)
        - Agent usage_count incremented
        - Agent last_used timestamp updated
    """
```

**Example**:
```python
result = await handler.handle_agent_use(session, name="coding")
# result.success = True
# result.message = "Switched to agent 'coding'"
# session.active_agent = "coding"
```

---

#### /agent delete

**Syntax**: `/agent delete <name>`

**Contract**:
```python
async def handle_agent_delete(
    self,
    session: Session,
    name: str
) -> CommandResult:
    """
    Delete agent from session.

    Args:
        session: Active session
        name: Agent name to delete

    Returns:
        CommandResult: Success or error

    Validates:
        - Agent exists in session
        - Agent is not currently active (must switch first)

    Side Effects:
        - Agent removed from session.agents
        - Agent entry deleted from database
        - PromptChain instance cleaned up
    """
```

**Example**:
```python
result = await handler.handle_agent_delete(session, name="old-agent")
# result.success = True
# result.message = "Agent 'old-agent' deleted"
```

---

### /session Commands

#### /session save

**Syntax**: `/session save [name]`

**Contract**:
```python
async def handle_session_save(
    self,
    session: Session,
    name: str | None = None
) -> CommandResult:
    """
    Save session (explicit save or rename).

    Args:
        session: Active session
        name: New session name (optional, for renaming)

    Returns:
        CommandResult: Success with save confirmation

    Side Effects:
        - Session saved to database
        - If name provided, session renamed
        - session.last_accessed updated
    """
```

**Example**:
```python
# Explicit save
result = await handler.handle_session_save(session)
# result.message = "Session 'my-project' saved"

# Rename and save
result = await handler.handle_session_save(session, name="new-name")
# result.message = "Session renamed to 'new-name' and saved"
```

---

#### /session list

**Syntax**: `/session list`

**Contract**:
```python
async def handle_session_list(self) -> CommandResult:
    """
    List all saved sessions.

    Returns:
        CommandResult: List of sessions with metadata

    Output Format:
        Saved sessions:
        - my-project (last accessed: 2 hours ago, 3 agents)
        - research (last accessed: 3 days ago, 1 agent)
    """
```

**Example**:
```python
result = await handler.handle_session_list()
# result.data = [
#   {"name": "my-project", "last_accessed": 1700000000.0, "agent_count": 3},
#   {"name": "research", "last_accessed": 1699900000.0, "agent_count": 1},
# ]
```

---

#### /session delete

**Syntax**: `/session delete <name>`

**Contract**:
```python
async def handle_session_delete(self, name: str) -> CommandResult:
    """
    Delete saved session.

    Args:
        name: Session name to delete

    Returns:
        CommandResult: Success or error

    Validates:
        - Session exists
        - Session is not currently active

    Side Effects:
        - Session deleted from database
        - Session files removed
    """
```

**Example**:
```python
result = await handler.handle_session_delete(name="old-session")
# result.success = True
# result.message = "Session 'old-session' deleted"
```

---

### /help Commands

#### /help

**Syntax**: `/help [topic]`

**Contract**:
```python
async def handle_help(self, topic: str | None = None) -> CommandResult:
    """
    Show help documentation.

    Args:
        topic: Optional help topic (commands, agents, sessions, files)

    Returns:
        CommandResult: Help text for topic or general help
    """
```

**Example**:
```python
result = await handler.handle_help()
# result.message = "Available commands: /agent, /session, /help, /exit, /status"

result = await handler.handle_help(topic="agent")
# result.message = "Agent commands: create, list, use, delete..."
```

---

### /status Command

**Syntax**: `/status`

**Contract**:
```python
async def handle_status(self, session: Session) -> CommandResult:
    """
    Show session status.

    Args:
        session: Active session

    Returns:
        CommandResult: Session status details

    Output Format:
        Session: my-project
        Working Directory: /home/user/projects/my-project
        Active Agent: coding (openai/gpt-4)
        Total Agents: 3
        Messages: 47
        Auto-save: Enabled (every 2 minutes)
    """
```

---

### /exit Command

**Syntax**: `/exit`

**Contract**:
```python
async def handle_exit(self, session: Session) -> CommandResult:
    """
    Exit session gracefully.

    Args:
        session: Active session

    Returns:
        CommandResult: Exit confirmation

    Side Effects:
        - Final auto-save triggered
        - Session state saved
        - Goodbye message prepared
    """
```

---

## Command Parsing Contract

### Method: parse_command

**Signature**:
```python
def parse_command(self, raw_input: str) -> ParsedCommand | None:
    """
    Parse slash command from raw input.

    Args:
        raw_input: User input string

    Returns:
        ParsedCommand: Parsed command with name, subcommand, args
        None: If input is not a slash command

    Parsing Rules:
        - Commands start with '/'
        - Format: /command [subcommand] [args]
        - Arguments can use --flag syntax or positional
    """
```

**Example**:
```python
parsed = parse_command("/agent create coding --model openai/gpt-4")
# ParsedCommand(
#   name="agent",
#   subcommand="create",
#   args={"name": "coding", "model": "openai/gpt-4"}
# )

parsed = parse_command("Regular message")
# None (not a command)
```

---

## CommandResult Contract

**Data Structure**:
```python
@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    message: str
    data: dict | list | None = None
    error: str | None = None

    def __str__(self) -> str:
        """User-friendly string representation."""
        if self.success:
            return self.message
        else:
            return f"Error: {self.error}"
```

**Example**:
```python
# Success result
result = CommandResult(
    success=True,
    message="Agent 'coding' created",
    data={"name": "coding", "model": "openai/gpt-4"}
)

# Error result
result = CommandResult(
    success=False,
    message="Command failed",
    error="Agent 'unknown' does not exist"
)
```

---

## Error Handling Contracts

### CommandNotFoundError

```python
class CommandNotFoundError(Exception):
    """Raised when command is not recognized."""
    def __init__(self, command: str):
        super().__init__(f"Unknown command: {command}")
```

### InvalidCommandSyntaxError

```python
class InvalidCommandSyntaxError(Exception):
    """Raised when command syntax is invalid."""
    def __init__(self, command: str, reason: str):
        super().__init__(f"Invalid syntax for '{command}': {reason}")
```

### CommandExecutionError

```python
class CommandExecutionError(Exception):
    """Raised when command execution fails."""
    def __init__(self, command: str, reason: str):
        super().__init__(f"Failed to execute '{command}': {reason}")
```

---

## Testing Contract

### Contract Tests

```python
# tests/cli/contract/test_command_handler_contract.py

async def test_agent_create_validation():
    """Contract: Agent creation validates inputs."""
    handler = CommandHandler(session_manager)

    # Valid creation
    result = await handler.handle_agent_create(
        session, name="coding", model="openai/gpt-4"
    )
    assert result.success

    # Invalid name (too long)
    result = await handler.handle_agent_create(
        session, name="a" * 33, model="openai/gpt-4"
    )
    assert not result.success
    assert "name" in result.error.lower()

async def test_agent_use_updates_session():
    """Contract: Agent use updates session state."""
    handler = CommandHandler(session_manager)

    # Create agents
    await handler.handle_agent_create(session, "coding", "openai/gpt-4")
    await handler.handle_agent_create(session, "research", "anthropic/claude-3-opus")

    # Switch agent
    result = await handler.handle_agent_use(session, "research")
    assert result.success
    assert session.active_agent == "research"

    # Verify usage count incremented
    research_agent = session.agents["research"]
    assert research_agent.usage_count == 1
    assert research_agent.last_used is not None

async def test_command_parsing():
    """Contract: Command parsing follows defined syntax."""
    handler = CommandHandler(session_manager)

    # Parse agent create
    parsed = handler.parse_command("/agent create coding --model openai/gpt-4")
    assert parsed.name == "agent"
    assert parsed.subcommand == "create"
    assert parsed.args["name"] == "coding"
    assert parsed.args["model"] == "openai/gpt-4"

    # Parse session list
    parsed = handler.parse_command("/session list")
    assert parsed.name == "session"
    assert parsed.subcommand == "list"

    # Not a command
    parsed = handler.parse_command("Just a message")
    assert parsed is None
```

---

## Integration Contract with Textual UI

**Command Palette Integration**:
```python
# Textual CommandPalette receives command suggestions
def get_command_completions(prefix: str) -> list[str]:
    """
    Get command completions for autocomplete.

    Args:
        prefix: Current user input (e.g., "/ag")

    Returns:
        list[str]: Matching commands (e.g., ["/agent create", "/agent list"])
    """
```

**UI Feedback**:
```python
# CommandResult → UI message
async def execute_command(raw_input: str) -> str:
    """
    Execute command and return UI-ready message.

    Args:
        raw_input: Raw command input

    Returns:
        str: Formatted message for UI display
    """
    parsed = handler.parse_command(raw_input)
    if not parsed:
        return None  # Not a command

    result = await handler.execute(parsed)
    return str(result)  # Uses CommandResult.__str__()
```

---

## Command Extension Points (Future)

**Plugin Commands** (FR-EXT-006):
```python
# Future: Plugin registration
def register_plugin_command(
    command: str,
    handler: Callable,
    description: str
) -> None:
    """
    Register custom slash command from plugin.

    Args:
        command: Command name (e.g., "toon")
        handler: Async function to handle command
        description: Help text for command
    """
```

**Example Plugin Command**:
```python
# Plugin: TOON extension
async def handle_toon_enable(session: Session) -> CommandResult:
    """Handle /toon enable command."""
    session.metadata["toon_enabled"] = True
    return CommandResult(success=True, message="TOON mode enabled")

# Register with CLI
register_plugin_command("toon", handle_toon_enable, "Enable TOON format")
```

---

## Backward Compatibility

**Version 1.0 (Initial)**:
- All commands listed above are stable
- Command syntax frozen (no breaking changes)
- New commands added without breaking existing ones
- Plugin system provides extension path (FR-EXT-006)

**Deprecation Policy**:
- Deprecated commands show warning for 2 versions before removal
- `--legacy` flag preserves old behavior during transition
