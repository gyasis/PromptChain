# Phase 0: Technology Research

**Feature**: PromptChain CLI Agent Interface
**Branch**: 001-cli-agent-interface
**Date**: 2025-11-16

## Research Objectives

Validate technology stack choices for building an interactive CLI agent interface, with focus on:
1. Terminal UI framework selection (Textual vs alternatives)
2. CLI framework for command parsing
3. Session persistence strategy
4. File reference parsing approach
5. Integration patterns with existing PromptChain components

## Technology Decisions

### 1. Terminal UI Framework: Textual

**Decision**: Use Textual 0.83+ for TUI components

**Rationale**:
- **Async-native**: Built on asyncio, perfectly aligned with Constitution Principle VI (Async-First Design)
- **Rich integration**: Built by same authors as Rich, seamless formatting capabilities
- **Reactive widgets**: Event-driven architecture matches PromptChain's ExecutionEvent system
- **Cross-platform**: Runs on Linux, macOS, Windows with identical behavior
- **Proven in CLI tools**: Used in production by multiple CLI agent tools

**Key Features Used**:
- `App` class for main event loop
- `Reactive` properties for status updates
- `ListView` for conversation display
- `TextArea` for multi-line input with history
- `Footer`/`Header` for status bar
- `CommandPalette` for slash command autocomplete

**Alternatives Considered**:
- **prompt_toolkit**: Lower-level, more manual widget management, no built-in reactive system
- **urwid**: Older framework, less active development, callback-heavy vs async
- **Rich only**: No input widgets, output-only library
- **Plain terminal**: Too much reinvention, no cross-platform guarantees

**Reference**: Textual documentation research via Context7 (performed in earlier conversation)

---

### 2. CLI Framework: Click

**Decision**: Use Click 8.1+ for command-line parsing and entry point

**Rationale**:
- **Decorator-based**: Clean, Pythonic API matching existing PromptChain style
- **Nested commands**: Supports `promptchain`, `promptchain --session NAME` patterns
- **Type safety**: Automatic type conversion and validation
- **Help generation**: Auto-generated `--help` documentation
- **Wide adoption**: Industry standard for Python CLIs

**Usage Pattern**:
```python
@click.command()
@click.option('--session', help='Resume existing session')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(session, verbose):
    """Launch PromptChain CLI interactive session."""
    asyncio.run(run_textual_app(session, verbose))
```

**Alternatives Considered**:
- **argparse**: Standard library but more verbose, no decorator pattern
- **Typer**: Built on Click, adds type hints, but introduces extra dependency
- **fire**: Too magical, less explicit than Click

---

### 3. Session Persistence: SQLite + JSON

**Decision**: Use SQLite for session metadata, JSON/JSONL for conversation history

**Rationale**:
- **Leverage existing pattern**: AgentChain already uses SQLite via `cache_config`
- **No external dependencies**: SQLite in Python stdlib
- **Schema flexibility**: Easy schema evolution for session metadata
- **JSONL for history**: Human-readable, line-by-line conversation logs (matches RunLogger pattern)

**Schema Design**:
```sql
-- sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    working_directory TEXT NOT NULL,
    active_agent TEXT,
    metadata_json TEXT  -- JSON blob for extensibility
);

-- agents table (per-session agent configurations)
CREATE TABLE agents (
    session_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    description TEXT,
    model_name TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, agent_name),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Conversation history stored in JSONL files
-- Path: ~/.promptchain/sessions/{session_id}/history.jsonl
```

**Alternatives Considered**:
- **All JSON files**: No query capabilities, slow session listing
- **PostgreSQL/MySQL**: Overkill, external dependencies, violates Simplicity principle
- **Pickle**: Not human-readable, version compatibility issues

---

### 4. File Reference Parsing: Regex + Path Resolution

**Decision**: Use regex for `@syntax` detection, pathlib for path resolution

**Rationale**:
- **Simple pattern**: `@` prefix is unambiguous in natural language
- **Standard library**: No external dependencies (regex + pathlib in stdlib)
- **Supports both files and dirs**: `@file.txt` vs `@directory/`
- **Safety**: Path resolution with checks for existence, permissions, symlinks

**Implementation Approach**:
```python
import re
from pathlib import Path

FILE_REF_PATTERN = r'@([\w\-./]+)'

def parse_file_references(message: str, working_dir: Path) -> List[FileReference]:
    """Extract @file.txt and @dir/ references from message."""
    matches = re.findall(FILE_REF_PATTERN, message)
    refs = []
    for match in matches:
        path = (working_dir / match).resolve()
        if path.exists():
            if path.is_file():
                refs.append(FileReference(path=path, content=read_file(path)))
            elif path.is_dir():
                refs.append(FileReference(path=path, files=discover_relevant_files(path)))
    return refs
```

**Large File Handling**:
- Files >100KB: Show preview (first 500 lines + last 100 lines) with token count
- Binary files: Detect via magic bytes, show metadata only
- Permission errors: Graceful failure with user-friendly error

**Alternatives Considered**:
- **LSP protocol**: Overkill for file references, not needed for MVP
- **Full file indexing**: Violates YAGNI, premature optimization
- **Git integration**: P4+ feature, defer to later iteration

---

### 5. Shell Command Execution: subprocess with Output Streaming

**Decision**: Use `asyncio.create_subprocess_shell()` for `!command` execution

**Rationale**:
- **Async-compatible**: Non-blocking, matches Textual event loop
- **Output streaming**: Real-time output display (no wait for completion)
- **Timeout support**: Built-in timeout to prevent hangs
- **Cross-platform**: Works on Unix and Windows

**Implementation Pattern**:
```python
async def execute_shell_command(command: str, timeout: int = 30) -> ShellOutput:
    """Execute shell command and stream output."""
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )
        return ShellOutput(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode
        )
    except asyncio.TimeoutError:
        process.kill()
        return ShellOutput(error="Command timed out")
```

**Safety Considerations**:
- Shell injection protection: Display command to user before execution
- Working directory context: Commands execute in session's working_directory
- Environment isolation: Use clean environment, no automatic env var injection

**Alternatives Considered**:
- **Synchronous subprocess**: Blocks event loop, poor UX
- **PTY-based execution**: Complex, not needed for basic output capture
- **Docker containers**: Overkill for MVP, adds security but violates Simplicity

---

### 6. Multi-line Input: Textual TextArea

**Decision**: Use Textual `TextArea` widget for input handling

**Rationale**:
- **Native multi-line**: Built-in support for Shift+Enter (newline) vs Enter (submit)
- **Command history**: Built-in history navigation with up/down arrows
- **Syntax highlighting**: Optional code highlighting for pasted code
- **Clipboard support**: Copy/paste works seamlessly

**Key Bindings**:
- `Enter`: Submit message (when not in code block)
- `Shift+Enter`: New line
- `Up/Down`: Navigate command history
- `Ctrl+C`: Cancel current operation (not exit)
- `Ctrl+D` or `/exit`: Graceful session exit

**Alternatives Considered**:
- **prompt_toolkit**: More powerful but requires manual integration with Textual
- **readline**: Too basic, no multi-line support

---

### 7. Agent Model Configuration: LiteLLM Provider Strings

**Decision**: Reuse existing LiteLLM model specification pattern

**Rationale**:
- **Already implemented**: PromptChain uses LiteLLM for model calls
- **Provider-agnostic**: Supports `openai/gpt-4`, `anthropic/claude-3`, `ollama/llama2`
- **No new code**: Directly pass model string to PromptChain constructor
- **User familiarity**: Matches existing PromptChain library usage

**Implementation**:
```python
# User creates agent with any model
/agent create coding --model openai/gpt-4
/agent create fast --model ollama/llama2
/agent create research --model anthropic/claude-3-opus-20240229

# CLI passes model string directly to PromptChain
agent_chain = PromptChain(
    models=[agent_config.model_name],  # e.g., "openai/gpt-4"
    instructions=agent_config.instructions,
    verbose=True
)
```

**Default Model Strategy**:
- Read from environment: `PROMPTCHAIN_DEFAULT_MODEL` (e.g., `openai/gpt-4o`)
- Fallback hierarchy: `openai/gpt-4` → `anthropic/claude-3-sonnet` → `ollama/llama2`
- User can override per-session with `/config default-model <model>`

---

### 8. Session Auto-save: Periodic Background Task

**Decision**: Use Textual `set_interval()` for auto-save timer

**Rationale**:
- **Built-in scheduler**: Textual provides timer primitives
- **Async-safe**: Runs in event loop, no threading issues
- **Configurable**: User can adjust auto-save frequency

**Implementation**:
```python
class PromptChainApp(App):
    def on_mount(self):
        # Auto-save every 2 minutes OR every 5 messages (whichever first)
        self.set_interval(120, self.auto_save_session)

    async def auto_save_session(self):
        await self.session_manager.save_current_session()
```

**Trigger Conditions** (SC-007):
- Every 5 messages (message count threshold)
- Every 2 minutes (time threshold)
- Before agent switch
- Before `/exit` (final save)

---

## Integration Points with Existing PromptChain

### 1. AgentChain Integration

**Existing Component**: `promptchain.utils.agent_chain.AgentChain`

**Integration Approach**:
- CLI creates AgentChain with `execution_mode="router"` for P2 (agent switching)
- Router configuration uses LLM-based agent selection (existing feature)
- CLI wraps `AgentChain.run_chat()` for conversation loop
- Session history passed via `auto_include_history=True` (existing feature)

**Key Methods Used**:
```python
agent_chain = AgentChain(
    agents={"default": default_agent, "coding": coding_agent},
    execution_mode="router",
    auto_include_history=True,
    agent_history_configs={...},  # Per-agent history (v0.4.2)
    verbose=True
)

# Async chat loop
await agent_chain.run_chat_async()  # If exists, else wrap run_chat()
```

---

### 2. ExecutionHistoryManager Integration

**Existing Component**: `promptchain.utils.execution_history_manager.ExecutionHistoryManager`

**Integration Approach**:
- One history manager per session
- Token-aware truncation prevents context overflow
- Structured entry types: `user_input`, `agent_output`, `tool_call`, `tool_result`
- Add new entry type: `cli_command` for slash commands

**Usage**:
```python
history_manager = ExecutionHistoryManager(
    max_tokens=8000,
    max_entries=100,
    truncation_strategy="oldest_first"
)

# Track user input
history_manager.add_entry("user_input", message, source="user")

# Track agent response
history_manager.add_entry("agent_output", response, source=active_agent_name)

# Track CLI commands
history_manager.add_entry("cli_command", "/agent create coding", source="cli")
```

---

### 3. MCPHelper Integration

**Existing Component**: `promptchain.utils.mcp_helpers.MCPHelper`

**Integration Approach**:
- MCP servers configured at CLI startup (from config file or flags)
- Tools available to all agents in session
- No CLI-specific changes needed (MCPHelper already complete)

**Configuration**:
```python
# ~/.promptchain/config.json
{
  "mcp_servers": [
    {"id": "filesystem", "type": "stdio", "command": "mcp-server-filesystem", "args": ["--root", "."]}
  ]
}

# CLI loads config and passes to PromptChain
mcp_config = load_mcp_config()
agent = PromptChain(mcp_servers=mcp_config, ...)
```

---

### 4. RunLogger Integration

**Existing Component**: `promptchain.utils.logging_utils.RunLogger`

**Integration Approach**:
- Session logs saved to `~/.promptchain/sessions/{session_id}/logs/`
- JSONL format for structured log analysis
- Events: `CLI_SESSION_START`, `CLI_COMMAND`, `CLI_AGENT_SWITCH`, `CLI_FILE_REF`

**Usage**:
```python
logger = RunLogger(log_dir=session_log_dir)
logger.log_run({
    "event": "CLI_SESSION_START",
    "session_id": session.id,
    "timestamp": time.time()
})
```

---

## Dependency Installation

**New Dependencies** (add to requirements.txt):
```
textual>=0.83.0
click>=8.1.0
```

**Existing Dependencies** (no changes):
```
litellm>=1.0.0
rich>=13.8.0
python-dotenv
tiktoken
```

**Development Dependencies** (add to requirements-dev.txt):
```
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0
```

---

## Performance Validation

### Benchmark Targets (from Technical Context)

| Metric | Target | Strategy |
|--------|--------|----------|
| Session startup | <10s | Lazy loading: Don't init agents until first use |
| File reference load | <500ms | Stream large files, truncate at 100KB |
| Session save | <2s | Incremental writes, async I/O |
| Session resume | <3s | SQLite indexes on session_name, created_at |
| 100+ message conversation | No degradation | ExecutionHistoryManager token truncation |

**Measurement Approach**:
- Integration tests with `time.perf_counter()` assertions
- Pytest benchmarks for session operations
- Manual testing with 1000+ saved sessions

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Textual learning curve | Medium | Extensive examples in docs, active community |
| Cross-platform terminal differences | Low | Textual abstracts platform differences |
| Session database corruption | Medium | SQLite WAL mode, periodic backups |
| Large file memory issues | Medium | Streaming reads, 100KB preview limit |
| Shell command security | High | User confirmation, no auto-execution, sandboxing (future) |

---

## Research Validation

✅ All technology choices align with Constitution Principles:
- Library-First: Textual, Click are standalone libraries
- Observable: RunLogger, ExecutionEvent integration
- Test-First: pytest-asyncio for async testing
- Token Economy: ExecutionHistoryManager reuse
- Async-First: Textual built on asyncio
- Simplicity: No complex frameworks, stdlib where possible

✅ All choices leverage existing PromptChain infrastructure:
- AgentChain for multi-agent orchestration
- ExecutionHistoryManager for conversation history
- MCPHelper for tool integration
- RunLogger for observability

**Research Phase Complete. Ready for Phase 1: Data Model & Contracts.**
