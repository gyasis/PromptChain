# PromptChain CLI - Interactive Terminal Interface

An interactive terminal interface for PromptChain that provides conversational AI interactions with persistent sessions, multi-agent orchestration, and seamless file/shell integration.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Development](#development)
- [Slash Commands Reference](#slash-commands-reference)

## Overview

PromptChain CLI provides a sophisticated terminal user interface (TUI) for managing conversational AI interactions. Built on the Textual framework, it combines the power of PromptChain's LLM orchestration with an intuitive, keyboard-driven interface.

**What it does:**
- Maintains persistent conversation sessions across CLI invocations
- Manages multiple specialized AI agents with different models
- Integrates file content directly into prompts using `@syntax`
- Executes shell commands within conversations using `!syntax`
- Provides conversation history management with token-aware truncation
- Auto-saves sessions to prevent data loss

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PromptChain CLI                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Textual TUI Application                    │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐    │ │
│  │  │ ChatView   │  │InputWidget │  │  StatusBar     │    │ │
│  │  │ (Messages) │  │ (Input)    │  │ (Session Info) │    │ │
│  │  └────────────┘  └────────────┘  └────────────────┘    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                    │
│  ┌────────────────────────┼──────────────────────────────────┐│
│  │         Core Components│                                  ││
│  │                        ▼                                  ││
│  │  ┌─────────────────────────────────────────────┐         ││
│  │  │         SessionManager (Persistence)        │         ││
│  │  │  • SQLite database (metadata, agents)       │         ││
│  │  │  • JSONL files (conversation history)       │         ││
│  │  │  • Auto-save logic (time + message count)   │         ││
│  │  └─────────────────────────────────────────────┘         ││
│  │                        │                                  ││
│  │  ┌─────────────────────┼─────────────────────┐           ││
│  │  │     CommandHandler  │  FileContextManager │           ││
│  │  │  • Slash commands   │  • @file references │           ││
│  │  │  • Agent mgmt       │  • Directory listing│           ││
│  │  │  • Session mgmt     │  • Glob patterns    │           ││
│  │  └─────────────────────┴─────────────────────┘           ││
│  │                        │                                  ││
│  │  ┌─────────────────────┼─────────────────────┐           ││
│  │  │    ShellExecutor    │  PromptChain (Core) │           ││
│  │  │  • !command exec    │  • LLM orchestration│           ││
│  │  │  • Output capture   │  • Multi-agent      │           ││
│  │  │  • Timeout handling │  • Tool integration │           ││
│  │  └─────────────────────┴─────────────────────┘           ││
│  └──────────────────────────────────────────────────────────┘│
│                           │                                    │
│  ┌────────────────────────┼──────────────────────────────────┐│
│  │         Data Models    ▼                                  ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐      ││
│  │  │  Session   │  │   Agent    │  │    Message     │      ││
│  │  │  Config    │  │   Config   │  │                │      ││
│  │  └────────────┘  └────────────┘  └────────────────┘      ││
│  └──────────────────────────────────────────────────────────┘│
│                           │                                    │
│  ┌────────────────────────┼──────────────────────────────────┐│
│  │         Storage        ▼                                  ││
│  │  ┌───────────────────────────────────────────┐            ││
│  │  │  ~/.promptchain/                          │            ││
│  │  │  ├── config.json (CLI configuration)      │            ││
│  │  │  └── sessions/                            │            ││
│  │  │      ├── sessions.db (SQLite metadata)    │            ││
│  │  │      └── <session-id>/                    │            ││
│  │  │          └── messages.jsonl (history)     │            ││
│  │  └───────────────────────────────────────────┘            ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**TUI Layer (Textual Framework):**
- `app.py` - Main application orchestrator, event routing, error handling
- `chat_view.py` - Scrollable message display with Rich formatting
- `input_widget.py` - Multi-line input with syntax highlighting, command history
- `status_bar.py` - Real-time session info (agent, model, message count)

**Session Management:**
- `session_manager.py` - CRUD operations for sessions, SQLite + JSONL persistence
- `models/session.py` - Session data model with auto-save logic, ExecutionHistoryManager integration
- `models/agent_config.py` - Agent configuration with usage tracking
- `models/message.py` - Message data model with metadata support
- `models/config.py` - CLI configuration model (T151-T153)

**Command Processing:**
- `command_handler.py` - Slash command parser and router (`/exit`, `/agent`, `/session`, `/help`)
- `shell_executor.py` - Shell command execution with timeout, output capture
- `file_context_manager.py` - File reference resolution (`@file`, `@dir/`, `@*.py`)

**Integration:**
- PromptChain core library for LLM orchestration
- ExecutionHistoryManager for token-aware conversation history
- LiteLLM for unified model access (OpenAI, Anthropic, Google, Ollama)

## Key Features

### 1. Persistent Sessions
- **SQLite Storage**: Session metadata, agent configurations, and settings
- **JSONL History**: Append-only conversation logs for efficient loading
- **Auto-Save**: Triggers every 5 messages or 2 minutes (configurable)
- **Session Switching**: Load/create sessions by name or ID

### 2. Multi-Agent Management
- **Specialized Agents**: Create agents with different models for specific tasks
- **Dynamic Switching**: Change active agent mid-conversation with `/agent use`
- **Usage Tracking**: Monitor agent invocation counts and last-used timestamps
- **Lazy Loading**: Agents initialized on-demand for faster startup (T148)

### 3. File Context Integration
- **Single Files**: `@path/to/file.py` - Include file content in prompt
- **Directories**: `@src/` - List directory contents
- **Glob Patterns**: `@**/*.ts` - Match multiple files with wildcards
- **Smart Truncation**: Large files (>100KB) show first 500 + last 100 lines

### 4. Shell Command Execution
- **Single Commands**: `!ls -la` - Execute and capture output
- **Shell Mode**: `!!` - Toggle mode where all input executes as commands
- **Formatted Output**: Color-coded exit codes, execution timing, stderr highlighting
- **Working Directory**: Commands run in session's working directory context

### 5. Advanced History Management
- **Token-Aware Truncation**: Uses tiktoken for accurate context window management
- **Configurable Limits**: Control max tokens (default: 6000) and max entries (default: 50)
- **Structured Entries**: Separate tracking for user input, agent output, tool calls, errors
- **Format Styles**: Chat format, full JSON, or filtered by entry type

### 6. Error Handling & Recovery
- **User-Friendly Messages**: Common errors (API keys, rate limits, network) show actionable guidance
- **Graceful Degradation**: Errors don't crash sessions, logged for debugging
- **Debug Mode**: Set `PROMPTCHAIN_DEBUG=1` for full tracebacks
- **Session Logging**: All errors persisted to JSONL with metadata (T143)

### 7. Configuration System (T151-T153)
- **Config File**: `~/.promptchain/config.json` for persistent settings
- **Default Settings**: Default model, agent, UI preferences, performance tuning
- **Validation**: Automatic validation on load with fallback to defaults
- **Hot Reload**: Future support for live config updates

### 8. Comprehensive Help System (T154-T156)
- **Topic-Based Help**: `/help commands`, `/help shell`, `/help files`, `/help shortcuts`, `/help config`
- **Context-Aware**: Provides relevant examples and best practices
- **Progressive Disclosure**: General overview with drill-down into specific topics

## Installation

### Prerequisites
- Python 3.8+
- PromptChain package installed

### Install PromptChain
```bash
# From PyPI (when published)
pip install promptchain

# From source (development)
git clone https://github.com/your-org/promptchain.git
cd promptchain
pip install -e .
```

### Verify Installation
```bash
promptchain --version
```

## Quick Start

### 1. Launch CLI
```bash
# Start with default session
promptchain

# Start with named session
promptchain --session my-project

# Use custom sessions directory
promptchain --sessions-dir /path/to/sessions
```

### 2. Basic Chat
```
> What is the capital of France?
[Agent responds with Paris]

> Tell me more about its history
[Agent responds with historical context]
```

### 3. Create Specialized Agent
```
> /agent create coder --model openai/gpt-4 --description "Python coding specialist"
Created agent 'coder' with model openai/gpt-4

> /agent use coder
Switched to agent 'coder' (openai/gpt-4)
```

### 4. Include File Context
```
> @main.py Review this code for bugs
[Agent reads main.py and provides code review]

> @src/ What's the overall architecture?
[Agent analyzes src/ directory structure]
```

### 5. Execute Shell Commands
```
> !git status
On branch main
nothing to commit, working tree clean

> !pytest tests/
[Test output displayed]

> Based on the test results, what failed?
[Agent analyzes test output from previous command]
```

### 6. Exit and Resume
```
> /exit
Session 'my-project' saved. Goodbye!

# Later...
$ promptchain --session my-project
[Full conversation history restored]
```

## Usage Examples

### Multi-Agent Workflow
```bash
$ promptchain --session code-review

# Create specialized agents
> /agent create reviewer --model anthropic/claude-3-opus-20240229 --description "Code review expert"
> /agent create coder --model openai/gpt-4 --description "Code generation specialist"
> /agent create tester --model openai/gpt-4 --description "Test writing specialist"

# Review code with Claude
> /agent use reviewer
> @src/auth.py Review for security issues
[Claude performs security analysis]

# Generate fixes with GPT-4
> /agent use coder
> Based on the review, refactor the authentication logic
[GPT-4 generates refactored code]

# Write tests
> /agent use tester
> Write comprehensive tests for the new auth logic
[GPT-4 generates test cases]
```

### File Discovery and Analysis
```bash
# Analyze project structure
> @. What's the high-level structure?
[Agent lists root directory]

# Deep dive into specific areas
> @src/**/*.py Summarize all Python modules
[Agent analyzes all Python files recursively]

# Compare files
> @tests/test_auth.py @src/auth.py Do the tests cover all edge cases?
[Agent compares test coverage against implementation]
```

### Shell Integration for DevOps
```bash
# Check system status
> !docker ps
[Container list displayed]

# Query agent about output
> Which containers are using the most memory?
[Agent analyzes docker ps output]

# Execute multiple commands
> !!
[Shell mode activated]
> git status
> git diff
> pytest
> !!
[Back to chat mode]

# Analyze all results
> Based on the git diff and test results, what should I commit?
[Agent provides recommendation]
```

### Debugging Workflow
```bash
# Run failing tests
> !pytest tests/test_api.py -v
[Test failures displayed]

# Analyze with specialized agent
> /agent create debugger --model anthropic/claude-3-opus-20240229
> /agent use debugger
> @tests/test_api.py @src/api.py Why is test_auth failing?
[Agent analyzes both files and test output to identify root cause]

# Apply fix
> Here's the fix: [paste code]
[Agent validates the fix]

# Re-run tests
> !pytest tests/test_api.py -v
[Tests now passing]
```

## Configuration

### Config File Location
```
~/.promptchain/config.json
```

### Default Configuration
```json
{
  "default_model": "openai/gpt-4",
  "default_agent": "default",
  "sessions_dir": null,
  "ui": {
    "max_displayed_messages": 100,
    "show_line_numbers": false,
    "theme": "default",
    "animation_fps": 10
  },
  "performance": {
    "lazy_load_agents": true,
    "history_max_tokens": 6000,
    "cache_enabled": true
  },
  "metadata": {}
}
```

### Configuration Options

**General Settings:**
- `default_model` - Default LiteLLM model string (e.g., `"openai/gpt-4"`, `"anthropic/claude-3-sonnet-20240229"`)
- `default_agent` - Name of default agent to use (default: `"default"`)
- `sessions_dir` - Custom sessions directory path (default: `~/.promptchain/sessions`)

**UI Settings (`ui`):**
- `max_displayed_messages` - Maximum messages shown in chat (pagination threshold, min: 10)
- `show_line_numbers` - Show line numbers in input widget (default: `false`)
- `theme` - Color theme: `"default"`, `"dark"`, or `"light"`
- `animation_fps` - Spinner animation frame rate (1-60 fps)

**Performance Settings (`performance`):**
- `lazy_load_agents` - Load agents on-demand vs all at startup (default: `true`)
- `history_max_tokens` - Max tokens for conversation history (min: 1000, default: 6000)
- `cache_enabled` - Enable session caching (default: `true`)

### View Current Configuration
```
> /help config
Configuration
Config File Location:
  /home/user/.promptchain/config.json

Current Settings:
  Default Model: openai/gpt-4
  Default Agent: default
  Max Displayed Messages: 100
  Lazy Load Agents: True
  History Max Tokens: 6000
```

### Modify Configuration
Edit `~/.promptchain/config.json` manually. Changes take effect on next CLI launch.

### Environment Variables

Required for LLM providers:
```bash
# .env file or shell environment
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Optional: Enable debug mode
PROMPTCHAIN_DEBUG=1  # Show full tracebacks
```

## Technical Details

### Textual TUI Framework
- **Event-Driven Architecture**: Async message passing between widgets
- **CSS Styling**: Terminal-based styling with pseudo-selectors
- **Rich Integration**: Advanced text formatting with syntax highlighting
- **Keyboard Bindings**: Vim-style keybindings for navigation (future enhancement)

### Storage Architecture

**SQLite Database (`sessions.db`):**
- Schema version tracking with migration support
- Foreign key constraints for referential integrity
- Indexes on frequently queried fields (name, last_accessed)

```sql
-- Core tables
sessions (id, name, created_at, last_accessed, working_directory, active_agent, ...)
agents (session_id, name, model_name, description, created_at, usage_count, ...)
schema_version (version, applied_at, description)
```

**JSONL Conversation History:**
- Append-only format for efficient writes
- Each line is a complete JSON message object
- Supports streaming playback for large histories
- Example entry:
```json
{"role": "user", "content": "Hello", "timestamp": 1699123456.789, "metadata": {}}
```

### ExecutionHistoryManager Integration

The CLI integrates PromptChain's `ExecutionHistoryManager` for sophisticated history management:

**Features:**
- **Token Counting**: Uses tiktoken for accurate GPT/Claude token estimation
- **Smart Truncation**: Multiple strategies (oldest_first, keep_last, sliding_window)
- **Entry Types**: user_input, agent_output, tool_call, tool_result, error, system
- **Filtering**: Include/exclude by type, source, time range
- **Format Styles**:
  - `chat`: OpenAI-style message list
  - `full_json`: Complete structured data
  - `markdown`: Human-readable conversation

**Session Integration:**
```python
# Lazy initialization on first access
session.history_manager  # Creates ExecutionHistoryManager with session settings

# Automatic population from messages
for msg in session.messages:
    history_manager.add_entry(entry_type=msg.role, content=msg.content, ...)

# Token-aware retrieval
formatted_history = session.history_manager.get_formatted_history(
    format_style='chat',
    max_tokens=6000  # Leaves room for response
)
```

### Auto-Save Logic

**Trigger Conditions (OR logic):**
- Message count threshold: 5+ messages since last save (configurable via `autosave_message_interval`)
- Time threshold: 2+ minutes since last save (configurable via `autosave_time_interval`)

**Implementation:**
```python
# In session.py
def check_autosave(self, session_manager):
    if not self.auto_save_enabled:
        return

    message_threshold_met = self.messages_since_save >= 5
    time_threshold_met = (current_time - self.last_save_time) >= 120

    if message_threshold_met or time_threshold_met:
        session_manager.save_session(self)
        self.messages_since_save = 0
        self.last_save_time = current_time
```

**Exit Behavior:**
- Always saves on graceful exit (`/exit` or Ctrl+D)
- Saves on exception/crash via `on_exit()` handler

### Error Handling Categories

The CLI provides context-aware error messages for:

1. **API Key Errors**: Detect missing/invalid keys, show provider-specific setup instructions
2. **Model Errors**: Invalid model names, region restrictions, API tier issues
3. **Rate Limit Errors**: HTTP 429, suggest wait times and tier upgrades
4. **Network Errors**: Connection failures, timeouts, firewall issues
5. **File Errors**: FileNotFoundError, PermissionError with working directory context
6. **Generic Errors**: Full traceback in debug mode, GitHub issue link

Example output:
```
Error: API key not configured

The model you're using requires authentication.

Set environment variable: OPENAI_API_KEY=your_key

Add your API keys to a .env file or export as environment variables.
Original error: AuthenticationError: No API key provided
```

### Lazy Loading Strategy (T148)

**Benefits:**
- Faster CLI startup (2-3x improvement for sessions with 5+ agents)
- Reduced memory footprint (only active agent loaded)
- On-demand initialization of LiteLLM clients

**Implementation:**
```python
# In app.py
def _initialize_agent_chain(self):
    # Only load active agent eagerly
    active_agent_name = self.session.active_agent
    self.agent_chains[active_agent_name] = PromptChain(models=[...])

def _get_or_create_agent_chain(self, agent_name: str):
    # Lazy load on first access
    if agent_name not in self.agent_chains:
        agent = self.session.agents[agent_name]
        self.agent_chains[agent_name] = PromptChain(models=[agent.model_name])
    return self.agent_chains[agent_name]
```

### History Pagination (T149)

**Features:**
- **Page-Based Loading**: Load 100 messages at a time
- **Infinite Scroll**: Load older messages on scroll-up
- **Message Caching**: Keep loaded messages in memory for fast navigation
- **Performance**: Constant-time scroll for sessions with 1000+ messages

### Model Support

**Via LiteLLM (100+ models):**

OpenAI:
```
openai/gpt-4o
openai/gpt-4-turbo
openai/gpt-4
openai/gpt-3.5-turbo
```

Anthropic:
```
anthropic/claude-3-opus-20240229
anthropic/claude-3-sonnet-20240229
anthropic/claude-3-haiku-20240307
```

Google:
```
gemini/gemini-pro
gemini/gemini-pro-vision
```

Local Models (Ollama):
```
ollama/llama2
ollama/codellama
ollama/mistral
```

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for full list.

## Development

### Project Structure
```
promptchain/cli/
├── __init__.py              # Package initialization
├── main.py                  # CLI entry point (Click commands)
├── session_manager.py       # Session CRUD and persistence
├── command_handler.py       # Slash command parser and router
├── shell_executor.py        # Shell command execution
├── schema.sql               # SQLite database schema
├── models/
│   ├── __init__.py
│   ├── session.py           # Session data model
│   ├── agent_config.py      # Agent configuration model
│   ├── message.py           # Message data model
│   └── config.py            # CLI configuration model
├── tui/
│   ├── __init__.py
│   ├── app.py               # Main Textual application
│   ├── chat_view.py         # Message display widget
│   ├── input_widget.py      # Multi-line input widget
│   └── status_bar.py        # Session info status bar
└── utils/
    ├── __init__.py
    ├── file_context_manager.py   # File reference resolution
    ├── file_loader.py            # File content loading
    ├── file_truncator.py         # Large file truncation
    ├── directory_discoverer.py   # Directory listing
    ├── output_formatter.py       # Shell/assistant output formatting
    └── file_reference_parser.py  # @syntax parsing
```

### Adding New Commands

See full development guide in the [Development section](#development) for detailed instructions on:
- Adding new slash commands
- Extending TUI widgets
- Testing strategies
- Debugging techniques

### Testing

**Run Tests:**
```bash
# Unit tests
pytest tests/cli/

# Integration tests (requires API keys)
pytest tests/cli/integration/

# Specific test file
pytest tests/cli/test_session_manager.py -v

# With coverage
pytest tests/cli/ --cov=promptchain.cli
```

### Debugging

**Enable Debug Mode:**
```bash
export PROMPTCHAIN_DEBUG=1
promptchain --session debug-session
```

**View Logs:**
```bash
# Session logs (JSONL format)
cat ~/.promptchain/sessions/<session-id>/messages.jsonl | jq

# SQLite inspection
sqlite3 ~/.promptchain/sessions/sessions.db
> SELECT * FROM sessions;
> SELECT * FROM agents WHERE session_id = '<id>';
```

## Keyboard Shortcuts

**Input Shortcuts:**
- `Enter` - Submit message
- `Shift+Enter` - Insert newline (multi-line input)
- `Tab` - Autocomplete slash commands (future)
- `Up/Down` - Navigate command history (future)
- `Ctrl+C` - Cancel current input (clears without exiting)

**App Shortcuts:**
- `Ctrl+D` - Exit application (saves session)
- `Ctrl+C` - Cancel operation (does NOT exit)

**Message Selection:**
- `Click` - Select/deselect message
- `c` - Copy focused message (when selected)
- `📋 Select All` button - Copy all conversation text to clipboard

## Slash Commands Reference

**Session Management:**
- `/session` - Show current session info (ID, state, working dir, message count)
- `/session list` - List all saved sessions with metadata (future)
- `/session delete <name>` - Delete a saved session (future)
- `/exit` - Save session and exit CLI

**Agent Management:**
- `/agent` - Show active agent info (future)
- `/agent create <name> --model <model> --description <desc>` - Create new agent (future)
- `/agent list` - List all agents in current session (future)
- `/agent use <name>` - Switch to specific agent (future)
- `/agent delete <name>` - Remove an agent (future)

**Help System:**
- `/help` - Show general help overview
- `/help commands` - Slash commands reference
- `/help shell` - Shell command integration
- `/help files` - File reference syntax
- `/help shortcuts` - Keyboard shortcuts
- `/help config` - Configuration settings

**Utility:**
- `/clear` - Clear chat history (future)

## Data Flow

### Typical Message Flow

1. **User Input**: User types message in InputWidget and presses Enter
2. **Command Detection**: PromptChainApp checks for slash command (`/`) or shell command (`!`)
3. **File Reference Parsing**: If message contains `@syntax`, FileContextManager loads file contents
4. **Message Augmentation**: User message combined with file contexts
5. **LLM Invocation**: PromptChain instance for active agent processes augmented message
6. **History Update**: ExecutionHistoryManager records user input and agent response
7. **Display**: ChatView renders new messages
8. **Auto-Save**: SessionManager checks if auto-save threshold reached

### Agent Switch Flow

1. **Command**: User types `/agent use <name>`
2. **Parsing**: CommandHandler parses command
3. **Validation**: Check if agent exists in session.agents
4. **Update**: SessionManager sets session.active_agent = <name>
5. **PromptChain Switch**: Next message uses the agent's PromptChain instance
6. **Status Update**: StatusBar displays new active agent and model
7. **Persistence**: Auto-save triggers to persist active_agent change

### Session Save/Resume Flow

**Save**:
1. User types `/session save [name]`
2. SessionManager updates session metadata in SQLite
3. SessionManager appends any unsaved messages to `messages.jsonl`
4. Agent configurations persisted to SQLite agents table
5. Confirmation displayed to user

**Resume**:
1. User runs `promptchain --session <name>`
2. SessionManager queries SQLite for session metadata
3. SessionManager loads conversation history from `messages.jsonl`
4. ExecutionHistoryManager reconstructed from loaded messages
5. Agent configurations loaded from SQLite agents table
6. PromptChain instances recreated for each agent with correct models (lazy)
7. ChatView populated with conversation history
8. Session continues seamlessly

## Extension Points

The CLI is designed for extensibility:

1. **Custom Commands**: Add new slash commands by extending CommandHandler
2. **Custom Widgets**: Add new Textual widgets for specialized views
3. **Custom File Handlers**: Extend FileLoader for specialized file types
4. **Custom Shell Integration**: Add shell command preprocessing or post-processing
5. **Plugin System**: Future plugin architecture for advanced features (TOON, APReL, APRICOT)

## Related Documentation

- **Main README**: `/README.md` - Project overview and installation
- **CLAUDE.md**: `/CLAUDE.md` - Development guidelines and CLI usage patterns
- **Spec**: `/specs/001-cli-agent-interface/spec.md` - Complete feature specification
- **Tasks**: `/specs/001-cli-agent-interface/tasks.md` - Implementation task breakdown

---

**Version:** CLI v1.0.0 | Framework: Textual 0.83+ | Python: 3.8+

**Last Updated**: 2025-11-18

**Status**: User Stories 1-5 Complete, Phase 8 In Progress (T148-T156)

**Documentation:** [PromptChain GitHub](https://github.com/your-org/promptchain)

**Issues:** [Report Bug](https://github.com/your-org/promptchain/issues)

**License:** MIT
