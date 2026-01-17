# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptChain is a Python library for building sophisticated LLM applications and agent frameworks. It provides two core capabilities:

1. **Advanced Prompt Engineering**: Create, test, and optimize prompts through systematic iteration and evaluation
2. **Flexible LLM Execution**: Chain multiple LLM calls and functions together in sophisticated processing pipelines

## Development Commands

### Core Development
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# Run tests (if pytest is available)
pytest

# Run linting
black . && isort . && flake8

# Type checking
mypy promptchain/
```

### Package Management
```bash
# Build package
python setup.py sdist bdist_wheel

# Install from local build
pip install dist/promptchain-<version>.tar.gz
```

## CLI Usage - Interactive Terminal Interface

PromptChain includes a powerful interactive terminal interface for conversational AI interactions with persistent sessions, multi-agent support, and file context integration.

### Quick Start

```bash
# Launch interactive CLI with default session
promptchain

# Launch with specific session name
promptchain --session my-project

# Launch with custom sessions directory
promptchain --sessions-dir /path/to/sessions
```

### Core Features

**Interactive Chat**: Natural language conversations with AI agents that maintain full context across exchanges.

**Multi-Agent Management**: Create specialized agents with different models (GPT-4, Claude, Gemini, local models via Ollama) for different tasks.

**Session Persistence**: Save and resume conversation sessions with full history, agent configurations, and working directory context.

**File Context Integration**: Reference files and directories using `@syntax` to automatically include content in your prompts.

**Shell Command Execution**: Execute shell commands with `!syntax` and feed output directly into conversation context.

### Available Commands

**Session Management**:
- `/session save [name]` - Save current session with optional name
- `/session list` - List all saved sessions
- `/session delete <name>` - Delete a saved session

**Agent Management**:
- `/agent create <name> --model <model> --description <desc>` - Create new agent
- `/agent list` - List all agents in current session
- `/agent use <name>` - Switch to specific agent
- `/agent delete <name>` - Remove an agent

**System Commands**:
- `/help` - Show available commands
- `/exit` - Exit CLI (or use Ctrl+D)

### Usage Examples

**Basic Conversation**:
```bash
$ promptchain
Welcome to PromptChain CLI!

> What are the main files in this project?
[Agent lists files and explains structure]

> Can you explain what main.py does?
[Agent explains main.py with context from previous message]

> /exit
Goodbye!
```

**Multi-Agent Workflow**:
```bash
$ promptchain --session code-review

> /agent create reviewer --model claude-3-opus --description "Code review specialist"
Created agent 'reviewer' with model claude-3-opus

> /agent create coder --model gpt-4 --description "Code generation specialist"
Created agent 'coder' with model gpt-4

> /agent list
  1. default (gpt-4) - Default agent
  2. reviewer (claude-3-opus) - Code review specialist
  3. coder (gpt-4) - Code generation specialist

> /agent use reviewer
Now using agent: reviewer

> @src/main.py Please review this file for potential issues
[Claude reviews the file content]

> /agent use coder
Now using agent: coder

> Based on the review, please refactor the authentication logic
[GPT-4 generates refactored code]
```

**File Context Integration**:
```bash
> @README.md Summarize the project goals
[Agent reads README.md and provides summary]

> @src/ What authentication patterns are used?
[Agent discovers relevant files in src/ and analyzes patterns]

> @tests/test_auth.py Fix the failing tests
[Agent reads test file and suggests fixes]
```

**Shell Integration**:
```bash
> !git status
[Git status output displayed]

> What files should I commit?
[Agent analyzes git status output and recommends files]

> !pytest tests/
[Test output displayed]

> Fix the failing test in test_auth.py
[Agent analyzes test failures and suggests fix]
```

**Session Persistence**:
```bash
# Day 1
$ promptchain --session feature-auth
> Let's implement JWT authentication
[Long conversation about implementation]
> /session save feature-auth
Session saved: feature-auth

# Day 2
$ promptchain --session feature-auth
[Full conversation history restored]
> Let's continue with the token refresh logic
[Agent has context from previous day]
```

### Model Configuration

Agents support any LiteLLM-compatible model string:

```bash
# OpenAI models
/agent create gpt --model gpt-4
/agent create fast --model gpt-3.5-turbo

# Anthropic models
/agent create claude --model claude-3-opus-20240229
/agent create sonnet --model claude-3-sonnet-20240229

# Google models
/agent create gemini --model gemini/gemini-pro

# Local models via Ollama
/agent create local --model ollama/llama2
/agent create codellama --model ollama/codellama
```

### File Reference Syntax

Use `@` prefix to reference files and directories:

```bash
# Single file
@path/to/file.py

# Multiple files
@src/main.py @tests/test_main.py

# Directory (discovers relevant files)
@src/

# Absolute paths
@/home/user/project/file.txt

# Relative paths (from working directory)
@./README.md
```

**File Truncation**: Large files (>100KB) are automatically truncated with preview of first 500 and last 100 lines.

**Binary Files**: Binary files display metadata instead of content.

### Shell Command Syntax

Use `!` prefix to execute shell commands:

```bash
# Single command
!ls -la

# Command with output captured
!git diff

# Long-running commands show progress
!npm test

# Shell mode (all inputs executed as commands)
!!
ls
pwd
git status
!!  # Exit shell mode
```

**Timeout**: Commands timeout after 30 seconds by default.

**Working Directory**: Commands execute in the session's working directory.

### Environment Setup

Required environment variables for LLM providers:

```bash
# .env file
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

### CLI Architecture

**Components**:
- `promptchain/cli/main.py` - CLI entry point with Click commands
- `promptchain/cli/tui/app.py` - Textual TUI application
- `promptchain/cli/session_manager.py` - Session persistence (SQLite + JSONL)
- `promptchain/cli/command_handler.py` - Slash command parsing and routing
- `promptchain/cli/models/` - Data models (Session, Agent, Message)

**Storage Locations**:
- Sessions: `~/.promptchain/sessions/` (or custom via --sessions-dir)
- SQLite database: `~/.promptchain/sessions/sessions.db`
- Conversation history: `~/.promptchain/sessions/<session-name>/history.jsonl`

**Auto-Save**: Sessions auto-save every 5 messages or 2 minutes (whichever comes first).

### Keyboard Shortcuts

- `Enter` - Send message
- `Ctrl+D` - Exit CLI
- `Ctrl+C` - Cancel current operation (does not exit)
- `Up/Down` - Navigate command history (planned in Phase 8)
- `Tab` - Autocomplete slash commands (planned in Phase 8)
- `Shift+Enter` - Multi-line input (planned in Phase 8)

## Architecture Overview

### Core Components

**PromptChain** (`promptchain/utils/promptchaining.py`): The main orchestration class that manages LLM chains. Handles:
- Sequential execution of instructions (prompts, functions, agentic steps)
- Model management and parameter passing
- Tool calling integration (local functions + MCP tools via MCPHelper)
- Memory management and conversation history
- Chain breaking mechanisms

**AgentChain** (`promptchain/utils/agent_chain.py`): Multi-agent orchestrator with multiple execution modes:
- Router mode: Dynamic agent selection using LLM or custom routing logic
- Pipeline mode: Sequential agent execution
- Round-robin mode: Cyclic agent execution
- Broadcast mode: Parallel execution with result synthesis

**AgenticStepProcessor** (`promptchain/utils/agentic_step_processor.py`): Enables complex agentic workflows within chains:
- Internal reasoning loops with multiple LLM calls
- Tool calling and result processing
- Automatic objective completion detection
- Error handling and clarification attempts

### Supporting Infrastructure

**ExecutionHistoryManager** (`promptchain/utils/execution_history_manager.py`): Advanced history management system:
- Token-aware truncation using tiktoken
- Structured entry types (user_input, agent_output, tool_call, tool_result, etc.)
- Filtering and formatting capabilities
- Automatic memory management to prevent context overflow

**MCPHelper** (`promptchain/utils/mcp_helpers.py`): Model Context Protocol integration:
- Manages connections to external MCP servers
- Tool discovery and schema management
- Automatic tool name prefixing to prevent conflicts
- Async context management for clean resource handling

**RunLogger** (`promptchain/utils/logging_utils.py`): Comprehensive logging system:
- JSONL file logging for detailed analysis
- Console logging with event-based structure
- Session-based log management
- Integration with AgentChain for conversation tracking

### Key Architectural Patterns

**Instruction Processing**: Three types of instructions supported:
- String templates: Processed by LLMs with variable substitution
- Python functions: Executed directly with string input/output
- AgenticStepProcessor instances: Complex multi-step reasoning workflows

**Tool Integration**: Unified tool system with conflict resolution:
- Local tools: Python functions registered directly in PromptChain
- MCP tools: External tools accessed via Model Context Protocol
- Automatic prefixing (`mcp_{server_id}_{tool_name}`) prevents naming conflicts
- Tool schemas standardized to OpenAI format

**Memory Management**: Multi-layered approach:
- ExecutionHistoryManager: Structured, filterable history with token limits
- Conversation history: Basic message storage with truncation
- Per-agent history configuration: Individual history settings per agent (v0.4.2)
- Step storage: Optional detailed step-by-step output tracking
- SQLite caching: Persistent conversation storage for AgentChain

**Async/Sync Pattern**: Consistent dual-interface design:
- All core methods have both sync and async variants
- Sync methods wrap async implementations with `asyncio.run()`
- Context managers handle resource cleanup automatically

## Important Conventions

### Environment Setup
Create `.env` file with API keys:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### Model Specification
Models can be specified as:
- Strings: `"openai/gpt-4"`, `"anthropic/claude-3-sonnet-20240229"`
- Dicts with parameters: `{"name": "openai/gpt-4", "params": {"temperature": 0.5}}`

### Function Integration
Functions must:
- Accept a single string input
- Return a string output
- Be registered via `chain.register_tool_function()` for tool calling

### Async/Sync Patterns
- All core methods have both sync and async variants
- Sync methods wrap async implementations with `asyncio.run()`
- Use async variants when working within async contexts

## Configuration Files

**pyproject.toml**: Build system configuration using setuptools_scm for versioning
**setup.py**: Package metadata and dependencies
**requirements.txt**: Core and development dependencies
**cursor_instruction.txt**: Development guidelines and code style preferences

## Common Workflows

### Basic Chain with History Management
```python
from promptchain import PromptChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

# Create history manager with token limits
history_manager = ExecutionHistoryManager(
    max_tokens=4000,
    max_entries=50,
    truncation_strategy="oldest_first"
)

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}", "Summarize: {input}"],
    verbose=True
)

# Track execution in history
history_manager.add_entry("user_input", "Your analysis request", source="user")
result = chain.process_prompt("Your input here")
history_manager.add_entry("agent_output", result, source="chain")

# Get formatted history
formatted = history_manager.get_formatted_history(
    format_style='chat',
    max_tokens=2000
)
```

### Agentic Step with Tool Integration
```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.tools.ripgrep_wrapper import RipgrepSearcher

# Create search tool
searcher = RipgrepSearcher()

def search_files(query: str) -> str:
    """Search through project files"""
    results = searcher.search(query, search_path="./src")
    return f"Found {len(results)} matches:\n" + "\n".join(results[:10])

# Create agentic step with specific objective
agentic_step = AgenticStepProcessor(
    objective="Find and analyze code patterns in the project",
    max_internal_steps=5,
    model_name="openai/gpt-4"
)

# Create chain with agentic reasoning
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Prepare search strategy: {input}",
        agentic_step,  # Complex reasoning step
        "Final summary: {input}"
    ]
)

# Register search tool
chain.register_tool_function(search_files)
chain.add_tools([{
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search through project files using ripgrep",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}])

result = chain.process_prompt("Find all database connection patterns")
```

### Multi-Agent Setup with MCP Integration
```python
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.logging_utils import RunLogger

# Setup logging
logger = RunLogger(log_dir="./logs")

# MCP server configuration for external tools
mcp_config = [{
    "id": "filesystem",
    "type": "stdio",
    "command": "mcp-server-filesystem",
    "args": ["--root", "./project"]
}]

# Create specialized agents with different capabilities
analyzer_agent = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze the data: {input}"],
    mcp_servers=mcp_config,  # Has file system access
    verbose=True
)

writer_agent = PromptChain(
    models=["anthropic/claude-3-sonnet-20240229"],
    instructions=["Write comprehensive report: {input}"],
    verbose=True
)

# Router configuration with decision templates
router_config = {
    "models": ["openai/gpt-4o-mini"],
    "instructions": [None, "{input}"],
    "decision_prompt_templates": {
        "single_agent_dispatch": """
Based on the user request: {user_input}

Available agents:
{agent_details}

Conversation history:
{history}

Choose the most appropriate agent and return JSON:
{{"chosen_agent": "agent_name", "refined_query": "optional refined query"}}
        """
    }
}

# Multi-agent system with caching and per-agent history configuration
agent_chain = AgentChain(
    agents={"analyzer": analyzer_agent, "writer": writer_agent},
    agent_descriptions={
        "analyzer": "Analyzes data and files with filesystem access",
        "writer": "Creates detailed written reports and documentation"
    },
    execution_mode="router",
    router=router_config,
    cache_config={
        "name": "my_project_session",
        "path": "./cache"
    },
    auto_include_history=True,  # Global history setting
    agent_history_configs={  # Per-agent history overrides (v0.4.2)
        "analyzer": {
            "enabled": True,
            "max_tokens": 8000,
            "max_entries": 20,
            "truncation_strategy": "oldest_first"
        },
        "writer": {
            "enabled": True,
            "max_tokens": 4000,
            "max_entries": 10,
            "truncation_strategy": "keep_last"
        }
    },
    verbose=True
)

# Interactive chat with full logging
await agent_chain.run_chat()
```

### Per-Agent History Configuration (v0.4.2)

The `agent_history_configs` parameter enables fine-grained control over conversation history for each agent, allowing you to optimize token usage and context relevance:

```python
from promptchain.utils.agent_chain import AgentChain

# Create agents with different history needs
code_runner = PromptChain(models=["openai/gpt-4"], instructions=["Execute: {input}"])
analyst = PromptChain(models=["openai/gpt-4"], instructions=["Analyze: {input}"])
writer = PromptChain(models=["openai/gpt-4"], instructions=["Document: {input}"])

agent_chain = AgentChain(
    agents={
        "code_runner": code_runner,
        "analyst": analyst,
        "writer": writer
    },
    auto_include_history=True,  # Global default
    agent_history_configs={
        # Terminal/execution agents: No history needed (saves 30-60% tokens)
        "code_runner": {
            "enabled": False  # Completely disable history for this agent
        },
        # Research/analysis agents: Full history for context
        "analyst": {
            "enabled": True,
            "max_tokens": 8000,  # Higher limit for detailed analysis
            "max_entries": 50,
            "truncation_strategy": "oldest_first",
            "include_types": ["user_input", "agent_output"],  # Filter history types
            "exclude_sources": ["system"]  # Exclude certain sources
        },
        # Documentation agents: Full history for comprehensive context
        "writer": {
            "enabled": True,
            "max_tokens": 6000,
            "max_entries": 30,
            "truncation_strategy": "keep_last"
        }
    }
)
```

**Configuration Options:**

- `enabled` (bool): Whether to include history for this agent (default: uses `auto_include_history`)
- `max_tokens` (int): Maximum tokens for history (uses ExecutionHistoryManager's token counting)
- `max_entries` (int): Maximum number of history entries
- `truncation_strategy` (str): How to truncate when limits exceeded ("oldest_first" or "keep_last")
- `include_types` (List[str]): Only include specific entry types (e.g., ["user_input", "agent_output"])
- `exclude_sources` (List[str]): Exclude entries from specific sources

**Token Savings Example:**

```python
# 6-agent system with selective history
agent_chain = AgentChain(
    agents={
        "terminal": terminal_agent,      # No history: -60% tokens
        "coding": coding_agent,          # Limited history: -40% tokens
        "research": research_agent,      # Full history
        "analysis": analysis_agent,      # Full history
        "documentation": doc_agent,      # Full history
        "synthesis": synthesis_agent     # Full history
    },
    agent_history_configs={
        "terminal": {"enabled": False},           # Saves ~3000 tokens per call
        "coding": {"enabled": True, "max_entries": 10},  # Saves ~2000 tokens per call
        # Others use full history
    }
)
# Total savings: ~5000 tokens per multi-agent conversation turn
```

**AgenticStepProcessor Internal History Isolation:**

Note that `agent_history_configs` controls **conversation-level history** (user inputs, agent outputs between agents), NOT the internal reasoning history within an `AgenticStepProcessor`. The AgenticStepProcessor maintains its own isolated history for multi-hop reasoning based on its `history_mode` parameter:

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# AgenticStepProcessor has its own internal history system
research_step = AgenticStepProcessor(
    objective="Research with multi-hop reasoning",
    max_internal_steps=8,
    history_mode="progressive"  # Internal reasoning history (not conversation history)
)

# This controls what the AGENT sees, not what's inside AgenticStepProcessor
agent_chain = AgentChain(
    agents={"research": PromptChain(instructions=[research_step])},
    agent_history_configs={
        "research": {
            "enabled": True,  # Controls conversation history passed TO the agent
            "max_tokens": 8000  # Limits conversation context, not internal reasoning
        }
    }
)
```

### Advanced Workflow Integration
```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.logging_utils import RunLogger
import asyncio

async def complex_analysis_workflow():
    # Initialize components
    history_manager = ExecutionHistoryManager(max_tokens=8000)
    logger = RunLogger(log_dir="./analysis_logs")
    
    # Create analysis pipeline
    data_processor = AgenticStepProcessor(
        objective="Process and validate input data",
        max_internal_steps=3
    )
    
    analysis_chain = PromptChain(
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        instructions=[
            data_processor,
            "Perform detailed analysis: {input}",
            "Generate actionable insights: {input}"
        ],
        store_steps=True,  # Keep step outputs
        verbose=True
    )
    
    # Execute workflow
    initial_data = "Complex dataset requiring multi-step analysis"
    history_manager.add_entry("user_input", initial_data, source="workflow")
    
    try:
        result = await analysis_chain.process_prompt_async(initial_data)
        history_manager.add_entry("agent_output", result, source="analysis_chain")
        
        # Log workflow completion
        logger.log_run({
            "event": "workflow_completed",
            "result_length": len(result),
            "steps_executed": len(analysis_chain.step_outputs)
        })
        
        # Get comprehensive history
        workflow_summary = history_manager.get_formatted_history(
            include_types=["user_input", "agent_output"],
            format_style="full_json"
        )
        
        return {
            "result": result,
            "step_details": analysis_chain.step_outputs,
            "history_summary": workflow_summary
        }
        
    except Exception as e:
        history_manager.add_entry("error", str(e), source="workflow")
        logger.log_run({"event": "workflow_error", "error": str(e)})
        raise

# Run the workflow
result = asyncio.run(complex_analysis_workflow())
```

## How Components Work Together

The PromptChain ecosystem follows a layered architecture where components complement each other:

### Execution Flow Integration
1. **AgentChain** receives user input and uses router logic to select an appropriate agent
2. Selected **PromptChain** processes instructions sequentially:
   - String templates → LLM calls via LiteLLM
   - Functions → Direct Python execution
   - **AgenticStepProcessor** → Internal reasoning loops with tool access
3. **MCPHelper** manages external tool discovery and execution
4. **ExecutionHistoryManager** tracks all interactions with token-aware memory management
5. **RunLogger** captures events and detailed execution data

### Data Flow Example
```
User Input → AgentChain Router → PromptChain Selection
    ↓
PromptChain Instructions:
    ↓
1. String prompt → LLM → Response
    ↓
2. AgenticStepProcessor:
   - Internal LLM reasoning
   - Tool calls via MCPHelper or local functions
   - Multi-step objective completion
    ↓
3. Final processing → LLM → Final result
    ↓
All steps logged by RunLogger and tracked in ExecutionHistoryManager
```

### Component Interactions
- **AgenticStepProcessor** uses PromptChain's tool ecosystem for reasoning
- **MCPHelper** provides tool discovery that integrates with both PromptChain and AgenticStepProcessor
- **ExecutionHistoryManager** works with all components to maintain context
- **RunLogger** captures events from AgentChain conversations and workflow executions
- **RipgrepSearcher** exemplifies local tool integration for file system operations

## Key Dependencies

### Core Libraries
- **litellm**: Unified LLM API access (OpenAI, Anthropic, etc.)
- **python-dotenv**: Environment variable management
- **asyncio**: Async/await support for concurrent operations

### Advanced Features
- **tiktoken**: Accurate token counting for history management
- **rich**: Enhanced console output for AgentChain chat interface
- **sqlite3**: Persistent conversation caching and session management
- **pydantic**: Data validation (optional, used by some model providers)

### External Tool Integration
- **mcp**: Model Context Protocol for external tool access
- **subprocess**: For tools like RipgrepSearcher that wrap command-line utilities

### Development Dependencies
- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

## Testing Strategy

When making changes, test the integration points:

1. **Basic Functionality**: Simple PromptChain with string instructions
2. **Function Integration**: Register and call local Python functions
3. **Agentic Workflows**: AgenticStepProcessor with tool access
4. **Multi-Agent Systems**: AgentChain routing and execution modes  
5. **Memory Management**: ExecutionHistoryManager with token limits
6. **MCP Integration**: External tool discovery and execution
7. **Logging**: RunLogger JSONL output and console formatting
8. **Async/Sync Compatibility**: Both execution modes work correctly

## Known Limitations

- **Token Management**: Varies by provider, requires manual monitoring
- **MCP Setup**: External MCP servers need separate installation and configuration
- **API Dependencies**: Heavy reliance on external service availability and rate limits
- **Sequential Processing**: Complex chains can be slow due to step-by-step execution
- **Tool Conflicts**: MCP and local tools require careful naming to avoid conflicts
- **Memory Overhead**: ExecutionHistoryManager and detailed logging can consume significant memory for long sessions
- from now own the current year is 2025

## Active Technologies
- Python 3.8+ (compatible with existing PromptChain codebase) + Textual 0.83+ (TUI framework), Rich 13.8+ (terminal formatting), Click 8.1+ (CLI framework), LiteLLM 1.0+ (existing), asyncio (stdlib), prompt_toolkit 3.0+ (input handling) (001-cli-agent-interface)
- SQLite 3 (session persistence via existing AgentChain cache_config pattern), JSON/JSONL (session exports, logs) (001-cli-agent-interface)
- Python 3.8+ (existing PromptChain codebase compatibility) (002-cli-orchestration)
- SQLite 3 (existing session persistence via AgentChain cache_config pattern), JSONL files (conversation logs, execution history) (002-cli-orchestration)
- Python 3.8+ (compatible with existing PromptChain codebase) + SQLite3 (existing), Textual (existing TUI), LiteLLM (existing), asyncio (stdlib) (003-multi-agent-communication)
- SQLite (extend existing sessions.db with new tables) (003-multi-agent-communication)
- Python 3.8+ (compatible with existing PromptChain codebase) + LiteLLM (existing), asyncio (stdlib), existing 003 infrastructure (MessageBus, Blackboard, CapabilityRegistry, TaskDelegation) (004-advanced-agentic-patterns)
- SQLite (existing session persistence), In-memory cache (speculative execution) (004-advanced-agentic-patterns)

## Development Orchestration Protocol

**CONSTITUTION PRINCIPLE VIII: Distributed Execution with File Locking**

This protocol governs how multi-agent development sessions execute tasks efficiently while preventing conflicts.

### Core Principles

1. **Token Efficiency Through Delegation**: Spawn specialized agents for task execution to preserve main agent context window
2. **Parallel Execution Where Safe**: Execute independent tasks concurrently via multiple agents
3. **File Locking Prevents Conflicts**: Explicit file ownership prevents simultaneous edits
4. **Milestone Synchronization**: Checkpoint state to memory-bank and git after each phase/wave

### Pre-Phase Analysis (MANDATORY)

Before executing ANY phase in tasks.md, the main orchestrator MUST:

```
1. ANALYZE dependency graph for all tasks in phase
2. IDENTIFY sequential dependencies (task B requires task A output)
3. GROUP independent tasks into parallel execution waves
4. ASSIGN file ownership to prevent edit conflicts
5. DOCUMENT wave structure before spawning agents
```

### Wave Execution Format

Tasks within a phase are grouped into waves for parallel execution:

```markdown
### Phase N: [Phase Name]

**Wave 1** (parallel - 3 agents):
| Task | Agent | Files Owned | Dependencies |
|------|-------|-------------|--------------|
| T001 | python-pro | models/task.py | None |
| T002 | python-pro | models/blackboard.py | None |
| T003 | sql-pro | schema.sql | None |

**Wave 2** (parallel - 2 agents, after Wave 1):
| Task | Agent | Files Owned | Dependencies |
|------|-------|-------------|--------------|
| T004 | python-pro | session_manager.py | T001, T002, T003 |
| T005 | test-automator | tests/test_models.py | T001, T002 |

**Wave 3** (sequential - requires all previous):
| Task | Agent | Files Owned | Dependencies |
|------|-------|-------------|--------------|
| T006 | code-reviewer | ALL | T004, T005 |
```

### File Locking System

**File Ownership Rules**:
- Each file can have ONLY ONE owner per wave
- Owner is assigned when agent spawns
- Ownership releases when agent completes task
- NO agent may edit files they don't own

**Checkout Format** (in agent spawn prompt):
```
FILE OWNERSHIP FOR THIS TASK:
- EXCLUSIVE: path/to/file.py (you may edit)
- EXCLUSIVE: path/to/other.py (you may edit)
- READ-ONLY: path/to/dependency.py (reference only)
- FORBIDDEN: path/to/locked.py (another agent owns)
```

**Conflict Prevention**:
```python
# Main orchestrator tracks ownership
file_locks = {
    "models/task.py": {"owner": "agent-1", "task": "T001", "status": "locked"},
    "models/blackboard.py": {"owner": "agent-2", "task": "T002", "status": "locked"},
    "session_manager.py": {"owner": None, "task": None, "status": "available"}
}
```

### Agent Spawning Strategy

**WITHIN Phases/Waves**: ALWAYS spawn agents for task execution
```
- Use Task tool with appropriate subagent_type
- Pass file ownership in prompt
- Agent executes task autonomously
- Returns completion report
```

**BETWEEN Phases**: Main agent orchestrates (NO spawning)
```
- Validate all tasks in previous phase completed
- Run checkpoint sync (memory + git)
- Analyze next phase for wave structure
- Prepare file ownership assignments
```

### Checkpoint Synchronization

After EACH phase completion, the main orchestrator MUST:

```markdown
**Phase N Complete - Checkpoint**

1. SPAWN memory-bank-keeper agent:
   - Update progress.md with completed tasks
   - Update activeContext.md with current state
   - Document any decisions or blockers

2. SPAWN git-version-manager agent:
   - Stage all modified files
   - Create semantic commit: "feat(003): Complete Phase N - [description]"
   - Tag if milestone: v0.X.0-alpha.N

3. VERIFY checkpoint:
   - All files saved and committed
   - No uncommitted changes
   - Memory bank reflects current state

4. PROCEED to Phase N+1 analysis
```

### Task Markers in tasks.md

Enhanced task format with orchestration metadata:

```markdown
- [ ] T001 [P] [W1] [US1] [python-pro] [models/task.py] Create Task dataclass
      ^     ^    ^    ^        ^              ^
      |     |    |    |        |              └── File ownership
      |     |    |    |        └── Recommended agent type
      |     |    |    └── User story reference
      |     |    └── Wave number within phase
      |     └── [P] = Parallelizable (independent)
      └── Task ID
```

### Execution Flow Example

```
Phase 2: Task Delegation Protocol
├── PRE-PHASE: Analyze 5 tasks, identify 2 waves
│
├── Wave 1 (spawn 3 agents in parallel):
│   ├── Agent 1: T008 [models/task.py]
│   ├── Agent 2: T009 [tools/library/delegation_tools.py]
│   └── Agent 3: T010 [schema.sql additions]
│
├── WAVE-SYNC: Verify all 3 complete, no conflicts
│
├── Wave 2 (spawn 2 agents in parallel):
│   ├── Agent 4: T011 [session_manager.py] depends on T008, T010
│   └── Agent 5: T012 [tests/] depends on T008, T009
│
├── PHASE-COMPLETE: All tasks done
│
└── CHECKPOINT:
    ├── memory-bank-keeper: Update progress
    └── git-version-manager: Commit phase
```

### Error Handling

**Agent Failure**:
```
1. Log failure with full context
2. Release file locks held by failed agent
3. Determine if task can be retried
4. If retry: spawn new agent with same ownership
5. If blocked: mark task failed, continue with independent tasks
```

**File Conflict Detected**:
```
1. HALT all agents in current wave
2. Identify conflict source
3. Rollback conflicting changes
4. Re-analyze wave structure
5. Resume with corrected ownership
```

### Performance Metrics

Track orchestration efficiency:
- **Parallelization Rate**: % of tasks executed in parallel
- **Context Savings**: Tokens saved by delegation vs inline execution
- **Checkpoint Overhead**: Time spent on sync vs execution
- **Conflict Rate**: File conflicts per phase (target: 0)

## Recent Changes
- 004-advanced-agentic-patterns: Added Python 3.8+ (compatible with existing PromptChain codebase) + LiteLLM (existing), asyncio (stdlib), existing 003 infrastructure (MessageBus, Blackboard, CapabilityRegistry, TaskDelegation)
- 001-cli-agent-interface: Added Python 3.8+ (compatible with existing PromptChain codebase) + Textual 0.83+ (TUI framework), Rich 13.8+ (terminal formatting), Click 8.1+ (CLI framework), LiteLLM 1.0+ (existing), asyncio (stdlib), prompt_toolkit 3.0+ (input handling)
