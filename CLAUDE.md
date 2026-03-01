# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PromptChain is a Python library for building sophisticated LLM applications and agent frameworks with:
1. **Advanced Prompt Engineering**: Create, test, and optimize prompts through systematic iteration
2. **Flexible LLM Execution**: Chain multiple LLM calls and functions in sophisticated pipelines

## Development Commands

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Run linting
black . && isort . && flake8

# Type checking
mypy promptchain/
```

## CLI Usage

PromptChain includes an interactive TUI with session persistence, multi-agent support, and file context integration.

**Quick Start:**
```bash
promptchain                          # Launch with default session
promptchain --session my-project    # Named session
```

**Key Features:**
- Interactive chat with context preservation
- Multi-agent management with different models
- File context with `@syntax` (e.g., `@src/main.py`)
- Shell integration with `!syntax` (e.g., `!git status`)
- Session persistence via SQLite + JSONL

**Essential Commands:**
- `/session save [name]` - Save session
- `/agent create <name> --model <model>` - Create agent
- `/agent use <name>` - Switch agent
- `/help` - Show all commands

**Storage:** `~/.promptchain/sessions/` (SQLite DB + JSONL history)

For detailed CLI documentation, see README.md or run `/help` in the CLI.

## Architecture Overview

### Core Components

**PromptChain** (`promptchain/utils/promptchaining.py`): Main orchestration class
- Sequential instruction execution (prompts, functions, agentic steps)
- Model management and parameter passing
- Tool calling integration (local + MCP tools via MCPHelper)
- Memory management and conversation history

**AgentChain** (`promptchain/utils/agent_chain.py`): Multi-agent orchestrator
- Router mode: Dynamic agent selection via LLM
- Pipeline mode: Sequential agent execution
- Round-robin mode: Cyclic execution
- Broadcast mode: Parallel execution with synthesis

**AgenticStepProcessor** (`promptchain/utils/agentic_step_processor.py`): Complex agentic workflows
- Internal reasoning loops with multiple LLM calls
- Tool calling and result processing
- Automatic objective completion detection
- Error handling and clarification

### Supporting Infrastructure

**ExecutionHistoryManager** (`promptchain/utils/execution_history_manager.py`): Advanced history management
- Token-aware truncation using tiktoken
- Structured entry types (user_input, agent_output, tool_call, tool_result)
- Filtering and formatting capabilities
- Automatic memory management

**MCPHelper** (`promptchain/utils/mcp_helpers.py`): Model Context Protocol integration
- External MCP server connections
- Tool discovery and schema management
- Automatic tool name prefixing to prevent conflicts

**RunLogger** (`promptchain/utils/logging_utils.py`): Comprehensive logging
- JSONL file logging for analysis
- Console logging with event structure
- Session-based log management

### Key Architectural Patterns

**Instruction Processing**: Three instruction types
- String templates → LLM calls with variable substitution
- Python functions → Direct execution
- AgenticStepProcessor instances → Multi-step reasoning workflows

**Tool Integration**: Unified tool system
- Local tools: Python functions registered in PromptChain
- MCP tools: External tools via Model Context Protocol
- Automatic prefixing: `mcp_{server_id}_{tool_name}` prevents conflicts

**Memory Management**: Multi-layered approach
- ExecutionHistoryManager: Structured, filterable history with token limits
- Conversation history: Basic message storage with truncation
- Per-agent history configuration: Individual settings per agent (v0.4.2)
- SQLite caching: Persistent conversation storage for AgentChain

**Async/Sync Pattern**: Consistent dual-interface design
- All core methods have both sync and async variants
- Sync methods wrap async implementations with `asyncio.run()`

## Important Conventions

### Environment Setup
Create `.env` file with API keys:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_anthropic_key
```

### Model Specification
- Strings: `"openai/gpt-4"`, `"anthropic/claude-3-sonnet-20240229"`
- Dicts: `{"name": "openai/gpt-4", "params": {"temperature": 0.5}}`

### Function Integration
Functions must:
- Accept a single string input
- Return a string output
- Be registered via `chain.register_tool_function()` for tool calling

## Configuration Files

- **pyproject.toml**: Build system configuration using setuptools_scm
- **setup.py**: Package metadata and dependencies
- **requirements.txt**: Core and development dependencies

## Common Workflows

For detailed code examples, see `examples/` directory:
- `examples/basic_chain.py` - Basic chain with history management
- `examples/agentic_step.py` - Agentic step with tool integration
- `examples/multi_agent.py` - Multi-agent setup with MCP integration
- `examples/advanced_workflow.py` - Advanced workflow integration

### Per-Agent History Configuration (v0.4.2)

`agent_history_configs` enables fine-grained control over conversation history per agent:

```python
agent_chain = AgentChain(
    agents={"terminal": terminal_agent, "analyst": analyst_agent},
    agent_history_configs={
        "terminal": {"enabled": False},  # Saves ~3000 tokens per call
        "analyst": {
            "enabled": True,
            "max_tokens": 8000,
            "max_entries": 50,
            "truncation_strategy": "oldest_first"
        }
    }
)
```

**Configuration Options:**
- `enabled` (bool): Include history for this agent
- `max_tokens` (int): Maximum tokens for history
- `max_entries` (int): Maximum history entries
- `truncation_strategy` (str): "oldest_first" or "keep_last"
- `include_types` (List[str]): Filter entry types
- `exclude_sources` (List[str]): Exclude specific sources

**Token Savings:** Terminal/execution agents with disabled history save 30-60% tokens.

## Component Interactions

**Execution Flow:**
1. AgentChain receives input → Router selects agent
2. PromptChain processes instructions sequentially
3. MCPHelper manages external tool discovery/execution
4. ExecutionHistoryManager tracks all interactions with token limits
5. RunLogger captures events and detailed execution data

**Integration Points:**
- AgenticStepProcessor uses PromptChain's tool ecosystem
- MCPHelper integrates with both PromptChain and AgenticStepProcessor
- ExecutionHistoryManager works with all components for context
- RunLogger captures AgentChain conversations and workflow executions

## Key Dependencies

### Core
- **litellm**: Unified LLM API access
- **python-dotenv**: Environment variable management
- **asyncio**: Async/await support

### Advanced Features
- **tiktoken**: Token counting for history management
- **rich**: Enhanced console output for AgentChain
- **sqlite3**: Persistent conversation caching
- **mcp**: Model Context Protocol for external tools

### Development
- **pytest**: Testing framework
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

## Testing Strategy

Test integration points:
1. Basic PromptChain with string instructions
2. Function integration with local Python functions
3. AgenticStepProcessor with tool access
4. AgentChain routing and execution modes
5. ExecutionHistoryManager with token limits
6. MCP integration for external tools
7. RunLogger JSONL output and console formatting
8. Async/sync compatibility

## Known Limitations

- **Token Management**: Varies by provider, requires manual monitoring
- **MCP Setup**: External MCP servers need separate installation
- **API Dependencies**: Heavy reliance on external service availability
- **Sequential Processing**: Complex chains can be slow
- **Tool Conflicts**: MCP and local tools require careful naming
- **Memory Overhead**: ExecutionHistoryManager and detailed logging consume memory for long sessions

## Active Technologies
- Python 3.12.11 + mypy 1.16.1, litellm 1.0+, Textual 0.83+, LightRAG (hybridrag) (008-type-safety-debt-pt2)
- N/A (annotation-only fixes; no data persistence changes) (008-type-safety-debt-pt2)

- Python 3.8+ + Textual 0.83+ (TUI), Rich 13.8+, Click 8.1+, LiteLLM 1.0+
- SQLite 3 (session persistence), JSON/JSONL (exports, logs)
- asyncio (stdlib), prompt_toolkit 3.0+ (input handling)

## Development Orchestration Protocol

**CONSTITUTION PRINCIPLE VIII: Distributed Execution with File Locking**

### Core Principles

1. **Token Efficiency Through Delegation**: Spawn specialized agents for task execution to preserve main agent context window
2. **Parallel Execution Where Safe**: Execute independent tasks concurrently via multiple agents
3. **File Locking Prevents Conflicts**: Explicit file ownership prevents simultaneous edits
4. **Milestone Synchronization**: Checkpoint state to memory-bank and git after each phase/wave

### Pre-Phase Analysis (MANDATORY)

Before executing ANY phase in tasks.md:
1. ANALYZE dependency graph for all tasks in phase
2. IDENTIFY sequential dependencies (task B requires task A output)
3. GROUP independent tasks into parallel execution waves
4. ASSIGN file ownership to prevent edit conflicts
5. DOCUMENT wave structure before spawning agents

### Wave Execution

Tasks within phases are grouped into waves for parallel execution:
- **Wave 1**: Independent tasks execute in parallel (multiple agents)
- **Wave 2**: Tasks dependent on Wave 1 execute in parallel
- **Wave 3**: Tasks dependent on previous waves

### File Locking Rules

**Ownership:**
- Each file has ONLY ONE owner per wave
- Owner assigned when agent spawns
- Ownership releases when agent completes task
- NO agent may edit files they don't own

**Checkout Format** (in agent spawn prompt):
```
FILE OWNERSHIP FOR THIS TASK:
- EXCLUSIVE: path/to/file.py (you may edit)
- READ-ONLY: path/to/dependency.py (reference only)
- FORBIDDEN: path/to/locked.py (another agent owns)
```

### Agent Spawning Strategy

**WITHIN Phases/Waves**: ALWAYS spawn agents for task execution
- Use Task tool with appropriate subagent_type
- Pass file ownership in prompt
- Agent executes task autonomously
- Returns completion report

**BETWEEN Phases**: Main agent orchestrates (NO spawning)
- Validate all tasks in previous phase completed
- Run checkpoint sync (memory + git)
- Analyze next phase for wave structure
- Prepare file ownership assignments

### Checkpoint Synchronization

After EACH phase completion:

1. **SPAWN memory-bank-keeper agent:**
   - Update progress.md with completed tasks
   - Update activeContext.md with current state
   - Document decisions or blockers

2. **SPAWN git-version-manager agent:**
   - Stage all modified files
   - Create semantic commit: "feat(003): Complete Phase N - [description]"
   - Tag if milestone: v0.X.0-alpha.N

3. **VERIFY checkpoint:**
   - All files saved and committed
   - No uncommitted changes
   - Memory bank reflects current state

4. **PROCEED to Phase N+1 analysis**

### Task Markers in tasks.md

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

### Error Handling

**Agent Failure:**
1. Log failure with full context
2. Release file locks held by failed agent
3. Determine if task can be retried
4. If retry: spawn new agent with same ownership
5. If blocked: mark task failed, continue with independent tasks

**File Conflict Detected:**
1. HALT all agents in current wave
2. Identify conflict source
3. Rollback conflicting changes
4. Re-analyze wave structure
5. Resume with corrected ownership

### Performance Metrics

Track orchestration efficiency:
- **Parallelization Rate**: % of tasks executed in parallel
- **Context Savings**: Tokens saved by delegation vs inline execution
- **Checkpoint Overhead**: Time spent on sync vs execution
- **Conflict Rate**: File conflicts per phase (target: 0)

## Recent Changes
- 008-type-safety-debt-pt2: Complete (41/41 tasks). Fixed 4 high-error files to 0 mypy errors each: state_agent.py (94→0), app.py (63→0), promptchaining.py (32→0), executors.py (31→0). Project total: 421→213 errors (49% reduction). Zero new test regressions. All SC met.
- 007-type-safety-debt: Complete (18/18 tasks). Fixed type errors across 20+ files. Reduced mypy errors 557→421 (24% reduction). Zero regressions.
- 006-promptchain-improvements: Complete (60/61 tasks). Added AsyncAgentInbox, JanitorAgent, ContextDistiller, PubSubBus, MicroCheckpoints, steering injection, global override, TUI interrupt command. Fixed Gemini MCP param bugs, TUI event loop, JSON parser robustness, MLflow shutdown, config cache. All 44 unit+integration tests green.


