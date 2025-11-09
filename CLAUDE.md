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