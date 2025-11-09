# PromptChain

A flexible prompt engineering library designed to create, refine, and specialize prompts for LLM applications and agent frameworks. PromptChain excels at creating specialized prompts that can be injected into agent frameworks and building sophisticated LLM processing pipelines.

## Primary Use Cases

### 🎯 Agent Framework Integration
Create specialized prompts for agent frameworks like AutoGen, LangChain, and CrewAI:

```python
from promptchain import PromptChain

# Generate optimized prompts for agents
prompt_engineer = PromptChain(
    models=["anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
    instructions=[
        "Analyze this agent task: {input}",
        "Create a specialized prompt with examples and constraints: {input}"
    ]
)

specialized_prompt = prompt_engineer.process_prompt("Research and summarize AI papers")

# Use with any agent framework
from autogen import AssistantAgent
agent = AssistantAgent(
    name="researcher", 
    system_message=specialized_prompt,
    llm_config={"config_list": [...]}
)
```

### ⚙️ LLM Processing Chains
Chain multiple LLM calls and functions for complex workflows:

```python
# Sequential processing with multiple models
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Extract key information from: {input}",
        "Create detailed analysis: {input}",
        "Generate actionable insights: {input}"
    ]
)

result = chain.process_prompt("Your content here...")
```

### 🔧 Custom Function Integration
Combine LLMs with custom processing functions:

```python
def validate_output(text: str) -> str:
    # Custom validation logic
    return f"Validated: {text}"

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Generate content: {input}",
        validate_output,  # Custom function
        "Finalize based on validation: {input}"
    ]
)
```

## Installation

```bash
# From PyPI (coming soon)
pip install promptchain

# From GitHub
pip install git+https://github.com/gyasis/promptchain.git

# Development installation
git clone https://github.com/gyasis/promptchain.git
cd promptchain
pip install -e .
```

## Quick Start

1. **Set up your environment:**
```bash
# Create .env file with your API keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

2. **Simple example:**
```python
from promptchain import PromptChain

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze and improve this text: {input}"]
)

result = chain.process_prompt("Write about quantum computing")
print(result)
```

## Key Features

- **Flexible Chain Construction**: Create sequences of LLM calls and functions
- **Multiple Model Support**: Use different models (GPT-4, Claude, etc.) in the same chain
- **Function Injection**: Insert custom processing between LLM calls
- **Advanced Prompt Engineering**: Systematic prompt optimization and testing
- **Agent Framework Integration**: Generate specialized prompts for any agent system
- **Async Support**: Full async/await support for modern applications
- **Memory Management**: State persistence and conversation history
- **🆕 Comprehensive Observability**: Event system, execution metadata, and monitoring (v0.4.1h)
- **🆕 Accurate Metrics Tracking**: Router steps, tools called, token usage tracking (v0.4.2)
- **🆕 Dual-Mode Logging**: Clean terminal output with full debug logs to file (v0.4.2)
- **🆕 Enhanced Orchestrator**: 5-step reasoning with execution vs knowledge detection (v0.4.2)

## What's New in v0.4.2 (2025-10-07)

### 🎯 Accurate Orchestrator Metrics
Now get real-time visibility into orchestrator performance:

```python
from promptchain.utils.agent_chain import AgentChain

agent_chain = AgentChain(
    agents={"terminal": terminal_agent, "research": research_agent},
    execution_mode="router"
)

result = await agent_chain.process_request("Check the current date")

# Access accurate metrics
print(f"Orchestrator Steps: {result.router_steps}")    # e.g., 5 (was always 0)
print(f"Tools Called: {result.tools_called}")         # e.g., 2 (was always 0)
print(f"Total Tokens: {result.total_tokens}")         # e.g., 1842 (was always None)
```

### 🧹 Clean Terminal Output
Dual-mode logging keeps your terminal clean while preserving full logs:

```bash
# Normal mode: Clean terminal + full DEBUG file logs
python agentic_chat/agentic_team_chat.py

# Development mode: Full observability in terminal
python agentic_chat/agentic_team_chat.py --dev

# Quiet mode: Errors/warnings only
python agentic_chat/agentic_team_chat.py --quiet
```

Logs automatically saved to `./agentic_chat/logs/YYYY-MM-DD/session_HHMMSS.log`

### 🧠 Smarter Agent Selection
Enhanced 5-step orchestrator reasoning distinguishes knowledge vs execution queries:

```python
# Knowledge query → Documentation agent (no tools needed)
result1 = await agent_chain.process_request("What year is it?")
# Orchestrator detects: KNOWLEDGE query, uses system context
# Selects: documentation agent

# Execution query → Terminal agent (runs actual command)
result2 = await agent_chain.process_request("Check the date")
# Orchestrator detects: EXECUTION query (verb: "check")
# Selects: terminal agent with execute_terminal_command tool
```

**See [CHANGELOG.md](CHANGELOG.md#042---2025-10-07) for complete v0.4.2 details**

## Observability System (v0.4.1h)

PromptChain now includes a comprehensive observability system for production monitoring and debugging:

### Event-Based Monitoring
```python
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

def performance_monitor(event: ExecutionEvent):
    if "execution_time_ms" in event.metadata:
        print(f"{event.event_type.name}: {event.metadata['execution_time_ms']}ms")

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}", "Summarize: {input}"]
)

# Register callback for real-time monitoring
chain.register_callback(
    performance_monitor,
    event_filter=[ExecutionEventType.STEP_END, ExecutionEventType.MODEL_CALL_END]
)

result = chain.process_prompt("Your input")
```

### Rich Execution Metadata
```python
from promptchain.utils.agent_chain import AgentChain

# Get comprehensive execution metadata
result = agent_chain.process_input(
    "Analyze quarterly data",
    return_metadata=True  # Returns AgentExecutionResult dataclass
)

print(f"Agent: {result.agent_name}")
print(f"Execution time: {result.execution_time_ms}ms")
print(f"Total tokens: {result.total_tokens}")
print(f"Tools called: {len(result.tools_called)}")
print(f"Router decision: {result.router_decision}")
```

### Public APIs for History Management
```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(max_tokens=4000)

# Use stable public APIs (no more private attributes!)
print(f"Current tokens: {history.current_token_count}")
print(f"History size: {history.history_size}")

# Get comprehensive statistics
stats = history.get_statistics()
print(f"Memory usage: {stats['memory_usage_bytes']} bytes")
```

**Features:**
- **33+ Event Types**: Complete lifecycle coverage (chain, step, model, tool, agentic, MCP)
- **Event Filtering**: Subscribe to specific events to reduce overhead
- **Async/Sync Callbacks**: Both patterns fully supported
- **Execution Metadata**: Rich dataclasses with timing, tokens, tools, errors
- **Public APIs**: Stable, tested APIs replacing private attributes
- **Zero Overhead**: No performance impact when features not used
- **Backward Compatible**: All features are opt-in

📚 **[Observability Documentation](docs/observability/README.md)** - Complete guide with examples

## Documentation

📚 **[Complete Documentation Index](docs/index.md)** - Start here for comprehensive guides

### Quick Links
- **[Quick Start Guide](docs/promptchain_quickstart_multistep.md)** - Get up and running in minutes
- **[Complete Framework Guide](docs/promptchain_guide.md)** - Comprehensive usage documentation
- **[Observability System](docs/observability/README.md)** - 🆕 Event system, metadata, and monitoring (v0.4.1h)
- **[Model Management System](docs/model_management.md)** - Automatic VRAM optimization for large models
- **[Prompt Engineering Guide](docs/prompt_engineer.md)** - Advanced prompt optimization
- **[Agent Integration Examples](docs/AgentChain_Usage.md)** - Using with agent frameworks
- **[Function Integration](docs/registering_functions_to_chain.md)** - Custom function examples
- **[Advanced Techniques](docs/chainbreaker_guide.md)** - Chain breakers, circular processing, etc.

## Requirements

- Python 3.8+
- litellm
- python-dotenv

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LiteLLM](https://github.com/BerriAI/litellm) for model integration
- OpenAI and Anthropic for LLM APIs