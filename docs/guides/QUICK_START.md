# Quick Start Guide

Get started with PromptChain in minutes.

## Installation

Choose the installation mode that fits your needs:

```bash
# Most users - core library only
pip install promptchain

# Interactive CLI users
pip install "promptchain[cli]"

# Developers and contributors
pip install "promptchain[dev]"
```

## Basic Usage

### 1. Simple Chain

```python
from promptchain import PromptChain

# Create a simple processing chain
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Analyze this text: {input}",
        "Provide 3 key insights: {input}"
    ]
)

# Process input
result = chain.process_prompt("Your text here")
print(result)
```

### 2. Multi-Agent System

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain

# Create specialized agents
analyst = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}"]
)

writer = PromptChain(
    models=["anthropic/claude-3-sonnet-20240229"],
    instructions=["Write report: {input}"]
)

# Create multi-agent system
system = AgentChain(
    agents={"analyst": analyst, "writer": writer},
    agent_descriptions={
        "analyst": "Data analysis specialist",
        "writer": "Report writing specialist"
    },
    execution_mode="router"
)

# Interactive chat
import asyncio
asyncio.run(system.run_chat())
```

### 3. With Tool Integration

```python
from promptchain import PromptChain

def search_files(query: str) -> str:
    """Search through project files"""
    # Your search logic here
    return "Search results..."

# Register tool
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Find relevant files: {input}"]
)

chain.register_tool_function(search_files)
chain.add_tools([{
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search project files",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
}])

result = chain.process_prompt("Find authentication code")
```

### 4. Interactive CLI

```bash
# Launch interactive terminal
promptchain

# Create specialized agent
> /agent create coder --model gpt-4 --description "Code specialist"

# Reference files in chat
> @src/main.py Explain this file

# Execute shell commands
> !git status

# Save session
> /session save my-project
```

## Configuration

Create a `.env` file in your project:

```bash
# Required: LLM API keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional: MLflow tracking (only if you installed with [dev])
PROMPTCHAIN_MLFLOW_ENABLED=false
MLFLOW_TRACKING_URI=http://localhost:5000
```

## MLflow Observability (Optional)

MLflow is **NOT required** for core functionality. Only install if you need observability:

```bash
# Install with MLflow
pip install "promptchain[dev]"

# Enable tracking
export PROMPTCHAIN_MLFLOW_ENABLED=true
```

When enabled, all LLM calls are automatically tracked with:
- Token usage and costs
- Response times
- Input/output samples
- Model parameters

## Next Steps

- Read [INSTALLATION.md](INSTALLATION.md) for detailed installation options
- Check [CLAUDE.md](CLAUDE.md) for complete API documentation
- See [examples/](examples/) for more code samples
- Visit [docs/](docs/) for in-depth guides

## Common Patterns

### Token-Efficient History Management

```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

history = ExecutionHistoryManager(
    max_tokens=4000,
    truncation_strategy="oldest_first"
)

history.add_entry("user_input", "Question here", source="user")
formatted = history.get_formatted_history(format_style="chat")
```

### Agentic Reasoning

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

agentic_step = AgenticStepProcessor(
    objective="Research and summarize topic",
    max_internal_steps=5
)

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[agentic_step]
)
```

### Session Persistence

```python
from promptchain.utils.agent_chain import AgentChain

system = AgentChain(
    agents={"main": agent},
    cache_config={
        "name": "project_session",
        "path": "./sessions"
    }
)
```

## Troubleshooting

### Import Errors

```bash
# Verify installation
python -c "from promptchain import PromptChain; print('OK')"

# Check dependencies
pip install --force-reinstall -r requirements.txt
```

### MLflow Not Found (Expected)

If you see `ImportError: No module named 'mlflow'`:
- This is normal if you installed core only
- MLflow is optional - tracking will be automatically disabled
- To enable: `pip install "promptchain[dev]"`

### API Key Errors

```python
# Check environment variables
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None
```

## Getting Help

- GitHub Issues: https://github.com/gyasis/promptchain/issues
- Documentation: [CLAUDE.md](CLAUDE.md)
- Examples: [examples/](examples/)
