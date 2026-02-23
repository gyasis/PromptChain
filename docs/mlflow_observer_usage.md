# MLflow Observer Plugin - Usage Guide

The MLflow Observer is an optional plugin that provides automatic observability by listening to CallbackManager events and logging them to MLflow.

## Overview

**Architecture**: Plugin design that integrates with the existing CallbackManager system
**Status**: Optional (works without MLflow installed)
**Integration**: No decorators needed - works through callbacks

## Requirements

1. **Install MLflow** (optional):
   ```bash
   pip install mlflow
   ```

2. **Enable the observer** (environment variable):
   ```bash
   export PROMPTCHAIN_MLFLOW_ENABLED=true
   ```

3. **Start MLflow server** (optional - can use local directory):
   ```bash
   # Option 1: MLflow UI server
   mlflow ui --port 5000

   # Option 2: Use local directory (no server needed)
   export MLFLOW_TRACKING_URI=./mlruns
   ```

## Activation

The MLflow observer activates automatically when:
- ✅ MLflow is installed (`pip install mlflow`)
- ✅ Environment variable `PROMPTCHAIN_MLFLOW_ENABLED=true`
- ✅ CLI is started with `--verbose` flag (for integration with ObservePanel)

If any condition is missing, the observer is silently disabled (no errors).

## Integration with CLI

The MLflow observer integrates seamlessly with the PromptChain CLI:

```bash
# Enable verbose mode with MLflow tracking
export PROMPTCHAIN_MLFLOW_ENABLED=true
promptchain --verbose --session my-project

# Observer auto-activates and logs:
# ✓ MLflow observer plugin activated (optional observability)
# ✓ MLflow observer connected (tracking enabled)
```

## What Gets Tracked

The observer automatically logs:

### 1. Chain Execution
- **Parameters**: Chain ID, model name, configuration
- **Metrics**: Total duration, total tokens used
- **Tags**: Chain metadata and settings
- **Status**: FINISHED, FAILED, or KILLED

### 2. LLM Calls
- **Parameters**: Model name per call
- **Metrics**:
  - Duration (ms)
  - Prompt tokens
  - Completion tokens
  - Total tokens
- **Artifacts**:
  - Prompts (if `auto_log_artifacts=True`)
  - Responses (if `auto_log_artifacts=True`)

### 3. Tool Calls
- **Parameters**: Tool name per call
- **Metrics**: Duration (ms)
- **Artifacts**:
  - Tool arguments (JSON)
  - Tool results (text)
- **Errors**: Logged as tags

### 4. Chain Steps
- **Nested Runs**: Each step gets its own nested MLflow run
- **Parameters**: Step number, instruction
- **Metrics**: Step duration

## Programmatic Usage

You can also use the MLflow observer programmatically:

```python
from promptchain import PromptChain
from promptchain.observability import MLflowObserver

# Create observer
observer = MLflowObserver(
    experiment_name="my-experiment",
    tracking_uri="http://localhost:5000",
    auto_log_artifacts=True  # Log prompts/responses as files
)

# Create chain
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}"]
)

# Register observer callback (if available)
if observer.is_available():
    chain.register_callback(observer.handle_event)
    print("MLflow tracking enabled")
else:
    print("MLflow not available (install with: pip install mlflow)")

# Execute chain - events automatically logged to MLflow
result = chain.process_prompt("Your input here")

# Cleanup on shutdown
observer.shutdown()
```

## Environment Variables

Configure MLflow observer behavior with environment variables:

```bash
# Enable/disable observer
export PROMPTCHAIN_MLFLOW_ENABLED=true

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Or use local directory (no server needed)
export MLFLOW_TRACKING_URI=./mlruns
```

## Viewing MLflow UI

Access the MLflow UI to view tracked experiments:

```bash
# Start MLflow UI (if not already running)
mlflow ui --port 5000

# Open browser to:
# http://localhost:5000
```

### UI Features:
- **Experiments**: View all tracked sessions
- **Runs**: Compare different chain executions
- **Metrics**: Visualize token usage, latency trends
- **Artifacts**: Download prompts, responses, tool I/O
- **Parameters**: Filter by model, configuration
- **Tags**: Search by chain metadata

## Graceful Degradation

The observer is designed to never break your application:

1. **MLflow not installed**: Observer silently disabled, logs debug message
2. **Environment variable not set**: Observer disabled, logs debug message
3. **MLflow server unreachable**: Observer disabled, logs warning
4. **Event handling errors**: Logged as warnings, execution continues
5. **Shutdown errors**: Logged as warnings, cleanup continues

## Best Practices

### 1. Use Experiment Names for Organization
```python
# Organize by project/session
observer = MLflowObserver(
    experiment_name=f"promptchain-{session_name}"
)
```

### 2. Disable Artifacts for Large-Scale Usage
```python
# Reduce storage for high-volume scenarios
observer = MLflowObserver(
    auto_log_artifacts=False  # Don't save prompts/responses
)
```

### 3. Use Local Storage for Development
```bash
# No MLflow server needed during development
export MLFLOW_TRACKING_URI=./mlruns
export PROMPTCHAIN_MLFLOW_ENABLED=true
```

### 4. Remote Tracking for Production
```bash
# Connect to centralized MLflow server
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export PROMPTCHAIN_MLFLOW_ENABLED=true
```

## Comparison with Decorators

The MLflow observer complements (not replaces) the existing decorator system:

| Feature | Decorators | Observer Plugin |
|---------|-----------|----------------|
| **Integration** | Manual (@track_llm_call) | Automatic (callbacks) |
| **Setup** | Decorator on functions | Register callback |
| **Coverage** | Explicit tracking | All CallbackManager events |
| **Flexibility** | Fine-grained control | Comprehensive coverage |
| **Best For** | Specific functions | Entire chain execution |

### Recommended Usage:
- **Observer**: Default for CLI and full chain tracking
- **Decorators**: Custom tracking in library integrations

## Troubleshooting

### Observer Not Activating

**Check environment variable**:
```bash
echo $PROMPTCHAIN_MLFLOW_ENABLED
# Should output: true
```

**Check MLflow installation**:
```bash
python -c "import mlflow; print(mlflow.__version__)"
```

**Check CLI logs** (with `--dev` flag):
```bash
promptchain --verbose --dev --session test
# Look for: "✓ MLflow observer plugin activated"
```

### No Data in MLflow UI

**Verify tracking URI**:
```bash
echo $MLFLOW_TRACKING_URI
# Should match your MLflow server
```

**Check experiment exists**:
```bash
mlflow experiments list | grep promptchain
```

**Verify runs were created**:
```bash
mlflow runs list --experiment-name promptchain-default
```

### Performance Impact

The observer is designed for minimal overhead:

- **Event handling**: <1ms per event (async logging)
- **Artifact logging**: Only if `auto_log_artifacts=True`
- **Network overhead**: Batched writes to MLflow server
- **Graceful errors**: Never blocks chain execution

**Disable if needed**:
```bash
export PROMPTCHAIN_MLFLOW_ENABLED=false
# Or simply don't set the variable
```

## Advanced Features

### Temporary Runs for Testing

```python
# Create one-off tracking without affecting main run
with observer.temporary_run("test-experiment"):
    mlflow.log_metric("test_metric", 42)
```

### Manual Cleanup

```python
# Explicitly cleanup before exit
observer.shutdown()
```

### Custom Tracking URI per Session

```python
observer = MLflowObserver(
    experiment_name="custom-session",
    tracking_uri="http://custom-server:5000"
)
```

## Examples

### Example 1: CLI Usage with MLflow

```bash
# Terminal 1: Start MLflow server
mlflow ui --port 5000

# Terminal 2: Enable observer and run CLI
export PROMPTCHAIN_MLFLOW_ENABLED=true
promptchain --verbose --session research-project

# In CLI:
> What are the key trends in AI research?
# Observer logs chain execution, LLM calls, tool usage to MLflow

# View in browser: http://localhost:5000
```

### Example 2: Programmatic Usage

```python
import os
from promptchain import PromptChain
from promptchain.observability import MLflowObserver

# Enable observer
os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

# Create observer
observer = MLflowObserver(
    experiment_name="code-review-bot",
    tracking_uri="http://localhost:5000"
)

# Create chain with tools
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Review code: {input}",
        "Generate report: {input}"
    ]
)

# Register observer
if observer.is_available():
    chain.register_callback(observer.handle_event)

# Execute
result = chain.process_prompt("Review my Python code for security issues")

# Cleanup
observer.shutdown()
```

### Example 3: Multi-Agent with MLflow

```python
from promptchain.utils.agent_chain import AgentChain
from promptchain.observability import MLflowObserver

# Create observer
observer = MLflowObserver(experiment_name="multi-agent-system")

# Create agents
researcher = PromptChain(models=["openai/gpt-4"], ...)
coder = PromptChain(models=["openai/gpt-4"], ...)
reviewer = PromptChain(models=["claude-3-opus"], ...)

# Multi-agent chain
agent_chain = AgentChain(
    agents={"researcher": researcher, "coder": coder, "reviewer": reviewer},
    execution_mode="router"
)

# Register observer for each agent
if observer.is_available():
    for agent in [researcher, coder, reviewer]:
        agent.register_callback(observer.handle_event)

# Execute - all agents tracked in MLflow
result = await agent_chain.run_chat()

# Cleanup
observer.shutdown()
```

## Summary

The MLflow observer provides:
- ✅ **Zero-code integration** via CallbackManager
- ✅ **Optional installation** - works without MLflow
- ✅ **Automatic tracking** of all chain events
- ✅ **Production-ready** error handling
- ✅ **Flexible configuration** via environment variables
- ✅ **Comprehensive coverage** - LLMs, tools, steps

**Get Started**:
```bash
pip install mlflow
export PROMPTCHAIN_MLFLOW_ENABLED=true
promptchain --verbose --session my-project
```
