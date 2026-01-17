# Quickstart: Multi-Agent Communication

**Feature**: 003-multi-agent-communication
**Estimated Setup Time**: 5 minutes after implementation

## Prerequisites

- PromptChain CLI installed with Phase 11 features
- Session with multiple agents configured
- SQLite database migrated to V3 schema

## Basic Usage

### 1. Agent Capability Discovery

```bash
# List all available capabilities
> /capabilities

# List capabilities for specific agent
> /capabilities DataAnalyst

# In agent code: discover what tools an agent can use
tools = registry.discover_capabilities("DataAnalyst")
```

### 2. Task Delegation

```python
# Supervisor agent delegating to worker
result = await delegate_task(
    task_description="Analyze sales data from Q4",
    target_agent="DataAnalyst",
    priority="high",
    context={"file_path": "/data/q4_sales.csv"}
)

# Check task status
> /tasks pending
> /tasks in_progress
```

### 3. Blackboard Collaboration

```python
# Agent A writes analysis results
await write_to_blackboard("analysis_results", {
    "total_sales": 1250000,
    "top_product": "Widget Pro",
    "growth_rate": 0.15
})

# Agent B reads and builds on results
results = await read_from_blackboard("analysis_results")
summary = f"Q4 showed {results['growth_rate']*100}% growth"

# List all available data
> /blackboard
```

### 4. Help Requests

```python
# Stuck agent requests help
result = await request_help(
    problem_description="Cannot parse malformed JSON in config file",
    context={"file": "config.json", "error": "Unexpected token at line 15"}
)
# System routes to agent with data_processing capability
```

### 5. Workflow Tracking

```bash
# View current workflow state
> /workflow

# Output:
# Workflow: wf_abc123
# Stage: execution
# Agents: [Supervisor, DataAnalyst, Writer]
# Progress: 3/5 tasks completed
# Current: "Generate quarterly report"
```

## Common Patterns

### Hierarchical Agent Team

```python
# Supervisor creates workflow
workflow = WorkflowState(
    workflow_id="quarterly_report",
    stage="planning",
    agents_involved=["Supervisor", "DataAnalyst", "Writer"]
)

# Supervisor delegates analysis
await delegate_task(
    task_description="Analyze Q4 data",
    target_agent="DataAnalyst",
    priority="high"
)

# DataAnalyst completes and shares results
await write_to_blackboard("q4_analysis", analysis_results)

# Supervisor delegates writing
await delegate_task(
    task_description="Write report using q4_analysis from blackboard",
    target_agent="Writer"
)
```

### Parallel Evaluation

```python
# Broadcast evaluation request
message_bus.broadcast(
    type=MessageType.REQUEST,
    payload={"task": "evaluate", "data": customer_data}
)

# Multiple agents write their evaluations
await write_to_blackboard("eval_agent1", {"score": 85})
await write_to_blackboard("eval_agent2", {"score": 92})
await write_to_blackboard("eval_agent3", {"score": 78})

# Aggregator reads all evaluations
keys = await list_blackboard_keys()
scores = [await read_from_blackboard(k) for k in keys if k.startswith("eval_")]
```

### Tool Registration with Capabilities

```python
@registry.register(
    category="analysis",
    description="Analyze CSV data files",
    allowed_agents=["DataAnalyst", "Supervisor"],
    capabilities=["data_processing", "statistics", "csv_handling"],
    parameters={
        "file_path": ParameterSchema(type="string", required=True),
        "columns": ParameterSchema(type="array", required=False)
    }
)
async def analyze_csv(file_path: str, columns: list = None) -> str:
    """Analyze CSV file and return statistics."""
    ...
```

## Troubleshooting

### Task Not Being Picked Up

```bash
# Check task exists
> /tasks

# Verify target agent exists
> /agent list

# Check agent capabilities match task requirements
> /capabilities target_agent_name
```

### Blackboard Key Not Found

```python
result = await read_from_blackboard("my_key")
if not result.get("found"):
    # Key doesn't exist or was cleared
    await write_to_blackboard("my_key", default_value)
```

### Message Handler Not Triggered

```python
# Verify handler is registered
@cli_communication_handler(type="request")  # Check type matches
async def my_handler(message, sender, receiver):
    print(f"Handler triggered: {message}")  # Add debug logging
    return message
```

## Performance Tips

1. **Batch blackboard writes**: Combine related data into single writes
2. **Use high priority sparingly**: Reserve for truly urgent tasks
3. **Clean up completed tasks**: Task queue grows over session lifetime
4. **Minimize message payloads**: Keep communication metadata light

## Next Steps

- Read [data-model.md](./data-model.md) for entity details
- Check [api-schema.json](./contracts/api-schema.json) for full API specs
- Review [research.md](./research.md) for design decisions
