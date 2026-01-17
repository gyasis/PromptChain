# Activity Logging Quick Start Guide

**For**: Developers using PromptChain's multi-agent systems
**Goal**: Capture and search ALL agent interactions without consuming tokens

## Two Ways to Use Activity Logging

### Option 1: Automatic (Recommended) - With SessionManager

**Zero setup required** - ActivityLogger is automatically created for every session:

```python
from promptchain.cli.session_manager import SessionManager
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from pathlib import Path

# 1. Create session (ActivityLogger automatically initialized)
session_manager = SessionManager(sessions_dir=Path("~/.promptchain/sessions"))
session = session_manager.create_session(
    name="my-project",
    working_directory=Path.cwd()
)

# 2. Create agents
agent1 = PromptChain(models=["gpt-4"], instructions=["Do task 1: {input}"])
agent2 = PromptChain(models=["gpt-4"], instructions=["Do task 2: {input}"])

# 3. Create AgentChain with session's ActivityLogger
chain = AgentChain(
    agents={"agent1": agent1, "agent2": agent2},
    agent_descriptions={"agent1": "Task 1 specialist", "agent2": "Task 2 specialist"},
    execution_mode="pipeline",
    activity_logger=session.activity_logger,  # ✅ Automatic from session
    verbose=True
)

# 4. Use normally - everything is logged automatically
result = await chain.process_input("Process this")

# 5. Search activities later
from promptchain.cli.activity_searcher import ActivitySearcher

searcher = ActivitySearcher(
    session_name=session.name,
    log_dir=session_manager.sessions_dir / session.id / "activity_logs",
    db_path=session_manager.sessions_dir / session.id / "activities.db"
)
```

### Option 2: Manual Setup - Without SessionManager

For standalone scripts or custom workflows:

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.cli.activity_logger import ActivityLogger
from promptchain.cli.activity_searcher import ActivitySearcher
from pathlib import Path

# 1. Create activity logger manually
logger = ActivityLogger(
    session_name="my-session",
    log_dir=Path("./logs"),
    db_path=Path("./activities.db")
)

# 2. Create your agents
agent1 = PromptChain(models=["gpt-4"], instructions=["Do task 1: {input}"])
agent2 = PromptChain(models=["gpt-4"], instructions=["Do task 2: {input}"])

# 3. Create AgentChain with logging
chain = AgentChain(
    agents={"agent1": agent1, "agent2": agent2},
    agent_descriptions={"agent1": "Task 1 specialist", "agent2": "Task 2 specialist"},
    execution_mode="pipeline",
    activity_logger=logger,  # ✅ That's it!
    verbose=True
)

# 4. Use normally - everything is logged automatically
result = await chain.process_input("Process this")

# 5. Search activities later
searcher = ActivitySearcher(
    session_name="my-session",
    log_dir=Path("./logs"),
    db_path=Path("./activities.db")
)

# Find all activities
activities = searcher.grep_logs(pattern=".*", max_results=100)
print(f"Found {len(activities)} activities")

# Get statistics
stats = searcher.get_statistics()
print(f"Total: {stats['total_activities']}")
print(f"By agent: {stats['activities_by_agent']}")
```

## What Gets Logged

Every agent interaction is captured:

✅ **User inputs** - What you send to AgentChain
✅ **Agent outputs** - What each agent produces
✅ **Router decisions** - Which agent was selected
✅ **Pipeline steps** - Each step in sequential execution
✅ **Errors** - Any agent failures with full context
✅ **Round robin** - Which agent in rotation
✅ **Broadcast** - All parallel agent outputs

## Search Examples

### Find Recent Errors
```python
errors = searcher.grep_logs(
    pattern="error",
    activity_type="error",
    max_results=10
)

for error in errors:
    print(f"{error['timestamp']} - {error['agent_name']}: {error['content']['error']}")
```

### Find Router Decisions
```python
decisions = searcher.grep_logs(
    pattern="router_decision",
    max_results=10
)

for decision in decisions:
    chosen = decision['content']['chosen_agent']
    available = decision['content']['available_agents']
    print(f"Chose '{chosen}' from {available}")
```

### Get Agent-Specific Activities
```python
researcher_activities = searcher.grep_logs(
    pattern=".*",
    agent_name="researcher",
    max_results=50
)

print(f"Researcher performed {len(researcher_activities)} activities")
```

### Search by Time Range
```python
from datetime import datetime, timedelta

last_hour = searcher.search_by_timerange(
    start_time=datetime.now() - timedelta(hours=1),
    agent_name="researcher"
)
```

### SQL Queries
```python
# Count by agent
results = searcher.sql_query("""
    SELECT agent_name, COUNT(*) as count
    FROM agent_activities
    WHERE session_name = ?
    GROUP BY agent_name
""", ("my-session",))

for row in results:
    print(f"{row['agent_name']}: {row['count']} activities")
```

### Get Full Chain
```python
# List all chains
chains = searcher.sql_query("""
    SELECT chain_id, status, total_activities, completed_at
    FROM interaction_chains
    WHERE session_name = ?
""", ("my-session",))

# Get full chain with content
chain = searcher.get_interaction_chain(
    chain_id=chains[0]['chain_id'],
    include_content=True,
    include_nested=True
)

print(f"Chain {chain['chain_id']} has {len(chain['activities'])} activities")
for activity in chain['activities']:
    print(f"  - {activity['activity_type']}: {activity.get('agent_name', 'System')}")
```

## Key Benefits

### 1. No Token Consumption
Activities are stored in files, not conversation history. Agents search when needed, not loaded into every prompt.

### 2. Complete History
Every interaction is captured, even across multiple chains, errors, and internal reasoning steps.

### 3. Fast Search
- Ripgrep for text search (100ms for 10K activities)
- SQL for structured queries (<10ms)
- Indexed for efficient filtering

### 4. Independent Storage
Doesn't interfere with:
- ExecutionHistoryManager (in-memory history)
- Token limits
- Conversation flow

## Execution Mode Examples

### Pipeline Mode
```python
# Sequential execution: agent1 → agent2 → agent3
chain = AgentChain(
    agents={"step1": agent1, "step2": agent2, "step3": agent3},
    agent_descriptions={"step1": "...", "step2": "...", "step3": "..."},
    execution_mode="pipeline",
    activity_logger=logger
)
```

**Logs**: user_input + 3× agent_output (one per step)

### Router Mode
```python
# LLM selects which agent to use
chain = AgentChain(
    agents={"researcher": agent1, "writer": agent2},
    agent_descriptions={"researcher": "...", "writer": "..."},
    execution_mode="router",
    router=router_config,
    activity_logger=logger
)
```

**Logs**: user_input + router_decision + agent_output

### Round Robin Mode
```python
# Agents take turns
chain = AgentChain(
    agents={"agent1": agent1, "agent2": agent2},
    agent_descriptions={"agent1": "...", "agent2": "..."},
    execution_mode="round_robin",
    activity_logger=logger
)
```

**Logs**: user_input + agent_output (alternating agents)

### Broadcast Mode
```python
# All agents run in parallel
chain = AgentChain(
    agents={"fast": agent1, "accurate": agent2, "creative": agent3},
    agent_descriptions={"fast": "...", "accurate": "...", "creative": "..."},
    execution_mode="broadcast",
    synthesizer_config={...},
    activity_logger=logger
)
```

**Logs**: user_input + 3× agent_output (parallel) + synthesizer_output

## Advanced Features

### Context Manager (Auto-Close)
```python
with ActivityLogger(
    session_name="temp-session",
    log_dir=Path("./logs"),
    db_path=Path("./activities.db")
) as logger:
    chain = AgentChain(
        agents={...},
        agent_descriptions={...},
        activity_logger=logger
    )
    result = await chain.process_input("Test")
    # Chain automatically closed on exit
```

### Statistics
```python
stats = searcher.get_statistics()
print(f"""
Total Activities: {stats['total_activities']}
Total Chains: {stats['total_chains']}
Active Chains: {stats['active_chains']}
Average Depth: {stats['avg_chain_depth']}
Total Errors: {stats['total_errors']}

By Type:
{stats['activities_by_type']}

By Agent:
{stats['activities_by_agent']}
""")
```

### Find Complex Reasoning Chains
```python
# Find chains with depth >= 2 (multi-hop reasoning)
complex_chains = searcher.find_reasoning_chains(
    agent_name="researcher",
    min_depth=2,
    limit=10
)

for chain in complex_chains:
    print(f"Chain {chain['chain_id']}: {chain['total_activities']} activities, depth {chain['max_depth_level']}")
```

## File Structure

```
./logs/
  ├── activities.jsonl          # Full activity log (grep-able)
  └── session_metadata.json     # Session info

./activities.db                  # SQLite database (indexed queries)
  ├── agent_activities           # Activity records
  └── interaction_chains         # Chain metadata
```

## Common Patterns

### Pattern 1: Debugging Agent Failures
```python
# Find all errors
errors = searcher.grep_logs(pattern="error", activity_type="error")

# Get details
for error in errors:
    chain_id = error['interaction_chain_id']

    # Get full chain for context
    chain = searcher.get_interaction_chain(chain_id, include_content=True)

    print(f"Error in chain {chain_id}:")
    print(f"  Agent: {error['agent_name']}")
    print(f"  Error: {error['content']['error']}")
    print(f"  Context: {chain['activities']}")
```

### Pattern 2: Analyzing Agent Performance
```python
# Count successes and failures per agent
results = searcher.sql_query("""
    SELECT
        agent_name,
        COUNT(*) as total,
        SUM(CASE WHEN activity_type = 'error' THEN 1 ELSE 0 END) as errors
    FROM agent_activities
    WHERE session_name = ?
    GROUP BY agent_name
""", ("my-session",))

for row in results:
    success_rate = (row['total'] - row['errors']) / row['total'] * 100
    print(f"{row['agent_name']}: {success_rate:.1f}% success rate")
```

### Pattern 3: Tracing User Requests
```python
# Find all chains that started with specific input
user_inputs = searcher.grep_logs(
    pattern="machine learning",
    activity_type="user_input"
)

for user_input in user_inputs:
    chain_id = user_input['interaction_chain_id']
    chain = searcher.get_interaction_chain(chain_id, include_content=True)

    print(f"Request: {user_input['content']['input']}")
    print(f"Agents involved: {[a['agent_name'] for a in chain['activities'] if a['agent_name']]}")
```

## Best Practices

### 1. Use Session Names
Organize activities by project/session:
```python
logger = ActivityLogger(session_name="project-alpha", ...)
```

### 2. Add Tags
Tag activities for easier filtering:
```python
logger.log_activity(
    activity_type="agent_output",
    agent_name="researcher",
    content={...},
    tags=["experiment", "v2", "production"]  # ✅ Custom tags
)
```

### 3. Filter Searches
Use filters to reduce results:
```python
# Only errors from last hour
errors = searcher.search_by_timerange(
    start_time=datetime.now() - timedelta(hours=1),
    activity_type="error",
    agent_name="researcher"
)
```

### 4. Clean Up Old Logs
Periodically archive or delete old activities:
```python
# Delete activities older than 30 days
searcher.sql_query("""
    DELETE FROM agent_activities
    WHERE session_name = ?
      AND timestamp < datetime('now', '-30 days')
""", ("my-session",))
```

## Troubleshooting

### Issue: No activities logged
**Solution**: Ensure `activity_logger` is passed to AgentChain:
```python
chain = AgentChain(
    agents={...},
    agent_descriptions={...},
    activity_logger=logger  # ✅ Don't forget this!
)
```

### Issue: Grep search is slow
**Solution**: Install ripgrep for 10x faster search:
```bash
# Ubuntu/Debian
sudo apt-get install ripgrep

# macOS
brew install ripgrep
```

### Issue: Database locked errors
**Solution**: Use context manager or explicitly close:
```python
with ActivityLogger(...) as logger:
    # Use logger
    pass
# Automatically closed

# Or manually:
logger.close()
```

### Issue: Large log files
**Solution**: Implement log rotation or filtering:
```python
# Only log specific activity types
if activity_type in ['user_input', 'agent_output', 'error']:
    logger.log_activity(...)
```

## Next Steps

- **Phase 3**: CLI commands (`/log search`, `/log stats`, etc.)
- **Phase 4**: TUI integration with real-time activity streaming
- **Future**: Tool call logging from AgenticStepProcessor

## More Examples

See:
- `test_activity_logger.py` - Core ActivityLogger tests
- `test_agentchain_activity_logging.py` - Integration tests
- `docs/cli/AGENT_ACTIVITY_LOGGING_DESIGN.md` - Full design
- `docs/cli/AGENTCHAIN_INTEGRATION_SUMMARY.md` - Implementation details

---

**Questions?** Check the full documentation or review the test files for comprehensive examples.
