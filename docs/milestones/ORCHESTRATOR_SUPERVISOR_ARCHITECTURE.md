# OrchestratorSupervisor - Complete Library Architecture

**Date**: 2025-10-04
**Version**: PromptChain v0.4.1k (Multi-Agent Orchestration with Oversight)
**Type**: Core Library Feature

---

## Executive Summary

We've implemented **OrchestratorSupervisor** as a core library class in PromptChain that provides:
- **Multi-hop reasoning** via AgenticStepProcessor for intelligent task decomposition
- **Strategy preferences** to guide orchestration behavior
- **Oversight metrics** for decision tracking and analysis
- **Full observability** with event emissions and logging
- **Backward compatibility** - existing code continues to work unchanged

---

## Architecture Overview

### Core Components

```
┌────────────────────────────────────────────────────────────────┐
│                    LIBRARY LEVEL                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          OrchestratorSupervisor                          │  │
│  │  (promptchain/utils/orchestrator_supervisor.py)          │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  - Multi-hop reasoning (AgenticStepProcessor)            │  │
│  │  - Strategy preferences (adaptive, prefer_multi, etc.)   │  │
│  │  - Oversight metrics and tracking                        │  │
│  │  - Event emissions (observability)                       │  │
│  │  - Logging integration                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               AgentChain                                 │  │
│  │  (promptchain/utils/agent_chain.py)                      │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  ENHANCED with supervisor support:                       │  │
│  │  - use_supervisor=True parameter                         │  │
│  │  - supervisor_strategy parameter                         │  │
│  │  - Automatic supervisor initialization                   │  │
│  │  - Router compatibility                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│                 APPLICATION LEVEL                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Simple Usage (agentic_chat/agentic_team_chat.py):            │
│                                                                │
│  agent_chain = AgentChain(                                     │
│      agents=agents,                                            │
│      agent_descriptions=descriptions,                          │
│      execution_mode="router",                                  │
│      use_supervisor=True,  # ✅ One line!                      │
│      supervisor_strategy="adaptive"  # ✅ Configure behavior   │
│  )                                                             │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Files Created/Modified

### New Library Files

1. **`/promptchain/utils/orchestrator_supervisor.py`** (NEW)
   - Core OrchestratorSupervisor class
   - Multi-hop reasoning engine
   - Strategy preference system
   - Oversight and metrics tracking
   - Event emission and logging

### Modified Library Files

2. **`/promptchain/utils/agent_chain.py`** (ENHANCED)
   - Added `use_supervisor` parameter (default: False)
   - Added `supervisor_strategy` parameter (default: "adaptive")
   - Added `supervisor_max_steps` parameter (default: 8)
   - Added `supervisor_model` parameter (default: "openai/gpt-4.1-mini")
   - Automatic OrchestratorSupervisor initialization when `use_supervisor=True`
   - Backward compatible - existing code unchanged

### Application Files

3. **`/agentic_chat/agentic_team_chat.py`** (SIMPLIFIED)
   - Changed import: `from promptchain.utils.orchestrator_supervisor import OrchestratorSupervisor`
   - Existing usage already compatible
   - Can now use simpler AgentChain-based approach

---

## Strategy Preferences

OrchestratorSupervisor supports 6 strategy preferences that guide (not force) the multi-hop reasoning:

### 1. **adaptive** (default)
```python
supervisor_strategy="adaptive"
```
- Let AgenticStepProcessor decide with no bias
- Analyzes task complexity thoroughly
- Chooses single or multi-agent based on merit
- Best for: General-purpose orchestration

### 2. **prefer_single**
```python
supervisor_strategy="prefer_single"
```
- Bias toward single-agent dispatch when possible
- Only use multi-agent if absolutely necessary
- Best for: Performance-critical scenarios, simple tasks

### 3. **prefer_multi**
```python
supervisor_strategy="prefer_multi"
```
- Bias toward multi-agent plans for thorough results
- Break complex tasks into specialized subtasks
- Best for: Quality-critical scenarios, comprehensive analysis

### 4. **always_research_first**
```python
supervisor_strategy="always_research_first"
```
- Start with research for any unknown topics
- Pattern: ["research", "documentation"] or ["research", "analysis"]
- Best for: Ensuring current information, fact-checking

### 5. **conservative**
```python
supervisor_strategy="conservative"
```
- Use single agent unless multiple distinct phases exist
- Require strong evidence for multi-agent
- Best for: Cost-sensitive scenarios, controlled experimentation

### 6. **aggressive**
```python
supervisor_strategy="aggressive"
```
- Liberally use multi-agent plans
- Break into fine-grained subtasks (2-3+ agents)
- Best for: Maximum quality, comprehensive coverage

---

## Usage Examples

### Example 1: Simple Usage (Recommended)

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain

# Create agents (same as before)
research_agent = PromptChain(...)
coding_agent = PromptChain(...)
documentation_agent = PromptChain(...)

agents = {
    "research": research_agent,
    "coding": coding_agent,
    "documentation": documentation_agent
}

agent_descriptions = {
    "research": "Research agent with web search tools",
    "coding": "Coding agent with script writing tools",
    "documentation": "Documentation agent for technical writing"
}

# ✅ NEW: Enable supervisor with one parameter!
agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    use_supervisor=True,  # ✅ Multi-hop reasoning + oversight
    supervisor_strategy="prefer_multi",  # ✅ Prefer multi-agent for quality
    router_strategy="static_plan"  # ✅ Sequential execution
)

# Use as normal
result = await agent_chain.process_input_async("Write tutorial about Zig sound manipulation")
```

### Example 2: Advanced Configuration

```python
agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    use_supervisor=True,
    supervisor_strategy="adaptive",
    supervisor_max_steps=10,  # More reasoning steps for complex tasks
    supervisor_model="openai/gpt-4.1",  # Use more powerful model
    router_strategy="static_plan",
    cache_config={"name": "my_session", "path": "./cache"},
    log_dir="./logs",
    verbose=True
)
```

### Example 3: Backward Compatible (No Changes)

```python
# Old code continues to work unchanged!
router_config = {
    "models": ["openai/gpt-4.1-mini"],
    "instructions": [None, "{input}"],
    "decision_prompt_templates": {...}
}

agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    router=router_config,  # Traditional router
    router_strategy="static_plan"
)
```

### Example 4: Direct Supervisor Usage

```python
from promptchain.utils.orchestrator_supervisor import OrchestratorSupervisor

# Create supervisor directly for more control
supervisor = OrchestratorSupervisor(
    agent_descriptions=agent_descriptions,
    log_event_callback=my_logger,
    dev_print_callback=my_printer,
    max_reasoning_steps=8,
    model_name="openai/gpt-4.1-mini",
    strategy_preference="always_research_first"
)

# Use as router function
agent_chain = AgentChain(
    agents=agents,
    agent_descriptions=agent_descriptions,
    execution_mode="router",
    router=supervisor.as_router_function()
)

# Get oversight metrics
print(supervisor.get_oversight_summary())
supervisor.print_oversight_report()
```

---

## Multi-Hop Reasoning Process

### How It Works

1. **User Query** → OrchestratorSupervisor
2. **AgenticStepProcessor** performs structured 4-step tool-aware reasoning:
   ```
   STEP 1: CLASSIFY TASK COMPLEXITY
   - Determine if task is SIMPLE (single action/question) or COMPLEX (multiple phases)
   - Examples:
     * SIMPLE: "What is X?", "Explain Y"
     * COMPLEX: "Research X and create tutorial", "Write script and run it"

   STEP 2: IDENTIFY REQUIRED CAPABILITIES & TOOLS
   - What does the task actually need?
     * Current/real-time web info? → Research agent ✅ (gemini_research, ask_gemini)
     * Create code/script files? → Coding agent ✅ (write_script)
     * Execute commands? → Terminal agent ✅ (execute_terminal_command)
     * Analyze data? → Analysis agent ❌ (no tools, reasoning only)
     * Explain concepts? → Documentation agent ❌ (no tools, training knowledge)
     * Synthesize info? → Synthesis agent ❌ (no tools, reasoning only)

   STEP 3: CHECK IF SINGLE AGENT HAS SUFFICIENT TOOLS
   - Does ONE agent have ALL the tools needed?
     * YES → Use single-agent plan ["agent_name"]
     * NO → Proceed to STEP 4 for multi-agent combination
   - Example: "What is Candle library?"
     * Needs: web search ✅
     * Research has: gemini_research ✅
     * SUFFICIENT: ["research"]

   STEP 4: DESIGN OPTIMAL AGENT COMBINATION
   - If single agent insufficient:
     1. List all required capabilities from STEP 2
     2. Map each capability to agent(s) that provide it
     3. Order by logical dependency:
        - Research BEFORE documentation (need info before writing)
        - Coding BEFORE terminal (need script before executing)
     4. Create sequential plan: ["agent1", "agent2", ...]
   - Verify plan accuracy and efficiency
   ```
3. **Strategy Preference** guides reasoning (doesn't force it)
4. **Return Plan**: `{"plan": ["agent1", "agent2"], "initial_input": "...", "reasoning": "..."}`

### Example Reasoning

**Query**: "Write tutorial about Zig sound manipulation"

**Multi-Hop Reasoning** (with `strategy="adaptive"`):
```
STEP 1: CLASSIFY TASK COMPLEXITY
- Task: "Write tutorial about Zig sound manipulation"
- Classification: COMPLEX
- Rationale: Two distinct phases - research current info + write tutorial

STEP 2: IDENTIFY REQUIRED CAPABILITIES & TOOLS
- Capability 1: Web search for current Zig audio libraries ✅
  → Reason: "Zig sound" is unknown/recent tech (post-2024 or niche)
  → Cannot rely on training data
- Capability 2: Documentation writing ✅
  → Reason: Need to format research findings into tutorial

STEP 3: CHECK IF SINGLE AGENT HAS SUFFICIENT TOOLS
- Research agent analysis:
  * Has gemini_research ✅ (can get current Zig audio info)
  * NO documentation expertise ❌
  * INSUFFICIENT for complete task
- Documentation agent analysis:
  * Has writing skills ✅ (can write tutorial)
  * NO web search tools ❌ (cannot get current Zig info)
  * INSUFFICIENT for complete task
- RESULT: Single agent INSUFFICIENT → Need multi-agent

STEP 4: DESIGN OPTIMAL AGENT COMBINATION
- Required capabilities: [web search, documentation]
- Agent mapping:
  * Web search → Research agent (gemini_research)
  * Documentation → Documentation agent (writing skills)
- Logical order:
  * Research FIRST (must gather current info before writing)
  * Documentation SECOND (formats research findings into tutorial)
- Final plan: ["research", "documentation"]
- Verification: ✅ Research gets data → Documentation formats it
```

**Output**:
```json
{
  "plan": ["research", "documentation"],
  "initial_input": "Research current Zig audio libraries and sound manipulation techniques",
  "reasoning": "Task needs current web info (Research has tools) + tutorial writing (Documentation). Research must go first to gather info, then Documentation formats it."
}
```

---

## Oversight Metrics

OrchestratorSupervisor tracks comprehensive metrics:

```python
supervisor.metrics = {
    "total_decisions": 10,
    "single_agent_dispatches": 3,
    "multi_agent_plans": 7,
    "total_reasoning_steps": 56,
    "total_tools_called": 12,
    "average_plan_length": 2.1,
    "decision_history": [...]  # Last 100 decisions
}

# Get summary
summary = supervisor.get_oversight_summary()
# Returns:
# {
#   "total_decisions": 10,
#   "single_agent_rate": 0.3,
#   "multi_agent_rate": 0.7,
#   "average_plan_length": 2.1,
#   "average_reasoning_steps": 5.6
# }

# Print detailed report
supervisor.print_oversight_report()
```

---

## Event Emissions

OrchestratorSupervisor emits comprehensive events for observability:

### Orchestrator Events

```json
// Input
{"event": "orchestrator_input", "user_query": "...", "history_length": 5}

// Agentic reasoning
{"event": "orchestrator_agentic_step", "total_steps": 4, "tools_called": 0, "execution_time_ms": 856}

// Decision
{"event": "orchestrator_decision", "plan": ["research", "documentation"], "reasoning": "...", "plan_length": 2, "decision_type": "multi_agent"}
```

### Integration with Existing Events

Works with all PromptChain v0.4.1 observability features:
- ✅ ExecutionHistoryManager (v0.4.1a)
- ✅ AgentExecutionResult Metadata (v0.4.1b)
- ✅ Event System Callbacks (v0.4.1d)
- ✅ MCP Event Emissions (v0.4.1j)
- ✅ OrchestratorSupervisor (v0.4.1k) ← NEW!

---

## Benefits

### 1. **Library-Level Feature**
- No custom code in application scripts
- Reusable across all projects
- Maintained as part of PromptChain core
- Version-controlled and tested

### 2. **Multi-Hop Reasoning**
- AgenticStepProcessor's 8-step reasoning
- Deep analysis of task requirements
- Tool capability awareness
- Knowledge boundary detection

### 3. **Strategy Flexibility**
- Guide behavior with strategy preferences
- Adaptive, conservative, aggressive modes
- Research-first for current info
- Single vs multi-agent biasing

### 4. **Oversight & Metrics**
- Track all decisions
- Analyze patterns
- Monitor performance
- Comprehensive reporting

### 5. **Full Observability**
- Event emissions
- Structured logging
- Dev mode terminal output
- Integration with existing observability

### 6. **Backward Compatible**
- Existing code unchanged
- Optional feature (use_supervisor=False by default)
- Can mix old and new approaches
- Gradual migration path

---

## Migration Guide

### From Old Approach

**Before** (custom orchestrator in script):
```python
def create_agentic_orchestrator(...):
    orchestrator_step = AgenticStepProcessor(...)
    orchestrator_chain = PromptChain(...)
    # ... 100+ lines of code
    return async_wrapper

agent_chain = AgentChain(
    ...,
    router=create_agentic_orchestrator(...)
)
```

**After** (library feature):
```python
agent_chain = AgentChain(
    ...,
    use_supervisor=True,  # ✅ One line!
    supervisor_strategy="adaptive"
)
```

### Migration Steps

1. **Remove custom orchestrator code** from application
2. **Add two parameters** to AgentChain:
   - `use_supervisor=True`
   - `supervisor_strategy="adaptive"` (or other preference)
3. **Test** - behavior should improve (better multi-agent decisions)
4. **Tune strategy** based on your needs
5. **Monitor metrics** via `supervisor.get_oversight_summary()`

---

## Testing

### Test Scenarios

1. **Simple query** (should use single agent):
   ```
   "Explain neural networks"
   Expected: {"plan": ["documentation"]}
   ```

2. **Unknown tech query** (should use research → documentation):
   ```
   "Write tutorial about Zig sound manipulation"
   Expected: {"plan": ["research", "documentation"]}
   ```

3. **Code + execute query** (should use coding → terminal):
   ```
   "Create log cleanup script and run it"
   Expected: {"plan": ["coding", "terminal"]}
   ```

4. **Strategy preference** (prefer_multi):
   ```
   "What is X?"
   Expected with prefer_multi: {"plan": ["research", "documentation"]}
   Expected with prefer_single: {"plan": ["research"]}
   ```

---

## Performance Considerations

### Token Usage

- Multi-hop reasoning: ~8 steps × 500 tokens/step = ~4k tokens
- Worthwhile for complex decisions
- Single queries benefit from cached reasoning patterns
- Consider `supervisor_max_steps` tuning

### Latency

- Multi-hop reasoning adds ~2-5 seconds
- Amortized across entire conversation
- Better decisions save tokens in agent execution
- Net positive for complex workflows

### Cost Optimization

Strategies for cost-sensitive scenarios:
- Use `supervisor_strategy="conservative"` → fewer multi-agent plans
- Use `supervisor_strategy="prefer_single"` → bias toward single agent
- Use `supervisor_max_steps=5` → less reasoning
- Use `supervisor_model="openai/gpt-4o-mini"` → cheaper model

---

## Future Enhancements

### Potential Additions

1. **Dynamic Decomposition Strategy**
   - Iterative planning (plan next step based on results)
   - Agent-to-agent feedback loops
   - Adaptive plan modification

2. **Policy Engine Integration**
   - Scoring-based agent selection
   - Performance-weighted routing
   - Cost-aware optimization

3. **Agent Builder**
   - Create specialized agents on-demand
   - Dynamic capability composition
   - Top-3 model integration (GPT-4.1, Claude Opus, Gemini Pro)

4. **Learning & Optimization**
   - Track successful patterns
   - Learn from user feedback
   - Optimize strategy preferences automatically

---

## Conclusion

OrchestratorSupervisor elevates PromptChain from **single-agent dispatching** to **intelligent multi-agent orchestration** at the library level.

**Key Achievements**:
- ✅ Multi-hop reasoning for complex task decomposition
- ✅ Strategy preferences for behavior guidance
- ✅ Oversight metrics for decision tracking
- ✅ Full observability integration
- ✅ Backward compatible
- ✅ Simple to use (one parameter!)

**Impact**:
- Applications get smarter orchestration with minimal code
- Multi-agent workflows become trivial to implement
- Better decisions through deep reasoning
- Comprehensive oversight and monitoring

The orchestrator is now **library-level, intelligent, and flexible** - exactly what PromptChain needs for sophisticated multi-agent applications!
