# PromptChain Architecture Vision: Hybrid Sequential + Agentic System
**Generated:** 2026-02-23
**Branch:** 005-mlflow-observability-clean

## Core Philosophy: "Both/And" Not "Either/Or"

**CRITICAL INSIGHT:** PromptChain's strength is deterministic sequential pipelines. We must **preserve this** while **adding** agentic flows, not replacing them.

### The Hybrid Model

```
┌─────────────────────────────────────────────────────────────┐
│                    PromptChain Hybrid Architecture           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────┐         ┌──────────────────────┐    │
│  │  Sequential Mode  │         │    Agentic Mode      │    │
│  │  (Current Strength)│         │   (New Capability)   │    │
│  └───────────────────┘         └──────────────────────┘    │
│           │                              │                   │
│           │                              │                   │
│  ┌────────▼────────────────────────────▼─────────┐         │
│  │         Unified Execution Engine               │         │
│  │  - Instruction processing                      │         │
│  - Tool calling (MCP + local)                   │         │
│  │  - History management                          │         │
│  │  - Event emission                              │         │
│  └────────────────────────────────────────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Mode 1: Sequential Pipelines (Preserve & Enhance)

**What we have today and must keep:**

### Strengths of Sequential Mode
1. **Deterministic**: Predictable execution order
2. **Debuggable**: Clear step-by-step flow
3. **Reliable**: No race conditions or timing issues
4. **Production-Ready**: Works in enterprise environments
5. **Composable**: Easy to understand and modify

### Use Cases Best Suited for Sequential
- **Data Processing Pipelines**: ETL workflows, data transformations
- **Document Generation**: Report generation, content creation
- **Code Generation**: Multi-step code scaffolding
- **Validation Workflows**: Schema validation, data quality checks
- **Deterministic Agents**: When order matters (legal, compliance, finance)

### Current Sequential Features (Keep & Improve)
```python
# This is PromptChain's core strength - preserve it!
chain = PromptChain(
    models=["gpt-4", "claude-3-opus"],
    instructions=[
        "Step 1: Analyze requirements {input}",
        analysis_function,  # Python function injection
        "Step 2: Generate design based on analysis {input}",
        design_function,
        "Step 3: Create implementation from design {input}"
    ]
)

result = chain.process_prompt("Build authentication system")
```

**Enhancement Ideas (without breaking sequential model):**
- Add optional checkpointing between steps
- Add conditional branching (if/else logic)
- Add loop constructs (for validation retries)
- Add step timeout and fallback handling
- Keep history and step outputs for debugging

---

## Mode 2: Agentic Flows (New Addition)

**What we need to add alongside sequential:**

### Characteristics of Agentic Mode
1. **Non-Deterministic**: Agent chooses next action dynamically
2. **Goal-Oriented**: Works toward objective, not fixed steps
3. **Adaptive**: Changes strategy based on feedback
4. **Conversational**: Multi-turn dialogue and reasoning
5. **Autonomous**: Makes decisions without predefined flow

### Use Cases Best Suited for Agentic
- **Research & Discovery**: Open-ended information gathering
- **Creative Problem Solving**: Brainstorming, ideation
- **Interactive Debugging**: Exploratory code analysis
- **Customer Support**: Dynamic conversation handling
- **Complex Reasoning**: Multi-hop question answering

### Agentic Features to Add
```python
# New agentic mode - doesn't replace sequential!
agentic_agent = AgenticAgent(
    objective="Debug why tests are failing",
    tools=[code_search, test_runner, gemini_debug],
    mode="autonomous",  # vs "sequential"
    max_iterations=10,
    allow_branching=True
)

# Agent decides its own steps:
# 1. Searches for test files
# 2. Runs tests to see failures
# 3. Analyzes error messages
# 4. Asks Gemini for debugging help
# 5. Proposes fixes
# 6. Validates fixes

result = await agentic_agent.execute()
```

---

## Hybrid Execution: Best of Both Worlds

### Pattern 1: Sequential Pipeline with Agentic Steps

```python
# Combine both modes in same workflow
hybrid_chain = PromptChain(
    instructions=[
        "Step 1: Validate input {input}",  # Sequential
        AgenticStepProcessor(  # Agentic reasoning embedded in pipeline
            objective="Research best implementation approach",
            max_internal_steps=5
        ),
        "Step 2: Implement based on research {input}",  # Sequential
        AgenticStepProcessor(  # Another agentic step
            objective="Test and debug implementation",
            max_internal_steps=3
        ),
        "Step 3: Generate documentation {input}"  # Sequential
    ]
)
```

**This already works today!** AgenticStepProcessor inside sequential chains.

### Pattern 2: Multi-Agent Round-Robin Chat

```python
# New pattern: Conversational multi-agent (like AG2 group chat)
round_robin_chat = AgentChain(
    agents={
        "researcher": researcher_agent,
        "coder": coder_agent,
        "reviewer": reviewer_agent
    },
    execution_mode="round-robin",  # New mode
    max_rounds=5,
    stopping_condition="consensus"  # Stop when agents agree
)

# Agents take turns discussing the problem
await round_robin_chat.chat("How should we implement authentication?")
```

**Output:**
```
Round 1:
  Researcher: "I recommend JWT tokens with refresh strategy..."
  Coder: "I can implement that with passport.js..."
  Reviewer: "Consider security implications of refresh tokens..."

Round 2:
  Researcher: "Good point, let me research secure refresh patterns..."
  Coder: "I'll add token rotation and blacklisting..."
  Reviewer: "That addresses my concerns, looks good..."

Round 3: [CONSENSUS] All agents agree on approach
```

### Pattern 3: Autonomous Delegation (Like AG2 Handoffs)

```python
# New pattern: Agents autonomously delegate to specialists
autonomous_team = AgentChain(
    agents={
        "orchestrator": orchestrator_agent,  # Coordinator
        "specialist_1": specialist_1_agent,
        "specialist_2": specialist_2_agent
    },
    execution_mode="autonomous_delegation",  # New mode
    allow_parallel=True  # Specialists work concurrently
)

# Orchestrator decides who to delegate to and when
result = await autonomous_team.execute("Build full-stack app")

# Behind the scenes:
# 1. Orchestrator analyzes task
# 2. Delegates backend to specialist_1
# 3. Delegates frontend to specialist_2 (parallel)
# 4. Synthesizes results
```

---

## Implementation Strategy: Preserve Sequential, Add Agentic

### Phase 1: Fix Critical Bugs (No Architecture Changes)
- Fix MCP tool parameter mismatches
- Fix event loop race conditions
- Fix JSON parsing errors
- **DO NOT touch sequential pipeline logic**

### Phase 2: Add Agentic Capabilities (Additive, Not Destructive)

#### 2.1 Round-Robin Chat Mode
```python
# Add to AgentChain without breaking existing modes
class AgentChain:
    def __init__(
        self,
        execution_mode: Literal[
            "router",      # Existing
            "pipeline",    # Existing
            "broadcast",   # Existing
            "round-robin", # NEW - circular agent turns
            "autonomous"   # NEW - self-organizing agents
        ]
    ):
        ...
```

#### 2.2 Non-Blocking Execution (Optional)
```python
# Add async variant, keep sync working
class AgenticStepProcessor:
    def execute(self, input_data):  # Keep existing sync
        ...

    async def execute_async(self, input_data):  # Add async variant
        # Non-blocking with interrupt queue
        ...

# Users choose which to use:
result = processor.execute(data)  # Blocking (sequential)
result = await processor.execute_async(data)  # Non-blocking (agentic)
```

#### 2.3 Interrupt Queue (For Agentic Only)
```python
# Only in async mode, doesn't affect sequential
processor = AgenticStepProcessor(
    mode="agentic",  # NEW
    allow_interrupts=True  # NEW
)

# In sequential mode, no interrupts (deterministic)
processor = AgenticStepProcessor(
    mode="sequential",  # DEFAULT
    allow_interrupts=False  # DEFAULT
)
```

---

## Comparison: Sequential vs Agentic

| Feature | Sequential Mode | Agentic Mode |
|---------|----------------|--------------|
| **Execution Order** | Predefined steps | Dynamic decisions |
| **Debuggability** | High (step-by-step) | Medium (emergent behavior) |
| **Predictability** | High (deterministic) | Low (adaptive) |
| **Speed** | Fast (no overhead) | Slower (reasoning loops) |
| **Use Cases** | Production pipelines | Exploration, research |
| **Control Flow** | Fixed at design time | Emerges at runtime |
| **Best For** | Known workflows | Unknown problems |
| **Reliability** | Very High | Medium (can wander) |
| **Token Efficiency** | High (minimal waste) | Lower (exploration cost) |

---

## Architectural Principles

### 1. Sequential is Default (Preserve Strength)
- All existing code continues to work
- Default `execution_mode="pipeline"`
- No breaking changes to core PromptChain

### 2. Agentic is Opt-In (Additive)
- New modes require explicit selection
- `execution_mode="round-robin"` or `"autonomous"`
- Clear documentation on when to use each

### 3. Hybrid Compositions (Best of Both)
- Sequential pipelines can embed agentic steps
- Agentic agents can call sequential chains
- User chooses the right tool for each problem

### 4. Backward Compatibility (No Regressions)
- All existing tests pass without changes
- Existing code examples still work
- Migration guide for new features

---

## Example: Preserving Sequential While Adding Agentic

### Before (Sequential - Keep This Working)
```python
# This is the strength of PromptChain - don't break it!
chain = PromptChain(
    models=["gpt-4"],
    instructions=[
        "Extract entities: {input}",
        "Classify sentiment: {input}",
        "Generate summary: {input}"
    ]
)

result = chain.process_prompt(document)  # Fast, predictable, reliable
```

### After (Agentic - Add This Alongside)
```python
# New capability without breaking sequential
agentic_chain = PromptChain(
    models=["gpt-4"],
    instructions=[
        AgenticRoundRobinChat(  # New instruction type
            agents=["researcher", "analyst", "writer"],
            max_rounds=3,
            objective="Analyze document comprehensively"
        )
    ]
)

result = await agentic_chain.process_prompt_async(document)
```

### Hybrid (Best of Both)
```python
# Combine sequential and agentic in one workflow
hybrid = PromptChain(
    models=["gpt-4"],
    instructions=[
        "Step 1: Extract entities (sequential) {input}",
        AgenticStepProcessor(  # Agentic reasoning
            objective="Research entity relationships",
            mode="agentic"
        ),
        "Step 2: Generate knowledge graph (sequential) {input}"
    ]
)
```

---

## Implementation Roadmap (Revised)

### Sprint 1: Fix Critical Bugs (No Architecture Changes)
- [x] Bug hunting complete (23 bugs found)
- [ ] Fix P0 bugs (Issues #1-3)
- [ ] Fix P1 bugs (Issues #4-5)
- **NO changes to sequential pipeline logic**

### Sprint 2: Add Round-Robin Chat Mode (New Capability)
- [ ] Add `execution_mode="round-robin"` to AgentChain
- [ ] Implement circular turn-taking
- [ ] Add stopping conditions (max_rounds, consensus)
- [ ] Keep all existing modes working

### Sprint 3: Add Autonomous Delegation (New Capability)
- [ ] Add `execution_mode="autonomous"` to AgentChain
- [ ] Implement delegation logic (orchestrator pattern)
- [ ] Add parallel execution for independent tasks
- [ ] Keep sequential mode as default

### Sprint 4: Add Non-Blocking Async (Optional Enhancement)
- [ ] Create `AsyncAgenticStepProcessor` (new class, not refactor)
- [ ] Add interrupt queue (async mode only)
- [ ] Keep sync `AgenticStepProcessor` working
- [ ] Document when to use each

### Sprint 5: Add Context Management (Both Modes)
- [ ] Context distillation (benefits both sequential and agentic)
- [ ] Semantic memo store (long-term memory)
- [ ] Improved history management
- [ ] Works for both modes

---

## Key Takeaways

1. **Sequential pipelines are PromptChain's killer feature** - preserve at all costs
2. **Agentic flows are new capabilities** - add alongside, not instead of
3. **Users choose the right mode** - provide clear guidance on when to use each
4. **Hybrid compositions are powerful** - enable mixing both paradigms
5. **Backward compatibility is critical** - no breaking changes

---

## Success Metrics (Revised)

### Sequential Mode (Preserve Performance)
- ✅ All existing tests pass without changes
- ✅ No performance regression in sequential execution
- ✅ No new dependencies required for sequential mode
- ✅ Existing code examples still work

### Agentic Mode (New Value)
- 🆕 Round-robin chat enables multi-agent collaboration
- 🆕 Autonomous delegation reduces manual coordination
- 🆕 Non-blocking execution improves UX
- 🆕 Interrupt queue enables real-time steering

### Hybrid Mode (Innovation)
- 🆕 Sequential pipelines can embed agentic reasoning
- 🆕 Agentic agents can leverage sequential chains
- 🆕 Users report better outcomes combining both modes
- 🆕 Clear patterns emerge for hybrid workflows

---

*This architecture vision preserves PromptChain's sequential strength while adding agentic capabilities for a hybrid system that offers both deterministic pipelines AND adaptive agents.*
