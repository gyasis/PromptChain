# Orchestrator Enhancement: Task Decomposition & Multi-Agent Pipelines

**Problem**: Current orchestrator picks ONE agent per query, even when complex tasks need multiple agents

**Example Failure**:
```
User: "write a tutorial with a zig tutorial about manipulating sound"

Current Behavior:
✗ Orchestrator → Documentation agent (no research tools)
✗ Documentation writes from possibly outdated/hallucinated knowledge

Desired Behavior:
✓ Orchestrator → Creates plan: [Research → Documentation]
✓ Research agent → Gets current Zig audio library info
✓ Documentation agent → Writes tutorial based on research
✓ Final output: Accurate, current tutorial
```

---

## Core Problem Analysis

### Current Limitations

1. **Single Agent Selection**: Router picks ONE agent per query
2. **No Task Decomposition**: Can't break complex tasks into subtasks
3. **No Planning**: No awareness of multi-step workflows
4. **Agent Capabilities Ignored**: Documentation chosen despite lacking research tools

### What We Need

1. **Task Complexity Analysis**: Identify when tasks need multiple agents
2. **Decomposition Strategy**: Break tasks into logical subtasks
3. **Agent Pipeline Planning**: Sequence agents appropriately
4. **Result Synthesis**: Combine multi-agent outputs coherently

---

## High-Level Architecture

### Phase 1: Task Analysis

```python
async def analyze_task_complexity(user_query: str, history: list) -> dict:
    """
    Determine if query needs single or multiple agents

    Returns:
    {
        "complexity": "simple" | "complex",
        "subtasks": ["subtask1", "subtask2", ...],
        "requires_research": bool,
        "requires_code": bool,
        "requires_documentation": bool
    }
    """
```

**Classification Logic**:
```
Simple Tasks (single agent):
- "What is 2+2?"
- "Explain neural networks" (known topic)
- "Create a backup script"

Complex Tasks (multi-agent):
- "Write tutorial about [UNKNOWN] topic" → research + documentation
- "Implement feature X with tests" → coding + terminal
- "Research Y and create presentation" → research + synthesis
- "Analyze data and visualize" → analysis + coding
```

### Phase 2: Plan Generation

```python
async def generate_execution_plan(task_analysis: dict, user_query: str) -> list:
    """
    Create agent execution plan with subtask routing

    Returns:
    [
        {
            "step": 1,
            "agent": "research",
            "objective": "Gather current info about Zig audio libraries",
            "output_format": "structured_summary"
        },
        {
            "step": 2,
            "agent": "documentation",
            "objective": "Write tutorial using research from step 1",
            "inputs_from": ["step_1"],
            "output_format": "markdown_tutorial"
        }
    ]
    """
```

**Plan Templates** (Pre-defined for common patterns):

```python
PLAN_TEMPLATES = {
    "research_then_write": [
        {"agent": "research", "objective": "Gather information"},
        {"agent": "documentation", "objective": "Write comprehensive guide"}
    ],

    "code_then_test": [
        {"agent": "coding", "objective": "Write implementation"},
        {"agent": "terminal", "objective": "Run tests and validate"}
    ],

    "research_analyze_synthesize": [
        {"agent": "research", "objective": "Gather data"},
        {"agent": "analysis", "objective": "Analyze findings"},
        {"agent": "synthesis", "objective": "Create final report"}
    ],

    "prototype_iterate": [
        {"agent": "coding", "objective": "Create initial version"},
        {"agent": "terminal", "objective": "Test and get feedback"},
        {"agent": "coding", "objective": "Refine based on results"}
    ]
}
```

### Phase 3: Sequential Execution

```python
async def execute_plan(plan: list, user_query: str, history: list) -> dict:
    """
    Execute multi-agent plan sequentially, passing context forward

    Returns:
    {
        "final_output": "Combined result",
        "steps_executed": [
            {"agent": "research", "output": "..."},
            {"agent": "documentation", "output": "..."}
        ],
        "total_time_ms": 5000,
        "plan_followed": true
    }
    """

    accumulated_context = []

    for step in plan:
        # Build input with context from previous steps
        step_input = build_step_input(
            original_query=user_query,
            step_objective=step["objective"],
            previous_outputs=accumulated_context
        )

        # Execute agent
        agent = agents[step["agent"]]
        step_output = await agent.process_prompt_async(step_input)

        # Accumulate context
        accumulated_context.append({
            "agent": step["agent"],
            "output": step_output
        })

    # Synthesize final output
    final_output = synthesize_results(accumulated_context, user_query)

    return {
        "final_output": final_output,
        "steps_executed": accumulated_context
    }
```

---

## Implementation Options

### Option 1: Enhanced AgenticStepProcessor (Recommended)

**Create `TaskPlanningOrchestrator`** - Meta-agent that uses AgenticStepProcessor for planning:

```python
class TaskPlanningOrchestrator:
    """
    Enhanced orchestrator with task decomposition and multi-agent planning
    """

    def __init__(self, agents: dict, agent_descriptions: dict):
        self.agents = agents
        self.agent_descriptions = agent_descriptions

        # Planning agent (uses AgenticStepProcessor)
        self.planner = AgenticStepProcessor(
            objective="""You are the Master Planner. Analyze complex tasks and create
            multi-agent execution plans. If task is simple, route to single agent.
            If complex, decompose into subtasks and assign agents.""",
            max_internal_steps=5,
            model_name="openai/gpt-4.1-mini"
        )

    async def process_query(self, user_query: str, history: list) -> dict:
        # Step 1: Analyze complexity
        analysis = await self.analyze_task(user_query)

        # Step 2: Simple or complex routing
        if analysis["complexity"] == "simple":
            return await self.simple_route(user_query)
        else:
            return await self.execute_multi_agent_plan(user_query, analysis)

    async def analyze_task(self, query: str) -> dict:
        """Use planner to determine if task needs decomposition"""
        analysis_prompt = f"""
        Analyze this query and determine complexity:
        Query: {query}

        Return JSON:
        {{
            "complexity": "simple" | "complex",
            "reasoning": "why this classification",
            "requires_research": bool,
            "suggested_plan": "template_name or custom"
        }}
        """

        result = await self.planner.run_async(analysis_prompt)
        return json.loads(result)

    async def execute_multi_agent_plan(self, query: str, analysis: dict) -> dict:
        """Execute multi-step agent pipeline"""

        # Generate plan
        plan = self.generate_plan(query, analysis)

        # Execute sequentially
        context = []
        for step in plan:
            agent = self.agents[step["agent"]]

            # Build input with accumulated context
            step_input = f"""
            Original Query: {query}

            Your Objective: {step["objective"]}

            Context from Previous Steps:
            {self.format_context(context)}

            Complete your objective now:
            """

            # Execute
            output = await agent.process_prompt_async(step_input)

            context.append({
                "agent": step["agent"],
                "objective": step["objective"],
                "output": output
            })

        # Return final output (last agent's result)
        return {
            "response": context[-1]["output"],
            "plan_executed": plan,
            "all_steps": context
        }
```

### Option 2: Pipeline Mode Enhancement

**Extend AgentChain** to support dynamic pipeline generation:

```python
# In create_agentic_orchestrator():

async def agentic_router_wrapper(user_input, history, agent_descriptions):
    # Check if task needs decomposition
    needs_multi_agent = await check_task_complexity(user_input)

    if needs_multi_agent:
        # Generate and execute plan
        plan = await generate_plan(user_input, agent_descriptions)
        result = await execute_sequential_plan(plan, user_input)
        return json.dumps({"chosen_agent": "multi_agent_pipeline", "result": result})
    else:
        # Original single-agent routing
        decision = await orchestrator_chain.process_prompt_async(user_input)
        return decision
```

---

## Enhanced Orchestrator Objective

```python
orchestrator_step = AgenticStepProcessor(
    objective=f"""You are the Master Orchestrator with two modes:

MODE 1: SIMPLE ROUTING (single agent)
- Use when query fits one agent's capabilities
- Direct questions, simple tasks, known topics

MODE 2: TASK DECOMPOSITION (multi-agent pipeline)
- Use when query needs multiple specialized skills
- Research + Write, Code + Test, Analyze + Visualize

CURRENT DATE: {{current_date}}

AVAILABLE AGENTS:
{agents_formatted}

COMMON MULTI-AGENT PATTERNS:
1. Research → Documentation (for tutorials/guides on unknown topics)
2. Research → Analysis → Synthesis (for comprehensive reports)
3. Coding → Terminal (for implementation + testing)
4. Research → Coding → Terminal (for building with new libraries)

DECISION PROCESS:
1. Analyze: Does query need multiple agent skills?
2. If NO: Return single agent choice
3. If YES: Return execution plan

OUTPUT FORMATS:

Single Agent:
{{"chosen_agent": "agent_name", "reasoning": "why"}}

Multi-Agent Plan:
{{
  "execution_mode": "pipeline",
  "plan": [
    {{"step": 1, "agent": "research", "objective": "gather info"}},
    {{"step": 2, "agent": "documentation", "objective": "write tutorial"}}
  ],
  "reasoning": "why multi-agent needed"
}}
""",
    max_internal_steps=5,
    model_name="openai/gpt-4.1-mini",
    history_mode="progressive"
)
```

---

## Example Execution Flow

### User Query: "Write a tutorial about manipulating sound with Zig"

**Step 1: Task Analysis**
```json
{
  "complexity": "complex",
  "reasoning": "Requires current knowledge of Zig audio libraries (research) + tutorial writing (documentation)",
  "execution_mode": "pipeline",
  "plan": [
    {
      "step": 1,
      "agent": "research",
      "objective": "Find current Zig audio libraries, tools, and best practices for sound manipulation",
      "output_format": "structured_summary"
    },
    {
      "step": 2,
      "agent": "documentation",
      "objective": "Write comprehensive tutorial using research findings",
      "output_format": "markdown_tutorial"
    }
  ]
}
```

**Step 2: Execute Research Agent**
```
Input: "Find current Zig audio libraries, tools, and best practices for sound manipulation"

Research Agent (with MCP gemini_research):
→ Calls gemini_research with topic
→ Gets current Zig audio ecosystem info
→ Returns: "Zig audio: miniaudio bindings, zound library, WAV parsing..."
```

**Step 3: Execute Documentation Agent**
```
Input:
"Original Query: Write tutorial about sound manipulation with Zig

Context from Research Agent:
- Zig has miniaudio bindings for cross-platform audio
- zound library for audio processing
- Direct C integration for libraries like libsndfile
- [full research output]

Write a comprehensive tutorial based on this current information."

Documentation Agent:
→ Writes tutorial using REAL, CURRENT info
→ Returns: Complete tutorial with actual libraries
```

**Final Output**: Accurate tutorial with current Zig audio libraries!

---

## Implementation Roadmap

### Phase 1: Basic Task Analysis (1-2 hours)
- Add complexity detection to orchestrator
- Identify queries needing multi-agent

### Phase 2: Simple Pipeline Execution (2-3 hours)
- Implement sequential agent execution
- Context passing between agents
- Result synthesis

### Phase 3: Plan Templates (1-2 hours)
- Pre-defined patterns (research→write, code→test)
- Template matching logic

### Phase 4: Dynamic Planning (3-4 hours)
- LLM-based plan generation for novel tasks
- Adaptive agent sequencing

### Phase 5: Testing & Refinement (2-3 hours)
- Test with various complex queries
- Tune planning logic
- Add --dev mode visibility

---

## Benefits

### 1. Accurate Results
- Research before writing = current, factual content
- Code before test = validated implementations
- Analyze before report = data-driven insights

### 2. Agent Specialization
- Each agent does what it's best at
- No "documentation agent trying to research"
- Proper tool usage (MCP tools where needed)

### 3. Comprehensive Solutions
- Complex tasks get thorough treatment
- Multiple perspectives on problems
- Better quality outputs

### 4. Transparency
- User sees multi-step execution
- Can track which agent did what
- Clear reasoning for agent choices

---

## Next Steps

1. **Decide on approach**: TaskPlanningOrchestrator class vs AgentChain enhancement
2. **Start with simple case**: Research → Documentation pipeline
3. **Test thoroughly**: Validate with tutorial request like Zig sound
4. **Expand gradually**: Add more plan templates
5. **Integrate with --dev mode**: Show plan execution clearly

Would you like me to implement Option 1 (TaskPlanningOrchestrator) or Option 2 (AgentChain enhancement)?
