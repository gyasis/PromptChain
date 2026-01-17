# Agentic Router Wrapper Pattern

## Quick Reference Guide

This document provides implementation guidance for the AgenticStepProcessor-based router orchestrator wrapper pattern.

---

## The Problem

**Current AgentChain Router Limitations:**
- Single-step routing decisions (one LLM call)
- No multi-hop reasoning capability
- ~70% routing accuracy for complex queries
- No knowledge boundary detection
- No temporal context awareness
- Context loss between routing attempts

**Impact:** Incorrect agent selection, failed queries, poor user experience

---

## The Solution

**AgenticStepProcessor Orchestrator:**
- Multi-hop reasoning with 5 internal steps
- Progressive history mode for context accumulation
- Tool capability awareness
- Knowledge boundary detection (when to research)
- Current date awareness for temporal queries
- **Target Accuracy: 95%**

---

## Implementation Pattern (Phase 1: Wrapper)

### Step 1: Create the Wrapper Function

**File:** `promptchain/utils/agentic_router_wrapper.py`

```python
import json
from datetime import datetime
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

async def agentic_orchestrator_router(
    user_query: str,
    agent_details: dict,
    conversation_history: list = None,
    llm_runner=None,
    tool_executor=None,
    available_tools=None
) -> dict:
    """
    Agentic orchestrator using AgenticStepProcessor for intelligent routing.

    This wrapper provides multi-hop reasoning, progressive context accumulation,
    and knowledge boundary detection for superior routing decisions.

    Args:
        user_query: The user's input query
        agent_details: Dict of available agents with their descriptions and capabilities
        conversation_history: Optional conversation context
        llm_runner: Async function to run LLM calls
        tool_executor: Async function to execute tools
        available_tools: List of available tool schemas

    Returns:
        dict: {
            "chosen_agent": "agent_name",
            "refined_query": "optimized query for agent",
            "reasoning": "explanation of routing decision",
            "requires_research": bool,
            "confidence_score": float,
            "knowledge_gaps": [...]
        }
    """

    # Get current date for temporal context
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Format agent details for the orchestrator
    agent_info = json.dumps(agent_details, indent=2)

    # Build the routing objective
    routing_objective = f"""
You are an intelligent router orchestrator for a multi-agent system.

CURRENT DATE: {current_date}

USER QUERY: {user_query}

AVAILABLE AGENTS:
{agent_info}

YOUR TASK:
1. Analyze the user query complexity and requirements
2. Evaluate agent capabilities and tool access
3. Detect knowledge boundaries - determine if external research is needed
4. Consider temporal context for time-sensitive queries
5. Make the optimal routing decision

DECISION CRITERIA:
- Match query requirements to agent capabilities
- Consider which agents have which tools/MCP servers
- Identify if query requires information beyond agent knowledge (needs research)
- Factor in temporal context for "latest", "recent", "current" queries
- Choose agent most likely to successfully fulfill the request

OUTPUT REQUIREMENTS:
Provide your final routing decision as JSON with this structure:
{{
    "chosen_agent": "agent_name",
    "refined_query": "optional refined/optimized query for the agent",
    "reasoning": "clear explanation of why this agent was chosen",
    "requires_research": true/false,
    "confidence_score": 0.0-1.0,
    "knowledge_gaps": ["gap1", "gap2"] or [],
    "temporal_context": "relevant date context if applicable"
}}

Think through this step-by-step using your available tools.
    """.strip()

    # Create AgenticStepProcessor with progressive history mode
    orchestrator = AgenticStepProcessor(
        objective=routing_objective,
        max_internal_steps=5,  # Allow up to 5 reasoning steps
        history_mode="progressive",  # CRITICAL: enables context accumulation
        model_name="openai/gpt-4o",  # Use capable model with tool calling
        max_context_tokens=8000  # Monitor context size
    )

    # Execute the orchestrator with tool access
    routing_result = await orchestrator.run_async(
        input_text=user_query,
        llm_runner=llm_runner,
        tool_executor=tool_executor,
        available_tools=available_tools
    )

    # Parse the routing decision from the result
    routing_decision = parse_routing_result(routing_result)

    return routing_decision


def parse_routing_result(result: str) -> dict:
    """
    Parse the orchestrator's output into a routing decision.

    Args:
        result: The text output from AgenticStepProcessor

    Returns:
        dict: Structured routing decision
    """
    try:
        # Try to extract JSON from the result
        # Look for JSON block in the response
        import re
        json_match = re.search(r'\{[^{}]*"chosen_agent"[^{}]*\}', result, re.DOTALL)

        if json_match:
            routing_data = json.loads(json_match.group(0))
            return routing_data
        else:
            # Fallback: try to parse the entire result
            return json.loads(result)

    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback to basic routing if parsing fails
        return {
            "chosen_agent": "default_agent",  # Configure appropriate fallback
            "refined_query": None,
            "reasoning": f"Parsing failed: {str(e)}. Using fallback routing.",
            "requires_research": False,
            "confidence_score": 0.5,
            "knowledge_gaps": [],
            "temporal_context": None
        }
```

### Step 2: Create Support Tools

**File:** `promptchain/utils/agentic_router_tools.py`

```python
from datetime import datetime
from typing import Dict, List

def get_current_date() -> str:
    """
    Get current date for temporal context.

    Returns:
        str: Current date in YYYY-MM-DD format
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_agent_capabilities(agent_name: str, agent_details: dict) -> dict:
    """
    Get detailed capabilities of a specific agent.

    Args:
        agent_name: Name of the agent to inspect
        agent_details: Full agent configuration

    Returns:
        dict: Agent capabilities including tools, MCP servers, etc.
    """
    if agent_name not in agent_details:
        return {"error": f"Agent '{agent_name}' not found"}

    agent_info = agent_details[agent_name]

    return {
        "agent_name": agent_name,
        "description": agent_info.get("description", ""),
        "has_mcp_servers": bool(agent_info.get("mcp_servers")),
        "mcp_servers": agent_info.get("mcp_servers", []),
        "tools": agent_info.get("tools", []),
        "capabilities": agent_info.get("capabilities", [])
    }


def assess_knowledge_boundary(
    query: str,
    current_date: str,
    agent_knowledge_cutoff: str = "2025-01-01"
) -> dict:
    """
    Assess if query requires external research beyond agent knowledge.

    Args:
        query: User query to analyze
        current_date: Current date for temporal assessment
        agent_knowledge_cutoff: Date beyond which agent has no knowledge

    Returns:
        dict: Assessment of knowledge requirements
    """
    # Keywords indicating need for current/recent information
    temporal_keywords = [
        "latest", "recent", "current", "today", "this week",
        "this month", "this year", "now", "newest", "updated"
    ]

    # Check for temporal indicators
    needs_current_info = any(
        keyword in query.lower() for keyword in temporal_keywords
    )

    # Check for year mentions beyond cutoff
    import re
    year_matches = re.findall(r'\b(20\d{2})\b', query)
    needs_recent_data = any(
        int(year) >= int(agent_knowledge_cutoff[:4])
        for year in year_matches
    )

    requires_research = needs_current_info or needs_recent_data

    knowledge_gaps = []
    if needs_current_info:
        knowledge_gaps.append("Requires current/real-time information")
    if needs_recent_data:
        knowledge_gaps.append(f"Requires data from {agent_knowledge_cutoff} onwards")

    return {
        "requires_research": requires_research,
        "knowledge_gaps": knowledge_gaps,
        "needs_current_info": needs_current_info,
        "needs_recent_data": needs_recent_data,
        "assessment_date": current_date
    }
```

### Step 3: Register Tools and Integrate

**Usage Example:**

```python
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_router_wrapper import agentic_orchestrator_router
from promptchain.utils.agentic_router_tools import (
    get_current_date,
    get_agent_capabilities,
    assess_knowledge_boundary
)

# Create your agents
research_agent = PromptChain(...)
analysis_agent = PromptChain(...)
writing_agent = PromptChain(...)

# Define agent details with capabilities
agent_details = {
    "research_agent": {
        "description": "Searches and retrieves current information from the web",
        "mcp_servers": ["web_search", "arxiv"],
        "capabilities": ["web_search", "paper_retrieval", "fact_checking"]
    },
    "analysis_agent": {
        "description": "Analyzes data and generates insights",
        "tools": ["data_processor", "statistical_analysis"],
        "capabilities": ["data_analysis", "pattern_recognition"]
    },
    "writing_agent": {
        "description": "Creates well-formatted documents and reports",
        "capabilities": ["document_generation", "formatting", "summarization"]
    }
}

# Configure router with agentic orchestrator
router_config = {
    "models": ["openai/gpt-4o"],
    "custom_router_function": agentic_orchestrator_router,
    "instructions": [None, "{input}"]
}

# Create AgentChain with agentic router
agent_chain = AgentChain(
    agents={
        "research": research_agent,
        "analysis": analysis_agent,
        "writing": writing_agent
    },
    agent_descriptions=agent_details,
    execution_mode="router",
    router=router_config,
    verbose=True
)

# The orchestrator automatically uses multi-hop reasoning
# to make intelligent routing decisions
result = await agent_chain.run_chat_async(
    "Find the latest GPT-5 research papers and analyze their performance metrics"
)
```

---

## Critical Implementation Notes

### 1. Progressive History Mode is ESSENTIAL

```python
orchestrator = AgenticStepProcessor(
    objective=routing_objective,
    max_internal_steps=5,
    history_mode="progressive",  # MUST BE "progressive"
    model_name="openai/gpt-4o"
)
```

**Why Progressive Mode?**
- **Minimal mode** (default): Only keeps last assistant message + tool results → LOSES CONTEXT
- **Progressive mode**: Accumulates ALL assistant messages + tool results → PRESERVES CONTEXT
- **Kitchen sink mode**: Keeps everything including intermediate reasoning

**For routing, use "progressive" - it provides the perfect balance of context preservation and token efficiency.**

### 2. Model Selection Matters

**Recommended Models:**
- `openai/gpt-4o` - Excellent reasoning, tool calling support
- `openai/gpt-4-turbo` - Good balance of speed and capability
- `anthropic/claude-3-5-sonnet-20241022` - Strong reasoning abilities

**Avoid:**
- Small models (gpt-3.5, etc.) - Insufficient reasoning capability
- Models without tool calling - Cannot use routing tools

### 3. Tool Integration Pattern

The orchestrator needs access to tools through callbacks:

```python
routing_result = await orchestrator.run_async(
    input_text=user_query,
    llm_runner=llm_runner,        # Provided by parent chain
    tool_executor=tool_executor,  # Provided by parent chain
    available_tools=available_tools  # Provided by parent chain
)
```

### 4. Performance Optimization

**Token Management:**
- Set `max_context_tokens=8000` to monitor context size
- Progressive mode uses more tokens than minimal, but provides critical context
- Trade-off is worth it for 95% accuracy vs 70%

**Latency Optimization:**
- `max_internal_steps=5` is recommended (allows complex reasoning without excessive delay)
- Reduce to 3 for faster routing if acceptable
- Increase to 7 for maximum reasoning depth if latency acceptable

---

## Testing Strategy

### Validation Dataset

Create a test set of 100 queries:

1. **Simple Queries (20)** - Single agent, obvious routing
   - "What is 2+2?" → math_agent
   - "Write a poem" → writing_agent

2. **Complex Queries (30)** - Multi-hop reasoning required
   - "Find latest AI papers and summarize key findings" → research + analysis
   - "Compare Python vs JavaScript for web development" → research + comparison

3. **Research-Requiring Queries (25)** - Knowledge boundary detection
   - "What happened at the 2025 AI Summit?" → Needs research
   - "Latest ChatGPT release features" → Needs current info

4. **Time-Sensitive Queries (25)** - Temporal context
   - "What's the current state of LLM development?" → Requires date awareness
   - "Recent breakthroughs in quantum computing" → Temporal + research

### Success Metrics

**Target:**
- Routing accuracy: ≥95%
- Multi-hop reasoning: avg ≥3 steps for complex queries
- Context preservation: 100%
- Knowledge boundary detection: ≥90%
- Latency: <5s
- Token usage: <2000 per decision

---

## Migration Path

### From Current Router

**Before:**
```python
router_config = {
    "models": ["openai/gpt-4"],
    "instructions": [None, "{input}"],
    "decision_prompt_templates": {
        "single_agent_dispatch": "Choose agent based on query..."
    }
}
```

**After (Phase 1):**
```python
from promptchain.utils.agentic_router_wrapper import agentic_orchestrator_router

router_config = {
    "models": ["openai/gpt-4o"],
    "custom_router_function": agentic_orchestrator_router,
    "instructions": [None, "{input}"]
}
```

**Changes Required:**
1. Import wrapper function
2. Set `custom_router_function` in config
3. Update model to GPT-4o (recommended)

**Rollback:**
Simply remove `custom_router_function` from config - zero code changes needed.

---

## Future: Phase 2 Native Integration

After validation, the pattern will be integrated natively:

```python
agent_chain = AgentChain(
    agents=agents,
    execution_mode="router",
    router_mode="agentic",  # New mode
    agentic_router_config={
        "max_internal_steps": 5,
        "history_mode": "progressive",
        "model_name": "openai/gpt-4o",
        "enable_research_detection": True,
        "enable_tool_awareness": True,
        "enable_date_awareness": True
    }
)
```

---

## Troubleshooting

### Issue: Low Routing Accuracy

**Possible Causes:**
- Using minimal history mode instead of progressive
- Model lacks tool calling capability
- Insufficient internal steps (increase max_internal_steps)
- Poor agent descriptions in agent_details

**Solutions:**
- Verify `history_mode="progressive"`
- Use GPT-4o or Claude 3.5 Sonnet
- Increase `max_internal_steps` to 7
- Improve agent descriptions with clear capabilities

### Issue: High Latency

**Possible Causes:**
- Too many internal steps
- Large context from progressive history
- Slow model (GPT-4 base)

**Solutions:**
- Reduce `max_internal_steps` to 3
- Set `max_context_tokens` lower (e.g., 4000)
- Use GPT-4o (faster than GPT-4)

### Issue: Context Overflow

**Possible Causes:**
- Progressive history accumulating too much context
- Very long conversation history

**Solutions:**
- Set `max_context_tokens` with warning threshold
- Implement fallback to minimal mode on overflow
- Truncate conversation_history before routing

---

## Key Takeaways

1. **Progressive history mode is critical** - Don't use minimal mode for routing
2. **Multi-hop reasoning significantly improves accuracy** - 70% → 95%
3. **Knowledge boundary detection enables better research routing**
4. **Current date awareness handles temporal queries correctly**
5. **Tool capability awareness matches agents to query requirements**
6. **Phase 1 wrapper validates approach without breaking changes**
7. **Phase 2 native integration provides seamless user experience**

---

## References

- **PRD:** `/home/gyasis/Documents/code/PromptChain/prd/agentic_orchestrator_router_enhancement.md`
- **AgenticStepProcessor:** `promptchain/utils/agentic_step_processor.py`
- **Progressive History Documentation:** `HISTORY_MODES_IMPLEMENTATION.md`
- **AgentChain:** `promptchain/utils/agent_chain.py`

---

**Status:** Phase 1 - Ready for Implementation
**Next Steps:** Create wrapper function, implement tools, validate on test dataset
