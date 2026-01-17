# Product Requirements Document: Agentic Orchestrator Router Enhancement

**Version:** 1.0
**Date:** October 4, 2025
**Status:** Phase 1 - Validation Stage
**Owner:** PromptChain Core Team

---

## Executive Summary

This PRD defines the enhancement of AgentChain's router system through an AgenticStepProcessor-based orchestrator that provides multi-hop reasoning, progressive context accumulation, and knowledge boundary detection. The implementation follows a phased approach: first validating the solution through an async wrapper function, then integrating it as a native library feature.

**Target Impact:**
- Router accuracy improvement: 70% → 95%
- Multi-hop reasoning capability (5+ internal reasoning steps)
- Knowledge boundary detection (when to research vs use internal knowledge)
- Progressive history accumulation for context preservation

---

## Problem Statement

### Current AgentChain Router Limitations

1. **Single-Step Routing Decisions**
   - Current router makes routing decisions in one LLM call
   - No iterative reasoning or refinement capability
   - Cannot decompose complex queries requiring multiple steps

2. **Context Loss Issues**
   - No progressive history accumulation across routing decisions
   - Router "forgets" previous reasoning when making subsequent decisions
   - Limited multi-hop reasoning capability

3. **Suboptimal Routing Accuracy**
   - Current accuracy: ~70% for complex queries
   - No mechanism for knowledge boundary detection
   - Cannot distinguish between tasks requiring research vs internal knowledge

4. **Knowledge Boundary Blindness**
   - Router cannot determine when additional research is needed
   - No awareness of what information is available vs what needs to be fetched
   - Cannot detect when current context is insufficient

5. **Temporal Context Unawareness**
   - No current date awareness for time-sensitive queries
   - Cannot factor recency into routing decisions
   - Missing temporal context for "latest" or "recent" query types

### Impact on User Experience

- **Incorrect Agent Selection:** Users frequently routed to wrong specialist agents
- **Failed Query Resolution:** Complex queries fail due to insufficient reasoning depth
- **Inefficient Workflows:** Manual intervention required to re-route failed queries
- **Poor Research Quality:** Missing research step when external knowledge needed

### Business Impact

- **Reduced User Trust:** Inconsistent routing leads to user frustration
- **Higher Support Costs:** Manual routing corrections increase operational overhead
- **Degraded Value Proposition:** Multi-agent system underperforms vs single-agent systems
- **Competitive Disadvantage:** Other frameworks with better routing outperform PromptChain

---

## Proposed Solution

### AgenticStepProcessor Orchestrator

Replace the current single-step router with an AgenticStepProcessor-based orchestrator that provides:

1. **Multi-Hop Reasoning**
   - Internal reasoning loop with configurable steps (default: 5)
   - Iterative refinement of routing decisions
   - Ability to decompose complex queries into sub-tasks

2. **Progressive History Mode**
   - Context accumulation across reasoning steps
   - Preservation of previous reasoning chains
   - Knowledge build-up over multiple internal iterations

3. **Tool Capability Awareness**
   - Knowledge of which agents have which tools/capabilities
   - Routing decisions based on tool requirements
   - Dynamic tool availability checking

4. **Knowledge Boundary Detection**
   - Determine when external research is needed
   - Identify gaps in available information
   - Route to research agents when appropriate

5. **Current Date Awareness**
   - Temporal context for time-sensitive queries
   - "Latest" and "recent" query understanding
   - Date-based routing logic

### Architecture Overview

```python
# Phase 1: Async Wrapper Function (Validation)
async def agentic_orchestrator_router(
    user_query: str,
    agent_details: dict,
    conversation_history: list,
    current_date: str
) -> dict:
    """
    Agentic orchestrator using AgenticStepProcessor for intelligent routing.

    Returns: {
        "chosen_agent": "agent_name",
        "refined_query": "optimized query for agent",
        "reasoning": "explanation of routing decision"
    }
    """

    # Create AgenticStepProcessor with routing objective
    orchestrator = AgenticStepProcessor(
        objective=f"""
        Analyze user query and route to the most appropriate agent.

        User Query: {user_query}
        Current Date: {current_date}

        Available Agents:
        {json.dumps(agent_details, indent=2)}

        Consider:
        1. Query complexity and requirements
        2. Agent capabilities and tools
        3. Knowledge boundaries - research if info unavailable
        4. Temporal context for time-sensitive queries
        5. Multi-step decomposition if needed

        Return final routing decision as JSON.
        """,
        max_internal_steps=5,
        history_mode="progressive",  # Critical for multi-hop reasoning
        model_name="openai/gpt-4o",
        max_context_tokens=8000
    )

    # Execute orchestrator with tool access
    result = await orchestrator.run_async(
        input_text=user_query,
        llm_runner=chain.run_model_async,
        tool_executor=chain.execute_tool_async,
        available_tools=chain.tools
    )

    # Parse and validate routing decision
    routing_decision = parse_routing_result(result)
    return routing_decision

# Phase 2: Native Library Integration (Future)
class AgentChain:
    def __init__(
        self,
        agents: dict,
        router_mode: str = "agentic",  # "simple", "llm", "agentic"
        agentic_router_config: dict = None,
        ...
    ):
        if router_mode == "agentic":
            self._init_agentic_router(agentic_router_config)
```

---

## Implementation Phases

### Phase 1: Validation Through Async Wrapper (CURRENT)

**Timeline:** Week 1-2
**Status:** In Progress
**Goal:** Validate approach without library changes

#### Implementation Details

1. **Create Wrapper Function**
   - Location: `promptchain/utils/agentic_router_wrapper.py`
   - Async function wrapping AgenticStepProcessor
   - Non-breaking - works alongside existing router

2. **Integration Pattern**
   ```python
   # In AgentChain router configuration
   router_config = {
       "models": ["openai/gpt-4o"],
       "custom_router_function": agentic_orchestrator_router,
       "instructions": [None, "{input}"]
   }
   ```

3. **Tool Integration**
   - Agent capability discovery tool
   - Current date fetching tool
   - Knowledge boundary assessment tool

4. **Validation Metrics**
   - Routing accuracy measurement (target: 95%)
   - Multi-hop reasoning effectiveness
   - Context preservation verification
   - Knowledge boundary detection accuracy

#### Success Criteria

- ✅ Routing accuracy ≥ 95% on test dataset
- ✅ Successful multi-hop reasoning demonstrated
- ✅ Progressive history mode preserves context
- ✅ Knowledge boundary detection works correctly
- ✅ Current date awareness functional
- ✅ No performance regression vs current router

### Phase 2: Native Library Integration (FUTURE)

**Timeline:** Week 3-6
**Status:** Planned
**Goal:** Integrate as native AgentChain feature

#### Implementation Details

1. **AgentChain Router Modes**
   - Add "agentic" router mode to existing "simple" and "llm" modes
   - Backward compatible - existing routers unchanged
   - Opt-in feature with configuration flag

2. **Router Configuration Schema**
   ```python
   agentic_router_config = {
       "max_internal_steps": 5,
       "history_mode": "progressive",
       "model_name": "openai/gpt-4o",
       "max_context_tokens": 8000,
       "enable_research_detection": True,
       "enable_tool_awareness": True,
       "enable_date_awareness": True
   }
   ```

3. **Tool Ecosystem Integration**
   - Built-in agent capability introspection
   - Automatic current date injection
   - Knowledge boundary assessment utilities

4. **Migration Path**
   - Simple config change to enable agentic router
   - Wrapper function remains compatible
   - Gradual rollout with A/B testing capability

#### Success Criteria

- ✅ Drop-in replacement for existing router
- ✅ Configuration-based activation
- ✅ Backward compatibility maintained
- ✅ Performance benchmarks met
- ✅ Documentation complete
- ✅ Migration guide available

---

## Technical Specifications

### Core Components

#### 1. AgenticStepProcessor Configuration

```python
orchestrator = AgenticStepProcessor(
    objective=routing_objective,
    max_internal_steps=5,           # Configurable reasoning depth
    history_mode="progressive",      # CRITICAL: enables context accumulation
    model_name="openai/gpt-4o",     # High-performance model for reasoning
    max_context_tokens=8000          # Context window management
)
```

**Key Parameters:**
- `max_internal_steps`: Controls reasoning depth (5 recommended)
- `history_mode`: MUST be "progressive" for multi-hop reasoning
- `model_name`: Should be capable of tool calling (GPT-4, Claude 3+)
- `max_context_tokens`: Prevents context overflow

#### 2. Tool Integration

**Required Tools:**

1. **Agent Capability Discovery**
   ```python
   def get_agent_capabilities(agent_name: str) -> dict:
       """Return tools, MCP servers, and capabilities of agent"""
   ```

2. **Current Date Tool**
   ```python
   def get_current_date() -> str:
       """Return current date for temporal context"""
   ```

3. **Knowledge Boundary Assessment**
   ```python
   def assess_knowledge_boundary(query: str, available_info: dict) -> dict:
       """Determine if external research needed"""
   ```

#### 3. Progressive History Implementation

**Critical Implementation Note:**
The `history_mode="progressive"` parameter is ESSENTIAL for multi-hop reasoning.

**How It Works:**
```python
# Minimal mode (OLD - loses context)
llm_history = [system, user, last_assistant, last_tools]

# Progressive mode (NEW - accumulates context)
conversation_history.append(last_assistant)
conversation_history.extend(last_tools)
llm_history = [system, user] + conversation_history
```

**Impact:**
- **Without progressive mode:** Router "forgets" previous reasoning steps
- **With progressive mode:** Each step builds on previous context
- **Result:** True multi-hop reasoning with knowledge accumulation

### Routing Decision Schema

```json
{
  "chosen_agent": "string",
  "refined_query": "string (optional)",
  "reasoning": "string",
  "requires_research": "boolean",
  "confidence_score": "float (0-1)",
  "multi_hop_steps": ["step1", "step2", ...],
  "knowledge_gaps": ["gap1", "gap2", ...],
  "temporal_context": "string"
}
```

### Performance Requirements

| Metric | Current | Target | Phase 1 | Phase 2 |
|--------|---------|--------|---------|---------|
| Routing Accuracy | ~70% | 95% | ✅ Validate | ✅ Maintain |
| Reasoning Steps | 1 | 5 | ✅ Implement | ✅ Optimize |
| Context Preservation | ❌ None | ✅ Full | ✅ Progressive | ✅ Enhanced |
| Knowledge Detection | ❌ None | ✅ Yes | ✅ Basic | ✅ Advanced |
| Latency | ~1-2s | <5s | ✅ Monitor | ✅ Optimize |
| Token Usage | ~500 | <2000 | ✅ Track | ✅ Optimize |

---

## Success Metrics

### Primary KPIs

1. **Routing Accuracy**
   - **Target:** 95% correct agent selection
   - **Measurement:** Human evaluation on test dataset (100 queries)
   - **Baseline:** Current router at ~70%

2. **Multi-Hop Reasoning Effectiveness**
   - **Target:** 90% of complex queries use multi-hop reasoning
   - **Measurement:** Track internal step usage in orchestrator
   - **Indicator:** avg_steps > 2 for complex queries

3. **Context Preservation**
   - **Target:** 100% context preservation across reasoning steps
   - **Measurement:** Verify progressive history accumulation
   - **Test:** Query references to previous reasoning steps

4. **Knowledge Boundary Detection**
   - **Target:** 90% accuracy in identifying research needs
   - **Measurement:** Compare router decisions to ground truth
   - **Dataset:** Queries requiring external knowledge vs internal knowledge

### Secondary KPIs

1. **Performance**
   - Latency: <5s for routing decision
   - Token usage: <2000 tokens per routing decision
   - Throughput: Handle 10+ concurrent routing requests

2. **User Experience**
   - Query success rate increase: 70% → 90%
   - User satisfaction score: >4.5/5
   - Re-routing frequency: <10%

3. **System Health**
   - Error rate: <1%
   - Fallback activation: <5%
   - Tool availability: >99%

---

## Testing Strategy

### Phase 1 Testing (Wrapper Validation)

#### 1. Unit Tests
```python
# Test progressive history accumulation
def test_progressive_history_preserves_context():
    """Verify context builds across reasoning steps"""

# Test multi-hop reasoning
def test_multi_hop_routing_decision():
    """Verify orchestrator uses multiple steps for complex queries"""

# Test knowledge boundary detection
def test_research_detection():
    """Verify correct identification of research requirements"""

# Test current date awareness
def test_temporal_context():
    """Verify date-based routing for time-sensitive queries"""
```

#### 2. Integration Tests
```python
# Test with real AgentChain
def test_wrapper_integration_with_agentchain():
    """Verify wrapper works with existing AgentChain"""

# Test tool ecosystem
def test_tool_integration():
    """Verify all required tools function correctly"""

# Test error handling
def test_error_recovery():
    """Verify graceful handling of failures"""
```

#### 3. Validation Dataset

**100 Query Test Set:**
- 20 simple queries (single-agent, no reasoning needed)
- 30 complex queries (multi-hop reasoning required)
- 25 research-requiring queries (knowledge boundary detection)
- 25 time-sensitive queries (current date awareness)

**Ground Truth:**
- Expert-labeled correct agent routing
- Reasoning step expectations
- Research requirement flags
- Expected temporal context

### Phase 2 Testing (Native Integration)

#### 1. Backward Compatibility Tests
```python
# Existing router modes still work
def test_simple_router_unchanged():
    """Verify simple router unaffected"""

def test_llm_router_unchanged():
    """Verify LLM router unaffected"""
```

#### 2. Migration Tests
```python
# Wrapper to native migration
def test_wrapper_to_native_equivalence():
    """Verify native implementation matches wrapper behavior"""
```

#### 3. Performance Benchmarks
```python
# Routing speed
def test_routing_latency():
    """Measure and validate routing decision speed"""

# Token efficiency
def test_token_usage():
    """Track and optimize token consumption"""

# Concurrent performance
def test_concurrent_routing():
    """Validate multi-request handling"""
```

---

## Migration Path

### For Existing Users (Phase 1)

**Before (Current Router):**
```python
router_config = {
    "models": ["openai/gpt-4"],
    "instructions": [None, "{input}"],
    "decision_prompt_templates": {
        "single_agent_dispatch": "Choose the best agent..."
    }
}

agent_chain = AgentChain(
    agents=agents,
    router=router_config,
    execution_mode="router"
)
```

**After (Agentic Orchestrator - Wrapper):**
```python
from promptchain.utils.agentic_router_wrapper import agentic_orchestrator_router

router_config = {
    "models": ["openai/gpt-4o"],
    "custom_router_function": agentic_orchestrator_router,
    "instructions": [None, "{input}"]
}

agent_chain = AgentChain(
    agents=agents,
    router=router_config,
    execution_mode="router"
)
```

**Key Changes:**
1. Add import for `agentic_orchestrator_router`
2. Set `custom_router_function` in router config
3. Optionally update model to GPT-4o for better performance

### For Existing Users (Phase 2)

**After (Native Agentic Router):**
```python
agent_chain = AgentChain(
    agents=agents,
    execution_mode="router",
    router_mode="agentic",  # NEW: specify agentic router
    agentic_router_config={
        "max_internal_steps": 5,
        "history_mode": "progressive",
        "model_name": "openai/gpt-4o",
        "enable_research_detection": True,
        "enable_date_awareness": True
    }
)
```

**Key Changes:**
1. Set `router_mode="agentic"`
2. Provide `agentic_router_config` with desired settings
3. Remove old `router` config (or keep for fallback)

### Rollback Strategy

**Phase 1 (Wrapper):**
- Simply remove `custom_router_function` from config
- Falls back to default LLM router
- Zero code changes required

**Phase 2 (Native):**
- Change `router_mode` back to "llm" or "simple"
- Remove `agentic_router_config`
- Previous router behavior restored

---

## Risk Assessment

### Technical Risks

1. **Performance Degradation**
   - **Risk:** Multi-hop reasoning increases latency
   - **Mitigation:** Optimize max_internal_steps, implement caching
   - **Fallback:** Timeout with single-step fallback
   - **Severity:** Medium

2. **Context Window Overflow**
   - **Risk:** Progressive history exceeds token limits
   - **Mitigation:** Implement max_context_tokens monitoring
   - **Fallback:** Switch to minimal history mode on overflow
   - **Severity:** Medium

3. **Tool Availability Issues**
   - **Risk:** Required tools fail or unavailable
   - **Mitigation:** Graceful degradation, fallback routing
   - **Fallback:** Revert to basic LLM router
   - **Severity:** Low

### Integration Risks

1. **Backward Compatibility**
   - **Risk:** Breaking changes to existing router API
   - **Mitigation:** Phase 1 wrapper approach, extensive testing
   - **Fallback:** Keep old router modes available
   - **Severity:** Low (Phase 1), Medium (Phase 2)

2. **Migration Complexity**
   - **Risk:** Users struggle to adopt new router
   - **Mitigation:** Clear documentation, gradual rollout
   - **Fallback:** Maintain wrapper and native options
   - **Severity:** Low

### Operational Risks

1. **Increased Token Costs**
   - **Risk:** Multi-hop reasoning increases API costs
   - **Mitigation:** Token usage monitoring, budget alerts
   - **Fallback:** Configurable max_internal_steps to control costs
   - **Severity:** Medium

2. **Model Availability**
   - **Risk:** GPT-4o or required models unavailable
   - **Mitigation:** Support multiple model backends
   - **Fallback:** Degrade to simpler models with reduced capabilities
   - **Severity:** Low

---

## Dependencies

### Required Components

1. **AgenticStepProcessor (Existing)**
   - Version: Latest (with progressive history mode)
   - Critical Feature: `history_mode="progressive"`
   - Location: `promptchain/utils/agentic_step_processor.py`

2. **AgentChain (Existing)**
   - Custom router function support
   - Tool integration capabilities
   - Location: `promptchain/utils/agent_chain.py`

3. **LiteLLM (Existing)**
   - Tool calling support
   - Multiple model provider support
   - Version: Latest stable

### New Components (Phase 1)

1. **Agentic Router Wrapper**
   - File: `promptchain/utils/agentic_router_wrapper.py`
   - Async wrapper function
   - Tool integration helpers

2. **Router Tools**
   - Agent capability discovery
   - Current date fetching
   - Knowledge boundary assessment

### New Components (Phase 2)

1. **AgentChain Router Modes**
   - Agentic router mode implementation
   - Configuration schema
   - Integration with existing router infrastructure

2. **Router Configuration Manager**
   - Config validation
   - Mode switching logic
   - Performance monitoring

---

## Timeline & Milestones

### Phase 1: Validation (Weeks 1-2)

**Week 1:**
- ✅ Day 1-2: Create agentic_router_wrapper.py
- ✅ Day 3-4: Implement required tools (capability discovery, date, knowledge boundary)
- ✅ Day 5: Integration testing with AgentChain

**Week 2:**
- ⏳ Day 1-2: Validation dataset creation and testing
- ⏳ Day 3-4: Performance optimization and tuning
- ⏳ Day 5: Documentation and examples

**Milestone 1 Exit Criteria:**
- Routing accuracy ≥ 95%
- Multi-hop reasoning demonstrated
- Progressive history verified
- Performance benchmarks met
- Documentation complete

### Phase 2: Native Integration (Weeks 3-6)

**Week 3-4:**
- ⏳ Router mode architecture design
- ⏳ AgentChain modifications for agentic mode
- ⏳ Configuration schema implementation
- ⏳ Backward compatibility testing

**Week 5:**
- ⏳ Migration tooling and documentation
- ⏳ Performance optimization
- ⏳ Integration testing
- ⏳ User acceptance testing

**Week 6:**
- ⏳ Final testing and bug fixes
- ⏳ Documentation finalization
- ⏳ Release preparation
- ⏳ Migration guide publication

**Milestone 2 Exit Criteria:**
- Native implementation complete
- Wrapper equivalence verified
- Migration path validated
- Performance targets met
- Release-ready documentation

---

## Success Validation

### Phase 1 Success Criteria

**Quantitative:**
- ✅ Routing accuracy: 95%+ on test dataset
- ✅ Multi-hop reasoning: avg 3+ steps for complex queries
- ✅ Context preservation: 100% across reasoning steps
- ✅ Knowledge boundary detection: 90%+ accuracy
- ✅ Latency: <5s per routing decision
- ✅ Token usage: <2000 tokens per routing decision

**Qualitative:**
- ✅ Code is maintainable and well-documented
- ✅ Integration is straightforward
- ✅ Error handling is robust
- ✅ Performance is acceptable

### Phase 2 Success Criteria

**Quantitative:**
- ✅ Zero regression in existing router modes
- ✅ Wrapper-to-native equivalence: 100%
- ✅ Migration success rate: >95% of test cases
- ✅ Performance parity or improvement vs wrapper

**Qualitative:**
- ✅ Migration path is clear and documented
- ✅ Configuration is intuitive
- ✅ User feedback is positive
- ✅ Maintenance burden is low

---

## Documentation Requirements

### Phase 1 Documentation

1. **Technical Documentation**
   - Agentic router wrapper API reference
   - Tool integration guide
   - Configuration options
   - Error handling patterns

2. **Usage Examples**
   - Basic integration example
   - Custom tool integration
   - Performance optimization tips
   - Troubleshooting guide

3. **Migration Guide**
   - Before/after configuration examples
   - Step-by-step migration instructions
   - Rollback procedures
   - FAQ section

### Phase 2 Documentation

1. **API Documentation**
   - AgentChain agentic router mode reference
   - Configuration schema documentation
   - Migration from wrapper to native
   - Backward compatibility notes

2. **User Guides**
   - Quick start guide
   - Advanced configuration guide
   - Performance tuning guide
   - Best practices

3. **Developer Documentation**
   - Architecture overview
   - Implementation details
   - Extension points
   - Contributing guide

---

## Appendix

### A. Technical Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              AgentChain Router Layer                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │   Router Mode Selection                          │  │
│  │   - simple    (regex patterns)                   │  │
│  │   - llm       (single LLM call)                  │  │
│  │   - agentic   (AgenticStepProcessor orchestrator)│  │
│  └──────────────────┬───────────────────────────────┘  │
└─────────────────────┼───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         AgenticStepProcessor Orchestrator               │
│                                                         │
│  Objective: Intelligent routing with multi-hop reasoning│
│                                                         │
│  ┌───────────────────────────────────────────────┐     │
│  │  Internal Reasoning Loop (max 5 steps)        │     │
│  │                                               │     │
│  │  Step 1: Analyze user query                  │     │
│  │  Step 2: Check agent capabilities            │     │
│  │  Step 3: Assess knowledge boundaries         │     │
│  │  Step 4: Consider temporal context           │     │
│  │  Step 5: Make final routing decision         │     │
│  └───────────────────────────────────────────────┘     │
│                                                         │
│  History Mode: PROGRESSIVE (context accumulation)       │
│  Model: openai/gpt-4o (tool calling support)          │
│  Tools: [capability_discovery, current_date,           │
│          knowledge_boundary_assessment]                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Routing Decision Output                    │
│                                                         │
│  {                                                      │
│    "chosen_agent": "research_agent",                   │
│    "refined_query": "Find latest GPT-5 papers",        │
│    "reasoning": "Query requires current info...",      │
│    "requires_research": true,                          │
│    "confidence_score": 0.95,                           │
│    "multi_hop_steps": [...],                           │
│    "knowledge_gaps": [...],                            │
│    "temporal_context": "2025-10-04"                    │
│  }                                                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Execute Chosen Agent with Query               │
└─────────────────────────────────────────────────────────┘
```

### B. Progressive History Accumulation Example

```python
# Reasoning Step 1
Messages: [system, user, assistant_step1]
Context: "Analyzing query complexity..."

# Reasoning Step 2 (PROGRESSIVE MODE)
Messages: [system, user, assistant_step1, tool_result1, assistant_step2]
Context: "Previous analysis + new agent capability check..."

# Reasoning Step 3 (PROGRESSIVE MODE)
Messages: [system, user, assistant_step1, tool_result1,
          assistant_step2, tool_result2, assistant_step3]
Context: "All previous reasoning + knowledge boundary assessment..."

# Final Step (PROGRESSIVE MODE)
Messages: [system, user, assistant_step1, tool_result1,
          assistant_step2, tool_result2, assistant_step3,
          tool_result3, final_assistant]
Context: "Complete reasoning chain + final routing decision"
```

**Key Insight:** Progressive mode accumulates ALL reasoning steps, enabling true multi-hop reasoning where each step builds on previous context.

### C. Knowledge Boundary Detection Logic

```python
def assess_knowledge_boundary(query: str, available_info: dict) -> dict:
    """
    Determine if query requires external research.

    Returns:
        {
            "requires_research": bool,
            "knowledge_gaps": List[str],
            "available_knowledge": List[str],
            "confidence": float
        }
    """

    # Check for temporal indicators
    temporal_keywords = ["latest", "recent", "current", "today", "2025"]
    needs_current_info = any(kw in query.lower() for kw in temporal_keywords)

    # Check against available agent knowledge
    knowledge_gaps = []
    available_knowledge = []

    # Analyze query requirements vs agent capabilities
    # ... implementation details ...

    return {
        "requires_research": needs_current_info or bool(knowledge_gaps),
        "knowledge_gaps": knowledge_gaps,
        "available_knowledge": available_knowledge,
        "confidence": calculate_confidence(query, available_info)
    }
```

### D. Validation Test Cases

**Test Case 1: Simple Query (No Multi-Hop)**
```
Query: "What is 2+2?"
Expected: Single-step routing to math_agent
Expected Steps: 1
Expected Research: False
```

**Test Case 2: Complex Query (Multi-Hop Required)**
```
Query: "Analyze the latest GPT-5 papers and compare performance metrics"
Expected: Multi-step reasoning
  - Step 1: Identify need for current papers
  - Step 2: Check research agent capabilities
  - Step 3: Determine need for analysis agent
  - Step 4: Plan multi-agent workflow
  - Step 5: Final routing decision
Expected Steps: 5
Expected Research: True
```

**Test Case 3: Knowledge Boundary Detection**
```
Query: "What happened in the 2025 AI Safety Summit?"
Expected: Detect knowledge gap (event after training cutoff)
Expected Research: True
Expected Agent: research_agent
Expected Reasoning: "Event is recent, requires external research"
```

**Test Case 4: Temporal Context**
```
Query: "What's the latest version of Python?"
Expected: Detect temporal context need
Expected: Check current_date tool
Expected Research: Likely true (depends on training date)
Expected Agent: research_agent or knowledge_agent
```

### E. References and Related Documents

1. **AgenticStepProcessor Implementation**
   - File: `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agentic_step_processor.py`
   - Key Feature: Progressive history mode (lines 300-343)
   - Documentation: HISTORY_MODES_IMPLEMENTATION.md

2. **AgentChain Router System**
   - File: `/home/gyasis/Documents/code/PromptChain/promptchain/utils/agent_chain.py`
   - Current Implementation: Single-step LLM router
   - Enhancement Target: Multi-hop agentic orchestrator

3. **Memory Bank Documentation**
   - Guide: memory-bank/systemPatterns.md
   - Architecture: Describes history accumulation patterns
   - Integration: Shows chat and router integration patterns

4. **Related PRDs**
   - MCP Tool Hijacker PRD: `/home/gyasis/Documents/code/PromptChain/prd/mcp_tool_hijacker_prd.md`
   - Terminal Execution Tool PRD: `/home/gyasis/Documents/code/PromptChain/prd/terminal_execution_tool_prd.md`

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-04 | PromptChain Core Team | Initial PRD creation |

---

**Next Steps:**
1. ✅ Phase 1 Implementation: Create agentic_router_wrapper.py
2. ✅ Implement required tools (capability discovery, current_date, knowledge_boundary)
3. ⏳ Validation testing with 100-query dataset
4. ⏳ Performance benchmarking and optimization
5. ⏳ Documentation and migration guide
6. ⏳ Phase 2 Planning: Native integration design

**Questions & Discussion:**
- What is the optimal `max_internal_steps` for most use cases? (Currently: 5)
- Should we support custom reasoning templates for orchestrator objective?
- How to balance latency vs reasoning depth in production?
- What fallback strategy if progressive mode causes token overflow?
