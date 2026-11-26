# 14 Agentic Patterns Gap Analysis
## Mapping Production-Grade Agentic Patterns to PromptChain

**Source**: Analysis of agentic patterns diagram and "Building the 14 Key Pillars of Agentic AI"  
**Purpose**: Comprehensive gap analysis comparing 14 production-grade agentic patterns to PromptChain's current capabilities, identifying what exists, what's missing, and how agent communication can address gaps.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Pattern-by-Pattern Analysis](#pattern-by-pattern-analysis)
3. [Current State Assessment](#current-state-assessment)
4. [Gap Analysis Matrix](#gap-analysis-matrix)
5. [How Agent Communication Addresses Gaps](#how-agent-communication-addresses-gaps)
6. [Additional Gaps Beyond Communication](#additional-gaps-beyond-communication)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Mental Models for Pattern Selection](#mental-models-for-pattern-selection)

---

## Executive Summary

### Key Findings

**What PromptChain Already Has:**
- ✅ **Agent Assembly Line** (Pipeline mode) - Fully supported
- ✅ **Competitive Agent Ensembles** (Broadcast mode) - Partially supported
- ✅ **Hierarchical Agent Teams** (Router mode with dynamic decomposition) - Partially supported
- ✅ **Parallel Tool Processing** - Supported via tool registration
- ⚠️ **Branching Thoughts** - Limited support (no multi-hypothesis generation)
- ⚠️ **Parallel Evaluation** - Not directly supported

**Critical Gaps:**
- ❌ **Blackboard Collaboration** - No shared workspace for agents
- ❌ **Parallel Query Expansion** - No query diversification
- ❌ **Sharded & Scattered Retrieval** - No distributed data source support
- ❌ **Parallel Multi-Hop Retrieval** - No parallel sub-question processing
- ❌ **Redundant Execution** - No fault tolerance via redundancy
- ❌ **Parallel Context Pre-processing** - No parallel LLM filtering
- ❌ **Hybrid Search Fusion** - No multi-modal search
- ❌ **Speculative Execution** - No predictive tool usage

**Agent Communication Can Address:**
- ✅ Blackboard Collaboration (via shared communication bus)
- ✅ Hierarchical Agent Teams (via capability discovery and task delegation)
- ✅ Competitive Agent Ensembles (via result comparison and selection)
- ✅ Parallel Evaluation (via multi-agent evaluation coordination)

**Additional Infrastructure Needed:**
- Query expansion and diversification
- Distributed retrieval systems
- Speculative execution engine
- Redundancy and fault tolerance
- Context pre-processing pipeline

---

## Pattern-by-Pattern Analysis

### 1. Parallel Tool Processing
**Description**: Tools process queries concurrently

**Current State**: ✅ **Supported**
- PromptChain supports tool registration and parallel execution
- Tools can be called concurrently within AgenticStepProcessor
- MCP tools can be loaded on-demand

**Gap**: None - Fully functional

**Implementation**: Already in `promptchain/utils/promptchaining.py`

---

### 2. Branching Thoughts
**Description**: Generates multiple hypotheses, evaluates with Judge agent, selects best path

**Current State**: ⚠️ **Partially Supported**
- AgenticStepProcessor supports multi-step reasoning
- No built-in multi-hypothesis generation
- No dedicated "Judge" agent for path selection

**Gap**: 
- Missing: Multi-hypothesis generation framework
- Missing: Judge agent for path evaluation
- Missing: Automatic best path selection

**How Agent Communication Helps**:
- Agents can generate hypotheses and share via communication bus
- Dedicated judge agent can evaluate hypotheses from multiple agents
- Result comparison and selection via competitive ensemble pattern

**Implementation Needed**:
```python
# New: HypothesisGenerator agent
# New: Judge agent for evaluation
# Enhanced: AgenticStepProcessor with branching support
```

---

### 3. Parallel Evaluation
**Description**: Simultaneous multi-expert evaluation with Editor Agent synthesis

**Current State**: ❌ **Not Supported**
- No parallel evaluation framework
- No multi-expert coordination
- No synthesis agent

**Gap**: Complete - No parallel evaluation infrastructure

**How Agent Communication Helps**:
- Multiple judge agents can evaluate in parallel
- Editor agent can synthesize evaluations via communication bus
- Agents can share evaluation criteria and results

**Implementation Needed**:
```python
# New: ParallelEvaluationCoordinator
# New: EditorAgent for synthesis
# New: Evaluation result aggregation
```

---

### 4. Parallel Query Expansion
**Description**: Diverse searches via multiple query formulations

**Current State**: ❌ **Not Supported**
- No query expansion mechanism
- No query diversification
- Single query per retrieval

**Gap**: Complete - No query expansion infrastructure

**How Agent Communication Helps**:
- Query expansion agent can generate diverse queries
- Agents can share query strategies
- Results can be aggregated via communication bus

**Implementation Needed**:
```python
# New: QueryExpansionAgent
# New: Query diversification strategies
# New: Result fusion mechanism
```

---

### 5. Sharded & Scattered Retrieval
**Description**: Diverse searches across distributed data sources

**Current State**: ❌ **Not Supported**
- No sharded retrieval support
- No distributed data source management
- Single retrieval endpoint

**Gap**: Complete - No distributed retrieval infrastructure

**How Agent Communication Helps**:
- Agents can coordinate retrieval across shards
- Results can be aggregated via communication bus
- Agents can discover available data sources

**Implementation Needed**:
```python
# New: ShardedRetrievalCoordinator
# New: Data source registry
# New: Result aggregation across shards
```

---

### 6. Competitive Agent Ensembles
**Description**: Diverse agents reduce risk, best output selected

**Current State**: ⚠️ **Partially Supported**
- Broadcast mode executes agents in parallel
- Synthesizer combines results
- No explicit "best output" selection mechanism
- No risk reduction strategy

**Gap**: 
- Missing: Explicit best output selection
- Missing: Risk assessment and reduction
- Missing: Agent diversity metrics

**How Agent Communication Helps**:
- Agents can share results for comparison
- Judge agent can select best output
- Agents can coordinate to ensure diversity

**Implementation Needed**:
```python
# Enhanced: Broadcast mode with best output selection
# New: RiskAssessmentAgent
# New: OutputComparisonAgent
```

---

### 7. Hierarchical Agent Teams
**Description**: Task split among specialists

**Current State**: ⚠️ **Partially Supported**
- Router mode with dynamic decomposition exists
- Task delegation is centralized (via router)
- No agent-initiated task splitting
- Limited specialist coordination

**Gap**: 
- Missing: Agent-initiated task decomposition
- Missing: Direct specialist-to-specialist communication
- Missing: Dynamic task assignment

**How Agent Communication Helps**:
- ✅ **Fully Addressable**: Agents can discover capabilities
- ✅ **Fully Addressable**: Agents can delegate tasks directly
- ✅ **Fully Addressable**: Agents can coordinate specialist work

**Implementation Needed**:
```python
# Enhanced: AgentCapabilityRegistry (from agent_interaction_design.md)
# Enhanced: AgentCommunicationBus (from agent_interaction_design.md)
# New: TaskDelegationProtocol
```

---

### 8. Blackboard Collaboration
**Description**: Coordinates specialist agents via shared workspace

**Current State**: ❌ **Not Supported**
- No shared workspace/blackboard
- No agent-to-agent indirect communication
- No collaborative knowledge building

**Gap**: Complete - No blackboard infrastructure

**How Agent Communication Helps**:
- ✅ **Fully Addressable**: Communication bus can act as blackboard
- ✅ **Fully Addressable**: Agents can read/write to shared space
- ✅ **Fully Addressable**: Agents can coordinate via blackboard

**Implementation Needed**:
```python
# New: BlackboardManager (extends AgentCommunicationBus)
# New: Shared workspace data structures
# New: Read/write protocols for agents
```

---

### 9. Parallel Multi-Hop Retrieval
**Description**: Parallel sub-questions, unified answer

**Current State**: ❌ **Not Supported**
- No parallel sub-question generation
- No multi-hop retrieval coordination
- Single retrieval path

**Gap**: Complete - No multi-hop retrieval infrastructure

**How Agent Communication Helps**:
- Query decomposition agent can generate sub-questions
- Agents can execute retrievals in parallel
- Synthesis agent can unify results

**Implementation Needed**:
```python
# New: MultiHopRetrievalCoordinator
# New: SubQuestionGenerator
# New: ResultUnificationAgent
```

---

### 10. Redundant Execution
**Description**: Monitored by specialists, first success wins

**Current State**: ❌ **Not Supported**
- No redundant execution framework
- No monitoring and cancellation
- No fault tolerance via redundancy

**Gap**: Complete - No redundancy infrastructure

**How Agent Communication Helps**:
- Monitoring agent can track execution
- Agents can signal success/failure
- Cancellation can be broadcast via communication bus

**Implementation Needed**:
```python
# New: RedundantExecutionManager
# New: ExecutionMonitor
# New: Cancellation protocol
```

---

### 11. Agent Assembly Line
**Description**: Process split into stations

**Current State**: ✅ **Fully Supported**
- Pipeline mode executes agents sequentially
- Output passed as input to next agent
- Well-established pattern

**Gap**: None - Fully functional

**Implementation**: Already in `promptchain/utils/agent_chain.py` (pipeline mode)

---

### 12. Parallel Context Pre-processing
**Description**: Parallel LLMs filter documents

**Current State**: ❌ **Not Supported**
- No parallel context filtering
- No document chunk processing
- Single LLM processing

**Gap**: Complete - No parallel pre-processing infrastructure

**How Agent Communication Helps**:
- Multiple agents can process chunks in parallel
- Results can be aggregated via communication bus
- Filtering criteria can be shared

**Implementation Needed**:
```python
# New: ParallelContextProcessor
# New: DocumentChunker
# New: FilteringAgent
```

---

### 13. Hybrid Search Fusion
**Description**: Fused search results from multiple techniques

**Current State**: ❌ **Not Supported**
- No hybrid search support
- No multi-modal search fusion
- Single search technique

**Gap**: Complete - No hybrid search infrastructure

**How Agent Communication Helps**:
- Multiple search agents can use different techniques
- Results can be fused via communication bus
- Agents can coordinate search strategies

**Implementation Needed**:
```python
# New: HybridSearchCoordinator
# New: EmbeddingSearchAgent
# New: KeywordSearchAgent
# New: ResultFusionAgent
```

---

### 14. Speculative Execution
**Description**: Predicting tool usage quickly

**Current State**: ❌ **Not Supported**
- No speculative execution
- No tool usage prediction
- No pre-computation

**Gap**: Complete - No speculative execution infrastructure

**How Agent Communication Helps**:
- Prediction agent can forecast tool needs
- Agents can share prediction results
- Speculative results can be validated

**Implementation Needed**:
```python
# New: SpeculativeExecutionEngine
# New: ToolUsagePredictor
# New: Prediction validation
```

---

## Current State Assessment

### What PromptChain Has (✅)

| Pattern | Support Level | Implementation |
|---------|--------------|----------------|
| **Agent Assembly Line** | ✅ Full | `execution_mode="pipeline"` |
| **Parallel Tool Processing** | ✅ Full | Tool registration + MCP |
| **Competitive Agent Ensembles** | ⚠️ Partial | `execution_mode="broadcast"` (needs best output selection) |
| **Hierarchical Agent Teams** | ⚠️ Partial | `execution_mode="router"` with dynamic decomposition |

### What's Missing (❌)

| Pattern | Gap Severity | Can Communication Fix? |
|---------|--------------|------------------------|
| **Blackboard Collaboration** | Critical | ✅ Yes - Communication bus as blackboard |
| **Branching Thoughts** | High | ⚠️ Partial - Needs hypothesis framework |
| **Parallel Evaluation** | High | ✅ Yes - Multi-agent evaluation |
| **Parallel Query Expansion** | Medium | ⚠️ Partial - Needs query agent |
| **Sharded & Scattered Retrieval** | Medium | ⚠️ Partial - Needs retrieval coordination |
| **Parallel Multi-Hop Retrieval** | Medium | ⚠️ Partial - Needs multi-hop coordinator |
| **Redundant Execution** | Medium | ✅ Yes - Monitoring + cancellation |
| **Parallel Context Pre-processing** | Low | ✅ Yes - Parallel processing agents |
| **Hybrid Search Fusion** | Low | ✅ Yes - Multiple search agents |
| **Speculative Execution** | Low | ⚠️ Partial - Needs prediction engine |

---

## Gap Analysis Matrix

### Pattern Support Matrix

| Pattern | Current | With Communication | Additional Infrastructure |
|---------|---------|-------------------|--------------------------|
| 1. Parallel Tool Processing | ✅ | ✅ | None |
| 2. Branching Thoughts | ⚠️ | ⚠️→✅ | Hypothesis framework |
| 3. Parallel Evaluation | ❌ | ✅ | Evaluation coordinator |
| 4. Parallel Query Expansion | ❌ | ⚠️ | Query expansion engine |
| 5. Sharded Retrieval | ❌ | ⚠️ | Shard coordinator |
| 6. Competitive Ensembles | ⚠️ | ✅ | Best output selector |
| 7. Hierarchical Teams | ⚠️ | ✅ | Task delegation protocol |
| 8. Blackboard Collaboration | ❌ | ✅ | Blackboard manager |
| 9. Multi-Hop Retrieval | ❌ | ⚠️ | Multi-hop coordinator |
| 10. Redundant Execution | ❌ | ✅ | Execution monitor |
| 11. Agent Assembly Line | ✅ | ✅ | None |
| 12. Context Pre-processing | ❌ | ✅ | Parallel processor |
| 13. Hybrid Search Fusion | ❌ | ✅ | Search fusion agent |
| 14. Speculative Execution | ❌ | ⚠️ | Prediction engine |

**Legend**:
- ✅ = Fully supported
- ⚠️ = Partially supported
- ❌ = Not supported

---

## How Agent Communication Addresses Gaps

### Patterns Fully Addressable via Communication

#### 1. Blackboard Collaboration ✅
**Solution**: Extend `AgentCommunicationBus` to act as shared blackboard

```python
class BlackboardManager(AgentCommunicationBus):
    """Shared workspace for agent collaboration"""
    
    def write_to_blackboard(self, agent_id: str, data: Dict):
        """Agent writes to shared workspace"""
        self.blackboard[agent_id] = data
        self.broadcast("blackboard_updated", agent_id, data)
    
    def read_from_blackboard(self, agent_id: str) -> Dict:
        """Agent reads from shared workspace"""
        return self.blackboard.get(agent_id, {})
    
    def get_full_blackboard(self) -> Dict:
        """Get complete shared state"""
        return self.blackboard.copy()
```

**Impact**: Enables flexible agent collaboration without central orchestrator

---

#### 2. Hierarchical Agent Teams ✅
**Solution**: `AgentCapabilityRegistry` + `AgentCommunicationBus`

```python
# Agent discovers capabilities
capabilities = capability_registry.discover_agent("specialist_agent")

# Agent delegates task
task = {
    "type": "data_analysis",
    "input": data,
    "delegated_by": "coordinator_agent"
}
communication_bus.send_task("specialist_agent", task)

# Specialist responds
result = communication_bus.get_task_result(task_id)
```

**Impact**: Enables agent-initiated task decomposition and delegation

---

#### 3. Competitive Agent Ensembles ✅
**Solution**: Parallel execution + result comparison via communication

```python
# Multiple agents execute in parallel
results = await agent_chain.broadcast_execute(user_input)

# Judge agent compares results
best_result = judge_agent.select_best(results)

# Communication enables result sharing
communication_bus.broadcast_results(results)
```

**Impact**: Risk reduction through diverse agent perspectives

---

#### 4. Parallel Evaluation ✅
**Solution**: Multiple judge agents + editor synthesis

```python
# Multiple judges evaluate in parallel
judge_results = await parallel_evaluate(plan, judge_agents)

# Editor agent synthesizes
final_evaluation = editor_agent.synthesize(judge_results)

# Communication coordinates evaluation
communication_bus.coordinate_evaluation(plan, judge_results)
```

**Impact**: Multi-expert evaluation reduces bias

---

#### 5. Redundant Execution ✅
**Solution**: Monitoring + cancellation via communication

```python
# Launch redundant agents
redundant_tasks = [agent.execute(task) for _ in range(3)]

# Monitor for first success
first_success = await wait_for_first_success(redundant_tasks)

# Cancel others via communication
communication_bus.cancel_tasks([t for t in redundant_tasks if t != first_success])
```

**Impact**: Fault tolerance and reliability

---

### Patterns Partially Addressable

#### 2. Branching Thoughts ⚠️→✅
**Needs**: Hypothesis generation framework + Judge agent

```python
# Hypothesis generation (NEW)
hypotheses = hypothesis_generator.generate_multiple(user_input)

# Judge agent evaluates (via communication)
evaluations = await judge_agent.evaluate_all(hypotheses)

# Best path selected
best_path = judge_agent.select_best(evaluations)
```

**Gap Remaining**: Hypothesis generation framework (not just communication)

---

#### 4. Parallel Query Expansion ⚠️
**Needs**: Query expansion engine (communication helps coordinate)

```python
# Query expansion (NEW engine needed)
expanded_queries = query_expander.diversify(user_query)

# Agents execute in parallel (communication coordinates)
results = await parallel_retrieve(expanded_queries)

# Results fused (communication aggregates)
fused_result = communication_bus.fuse_results(results)
```

**Gap Remaining**: Query expansion algorithms (communication enables coordination)

---

## Additional Gaps Beyond Communication

### Infrastructure Gaps

1. **Query Expansion Engine**
   - Query diversification algorithms
   - Semantic query variation
   - Query quality assessment

2. **Distributed Retrieval System**
   - Shard management
   - Load balancing across shards
   - Result aggregation

3. **Speculative Execution Engine**
   - Tool usage prediction
   - Pre-computation framework
   - Prediction validation

4. **Multi-Hop Retrieval Coordinator**
   - Sub-question generation
   - Dependency tracking
   - Result unification

5. **Context Pre-processing Pipeline**
   - Document chunking
   - Parallel LLM filtering
   - Quality filtering

6. **Hybrid Search Fusion**
   - Embedding search integration
   - Keyword search integration
   - Result fusion algorithms

---

## Implementation Roadmap

### Phase 1: Core Communication Infrastructure (Addresses 4 patterns)
**Timeline**: 2-3 months

1. **AgentCommunicationBus** (from `agent_interaction_design.md`)
   - Enables: Blackboard Collaboration, Hierarchical Teams
   - Priority: Critical

2. **AgentCapabilityRegistry** (from `agent_interaction_design.md`)
   - Enables: Hierarchical Teams, Task Delegation
   - Priority: Critical

3. **Task Delegation Protocol**
   - Enables: Hierarchical Teams
   - Priority: High

**Deliverables**:
- ✅ Blackboard Collaboration pattern
- ✅ Hierarchical Agent Teams (full support)
- ✅ Competitive Agent Ensembles (enhanced)

---

### Phase 2: Evaluation and Redundancy (Addresses 2 patterns)
**Timeline**: 1-2 months

1. **Parallel Evaluation Framework**
   - Multiple judge agents
   - Editor agent for synthesis
   - Evaluation coordination

2. **Redundant Execution Manager**
   - Execution monitoring
   - Success detection
   - Cancellation protocol

**Deliverables**:
- ✅ Parallel Evaluation pattern
- ✅ Redundant Execution pattern

---

### Phase 3: Query and Retrieval Enhancement (Addresses 3 patterns)
**Timeline**: 2-3 months

1. **Query Expansion Engine**
   - Query diversification
   - Semantic variation
   - Quality assessment

2. **Multi-Hop Retrieval Coordinator**
   - Sub-question generation
   - Parallel execution
   - Result unification

3. **Sharded Retrieval System**
   - Shard management
   - Distributed coordination
   - Result aggregation

**Deliverables**:
- ✅ Parallel Query Expansion pattern
- ✅ Parallel Multi-Hop Retrieval pattern
- ✅ Sharded & Scattered Retrieval pattern

---

### Phase 4: Advanced Patterns (Addresses 3 patterns)
**Timeline**: 2-3 months

1. **Branching Thoughts Framework**
   - Hypothesis generation
   - Path evaluation
   - Best path selection

2. **Context Pre-processing Pipeline**
   - Document chunking
   - Parallel filtering
   - Quality assessment

3. **Hybrid Search Fusion**
   - Multi-modal search
   - Result fusion
   - Quality ranking

**Deliverables**:
- ✅ Branching Thoughts pattern
- ✅ Parallel Context Pre-processing pattern
- ✅ Hybrid Search Fusion pattern

---

### Phase 5: Speculative Execution (Addresses 1 pattern)
**Timeline**: 1-2 months

1. **Speculative Execution Engine**
   - Tool usage prediction
   - Pre-computation
   - Validation and commitment

**Deliverables**:
- ✅ Speculative Execution pattern

---

## Mental Models for Pattern Selection

### Decision Framework

Agents can use mental models to select appropriate patterns:

#### 1. **Decomposition** → Hierarchical Agent Teams
- When: Complex task needs breaking down
- Mental Model: "decomposition"
- Pattern: Hierarchical Agent Teams

#### 2. **Trade-off Matrix** → Competitive Agent Ensembles
- When: Need to reduce risk, multiple perspectives valuable
- Mental Model: "trade-off-matrix"
- Pattern: Competitive Agent Ensembles

#### 3. **Pre-mortem** → Redundant Execution
- When: High-stakes task, fault tolerance critical
- Mental Model: "pre-mortem"
- Pattern: Redundant Execution

#### 4. **Abstraction Laddering** → Parallel Query Expansion
- When: Need diverse search perspectives
- Mental Model: "abstraction-laddering"
- Pattern: Parallel Query Expansion

#### 5. **Five Whys** → Multi-Hop Retrieval
- When: Need to drill down through multiple information layers
- Mental Model: "five-whys"
- Pattern: Parallel Multi-Hop Retrieval

#### 6. **Steelmanning** → Parallel Evaluation
- When: Need balanced, unbiased evaluation
- Mental Model: "steelmanning"
- Pattern: Parallel Evaluation

#### 7. **Constraint Relaxation** → Branching Thoughts
- When: Need to explore multiple solution paths
- Mental Model: "constraint-relaxation"
- Pattern: Branching Thoughts

---

### Pattern Selection Algorithm

```python
def select_pattern(task_description: str, mental_model: str) -> str:
    """Select agentic pattern based on task and mental model"""
    
    model_to_pattern = {
        "decomposition": "hierarchical_teams",
        "trade-off-matrix": "competitive_ensembles",
        "pre-mortem": "redundant_execution",
        "abstraction-laddering": "parallel_query_expansion",
        "five-whys": "multi_hop_retrieval",
        "steelmanning": "parallel_evaluation",
        "constraint-relaxation": "branching_thoughts",
    }
    
    # Select pattern based on mental model
    pattern = model_to_pattern.get(mental_model)
    
    # Fallback: Analyze task characteristics
    if not pattern:
        if "complex" in task_description.lower():
            pattern = "hierarchical_teams"
        elif "risk" in task_description.lower():
            pattern = "competitive_ensembles"
        else:
            pattern = "agent_assembly_line"  # Default
    
    return pattern
```

---

## Summary

### Current Coverage
- **Fully Supported**: 2 patterns (14%)
- **Partially Supported**: 2 patterns (14%)
- **Not Supported**: 10 patterns (72%)

### With Agent Communication
- **Fully Supported**: 8 patterns (57%)
- **Partially Supported**: 4 patterns (29%)
- **Not Supported**: 2 patterns (14%)

### With Full Implementation
- **Fully Supported**: 14 patterns (100%)

### Key Insights

1. **Agent Communication is Critical**: Enables 6 additional patterns (43% improvement)

2. **Infrastructure Still Needed**: Query expansion, speculative execution, etc. require dedicated engines

3. **Mental Models Guide Selection**: Structured reasoning frameworks help agents choose appropriate patterns

4. **Phased Approach Works**: Can achieve 57% coverage with communication infrastructure alone

5. **Production Readiness**: Need all 14 patterns for production-grade agentic systems

---

## Next Steps

1. **Immediate**: Implement Phase 1 (Agent Communication Infrastructure)
2. **Short-term**: Implement Phase 2 (Evaluation and Redundancy)
3. **Medium-term**: Implement Phases 3-4 (Query/Retrieval and Advanced Patterns)
4. **Long-term**: Implement Phase 5 (Speculative Execution)

**Priority**: Focus on patterns that agent communication can fully address first, then build additional infrastructure for remaining patterns.

---

**Status**: Comprehensive gap analysis complete - Ready for implementation planning

**Related Documents**:
- [agent_interaction_design.md](./agent_interaction_design.md) - Communication infrastructure design
- [thoughtbox_mental_models_integration.md](./thoughtbox_mental_models_integration.md) - Mental models for pattern selection
- [gap_analysis_and_solution_mapping.md](./gap_analysis_and_solution_mapping.md) - General gap analysis

