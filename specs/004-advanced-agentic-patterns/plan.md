# Implementation Plan: Advanced Agentic Patterns

**Branch**: `004-advanced-agentic-patterns` | **Date**: 2025-11-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-advanced-agentic-patterns/spec.md`

## Summary

Complete the remaining 6 agentic AI patterns (Branching Thoughts, Parallel Query Expansion, Sharded Retrieval, Multi-Hop Retrieval, Hybrid Search Fusion, Speculative Execution) to achieve 100% coverage of the 14 pillars of production-grade agentic AI.

**Architecture Decision**: Integrate existing `hybridrag` project (LightRAG-based) instead of building patterns from scratch. See [research.md](./research.md) for detailed rationale.

**Installation**: `pip install git+https://github.com/gyasis/hybridrag.git`

## Technical Context

**Language/Version**: Python 3.8+ (compatible with existing PromptChain codebase)
**Primary Dependencies**: LiteLLM (existing), asyncio (stdlib), existing 003 infrastructure (MessageBus, Blackboard, CapabilityRegistry), hybridrag (LightRAG integration - `pip install git+https://github.com/gyasis/hybridrag.git`)
**Storage**: SQLite (existing session persistence), In-memory cache (speculative execution)
**Testing**: pytest (existing framework)
**Target Platform**: Linux/macOS/Windows (Python cross-platform)
**Project Type**: Single project (library extension)
**Performance Goals**: <2x latency for parallel operations, 60%+ prediction accuracy for speculative execution, 30%+ recall improvement for query expansion
**Constraints**: <200ms overhead per pattern, backward compatible with existing APIs
**Scale/Scope**: Integration with existing AgentChain/PromptChain ecosystem, 6 new pattern modules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Library-First Architecture | ✅ PASS | Each pattern is self-contained module in `promptchain/patterns/` |
| II. Observable Systems | ✅ PASS | All patterns emit events via existing observability infrastructure |
| III. Test-First Development | ✅ PASS | Each pattern has dedicated test suite, TDD workflow |
| IV. Integration Testing | ✅ PASS | Integration with 003 infrastructure (MessageBus, Blackboard) tested |
| V. Token Economy & Performance | ✅ PASS | Parallel execution reduces total latency, speculative caching saves tokens |
| VI. Async-First Design | ✅ PASS | All patterns implement async methods with sync wrappers |
| VII. Simplicity & Maintainability | ✅ PASS | Each pattern is independent, minimal interdependencies |

**Gate Status**: ✅ PASSED - All principles satisfied

## Project Structure

### Documentation (this feature)

```text
specs/004-advanced-agentic-patterns/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── branching_thoughts.yaml
│   ├── query_expansion.yaml
│   ├── sharded_retrieval.yaml
│   ├── multi_hop_retrieval.yaml
│   ├── hybrid_search.yaml
│   └── speculative_execution.yaml
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
promptchain/
├── patterns/                          # NEW: Pattern base classes
│   ├── __init__.py
│   └── base.py                        # Base pattern class, configs, results
├── integrations/                      # NEW: External integrations
│   ├── __init__.py
│   └── lightrag/                      # LightRAG via hybridrag
│       ├── __init__.py                # Imports, LIGHTRAG_AVAILABLE flag
│       ├── core.py                    # LightRAGIntegration wrapper
│       ├── branching.py               # US1: Branching Thoughts via LightRAG
│       ├── query_expansion.py         # US2: Query Expansion via LightRAG
│       ├── sharded.py                 # US3: Sharded Retrieval via LightRAG
│       ├── multi_hop.py               # US4: Multi-Hop via agentic_search
│       ├── hybrid_search.py           # US5: Hybrid Search via LightRAG modes
│       ├── speculative.py             # US6: Speculative Execution
│       ├── messaging.py               # MessageBus integration mixin
│       ├── state.py                   # Blackboard integration mixin
│       └── events.py                  # Pattern event definitions
└── cli/
    └── commands/
        └── patterns.py                 # NEW: CLI pattern commands

tests/
├── unit/
│   └── patterns/                       # NEW: Pattern unit tests
│       ├── test_branching_thoughts.py
│       ├── test_query_expansion.py
│       ├── test_sharded_retrieval.py
│       ├── test_multi_hop_retrieval.py
│       ├── test_hybrid_search.py
│       └── test_speculative_execution.py
├── integration/
│   └── patterns/                       # NEW: Pattern integration tests
│       ├── test_pattern_message_bus.py
│       ├── test_pattern_blackboard.py
│       └── test_pattern_orchestration.py
└── e2e/
    └── patterns/                       # NEW: End-to-end workflow tests
        └── test_multi_pattern_workflow.py

# External dependency (installed via pip)
hybridrag/  # pip install git+https://github.com/gyasis/hybridrag.git
├── src/
│   ├── lightrag_core.py              # HybridLightRAGCore class
│   ├── search_interface.py           # SearchInterface with agentic_search
│   └── ...
└── hybridrag_mcp_server.py           # FastMCP 2.0 server
```

**Structure Decision**: Integration-first architecture. The `promptchain/integrations/lightrag/` module wraps the existing `hybridrag` project components as PromptChain patterns. This reduces implementation effort by ~60% while leveraging battle-tested LightRAG integration.

## Complexity Tracking

> No violations to justify - all gates passed.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Phase Dependencies

### Phase 0: Research ✅ COMPLETE
- ✅ Resolved best practices for each pattern implementation
- ✅ Evaluated fusion algorithms (RRF via LightRAG hybrid mode)
- ✅ Research prediction models for speculative execution
- ✅ **KEY DECISION**: Integrate existing `hybridrag` project instead of building from scratch
  - HybridLightRAGCore already implements local/global/hybrid query modes
  - SearchInterface already has agentic_search (multi-hop) and multi_query_search (expansion)
  - See [research.md](./research.md) for full analysis

### Phase 1: Design ✅ COMPLETE
- ✅ Define data models for all 6 patterns (see [data-model.md](./data-model.md))
- ✅ Create API contracts for pattern interfaces (see [contracts/](./contracts/))
- ✅ Design integration with 003 infrastructure (MessageBus, Blackboard)
- ✅ Create quickstart examples (see [quickstart.md](./quickstart.md))

### Phase 2: Tasks ✅ COMPLETE
- ✅ Implementation tasks generated with Wave-based parallelization
- ✅ File locking strategy defined to prevent conflicts
- See [tasks.md](./tasks.md) for full implementation plan

### Phase 3: Implementation (via /speckit.implement)
- Execute tasks.md following Wave structure
- Parallel agent execution where safe
- Checkpoint sync after each Wave
