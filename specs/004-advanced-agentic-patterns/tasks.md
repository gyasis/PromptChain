# Tasks: Advanced Agentic Patterns

**Feature**: 004-advanced-agentic-patterns
**Date**: 2025-11-29
**Status**: Ready for Implementation

## Architecture Decision

**Integration over Recreation**: Leverage existing `/home/gyasis/Documents/code/hybridrag` project (LightRAG-based) instead of building patterns from scratch.

**Installation**: `pip install git+https://github.com/gyasis/hybridrag.git`

---

## Dependency Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY GRAPH                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Wave 1: Foundation (Sequential - Core Setup)                   │
│  ├── T001: Create integrations module structure                 │
│  └── T002: Create base pattern abstractions                     │
│                                                                  │
│  Wave 2: Pattern Wrappers (Parallel - No File Conflicts)        │
│  ├── T003: Branching Thoughts adapter ──────────┐               │
│  ├── T004: Query Expansion adapter ─────────────┤ Independent   │
│  ├── T005: Sharded Retrieval adapter ───────────┤ (parallel)    │
│  ├── T006: Multi-Hop Retrieval adapter ─────────┤               │
│  ├── T007: Hybrid Search Fusion adapter ────────┤               │
│  └── T008: Speculative Execution adapter ───────┘               │
│                                                                  │
│  Wave 3: Integration Layer (Sequential - Shared Files)          │
│  ├── T009: MessageBus integration                               │
│  ├── T010: Blackboard integration                               │
│  └── T011: Event emission system                                │
│                                                                  │
│  Wave 4: Testing (Parallel - Separate Test Files)               │
│  ├── T012: Unit tests for pattern adapters ─────┐               │
│  ├── T013: Integration tests with 003 infra ────┤ Independent   │
│  └── T014: End-to-end workflow tests ───────────┘               │
│                                                                  │
│  Wave 5: CLI & Documentation (Parallel)                         │
│  ├── T015: CLI pattern commands                                 │
│  └── T016: Documentation and examples                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation

### Wave 1 (Sequential - Core Setup)

| Task | Agent | Files Owned (Exclusive) | Dependencies | Est. Complexity |
|------|-------|-------------------------|--------------|-----------------|
| T001 | python-pro | `promptchain/integrations/__init__.py`, `promptchain/integrations/lightrag/__init__.py` | None | Low |
| T002 | python-pro | `promptchain/patterns/base.py` | T001 | Low |

---

- [X] T001 [S] [W1] [INFRA] [python-pro] [integrations/__init__.py, integrations/lightrag/__init__.py]
  **Create LightRAG Integration Module Structure**

  Create the integration layer that wraps hybridrag components:

  ```python
  # promptchain/integrations/__init__.py
  """Integration modules for external RAG systems."""
  from promptchain.integrations.lightrag import LightRAGIntegration

  __all__ = ["LightRAGIntegration"]

  # promptchain/integrations/lightrag/__init__.py
  """LightRAG integration via hybridrag project."""
  try:
      from hybridrag.src.lightrag_core import HybridLightRAGCore
      from hybridrag.src.search_interface import SearchInterface
      LIGHTRAG_AVAILABLE = True
  except ImportError:
      LIGHTRAG_AVAILABLE = False
      HybridLightRAGCore = None
      SearchInterface = None

  from promptchain.integrations.lightrag.core import LightRAGIntegration
  from promptchain.integrations.lightrag.patterns import (
      LightRAGBranchingThoughts,
      LightRAGQueryExpander,
      LightRAGShardedRetriever,
      LightRAGMultiHop,
      LightRAGHybridSearcher,
      LightRAGSpeculativeExecutor,
  )

  __all__ = [
      "LIGHTRAG_AVAILABLE",
      "LightRAGIntegration",
      "LightRAGBranchingThoughts",
      "LightRAGQueryExpander",
      "LightRAGShardedRetriever",
      "LightRAGMultiHop",
      "LightRAGHybridSearcher",
      "LightRAGSpeculativeExecutor",
  ]
  ```

  **Acceptance Criteria**:
  - [ ] Integration module imports work
  - [ ] Graceful fallback when hybridrag not installed
  - [ ] `LIGHTRAG_AVAILABLE` flag accessible

---

- [X] T002 [S] [W1] [INFRA] [python-pro] [patterns/base.py]
  **Create Base Pattern Abstractions**

  Define abstract base class for all patterns with event emission hooks:

  ```python
  # promptchain/patterns/base.py
  from abc import ABC, abstractmethod
  from dataclasses import dataclass, field
  from typing import Any, Dict, Optional, List
  from datetime import datetime
  import uuid

  @dataclass
  class PatternConfig:
      pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
      enabled: bool = True
      timeout_seconds: float = 30.0
      emit_events: bool = True
      use_blackboard: bool = False

  @dataclass
  class PatternResult:
      pattern_id: str
      success: bool
      result: Any
      execution_time_ms: float
      metadata: Dict[str, Any] = field(default_factory=dict)
      errors: List[str] = field(default_factory=list)
      timestamp: datetime = field(default_factory=datetime.utcnow)

  class BasePattern(ABC):
      def __init__(self, config: Optional[PatternConfig] = None):
          self.config = config or PatternConfig()
          self._event_handlers = []
          self._blackboard = None

      @abstractmethod
      async def execute(self, **kwargs) -> PatternResult:
          """Execute the pattern."""
          pass

      def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
          """Emit pattern event to MessageBus if configured."""
          pass

      def set_blackboard(self, blackboard) -> None:
          """Set Blackboard for state sharing."""
          self._blackboard = blackboard
  ```

  **Acceptance Criteria**:
  - [ ] BasePattern ABC defined with execute method
  - [ ] PatternConfig and PatternResult dataclasses
  - [ ] Event emission hooks present
  - [ ] Blackboard integration hooks present

---

## Phase 2: Pattern Adapters

### Wave 2 (Parallel - 6 Agents, No File Conflicts)

| Task | Agent | Files Owned (Exclusive) | Dependencies | Est. Complexity |
|------|-------|-------------------------|--------------|-----------------|
| T003 | python-pro | `integrations/lightrag/branching.py` | T001, T002 | Medium |
| T004 | python-pro | `integrations/lightrag/query_expansion.py` | T001, T002 | Medium |
| T005 | python-pro | `integrations/lightrag/sharded.py` | T001, T002 | Medium |
| T006 | python-pro | `integrations/lightrag/multi_hop.py` | T001, T002 | Medium |
| T007 | python-pro | `integrations/lightrag/hybrid_search.py` | T001, T002 | Medium |
| T008 | python-pro | `integrations/lightrag/speculative.py` | T001, T002 | Medium |

**FILE LOCKING**: Each task owns exactly ONE pattern file. No conflicts possible.

---

- [X] T003 [P] [W2] [US1] [python-pro] [integrations/lightrag/branching.py]
  **Branching Thoughts Pattern via LightRAG**

  Use LightRAG multi-mode queries as hypothesis generators:

  ```python
  # promptchain/integrations/lightrag/branching.py
  from promptchain.patterns.base import BasePattern, PatternResult
  from promptchain.integrations.lightrag import HybridLightRAGCore
  import asyncio

  class LightRAGBranchingThoughts(BasePattern):
      """Generate hypotheses using LightRAG's local/global/hybrid modes."""

      def __init__(self, lightrag_core: HybridLightRAGCore, **kwargs):
          super().__init__(**kwargs)
          self.lightrag = lightrag_core

      async def execute(self, problem: str, hypothesis_count: int = 3) -> PatternResult:
          # Generate hypotheses via different query modes
          hypotheses = await asyncio.gather(
              self.lightrag.local_query(problem),   # Entity-focused
              self.lightrag.global_query(problem),  # Overview
              self.lightrag.hybrid_query(problem),  # Balanced
          )

          # Score and select best hypothesis
          # Use AgenticStepProcessor for evaluation
          ...
  ```

  **Acceptance Criteria**:
  - [ ] Implements BasePattern interface
  - [ ] Uses HybridLightRAGCore for hypothesis generation
  - [ ] Scores hypotheses using LLM judge
  - [ ] Emits pattern.branching.* events

---

- [X] T004 [P] [W2] [US2] [python-pro] [integrations/lightrag/query_expansion.py]
  **Query Expansion Pattern via LightRAG**

  Leverage `SearchInterface.multi_query_search()`:

  ```python
  # promptchain/integrations/lightrag/query_expansion.py
  from promptchain.patterns.base import BasePattern, PatternResult
  from promptchain.integrations.lightrag import SearchInterface

  class LightRAGQueryExpander(BasePattern):
      """Expand queries using LightRAG context extraction."""

      def __init__(self, search_interface: SearchInterface, **kwargs):
          super().__init__(**kwargs)
          self.search = search_interface

      async def execute(self, query: str, strategies: List[str] = None) -> PatternResult:
          # Use context extraction for semantic expansion
          context = await self.search.lightrag.extract_context(query)

          # Generate variations based on extracted entities
          expanded = self._generate_variations(query, context)

          # Execute multi_query_search
          results = await self.search.multi_query_search(expanded)
          ...
  ```

  **Acceptance Criteria**:
  - [ ] Implements BasePattern interface
  - [ ] Uses SearchInterface.multi_query_search
  - [ ] Supports SYNONYM, SEMANTIC, ACRONYM strategies
  - [ ] Deduplicates results with proper scoring

---

- [X] T005 [P] [W2] [US3] [python-pro] [integrations/lightrag/sharded.py]
  **Sharded Retrieval Pattern via LightRAG**

  Register multiple LightRAG databases as shards:

  ```python
  # promptchain/integrations/lightrag/sharded.py
  from promptchain.patterns.base import BasePattern, PatternResult
  from promptchain.integrations.lightrag import HybridLightRAGCore
  import asyncio

  class LightRAGShardRegistry:
      """Registry for multiple LightRAG database shards."""

      def __init__(self):
          self.shards: Dict[str, HybridLightRAGCore] = {}

      def register_shard(self, name: str, working_dir: str) -> None:
          self.shards[name] = HybridLightRAGCore(working_dir=working_dir)

  class LightRAGShardedRetriever(BasePattern):
      """Query across multiple LightRAG shards in parallel."""

      async def execute(self, query: str) -> PatternResult:
          results = await asyncio.gather(*[
              shard.hybrid_query(query) for shard in self.registry.shards.values()
          ])
          return self._aggregate_results(results)
  ```

  **Acceptance Criteria**:
  - [ ] Implements BasePattern interface
  - [ ] ShardRegistry for multiple LightRAG databases
  - [ ] Parallel query execution across shards
  - [ ] Result aggregation with source tracking

---

- [X] T006 [P] [W2] [US4] [python-pro] [integrations/lightrag/multi_hop.py]
  **Multi-Hop Retrieval Pattern via LightRAG**

  Wrap `SearchInterface.agentic_search()`:

  ```python
  # promptchain/integrations/lightrag/multi_hop.py
  from promptchain.patterns.base import BasePattern, PatternResult
  from promptchain.integrations.lightrag import SearchInterface

  class LightRAGMultiHop(BasePattern):
      """Multi-hop retrieval using LightRAG's agentic search."""

      def __init__(self, search_interface: SearchInterface, max_hops: int = 5, **kwargs):
          super().__init__(**kwargs)
          self.search = search_interface
          self.max_hops = max_hops

      async def execute(self, question: str) -> PatternResult:
          # Already implemented in SearchInterface.agentic_search()
          result = await self.search.agentic_search(
              query=question,
              objective=f"Comprehensively answer: {question}",
              max_steps=self.max_hops
          )

          return PatternResult(
              pattern_id=self.config.pattern_id,
              success=True,
              result=result,
              execution_time_ms=...,
              metadata={"hops_executed": result.steps_taken}
          )
  ```

  **Acceptance Criteria**:
  - [ ] Implements BasePattern interface
  - [ ] Wraps SearchInterface.agentic_search
  - [ ] Tracks hops/steps executed
  - [ ] Identifies unanswered aspects

---

- [X] T007 [P] [W2] [US5] [python-pro] [integrations/lightrag/hybrid_search.py]
  **Hybrid Search Fusion Pattern via LightRAG**

  LightRAG hybrid mode + explicit RRF control:

  ```python
  # promptchain/integrations/lightrag/hybrid_search.py
  from promptchain.patterns.base import BasePattern, PatternResult
  from promptchain.integrations.lightrag import HybridLightRAGCore

  class LightRAGHybridSearcher(BasePattern):
      """Hybrid search with explicit RRF fusion control."""

      async def execute(self, query: str) -> PatternResult:
          # Get results from different modes
          local_result = await self.lightrag.local_query(query)
          global_result = await self.lightrag.global_query(query)

          # Apply Reciprocal Rank Fusion
          fused = self._reciprocal_rank_fusion([local_result, global_result])

          return PatternResult(
              result=fused,
              metadata={
                  "technique_contributions": {
                      "local": local_result.score,
                      "global": global_result.score
                  }
              }
          )

      def _reciprocal_rank_fusion(self, results: List, k: int = 60) -> List:
          """RRF formula: score = sum(1 / (k + rank))"""
          ...
  ```

  **Acceptance Criteria**:
  - [ ] Implements BasePattern interface
  - [ ] Uses local + global query modes
  - [ ] Implements RRF fusion algorithm
  - [ ] Tracks per-technique contributions

---

- [X] T008 [P] [W2] [US6] [python-pro] [integrations/lightrag/speculative.py]
  **Speculative Execution Pattern via LightRAG**

  Predict next LightRAG queries based on conversation patterns:

  ```python
  # promptchain/integrations/lightrag/speculative.py
  from promptchain.patterns.base import BasePattern, PatternResult
  from promptchain.integrations.lightrag import HybridLightRAGCore
  from collections import deque

  class LightRAGSpeculativeExecutor(BasePattern):
      """Predict and pre-execute likely LightRAG queries."""

      def __init__(self, lightrag_core: HybridLightRAGCore,
                   history_window: int = 20, min_confidence: float = 0.7, **kwargs):
          super().__init__(**kwargs)
          self.lightrag = lightrag_core
          self.history = deque(maxlen=history_window)
          self.min_confidence = min_confidence
          self.cache = {}

      def record_call(self, query: str, mode: str) -> None:
          self.history.append({"query": query, "mode": mode})

      def predict_next_queries(self, context: str) -> List[Dict]:
          """Analyze patterns to predict next queries."""
          # Frequency-based prediction
          # Pattern matching (follow-up questions)
          ...

      async def execute(self, context: str) -> PatternResult:
          predictions = self.predict_next_queries(context)

          # Speculatively execute high-confidence predictions
          for pred in predictions:
              if pred["confidence"] >= self.min_confidence:
                  result = await self.lightrag.hybrid_query(pred["query"])
                  self.cache[pred["query"]] = result
          ...
  ```

  **Acceptance Criteria**:
  - [ ] Implements BasePattern interface
  - [ ] Tracks query history for pattern analysis
  - [ ] Predicts likely next queries
  - [ ] Caches speculative results with TTL
  - [ ] Reports cache hit rate and latency savings

---

## Phase 3: Integration Layer

### Wave 3 (Sequential - Shared Integration Files)

| Task | Agent | Files Owned (Exclusive) | Dependencies | Est. Complexity |
|------|-------|-------------------------|--------------|-----------------|
| T009 | python-pro | `integrations/lightrag/messaging.py` | T003-T008 | Medium |
| T010 | python-pro | `integrations/lightrag/state.py` | T003-T008, T009 | Medium |
| T011 | python-pro | `integrations/lightrag/events.py` | T003-T008, T009, T010 | Low |

---

- [X] T009 [S] [W3] [INFRA] [python-pro] [integrations/lightrag/messaging.py]
  **MessageBus Integration for Patterns**

  Enable patterns to publish/subscribe via 003 MessageBus:

  ```python
  # promptchain/integrations/lightrag/messaging.py
  from promptchain.cli.models import MessageBus
  from promptchain.patterns.base import BasePattern

  class PatternMessageBusMixin:
      """Mixin to add MessageBus integration to patterns."""

      def connect_messagebus(self, bus: MessageBus) -> None:
          self._bus = bus

      def emit_event(self, event_type: str, data: Dict) -> None:
          if hasattr(self, '_bus') and self._bus:
              self._bus.publish(event_type, data)

      def subscribe_to(self, pattern: str, handler: Callable) -> None:
          if hasattr(self, '_bus') and self._bus:
              self._bus.subscribe(pattern, handler)
  ```

  **Acceptance Criteria**:
  - [ ] Mixin class for MessageBus integration
  - [ ] Pattern events published to bus
  - [ ] Cross-pattern subscriptions work

---

- [X] T010 [S] [W3] [INFRA] [python-pro] [integrations/lightrag/state.py]
  **Blackboard Integration for Patterns**

  Enable patterns to share state via 003 Blackboard:

  ```python
  # promptchain/integrations/lightrag/state.py
  from promptchain.cli.models import Blackboard
  from promptchain.patterns.base import BasePattern

  class PatternBlackboardMixin:
      """Mixin to add Blackboard state sharing to patterns."""

      def connect_blackboard(self, blackboard: Blackboard) -> None:
          self._blackboard = blackboard

      def share_result(self, key: str, value: Any) -> None:
          if self._blackboard:
              self._blackboard.write(key, value, source=self.config.pattern_id)

      def read_shared(self, key: str) -> Optional[Any]:
          if self._blackboard:
              return self._blackboard.read(key)
          return None
  ```

  **Acceptance Criteria**:
  - [ ] Mixin class for Blackboard integration
  - [ ] Patterns can share results via Blackboard
  - [ ] Cross-pattern state access works

---

- [X] T011 [S] [W3] [INFRA] [python-pro] [integrations/lightrag/events.py]
  **Event Emission System**

  Standardize pattern event types and payloads:

  ```python
  # promptchain/integrations/lightrag/events.py
  from dataclasses import dataclass
  from datetime import datetime
  from typing import Any, Dict

  @dataclass
  class PatternEvent:
      event_type: str
      pattern_id: str
      timestamp: datetime
      data: Dict[str, Any]

  # Standard event types
  PATTERN_EVENTS = {
      "branching": [
          "pattern.branching.started",
          "pattern.branching.hypothesis_generated",
          "pattern.branching.judging",
          "pattern.branching.selected",
          "pattern.branching.completed",
      ],
      "query_expansion": [
          "pattern.query_expansion.started",
          "pattern.query_expansion.expanded",
          "pattern.query_expansion.searching",
          "pattern.query_expansion.completed",
      ],
      # ... etc for all patterns
  }
  ```

  **Acceptance Criteria**:
  - [ ] PatternEvent dataclass defined
  - [ ] Standard event types for each pattern
  - [ ] Event payloads documented

---

## Phase 4: Testing

### Wave 4 (Parallel - Separate Test Files)

| Task | Agent | Files Owned (Exclusive) | Dependencies | Est. Complexity |
|------|-------|-------------------------|--------------|-----------------|
| T012 | test-automator | `tests/unit/patterns/` | T003-T011 | Medium |
| T013 | test-automator | `tests/integration/patterns/` | T003-T011 | Medium |
| T014 | test-automator | `tests/e2e/patterns/` | T003-T011 | High |

**FILE LOCKING**: Each test task owns distinct test directories.

---

- [X] T012 [P] [W4] [TEST] [test-automator] [tests/unit/patterns/]
  **Unit Tests for Pattern Adapters**

  Test each pattern adapter in isolation with mocked LightRAG:

  ```python
  # tests/unit/patterns/test_branching_thoughts.py
  import pytest
  from unittest.mock import AsyncMock, MagicMock
  from promptchain.integrations.lightrag import LightRAGBranchingThoughts

  @pytest.fixture
  def mock_lightrag():
      mock = MagicMock()
      mock.local_query = AsyncMock(return_value="Local hypothesis")
      mock.global_query = AsyncMock(return_value="Global hypothesis")
      mock.hybrid_query = AsyncMock(return_value="Hybrid hypothesis")
      return mock

  @pytest.mark.asyncio
  async def test_branching_generates_hypotheses(mock_lightrag):
      brancher = LightRAGBranchingThoughts(lightrag_core=mock_lightrag)
      result = await brancher.execute(problem="Test problem")

      assert result.success
      assert len(result.result.hypotheses) == 3
      mock_lightrag.local_query.assert_called_once()
  ```

  **Acceptance Criteria**:
  - [ ] Unit tests for each pattern (6 files)
  - [ ] 80%+ code coverage
  - [ ] Mocked LightRAG dependencies
  - [ ] Tests pass without external dependencies

---

- [X] T013 [P] [W4] [TEST] [test-automator] [tests/integration/patterns/]
  **Integration Tests with 003 Infrastructure**

  Test patterns with real MessageBus and Blackboard:

  ```python
  # tests/integration/patterns/test_pattern_messaging.py
  import pytest
  from promptchain.cli.models import MessageBus, Blackboard
  from promptchain.integrations.lightrag import LightRAGBranchingThoughts

  @pytest.fixture
  def session_infrastructure():
      bus = MessageBus(session_id="test-session")
      blackboard = Blackboard(session_id="test-session")
      return bus, blackboard

  @pytest.mark.asyncio
  async def test_pattern_emits_events(session_infrastructure, mock_lightrag):
      bus, blackboard = session_infrastructure
      events_received = []
      bus.subscribe("pattern.branching.*", lambda e: events_received.append(e))

      brancher = LightRAGBranchingThoughts(lightrag_core=mock_lightrag)
      brancher.connect_messagebus(bus)

      await brancher.execute(problem="Test")

      assert len(events_received) > 0
      assert any("started" in e.event_type for e in events_received)
  ```

  **Acceptance Criteria**:
  - [ ] Tests patterns with MessageBus
  - [ ] Tests patterns with Blackboard
  - [ ] Cross-pattern communication tested
  - [ ] Event propagation verified

---

- [X] T014 [P] [W4] [TEST] [test-automator] [tests/e2e/patterns/]
  **End-to-End Workflow Tests**

  Test complete workflows with real (or simulated) LightRAG:

  ```python
  # tests/e2e/patterns/test_multi_pattern_workflow.py
  import pytest
  from promptchain.integrations.lightrag import (
      LightRAGQueryExpander,
      LightRAGMultiHop,
      LightRAGHybridSearcher,
  )

  @pytest.mark.e2e
  @pytest.mark.asyncio
  async def test_research_workflow():
      """Test a complete research workflow using multiple patterns."""

      # 1. Expand query
      expander = LightRAGQueryExpander(...)
      expanded = await expander.execute(query="ML model optimization")

      # 2. Multi-hop retrieval on expanded queries
      multi_hop = LightRAGMultiHop(...)
      answers = await asyncio.gather(*[
          multi_hop.execute(q) for q in expanded.result.expanded_queries
      ])

      # 3. Hybrid search fusion on answers
      hybrid = LightRAGHybridSearcher(...)
      final = await hybrid.execute(query=..., context=answers)

      assert final.success
      assert final.result.fused_results
  ```

  **Acceptance Criteria**:
  - [ ] Multi-pattern workflow tests
  - [ ] Real LightRAG integration (optional, marked)
  - [ ] Performance benchmarks captured
  - [ ] Error scenarios tested

---

## Phase 5: CLI & Documentation

### Wave 5 (Parallel - Distinct Files)

| Task | Agent | Files Owned (Exclusive) | Dependencies | Est. Complexity |
|------|-------|-------------------------|--------------|-----------------|
| T015 | python-pro | `promptchain/cli/commands/patterns.py` | T003-T014 | Medium |
| T016 | python-pro | `docs/patterns/`, `examples/patterns/` | T003-T014 | Low |

---

- [X] T015 [P] [W5] [CLI] [python-pro] [cli/commands/patterns.py]
  **CLI Pattern Commands**

  Add CLI commands for pattern execution:

  ```python
  # promptchain/cli/commands/patterns.py
  import click
  from promptchain.integrations.lightrag import (
      LightRAGBranchingThoughts,
      LightRAGQueryExpander,
      LightRAGMultiHop,
  )

  @click.group()
  def patterns():
      """Advanced agentic pattern commands."""
      pass

  @patterns.command()
  @click.argument('problem')
  @click.option('--count', default=3, help='Number of hypotheses')
  def branch(problem: str, count: int):
      """Generate branching hypotheses for a problem."""
      ...

  @patterns.command()
  @click.argument('query')
  @click.option('--strategies', multiple=True, default=['semantic'])
  def expand(query: str, strategies: tuple):
      """Expand a query using multiple strategies."""
      ...

  @patterns.command()
  @click.argument('question')
  @click.option('--max-hops', default=5)
  def multihop(question: str, max_hops: int):
      """Answer a complex question via multi-hop retrieval."""
      ...
  ```

  **Acceptance Criteria**:
  - [ ] `promptchain patterns branch` command
  - [ ] `promptchain patterns expand` command
  - [ ] `promptchain patterns multihop` command
  - [ ] `promptchain patterns hybrid` command
  - [ ] `promptchain patterns speculate` command
  - [ ] Help text and examples

---

- [X] T016 [P] [W5] [DOCS] [python-pro] [docs/patterns/, examples/patterns/]
  **Documentation and Examples**

  Create comprehensive documentation:

  ```markdown
  # docs/patterns/README.md

  # Advanced Agentic Patterns

  PromptChain provides 6 advanced patterns powered by LightRAG integration.

  ## Installation

  ```bash
  pip install promptchain[lightrag]
  # Or install hybridrag directly:
  pip install git+https://github.com/gyasis/hybridrag.git
  ```

  ## Patterns

  ### 1. Branching Thoughts
  [Usage examples...]

  ### 2. Query Expansion
  [Usage examples...]

  ...
  ```

  **Acceptance Criteria**:
  - [ ] docs/patterns/README.md
  - [ ] Pattern-specific documentation (6 files)
  - [ ] examples/patterns/ with runnable scripts
  - [ ] quickstart.md updated with real examples

---

## Execution Summary

| Phase | Wave | Tasks | Parallel? | File Conflicts | Est. Duration |
|-------|------|-------|-----------|----------------|---------------|
| 1 | W1 | T001-T002 | No (Sequential) | N/A | Short |
| 2 | W2 | T003-T008 | Yes (6 agents) | None (separate files) | Medium |
| 3 | W3 | T009-T011 | No (Sequential) | Shared integration | Short |
| 4 | W4 | T012-T014 | Yes (3 agents) | None (separate dirs) | Medium |
| 5 | W5 | T015-T016 | Yes (2 agents) | None (separate dirs) | Short |

**Total Tasks**: 16
**Parallelizable**: 11 (69%)
**Sequential**: 5 (31%)
**Max Parallel Agents**: 6 (Wave 2)

---

## Checkpoint Protocol

After each Wave completion:

1. **Memory Bank Update** (via memory-bank-keeper):
   - Update `progress.md` with completed tasks
   - Update `activeContext.md` with current state

2. **Git Commit** (via git-version-manager):
   - Stage all modified files
   - Semantic commit: `feat(004): Complete Wave N - [description]`

3. **Verification**:
   - All tests pass for completed tasks
   - No uncommitted changes
   - File ownership released

---

## Risk Mitigation

| Risk | Mitigation | Owner |
|------|------------|-------|
| hybridrag not installed | Graceful fallback, clear error messages | T001 |
| LightRAG API changes | Abstract behind integration layer | T001-T002 |
| File conflicts in Wave 2 | Strict file ownership, 1 file per task | Orchestrator |
| Test flakiness | Mock external dependencies, use fixtures | T012-T014 |

---

## Success Criteria

- [ ] All 16 tasks completed
- [ ] 80%+ test coverage for new code
- [ ] All patterns integrate with MessageBus/Blackboard
- [ ] CLI commands functional
- [ ] Documentation complete
- [ ] No regressions in existing functionality
