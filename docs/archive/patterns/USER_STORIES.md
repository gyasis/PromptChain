# User Stories: 003-004 Integration Scenarios

This document shows how the **003-multi-agent-communication** infrastructure (MessageBus, Blackboard, native CLI) integrates with **004-advanced-agentic-patterns** (LightRAG patterns).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PromptChain CLI (main.py)                       │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Native TUI Mode       │    │   Patterns Subcommand           │ │
│  │   (default behavior)    │    │   promptchain patterns <cmd>    │ │
│  │                         │    │                                 │ │
│  │   - Interactive chat    │    │   - branch "query"              │ │
│  │   - Multi-agent mgmt    │    │   - expand "query"              │ │
│  │   - Session persistence │    │   - multihop "query"            │ │
│  │   - @file context       │    │   - hybrid "query"              │ │
│  │   - !shell commands     │    │   - sharded "query"             │ │
│  └──────────┬──────────────┘    │   - speculate "query"           │ │
│             │                   └──────────┬──────────────────────┘ │
└─────────────┼──────────────────────────────┼───────────────────────┘
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   003: Multi-Agent Communication                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │    MessageBus    │  │    Blackboard    │  │  AgentRegistry   │   │
│  │  - publish()     │  │  - write()       │  │  - register()    │   │
│  │  - subscribe()   │  │  - read()        │  │  - get()         │   │
│  │  - broadcast()   │  │  - snapshot()    │  │  - list()        │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
│           │                     │                     │             │
│           └─────────────────────┼─────────────────────┘             │
│                                 │                                   │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   004: Advanced Agentic Patterns                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │  Branching  │ │   Query     │ │  Sharded    │ │  Multi-Hop  │   │
│  │  Thoughts   │ │  Expansion  │ │  Retrieval  │ │  Retrieval  │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │
│  ┌─────────────┐ ┌─────────────┐                                   │
│  │   Hybrid    │ │ Speculative │   All patterns inherit from       │
│  │   Search    │ │  Execution  │   BasePattern with MessageBus     │
│  └─────────────┘ └─────────────┘   and Blackboard integration      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## User Story 1: Basic Pattern Execution (CLI Only)

**As a** developer exploring LightRAG patterns
**I want to** run patterns from the command line
**So that** I can quickly test retrieval strategies without writing code

### Scenario: Run branching thoughts on a research query

```bash
# User runs pattern command
$ promptchain patterns branch "What are the security implications of LLM tool use?"

# Output shows hypothesis generation + judge evaluation
┌─ Branching Thoughts Analysis ─────────────────────────────────┐
│ Query: What are the security implications of LLM tool use?    │
│ Mode: hybrid                                                  │
│ Branches: 3                                                   │
├───────────────────────────────────────────────────────────────┤
│ Branch 1 (Score: 0.89)                                        │
│ ├─ Hypothesis: Prompt injection via tool parameters           │
│ └─ Evidence: Retrieved 5 relevant documents                   │
│                                                               │
│ Branch 2 (Score: 0.76)                                        │
│ ├─ Hypothesis: Unintended data exfiltration                   │
│ └─ Evidence: Retrieved 3 relevant documents                   │
│                                                               │
│ Branch 3 (Score: 0.71)                                        │
│ ├─ Hypothesis: Resource exhaustion attacks                    │
│ └─ Evidence: Retrieved 4 relevant documents                   │
├───────────────────────────────────────────────────────────────┤
│ Final Synthesis:                                              │
│ The primary security concerns for LLM tool use are...         │
└───────────────────────────────────────────────────────────────┘
```

### What happens behind the scenes:

1. CLI parses `patterns branch` subcommand
2. `LightRAGBranchingThoughts` pattern instantiated
3. Pattern emits `pattern.branching.started` event
4. Hypotheses generated, each emits `pattern.branching.branch_created`
5. Judge evaluates branches, emits `pattern.branching.progress`
6. Final synthesis, emits `pattern.branching.completed`
7. Results formatted and displayed

---

## User Story 2: Pattern + MessageBus Integration (Python API)

**As a** developer building a research assistant
**I want to** patterns to communicate via MessageBus
**So that** I can coordinate multiple patterns and track progress

### Scenario: Monitor pattern execution in real-time

```python
from promptchain.cli.models import MessageBus, Blackboard
from promptchain.integrations.lightrag import (
    LightRAGBranchingThoughts,
    LightRAGQueryExpander,
    PatternConfig
)

# Initialize 003 infrastructure
bus = MessageBus()
blackboard = Blackboard()

# Create patterns with 003 integration
branching = LightRAGBranchingThoughts(
    config=PatternConfig(
        pattern_id="research-branching",
        emit_events=True,
        use_blackboard=True
    )
)
branching.connect_messagebus(bus)
branching.connect_blackboard(blackboard)

# Subscribe to pattern events
def on_branch_created(event_type, data):
    print(f"New branch: {data['hypothesis']}")
    print(f"  Score: {data.get('score', 'pending')}")

def on_pattern_complete(event_type, data):
    print(f"Pattern {data['pattern_id']} completed in {data['execution_time_ms']}ms")
    # Read final results from blackboard
    results = blackboard.read("research-branching:final_results")
    print(f"Final synthesis available: {results is not None}")

bus.subscribe("pattern.branching.branch_created", on_branch_created)
bus.subscribe("pattern.*.completed", on_pattern_complete)

# Execute pattern
import asyncio
result = asyncio.run(branching.execute_with_timeout(
    query="How do transformer attention mechanisms work?"
))

# Results also available on blackboard
synthesis = blackboard.read("research-branching:synthesis")
```

### Output:

```
New branch: Self-attention computes weighted relationships
  Score: pending
New branch: Multi-head attention enables diverse representations
  Score: pending
New branch: Positional encoding provides sequence awareness
  Score: pending
Pattern research-branching completed in 2341ms
Final synthesis available: True
```

---

## User Story 3: Multi-Pattern Coordination via Blackboard

**As a** developer building a complex RAG pipeline
**I want to** patterns to share state via Blackboard
**So that** downstream patterns can use upstream results

### Scenario: Query expansion feeds into multi-hop retrieval

```python
from promptchain.cli.models import MessageBus, Blackboard
from promptchain.integrations.lightrag import (
    LightRAGQueryExpander,
    LightRAGMultiHop,
    PatternConfig,
    PatternStateCoordinator
)
import asyncio

# Shared infrastructure
bus = MessageBus()
blackboard = Blackboard()

# State coordinator for cross-pattern orchestration
coordinator = PatternStateCoordinator(blackboard)

# Pattern 1: Query Expansion
expander = LightRAGQueryExpander(
    config=PatternConfig(
        pattern_id="step1-expand",
        emit_events=True,
        use_blackboard=True
    )
)
expander.connect_messagebus(bus)
expander.connect_blackboard(blackboard)

# Pattern 2: Multi-Hop (will consume expansion results)
multihop = LightRAGMultiHop(
    config=PatternConfig(
        pattern_id="step2-multihop",
        emit_events=True,
        use_blackboard=True
    )
)
multihop.connect_messagebus(bus)
multihop.connect_blackboard(blackboard)

async def coordinated_pipeline(query: str):
    # Step 1: Expand query into variations
    expand_result = await expander.execute_with_timeout(query=query)

    # Expansion writes to blackboard automatically:
    # - "step1-expand:expanded_queries" = ["query1", "query2", "query3"]

    # Step 2: Multi-hop reads expansion from blackboard
    expanded = blackboard.read("step1-expand:expanded_queries")

    # Run multi-hop on each expanded query
    all_results = []
    for eq in expanded:
        result = await multihop.execute_with_timeout(query=eq)
        all_results.append(result)

    # Coordinator creates unified snapshot
    snapshot = coordinator.create_snapshot(["step1-expand", "step2-multihop"])

    return {
        "original_query": query,
        "expanded_queries": expanded,
        "multihop_results": all_results,
        "state_snapshot": snapshot
    }

# Execute pipeline
result = asyncio.run(coordinated_pipeline(
    "What causes transformer models to hallucinate?"
))

print(f"Original: {result['original_query']}")
print(f"Expanded into {len(result['expanded_queries'])} queries")
print(f"Multi-hop found {sum(r.success for r in result['multihop_results'])} successful paths")
```

---

## User Story 4: TUI + Patterns Integration (Native Support)

**As a** power user
**I want to** use patterns within the interactive TUI
**So that** I can combine conversational AI with advanced retrieval without interrupting my session

### Scenario: Research session with pattern-enhanced retrieval

```bash
# Start interactive session
$ promptchain --session research-project

Welcome to PromptChain CLI!

# Normal conversational query
> What are the main approaches to RAG?
[Agent provides general overview from training data]

# User wants deeper research - run pattern directly in TUI!
> /multihop "What are the main approaches to RAG?" --max-hops=3

🔗 Multi-hop retrieval for: What are the main approaches to RAG?

✅ Completed 3 hops:

Hop 1: RAG architectures
  └─ Found: Dense retrieval, sparse retrieval, hybrid

Hop 2: Dense vs sparse retrieval comparison
  └─ Found: BM25, DPR, ColBERT, hybrid fusion

Hop 3: ColBERT late interaction mechanism
  └─ Found: MaxSim scoring, token-level matching

📊 Synthesized Answer:
The main RAG approaches are:
1. Dense retrieval (DPR): Semantic embeddings...
2. Sparse retrieval (BM25): Lexical matching...
3. Hybrid fusion: Combines dense + sparse...
4. Late interaction (ColBERT): Token-level MaxSim...

⏱️ Execution time: 3102ms

# Continue conversation with pattern results in session context
> Based on this multi-hop research, can you help me implement a hybrid RAG system?
[Agent has pattern results in session history and can reference them]

# Try other patterns without leaving TUI
> /branch "How do vector databases handle dimensionality reduction?" --count=5

🌳 Generating 5 branching hypotheses...

✅ Generated 5 hypotheses:
1. Approximate nearest neighbor algorithms like HNSW
2. Dimensionality reduction techniques like PCA
3. Quantization methods for compression
4. Hierarchical indexing strategies
5. Locality-sensitive hashing (LSH)

⏱️ Execution time: 2341ms

# Check available pattern commands
> /patterns

Pattern Commands:

🌳 /branch "query" [--count=N] [--mode=local|global|hybrid]
   Generate branching hypotheses for exploration

🔄 /expand "query" [--strategies=semantic,synonym] [--max=N]
   Expand query with variations

🔗 /multihop "query" [--max-hops=N] [--mode=hybrid]
   Multi-hop retrieval with reasoning chains

🔀 /hybrid "query" [--fusion=rrf|linear|borda] [--top-k=N]
   Hybrid search combining dense and sparse retrieval

🗂️  /sharded "query" --shards=shard1,shard2 [--aggregation=rrf]
   Search across multiple sharded indexes

⚡ /speculate "context" [--min-confidence=0.7] [--prefetch=N]
   Speculative execution with prefetching

📖 /patterns
   Show this help message

Examples:
  /branch "quantum computing applications" --count=5
  /expand "machine learning" --strategies=semantic,synonym
  /multihop "what caused the 2008 financial crisis" --max-hops=3
```

### What happens behind the scenes:

1. TUI parses `/multihop` command with arguments
2. `_handle_multihop_pattern()` method called in `app.py`
3. `_parse_pattern_command()` extracts query and options
4. `execute_multihop()` executor called with session MessageBus/Blackboard
5. Pattern emits events to session MessageBus (if available)
6. Results formatted for chat display with emoji indicators
7. Command and results added to session message history
8. Session continues without interruption

---

## User Story 5: Event-Driven Pattern Monitoring (DevOps/Observability)

**As a** DevOps engineer
**I want to** monitor pattern execution across the system
**So that** I can track performance and debug issues

### Scenario: Centralized event logging

```python
from promptchain.cli.models import MessageBus
from promptchain.integrations.lightrag import (
    PATTERN_EVENTS,
    PatternEvent,
    EventSeverity
)
import json
from datetime import datetime

# Create monitoring bus
monitor_bus = MessageBus()

# Event aggregator
event_log = []

def log_all_events(event_type: str, data: dict):
    """Central event logger for all pattern activity."""
    event_log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "pattern_id": data.get("pattern_id"),
        "severity": data.get("severity", "info"),
        "data": data
    })

    # Alert on errors
    if "error" in event_type or "timeout" in event_type:
        print(f"ALERT: {event_type} - {data.get('error', 'timeout')}")

# Subscribe to ALL pattern events
monitor_bus.subscribe("pattern.*", log_all_events)

# Connect patterns to monitored bus
from promptchain.integrations.lightrag import LightRAGBranchingThoughts
pattern = LightRAGBranchingThoughts()
pattern.connect_messagebus(monitor_bus)

# After execution, analyze events
print(f"Total events captured: {len(event_log)}")

# Event breakdown by type
from collections import Counter
event_types = Counter(e["event_type"] for e in event_log)
for event_type, count in event_types.most_common():
    print(f"  {event_type}: {count}")

# Performance analysis
completed_events = [e for e in event_log if "completed" in e["event_type"]]
for e in completed_events:
    print(f"Pattern {e['pattern_id']} completed in {e['data'].get('execution_time_ms')}ms")
```

### Sample output:

```
Total events captured: 8
  pattern.branching.started: 1
  pattern.branching.branch_created: 3
  pattern.branching.progress: 3
  pattern.branching.completed: 1

Pattern research-branching completed in 2341ms
```

---

## User Story 6: Graceful Degradation (No LightRAG Installed)

**As a** user without the hybridrag dependency
**I want to** get helpful error messages
**So that** I know what to install

### Scenario: Pattern command with missing dependency

```bash
# User hasn't installed hybridrag
$ promptchain patterns branch "test query"

Error: LightRAG integration not available.

To use pattern commands, install the hybridrag package:
    pip install git+https://github.com/gyasis/hybridrag.git

Once installed, the following patterns will be available:
  - branch:     Branching Thoughts (hypothesis generation + judge)
  - expand:     Query Expansion (parallel diversification)
  - multihop:   Multi-Hop Retrieval (question decomposition)
  - hybrid:     Hybrid Search Fusion (technique combination)
  - sharded:    Sharded Retrieval (multi-source parallel)
  - speculate:  Speculative Execution (predictive tool calling)

For more information, see: docs/patterns/README.md
```

---

## User Story 7: Combined Pattern Execution (Power User)

**As a** researcher doing comprehensive analysis
**I want to** run multiple patterns and compare results
**So that** I can choose the best retrieval strategy

### Scenario: Compare branching vs expansion vs multi-hop

```python
from promptchain.integrations.lightrag import (
    LightRAGBranchingThoughts,
    LightRAGQueryExpander,
    LightRAGMultiHop,
    PatternConfig
)
import asyncio

async def compare_patterns(query: str):
    """Run multiple patterns and compare results."""

    patterns = {
        "branching": LightRAGBranchingThoughts(
            config=PatternConfig(pattern_id="compare-branch")
        ),
        "expansion": LightRAGQueryExpander(
            config=PatternConfig(pattern_id="compare-expand")
        ),
        "multihop": LightRAGMultiHop(
            config=PatternConfig(pattern_id="compare-multihop")
        )
    }

    # Execute all patterns concurrently
    tasks = {
        name: pattern.execute_with_timeout(query=query)
        for name, pattern in patterns.items()
    }

    results = {}
    for name, task in tasks.items():
        results[name] = await task

    # Compare results
    print(f"Query: {query}\n")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name.upper()}")
        print(f"  Success: {result.success}")
        print(f"  Time: {result.execution_time_ms:.0f}ms")
        if result.success:
            # Pattern-specific result summary
            if name == "branching":
                branches = result.result.get("branches", [])
                print(f"  Branches: {len(branches)}")
                for b in branches[:2]:
                    print(f"    - {b.get('hypothesis', 'N/A')[:50]}...")
            elif name == "expansion":
                queries = result.result.get("expanded_queries", [])
                print(f"  Expanded queries: {len(queries)}")
            elif name == "multihop":
                hops = result.result.get("hops", [])
                print(f"  Hops completed: {len(hops)}")

# Run comparison
asyncio.run(compare_patterns("How do vector databases handle high-dimensional data?"))
```

### Output:

```
Query: How do vector databases handle high-dimensional data?

============================================================

BRANCHING
  Success: True
  Time: 2341ms
  Branches: 3
    - Approximate nearest neighbor algorithms like HNSW...
    - Dimensionality reduction techniques like PCA...

EXPANSION
  Success: True
  Time: 1856ms
  Expanded queries: 5

MULTIHOP
  Success: True
  Time: 3102ms
  Hops completed: 3
```

---

## Integration Matrix

| Feature | Native TUI (003) | Pattern CLI (004) | Pattern TUI (004a) | Python API |
|---------|------------------|-------------------|--------------------|------------|
| Interactive chat | ✅ | ❌ | ✅ | ✅ |
| Session persistence | ✅ | ❌ (stateless) | ✅ | ✅ |
| Multi-agent routing | ✅ | ❌ | ✅ | ✅ |
| Branching Thoughts | ❌ | ✅ `patterns branch` | ✅ `/branch` | ✅ |
| Query Expansion | ❌ | ✅ `patterns expand` | ✅ `/expand` | ✅ |
| Multi-Hop | ❌ | ✅ `patterns multihop` | ✅ `/multihop` | ✅ |
| Hybrid Search | ❌ | ✅ `patterns hybrid` | ✅ `/hybrid` | ✅ |
| Sharded Retrieval | ❌ | ✅ `patterns sharded` | ✅ `/sharded` | ✅ |
| Speculative Exec | ❌ | ✅ `patterns speculate` | ✅ `/speculate` | ✅ |
| MessageBus events | ✅ | ✅ (emits) | ✅ (integrated) | ✅ |
| Blackboard state | ✅ | ❌ (stateless) | ✅ (integrated) | ✅ |
| Event monitoring | ✅ | ✅ (verbose) | ✅ (session) | ✅ |
| Results in history | ✅ | ❌ | ✅ | ✅ |
| No exit required | ✅ | ❌ | ✅ | ✅ |

**Legend**:
- **Native TUI (003)**: Standard interactive chat (`promptchain`)
- **Pattern CLI (004)**: Standalone pattern commands (`promptchain patterns branch "query"`)
- **Pattern TUI (004a)**: ✨ NEW - Patterns inside TUI session (`/branch "query"`)
- **Python API**: Programmatic pattern usage

---

## What's Integrated (as of 004a - November 2025)

✅ **COMPLETE: TUI Pattern Integration** (spec 004a-tui-pattern-commands)
- All 6 patterns available as TUI slash commands (/branch, /expand, /multihop, /hybrid, /sharded, /speculate)
- Shell-style argument parsing with --flag=value syntax
- Results formatted for chat display with emoji indicators
- Commands added to session message history
- MessageBus/Blackboard integration for event tracking
- No need to exit TUI to use patterns

✅ **COMPLETE: Session-Aware Patterns**
- Pattern results automatically saved to session history
- Results available for agent reference in subsequent messages
- Full session persistence across TUI restarts

## What's NOT Integrated Yet

These are potential future enhancements:

1. **Pattern Chaining in TUI**: Automatic chaining of multiple patterns in conversation flow
2. **Agent + Pattern Hybrid**: Use patterns as tools within agent conversations (agent can invoke patterns)
3. **Pattern Results as Agent Context**: Automatic injection of pattern outputs into agent system prompts
4. **Pattern Composition UI**: Visual interface for building multi-pattern workflows
5. **Pattern Performance Dashboard**: Real-time monitoring of pattern execution metrics

---

## Quick Reference: How to Test Integration

```bash
# 1. Test CLI patterns work
promptchain patterns --help
promptchain patterns branch "test query" --dry-run

# 2. Test native TUI still works
promptchain --session test
# Then /exit

# 3. Test Python API integration
python -c "
from promptchain.integrations.lightrag import LIGHTRAG_AVAILABLE, LightRAGBranchingThoughts
print(f'LightRAG available: {LIGHTRAG_AVAILABLE}')
if LIGHTRAG_AVAILABLE:
    pattern = LightRAGBranchingThoughts()
    print(f'Pattern created: {pattern.config.pattern_id}')
"

# 4. Test MessageBus integration
python -c "
from promptchain.cli.models import MessageBus
from promptchain.integrations.lightrag import LightRAGBranchingThoughts
bus = MessageBus()
pattern = LightRAGBranchingThoughts()
pattern.connect_messagebus(bus)
print('MessageBus connected successfully')
"
```
