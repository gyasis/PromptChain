# 004a Tasks: TUI Pattern Slash Commands

## Overview
- **Spec**: 004a-tui-pattern-commands
- **Estimated Effort**: 2-3 hours
- **Dependencies**: 004-advanced-agentic-patterns (complete)

---

## Wave 1: Core Executor Extraction (Sequential)

### T001 [S] Create pattern executors module
**File**: `promptchain/patterns/executors.py`
**Description**: Extract core async execution logic from Click commands

```python
# New file with pattern execution functions
async def execute_branch(query: str, count: int = 3, mode: str = "hybrid",
                         deeplake_path: str = None, verbose: bool = False) -> dict:
    """Execute branching thoughts pattern. Returns dict with results."""

async def execute_expand(query: str, strategies: list = None,
                         max_expansions: int = 5, ...) -> dict:
    """Execute query expansion pattern."""

async def execute_multihop(query: str, max_hops: int = 5, ...) -> dict:
async def execute_hybrid(query: str, fusion: str = "rrf", ...) -> dict:
async def execute_sharded(query: str, shards: list, ...) -> dict:
async def execute_speculate(context: str, ...) -> dict:
```

**Acceptance Criteria**:
- [x] All 6 execute_* functions created
- [x] Each returns standardized dict with {success, result, error, execution_time_ms}
- [x] Functions work standalone without Click dependency
- [x] Graceful error handling for missing hybridrag

---

### T002 [S] [T001] Refactor Click commands to use executors
**File**: `promptchain/cli/commands/patterns.py`
**Description**: Make Click commands thin wrappers around executors

```python
# Before:
@patterns.command()
def branch(problem, count, mode, ...):
    async def execute():
        branching = LightRAGBranchingThoughts(...)
        result = await branching.process_step(problem)
        return result
    result = run_async(execute())
    # display...

# After:
@patterns.command()
def branch(problem, count, mode, ...):
    from promptchain.patterns.executors import execute_branch
    result = run_async(execute_branch(problem, count, mode, ...))
    # display (unchanged)...
```

**Acceptance Criteria**:
- [x] All 6 Click commands refactored
- [x] CLI behavior unchanged (same output)
- [x] `promptchain patterns --help` still works

---

## Wave 2: TUI Integration (Parallel after Wave 1)

### T003 [P] [T001] Add pattern commands to registry
**File**: `promptchain/cli/command_handler.py`
**Description**: Register pattern slash commands for autocomplete

```python
COMMAND_REGISTRY: Dict[str, Dict[str, str]] = {
    # ... existing commands ...

    # Pattern commands (004a)
    "/branch": {"description": "Generate branching hypotheses", "usage": '/branch "query" [--count=3] [--mode=hybrid]'},
    "/expand": {"description": "Expand query variations", "usage": '/expand "query" [--strategies=semantic]'},
    "/multihop": {"description": "Multi-hop retrieval", "usage": '/multihop "query" [--max-hops=5]'},
    "/hybrid": {"description": "Hybrid search fusion", "usage": '/hybrid "query" [--fusion=rrf]'},
    "/sharded": {"description": "Sharded retrieval", "usage": '/sharded "query" --shards=shard1,shard2'},
    "/speculate": {"description": "Speculative execution", "usage": '/speculate "context"'},
}
```

**Acceptance Criteria**:
- [x] All 6 pattern commands in registry
- [x] Tab autocomplete works for `/bra` -> `/branch`
- [x] Help text accurate

---

### T004 [P] [T001] Add TUI pattern handlers
**File**: `promptchain/cli/tui/app.py`
**Description**: Add elif handlers for each pattern command

```python
async def handle_command(self, command: str):
    # ... existing handlers ...

    elif command.startswith("/branch"):
        await self._handle_pattern_branch(command)
    elif command.startswith("/expand"):
        await self._handle_pattern_expand(command)
    # ... etc for all 6 patterns

async def _handle_pattern_branch(self, command: str):
    """Handle /branch slash command."""
    from promptchain.patterns.executors import execute_branch

    # Parse command: /branch "query" --count=3
    query, opts = self._parse_pattern_command(command, "branch")

    # Show progress
    chat_view = self.query_one("#chat-view", ChatView)
    chat_view.add_message(Message(role="system", content="Generating hypotheses..."))

    # Execute pattern
    result = await execute_branch(query, **opts)

    # Display results
    formatted = self._format_pattern_result("branch", result)
    chat_view.add_message(Message(role="assistant", content=formatted))

    # Add to session history
    if self.session:
        self.session.messages.append(Message(role="user", content=command))
        self.session.messages.append(Message(role="assistant", content=formatted))
```

**Acceptance Criteria**:
- [x] All 6 pattern handlers implemented
- [x] Pattern results display in chat
- [x] Results added to session history
- [x] Progress indicator while executing

---

### T005 [P] [T001] Add pattern command parser
**File**: `promptchain/cli/tui/app.py`
**Description**: Parse pattern command syntax

```python
def _parse_pattern_command(self, command: str, pattern_name: str) -> tuple[str, dict]:
    """Parse pattern command into query and options.

    Examples:
        /branch "What causes X?"
        /branch "What causes X?" --count=5 --mode=local
        /expand "ML optimization" --strategies=semantic,synonym

    Returns:
        (query_string, options_dict)
    """
    import shlex
    parts = shlex.split(command)  # Handles quoted strings
    # ... parsing logic
```

**Acceptance Criteria**:
- [x] Handles quoted queries: `/branch "multi word query"`
- [x] Parses options: `--count=3 --mode=hybrid`
- [x] Returns (query, opts_dict)
- [x] Error message for malformed commands

---

### T006 [P] [T001] Add pattern result formatter
**File**: `promptchain/cli/tui/app.py`
**Description**: Format pattern results for chat display

```python
def _format_pattern_result(self, pattern_name: str, result: dict) -> str:
    """Format pattern result as markdown for chat display.

    Uses Rich-compatible markdown since ChatView renders it.
    """
    if pattern_name == "branch":
        return self._format_branch_result(result)
    elif pattern_name == "expand":
        return self._format_expand_result(result)
    # ...

def _format_branch_result(self, result: dict) -> str:
    """Format branching thoughts result."""
    if not result.get("success"):
        return f"**Error**: {result.get('error', 'Unknown error')}"

    lines = ["**Branching Hypotheses**\n"]
    for i, hyp in enumerate(result.get("hypotheses", []), 1):
        lines.append(f"**{i}.** {hyp.get('approach', 'N/A')}")
        lines.append(f"   _Rationale_: {hyp.get('rationale', 'N/A')}")
    return "\n".join(lines)
```

**Acceptance Criteria**:
- [x] All 6 pattern formatters
- [x] Markdown renders correctly in ChatView
- [x] Error states handled gracefully
- [x] Consistent formatting style

---

## Wave 3: Session Integration (Sequential after Wave 2)

### T007 [S] [T004] Connect patterns to session MessageBus
**File**: `promptchain/cli/tui/app.py`
**Description**: Wire pattern execution to session's 003 infrastructure

```python
async def _handle_pattern_branch(self, command: str):
    from promptchain.patterns.executors import execute_branch

    query, opts = self._parse_pattern_command(command, "branch")

    # Connect to session's MessageBus/Blackboard
    if self.session and hasattr(self.session, 'message_bus'):
        opts['message_bus'] = self.session.message_bus
    if self.session and hasattr(self.session, 'blackboard'):
        opts['blackboard'] = self.session.blackboard

    result = await execute_branch(query, **opts)
    # ...
```

**Acceptance Criteria**:
- [x] Pattern events emit to session MessageBus
- [x] Pattern results written to session Blackboard
- [x] Works gracefully when session has no bus/blackboard

---

### T008 [S] [T007] Add /patterns help command
**File**: `promptchain/cli/tui/app.py`
**Description**: Add help for pattern commands

```python
elif command == "/patterns" or command == "/patterns help":
    help_text = """
**Pattern Commands** (004 Agentic Patterns)

`/branch "query"` - Generate branching hypotheses
  Options: --count=N --mode=local|global|hybrid

`/expand "query"` - Query expansion
  Options: --strategies=semantic,synonym --max=N

`/multihop "query"` - Multi-hop retrieval
  Options: --max-hops=N --mode=local|global|hybrid

`/hybrid "query"` - Hybrid search fusion
  Options: --fusion=rrf|linear|borda --weights=0.5,0.5

`/sharded "query"` - Sharded retrieval
  Options: --shards=shard1,shard2 --aggregation=rrf

`/speculate "context"` - Speculative execution
  Options: --min-confidence=0.7 --prefetch=3

**Note**: Requires hybridrag package installed.
    """
    chat_view.add_message(Message(role="system", content=help_text))
```

**Acceptance Criteria**:
- [x] `/patterns` shows help
- [x] All 6 commands documented with options
- [x] Mentions hybridrag requirement

---

## Summary

| Wave | Tasks | Parallel? | Dependencies |
|------|-------|-----------|--------------|
| 1 | T001, T002 | Sequential | None |
| 2 | T003, T004, T005, T006 | Parallel | T001 |
| 3 | T007, T008 | Sequential | T004 |

**Total**: 8 tasks, ~2-3 hours

## Verification

```bash
# After implementation:
promptchain --session test

> /patterns
[Shows help for all pattern commands]

> /branch "What causes transformer hallucinations?"
[Displays branching hypotheses in chat]

> /expand "ML optimization" --strategies=semantic
[Displays expanded queries in chat]

> /exit
[Session saved with pattern results in history]
```
