# 004a: TUI Pattern Slash Commands

## Problem Statement

Currently, to use agentic patterns (branching, query expansion, etc.), users must:
1. Exit the TUI (`/exit`)
2. Run CLI command (`promptchain patterns branch "query"`)
3. Restart TUI (`promptchain --session name`)

This breaks flow and loses session context.

## Solution

Add pattern slash commands inside the TUI:
- `/branch "query"` - Branching thoughts
- `/expand "query"` - Query expansion
- `/multihop "query"` - Multi-hop retrieval
- `/hybrid "query"` - Hybrid search fusion
- `/sharded "query"` - Sharded retrieval
- `/speculate "query"` - Speculative execution

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Pattern Core (NEW)                     │
│         promptchain/patterns/executors.py           │
│                                                     │
│   async def execute_branch(query, **opts) -> dict  │
│   async def execute_expand(query, **opts) -> dict  │
│   async def execute_multihop(query, **opts) -> dict│
│   ...                                               │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────────┐      ┌───────────────────┐
│  Click Commands   │      │   TUI Handlers    │
│  patterns.py      │      │   app.py          │
│                   │      │                   │
│ @click.command()  │      │ /branch "query"   │
│ def branch():     │      │ /expand "query"   │
│   execute_branch()│      │   execute_branch()│
└───────────────────┘      └───────────────────┘
```

## Benefits

1. **Seamless UX**: Switch between chat and patterns without leaving session
2. **Context preserved**: Pattern results added to conversation history
3. **003 Integration**: Patterns connect to session's MessageBus/Blackboard
4. **Code reuse**: Single execution logic for both CLI and TUI

## Non-Goals

- Not adding new patterns (004 already complete)
- Not changing pattern behavior
- Not modifying Click CLI interface (remains as fallback)
