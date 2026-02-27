# Data Model: 008-type-safety-debt-pt2

**Date**: 2026-02-27
**Branch**: `008-type-safety-debt-pt2`
**Purpose**: Catalogue of type annotation patterns being fixed. No new data structures are
introduced — this documents the correct type annotations that will replace the incorrect ones.

---

## Type Annotation Patterns

### Pattern 1: Implicit Optional Fix

**Before (broken)**:
```python
def method(self, max_entries: int = None): ...
```

**After (correct)**:
```python
def method(self, max_entries: Optional[int] = None): ...
```

**Files affected**: state_agent.py line 1014, promptchaining.py lines 148, 152, 1806, 1807,
1838, 1839.

---

### Pattern 2: Collection → Optional[list] for Mutable Fields

**Before (broken)**:
```python
self.session_items: Collection[Any] | None = None
# later:
self.session_items.append(x)  # Error: Collection has no .append
```

**After (correct)**:
```python
self.session_items: Optional[list[Any]] = None
# later:
if self.session_items is not None:
    self.session_items.append(x)
```

**Files affected**: state_agent.py — all `Collection[Any] | None` fields used mutably.

---

### Pattern 3: TYPE_CHECKING Guard for Conditional Imports

**Before (broken)**:
```python
try:
    from mcp import ClientSession, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None        # Error: assigning None to type[ClientSession]
    StdioServerParameters = None
```

**After (correct)**:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

MCP_AVAILABLE = False
_ClientSession: Optional[type] = None
_StdioServerParameters: Optional[type] = None
try:
    from mcp import ClientSession as _CS, StdioServerParameters as _SP
    _ClientSession = _CS
    _StdioServerParameters = _SP
    MCP_AVAILABLE = True
except ImportError:
    pass
```

**Files affected**: promptchaining.py lines 119–123.

---

### Pattern 4: Session | None Narrowing

**Before (broken)**:
```python
def _some_method(self):
    agents = self.session.agents  # Error: Session | None has no .agents
```

**After (correct)**:
```python
def _some_method(self):
    if self.session is None:
        return
    agents = self.session.agents  # OK: narrowed to Session
```

**Files affected**: app.py — all Session|None access sites.

---

### Pattern 5: Widget Subtype Narrowing for Spinner

**Before (broken)**:
```python
last_item = chat_view.children[-1]  # type: Widget
last_item.start_spinner()  # Error: Widget has no start_spinner
```

**After (correct)**:
```python
from promptchain.cli.tui.chat_view import MessageItem
last_item = chat_view.children[-1]
if isinstance(last_item, MessageItem):
    last_item.start_spinner()  # OK: narrowed to MessageItem
```

**Files affected**: app.py lines 2598, 2613, 2637, 3071, 3359, 3446.

---

### Pattern 6: Wrong Kwarg Names at Call Site

**Before (broken — app.py)**:
```python
result = await execute_hybrid(
    query=query,
    fusion_method=options.get("fusion", "rrf"),  # Error: no such param
)
result = await execute_sharded(
    query=query,
    shard_paths=shards,                           # Error: no such param
    aggregation_method=options.get("aggregation", "rrf"),  # Error: no such param
)
```

**After (correct)**:
```python
result = await execute_hybrid(
    query=query,
    fusion=options.get("fusion", "rrf"),          # correct param name
)
result = await execute_sharded(
    query=query,
    shards=shards,                                # correct param name
    aggregation=options.get("aggregation", "rrf"), # correct param name
)
```

**Files affected**: app.py lines 2268–2270, 2363–2366. Also affects executors.py.

---

### Pattern 7: LightRAG Constructor Call Correction

**Before (broken — executors.py)**:
```python
expander = LightRAGQueryExpander(
    deeplake_path=path,
    expansion_strategies=strategies,
    max_expansions=5,
    verbose=True
)
```

**After (correct)**:
```python
from promptchain.integrations.lightrag.query_expansion import QueryExpansionConfig
expander = LightRAGQueryExpander(
    lightrag_integration=integration,
    config=QueryExpansionConfig(
        strategies=strategies,
        max_expansions_per_strategy=5
    )
)
```

**Files affected**: executors.py lines 82, 162, 202/241, 323, 404, 486.

---

### Pattern 8: Var-Annotated Fix

**Before (broken)**:
```python
sessions = {}       # Error: Need type annotation for "sessions"
step_outputs = {}   # Error: Need type annotation for "step_outputs"
```

**After (correct)**:
```python
sessions: dict[str, Any] = {}
step_outputs: dict[str, Any] = {}
```

**Files affected**: state_agent.py line 1367; promptchaining.py line 220.

---

### Pattern 9: isinstance Narrowing Before Union Member Access

**Before (broken)**:
```python
instruction: str | Callable | AgenticStepProcessor
objective = instruction.objective  # Error: str and Callable have no .objective
```

**After (correct)**:
```python
if isinstance(instruction, AgenticStepProcessor):
    objective = instruction.objective  # OK
```

**Files affected**: promptchaining.py line 1619.

---

## No New Entities

This sprint introduces no new data classes, no new database tables, and no new API models.
All type annotations listed above are corrections to existing declarations.
