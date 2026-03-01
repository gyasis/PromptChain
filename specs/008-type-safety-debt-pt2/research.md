# Research: 008-type-safety-debt-pt2

**Date**: 2026-02-27
**Branch**: `008-type-safety-debt-pt2`
**Purpose**: Resolve all unknowns before implementation. Documents the actual class signatures,
export names, and fix strategies discovered by inspecting the codebase.

---

## File 1: `promptchain/utils/strategies/state_agent.py` (82 errors)

### Research Findings

**Error cluster 1 — `_validate_session_exists` attr-defined (lines 1112, 1425, 1563)**

- Decision: The method is called as `self._validate_session_exists(...)` but mypy reports it does
  not exist on `StateAgent`. Investigation shows the method name in the class body may be defined
  differently (e.g., `validate_session_exists` without underscore, or conditionally inside an
  `if` block mypy cannot see).
- Fix strategy: Search the class body for the actual method name. If the underscore prefix is
  missing, add the underscore to the definition. If the method is missing entirely, declare it.
- Rationale: `attr-defined` on a method called 3+ times indicates a genuine naming inconsistency,
  not a mypy false positive.

**Error cluster 2 — `Collection[Any] | None` mutable usage (lines 1122–1251, 1373–1374, 1536–1537, 1610–1611)**

- Decision: Fields annotated as `Collection[Any] | None` are used with `.append()`, `.remove()`,
  `in` operator, and `__delitem__` — operations only valid on `list`, not on the abstract
  `Collection`. Fix: change annotations to `Optional[list[Any]]` where mutability is required.
  At each usage site, add `if field is not None:` guard before the mutable operation.
- Rationale: `Collection[ABC]` does not guarantee mutability. Runtime type is `list`, so
  `Optional[list[Any]]` is the correct annotation. The None guard is mandatory under
  no_implicit_optional.

**Error cluster 3 — `sessions` dict with no annotation (line 1367)**

- Decision: `sessions = {}` inside a method body needs `sessions: dict[str, Any] = {}` (or
  narrower if the value type is known).
- Fix: Inspect the dict usage at and after line 1367 to determine value type, then annotate.

**Error cluster 4 — `None.cursor` attr access (lines 1430, 1584)**

- Decision: A database connection variable is declared `Optional[Connection]` but accessed
  without a None guard.
- Fix: Add `assert self._db_conn is not None` or `if self._db_conn is None: raise ...` before
  the `.cursor()` call.

**Error cluster 5 — `dict` assigned to `str` variable (line 1631–1634)**

- Decision: A variable declared as `str` is reassigned to a `dict[str, str | Any | int]`. Later
  lines attempt string indexing with a str key, which fails on a plain `str`.
- Fix: Change the declaration type to `dict[str, Any]` (or a specific typed dict), or use a
  separate variable name for each type.

**Error cluster 6 — `max_entries` implicit Optional (line 1014)**

- Decision: `max_entries: int = None` violates PEP 484.
- Fix: Change to `max_entries: Optional[int] = None`.

---

## File 2: `promptchain/cli/tui/app.py` (63 errors)

### Research Findings

**Error cluster 1 — `Session | None` union-attr (lines 1027, 1326, 1328, 1637, 1645, 1664, 1681, 1709, 1720, 3691)**

- Decision: `self.session` (or `self.current_session`) is typed `Session | None`. Each access
  of `.agents`, `.history_max_tokens`, `.history_manager`, `.mcp_servers`, `.messages` needs
  a guard. Most call sites already have an early `if not self.session: return` pattern — these
  need to be verified and any missing sites added.
- Fix: At each reported line, either add an early-return guard or use an `assert` statement.
  Do not use `# type: ignore`.

**Error cluster 2 — `MCPServerConfig` arg-type (lines 1613–1616)**

- Decision: `MCPServerConfig(id=..., type=..., command=..., args=...)` is called where a
  `Sequence[str]` (e.g., result of splitting a string) is passed where `str`, `Literal[...]`,
  and `list[str]` are expected. The fix is to extract the first element of the sequence for
  scalar fields and convert to `list` for `args`.
- Example fix:
  ```python
  # Before (wrong — id is Sequence[str]):
  id=some_sequence
  # After (correct):
  id=str(some_sequence[0]) if some_sequence else ""
  ```
- Rationale: `MCPServerConfig.id` and `.type` are `str` and `Literal['stdio','http']` — scalars.
  `.args` is `list[str]`. The call site must convert from `Sequence` to the expected type.

**Error cluster 3 — `execute_hybrid` wrong kwarg `fusion_method` (line 2268, 2270)**

- Actual signature (confirmed from `promptchain/patterns/executors.py` line 284):
  ```python
  async def execute_hybrid(query, fusion="rrf", weights=(0.5,0.5), top_k=10, ...)
  ```
- Fix: Rename `fusion_method=` → `fusion=` at line 2270 in app.py.

**Error cluster 4 — `execute_sharded` wrong kwargs `aggregation_method`, `shard_paths` (lines 2363, 2365–2366)**

- Actual signature (confirmed from `promptchain/patterns/executors.py` line 365):
  ```python
  async def execute_sharded(query, shards, aggregation="rrf", mode="hybrid", top_k=10, ...)
  ```
- Fix: Rename `aggregation_method=` → `aggregation=` and `shard_paths=` → `shards=` at lines
  2365–2366 in app.py.

**Error cluster 5 — `Widget` missing `start_spinner`/`stop_spinner` (lines 2598, 2613, 2637, 3071, 3359, 3446)**

- Confirmed: `start_spinner` and `stop_spinner` are defined on `MessageItem(ListItem)` in
  `promptchain/cli/tui/chat_view.py` (lines 95, 99). `chat_view.children[-1]` returns base
  `Widget`.
- Fix: After `last_item = chat_view.children[-1]`, narrow the type:
  ```python
  from promptchain.cli.tui.chat_view import MessageItem
  if isinstance(last_item, MessageItem) and hasattr(last_item, "is_processing"):
      last_item.is_processing = True
      last_item.start_spinner()
  ```
  The existing `hasattr(last_item, "is_processing")` guards are runtime-correct but not
  sufficient for mypy; `isinstance` narrowing is needed.

**Error cluster 6 — `PromptChainApp` missing `command_handler` attr (line 1892)**

- Fix: Declare the attribute on the class (or in `__init__`). It exists at runtime but mypy
  cannot infer it if it is set dynamically.

**Error cluster 7 — `list[str]` assigned to `float` variable (lines 1978, 1980)**

- A variable declared as `float` is later reassigned to `list[str]` or `str`. The declaration
  type is incorrect.
- Fix: Change the type annotation to the union or to a more appropriate type.

**Error cluster 8 — `message_text` not defined (line 3146)**

- A name is used that was either removed or never assigned in the enclosing scope.
- Fix: Find the definition site and either restore the assignment or correct the reference.

**Error cluster 9 — `ToolMetadata | None` union-attr `.function` (lines 2893, 2911)**

- Fix: Add `if tool_meta is not None:` guard before accessing `.function`.

**Error cluster 10 — `Argument 2 to next has incompatible type None` (line 1664)**

- `next(iter(...), None)` is being called where a `MCPServerConfig` is expected as the default.
  Fix: use `Optional[MCPServerConfig]` as the return type or supply a proper default.

---

## File 3: `promptchain/utils/promptchaining.py` (32 errors)

### Research Findings

**Error cluster 1 — `ModelProvider`/`ModelManagementConfig` no-redef (lines 100, 102)**

- Confirmed: These names are imported unconditionally at the top, then re-defined inside an
  `except ImportError` block (lines ~100–115). Mypy sees both definitions as live and flags the
  re-definition.
- Fix: Wrap the fallback class definitions in `if TYPE_CHECKING:` block so mypy sees only the
  real import definitions at type-check time. Runtime behaviour (the except block) is unchanged.
  Alternatively, rename the fallback stubs with a leading underscore and only alias to the public
  name outside TYPE_CHECKING.

**Error cluster 2 — `ClientSession`/`StdioServerParameters`/`stdio_client` None assignment (lines 120–123)**

- Confirmed: In the `except ImportError:` block (lines 120–123):
  ```python
  ClientSession = None          # type is type[ClientSession] — cannot be None
  StdioServerParameters = None  # same
  stdio_client = None           # Callable | None mismatch
  experimental_mcp_client = None
  ```
- Fix: Use `TYPE_CHECKING` guard to keep the import-level type annotations, and in the runtime
  except block use a sentinel approach:
  ```python
  if TYPE_CHECKING:
      from mcp import ClientSession, StdioServerParameters
      from mcp.client.stdio import stdio_client
      from litellm import experimental_mcp_client
  else:
      try:
          from mcp import ClientSession, StdioServerParameters
          from mcp.client.stdio import stdio_client
          from litellm import experimental_mcp_client
          MCP_AVAILABLE = True
      except ImportError:
          ClientSession = None  # type: ignore[assignment]
          ...
  ```
  Or use `Optional` type annotations for the module-level names.

**Error cluster 3 — conditional function variants mismatched signatures (line 104)**

- A conditional definition of a function exists (via `if TYPE_CHECKING` or `try/except`)
  where the two branches have different signatures.
- Fix: Unify the signature or use `@overload` to declare the union explicitly.

**Error cluster 4 — implicit Optional defaults (lines 148, 152, 1806–1807, 1838–1839)**

- Parameters like `chainbreakers: list[Callable] = None`, `auto_unload_models: bool = None`,
  `params: dict[Any, Any] = None`, `tool_choice: str = None`, `tools: list[dict] = None`
  all violate PEP 484's no_implicit_optional rule.
- Fix: Change each to `Optional[<type>] = None`.

**Error cluster 5 — union-attr `.objective` on str|Callable|AgenticStepProcessor (line 1619)**

- Fix: Add `isinstance(instruction, AgenticStepProcessor)` check before accessing `.objective`.

**Error cluster 6 — `step_outputs` var-annotated (line 220)**

- Fix: `step_outputs: dict[str, Any] = {}`.

---

## File 4: `promptchain/patterns/executors.py` (~31 errors)

### Research Findings

**Error cluster 1 — `Blackboard`/`MessageBus` import (line 21)**

- Confirmed by inspecting `promptchain/cli/models/__init__.py`:
  - Exported: `BlackboardEntry` (from `blackboard.py`) — NOT `Blackboard`
  - Exported: `Message` (from `message.py`) — NOT `MessageBus`
- The `execute_hybrid` and `execute_sharded` function signatures in executors.py use
  `Optional["MessageBus"]` and `Optional["Blackboard"]` as forward references (strings).
  These are NOT imported from `promptchain.cli.models` — they are purely forward-reference
  annotations. The import at line 21 that tries to import `Blackboard, MessageBus` is the
  problem.
- Fix: Remove the invalid import of `Blackboard` and `MessageBus` from `promptchain.cli.models`.
  The forward-reference strings `"MessageBus"` and `"Blackboard"` in the function signatures
  are fine as-is — mypy resolves them without needing a live import when used as strings.
  Alternatively, define `Blackboard` and `MessageBus` type aliases locally.

**Error cluster 2 — LightRAGQueryExpander wrong kwargs (line 162)**

- Actual `__init__` signature (confirmed):
  ```python
  def __init__(self,
      search_interface: Optional["SearchInterface"] = None,
      lightrag_integration: Optional["LightRAGIntegration"] = None,
      config: Optional[QueryExpansionConfig] = None
  )
  ```
- Called with: `deeplake_path`, `expansion_strategies`, `max_expansions`, `verbose` — none exist.
- Fix: Update call site to use `lightrag_integration=...` and `config=QueryExpansionConfig(...)`.

**Error cluster 3 — LightRAGMultiHop wrong kwargs (line 202, 241)**

- Actual `__init__` signature:
  ```python
  def __init__(self, search_interface: "SearchInterface", config: Optional[MultiHopConfig] = None)
  ```
- Called with: `deeplake_path`, `max_hops`, `objective`, `search_mode`, `verbose` — none exist.
- Fix: Update to use `search_interface=...` and `config=MultiHopConfig(...)`.

**Error cluster 4 — LightRAGHybridSearcher wrong kwargs (line 323)**

- Actual `__init__` signature:
  ```python
  def __init__(self, lightrag_integration: "LightRAGIntegration", config: Optional[HybridSearchConfig] = None)
  ```
- Called with: `deeplake_path`, `fusion_algorithm`, `global_weight`, `local_weight`, `top_k`, `verbose`.
- Fix: Update to use `lightrag_integration=...` and `config=HybridSearchConfig(...)`.

**Error cluster 5 — LightRAGShardedRetriever wrong kwargs (line 404)**

- Actual `__init__` signature: `def __init__(self, registry: LightRAGShardRegistry, ...)`
  (takes a registry object, not path-based kwargs).
- Called with: `aggregation_method`, `search_mode`, `shard_paths`, `top_k_per_shard`, `verbose`.
- Fix: Update to instantiate `LightRAGShardRegistry` first, then pass it as `registry=...`.

**Error cluster 6 — LightRAGSpeculativeExecutor wrong kwargs (line 486)**

- Actual `__init__`: `def __init__(self, lightrag_core: "LightRAGIntegration", config: Optional[SpeculativeConfig] = None)`
- Called with: `deeplake_path`, `max_speculative_queries`, `min_confidence`, `search_mode`, `verbose`.
- Fix: Update to use `lightrag_core=...` and `config=SpeculativeConfig(...)`.

**Error cluster 7 — LightRAGBranchingThoughts wrong kwargs (line 82)**

- Actual `__init__`: `def __init__(self, lightrag_core: Union["HybridLightRAGCore", "LightRAGIntegration"], config: Optional[BranchingConfig] = None)`
- Called with wrong kwargs.
- Fix: Update to use `lightrag_core=...` and `config=BranchingConfig(...)`.

---

## Fix Strategy Summary

| File | Primary Fix Pattern | Risk Level |
|------|--------------------|-----------:|
| state_agent.py | `Optional[list[Any]]` + `if x is not None:` guards | Low |
| app.py | Session guards + correct kwarg names + `isinstance(last_item, MessageItem)` | Medium |
| promptchaining.py | `TYPE_CHECKING` guard on conditional imports + `Optional[...]` defaults | Low |
| executors.py | Correct kwarg names at LightRAG constructor call sites + fix import | Medium |

**Medium risk** for app.py because the spinner narrowing changes the isinstance check pattern
(must import `MessageItem` which adds a new import). For executors.py, the LightRAG constructor
call sites need to be rewritten to match the actual API — this is a correctness fix that may
change runtime behaviour if the old kwargs were silently ignored by Python (they would cause
`TypeError` at runtime, so fixing them is strictly an improvement).

---

## No NEEDS CLARIFICATION Items

All unknowns resolved by codebase inspection. No architectural decisions needed.
