# Feature Specification: 008-type-safety-debt-pt2 — Fix High-Error Type Safety Files

**Feature Branch**: `008-type-safety-debt-pt2`
**Created**: 2026-02-27
**Status**: Draft
**Input**: Second sprint of type safety debt reduction targeting 4 high-error files

---

## Problem

Branch 007-type-safety-debt eliminated type errors across 20+ smaller files, reducing the project
from 557 to 329 mypy errors. The 4 remaining files each carry a disproportionately high error
load and were deferred from sprint 1 because they require deeper investigation:

| File | Errors | Primary Error Classes |
|------|--------|-----------------------|
| `promptchain/utils/strategies/state_agent.py` | 82 | `union-attr`, `operator`, `attr-defined`, `var-annotated`, `assignment` |
| `promptchain/cli/tui/app.py` | 63 | `union-attr` (Session\|None), `arg-type`, `attr-defined`, `call-arg`, `name-defined` |
| `promptchain/utils/promptchaining.py` | 32 | `no-redef`, `assignment`, `misc`, `union-attr`, `var-annotated` |
| `promptchain/patterns/executors.py` | ~31 | `attr-defined`, `call-arg` (wrong kwarg names to LightRAG constructors) |

**Total project errors at sprint start**: 329 across 45 files (checked 1 source file per run).
**Sprint target**: reduce to < 220 (33% further reduction), with all 4 files reaching 0 errors.

---

## User Scenarios & Testing

### User Story 1 — state_agent.py Reaches 0 Errors (Priority: P1)

A developer running `python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports`
receives zero errors. The StateAgent class properly annotates all session-related collection
fields so that mypy can verify None-safety and mutability contracts without false positives.

**Why this priority**: Largest single-file error source (82 errors). Fixing it delivers the
greatest raw error reduction and unblocks IDE type-checking for the multi-step planning agent,
which is a core framework component.

**Independent Test**: Run mypy on state_agent.py alone and confirm 0 errors.
Runtime behaviour is unchanged — only annotations and None-guard `if x is not None:` statements
are added.

**Acceptance Scenarios**:

1. **Given** state_agent.py with 82 type errors, **When** the Collection[Any]|None fields are
   narrowed to `Optional[list[Any]]` and guarded before append/remove/in operations, **Then**
   mypy reports 0 errors for that file.
2. **Given** the `_validate_session_exists` method referenced but not found by mypy, **When** it
   is confirmed present (or the typo corrected), **Then** the `attr-defined` error disappears.
3. **Given** the `sessions` dict with no type annotation, **When** `sessions: dict[str, Any] = {}`
   annotation is added, **Then** the `var-annotated` error resolves.
4. **Given** `None.cursor` attribute access, **When** the database connection variable is narrowed
   with a None guard, **Then** the `attr-defined` error resolves.
5. **Given** a variable annotated as `str` then assigned a `dict`, **When** the annotation is
   corrected to `dict[str, Any]` or `str | dict[str, Any]`, **Then** the `assignment` error
   resolves.
6. **Given** `max_entries` parameter with `None` default but `int` annotation, **When** changed
   to `Optional[int] = None`, **Then** the implicit Optional error resolves.

---

### User Story 2 — app.py Reaches 0 Errors (Priority: P1)

A developer running `python -m mypy promptchain/cli/tui/app.py --ignore-missing-imports`
receives zero errors. The TUI application's `Session | None` accesses are all guarded, API call
sites use correct keyword argument names, and the spinner methods are resolved.

**Why this priority**: 63 errors, second-highest count. The TUI is user-facing and developer-facing;
type safety here prevents runtime AttributeError crashes on None session access.

**Independent Test**: Run mypy on app.py alone and confirm 0 errors. All existing TUI
functionality is preserved — no behavioural changes.

**Acceptance Scenarios**:

1. **Given** `self.current_session` accessed as `Session | None` without guards,
   **When** `assert self.current_session is not None` or early-return guards are added before
   each `.agents`, `.history_max_tokens`, `.history_manager`, `.mcp_servers`, `.messages` access,
   **Then** the union-attr errors resolve.
2. **Given** `MCPServerConfig(id=..., type=..., command=..., args=...)` called with
   `Sequence[str]` values, **When** the values are cast/converted to `str` and `list[str]`,
   **Then** the `arg-type` errors resolve.
3. **Given** `execute_hybrid` called with unknown kwarg `fusion_method`, **When** the kwarg is
   renamed to the actual parameter name (or the call is corrected to match the executor API),
   **Then** the `call-arg` error resolves.
4. **Given** `execute_sharded` called with `aggregation_method` and `shard_paths` (wrong names),
   **When** renamed to the correct parameter names (`aggregation` / correct path param),
   **Then** the `call-arg` errors resolve.
5. **Given** `Widget.start_spinner()` / `Widget.stop_spinner()` called on a plain `Widget` base,
   **When** the widget variables are typed to the concrete subclass that has these methods (or
   the calls are guarded with `hasattr` and cast), **Then** the `attr-defined` errors resolve.
6. **Given** `PromptChainApp` missing `command_handler` attribute, **When** the attribute is
   declared on the class (it exists at runtime but mypy cannot see it), **Then** the
   `attr-defined` error resolves.
7. **Given** name `message_text` not defined at line 3146, **When** the variable is defined or
   the reference is corrected, **Then** the `name-defined` error resolves.

---

### User Story 3 — promptchaining.py Reaches 0 Errors (Priority: P2)

A developer running `python -m mypy promptchain/utils/promptchaining.py --ignore-missing-imports`
receives zero errors. The core PromptChain class handles conditional MCP imports safely and
documents all implicit-Optional parameters.

**Why this priority**: 32 errors. Core class that all agents depend on. The `no-redef` conditional
import pattern is a common mypy pitfall and its fix will serve as a pattern for other files.

**Independent Test**: Run mypy on promptchaining.py alone and confirm 0 errors.

**Acceptance Scenarios**:

1. **Given** `ModelProvider` and `ModelManagementConfig` defined twice (once by import, once in
   conditional block), **When** the conditional block uses `TYPE_CHECKING` guard or the fallback
   is restructured to avoid the redefinition, **Then** the `no-redef` errors resolve.
2. **Given** `ClientSession`, `StdioServerParameters`, `stdio_client` assigned `None` as fallback
   in the `except ImportError` branch, **When** a `TYPE_CHECKING` guard wraps the type-level
   assignments and `None` is used only for the runtime fallback, **Then** the assignment errors
   resolve.
3. **Given** conditional function variants with mismatched signatures, **When** overload
   decorators or a unified signature with `Optional` parameters are used, **Then** the `misc`
   error resolves.
4. **Given** parameters like `chainbreakers`, `auto_unload_models` with `None` defaults but
   non-Optional annotations, **When** changed to `Optional[list[...]] = None` and
   `Optional[bool] = None`, **Then** the implicit Optional errors resolve.
5. **Given** union-attr `.objective` on `str | Callable | AgenticStepProcessor`, **When** the
   code narrows to `AgenticStepProcessor` with `isinstance()` before accessing `.objective`,
   **Then** the union-attr error resolves.
6. **Given** `step_outputs = {}` with no annotation, **When** annotated as
   `step_outputs: dict[str, Any] = {}`, **Then** the `var-annotated` error resolves.

---

### User Story 4 — executors.py Reaches 0 Errors (Priority: P2)

A developer running `python -m mypy promptchain/patterns/executors.py --ignore-missing-imports`
receives zero errors. All LightRAG class constructors are called with the correct parameter names,
and the `Blackboard` / `MessageBus` imports are resolved.

**Why this priority**: ~31 errors, almost entirely wrong-kwarg-name call sites. These are latent
bugs — if the LightRAG API changed, calls are silently broken. Fixing them improves correctness,
not just annotation coverage.

**Independent Test**: Run mypy on executors.py alone and confirm 0 errors.

**Acceptance Scenarios**:

1. **Given** `from promptchain.cli.models import Blackboard, MessageBus` where neither attribute
   exists (maybe `BlackboardEntry` and `Message`), **When** the import is corrected to use the
   actual exported names, **Then** the `attr-defined` errors resolve.
2. **Given** `LightRAGQueryExpander(deeplake_path=..., expansion_strategies=..., max_expansions=..., verbose=...)`
   with kwargs not accepted by that class, **When** the constructor call is updated to use the
   actual `__init__` parameters, **Then** the `call-arg` errors resolve.
3. **Given** `LightRAGMultiHop`, `LightRAGHybridSearcher`, `LightRAGShardedRetriever`,
   `LightRAGSpeculativeExecutor`, `LightRAGBranchingThoughts` all called with wrong kwargs,
   **When** each constructor call is corrected to match the actual class signatures,
   **Then** all `call-arg` errors resolve.
4. **Given** `execute_hybrid` called with `fusion_method` kwarg (wrong name) and `execute_sharded`
   called with `aggregation_method` / `shard_paths` (wrong names), **When** renamed to the actual
   parameter names accepted by those functions, **Then** the `call-arg` errors resolve.

---

### Edge Cases

- A fix to state_agent.py must not change the runtime type of session collections — they must
  remain mutable lists at runtime even after being annotated as `Optional[list[Any]]`.
- The app.py `Session | None` guards must preserve existing control flow — early returns must not
  silently skip code that should execute when session is None (raise instead where appropriate).
- The executors.py kwarg fixes may reveal that the LightRAG classes themselves have changed API;
  if so, the fix is to match the current class signatures, not to restore the old API.
- Conditional import patterns in promptchaining.py must not break the `TYPE_CHECKING` imports
  used downstream by other modules that import from promptchaining.

---

## Requirements

### Functional Requirements

- **FR-001**: All 4 target files MUST reach 0 mypy errors when checked individually with
  `--ignore-missing-imports`.
- **FR-002**: All fixes MUST be annotation-only or control-flow guards (`if x is not None:`,
  `isinstance()`, `assert`) — no changes to business logic or algorithm.
- **FR-003**: The `pytest` suite MUST pass with no new failures introduced by this sprint
  (pre-existing failures from before this branch are acceptable and documented).
- **FR-004**: All modified files MUST pass `black` formatting and `isort` import ordering checks.
- **FR-005**: Each LightRAG constructor call in executors.py MUST be updated to use the actual
  parameter names accepted by the current class definitions.
- **FR-006**: Session|None accesses in app.py MUST be protected by explicit None guards rather
  than blanket `# type: ignore` suppressors.
- **FR-007**: Conditional MCP import fallbacks in promptchaining.py MUST be restructured to use
  `TYPE_CHECKING` guard so that runtime `None` assignments do not conflict with type-level
  declarations.
- **FR-008**: A mypy baseline snapshot MUST be captured at the start of the sprint and a
  test-results record committed at the end for regression tracking.

### Key Entities

- **StateAgent**: Multi-step planning agent whose session-related fields are typed as
  `Collection[Any] | None` but used as mutable lists.
- **PromptChainApp (TUI)**: Textual application class with a `current_session: Session | None`
  field accessed unsafely in many methods.
- **PromptChain**: Core orchestration class with conditional MCP imports and mixed-type instruction
  parameters.
- **Executors**: Pattern-execution wrappers that instantiate LightRAG retrieval classes with
  arguments that have drifted from the current class signatures.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: `python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports`
  reports **0 errors** (down from 82).
- **SC-002**: `python -m mypy promptchain/cli/tui/app.py --ignore-missing-imports`
  reports **0 errors** (down from 63).
- **SC-003**: `python -m mypy promptchain/utils/promptchaining.py --ignore-missing-imports`
  reports **0 errors** (down from 32).
- **SC-004**: `python -m mypy promptchain/patterns/executors.py --ignore-missing-imports`
  reports **0 errors** (down from ~31).
- **SC-005**: `python -m mypy promptchain/ --ignore-missing-imports` reports **fewer than 220
  errors** total (down from 329 — a 33%+ reduction).
- **SC-006**: `pytest` exits with the same number of passing tests as the pre-sprint baseline;
  zero new test failures introduced.
- **SC-007**: `black --check . && isort --check .` exits cleanly on all modified files.
- **SC-008**: No `# type: ignore` blanket suppressors introduced; every error is fixed at root
  cause.

---

## Assumptions

- The LightRAG class constructors (`LightRAGQueryExpander`, `LightRAGMultiHop`, etc.) have stable
  `__init__` signatures accessible to mypy in the current codebase; the executor call sites will
  be updated to match whatever signatures currently exist.
- `_validate_session_exists` in state_agent.py is a real method on `StateAgent` that mypy cannot
  see due to a conditional definition or typo — investigation will determine whether it is a
  naming error or a missing declaration.
- The spinner methods (`start_spinner`, `stop_spinner`) exist on a specific Textual widget
  subclass, not on the base `Widget` type; the fix is to tighten the type annotation of the
  widget variable.
- Fixing the 4 target files will not require changes to any file already fixed in sprint 007.
- The `pytest` baseline is the state of the test suite on the `007-type-safety-debt` branch
  before merging into main.
