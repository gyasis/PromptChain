---
description: "Task list for 008-type-safety-debt-pt2 — Fix High-Error Type Safety Files"
---

# Tasks: 008-type-safety-debt-pt2 — Fix High-Error Type Safety Files

**Input**: Design documents from `specs/008-type-safety-debt-pt2/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, quickstart.md ✓

**Goal**: Reduce mypy errors from 329 → < 220 by fixing all 4 high-error files to 0 errors each.

**No TDD required**: These are annotation-only and control-flow-guard fixes. Each task verifies
by running `python -m mypy <file> --ignore-missing-imports` and confirming error count drops.

## Format: `[ID] [P?] [Story?] Description — File(s)`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[US#]**: User story association

---

## Phase 1: Setup

**Purpose**: Capture baseline and prepare the sprint.

- [x] T001 Run `python -m mypy promptchain/ --ignore-missing-imports 2>&1 | grep "error:" | sort -u > specs/008-type-safety-debt-pt2/mypy-baseline.txt` and commit the output as the baseline snapshot for regression tracking

---

## Phase 2: User Story 1 — state_agent.py (Priority: P1) — 82 errors → 0

**Goal**: Fix `promptchain/utils/strategies/state_agent.py` to 0 mypy errors.

**Independent Test**:
```bash
python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports 2>&1 | grep "error:"
# Expected: (empty)
```

**Why P1**: Largest single-file error source (82 errors); greatest raw error reduction per fix.

### Implementation for User Story 1

- [x] T002 [US1] Fix `max_entries` implicit Optional at line 1014: change `max_entries: int = None` to `max_entries: Optional[int] = None` in `promptchain/utils/strategies/state_agent.py`

- [x] T003 [US1] Resolve `_validate_session_exists` attr-defined errors at lines 1112, 1425, 1563: search the StateAgent class body for the actual method name; add underscore prefix to the definition if missing, or add the method declaration if absent, in `promptchain/utils/strategies/state_agent.py`

- [x] T004 [US1] Fix `Collection[Any] | None` mutable usage errors (lines 1122, 1123, 1244, 1245, 1250, 1251, 1373, 1374, 1536, 1537, 1610, 1611): change all `Collection[Any] | None` field annotations to `Optional[list[Any]]`, then add `if field is not None:` guards before every `.append()`, `.remove()`, `in` operator, and `__delitem__` call in `promptchain/utils/strategies/state_agent.py`

- [x] T005 [US1] Fix `sessions` var-annotated error at line 1367: change `sessions = {}` to `sessions: dict[str, Any] = {}` in `promptchain/utils/strategies/state_agent.py`

- [x] T006 [US1] Fix `None.cursor` attr-defined errors at lines 1430 and 1584: add `if self._db_conn is None: raise RuntimeError("Database connection not initialized")` (or `assert self._db_conn is not None`) immediately before each `.cursor()` call in `promptchain/utils/strategies/state_agent.py`

- [x] T007 [US1] Fix `dict` assigned to `str` variable errors at lines 1631–1634: change the variable declaration from `str` to `dict[str, Any]` (or rename the variable to avoid the type clash), then fix any downstream string-index accesses to use dict-key lookups in `promptchain/utils/strategies/state_agent.py`

- [x] T008 [US1] Verify US1 complete: run `python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports 2>&1 | grep "error:"` and confirm empty output (0 errors)

**Checkpoint**: state_agent.py at 0 errors. User Story 1 is independently complete.

---

## Phase 3: User Story 2 — app.py (Priority: P1) — 63 errors → 0

**Goal**: Fix `promptchain/cli/tui/app.py` to 0 mypy errors.

**Independent Test**:
```bash
python -m mypy promptchain/cli/tui/app.py --ignore-missing-imports 2>&1 | grep "app.py.*error:"
# Expected: (empty)
```

**Why P1**: 63 errors; second-highest. TUI is user-facing — None session access errors are latent runtime crashes.

**Note**: US2 can proceed in parallel with US1 as they touch different files.

### Implementation for User Story 2

- [x] T009 [P] [US2] Fix `Session | None` union-attr errors (lines 1027, 1326, 1328, 1637, 1645, 1664, 1681, 1709, 1720, 3691): at each line that accesses `.agents`, `.history_max_tokens`, `.history_manager`, `.mcp_servers`, or `.messages`, add an early-return guard `if self.session is None: return` (or `assert self.session is not None`) immediately before the access in `promptchain/cli/tui/app.py`

- [x] T010 [P] [US2] Fix `MCPServerConfig` arg-type errors (lines 1613–1616): convert `Sequence[str]` values to scalar `str` for `id` and `type` fields (e.g., `id=str(val[0])`) and to `list[str]` for the `args` field (e.g., `args=list(val)`) before passing to `MCPServerConfig(...)` in `promptchain/cli/tui/app.py`

- [x] T011 [P] [US2] Fix `execute_hybrid` wrong kwarg error at line 2268/2270: rename `fusion_method=` to `fusion=` (the actual parameter name in `promptchain/patterns/executors.py:execute_hybrid`) in `promptchain/cli/tui/app.py`

- [x] T012 [P] [US2] Fix `execute_sharded` wrong kwargs at lines 2363–2366: rename `shard_paths=` to `shards=` and `aggregation_method=` to `aggregation=` (the actual parameter names in `promptchain/patterns/executors.py:execute_sharded`) in `promptchain/cli/tui/app.py`

- [x] T013 [P] [US2] Fix `Widget` missing `start_spinner`/`stop_spinner` attr-defined errors (lines 2598, 2613, 2637, 3071, 3359, 3446, 3534, 3589, 3670): at each `last_item = chat_view.children[-1]` site, replace the `hasattr(last_item, "is_processing")` guard with `isinstance(last_item, MessageItem)` narrowing, importing `MessageItem` from `promptchain.cli.tui.chat_view` at the top of `promptchain/cli/tui/app.py`

- [x] T014 [P] [US2] Fix `PromptChainApp` missing `command_handler` attr-defined at line 1892: declare `command_handler: CommandHandler` (or appropriate type) as a class-level attribute or add it to `__init__` with the correct type in `promptchain/cli/tui/app.py`

- [x] T015 [P] [US2] Fix `list[str]` assigned to `float` variable errors at lines 1978 and 1980: correct the type annotation of the affected variable from `float` to `list[str] | str | float` (or split into separate variables with appropriate types) in `promptchain/cli/tui/app.py`

- [x] T016 [P] [US2] Fix `name message_text not defined` error at line 3146: trace the enclosing scope to find where `message_text` should be assigned; add the assignment before line 3146 or correct the variable reference in `promptchain/cli/tui/app.py`

- [x] T017 [P] [US2] Fix `ToolMetadata | None` union-attr `.function` errors at lines 2893 and 2911: add `if tool_meta is not None:` guard before each `.function` access in `promptchain/cli/tui/app.py`

- [x] T018 [P] [US2] Fix `Argument 2 to next has incompatible type None` error at line 1664: change the `next(iter(...), None)` default to a proper `MCPServerConfig` sentinel or change the variable type to `Optional[MCPServerConfig]` and handle the None downstream in `promptchain/cli/tui/app.py`

- [x] T019 [US2] Verify US2 complete: run `python -m mypy promptchain/cli/tui/app.py --ignore-missing-imports 2>&1 | grep "app.py.*error:"` and confirm empty output (0 errors); also manually test TUI starts without errors with `promptchain --help`

**Checkpoint**: app.py at 0 errors. User Story 2 is independently complete.

---

## Phase 4: User Story 3 — promptchaining.py (Priority: P2) — 32 errors → 0

**Goal**: Fix `promptchain/utils/promptchaining.py` to 0 mypy errors.

**Independent Test**:
```bash
python -m mypy promptchain/utils/promptchaining.py --ignore-missing-imports 2>&1 | grep "promptchaining.py.*error:"
# Expected: (empty)
```

**Why P2**: 32 errors; depends on US1/US2 being complete only for ordering — technically independent.

**Note**: US3 can proceed in parallel with US4 as they touch different files.

### Implementation for User Story 3

- [x] T020 [P] [US3] Fix `ModelProvider`/`ModelManagementConfig` no-redef errors at lines 100 and 102: wrap the fallback class stubs inside the `except ImportError:` block in a `if not TYPE_CHECKING:` guard (or rename the fallback stubs to avoid collision with the real import names) in `promptchain/utils/promptchaining.py`

- [x] T021 [P] [US3] Fix `ClientSession`/`StdioServerParameters`/`stdio_client`/`experimental_mcp_client` None assignment errors at lines 120–123: restructure the MCP import block to use `if TYPE_CHECKING:` for the type-level names and keep only runtime values (flags or sentinel objects) in the except branch; add `# type: ignore[assignment]` only where truly unavoidable in `promptchain/utils/promptchaining.py`

- [x] T022 [P] [US3] Fix conditional function variant mismatched signatures at line 104: unify the two conditional definitions into a single signature (using `Optional` parameters) or add `@overload` declarations so mypy sees a consistent signature in `promptchain/utils/promptchaining.py`

- [x] T023 [P] [US3] Fix implicit Optional defaults at lines 148, 152, 1806, 1807, 1838, 1839: change `chainbreakers: list[Callable[..., Any]] = None` to `Optional[list[Callable[..., Any]]] = None`, `auto_unload_models: bool = None` to `Optional[bool] = None`, and similarly for `params`, `tool_choice`, `tools` in `promptchain/utils/promptchaining.py`

- [x] T024 [P] [US3] Fix union-attr `.objective` on `str | Callable | AgenticStepProcessor` at line 1619: add `if isinstance(instruction, AgenticStepProcessor):` check before the `.objective` access; handle the other union members explicitly in `promptchain/utils/promptchaining.py`

- [x] T025 [P] [US3] Fix `step_outputs` var-annotated error at line 220: change `step_outputs = {}` to `step_outputs: dict[str, Any] = {}` in `promptchain/utils/promptchaining.py`

- [x] T026 [US3] Verify US3 complete: run `python -m mypy promptchain/utils/promptchaining.py --ignore-missing-imports 2>&1 | grep "promptchaining.py.*error:"` and confirm empty output (0 errors)

**Checkpoint**: promptchaining.py at 0 errors. User Story 3 is independently complete.

---

## Phase 5: User Story 4 — executors.py (Priority: P2) — ~31 errors → 0

**Goal**: Fix `promptchain/patterns/executors.py` to 0 mypy errors.

**Independent Test**:
```bash
python -m mypy promptchain/patterns/executors.py --ignore-missing-imports 2>&1 | grep "executors.py.*error:"
# Expected: (empty)
```

**Why P2**: ~31 errors; mostly wrong kwarg names at LightRAG constructor call sites. Fixing makes latent runtime TypeErrors visible.

**Note**: US4 can proceed in parallel with US3.

### Implementation for User Story 4

- [x] T027 [P] [US4] Fix `Blackboard`/`MessageBus` attr-defined import errors at line 21: remove the invalid import `from promptchain.cli.models import Blackboard, MessageBus` (these names do not exist; the forward-reference strings `"Blackboard"` and `"MessageBus"` in function signatures require no live import) in `promptchain/patterns/executors.py`

- [x] T028 [P] [US4] Fix `LightRAGBranchingThoughts` wrong kwargs at line 82 (execute_branch function): update the constructor call to use `lightrag_core=<integration_instance>` and wrap settings in `BranchingConfig(...)` matching the actual `__init__(self, lightrag_core, config=None)` signature; inspect `promptchain/integrations/lightrag/branching.py` for exact config fields in `promptchain/patterns/executors.py`

- [x] T029 [P] [US4] Fix `LightRAGQueryExpander` wrong kwargs at line 162 (execute_expand function): update constructor call to use `lightrag_integration=<integration_instance>` and `config=QueryExpansionConfig(strategies=..., ...)` matching `__init__(self, search_interface=None, lightrag_integration=None, config=None)` signature; inspect `promptchain/integrations/lightrag/query_expansion.py` for QueryExpansionConfig fields in `promptchain/patterns/executors.py`

- [x] T030 [P] [US4] Fix `LightRAGMultiHop` wrong kwargs at lines 202/241 (execute_multihop function): update constructor call to use `search_interface=<interface>` and `config=MultiHopConfig(...)` matching `__init__(self, search_interface, config=None)` signature; inspect `promptchain/integrations/lightrag/multi_hop.py` for MultiHopConfig fields in `promptchain/patterns/executors.py`

- [x] T031 [P] [US4] Fix `execute_hybrid` wrong kwarg `fusion_method` at app.py-facing call in executors.py and/or the execute_hybrid function body: confirm the internal call at line 284 area uses `fusion=` not `fusion_method=`; if the error is in a nested call within execute_hybrid, rename to `fusion=` in `promptchain/patterns/executors.py`

- [x] T032 [P] [US4] Fix `LightRAGHybridSearcher` wrong kwargs at line 323 (execute_hybrid function): update constructor call to use `lightrag_integration=<integration>` and `config=HybridSearchConfig(...)` matching `__init__(self, lightrag_integration, config=None)` signature; inspect `promptchain/integrations/lightrag/hybrid_search.py` for HybridSearchConfig fields in `promptchain/patterns/executors.py`

- [x] T033 [P] [US4] Fix `execute_sharded` wrong kwarg names and `LightRAGShardedRetriever` wrong kwargs at lines 365/404: rename `aggregation_method=` → `aggregation=` and `shard_paths=` → `shards=` in the execute_sharded call site; update `LightRAGShardedRetriever` constructor to use `registry=LightRAGShardRegistry(...)` (instantiate registry first, populate shards, then pass) in `promptchain/patterns/executors.py`

- [x] T034 [P] [US4] Fix `LightRAGSpeculativeExecutor` wrong kwargs at line 486 (execute_speculate function): update constructor call to use `lightrag_core=<integration>` and `config=SpeculativeConfig(...)` matching `__init__(self, lightrag_core, config=None)` signature; inspect `promptchain/integrations/lightrag/speculative.py` for SpeculativeConfig fields in `promptchain/patterns/executors.py`

- [x] T035 [US4] Verify US4 complete: run `python -m mypy promptchain/patterns/executors.py --ignore-missing-imports 2>&1 | grep "executors.py.*error:"` and confirm empty output (0 errors)

**Checkpoint**: executors.py at 0 errors. User Story 4 is independently complete.

---

## Phase 6: Validation & Polish

**Purpose**: Confirm all success criteria, run linting, regression tests, and record results.

- [x] T036 [P] Run full project mypy check: `python -m mypy promptchain/ --ignore-missing-imports 2>&1 | grep "Found.*error" | tail -1` — confirm total error count < 220 (down from 329); record result

- [x] T037 [P] Run `black promptchain/utils/strategies/state_agent.py promptchain/cli/tui/app.py promptchain/utils/promptchaining.py promptchain/patterns/executors.py` — fix any formatting issues introduced by annotation changes in the 4 modified files

- [x] T038 [P] Run `isort promptchain/utils/strategies/state_agent.py promptchain/cli/tui/app.py promptchain/utils/promptchaining.py promptchain/patterns/executors.py` — fix any import ordering issues (e.g., `Optional`, `TYPE_CHECKING` import additions)

- [x] T039 [P] Run `pytest --tb=short -q` — confirm no new test failures vs pre-sprint baseline; annotate any pre-existing failures as pre-existing in `specs/008-type-safety-debt-pt2/test-results.md`

- [x] T040 Write results to `specs/008-type-safety-debt-pt2/test-results.md` using the template from `specs/008-type-safety-debt-pt2/quickstart.md` — record before/after error counts per file, pytest pass/fail counts, and linting status

- [x] T041 Update `CLAUDE.md` "Recent Changes" section to record 008-type-safety-debt-pt2 as complete with final error count

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately (T001)
- **US1 / US2 (Phases 2–3)**: Depend only on T001 — can run in parallel (different files)
- **US3 / US4 (Phases 4–5)**: Depend only on T001 — can run in parallel with each other and with US1/US2 (different files)
- **Validation (Phase 6)**: Depends on ALL fix phases (Phases 2–5) complete

### Parallel Opportunities

```
# Wave 1: Baseline capture (sequential — must commit before fixes)
T001

# Wave 2: All user story fixes run in parallel (independent files)
T002–T008   (state_agent.py — US1)
T009–T019   (app.py — US2)
T020–T026   (promptchaining.py — US3)
T027–T035   (executors.py — US4)

# Wave 3: Validation (after all fix waves committed)
T036 (mypy total)
T037 (black)
T038 (isort)
T039 (pytest)
T040 (test-results.md)
T041 (CLAUDE.md)
```

### Within Each User Story

- Fixes within a story can all run in parallel (they edit non-overlapping line ranges)
- The final verify task (T008, T019, T026, T035) MUST run after all fix tasks in that story
- T037/T038 (formatting) MUST run after all fix tasks but can run concurrently with each other

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete T001: Setup baseline
2. Complete T002–T008: Fix state_agent.py
3. VALIDATE: `python -m mypy promptchain/utils/strategies/state_agent.py --ignore-missing-imports` → 0 errors
4. Commit: `fix(008): state_agent.py — 82 errors resolved`

### Incremental Delivery

1. T001 → Baseline committed
2. T002–T008 → state_agent.py fixed (82 errors gone) → Commit
3. T009–T019 → app.py fixed (63 errors gone) → Commit
4. T020–T026 → promptchaining.py fixed (32 errors gone) → Commit
5. T027–T035 → executors.py fixed (~31 errors gone) → Commit
6. T036–T041 → Validate, lint, record → Commit

### Parallel Team Strategy

With multiple developers or agents:

1. All complete T001 together (or one person/agent, then unblock the rest)
2. Once T001 done:
   - Agent A: US1 (state_agent.py, T002–T008)
   - Agent B: US2 (app.py, T009–T019)
   - Agent C: US3 (promptchaining.py, T020–T026)
   - Agent D: US4 (executors.py, T027–T035)
3. All join for T036–T041 validation

---

## Notes

- **[P] tasks** = different files or non-overlapping line ranges, no blocking dependencies
- **No `# type: ignore` blanket suppressors** — fix root causes only
- **No business logic changes** — annotation-only fixes and `if x is not None:` / `isinstance()` guards
- Commit after each story with message: `fix(008): <filename> — <N> errors resolved`
- Use `Optional[T]` from `typing` (not `T | None` union syntax) for consistency with the rest of the codebase (Python 3.8+ compatibility if needed)
- If a fix in one file accidentally resolves errors in another file (shared type stubs), document in test-results.md
- Do not modify tests — if a test was previously passing incorrectly due to a type error that is now fixed, investigate before changing the test
