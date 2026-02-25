---
description: "Task list for 007-type-safety-debt"
---

# Tasks: Fix Pre-Existing Type Safety Debt

**Input**: Mypy error report from `python -m mypy promptchain/ --ignore-missing-imports`
**Prerequisites**: spec.md ✓

**Goal**: Reduce mypy errors from 312+ to < 10 by fixing pre-existing type
annotations across 20+ files without changing runtime behaviour.

**No TDD required**: These are annotation-only fixes. Each task verifies by
running mypy on the touched file(s) and confirming error count drops.

## Format: `[ID] [P?] Description — File(s)`

- **[P]**: Can run in parallel (different files, no blocking dependencies)

---

## Phase 1: Setup

**Purpose**: Branch setup and baseline error snapshot.

- [ ] T001 Create feature branch `007-type-safety-debt` from `main`
- [ ] T002 [P] Run `python -m mypy promptchain/ --ignore-missing-imports 2>&1 | grep "error:" | sort -u > /tmp/mypy-baseline.txt` and commit the output as `specs/007-type-safety-debt/mypy-baseline.txt` for regression tracking

---

## Phase 2: High-Impact Files (15+ errors each)

**Purpose**: Eliminate the largest error clusters first for maximum error reduction per PR.

- [ ] T003 Fix `promptchain/cli/session_manager.py` (15 errors) — add `TYPE_CHECKING` guard to import `MCPServerConfig`, `WorkflowState`, `WorkflowStep`, `Task`, `BlackboardEntry`, `MultiAgentWorkflow` from their respective modules; fix `append` `arg-type` (int appended to `list[str]`) at lines 1784 and 2189

- [ ] T004 Fix `promptchain/cli/command_handler.py` (13 errors) — change bare `{}` / `[]` vars to typed annotations (`dict[str, list[Any]]`, `dict[str, Any]`); replace `Collection[str]` mutable usage with `list[str]` where items are appended/indexed; add `if self.agent_config.history is not None:` guards at lines 317 and 323; add `# type: ignore[import-untyped]` comment on yaml import

---

## Phase 3: Medium-Impact Files (7–9 errors each)

**Purpose**: Second-largest clusters — can all run in parallel (different files).

- [ ] T005 [P] Fix `promptchain/cli/config/yaml_translator.py` (9 errors) — add `types-PyYAML` stub note or `# type: ignore[import-untyped]`; fix list type declarations (`list[str | AgenticStepProcessor]`); change `OrchestrationConfig(execution_mode=...)` call to cast/validate the string to `Literal['router', 'pipeline', 'round-robin', 'broadcast']`; fix `list[Any]` vs `dict[str, Any]` assignment at line 436

- [ ] T006 [P] Fix `promptchain/cli/tools/sandbox/docker_sandbox.py` (8 errors) — filter `None` from command lists before passing to `subprocess.run` (e.g., `[x for x in cmd if x is not None]`); fix missing `project_root` positional argument to `SafetyValidator` constructor at line 560; rename `validate_code` → `validate_command` at line 561 (matches actual `SafetyValidator` API)

- [ ] T007 [P] Fix `promptchain/cli/tools/safety.py` (8 errors) — replace `any` (builtin function used as type) with `Any` from `typing` at lines 398 and 583; change `result: str` declarations at lines 452, 457, 466, 468 to `result: Path | list[str] | int | str` or use separate typed variables; add `Path | None` cast before passing to `_validate_delete_operation` / `_validate_write_operation`

- [ ] T008 [P] Fix `promptchain/tools/terminal/terminal_tool.py` (7 errors) — add `Optional[str]` and `Optional[dict[str, str]]` to `working_directory` and `environment_variables` parameter annotations (implicit Optional fix); add `assert result is not None` / early-return guards before accessing `.return_code` at lines 642, 684, 710; guard `session.name` with `if session is not None:` at line 823

- [ ] T009 [P] Fix `promptchain/cli/communication/handlers.py` (7 errors) — move `_handlers: dict[str, list[Callable[..., Any]]]` declaration to the class body (not inside a method), initialised as a proper class variable or instance variable in `__init__` to resolve `has-type` and non-self attribute errors

---

## Phase 4: Lower-Impact Files (1–5 errors each)

**Purpose**: Sweep remaining error clusters — all parallel (independent files).

- [ ] T010 [P] Fix `promptchain/tools/terminal/session_manager.py` (5 errors) — add `Optional[str]` / `Optional[dict[str, str]]` to `working_directory` and `environment_variables` parameters at lines 51–52 and 464–465; annotate `command_history: list[dict[str, Any]] = []` at line 57

- [ ] T011 [P] Fix `promptchain/integrations/lightrag/events.py` (4 errors) — change `details: dict[str, Any] = None` to `details: Optional[dict[str, Any]] = None` at line 167 and similarly for `metadata` at line 184; change return type of the `Sequence[str]` method to `list[str]` and add `list(...)` cast; annotate `all_events: list[...]` at line 264

- [ ] T012 [P] Fix `promptchain/cli/tools/filesystem_tools.py` (4 errors) — annotate `all_paths: list[Path] = []` and `results: list[Any] = []`; add `if metadata is not None:` guards before accessing `metadata.function` at lines 515–516

- [ ] T013 [P] Fix `promptchain/cli/tools/executor.py` (3 errors) — add `if self.safety_validator is not None:` guards before `.validate_path()` and `.validate_command()` calls at lines 484, 494, 496

- [ ] T014 [P] Fix remaining 1–2 error files in a single pass:
  - `promptchain/utils/model_management.py`: add `Optional[str]` to `base_url` param; fix `None` assignment to `dict[ModelProvider, ...]` var
  - `promptchain/cli/tui/token_bar.py`: add `Optional[int]` to `api_prompt_tokens` / `api_completion_tokens` params
  - `promptchain/cli/security/input_sanitizer.py`: separate `result` into typed vars; fix return type to `list[str | dict[Any, Any]]`
  - `promptchain/cli/prompt_manager.py`: annotate `strategies: list[Any]` and `templates: list[Any]`
  - `promptchain/utils/execution_events.py`: fix `int` assigned to `str` var at line 122
  - `promptchain/utils/dry_run.py`: annotate `accuracy_history: list[float]`
  - `promptchain/utils/prompt_loader.py`: annotate `organized_prompts: dict[str, Any]`
  - `promptchain/cli/__init__.py`: resolve `registration` name conflict (rename import or use alias)
  - `promptchain/cli/main.py`: annotate `results: list[Any]`
  - `promptchain/cli/models/agent_config.py`: cast `object` to `dict[str, Any]` before passing to `HistoryConfig.from_dict()`
  - `promptchain/utils/ollama_model_manager.py`: fix `AsyncClient(**kwargs)` call by unpacking only valid params or using typed kwargs

---

## Phase 5: Validation & Polish

**Purpose**: Confirm success criteria, linting, and documentation.

- [ ] T015 [P] Run `python -m mypy promptchain/ --ignore-missing-imports 2>&1 | grep "error:" | sort -u | wc -l` — confirm count < 10; record result in `specs/007-type-safety-debt/test-results.md`

- [ ] T016 [P] Run `black . && isort . && flake8 promptchain/` — fix any linting issues introduced by annotation changes in modified files

- [ ] T017 [P] Run `pytest` — confirm no regressions vs baseline; annotate any pre-existing failures as pre-existing in `specs/007-type-safety-debt/test-results.md`

- [ ] T018 Update `CLAUDE.md` "Recent Changes" section to record 007-type-safety-debt as complete

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **High-Impact (Phase 2)**: Depends on Phase 1 — T003 and T004 are sequential to each other only if the same agent handles both; otherwise parallel (different files)
- **Medium-Impact (Phase 3)**: Depends on Phase 1 only — all parallel (T005–T009 touch different files)
- **Lower-Impact (Phase 4)**: Depends on Phase 1 only — all parallel (T010–T014 touch different files)
- **Validation (Phase 5)**: Depends on ALL previous phases complete

### Parallel Opportunities

```
# Wave 1: Setup (sequential — branch then baseline)
T001 → T002

# Wave 2: All fix tasks run in parallel (independent files)
T003 (session_manager.py)
T004 (command_handler.py)
T005 (yaml_translator.py)
T006 (docker_sandbox.py)
T007 (safety.py)
T008 (terminal_tool.py)
T009 (handlers.py)
T010 (tools/terminal/session_manager.py)
T011 (lightrag/events.py)
T012 (filesystem_tools.py)
T013 (executor.py)
T014 (remaining 1–2 error files)

# Wave 3: Validate (after all fixes committed)
T015 (mypy check)
T016 (linting)
T017 (pytest)
T018 (docs)
```

---

## Notes

- Fixes must not change runtime behaviour — type annotations and `if x is not None:` guards only
- `types-PyYAML` install resolves all `import-untyped` yaml errors across multiple files; add to `requirements.txt` dev dependencies as part of T014
- Do not use `# type: ignore` as a blanket suppressor — fix the root cause
- Commit after each phase with message: `fix(007): <description>`
- Do not modify test logic to force passes — fix the implementation
