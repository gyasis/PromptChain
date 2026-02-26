# Spec: 007-type-safety-debt — Fix Pre-Existing Type Errors

**Branch**: 007-type-safety-debt
**Priority**: P3 (maintenance)
**Status**: Backlog

## Problem

Branch 006-promptchain-improvements revealed 312+ pre-existing mypy errors and
several Pyright errors in files that were not touched by branch 006. These
are technical debt accumulated from earlier branches.

The mypy run (`python -m mypy promptchain/ --ignore-missing-imports`) reports
errors across 20+ files spanning the CLI, integrations, tools, and utils
layers. The most common error classes are:

- `implicit Optional` — non-Optional type hints with `None` defaults (PEP 484 violation)
- `name-defined` — missing imports / undefined names used in type positions
- `union-attr` — un-narrowed `Optional` access without None guard
- `valid-type` — `builtins.any` used as a type annotation instead of `typing.Any`
- `var-annotated` — untyped collection literals that mypy cannot infer
- `assignment` — value type mismatches on reassignment
- `arg-type` — wrong types passed to typed function parameters
- `no-redef` — duplicate name binding on import

## Affected Files (Pre-existing errors NOT from branch 006)

Sorted by error count (descending), derived from mypy output:

| File | Error Count | Primary Error Classes |
|------|-------------|----------------------|
| `promptchain/cli/session_manager.py` | 15 | `name-defined` (missing MCPServerConfig, WorkflowState, WorkflowStep, Task, BlackboardEntry, MultiAgentWorkflow), `arg-type` |
| `promptchain/cli/command_handler.py` | 13 | `var-annotated`, `operator`, `attr-defined`, `union-attr`, `index`, `import-untyped` |
| `promptchain/cli/config/yaml_translator.py` | 9 | `import-untyped`, `arg-type`, `return-value`, `assignment` |
| `promptchain/cli/tools/sandbox/docker_sandbox.py` | 8 | `arg-type` (subprocess.run with `str | None`), `list-item`, `call-arg`, `attr-defined` |
| `promptchain/cli/tools/safety.py` | 8 | `valid-type` (builtins.any), `assignment`, `arg-type` |
| `promptchain/tools/terminal/terminal_tool.py` | 7 | `assignment`, `attr-defined` (None.return_code), `assignment` (implicit Optional), `union-attr` |
| `promptchain/cli/communication/handlers.py` | 7 | `has-type`, `misc` (non-self attribute declaration) |
| `promptchain/tools/terminal/session_manager.py` | 5 | `assignment` (implicit Optional), `var-annotated` |
| `promptchain/integrations/lightrag/events.py` | 4 | `assignment` (implicit Optional), `return-value`, `var-annotated` |
| `promptchain/cli/tools/filesystem_tools.py` | 4 | `var-annotated`, `union-attr` |
| `promptchain/cli/tools/executor.py` | 3 | `union-attr` (SafetyValidator | None) |
| `promptchain/utils/model_management.py` | 2 | `assignment` (implicit Optional, None assignment) |
| `promptchain/cli/tui/token_bar.py` | 2 | `assignment` (implicit Optional int args) |
| `promptchain/cli/security/input_sanitizer.py` | 2 | `assignment`, `return-value` |
| `promptchain/cli/prompt_manager.py` | 2 | `var-annotated` |
| `promptchain/utils/ollama_model_manager.py` | 1 | `arg-type` (AsyncClient overload mismatch) |
| `promptchain/utils/mcp_schema_validator.py` | 1 | (schema validator error) |
| `promptchain/utils/execution_events.py` | 1 | `assignment` (int assigned to str var) |
| `promptchain/utils/dry_run.py` | 1 | `var-annotated` |
| `promptchain/utils/prompt_loader.py` | 1 | `var-annotated` |
| `promptchain/cli/__init__.py` | 1 | `no-redef` (registration already defined) |
| `promptchain/cli/main.py` | 1 | `var-annotated` |
| `promptchain/cli/models/agent_config.py` | 1 | `arg-type` |
| `promptchain/cli/security/yaml_validator.py` | 1 | `import-untyped` |
| `promptchain/observability/config.py` | 1 | `import-untyped` (yaml stubs) |

**Total captured distinct errors: ~100+ unique error lines**
(Full run of `python -m mypy promptchain/ --ignore-missing-imports` produces 312+ errors)

## Scope

Fix type errors in:
- `promptchain/cli/session_manager.py` — add missing TYPE_CHECKING imports for MCPServerConfig, WorkflowState, WorkflowStep, Task, BlackboardEntry, MultiAgentWorkflow
- `promptchain/cli/command_handler.py` — fix Collection[str] mutation attempts, operator types, HistoryConfig None guards
- `promptchain/cli/config/yaml_translator.py` — fix list type declarations, return type annotations, OrchestrationConfig Literal arg
- `promptchain/cli/tools/sandbox/docker_sandbox.py` — fix subprocess.run call with `str | None` items, missing SafetyValidator arg
- `promptchain/cli/tools/safety.py` — replace `any` type annotation with `Any`, fix str/Path/int assignment mismatches
- `promptchain/tools/terminal/terminal_tool.py` — add None guards on CommandResult, fix implicit Optional args
- `promptchain/cli/communication/handlers.py` — fix `_handlers` class-level type declaration
- `promptchain/tools/terminal/session_manager.py` — fix implicit Optional parameter annotations
- `promptchain/integrations/lightrag/events.py` — fix implicit Optional args, Sequence vs list return type
- `promptchain/cli/tools/filesystem_tools.py` — add type annotations to collection vars, guard ToolMetadata | None
- `promptchain/cli/tools/executor.py` — add None guards on SafetyValidator access
- Any other files with > 1 mypy error listed above

## Out of Scope

- New features
- Refactoring beyond fixing type annotations
- The 67 pre-existing test failures in `tests/unit/patterns/` (separate issue)
- Fixing `import-untyped` yaml stubs (requires `pip install types-PyYAML`; tracked separately)
- Changes to `promptchain/cli/tui/app.py` Pyright errors that require Textual API investigation

## Success Criteria

- `mypy promptchain/ --ignore-missing-imports` reports < 10 errors (down from 312+)
- All Pyright errors (not informational hints) in touched files resolved
- No regressions in existing test suite (`pytest` passes same tests as before)
- `black . && isort . && flake8 promptchain/` clean on all modified files
