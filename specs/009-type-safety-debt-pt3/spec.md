# Feature Specification: 009-type-safety-debt-pt3 — Full Mypy Sweep to < 100 Errors

**Feature Branch**: `009-type-safety-debt-pt3`
**Created**: 2026-03-03
**Status**: Draft
**Input**: Third sprint of type safety debt reduction — 213 remaining errors across 55 files

---

## Problem

Sprints 007 (557→421, −24%) and 008 (421→213, −49%) fixed the largest single-file offenders.
The remaining 213 errors are distributed across 55 files in three tiers:

| Tier | Files | Errors | Description |
|------|-------|--------|-------------|
| Tier 1 | 5 | 79 | High-error files (13–21 errors each) |
| Tier 2 | 6 | 52 | Medium files (7–11 errors each) |
| Tier 3 | 44 | 82 | Long-tail (1–6 errors each, mostly trivial fixes) |

**Dominant error classes** (by frequency):

| Error Class | Count | Typical Fix |
|-------------|-------|-------------|
| `assignment` | 57 | Wrong annotation, implicit Optional |
| `attr-defined` | 40 | Wrong attribute name / None guard missing |
| `arg-type` | 32 | Argument type mismatch at call site |
| `misc` | 20 | Conditional import / override patterns |
| `var-annotated` | 17 | Missing annotation on `{}` or `[]` literal |
| `union-attr` | 11 | Access on union type without narrowing |
| `override` | 7 | Method signature mismatch in subclass |
| `return-value` | 5 | Return type declared but wrong type returned |

**Sprint target**: 213 → **< 100** errors (53% further reduction). All Tier 1 and Tier 2
files reach 0 errors. Tier 3 long-tail files are batch-fixed in parallel passes.

---

## User Scenarios & Testing

### User Story 1 — Tier 1 Files Reach 0 Errors (Priority: P1)

A developer running mypy on any of the five Tier 1 files receives zero errors.
These are framework infrastructure files; clean type-checking here prevents
silent bugs in model management, MCP connectivity, and agentic step execution.

**Tier 1 targets** (in order of error count):

| File | Errors |
|------|--------|
| `promptchain/observability/mlflow_adapter.py` | 21 |
| `promptchain/utils/ollama_model_manager.py` | 16 |
| `promptchain/utils/mcp_helpers.py` | 15 |
| `promptchain/utils/orchestrator_supervisor.py` | 14 |
| `promptchain/utils/agentic_step_processor.py` | 13 |

**Why this priority**: Highest per-file error counts; fixing them delivers 79 errors → 0 (37%
of total). These files are imported by most of the rest of the project, so their annotations
unblock downstream type inference across the codebase.

**Independent Test**: Run `python -m mypy <file> --ignore-missing-imports 2>&1 | grep "<filename>.*error:"` for each file and confirm empty output. Each file is independently verifiable.

**Acceptance Scenarios**:

1. **Given** `mlflow_adapter.py` with 21 errors (primarily `assignment` and `attr-defined`),
   **When** the MLflow optional import is restructured with `TYPE_CHECKING` guard and `Optional`
   annotations are added to fields that may be `None`, **Then** mypy reports 0 errors on this file.

2. **Given** `ollama_model_manager.py` with 16 errors (`assignment`, `arg-type`, `var-annotated`),
   **When** untyped dict/list literals are annotated and method parameter types are corrected,
   **Then** mypy reports 0 errors on this file.

3. **Given** `mcp_helpers.py` with 15 errors (`assignment`, `attr-defined`, `arg-type`),
   **When** the MCP conditional import block uses the `TYPE_CHECKING` guard pattern established
   in sprint 008, **Then** mypy reports 0 errors on this file.

4. **Given** `orchestrator_supervisor.py` with 14 errors (`assignment`, `union-attr`),
   **When** supervisor state fields are annotated and union-attr accesses are narrowed with
   `isinstance()` guards, **Then** mypy reports 0 errors on this file.

5. **Given** `agentic_step_processor.py` with 13 errors (`assignment`, `attr-defined`, `misc`),
   **When** step processor result type annotations are corrected and optional-chaining patterns
   are guarded, **Then** mypy reports 0 errors on this file.

---

### User Story 2 — Tier 2 Files Reach 0 Errors (Priority: P1)

A developer running mypy on any of the six Tier 2 files receives zero errors.

**Tier 2 targets**:

| File | Errors |
|------|--------|
| `promptchain/utils/dynamic_chain_builder.py` | 11 |
| `promptchain/utils/agent_chain.py` | 9 |
| `promptchain/cli/tui/task_list_widget.py` | 9 |
| `promptchain/utils/mcp_connection_manager.py` | 8 |
| `promptchain/tools/terminal/environment.py` | 8 |
| `promptchain/utils/mcp_client_manager.py` | 7 |

**Why this priority**: Combined 52 errors (24% of total); all are critical-path framework
modules. `agent_chain.py` and `dynamic_chain_builder.py` are the orchestration backbone.
The MCP manager pair shares the same import pattern fix as `mcp_helpers.py`.

**Independent Test**: Same pattern — run per-file mypy and confirm empty output.

**Acceptance Scenarios**:

1. **Given** `dynamic_chain_builder.py` with 11 errors (`assignment`, `var-annotated`),
   **When** builder method return types and intermediate variable annotations are corrected,
   **Then** 0 errors.

2. **Given** `agent_chain.py` with 9 errors (`override`, `assignment`),
   **When** subclass method signatures match base class and `Optional` annotations are added,
   **Then** 0 errors.

3. **Given** `task_list_widget.py` with 9 errors (`attr-defined`, `assignment`),
   **When** Textual widget attribute accesses are guarded and widget field types are declared,
   **Then** 0 errors.

4. **Given** `mcp_connection_manager.py` and `mcp_client_manager.py` with 8 and 7 errors
   (same MCP import pattern as `mcp_helpers.py`),
   **When** the same `TYPE_CHECKING` guard fix is applied to both,
   **Then** 0 errors each.

5. **Given** `environment.py` with 8 errors (`assignment`, `return-value`),
   **When** environment variable lookup return types use `Optional[str]` and return annotations
   are corrected, **Then** 0 errors.

---

### User Story 3 — Tier 3 Long-Tail Sweep (Priority: P2)

A developer running the full project mypy check sees fewer than 100 total errors.
The 44 long-tail files (1–6 errors each) are batch-fixed in parallel agent passes
grouped by error class.

**Why this priority**: Individually small, collectively 82 errors (38% of total). Fixing them
in parallel by error class (not file-by-file) is the most efficient strategy.

**Grouping strategy for parallel execution**:
- **Group A** (`var-annotated`): Add `Dict[str, Any]` / `List[T]` annotations to bare `{}` / `[]`
- **Group B** (`assignment` in 1–2 error files): Implicit Optional or wrong literal type fixes
- **Group C** (`attr-defined` in 1–2 error files): None guard or attribute rename
- **Group D** (`arg-type` in 1–2 error files): Coerce or correct call sites

**Independent Test**: `python -m mypy promptchain/ --ignore-missing-imports 2>&1 | tail -1` reports fewer than 100 errors.

**Acceptance Scenarios**:

1. **Given** 44 files each with 1–6 errors, **When** each is fixed using patterns established
   in sprints 007 and 008, **Then** the project total drops below 100.

2. **Given** `observability/__init__.py` with 6 errors, **When** the MLflow import block is
   guarded consistently with `mlflow_adapter.py`, **Then** 0 errors.

3. **Given** the 7 LightRAG integration files (branching.py 3, state.py 3, hybrid_search.py 5,
   plus 4 files with 1 error each) with scattered errors, **When** annotation fixes are applied
   consistently with the executor.py corrections from sprint 008, **Then** 0 errors across all
   LightRAG integration files.

---

### Edge Cases

- `agentic_step_processor.py` uses complex generic types for step results — narrowing must not
  break the existing `isinstance()` dispatch logic.
- `mcp_helpers.py` and `mcp_connection_manager.py` share some type stubs — fixing one may
  partially resolve the other; verify both independently after each fix.
- LightRAG integration files may have errors that are downstream effects of executor.py fixes
  from sprint 008 (corrected kwarg names propagate); run mypy fresh before fixing each to
  avoid double-work.
- `task_list_widget.py` is a Textual widget — Textual's type stubs may be incomplete; `# type: ignore[attr-defined]` is permitted only for genuine Textual framework gaps, not project code errors.
- `agent_chain.py` `override` errors — fix by aligning subclass signature with base class;
  do not change base class signatures unless the subclass signature is clearly correct.

---

## Requirements

### Functional Requirements

- **FR-001**: All 5 Tier 1 files MUST reach 0 mypy errors when checked individually with `--ignore-missing-imports`.
- **FR-002**: All 6 Tier 2 files MUST reach 0 mypy errors when checked individually with `--ignore-missing-imports`.
- **FR-003**: The full project `python -m mypy promptchain/ --ignore-missing-imports` MUST report fewer than 100 errors (down from 213).
- **FR-004**: All fixes MUST be annotation-only or control-flow guards — no business logic changes.
- **FR-005**: `pytest` MUST pass with no new failures vs pre-sprint baseline.
- **FR-006**: All modified files MUST pass `black` and `isort` formatting checks.
- **FR-007**: No blanket `# type: ignore` suppressors — every error fixed at root cause.
- **FR-008**: MCP conditional import blocks in `mcp_helpers.py`, `mcp_connection_manager.py`, and `mcp_client_manager.py` MUST use the `TYPE_CHECKING` guard pattern from sprint 008, not `# type: ignore[assignment]` hacks.
- **FR-009**: A mypy baseline snapshot MUST be captured at sprint start and test-results committed at the end.

### Key Entities

- **MCP Infrastructure** (`mcp_helpers.py`, `mcp_connection_manager.py`, `mcp_client_manager.py`): Three files sharing the same conditional MCP import pattern — one fix pattern applies to all.
- **Observability** (`mlflow_adapter.py`, `observability/__init__.py`): MLflow optional import pattern identical across both files.
- **Agentic Core** (`agentic_step_processor.py`, `orchestrator_supervisor.py`, `agent_chain.py`, `dynamic_chain_builder.py`): Orchestration backbone; annotation fixes here improve downstream type inference.
- **LightRAG integrations** (7 files): 1–5 errors each, share fix patterns with executor.py from sprint 008.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: `python -m mypy promptchain/observability/mlflow_adapter.py --ignore-missing-imports` → **0 errors** (from 21)
- **SC-002**: `python -m mypy promptchain/utils/ollama_model_manager.py --ignore-missing-imports` → **0 errors** (from 16)
- **SC-003**: `python -m mypy promptchain/utils/mcp_helpers.py --ignore-missing-imports` → **0 errors** (from 15)
- **SC-004**: `python -m mypy promptchain/utils/orchestrator_supervisor.py --ignore-missing-imports` → **0 errors** (from 14)
- **SC-005**: `python -m mypy promptchain/utils/agentic_step_processor.py --ignore-missing-imports` → **0 errors** (from 13)
- **SC-006**: All 6 Tier 2 files → **0 errors** each
- **SC-007**: `python -m mypy promptchain/ --ignore-missing-imports` → **< 100 errors** total (from 213)
- **SC-008**: `pytest` exits with same passing count as pre-sprint baseline; zero new failures
- **SC-009**: `black --check` and `isort --check` pass on all modified files
- **SC-010**: No `# type: ignore` blanket suppressors introduced

---

## Assumptions

- The `TYPE_CHECKING` guard pattern from sprint 008 is the approved approach for all conditional MCP and MLflow imports — no new patterns needed.
- `agentic_step_processor.py` errors are annotation-only; if investigation reveals logic bugs, they are documented and deferred.
- Tier 3 long-tail files with 1–2 errors each are genuinely trivial — if any reveal deeper structural issues, they are deferred to a 010 sprint.
- The 57 `assignment` errors are a mix of implicit Optional and wrong-type annotations; none require runtime behavior changes.
- `agent_chain.py` `override` errors are signature mismatches fixable by aligning subclass to base class, not by changing logic.
- Running the full project mypy after Tier 1+2 fixes will reveal if Tier 3 errors have already been resolved transitively — Tier 3 effort may be less than 82 errors.
