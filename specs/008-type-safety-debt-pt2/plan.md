# Implementation Plan: 008-type-safety-debt-pt2 — Fix High-Error Type Safety Files

**Branch**: `008-type-safety-debt-pt2` | **Date**: 2026-02-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/008-type-safety-debt-pt2/spec.md`

## Summary

Fix the 4 remaining high-error files in the PromptChain codebase to reach 0 mypy errors each,
reducing total project errors from 329 to < 220. The approach is annotation-only: add
`Optional[...]` wrappers, `isinstance()` narrowing, `if x is not None:` guards, correct kwarg
names at LightRAG call sites, and `TYPE_CHECKING` guards around conditional import fallbacks.
No business logic changes.

## Technical Context

**Language/Version**: Python 3.12.11
**Primary Dependencies**: mypy 1.16.1, litellm 1.0+, Textual 0.83+, LightRAG (hybridrag)
**Storage**: N/A (annotation-only fixes; no data persistence changes)
**Testing**: pytest (existing test suite for regression), mypy per-file runs for acceptance
**Target Platform**: Linux (development), Python 3.12
**Project Type**: Single Python library with CLI/TUI layer
**Performance Goals**: N/A (annotation changes have no runtime performance impact)
**Constraints**: Zero runtime behaviour change — no business logic modifications allowed
**Scale/Scope**: 4 files, ~208 error lines to resolve

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Library-First Architecture | PASS | Fixes are internal to existing libraries; no new dependencies |
| II. Observable Systems | PASS | No changes to event emission or observability wiring |
| III. Test-First Development | MODIFIED | TDD waived for annotation-only fixes per project precedent (007 established this). Verification: mypy per-file before and after each fix |
| IV. Integration Testing | PASS | pytest regression run required before merge |
| V. Token Economy & Performance | PASS | No runtime changes; token economy unaffected |
| VI. Async-First Design | PASS | No async/sync interface changes |
| VII. Simplicity & Maintainability | PASS | Fixes strictly reduce type debt, improving maintainability |

**Constitution verdict**: APPROVED with one noted exception:

> **Principle III exception**: TDD is waived for pure type-annotation fixes following the
> precedent set by sprint 007. Rationale: there is no meaningful way to write a failing test
> for "this parameter should be annotated `Optional[int]`". Verification is done via mypy
> (before = N errors, after = 0 errors for the target file).

## Project Structure

### Documentation (this feature)

```text
specs/008-type-safety-debt-pt2/
├── plan.md              # This file
├── research.md          # Phase 0: error patterns and fix strategies
├── data-model.md        # Phase 1: type annotation patterns catalogue
├── quickstart.md        # Phase 1: how to verify fixes locally
└── tasks.md             # Phase 2: task list (/speckit.tasks output)
```

### Source Code (files touched by this sprint)

```text
promptchain/utils/strategies/state_agent.py       # 82 errors → 0
promptchain/cli/tui/app.py                        # 63 errors → 0
promptchain/utils/promptchaining.py               # 32 errors → 0
promptchain/patterns/executors.py                 # ~31 errors → 0

# Read-only dependencies (do not modify):
promptchain/integrations/lightrag/query_expansion.py   # LightRAGQueryExpander signature
promptchain/integrations/lightrag/multi_hop.py          # LightRAGMultiHop signature
promptchain/integrations/lightrag/hybrid_search.py      # LightRAGHybridSearcher signature
promptchain/integrations/lightrag/sharded.py            # LightRAGShardedRetriever signature
promptchain/integrations/lightrag/speculative.py        # LightRAGSpeculativeExecutor signature
promptchain/integrations/lightrag/branching.py          # LightRAGBranchingThoughts signature
promptchain/cli/tui/chat_view.py                        # MessageItem class (has spinners)
promptchain/cli/models/__init__.py                      # Exports: BlackboardEntry, not Blackboard
```

**Structure Decision**: Single project. Only the 4 target files receive edits. All integration
modules are read-only references to verify correct signatures and export names.
