# Implementation Plan: CLI Multi-Agent Communication Architecture

**Branch**: `003-multi-agent-communication` | **Date**: 2025-11-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-multi-agent-communication/spec.md`

## Summary

Implement a HYBRID EXTEND-FIRST architecture for multi-agent communication in PromptChain CLI to increase pattern coverage from 14% to 57% of production-grade agentic patterns. The approach extends existing infrastructure (ToolRegistry, SessionManager, AgentChain) rather than building new systems, adding targeted new modules only where extension is infeasible.

## Technical Context

**Language/Version**: Python 3.8+ (compatible with existing PromptChain codebase)
**Primary Dependencies**: SQLite3 (existing), Textual (existing TUI), LiteLLM (existing), asyncio (stdlib)
**Storage**: SQLite (extend existing sessions.db with new tables)
**Testing**: pytest with existing test infrastructure in tests/cli/
**Target Platform**: Linux/macOS/Windows (CLI application)
**Project Type**: Single project (extend existing promptchain/cli/)
**Performance Goals**: Communication < 10ms overhead, blackboard < 5ms operations
**Constraints**: Backward compatible with existing sessions, maintain 90% token optimization
**Scale/Scope**: Multi-agent workflows with 2-10 concurrent agents

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Library-First Architecture | ✅ PASS | New components (communication/, delegation_tools.py, blackboard_tools.py) are self-contained with clear boundaries |
| II. Observable Systems | ✅ PASS | FR-019 mandates activity logger captures all communication; events emitted for all state changes |
| III. Test-First Development | ✅ PASS | Test locations defined per component; TDD workflow will be followed |
| IV. Integration Testing | ✅ PASS | E2E test (SC-009) covers inter-component communication; contract tests for API stability |
| V. Token Economy & Performance | ✅ PASS | Communication metadata minimal in LLM context; FR-030 maintains 90% token optimization |
| VI. Async-First Design | ✅ PASS | All tools async with sync wrappers; handlers async by default |
| VII. Simplicity & Maintainability | ✅ PASS | EXTEND-FIRST approach minimizes new code; clear documentation required |

**Gate Result**: PASSED - No violations requiring justification

## Project Structure

### Documentation (this feature)

```text
specs/003-multi-agent-communication/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── api-schema.json
├── checklists/
│   └── requirements.md  # Already created
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
promptchain/
├── cli/
│   ├── communication/           # NEW: Message bus and handlers
│   │   ├── __init__.py
│   │   ├── handlers.py          # @cli_communication_handler decorator
│   │   └── message_bus.py       # Message routing and dispatch
│   ├── tools/
│   │   ├── registry.py          # EXTEND: Add allowed_agents, capabilities
│   │   └── library/
│   │       ├── registration.py  # EXTEND: Add capability tags to 19 tools
│   │       ├── delegation_tools.py   # NEW: delegate_task, request_help
│   │       └── blackboard_tools.py   # NEW: blackboard read/write
│   ├── models/
│   │   ├── session.py           # EXTEND: Add Task, BlackboardEntry models
│   │   └── workflow.py          # NEW: WorkflowState, WorkflowStage
│   ├── session_manager.py       # EXTEND: Add new tables
│   ├── command_handler.py       # EXTEND: Add /capabilities, /tasks, /blackboard, /workflow, /mentalmodels
│   └── schema.sql               # EXTEND: Add 3 new tables
└── utils/
    ├── mental_models.py         # NEW: **CRITICAL** Mental model reasoning frameworks
    ├── agent_chain.py           # EXTEND: Add enable_mental_models, mental model tools
    └── strategies/
        └── dynamic_decomposition_strategy.py  # EXTEND: Agent-initiated delegation

tests/
├── cli/
│   ├── communication/           # NEW
│   │   └── test_handlers.py
│   ├── tools/
│   │   ├── test_delegation.py   # NEW
│   │   ├── test_blackboard.py   # NEW
│   │   └── test_mental_models.py # NEW: **CRITICAL** Mental model tests
│   ├── integration/
│   │   ├── test_workflow.py     # NEW
│   │   └── test_mental_models.py # NEW: Mental model integration tests
│   └── e2e/
│       ├── test_multi_agent.py  # NEW
│       └── test_mental_models_e2e.py  # NEW: E2E mental models validation
```

**Structure Decision**: Extend existing `promptchain/cli/` structure. New `communication/` module is the only new top-level directory. All other changes extend existing files or add to existing directories.

## Mental Models Integration (CRITICAL)

**Source Document**: `/docs/agent_communication/thoughtbox_mental_models_integration.md`

Mental models are **process scaffolds** that tell agents **HOW to think** about problems. This is a **critical** component for agent behavior quality.

### 15 Mental Models

| Model | Tags | Purpose |
|-------|------|---------|
| rubber-duck | debugging, communication | Explain problems step-by-step |
| five-whys | debugging, validation | Root cause analysis |
| pre-mortem | risk-analysis, planning | Identify failure points |
| assumption-surfacing | validation, planning | Validate hidden assumptions |
| steelmanning | decision-making, validation | Balanced decision making |
| trade-off-matrix | decision-making, prioritization | Map competing concerns |
| fermi-estimation | estimation | Order-of-magnitude estimates |
| abstraction-laddering | architecture, communication | Find right abstraction level |
| decomposition | planning, architecture | Break down complexity |
| adversarial-thinking | risk-analysis, validation | Security/edge case analysis |
| opportunity-cost | decision-making, prioritization | Resource allocation |
| constraint-relaxation | planning, architecture | Explore solution space |
| time-horizon-shifting | planning, decision-making | Multi-scale evaluation |
| impact-effort-grid | prioritization | Task prioritization |
| inversion | risk-analysis, planning | Avoid paths to failure |

### Architecture

```
Agent Task → Mental Model Selector → Selected Model → Apply Process → Task Execution
                ↓
         Tag-Based Discovery
                ↓
         Model Registry (15 models)
```

### Components

1. **MentalModelRegistry** (`promptchain/utils/mental_models.py`)
   - Stores all 15 mental models with process prompts
   - Tag-based organization and discovery
   - `find_models_for_task()` keyword matching

2. **MentalModelSelector** (`promptchain/utils/mental_models.py`)
   - Uses LLM to select appropriate model
   - Builds selection prompt with candidates
   - Returns model name or "none"

3. **MentalModelApplicator** (`promptchain/utils/mental_models.py`)
   - Applies model's process to task
   - Integrates with AgenticStepProcessor
   - Returns reasoning result

4. **AgentChain Integration** (`promptchain/utils/agent_chain.py`)
   - `enable_mental_models=True` parameter
   - Mental model tools: `select_mental_model`, `get_mental_model`, `list_mental_models`
   - Auto-selection in `process_input()` when enabled

### Benefits

- **No External Dependencies**: Works without MCP server
- **Seamless Integration**: Part of agent's native toolset
- **Automatic Selection**: Agents auto-select based on task
- **Flexible Application**: Models can be applied at any point
- **Extensible**: Easy to add new models or customize

## Complexity Tracking

> No Constitution violations to justify. EXTEND-FIRST approach keeps complexity minimal.

| Decision | Rationale |
|----------|-----------|
| Extend ToolRegistry vs new AgentCapabilityRegistry | Reuse existing decorator pattern, parameter validation, and discovery |
| SQLite tables in sessions.db vs separate DB | Single transaction context, existing backup/restore, simpler migration |
| Decorator-based handlers vs event bus | Matches existing CLI patterns (command_handler decorators), simpler testing |
