# PromptChain Polish Checklist
**Created:** 2026-02-24
**Purpose:** Consolidate all genuinely remaining tasks across specs 001, 002, 003 into themed groups for a final polish pass.
**Scope:** Excludes spec 004, 004a (100% done), spec 005 (100% done), spec 006 (in progress).

---

## Summary

| Group | Theme | Tasks | Source | Priority |
|-------|-------|-------|--------|----------|
| A | Missing Test Coverage | 32 | 002, 003 | High |
| B | Mental Models Feature | 39 | 003 Phase 9 | High (large feature) |
| C | Workflow CRUD Methods | 5 | 003 Phase 7 | High (unblocks tests) |
| D | Activity Logger Integration | 2 | 003 | Medium |
| E | Documentation | 5 | 001 | Medium |
| F | UX Polish | 4 | 001 | Medium |
| G | Security & Validation | 3 | 001, 003 | Medium |
| H | Deferred Enhancements | 3 | 002, 003 | Low |

**Total remaining: 93 tasks**

---

## Group A — Missing Test Coverage
**Rationale:** All underlying features are implemented; these tests exist in spec as contracts but were never written. Grouped because they can all be handled by the `test-automator` agent in parallel waves. Prerequisite: Group C must complete first (Workflow CRUD) to unblock workflow tests.

### From Spec 002
- [ ] **002-T032** Contract test for AgentConfig schema validation (`tests/cli/contract/test_agent_config_contract.py`)
- [ ] **002-T033** Contract test for router decision prompt template (`tests/cli/contract/test_router_contract.py`)
- [ ] **002-T034** Integration test for router mode agent selection (`tests/cli/integration/test_agentchain_routing.py`)
- [ ] **002-T035** Integration test for multi-agent conversation flow with automatic switching
- [ ] **002-T036** Unit test for agent description matching logic (`tests/cli/unit/test_agent_selection.py`)
- [ ] **002-T058** Integration test for MCP tool discovery and registration (needs mocking framework)
- [ ] **002-T059** Integration test for MCP tool execution within agent conversation (needs mocking framework)
- [ ] **002-T083** Integration test for workflow persistence across session restarts
- [ ] **002-T084** Integration test for workflow resume with context restoration
- [ ] **002-T085** Unit test for workflow step state transitions (pending→in_progress→completed)

### From Spec 003
- [ ] **003-T014** Create test directory structure: `tests/cli/communication/`, `tests/cli/e2e/`, `tests/cli/performance/`
- [ ] **003-T015** Unit test for ToolMetadata extensions in `tests/cli/unit/test_registry.py`
- [ ] **003-T016** Unit test for `discover_capabilities()` in `tests/cli/unit/test_registry.py`
- [ ] **003-T017** Integration test for capability discovery in `tests/cli/integration/test_capability_discovery.py`
- [ ] **003-T021** Backward compatibility test — tools without `allowed_agents` still work
- [ ] **003-T027** Unit test for Task model in `tests/cli/tools/test_delegation.py`
- [ ] **003-T028** Unit test for `delegate_task` tool in `tests/cli/tools/test_delegation.py`
- [ ] **003-T029** Integration test for task status transitions in `tests/cli/integration/test_task_delegation.py`
- [ ] **003-T037** Unit test for BlackboardEntry model (`tests/cli/tools/test_blackboard.py`)
- [ ] **003-T038** Unit test for `write_to_blackboard` tool
- [ ] **003-T039** Unit test for `read_from_blackboard` tool
- [ ] **003-T040** Integration test for blackboard operations
- [ ] **003-T048** Unit test for Message dataclass (`tests/cli/communication/test_handlers.py`)
- [ ] **003-T049** Unit test for `@cli_communication_handler` decorator
- [ ] **003-T050** Unit test for `message_bus` send/broadcast (`tests/cli/communication/test_message_bus.py`)
- [ ] **003-T051** Integration test for handler filtering
- [ ] **003-T060** Unit test for WorkflowState model (`tests/cli/integration/test_workflow.py`)
- [ ] **003-T061** Unit test for workflow stage transitions
- [ ] **003-T062** Integration test for workflow persistence
- [ ] **003-T070** Unit test for `request_help` tool (`tests/cli/tools/test_delegation.py`)
- [ ] **003-T071** Integration test for help routing (`tests/cli/integration/test_help_request.py`)
- [ ] **003-T076** Unit test for MentalModel dataclass
- [ ] **003-T077** Unit test for MentalModelRegistry
- [ ] **003-T078** Unit test for MentalModelSelector
- [ ] **003-T079** Unit test for `find_models_for_task()`
- [ ] **003-T080** Integration test for mental model selection
- [ ] **003-T081** Integration test for model application

---

## Group B — Mental Models Feature (Phase 9)
**Rationale:** This is the only entirely unimplemented feature across all specs — `promptchain/utils/mental_models.py` does not exist. It's a substantial new capability (15 reasoning process prompts + registry + selector + applicator + AgentChain integration + CLI command). Should be treated as a mini-sprint of its own. A test file exists but has nothing to test yet — Group A mental model tests (T076-T081) are blocked until this group ships.

### Core Data Layer
- [ ] **003-T082** Create `MentalModel` dataclass in `promptchain/utils/mental_models.py`
- [ ] **003-T083** Create `Tag` dataclass
- [ ] **003-T084** Implement `MentalModelRegistry` class
- [ ] **003-T085** Implement `get_model()` method
- [ ] **003-T086** Implement `list_models()` method
- [ ] **003-T087** Implement `list_tags()` method
- [ ] **003-T088** Implement `find_models_for_task()` method

### 15 Process Prompts
- [ ] **003-T089** Rubber-duck debugging prompt
- [ ] **003-T090** Five-whys root cause prompt
- [ ] **003-T091** Pre-mortem risk analysis prompt
- [ ] **003-T092** Assumption-surfacing prompt
- [ ] **003-T093** Steelmanning prompt
- [ ] **003-T094** Trade-off matrix prompt
- [ ] **003-T095** Fermi estimation prompt
- [ ] **003-T096** Abstraction laddering prompt
- [ ] **003-T097** Decomposition prompt
- [ ] **003-T098** Adversarial thinking prompt
- [ ] **003-T099** Opportunity cost prompt
- [ ] **003-T100** Constraint relaxation prompt
- [ ] **003-T101** Time-horizon shifting prompt
- [ ] **003-T102** Impact-effort grid prompt
- [ ] **003-T103** Inversion prompt

### Selector & Applicator
- [ ] **003-T104** Implement `MentalModelSelector` class
- [ ] **003-T105** Implement `select_model()` async method
- [ ] **003-T106** Implement `_build_selection_prompt()`
- [ ] **003-T107** Implement `MentalModelApplicator` class
- [ ] **003-T108** Implement `apply_model()` async method

### AgentChain Integration
- [ ] **003-T109** Add `enable_mental_models` parameter to `AgentChain`
- [ ] **003-T110** Add `_register_mental_model_tools()` method
- [ ] **003-T111** Add `_create_mental_model_tools()` method
- [ ] **003-T112** Implement `select_mental_model` tool schema
- [ ] **003-T113** Implement `get_mental_model` tool schema
- [ ] **003-T114** Implement `list_mental_models` tool schema
- [ ] **003-T115** Implement `_handle_mental_model_tool()` async method
- [ ] **003-T116** Add auto-select mental model logic in `process_input()`
- [ ] **003-T117** Enhance user input with mental model guidance when model selected
- [ ] **003-T118** Add mental model context to conversation history

### CLI Integration
- [ ] **003-T119** Add `/mentalmodels` CLI command in `command_handler.py`
- [ ] **003-T120** Add mental model tools to CLI tool registry

### E2E & Performance (Phase 10 — Mental Models subset)
- [ ] **003-T129** E2E test for mental models integration with multi-agent workflow
- [ ] **003-T130** Performance test for mental model selection < 100ms

---

## Group C — Workflow CRUD Methods
**Rationale:** `SessionManager` is missing 4 concrete methods that the spec defines. These are small, focused implementations that unblock Group A workflow tests (003-T060, T061, T062) and Group B AgentChain integration. Should be done before writing workflow tests.

- [ ] **003-T063** Implement `create_workflow()` in `promptchain/cli/session_manager.py`
- [ ] **003-T064** Implement `update_workflow_stage()` in `session_manager.py`
- [ ] **003-T065** Implement `add_completed_task()` in `session_manager.py`
- [ ] **003-T066** Implement `get_workflow_state()` in `session_manager.py`
- [ ] **003-T068** Integrate workflow updates with task completion callbacks in `delegation_tools.py`

---

## Group D — Activity Logger Integration
**Rationale:** Both delegation and blackboard tools emit events but don't pipe them through the activity logger. Small, isolated additions. No blockers.

- [ ] **003-T036** Add activity logger integration for task events in `delegation_tools.py`
- [ ] **003-T047** Add activity logger integration for blackboard events in `blackboard_tools.py`

---

## Group E — Documentation
**Rationale:** All code exists; these are pure documentation tasks. Can be done independently at any time. Not blocking any feature work.

- [ ] **001-T138** Update `CLAUDE.md` with CLI usage patterns and examples
- [ ] **001-T139** Create `promptchain/cli/README.md` with architecture overview
- [ ] **001-T140** Add inline documentation to all CLI classes and methods
- [ ] **001-T156** Create help text formatting in ChatView with usage examples
- [ ] **001-T158** Validate quickstart.md workflows manually (code review, research, quick questions)

---

## Group F — UX Polish
**Rationale:** Input widget enhancements that improve developer experience. None are blockers; all are independent of each other and can be tackled one at a time.

- [ ] **001-T144** Add command history navigation with Up/Down arrow keys in `InputWidget`
- [ ] **001-T145** Implement slash command autocomplete with Tab completion in `InputWidget`
- [ ] **001-T146** Add multi-line input support with Shift+Enter for newlines
- [ ] **001-T147** Add Ctrl+C handling to cancel current operation without exiting

---

## Group G — Security & Validation
**Rationale:** Quality gates that should run before any production release. mypy typing closes silent bug surface; security review prevents path traversal/injection; success criteria validation confirms the product meets its stated goals.

- [ ] **001-T150** Run performance benchmarks and validate against spec success criteria (SC-001 to SC-012)
- [ ] **001-T164** Code review for security issues (path traversal, command injection)
- [ ] **003-T128** Validate all 15 success criteria (SC-001 to SC-015) for spec 003

---

## Group H — Deferred Enhancements
**Rationale:** These were explicitly deferred during original development — either waiting on upstream changes or marked low-priority. Include in polish only if time allows.

- [ ] **002-T091** Integrate workflow objective with `AgenticStepProcessor` goals for automatic step execution
- [ ] **003-T125** Update `promptchain/cli/__init__.py` exports for new communication/blackboard modules
- [ ] **003-T126** Run quickstart.md validation scenarios manually
- [ ] **003-T127** Update `checklists/requirements.md` with final completion status

---

## Recommended Execution Order

```
Phase 1 (unblocks everything):  Group C (Workflow CRUD) → Group D (Activity Logger)
Phase 2 (parallelizable):        Group A (Tests) + Group B (Mental Models — mini sprint)
Phase 3 (independent):           Group E (Docs) + Group F (UX) + Group G (Security)
Phase 4 (final):                 Group H (Deferred)
```

> **Note:** Group B (Mental Models) is large enough to warrant its own branch and spec entry. Consider creating `specs/007-mental-models/` after spec 006 is complete.
