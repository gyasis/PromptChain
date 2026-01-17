# Implementation Plan: CLI Orchestration Integration

**Branch**: `002-cli-orchestration` | **Date**: 2025-11-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-cli-orchestration/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Integrate the PromptChain CLI with full library infrastructure including AgentChain orchestration for multi-agent routing, AgenticStepProcessor for complex multi-hop reasoning, MCPHelper for external tool ecosystem integration, and ExecutionHistoryManager for token-efficient per-agent history management. This transforms the CLI from a simple chat interface into a sophisticated agent orchestration platform that showcases the full capabilities of the PromptChain library while maintaining sensible defaults for immediate usability and deep configuration options for power users.

**Technical Approach**: Refactor existing CLI to replace individual PromptChain instances with single AgentChain orchestrator, extend agent configuration model to support instruction chains (strings/functions/AgenticStepProcessor), integrate MCP server lifecycle management, implement per-agent history configs with token-aware truncation, add workflow state persistence, and provide pre-configured agent templates.

## Technical Context

**Language/Version**: Python 3.8+ (existing PromptChain codebase compatibility)
**Primary Dependencies**:
- LiteLLM 1.0+ (existing - unified LLM API)
- Textual 0.83+ (existing - TUI framework)
- Rich 13.8+ (existing - terminal formatting)
- Click 8.1+ (existing - CLI framework)
- tiktoken (existing - token counting for history management)
- asyncio (stdlib - concurrent agent execution)

**Storage**: SQLite 3 (existing session persistence via AgentChain cache_config pattern), JSONL files (conversation logs, execution history)

**Testing**: pytest (existing test framework), contract tests for agent configurations, integration tests for AgentChain ↔ PromptChain ↔ MCPHelper interactions

**Target Platform**: Linux/macOS/Windows terminal environments (cross-platform CLI)

**Project Type**: Single project - extension of existing `promptchain/cli/` module structure

**Performance Goals**:
- Agent routing decision < 500ms
- MCP tool discovery < 2 seconds at session start
- Token savings 30-60% via per-agent history configs
- Concurrent agent execution (5+ agents in broadcast mode) without deadlocks

**Constraints**:
- Backward compatibility with existing CLI commands and session storage
- Zero-config defaults (works out-of-box with single default agent)
- Deep config options via agent creation commands for power users
- Memory efficient (2GB+ recommended for concurrent execution with full history buffers)

**Scale/Scope**:
- 6-10 concurrent agents in multi-agent workflows
- Conversation histories up to 8000 tokens per agent
- 100+ MCP tools registered across filesystem/code_execution/web_search servers
- Multi-session workflows spanning days/weeks via persistent state

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Library-First Architecture ✅

**Status**: PASS

**Evidence**:
- Feature exclusively leverages existing library components (AgentChain, AgenticStepProcessor, MCPHelper, ExecutionHistoryManager)
- No new standalone libraries created - purely integration work
- CLI acts as demonstration layer showcasing library capabilities
- All orchestration logic resides in proven library infrastructure (AgentChain v0.4.2+)

**Rationale**: This feature epitomizes library-first design by transforming the CLI into a showcase for PromptChain's library infrastructure, increasing library utilization from 15% to 85%+.

### II. Observable Systems ✅

**Status**: PASS

**Evidence**:
- AgentChain provides 30+ lifecycle events (agent_selected, reasoning_step_complete, tool_call_started, etc.)
- ExecutionHistoryManager tracks structured entries (user_input, agent_output, tool_call, tool_result)
- JSONL logging already implemented via RunLogger for conversation tracking
- TUI will display agent routing decisions and reasoning step progress
- Execution metadata includes timing, token counts, tool calls, errors

**Rationale**: Existing library components already meet observability standards. CLI integration extends visibility to user via TUI status updates.

### III. Test-First Development (NON-NEGOTIABLE) ⚠️

**Status**: PENDING (Will be enforced in Phase 2 - `/speckit.tasks`)

**Plan**:
- Tests written FIRST before implementation (TDD cycle strictly enforced)
- User approval required before proceeding to implementation
- Test categories: contract tests (AgentChain configuration), integration tests (multi-agent routing), unit tests (command parsing, history management)
- Red-Green-Refactor cycle documented in tasks.md

**Note**: Phase 0-1 (research, design, contracts) precedes test writing. Tests authored during `/speckit.tasks` execution in Phase 2.

### IV. Integration Testing ✅

**Status**: PASS (Design supports required integration points)

**Required Integration Tests** (to be written in Phase 2):
- AgentChain ↔ PromptChain contract: Router decision prompt execution, instruction chain processing
- AgentChain ↔ MCPHelper: Tool discovery, registration, execution with proper prefixing
- AgenticStepProcessor ↔ Tool Ecosystem: Multi-hop reasoning with tool calls (local + MCP)
- ExecutionHistoryManager ↔ AgentChain: Per-agent history configs, token-aware truncation
- Session Manager ↔ Workflow State: Persistence and restoration across session restarts
- Command Handler ↔ Agent Management: Create/list/delete/switch agents via `/agent` commands

**Rationale**: Feature heavily relies on inter-component communication. Integration tests critical for verifying orchestration boundaries.

### V. Token Economy & Performance ✅

**Status**: PASS

**Evidence**:
- Per-agent history configs (v0.4.2 feature) explicitly integrated for 30-60% token savings
- ExecutionHistoryManager provides tiktoken-based token counting and truncation
- Terminal agents configured with `history_enabled=False` (saves ~60% tokens)
- Research agents configured with `max_tokens=8000` and truncation strategies
- Async/await support for concurrent agent execution (broadcast mode)
- Performance metrics: routing < 500ms, tool discovery < 2s, stable 5+ agent concurrency

**Rationale**: Token optimization is a PRIMARY feature goal (SC-003: 30-60% token reduction). Design inherently optimized via library infrastructure.

### VI. Async-First Design ✅

**Status**: PASS

**Evidence**:
- AgentChain provides both sync and async interfaces (`run_chat()` and `run_chat_async()`)
- PromptChain offers dual APIs (`process_prompt()` and `process_prompt_async()`)
- MCPHelper uses async context managers for server lifecycle
- TUI application will use async event loop for concurrent operations
- Backward compatibility maintained via sync wrappers

**Rationale**: All underlying library components follow async-first pattern. CLI integration inherits this design naturally.

### VII. Simplicity & Maintainability ✅

**Status**: PASS

**Evidence**:
- Leverages existing components (YAGNI - no reinvention)
- Zero-config defaults: Works out-of-box with single default agent
- Clear component boundaries: AgentChain (orchestration), AgenticStepProcessor (reasoning), MCPHelper (tools), ExecutionHistoryManager (memory)
- Explicit agent configuration via commands (no magic routing logic)
- Documentation in CLAUDE.md updated for new workflows
- Modular structure preserved: `promptchain/cli/` with clear subdirectories

**Rationale**: Feature adds sophistication through composition of existing simple components, not new complexity. CLI remains approachable for basic use while enabling advanced workflows.

### Constitution Compliance Summary

**Gates Passed**: 6/7 ✅
**Gates Pending**: 1/7 (Test-First Development - enforced in Phase 2)
**Violations Requiring Justification**: NONE

**Proceed to Phase 0**: ✅ APPROVED

## Project Structure

### Documentation (this feature)

```text
specs/002-cli-orchestration/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   ├── agent-config-schema.json      # Agent configuration contract
│   ├── history-config-schema.json    # Per-agent history config contract
│   ├── mcp-server-schema.json        # MCP server connection contract
│   └── workflow-state-schema.json    # Workflow persistence contract
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
promptchain/cli/
├── models/
│   ├── agent_config.py       # EXTENDED: Add instruction_chain, objectives, history_config fields
│   ├── session.py            # EXTENDED: Add mcp_servers, workflow_state fields
│   ├── message.py            # EXISTING: No changes
│   └── config.py             # EXISTING: No changes
├── tui/
│   ├── app.py                # REFACTORED: Replace PromptChain with AgentChain, update UI for routing
│   ├── status_bar.py         # EXTENDED: Show active agent, reasoning progress
│   └── widgets/              # NEW: Agent selector, reasoning step display widgets
├── utils/
│   ├── file_context_manager.py    # EXISTING: No changes
│   ├── error_logger.py            # EXISTING: No changes
│   └── agent_templates.py         # NEW: Pre-configured agent templates (researcher, coder, analyst, terminal)
├── main.py               # EXISTING: Entry point (minor updates for new commands)
├── session_manager.py    # EXTENDED: Workflow state persistence, MCP server config
├── command_handler.py    # EXTENDED: /tools, /workflow commands, template creation
└── error_handler.py      # EXISTING: No changes

promptchain/utils/
├── agent_chain.py             # EXISTING: Core orchestration (already supports v0.4.2 features)
├── agentic_step_processor.py  # EXISTING: Multi-hop reasoning
├── mcp_helpers.py             # EXISTING: MCP server lifecycle
├── execution_history_manager.py  # EXISTING: Token-aware history
└── promptchaining.py          # EXISTING: Base chain execution

tests/cli/
├── contract/
│   ├── test_agent_config_contract.py     # NEW: Validate AgentConfig schema
│   ├── test_history_config_contract.py   # NEW: Validate per-agent history
│   └── test_mcp_server_contract.py       # NEW: Validate MCP server configs
├── integration/
│   ├── test_agentchain_routing.py        # NEW: Router mode agent selection
│   ├── test_agentic_reasoning.py         # NEW: Multi-hop with tool calls
│   ├── test_mcp_integration.py           # NEW: Tool discovery and execution
│   └── test_workflow_persistence.py      # NEW: Multi-session workflows
└── unit/
    ├── test_agent_templates.py           # NEW: Template creation and validation
    ├── test_command_handler.py           # EXTENDED: /tools, /workflow commands
    └── test_session_manager.py           # EXTENDED: Workflow state operations
```

**Structure Decision**: Single project structure maintained. All changes confined to existing `promptchain/cli/` module with extensions to models, TUI, and command handling. Library infrastructure (`promptchain/utils/`) remains unchanged - CLI purely integrates existing capabilities.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**No violations requiring justification**. All constitution principles satisfied by design.

**Complexity Mitigation Strategies** (proactive, not violations):

| Potential Complexity | Mitigation Strategy |
|---------------------|---------------------|
| AgentChain router configuration with decision prompts | Provide sensible defaults in agent templates; document decision prompt customization in quickstart.md |
| Per-agent history configs with 8 parameters | Default to reasonable values (terminal: disabled, research: 8000 tokens); expose only when using `/agent create` with `--advanced` flag |
| MCP server lifecycle management | Wrap MCPHelper in session manager; auto-connect default servers (filesystem, code_execution); handle failures gracefully |
| Workflow state persistence (SQLite + JSONL) | Reuse existing AgentChain cache_config pattern; extend SessionManager with workflow_state column; no new storage mechanisms |
| Agent template system | Start with 4 core templates (researcher, coder, analyst, terminal); templates are just pre-filled AgentConfig objects; users can customize post-creation |

**Zero-Config Defaults Philosophy**:
- Default agent created automatically if user doesn't create custom agents
- MCP servers attempt auto-connect but CLI functions without them
- History management uses reasonable defaults (4000 tokens, oldest_first truncation)
- Router mode enabled only when multiple agents exist; single-agent mode is simple pass-through
- Workflow state is optional feature (CLI works fine without `/workflow` commands)

**Deep Config Options for Power Users**:
- `/agent create --model <model> --description <desc> --history-max-tokens <N> --history-strategy <strategy>`
- `/tools add <mcp_server_config_path>` for custom MCP servers
- `/workflow create <objective> --max-steps <N>` for complex multi-session objectives
- Agent instruction chains editable via configuration files for advanced customization

---

## Post-Design Constitution Re-Evaluation

*GATE: Re-check after Phase 1 design completion*

### Constitution Compliance Review

All seven constitution principles remain satisfied after Phase 1 design:

**I. Library-First Architecture** ✅
- **Design evidence**: No new libraries created. All components (AgentConfig, Session, WorkflowState) extend existing CLI models. YAML translator uses library components (AgentChain, PromptChain, AgenticStepProcessor).
- **Status**: PASS - Pure integration layer over proven library infrastructure.

**II. Observable Systems** ✅
- **Design evidence**: Data model captures all required metadata (agent selections, workflow progress, MCP server status, tool calls). JSON schemas define structured logging formats. Execution history preserved in JSONL.
- **Status**: PASS - Full observability maintained through design.

**III. Test-First Development** ⚠️
- **Status**: PENDING - Tests will be written in Phase 2 (`/speckit.tasks`)
- **Design readiness**: Data models define clear contracts for testing. JSON schemas enable contract test automation.
- **Note**: Constitution requires tests BEFORE implementation. Tasks.md will enforce TDD cycle.

**IV. Integration Testing** ✅
- **Design evidence**: Data model defines integration points (AgentChain ↔ PromptChain via AgentConfig, Session ↔ MCPHelper via MCPServerConfig, Workflow ↔ Session persistence). Contract schemas specify validation rules.
- **Status**: PASS - Integration boundaries clearly documented, testable.

**V. Token Economy & Performance** ✅
- **Design evidence**: HistoryConfig model implements per-agent token limits (100-16000 range). Research shows 44% token savings via tiered configs. Quickstart demonstrates optimization patterns.
- **Status**: PASS - Token optimization built into data model and configuration system.

**VI. Async-First Design** ✅
- **Design evidence**: Data models are async-agnostic (plain dataclasses). Library components already async-first (AgentChain, PromptChain, MCPHelper). CLI integration inherits async patterns. Router configuration follows patterns established in [research.md R9](./research.md#9-router-configuration-and-performance).
- **Status**: PASS - Design compatible with async library infrastructure.

**VII. Simplicity & Maintainability** ✅
- **Design evidence**:
  - Zero-config defaults (auto-create default agent, attempt MCP auto-connect)
  - YAML configuration for declarative setup (human-readable)
  - Clear data model with 10 entities vs potential 50+ if overengineered
  - Schema versioning enables evolution without breaking changes
  - Template system provides guided starting points (4 templates vs unlimited customization)
- **Status**: PASS - Design balances power with simplicity via sensible defaults.

### Final Constitution Score

**Gates Passed**: 6/7 ✅
**Gates Pending**: 1/7 (Test-First Development - enforced in Phase 2)
**Violations**: NONE
**Design Quality**: HIGH - All principles satisfied by design

**Proceed to Phase 2** (/speckit.tasks): ✅ APPROVED

---

## Phase 1 Completion Summary

### Artifacts Generated

✅ **Technical Context**: Language, dependencies, performance goals, constraints documented
✅ **Constitution Check**: All 7 principles evaluated (6 pass, 1 pending for Phase 2)
✅ **Research (Phase 0)**: 8 research areas resolved with clear decisions
✅ **Data Models (Phase 1)**: 10 entities defined with validation rules, state transitions, relationships
✅ **API Contracts (Phase 1)**: 4 JSON schemas (agent-config, mcp-server, workflow-state, yaml-config)
✅ **Quickstart Guide (Phase 1)**: Comprehensive examples covering all major features
✅ **Agent Context Update (Phase 1)**: CLAUDE.md updated with new technologies

### Key Design Decisions

| Decision | Outcome |
|----------|---------|
| Orchestration Pattern | Single AgentChain per session with router mode |
| Configuration Format | YAML with translation layer + zero-config defaults |
| History Optimization | 3-tier system (disabled/medium/full) for 44% token savings |
| MCP Defaults | filesystem + code_execution auto-connect, graceful degradation |
| Workflow State | Minimal persistence (objective/steps/progress) |
| Agent Templates | 4 specialized (researcher/coder/analyst/terminal) |
| Schema Versioning | V1→V2 migration with backward compatibility |
| Tool Naming | MCP prefix (mcp_{server_id}_{tool_name}) prevents conflicts |

### Next Phase Preview

**Phase 2** (`/speckit.tasks`): Generate tasks.md with:
- TDD test specifications (write tests FIRST)
- Implementation tasks (RED → GREEN → REFACTOR cycle)
- Integration validation checkpoints
- Rollback procedures
- User acceptance criteria

**Command**: `/speckit.tasks` (when ready to proceed)
