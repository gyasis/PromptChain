# Implementation Plan: PromptChain CLI Agent Interface

**Branch**: `001-cli-agent-interface` | **Date**: 2025-11-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-cli-agent-interface/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build an interactive CLI tool (`promptchain` command) that transforms PromptChain from a library into a full-featured terminal agent interface similar to Claude Code, Aider, Goose CLI, and Gemini CLI. The CLI will provide persistent conversation sessions, model-agnostic multi-agent orchestration, session management, file operations with `@syntax`, and shell command execution with `!syntax`. The implementation leverages Textual for TUI components, Rich for terminal formatting, and Click for command-line parsing, while integrating with PromptChain's existing AgentChain, MCPHelper, and ExecutionHistoryManager infrastructure.

## Technical Context

**Language/Version**: Python 3.8+ (compatible with existing PromptChain codebase)
**Primary Dependencies**: Textual 0.83+ (TUI framework), Rich 13.8+ (terminal formatting), Click 8.1+ (CLI framework), LiteLLM 1.0+ (existing), asyncio (stdlib), prompt_toolkit 3.0+ (input handling)
**Storage**: SQLite 3 (session persistence via existing AgentChain cache_config pattern), JSON/JSONL (session exports, logs)
**Testing**: pytest 7.0+ (existing test framework), pytest-asyncio (async test support), pytest-mock (mocking MCP/LLM calls)
**Target Platform**: Cross-platform (Linux, macOS, Windows) terminal environments with ANSI support
**Project Type**: Single project with CLI entry point
**Performance Goals**: <10s session startup, <500ms file reference loading (<10MB files), <2s session save, <3s session resume
**Constraints**: Must preserve existing PromptChain library API (no breaking changes), CLI is additive feature accessed via `promptchain` command entry point
**Scale/Scope**: Support 100+ message conversations, 10+ concurrent agents per session, sessions scalable to 1000+ saved sessions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ I. Library-First Architecture
**Status**: PASS
- CLI components built as reusable modules (`promptchain/cli/`)
- Session management, command parsing, TUI widgets are independently testable
- Clear separation: CLI layer (Textual/Click) → AgentChain orchestration → PromptChain execution
- No tight coupling to CLI; existing library APIs remain unchanged

### ✅ II. Observable Systems
**Status**: PASS
- Leverage existing ExecutionEvent system for CLI lifecycle events
- Extend event types: `CLI_SESSION_START`, `CLI_COMMAND_EXECUTED`, `CLI_AGENT_SWITCHED`, `CLI_FILE_REFERENCED`
- RunLogger integration for JSONL session logs
- Execution metadata includes: CLI command, active agent, file references, timing

### ✅ III. Test-First Development (NON-NEGOTIABLE)
**Status**: PASS - TDD MANDATORY
- Write tests FIRST for each CLI component before implementation
- User approval of tests before writing implementation code
- Tests must FAIL initially (red), then implementation makes them pass (green)
- Contract tests for CLI ↔ AgentChain interaction
- Integration tests for session persistence, agent switching, file operations

### ✅ IV. Integration Testing
**Status**: PASS
- Critical integration points:
  - CLI Session Manager ↔ AgentChain (agent selection, conversation flow)
  - File Reference Parser ↔ PromptChain (content injection)
  - Command Handler ↔ ExecutionHistoryManager (history management)
  - Session Persistence ↔ SQLite (save/resume operations)
- Contract tests verify CLI doesn't break existing AgentChain/PromptChain APIs

### ✅ V. Token Economy & Performance
**Status**: PASS
- ExecutionHistoryManager already provides token-aware truncation
- CLI session history leverages per-agent history configs (v0.4.2 feature)
- File references use intelligent truncation for large files (>100KB preview mode)
- Agent switching preserves conversation context without re-sending full history

### ✅ VI. Async-First Design
**Status**: PASS
- CLI main loop uses async/await (Textual is async-native)
- AgentChain.run_chat() async integration
- Sync wrappers for Click command entry points (`promptchain` command bootstraps async event loop)
- Non-blocking UI updates during LLM response generation

### ✅ VII. Simplicity & Maintainability
**Status**: PASS
- Start with P1 (interactive chat) before adding P2-P5 features
- YAGNI: No advanced extensions (TOON, APReL, APRICOT) in initial implementation
- Plugin architecture requirements (FR-EXT-001 to FR-EXT-010) deferred to future iterations
- Clear component boundaries: CLI → AgentChain → PromptChain → MCPHelper

**OVERALL CONSTITUTION CHECK: ✅ PASS**

## Project Structure

### Documentation (this feature)

```text
specs/001-cli-agent-interface/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (technology decisions, rationale)
├── data-model.md        # Phase 1 output (Session, Agent, Message entities)
├── quickstart.md        # Phase 1 output (getting started guide)
├── contracts/           # Phase 1 output (CLI API contracts)
│   ├── session-manager.md
│   ├── command-handler.md
│   └── file-reference-parser.md
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
promptchain/
├── __init__.py
├── utils/
│   ├── promptchaining.py         # Existing PromptChain
│   ├── agent_chain.py            # Existing AgentChain
│   ├── agentic_step_processor.py # Existing AgenticStepProcessor
│   ├── execution_history_manager.py # Existing ExecutionHistoryManager
│   ├── mcp_helpers.py            # Existing MCPHelper
│   ├── logging_utils.py          # Existing RunLogger
│   └── strategies/
└── cli/                          # NEW: CLI components
    ├── __init__.py
    ├── main.py                   # Entry point, Click command definitions
    ├── session_manager.py        # Session lifecycle, persistence (SQLite)
    ├── command_handler.py        # Slash command routing (/agent, /session, /help)
    ├── file_reference_parser.py  # @file.txt and @directory/ handling
    ├── shell_executor.py         # !command execution and output capture
    ├── tui/                      # Textual UI components
    │   ├── __init__.py
    │   ├── app.py                # Main Textual app
    │   ├── chat_view.py          # Conversation display widget
    │   ├── input_widget.py       # Multi-line input with history
    │   ├── status_bar.py         # Session info, active agent, status
    │   └── command_palette.py    # Slash command autocomplete
    └── models/                   # Data models for CLI
        ├── __init__.py
        ├── session.py            # Session entity
        ├── message.py            # Message entity
        └── agent_config.py       # Agent configuration entity

tests/
├── cli/                          # NEW: CLI tests
│   ├── contract/
│   │   ├── test_session_manager_contract.py
│   │   ├── test_command_handler_contract.py
│   │   └── test_agentchain_integration.py
│   ├── integration/
│   │   ├── test_session_persistence.py
│   │   ├── test_agent_switching.py
│   │   ├── test_file_references.py
│   │   └── test_shell_execution.py
│   └── unit/
│       ├── test_file_reference_parser.py
│       ├── test_command_handler.py
│       └── test_session_models.py
└── [existing test directories]

setup.py                          # Updated: Add CLI entry point
pyproject.toml                    # Updated: Add CLI dependencies
```

**Structure Decision**: Single project structure (Option 1) selected. The CLI is an additive feature to the existing PromptChain library, packaged as `promptchain.cli` module. The `promptchain` command will be registered as a console script entry point in `setup.py`. This preserves the existing library structure while cleanly separating CLI-specific code from core orchestration logic.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**No violations detected.** All constitution principles are satisfied by the proposed architecture.
