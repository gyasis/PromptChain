# Tasks: PromptChain CLI Agent Interface

**Input**: Design documents from `/specs/001-cli-agent-interface/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: ✅ MANDATORY - TDD is required by Constitution Principle III (Test-First Development NON-NEGOTIABLE)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Project uses single project structure:
- `promptchain/cli/` - New CLI components
- `tests/cli/` - New CLI tests
- `setup.py`, `pyproject.toml` - Package configuration

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and CLI structure creation

- [x] T001 Create CLI directory structure at promptchain/cli/ with subdirectories: tui/, models/, __init__.py
- [x] T002 Create CLI test directory structure at tests/cli/ with subdirectories: contract/, integration/, unit/
- [x] T003 [P] Add CLI dependencies to setup.py: textual>=0.83.0, click>=8.1.0
- [x] T004 [P] Add CLI test dependencies to requirements-dev.txt: pytest-asyncio>=0.21.0, pytest-mock>=3.10.0
- [x] T005 [P] Create CLI models in promptchain/cli/models/__init__.py
- [x] T006 [P] Create Textual TUI directory promptchain/cli/tui/ with __init__.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core CLI infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Data Models (Foundation)

- [x] T007 [P] Create Session model in promptchain/cli/models/session.py with attributes from data-model.md
- [x] T008 [P] Create Agent model in promptchain/cli/models/agent_config.py with model_name attribute
- [x] T009 [P] Create Message model in promptchain/cli/models/message.py with role, content, timestamp
- [x] T010 Create FileReference dataclass in promptchain/cli/file_reference_parser.py with to_message_context() method

### Core CLI Components (Foundation)

- [x] T011 Create SessionManager class skeleton in promptchain/cli/session_manager.py with __init__()
- [x] T012 Create CommandHandler class skeleton in promptchain/cli/command_handler.py with __init__()
- [x] T013 Create FileReferenceParser class skeleton in promptchain/cli/file_reference_parser.py with __init__()
- [x] T014 Create ShellExecutor class skeleton in promptchain/cli/shell_executor.py with __init__()

### Database Schema (Foundation)

- [x] T015 Create database schema file promptchain/cli/schema.sql with sessions and agents tables from data-model.md
- [x] T016 Implement database initialization in SessionManager.__init__() to create tables if not exist
- [x] T017 Add schema version tracking table and migration framework to SessionManager

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Interactive Chat Session (Priority: P1) 🎯 MVP

**Goal**: Users can start interactive terminal session, have conversation with AI agent, maintain context, and exit gracefully

**Independent Test**: Run `promptchain` command, type question, receive response, ask follow-up with context awareness, then `/exit`

### Tests for User Story 1 (TDD - Write FIRST, Must FAIL) ⚠️

> **CRITICAL**: Write these tests FIRST, get user approval, verify they FAIL, then implement

- [x] T018 [P] [US1] Write contract test for basic session lifecycle in tests/cli/contract/test_session_lifecycle.py (test_create_session, test_session_has_default_agent, test_session_working_directory)
- [x] T019 [P] [US1] Write integration test for CLI startup in tests/cli/integration/test_cli_startup.py (test_promptchain_command_launches, test_welcome_message_displayed, test_prompt_ready)
- [x] T020 [P] [US1] Write integration test for conversation flow in tests/cli/integration/test_conversation_flow.py (test_user_sends_message, test_agent_responds, test_follow_up_maintains_context)
- [x] T021 [P] [US1] Write integration test for exit behavior in tests/cli/integration/test_exit_behavior.py (test_slash_exit_command, test_ctrl_d_exit, test_goodbye_message, test_session_cleanup)
- [x] T022 [P] [US1] Write unit test for ExecutionHistoryManager integration in tests/cli/unit/test_history_integration.py (test_messages_added_to_history, test_history_token_limits)

**USER APPROVAL REQUIRED**: Review and approve tests T018-T022 before proceeding to implementation

### Implementation for User Story 1

**SessionManager - Basic Operations**

- [x] T023 [P] [US1] Implement SessionManager.create_session() in promptchain/cli/session_manager.py with SQLite insert and directory creation
- [x] T024 [P] [US1] Implement SessionManager.load_session() in promptchain/cli/session_manager.py with SQLite query and JSONL loading
- [x] T025 [US1] Implement Session.add_message() method to append messages to conversation history
- [x] T026 [US1] Implement SessionManager.save_session() in promptchain/cli/session_manager.py with SQLite update and JSONL append

**Textual TUI - Basic Interface**

- [x] T027 [P] [US1] Create PromptChainApp class in promptchain/cli/tui/app.py extending textual.app.App
- [x] T028 [P] [US1] Implement ChatView widget in promptchain/cli/tui/chat_view.py for message display using ListView
- [x] T029 [P] [US1] Implement InputWidget in promptchain/cli/tui/input_widget.py using TextArea with Enter to submit
- [x] T030 [P] [US1] Implement StatusBar widget in promptchain/cli/tui/status_bar.py showing session name and agent

**AgentChain Integration**

- [x] T031 [US1] Create default agent initialization in SessionManager.create_session() using PromptChain with default model
- [x] T032 [US1] Implement conversation loop in PromptChainApp.on_message_submit() calling AgentChain
- [x] T033 [US1] Integrate ExecutionHistoryManager in SessionManager for context tracking
- [x] T034 [US1] Add message history to agent prompts via ExecutionHistoryManager.get_formatted_history()

**CLI Entry Point**

- [x] T035 [US1] Create main Click command in promptchain/cli/main.py with @click.command() decorator
- [x] T036 [US1] Add promptchain entry point to setup.py console_scripts
- [x] T037 [US1] Implement async event loop bootstrap in main() to run PromptChainApp
- [x] T038 [US1] Add welcome message display on PromptChainApp.on_mount()

**Exit Handling**

- [x] T039 [US1] Implement /exit command handler in CommandHandler.handle_exit()
- [x] T040 [US1] Add Ctrl+D handling in PromptChainApp binding to exit gracefully
- [x] T041 [US1] Trigger final save in PromptChainApp.on_exit() before shutdown
- [x] T042 [US1] Display goodbye message in PromptChainApp.on_exit()

**Checkpoint**: At this point, User Story 1 should be fully functional - basic chat works, context maintained, graceful exit

---

## Phase 4: User Story 2 - Agent Creation and Management (Priority: P2)

**Goal**: Users can create specialized agents with different models, list agents, switch between them, and delete agents

**Independent Test**: Create agents with `/agent create coding --model gpt-4`, `/agent create fast --model ollama/llama2`, verify with `/agent list`, switch with `/agent use coding`, delete with `/agent delete fast`

### Tests for User Story 2 (TDD - Write FIRST, Must FAIL) ⚠️

> **CRITICAL**: Write these tests FIRST, get user approval, verify they FAIL, then implement

- [x] T043 [P] [US2] Write contract test for agent CRUD operations in tests/cli/contract/test_agent_crud.py (test_create_agent, test_list_agents, test_delete_agent)
- [x] T044 [P] [US2] Write contract test for agent model configuration in tests/cli/contract/test_agent_models.py (test_agent_uses_specified_model, test_agent_without_model_uses_default, test_litellm_model_strings)
- [x] T045 [P] [US2] Write integration test for agent switching in tests/cli/integration/test_agent_switching.py (test_switch_agent, test_active_agent_responds, test_usage_count_incremented)
- [x] T046 [P] [US2] Write integration test for multi-agent conversation in tests/cli/integration/test_multi_agent.py (test_different_agents_different_responses, test_agent_history_isolation)
- [x] T047 [P] [US2] Write unit test for CommandHandler agent commands in tests/cli/unit/test_command_handler.py (test_parse_agent_create, test_parse_agent_use, test_parse_agent_delete)

**USER APPROVAL REQUIRED**: Review and approve tests T043-T047 before proceeding to implementation

### Implementation for User Story 2

**Command Parsing**

- [x] T048 [P] [US2] Implement CommandHandler.parse_command() to extract slash commands from input
- [x] T049 [P] [US2] Create ParsedCommand dataclass in promptchain/cli/command_handler.py with name, subcommand, args
- [x] T050 [US2] Implement command validation in CommandHandler.parse_command() for agent subcommands

**Agent CRUD Operations**

- [x] T051 [P] [US2] Implement CommandHandler.handle_agent_create() with name, model, description parameters
- [x] T052 [P] [US2] Implement Agent model validation in promptchain/cli/models/agent_config.py (name format, model string)
- [x] T053 [US2] Add agent to session.agents dict and persist to SQLite agents table in handle_agent_create()
- [x] T054 [P] [US2] Implement CommandHandler.handle_agent_list() formatting agent details with model info
- [x] T055 [P] [US2] Implement CommandHandler.handle_agent_delete() with validation (not active agent)
- [x] T056 [US2] Remove agent from SQLite and session.agents in handle_agent_delete()

**Agent Switching**

- [x] T057 [US2] Implement CommandHandler.handle_agent_use() updating session.active_agent
- [x] T058 [US2] Create PromptChain instance for each agent with specified model in SessionManager
- [x] T059 [US2] Update conversation routing to use active agent's PromptChain instance
- [x] T060 [US2] Increment agent.usage_count and update agent.last_used timestamp on switch
- [x] T061 [US2] Update StatusBar to display active agent name and model

**Model Configuration**

- [x] T062 [P] [US2] Add default_model configuration to SessionManager from config or env variable
- [x] T063 [P] [US2] Pass model string directly to PromptChain constructor (leverages existing LiteLLM integration)
- [x] T064 [US2] Validate model string format (provider/model-name) in Agent model __post_init__()

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - basic chat works, multiple agents with different models work

---

## Phase 5: User Story 3 - Session Persistence and Resumption (Priority: P3)

**Goal**: Users can save sessions, list saved sessions, resume previous sessions with full history, and delete old sessions

**Independent Test**: Have conversation, `/session save my-project`, exit, run `promptchain --session my-project`, verify history restored, `/session delete old-project`

### Tests for User Story 3 (TDD - Write FIRST, Must FAIL) ⚠️

> **CRITICAL**: Write these tests FIRST, get user approval, verify they FAIL, then implement

- [x] T065 [P] [US3] Write contract test for session save/load roundtrip in tests/cli/contract/test_session_persistence.py (test_save_session, test_load_session, test_history_preserved, test_agents_restored)
- [x] T066 [P] [US3] Write contract test for session performance in tests/cli/contract/test_session_performance.py (test_save_under_2_seconds, test_load_under_3_seconds per SC-003)
- [x] T067 [P] [US3] Write integration test for session list in tests/cli/integration/test_session_list.py (test_list_sessions, test_session_metadata_display, test_sorted_by_last_accessed)
- [x] T068 [P] [US3] Write integration test for session deletion in tests/cli/integration/test_session_delete.py (test_delete_session, test_files_removed, test_cascade_agents_deleted)
- [x] T069 [P] [US3] Write unit test for auto-save logic in tests/cli/unit/test_autosave.py (test_autosave_every_5_messages, test_autosave_every_2_minutes per SC-007)

**USER APPROVAL REQUIRED**: Review and approve tests T065-T069 before proceeding to implementation

### Implementation for User Story 3

**Session Persistence**

- [x] T070 [P] [US3] Implement conversation history JSONL writer in SessionManager.save_session()
- [x] T071 [P] [US3] Implement JSONL reader in SessionManager.load_session() to reconstruct messages
- [x] T072 [US3] Add SQLite transaction support in save_session() for atomic updates
- [x] T073 [US3] Implement agent configuration save/restore in SQLite agents table

**Session Listing**

- [x] T074 [P] [US3] Implement CommandHandler.handle_session_list() querying SQLite sessions table
- [x] T075 [P] [US3] Format session list with name, last_accessed (human-readable), and agent count
- [x] T076 [US3] Add SQLite indexes on sessions.name and sessions.last_accessed for performance

**Session Resumption**

- [x] T077 [US3] Add --session CLI argument to main.py Click command
- [x] T078 [US3] Implement session resume logic in main() calling SessionManager.load_session()
- [x] T079 [US3] Restore ExecutionHistoryManager from loaded conversation history
- [x] T080 [US3] Recreate PromptChain instances for each loaded agent with correct models
- [x] T081 [US3] Set session.active_agent to last used agent from loaded session

**Session Deletion**

- [x] T082 [P] [US3] Implement CommandHandler.handle_session_delete() with confirmation
- [x] T083 [US3] Delete session from SQLite with CASCADE for agents
- [x] T084 [US3] Remove session directory and all files (history.jsonl, logs)

**Auto-Save**

- [x] T085 [US3] Implement SessionManager.auto_save_if_needed() checking message count and time interval
- [x] T086 [US3] Add Textual set_interval() in PromptChainApp.on_mount() for 2-minute auto-save (SC-007)
- [x] T087 [US3] Track message count since last save and trigger at 5 messages (SC-007)
- [x] T088 [US3] Trigger auto-save before agent switch in handle_agent_use()

**Session Naming**

- [x] T089 [P] [US3] Implement CommandHandler.handle_session_save() with optional rename parameter
- [x] T090 [US3] Add session name validation (1-64 chars, alphanumeric+dashes) in SessionManager

**Checkpoint**: All three user stories should now be independently functional - basic chat, multi-agent, and session persistence all work

---

## Phase 6: User Story 4 - File Operations and Context (Priority: P4)

**Goal**: Users can reference files with @syntax, directories with @dir/, see file contents in context, and have edits applied with confirmation

**Independent Test**: Type `@README.md` in prompt, verify file content included, type `@src/` and verify relevant files discovered

### Tests for User Story 4 (TDD - Write FIRST, Must FAIL) ⚠️

> **CRITICAL**: Write these tests FIRST, get user approval, verify they FAIL, then implement

- [x] T091 [P] [US4] Write contract test for file reference parsing in tests/cli/contract/test_file_reference_parsing.py (test_parse_single_file, test_parse_multiple_files, test_parse_directory)
- [x] T092 [P] [US4] Write contract test for file loading performance in tests/cli/contract/test_file_loading_performance.py (test_file_load_under_500ms per SC-004)
- [x] T093 [P] [US4] Write integration test for file context injection in tests/cli/integration/test_file_context.py (test_file_content_in_prompt, test_agent_sees_file_content, test_binary_file_handling)
- [x] T094 [P] [US4] Write integration test for directory discovery in tests/cli/integration/test_directory_discovery.py (test_relevant_files_found, test_code_files_prioritized, test_max_files_limit)
- [x] T095 [P] [US4] Write unit test for file truncation in tests/cli/unit/test_file_truncation.py (test_large_file_truncated, test_preview_format, test_truncation_indicator)

**USER APPROVAL REQUIRED**: Review and approve tests T091-T095 before proceeding to implementation

### Implementation for User Story 4

**File Reference Parsing**

- [x] T096 [P] [US4] Implement FileReferenceParser.parse_message() with regex FILE_REF_PATTERN to detect @syntax
- [x] T097 [P] [US4] Implement FileReferenceParser.resolve_reference() to resolve paths relative to working_directory
- [x] T098 [P] [US4] Add path validation in resolve_reference() checking existence and permissions
- [x] T099 [US4] Implement graceful error handling for non-existent files (return FileReference with error field)

**File Content Loading**

- [x] T100 [P] [US4] Implement FileReferenceParser.load_file_content() with size-based truncation (100KB limit)
- [x] T101 [P] [US4] Add preview mode for large files: first 500 lines + separator + last 100 lines
- [x] T102 [P] [US4] Implement binary file detection using magic bytes and return metadata string
- [x] T103 [US4] Add UTF-8 decoding with latin-1 fallback for encoding errors

**Directory File Discovery**

- [x] T104 [P] [US4] Implement FileReferenceParser.discover_directory_files() with relevance heuristics
- [x] T105 [P] [US4] Prioritize common code file extensions (.py, .js, .ts, .md, .txt) in discovery
- [x] T106 [P] [US4] Skip hidden files/directories (starting with .) and common ignore patterns (node_modules, __pycache__, .git)
- [x] T107 [US4] Limit directory discovery to max_files (default 20) sorted by relevance

**Message Augmentation**

- [x] T108 [US4] Implement FileReference.to_message_context() formatting file/directory content for LLM
- [x] T109 [US4] Create augment_message_with_references() function to combine message with file contexts
- [x] T110 [US4] Integrate file reference parsing in PromptChainApp message handling before sending to agent
- [x] T111 [US4] Display file references in ChatView with visual indicators (file icon, path)

**Performance Optimization**

- [x] T112 [P] [US4] Implement streaming file reads for large files to avoid loading entire file into memory
- [x] T113 [P] [US4] Add file size check before reading to skip files >10MB (SC-004)

**Checkpoint**: User Stories 1-4 should all work independently - basic chat, multi-agent, persistence, and file references all functional

---

## Phase 7: User Story 5 - Shell Command Execution (Priority: P5)

**Goal**: Users can execute shell commands with !syntax, see output in chat, have long-running commands show progress, and toggle shell mode with !!

**Independent Test**: Type `!ls -la` in chat, verify command executes and output displayed, ask AI to interpret results

### Tests for User Story 5 (TDD - Write FIRST, Must FAIL) ⚠️

> **CRITICAL**: Write these tests FIRST, get user approval, verify they FAIL, then implement

- [x] T114 [P] [US5] Write contract test for shell execution in tests/cli/contract/test_shell_execution.py (test_execute_command, test_capture_stdout, test_capture_stderr, test_return_code)
- [x] T115 [P] [US5] Write contract test for command timeout in tests/cli/contract/test_shell_timeout.py (test_timeout_after_30_seconds, test_timeout_cancellable)
- [x] T116 [P] [US5] Write integration test for shell output in chat in tests/cli/integration/test_shell_output_display.py (test_output_displayed, test_ansi_colors_preserved per FR-026)
- [x] T117 [P] [US5] Write integration test for shell mode toggle in tests/cli/integration/test_shell_mode.py (test_double_bang_toggles_mode, test_consecutive_commands)
- [x] T118 [P] [US5] Write unit test for command parsing in tests/cli/unit/test_shell_parser.py (test_detect_shell_command, test_extract_command_string)

**USER APPROVAL REQUIRED**: Review and approve tests T114-T118 before proceeding to implementation

### Implementation for User Story 5

**Shell Command Execution**

- [x] T119 [P] [US5] Implement ShellExecutor.execute_shell_command() using asyncio.create_subprocess_shell()
- [x] T120 [P] [US5] Add output streaming in execute_shell_command() to capture stdout and stderr asynchronously
- [x] T121 [P] [US5] Implement timeout support with asyncio.wait_for() (default 30 seconds)
- [x] T122 [US5] Add command timeout cancellation handling (kill process on timeout)

**Shell Command Detection**

- [x] T123 [P] [US5] Implement shell command detection in PromptChainApp checking for ! prefix
- [x] T124 [P] [US5] Extract command string from !syntax and pass to ShellExecutor
- [x] T125 [US5] Add working directory context (execute commands in session.working_directory)

**Output Display**

- [x] T126 [P] [US5] Create ShellOutput dataclass in promptchain/cli/shell_executor.py with stdout, stderr, return_code
- [x] T127 [P] [US5] Format shell output for ChatView display with syntax highlighting
- [x] T128 [P] [US5] Preserve ANSI color codes and formatting in shell output display (FR-026)
- [x] T129 [US5] Add shell output to conversation history as Message with command_executed field

**Progress Indicators**

- [x] T130 [P] [US5] Add progress spinner in ChatView during long-running command execution
- [x] T131 [US5] Update progress indicator with elapsed time for commands >5 seconds

**Shell Mode Toggle**

- [x] T132 [P] [US5] Implement !! detection to toggle shell-only mode in PromptChainApp
- [x] T133 [US5] Add shell mode state to PromptChainApp (shell_mode: bool)
- [x] T134 [US5] When in shell mode, execute all inputs as shell commands until !! typed again
- [x] T135 [US5] Update StatusBar to show "Shell Mode" indicator when active

**Security Considerations**

- [x] T136 [P] [US5] Display command to user before execution (confirmation not required, just visibility)
- [x] T137 [US5] Execute commands in session's working_directory (no automatic directory traversal)

**Checkpoint**: All five user stories should now be independently functional - complete CLI feature set

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and overall quality

### Documentation

- [ ] T138 [P] Update CLAUDE.md with CLI usage patterns and examples
- [ ] T139 [P] Create CLI README in promptchain/cli/README.md with architecture overview
- [ ] T140 [P] Add inline documentation to all CLI classes and methods

### Error Handling

- [x] T141 [P] Implement global error handler in PromptChainApp for graceful crash recovery
- [x] T142 [P] Add user-friendly error messages for common failures (API key missing, model not available, file not found)
- [x] T143 Add error logging to JSONL session logs for debugging

### UI/UX Improvements

- [ ] T144 [P] Add command history navigation with Up/Down arrow keys in InputWidget
- [ ] T145 [P] Implement slash command autocomplete in InputWidget with Tab completion
- [ ] T146 [P] Add multi-line input support with Shift+Enter for newlines
- [ ] T147 Add Ctrl+C handling to cancel current operation without exiting (FR-019)

### Performance Optimization

- [x] T148 [P] Add lazy loading for agents (don't initialize PromptChain until first use)
- [x] T149 [P] Implement conversation history pagination in ChatView for 100+ message sessions
- [ ] T150 Run performance benchmarks and validate against targets (SC-001 through SC-012)

### Configuration

- [x] T151 [P] Create default config file ~/.promptchain/config.json with default_model, auto_save_interval, max_file_size
- [x] T152 [P] Add config file loading in SessionManager.__init__()
- [x] T153 Add config validation and user-friendly error messages for invalid config

### Help System

- [x] T154 [P] Implement CommandHandler.handle_help() with general help text
- [x] T155 [P] Add topic-specific help for /help commands, /help agents, /help sessions
- [ ] T156 Create help text formatting in ChatView with examples

### Testing Validation

- [x] T157 Run full integration test suite and verify all user stories pass independently
- [ ] T158 Validate quickstart.md workflows manually (code review, research, quick questions examples)
- [x] T159 Run contract tests to ensure no existing PromptChain APIs were broken

### Code Quality

- [x] T160 [P] Run black formatter on all CLI code
- [x] T161 [P] Run isort on all imports
- [x] T162 [P] Run flake8 linter and fix issues
- [x] T163 [P] Run mypy type checker on CLI code
- [ ] T164 Code review for security issues (path traversal, command injection, etc.)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5)
- **Polish (Phase 8)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories ✅ FULLY INDEPENDENT
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Builds on US1 but can be tested independently ✅ MOSTLY INDEPENDENT
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Requires US1 session infrastructure ⚠️ Depends on US1
- **User Story 4 (P4)**: Can start after Foundational (Phase 2) - Works with US1 conversation flow ⚠️ Depends on US1
- **User Story 5 (P5)**: Can start after Foundational (Phase 2) - Works with US1 conversation flow ⚠️ Depends on US1

### Within Each User Story

- Tests (TDD) MUST be written and approved FIRST, then verified to FAIL before implementation
- Models before services
- Services before UI components
- Core implementation before integration
- Story complete and independently tested before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**:
- T003, T004, T005, T006 can run in parallel (different files)

**Phase 2 (Foundational)**:
- T007, T008, T009, T010 (data models) can run in parallel
- T011, T012, T013, T014 (component skeletons) can run in parallel after models

**User Story Tests** (within each story):
- All test tasks marked [P] can run in parallel (different test files)

**User Story Implementation** (within each story):
- Models marked [P] can run in parallel
- Independent components marked [P] can run in parallel

**Cross-Story Parallelization**:
- US2, US3, US4, US5 can be worked on in parallel by different developers after US1 completes
- With 3 developers: Dev A (US2), Dev B (US4), Dev C (US5) can proceed concurrently

---

## Parallel Example: User Story 1 (Interactive Chat)

```bash
# TDD Phase - All tests can be written in parallel:
Task T018: "Contract test for basic session lifecycle"
Task T019: "Integration test for CLI startup"
Task T020: "Integration test for conversation flow"
Task T021: "Integration test for exit behavior"
Task T022: "Unit test for ExecutionHistoryManager integration"

# After tests approved and verified failing, parallel implementation:
Task T023: "Implement SessionManager.create_session()"
Task T024: "Implement SessionManager.load_session()"
Task T027: "Create PromptChainApp class"
Task T028: "Implement ChatView widget"
Task T029: "Implement InputWidget"
Task T030: "Implement StatusBar widget"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T006)
2. Complete Phase 2: Foundational (T007-T017) - **CRITICAL BLOCKING PHASE**
3. Complete Phase 3: User Story 1 (T018-T042)
   - Write tests FIRST (T018-T022)
   - Get user approval of tests
   - Verify tests FAIL
   - Implement (T023-T042)
   - Verify tests PASS
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo basic interactive chat

**MVP Scope**: At this point you have a working CLI tool - users can chat with AI in terminal with context awareness!

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP - basic chat!)
3. Add User Story 2 → Test independently → Deploy/Demo (multi-agent with different models!)
4. Add User Story 3 → Test independently → Deploy/Demo (session persistence!)
5. Add User Story 4 → Test independently → Deploy/Demo (file references!)
6. Add User Story 5 → Test independently → Deploy/Demo (shell commands!)
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T017)
2. Once Foundational is done, US1 completed first (blocking)
3. After US1 complete:
   - Developer A: User Story 2 (agent management)
   - Developer B: User Story 4 (file references)
   - Developer C: User Story 5 (shell commands)
4. User Story 3 (persistence) completed by available developer
5. Stories integrate and test independently

---

## TDD Workflow (Constitution Principle III)

**CRITICAL**: Test-Driven Development is MANDATORY for this feature

### Red-Green-Refactor Cycle

For EVERY user story phase:

1. **RED**: Write tests FIRST
   - Write all tests for the user story (T018-T022 for US1, T043-T047 for US2, etc.)
   - Get user/stakeholder approval of tests
   - Run tests and verify they FAIL (red)
   - If tests pass before implementation, fix the tests!

2. **USER APPROVAL**: Show failing tests to user
   - Demonstrate that tests capture requirements correctly
   - Get explicit approval before proceeding to implementation

3. **GREEN**: Implement to make tests pass
   - Implement features (T023-T042 for US1, T048-T064 for US2, etc.)
   - Run tests frequently
   - Stop when all tests pass (green)

4. **REFACTOR**: Clean up while keeping tests green
   - Improve code quality
   - Remove duplication
   - Ensure tests still pass

### Test Approval Checkpoints

- ✋ **Checkpoint after T022**: User approves US1 tests before T023
- ✋ **Checkpoint after T047**: User approves US2 tests before T048
- ✋ **Checkpoint after T069**: User approves US3 tests before T070
- ✋ **Checkpoint after T095**: User approves US4 tests before T096
- ✋ **Checkpoint after T118**: User approves US5 tests before T119

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability (US1-US5)
- Each user story should be independently completable and testable
- **TDD is MANDATORY**: Tests written FIRST, approved by user, verified to FAIL, then implement
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Constitution Principle III enforced: No implementation without tests first
- All tasks include exact file paths for clarity
- Performance targets from plan.md baked into contract tests (SC-001 through SC-012)

---

**Total Tasks**: 164 tasks
**Task Count by User Story**:
- Setup (Phase 1): 6 tasks
- Foundational (Phase 2): 11 tasks (BLOCKING)
- User Story 1 (P1 - MVP): 25 tasks (5 tests + 20 implementation)
- User Story 2 (P2): 22 tasks (5 tests + 17 implementation)
- User Story 3 (P3): 26 tasks (5 tests + 21 implementation)
- User Story 4 (P4): 23 tasks (5 tests + 18 implementation)
- User Story 5 (P5): 24 tasks (5 tests + 19 implementation)
- Polish (Phase 8): 27 tasks

**Parallel Opportunities**: 45+ tasks marked [P] can run in parallel
**Independent Testing**: Each user story has 5 dedicated test tasks to verify independent functionality
**Suggested MVP Scope**: Phase 1 + Phase 2 + Phase 3 (User Story 1 only) = 42 tasks for working interactive CLI
