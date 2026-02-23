# Feature Specification: PromptChain CLI Agent Interface

**Feature Branch**: `001-cli-agent-interface`
**Created**: 2025-11-16
**Status**: Draft
**Input**: User description: "Build a CLI agent interface for PromptChain similar to Claude Code, Aider, Goose CLI, and Gemini CLI with interactive sessions, agent creation, and session management"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Interactive Chat Session (Priority: P1)

Users can start an interactive terminal session with PromptChain where they have a persistent conversation with AI agents, similar to how they would use Claude Code or Gemini CLI. The session maintains context across multiple exchanges and allows natural language interaction with file system awareness.

**Why this priority**: This is the core value proposition - transforming PromptChain from a library into a usable CLI tool. Without this, users cannot experience the CLI at all.

**Independent Test**: Can be fully tested by running `promptchain` command, typing a question, receiving a response, and verifying that follow-up questions maintain context.

**Acceptance Scenarios**:

1. **Given** user is in any directory, **When** they run `promptchain` command, **Then** an interactive chat session starts with a welcome message and prompt
2. **Given** user is in an interactive session, **When** they type "What files are in this directory?", **Then** system responds with directory contents
3. **Given** user asks a follow-up question, **When** referencing previous conversation context, **Then** system responds with awareness of prior exchanges
4. **Given** user wants to exit, **When** they type `/exit` or press Ctrl+D, **Then** session terminates gracefully with a goodbye message

---

### User Story 2 - Agent Creation and Management (Priority: P2)

Users can create, configure, and switch between different AI agents within a session. Each agent can be configured with ANY AI model (GPT-4, Claude, Gemini, local models, etc.), allowing for model-agnostic agent creation. Each agent can have specialized capabilities (coding, research, documentation) and users can invoke specific agents for specific tasks.

**Why this priority**: Enables the multi-agent orchestration AND model flexibility that differentiates PromptChain from simpler CLI tools. Users can mix and match models based on task requirements (e.g., fast local model for simple tasks, powerful cloud model for complex reasoning). Once basic chat works (P1), users need specialized agents with different model backends for different tasks.

**Independent Test**: Can be fully tested by creating agents with different models `/agent create coding --model gpt-4` and `/agent create fast --model ollama/llama2`, verifying they appear in `/agent list` with model info, switching between them, and testing that they use their respective models.

**Acceptance Scenarios**:

1. **Given** user is in a session, **When** they run `/agent create research --description "Research specialist" --model claude-3-opus`, **Then** a new agent named "research" is created with Claude 3 Opus model
2. **Given** multiple agents with different models exist, **When** user runs `/agent list`, **Then** all agents are displayed with their names, descriptions, and configured models
3. **Given** user types `/agent use coding`, **When** they ask a question, **Then** the "coding" agent responds using its configured model instead of the default
4. **Given** user wants to remove an agent, **When** they run `/agent delete research`, **Then** agent is removed from the session
5. **Given** user creates an agent without specifying a model, **When** agent is created, **Then** it uses the default powerful model configured for the CLI

---

### User Story 3 - Session Persistence and Resumption (Priority: P3)

Users can save their conversation sessions at any point and resume them later, preserving full conversation history, agent configurations, and working directory context. This enables long-running projects that span multiple work sessions.

**Why this priority**: Critical for professional workflows but not needed for initial evaluation. Users can still have valuable single-session experiences with P1+P2.

**Independent Test**: Can be fully tested by having a conversation, running `/session save my-project`, exiting, running `promptchain --session my-project`, and verifying the conversation history is restored.

**Acceptance Scenarios**:

1. **Given** user has an active conversation, **When** they run `/session save feature-x`, **Then** session is saved with name "feature-x" and confirmation message displayed
2. **Given** saved sessions exist, **When** user runs `/session list`, **Then** all saved sessions are displayed with names and timestamps
3. **Given** user runs `promptchain --session feature-x`, **When** session loads, **Then** full conversation history and agents are restored
4. **Given** user wants to remove old sessions, **When** they run `/session delete old-project`, **Then** session data is permanently removed

---

### User Story 4 - File Operations and Context (Priority: P4)

Users can reference specific files or directories in their prompts, have the CLI read file contents automatically, and request file edits that are applied with user confirmation. This provides seamless integration with code and documentation files.

**Why this priority**: Enhances productivity for coding tasks but not essential for initial CLI functionality. Can be added after core chat and agent features work.

**Independent Test**: Can be fully tested by typing `@src/main.py` in a prompt and verifying the file contents are included in the conversation context, then requesting an edit and confirming it's applied.

**Acceptance Scenarios**:

1. **Given** user types `@file.txt` in a prompt, **When** message is sent, **Then** file contents are automatically read and included in conversation
2. **Given** user types `@src/` (directory), **When** message is sent, **Then** relevant files in directory are discovered and included
3. **Given** AI suggests a file edit, **When** user is prompted for confirmation, **Then** user can approve/reject before changes are applied
4. **Given** file edit is approved, **When** applied, **Then** file is modified and change is logged to session history

---

### User Story 5 - Shell Command Execution (Priority: P5)

Users can execute shell commands directly from the chat interface and have the output fed back into the conversation context. This enables workflows like "run the tests and fix any failures" without leaving the chat.

**Why this priority**: Nice-to-have feature that streamlines certain workflows but not core to CLI agent functionality.

**Independent Test**: Can be fully tested by typing `!ls -la` in the chat, verifying the command executes and output is displayed, and asking the AI to interpret the results.

**Acceptance Scenarios**:

1. **Given** user types `!git status`, **When** command executes, **Then** git status output is displayed in the chat
2. **Given** user asks "What files changed?", **When** AI needs current git state, **Then** AI can request shell command execution
3. **Given** a long-running command (e.g., `!npm test`), **When** it's executing, **Then** user sees a progress indicator
4. **Given** user wants command-only mode, **When** they type `!!`, **Then** session switches to shell mode until `!!` typed again

---

### Edge Cases

- What happens when user tries to resume a session that doesn't exist? Show friendly error with list of available sessions
- How does system handle file reference to non-existent file? Display clear error indicating file not found and prompt user to check path
- What happens when multiple agents are created with the same name? Prevent duplicate names or auto-append number (agent-2, agent-3)
- How does system handle interrupted sessions (crash/network loss)? Auto-save session state every N messages for recovery
- What happens when working directory changes mid-session? Update context automatically and notify user of directory change
- How does system handle very large files referenced with @? Implement size limits and show preview instead of full content for files >100KB
- What happens when shell command takes too long or hangs? Implement timeout (default 30s) with option to continue waiting or cancel
- How does system handle terminal window resize? Automatically reflow content to new dimensions

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a `promptchain` command that launches an interactive terminal session
- **FR-002**: System MUST display a welcome message and prompt when session starts
- **FR-003**: System MUST maintain conversation history within a session for context awareness
- **FR-004**: System MUST support `/command` syntax for session management and agent operations
- **FR-005**: System MUST allow users to create named agents with optional descriptions and model selection
- **FR-006**: System MUST support model-agnostic agent creation (GPT-4, Claude, Gemini, Ollama, any LiteLLM-compatible model)
- **FR-007**: System MUST allow users to list all available agents in the current session with their configured models
- **FR-008**: System MUST allow users to switch between agents during a conversation
- **FR-009**: System MUST allow users to delete agents they've created
- **FR-010**: System MUST use a default powerful model for CLI interactions when no agent model is specified
- **FR-011**: System MUST save conversation sessions with user-specified names
- **FR-012**: System MUST list all saved sessions with timestamps
- **FR-013**: System MUST restore full conversation history and agent configurations when resuming a saved session
- **FR-014**: System MUST support file reference syntax `@file.txt` to include file contents in prompts
- **FR-015**: System MUST support directory reference syntax `@directory/` to discover relevant files
- **FR-016**: System MUST prompt users for confirmation before applying file edits
- **FR-017**: System MUST execute shell commands prefixed with `!` and display output
- **FR-018**: System MUST provide a shell mode toggle with `!!` for consecutive command execution
- **FR-019**: System MUST handle Ctrl+C without terminating session (cancel current operation only)
- **FR-020**: System MUST handle Ctrl+D or `/exit` to gracefully terminate session
- **FR-021**: System MUST preserve working directory context across conversation exchanges
- **FR-022**: System MUST support command history navigation with up/down arrow keys
- **FR-023**: System MUST provide help documentation via `/help` command
- **FR-024**: System MUST display available slash commands with `/help commands`
- **FR-025**: System MUST support multi-line input when users need to paste code or complex prompts
- **FR-026**: System MUST preserve ANSI color codes and formatting in shell command output
- **FR-027**: System MUST auto-save session state periodically for crash recovery

### Key Entities

- **Session**: Represents a persistent conversation instance with unique name, creation timestamp, conversation history (user inputs and AI responses), active agents, working directory, and auto-save state
- **Agent**: Represents an AI agent configuration with unique name, description, system prompt/capabilities, model selection, and usage statistics
- **Message**: Represents a single exchange in conversation with role (user/assistant/system), content (text/file references), timestamp, agent that generated it (for assistant messages), and associated file operations or commands
- **File Reference**: Represents a file or directory mentioned in conversation with absolute path, content snapshot (for referenced files), modification timestamp, and inclusion method (@syntax, shell output, or agent request)
- **Command**: Represents a slash command execution with command name, arguments, execution timestamp, result/output, and error state (if failed)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can start an interactive session and have a basic conversation within 10 seconds of running `promptchain`
- **SC-002**: Users can create and switch between 3 different agents in under 1 minute
- **SC-003**: Sessions save in under 2 seconds and resume with full history in under 3 seconds
- **SC-004**: File references (@syntax) load and include file contents in under 500ms for files under 10MB
- **SC-005**: Shell commands execute and return output within expected command duration + 100ms overhead
- **SC-006**: 90% of users successfully complete their first conversation without consulting documentation
- **SC-007**: Session auto-save triggers every 5 messages or 2 minutes, whichever comes first
- **SC-008**: Users can navigate command history with arrow keys and recall any previous command instantly
- **SC-009**: Multi-line input supports pasting up to 10,000 characters without data loss
- **SC-010**: Terminal UI remains responsive during AI response generation with progress indicators
- **SC-011**: System gracefully handles 100+ message conversations without performance degradation
- **SC-012**: Help documentation is accessible within 2 keystrokes (/, h, enter) from any state

### Assumptions

1. Users have Python 3.8+ installed and in their PATH
2. Users have basic terminal/command-line familiarity
3. Users have configured API keys for LLM providers (OpenAI, Anthropic, etc.) via environment variables or config files
4. Users operate on Unix-like systems (Linux, macOS) or Windows with appropriate shell
5. Terminal supports ANSI escape codes for colors and formatting
6. Users have write permissions in their home directory for session storage
7. Default model is configured in existing PromptChain configuration (or uses sensible defaults)
8. File system operations respect existing file permissions and user privileges
9. Shell commands execute in the user's default shell environment
10. Network connectivity is available for LLM API calls (no offline mode initially)

## Advanced Extensions and Plugins *(optional)*

The following advanced techniques represent **optional plugins and extensions** that can be added to the CLI after core functionality is established. These are NOT core features but extensibility points for advanced users.

### Extension 1: TOON (Token-Oriented Object Notation) Support

**What it is**: TOON is a token-efficient data format (November 2024) designed specifically for LLM-native workflows, reducing token usage by 30-60% compared to JSON while maintaining human readability.

**Use case for CLI**: When agents need to exchange structured data (configuration files, agent state, tool schemas), TOON format can reduce token costs and improve response times. Particularly valuable for multi-agent coordination where agents communicate frequently.

**Plugin behavior**:
- Users can enable TOON mode with `/toon enable`
- Agent-to-agent data exchange automatically uses TOON format instead of JSON
- Session state files can be saved in TOON format (`.toon` extension)
- Tool schemas and responses can optionally use TOON format to reduce token overhead

**Libraries**: Python TOON libraries (python-toon, pytoon) available on GitHub (November 2024)

**Example benefit**: Multi-agent workflow with 5 agents exchanging state → 150K tokens reduced to ~75K tokens

---

### Extension 2: APReL (Active Preference-based Reward Learning) Integration

**What it is**: APReL is Stanford's library for active preference-based reward learning (arXiv 2021, actively maintained 2024), enabling agents to learn user preferences through minimal queries rather than extensive training data.

**Use case for CLI**: The CLI can learn user's coding style preferences, documentation preferences, or workflow preferences by asking minimal clarifying questions. Over time, agents adapt to user's demonstrated preferences.

**Plugin behavior**:
- `/preferences enable` activates preference learning mode
- When agent encounters ambiguous choices (code style, naming conventions, file organization), it queries user: "Option A or Option B?"
- APReL algorithm learns from binary preference comparisons to build user preference model
- Future sessions automatically apply learned preferences without asking

**Example flow**:
```
Agent: "For this function, prefer verbose names (get_user_data_from_database) or concise (get_data)?"
User: "Verbose"
Agent: [Records preference, applies to future code generation]
```

**Libraries**: APReL available at github.com/Stanford-ILIAD/APReL, supports OpenAI Gym integration

**Example benefit**: After 10-15 preference queries, agent generates code matching user's style 85%+ of the time

---

### Extension 3: APRICOT (Active Preference Learning with Constraint-Aware Planning)

**What it is**: APRICOT is Stanford/Cornell research (2024) combining LLM-based Bayesian active preference learning with constraint-aware task planning. Addresses ambiguity in user intent while respecting environmental constraints.

**Use case for CLI**: When users request complex refactorings or multi-file changes, APRICOT helps the agent understand user intent through minimal questions while ensuring generated plans respect project constraints (dependencies, type safety, API contracts).

**Plugin behavior**:
- `/apricot enable` activates constraint-aware planning mode
- Agent converts user's natural language request into structured plan
- Before executing, agent identifies ambiguities and queries user for clarification (2-3 questions average)
- Generated plan respects discovered constraints (import dependencies, type signatures, existing patterns)
- Achieves 96%+ feasible plan success rate

**Example flow**:
```
User: "Refactor authentication system to use JWT"
Agent: [Analyzes codebase constraints]
Agent: "Should JWT tokens be stored in localStorage or httpOnly cookies?"
User: "httpOnly cookies"
Agent: [Generates constraint-aware plan respecting CORS, security headers, existing auth flow]
```

**Libraries**: Research implementation available at portal.cs.cornell.edu/apricot, arXiv:2410.19656

**Example benefit**: Reduces invalid refactoring attempts by 40%, cuts clarification questions from 8-10 to 2-3

---

### Extension 4: Deep Agent Patterns (Multi-Hop Reasoning & Tool Discovery)

**What it is**: Advanced agent orchestration patterns (2024-2025) enabling agents to dynamically discover and chain tools, perform multi-step reasoning with memory folding, and learn from simulated environments.

**Use case for CLI**: For complex research or debugging tasks requiring multiple tool calls and iterative refinement, deep agent patterns enable autonomous tool discovery and adaptive reasoning.

**Plugin behavior**:
- `/deepagent enable` activates advanced reasoning mode
- Agent can search for and register new tools dynamically (MCP servers, local scripts, APIs)
- Uses "memory folding" to compress episodic history when context grows too large
- Supports multi-hop reasoning: "find bug → identify root cause → search similar issues → propose fix → validate"

**Example flow**:
```
User: "Debug the failing test in authentication module"
Agent: [Autonomous reasoning loop]
  1. Runs test, captures failure
  2. Searches codebase for auth logic
  3. Discovers auth token is undefined
  4. Searches for where token should be set
  5. Finds missing initialization in setup()
  6. Proposes fix with explanation
```

**Patterns**: ToolPO (reinforcement learning for tool selection), Autonomous Memory Folding, ReAct framework

**Example benefit**: Complex debugging tasks complete in 1 session instead of 3-5 back-and-forth exchanges

---

### Extension 5: MCP Code Execution (Token Reduction)

**What it is**: Anthropic's Model Context Protocol code execution pattern (2024) where agents write code to interact with tools instead of making individual tool calls, reducing token usage from 150K to ~2K for complex workflows.

**Use case for CLI**: When agent needs to process large datasets, chain multiple API calls, or perform complex data transformations, code execution mode drastically reduces token overhead.

**Plugin behavior**:
- `/mcp-exec enable` activates code execution mode
- Instead of serializing tool call results in context, agent writes Python code that executes in sandbox
- Only final results returned to conversation context (not intermediate steps)
- Supports reusable "skills" - code snippets saved for future sessions

**Example comparison**:
```
Traditional: [Call API] → [150KB JSON] → [Filter] → [Transform] → [Aggregate] = 150K tokens
Code Exec:   [Write Python code] → [Execute in sandbox] → [Return summary] = 2K tokens
```

**Libraries**: MCP servers, code sandbox execution environment

**Example benefit**: 98.7% token reduction for data processing workflows, 5-10x faster responses

---

### Plugin Architecture Requirements

**FR-EXT-001**: System SHOULD provide a plugin registry where extensions can be installed via `/plugin install <name>`

**FR-EXT-002**: System SHOULD allow users to enable/disable plugins per session with `/plugin enable <name>` and `/plugin disable <name>`

**FR-EXT-003**: System SHOULD store plugin configurations separately from core session state

**FR-EXT-004**: System SHOULD display active plugins in session status with `/status` command

**FR-EXT-005**: System SHOULD provide plugin compatibility warnings if plugin requires features not available in current PromptChain version

**FR-EXT-006**: System SHOULD allow plugins to register custom slash commands (e.g., `/toon`, `/preferences`, `/apricot`)

**FR-EXT-007**: System SHOULD provide plugin API for extensions to hook into agent lifecycle (pre-prompt, post-response, tool-calling)

**FR-EXT-008**: System SHOULD isolate plugin failures to prevent core CLI crashes

**FR-EXT-009**: System SHOULD track plugin usage statistics (invocations, token savings, preference accuracy) for user insight

**FR-EXT-010**: System SHOULD support plugin versioning and compatibility declarations
