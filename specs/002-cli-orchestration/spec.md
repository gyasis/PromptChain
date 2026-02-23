# Feature Specification: CLI Orchestration Integration

**Feature Branch**: `002-cli-orchestration`
**Created**: 2025-11-18
**Status**: Draft
**Input**: User description: "Integrate CLI with AgentChain orchestration, AgenticStepProcessor multi-hop reasoning, and MCP tool ecosystem for sophisticated agent workflows"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Intelligent Multi-Agent Conversations (Priority: P1)

Users engage in natural language conversations where the system automatically selects the most appropriate specialized agent based on their query context, without requiring manual agent switching.

**Why this priority**: Core value proposition - transforms CLI from simple chat to intelligent orchestration. This is the foundation that all other features build upon. Without automatic routing, users must manually manage agents, defeating the purpose of orchestration.

**Independent Test**: Can be fully tested by sending diverse queries ("analyze this code", "search the web", "write documentation") and verifying correct agent selection without manual `/agent use` commands. Delivers immediate value by simplifying multi-agent workflows.

**Acceptance Scenarios**:

1. **Given** user starts new session, **When** user asks "analyze the authentication patterns in src/", **Then** system automatically routes to analysis agent and provides detailed code analysis
2. **Given** active conversation about code, **When** user asks "now write documentation for this", **Then** system automatically switches to documentation agent without manual command
3. **Given** multi-turn conversation, **When** user asks follow-up question, **Then** system maintains context from previous exchanges and selects appropriate agent

---

### User Story 2 - Complex Multi-Hop Reasoning (Priority: P1)

Users request complex tasks requiring multiple reasoning steps, tool calls, and information synthesis. The system autonomously breaks down the problem, executes necessary research/analysis steps, and synthesizes results without requiring step-by-step user guidance.

**Why this priority**: Demonstrates sophisticated capabilities beyond simple LLM chat. This is what differentiates the platform from basic chat interfaces. Essential for complex workflows like "research this topic, analyze findings, generate recommendations".

**Independent Test**: Can be tested by providing complex objectives like "Research authentication best practices, analyze our current implementation, identify gaps, and recommend improvements". System should autonomously execute multi-step reasoning with tool calls and deliver comprehensive analysis.

**Acceptance Scenarios**:

1. **Given** user provides complex objective, **When** system begins processing, **Then** system autonomously executes 3-8 reasoning steps with appropriate tool calls
2. **Given** multi-step reasoning in progress, **When** step requires external information, **Then** system automatically calls appropriate tools (file search, web search, code analysis) without asking user
3. **Given** reasoning steps complete, **When** results ready, **Then** system synthesizes findings into coherent response addressing original objective

---

### User Story 3 - External Tool Integration (Priority: P2)

Users leverage external capabilities (file system operations, code execution, web search, database queries) through natural language requests. The system discovers available tools, routes requests appropriately, and executes tool calls seamlessly within conversation flow.

**Why this priority**: Expands agent capabilities beyond language generation to actionable operations. Critical for real-world workflows but depends on foundational orchestration (P1). Can be demonstrated independently once routing works.

**Independent Test**: Can be tested by requesting file operations ("list Python files in src/"), web searches ("find latest React documentation"), or code execution ("run the test suite"). System should discover MCP servers, register tools, and execute requests successfully.

**Acceptance Scenarios**:

1. **Given** MCP servers configured, **When** session starts, **Then** system discovers and registers all available tools from connected servers
2. **Given** user requests file operation, **When** agent processes request, **Then** system automatically calls appropriate MCP filesystem tool and returns results
3. **Given** user needs web information, **When** query requires external data, **Then** system calls web search tool and incorporates results into response

---

### User Story 4 - Token-Efficient History Management (Priority: P2)

The system optimizes token usage across multi-agent conversations by applying different history configurations per agent type. Execution/terminal agents operate without history (saving 60% tokens), while research/analysis agents maintain full context.

**Why this priority**: Critical for cost management and scalability but secondary to core functionality. Users benefit from reduced costs/latency but won't notice unless specifically monitoring token usage. Can be implemented and measured independently.

**Independent Test**: Can be tested by conducting multi-agent conversations while monitoring token usage. Compare baseline (all agents with full history) vs optimized (per-agent configs). Should demonstrate 30-60% token reduction without functionality loss.

**Acceptance Scenarios**:

1. **Given** terminal agent executing commands, **When** agent processes request, **Then** system sends request without conversation history, reducing token count by ~60%
2. **Given** research agent analyzing data, **When** agent processes request, **Then** system includes relevant conversation history up to configured token limit (8000 tokens)
3. **Given** multi-agent conversation spanning 20 exchanges, **When** comparing token usage, **Then** optimized configuration uses 30-60% fewer tokens than baseline

---

### User Story 5 - Persistent Workflow State (Priority: P3)

Users work on complex objectives spanning multiple sessions. The system tracks workflow progress, completed steps, and pending tasks. When users return, they can resume exactly where they left off with full context preservation.

**Why this priority**: Valuable for long-running projects but not essential for basic operation. Users can work around this by manually tracking progress. Nice-to-have enhancement that improves user experience for power users.

**Independent Test**: Can be tested by creating workflow with objective, completing some steps, exiting session, restarting, and verifying ability to resume with full context. Should preserve objective, completed steps, and conversation history.

**Acceptance Scenarios**:

1. **Given** user creates workflow with multi-step objective, **When** user completes 3 of 7 steps and exits, **Then** workflow state persists to database with progress markers
2. **Given** saved workflow exists, **When** user resumes session, **Then** system loads workflow state and presents progress summary with remaining steps
3. **Given** workflow resumed, **When** user continues work, **Then** system picks up from last completed step with full conversation context

---

### User Story 6 - Specialized Agent Templates (Priority: P3)

Users quickly create sophisticated agents using pre-configured templates. Templates showcase platform capabilities with optimized instructions, tool configurations, and history settings for common use cases (research, coding, analysis, terminal operations).

**Why this priority**: Convenience feature that reduces setup time but not core functionality. Users can manually configure agents with same capabilities. Primarily benefits new users exploring platform features.

**Independent Test**: Can be tested by creating agents from templates and verifying pre-configured settings match documented specifications. Each template should work immediately without additional configuration.

**Acceptance Scenarios**:

1. **Given** user executes template creation command, **When** specifying "researcher" template, **Then** system creates agent with AgenticStepProcessor, web search tools, and multi-hop reasoning capabilities
2. **Given** coder template agent created, **When** user requests code operation, **Then** agent has pre-registered file operation tools and code execution capabilities
3. **Given** terminal template agent created, **When** processing commands, **Then** agent operates with history disabled for maximum token efficiency

---

### Edge Cases

- **Concurrent Agent Execution**: What happens when user requests broadcast mode operation across 5 agents simultaneously? System should manage execution threads, collect results, and synthesize without deadlocks or resource exhaustion.
- **MCP Server Failures**: How does system handle unavailable MCP servers at session start? System should gracefully degrade, log missing servers, continue with available tools, and notify user of limited capabilities.
- **Token Limit Exceeded**: What happens when conversation history + new request exceeds model token limit? System should automatically truncate history using configured strategy (oldest_first or keep_last) and continue operation.
- **Conflicting Tool Names**: How does system handle MCP tool names that conflict with local functions? System should automatically prefix MCP tools with server ID (`mcp_filesystem_read_file`) to prevent naming collisions.
- **AgenticStepProcessor Max Steps**: What happens when multi-hop reasoning exceeds max_internal_steps without completing objective? System should return partial results with explanation of completion status and allow user to adjust parameters or refine objective.
- **Session State Corruption**: How does system recover from corrupted SQLite session database? System should detect corruption, backup corrupted file, initialize fresh database, and notify user to restore from recent backup if needed.
- **Circular Agent Routing**: What happens if router logic creates infinite loop selecting same agent repeatedly? System should detect routing cycles after 3 identical selections and break loop with error message.

## Requirements *(mandatory)*

### Functional Requirements

#### Core Orchestration (P1)

- **FR-001**: System MUST replace individual agent instances with single AgentChain orchestrator managing all agents within a session
  - **Success Criteria**: Router achieves ≥95% agent selection accuracy (correct agent for task type); Router decision latency <100ms for sessions with ≤10 agents; Zero routing failures in single-agent mode (100% pass-through)
- **FR-002**: System MUST support router execution mode for automatic agent selection based on query analysis
  - **Success Criteria**: 95th percentile routing latency <100ms (measured from prompt submission to agent selection); Router handles up to 20 concurrent agents without performance degradation; Custom decision prompts validated against schema before use
- **FR-003**: System MUST support pipeline, round-robin, and broadcast execution modes for different workflow patterns
- **FR-004**: System MUST configure router with decision prompt templates that analyze user input, conversation history, and agent descriptions to select optimal agent
- **FR-005**: System MUST maintain single AgentChain instance per session with persistent cache configuration tied to session identifier

#### Multi-Hop Reasoning (P1)

- **FR-006**: System MUST support AgenticStepProcessor instances within agent instruction chains for complex multi-step reasoning
- **FR-007**: System MUST allow per-agent configuration of reasoning objectives, max internal steps (3-10 range), and step completion criteria
- **FR-008**: System MUST enable AgenticStepProcessor to access full tool ecosystem (local functions + MCP tools) during internal reasoning loops
- **FR-009**: System MUST track agentic reasoning progress and expose step-by-step outputs for transparency and debugging
- **FR-010**: System MUST detect objective completion or exhaustion of max steps and return results with completion status

#### Agent Configuration (P1)

- **FR-011**: System MUST extend agent configuration model to support instruction chains combining strings, functions, and AgenticStepProcessor instances
- **FR-012**: System MUST store agent descriptions used by router for decision-making during automatic agent selection
- **FR-013**: System MUST allow per-agent configuration of: model selection, instruction pipelines, tool access, and history settings
- **FR-014**: Users MUST be able to create, list, switch between, and delete agents using existing `/agent` commands
- **FR-015**: System MUST validate agent configurations at creation time to prevent runtime errors from misconfigured instruction chains

#### MCP Tool Integration (P2)

- **FR-016**: System MUST connect to configured MCP servers (filesystem, code execution, web search) at session initialization
- **FR-017**: System MUST discover and register tools from all connected MCP servers using automatic tool discovery
- **FR-018**: System MUST prefix MCP tool names with server identifier to prevent conflicts with local functions
  - Prefix format: `mcp_{server_id}_{tool_name}` (e.g., `mcp_filesystem_read`)
  - Local tools registered without prefix (e.g., `search_files`)
  - **Conflict resolution**: If local tool name matches MCP prefixed name, reject MCP tool with warning
    - Example: Local tool `mcp_filesystem_read` exists → MCP tool registration fails with error
    - Log warning: "MCP tool 'mcp_filesystem_read' from server 'filesystem' conflicts with local tool, skipping"
  - **Duplicate MCP tools**: If two MCP servers provide same tool name, use first-registered server
    - Example: `filesystem_server_1.read` and `filesystem_server_2.read` both register as `mcp_filesystem_server_1_read` and `mcp_filesystem_server_2_read` (no conflict due to server_id prefix)
- **FR-019**: System MUST manage MCP server lifecycle including connection startup, health checking, and graceful shutdown
- **FR-020**: Users MUST be able to list available tools, add/remove MCP servers, and view tool schemas using `/tools` commands
- **FR-021**: System MUST handle MCP server connection failures gracefully by logging errors, continuing with available tools, and notifying user

#### History Management (P2)

- **FR-022**: System MUST implement per-agent history configurations using agent_history_configs parameter from AgentChain v0.4.2
- **FR-023**: System MUST support disabling history entirely for terminal/execution agents to save token costs
- **FR-024**: System MUST support configurable history limits (max_tokens, max_entries) per agent with automatic truncation strategies
- **FR-025**: System MUST use ExecutionHistoryManager with structured entry types (user_input, agent_output, tool_call, tool_result)
- **FR-026**: System MUST support history filtering by entry type and source for precise context control
- **FR-027**: System MUST automatically apply token-aware truncation when conversation history exceeds configured limits

#### Workflow State (P3)

- **FR-028**: System MUST persist workflow state (objective, completed steps, pending steps) to session SQLite database
- **FR-029**: Users MUST be able to create workflows with explicit objectives using `/workflow create <objective>` command
- **FR-030**: Users MUST be able to view workflow progress, resume incomplete workflows, and track completion status
- **FR-031**: System MUST integrate workflow objectives with AgenticStepProcessor goals for automatic step execution
- **FR-032**: System MUST preserve workflow state across session restarts with full conversation context

#### Agent Templates (P3)

- **FR-033**: System MUST provide pre-configured agent templates for common use cases: researcher, coder, analyst, terminal
- **FR-034**: Users MUST be able to create agents from templates using `/agent create-from-template <template_name> <agent_name>`
- **FR-035**: Researcher template MUST include AgenticStepProcessor with web search tools and max_internal_steps=8
- **FR-036**: Coder template MUST include file operation tools, code execution capabilities, and validation instruction chains
- **FR-037**: Terminal template MUST configure history as disabled for maximum token efficiency
- **FR-038**: System MUST allow customization of template-created agents after instantiation

### Key Entities

- **AgentChain Instance**: Single orchestrator managing all agents in session, configured with execution mode (router/pipeline/round-robin/broadcast), cache settings tied to session ID, and global history settings
- **Agent Configuration**: Individual agent definition including model selection, instruction chain (strings/functions/AgenticStepProcessor), tool access permissions, history settings, and routing metadata (description for router)
- **Instruction Chain**: Ordered sequence of processing steps including string templates processed by LLMs, Python functions executed directly, and AgenticStepProcessor instances for multi-hop reasoning
- **AgenticStepProcessor Instance**: Multi-hop reasoning component with objective definition, max internal steps limit (3-10), tool access configuration, and step tracking state
- **MCP Server Connection**: External tool server reference including server ID, connection type (stdio/http), command/URL, lifecycle state (connected/disconnected), and discovered tool registry
- **Tool Registration**: Tool definition including name (with mcp_ prefix for external tools), schema (parameters/descriptions), source (local function or MCP server ID), and execution handler
- **History Configuration**: Per-agent history settings including enabled flag, max_tokens limit, max_entries limit, truncation strategy (oldest_first/keep_last), included entry types filter, and excluded sources filter
- **Execution History Entry**: Structured conversation record including entry type (user_input/agent_output/tool_call/tool_result), content, timestamp, source identifier, and associated metadata
- **Workflow State**: Persistent multi-session objective tracker including workflow ID, objective description, completed steps list, pending steps list, current step index, and associated conversation history
- **Session Cache**: SQLite database persisting agent configurations, conversation history, workflow states, and session metadata for resumption across restarts

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Multi-agent conversations demonstrate automatic agent routing with 95% selection accuracy (correct agent chosen for query type) without manual `/agent use` commands
- **SC-002**: Complex tasks utilizing AgenticStepProcessor complete multi-hop reasoning with 3-8 autonomous steps including appropriate tool calls and synthesis
- **SC-003**: Per-agent history configuration reduces token consumption by 30-60% compared to baseline (all agents with full history) while maintaining response quality
- **SC-004**: MCP tool integration enables successful execution of external operations (file system access, web search, code execution) within 2 seconds per tool call
- **SC-005**: Session persistence allows users to exit and resume workflows with complete context restoration including conversation history and workflow progress
- **SC-006**: Agent template creation (using `/agent create-from-template`) produces functional specialized agents in under 5 seconds with zero additional configuration required
- **SC-007**: System maintains stable operation during concurrent agent execution (broadcast mode with 5+ agents) without memory leaks or thread deadlocks
- **SC-008**: Infrastructure utilization increases from current 15% to 85%+ as measured by active usage of AgentChain, AgenticStepProcessor, MCPHelper, and ExecutionHistoryManager components

## Assumptions

- **Model API Access**: Users have configured API keys for at least one LLM provider (OpenAI, Anthropic, Google) in environment variables or .env file
- **MCP Server Availability**: External MCP servers (filesystem, code execution, web search) are installed and accessible via stdio or http protocols
- **SQLite Support**: Runtime environment supports SQLite 3 for session persistence and conversation caching
- **Token Counting**: tiktoken library is available for accurate token counting across different model families for history management
- **Async Runtime**: Python 3.8+ asyncio support is available for concurrent agent execution and MCP server communication
- **File System Access**: CLI has read/write permissions to session directory (~/.promptchain/sessions/) for cache and history storage
- **Network Connectivity**: System has network access for MCP server connections and LLM API calls
- **Resource Limits**: System has sufficient memory (2GB+ recommended) for concurrent agent execution with history buffers
- **Default Agent**: If user doesn't create custom agents, system provides default agent with basic capabilities as fallback

## Dependencies

- **AgentChain (v0.4.2+)**: Core orchestration with router mode, per-agent history configs, cache persistence
- **AgenticStepProcessor**: Multi-hop reasoning with tool calling and objective completion detection
- **MCPHelper**: MCP server lifecycle management, tool discovery, and execution
- **ExecutionHistoryManager**: Token-aware history management with structured entry types and filtering
- **LiteLLM**: Unified LLM API access across providers
- **Session Manager**: Existing SQLite-based session persistence extended for workflow state
- **Command Handler**: Existing slash command parsing extended for new `/tools` and `/workflow` commands
- **TUI Application**: Existing Textual interface updated to display agent routing decisions and reasoning steps

## Out of Scope

- **Real-time Collaboration**: Multi-user session sharing with concurrent editing not included
- **Custom LLM Endpoints**: Support for self-hosted LLM servers beyond Ollama (already supported via LiteLLM)
- **Voice/Audio Input**: Speech-to-text integration for voice commands
- **Browser-Based UI**: Web interface alternative to terminal TUI (CLI remains terminal-only)
- **Mobile Clients**: Native mobile apps for iOS/Android
- **Plugin Marketplace**: Third-party agent template distribution system
- **Usage Analytics Dashboard**: Graphical visualization of token usage, agent performance metrics
- **A/B Testing Framework**: Automated comparison of different agent configurations or routing strategies
- **Distributed Execution**: Running agents across multiple machines/containers for scalability
- **LLM Fine-Tuning**: Custom model training or fine-tuning capabilities
- **Version Control Integration**: Automatic git operations tied to agent actions
- **Database Query Builder**: Visual interface for constructing complex database queries
- **Cost Budgeting**: Hard limits on token usage or API spending per session
