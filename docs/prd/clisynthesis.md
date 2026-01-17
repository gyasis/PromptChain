 Plan: Upgrade CLI to Leverage Full PromptChain Infrastructure

 Phase 1: Core AgentChain Integration (CRITICAL)

 Replace individual PromptChain instances with AgentChain orchestrator

 1. Refactor Agent Management (promptchain/cli/tui/app.py)
   - Replace self.agent_chain: Optional[PromptChain] with self.agent_chain: Optional[AgentChain]
   - Modify _get_or_create_agent_chain() to return single AgentChain instance managing all agents
   - Configure with router mode for automatic agent selection
 2. Implement Router Configuration
   - Add decision prompt templates for intelligent agent selection
   - Configure execution modes (router, pipeline, round-robin, broadcast)
   - Set up cache_config for persistent sessions
   - Enable auto_include_history with per-agent history configs (v0.4.2)
 3. Update Agent Creation (promptchain/cli/command_handler.py)
   - Modify /agent create to register agents within AgentChain
   - Store agent descriptions for router decision-making
   - Configure specialized instructions per agent type

 Expected Outcome: Single AgentChain orchestrating all agents with automatic routing

 ---
 Phase 2: AgenticStepProcessor Integration (CRITICAL)

 Add multi-hop reasoning capabilities for complex tasks

 1. Add AgenticStep Support
   - Create agent templates with AgenticStepProcessor for research, analysis, debugging
   - Configure max_internal_steps and objectives per agent type
   - Integrate with AgentChain instruction chains
 2. Update Agent Configuration Model (promptchain/cli/models/agent_config.py)
   - Add fields: instructions: List[Union[str, Callable, AgenticStepProcessor]]
   - Add fields: objectives: Optional[str], max_reasoning_steps: int
   - Store agent-specific tool configurations
 3. Implement Complex Instruction Chains
   - Replace ["{input}"] with multi-step instruction pipelines
   - Example: ["Analyze: {input}", agentic_step, "Synthesize: {input}"]

 Expected Outcome: Agents capable of multi-hop reasoning with internal tool calling loops

 ---
 Phase 3: MCP Tool Ecosystem Integration (HIGH PRIORITY)

 Connect external tool servers for file ops, code execution, etc.

 1. Add MCP Configuration (promptchain/cli/session_manager.py)
   - Add mcp_servers field to Session model
   - Default MCP servers: filesystem, code_execution, web_search
   - Per-session MCP server customization
 2. Integrate MCPHelper
   - Pass mcp_servers config to AgentChain/PromptChain instances
   - Enable tool discovery and registration
   - Handle MCP server lifecycle (startup/shutdown)
 3. Add Tool Management Commands
   - /tools list - Show available MCP and local tools
   - /tools add <server> - Add MCP server to session
   - /tools remove <server> - Remove MCP server

 Expected Outcome: Agents can call external tools for file operations, code execution, web search

 ---
 Phase 4: Advanced History Management (MEDIUM PRIORITY)

 Implement token-efficient per-agent history configuration

 1. Configure Per-Agent History (v0.4.2 feature)
   - Implement agent_history_configs in AgentChain setup
   - Terminal/execution agents: {"enabled": False} (save 60% tokens)
   - Research/analysis agents: {"enabled": True, "max_tokens": 8000}
   - Documentation agents: {"enabled": True, "max_tokens": 6000}
 2. Add Structured Entry Types
   - Use ExecutionHistoryManager's entry types: tool_call, tool_result, agent_output
   - Implement filtering by type and source
   - Add metadata tracking for tools and workflows
 3. Enhance History Commands
   - /history filter --types user_input,agent_output
   - /history export --format full_json
   - /history stats - Show token usage, entry counts

 Expected Outcome: 30-60% token savings per multi-agent conversation turn

 ---
 Phase 5: Persistent Workflow State (MEDIUM PRIORITY)

 Track multi-turn objectives and plans across sessions

 1. Add Workflow State Model
   - Create WorkflowState model with: current_objective, completed_steps, pending_steps
   - Store workflow state in session SQLite database
   - Resume workflows across session restarts
 2. Implement Workflow Commands
   - /workflow create <objective> - Start new workflow with objective
   - /workflow status - Show progress on current workflow
   - /workflow resume - Continue previous workflow
 3. Integrate with AgenticStepProcessor
   - Workflow objectives drive AgenticStepProcessor goals
   - Track step completion and update workflow state
   - Persist agentic reasoning chains

 Expected Outcome: Complex multi-session workflows with persistent objectives

 ---
 Phase 6: Specialized Agent Templates (MEDIUM PRIORITY)

 Pre-configured agents showcasing library capabilities

 1. Create Agent Templates
   - Researcher: AgenticStepProcessor + MCP web_search + multi-hop reasoning
   - Coder: Tool calling for file ops + code execution + validation loops
   - Analyst: ExecutionHistoryManager with full context + synthesis patterns
   - Terminal: No history (token efficient) + shell command execution
 2. Add Template Command
   - /agent create-from-template <template_name> <agent_name>
   - Pre-configured instructions, tools, and history settings
   - Customizable after creation

 Expected Outcome: One-command creation of sophisticated agents

 ---
 Phase 7: Function and Tool Registration (MEDIUM PRIORITY)

 Enable local function calling and tool chains

 1. Add Tool Registration
   - chain.register_tool_function() support in CLI
   - Pre-register common tools: file search, code analysis, shell execution
   - Custom tool registration via commands
 2. Implement Tool Calling Patterns
   - Multi-step chains with function calls
   - Tool result processing and refinement
   - Error handling and retry logic

 Expected Outcome: Agents can orchestrate local functions and external tools

 ---
 Implementation Priority

 CRITICAL (Must Have):
 1. AgentChain Integration (Phase 1)
 2. AgenticStepProcessor Support (Phase 2)

 HIGH (Should Have):
 3. MCP Tool Ecosystem (Phase 3)
 4. Per-Agent History Config (Phase 4)

 MEDIUM (Nice to Have):
 5. Workflow State (Phase 5)
 6. Agent Templates (Phase 6)
 7. Tool Registration (Phase 7)

 ---
 Success Criteria

 ✅ CLI uses AgentChain for orchestration, not individual PromptChain instances
 ✅ Agents support multi-hop reasoning via AgenticStepProcessor
 ✅ Intelligent agent routing based on user query
 ✅ MCP tools integrated for external capabilities
 ✅ Per-agent history configuration saving 30-60% tokens
 ✅ Persistent workflow state across sessions
 ✅ Specialized agent templates showcasing library features
 ✅ Infrastructure usage: 15% → 85%+

 ---
 Estimated Impact

 Token Efficiency: 30-60% reduction via per-agent history configs
 Capability Expansion: 6x increase in agent sophistication
 User Experience: Intelligent routing, complex reasoning, tool access
 Library Showcase: CLI becomes demonstration of PromptChain's full power

 This transforms the CLI from a simple chat interface into a sophisticated agent orchestration platform.