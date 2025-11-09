---
noteId: "0fcda2b0055111f0b67657686c686f9a"
tags: []

---

# System Patterns: PromptChain

## System Architecture

**Research Agent Frontend Architecture Achievement**

The Research Agent project recently achieved a major frontend architecture milestone by successfully resolving a critical TailwindCSS 4.x configuration crisis. Key architectural decisions:

- **CSS-First Architecture**: Migrated from TailwindCSS 3.x JavaScript config to 4.x @theme directive approach
- **Component System**: Established comprehensive CSS component architecture using custom CSS variables
- **Design System**: Implemented professional orange/coral (#ff7733) brand identity
- **Technology Stack**: SvelteKit + TypeScript + TailwindCSS 4.x + CSS-first configuration
- **Visual Verification**: Playwright testing integration for design system validation

PromptChain follows a modular, pipeline-based architecture that enables flexible prompt processing through a series of steps:

```
[Input] → [Step 1] → [Step 2] → ... → [Step N] → [Final Output]
```

Each step can be either:
- An LLM call (using a specified model)
- A custom processing function

### Core Components

1. **PromptChain Class**: Central coordinator that manages the flow of data through the chain
2. **ChainStep Model**: Data structure for tracking individual step information
3. **Utility Functions**: Supporting components for model execution and data handling
4. **Chainbreakers**: Optional functions that can conditionally terminate chain execution
5. **Prompt Templates**: Reusable prompt patterns stored as instructions
6. **Memory Bank**: Persistent storage system for maintaining state across chain executions
7. **Terminal Tool with Persistent Sessions**: Advanced terminal execution with state persistence across commands
8. **Research Agent Frontend**: Professional SvelteKit-based dashboard with TailwindCSS 4.x design system

## Key Technical Decisions

### 1. Model Parameterization
- Models are specified as either strings or dictionaries with parameters
- If a single model is provided, it's used for all non-function steps
- Model parameters can be customized for each step

```python
models = [
    "openai/gpt-4",                          # Simple string format
    {"name": "anthropic/claude-3-opus-20240229", "params": {"max_tokens": 1000}}  # Dict with params
]
```

### 2. Function Injection
- Custom Python functions can be inserted at any point in the chain
- Functions receive the output of the previous step and return a result for the next step
- This enables custom processing, validation, or transformation between LLM calls

```python
def analyze_sentiment(text: str) -> str:
    # Custom processing logic
    return processed_result

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Extract key points: {input}", analyze_sentiment, "Summarize: {input}"]
)
```

### 3. History Management
- Optional full history tracking (`full_history=True`)
- Step storage for selective access (`store_steps=True`)
- Detailed information about each step (type, input, output, model params)

### 4. Chain Breaking Logic
- Conditional termination of chains based on custom logic
- Chainbreakers can modify the final output if needed
- Each breaker returns `(should_break, reason, modified_output)`

### 5. Dynamic Chain Execution
- **Execution Modes**:
  - Serial: Sequential execution with dependency tracking
  - Parallel: Concurrent execution of independent steps
  - Independent: No dependency tracking, flexible execution timing
- **Group Management**:
  - Chains organized into logical groups
  - Mixed execution modes within groups
  - Group-level execution control
- **Execution Flow Control**:
  - Dependency validation
  - Status tracking
  - Output preservation
  - Chain merging and insertion

### 6. Chain Builder Architecture
- **Base Configuration**:
  - Common model settings
  - Base instruction templates
  - Shared execution parameters
- **Chain Registry**:
  - Chain metadata storage
  - Execution mode tracking
  - Group membership
  - Dependency relationships
- **Output Management**:
  - Per-chain output storage
  - Group result aggregation
  - Status history

### 7. Memory Bank Architecture
- **Namespace Organization**:
  - Memories organized into logical namespaces
  - Default namespace for common storage
  - Specialized namespaces for isolating memory contexts
- **Core Operations**:
  - Store: `store_memory(key, value, namespace)`
  - Retrieve: `retrieve_memory(key, namespace, default)`
  - Check: `memory_exists(key, namespace)`
  - List: `list_memories(namespace)`
  - Clear: `clear_memories(namespace)`
- **Memory Functions**:
  - Specialized memory access functions for chain steps
  - Command parsing for storing and retrieving from memory
  - Integration with prompt templates
- **Memory Chains**:
  - Dedicated chains with memory capabilities
  - Built-in memory processing
  - Template-based memory operations
- **Chat Integration**:
  - Conversation history storage across sessions
  - User preference management in dedicated namespaces
  - Contextual memory for maintaining conversation state
  - WebSocket server integration for real-time applications
  - Session identification and management

### 8. MCP Tool Hijacker Architecture (NEW - January 2025)
- **Direct Tool Execution System**:
  - **MCPToolHijacker**: Main orchestrator class for bypassing LLM agent processing overhead
  - **ToolParameterManager**: Static/dynamic parameter management with transformation capabilities
  - **MCPConnectionManager**: Connection pooling, session management, and tool discovery system
  - **ToolSchemaValidator**: Parameter validation against MCP tool schemas with type checking
  - **Integration Pattern**: Optional hijacker property in PromptChain for non-breaking adoption
  - **Performance Architecture**: Direct tool execution targeting sub-100ms latency
- **Modular Design Philosophy**:
  - Clean separation of concerns with focused component responsibilities
  - Non-breaking integration with existing MCP infrastructure
  - Extensible parameter transformation system with custom functions
  - Robust error handling and connection recovery mechanisms
  - Comprehensive schema validation and parameter sanitization
- **Tool Execution Flow**:
  - Parameter merging: Static parameters + dynamic overrides
  - Parameter transformation: Custom transformation functions applied
  - Schema validation: Parameters validated against MCP tool schemas
  - Direct execution: Tool called directly on MCP server bypassing LLM processing
  - Result processing: Raw tool results returned with minimal overhead
- **Integration Architecture**:
  - Optional hijacker integration via PromptChain.mcp_hijacker property
  - Backward compatibility with existing MCP tool execution workflows
  - Seamless parameter management for repetitive tool operations
  - Performance optimization for tool-heavy workflows and batch operations

### 9. Research Agent Frontend Architecture
- **Real-Time Progress Tracking System**:
  - **ProgressTracker.svelte**: Core WebSocket-integrated progress display with step visualization
  - **ProgressModal.svelte**: Full-screen detailed progress view with controls and connection status
  - **ProgressWidget.svelte**: Minimized corner widget for ongoing session monitoring
  - **progress.ts**: Reactive Svelte 5 store managing state across all progress components
  - **progressDemo.ts**: Simulation system providing realistic research workflow testing
  - **WebSocket Architecture**: Ready for real backend integration with automatic reconnection
  - **Multi-Session Support**: Unique session IDs with proper state management and cleanup
- **TailwindCSS 4.x Integration**:
  - CSS-first configuration using @theme directive
  - Custom CSS variables for consistent theming
  - Professional component system with hover states and animations
  - Orange/coral primary color palette (#ff7733) for brand identity
- **Component Design System**:
  - Custom button variants (primary, secondary, ghost)
  - Card components with hover effects and shadows
  - Input components with focus states and validation styling
  - Badge components for status indicators
  - Navigation components with active state management
  - Progress components with smooth animations and real-time updates
- **SvelteKit Integration**:
  - TypeScript support for type safety
  - Reactive component architecture with Svelte 5 stores
  - API integration structure for backend communication
  - Multi-view navigation system (Dashboard, Sessions, Chat)
  - Real-time state synchronization across components
- **Visual Standards**:
  - Consistent border radius using CSS variables
  - Standardized shadow system (soft, medium, large)
  - Responsive grid systems for adaptive layouts
  - Smooth animations and transitions
  - Professional typography hierarchy

### 10. Chat Architecture
- **WebSocket Integration**:
  - Real-time message processing through WebSocket connections
  - Session management for persistent conversations
  - Event-driven message handling
- **Conversation Context**:
  - Memory-based conversation history tracking
  - Context windowing for focusing on relevant history
  - User and session identification
- **Message Processing Pipeline**:
  - Message preprocessing and normalization
  - Prompt construction with conversation context
  - Response generation and formatting
  - Memory updates for context maintenance
- **Session identification and management**

#### Simple Agent-User Chat Loop
- Implements a basic chat loop between an agent and a user using PromptChain
- Integrates with the Memory Bank for session persistence and conversation history
- Demonstrates core chat functionality without advanced routing or multi-agent orchestration

## Design Patterns

### 1. Builder Pattern
- The `PromptChain` constructor acts as a builder for complex chain configurations
- Configuration options set during initialization determine chain behavior

### 2. Pipeline Pattern
- Data flows sequentially through a series of processing steps
- Each step's output becomes the input for the next step

### 3. Strategy Pattern
- Different models or functions can be swapped in and out for different steps
- This allows flexible composition of processing strategies

### 4. Decorator Pattern (for function steps)
- Function steps act as decorators that transform data between model calls
- They can add, remove, or modify information without changing the core flow

### 5. Repository Pattern (for Memory Bank)
- Memory Bank acts as a repository for storing and retrieving persistent data
- Provides a consistent interface for memory operations regardless of backend
- Abstracts the details of memory storage from chain execution

## Component Relationships

1. **PromptChain and Models**:
   - PromptChain initializes and manages the execution of models
   - Models process prompts and generate outputs for subsequent steps

2. **PromptChain and Instructions**:
   - Instructions define the transformation applied at each step
   - They can be templates with placeholder variables or functions

3. **PromptChain and ChainStep**:
   - ChainStep objects represent the state at each point in the chain
   - PromptChain creates and manages these steps during execution

4. **PromptChain and Chainbreakers**:
   - Chainbreakers conditionally interrupt chain execution
   - They have access to step information for making decisions

5. **PromptChain and Memory Bank**:
   - Memory Bank provides persistent storage across chain executions
   - Chains can store and retrieve information using the Memory Bank
   - Memory functions enable direct memory operations within chain steps
   - Chat applications use Memory Bank for conversation state management

6. **PromptChain and Chat Systems**:
   - WebSocket servers interface with PromptChain for message processing
   - Conversation history maintained in Memory Bank namespaces
   - Asynchronous processing handles concurrent chat sessions
   - MCP servers provide specialized tools for chat operations

## Technical Constraints

1. **Terminal Tool Persistent Sessions Architecture (IMPLEMENTED)**:
   - **File-Based State Management**: SimplePersistentSession uses bash scripts to persist environment variables and working directory
   - **Session Isolation**: Each named session maintains completely separate state using dedicated directories
   - **Command Substitution Handling**: Properly processes command substitution syntax like `export VAR=$(pwd)` with clean output
   - **Multi-Session Management**: SimpleSessionManager coordinates multiple isolated sessions with unique state
   - **Backward Compatibility**: Enhanced TerminalTool maintains existing API while adding optional session management
   - **State Persistence**: Environment variables and working directory persist across separate command invocations
   - **Clean Output Processing**: Reliable separation of command output from state management overhead

2. **MCP Tool Hijacker Development Constraints**:
   - **MCP Protocol Dependency**: Tool execution depends on MCP server availability and protocol stability
   - **Schema Validation Complexity**: Need to handle diverse MCP tool schemas and parameter types
   - **Connection Management**: Requires robust connection pooling and failure recovery mechanisms
   - **Parameter Transformation**: Must support flexible parameter transformation without breaking existing workflows
   - **Performance Requirements**: Sub-100ms latency targets require careful optimization of connection overhead
   - **Integration Compatibility**: Must maintain backward compatibility with existing PromptChain MCP integration
   - **Testing Complexity**: Requires comprehensive mock MCP server infrastructure for reliable testing

2. **Frontend Development Constraints (Resolved)**:
   - **TailwindCSS 4.x Learning Curve**: Successfully navigated migration from 3.x patterns
   - **CSS-First Architecture**: Required fundamental approach change from JavaScript config
   - **Component Integration**: Resolved conflicts between @apply directives and CSS variables
   - **Design System Consistency**: Established comprehensive design token system
   - **Browser Compatibility**: Ensured modern CSS features work across target browsers

2. **API Dependencies**:
   - Requires valid API keys for each model provider
   - Environment variables must be set up correctly

2. **Function Compatibility**:
   - Custom functions must accept a single string input and return a string output
   - Functions should handle exceptions to prevent chain failures

3. **Model Availability**:
   - Chain execution depends on the availability of specified models
   - Rate limits and quotas from providers may impact performance

4. **Memory Management**:
   - Full history tracking increases memory usage with chain length
   - For very long chains, selective history or step storage is recommended

5. **Memory Persistence**:
   - Current implementation uses in-memory storage (non-persistent between processes)
   - Memory access is synchronous and immediate
   - Future implementations will support persistent storage with potential latency
   - Memory operations require proper error handling for missing or invalid values 

## Agent Orchestration (AgentChain)

- **Purpose**: To manage and route tasks between multiple specialized `PromptChain` agents.
- **Structure**: 
    - Uses a central `AgentChain` class.
    - Contains a dictionary of named `PromptChain` agent instances.
    - Employs a configurable routing mechanism.
- **Routing**: 
    1. **Simple Router**: First checks input against basic rules (e.g., regex for math).
    2. **Complex Router**: If no simple match, invokes either:
        - A default 2-step LLM chain (Prepare Prompt -> Execute LLM for JSON decision).
        - A user-provided custom asynchronous Python function.
    3. **Direct Execution**: Allows bypassing routing via `@agent_name:` syntax in `run_chat`.
- **Agent Responsibility**: Individual agents handle their own logic, tools, and MCP connections; they must be pre-configured. 

## Router Error Recovery Pattern

- **Problem**: When handling complex inputs like multiline code blocks with special characters, the router's JSON output can become malformed.
- **Solution**: A multi-layered approach to error detection and recovery:
  1. **Router Prompt Enhancement**: 
     - Explicit instructions for properly escaping special characters
     - Example JSON with escaped quotes and newlines
     - Clear formatting guidelines for different input types
  2. **Error Detection**:
     - Primary approach attempts regular routing
     - Detects specific error message patterns
     - Identifies JSON parsing errors in router output
  3. **Plan Extraction**:
     - When errors occur, attempts to extract the intended plan from malformed JSON
     - Uses multiple parsing strategies (direct parsing, regex extraction)
     - Preserves the router's intended agent sequence
  4. **Manual Plan Execution**:
     - Executes extracted plans by calling agents in sequence
     - Passes output from one agent as input to the next
     - Mimics the normal multi-agent plan execution flow
  5. **Fallback Mechanism**:
     - When extraction fails, uses sensible defaults for specific content types
     - For code inputs, defaults to library_verification_agent → engineer_agent
     - Handles direct agent calls when specified with @agent_name syntax
  6. **Partial Execution Recovery**:
     - Returns partial results if an agent in the sequence fails
     - Clearly indicates which part of the plan completed successfully
     - Maintains as much functionality as possible even with errors

## Agentic Orchestrator Router Pattern (NEW - October 2025)

- **Concept:** Transform AgentChain's router from single-step decision-maker to intelligent orchestrator using AgenticStepProcessor for multi-hop reasoning and progressive context accumulation.

- **Problem Addressed:**
  - Current router: Single LLM call, ~70% accuracy, no multi-hop reasoning
  - Context loss: No progressive history accumulation across routing decisions
  - Knowledge blindness: Cannot detect when external research is needed
  - Temporal unawareness: Missing current date context for time-sensitive queries

- **Core Architecture:**
  - **AgenticStepProcessor Orchestrator**: Multi-hop reasoning engine with 5 internal steps
  - **Progressive History Mode**: CRITICAL - accumulates context across reasoning steps
  - **Tool Capability Awareness**: Knows which agents have which tools/MCP servers
  - **Knowledge Boundary Detection**: Identifies when research vs internal knowledge needed
  - **Current Date Awareness**: Temporal context for "latest" and "recent" queries

- **Implementation Strategy:**
  - **Phase 1 (Validation)**: Async wrapper function - non-breaking, validates approach
  - **Phase 2 (Integration)**: Native AgentChain router mode after validation
  - **Target Accuracy**: 95% (from current ~70%)
  - **Performance Budget**: <5s routing decision, <2000 tokens

- **Technical Pattern:**
  ```python
  # Phase 1: Wrapper Function
  async def agentic_orchestrator_router(
      user_query: str,
      agent_details: dict,
      conversation_history: list,
      current_date: str
  ) -> dict:
      orchestrator = AgenticStepProcessor(
          objective=f"Intelligent routing with multi-hop reasoning...",
          max_internal_steps=5,
          history_mode="progressive",  # CRITICAL for context accumulation
          model_name="openai/gpt-4o"
      )
      # Multi-step reasoning with tool access
      result = await orchestrator.run_async(...)
      return routing_decision

  # Phase 2: Native Integration
  agent_chain = AgentChain(
      agents=agents,
      router_mode="agentic",
      agentic_router_config={
          "max_internal_steps": 5,
          "history_mode": "progressive",
          "enable_research_detection": True,
          "enable_date_awareness": True
      }
  )
  ```

- **Key Technical Insights:**
  - **Progressive History is Essential**: Without it, orchestrator "forgets" previous reasoning
  - **Multi-Hop Reasoning Flow**:
    1. Analyze query complexity
    2. Check agent capabilities
    3. Assess knowledge boundaries
    4. Consider temporal context
    5. Make final routing decision
  - **Knowledge Boundary Detection**: Identifies queries needing external research vs available knowledge
  - **Tool Awareness**: Routes based on which agents have required tools/MCP servers

- **Migration Path:**
  - **Current**: `router_config = {"models": [...], "instructions": [...]}`
  - **Phase 1**: Add `"custom_router_function": agentic_orchestrator_router`
  - **Phase 2**: Set `router_mode="agentic"` with config dict
  - **Rollback**: Simple config change, zero code changes

- **Success Metrics:**
  - Routing accuracy: 95%+ on validation dataset
  - Multi-hop reasoning: avg 3+ steps for complex queries
  - Context preservation: 100% across reasoning steps
  - Knowledge boundary detection: 90%+ accuracy
  - Performance: <5s latency, <2000 tokens per decision

- **Use Cases:**
  - **Complex Query Routing**: Multi-step analysis for sophisticated queries
  - **Research Detection**: Identify when external knowledge needed
  - **Temporal Queries**: Handle "latest", "recent", "current" queries correctly
  - **Tool-Based Routing**: Route based on agent tool capabilities
  - **Multi-Agent Workflows**: Decompose queries into agent sequences

- **Location:**
  - PRD: `/home/gyasis/Documents/code/PromptChain/prd/agentic_orchestrator_router_enhancement.md`
  - Phase 1 Implementation: `promptchain/utils/agentic_router_wrapper.py` (planned)
  - Phase 2 Integration: `promptchain/utils/agent_chain.py` (future enhancement)

## MCP Tool Hijacker Pattern (NEW - January 2025)

- **Concept:** To enable direct MCP tool execution without LLM agent processing overhead, the `MCPToolHijacker` provides a high-performance interface to MCP tools.
- **Core Components:**
  - **MCPToolHijacker**: Main class orchestrating direct tool execution with parameter management
  - **ToolParameterManager**: Handles static parameter storage, dynamic merging, and custom transformations
  - **MCPConnectionManager**: Manages MCP server connections, tool discovery, and session handling
  - **ToolSchemaValidator**: Validates parameters against MCP tool schemas with type checking

- **Execution Flow:**
  - **Parameter Resolution**: Static parameters merged with dynamic overrides using ToolParameterManager
  - **Parameter Transformation**: Custom transformation functions applied to normalize and validate inputs
  - **Schema Validation**: Parameters validated against discovered MCP tool schemas for type safety
  - **Direct Tool Execution**: Tool called directly on appropriate MCP server bypassing LLM processing
  - **Result Processing**: Raw tool results returned with minimal processing overhead

- **Integration Pattern:**
  - **Optional Integration**: Available via `chain.mcp_hijacker` property for non-breaking adoption
  - **Performance Optimization**: Targets sub-100ms tool execution for performance-critical workflows
  - **Parameter Management**: Supports static parameter presets for repetitive tool operations
  - **Batch Operations**: Optimized for executing same tool with different parameters efficiently

- **Use Cases:**
  - **API Wrappers**: Creating simple interfaces to MCP tools without agent complexity
  - **Batch Processing**: Executing same tool with different parameters repeatedly
  - **Performance-Critical Operations**: Where LLM processing latency is unacceptable
  - **Testing and Validation**: Direct tool testing without full agent workflow setup
  - **Parameter Experimentation**: Iterative parameter adjustment with immediate feedback

- **Technical Implementation:**
  - **Connection Pooling**: Efficient reuse of MCP server connections for multiple tool calls
  - **Error Recovery**: Robust handling of connection failures and parameter validation errors
  - **Type Safety**: Comprehensive parameter validation and type conversion capabilities
  - **Extensibility**: Plugin-like parameter transformation system for custom processing logic

- **Location:** Planned for `promptchain/utils/mcp_tool_hijacker.py` with supporting modules for parameter management and connection handling

## Terminal Tool Persistent Sessions Pattern (IMPLEMENTED - January 2025)

- **Problem Solved**: Terminal commands previously ran in separate subprocesses, losing all state (environment variables, working directory, etc.) between commands, preventing multi-step workflows from functioning correctly.

- **Core Architecture:**
  - **SimplePersistentSession**: File-based state management using bash scripts to persist environment variables and working directory
  - **SimpleSessionManager**: Multi-session coordination allowing multiple isolated terminal environments
  - **Enhanced TerminalTool**: Backward-compatible integration with new session management capabilities

- **Technical Implementation:**
  - **State Persistence**: Environment variables stored in `.env` file, working directory tracked in `.pwd` file
  - **Command Processing**: Bash script execution preserves state between separate command invocations
  - **Session Isolation**: Each named session maintains completely separate state in dedicated directories
  - **Command Substitution**: Proper handling of complex bash syntax like `export VAR=$(pwd)` with clean output processing
  - **Output Cleansing**: Reliable separation of actual command output from state management overhead

- **Key Features:**
  - **Environment Variable Persistence**: `export TEST_VAR=value` → `echo $TEST_VAR` correctly returns `value` in subsequent commands
  - **Working Directory Persistence**: `cd /tmp` → `pwd` correctly returns `/tmp` in subsequent commands
  - **Multiple Session Management**: Create and switch between isolated terminal sessions with independent environments
  - **Command Substitution Support**: Complex bash operations work correctly across command boundaries
  - **Backward Compatibility**: Existing TerminalTool usage patterns work unchanged (opt-in feature)

- **Integration Pattern:**
  - **Session Creation**: `terminal_tool.create_persistent_session("my_session")`
  - **Session Switching**: `terminal_tool.switch_to_session("my_session")`
  - **Persistent Execution**: Commands executed in sessions maintain state automatically
  - **Session Management**: List, delete, and manage multiple named sessions

- **Use Cases:**
  - **Multi-Step Development Workflows**: Activate Node.js version, install dependencies, run build commands with state continuity
  - **Environment Setup**: Configure development environments that persist across multiple commands
  - **Project Context Switching**: Maintain separate environments for different projects or development contexts
  - **Complex Build Processes**: Execute multi-step build processes where each step depends on previous environment changes
  - **Training and Demos**: Reliable demonstration of terminal workflows with predictable state management

- **Verification Results:**
  - All test scenarios passing: environment variables, directory persistence, command substitution, session switching
  - Complex workflows tested and verified working correctly
  - Clean output processing ensures reliable command result extraction
  - Multi-session isolation verified with independent environment states

- **File Locations:**
  - `/promptchain/tools/terminal/simple_persistent_session.py` (core session management)
  - `/promptchain/tools/terminal/terminal_tool.py` (enhanced with session methods)
  - `/examples/session_persistence_demo.py` (comprehensive demonstration)

## Agentic Step Pattern

- **Concept:** To handle complex tasks within a single logical step of a `PromptChain`, the `AgenticStepProcessor` class provides an internal execution loop.
- **Core Components:**
  - **AgenticStepProcessor**: Encapsulates the internal agentic loop logic
  - **Tool Integration**: Uses functions registered in parent PromptChain
  - **Robust Function Name Extraction**: Helper utility to handle different tool call formats
  - **Internal Tool Execution**: Manages tool calls within the agentic step
  - **History Accumulation Modes**: Three modes for managing conversation context during multi-hop reasoning

- **Mechanism:**
  - An instance of `AgenticStepProcessor` is included in the `PromptChain`'s `instructions` list.
  - `PromptChain.process_prompt_async` detects this type and calls its `run_async` method.
  - `run_async` manages an internal loop involving:
    - LLM calls to determine the next action (tool call or final answer).
    - Execution of necessary tools (local or MCP) via callbacks provided by `PromptChain`.
    - Evaluation of results to decide whether to continue the loop or finalize the step.

- **Configuration:**
  - **objective**: Defines the specific goal for the agentic step
  - **max_internal_steps**: Controls loop termination to prevent infinite execution
  - **model_name**: Optional parameter to specify a different model than the parent chain
  - **model_params**: Optional parameters for the LLM model used in the step
  - **history_mode**: Controls history accumulation strategy (minimal/progressive/kitchen_sink)
  - **max_context_tokens**: Optional token limit for warning when context grows too large

- **History Accumulation Modes (NEW - January 2025):**
  - **minimal** (default): Only keeps last assistant message + tool results (original behavior)
    - Use for: Simple single-tool tasks, token efficiency
    - ⚠️ May be deprecated in future versions
  - **progressive** (RECOMMENDED): Accumulates assistant messages + tool results progressively
    - Use for: Multi-hop reasoning, knowledge accumulation across tool calls
    - Fixes the context loss problem in complex agentic workflows
  - **kitchen_sink**: Keeps everything - all reasoning, tool calls, and results
    - Use for: Maximum context retention, complex reasoning chains
    - Trade-off: Uses most tokens

- **Function Name Extraction Logic:**
  - New helper function `get_function_name_from_tool_call` extracts function names from:
    - Dictionary format (common in direct API responses)
    - Object with attributes (common in LiteLLM's response objects)
    - Nested objects with various property structures
    - Objects with model_dump capability (Pydantic models)
  - This robust extraction prevents infinite loops and improves reliability

- **Tool Execution Flow:**
  1. LLM generates a tool call based on the objective and context
  2. `get_function_name_from_tool_call` extracts the function name
  3. Arguments are extracted from the tool call
  4. The tool is executed via callbacks to the parent PromptChain
  5. Results are formatted and added to history based on history_mode
  6. Context is accumulated according to selected history mode
  7. Process repeats until a final answer is reached or max steps are hit

- **History Management Implementation (lines 300-343):**
  ```python
  if history_mode == "minimal":
      # Only last interaction - backward compatible
      llm_history = [system, user, last_assistant, last_tools]

  elif history_mode == "progressive":
      # Accumulate progressively - RECOMMENDED for multi-hop
      conversation_history.append(last_assistant)
      conversation_history.extend(last_tools)
      llm_history = [system, user] + conversation_history

  elif history_mode == "kitchen_sink":
      # Keep everything for maximum context
      conversation_history.append(last_assistant)
      conversation_history.extend(last_tools)
      llm_history = [system, user] + conversation_history

  # Token limit warning
  if max_context_tokens and estimate_tokens(llm_history) > max_context_tokens:
      logger.warning("Context size exceeds limit...")
  ```

- **PromptChain Integration:**
  - When using `AgenticStepProcessor`, set `models=[]` in PromptChain initialization
  - Model is set directly on the AgenticStepProcessor using `model_name` parameter
  - Tools must be registered with the parent PromptChain using `add_tools()` and `register_tool_function()`

- **Error Handling:**
  - Robust error handling for failed function name extraction
  - Clear error messages for missing tool functions
  - Proper handling of tool execution failures
  - Prevention of infinite loops through max_internal_steps

- **Purpose:** Allows for dynamic, multi-turn reasoning and tool use to accomplish a specific sub-goal defined by the agentic step's objective, before the main chain proceeds.
- **Location:** `promptchain/utils/agentic_step_processor.py` with supporting code in `promptchain/utils/promptchaining.py` 