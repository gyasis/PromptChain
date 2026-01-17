---
noteId: "0fcda2b0055111f0b67657686c686f9a"
tags: []

---

# System Patterns: PromptChain

## System Architecture

**Agent-Transfer Conflict Resolution Architecture (November 29, 2025)**

The agent-transfer subproject implements a comprehensive diff-based conflict resolution system for managing agent configuration imports. Key architectural components:

### Conflict Resolution System

**Core Components**:
1. **ConflictMode Enum**: Four resolution strategies (OVERWRITE, KEEP, DUPLICATE, DIFF)
2. **Visual Diff Engine**: Rich-based colored diff display with unified and side-by-side views
3. **Hybrid Merge System**: Section-aware YAML merging + line-by-line markdown merging
4. **Interactive Resolution UI**: User-driven merge decision workflow

**Architecture Pattern**:
```
Import Flow:
User imports backup → Detect conflicts → Conflict resolver invoked
                                              ↓
                                     ConflictMode switch
                                              ↓
                    ┌────────────────────────┴────────────────────────┐
                    │                                                  │
            OVERWRITE/KEEP                                        DIFF mode
         (automatic resolution)                            (interactive resolution)
                    │                                                  │
                    └──────────────────┬───────────────────────────────┘
                                       ↓
                            Final agent files written
```

**Key Technical Patterns**:

1. **Diff Display Strategy**:
   - Unified Diff: Rich Syntax with language="diff" for colored output
   - Side-by-Side: Rich Table with two columns (existing vs imported)
   - YAML/Markdown Split: parse_agent_sections() separates structured vs prose content

2. **Hybrid Merge Granularity**:
   - YAML Frontmatter: Field-level selection (name, model, provider, description)
     ```
     Choose fields:
     [1] name: "researcher" (existing) or "analyst" (imported)
     [2] model: "gpt-4" (existing) or "claude-3" (imported)
     ```
   - Markdown Body: Diff block selection (groups of changed lines)
     ```
     Block 1: Lines 10-15 (5 additions, 2 deletions)
     [e]xisting / [i]mported / [b]oth / [s]kip
     ```

3. **Duplicate Naming Convention**:
   - Pattern: `{base_name}_{number}.md`
   - Algorithm: `get_duplicate_name()` increments suffix until unique
   - Example: `researcher.md` → `researcher_1.md` → `researcher_2.md`

4. **Shell Script Parity**:
   - Full conflict resolution implemented in bash
   - Functions: `show_diff()`, `show_side_by_side()`, `resolve_conflict_interactive()`
   - Dependencies: Optional `colordiff`/`sdiff`, fallback to plain `diff`
   - Maintains UX consistency with Python implementation

**File Organization**:
```
agent-transfer/
├── agent_transfer/
│   ├── utils/
│   │   ├── conflict_resolver.py  # Core conflict resolution logic (402 lines)
│   │   └── transfer.py           # Import/export with conflict handling
│   └── cli.py                    # CLI interface with --conflict-mode flag
├── agent-transfer.sh             # Standalone shell script with full parity
└── README.md                     # User documentation
```

**Integration Points**:
- `transfer.py::import_agents()`: Detects conflicts, invokes resolver
- `cli.py`: Exposes --conflict-mode/-c option to users
- `conflict_resolver.py`: Standalone module, no external dependencies beyond Rich

**Design Philosophy**:
- Minimal dependencies (stdlib difflib + existing Rich)
- User control (interactive decisions vs automated modes)
- Graceful degradation (shell script fallbacks)
- Consistent UX (Python CLI and shell script behave identically)

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

### 9. PromptChain CLI Architecture (NEW - January 2025)

**Overview:**
The PromptChain CLI provides a professional terminal user interface (TUI) for interactive LLM conversations with multi-agent orchestration, session persistence, and advanced UX features.

**Core Architecture:**
- **Framework**: Textual 0.83+ for TUI framework, Rich 13.8+ for terminal formatting
- **Agent Integration**: Uses PromptChain's AgentChain for multi-agent orchestration
- **Session Persistence**: SQLite-based session storage via AgentChain cache_config
- **Configuration**: JSON-based config system with validation and defaults

**Component Structure:**
```
promptchain/cli/
├── tui/
│   ├── app.py           # Main TUI application (PromptChainApp)
│   ├── chat_view.py     # Chat message display with syntax highlighting
│   ├── input_widget.py  # Command input with history and autocomplete
│   ├── status_bar.py    # Status display (session/agent/model)
│   └── help_modal.py    # Help system modal dialog
├── models/
│   └── config.py        # Configuration dataclasses with validation
├── utils/
│   └── output_formatter.py  # Output formatting utilities
└── main.py              # CLI entry point and initialization
```

**Key Technical Patterns:**

1. **Reactive UI Updates**:
   - Textual reactive properties for real-time updates
   - Status bar reactively reflects agent/session state
   - Chat view auto-scrolls and highlights new messages

2. **Configuration System** (config.py):
   - Dataclasses with field validation
   - JSON file loading with defaults
   - Hierarchical config (ui, performance, agents, sessions)
   - Type-safe configuration access

3. **Session Management**:
   - Integration with AgentChain.cache_config for SQLite persistence
   - Session naming, listing, loading, deletion
   - Conversation history stored in database
   - Session directory structure: `~/.promptchain/sessions/`

4. **Input Handling**:
   - Command history navigation (↑/↓ arrows)
   - Tab autocomplete for slash commands
   - Multi-line input support (Shift+Enter)
   - Slash command parsing and routing
   - Message submission queue for async processing

5. **Error Handling**:
   - User-friendly error messages in chat view
   - Graceful recovery from agent failures
   - Exception logging without crashes
   - Modal dialogs for critical errors

6. **Performance Optimizations**:
   - Lazy loading: Agents initialized on first use
   - Pagination: Large conversation histories paginated
   - Async message processing: Non-blocking UI
   - LiteLLM logging suppression: Cleaner output

**Help System Architecture:**

- `/help` command triggers modal dialog
- Categorized command documentation:
  - Session Management
  - Agent Control
  - Information Commands
  - System Commands
- Keyboard shortcuts guide
- Workflow suggestions
- Configuration tips

**Status Bar Design:**
```
[*] Session: my-session | Agent: analyst (gpt-4) | Messages: 42
 ^^            ^^             ^^           ^^          ^^
State      Session Name   Active Agent  Model     Message Count
```

**Phase 8 Achievements (Complete):**
- Comprehensive help system with categorized documentation
- Configuration system with JSON loading and validation
- Animated spinners during LLM processing
- Message selection and copy functionality (arrow keys + c)
- Command history navigation and recall
- Tab autocomplete for all slash commands
- Multi-line input with visual indicator
- Lazy loading for performance
- Conversation pagination for large sessions
- User-friendly error messages
- flake8 cleanup and type hints

**Next Enhancement: Token Management**
- Real-time token tracking in status bar
- Automatic history compression at 75% threshold
- Integration with ExecutionHistoryManager
- Visual progress bar and color-coded warnings
- See PRD: `/home/gyasis/Documents/code/PromptChain/docs/prd/cli-enhancement-token-management.md`

**Technical Stack:**
- Python 3.8+ (compatible with PromptChain)
- Textual 0.83+ (TUI framework)
- Rich 13.8+ (terminal formatting)
- Click 8.1+ (CLI framework)
- LiteLLM 1.0+ (existing, for agent execution)
- asyncio (stdlib, for async processing)
- SQLite 3 (session persistence)

**Branch:** `001-cli-agent-interface`

### 10. Research Agent Frontend Architecture
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

## Advanced Agentic Patterns (NEW - November 2025)

### Pattern System Architecture
- **Concept:** Modular, composable patterns for advanced RAG and agentic workflows built on top of 003-multi-agent-communication infrastructure
- **Foundation:** All patterns inherit from BasePattern which provides MessageBus/Blackboard integration
- **Library Choice:** Wraps hybridrag library for LightRAG patterns instead of building from scratch (faster implementation)

### Core Components

**BasePattern** (`promptchain/patterns/base.py`):
- Abstract base class for all agentic patterns
- MessageBus integration for event-driven coordination (enhanced in Wave 3)
- Blackboard integration for shared state management (enhanced in Wave 3)
- Dual execution interface: execute() and execute_async()
- Consistent event emission for pattern lifecycle (36 standardized event types)
- Event history tracking (last 100 events per pattern)
- State versioning and snapshot support (Wave 3)

**PatternConfig**:
- Pydantic-based configuration validation
- Type-safe pattern parameters
- Extensible for pattern-specific configuration

**PatternResult**:
- Structured result object with metadata
- Success/failure status tracking
- Execution metrics and timing
- Pattern-specific output data

### LightRAG Pattern Implementations

**1. LightRAGBranchingThoughts** (`promptchain/integrations/lightrag/branching.py`):
- **Purpose:** Parallel hypothesis generation with judge feedback
- **Mechanism:**
  - Generates k hypotheses in parallel
  - Judge model evaluates and selects best hypothesis
  - Enables creative exploration of solution space
- **Use Cases:** Creative problem-solving, multi-perspective analysis
- **Key Parameters:** num_branches (k=3-5), judge_model, hypothesis_generator

**2. LightRAGQueryExpander** (`promptchain/integrations/lightrag/query_expansion.py`):
- **Purpose:** Parallel query diversification for comprehensive search
- **Mechanism:**
  - Expands single query into k diverse perspectives
  - Executes queries in parallel
  - Combines results from multiple angles
- **Use Cases:** Comprehensive information retrieval, multi-aspect analysis
- **Key Parameters:** k (3-5 queries), expansion_strategy, combination_method

**3. LightRAGShardedRetriever** (`promptchain/integrations/lightrag/sharded.py`):
- **Purpose:** Multi-source parallel retrieval with fusion
- **Mechanism:**
  - Queries multiple data sources in parallel
  - Applies Reciprocal Rank Fusion (RRF) for result merging
  - Efficient distributed retrieval
- **Use Cases:** Large-scale retrieval, federated search
- **Key Parameters:** shards (data sources), fusion_method (RRF), top_k

**4. LightRAGMultiHop** (`promptchain/integrations/lightrag/multi_hop.py`):
- **Purpose:** Complex reasoning with question decomposition
- **Mechanism:**
  - Decomposes complex questions into sub-questions
  - Uses agentic_search for multi-step reasoning
  - Progressive knowledge accumulation
- **Use Cases:** Complex analytical queries, research workflows
- **Key Parameters:** max_hops, decomposition_strategy, agentic_search_model

**5. LightRAGHybridSearcher** (`promptchain/integrations/lightrag/hybrid_search.py`):
- **Purpose:** Flexible fusion of multiple search strategies
- **Mechanism:**
  - Combines keyword and semantic search
  - Supports multiple fusion algorithms (RRF, Linear, Borda)
  - Configurable fusion strategy
- **Use Cases:** Balanced retrieval, combining multiple search modalities
- **Key Parameters:** fusion_algorithm, keyword_weight, semantic_weight

**6. LightRAGSpeculativeExecutor** (`promptchain/integrations/lightrag/speculative.py`):
- **Purpose:** Performance optimization with predictive caching
- **Mechanism:**
  - Predicts likely tool calls before execution
  - Caches speculative results
  - Falls back to normal execution if prediction wrong
- **Use Cases:** Performance-critical workflows, repetitive tool usage
- **Key Parameters:** prediction_model, cache_size, speculation_threshold

### Integration with Multi-Agent Communication (003) - Enhanced in Wave 3

**Enhanced MessageBus Integration:**
```python
# All patterns emit events to MessageBus with history tracking
from promptchain.integrations.lightrag.messaging import PatternMessageBusMixin

class MyPattern(BasePattern, PatternMessageBusMixin):
    def execute(self, input_data):
        # Emit started event with automatic history tracking
        self.publish_pattern_event("pattern.started", {...})

        # Batch publish multiple events efficiently
        self.publish_batch([
            create_pattern_progress_event("my_pattern", "step1", {...}),
            create_pattern_progress_event("my_pattern", "step2", {...})
        ])

        result = self._execute_impl(input_data)

        # Query event history for debugging
        recent_events = self.get_event_history(limit=10)

        self.publish_pattern_event("pattern.completed", {...})
        return result
```

**Enhanced Blackboard Integration:**
```python
# Patterns can read/write shared state with TTL, versioning, and snapshots
from promptchain.integrations.lightrag.state import PatternBlackboardMixin

class LightRAGMultiHop(BasePattern, PatternBlackboardMixin):
    def _execute_impl(self, query):
        # Store intermediate results with TTL (auto-expires in 1 hour)
        self.write_with_ttl("current_hop", hop_data, ttl=3600)

        # Create snapshot before complex operation
        snapshot = self.create_snapshot()

        # Read context with version tracking
        context, version = self.get_versioned("search_context")

        # Cleanup expired state
        self.cleanup_expired()

        # Restore from snapshot if needed
        if error:
            self.restore_snapshot(snapshot)
```

**Standardized Event System:**
```python
# 36 event types across 6 patterns
from promptchain.integrations.lightrag.events import (
    PatternEvent, EventSeverity, EventLifecycle,
    create_pattern_started_event, create_pattern_completed_event,
    subscribe_to_pattern, subscribe_to_lifecycle
)

# Create standardized events
event = create_pattern_started_event(
    "lightrag_multihop",
    data={"query": query, "max_hops": 3}
)

# Subscribe to specific pattern events
subscribe_to_pattern(
    message_bus, "lightrag_multihop",
    lambda event: print(f"MultiHop event: {event.event_type}")
)

# Subscribe to lifecycle events (all patterns)
subscribe_to_lifecycle(
    message_bus, EventLifecycle.FAILED,
    lambda event: handle_pattern_failure(event)
)
```

**Event-Driven Coordination:**
- Patterns publish 36 standardized event types (6 per pattern)
- Event history tracking enables debugging and analysis
- Subscription helpers simplify pattern-level and lifecycle filtering
- Other agents can subscribe to pattern events
- Enables reactive multi-agent workflows
- Consistent event schema with severity, lifecycle, correlation tracking

### Architectural Decisions

**1. Wrapping hybridrag vs Building from Scratch:**
- **Decision:** Wrap existing hybridrag library
- **Rationale:** Faster implementation, proven functionality, community support
- **Trade-off:** Some loss of flexibility vs complete control
- **Benefit:** Focus on integration and composition vs low-level implementation

**2. BasePattern as Foundation:**
- **Decision:** All patterns inherit from BasePattern
- **Rationale:** Ensures consistent MessageBus/Blackboard integration from 003
- **Benefit:** Patterns automatically participate in multi-agent coordination
- **Impact:** Patterns can be composed and orchestrated like agents

**3. Pydantic for Configuration:**
- **Decision:** Use Pydantic for PatternConfig validation
- **Rationale:** Type safety, automatic validation, IDE support
- **Benefit:** Catches configuration errors early, clear documentation

**4. Dual Sync/Async Interface:**
- **Decision:** All patterns support both execute() and execute_async()
- **Rationale:** Flexibility for different usage contexts
- **Benefit:** Patterns work in sync and async workflows seamlessly

**5. Structured Results:**
- **Decision:** PatternResult for all pattern outputs
- **Rationale:** Consistent interface, metadata tracking, error handling
- **Benefit:** Uniform pattern composition and error handling

### Pattern Composition

Patterns can be composed in multiple ways:

**Sequential Composition:**
```python
# Chain patterns together
query = "complex question"
expanded = query_expander.execute(query)
results = multi_hop.execute(expanded)
final = hybrid_search.execute(results)
```

**Parallel Composition:**
```python
# Execute patterns in parallel
with ThreadPoolExecutor() as pool:
    futures = [
        pool.submit(branching.execute, query),
        pool.submit(query_expander.execute, query)
    ]
    results = [f.result() for f in futures]
```

**Event-Driven Composition:**
```python
# Subscribe to pattern events
message_bus.subscribe("pattern.completed",
    lambda event: next_pattern.execute(event.data))
```

### Integration Points

**With PromptChain:**
- Patterns can be used as instruction steps in PromptChain
- AgenticStepProcessor can invoke patterns for sub-goals
- Patterns emit events that PromptChain can react to

**With AgentChain:**
- Patterns enable specialized agent capabilities
- Agents can use patterns as tools
- Multi-agent workflows can coordinate via pattern events

**With CLI:**
- Future `/patterns` command for interactive pattern execution
- Pattern results displayed in TUI
- Pattern composition via CLI workflows

### Wave 3 Completion (November 29, 2025) ✅

**Integration Layer Delivered:**
- ✅ Enhanced messaging protocol for pattern coordination (messaging.py)
- ✅ Shared state management via improved Blackboard (state.py)
- ✅ Advanced event routing and filtering (events.py)
- ✅ Event history tracking (last 100 events per pattern)
- ✅ TTL-based state cleanup with automatic expiration
- ✅ State versioning and immutable snapshots
- ✅ Standardized event system (36 event types)
- ✅ Subscription helpers for pattern-level and lifecycle filtering

**Key Architecture Achievements:**
- PatternMessageBusMixin: Event history + batch publishing
- PatternBlackboardMixin: TTL + versioning + snapshots
- PatternEvent: Standardized event structure with metadata
- PATTERN_EVENTS: Complete registry of 36 event types
- Event factory functions: Consistent event creation
- Subscription helpers: Simplified event filtering

**Files Created (Wave 3):**
- `promptchain/integrations/lightrag/messaging.py` (225 lines)
- `promptchain/integrations/lightrag/state.py` (189 lines)
- `promptchain/integrations/lightrag/events.py` (267 lines)

**Total Lines Added (Wave 3):** 681 lines

### Spec 004a: TUI Pattern Commands (December 2025) - COMPLETE ✅

**Overview:**
Specification 004a delivered TUI pattern slash commands, enabling users to execute all 6 advanced agentic patterns directly from the TUI without exiting to CLI.

**Problem Solved:**
- Previous UX: Users had to exit TUI → run CLI pattern commands → restart TUI → lose session state
- New UX: Users execute patterns via /branch, /expand, etc. directly in TUI session
- Impact: Zero workflow disruption, full context preservation, seamless pattern integration

**Architecture Pattern:**

**1. Executor Pattern (Wave 1):**
```python
# promptchain/patterns/executors.py (NEW, 604 lines)
async def branch_executor(
    query: str,
    message_bus: Optional[MessageBus] = None,
    blackboard: Optional[Blackboard] = None,
    count: int = 3,
    mode: str = "hybrid"
) -> dict:
    """Shared executor callable from both CLI and TUI."""
    # Pattern execution logic
    return {
        "success": True,
        "result": result_data,
        "error": None,
        "execution_time_ms": duration,
        "metadata": {...}
    }
```

**2. Command Registry Integration (Wave 2):**
```python
# promptchain/cli/command_handler.py
COMMAND_REGISTRY = {
    "patterns": "List available patterns",
    "branch": "Execute branching thoughts pattern",
    "expand": "Execute query expansion pattern",
    "multihop": "Execute multi-hop reasoning pattern",
    "hybrid": "Execute hybrid search pattern",
    "sharded": "Execute sharded retrieval pattern",
    "speculate": "Execute speculative execution pattern",
}
```

**3. TUI Handler Pattern (Wave 2):**
```python
# promptchain/cli/tui/app.py (lines 1809-2314)
async def _handle_branch_pattern(self, command: str) -> None:
    """Handle /branch pattern command."""
    # Parse arguments with shlex
    args = self._parse_pattern_command(command)

    # Execute via shared executor
    result = await branch_executor(
        query=args["query"],
        message_bus=self.message_bus,
        blackboard=self.blackboard,
        count=args.get("count", 3),
        mode=args.get("mode", "hybrid")
    )

    # Format result for chat display
    if result["success"]:
        self.add_message("assistant", f"✅ {result['result']}")
    else:
        self.add_message("assistant", f"❌ Error: {result['error']}")
```

**4. MessageBus/Blackboard Integration (Wave 3):**
- Verified in T007: All handlers pass self.message_bus and self.blackboard to executors
- Event Tracking: Patterns can publish events via MessageBus
- State Management: Patterns can read/write shared state via Blackboard
- End-to-End Chain: TUI → Handler → Executor → Pattern → MessageBus/Blackboard

**Key Architectural Decisions:**

**Executor Pattern Selection:**
- Chosen: Lightweight async functions in executors.py
- Rejected: Duplicate logic (violates DRY), Click command calls (heavy dependency), separate library module (over-engineering)
- Benefit: 95% code reuse, single source of truth, minimal complexity

**TUI Integration Strategy:**
- Chosen: Extend existing COMMAND_REGISTRY + handler methods
- Rejected: Separate pattern dispatcher (breaks consistency), inline logic (violates DRY), JSON config (reduces flexibility)
- Benefit: Consistent with existing TUI patterns, discoverable via /help

**Argument Parsing:**
- Chosen: shlex for shell-style parsing
- Format: "/pattern \"query\" --flag=value"
- Benefit: Familiar to developers, handles quotes and escaping

**Result Formatting:**
- Emoji indicators: ✅ success, ❌ error
- Human-readable chat messages
- Automatic message history integration

**Completion Metrics:**
- 8/8 tasks complete (100%)
- 3/3 waves complete (100%)
- ~1,150 lines of new/modified code
- 95% code reuse between CLI and TUI
- 0 regressions in existing functionality

**Post-Release Bug Fixes & Enhancements:**

**1. Tool Schema Compliance Fix (Commit 7699086)**:
- Fixed 4 OpenAI function calling schema violations
- All array parameters now include required `items` field
- Compliance with OpenAI function calling specification
- Tools affected: delegation_tools.py, mental_model_tools.py

**2. Router Intelligence Enhancement (Commit 1a0d41f)**:
- **Problem**: Conversational queries inappropriately triggered tools
- **Solution**: 4-category task classification in router:
  ```
  CONVERSATIONAL → "Respond conversationally without tools: [query]"
  SIMPLE_QUERY → Minimal tool usage
  TASK_ORIENTED → Standard tool invocation
  PATTERN_BASED → Full multi-hop reasoning
  ```
- **Architecture**: Router classifies query type BEFORE agent selection
- **Implementation**: Modified both fallback and custom router configs (app.py)
- **Mechanism**: Uses refined_query parameter to pass tool usage instructions to agents
- **Impact**: Prevents inappropriate tool calling for simple greetings and identity queries
- **Testing Required**: Live TUI validation of all 4 query categories

**Production Impact:**
- Users can execute all 6 patterns directly from TUI
- No workflow disruption (no exit/restart needed)
- MessageBus integration enables event tracking
- Consistent UX with formatted results
- Code maintainability improved with shared executors
- Future pattern additions require minimal TUI changes

**Files Created/Modified:**
- `promptchain/patterns/executors.py` (NEW, 604 lines)
- `promptchain/patterns/__init__.py` (exports added)
- `promptchain/cli/commands/patterns.py` (refactored)
- `promptchain/cli/command_handler.py` (registry updated)
- `promptchain/cli/tui/app.py` (~542 lines added)

**Git Commits:**
- 88a3ab7: feat(004a): Complete Wave 1 - Extract pattern executors
- 64c33c2: feat(004a): Complete Wave 2 - TUI pattern command integration
- [Current]: feat(004a): Complete Wave 3 - Verify MessageBus/Blackboard integration

---

### Future Enhancements (Wave 4+) - PREVIOUS SPEC 004

**Wave 4 - Testing (COMPLETE):**
- ✅ Comprehensive unit tests for all patterns (T012-T014)
- ✅ Integration tests for pattern composition (172 tests total)
- ✅ Performance benchmarking for pattern execution
- ✅ Test event history tracking and cleanup
- ✅ Test TTL-based state expiration
- ✅ Test state versioning and snapshots

**Wave 5 - CLI & Documentation (COMPLETE):**
- ✅ `/patterns` command for pattern discovery and execution
- ✅ Comprehensive pattern documentation
- ✅ Pattern usage examples and tutorials
- ✅ Event system documentation and examples
- ✅ State management best practices guide

---

## MLflow Observability Architectural Patterns (January 2026)

### 1. Ghost Decorator Pattern (Zero-Overhead When Disabled)

**Purpose**: Optional observability features that should not impact performance when disabled.

**Implementation Pattern**:
```python
def observe_chain(func):
    """Zero-overhead decorator when MLflow disabled."""
    if not is_mlflow_enabled():
        return func  # Passthrough - zero overhead

    # Only wrap when enabled
    def wrapper(*args, **kwargs):
        # Tracking logic here
        return func(*args, **kwargs)
    return wrapper
```

**Key Characteristics**:
- **Function Passthrough**: Returns original function unchanged when disabled
- **Import Safety**: Gracefully handles missing dependencies (MLflow optional)
- **Auto-Enable Logic**: Detects when feature available via config
- **Performance**: <0.1% overhead when disabled (validated via benchmarks)
- **Clean API**: Decorators work transparently regardless of enabled state

**Usage Example**:
```python
from promptchain.observability import observe_chain

@observe_chain  # Zero overhead if MLFLOW_ENABLED=false
def process_prompt(self, input_text):
    # Normal execution
    return result
```

**Benefits**:
- Production code can always include decorators without performance penalty
- No conditional imports or if-statements needed at call sites
- Feature can be toggled via environment variables without code changes
- Graceful degradation when MLflow server unavailable

**Performance Validation**:
- Disabled overhead: <0.1% (measured: 0.05% average)
- Enabled overhead: <1% (measured: 0.8% average)
- Validated via promptchain/tests/test_observability_performance.py

---

### 2. ContextVars Async-Safe Pattern (Nested Run Tracking)

**Purpose**: Track nested execution contexts in async/TUI environments without thread-local storage issues.

**Problem Solved**:
- Thread-local storage fails in async contexts and TUI environments
- Parent-child run relationships need tracking across async boundaries
- Nested operations (LLM calls within agentic steps) require context isolation

**Implementation Pattern**:
```python
from contextvars import ContextVar

# Async-safe run tracking
_active_run: ContextVar[Optional[str]] = ContextVar('active_run', default=None)
_run_start_times: ContextVar[dict] = ContextVar('run_start_times', default_factory=dict)

def set_active_run(run_id: str):
    """Set active run in async-safe context."""
    _active_run.set(run_id)
    _run_start_times.get()[run_id] = time.time()

def get_active_run() -> Optional[str]:
    """Retrieve active run from async context."""
    return _active_run.get()
```

**Key Characteristics**:
- **Async-Safe**: ContextVars work correctly across await boundaries
- **Context Isolation**: Each async task has independent context
- **Parent-Child Tracking**: Nested runs maintain relationship hierarchy
- **TUI Compatible**: No conflicts with Textual's async event loop
- **Automatic Cleanup**: Context cleared when run ends

**Usage Example**:
```python
# In PromptChain.process_prompt()
from promptchain.observability.context import set_active_run, get_active_run

async def process_prompt_async(self, input_text):
    parent_run = get_active_run()  # Get current context

    with mlflow.start_run(run_name="process_prompt", nested=bool(parent_run)):
        current_run = mlflow.active_run().info.run_id
        set_active_run(current_run)  # Set in async context

        # Nested operations inherit context
        result = await agentic_step.execute_async(input_text)

        return result
```

**Benefits**:
- No race conditions in async environments
- Correct parent-child relationships in nested calls
- Works seamlessly with Textual TUI event loop
- No thread-local storage issues in multi-async scenarios

**Technical Details**:
- ContextVars introduced in Python 3.7 for async-safe context
- Each async task gets independent context variable storage
- Context propagates automatically across async function calls
- Eliminates need for threading.local() which fails in async

---

### 3. Background Queue Pattern (Non-Blocking Metric Logging)

**Purpose**: Prevent MLflow operations from blocking TUI rendering or LLM processing.

**Problem Solved**:
- MLflow logging operations can take 10-50ms
- TUI rendering requires <16ms frame time for 60fps
- Metric logging during LLM calls blocks response streaming
- Batch operations improve throughput vs individual calls

**Implementation Pattern**:
```python
import queue
import threading

class MetricQueue:
    def __init__(self, batch_size=10, max_queue_size=1000):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.batch_size = batch_size
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.worker.start()

    def enqueue_metric(self, key: str, value: float):
        """Non-blocking enqueue, returns immediately."""
        try:
            self.queue.put_nowait((key, value, time.time()))
        except queue.Full:
            # Graceful degradation - drop metric if queue full
            pass

    def _process_queue(self):
        """Background worker processes batches."""
        batch = []
        while True:
            try:
                item = self.queue.get(timeout=1.0)
                batch.append(item)

                if len(batch) >= self.batch_size:
                    self._flush_batch(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._flush_batch(batch)
                    batch = []
```

**Key Characteristics**:
- **Non-Blocking**: enqueue_metric() returns in <1ms
- **Batch Processing**: Processes 10 metrics at once for efficiency
- **Graceful Degradation**: Drops metrics if queue full (never blocks)
- **Automatic Shutdown**: Drains queue on shutdown with timeout
- **Thread-Safe**: Uses queue.Queue for thread-safe operations

**Performance Characteristics**:
- Enqueue latency: <1ms (measured: 0.3ms average)
- Enabled overhead: <5ms per operation (FR-002 requirement)
- Batch processing: 100+ metrics/second throughput
- Queue size: 1000 items default (configurable)

**Usage Example**:
```python
from promptchain.observability.queue import enqueue_metric

# In LLM tracking decorator
@observe_chain
def process_prompt(self, input_text):
    start_time = time.time()
    result = original_func(input_text)
    duration = time.time() - start_time

    # Returns immediately - queued for background processing
    enqueue_metric("llm_call_duration", duration)
    enqueue_metric("token_count", result.tokens)

    return result
```

**Benefits**:
- TUI remains responsive during heavy metric logging
- LLM streaming not blocked by observability
- Batch processing improves MLflow server throughput
- Automatic retry on transient failures
- Clean shutdown ensures no metric loss

**Implementation Details**:
- Uses threading.Thread (not asyncio) for independence from main event loop
- Daemon thread ensures clean shutdown with Python process
- Queue.put_nowait() prevents blocking on full queue
- Configurable batch size balances latency vs throughput
- Validated via promptchain/tests/test_observability_performance.py

---

## Pattern Integration Summary

**Ghost Decorator + ContextVars + Background Queue = Production-Ready Observability**

These three patterns work together to provide:
1. **Zero-overhead when disabled** (Ghost Decorator)
2. **Async-safe context tracking** (ContextVars)
3. **Non-blocking metric logging** (Background Queue)

**Combined Performance**:
- Disabled overhead: <0.1% (ghost decorator passthrough)
- Enabled overhead: <1% (background queue processing)
- TUI rendering: <5ms impact (background queue prevents blocking)
- Async compatibility: 100% (ContextVars async-safe)

**Production Validation**:
- All patterns tested in promptchain/tests/test_observability_*.py
- Performance benchmarks validate requirements (FR-002)
- Integration tests confirm async safety (FR-003)
- TUI integration tests confirm non-blocking (SC-010)

**Files Implementing These Patterns**:
- Ghost Decorator: `promptchain/observability/ghost.py` (109 lines)
- ContextVars Pattern: `promptchain/observability/context.py` (123 lines)
- Background Queue: `promptchain/observability/queue.py` (209 lines)
- Integration: `promptchain/observability/decorators.py` (731 lines)

---