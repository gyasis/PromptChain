---
noteId: "2c8f15f0055111f0b67657686c686f9a"
tags: []

---

# Active Context: PromptChain

## Current Work Focus

**LATEST MAJOR ARCHITECTURE MILESTONE: Agentic Orchestrator Router Enhancement Design (October 2025)**

A critical architectural decision has been made to enhance AgentChain's router system through an AgenticStepProcessor-based orchestrator. This addresses fundamental limitations in the current single-step router that achieves only ~70% accuracy. The new design targets 95% accuracy through multi-hop reasoning, progressive context accumulation, and knowledge boundary detection.

**Strategic Implementation Approach:**
- **Phase 1 (CURRENT):** Validate solution through async wrapper function (non-breaking)
- **Phase 2 (FUTURE):** After validation, integrate as native AgentChain router mode

**Key Technical Decisions:**
- Leverage existing AgenticStepProcessor with `history_mode="progressive"` (critical for multi-hop reasoning)
- 5 internal reasoning steps for complex routing decisions
- Tool capability awareness and knowledge boundary detection
- Current date awareness for temporal context
- Seamless migration path from wrapper to native implementation

**Comprehensive PRD Created:** `/home/gyasis/Documents/code/PromptChain/prd/agentic_orchestrator_router_enhancement.md`

**PREVIOUS MAJOR ENHANCEMENT: AgenticStepProcessor History Modes Implementation Complete (January 2025)**

A critical enhancement to the AgenticStepProcessor has been completed: History Accumulation Modes. This fixes a fundamental limitation in multi-hop reasoning where context from previous tool calls was being lost, breaking true ReACT methodology for agentic workflows.

**NEW MAJOR IMPLEMENTATION MILESTONE: Terminal Tool Persistent Sessions Complete**

A critical infrastructure enhancement has been completed: Terminal Tool Persistent Sessions. This resolves a fundamental limitation that prevented multi-step terminal workflows from working properly in PromptChain applications.

**PREVIOUS MAJOR DEVELOPMENT MILESTONE: MCP Tool Hijacker Implementation Started**

A significant new feature development has begun: the MCP Tool Hijacker system. This represents a major enhancement to the PromptChain library that will enable direct MCP tool execution without LLM agent processing overhead.

**MCP Tool Hijacker Development Milestone (January 2025):**
- **Feature Purpose**: Direct MCP tool execution bypassing LLM agents for performance optimization
- **Architecture**: Modular design with MCPToolHijacker, ToolParameterManager, MCPConnectionManager classes
- **PRD Completed**: Comprehensive Product Requirements Document created with detailed technical specifications
- **Implementation Approach**: Non-breaking modular changes to existing PromptChain library
- **Integration Strategy**: Optional hijacker property in PromptChain class for seamless adoption
- **Performance Goals**: Sub-100ms tool execution latency, improved throughput for tool-heavy workflows
- **Development Phase**: Phase 1 - Core Infrastructure planning and design

**PREVIOUS MAJOR MILESTONE ACHIEVED: Real-Time Progress Tracking System Complete**

The Research Agent frontend has successfully completed implementation of a comprehensive real-time progress tracking system. This achievement builds upon the previous TailwindCSS 4.x crisis resolution and delivers a production-ready user experience for research session monitoring.

**Recent Major Achievement (January 2025):**
- **Real-Time Progress Tracking System**: Complete implementation with WebSocket architecture, reactive state management, and professional UX components
- **Progress Widget & Modal**: Seamless expand/minimize functionality with detailed progress visualization
- **Step-by-Step Visualization**: Real-time display of research workflow stages (Init → Search → Process → Analyze → Synthesize)
- **Demo System**: Comprehensive testing environment with realistic progress simulation
- **Ready for Production**: Prepared for backend WebSocket integration

The project continues to focus on building a flexible prompt engineering library that enables:

1. **Core Functionality Enhancement**: 
   - Refining the core PromptChain class implementation
   - Enhancing function injection capabilities
   - Improving chainbreaker logic
   - Expanding PromptEngineer configuration options
   - Implementing comprehensive async support
   - Adding MCP server integration
   - Enhancing Memory Bank capabilities
   - Implementing robust tool integration in agentic steps
   - Adding simple agent-user chat loop with session persistence using Memory Bank

2. **Integration Examples**: 
   - Developing examples for major agent frameworks (AutoGen, LangChain, CrewAI)
   - Creating specialized prompt templates for common agent tasks
   - Implementing comprehensive prompt improvement techniques
   - Demonstrating async/MCP usage patterns
   - Building tool integration examples
   - Creating examples for using AgenticStepProcessor with tools

3. **Documentation**: 
   - Documenting the API and usage patterns
   - Creating examples and tutorials
   - Comprehensive documentation of PromptEngineer parameters and techniques
   - Detailing async capabilities and best practices
   - Explaining MCP server integration
   - Documenting AgenticStepProcessor usage and best practices

4. **Multimodal Processing**:
   - Enhancing the multimodal_ingest.py implementation
   - Improving integration between ingestors and LiteLLM
   - Expanding Gemini multimedia processing capabilities
   - Optimizing performance for large media files
   - Supporting async processing for media

## Recent Changes

1. **History Accumulation Modes for AgenticStepProcessor (January 2025)**:
   - **Core Problem Solved**: AgenticStepProcessor only kept the last assistant message + tool results, causing the AI to "forget" information from earlier tool calls during multi-hop reasoning workflows
   - **Three History Modes Implemented**: minimal (default, backward compatible), progressive (RECOMMENDED), kitchen_sink (maximum context)
   - **HistoryMode Enum Added**: Type-safe enum-based parameter for history mode selection (lines 35-39)
   - **Token Estimation Function**: Added estimate_tokens() for context size monitoring (lines 9-32)
   - **New Parameters**: history_mode (default: "minimal"), max_context_tokens (optional token limit warning)
   - **Mode-Based History Logic**: Lines 300-343 implement different accumulation strategies based on selected mode
   - **Backward Compatibility**: Fully non-breaking - defaults to original "minimal" behavior with deprecation warning
   - **Progressive Mode Details**: Accumulates assistant messages + tool results progressively for true multi-hop reasoning
   - **Kitchen Sink Mode**: Keeps everything including all intermediate reasoning steps for maximum context
   - **Deprecation Warning**: Lines 132-136 warn that minimal mode may be deprecated in future versions
   - **Testing Complete**: All three modes tested and passing (test_history_modes_simple.py)
   - **HybridRAG Integration**: query_with_promptchain.py updated to use history_mode="progressive" for RAG workflows
   - **Impact**: Fixes multi-hop reasoning for RAG workflows, enables true context accumulation across tool calls
   - **Files Modified**: promptchain/utils/agentic_step_processor.py, hybridrag/query_with_promptchain.py
   - **Documentation Created**: HISTORY_MODES_IMPLEMENTATION.md provides comprehensive implementation guide

2. **Terminal Tool Persistent Sessions Implementation Complete (January 2025)**:
   - **Core Problem Solved**: Terminal commands previously ran in separate subprocesses, losing all state (environment variables, working directory, etc.) between commands
   - **Persistent Terminal Sessions**: Named sessions maintain state across commands using simplified file-based approach
   - **Environment Variable Persistence**: `export VAR=value` persists across separate commands
   - **Working Directory Persistence**: `cd /path` persists across separate commands
   - **Multiple Session Management**: Create, switch, and manage multiple isolated sessions
   - **Command Substitution Support**: `export VAR=$(pwd)` works correctly with clean output processing
   - **Session Isolation**: Different sessions maintain completely separate environments
   - **Backward Compatibility**: Existing TerminalTool code works unchanged (opt-in feature)
   - **Technical Implementation**: SimplePersistentSession (file-based state), SimpleSessionManager (multi-session management), enhanced TerminalTool with session methods
   - **Verification Complete**: All test cases passing including environment variables, directory persistence, command substitution, session switching, and complex workflows
   - **Files Created/Modified**: `/promptchain/tools/terminal/simple_persistent_session.py`, `/promptchain/tools/terminal/terminal_tool.py`, `/examples/session_persistence_demo.py`
   - **User Questions Resolved**: "Do terminal commands persist?" → YES, "If I activate NVM 22.9.0, does next command use it?" → YES, "Can I have named sessions?" → YES

2. **Real-Time Progress Tracking System Implementation (January 2025)**:
   - **Core Components Built**: ProgressTracker.svelte, ProgressModal.svelte, ProgressWidget.svelte with complete WebSocket integration architecture
   - **Reactive State Management**: Implemented progress.ts Svelte 5 store with TypeScript for seamless state synchronization across components
   - **Professional UX Features**: Real-time percentage updates (tested 4% → 46%), step-by-step workflow visualization, smooth animations
   - **Interactive Controls**: Expand widget to modal, minimize modal to widget, connection status with reconnection logic
   - **Dashboard Integration**: Seamlessly integrated with main Research Agent dashboard, automatic session tracking
   - **Demo Simulation**: Complete testing environment with realistic research workflow progression
   - **User Flow Verified**: Tested complete cycle from session creation through progress tracking to completion
   - **Production Ready**: WebSocket architecture prepared, only backend endpoint connections needed

2. **Research Agent Frontend Crisis Resolution (January 2025)**:
   - **Critical Issue Resolved**: TailwindCSS 4.x @apply directive failures causing complete CSS breakdown
   - **Root Cause Identified**: Mixing TailwindCSS 3.x JavaScript config patterns with 4.x CSS-first architecture
   - **Research & Discovery**: Used Gemini debugging + Context7 official documentation for deep technical investigation
   - **Solution Implemented**: Complete migration to CSS-first @theme directive approach in app.css
   - **Design System Established**: Orange/coral primary colors (#ff7733) with professional component architecture
   - **Visual Verification**: Playwright browser testing confirmed perfect styling implementation
   - **Impact**: Unblocked entire frontend development pipeline for Research Agent dashboard

2. **Asynchronous Capabilities**:
   - Added comprehensive async support across core functionality
   - Implemented async versions of key methods (process_prompt, run_model, etc.)
   - Added MCP server integration with async connection handling
   - Maintained backward compatibility with synchronous methods
   - Enhanced error handling for async operations

2. **MCP Integration**:
   - Added support for Model Context Protocol (MCP) servers
   - Implemented flexible server configuration options
   - Added tool discovery and management
   - Created async connection handling with proper lifecycle management
   - Enhanced error handling for MCP operations
   - Integrated Context7 for library documentation
   - Added support for sequential thinking tools

3. **Memory Bank Implementation**:
   - Added comprehensive Memory Bank functionality for state persistence
   - Implemented namespace-based memory organization
   - Created core memory operations (store, retrieve, check, list, clear)
   - Added memory function creation for chain steps
   - Implemented specialized memory chains with built-in memory capabilities
   - Integrated with async operations for concurrent contexts
   - Added MCP server integration with memory capabilities
   - Created documentation in memory_bank_guide.md
   - Implemented chat context management for conversational applications
   - Added memory-based conversation history tracking
   - Created specialized chat memory namespaces for user preferences and session data
   - Integrated with WebSocket servers for real-time chat applications

4. **Router Improvements**:
   - Enhanced chat endpoint to detect and handle router JSON parsing errors
   - Implemented smart extraction of router plans from error messages
   - Added fallback processing for multi-agent plans when JSON parsing fails
   - Created robust handling for code blocks in user inputs
   - Updated router prompt to properly handle and escape special characters in JSON
   - Added example JSON formatting for code with special characters
   - Implemented plan extraction from malformed JSON to preserve intended agent sequences
   - Created graceful fallback to direct agent execution when needed

5. **PromptEngineer Enhancements**:
   - Added comprehensive documentation for command line parameters
   - Implemented configurable model parameters (temperature, max_tokens, etc.)
   - Added support for multiple improvement techniques
   - Created focus areas for targeted prompt enhancement
   - Improved interactive mode functionality

6. **Core Implementation**:
   - Implemented the primary PromptChain class in `promptchain/utils/promptchaining.py`
   - Added support for multiple model providers through LiteLLM
   - Implemented function injection, history tracking, and chainbreakers
   - Added async/sync dual interface support
   - Integrated MCP server management
   - Improved tool execution with robust function name extraction

7. **Support Utilities**:
   - Added logging utilities in `logging_utils.py`
   - Created prompt loading functionality in `prompt_loader.py`
   - Implemented async utility functions
   - Added MCP connection management utilities

8. **Project Structure**:
   - Established core package structure
   - Created initial examples and test directories
   - Set up development environment with requirements

9. **Ingestors Implementation**:
   - Developed specialized ingestors for different content types:
     - ArXiv papers (`arxiv.py`)
     - Web content (`crawler.py`, `singlepage_advanced.py`)
     - YouTube videos (`youtube_subtitles_processor`)
     - Technical blog posts (`marktechpost.py`)
   - Created `multimodal_ingest.py` for handling various content types
   - Implemented Gemini integration for multimedia processing

10. **Dynamic Chain Execution**:
   - Implemented three execution modes (serial, parallel, independent)
   - Added group-based chain organization
   - Created execution group management
   - Added parallel execution support
   - Enhanced chain merging capabilities

11. **Documentation Updates**:
   - Created comprehensive examples documentation
   - Updated system patterns documentation
   - Added best practices and common patterns
   - Documented parallel execution patterns

12. **Core Implementation**:
   - Enhanced DynamicChainBuilder class
   - Added execution mode validation
   - Implemented group-based execution
   - Added status tracking and monitoring
   - Enhanced chain insertion and merging

13. **Agent Orchestration**: Implemented and documented the `AgentChain` class for orchestrating multiple agents.
14. **Key Features Added**: Flexible routing (simple rules, configurable LLM/custom function), direct agent execution (`@agent_name:` syntax), structured logging.
15. **Chat Loop Feature**: Added a simple agent-user chat loop using PromptChain, with session persistence and conversation history via Memory Bank.

16. **AgenticStepProcessor**:
   - Added support for an optional model_name parameter
   - Updated llm_runner_callback to support this
   - Ensured backward compatibility
   - Fixed tool execution with robust function name extraction
   - Added comprehensive documentation on usage patterns
   - Implemented proper integration with PromptChain
   - Fixed issues with endless loops in tool execution
   - Added support for various tool call formats

## Next Steps

1. **MCP Tool Hijacker Implementation (New High Priority)**:
   - Begin Phase 1 implementation: Core Infrastructure development
   - Create MCPConnectionManager class with connection pooling and tool discovery
   - Implement basic MCPToolHijacker skeleton with direct tool execution framework
   - Develop ToolParameterManager for static parameter handling and transformation
   - Set up comprehensive unit testing infrastructure for MCP tool hijacking
   - Create integration points with existing PromptChain MCP infrastructure
   - Design parameter validation and schema checking system
   - Plan Phase 2: Parameter Management and validation framework

2. **Research Agent Backend Integration (Immediate Priority)**:
   - Connect progress tracking system to actual FastAPI WebSocket endpoints at `/ws/progress/{sessionId}`
   - Replace demo simulation with real research pipeline event streams
   - Implement session persistence and recovery for page reloads
   - Integrate real research metrics and timing data from backend
   - Build interactive chat interface using established design system
   - Create data visualization components for research results
   - Add file upload functionality for document analysis
   - Build export functionality for research findings (PDF, Word, JSON)

2. **Short-term Priorities**:
   - Enhance MCP server connection reliability
   - Add more comprehensive async error handling
   - Implement connection pooling for MCP servers
   - Create examples demonstrating async/MCP usage
   - Add monitoring for async operations
   - Expand test coverage for async functionality
   - Implement graceful shutdown for MCP connections
   - Implement persistent storage backends for Memory Bank
   - Add memory expiration and TTL features
   - Create examples demonstrating Memory Bank usage patterns
   - Develop comprehensive examples for AgenticStepProcessor usage
   - Enhance error handling for tool execution

2. **Medium-term Goals**:
   - Develop advanced MCP server management
   - Create a unified tool registry system
   - Build visualization tools for async operations
   - Enhance async/parallel processing capabilities
   - Implement sophisticated error recovery for MCP
   - Add support for more MCP server types
   - Create MCP server templates for common use cases
   - Improve tool handling with more advanced extraction and execution patterns

3. **Long-term Vision**:
   - Create a community-contributed prompt template repository
   - Build a web interface for designing and testing chains
   - Develop advanced prompt optimization techniques
   - Explore automated prompt engineering approaches
   - Support additional media types (3D models, specialized documents)
   - Implement content chunking for very large files
   - Create distributed MCP server architecture
   - Develop advanced agentic capabilities with multi-step reasoning

4. **Performance Optimization**:
   - Optimize parallel execution
   - Implement caching for chain outputs
   - Add resource usage monitoring
   - Enhance async operation efficiency
   - Optimize MCP server communication
   - Improve tool execution performance

5. **Error Handling**:
   - Enhance error recovery mechanisms
   - Add retry logic for failed chains
   - Implement chain rollback capabilities
   - Improve async error handling
   - Add MCP server error recovery
   - Enhance tool execution error handling

6. **Monitoring and Debugging**:
   - Add detailed execution logging
   - Create visualization tools for chain dependencies
   - Implement performance metrics collection
   - Monitor async operations
   - Track MCP server health
   - Add detailed logging for tool executions

7. **Integration Features**:
   - Add support for distributed execution
   - Implement chain persistence
   - Create chain templates system
   - Enhance MCP tool integration
   - Support advanced async patterns
   - Improve integration with external tool ecosystems

8. **AgentChain Development**: Test `AgentChain` with more complex scenarios, potentially refine the default LLM router prompt, explore passing refined queries from router to agents.

9. **AgenticStepProcessor Enhancements**:
   - Add support for memory persistence across steps
   - Implement more sophisticated tool selection logic
   - Enhance context management between steps
   - Create specialized templates for common agentic tasks
   - Develop visualization tools for agentic reasoning flow

## Active Decisions and Considerations

1. **MCP Tool Hijacker Architecture Strategy**:
   - **Modular Design Approach**: Implementing as separate, focused components (MCPToolHijacker, ToolParameterManager, MCPConnectionManager)
   - **Non-Breaking Integration**: Ensuring existing PromptChain MCP functionality remains unchanged
   - **Performance Optimization**: Targeting sub-100ms tool execution for performance-critical operations
   - **Parameter Management Strategy**: Supporting static/default parameters with dynamic override capabilities
   - **Schema Validation**: Implementing robust parameter validation against MCP tool schemas
   - **Testing Strategy**: Comprehensive mock infrastructure for reliable MCP tool testing
   - **Integration Pattern**: Optional hijacker property for backward compatibility

2. **Frontend Architecture Strategy**:
   - **TailwindCSS 4.x Adoption**: Committed to CSS-first architecture using @theme directive
   - **Component Design System**: Established orange/coral brand identity with professional styling
   - **CSS Variable Strategy**: Using custom CSS variables for consistent theming across components
   - **Svelte Integration**: Leveraging Svelte with TypeScript for reactive UI components
   - **API Integration**: Planning WebSocket connections for real-time features
   - **Mobile Responsiveness**: Ensuring mobile-first design patterns throughout

2. **Async Implementation Strategy**:
   - Balancing sync and async interfaces
   - Managing connection lifecycles effectively
   - Handling errors in async contexts
   - Optimizing async performance
   - Supporting both sync and async workflows

2. **MCP Integration Approach**:
   - Standardizing server configurations
   - Managing tool discovery and updates
   - Handling server connection states
   - Implementing proper error recovery
   - Supporting multiple server types

3. **API Design**:
   - Keeping the core API simple while enabling advanced functionality
   - Balancing flexibility vs. prescriptive patterns
   - Determining the right level of abstraction for model interactions
   - Supporting both sync and async patterns
   - Integrating MCP capabilities seamlessly

4. **Model Provider Strategy**:
   - Using LiteLLM for provider abstraction
   - Supporting model-specific parameters while maintaining a consistent interface
   - Planning for new model architectures and capabilities
   - Integrating multimodal capabilities across different providers
   - Supporting async operations across providers

5. **Memory Bank Architecture**:
   - Evaluating persistent storage options (Redis, SQLite)
   - Determining best approaches for memory lifecycle management
   - Balancing memory persistence vs. performance
   - Designing secure memory storage for sensitive information
   - Implementing efficient memory sharing between chains

6. **Tool Integration Strategy**:
   - Balancing between different tool call formats
   - Standardizing function name and argument extraction
   - Handling errors in tool execution gracefully
   - Supporting diverse tool response formats
   - Integrating with external tool ecosystems

7. **Architectural Refactor for History Management**:
   - **Objective**: Overhaul the history management in `AgentChain` by making the `auto_include_history` parameter more powerful.
   - **Key Implementation Details**:
       - `auto_include_history=False`: (Stateless) No history is tracked.
       - `auto_include_history=True`: (Simple Stateful) The default internal list-based history is used.
       - `auto_include_history=<Dict>`: (Managed Stateful) A dictionary of parameters is passed to instantiate the `ExecutionHistoryManager` for advanced, token-aware history management.

## Current Challenges

1. **Async Operation Management**:
   - Ensuring proper connection cleanup
   - Handling concurrent operations efficiently
   - Managing async context properly
   - Debugging async workflows effectively
   - Optimizing async performance

2. **MCP Server Reliability**:
   - Handling connection failures gracefully
   - Managing server state effectively
   - Implementing proper error recovery
   - Optimizing tool discovery process
   - Supporting various server configurations

3. **Performance Optimization**:
   - Improving response times for long chains
   - Optimizing memory usage during execution
   - Enhancing parallel processing efficiency
   - Managing provider rate limits effectively
   - Balancing flexibility vs. performance

4. **Tool Execution Reliability**:
   - Handling different tool call formats consistently
   - Managing errors during tool execution
   - Preventing infinite loops in tool calls
   - Supporting different model providers' tool calling formats
   - Optimizing tool execution performance 