---
noteId: "0fcda2b0055111f0b67657686c686f9a"
tags: []

---

# System Patterns: PromptChain

## System Architecture

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

### 8. Chat Architecture
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

1. **API Dependencies**:
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

## Agentic Step Pattern

- **Concept:** To handle complex tasks within a single logical step of a `PromptChain`, the `AgenticStepProcessor` class provides an internal execution loop.
- **Core Components:**
  - **AgenticStepProcessor**: Encapsulates the internal agentic loop logic
  - **Tool Integration**: Uses functions registered in parent PromptChain
  - **Robust Function Name Extraction**: Helper utility to handle different tool call formats
  - **Internal Tool Execution**: Manages tool calls within the agentic step

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
  5. Results are formatted and incorporated into the next LLM call
  6. Process repeats until a final answer is reached or max steps are hit

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