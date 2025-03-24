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