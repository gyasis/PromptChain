# PromptChain Framework: Complete Implementation Guide

PromptChain is a powerful framework that specializes in two core capabilities:
1. **Advanced Prompt Engineering**: Create, test, and optimize prompts through systematic iteration and evaluation
2. **Flexible LLM Execution**: Chain multiple LLM calls and functions together in sophisticated processing pipelines

## Key Features

### Prompt Engineering Capabilities
- Iterative prompt refinement with automated evaluation
- Rich technique library for prompt optimization
- Built-in testing and validation
- Human-in-the-loop feedback integration

### LLM Execution Features
- Sequential and parallel processing chains
- Function integration for custom logic
- Multiple model support with parameter control
- Memory management and state persistence

### Quick Start Examples

#### Simple LLM Chain
```python
# Basic chain for content generation
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Write about: {input}",
        "Improve the writing: {input}",
        "Add examples to: {input}"
    ]
)
result = chain.process_prompt("Quantum computing basics")
```

#### Prompt Engineering Chain
```python
# Chain for prompt optimization
engineer = PromptEngineer(
    max_iterations=3,
    use_human_evaluation=False,
    verbose=True
)

# Create and optimize a specialized prompt
task = "Create an agent that helps with code review"
optimized_prompt = engineer.create_specialized_prompt(task)

# Test the optimized prompt
test_results = engineer.test_prompt(
    prompt=optimized_prompt,
    test_inputs=[
        "Review a Python web app",
        "Check a JavaScript library",
        "Analyze a C++ game engine"
    ]
)
```

#### Advanced Processing Chain
```python
# Chain combining LLMs, functions, and memory
def validate_code(text: str) -> str:
    """Custom validation function"""
    return "Code validation results..."

chain = PromptChain(
    models=[
        {
            "name": "openai/gpt-4",
            "params": {"temperature": 0.7}
        },
        "anthropic/claude-3-sonnet-20240229"
    ],
    instructions=[
        "Analyze this code: {input}",
        validate_code,
        "Generate improvements: {input}",
        "Create test cases: {input}"
    ],
    store_steps=True,
    verbose=True
)
```

## When to Use PromptChain

### For Prompt Engineering
- When you need systematic prompt optimization
- For creating reusable prompt templates
- When testing prompts across different scenarios
- To maintain prompt quality and consistency

### For LLM Execution
- When combining multiple LLM calls in sequence
- For mixing LLMs with custom processing functions
- When you need sophisticated flow control
- For handling complex data transformations

### For Both
- Building AI agents with optimized prompts
- Creating robust data processing pipelines
- Implementing quality control in LLM applications
- Developing maintainable AI workflows

## Table of Contents
1. [Core Components Overview](#core-components-overview)
2. [PromptChain: The Orchestrator](#promptchain-the-orchestrator)
3. [DynamicChainBuilder: Flexible Chain Creation](#dynamicchainbuilder-flexible-chain-creation)
4. [PromptEngineer: Optimizing Prompts](#promptengineer-optimizing-prompts)
5. [Integration Examples](#integration-examples)
6. [Best Practices](#best-practices)
7. [Function Integration in PromptChain](#function-integration-in-promptchain)
8. [Advanced PromptChain Example](#advanced-promptchain-example)
9. [Decision Chain Example](#decision-chain-example)

## Core Components Overview

The PromptChain framework consists of three main components that work together to create powerful, flexible, and optimized prompt chains:

1. **PromptChain**: The central orchestrator that manages prompt sequences
2. **DynamicChainBuilder**: Creates and modifies chains dynamically
3. **PromptEngineer**: Optimizes and refines individual prompts

## Asynchronous Capabilities

PromptChain supports both synchronous and asynchronous operation modes, making it flexible for different use cases:

### Synchronous Usage (Default)
```python
from promptchain.utils.promptchaining import PromptChain

chain = PromptChain(
    models=["gpt-4"],
    instructions=["Process this: {input}"]
)

# Synchronous execution
result = chain.process_prompt("Hello!")
```

### Asynchronous Usage
```python
import asyncio
from promptchain.utils.promptchaining import PromptChain

async def main():
    chain = PromptChain(
        models=["gpt-4"],
        instructions=["Process this: {input}"],
        mcp_servers=[{  # Optional MCP server configuration
            "id": "local_server",
            "type": "stdio",
            "command": "python",
            "args": ["path/to/server.py"],
            "env": {"CUSTOM_VAR": "value"}
        }]
    )

    # Async execution
    result = await chain.process_prompt_async("Hello!")
    
    # When using MCP servers, remember to close connections
    await chain.close_mcp_async()

# Run the async code
asyncio.run(main())
```

### Available Async Methods
All core PromptChain methods have both synchronous and asynchronous versions:

| Synchronous Method | Asynchronous Version | Description |
|-------------------|---------------------|-------------|
| `process_prompt()` | `process_prompt_async()` | Process input through the chain |
| `run_model()` | `run_model_async()` | Execute a single model call |
| `connect_mcp()` | `connect_mcp_async()` | Connect to MCP servers |
| `close_mcp()` | `close_mcp_async()` | Close MCP connections |

### When to Use Async
- When integrating with async web frameworks (FastAPI, aiohttp)
- When using Model Context Protocol (MCP) servers
- For parallel processing of multiple chains
- When making multiple concurrent API calls

### MCP Integration
The async capabilities are particularly useful when working with Model Context Protocol (MCP) servers:

```python
chain = PromptChain(
    models=["gpt-4"],
    instructions=["Use available tools to: {input}"],
    mcp_servers=[
        {
            "id": "tools_server",
            "type": "stdio",
            "command": "python",
            "args": ["tools_server.py"]
        },
        {
            "id": "data_server",
            "type": "stdio",
            "command": "/usr/local/bin/node",
            "args": ["-y", "@modelcontextprotocol/server-data"]
        }
    ]
)

# Tools from both servers will be automatically discovered
# and made available to the chain
```

## PromptChain: The Orchestrator

### Initialization
```python
from promptchain.utils.promptchaining import PromptChain

chain = PromptChain(
    models=["gpt-3.5"],  # List of models to use
    instructions=[
        "Translate the following English text to French: {input_text}",
        "Summarize the following text: {translated_text}",
    ],
    full_history=True,   # Keep track of all steps
    store_steps=False,   # Don't store intermediate steps
    verbose=True        # Show detailed output
)
```

### Key Parameters
- `models`: List of LLM models to use
- `instructions`: List of templated instructions
- `full_history`: Boolean to track execution history
- `store_steps`: Boolean to store intermediate results
- `verbose`: Boolean for detailed logging

### Usage Example
```python
# Process a single prompt through the chain
output = chain.process_prompt(input_text="Hello, how are you?")

# Access results
print(output)
```

## DynamicChainBuilder: Flexible Chain Creation

### Initialization
```python
from promptchain.utils.promptchaining import DynamicChainBuilder

builder = DynamicChainBuilder(
    base_model="gpt-3",
    base_instruction="Explain the concept of {topic}."
)
```

### Key Parameters
- `base_model`: The default LLM model
- `base_instruction`: Template for basic instructions
- `chain_id`: Unique identifier for each chain
- `execution_mode`: How steps are executed ("sequential" or "parallel")

### Creating and Executing Chains
```python
# Create a new chain
chain = builder.create_chain(
    chain_id="topic_explainer",
    instructions=[
        "Define {topic}.",
        "Discuss the importance of {topic}.",
        "Provide examples of {topic}."
    ],
    execution_mode="sequential"
)

# Execute the chain
output = builder.execute_chain(
    chain_id="topic_explainer",
    input_data={"topic": "Dynamic Programming"}
)

# Add advanced techniques
builder.add_techniques([
    "role_playing:teacher",
    "style_mimicking:Albert Einstein"
])
```

## PromptEngineer: Optimizing Prompts

PromptChain includes a powerful prompt engineering system for systematic prompt optimization. For comprehensive documentation on prompt engineering features, techniques, and usage examples, see:

**📚 [Complete Prompt Engineering Guide](prompt_engineer.md)**

### Quick Overview

The PromptEngineer component provides:
- **Iterative Prompt Improvement**: Automated refinement through multiple iterations
- **Technique Library**: 20+ prompt engineering techniques (role playing, chain of thought, etc.)
- **Human & LLM Evaluation**: Choose between automated or human-guided feedback
- **Protected Content**: Keep code blocks and variables unchanged during improvement
- **CLI & Programmatic Access**: Use via command line or Python API

### Basic Usage

```python
from promptchain.utils.prompt_engineer import PromptEngineer

# Initialize with custom settings
engineer = PromptEngineer(
    max_iterations=5,
    verbose=True,
    evaluator_model="anthropic/claude-3-sonnet-20240229",
    protect_content=True
)

# Generate optimized prompt
task = "Create a technical documentation writer agent"
optimized_prompt = engineer.create_specialized_prompt(task)
```

### Command Line Usage

```bash
# Interactive mode with technique selection
python -m promptchain.utils.prompt_engineer -i

# Direct prompt improvement
python -m promptchain.utils.prompt_engineer \
    --task "Generate SQL queries from natural language" \
    --techniques role_playing:database_expert step_by_step \
    --max-iterations 5 \
    --output-file optimized_prompt.txt
```

### Available Techniques
- **Role Playing**: `role_playing:scientist`, `role_playing:teacher`
- **Reasoning**: `chain_of_thought`, `step_by_step`, `tree_of_thought:3`
- **Examples**: `few_shot:3`, `reverse_prompting:5`
- **Style**: `style_mimicking:technical`, `persona_emulation:expert`
- **Quality Control**: `forbidden_words:maybe,probably`, `iterative_refinement`

For detailed technique documentation, parameters, and advanced usage patterns, see the [Complete Prompt Engineering Guide](prompt_engineer.md).

## Advanced Workflow: Chain followed by Interactive Chat

A common pattern is to use a `PromptChain` for automated processing or generation and then allow the user to interactively discuss or refine the output.

Since an interactive chat loop (using `input()`) cannot be a direct step within an automated `PromptChain`, this workflow is typically implemented by running the initial chain first and then passing its output as context to start a managed, interactive chat session.

1.  **Phase 1: Initial Processing Chain**
    - Define a standard `PromptChain` to perform the initial task (e.g., summarization, data extraction, content generation).
    - Run this chain with the initial input.
    - Capture the final output.

2.  **Phase 2: Interactive Discussion Chat**
    - Use a class like `ManagedChatSession` (as demonstrated in examples) which encapsulates its own internal `PromptChain`, chat loop logic, and history management.
    - Initialize this chat session class, crucially passing the output from Phase 1 as an `initial_context` argument.
    - The chat session's internal prompt/chain should be configured to utilize this initial context along with the ongoing conversation history.
    - Call the chat session's `run()` method to start the interactive loop where the user can ask questions or provide feedback related to the initial output.

**Example Structure (`task_chain_to_chat.py`):**

```python
# (Import ManagedChatSession and PromptChain)

async def main_workflow():
    # --- Phase 1: Initial Processing ---
    print("--- Phase 1: Initial Processing ---")
    initial_task = input("Enter the initial task: ")
    
    initial_processor_chain = PromptChain(...)
    initial_output = await initial_processor_chain.process_prompt_async(initial_task)
    
    print("\n--- Initial Processing Output ---")
    print(initial_output)
    print("--------------------------------")
    
    # --- Phase 2: Interactive Chat Session ---
    print("\n--- Phase 2: Starting Chat Session about the Output ---")
    
    # Define template/config for the chat session's internal chain
    chat_instruction_template = [...]
    chat_history_injection_index = ...
    
    # Create the chat session, passing Phase 1 output as context
    chat_session = ManagedChatSession(
        ...,
        initial_context=initial_output # Key step!
    )
    
    # Start the interactive chat
    await chat_session.run()
    
    print("\n--- Workflow Complete ---")

if __name__ == "__main__":
    asyncio.run(main_workflow())
```

This pattern allows for automated processing followed by focused, context-aware user interaction, leveraging the strengths of both automated chains and interactive loops.

## Decision Chain Example

A "Decision Chain" uses an early step, often involving an LLM, to classify the input or determine the required processing path. Subsequent steps then branch based on this decision. This is useful for directing requests to the most appropriate logic (e.g., simple vs. complex handling, different specialized tools, etc.).

**Concept:**

1.  **Input Analysis:** An LLM analyzes the user's input using a carefully crafted prompt.
2.  **Classification Output:** The LLM is instructed to output a simple, standardized token representing the decision (e.g., "QUICK", "DEEP").
3.  **Routing Function:** A Python function receives the LLM's decision token.
4.  **Conditional Execution:** The routing function calls different downstream functions (or could trigger other `PromptChain` instances) based on the token.

**Implementation:**

```python
# decision_chain_example.py
import asyncio
from promptchain.utils.promptchaining import PromptChain

# --- Placeholder Functions ---
def handle_quick_answer(decision_context: str) -> str:
    """Placeholder for the quick answer processing path."""
    print(f"LLM Decision Context: {decision_context}")
    # In a real scenario, this could trigger another simple chain or function.
    return "-> Quick Answer Path Chosen. Processing initiated."

def handle_deep_investigation(decision_context: str) -> str:
    """Placeholder for the deep investigation processing path."""
    print(f"LLM Decision Context: {decision_context}")
    # In a real scenario, this could trigger a complex chain, research agent, etc.
    return "-> Deep Investigation Path Chosen. Processing initiated."

# --- Routing Function ---
def route_based_on_decision(llm_decision: str) -> str:
    """Routes processing based on the LLM's simple output ('QUICK' or 'DEEP')."""
    decision = llm_decision.strip().upper()
    # Use 'in' for robustness against minor variations (e.g., extra spaces)
    if "QUICK" in decision:
        return handle_quick_answer(llm_decision)
    elif "DEEP" in decision:
        return handle_deep_investigation(llm_decision)
    else:
        # Fallback or error handling for unexpected LLM output
        print(f"Warning: Unexpected decision '{llm_decision}'. Defaulting to quick.")
        return handle_quick_answer(f"Unexpected: {llm_decision}")

# --- Decision Prompt Template ---
# This prompt is crucial. It instructs the LLM to act as a classifier.
decision_prompt_template = \"\"\"
Analyze the user's request below. Determine if the user wants a brief, quick answer or a detailed, deep investigation.

Consider keywords:
- Quick/Brief: 'summary', 'define', 'what is', 'short', 'overview', 'quick question'
- Deep/Detailed: 'explain in detail', 'how does', 'compare', 'analyze', 'examples', 'elaborate', 'deep dive'

Based on your analysis, output ONLY the single word 'QUICK' or 'DEEP'. Do not add any explanation or other text.

User Request:
"{input}"

Your Decision (QUICK or DEEP):
\"\"\"

# --- Create the Decision Chain ---
decision_chain = PromptChain(
    # Using a potentially faster/cheaper model suitable for classification
    models=["openai/gpt-4o-mini"],
    instructions=[
        decision_prompt_template, # LLM makes the decision
        route_based_on_decision  # Python function routes based on decision
    ],
    verbose=True # Set to False for cleaner output once tested
)

# --- Test Cases ---
async def run_tests():
    print("--- Testing Decision Chain ---")

    # Test Case 1: Likely "QUICK"
    quick_request = "What is the definition of photosynthesis?"
    print(f"\n[Test 1] Input: {quick_request}")
    result_quick = await decision_chain.process_prompt_async(quick_request)
    print(f"[Test 1] Final Output: {result_quick}")
    print("-" * 20)

    # Test Case 2: Likely "DEEP"
    deep_request = "Explain in detail how photosynthesis works, including the light-dependent and light-independent reactions, and provide examples of different types of plants."
    print(f"\n[Test 2] Input: {deep_request}")
    result_deep = await decision_chain.process_prompt_async(deep_request)
    print(f"[Test 2] Final Output: {result_deep}")
    print("-" * 20)

    # Test Case 3: Ambiguous (Tests robustness/fallback)
    ambiguous_request = "Tell me about photosynthesis."
    print(f"\n[Test 3] Input: {ambiguous_request}")
    result_ambiguous = await decision_chain.process_prompt_async(ambiguous_request)
    print(f"[Test 3] Final Output: {result_ambiguous}")
    print("-" * 20)

if __name__ == "__main__":
    # Save this code as e.g., 'decision_chain_example.py' and run it
    asyncio.run(run_tests())
```

This example showcases how `PromptChain` can integrate LLM-based decision-making with standard Python function logic to create dynamic processing workflows. The key is the carefully designed prompt that forces the LLM into a classification role with a simple, predictable output format.

## Function Integration in PromptChain

### Basic Function Usage
Functions can be mixed with model instructions in a PromptChain:

```python
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text and return a summary."""
    # Your sentiment analysis logic here
    return f"Sentiment Analysis: {text}"

# Mix functions with model instructions
chain = PromptChain(
    models=["openai/gpt-4"],  # Only needed for non-function instructions
    instructions=[
        "Generate a product review: {input}",
        analyze_sentiment,  # Function doesn't need a model
        "Summarize the analysis: {input}"
    ]
)
```

### Function Requirements
- Functions must accept a single string input
- Functions must return a string output
- Functions are executed directly, not through models
- No model parameters needed for function steps

### Advanced Function Integration

#### Preprocessing Functions
```python
def preprocess_multimodal_input(input_data: str) -> str:
    """Convert various input types to text."""
    if input_data.endswith('.jpg'):
        return extract_text_from_image(input_data)
    elif input_data.endswith('.pdf'):
        return extract_text_from_pdf(input_data)
    return input_data

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        preprocess_multimodal_input,
        "Analyze the processed content: {input}",
        "Generate insights: {input}"
    ]
)
```

#### Data Transformation Functions
```python
def format_as_json(text: str) -> str:
    """Convert unstructured text to JSON format."""
    data = parse_text(text)
    return json.dumps(data, indent=2)

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Extract key information: {input}",
        format_as_json,
        "Analyze the structured data: {input}"
    ]
)
```

## Advanced PromptChain Example

Here's a comprehensive example that combines multiple advanced features:

```python
from typing import Dict, Any, Tuple
import json

# 1. Define utility functions
def preprocess_input(text: str) -> str:
    """Clean and standardize input text."""
    return text.strip().lower()

def validate_data(text: str) -> str:
    """Validate data format and content."""
    try:
        data = json.loads(text)
        return "Valid JSON data"
    except:
        return "Invalid data format"

def analyze_metrics(text: str) -> str:
    """Extract and analyze key metrics."""
    # Simulate metric analysis
    return "Metrics Analysis: Positive trends detected"

# 2. Define chainbreakers
def break_on_validation_error(step: int, output: str, step_info: Dict[str, Any]) -> Tuple[bool, str, str]:
    """Break chain if validation fails."""
    if step_info.get('type') == 'function' and 'Invalid' in output:
        return (True, "Validation failed", "Process stopped: Invalid data")
    return (False, "", None)

def break_on_low_confidence(step: int, output: str) -> Tuple[bool, str, str]:
    """Break chain if confidence is low."""
    if 'low confidence' in output.lower():
        return (True, "Low confidence detected", output)
    return (False, "", None)

# 3. Create memory management function
def create_memory_function(namespace: str = "default"):
    """Create a function for managing chain memory."""
    memory_store = {}
    
    def memory_manager(input_text: str) -> str:
        """Store or retrieve data from memory."""
        if input_text.startswith("STORE:"):
            key, value = input_text[6:].split("=", 1)
            memory_store[key.strip()] = value.strip()
            return f"Stored: {key}"
        elif input_text.startswith("GET:"):
            key = input_text[4:].strip()
            return memory_store.get(key, "Not found")
        return input_text
    
    return memory_manager

# 4. Create the advanced chain
def create_advanced_analysis_chain():
    """Create an advanced analysis chain with multiple features."""
    memory_func = create_memory_function("analysis")
    
    chain = PromptChain(
        models=[
            {
                "name": "openai/gpt-4",
                "params": {"temperature": 0.7}
            },
            "anthropic/claude-3-sonnet-20240229"
        ],
        instructions=[
            preprocess_input,  # Clean input
            "Extract key information from: {input}",
            validate_data,    # Validate format
            memory_func,      # Store intermediate results
            analyze_metrics,  # Analyze data
            """Generate comprehensive report:
            Previous Analysis: {input}
            Consider:
            1. Key trends
            2. Potential risks
            3. Recommendations
            """,
            "Summarize findings in JSON format: {input}"
        ],
        chainbreakers=[
            break_on_validation_error,
            break_on_low_confidence
        ],
        full_history=True,
        store_steps=True,
        verbose=True
    )
    
    return chain

# 5. Usage example
def process_data_with_advanced_chain(input_data: str):
    """Process data using the advanced chain."""
    chain = create_advanced_analysis_chain()
    
    try:
        result = chain.process_prompt(input_data)
        
        # Access step outputs if needed
        validation_step = chain.get_step_output(2)
        metrics_step = chain.get_step_output(4)
        
        return {
            "final_result": result,
            "validation": validation_step,
            "metrics": metrics_step
        }
    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    input_data = '''
    {
        "metrics": {
            "revenue": 1000000,
            "growth": 15.5,
            "satisfaction": 4.8
        }
    }
    '''
    
    results = process_data_with_advanced_chain(input_data)
    print(json.dumps(results, indent=2))
```

This advanced example demonstrates:
1. Function integration for data processing
2. Multiple chainbreakers for flow control
3. Memory management for state persistence
4. Mixed model usage with parameters
5. Error handling and validation
6. Step output tracking and access
7. Comprehensive data processing pipeline

The chain will:
1. Preprocess and clean input
2. Extract key information using GPT-4
3. Validate data format
4. Store intermediate results in memory
5. Analyze metrics
6. Generate a comprehensive report using Claude
7. Format the final output as JSON

It also includes:
- Automatic breaking on validation errors
- Confidence-based chain breaking
- Full history tracking
- Step output storage
- Verbose logging

---

This guide provides a comprehensive overview of the PromptChain framework and its components. For more detailed information about specific features or advanced usage, please refer to the individual component documentation.

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge) 