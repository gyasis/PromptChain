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

### Initialization
```python
from promptchain.utils.promptengineer import PromptEngineer

engineer = PromptEngineer(
    max_iterations=5,           # Maximum improvement iterations
    use_human_evaluation=True,  # Use human feedback
    verbose=False              # Detailed output setting
)
```

### Parameter Configuration

#### Programming Interface Parameters
Available when using the PromptEngineer class in Python scripts:
```python
# Core Parameters
max_iterations: int = 3        # Maximum improvement cycles
use_human_evaluation: bool = False  # Use human feedback instead of LLM
verbose: bool = False         # Enable detailed logging
store_steps: bool = True      # Store intermediate steps
full_history: bool = False    # Keep complete history

# Model Configuration
evaluator_model: str = "anthropic/claude-3-sonnet-20240229"  # Model for evaluation
improver_model: str = "openai/gpt-4o"  # Model for improvements
creator_models: List[str] = [  # Models for initial prompt creation
    "anthropic/claude-3-sonnet-20240229",
    "openai/gpt-4o-mini"
]

# Advanced Settings
test_inputs: Optional[List[str]] = None  # Test cases for validation
focus_area: str = "all"  # Specific aspect to focus improvements on
```

#### Command Line Interface Parameters
Available when using the prompt engineer via command line:
```bash
# Core Parameters
--max-iterations INT          # Maximum improvement cycles (default: 3)
--feedback [llm|human]        # Evaluation type (default: llm)
--verbose                     # Enable detailed logging
-i, --interactive            # Enable interactive mode

# Model Selection
--evaluator-model STRING     # Model for evaluation
--improver-model STRING      # Model for improvements

# Input/Output
--task STRING               # Task description for prompt creation
--initial-prompt STRING     # Starting prompt to improve
--output-file STRING        # Save location for final prompt
--test-inputs STRING        # Comma-separated test inputs
--focus [clarity|completeness|task_alignment|output_quality|all]

# Technique Configuration
--techniques STRING         # Space-separated list of techniques
```

### Available Techniques

#### Required Parameter Techniques
These techniques must include a parameter:
- `role_playing:profession` - Define expert role (e.g., `role_playing:scientist`)
- `style_mimicking:author` - Mimic writing style (e.g., `style_mimicking:Richard Feynman`)
- `persona_emulation:expert` - Emulate expert perspective (e.g., `persona_emulation:Warren Buffett`)
- `forbidden_words:words` - Specify words to avoid (e.g., `forbidden_words:maybe,probably,perhaps`)

#### Optional Parameter Techniques
These techniques can be used with or without parameters:
- `few_shot:[number]` - Include specific number of examples
- `reverse_prompting:[number]` - Generate self-reflection questions
- `context_expansion:[type]` - Expand context in specific direction
- `comparative_answering:[aspects]` - Compare specific aspects
- `tree_of_thought:[paths]` - Explore multiple reasoning paths

#### No-Parameter Techniques
Simple flags that modify prompt behavior:
- `step_by_step` - Break down complex processes
- `chain_of_thought` - Show detailed reasoning process
- `iterative_refinement` - Gradually improve output
- `contrarian_perspective` - Challenge assumptions
- `react` - Implement Reason-Act-Observe cycle

### Model Configuration

#### Single Model Usage
When using the same model for all evaluations:
```python
engineer = PromptEngineer(
    evaluator_model="openai/gpt-4",
    max_iterations=3
)
```

#### Multiple Models
Using different models for different aspects:
```python
engineer = PromptEngineer(
    evaluator_model="anthropic/claude-3-sonnet-20240229",
    improver_model="openai/gpt-4o",
    creator_models=[
        "anthropic/claude-3-sonnet-20240229",
        "openai/gpt-4o-mini"
    ]
)
```

### Output Storage and Access

#### Configuring Storage
```python
engineer = PromptEngineer(
    store_steps=True,     # Store intermediate steps
    full_history=False    # Don't return full history
)

# Process and store results
result = engineer.create_specialized_prompt(task_description)

# Access specific steps
evaluation_step = engineer.evaluator.get_step_output(1)
improvement_step = engineer.improver.get_step_output(1)
```

#### Tracking Progress
```python
# Enable verbose mode for detailed logging
engineer = PromptEngineer(verbose=True)

# Track improvements with full history
results = engineer.improve_prompt_continuously(
    initial_prompt,
    full_history=True
)

# Access improvement history
for step in results:
    print(f"Step {step['step']}:")
    print(f"Evaluation: {step.get('evaluation', 'N/A')}")
    print(f"Improvements: {step.get('improvements', 'N/A')}\n")
```

### Technique Combinations

You can combine multiple techniques for sophisticated prompt engineering:

```python
# Via programming interface
engineer.evaluator.add_techniques([
    "role_playing:scientist",
    "step_by_step",
    "few_shot:3"
])

# Via command line
python -m promptchain.utils.prompt_engineer \
    --techniques "role_playing:scientist step_by_step few_shot:3" \
    --focus clarity \
    --max-iterations 5
```

### Usage Examples
```python
# Create and optimize a prompt
specialized_prompt = engineer.create_specialized_prompt(
    task_description="Generate creative writing prompts"
)

# Test the prompt with different inputs
test_results = engineer.test_prompt(
    prompt=specialized_prompt,
    test_inputs=[
        "Write a story about space exploration",
        "Create a mystery plot"
    ]
)

# Continuous improvement with specific techniques
engineer.evaluator.add_techniques([
    "role_playing:writer",
    "style_mimicking:Ernest Hemingway",
    "step_by_step"
])
improved_prompt = engineer.improve_prompt_continuously(initial_prompt)
```

### Best Practices

1. **Technique Selection**
   - Start with role-based techniques for expertise
   - Add structural techniques (step_by_step, chain_of_thought) for clarity
   - Use optional parameters to fine-tune behavior
   - Combine complementary techniques

2. **Testing Strategy**
   - Test prompts with diverse inputs
   - Use `test_prompt()` to evaluate consistency
   - Monitor improvements across iterations
   - Validate against edge cases

3. **Improvement Process**
   - Start with basic techniques
   - Add complexity gradually
   - Monitor changes between iterations
   - Use verbose mode for debugging

4. **Parameter Guidelines**
   - Be specific with role parameters
   - Use descriptive style references
   - Keep forbidden word lists focused
   - Adjust parameters based on results

5. **Storage and History**
   - Enable `store_steps` for debugging
   - Use `full_history` for analysis
   - Track model parameters when needed
   - Clean up stored steps periodically

## Integration Examples

### Complete Workflow Example
```python
# Initialize components
chain = PromptChain(models=["gpt-3.5"], full_history=True)
builder = DynamicChainBuilder(base_model="gpt-3")
engineer = PromptEngineer(max_iterations=3)

# Create optimized prompts
base_prompt = engineer.create_specialized_prompt(
    "Generate creative writing prompts"
)

# Build dynamic chain
writing_chain = builder.create_chain(
    chain_id="creative_writing",
    instructions=[
        base_prompt,
        "Expand the prompt into a detailed outline",
        "Generate a first draft"
    ]
)

# Execute the complete workflow
final_output = chain.process_prompt(
    input_text="Write a story about time travel",
    chain=writing_chain
)
```

## Best Practices

1. **PromptChain Best Practices**
   - Keep instructions clear and specific
   - Use verbose mode during development
   - Test chains incrementally
   - Leverage memory functions for context

2. **DynamicChainBuilder Best Practices**
   - Start with simple chains
   - Use descriptive chain IDs
   - Test different execution modes
   - Document chain dependencies

3. **PromptEngineer Best Practices**
   - Begin with basic prompts
   - Incorporate human feedback when possible
   - Test with diverse inputs
   - Monitor improvement iterations

4. **General Tips**
   - Document your chains and prompts
   - Use version control for prompt templates
   - Monitor performance metrics
   - Regular testing and validation
   - Keep error handling robust

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