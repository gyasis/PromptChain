---
noteId: "4ef50b40088511f09aa1cf0704e2b01f"
tags: []

---

# Head-Tail Technique for Dynamic Chain Building

The Head-Tail technique is a powerful pattern for creating dynamic, adaptive prompt chains that can respond to content analysis and context. Unlike traditional linear chains, head-tail chains use a "head" step to generate instructions or decompose tasks, followed by dynamic "tail" steps that adapt based on the head's output.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Structure Requirements](#structure-requirements)
3. [Implementation Guide](#implementation-guide)
4. [Advanced Pattern: Processing Sequences](#advanced-pattern-processing-sequences)
5. [Required Parameters](#required-parameters)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

## Core Concepts

The Head-Tail technique splits chain processing into two distinct phases:

- **Head Phase**: Analyzes input and determines what processing is needed
- **Tail Phase**: Executes the specific processing steps identified by the head

This creates a workflow where one chain (the head) instructs the creation or execution of other chains (the tail), allowing for dynamic adaptation to content.

## Structure Requirements

For a valid head-tail implementation, the following structure is **required**:

1. **Technique Declaration**: The builder must be initialized with `technique="head-tail"`
2. **Head Chain**: Must be a single instruction (string) focused on task decomposition
3. **Tail Chains**: Generated dynamically based on the head's output
4. **Group Organization**: All chains must be organized within named groups
5. **Dependencies**: Tail chains must specify dependencies appropriately

### Technical Requirements

```python
# Valid Head-Tail Structure
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction=HEAD_TEMPLATE,
    technique="head-tail"  # Required technique declaration
)

# Head chain (single instruction)
head_chain = builder.create_chain(
    "decomposer",
    "Analyze the following input and identify the necessary processing steps...",
    execution_mode="serial",
    group="analysis_system"  # Group parameter is required
)

# Tail chains are created dynamically based on head output
```

## Implementation Guide

### Step 1: Initialize the Builder

```python
from promptchain.utils.promptchaining import DynamicChainBuilder

# Head template for task decomposition
HEAD_TEMPLATE = """
Task Decomposition Framework
============================
Your task: {instruction}
Input context: {input}

Guidelines:
- Break down the complex task into specific sub-tasks
- Identify the necessary processing for each sub-task
- Format output as a structured list
"""

# Create builder with head-tail technique
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction=HEAD_TEMPLATE,
    technique="head-tail"  # Required
)
```

### Step 2: Create the Head Chain

```python
# Head chain for task decomposition
head_chain = builder.create_chain(
    "task_decomposer",
    "Analyze the input and identify specific sub-tasks that need to be performed.",
    execution_mode="serial",
    group="task_processing",  # Required group parameter
    dependencies=None  # Head typically has no dependencies
)
```

### Step 3: Execute the Head Chain

```python
# Execute the head chain to get sub-tasks
decomposition_result = builder.execute_chain("task_decomposer", "Complex task input...")
print(f"Task decomposition result:\n{decomposition_result}")
```

### Step 4: Parse Head Results and Create Tail Chains

```python
def create_processing_chains(decomposition: str) -> list:
    """Parse the head output and create appropriate tail chains."""
    # Extract sub-tasks from the decomposition
    sub_tasks = []
    for line in decomposition.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Extract the task from formatted lines
            task = line.lstrip("0123456789-. ")
            sub_tasks.append(task)
    
    # Create a chain for each sub-task
    chain_ids = []
    for i, task in enumerate(sub_tasks):
        chain_id = f"subtask_{i+1}"
        
        # Create a chain to process this specific sub-task
        builder.create_chain(
            chain_id,
            f"Process the following sub-task: {task}\n\n{{input}}",
            execution_mode="parallel",  # Can run in parallel with other sub-tasks
            group="task_processing",  # Same group as head chain (required)
            dependencies=["task_decomposer"]  # Depends on head chain
        )
        chain_ids.append(chain_id)
    
    return chain_ids
```

### Step 5: Execute the Tail Chains

```python
# Create and execute tail chains
tail_chains = create_processing_chains(decomposition_result)

# Execute all sub-task chains
results = {}
for chain_id in tail_chains:
    results[chain_id] = builder.execute_chain(chain_id, "Input for processing...")

# Final results
for chain_id, result in results.items():
    print(f"\nResult from {chain_id}:\n{result[:100]}...")
```

## Advanced Pattern: Processing Sequences

For complex processing, each tail chain can implement a sequence of operations:

```python
def create_processing_sequence(builder, task, sequence_id):
    """Create a complete processing sequence for a task."""
    # Create specialized functions for this sequence
    process_fn = create_processor_function(task)
    analyze_fn = create_analysis_function(task)
    summarize_fn = create_summary_function()
    
    # Register the sequence chain with multiple processing steps
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            f"Processing sequence for task: {task}\n\n{{input}}",
            process_fn,    # Step 1: Initial processing
            analyze_fn,    # Step 2: Analysis
            summarize_fn   # Step 3: Summary
        ],
        store_steps=True
    )
    
    # Register with required parameters
    builder.chain_registry[sequence_id] = {
        "chain": chain,
        "execution_mode": "parallel",
        "group": "processing_sequences",  # Required group parameter
        "dependencies": ["task_decomposer"],  # Depends on head chain
        "status": "created"
    }
    
    return sequence_id
```

## Required Parameters

### The `group` Parameter

The `group` parameter is **required** in head-tail chains for several critical reasons:

1. **Execution Management**: Groups organize related chains for collective execution
2. **Dependency Tracking**: Ensures proper sequencing between head and tail
3. **Resource Management**: Allows parallel execution of chains within a group
4. **State Persistence**: Maintains context across related chains
5. **Error Handling**: Isolates failures to specific groups

```python
# Creating a chain with the required group parameter
builder.create_chain(
    "chain_id",
    "Process the following input: {input}",
    execution_mode="serial",
    group="processing_group",  # Required in head-tail technique
    dependencies=["head_chain"]
)
```

### Execution Modes

The execution mode parameter determines how chains are scheduled:

| Mode | Description | Use Case |
|------|-------------|----------|
| `serial` | Runs in sequence with dependencies | For chains that depend on previous outputs |
| `parallel` | Can run simultaneously | For independent processing tasks |
| `independent` | Runs anytime, no specific scheduling | For utility chains |

## Best Practices

1. **Head Chain Design**
   - Keep the head chain focused solely on decomposition
   - Format output consistently for reliable parsing
   - Include clear instructions for sub-task identification

2. **Tail Chain Management**
   - Create specific chains for distinct sub-tasks
   - Consider execution mode carefully for each tail chain
   - Establish clear dependencies between chains

3. **Group Parameter Usage**
   - Use meaningful group names that reflect function
   - Keep related chains in the same group
   - Different processing domains should use different groups

4. **Performance Optimization**
   - Use parallel execution mode for independent tail chains
   - Leverage memory bank for sharing information between chains
   - Implement chainbreakers for early termination when appropriate

## Examples

### Task Decomposition and Research

```python
# Initialize builder with head-tail technique
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction="""
    Task Decomposition Framework
    ===========================
    Your task: {instruction}
    Input: {input}
    
    Break this down into research questions.
    """,
    technique="head-tail"
)

# Create head chain
head_chain = builder.create_chain(
    "research_decomposer",
    "Analyze this topic and generate 3-5 specific research questions.",
    execution_mode="serial",
    group="research_project"  # Required group
)

# Execute head chain
questions = builder.execute_chain("research_decomposer", "Impact of artificial intelligence on healthcare")

# Parse and create tail chains
research_chains = []
for i, question in enumerate(questions.strip().split('\n')):
    if question.strip() and (question[0].isdigit() or question.startswith('-')):
        question_text = question.lstrip("0123456789-. ")
        chain_id = f"research_{i+1}"
        
        # Create research chain for this question
        builder.create_chain(
            chain_id,
            f"Research the following question thoroughly: {question_text}\n\n{{input}}",
            execution_mode="parallel",
            group="research_project",  # Same group as head
            dependencies=["research_decomposer"]
        )
        research_chains.append(chain_id)

# Create synthesis chain
builder.create_chain(
    "research_synthesis",
    "Synthesize all research findings into a comprehensive report.",
    execution_mode="serial",
    group="research_project",  # Same group
    dependencies=research_chains  # Depends on all research chains
)

# Execute chains
for chain_id in research_chains:
    builder.execute_chain(chain_id, "")

# Final synthesis
final_report = builder.execute_chain("research_synthesis", "")
print(f"Research Report:\n{final_report}")
```

### Multi-Modal Processing System

```python
# Initialize builder for multimedia processing
builder = DynamicChainBuilder(
    base_model="gemini-pro",
    base_instruction="""
    Multimedia Analysis Framework
    ===========================
    Task: {instruction}
    Content: {input}
    
    Identify the type of content and necessary processing steps.
    """,
    technique="head-tail"
)

# Create content analyzer head
head_chain = builder.create_chain(
    "content_analyzer",
    "Analyze this content and identify what type of media it contains and what processing is needed.",
    execution_mode="serial",
    group="multimedia_processing"  # Required group
)

# Execute head to analyze content
analysis = builder.execute_chain("content_analyzer", "https://example.com/mixed_media_page.html")

# Create specialized processors based on head output
if "image" in analysis.lower():
    builder.create_chain(
        "image_processor",
        "Extract and analyze all images from the content.",
        execution_mode="parallel",
        group="multimedia_processing",
        dependencies=["content_analyzer"]
    )

if "text" in analysis.lower():
    builder.create_chain(
        "text_processor",
        "Extract and analyze all textual content.",
        execution_mode="parallel",
        group="multimedia_processing",
        dependencies=["content_analyzer"]
    )

if "video" in analysis.lower():
    builder.create_chain(
        "video_processor",
        "Extract key frames and analyze video content.",
        execution_mode="parallel",
        group="multimedia_processing",
        dependencies=["content_analyzer"]
    )

# Create integrator chain
builder.create_chain(
    "content_integrator",
    "Integrate all analyzed components into a comprehensive summary.",
    execution_mode="serial",
    group="multimedia_processing",
    dependencies=["image_processor", "text_processor", "video_processor"]
)

# Execute the processing group
results = builder.execute_group("multimedia_processing", "https://example.com/mixed_media_page.html")
```

By following this head-tail pattern with proper structure and required parameters, you can create dynamic, adaptive chains that respond intelligently to content and context while maintaining proper execution flow and dependencies. 