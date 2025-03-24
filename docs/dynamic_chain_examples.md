---
noteId: "6e527530087c11f09aa1cf0704e2b01f"
tags: []

---

# Dynamic Chain Execution Examples

This guide demonstrates various ways to use PromptChain's dynamic chain execution features. The examples cover different execution modes, group management, and advanced use cases.

## Basic Setup

First, create a DynamicChainBuilder with base configuration:

```python
from promptchain.utils.promptchaining import DynamicChainBuilder

builder = DynamicChainBuilder(
    base_model={
        "name": "openai/gpt-4",
        "params": {"temperature": 0.7}
    },
    base_instruction="Base analysis: {input}"
)
```

## 1. Serial Execution (Dependencies)

Serial execution ensures chains run in a specific order:

```python
# Create a chain for initial analysis
builder.create_chain(
    "extract",
    ["Extract key information: {input}"],
    execution_mode="serial",
    group="document_analysis"
)

# Create a dependent chain that runs after extract
builder.create_chain(
    "summarize",
    ["Create a summary based on: {input}"],
    execution_mode="serial",
    dependencies=["extract"],
    group="document_analysis"
)

# Create final chain dependent on summary
builder.create_chain(
    "recommendations",
    ["Generate recommendations based on: {input}"],
    execution_mode="serial",
    dependencies=["summarize"],
    group="document_analysis"
)

# Execute the entire group
results = builder.execute_group("document_analysis", "Long document text here...")
```

## 2. Parallel Execution

Run multiple chains simultaneously for better performance:

```python
# Create parallel analysis chains
builder.create_chain(
    "sentiment",
    ["Analyze sentiment in: {input}"],
    execution_mode="parallel",
    group="content_analysis"
)

builder.create_chain(
    "keywords",
    ["Extract keywords from: {input}"],
    execution_mode="parallel",
    group="content_analysis"
)

builder.create_chain(
    "entities",
    ["Identify entities in: {input}"],
    execution_mode="parallel",
    group="content_analysis"
)

# Define parallel executor (using asyncio)
async def parallel_executor(chain_inputs):
    import asyncio
    
    async def process_chain(chain_id, input_data):
        # Simulate async processing
        return await asyncio.to_thread(
            builder.execute_chain, chain_id, input_data
        )
    
    tasks = [
        process_chain(chain_id, input_data)
        for chain_id, input_data in chain_inputs
    ]
    
    results = await asyncio.gather(*tasks)
    return dict(zip([ci[0] for ci in chain_inputs], results))

# Execute parallel chains
results = builder.execute_group(
    "content_analysis",
    "Text to analyze",
    parallel_executor=parallel_executor
)
```

## 3. Independent Execution

Chains that can run at any time without dependencies:

```python
# Create independent validation chains
builder.create_chain(
    "spell_check",
    ["Check spelling in: {input}"],
    execution_mode="independent",
    group="validation"
)

builder.create_chain(
    "grammar_check",
    ["Check grammar in: {input}"],
    execution_mode="independent",
    group="validation"
)

builder.create_chain(
    "style_check",
    ["Check writing style in: {input}"],
    execution_mode="independent",
    group="validation"
)

# These can be executed in any order
results = builder.execute_group("validation", "Text to validate")
```

## 4. Mixed Execution Modes

Combine different execution modes in a single workflow:

```python
# Independent preprocessing
builder.create_chain(
    "cleanup",
    ["Clean and normalize: {input}"],
    execution_mode="independent",
    group="analysis"
)

# Parallel analysis after cleanup
builder.create_chain(
    "topic_analysis",
    ["Identify topics in: {input}"],
    execution_mode="parallel",
    group="analysis"
)

builder.create_chain(
    "tone_analysis",
    ["Analyze tone in: {input}"],
    execution_mode="parallel",
    group="analysis"
)

# Serial final processing
builder.create_chain(
    "final_summary",
    ["Synthesize findings: {input}"],
    execution_mode="serial",
    dependencies=["cleanup"],
    group="analysis"
)

# Execute mixed mode group
results = builder.execute_group("analysis", "Input text")
```

## 5. Dynamic Chain Insertion

Add new steps to existing chains:

```python
# Create initial chain
builder.create_chain(
    "process",
    ["Initial processing: {input}"],
    execution_mode="serial",
    group="dynamic"
)

# Later, insert new steps
builder.insert_chain(
    "process",
    ["Additional analysis: {input}"],
    position=1  # Insert after first step
)

# Execute updated chain
results = builder.execute_chain("process", "Input data")
```

## 6. Chain Merging

Combine multiple chains into a new one:

```python
# Create source chains
builder.create_chain(
    "analysis_1",
    ["First analysis: {input}"],
    execution_mode="independent",
    group="source"
)

builder.create_chain(
    "analysis_2",
    ["Second analysis: {input}"],
    execution_mode="independent",
    group="source"
)

# Merge into new chain
merged = builder.merge_chains(
    ["analysis_1", "analysis_2"],
    "combined_analysis",
    execution_mode="serial",
    group="merged"
)
```

## 7. Real-World Use Case: Document Processing Pipeline

Complex example combining multiple features:

```python
# 1. Independent preprocessing chains
builder.create_chain(
    "format_check",
    ["Validate document format: {input}"],
    execution_mode="independent",
    group="doc_processing"
)

builder.create_chain(
    "sanitize",
    ["Remove sensitive information: {input}"],
    execution_mode="independent",
    group="doc_processing"
)

# 2. Parallel analysis chains
builder.create_chain(
    "content_analysis",
    ["Analyze main content: {input}"],
    execution_mode="parallel",
    group="doc_processing"
)

builder.create_chain(
    "metadata_extraction",
    ["Extract document metadata: {input}"],
    execution_mode="parallel",
    group="doc_processing"
)

# 3. Serial processing chains
builder.create_chain(
    "summarize",
    ["Create document summary: {input}"],
    execution_mode="serial",
    dependencies=["content_analysis"],
    group="doc_processing"
)

builder.create_chain(
    "categorize",
    ["Categorize document: {input}"],
    execution_mode="serial",
    dependencies=["summarize"],
    group="doc_processing"
)

# 4. Execute the entire pipeline
async def process_document(doc_text):
    # Custom parallel executor
    async def parallel_executor(chain_inputs):
        import asyncio
        tasks = {
            chain_id: asyncio.create_task(
                asyncio.to_thread(builder.execute_chain, chain_id, input_data)
            )
            for chain_id, input_data in chain_inputs
        }
        return {
            chain_id: await task
            for chain_id, task in tasks.items()
        }
    
    # Process document
    results = await builder.execute_group(
        "doc_processing",
        doc_text,
        parallel_executor=parallel_executor
    )
    
    return results

# 5. Monitor processing status
def monitor_processing(group="doc_processing"):
    status = builder.get_group_status(group)
    print(f"Processing status: {status}")
    
    # Get specific outputs
    summary = builder.get_chain_output("summarize")
    category = builder.get_chain_output("categorize")
    
    return {
        "status": status,
        "summary": summary,
        "category": category
    }
```

## Best Practices

1. **Group Organization**:
   - Group related chains together
   - Use meaningful group names
   - Keep groups focused on specific tasks

2. **Execution Mode Selection**:
   - Use `serial` when order matters
   - Use `parallel` for independent, time-consuming operations
   - Use `independent` for validation and preprocessing

3. **Dependency Management**:
   - Keep dependency chains clear and minimal
   - Avoid circular dependencies
   - Document dependencies in comments

4. **Error Handling**:
   - Monitor chain status
   - Implement proper error handling in parallel executors
   - Save intermediate outputs for debugging

5. **Performance Optimization**:
   - Use parallel execution for CPU/IO-bound tasks
   - Group similar operations together
   - Monitor resource usage

## Common Patterns

1. **Validation Pipeline**:
   ```python
   # Independent validation chains
   for check in ["format", "schema", "content"]:
       builder.create_chain(
           f"{check}_check",
           [f"Validate {check}: {{input}}"],
           execution_mode="independent",
           group="validation"
       )
   ```

2. **Analysis Pipeline**:
   ```python
   # Serial analysis with parallel components
   builder.create_chain("preprocess", ["Preprocess: {input}"],
                       execution_mode="serial", group="analysis")
   
   for analysis in ["sentiment", "topics", "entities"]:
       builder.create_chain(
           analysis,
           [f"Analyze {analysis}: {{input}}"],
           execution_mode="parallel",
           group="analysis"
       )
   
   builder.create_chain("synthesize", ["Synthesize results: {input}"],
                       execution_mode="serial",
                       dependencies=["preprocess"],
                       group="analysis")
   ```

3. **Document Processing**:
   ```python
   # Mixed mode document processing
   stages = [
       ("validate", "independent", []),
       ("extract", "parallel", []),
       ("analyze", "parallel", []),
       ("summarize", "serial", ["analyze"]),
       ("format", "serial", ["summarize"])
   ]
   
   for name, mode, deps in stages:
       builder.create_chain(
           name,
           [f"{name.capitalize()}: {{input}}"],
           execution_mode=mode,
           dependencies=deps,
           group="document"
       )
   ``` 