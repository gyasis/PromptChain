# Chain Injection Examples

This guide demonstrates how to use dynamic chain injection with both regular PromptChains and DynamicChainBuilder. It shows how dynamic chains can be injected into regular chains and function-based steps.

## 1. Chain Building Techniques

The DynamicChainBuilder supports two techniques for chain creation. Both techniques use a base instruction template that gets copied and used for each step in the chain.

### Base Instruction Template
The base instruction is a template that defines the structure for each step. It gets copied and combined with step-specific instructions:

```python
BASE_INSTRUCTION = """
You are performing a specific step in an analysis chain.
Current step instruction: {instruction}
Input to analyze: {input}

Provide your analysis following these guidelines:
- Be specific and focused on the current step
- Maintain consistency with previous steps
- Format output clearly and concisely
"""
```

### Normal Technique (List-based)
- Requires a predefined list of instructions
- Used when you know all steps in advance
- Validates that first chain must provide instruction list
- Each step combines base instruction with step-specific instruction

### Head-Tail Technique
- Head: Generates task list dynamically from input
- Tail: Executes each generated task
- More flexible for dynamic workflows
- Tasks are generated based on input context
- Each generated task is combined with base instruction

Example of both techniques:

```python
from promptchain.utils.promptchaining import DynamicChainBuilder

# Define base instruction template
BASE_INSTRUCTION = """
Step Analysis Framework:
Current Task: {instruction}
Input Data: {input}

Provide analysis following these guidelines:
- Focus on the specific task
- Be precise and detailed
- Format output clearly
"""

# Normal Technique (List-based)
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction=BASE_INSTRUCTION,  # Will be used for each step
    technique="normal"
)

# Invalid examples:
try:
    # Invalid: Single string instead of list
    invalid_chain1 = builder.create_chain(
        "invalid1",
        "Extract key data points from the input",  # Error: must be a list
        execution_mode="serial"
    )
except ValueError as e:
    print("Error: First chain must provide instruction list")

try:
    # Invalid: Empty list
    invalid_chain2 = builder.create_chain(
        "invalid2",
        [],  # Error: list cannot be empty
        execution_mode="serial"
    )
except ValueError as e:
    print("Error: Instruction list cannot be empty")

# Valid chain - meets all requirements:
# 1. Provides a list of instructions (required for first chain)
# 2. Each instruction is a task description that can fill {instruction} in template
# 3. Instructions are properly formatted for combining with base template
valid_chain = builder.create_chain(
    "valid",
    [
        "Extract key data points from the input",  # Becomes: BASE_INSTRUCTION.format(instruction="Extract key data points...", input="{input}")
        "Analyze relationships between data points",
        "Generate summary of findings"
    ],
    execution_mode="serial"
)

# Example of how the first instruction gets combined with base template:
"""
Step Analysis Framework:
Current Task: Extract key data points from the input
Input Data: {input}

Provide analysis following these guidelines:
- Focus on the specific task
- Be precise and detailed
- Format output clearly
"""

# Head-Tail Technique
head_tail_builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction=BASE_INSTRUCTION,  # Used for both head and generated tasks
    technique="head-tail"
)

# Create head chain that generates task list
head_chain = head_tail_builder.create_chain(
    "task_generator",
    """Generate a list of analysis tasks for the input.
    Each task should be specific and actionable.
    Return tasks in a numbered list format.""",  # Combined with BASE_INSTRUCTION
    execution_mode="serial"
)

# Example of how tasks get combined with base instruction:
"""
Generated Task: "Analyze sentiment of text"
Becomes:
Step Analysis Framework:
Current Task: Analyze sentiment of text
Input Data: {actual_input}

Provide analysis following these guidelines:
- Focus on the specific task
- Be precise and detailed
- Format output clearly
"""
```

## 2. Regular PromptChain vs Dynamic Injection

### Regular PromptChain

First, let's see a regular PromptChain:

```python
from promptchain.utils.promptchaining import PromptChain

# Define a regular processing function
def analyze_sentiment(text: str) -> str:
    return f"Sentiment analysis of: {text}"

# Create a regular chain with mixed instructions and functions
regular_chain = PromptChain(
    models=["openai/gpt-4"] * 3,
    instructions=[
        "Extract main points from: {input}",
        analyze_sentiment,  # Function step
        "Summarize findings: {input}"
    ],
    store_steps=True
)

# Execute regular chain
result = regular_chain.process_prompt("Sample text to analyze")
```

### Dynamic Chain Injection

Now, let's see how we can inject a dynamic chain into this regular chain:

```python
from promptchain.utils.promptchaining import DynamicChainBuilder

# Create dynamic builder
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction="Dynamic analysis: {input}"
)

# Create a dynamic chain to inject
dynamic_chain = builder.create_chain(
    "detail_analysis",
    [
        "Analyze tone: {input}",
        "Check bias: {input}",
        "Evaluate complexity: {input}"
    ],
    execution_mode="serial"
)

# Create the regular chain that will receive the injection
main_chain = PromptChain(
    models=["openai/gpt-4"] * 4,
    instructions=[
        "Initial processing: {input}",
        "Extract key information: {input}",
        lambda x: "Function processing: " + x,  # Function step
        "Final summary: {input}"
    ],
    store_steps=True
)

# Register the regular chain with the builder
builder.chain_registry["main"] = {
    "chain": main_chain,
    "execution_mode": "serial",
    "group": "main",
    "dependencies": [],
    "status": "created"
}

# Inject the dynamic chain after the second step
builder.inject_chain(
    target_chain_id="main",
    source_chain_id="detail_analysis",
    position=2  # Insert after "Extract key information"
)

# Now the chain structure looks like:
# 1. Initial processing
# 2. Extract key information
# 3. Analyze tone (injected)
# 4. Check bias (injected)
# 5. Evaluate complexity (injected)
# 6. Function processing
# 7. Final summary
```

## 3. Real-World Example: Document Analysis Pipeline with Head-Tail

Here's a complete example showing how base instructions are used:

```python
# 1. Define base instruction template
DOCUMENT_ANALYSIS_TEMPLATE = """
Document Analysis Step
=====================
Current Operation: {instruction}
Document Content: {input}

Analysis Guidelines:
- Focus on the specified operation
- Consider document context
- Provide structured output
- Link findings to specific sections
"""

# 2. Create the dynamic builder
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction=DOCUMENT_ANALYSIS_TEMPLATE,
    technique="head-tail"
)

# 3. Create head chain for task generation
head_chain = builder.create_chain(
    "task_generator",
    """Analyze the document and generate required analysis tasks.
    Consider:
    - Document type and structure
    - Content complexity
    - Required analysis depth
    - Specific domain requirements
    
    Return a numbered list of tasks.""",  # Combined with DOCUMENT_ANALYSIS_TEMPLATE
    execution_mode="serial"
)

# 4. Process a document with dynamic task generation
async def process_document(text: str):
    # Generate tasks
    tasks = await builder.execute_chain("task_generator", text)
    
    # Create dynamic analysis chains based on generated tasks
    # Each task gets combined with DOCUMENT_ANALYSIS_TEMPLATE
    for i, task in enumerate(tasks):
        task_chain = builder.create_chain(
            f"task_{i}",
            [task],  # Task gets combined with base template
            execution_mode="parallel"
        )
        
        builder.inject_chain(
            target_chain_id="doc_processing",
            source_chain_id=f"task_{i}",
            position=1
        )
    
    return await builder.execute_chain("doc_processing", text)

# Example of how a task gets processed:
"""
Generated Task: "Extract technical requirements"

Gets transformed into:
Document Analysis Step
=====================
Current Operation: Extract technical requirements
Document Content: {actual_document_content}

Analysis Guidelines:
- Focus on the specified operation
- Consider document context
- Provide structured output
- Link findings to specific sections
"""
```

## 4. Best Practices for Base Instructions

1. **Template Design**:
   ```python
   # Create clear, structured base instruction templates
   BASE_TEMPLATE = """
   Analysis Step: {instruction}
   Input: {input}
   
   Requirements:
   1. Focus on the specific task
   2. Maintain consistent format
   3. Link to previous steps if needed
   
   Output Format:
   - Key findings: bullet points
   - Details: paragraph form
   - References: numbered list
   """
   ```

2. **Context Management**:
   ```python
   # Include relevant context in base template
   CONTEXT_TEMPLATE = """
   Step Context
   ===========
   Chain ID: {chain_id}
   Step Number: {step_number}
   Current Task: {instruction}
   Input Data: {input}
   
   Previous Context: {context}
   
   Task Instructions:
   {specific_instructions}
   """
   ```

3. **Specialized Templates**:
   ```python
   # Different templates for different analysis types
   TEMPLATES = {
       "technical": """
           Technical Analysis Step
           =====================
           Operation: {instruction}
           Input: {input}
           
           Technical Guidelines:
           - Use precise terminology
           - Include measurements/metrics
           - Reference standards
       """,
       "creative": """
           Creative Analysis Step
           ====================
           Creative Task: {instruction}
           Content: {input}
           
           Creative Guidelines:
           - Consider style and tone
           - Evaluate emotional impact
           - Assess originality
       """
   }
   
   # Use appropriate template based on content
   def create_specialized_chain(content_type: str, instructions: list):
       return DynamicChainBuilder(
           base_model="openai/gpt-4",
           base_instruction=TEMPLATES[content_type],
           technique="normal"
       )
   ```

4. **Template Validation**:
   ```python
   def validate_template(template: str):
       """Ensure template has required placeholders"""
       required = ["{instruction}", "{input}"]
       for req in required:
           if req not in template:
               raise ValueError(f"Template missing required placeholder: {req}")
               
   # Use in builder
   class DynamicChainBuilder:
       def __init__(self, base_model, base_instruction, technique):
           validate_template(base_instruction)
           self.base_instruction = base_instruction
   ```

## 5. Common Patterns

1. **Head-Tail Pattern**:
   ```python
   def create_head_tail_chain(content: str):
       # Create head chain for task generation
       head = builder.create_chain(
           "head",
           """Analyze input and generate task list.
           Tasks should be specific and actionable.
           Return numbered list.""",
           execution_mode="serial"
       )
       
       # Generate tasks
       tasks = builder.execute_chain("head", content)
       
       # Create and inject task chains
       for i, task in enumerate(tasks):
           task_chain = builder.create_chain(
               f"task_{i}",
               [f"Execute: {task}"],
               execution_mode="parallel"
           )
           builder.inject_chain("main", f"task_{i}", i + 1)
   ```

2. **Dynamic Task Generation**:
   ```python
   def generate_analysis_tasks(text: str):
       # Head chain analyzes content and suggests tasks
       analysis_plan = builder.create_chain(
           "analysis_planner",
           """Analyze the content and determine required tasks.
           Consider:
           - Content type
           - Complexity
           - Required analysis depth
           Return task list.""",
           execution_mode="serial"
       )
       
       # Generate tasks
       tasks = builder.execute_chain("analysis_planner", text)
       
       # Create task-specific chains
       for task in tasks:
           chain_id = f"task_{hash(task)}"
           builder.create_chain(chain_id, [task])
           builder.inject_chain("main", chain_id, "end")
   ```

3. **Conditional Head-Tail**:
   ```python
   def smart_task_generation(content: str):
       # Head determines content type and generates appropriate tasks
       content_analyzer = builder.create_chain(
           "content_analyzer",
           """Analyze content type and generate appropriate tasks.
           If technical: focus on accuracy and specifications
           If creative: focus on style and engagement
           If data-heavy: focus on statistical analysis
           Return task list.""",
           execution_mode="serial"
       )
       
       # Generate and execute task chains based on content type
       tasks = builder.execute_chain("content_analyzer", content)
       for task in tasks:
           task_chain = builder.create_chain(
               f"task_{hash(task)}",
               [task],
               execution_mode="parallel"
           )
   ``` 
noteId: "93052ed0087d11f09aa1cf0704e2b01f"
tags: []

---

 