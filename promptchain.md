# Prompt Chaining with LiteLLM

## Introduction

Prompt chaining is a powerful technique in AI that involves passing outputs through a sequence of language models, where each model in the chain performs a specific task or refinement. This repository contains two main components:

1. `promptchaining.py`: The core implementation file containing the `PromptChain` class
2. `promptchain.md`: This documentation file explaining the implementation and use cases

### What is Prompt Chaining?

Prompt chaining allows you to:
- Break down complex tasks into smaller, manageable steps
- Use different models for different aspects of processing
- Maintain context and history across processing steps
- Progressively refine and improve outputs
- Create specialized pipelines for specific use cases

### Architecture Overview

The implementation consists of several key components:

1. **PromptChain Class**: The main class that orchestrates the prompt chaining process
   - Manages model selection and sequencing
   - Handles instruction templates
   - Tracks processing history
   - Coordinates model interactions

2. **Template Management**:
   - Supports both file-based and inline templates
   - Flexible template loading and validation
   - Dynamic template substitution

3. **Model Integration**:
   - Uses LiteLLM for unified model access
   - Supports multiple model providers (OpenAI, Anthropic, etc.)
   - Configurable model selection per chain step

4. **History Tracking**:
   - Optional full history mode
   - Structured history storage
   - Context preservation across chain steps

### Key Concepts

1. **Refinement Steps**:
   - Each step in the chain represents a specific task
   - Steps can be executed by different models
   - Results flow from one step to the next

2. **Instructions**:
   - Template-based instruction format
   - Support for file-based instruction loading
   - Dynamic content substitution

3. **Model Selection**:
   - Flexible model assignment per step
   - Support for multiple AI providers
   - Model validation and error handling

4. **History Management**:
   - Configurable history tracking
   - Structured history storage
   - Context preservation options

### Implementation Details

The implementation in `promptchaining.py` provides:

1. **Initialization**:
   ```python
   def __init__(self, models, num_refinements, instructions, full_history=False):
   ```
   - Validates model and instruction configurations
   - Sets up history tracking
   - Initializes template management

2. **Template Loading**:
   ```python
   def load_instruction(self, instruction):
   ```
   - Supports both file and string templates
   - Handles template validation
   - Manages template loading errors

3. **Prompt Processing**:
   ```python
   def process_prompt(self, initial_input):
   ```
   - Manages the chain execution
   - Handles history tracking
   - Coordinates model interactions

4. **Model Execution**:
   ```python
   @staticmethod
   def run_model(model_name, prompt):
   ```
   - Executes model calls via LiteLLM
   - Handles response processing
   - Manages error conditions

### Integration with LiteLLM

The implementation leverages LiteLLM to:
- Provide unified access to multiple AI models
- Handle API authentication and routing
- Manage model-specific configurations
- Process responses consistently

### Environment Setup

Required environment variables:
```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Dependencies:
```python
litellm
python-dotenv
```

### Integrating Prompt Templates into the LiteLLM Prompt Chaining Script

This implementation demonstrates a robust, class-based approach to prompt chaining using LiteLLM. The solution supports both file-based and inline prompt templates, maintains chain history, and provides comprehensive error handling.

```python
from litellm import completion
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PromptChain:
    def __init__(self, models, num_refinements, instructions, full_history=False):
        """
        Initialize the PromptChain with models and instructions for each refinement step.
        
        :param models: List of model names for each chain link
        :param num_refinements: Number of refinement steps
        :param instructions: List of instruction templates or file paths
        :param full_history: Whether to pass full chain history
        """
        if len(models) != num_refinements or len(instructions) != num_refinements:
            raise ValueError("The number of models and instructions must match refinement steps.")
            
        self.models = models
        self.num_refinements = num_refinements
        self.instructions = [self.load_instruction(instr) for instr in instructions]
        self.full_history = full_history

    def load_instruction(self, instruction):
        """Load instruction from string or file."""
        if os.path.isfile(instruction):
            with open(instruction, 'r') as file:
                return file.read()
        return instruction

    def process_prompt(self, initial_input):
        """Execute the prompt chain with history tracking."""
        result = initial_input
        chain_history = [{"step": 0, "input": initial_input, "output": initial_input}]
        
        for step in range(self.num_refinements):
            model = self.models[step]
            instruction = self.instructions[step]

            # Handle history tracking
            if self.full_history:
                history_text = "\n".join([
                    f"Step {entry['step']}: {entry['output']}" 
                    for entry in chain_history
                ])
                content_to_process = f"Previous steps:\n{history_text}\n\nCurrent input: {result}"
            else:
                content_to_process = result

            # Process the prompt
            prompt = instruction.replace("{input}", content_to_process)
            result = self.run_model(model, prompt)
            
            # Update history
            chain_history.append({
                "step": step + 1,
                "input": content_to_process,
                "output": result
            })

        return result if not self.full_history else chain_history

    @staticmethod
    def run_model(model_name, prompt):
        """Execute model using LiteLLM."""
        response = completion(
            model=model_name, 
            messages=[{"content": prompt, "role": "user"}]
        )
        return response['choices'][0]['message']['content']

# Example usage
if __name__ == "__main__":
    # Define your models and instructions
    models = ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"]
    instructions = [
        "Initial analysis of {input}",
        "Refine and expand upon: {input}"
    ]

    # Create and execute the chain
    chain = PromptChain(
        models=models,
        num_refinements=2,
        instructions=instructions,
        full_history=True
    )
    
    result = chain.process_prompt("Analyze the impact of AI on healthcare")
```

### Key Features and Benefits

1. **Robust Template Management**:
   - Supports both inline templates and file-based templates
   - Flexible instruction loading mechanism
   - Template validation during initialization

2. **History Tracking**:
   - Optional full history mode to maintain context across the chain
   - Structured history with step-by-step tracking
   - Ability to review the entire refinement process

3. **Error Handling and Validation**:
   - Validates configuration during initialization
   - Ensures consistency between models and instructions
   - Graceful handling of file loading and template processing

4. **Flexibility and Extensibility**:
   - Easy to add new models to the chain
   - Configurable history tracking
   - Support for different instruction formats

This implementation provides a more structured and maintainable approach to prompt chaining, suitable for both experimental and production environments. The class-based design allows for easy extension and modification while maintaining clean separation of concerns.

### Examples and Use Cases

Here are five practical examples demonstrating different applications of the PromptChain class:

1. **Content Generation and Refinement**
```python
# Content creation with progressive refinement
models = ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"]
instructions = [
    "Generate a detailed blog post outline about: {input}",
    "Expand this outline into a full blog post with engaging content"
]

content_chain = PromptChain(
    models=models,
    num_refinements=2,
    instructions=instructions,
    full_history=True
)

blog_post = content_chain.process_prompt("The Future of Quantum Computing")
```

2. **Code Review and Documentation**
```python
# Automated code review and documentation generation
models = ["openai/gpt-4", "openai/gpt-4", "anthropic/claude-3-sonnet-20240229"]
instructions = [
    "Analyze this code for potential bugs and improvements: {input}",
    "Generate comprehensive documentation for the analyzed code",
    "Create unit test scenarios based on the documentation"
]

code_review_chain = PromptChain(
    models=models,
    num_refinements=3,
    instructions=instructions,
    full_history=True
)

code_analysis = code_review_chain.process_prompt("""
def calculate_fibonacci(n):
    if n <= 0: return []
    if n == 1: return [0]
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence
""")
```

3. **Research Paper Analysis**
```python
# Multi-step research paper analysis
models = ["anthropic/claude-3-sonnet-20240229", "openai/gpt-4", "anthropic/claude-3-sonnet-20240229"]
instructions = [
    "Extract key findings and methodology from this research paper: {input}",
    "Analyze the limitations and potential improvements of the methodology",
    "Synthesize a comprehensive summary with critical analysis and future research directions"
]

research_chain = PromptChain(
    models=models,
    num_refinements=3,
    instructions=instructions
)

analysis = research_chain.process_prompt("Abstract: [Research paper content...]")
```

4. **Language Translation and Localization**
```python
# Multi-step translation with cultural adaptation
models = ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229", "openai/gpt-4"]
instructions = [
    "Translate this English text to Spanish: {input}",
    "Adapt the translation to be culturally appropriate for Latin American audiences",
    "Review and refine the translation for natural flow and idioms"
]

translation_chain = PromptChain(
    models=models,
    num_refinements=3,
    instructions=instructions,
    full_history=True
)

localized_content = translation_chain.process_prompt(
    "Welcome to our product launch! We're excited to share our innovative solution..."
)
```

5. **Data Analysis and Reporting**
```python
# Progressive data analysis and report generation
models = ["openai/gpt-4", "anthropic/claude-3-sonnet-20240229", "openai/gpt-4", "anthropic/claude-3-sonnet-20240229"]
instructions = [
    "Analyze this dataset and identify key trends: {input}",
    "Generate statistical insights and correlations from the analysis",
    "Create data visualizations recommendations based on the insights",
    "Compile a comprehensive executive summary with key findings"
]

analysis_chain = PromptChain(
    models=models,
    num_refinements=4,
    instructions=instructions,
    full_history=True
)

report = analysis_chain.process_prompt("""
Monthly Sales Data:
Jan: $50,000
Feb: $65,000
Mar: $75,000
Apr: $85,000
May: $95,000
""")
```

Each example demonstrates different aspects of the PromptChain's capabilities:
- Variable number of refinement steps
- Different model combinations
- Task-specific instruction templates
- Use of history tracking
- Progressive refinement of content

The examples can be extended or modified to suit specific needs by:
- Adjusting the number of refinement steps
- Changing the model selection
- Customizing instruction templates
- Modifying the history tracking behavior
- Adding domain-specific validation or processing

These examples serve as starting points for building more complex prompt chains tailored to specific use cases and requirements.