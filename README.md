# PromptChain 

A flexible prompt engineering library designed to create, refine, and specialize prompts for LLM applications and agent frameworks. While it can be used as a simple sequential chain processor, its primary strength lies in dynamic prompt engineering and optimization.

## Primary Use Case: Prompt Engineering for Agent Frameworks

PromptChain excels at creating specialized prompts that can be injected into agent frameworks. Here's a typical workflow:

```python
from autogen import AssistantAgent, UserProxyAgent
from langchain.agents import initialize_agent, Tool
from crewai import Agent, Task

# 1. Create a specialized prompt using PromptChain
prompt_engineer = PromptChain(
    models=[
        "anthropic/claude-3-sonnet-20240229",  # Task analysis
        "openai/gpt-4",                        # Prompt optimization
        analyze_prompt_effectiveness,           # Custom testing
    ],
    instructions=[
        """Analyze this agent task and create a specialized prompt:
        - Identify key capabilities needed
        - Define constraints and guidelines
        - Specify expected output format
        
        Task to analyze: {input}""",
        
        """Optimize the prompt for the agent framework:
        - Add relevant context
        - Include example interactions
        - Define error handling
        
        Initial prompt: {input}""",
        
        analyze_prompt_effectiveness  # Custom validation
    ],
    full_history=True
)

# 2. Generate specialized prompt for different agent frameworks
task = "Research and summarize recent AI papers with critical analysis"
specialized_prompt = prompt_engineer.process_prompt(task)

# 3. Use with AutoGen
autogen_agent = AssistantAgent(
    name="researcher",
    system_message=specialized_prompt['output'],
    llm_config={"config_list": [...]}
)

# 4. Use with LangChain
langchain_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=[...],
    llm=ChatOpenAI(),
    agent_kwargs={
        "system_message": specialized_prompt['output']
    }
)

# 5. Use with CrewAI
crewai_agent = Agent(
    role="Researcher",
    goal="Conduct thorough research",
    backstory=specialized_prompt['output'],
    tools=[...]
)

# 6. Dynamic prompt adaptation
def adapt_prompt_for_context(context: str) -> str:
    adaptation_chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            f"""Adapt this specialized prompt for the current context:
            Original Prompt: {specialized_prompt['output']}
            Current Context: {context}
            
            Maintain the core functionality while optimizing for:
            - Relevant context integration
            - Specific task requirements
            - Performance constraints
            """
        ]
    )
    return adaptation_chain.process_prompt(context)

# Use the adapted prompt
current_context = "Focus on quantum computing papers"
adapted_prompt = adapt_prompt_for_context(current_context)
updated_agent = AssistantAgent(
    name="researcher",
    system_message=adapted_prompt,
    llm_config={"config_list": [...]}
)
```

This approach enables:
- Creation of highly specialized agent prompts
- Dynamic prompt adaptation based on context
- Consistent prompt quality across different frameworks
- Reusable prompt templates for similar tasks
- Systematic prompt testing and validation

## Additional Use Cases

PromptChain can be used in two main ways:

### 1. Sequential Processing Pipeline
Chain multiple LLMs and processing steps together to create sophisticated workflows:
```python
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Extract key information from: {input}",
        "Create a detailed analysis based on: {input}"
    ]
)
result = chain.process_prompt("Raw data here...")
```

### 2. Specialized Prompt Engineering
Build and refine prompts through multiple iterations to create the perfect instruction:
```python
refinement_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Create initial prompt for task: {input}",
        analyze_prompt_quality,  # Custom function
        "Improve prompt based on analysis: {input}"
    ],
    full_history=True
)
final_prompt = refinement_chain.process_prompt("Generate marketing copy")
```

## Current Features

- **Flexible Chain Construction**: Create chains of prompts using different models
- **Function Injection**: Insert custom processing functions between LLM calls
- **Multiple Model Support**: Use different models (GPT-4, Claude, etc.) in the same chain
- **History Tracking**: Optional tracking of full chain history
- **Simple API**: Easy-to-use interface for creating and running prompt chains

## Installation

### From PyPI (Coming Soon)
```bash
pip install promptchain
```

### From GitHub
```bash
pip install git+https://github.com/yourusername/promptchain.git
```

### Development Installation
```bash
git clone https://github.com/yourusername/promptchain.git
cd promptchain
pip install -e .
```

## Quick Start

```python
# Import the main class
from promptchain import PromptChain

# Create a simple chain
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Write initial content about: {input}",
        "Improve and expand the content: {input}"
    ],
    full_history=False
)

# Run the chain
result = chain.process_prompt("AI in healthcare")
```

For more advanced usage, you can also import the prompt loading utilities:
```python
from promptchain import load_prompts, get_prompt_by_name

# Load prompts from files
prompts = load_prompts()
analysis_prompt = get_prompt_by_name("ANALYSIS_INITIAL_ANALYSIS")
```

## Advanced Usage

### Function Injection
You can inject custom processing functions into the chain:

```python
def analyze_sentiment(text: str) -> str:
    # Custom sentiment analysis
    return f"Sentiment score: {score}"

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Write about: {input}",
        analyze_sentiment,
        "Summarize the analysis: {input}"
    ],
    full_history=True
)
```

### Chain History
Track the full history of processing:

```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze: {input}"],
    full_history=True
)

history = chain.process_prompt("Topic")
for step in history:
    print(f"Step {step['step']}: {step['output']}")
```

### Using Structured Prompts
For more complex instructions, you can define prompts as variables:

```python
ANALYSIS_PROMPT = """
Analyze the following topic in detail:
- Key concepts and principles
- Historical context
- Current applications
- Future implications

Topic: {input}
"""

REFINEMENT_PROMPT = """
Based on the previous analysis, enhance and structure the content:
1. Add specific examples
2. Include relevant statistics
3. Address potential counterarguments
4. Provide actionable insights

Previous analysis: {input}
"""

SUMMARY_PROMPT = """
Create a comprehensive yet concise summary:
- Main findings
- Critical insights
- Key recommendations
- Next steps

Full analysis: {input}
"""

# Create chain with structured prompts
analysis_chain = PromptChain(
    models=[
        "openai/gpt-4",
        "anthropic/claude-3-sonnet-20240229",
        "openai/gpt-4"
    ],
    instructions=[
        ANALYSIS_PROMPT,
        REFINEMENT_PROMPT,
        SUMMARY_PROMPT
    ],
    full_history=True
)

# Run the chain
result = analysis_chain.process_prompt("Impact of quantum computing on cryptography")
```

This approach helps:
- Keep prompts organized and reusable
- Make instructions more detailed and structured
- Maintain consistent formatting across chain steps
- Improve readability of chain definitions

### Loading Prompts from Files
You can organize complex prompts in markdown files:

```plaintext
project_root/
├── src/
│   ├── prompts/
│   │   ├── analysis/
│   │   │   ├── initial_analysis.md
│   │   │   ├── refinement.md
│   │   │   └── summary.md
│   │   ├── story/
│   │   │   ├── outline.md
│   │   │   └── development.md
│   │   └── translation/
│   │       ├── translate.md
│   │       └── localize.md
```

Example prompt files:

```markdown:src/prompts/analysis/initial_analysis.md
# Initial Analysis Template

Perform a detailed analysis of the following topic:

## Required Aspects
- Key concepts and principles
- Historical context
- Current applications
- Future implications

## Analysis Guidelines
- Use concrete examples
- Cite relevant research
- Consider multiple perspectives

Topic: {input}
```

```markdown:src/prompts/analysis/refinement.md
# Refinement Guidelines

Enhance the previous analysis with:

1. Supporting Evidence
   - Statistical data
   - Case studies
   - Expert opinions

2. Critical Evaluation
   - Strengths
   - Limitations
   - Counterarguments

Previous analysis: {input}
```

Loading and using the prompts:

```python
import os

def load_prompt(prompt_path: str) -> str:
    """Load prompt from markdown file"""
    with open(prompt_path, 'r') as file:
        return file.read()

# Load prompts from files
ANALYSIS_PROMPT = load_prompt('src/prompts/analysis/initial_analysis.md')
REFINEMENT_PROMPT = load_prompt('src/prompts/analysis/refinement.md')
SUMMARY_PROMPT = load_prompt('src/prompts/analysis/summary.md')

# Create chain with loaded prompts
analysis_chain = PromptChain(
    models=[
        "openai/gpt-4",
        "anthropic/claude-3-sonnet-20240229",
        "openai/gpt-4"
    ],
    instructions=[
        ANALYSIS_PROMPT,
        REFINEMENT_PROMPT,
        SUMMARY_PROMPT
    ],
    full_history=True
)
```

Benefits of this approach:
- Organize prompts in a clear directory structure
- Use markdown formatting for better readability
- Version control your prompts
- Share and reuse prompts across projects
- Easier collaboration on prompt development
- Support for prompt documentation and metadata

### Automatic Prompt Loading
The system can automatically load prompts from the prompts directory structure:

```plaintext
project_root/
├── src/
│   ├── prompts/
│   │   ├── analysis/
│   │   │   ├── initial_analysis.md  -> ANALYSIS_INITIAL_ANALYSIS
│   │   │   ├── refinement.md        -> ANALYSIS_REFINEMENT
│   │   │   └── summary.md           -> ANALYSIS_SUMMARY
│   │   ├── story/
│   │   │   ├── outline.md           -> STORY_OUTLINE
│   │   │   └── development.md       -> STORY_DEVELOPMENT
│   │   └── translation/
│   │       ├── translate.md         -> TRANSLATION_TRANSLATE
│   │       └── localize.md          -> TRANSLATION_LOCALIZE
```

Using the automatic prompt loader:

```python
from src.utils.prompt_loader import load_prompts, get_prompt_by_name

# Load all prompts
prompts = load_prompts()

# Create chain using prompt names
analysis_chain = PromptChain(
    models=[
        "openai/gpt-4",
        "anthropic/claude-3-sonnet-20240229",
        "openai/gpt-4"
    ],
    instructions=[
        get_prompt_by_name("ANALYSIS_INITIAL_ANALYSIS"),
        get_prompt_by_name("ANALYSIS_REFINEMENT"),
        get_prompt_by_name("ANALYSIS_SUMMARY")
    ],
    full_history=True
)

# Or access prompts directly from the loaded dictionary
for name, (category, content) in prompts.items():
    print(f"Prompt: {name}")
    print(f"Category: {category}")
    print(f"Content length: {len(content)}")
```

This approach provides:
- Automatic variable naming based on directory structure
- Category organization through subdirectories
- Easy access to all available prompts
- Consistent naming conventions
- Runtime prompt loading and validation

## Examples

Check the `src/examples` directory for various use cases:
- Basic chain examples (`chain_examples.py`)
- Function injection examples (`function_chain_example.py`)
- Advanced processing examples (`advanced_chain_example.py`)

These examples serve as starting points for building more complex prompt chains tailored to specific use cases and requirements.

### Prompt Engineering Techniques

PromptChain provides a simple yet powerful way to enhance your prompts using various techniques. Each technique can be added using a simple string format: `technique:parameter` (when parameters are needed).

#### Technique Categories

1. **Techniques Requiring Parameters**
```python
chain.add_techniques([
    "role_playing:scientist",                # Adopts a specific professional role
    "style_mimicking:Richard Feynman",      # Mimics a specific writing style
    "persona_emulation:Warren Buffett",     # Emulates a specific expert
    "forbidden_words:maybe,probably,perhaps" # Specifies words to avoid
])
```

2. **Techniques with Optional Parameters**
```python
chain.add_techniques([
    "few_shot:3",                    # Number of examples to include
    "reverse_prompting:5",           # Number of questions to generate
    "context_expansion:historical",   # Type of context to consider
    "comparative_answering:3",        # Number of aspects to compare
    "tree_of_thought:4"              # Number of solution paths
])
```

3. **Techniques Without Parameters**
```python
chain.add_techniques([
    "step_by_step",           # Step-by-step reasoning
    "chain_of_thought",       # Multiple-step reasoning
    "iterative_refinement",   # Progressive improvement
    "contrarian_perspective", # Challenge common beliefs
    "react"                  # Reason-Act-Observe process
])
```

#### Validation Rules

The library enforces strict validation for technique usage:

1. **Required Parameters**
```python
# ❌ This will raise an error
chain.add_techniques(["role_playing"])  
# Error: Technique 'role_playing' requires a profession/role parameter.

# ✅ This is correct
chain.add_techniques(["role_playing:quantum physicist"])
```

2. **Optional Parameters**
```python
# ✅ Both are valid
chain.add_techniques(["few_shot"])         # Uses default
chain.add_techniques(["few_shot:3"])       # Uses specified number
```

3. **No Parameters**
```python
# ⚠️ This will show a warning
chain.add_techniques(["step_by_step:detailed"])  
# Warning: Technique 'step_by_step' doesn't use parameters

# ✅ This is correct
chain.add_techniques(["step_by_step"])
```

#### Examples

1. **Expert Analysis Chain**
```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze this topic: {input}"]
)

chain.add_techniques([
    "role_playing:data scientist",
    "step_by_step",
    "tree_of_thought:3"
])

result = chain.process_prompt("Analyze customer churn patterns")
```

2. **Creative Writing Chain**
```python
chain = PromptChain(
    models=["anthropic/claude-3-sonnet-20240229"],
    instructions=["Write a story about: {input}"]
)

chain.add_techniques([
    "style_mimicking:Ernest Hemingway",
    "forbidden_words:very,really,quite",
    "context_expansion:emotional"
])

story = chain.process_prompt("A fisherman's last voyage")
```

3. **Problem-Solving Chain**
```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Analyze the problem: {input}",
        "Generate solutions: {input}"
    ]
)

chain.add_techniques([
    "persona_emulation:Systems Engineer",
    "react",
    "comparative_answering:5",
    "tree_of_thought:3"
])

solution = chain.process_prompt("Optimize server response times")
```

4. **Educational Content Chain**
```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Explain this concept: {input}"]
)

chain.add_techniques([
    "role_playing:professor",
    "few_shot:3",
    "step_by_step",
    "context_expansion:historical"
])

explanation = chain.process_prompt("Quantum entanglement")
```

#### Best Practices

1. **Combine Techniques Thoughtfully**
   - Mix different types of techniques for better results
   - Consider how techniques complement each other
   - Don't overload with too many techniques

2. **Parameter Usage**
   - Use specific, relevant parameters
   - Keep parameters concise and clear
   - Test different parameter values

3. **Validation Awareness**
   - Always provide parameters for required techniques
   - Use optional parameters when more specificity is needed
   - Remove parameters from techniques that don't use them

## Future Roadmap

### Planned Multimodal Support
- **Image Input/Output**: Process and generate images within chains
- **Audio Processing**: Handle audio input and output
- **Video Integration**: Support for video processing steps
- **Mixed Media Chains**: Combine different media types in single chains

### Upcoming Features
- Async processing support
- Parallel chain execution
- Custom model integration
- Enhanced error handling
- Progress tracking
- Chain visualization tools

## Current Limitations

- Text-only input/output
- Sequential processing only
- Single-thread execution
- Limited error recovery
- No streaming support

## Requirements

- Python 3.8+
- litellm
- python-dotenv

## Environment Setup

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License

## Acknowledgments

- LiteLLM for model integration
- OpenAI and Anthropic for LLM APIs

### Model Parameters and Customization
You can customize model behavior by passing parameters specific to each LLM provider:

```python
# Simple usage with default parameters
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Write about: {input}",
        "Enhance the content: {input}"
    ]
)

# Advanced usage with custom parameters
chain = PromptChain(
    models=[
        {
            "name": "openai/gpt-4",
            "params": {
                "temperature": 0.7,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            }
        },
        {
            "name": "anthropic/claude-3-sonnet-20240229",
            "params": {
                "temperature": 0.3,
                "top_k": 40,
                "metadata": {"user_id": "123"}
            }
        }
    ],
    instructions=[
        "Generate JSON analysis of: {input}",
        "Improve the analysis: {input}"
    ]
)

# Mixed usage - both simple and detailed configurations
chain = PromptChain(
    models=[
        "openai/gpt-4",  # Use defaults
        {  # Custom parameters
            "name": "anthropic/claude-3-sonnet-20240229",
            "params": {
                "temperature": 0.3,
                "max_tokens": 200
            }
        },
        "openai/gpt-4"  # Use defaults
    ],
    instructions=[...]
)
```

#### Available Parameters

##### OpenAI Models
```python
openai_params = {
    "temperature": 0.7,        # Controls randomness (0-1)
    "max_tokens": 150,         # Maximum length of response
    "top_p": 1.0,             # Nucleus sampling
    "frequency_penalty": 0.2,  # Reduce repetition
    "presence_penalty": 0.1,   # Encourage new topics
    "stop": ["###"],          # Custom stop sequences
    "response_format": {       # Force specific formats
        "type": "json_object"
    }
}
```

##### Anthropic Models
```python
anthropic_params = {
    "temperature": 0.3,     # Controls randomness
    "max_tokens": 200,      # Maximum length
    "top_k": 40,           # Top-k sampling
    "top_p": 0.9,          # Nucleus sampling
    "metadata": {          # Custom metadata
        "user_id": "123",
        "session_id": "456"
    }
}
```

#### Tracking Parameters in Chain History
When using `full_history=True`, you can see which parameters were used at each step:

```python
results = chain.process_prompt("Analysis topic")
for step in results:
    if step['type'] == 'model':
        print(f"Step {step['step']}:")
        print(f"Parameters used: {step['model_params']}")
        print(f"Output: {step['output']}\n")
```

Check the `src/examples/` directory for complete examples:
- `model_params_example.py`: Detailed parameter customization
- `mixed_params_example.py`: Mixed default and custom parameters
- `chain_examples.py`: Various chain configurations

## API Key Setup

There are two ways to set up your API keys:

### 1. Using .env File (Recommended)

1. Copy the sample environment file:
```bash
cp .env.sample .env
```

2. Edit .env and add your API keys:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
# Add other API keys as needed
```

3. The keys will be automatically loaded when you run the application

### 2. Setting Keys Programmatically

You can also set API keys during runtime:

```python
import os
os.environ["OPENAI_API_KEY"] = "your_key_here"
os.environ["ANTHROPIC_API_KEY"] = "your_key_here"

# Then create your chain
chain = PromptChain(...)
```

### Available API Keys

Different models require different API keys:

| Model Provider | Environment Variable | Required For |
|---------------|---------------------|--------------|
| OpenAI | OPENAI_API_KEY | GPT-3.5, GPT-4 |
| Anthropic | ANTHROPIC_API_KEY | Claude models |
| Groq | GROQ_API_KEY | Groq models |
| Google | GOOGLE_GENERATIVE_AI_API_KEY | Gemini models |
| Mistral | MISTRAL_API_KEY | Mistral models |
| OpenRouter | OPEN_ROUTER_API_KEY | OpenRouter models |
| Ollama | OLLAMA_API_BASE_URL | Local Ollama models |

Note: You only need to set the API keys for the models you plan to use.

### Advanced Prompt Engineering Use Cases

#### 1. Task Decomposition and Chain-of-Thought
Use PromptChain to break down complex tasks and generate structured prompts:

```python
TASK_DECOMPOSITION_PROMPT = """
Analyze this task and break it down into sequential steps:
1. Identify the main objective
2. List required sub-tasks
3. Specify dependencies between tasks
4. Identify potential challenges

Task to analyze: {input}
"""

COT_GENERATION_PROMPT = """
For each step identified, create a chain-of-thought prompt that:
- Explains the reasoning process
- Identifies key decision points
- Lists required information
- Suggests validation steps

Steps to analyze: {input}
"""

PROMPT_ASSEMBLY_PROMPT = """
Create a comprehensive prompt that:
1. Incorporates the task breakdown
2. Includes chain-of-thought guidance
3. Specifies expected output format
4. Adds relevant examples

Input analysis: {input}
"""

# Create a chain for prompt engineering
prompt_engineering_chain = PromptChain(
    models=[
        "anthropic/claude-3-sonnet-20240229",  # Good at task analysis
        "openai/gpt-4",                        # Strong at COT reasoning
        analyze_prompt_complexity,              # Custom validation function
        "anthropic/claude-3-sonnet-20240229"   # Final prompt assembly
    ],
    instructions=[
        TASK_DECOMPOSITION_PROMPT,
        COT_GENERATION_PROMPT,
        analyze_prompt_complexity,
        PROMPT_ASSEMBLY_PROMPT
    ],
    full_history=True
)

# Generate specialized prompt
task = "Create a trading algorithm that considers market sentiment"
final_prompt = prompt_engineering_chain.process_prompt(task)
```

#### 2. Iterative Prompt Refinement
Use the chain to iteratively improve a prompt based on specific criteria:

```python
def evaluate_prompt_clarity(prompt: str) -> str:
    """Analyze prompt clarity and suggest improvements"""
    metrics = {
        "specificity": check_specificity(prompt),
        "completeness": check_completeness(prompt),
        "ambiguity": check_ambiguity(prompt)
    }
    return json.dumps(metrics, indent=2)

def test_prompt_output(prompt: str) -> str:
    """Test prompt with different inputs and analyze results"""
    test_inputs = ["simple case", "edge case", "complex case"]
    results = []
    for test in test_inputs:
        # Simulate prompt testing
        result = simulate_prompt_execution(prompt, test)
        results.append({"input": test, "output": result})
    return json.dumps(results, indent=2)

refinement_chain = PromptChain(
    models=[
        "openai/gpt-4",
        evaluate_prompt_clarity,    # Custom evaluation
        test_prompt_output,        # Custom testing
        "anthropic/claude-3-sonnet-20240229"
    ],
    instructions=[
        "Create initial prompt for: {input}",
        evaluate_prompt_clarity,
        test_prompt_output,
        """
        Improve the prompt based on evaluation and tests:
        - Address clarity issues
        - Fix ambiguities
        - Handle edge cases
        - Add constraints and examples
        
        Evaluation results: {input}
        """
    ],
    full_history=True
)
```

#### 3. Using Refined Prompts
The generated prompts can then be used with any LLM:

```python
# Generate specialized prompt
task_description = "Generate product descriptions for e-commerce"
refined_prompt = refinement_chain.process_prompt(task_description)

# Use the refined prompt with any model
execution_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[refined_prompt['output']],
    full_history=False
)

# Process actual content
product_data = "Raw product specifications..."
final_description = execution_chain.process_prompt(product_data)
```

This approach helps:
- Create more robust and reliable prompts
- Handle edge cases systematically
- Validate prompt effectiveness
- Generate reusable prompt templates
- Improve consistency in outputs

### Step Output Storage
You can store step outputs for later access without returning the full history:

```python
# Initialize chain with step storage
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Initial analysis: {input}",
        "Detailed summary: {input}"
    ],
    full_history=False,  # Don't return full history
    store_steps=True     # But store steps for later access
)

# Get only final result
final_result = chain.process_prompt("Topic")

# Later access specific steps if needed
initial_analysis = chain.get_step_output(1)
detailed_summary = chain.get_step_output(2)

# Access all stored steps
for step_num in range(len(chain.step_outputs)):
    step_data = chain.get_step_output(step_num)
    print(f"Step {step_num}: {step_data['type']}")
```

This approach:
- Returns only the final result for efficiency
- Stores step outputs for later reference
- Allows selective access to intermediate results
- Maintains clean API while preserving history

### Listing Available Prompts
You can list all available prompts and their categories:

```python
from promptchain import list_available_prompts, print_available_prompts

# Get dictionary of prompts
prompts = list_available_prompts()
for category, prompt_list in prompts.items():
    print(f"\nCategory: {category}")
    for prompt in prompt_list:
        print(f"  - {prompt['name']}: {prompt['description']}")

# Or use the pretty print function
print_available_prompts()
```

Example output:
```
Available Prompts:
=================

ANALYSIS:
---------
  ANALYSIS_INITIAL_ANALYSIS:
    Description: Perform a detailed analysis of the given topic
    Path: prompts/analysis/initial_analysis.md

  ANALYSIS_REFINEMENT:
    Description: Enhance and structure the previous analysis
    Path: prompts/analysis/refinement.md

STORY:
------
  STORY_OUTLINE:
    Description: Create a structured outline for a story
    Path: prompts/story/outline.md
...
```

### Single Model Simplification
When using the same model for all steps, you can provide just one model:

```python
# All instructions will use GPT-4
chain = PromptChain(
    models=["openai/gpt-4"],  # Single model for all instructions
    instructions=[
        "Initial analysis: {input}",
        "Detailed expansion: {input}",
        "Final summary: {input}"
    ]
)

# Same with custom parameters
chain = PromptChain(
    models=[{
        "name": "openai/gpt-4",
        "params": {
            "temperature": 0.7,
            "max_tokens": 150
        }
    }],  # Same model and parameters for all instructions
    instructions=[
        "Write initial draft: {input}",
        "Improve content: {input}",
        "Create summary: {input}"
    ]
)

# Mixed with functions
chain = PromptChain(
    models=["openai/gpt-4"],  # Will be used for both non-function instructions
    instructions=[
        "Generate content: {input}",
        analyze_text,  # Function doesn't need a model
        "Summarize analysis: {input}"
    ]
)
```

This simplifies chain creation when you want to use the same model throughout the process.

## Multimodal Input Processing

PromptChain can be extended to handle multimodal inputs such as documents, images, videos, and audio. This is achieved by introducing a preprocessing function that converts these inputs into a text format suitable for the prompt chain.

### Example: Preprocessing Multimodal Inputs

Here's how you can set up a preprocessing function and integrate it into a prompt chain:

```python
def preprocess_multimodal_input(input_data):
    # Example function to process different types of media
    if isinstance(input_data, str):
        # Assume it's a document
        return extract_text_from_document(input_data)
    elif isinstance(input_data, Image):
        return extract_text_from_image(input_data)
    elif isinstance(input_data, Video):
        return extract_text_from_video(input_data)
    elif isinstance(input_data, Audio):
        return transcribe_audio(input_data)
    else:
        raise ValueError("Unsupported input type")

# Initialize the PromptChain with the preprocessing function as the first step
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        preprocess_multimodal_input,  # Preprocessing step
        "Analyze the processed text: {input}",
        "Generate insights based on the analysis: {input}"
    ]
)

# Run the chain with a multimodal input
result = chain.process_prompt(multimodal_input)
```

### Considerations

- **Function Integration**: Ensure that the `PromptChain` can handle functions as part of its instructions. This might require checking or modifying the library's implementation.
  
- **Error Handling**: Implement robust error handling within the preprocessing function to manage different input types and potential processing errors.

- **Performance**: Consider the performance implications of processing large or complex media files, and optimize the preprocessing function accordingly.
