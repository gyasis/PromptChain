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

### Structured Prompts
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

### Prompt Engineering Tools

### Interactive Prompt Improver

The package includes a powerful command-line tool for iteratively improving prompts. Located at `examples/prompt_improver.py`, this tool provides multiple ways to input and improve prompts with different focus areas.

#### Basic Usage

```bash
# Improve a simple prompt
python examples/prompt_improver.py "Write a story about space exploration"

# Use interactive mode for multiline prompts
python examples/prompt_improver.py -i

# Read prompt from clipboard
python examples/prompt_improver.py -c
```

#### Input Methods

1. **Interactive Mode** (`-i` or `--interactive`):
   - Enter multiline prompts directly
   - Use Ctrl+D (Unix) or Ctrl+Z (Windows) to finish input
   ```bash
   python examples/prompt_improver.py -i
   ```

2. **Clipboard Mode** (`-c` or `--clipboard`):
   - Read prompt directly from system clipboard
   - Supports pbpaste (macOS) and xclip (Linux)
   ```bash
   python examples/prompt_improver.py -c
   ```

3. **Direct Input**:
   - Pass prompt as command line argument
   ```bash
   python examples/prompt_improver.py "Your prompt here"
   ```

4. **Pipe Input**:
   - Pipe content from another command
   ```bash
   pbpaste | python examples/prompt_improver.py
   ```

#### Improvement Modes

1. **LLM Evaluation** (default):
   ```bash
   python examples/prompt_improver.py -i --mode llm
   ```

2. **Human Evaluation**:
   ```bash
   python examples/prompt_improver.py -i --mode human
   ```

3. **Continuous Improvement** (no evaluation):
   ```bash
   python examples/prompt_improver.py -i --mode continuous
   ```

#### Focus Areas

Specify improvement focus with `--focus`:

- `clarity`: Enhance clarity and specificity
- `completeness`: Add comprehensive details
- `creativity`: Make prompts more engaging
- `technical`: Optimize for technical accuracy

```bash
python examples/prompt_improver.py -i --focus technical
```

##### Focus Area Details

Each focus area comes with specific improvement criteria:

1. **Clarity** (`--focus clarity`):
   ```text
   - Eliminate ambiguity
   - Use precise language
   - Provide clear examples
   - Structure information logically
   - Define unfamiliar terms or acronyms
   - Use active voice for direct instructions
   - Prioritize key information first
   - Specify the intended audience's knowledge level
   - Include visual/spatial descriptors for technical prompts
   - Use consistent terminology throughout
   ```

2. **Completeness** (`--focus completeness`):
   ```text
   - Add missing requirements
   - Include edge cases
   - Specify constraints
   - Define success criteria
   - Anticipate and address potential misinterpretations
   - Add validation criteria for outputs
   - Specify required data sources/materials
   - Include time/scope boundaries
   - Define acceptable abstraction levels
   - Add fallback options for ambiguous situations
   - Require progress checkpoints for multi-step tasks
   ```

3. **Creativity** (`--focus creativity`):
   ```text
   - Add interesting scenarios
   - Include thought-provoking elements
   - Encourage innovative thinking
   - Balance creativity with practicality
   - Incorporate metaphorical frameworks
   - Add constraints to spark ingenuity
   - Request multiple alternative solutions
   - Include cross-domain inspiration sources
   - Use narrative/storytelling elements
   - Add "what-if" hypothetical modifiers
   - Encourage redefinition of problem boundaries
   ```

4. **Technical** (`--focus technical`):
   ```text
   - Add technical specifications
   - Include implementation details
   - Specify input/output formats
   - Define error handling requirements
   - Include performance benchmarks/metrics
   - Specify security/privacy requirements
   - Define compatibility constraints
   - Add versioning/update considerations
   - Include failure mode analysis
   - Specify testing/validation protocols
   - Define documentation requirements
   - Add resource optimization targets
   ```

Example using focus with other options:
```bash
# Improve a prompt with technical focus and human evaluation
python examples/prompt_improver.py -i \
    --focus technical \
    --mode human \
    --verbose

# Read from clipboard and improve for clarity
python examples/prompt_improver.py -c \
    --focus clarity \
    --max-iterations 5

# Pipe content and improve for completeness
cat prompt.txt | python examples/prompt_improver.py \
    --focus completeness \
    --mode continuous
```

#### Additional Options

- `--max-iterations`: Set maximum improvement iterations (default: 5)
- `--verbose`: Show detailed improvement process

#### Complete Example

```bash
# Interactive mode with technical focus, human evaluation, and verbose output
python examples/prompt_improver.py -i \
    --mode human \
    --focus technical \
    --max-iterations 3 \
    --verbose
```

#### Requirements

For clipboard support:
- macOS: Uses built-in `pbpaste`
- Linux: Requires `xclip`
  ```bash
  # Ubuntu/Debian
  sudo apt-get install xclip
  
  # CentOS/RHEL
  sudo yum install xclip
  ```

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

## Prompt Engineering Tools

### Interactive Prompt Improver

The package includes a powerful command-line tool for iteratively improving prompts. Located at `examples/prompt_improver.py`, this tool provides multiple ways to input and improve prompts with different focus areas.

#### Basic Usage

```bash
# Improve a simple prompt
python examples/prompt_improver.py "Write a story about space exploration"

# Use interactive mode for multiline prompts
python examples/prompt_improver.py -i

# Read prompt from clipboard
python examples/prompt_improver.py -c
```

#### Input Methods

1. **Interactive Mode** (`-i` or `--interactive`):
   - Enter multiline prompts directly
   - Use Ctrl+D (Unix) or Ctrl+Z (Windows) to finish input
   ```bash
   python examples/prompt_improver.py -i
   ```

2. **Clipboard Mode** (`-c` or `--clipboard`):
   - Read prompt directly from system clipboard
   - Supports pbpaste (macOS) and xclip (Linux)
   ```bash
   python examples/prompt_improver.py -c
   ```

3. **Direct Input**:
   - Pass prompt as command line argument
   ```bash
   python examples/prompt_improver.py "Your prompt here"
   ```

4. **Pipe Input**:
   - Pipe content from another command
   ```bash
   pbpaste | python examples/prompt_improver.py
   ```

#### Improvement Modes

1. **LLM Evaluation** (default):
   ```bash
   python examples/prompt_improver.py -i --mode llm
   ```

2. **Human Evaluation**:
   ```bash
   python examples/prompt_improver.py -i --mode human
   ```

3. **Continuous Improvement** (no evaluation):
   ```bash
   python examples/prompt_improver.py -i --mode continuous
   ```

#
#### Additional Options

- `--max-iterations`: Set maximum improvement iterations (default: 5)
- `--verbose`: Show detailed improvement process

#### Complete Example

```bash
# Interactive mode with technical focus, human evaluation, and verbose output
python examples/prompt_improver.py -i \
    --mode human \
    --focus technical \
    --max-iterations 3 \
    --verbose
```

#### Requirements

For clipboard support:
- macOS: Uses built-in `pbpaste`
- Linux: Requires `xclip`
  ```bash
  # Ubuntu/Debian
  sudo apt-get install xclip
  
  # CentOS/RHEL
  sudo yum install xclip
  ```

### Chain Breakers

Chain breakers allow you to interrupt the prompt chain processing when certain conditions are met. This is useful for:

- Stopping on error conditions
- Halting when confidence is too low
- Limiting output length
- Validating domain-specific data
- Detecting quality issues

```python
from promptchain.utils.promptchaining import PromptChain

# Define a chain breaker function
def break_on_keywords(step: int, output: str):
    """Break the chain if specific keywords are found in the output."""
    keywords = ["error", "invalid", "cannot", "impossible"]
    
    for keyword in keywords:
        if keyword.lower() in output.lower():
            return (
                True,  # Yes, break the chain
                f"Keyword '{keyword}' detected in output",  # Reason
                f"CHAIN INTERRUPTED: The process was stopped because '{keyword}' was detected."  # Modified output
            )
    
    # If no keywords found, continue the chain
    return (False, "", None)

# Create a chain with the breaker
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Analyze this problem: {input}",
        "Propose solutions for this problem: {input}"
    ],
    chainbreakers=[break_on_keywords]  # Add the chain breaker
)

result = chain.process_prompt("How to implement a perpetual motion machine?")
```

#### Creating Factory Functions for Configurable Chain Breakers

You can create factory functions that return configured chain breakers:

```python
def break_after_steps(max_steps: int):
    """Create a chainbreaker that stops after a specific number of steps."""
    def _breaker(step: int, output: str):
        if step >= max_steps:
            return (
                True,  # Yes, break the chain
                f"Maximum step count reached ({max_steps})",  # Reason
                output  # Keep the original output
            )
        return (False, "", None)
    
    return _breaker

# Use the factory function to create a breaker that stops after 2 steps
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
    instructions=[
        "Step 1: Outline a story about: {input}",
        "Step 2: Develop the characters: {input}",
        "Step 3: Write the first chapter: {input}"
    ],
    chainbreakers=[break_after_steps(2)]  # Only run first 2 steps
)
```

#### Using Multiple Chain Breakers

You can use multiple chain breakers together:

```python
chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Analyze this data: {input}",
        "Make predictions based on the analysis: {input}"
    ],
    chainbreakers=[
        break_on_keywords,
        break_on_low_confidence,
        break_on_length(max_length=1000)
    ]
)
```

#### Breaking on Functions

You can also create chainbreakers that specifically target function steps in your chain:

```python
# Define a chainbreaker that stops when a specific function is encountered
def break_on_function(function_name: str):
    """Create a chainbreaker that stops when a specific function is encountered."""
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # Check if step_info is provided and contains function information
        if step_info and step_info.get('type') == 'function':
            # Get the function name from the step info
            current_function = step_info.get('function_name', '')
            
            # Check if this is the function we want to break on
            if current_function == function_name:
                return (
                    True,  # Yes, break the chain
                    f"Function '{function_name}' encountered",  # Reason
                    output  # Keep the original output
                )
        
        return (False, "", None)
    
    return _breaker

# Define a chainbreaker that stops when a function output meets a condition
def break_on_function_condition(condition_func):
    """Create a chainbreaker that stops when a function output meets a condition."""
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # Only check function steps
        if step_info and step_info.get('type') == 'function':
            # Check if the condition is met
            if condition_func(output, step_info):
                return (
                    True,  # Yes, break the chain
                    "Function condition met",  # Reason
                    output  # Keep the original output
                )
        
        return (False, "", None)
    
    return _breaker

# Example usage with functions
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    # Sentiment analysis logic here
    return f"Sentiment Analysis: The text has a positive sentiment."

# Define a condition: break if sentiment is negative
def negative_sentiment_condition(output: str, step_info: dict) -> bool:
    return "negative sentiment" in output.lower()

# Create a chain with function chainbreakers
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Write a review of: {input}",
        analyze_sentiment,  # Function that might trigger the break
        "Provide recommendations based on the review: {input}"
    ],
    chainbreakers=[
        break_on_function("analyze_sentiment"),  # Break on specific function
        break_on_function_condition(negative_sentiment_condition)  # Break on condition
    ]
)
```

This allows you to:
- Stop processing when specific functions are encountered
- Break the chain based on function output conditions
- Implement domain-specific validation logic
- Create complex decision trees in your processing flow

See `examples/chainbreaker_examples.py` for more detailed examples of chain breakers.

### Circular Processing

For detailed documentation on using circular processing patterns in PromptChain, see `docs/circular_runs.md`. This feature allows you to create iterative refinement loops by feeding chain outputs back into the beginning of the chain.
