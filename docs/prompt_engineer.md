# Prompt Engineer Guide

The Prompt Engineer is a specialized component of PromptChain designed for iterative prompt engineering with meta-evaluation. It provides tools for creating, testing, and optimizing prompts through both automated and human-guided processes.

## Core Components

### Class Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 3 | Maximum number of improvement iterations |
| `use_human_evaluation` | bool | False | Whether to use human evaluation instead of LLM |
| `verbose` | bool | False | Enable detailed logging |

### Internal Chains

1. **Evaluator Chain**
   ```python
   # Evaluates if prompts need improvement
   evaluator = PromptChain(
       models=["anthropic/claude-3-sonnet-20240229"],
       instructions=["""Evaluate if this prompt is acceptable or needs improvement.
           Consider:
           1. Clarity: Are the instructions clear and specific?
           2. Completeness: Are all necessary guidelines included?
           3. Task Alignment: Does it match the intended task?
           4. Output Quality: Will it generate high quality outputs?

           Respond with either:
           PASS: [Brief explanation why it's acceptable]
           or
           FAIL: [List specific improvements needed]"""]
   )
   ```

2. **Improver Chain**
   ```python
   # Improves prompts based on feedback
   improver = PromptChain(
       models=["openai/gpt-4o"],
       instructions=["""You are a prompt engineering expert. Your task is to improve the given prompt based on specific feedback.

           IMPORTANT: Do NOT change the core purpose or domain of the prompt. Maintain the original intent and subject matter.

           Current Prompt:
           {input}

           Follow these steps to improve the prompt:
           1. Understand the original prompt's purpose and domain
           2. Apply the specific feedback provided
           3. Make targeted improvements while preserving the original intent
           4. Return ONLY the improved prompt without explanations"""]
   )
   ```

## Available Techniques and Choices

### Required Parameter Techniques

These techniques require specific parameters to function:

| Technique | Parameter Format | Description | Examples |
|-----------|-----------------|-------------|----------|
| `role_playing` | `role_playing:profession` | Adopt a specific professional role or expertise | - `role_playing:scientist`<br>- `role_playing:writer`<br>- `role_playing:teacher` |
| `style_mimicking` | `style_mimicking:author` | Mimic the writing style of a specific author or source | - `style_mimicking:Richard Feynman`<br>- `style_mimicking:Ernest Hemingway`<br>- `style_mimicking:technical documentation` |
| `persona_emulation` | `persona_emulation:expert` | Emulate the perspective and approach of a specific expert | - `persona_emulation:Warren Buffett`<br>- `persona_emulation:Steve Jobs`<br>- `persona_emulation:Marie Curie` |
| `forbidden_words` | `forbidden_words:word1,word2` | Specify words to avoid in the output | - `forbidden_words:maybe,probably,perhaps`<br>- `forbidden_words:basically,actually,literally`<br>- `forbidden_words:very,really,quite` |

### Optional Parameter Techniques

These techniques can be used with or without parameters:

| Technique | Parameter Format | Description | Examples |
|-----------|-----------------|-------------|----------|
| `few_shot` | `few_shot:number` | Include specific number of examples | - `few_shot:3`<br>- `few_shot:5`<br>- `few_shot` (uses default) |
| `reverse_prompting` | `reverse_prompting:number` | Generate specified number of self-reflection questions | - `reverse_prompting:3`<br>- `reverse_prompting:5`<br>- `reverse_prompting` |
| `context_expansion` | `context_expansion:type` | Expand context in a specific direction | - `context_expansion:historical`<br>- `context_expansion:technical`<br>- `context_expansion:cultural` |
| `comparative_answering` | `comparative_answering:aspects` | Compare specific aspects or approaches | - `comparative_answering:pros_cons`<br>- `comparative_answering:cost_benefit`<br>- `comparative_answering:risk_reward` |
| `tree_of_thought` | `tree_of_thought:paths` | Explore multiple reasoning paths | - `tree_of_thought:3`<br>- `tree_of_thought:5`<br>- `tree_of_thought` |

### No-Parameter Techniques

These techniques are used as flags without additional parameters:

| Technique | Description | Use Case |
|-----------|-------------|----------|
| `step_by_step` | Break down complex processes into discrete steps | Complex procedures, tutorials |
| `chain_of_thought` | Show detailed reasoning process | Problem-solving, analysis |
| `iterative_refinement` | Gradually improve output through iterations | Content optimization |
| `contrarian_perspective` | Challenge assumptions and common beliefs | Critical analysis |
| `react` | Implement Reason-Act-Observe cycle | Interactive problem-solving |

### Combining Techniques

You can combine multiple techniques to create sophisticated prompt engineering strategies:

```bash
# Basic combination
--techniques role_playing:scientist step_by_step

# Multiple techniques with parameters
--techniques role_playing:writer chain_of_thought few_shot:3

# Complex combination
--techniques role_playing:analyst style_mimicking:technical tree_of_thought:3 step_by_step
```

### Technique Selection Guidelines

1. **Role-Based Tasks**
   - Use `role_playing` for expertise-specific tasks
   - Combine with `style_mimicking` for authentic voice
   - Example: `role_playing:scientist style_mimicking:research_paper`

2. **Educational Content**
   - Use `step_by_step` for clear instruction
   - Add `few_shot` for examples
   - Example: `step_by_step few_shot:3 role_playing:teacher`

3. **Analysis Tasks**
   - Use `chain_of_thought` for reasoning
   - Add `tree_of_thought` for multiple perspectives
   - Example: `chain_of_thought tree_of_thought:3 role_playing:analyst`

4. **Content Creation**
   - Use `style_mimicking` for tone
   - Add `iterative_refinement` for polish
   - Example: `style_mimicking:journalist iterative_refinement`

5. **Problem Solving**
   - Use `react` for systematic approach
   - Add `comparative_answering` for options
   - Example: `react comparative_answering:pros_cons`

## Usage Modes

### 1. Interactive Mode

```bash
python -m promptchain.utils.prompt_engineer -i
```

Features:
- Guided technique selection
- Parameter configuration
- Settings customization
- Progress tracking
- Real-time feedback

### 2. Command Line Mode

```bash
python -m promptchain.utils.prompt_engineer \
    --techniques role_playing:scientist step_by_step few_shot:3 \
    --feedback llm \
    --max-iterations 5 \
    --verbose
```

#### Command Line Parameters

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `--techniques` | List[str] | [] | Space-separated list of techniques to apply | `--techniques role_playing:scientist step_by_step` |
| `--interactive` or `-i` | bool | False | Enable interactive mode for technique selection | `-i` |
| `--feedback` | str | "llm" | Evaluation type ("human" or "llm") | `--feedback human` |
| `--max-iterations` | int | 3 | Maximum number of improvement iterations | `--max-iterations 5` |
| `--verbose` | bool | False | Enable detailed logging | `--verbose` |
| `--output-file` | str | None | File to save the final prompt to | `--output-file prompt.txt` |
| `--task` | str | None | Task description for prompt creation | `--task "Generate SQL queries"` |
| `--initial-prompt` | str | None | Initial prompt to improve | `--initial-prompt "Write a story..."` |
| `--evaluator-model` | str | "anthropic/claude-3-sonnet-20240229" | Model to use for evaluation | `--evaluator-model "openai/gpt-4"` |
| `--improver-model` | str | "openai/gpt-4o" | Model to use for improvements | `--improver-model "anthropic/claude-3"` |
| `--test` | bool | False | Enable prompt testing | `--test` |
| `--test-inputs` | List[str] | None | Comma-separated test inputs | `--test-inputs "input1,input2,input3"` |
| `--focus` | str | "all" | Focus area for improvements | `--focus clarity` |

#### Example Commands

1. **Basic Usage**
```bash
python -m promptchain.utils.prompt_engineer \
    --task "Create a story prompt" \
    --max-iterations 3
```

2. **Interactive Mode with Custom Models**
```bash
python -m promptchain.utils.prompt_engineer \
    -i \
    --evaluator-model "openai/gpt-4" \
    --improver-model "anthropic/claude-3-sonnet-20240229"
```

3. **Full Testing Configuration**
```bash
python -m promptchain.utils.prompt_engineer \
    --techniques role_playing:writer chain_of_thought few_shot:3 \
    --test \
    --test-inputs "fantasy story,sci-fi story,mystery story" \
    --focus clarity \
    --output-file tested_prompt.txt \
    --verbose
```

4. **Improve Existing Prompt**
```bash
python -m promptchain.utils.prompt_engineer \
    --initial-prompt "Write a story about space exploration" \
    --feedback human \
    --max-iterations 5 \
    --verbose
```

5. **Focus-Specific Improvement**
```bash
python -m promptchain.utils.prompt_engineer \
    --task "Generate product descriptions" \
    --focus completeness \
    --techniques style_mimicking:copywriter step_by_step \
    --max-iterations 4
```

#### Focus Areas

| Focus Area | Description | Use Case |
|------------|-------------|-----------|
| `clarity` | Improve instruction clarity and specificity | Complex technical prompts |
| `completeness` | Ensure all necessary components are included | Comprehensive guides |
| `task_alignment` | Optimize for specific task requirements | Specialized applications |
| `output_quality` | Focus on generating high-quality results | Creative content |
| `all` | Balance all improvement aspects | General optimization |

### 3. Programmatic Usage

```python
from promptchain.utils.prompt_engineer import PromptEngineer

# Initialize engineer
engineer = PromptEngineer(
    max_iterations=3,
    use_human_evaluation=False,
    verbose=True
)

# Create specialized prompt
task = """Create a prompt for an AI agent that helps users analyze financial data.
The agent should:
1. Extract key metrics
2. Identify trends
3. Provide actionable insights
4. Format output as a structured report"""

optimized_prompt = engineer.create_specialized_prompt(task)
```

## Core Functions

### 1. create_specialized_prompt()

Creates and iteratively improves a specialized prompt for a given task.

```python
def create_specialized_prompt(self, task_description: str) -> str:
    """
    Args:
        task_description: Description of the task to create a prompt for
    Returns:
        The final optimized prompt
    """
```

### 2. test_prompt()

Tests a prompt with multiple inputs and evaluates consistency.

```python
def test_prompt(self, prompt: str, test_inputs: List[str]) -> Dict:
    """
    Args:
        prompt: The prompt to test
        test_inputs: List of test inputs
    Returns:
        Evaluation results with consistency metrics
    """
```

### 3. improve_prompt_continuously()

Continuously improves a prompt using the improvement prompt.

```python
def improve_prompt_continuously(self, initial_prompt: str) -> str:
    """
    Args:
        initial_prompt: The initial prompt to improve
    Returns:
        The final improved prompt
    """
```

## Interactive Features

### Kitchen Sink Mode

Allows bulk editing of all parameters:
- Configure all techniques at once
- View current command parameters
- Quick parameter updates
- Command preview

### Progress Tracking

```python
def track_chain_progress(output: str, history: list = None) -> None:
    """
    Tracks progress and provides detailed logging:
    - Step number
    - Quality score
    - Changes made
    - Progress indicators
    """
```

### Configuration Settings

Available settings:
- Feedback type (human/llm)
- Max iterations
- Verbose output
- Task description
- Initial prompt
- Model selection
- Testing options
- Focus area
- Output file

## Best Practices

1. **Evaluation Strategy**
   - Use LLM evaluation for rapid iteration
   - Switch to human evaluation for critical prompts
   - Combine both for optimal results

2. **Iteration Management**
   - Start with 3-5 iterations
   - Monitor improvement delta
   - Stop when convergence reached
   - Use verbose mode for debugging

3. **Technique Selection**
   - Combine complementary techniques
   - Start with role-based techniques
   - Add structural techniques
   - Include quality controls

4. **Testing and Validation**
   - Use diverse test inputs
   - Check consistency across outputs
   - Validate against edge cases
   - Monitor resource usage

## Example Scripts

Check the `examples/prompt_engineer/` directory for:
- Basic usage examples
- Advanced configurations
- Testing implementations
- Custom technique combinations

## Troubleshooting

### Common Issues

1. **Evaluation Loop**
   - Check max_iterations setting
   - Verify evaluation criteria
   - Monitor improvement metrics

2. **Quality Issues**
   - Review technique combinations
   - Adjust model parameters
   - Increase test coverage

3. **Resource Management**
   - Monitor token usage
   - Optimize chain length
   - Use appropriate models

### Debug Mode

```python
# Enable comprehensive debugging
engineer = PromptEngineer(
    max_iterations=3,
    verbose=True,
    use_human_evaluation=False
)

# Add debug callbacks
engineer.evaluator.add_debug_callbacks({
    "on_start": log_start,
    "on_complete": log_complete,
    "on_error": log_error
})
```

## Command Line Parameters

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `--techniques` | List[str] | [] | Space-separated list of techniques to apply | `--techniques role_playing:scientist step_by_step` |
| `--feedback` | str | "llm" | Type of feedback ("llm" or "human") | `--feedback human` |
| `--max-iterations` | int | 3 | Maximum number of improvement iterations | `--max-iterations 5` |
| `--verbose` | bool | False | Enable detailed logging | `--verbose` |
| `--task` | str | None | Task description for prompt creation | `--task "Generate SQL queries"` |
| `--initial-prompt` | str | None | Initial prompt to improve (use "-" for stdin) | `--initial-prompt "Write a story..."` |
| `--evaluator-model` | str | "anthropic/claude-3-sonnet-20240229" | Model to use for evaluation | `--evaluator-model "openai/gpt-4"` |
| `--improver-model` | str | "openai/gpt-4o" | Model to use for improvements | `--improver-model "anthropic/claude-3"` |
| `--test` | bool | False | Enable prompt testing | `--test` |
| `--test-inputs` | List[str] | None | Test inputs for prompt testing | `--test-inputs "input1,input2,input3"` |
| `--focus` | str | "all" | Focus area for improvements | `--focus clarity` |
| `-i, --interactive` | bool | False | Use interactive mode | `-i` |
| `--output-file` | str | None | File to save the final prompt | `--output-file prompt.txt` |

### Focus Areas

| Focus Area | Description | Use Case |
|------------|-------------|-----------|
| `clarity` | Improve instruction clarity and specificity | Complex technical prompts |
| `completeness` | Ensure all necessary components are included | Comprehensive guides |
| `task_alignment` | Optimize for specific task requirements | Specialized applications |
| `output_quality` | Focus on generating high-quality results | Creative content |
| `all` | Balance all improvement aspects | General optimization |

## Available Techniques

### Required Parameter Techniques

| Technique | Parameter Format | Description | Example |
|-----------|-----------------|-------------|----------|
| `role_playing` | `role_playing:profession` | Define expert role | `role_playing:scientist` |
| `style_mimicking` | `style_mimicking:author` | Mimic writing style | `style_mimicking:Richard Feynman` |
| `persona_emulation` | `persona_emulation:expert` | Emulate expert perspective | `persona_emulation:Warren Buffett` |
| `forbidden_words` | `forbidden_words:word1,word2` | Words to avoid | `forbidden_words:maybe,probably,perhaps` |

### Optional Parameter Techniques

| Technique | Parameter Format | Description | Example |
|-----------|-----------------|-------------|----------|
| `few_shot` | `few_shot:number` | Number of examples to include | `few_shot:3` |
| `reverse_prompting` | `reverse_prompting:number` | Number of self-reflection questions | `reverse_prompting:5` |
| `context_expansion` | `context_expansion:type` | Type of context to expand | `context_expansion:historical` |
| `comparative_answering` | `comparative_answering:aspects` | Aspects to compare | `comparative_answering:pros_cons` |
| `tree_of_thought` | `tree_of_thought:paths` | Number of reasoning paths | `tree_of_thought:3` |

### No-Parameter Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| `step_by_step` | Break down complex processes | Tutorials, procedures |
| `chain_of_thought` | Show detailed reasoning process | Problem-solving |
| `iterative_refinement` | Gradually improve output | Content optimization |
| `contrarian_perspective` | Challenge assumptions | Critical analysis |
| `react` | Reason-Act-Observe cycle | Interactive problem-solving |

## Model Parameters

### Evaluator Model

Default: `anthropic/claude-3-sonnet-20240229`

Parameters can be specified as a JSON object:
```bash
--evaluator-model '{"name": "openai/gpt-4", "params": {"temperature": 0.7, "max_tokens": 1000}}'
```

### Improver Model

Default: `openai/gpt-4o`

Parameters can be specified as a JSON object:
```bash
--improver-model '{"name": "anthropic/claude-3", "params": {"temperature": 0.5, "max_tokens": 2000}}'
```

### Creator Models

Default: `["anthropic/claude-3-sonnet-20240229", "openai/gpt-4o-mini"]`

Can be specified as comma-separated list in interactive mode:
```bash
python -m promptchain.utils.prompt_engineer -i
# Then select option 8 to configure creator models
```

## Interactive Mode Features

### 1. Technique Selection

- Required Parameter Techniques
  - Select technique number
  - Enter required parameter
  - Validates parameter format

- Optional Parameter Techniques
  - Select technique number
  - Choose to add parameter or not
  - Enter parameter if chosen

- No-Parameter Techniques
  - Select technique number
  - Automatically added to configuration

### 2. Kitchen Sink Mode

Allows bulk configuration of:
- All technique parameters
- Model settings
- Evaluation options
- Testing configuration
- Output preferences

### 3. Configuration Settings

| Setting | Options | Description |
|---------|---------|-------------|
| Feedback type | "llm", "human" | Type of evaluation feedback |
| Max iterations | Integer > 0 | Maximum improvement cycles |
| Verbose output | Boolean | Enable detailed logging |
| Task description | String | Task to create prompt for |
| Initial prompt | String | Starting prompt to improve |
| Evaluator model | String/JSON | Model for evaluation |
| Improver model | String/JSON | Model for improvements |
| Creator models | List[String] | Models for initial creation |
| Testing | Boolean | Enable prompt testing |
| Test inputs | List[String] | Inputs for testing |
| Focus area | Choice | Area to focus improvements |
| Output file | String | Save location for prompt |

## Example Commands

1. **Basic Usage with Required Technique**
```bash
python -m promptchain.utils.prompt_engineer \
    --techniques role_playing:scientist \
    --task "Explain quantum mechanics" \
    --max-iterations 3
```

2. **Multiple Techniques with Parameters**
```bash
python -m promptchain.utils.prompt_engineer \
    --techniques role_playing:teacher few_shot:3 step_by_step \
    --focus clarity \
    --verbose
```

3. **Custom Models with Parameters**
```bash
python -m promptchain.utils.prompt_engineer \
    --evaluator-model '{"name": "openai/gpt-4", "params": {"temperature": 0.7}}' \
    --improver-model '{"name": "anthropic/claude-3", "params": {"max_tokens": 2000}}' \
    --techniques style_mimicking:technical comparative_answering:pros_cons
```

4. **Testing Configuration**
```bash
python -m promptchain.utils.prompt_engineer \
    --test \
    --test-inputs "beginner query" "advanced query" "edge case" \
    --focus completeness \
    --output-file tested_prompt.txt
```

5. **Interactive Mode with Initial Setup**
```bash
python -m promptchain.utils.prompt_engineer \
    -i \
    --initial-prompt "Base prompt to improve" \
    --feedback human
```