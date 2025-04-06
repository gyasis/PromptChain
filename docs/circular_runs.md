# Circular Runs in PromptChain

This guide explains how to use circular processing patterns in PromptChain, where the output of a chain can be fed back into the beginning, creating an iterative refinement loop.

## Overview

Circular runs are particularly useful for:

- Progressive refinement of content
- Iterative problem solving
- Continuous improvement loops
- Self-correcting processes
- Evolutionary text generation

## Implementation Guide

### Basic Circular Run

Here's a simple example of a circular run:

```python
from promptchain import PromptChain

# Create a chain that can improve content iteratively
improvement_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        """
        Improve the following content. If it needs more improvement, return 'NEEDS_IMPROVEMENT: improved_content'.
        If it's good enough, return 'FINAL: final_content'
        
        Content to improve: {input}
        """,
    ],
    circular=True,  # Enable circular processing
    max_iterations=5  # Prevent infinite loops
)

# Initial content
initial_content = "AI is good for automation."

# Run the chain with circular processing
result = improvement_chain.process_prompt(
    initial_content,
    circular_condition=lambda output: output.startswith("NEEDS_IMPROVEMENT:"),
    circular_transform=lambda output: output.replace("NEEDS_IMPROVEMENT:", "").strip()
)
```

### Advanced Circular Processing

For more complex scenarios, you can use the full history in each iteration:

```python
def should_continue(output: str, history: list) -> bool:
    """Determine if another iteration is needed based on output and history."""
    if len(history) >= 5:  # Stop after 5 iterations
        return False
        
    # Check if significant improvements are still being made
    if len(history) > 1:
        current = output
        previous = history[-1]['output']
        similarity = calculate_similarity(current, previous)
        return similarity < 0.95  # Continue if changes are significant
        
    return True

def transform_for_next_iteration(output: str, history: list) -> str:
    """Prepare the output for the next iteration using history."""
    # Add context from history
    iterations = len(history)
    return f"""
    Previous iterations: {iterations}
    History summary: {summarize_history(history)}
    Current content: {output}
    """

# Create a chain with history-aware circular processing
advanced_chain = PromptChain(
    models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
    instructions=[
        "Analyze and improve this content: {input}",
        "Refine the improvements further: {input}"
    ],
    circular=True,
    full_history=True,  # Keep track of all iterations
    max_iterations=10
)

# Run with advanced circular processing
result = advanced_chain.process_prompt(
    initial_content,
    circular_condition=should_continue,
    circular_transform=transform_for_next_iteration
)
```

### Multiple Exit Points

You can create chains that can exit the circular processing at different points:

```python
def multi_exit_condition(output: str, history: list) -> bool:
    """Complex exit condition with multiple criteria."""
    # Exit if perfect score achieved
    if "SCORE: 10/10" in output:
        return False
        
    # Exit if no improvement in last 3 iterations
    if len(history) >= 3:
        recent_scores = extract_scores(history[-3:])
        if all(score == recent_scores[0] for score in recent_scores):
            return False
            
    # Exit if maximum quality threshold reached
    if quality_score(output) >= 0.95:
        return False
        
    # Continue if none of the above conditions met
    return True

# Create a chain with multiple exit points
quality_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        """
        Improve this content and rate it (0-10):
        {input}
        
        Provide output in format:
        SCORE: X/10
        CONTENT: improved_text
        """,
    ],
    circular=True,
    max_iterations=10
)

# Run with multiple exit conditions
result = quality_chain.process_prompt(
    initial_content,
    circular_condition=multi_exit_condition,
    circular_transform=lambda output: extract_content(output)
)
```

### Progress Tracking

Monitor the progress of circular processing:

```python
def track_progress(output: str, history: list) -> None:
    """Track progress of the circular processing."""
    iteration = len(history) + 1
    current_score = extract_score(output)
    initial_score = extract_score(history[0]['output']) if history else 0
    
    print(f"Iteration {iteration}:")
    print(f"Initial Score: {initial_score}")
    print(f"Current Score: {current_score}")
    print(f"Improvement: {current_score - initial_score}")
    print("-" * 40)

# Create a chain with progress tracking
tracked_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Improve and score: {input}"],
    circular=True,
    max_iterations=5,
    on_iteration=track_progress  # Called after each iteration
)

# Run with progress tracking
result = tracked_chain.process_prompt(
    initial_content,
    circular_condition=lambda output: "NEEDS_IMPROVEMENT" in output
)
```

### Exit Strategies with Chainbreakers

Chainbreakers provide a powerful way to control circular processing flow. Here are examples of different exit strategies:

```python
from promptchain import PromptChain

# 1. Exit on Quality Threshold
def quality_chainbreaker(step: int, output: str, history: list = None) -> tuple:
    """Break the chain when quality threshold is met."""
    try:
        # Extract quality score from output
        quality_score = float(output.split('QUALITY_SCORE:')[1].split('\n')[0].strip())
        
        if quality_score >= 0.95:
            return (
                True,  # Yes, break the chain
                f"Quality threshold met: {quality_score}",
                f"FINAL OUTPUT (Quality: {quality_score}):\n{output}"
            )
        return (False, "", None)
    except:
        return (False, "", None)

# 2. Exit on Convergence
def convergence_chainbreaker(step: int, output: str, history: list = None) -> tuple:
    """Break when output converges (stops changing significantly)."""
    if not history or len(history) < 2:
        return (False, "", None)
        
    current = output
    previous = history[-1]['output']
    similarity = calculate_similarity(current, previous)
    
    if similarity > 0.98:  # 98% similar
        return (
            True,
            "Output converged - no significant changes",
            f"FINAL (Converged):\n{output}"
        )
    return (False, "", None)

# 3. Exit on Error Detection
def error_chainbreaker(step: int, output: str, history: list = None) -> tuple:
    """Break if errors are detected in the output."""
    error_keywords = ['error', 'invalid', 'failed', 'cannot']
    
    for keyword in error_keywords:
        if keyword.lower() in output.lower():
            return (
                True,
                f"Error detected: '{keyword}'",
                f"PROCESS HALTED - Error found:\n{output}"
            )
    return (False, "", None)

# 4. Exit on Resource Limit
def resource_chainbreaker(step: int, output: str, history: list = None) -> tuple:
    """Break if resource usage exceeds limits."""
    total_tokens = sum(len(h['output'].split()) for h in (history or []))
    total_tokens += len(output.split())
    
    if total_tokens > 5000:  # Token budget
        return (
            True,
            "Token budget exceeded",
            f"HALTED - Token limit reached:\n{output}"
        )
    return (False, "", None)

# Example using multiple chainbreakers
improvement_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        """
        Improve this content and rate its quality (0-1):
        
        {input}
        
        Respond in format:
        QUALITY_SCORE: <score>
        IMPROVED_CONTENT: <content>
        """,
    ],
    circular=True,
    max_iterations=10,
    chainbreakers=[
        quality_chainbreaker,      # Exit on quality threshold
        convergence_chainbreaker,  # Exit on convergence
        error_chainbreaker,        # Exit on error
        resource_chainbreaker      # Exit on resource limit
    ]
)

# Example with combined exit strategy
def combined_exit_strategy(step: int, output: str, history: list = None) -> tuple:
    """Complex exit strategy combining multiple conditions."""
    # Check quality
    try:
        quality_score = float(output.split('QUALITY_SCORE:')[1].split('\n')[0].strip())
        if quality_score >= 0.95:
            return (True, "Quality threshold met", output)
    except:
        pass
    
    # Check convergence
    if history and len(history) >= 2:
        similarity = calculate_similarity(output, history[-1]['output'])
        if similarity > 0.98:
            return (True, "Output converged", output)
    
    # Check improvement rate
    if history and len(history) >= 3:
        recent_scores = [
            float(h['output'].split('QUALITY_SCORE:')[1].split('\n')[0].strip())
            for h in history[-3:]
        ]
        if max(recent_scores) - min(recent_scores) < 0.01:  # No significant improvement
            return (True, "Improvement stagnated", output)
    
    # Check resource usage
    total_tokens = sum(len(h['output'].split()) for h in (history or []))
    if total_tokens > 5000:
        return (True, "Resource limit reached", output)
    
    return (False, "", None)

# Create chain with combined exit strategy
advanced_chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        """
        Improve and rate the content:
        {input}
        
        Provide:
        QUALITY_SCORE: <0-1>
        IMPROVED_CONTENT: <content>
        CHANGES_MADE: <list of improvements>
        """,
    ],
    circular=True,
    max_iterations=15,
    chainbreakers=[combined_exit_strategy]
)

# Example usage with progress tracking
def track_chain_progress(output: str, history: list = None) -> None:
    """Track progress and provide detailed logging."""
    step = len(history) + 1 if history else 1
    try:
        score = float(output.split('QUALITY_SCORE:')[1].split('\n')[0].strip())
        print(f"\nStep {step}:")
        print(f"Quality Score: {score:.3f}")
        print("Changes Made:")
        changes = output.split('CHANGES_MADE:')[1].strip()
        print(changes)
        print("-" * 40)
    except:
        print(f"Step {step}: Error parsing output")

# Run the chain with tracking
result = advanced_chain.process_prompt(
    "Initial content to improve",
    on_iteration=track_chain_progress
)
```

This approach offers several benefits:

1. **Flexible Exit Conditions**
   - Quality-based exits
   - Convergence detection
   - Error handling
   - Resource management

2. **Comprehensive Monitoring**
   - Progress tracking
   - Quality metrics
   - Resource usage
   - Error detection

3. **Graceful Termination**
   - Clean exit states
   - Informative messages
   - State preservation
   - Result formatting

4. **Resource Optimization**
   - Token usage tracking
   - Performance monitoring
   - Early stopping
   - Efficiency metrics

The chainbreakers can be combined with circular runs' built-in features:
- `max_iterations`
- `circular_condition`
- `circular_transform`
- Progress callbacks

This creates a robust system for controlling and monitoring circular processing flows.

## Best Practices

### 1. Setting Maximum Iterations
- Always set `max_iterations` to prevent infinite loops
- Choose reasonable limits based on your use case
- Consider computational costs and API usage

### 2. Exit Conditions
- Define clear success criteria
- Include multiple exit points when needed
- Consider diminishing returns
- Use history to make informed decisions

### 3. Transform Functions
- Maintain relevant context from previous iterations
- Remove unnecessary information
- Format input consistently
- Consider using summarization for long histories

### 4. Progress Monitoring
- Track improvement metrics
- Log intermediate results
- Monitor resource usage
- Implement early stopping when appropriate

### 5. Error Handling
- Handle edge cases gracefully
- Provide fallback options
- Maintain error state across iterations
- Implement recovery mechanisms

## Common Use Cases

1. **Content Improvement**
   - Iterative text refinement
   - Grammar and style enhancement
   - Translation quality improvement

2. **Problem Solving**
   - Mathematical optimization
   - Algorithm refinement
   - Solution space exploration

3. **Quality Assurance**
   - Code review and improvement
   - Documentation enhancement
   - Test case generation

4. **Creative Tasks**
   - Story development
   - Idea generation
   - Design iteration

## Examples

Check out the following examples in the `examples/` directory:
- `basic_circular_run.py`: Simple improvement loop
- `advanced_circular_processing.py`: History-aware processing
- `multi_exit_chain.py`: Complex exit conditions
- `progress_tracking.py`: Monitoring and logging

## Troubleshooting

### Common Issues

1. **Infinite Loops**
   - Check max_iterations setting
   - Verify exit conditions
   - Monitor transformation logic

2. **Memory Usage**
   - Implement history pruning
   - Use summarization for long histories
   - Monitor resource usage

3. **Performance**
   - Optimize exit conditions
   - Implement early stopping
   - Use efficient transformations

### Debug Tips

1. Enable verbose logging:
```python
chain = PromptChain(
    ...,
    verbose=True,
    log_level="DEBUG"
)
```

2. Use progress callbacks:
```python
def debug_callback(output, history):
    print(f"Step {len(history)}")
    print(f"Output length: {len(output)}")
    print("---")

chain = PromptChain(
    ...,
    on_iteration=debug_callback
)
```

3. Implement validation checks:
```python
def validate_output(output: str) -> bool:
    """Validate output format and content."""
    try:
        # Add validation logic
        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False
``` 