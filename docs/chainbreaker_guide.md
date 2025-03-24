---
noteId: "c5d71d80088411f09aa1cf0704e2b01f"
tags: []

---

# Chainbreaker Guide: From Simple to Complex

Chainbreakers are powerful tools for controlling prompt chain execution by conditionally interrupting the processing flow based on specific criteria. This guide provides a comprehensive overview of chainbreaker techniques, from basic to advanced use cases.

## Table of Contents

1. [Introduction to Chainbreakers](#introduction-to-chainbreakers)
2. [Simple Chainbreakers](#simple-chainbreakers)
3. [Intermediate Chainbreakers](#intermediate-chainbreakers)
4. [Advanced Chainbreakers](#advanced-chainbreakers)
5. [Complex Chainbreakers with DynamicChainBuilder](#complex-chainbreakers-with-dynamicchainbuilder)
6. [Best Practices](#best-practices)

## Introduction to Chainbreakers

Chainbreakers are functions that conditionally interrupt the flow of a prompt chain. Each chainbreaker receives information about the current step and its output, and decides whether to break the chain or let it continue.

### Chainbreaker Function Structure

```python
def chainbreaker_function(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
    """
    Args:
        step: Current step number (1-indexed)
        output: Output text from the current step
        step_info: Optional dictionary with additional step information
        
    Returns:
        Tuple of:
        - should_break (bool): Whether to break the chain
        - break_reason (str): Explanation for breaking
        - modified_output (Any): Optional modified output to return
    """
    # Analyze output and decide whether to break
    if breaking_condition:
        return (True, "Reason for breaking", "Modified output (optional)")
    
    # Continue chain processing
    return (False, "", None)
```

## Simple Chainbreakers

### 1. Break on Keywords

The simplest chainbreaker interrupts the chain when specific keywords are found in the output:

```python
def break_on_keywords(step: int, output: str) -> Tuple[bool, str, Any]:
    """Break the chain if specific keywords are found in the output."""
    keywords = ["error", "invalid", "cannot", "impossible"]
    
    for keyword in keywords:
        if keyword.lower() in output.lower():
            return (
                True,  # Yes, break the chain
                f"Keyword '{keyword}' detected in output",  # Reason
                f"CHAIN INTERRUPTED: The process was stopped because '{keyword}' was detected. " +
                f"Original output: {output}"  # Modified output
            )
    
    # If no keywords found, continue the chain
    return (False, "", None)

# Usage
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Analyze this problem: {input}",
        "Propose solutions for this problem: {input}"
    ],
    chainbreakers=[break_on_keywords]
)

result = chain.process_prompt("How to implement a perpetual motion machine?")
```

### 2. Break After Specific Steps

Limit processing to a specific number of steps:

```python
def break_after_steps(max_steps: int):
    """Create a chainbreaker that stops after a specific number of steps."""
    def _breaker(step: int, output: str) -> Tuple[bool, str, Any]:
        if step >= max_steps:
            return (
                True,  # Yes, break the chain
                f"Maximum step count reached ({max_steps})",  # Reason
                output  # Keep the original output
            )
        return (False, "", None)
    
    return _breaker

# Usage
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Step 1: Outline a story about: {input}",
        "Step 2: Develop the characters: {input}",
        "Step 3: Write the first chapter: {input}",
        "Step 4: Write the second chapter: {input}"
    ],
    chainbreakers=[break_after_steps(2)]  # Only run first 2 steps
)
```

### 3. Break on Output Length

Stop processing if output is too short or too long:

```python
def break_on_length(min_length: int = None, max_length: int = None):
    """Create a chainbreaker that stops if output is too short or too long."""
    def _breaker(step: int, output: str) -> Tuple[bool, str, Any]:
        output_length = len(output)
        
        if min_length and output_length < min_length:
            return (
                True,
                f"Output too short: {output_length} chars (min: {min_length})",
                f"CHAIN INTERRUPTED: The output was too short ({output_length} chars). " +
                f"Please provide more detailed information.\n\n{output}"
            )
            
        if max_length and output_length > max_length:
            # Truncate the output and add a note
            truncated = output[:max_length] + "..."
            return (
                True,
                f"Output too long: {output_length} chars (max: {max_length})",
                f"CHAIN INTERRUPTED: The output was truncated due to length.\n\n{truncated}"
            )
            
        return (False, "", None)
    
    return _breaker
```

## Intermediate Chainbreakers

### 1. Break on Low Confidence

Interrupt the chain if the model expresses low confidence:

```python
def break_on_low_confidence(step: int, output: str) -> Tuple[bool, str, Any]:
    """Break the chain if confidence level is below threshold."""
    # Look for confidence pattern like "Confidence: 75%" or "confidence level: 0.32"
    confidence_patterns = [
        r"confidence:?\s*(\d+)%",
        r"confidence:?\s*level:?\s*(\d+\.?\d*)",
        r"confidence:?\s*(\d+\.?\d*)"
    ]
    
    for pattern in confidence_patterns:
        matches = re.findall(pattern, output.lower())
        if matches:
            confidence = float(matches[0])
            # Normalize to 0-1 scale if it's a percentage
            if confidence > 1 and confidence <= 100:
                confidence /= 100
                
            if confidence < 0.7:  # 70% confidence threshold
                return (
                    True,
                    f"Low confidence detected: {confidence:.2%}",
                    f"CHAIN INTERRUPTED: Process stopped due to low confidence ({confidence:.2%}). " +
                    f"Please verify results manually."
                )
    
    return (False, "", None)
```

### 2. Break on Quality Issues

Detect repetitive or uncertain content:

```python
def break_on_quality_issues(step: int, output: str) -> Tuple[bool, str, Any]:
    """Break the chain if quality issues are detected."""
    # Check for repetition
    paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3:
        # Check if consecutive paragraphs are too similar
        for i in range(len(paragraphs) - 1):
            similarity = len(set(paragraphs[i].split()) & set(paragraphs[i+1].split())) / len(set(paragraphs[i].split()))
            if similarity > 0.7:  # 70% similarity threshold
                return (
                    True,
                    f"Repetitive content detected (similarity: {similarity:.2%})",
                    f"CHAIN INTERRUPTED: Repetitive content detected. Please review and refine.\n\n{output}"
                )
    
    # Check for hallucination indicators
    hallucination_phrases = [
        "I'm not sure", "I don't know", "I'm uncertain", 
        "I can't verify", "I can't confirm", "might be", "could be"
    ]
    
    hallucination_count = sum(1 for phrase in hallucination_phrases if phrase.lower() in output.lower())
    if hallucination_count >= 3:
        return (
            True,
            f"Potential hallucination detected ({hallucination_count} indicators)",
            f"CHAIN INTERRUPTED: Multiple uncertainty indicators detected. " +
            f"The information may not be reliable.\n\n{output}"
        )
    
    return (False, "", None)
```

### 3. Domain-Specific Validation

Create specialized validators for specific domains:

```python
def financial_data_validator(step: int, output: str) -> Tuple[bool, str, Any]:
    """Break the chain if financial data appears invalid."""
    # Check for negative revenue or profit margins over 100%
    revenue_pattern = r"revenue:?\s*\$?(-?\d+\.?\d*)"
    profit_pattern = r"profit margin:?\s*(\d+\.?\d*)%"
    
    revenue_matches = re.findall(revenue_pattern, output.lower())
    profit_matches = re.findall(profit_pattern, output.lower())
    
    if revenue_matches and float(revenue_matches[0]) < 0:
        return (
            True,
            "Negative revenue detected",
            f"CHAIN INTERRUPTED: Invalid financial data - negative revenue detected. " +
            f"Please verify the calculations.\n\n{output}"
        )
    
    if profit_matches and float(profit_matches[0]) > 100:
        return (
            True,
            f"Unrealistic profit margin: {profit_matches[0]}%",
            f"CHAIN INTERRUPTED: Invalid financial data - profit margin exceeds 100%. " +
            f"Please verify the calculations.\n\n{output}"
        )
    
    return (False, "", None)
```

## Advanced Chainbreakers

### 1. Function-Specific Chainbreakers

Break the chain on specific functions or function outputs:

```python
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
                    True,
                    f"Function '{function_name}' encountered at step {step}",
                    f"CHAIN INTERRUPTED: Processing stopped at function '{function_name}'.\n\n{output}"
                )
        
        return (False, "", None)
    
    return _breaker

def break_on_function_condition(condition_func: Callable[[str, dict], bool]):
    """
    Create a chainbreaker that stops when a function output meets a specific condition.
    
    Args:
        condition_func: A function that takes (output, step_info) and returns True if chain should break
    """
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # Only check function steps
        if step_info and step_info.get('type') == 'function':
            # Check if the condition is met
            if condition_func(output, step_info):
                return (
                    True,
                    f"Function condition met at step {step}",
                    f"CHAIN INTERRUPTED: Function output met breaking condition.\n\n{output}"
                )
        
        return (False, "", None)
    
    return _breaker

# Usage example
def negative_sentiment_condition(output: str, step_info: dict) -> bool:
    return "negative sentiment" in output.lower()

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Write a review of: {input}",
        analyze_sentiment,  # Function that might trigger the break
        "Provide recommendations based on the review: {input}"
    ],
    chainbreakers=[break_on_function_condition(negative_sentiment_condition)]
)
```

### 2. Multiple Chainbreakers

Combine multiple breakers to create sophisticated interruption logic:

```python
chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=[
        "Analyze this financial data: {input}",
        "Make predictions based on the analysis: {input}",
        "Recommend investment strategies: {input}"
    ],
    chainbreakers=[
        financial_data_validator,  # Domain-specific validation
        break_on_keywords,         # Check for problematic keywords
        break_on_quality_issues,   # Quality control
        break_after_steps(5)       # Failsafe step limit
    ]
)
```

### 3. Stateful Chainbreakers

Create chainbreakers that maintain state between steps:

```python
def create_trend_detector(threshold: float = 0.2):
    """Create a chainbreaker that detects significant changes in output length."""
    history = []
    
    def _breaker(step: int, output: str) -> Tuple[bool, str, Any]:
        # Store output length
        current_length = len(output)
        history.append(current_length)
        
        # Need at least 3 steps to detect a trend
        if len(history) < 3:
            return (False, "", None)
        
        # Calculate growth rates
        growth_rates = [(history[i] - history[i-1]) / max(1, history[i-1]) 
                      for i in range(1, len(history))]
        
        # Check for consistent growth or decline
        if all(rate > threshold for rate in growth_rates[-2:]):
            return (
                True,
                f"Consistent output growth detected: {growth_rates[-2:]}", 
                f"CHAIN INTERRUPTED: The output is growing significantly each step, " +
                f"which may indicate a runaway expansion pattern.\n\n{output}"
            )
            
        if all(rate < -threshold for rate in growth_rates[-2:]):
            return (
                True,
                f"Consistent output shrinkage detected: {growth_rates[-2:]}", 
                f"CHAIN INTERRUPTED: The output is shrinking significantly each step, " +
                f"which may indicate a convergence problem.\n\n{output}"
            )
                
        return (False, "", None)
    
    return _breaker
```

## Complex Chainbreakers with DynamicChainBuilder

### 1. Global Chain Monitoring

Monitor the entire chain ecosystem to make breaking decisions:

```python
def create_global_chain_monitor(builder: DynamicChainBuilder, max_running_chains: int = 5):
    """
    Create a chainbreaker that monitors the entire system and breaks chains 
    if too many are running simultaneously.
    """
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # Count running chains
        running_chains = sum(1 for info in builder.chain_registry.values() 
                           if info["status"] == "running")
        
        if running_chains > max_running_chains:
            return (
                True,
                f"Too many chains running simultaneously ({running_chains})",
                f"CHAIN INTERRUPTED: System load exceeded threshold with {running_chains} " + 
                f"chains running simultaneously.\n\n{output}"
            )
                
        return (False, "", None)
    
    return _breaker

# Usage
builder = DynamicChainBuilder(
    base_model="openai/gpt-4",
    base_instruction="Process the task: {instruction} with this input: {input}"
)

# Create a chainbreaker that monitors the whole system
system_monitor = create_global_chain_monitor(builder, max_running_chains=3)

# Add to all chains as they're created
chain1 = builder.create_chain(
    "chain1",
    ["Step 1: {input}", "Step 2: {input}"],
    chainbreakers=[system_monitor]  # Add the system monitor
)
```

### 2. Memory-Based Chainbreaker

Use the builder's memory bank to make decisions:

```python
def create_memory_aware_breaker(builder: DynamicChainBuilder, 
                               namespace: str = "errors",
                               max_errors: int = 3):
    """
    Create a chainbreaker that stops processing if too many errors
    have been recorded in the memory bank.
    """
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # Get error count from memory
        error_count = builder.retrieve_memory("error_count", namespace, 0)
        
        # Check for error indicators in the current output
        error_indicators = ["error", "exception", "failed", "invalid"]
        
        if any(indicator in output.lower() for indicator in error_indicators):
            # Increment the error count
            error_count += 1
            builder.store_memory("error_count", error_count, namespace)
            
            # Also store the specific error for review
            builder.store_memory(f"error_{error_count}", output[:100], namespace)
            
            # Check if we've exceeded the threshold
            if error_count >= max_errors:
                error_log = "\n".join([
                    f"Error {i+1}: {builder.retrieve_memory(f'error_{i+1}', namespace, '')}"
                    for i in range(error_count)
                ])
                
                return (
                    True,
                    f"Error threshold exceeded ({error_count} errors)",
                    f"CHAIN INTERRUPTED: Too many errors encountered ({error_count}).\n\n" +
                    f"Error Log:\n{error_log}\n\nLast Output:\n{output}"
                )
                
        return (False, "", None)
    
    return _breaker
```

### 3. Adaptive Chainbreaker with Dynamic Thresholds

Adapt thresholds based on chain performance:

```python
def create_adaptive_quality_breaker(builder: DynamicChainBuilder):
    """
    Create a chainbreaker with adaptive thresholds that learn from successful chains.
    """
    # Default quality thresholds
    thresholds = {
        "min_length": 100,
        "max_hallucination_indicators": 3,
        "max_repetition_similarity": 0.7
    }
    
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # Check output length
        if len(output) < thresholds["min_length"]:
            return (
                True,
                f"Output too short: {len(output)} chars (min: {thresholds['min_length']})",
                f"CHAIN INTERRUPTED: Output below minimum length threshold.\n\n{output}"
            )
        
        # Check for hallucination indicators
        hallucination_phrases = [
            "I'm not sure", "I don't know", "I'm uncertain", 
            "I can't verify", "I can't confirm", "might be", "could be"
        ]
        
        hallucination_count = sum(1 for phrase in hallucination_phrases 
                                if phrase.lower() in output.lower())
        
        if hallucination_count > thresholds["max_hallucination_indicators"]:
            return (
                True,
                f"Hallucination threshold exceeded: {hallucination_count} indicators",
                f"CHAIN INTERRUPTED: Too many uncertainty indicators.\n\n{output}"
            )
        
        # Check for repetition
        paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            for i in range(len(paragraphs) - 1):
                similarity = len(set(paragraphs[i].split()) & set(paragraphs[i+1].split())) / len(set(paragraphs[i].split()))
                if similarity > thresholds["max_repetition_similarity"]:
                    return (
                        True,
                        f"Repetition threshold exceeded: {similarity:.2%} similarity",
                        f"CHAIN INTERRUPTED: Content too repetitive.\n\n{output}"
                    )
        
        # If this is the last step in a successful chain, update thresholds
        if step_info and step_info.get("step") == len(builder.chain_registry[step_info.get("chain_id", "")]["chain"].instructions):
            # Decrease minimum length if current output is good but shorter
            if 50 < len(output) < thresholds["min_length"]:
                thresholds["min_length"] = max(50, int(thresholds["min_length"] * 0.9))
            
            # Increase hallucination tolerance if chain completed with some indicators
            if 0 < hallucination_count < thresholds["max_hallucination_indicators"]:
                thresholds["max_hallucination_indicators"] += 1
        
        return (False, "", None)
    
    return _breaker
```

### 4. Chain Injection Control with Chainbreakers

Control dynamic chain injection based on output analysis:

```python
def create_injection_controller(builder: DynamicChainBuilder):
    """
    Create a chainbreaker that controls whether chain injection should proceed
    based on an analysis of the current output.
    """
    def _breaker(step: int, output: str, step_info: dict = None) -> Tuple[bool, str, Any]:
        # If this is an injection point (marked by a special tag)
        if "[[INJECTION_POINT]]" in output:
            # Extract the injection parameters
            injection_pattern = r"\[\[INJECTION_POINT:([^\]]+)\]\]"
            matches = re.findall(injection_pattern, output)
            
            if matches:
                injection_params = matches[0].split(',')
                target_chain_id = step_info.get("chain_id", "")
                
                # Check if injection should be prevented
                prevent_reasons = []
                
                # Check for circular dependencies
                for param in injection_params:
                    if ":" in param:
                        key, value = param.split(":", 1)
                        if key.strip() == "source_chain" and value.strip() in builder.chain_registry:
                            # Check if this would create a circular dependency
                            if not builder.validate_injection(target_chain_id, value.strip(), step):
                                prevent_reasons.append(f"Would create circular dependency with {value.strip()}")
                
                # If there are reasons to prevent injection
                if prevent_reasons:
                    return (
                        True,
                        f"Injection prevented: {'; '.join(prevent_reasons)}",
                        f"CHAIN INTERRUPTED: Dynamic injection prevented for safety reasons:\n" +
                        f"- {'\n- '.join(prevent_reasons)}\n\nPlease review the chain design.\n\n{output}"
                    )
                
                # Remove the injection point markers for cleaner output
                cleaned_output = re.sub(r"\[\[INJECTION_POINT:[^\]]+\]\]", "", output)
                return (False, "", cleaned_output)
                
        return (False, "", None)
    
    return _breaker
```

## Best Practices

1. **Start Simple**: Begin with basic chainbreakers and add complexity as needed.

2. **Combine Strategically**: Use multiple chainbreakers that check different aspects.

3. **Provide Helpful Messages**: When breaking a chain, include clear explanations and suggestions.

4. **Handle Edge Cases**: Consider potential false positives and edge cases in your conditions.

5. **Test Thoroughly**: Ensure your chainbreakers work as expected with various inputs.

6. **Monitor Performance**: Track how often chains are broken and why to refine your approach.

7. **Consider Chain-Specific Breakers**: Different chains may need different breaking conditions.

8. **Use Memory for Context**: Leverage the memory bank to make more informed breaking decisions.

9. **Balance Control and Flexibility**: Don't make breaking conditions too strict or too lenient.

10. **Document Breaking Conditions**: Make sure users understand why chains might be interrupted.

By applying these chainbreaker techniques, you can create more robust and reliable prompt chains that handle edge cases gracefully and maintain quality control throughout the execution process. 