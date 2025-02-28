"""
Examples of using PromptChain with different prompt engineering techniques.
This file demonstrates various combinations and use cases of the techniques.
"""

import os
from dotenv import load_dotenv
from promptchain.utils.promptchaining import PromptChain

# Load environment variables from .env file
load_dotenv()

# Verify required API keys are present
required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}. "
                    f"Please add them to your .env file.")

def test_basic_usage_without_techniques():
    """Demonstrate basic usage without any techniques."""
    # Single instruction chain
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Explain this topic: {input}"]
    )
    
    result = chain.process_prompt("How do computers work?")
    print("\n=== Basic Single Instruction ===")
    print(result)
    
    # Multiple instruction chain
    chain = PromptChain(
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "Analyze this topic: {input}",
            "Summarize the analysis: {input}"
        ],
        full_history=True
    )
    
    result = chain.process_prompt("The impact of social media")
    print("\n=== Basic Multiple Instructions ===")
    print(result)

def test_required_parameter_techniques():
    """Test techniques that require parameters."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Analyze the following topic: {input}"]
    )
    
    # Using role-playing and style-mimicking
    chain.add_techniques([
        "role_playing:quantum physicist",
        "style_mimicking:Richard Feynman"
    ])
    
    result = chain.process_prompt("Explain quantum entanglement")
    print("\n=== Expert Physics Explanation ===")
    print(result)
    
    # Using persona emulation and forbidden words
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Provide investment advice about: {input}"]
    )
    
    chain.add_techniques([
        "persona_emulation:Warren Buffett",
        "forbidden_words:maybe,possibly,perhaps,might"
    ])
    
    result = chain.process_prompt("Long-term investment strategies")
    print("\n=== Investment Advice ===")
    print(result)

def test_optional_parameter_techniques():
    """Test techniques with optional parameters."""
    chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229"],
        instructions=["Explain this concept: {input}"]
    )
    
    # Using few-shot and context expansion
    chain.add_techniques([
        "few_shot:3",                     # Provide 3 examples
        "context_expansion:historical",    # Add historical context
        "reverse_prompting:5"             # Generate 5 questions
    ])
    
    result = chain.process_prompt("The evolution of artificial intelligence")
    print("\n=== Concept Explanation with Examples ===")
    print(result)
    
    # Using tree of thought and comparative answering
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Compare and analyze: {input}"]
    )
    
    chain.add_techniques([
        "tree_of_thought:3",           # Explore 3 paths
        "comparative_answering:4"      # Compare 4 aspects
    ])
    
    result = chain.process_prompt("Different machine learning frameworks")
    print("\n=== Comparative Analysis ===")
    print(result)

def test_no_parameter_techniques():
    """Test techniques that don't use parameters."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Solve this problem: {input}"]
    )
    
    # Using step-by-step and chain of thought
    chain.add_techniques([
        "step_by_step",
        "chain_of_thought",
        "react"
    ])
    
    result = chain.process_prompt("How to optimize a sorting algorithm")
    print("\n=== Step-by-Step Problem Solving ===")
    print(result)
    
    # Using iterative refinement and contrarian perspective
    chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229"],
        instructions=["Analyze this argument: {input}"]
    )
    
    chain.add_techniques([
        "iterative_refinement",
        "contrarian_perspective"
    ])
    
    result = chain.process_prompt("Social media's impact on society")
    print("\n=== Contrarian Analysis ===")
    print(result)

def test_combined_techniques():
    """Test combining different types of techniques."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Research and analyze: {input}",
            "Provide recommendations based on: {input}"
        ],
        full_history=True
    )
    
    # Combining various techniques
    chain.add_techniques([
        # Required parameter techniques
        "role_playing:research scientist",
        "style_mimicking:Academic Journal",
        
        # Optional parameter techniques
        "few_shot:2",
        "tree_of_thought:3",
        
        # No parameter techniques
        "step_by_step",
        "react"
    ])
    
    result = chain.process_prompt("The future of renewable energy storage")
    print("\n=== Comprehensive Analysis with Multiple Techniques ===")
    print(result)

def test_error_cases():
    """Test error cases and validation."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Analyze: {input}"]
    )
    
    try:
        # Should raise error - missing required parameter
        chain.add_techniques(["role_playing"])
    except ValueError as e:
        print("\n=== Expected Error (Missing Parameter) ===")
        print(e)
    
    try:
        # Should raise error - unknown technique
        chain.add_techniques(["unknown_technique"])
    except ValueError as e:
        print("\n=== Expected Error (Unknown Technique) ===")
        print(e)
    
    # Should show warning - parameter provided for no-parameter technique
    chain.add_techniques(["step_by_step:detailed"])

if __name__ == "__main__":
    print("Testing PromptChain Techniques...")
    
    print("\nTesting Basic Usage Without Techniques:")
    test_basic_usage_without_techniques()
    
    print("\nTesting Required Parameter Techniques:")
    test_required_parameter_techniques()
    
    print("\nTesting Optional Parameter Techniques:")
    test_optional_parameter_techniques()
    
    print("\nTesting No Parameter Techniques:")
    test_no_parameter_techniques()
    
    print("\nTesting Combined Techniques:")
    test_combined_techniques()
    
    print("\nTesting Error Cases:")
    test_error_cases()