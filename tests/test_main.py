"""
Test suite for PromptChain functionality.
"""

import os
import pytest
from dotenv import load_dotenv
from promptchain.utils.promptchaining import PromptChain

# Load environment variables
load_dotenv()

@pytest.fixture(scope="module")
def setup_environment():
    """Setup test environment and verify API keys."""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        pytest.skip(f"Missing required API keys: {', '.join(missing_keys)}")
    return True

def test_basic_chain_without_techniques(setup_environment):
    """Test that PromptChain works without any techniques."""
    # Single model, single instruction
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Process this: {input}"]
    )
    result = chain.process_prompt("Hello")
    assert result is not None

    # Multiple models, multiple instructions
    chain = PromptChain(
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "First step: {input}",
            "Second step: {input}"
        ],
        full_history=True
    )
    result = chain.process_prompt("Test input")
    assert result is not None

def test_prompt_chain_initialization(setup_environment):
    """Test basic PromptChain initialization."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Test instruction: {input}"]
    )
    assert chain is not None
    assert len(chain.instructions) == 1

def test_technique_validation(setup_environment):
    """Test technique validation rules."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Test: {input}"]
    )
    
    # Test required parameter validation
    with pytest.raises(ValueError):
        chain.add_techniques(["role_playing"])  # Should fail without parameter
    
    # Test valid required parameter
    chain.add_techniques(["role_playing:scientist"])  # Should work
    
    # Test invalid technique
    with pytest.raises(ValueError):
        chain.add_techniques(["invalid_technique"])

def test_technique_combinations(setup_environment):
    """Test combining different techniques."""
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Test: {input}"]
    )
    
    # Test combining different types of techniques
    chain.add_techniques([
        "role_playing:scientist",    # Required parameter
        "few_shot:3",               # Optional parameter
        "step_by_step"              # No parameter
    ])
    
    assert len(chain.instructions) == 1  # Should still have one instruction
