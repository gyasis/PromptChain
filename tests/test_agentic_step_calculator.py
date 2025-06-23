import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Set up more verbose logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Make sure LiteLLM and PromptChain loggers are at DEBUG level
logging.getLogger('litellm').setLevel(logging.DEBUG)
logging.getLogger('promptchain').setLevel(logging.DEBUG)

def simple_calculator(expression: str) -> str:
    """
    Evaluates a simple mathematical expression securely.
    
    Args:
        expression: A math expression to evaluate, e.g., "2 + 2", "7 * 8"
    
    Returns:
        Result of the calculation as a string
    """
    try:
        # Create a safe evaluation environment with no builtins
        allowed_names = {"abs": abs, "pow": pow, "round": round, "int": int, "float": float}
        
        # Since eval is used here, we need to be careful and limit the scope
        result = eval(expression, {"__builtins__": None}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"

async def test_agentic_calculator():
    """Test the calculator tool with AgenticStepProcessor"""
    print("\n\n=== RUNNING TEST WITH AGENTIC STEP PROCESSOR ===")
    
    # Define a simple calculator tool schema
    calculator_schema = {
        "type": "function",
        "function": {
            "name": "simple_calculator",
            "description": "Evaluates a simple mathematical expression (e.g., '2 + 2', '5 * 8').",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The mathematical expression to evaluate."}
                },
                "required": ["expression"]
            }
        }
    }

    # Create the PromptChain with an agentic step
    agentic_objective = (
        "You are a math assistant. If the user asks a math question, use the calculator tool to compute the answer. "
        "Always show your reasoning and the final answer."
    )
    
    chain = PromptChain(
        models=[],  # No models since AgenticStepProcessor handles its own model
        instructions=[
            AgenticStepProcessor(
                objective=agentic_objective, 
                max_internal_steps=3,
                model_name="openai/gpt-4o",  # Set the model here
                model_params={
                    # Use auto tool choice to let the model decide
                    "tool_choice": "auto"
                }
            )
        ],
        verbose=True
    )
    
    # Add tools and register function
    chain.add_tools([calculator_schema])
    chain.register_tool_function(simple_calculator)

    try:
        # Run the chain with a math question
        math_question = "What is 7 * 8?"
        output = await chain.process_prompt_async(math_question)
        print("\nAgentic Step Output:\n", output)
        
        # Check if we got a successful answer
        if "56" in output:
            print("\nTEST PASSED: Correct result received!")
        else:
            print("\nTEST FAILED: Did not find expected answer (56) in output")
            
    except Exception as e:
        print(f"Error running the chain: {e}")
        raise

async def main():
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        return
    
    # Run the test
    await test_agentic_calculator()

if __name__ == "__main__":
    asyncio.run(main()) 