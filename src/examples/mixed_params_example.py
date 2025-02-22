from src.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import os

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_path)

def mixed_params_example():
    """Example showing both simple string models and detailed model configs"""
    
    # Create chain with mixed parameter styles
    analysis_chain = PromptChain(
        models=[
            "openai/gpt-4",  # Simple string model name
            {  # Detailed model config
                "name": "anthropic/claude-3-sonnet-20240229",
                "params": {
                    "temperature": 0.3,
                    "max_tokens": 200
                }
            },
            "openai/gpt-4"  # Another simple string model name
        ],
        instructions=[
            "Write initial analysis of: {input}",
            "Enhance with specific details: {input}",
            "Create final summary: {input}"
        ],
        full_history=True
    )

    return analysis_chain.process_prompt("Future of remote work")

def run_mixed_example():
    """Run the mixed parameters example"""
    load_env()
    results = mixed_params_example()
    
    print("\n=== Mixed Model Parameters Results ===")
    for step in results:
        print(f"\nStep {step['step']}:")
        print(f"Type: {step['type']}")
        if step['type'] == 'model':
            if step.get('model_params'):
                print("Using custom parameters:")
                print(f"Parameters: {step['model_params']}")
            else:
                print("Using default parameters")
        print(f"Output: {step['output']}")

if __name__ == "__main__":
    run_mixed_example() 