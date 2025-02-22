from src.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import os

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_path)

def custom_model_example():
    """Example showing custom model parameters for different LLMs"""
    
    # OpenAI specific parameters
    openai_params = {
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 1.0,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "stop": ["###"],  # Custom stop sequence
        "response_format": {"type": "json_object"}  # Force JSON response
    }

    # Anthropic specific parameters
    anthropic_params = {
        "temperature": 0.3,
        "max_tokens": 200,
        "top_k": 40,
        "top_p": 0.9,
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456"
        }
    }

    # Create chain with custom parameters
    analysis_chain = PromptChain(
        models=[
            {
                "name": "openai/gpt-4",
                "params": openai_params
            },
            {
                "name": "anthropic/claude-3-sonnet-20240229",
                "params": anthropic_params
            }
        ],
        instructions=[
            "Generate a detailed JSON analysis of: {input}",
            "Improve and structure the analysis: {input}"
        ],
        full_history=True
    )

    return analysis_chain.process_prompt("Impact of AI on healthcare")

def run_model_params_example():
    """Run the model parameters example"""
    load_env()
    results = custom_model_example()
    
    print("\n=== Custom Model Parameters Results ===")
    for step in results:
        print(f"\nStep {step['step']}:")
        print(f"Type: {step['type']}")
        print(f"Model Parameters Used:")
        if step['type'] == 'model':
            print(f"Temperature: {step.get('model_params', {}).get('temperature')}")
            print(f"Max Tokens: {step.get('model_params', {}).get('max_tokens')}")
        print(f"Output: {step['output']}")

if __name__ == "__main__":
    run_model_params_example() 