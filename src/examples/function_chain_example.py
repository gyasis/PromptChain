from src.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import os
import json
from typing import Dict, Any

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_path)

def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis function"""
    # This is a simple example - you could use a real sentiment analyzer
    positive_words = ['good', 'great', 'excellent', 'happy', 'positive']
    negative_words = ['bad', 'poor', 'negative', 'sad', 'unfortunate']
    
    words = text.lower().split()
    sentiment_score = sum(1 for word in words if word in positive_words)
    sentiment_score -= sum(1 for word in words if word in negative_words)
    
    return f"Sentiment Analysis Result: {sentiment_score} (Original text: {text})"

def add_metadata(text: str) -> str:
    """Add metadata to the text"""
    metadata = {
        "word_count": len(text.split()),
        "character_count": len(text),
        "content": text
    }
    return json.dumps(metadata, indent=2)

def function_chain_example():
    """Example using functions in the instruction chain"""
    
    chain = PromptChain(
        # Only need 2 models for the 2 non-function instructions
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "Write a short negative opnion piece politically charged about health implications of : {input}",  # Uses first model
            analyze_sentiment,                      # Function (no model needed)
            add_metadata,                          # Function (no model needed)
            "Summarize all the above analysis about: {input}"  # Uses second model
        ],
        full_history=True
    )

    initial_topic = "The latest smartphone technology"
    return chain.process_prompt(initial_topic)

def run_function_example():
    """Run the function chain example"""
    load_env()
    results = function_chain_example()
    
    print("\n=== Function Chain Results ===")
    for step in results:
        print(f"\nStep {step['step']}:")
        print(f"Type: {step['type']}")
        print(f"Output: {step['output']}")

if __name__ == "__main__":
    run_function_example() 