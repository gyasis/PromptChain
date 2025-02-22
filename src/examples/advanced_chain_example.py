from src.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import os
import json

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_path)

def format_as_json(text):
    """Custom processor that formats text as JSON structure"""
    try:
        # Add some structure to the text
        formatted = {
            "original_text": text,
            "word_count": len(text.split()),
            "processed_timestamp": "2024-03-20",  # You could use real timestamp
            "metadata": {
                "type": "story_element",
                "version": "1.0"
            }
        }
        return json.dumps(formatted, indent=2)
    except Exception as e:
        print(f"Error in JSON formatting: {e}")
        return text

def extract_from_json(json_text):
    """Custom processor that extracts original text from JSON"""
    try:
        data = json.loads(json_text)
        return data.get("original_text", json_text)
    except Exception as e:
        print(f"Error extracting from JSON: {e}")
        return json_text

def advanced_story_generation():
    """Example: Story Generation with intermediate processing"""
    
    # First chain: Generate initial story elements
    initial_chain = PromptChain(
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "Create a list of story elements (characters, setting, conflict) for a story about: {input}",
            "Expand these elements with detailed descriptions: {input}"
        ],
        full_history=False
    )

    # Second chain: Convert elements into story
    story_chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
        instructions=[
            "Write a first draft of a story using these structured elements: {input}",
            "Polish and enhance the story, maintaining all key elements: {input}"
        ],
        full_history=False
    )

    # Initial topic
    initial_topic = "a time-traveling chef who discovers ancient recipes"

    try:
        # Step 1: Generate initial story elements
        print("\n=== Generating Story Elements ===")
        elements = initial_chain.process_prompt(initial_topic)
        print(f"Initial elements:\n{elements}\n")

        # Step 2: Custom Processing - Format as JSON
        print("\n=== Custom Processing: Converting to JSON ===")
        json_elements = format_as_json(elements)
        print(f"Formatted as JSON:\n{json_elements}\n")

        # Step 3: Custom Processing - Extract from JSON
        print("\n=== Custom Processing: Extracting from JSON ===")
        processed_elements = extract_from_json(json_elements)
        print(f"Extracted elements:\n{processed_elements}\n")

        # Step 4: Generate final story
        print("\n=== Generating Final Story ===")
        final_story = story_chain.process_prompt(processed_elements)
        print(f"Final story:\n{final_story}")

        return {
            "initial_elements": elements,
            "processed_json": json_elements,
            "final_story": final_story
        }

    except Exception as e:
        print(f"Error in advanced story generation: {str(e)}")
        return None

def run_advanced_example():
    """Run the advanced chain example"""
    load_env()
    result = advanced_story_generation()
    
    if result:
        print("\n=== Advanced Chain Results ===")
        print("\nInitial Elements:")
        print(result["initial_elements"])
        print("\nProcessed JSON:")
        print(result["processed_json"])
        print("\nFinal Story:")
        print(result["final_story"])

if __name__ == "__main__":
    run_advanced_example() 