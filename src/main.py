from utils.promptchaining import PromptChain
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), "../.env")
    load_dotenv(env_path)

    # First chain - generates a story outline
    outline_chain = PromptChain(
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        num_refinements=2,
        instructions=[
            "Create a brief outline for a story about: {input}",
            "Add character details to the outline: {input}"
        ],
        full_history=False  # We only want the final output
    )

    # Second chain - uses the outline to write the story
    story_chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
        num_refinements=2,
        instructions=[
            "Write a first draft of a story based on this outline: {input}",
            "Polish and enhance the story: {input}"
        ],
        full_history=False  # Again, only final output needed
    )

    # Initial topic
    initial_topic = "a robot learning to paint"

    try:
        # Process the first chain to get the outline
        print("\n=== Creating Story Outline ===")
        outline_result = outline_chain.process_prompt(initial_topic)
        print(f"Outline created:\n{outline_result}\n")

        # Use the outline result as input for the story chain
        print("\n=== Writing Story from Outline ===")
        final_story = story_chain.process_prompt(outline_result)
        print(f"Final story:\n{final_story}")

    except Exception as e:
        print(f"Error in chain processing: {str(e)}")

if __name__ == "__main__":
    main()
