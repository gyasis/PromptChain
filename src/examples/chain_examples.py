from src.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import os

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    load_dotenv(env_path)

def story_generation_example():
    """Example 1: Story Generation Chain"""
    outline_chain = PromptChain(
        models=["openai/gpt-4", "anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "Create a brief outline for a story about: {input}",
            "Add character details to the outline: {input}"
        ],
        full_history=False
    )

    story_chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
        instructions=[
            "Write a first draft of a story based on this outline: {input}",
            "Polish and enhance the story: {input}"
        ],
        full_history=False
    )

    initial_topic = "a robot learning to paint"
    outline = outline_chain.process_prompt(initial_topic)
    final_story = story_chain.process_prompt(outline)
    return final_story

def code_review_example():
    """Example 2: Code Review Chain"""
    analysis_chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Analyze this code for bugs and potential improvements: {input}"
        ],
        full_history=False
    )

    documentation_chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "Generate documentation based on this code analysis: {input}"
        ],
        full_history=False
    )

    test_chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Create unit test cases based on this documentation: {input}"
        ],
        full_history=False
    )

    sample_code = """
    def fibonacci(n):
        if n <= 0: return []
        if n == 1: return [0]
        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        return sequence
    """

    analysis = analysis_chain.process_prompt(sample_code)
    docs = documentation_chain.process_prompt(analysis)
    tests = test_chain.process_prompt(docs)
    return {"analysis": analysis, "documentation": docs, "tests": tests}

def translation_example():
    """Example 3: Translation and Localization Chain"""
    translation_chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Translate this English text to Spanish: {input}"
        ],
        full_history=False
    )

    localization_chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229"],
        instructions=[
            "Adapt this Spanish translation for Mexican audience, considering cultural context: {input}"
        ],
        full_history=False
    )

    english_text = "Welcome to our product launch! We're excited to share our innovative solution."
    spanish = translation_chain.process_prompt(english_text)
    localized = localization_chain.process_prompt(spanish)
    return {"translation": spanish, "localized": localized}

def research_analysis_example():
    """Example 4: Research Paper Analysis Chain with History"""
    analysis_chain = PromptChain(
        models=["anthropic/claude-3-sonnet-20240229", "openai/gpt-4"],
        instructions=[
            "Extract key findings and methodology from this research paper: {input}",
            "Analyze the limitations and potential improvements: {input}"
        ],
        full_history=True  # Keep history to build upon previous analysis
    )

    summary_chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Create an executive summary based on this analysis: {input}"
        ],
        full_history=False
    )

    paper_abstract = """
    This study examines the impact of artificial intelligence on workplace productivity...
    """

    analysis_results = analysis_chain.process_prompt(paper_abstract)
    final_summary = summary_chain.process_prompt(analysis_results[-1]['output'])
    return {"analysis": analysis_results, "summary": final_summary}

def run_all_examples():
    """Run all example chains and display results"""
    load_env()

    print("\n=== Story Generation Example ===")
    story = story_generation_example()
    print(f"Final Story:\n{story}\n")

    print("\n=== Code Review Example ===")
    code_review = code_review_example()
    print(f"Analysis:\n{code_review['analysis']}\n")
    print(f"Documentation:\n{code_review['documentation']}\n")
    print(f"Tests:\n{code_review['tests']}\n")

    print("\n=== Translation Example ===")
    translation = translation_example()
    print(f"Spanish Translation:\n{translation['translation']}\n")
    print(f"Localized Version:\n{translation['localized']}\n")

    print("\n=== Research Analysis Example ===")
    research = research_analysis_example()
    print("Analysis Steps:")
    for step in research['analysis']:
        print(f"\nStep {step['step']}:\n{step['output']}")
    print(f"\nFinal Summary:\n{research['summary']}")

if __name__ == "__main__":
    run_all_examples() 