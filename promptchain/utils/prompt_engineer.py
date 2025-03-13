"""A module for iterative prompt engineering with meta-evaluation."""

from typing import List, Dict, Union, Callable, Optional
from promptchain.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

def display_final_prompt(prompt: str):
    """Display the final prompt in a rich formatted panel."""
    # Clear the console for clean output
    console.clear()
    # Create a markdown object from the prompt
    md = Markdown(prompt)
    # Display in a nice panel
    console.print(Panel(md, title="[bold green]Final Prompt[/bold green]", 
                       border_style="green",
                       padding=(1, 2)))
    # Return the prompt instead of exiting
    return prompt

class PromptEngineer:
    # Generic prompt for continuous improvement
    GENERIC_IMPROVEMENT_PROMPT = """│                                                                                                                                                                                                                                   │
│  Your task is to evaluate and improve given prompt by assessing its clarity and effectiveness. Follow these improved steps:                                                                                                                 │
│                                                                                                                                                                                                                                   │
│   1 Define the Objective: Clearly state what the prompt aims to accomplish.                                                                                                                                                       │
│   2 Assess Positives: Identify what the prompt does well, such as clarity and engagement.                                                                                                                                         │
│   3 Identify Areas for Improvement: Note where the prompt lacks clarity or effectiveness.                                                                                                                                         │
│   4 Consider Scenarios for Use: List situations where the prompt would excel and where it may face challenges.                                                                                                                    │
│   5 Enhance the Prompt: Rewrite it to maintain its strengths and address identified weaknesses.                                                                                                                                   │
│   6 Conclude Your Analysis:                                                                                                                                                                                                       │
│      • Summarize the initial prompt briefly.                                                                                                                                                                                      │
│      • Outline positives and areas needing enhancement.                                                                                                                                                                           │
│      • Discuss suitable scenarios and limitations.                                                                                                                                                                                │
│      • Provide the refined prompt with an explanation of its improvements.                                                                                                                                                        │
│     

    This is the prompt to modify and improve:                                                                                                                                                                                                                               │
│     """

    def __init__(self, 
                 max_iterations: int = 3,
                 use_human_evaluation: bool = False,
                 verbose: bool = False):
        """
        Initialize the PromptEngineer.
        
        Args:
            max_iterations: Maximum number of improvement iterations
            use_human_evaluation: Whether to use human evaluation instead of LLM
            verbose: Whether to print detailed output
        """
        self.max_iterations = max_iterations
        self.use_human_evaluation = use_human_evaluation
        self.verbose = verbose
        
        # Initialize the evaluation chain with simple pass/fail evaluation
        self.evaluator = PromptChain(
            models=["anthropic/claude-3-sonnet-20240229"],
            instructions=["""Evaluate if this prompt is acceptable or needs improvement.
Consider:
1. Clarity: Are the instructions clear and specific?
2. Completeness: Are all necessary guidelines included?
3. Task Alignment: Does it match the intended task?
4. Output Quality: Will it generate high quality outputs?

Respond with either:
PASS: [Brief explanation why it's acceptable]
or
FAIL: [List specific improvements needed]

Prompt to evaluate: {input}"""],
            store_steps=True
        )
        
        self.evaluator.add_techniques([
            "role_playing:prompt engineering expert"
        ])
        
        # Initialize the improvement chain with a more structured improvement prompt
        self.improver = PromptChain(
            models=["openai/gpt-4o"],
            instructions=["""You are a prompt engineering expert. Your task is to improve the given prompt based on specific feedback.

IMPORTANT: Do NOT change the core purpose or domain of the prompt. Maintain the original intent and subject matter.

Current Prompt:
{input}

Follow these steps to improve the prompt:
1. Understand the original prompt's purpose and domain
2. Apply the specific feedback provided
3. Make targeted improvements while preserving the original intent
4. Return ONLY the improved prompt without explanations

The improved prompt should be clearly formatted and ready to use.

Feedback to address:
"""],
            store_steps=True
        )
        
        self.improver.add_techniques([
            "role_playing:prompt engineer"
        ])

    def get_human_evaluation(self, prompt: str) -> tuple[bool, str]:
        """Get evaluation from human user."""
        if self.verbose:
            print("\n=== Current Prompt ===")
            print(prompt)
            print("\n=== Evaluation Options ===")
            print("1: Accept prompt (PASS)")
            print("2: Request improvements (FAIL)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "1":
                # Return PASS result without exiting
                return True, "PASS: Prompt accepted by human evaluator."
            elif choice == "2":
                if self.verbose:
                    print("\n=== Improvement Suggestions ===")
                    print("Please enter your suggestions for improving the prompt.")
                    print("Consider:")
                    print("- Clarity of instructions")
                    print("- Completeness of guidelines")
                    print("- Task alignment")
                    print("- Output quality")
                print("\nEnter your suggestions below:")
                feedback = input("> ").strip()
                while not feedback:
                    print("Please provide some suggestions for improvement:")
                    feedback = input("> ").strip()
                return False, f"FAIL: {feedback}"
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def create_specialized_prompt(self, task_description: str) -> str:
        """
        Create and iteratively improve a specialized prompt for a given task.
        
        Args:
            task_description: Description of the task to create a prompt for
            
        Returns:
            The final optimized prompt
        """
        # Initial prompt creation chain
        creator = PromptChain(
            models=[
                "anthropic/claude-3-sonnet-20240229",
                "openai/gpt-4o-mini",
            ],
            instructions=[
                """Analyze this task and identify:
                    - Key capabilities needed
                    - Constraints and requirements
                    - Expected input/output formats
                    - Edge cases to handle

                    Task to analyze: {input}""",
                                    
                                    """Create a specialized prompt based on the analysis:
                    - Include clear instructions
                    - Specify constraints and guidelines
                    - Define expected output format
                    - Add relevant examples
                    - Include error handling guidance

                    Analysis: {input}"""
            ],
            full_history=False,  # Don't store history
            store_steps=False    # Don't store steps
        )
        
        creator.add_techniques([
            "role_playing:task analyst",
            "role_playing:prompt engineer"
        ])
        
        # Generate initial prompt
        result = creator.process_prompt(task_description)
        current_prompt = result[-1]['output'] if isinstance(result, list) else result
        
        if self.verbose:
            print("\n=== Initial Prompt ===")
            print(current_prompt)
        
        # Iterative improvement loop
        for iteration in range(self.max_iterations):
            if self.verbose and not iteration == 0:
                print(f"\n=== Iteration {iteration + 1} ===")
            
            # Get evaluation (either human or LLM)
            if self.use_human_evaluation:
                passed, evaluation = self.get_human_evaluation(current_prompt)
                if passed:
                    # Display final prompt in rich format and exit
                    return display_final_prompt(current_prompt)
            else:
                evaluation = self.evaluator.process_prompt(current_prompt)
                passed = evaluation.startswith("PASS:")
                if passed:
                    # Display final prompt in rich format and exit
                    return display_final_prompt(current_prompt)
            
            if self.verbose:
                print("\nEvaluation:")
                print(evaluation)
            
            # Extract improvement feedback
            feedback = evaluation.replace("FAIL:", "").strip()
            
            if self.verbose:
                print("\nImproving prompt based on feedback...")
            
            # Improve the prompt with the feedback
            # Construct a complete input with both the prompt and feedback
            improvement_input = f"""Current Prompt:
{current_prompt}

Feedback to address:
{feedback}"""
            
            # Process the improvement
            improved = self.improver.process_prompt(improvement_input)
            current_prompt = improved[-1]['output'] if isinstance(improved, list) else improved
        
        # Display final prompt in rich format if max iterations reached
        return display_final_prompt(current_prompt)

    def test_prompt(self, prompt: str, test_inputs: List[str]) -> Dict:
        """
        Test a prompt with multiple inputs and evaluate consistency.
        
        Args:
            prompt: The prompt to test
            test_inputs: List of test inputs
            
        Returns:
            Evaluation results
        """
        # Create test chain
        tester = PromptChain(
            models=["openai/gpt-4o-mini"] * len(test_inputs),
            instructions=[prompt] * len(test_inputs),
            store_steps=True
        )
        
        tester.add_techniques([
            "role_playing:test executor"
        ])
        
        # Generate outputs for all test inputs
        outputs = []
        for input_text in test_inputs:
            tester.reset_model_index()
            result = tester.process_prompt(input_text)
            output = result[-1]['output'] if isinstance(result, list) else result
            outputs.append(output)
        
        if self.use_human_evaluation:
            print("\n=== Test Outputs ===")
            for i, (input_text, output) in enumerate(zip(test_inputs, outputs), 1):
                print(f"\nTest {i}:")
                print("Input:", input_text)
                print("Output:", output)
            
            passed, feedback = self.get_human_evaluation("\n".join(outputs))
            return {
                "passed": passed,
                "feedback": feedback.replace("PASS:", "").replace("FAIL:", "").strip(),
                "outputs": outputs
            }
        
        # Use LLM evaluation
        consistency_evaluator = PromptChain(
            models=["anthropic/claude-3-sonnet-20240229"],
            instructions=["""Evaluate if these outputs are consistently high quality.
                            Consider:
                            1. Format Consistency
                            2. Style Consistency
                            3. Quality Consistency
                            4. Logic Consistency

                            Respond with either:
                            PASS: [Brief explanation of consistency]
                            or
                            FAIL: [List specific consistency issues]

                            Test Inputs:
                            {input}

                            Generated Outputs:
                            {outputs}"""],
            store_steps=True
        )
        
        consistency_evaluator.add_techniques([
            "role_playing:consistency evaluator"
        ])
        
        consistency_evaluator.reset_model_index()
        
        # Evaluate consistency
        evaluation = consistency_evaluator.process_prompt(
            f"Test Inputs: {test_inputs}\n\nOutputs: {outputs}"
        )
        
        return {
            "passed": evaluation.startswith("PASS:"),
            "feedback": evaluation.replace("PASS:", "").replace("FAIL:", "").strip(),
            "outputs": outputs
        }

    def improve_prompt_continuously(self, initial_prompt: str) -> str:
        """
        Continuously improve a prompt using the improvement prompt without human evaluation.
        
        Args:
            initial_prompt: The initial prompt to improve
            
        Returns:
            The final improved prompt after max_iterations
        """
        current_prompt = initial_prompt
        
        if self.verbose:
            print("\n=== Initial Prompt ===")
            print(current_prompt)
        
        # Iterative improvement loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n=== Iteration {iteration + 1} ===")
            
            # Improve the prompt using the improvement prompt
            result = self.improver.process_prompt(current_prompt)
            
            # Extract the final prompt from the structured output
            output = result[-1]['output'] if isinstance(result, list) else result
            
            # Try to extract the improved prompt using various methods
            # Method 1: Look for tags
            start_tag = "<final_prompt>"
            end_tag = "</final_prompt>"
            start_idx = output.find(start_tag)
            end_idx = output.find(end_tag)
            
            if start_idx != -1 and end_idx != -1:
                # Extract the prompt between tags
                current_prompt = output[start_idx + len(start_tag):end_idx].strip()
            else:
                # Method 2: If no tags, use the entire output as the improved prompt
                # This assumes the model followed instructions to return only the improved prompt
                current_prompt = output.strip()
            
            if self.verbose:
                print("\nImproved Prompt:")
                print(current_prompt)
        
        # Return the final prompt
        return current_prompt

# Example usage
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Prompt Engineering Tool')
    parser.add_argument('--feedback', 
                       choices=['human', 'llm'],
                       default='llm',
                       help='Choose feedback type: human for human evaluation, llm for LLM evaluation')
    parser.add_argument('--max-iterations',
                       type=int,
                       default=3,
                       help='Maximum number of improvement iterations')
    parser.add_argument('--verbose',
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize prompt engineer with command line arguments
    engineer = PromptEngineer(
        max_iterations=args.max_iterations,
        use_human_evaluation=(args.feedback == 'human'),
        verbose=args.verbose
    )
    
    # Example task
    task = """Create a prompt for an AI agent that helps users analyze financial data.
    The agent should:
    1. Extract key metrics
    2. Identify trends
    3. Provide actionable insights
    4. Format output as a structured report"""
    
    if args.verbose:
        print(f"\nUsing {'human' if args.feedback == 'human' else 'LLM'} feedback")
        print(f"Maximum iterations: {args.max_iterations}")
    
    # Create specialized prompt and exit when accepted
    optimized_prompt = engineer.create_specialized_prompt(task)
    # Program ends in display_final_prompt
    exit(0) 