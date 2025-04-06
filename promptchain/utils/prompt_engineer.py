"""A module for iterative prompt engineering with meta-evaluation."""

from typing import List, Dict, Union, Callable, Optional
from promptchain.utils.promptchaining import PromptChain
from dotenv import load_dotenv
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.style import Style
import sys

# Load environment variables
load_dotenv()

# Initialize rich console
console = Console()

# Initialize argument parser
parser = argparse.ArgumentParser(description='Prompt Engineering Tool')
parser.add_argument('--techniques',
                   type=str,
                   nargs='+',
                   default=[],
                   help='''Space-separated list of techniques to apply. Use --interactive for guided selection.
Available techniques:

Required parameters:
- role_playing:profession      (e.g., role_playing:scientist)
- style_mimicking:author      (e.g., style_mimicking:Richard Feynman)
- persona_emulation:expert    (e.g., persona_emulation:Warren Buffett)
- forbidden_words:words       (e.g., forbidden_words:maybe,probably,perhaps)

Optional parameters:
- few_shot:[examples]         (number of examples)
- reverse_prompting:[questions] (number of questions)
- context_expansion:[type]    (context type)
- comparative_answering:[aspects] (aspects to compare)
- tree_of_thought:[paths]     (number of paths)

No parameters:
- step_by_step               (step-by-step reasoning)
- chain_of_thought          (detailed thought process)
- iterative_refinement      (iterative improvement)
- contrarian_perspective    (challenge common beliefs)
- react                     (reason-act-observe process)

Multiple techniques can be combined. Example:
--techniques role_playing:scientist step_by_step few_shot:3''')
parser.add_argument('--feedback',
                   type=str,
                   default='llm',
                   choices=['llm', 'human'],
                   help='Type of feedback to use (llm or human)')
parser.add_argument('--max-iterations',
                   type=int,
                   default=3,
                   help='Maximum number of improvement iterations')
parser.add_argument('--verbose',
                   action='store_true',
                   help='Enable verbose output')
parser.add_argument('--task',
                   type=str,
                   help='Task description for prompt creation')
parser.add_argument('--initial-prompt',
                   type=str,
                   help='Initial prompt to improve. Use "-" to read from stdin')
parser.add_argument('--evaluator-model',
                   type=str,
                   help='Model to use for evaluation')
parser.add_argument('--improver-model',
                   type=str,
                   help='Model to use for improvements')
parser.add_argument('--test',
                   action='store_true',
                   help='Enable prompt testing')
parser.add_argument('--test-inputs',
                   type=str,
                   nargs='+',
                   help='Test inputs for prompt testing')
parser.add_argument('--focus',
                   type=str,
                   default='all',
                   choices=['clarity', 'completeness', 'task_alignment', 'output_quality', 'all'],
                   help='Focus area for improvements')
parser.add_argument('-i', '--interactive',
                   action='store_true',
                   help='Use interactive mode to select techniques and parameters')
parser.add_argument('--output-file',
                   type=str,
                   help='File to save the final prompt to')

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

def get_interactive_techniques():
    """Interactive prompt for selecting techniques and their parameters."""
    # Define technique categories
    REQUIRED_PARAMS = {
        "role_playing": "profession/role (e.g., scientist)",
        "style_mimicking": "author/style (e.g., Richard Feynman)",
        "persona_emulation": "expert name (e.g., Warren Buffett)",
        "forbidden_words": "comma-separated words to avoid"
    }
    
    OPTIONAL_PARAMS = {
        "few_shot": "number of examples",
        "reverse_prompting": "number of questions",
        "context_expansion": "context type",
        "comparative_answering": "aspects to compare",
        "tree_of_thought": "number of paths"
    }
    
    NO_PARAMS = {
        "step_by_step": "step-by-step reasoning",
        "chain_of_thought": "detailed thought process",
        "iterative_refinement": "iterative improvement",
        "contrarian_perspective": "challenge common beliefs",
        "react": "reason-act-observe process"
    }
    
    selected_techniques = []
    navigation_stack = []
    
    # Store configuration settings
    config = {
        "feedback": "llm",
        "max_iterations": 3,
        "verbose": False,
        "task": None,
        "initial_prompt": None,
        "evaluator_model": "anthropic/claude-3-sonnet-20240229",
        "improver_model": "openai/gpt-4o",
        "creator_models": ["anthropic/claude-3-sonnet-20240229", "openai/gpt-4o-mini"],
        "test": False,
        "test_inputs": None,
        "focus": "all",
        "output_file": None
    }

    def show_current_status():
        """Display current selected techniques in a table."""
        table = Table(title="Selected Techniques", show_header=True)
        table.add_column("Technique", style="cyan")
        table.add_column("Parameter", style="green")
        table.add_column("Category", style="yellow")
        
        for tech in selected_techniques:
            if ":" in tech:
                name, param = tech.split(":", 1)
            else:
                name, param = tech, "N/A"
                
            category = "Required" if name in REQUIRED_PARAMS else \
                      "Optional" if name in OPTIONAL_PARAMS else "No Parameters"
            
            table.add_row(name, param, category)
        
        console.print(table)

    def kitchen_sink_mode():
        """Show all parameters at once and allow bulk editing."""
        # Define default parameters for required fields
        DEFAULT_PARAMS = {
            "role_playing": "prompt engineer",
            "style_mimicking": "technical writer",
            "persona_emulation": "expert analyst",
            "forbidden_words": "maybe,probably,perhaps"
        }

        # Define all techniques at this scope
        all_techniques = {
            **{k: ("Required", v) for k, v in REQUIRED_PARAMS.items()},
            **{k: ("Optional", v) for k, v in OPTIONAL_PARAMS.items()},
            **{k: ("None", v) for k, v in NO_PARAMS.items()}
        }

        def show_kitchen_sink_table():
            """Display all techniques with their current values."""
            console.print("\n[bold yellow]Kitchen Sink Mode - All Parameters[/bold yellow]")
            
            table = Table(show_header=True)
            table.add_column("Number", style="cyan")
            table.add_column("Technique", style="green")
            table.add_column("Parameter", style="yellow")
            table.add_column("Required", style="red")
            table.add_column("Current Value", style="magenta")
            table.add_column("Command Parameter", style="blue")
            
            # Get current values and command parameters
            current_values = {}
            command_params = {}
            for tech in selected_techniques:
                if ":" in tech:
                    name, value = tech.split(":", 1)
                    current_values[name] = value
                    command_params[name] = f"{name}:{value}"
                else:
                    current_values[tech] = "Enabled"
                    command_params[tech] = tech
            
            for i, (tech, (req_type, desc)) in enumerate(all_techniques.items(), 1):
                current_val = current_values.get(tech, "Not set")
                if current_val == "Not set" and req_type == "Required":
                    current_val = f"Default: {DEFAULT_PARAMS[tech]}"
                
                command_param = command_params.get(tech, "")
                
                table.add_row(
                    str(i),
                    tech,
                    desc,
                    req_type,
                    current_val,
                    command_param
                )
            
            console.print(table)
            
            # Show resulting command
            if selected_techniques:
                console.print("\n[bold green]Current Command:[/bold green]")
                command = "--techniques " + " ".join(command_params.values())
                console.print(f"[blue]{command}[/blue]")
        
        while True:
            show_kitchen_sink_table()
            
            choice = Prompt.ask(
                "\nEnter technique numbers to configure (comma-separated) or 'done'",
                default="done"
            )
            
            if choice.lower() == "done":
                break
            
            try:
                choices = [int(x.strip()) for x in choice.split(",")]
                for num in choices:
                    if 1 <= num <= len(all_techniques):
                        tech = list(all_techniques.keys())[num-1]
                        req_type = list(all_techniques.values())[num-1][0]
                        
                        # Remove existing technique if it's already selected
                        selected_techniques[:] = [t for t in selected_techniques if not t.startswith(tech + ":") and t != tech]
                        
                        if req_type == "Required":
                            default_val = DEFAULT_PARAMS[tech]
                            param = Prompt.ask(
                                f"Enter {REQUIRED_PARAMS[tech]}",
                                default=default_val
                            )
                            selected_techniques.append(f"{tech}:{param}")
                        elif req_type == "Optional":
                            if Confirm.ask(f"Add parameter for {tech}?"):
                                param = Prompt.ask(f"Enter {OPTIONAL_PARAMS[tech]}")
                                selected_techniques.append(f"{tech}:{param}")
                            else:
                                selected_techniques.append(tech)
                        else:
                            selected_techniques.append(tech)
                            
                        # Show updated table after each change
                        show_kitchen_sink_table()
            except ValueError:
                console.print("[red]Invalid input. Use comma-separated numbers.[/red]")
        
        show_current_status()

    def show_config_status():
        """Display current configuration in a table."""
        table = Table(title="Current Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config.items():
            if isinstance(value, list):
                value = ", ".join(value)
            elif value is None:
                value = "[red]Not set[/red]"
            elif isinstance(value, bool):
                value = "[green]Yes[/green]" if value else "[red]No[/red]"
            table.add_row(key, str(value))
        
        console.print(table)

    def configure_settings():
        """Configure command-line arguments interactively."""
        while True:
            show_config_status()
            
            console.print("\n[bold]Available Settings:[/bold]")
            console.print("1. Feedback type (human/llm)")
            console.print("2. Max iterations")
            console.print("3. Verbose output")
            console.print("4. Task description")
            console.print("5. Initial prompt")
            console.print("6. Evaluator model")
            console.print("7. Improver model")
            console.print("8. Creator models")
            console.print("9. Enable testing")
            console.print("10. Test inputs")
            console.print("11. Focus area")
            console.print("12. Output file")
            console.print("13. Back to main menu")
            
            choice = Prompt.ask("Choose setting to configure", choices=[str(i) for i in range(1, 14)])
            
            if choice == "13":
                break
                
            if choice == "1":
                config["feedback"] = Prompt.ask(
                    "Choose feedback type",
                    choices=["human", "llm"],
                    default=config["feedback"]
                )
            elif choice == "2":
                config["max_iterations"] = int(Prompt.ask(
                    "Enter maximum iterations",
                    default=str(config["max_iterations"])
                ))
            elif choice == "3":
                config["verbose"] = Confirm.ask("Enable verbose output?", default=config["verbose"])
            elif choice == "4":
                config["task"] = Prompt.ask("Enter task description", default=config["task"] or "")
            elif choice == "5":
                config["initial_prompt"] = Prompt.ask("Enter initial prompt", default=config["initial_prompt"] or "")
            elif choice == "6":
                config["evaluator_model"] = Prompt.ask(
                    "Enter evaluator model",
                    default=config["evaluator_model"]
                )
            elif choice == "7":
                config["improver_model"] = Prompt.ask(
                    "Enter improver model",
                    default=config["improver_model"]
                )
            elif choice == "8":
                models = Prompt.ask(
                    "Enter creator models (comma-separated)",
                    default=",".join(config["creator_models"])
                )
                config["creator_models"] = [m.strip() for m in models.split(",")]
            elif choice == "9":
                config["test"] = Confirm.ask("Enable testing?", default=config["test"])
            elif choice == "10":
                if config["test"]:
                    inputs = Prompt.ask(
                        "Enter test inputs (comma-separated)",
                        default=",".join(config["test_inputs"]) if config["test_inputs"] else ""
                    )
                    config["test_inputs"] = [i.strip() for i in inputs.split(",")] if inputs else None
                else:
                    console.print("[red]Enable testing first[/red]")
            elif choice == "11":
                config["focus"] = Prompt.ask(
                    "Choose focus area",
                    choices=["clarity", "completeness", "task_alignment", "output_quality", "all"],
                    default=config["focus"]
                )
            elif choice == "12":
                config["output_file"] = Prompt.ask("Enter output file path", default=config["output_file"] or "")

    # Show current status
    if selected_techniques:
        show_current_status()
    
    while True:
        # Show main menu
        console.print("\n[bold]Available Actions:[/bold]")
        console.print("1. Add techniques requiring parameters")
        console.print("2. Add techniques with optional parameters")
        console.print("3. Add techniques without parameters")
        console.print("4. Kitchen sink mode (configure all at once)")
        console.print("5. Remove last technique")
        console.print("6. Clear all techniques")
        console.print("7. Configure settings")
        console.print("8. Done")
        console.print("9. Back to previous menu" if navigation_stack else "")
        
        choices = ["1", "2", "3", "4", "5", "6", "7", "8"]
        if navigation_stack:
            choices.append("9")
        
        category = Prompt.ask("Choose action", choices=choices)
        
        if category == "9" and navigation_stack:
            # Go back to previous menu
            previous = navigation_stack.pop()
            continue
        
        if category == "8":
            # Return both techniques and config
            return selected_techniques, config
            
        if category == "7":
            navigation_stack.append(category)
            configure_settings()
            continue
            
        if category == "6":
            if Confirm.ask("Are you sure you want to clear all techniques?"):
                selected_techniques = []
            continue
            
        if category == "5":
            if selected_techniques:
                removed = selected_techniques.pop()
                console.print(f"[yellow]Removed: {removed}[/yellow]")
            else:
                console.print("[red]No techniques to remove[/red]")
            continue
            
        if category == "4":
            navigation_stack.append(category)
            kitchen_sink_mode()
            continue
            
        if category == "1":
            # Show techniques requiring parameters
            console.print("\n[bold]Techniques Requiring Parameters:[/bold]")
            for i, (tech, desc) in enumerate(REQUIRED_PARAMS.items(), 1):
                console.print(f"{i}. {tech}: {desc}")
            
            choice = Prompt.ask("Choose technique number or 'back'")
            
            if choice.lower() == "back":
                continue
                
            try:
                tech_idx = int(choice)
                if 1 <= tech_idx <= len(REQUIRED_PARAMS):
                    tech = list(REQUIRED_PARAMS.keys())[tech_idx - 1]
                    param = Prompt.ask(f"Enter {REQUIRED_PARAMS[tech]}")
                    selected_techniques.append(f"{tech}:{param}")
            except ValueError:
                console.print("[red]Invalid choice[/red]")
            
        elif category == "2":
            # Show techniques with optional parameters
            console.print("\n[bold]Techniques with Optional Parameters:[/bold]")
            for i, (tech, desc) in enumerate(OPTIONAL_PARAMS.items(), 1):
                console.print(f"{i}. {tech}: {desc}")
            
            choice = Prompt.ask("Choose technique number or 'back'")
            
            if choice.lower() == "back":
                continue
                
            try:
                tech_idx = int(choice)
                if 1 <= tech_idx <= len(OPTIONAL_PARAMS):
                    tech = list(OPTIONAL_PARAMS.keys())[tech_idx - 1]
                    if Confirm.ask("Add a parameter?"):
                        param = Prompt.ask(f"Enter {OPTIONAL_PARAMS[tech]}")
                        selected_techniques.append(f"{tech}:{param}")
                    else:
                        selected_techniques.append(tech)
            except ValueError:
                console.print("[red]Invalid choice[/red]")
                
        elif category == "3":
            # Show techniques without parameters
            console.print("\n[bold]Techniques without Parameters:[/bold]")
            for i, (tech, desc) in enumerate(NO_PARAMS.items(), 1):
                console.print(f"{i}. {tech}: {desc}")
            
            choice = Prompt.ask("Choose technique number or 'back'")
            
            if choice.lower() == "back":
                continue
                
            try:
                tech_idx = int(choice)
                if 1 <= tech_idx <= len(NO_PARAMS):
                    tech = list(NO_PARAMS.keys())[tech_idx - 1]
                    selected_techniques.append(tech)
            except ValueError:
                console.print("[red]Invalid choice[/red]")
        
        navigation_stack.append(category)
    
    return selected_techniques, config

# Add stdin support
def read_stdin_if_dash(value):
    """Read from stdin if value is '-'"""
    if value == '-':
        return sys.stdin.read().strip()
    return value

# Example usage
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Handle stdin for initial prompt
    if args.initial_prompt:
        args.initial_prompt = read_stdin_if_dash(args.initial_prompt)
    
    # If interactive mode is selected, get techniques interactively
    if args.interactive:
        techniques, config = get_interactive_techniques()
        args.techniques = techniques
        
        # Update args with interactive config
        for key, value in config.items():
            if value is not None:  # Only override if value was set
                setattr(args, key, value)
    
    # Initialize prompt engineer with command line arguments
    engineer = PromptEngineer(
        max_iterations=args.max_iterations,
        use_human_evaluation=(args.feedback == 'human'),
        verbose=args.verbose
    )
    
    # Override default models if specified
    if args.evaluator_model:
        engineer.evaluator.models = [args.evaluator_model]
    
    if args.improver_model:
        engineer.improver.models = [args.improver_model]
    
    # Apply additional techniques
    if args.techniques:
        engineer.evaluator.add_techniques(args.techniques)
        engineer.improver.add_techniques(args.techniques)
    
    # Example task if not provided
    task = args.task if args.task else """Create a prompt for an AI agent that helps users analyze financial data.
    The agent should:
    1. Extract key metrics
    2. Identify trends
    3. Provide actionable insights
    4. Format output as a structured report"""
    
    if args.verbose:
        print(f"\nUsing {'human' if args.feedback == 'human' else 'LLM'} feedback")
        print(f"Maximum iterations: {args.max_iterations}")
    
    # Process based on mode
    if args.initial_prompt:
        # Improve existing prompt
        optimized_prompt = engineer.improve_prompt_continuously(args.initial_prompt)
    else:
        # Create specialized prompt and exit when accepted
        optimized_prompt = engineer.create_specialized_prompt(task)
    
    # Test prompt if requested
    if args.test and args.test_inputs:
        test_results = engineer.test_prompt(optimized_prompt, args.test_inputs)
        if args.verbose:
            print("\n=== Test Results ===")
            print(f"Passed: {test_results['passed']}")
            print(f"Feedback: {test_results['feedback']}")
            
            print("\n=== Test Outputs ===")
            for i, (input_text, output) in enumerate(zip(args.test_inputs, test_results['outputs']), 1):
                print(f"\nTest {i}:")
                print("Input:", input_text)
                print("Output:", output)
    
    # Save to output file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(optimized_prompt)
        if args.verbose:
            print(f"\nPrompt saved to {args.output_file}")
    
    # Program ends in display_final_prompt
    exit(0) 