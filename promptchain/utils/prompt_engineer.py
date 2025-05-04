"""A module for iterative prompt engineering with meta-evaluation."""

from typing import List, Dict, Union, Callable, Optional, Any, Tuple
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
import re

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
parser.add_argument('--protect-content',
                   action='store_true',
                   default=True,
                   help='Protect content between triple backticks from modification')
parser.add_argument('--no-protect-content',
                   action='store_false',
                   dest='protect_content',
                   help='Disable protection of content between triple backticks')

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

class PromptEvaluator:
    """Evaluates and improves prompts based on specific techniques."""
    
    def __init__(self, model: str):
        self.model = model
        self.techniques = []
    
    def add_techniques(self, techniques: List[str]):
        """Add evaluation techniques to use."""
        self.techniques = techniques
    
    def evaluate(self, prompt: str) -> dict:
        """Translate selected techniques into actionable suggestions for the improver."""
        suggestions = []

        # Directly generate suggestions based on the enabled techniques
        for technique in self.techniques:
            if technique == "step_by_step":
                suggestions.append("Add step-by-step instructions")
            elif technique == "chain_of_thought":
                suggestions.append("Add Chain of Thought reasoning (e.g., 'Think step by step')")
            elif technique == "iterative_refinement" or technique == "self_refine":
                suggestions.append("Add instructions for iterative refinement or self-critique")
            elif technique.startswith("role_playing:"):
                param = technique.split(":", 1)[1]
                suggestions.append(f"Assign the specific role: '{param}' (e.g., 'Act as a {param}')")
            elif technique.startswith("style_mimicking:"):
                param = technique.split(":", 1)[1]
                suggestions.append(f"Specify the writing style to mimic: '{param}' (e.g., 'Write in the style of {param}')")
            elif technique.startswith("persona_emulation:"):
                param = technique.split(":", 1)[1]
                suggestions.append(f"Instruct the prompt to emulate the persona: '{param}' (e.g., 'Emulate {param}')")
            elif technique.startswith("few_shot"):
                suggestions.append("Include relevant few-shot examples")
            elif technique.startswith("tree_of_thought"):
                 suggestions.append("Encourage exploring multiple reasoning paths (Tree of Thought)")
            elif technique.startswith("context_expansion"):
                 suggestions.append("Add instructions to provide relevant context")
            elif technique.startswith("reverse_prompting"):
                 suggestions.append("Add instructions for the prompt to ask clarifying questions or reflect")
            elif technique == "contrarian_perspective":
                 suggestions.append("Add instructions to consider a contrarian perspective")
            elif technique == "react":
                 suggestions.append("Structure the prompt using the Reason-Act-Observe (ReAct) cycle")
            elif technique.startswith("forbidden_words:"):
                 param = technique.split(":", 1)[1]
                 suggestions.append(f"Ensure the following words are avoided: {param}")
            elif technique.startswith("comparative_answering:"):
                 param = technique.split(":", 1)[1]
                 suggestions.append(f"Add comparison criteria based on: {param}")
            # Add other techniques if needed

        # The score is less meaningful now, but we can keep the structure
        score = max(0, 5 - len(suggestions)) # Less relevant score calculation

        return {
            'score': score, # Score might not be very useful with this logic
            'suggestions': suggestions
        }

class PromptEngineer:
    """Engineers better prompts through iterative improvement."""

    def __init__(self, 
                 max_iterations: int = 3,
                 verbose: bool = False,
                 evaluator_model: str = "openai/gpt-4",
                 improver_model: Optional[str] = None,
                 use_human_evaluation: bool = False,
                 protect_content: bool = True,
                 focus_area: str = "all"):
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.evaluator = PromptEvaluator(evaluator_model)
        self.improver_model = improver_model if improver_model else evaluator_model
        self.use_human_evaluation = use_human_evaluation
        self.protect_content = protect_content
        self.focus_area = focus_area
        
    def _extract_protected_content(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Extract content between triple backticks and replace with placeholders.
        
        Args:
            text: The input text to process
            
        Returns:
            Tuple containing:
                - Modified text with placeholders
                - Dictionary mapping placeholders to original content
        """
        # Only process if triple backticks are present
        if '```' not in text and not any(f"${{{i}}}" in text for i in range(1, 100)) and not self.protect_content:
            return text, {}
            
        protected_sections = {}
        counter = 0
        
        # First, handle triple backticks
        def replace_backticks(match):
            nonlocal counter
            # Create a unique variable name (not visible in text)
            variable_name = f"__PROTECTED_CONTENT_{counter}__"
            protected_sections[variable_name] = match.group(1)
            counter += 1
            # Return a special marker that won't appear naturally in text
            return f"__CONTENT_MARKER_{counter-1}__"
        
        # Extract content between triple backticks
        modified_text = re.sub(r'```(.*?)```', replace_backticks, text, flags=re.DOTALL)
        
        # Handle explicit variable protection like ${1}, ${2}, etc.
        def replace_variables(match):
            var_id = match.group(1)
            var_content = match.group(2)
            variable_name = f"__PROTECTED_VAR_{var_id}__"
            protected_sections[variable_name] = var_content
            return f"__VAR_MARKER_{var_id}__"
            
        # Look for ${n}content${n} patterns
        modified_text = re.sub(r'\${(\d+)}(.*?)\${(\1)}', replace_variables, modified_text, flags=re.DOTALL)
        
        if self.verbose and protected_sections:
            print(f"Protected {len(protected_sections)} content blocks")
            
        return modified_text, protected_sections
    
    def _restore_protected_content(self, text: str, protected_sections: Dict[str, str]) -> str:
        """
        Restore protected content from placeholders.
        
        Args:
            text: Text with placeholders
            protected_sections: Dictionary mapping placeholders to original content
            
        Returns:
            Text with original content restored
        """
        if not protected_sections:
            return text
            
        result = text
        
        # Restore backtick content
        for i, (variable_name, content) in enumerate(protected_sections.items()):
            if variable_name.startswith("__PROTECTED_CONTENT_"):
                marker_match = re.search(f"__CONTENT_MARKER_(\d+)__", variable_name)
                if marker_match:
                    marker_index = marker_match.group(1)
                    marker = f"__CONTENT_MARKER_{marker_index}__"
                    # Replace marker with original content wrapped in backticks
                    result = result.replace(marker, f"```{content}```")
            elif variable_name.startswith("__PROTECTED_VAR_"):
                var_id = variable_name.split("_")[3].rstrip("__")
                marker = f"__VAR_MARKER_{var_id}__"
                # Replace marker with original content (no wrapping)
                result = result.replace(marker, content)
            
        if self.verbose:
            print(f"Restored {len(protected_sections)} protected content blocks")
            
        return result
        
    def create_specialized_prompt(self, base_prompt: str) -> str:
        """
        Create a specialized prompt through iterative improvement.
        
        Args:
            base_prompt: The starting prompt to improve
            
        Returns:
            The improved prompt
        """
        # Handle protected content if enabled
        current_prompt, protected_sections = self._extract_protected_content(base_prompt)
            
        best_score = -1 # Initialize best score to allow improvement even with 0 suggestions initially
        best_prompt = current_prompt
        
        for i in range(self.max_iterations):
            # Evaluate current prompt based on selected techniques
            evaluation = self.evaluator.evaluate(current_prompt) # Evaluate method now ignores prompt content
            
            # Handle human feedback simulation/warning
            if self.use_human_evaluation:
                 warning_msg = "[Warning] Human feedback mode requested but not implemented for interactive use. Proceeding with LLM evaluation."
                 print(warning_msg) # Also print to server console
                 # Potential: Send warning back via logs if a mechanism exists
                 if self.verbose:
                     print("--- Human Feedback Point (Simulated) ---")
                     print(f"Current Prompt:\n{current_prompt}")
                     print(f"Suggestions:\n{evaluation['suggestions']}")
                     print("--- (Proceeding Automatically) ---")
                     # NOTE: No actual waiting or input is handled here.

            if self.verbose:
                print(f"\nIteration {i+1}: Suggestions based on selected techniques: {evaluation['suggestions']}")
                # print(f"Current score: {evaluation['score']}") # Score is less relevant now
            
            # If no suggestions were generated based on selected techniques, break the loop
            if not evaluation['suggestions']:
                if i == 0: # If no suggestions on the first try, return original
                    best_prompt = current_prompt 
                # else: best_prompt already holds the result from the previous iteration
                break 
            
            # --- Improvement Step --- 
            # Prepare focus instruction if applicable
            focus_instruction = ""
            if self.focus_area != "all":
                focus_map = {
                    "task_alignment": "alignment with the original task",
                    "output_quality": "quality and relevance of the expected output"
                }
                focus_display = focus_map.get(self.focus_area, self.focus_area) 
                focus_instruction = f"Focus specifically on improving the prompt's {focus_display}."

            # Create improvement chain with protection and focus awareness
            improvement_chain = PromptChain(
                models=[self.improver_model], 
                instructions=[
                    f"""
                    Improve this prompt based on the following suggestions.
                    Return ONLY the improved prompt without any introductory text or explanations.
                    Do NOT start with phrases like 'Here is' or 'The improved prompt'.
                    
                    You are an expert prompt engineer. Refine the 'Current prompt' below by carefully applying the 'Suggestions to address'. 
                    Integrate these suggestions thoughtfully to enhance the prompt's effectiveness according to the specified 'Focus area' (if any), 
                    while preserving the original core intent. Ensure the final output is clear, concise, and directly usable.
                    
                    Follow this thinking process while applying the suggestions:
                    '''
                    Think step by step:
                    1) Analyze the given techniques (represented by the suggestions) and understand their purpose and significance in the context of the current prompt.
                    2) Translate each suggestion into concrete changes or additions to the prompt text.
                    3) Implement the suggestions iteratively if multiple are present, ensuring coherence.
                    4) After applying suggestions, mentally critique the result: Does it clearly reflect the technique's intent? Is it well-integrated?
                    5) Refine the implementation based on this self-critique before finalizing the output for this iteration.
                    6) Consider if the applied techniques could be combined or phrased more effectively.
                    7) Reflect on the overall clarity and effectiveness of the resulting prompt.
                    '''
                    
                    {"IMPORTANT: Do NOT modify any content marked with __CONTENT_MARKER_X__ or __VAR_MARKER_X__." if protected_sections else ""}
                    
                    Suggestions to address:
                    {evaluation['suggestions']}
                    
                    {focus_instruction}
                    
                    Current prompt:
                    {current_prompt}
                    """
                ],
                verbose=self.verbose
            )
            
            # Get improved version and handle different result formats
            try:
                # Provide a non-empty placeholder input to avoid Anthropic API error
                result = improvement_chain.process_prompt("Improve the prompt based on the instructions.") 
                
                # Handle different result formats
                if isinstance(result, dict):
                    current_prompt = result.get('output', '')
                elif isinstance(result, list) and result:
                    current_prompt = result[-1].get('output', '') if isinstance(result[-1], dict) else str(result[-1])
                else:
                    current_prompt = str(result)
                
                # Strip any unwanted prefix text
                common_prefixes = [
                    "here is the improved prompt:",
                    "here is the improved version:",
                    "the improved prompt:",
                    "improved prompt:",
                    "here's the improved prompt:"
                ]
                
                result_lower = current_prompt.lower()
                for prefix in common_prefixes:
                    if result_lower.startswith(prefix):
                        current_prompt = current_prompt[len(prefix):].strip()
                        break
                
                # Update best prompt after a successful improvement iteration
                best_prompt = current_prompt
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error in improvement iteration {i+1}: {str(e)}")
                # If an error occurs during improvement, stop iterating and use the previous best
                break 

            # Optional: Add a check here to see if the prompt *actually* changed. 
            # If current_prompt == best_prompt after improvement, could break early.
        
        # Return the best prompt found, with protected content restored
        return self._restore_protected_content(best_prompt, protected_sections)

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
            console.print("13. Protected content")
            console.print("14. Back to main menu")
            
            choice = Prompt.ask("Choose setting to configure", choices=[str(i) for i in range(1, 15)])
            
            if choice == "14":
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
            elif choice == "13":
                config["protect_content"] = Confirm.ask(
                    "Enable protection of content between triple backticks?",
                    default=config.get("protect_content", True)
                )

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
        verbose=args.verbose,
        evaluator_model=args.evaluator_model if args.evaluator_model else "openai/gpt-4",
        improver_model=args.improver_model,
        use_human_evaluation=args.feedback == 'human',
        protect_content=args.protect_content,
        focus_area=args.focus
    )
    
    # Override default models if specified
    if args.evaluator_model:
        engineer.evaluator.model = args.evaluator_model
    
    # Apply additional techniques
    if args.techniques:
        engineer.evaluator.add_techniques(args.techniques)
    
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
        print(f"Protected content: {'enabled' if args.protect_content else 'disabled'}")
    
    # Process based on mode
    if args.initial_prompt:
        # Improve existing prompt
        optimized_prompt = engineer.create_specialized_prompt(args.initial_prompt)
    else:
        # Create specialized prompt and exit when accepted
        optimized_prompt = engineer.create_specialized_prompt(task)
    
    # Save to output file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(optimized_prompt)
        if args.verbose:
            print(f"\nPrompt saved to {args.output_file}")
    
    # Program ends in display_final_prompt
    display_final_prompt(optimized_prompt) 