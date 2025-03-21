"""Certainly! Below is an example of a Python-based class that facilitates a prompt chain workflow, allowing you to specify a number of refinement steps, a list of instructions for each step, and the models to be used for each link in the chain. This class encapsulates the logic for initializing models, processing prompts, and dynamically updating prompts with outputs from previous steps."""

from litellm import completion
import os
from typing import Union, Callable, List, Literal
from dotenv import load_dotenv
import inspect
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv("../../.env")

# Configure environment variables for API keys
# These will be loaded from the .env file
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your actual OpenAI API key
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"  # Replace with your actual Anthropic API key

### Python Class for Prompt Chaining

class ChainStep(BaseModel):
    step: int
    input: str
    output: str
    type: Literal["initial", "model", "function"]

class PromptChain:
    def __init__(self, models: List[Union[str, dict]], 
                 instructions: List[Union[str, Callable]], 
                 full_history: bool = False,
                 store_steps: bool = False,
                 verbose: bool = False):
        """
        Initialize the PromptChain with optional step storage and verbose output.

        :param models: List of model names or dicts with model config. 
                      If single model provided, it will be used for all instructions.
        :param instructions: List of instruction templates or callable functions
        :param full_history: Whether to pass full chain history
        :param store_steps: If True, stores step outputs in self.step_outputs without returning full history
        :param verbose: If True, prints detailed output for each step with formatting
        """
        self.verbose = verbose
        # Extract model names and parameters
        self.models = []
        self.model_params = []
        
        # Count non-function instructions to match with models
        model_instruction_count = sum(1 for instr in instructions if not callable(instr))
        
        # Handle single model case
        if len(models) == 1:
            # Replicate the single model for all instructions
            models = models * model_instruction_count
        
        # Process models
        for model in models:
            if isinstance(model, dict):
                self.models.append(model["name"])
                self.model_params.append(model.get("params", {}))
            else:
                self.models.append(model)
                self.model_params.append({})

        # Validate model count
        if len(self.models) != model_instruction_count:
            raise ValueError(
                f"Number of models ({len(self.models)}) must match number of non-function instructions ({model_instruction_count})"
                "\nOr provide a single model to use for all instructions."
            )
            
        self.instructions = instructions
        self.full_history = full_history
        self.model_index = 0
        self.store_steps = store_steps
        self.step_outputs = {}

    def is_function(self, instruction: Union[str, Callable]) -> bool:
        """Check if an instruction is actually a function"""
        return callable(instruction)

    def process_prompt(self, initial_input: str):
        """Execute the prompt chain with optional step storage and verbose output."""
        result = initial_input
        
        if self.verbose:
            print("\n" + "="*50)
            print("🔄 Starting Prompt Chain")
            print("="*50)
            print("\n📝 Initial Input:")
            print(f"{initial_input}\n")
        
        chain_history = [{
            "step": 0, 
            "input": initial_input, 
            "output": initial_input,
            "type": "initial"
        }]
        
        # Store initial step if requested
        if self.store_steps:
            self.step_outputs["step_0"] = {
                "type": "initial",
                "output": initial_input
            }
        
        for step, instruction in enumerate(self.instructions):
            if self.verbose:
                print("\n" + "-"*50)
                print(f"Step {step + 1}:")
                print("-"*50)
            
            if self.full_history:
                history_text = "\n".join([
                    f"Step {entry['step']}: {entry['output']}" 
                    for entry in chain_history
                ])
                content_to_process = f"Previous steps:\n{history_text}\n\nCurrent input: {result}"
            else:
                content_to_process = result

            # Check if this step is a function or an instruction
            if self.is_function(instruction):
                if self.verbose:
                    print(f"\n🔧 Executing Function: {instruction.__name__}")
                    print(f"\nInput:\n{content_to_process}")
                
                result = instruction(content_to_process)
                step_type = "function"
                
                if self.verbose:
                    print(f"\nOutput:\n{result}")
            else:
                # Load instruction if it's a file path
                if os.path.isfile(str(instruction)):
                    with open(instruction, 'r') as file:
                        instruction = file.read()
                
                prompt = instruction.replace("{input}", content_to_process)
                model = self.models[self.model_index]
                self.model_index += 1  # Move to next model
                result = self.run_model(model, prompt, self.model_params[self.model_index - 1])
                step_type = "model"
                if self.verbose:
                    print(f"\n🤖 Using Model: {model}")
                    if self.model_params[self.model_index - 1]:
                        print(f"Parameters: {self.model_params[self.model_index - 1]}")
                    print(f"\nPrompt:\n{instruction.replace('{input}', '...')}")
                    print(f"\nInput:\n{content_to_process}")
                
                if self.verbose:
                    print(f"\nOutput:\n{result}")

            chain_history.append({
                "step": step + 1,
                "input": content_to_process,
                "output": result,
                "type": step_type,
                "model_params": self.model_params[self.model_index - 1]
            })

            # Store step output if requested
            if self.store_steps:
                self.step_outputs[f"step_{step + 1}"] = {
                    "type": step_type,
                    "output": result,
                    "model_params": self.model_params[self.model_index - 1] if step_type == "model" else None
                }

        if self.verbose:
            print("\n" + "="*50)
            print("✅ Chain Completed")
            print("="*50)
            print("\n📊 Final Output:")
            print(f"{result}\n")
        
        return result if not self.full_history else chain_history

    @staticmethod
    def run_model(model_name: str, prompt: str, params: dict = None) -> str:
        """Execute model using LiteLLM with custom parameters."""
        model_params = {
            "model": model_name,
            "messages": [{"content": prompt, "role": "user"}]
        }
        
        # Add any custom parameters
        if params:
            model_params.update(params)
        
        response = completion(**model_params)
        return response['choices'][0]['message']['content']

    def get_step_output(self, step_number: int) -> dict:
        """Retrieve output for a specific step."""
        if not self.store_steps:
            raise ValueError("Step storage is not enabled. Initialize with store_steps=True")
        
        step_key = f"step_{step_number}"
        if step_key not in self.step_outputs:
            raise ValueError(f"Step {step_number} not found")
        
        return self.step_outputs[step_key]

    def add_techniques(self, techniques: List[str]) -> None:
        """
        Injects additional prompt engineering techniques into all string-based instructions.
        Each technique can include an optional parameter using the format "technique:parameter"
        
        Some techniques REQUIRE parameters:
        - role_playing: requires profession/role (e.g., "role_playing:scientist")
        - style_mimicking: requires author/style (e.g., "style_mimicking:Richard Feynman")
        - persona_emulation: requires expert name (e.g., "persona_emulation:Warren Buffett")
        - forbidden_words: requires comma-separated words (e.g., "forbidden_words:maybe,probably,perhaps")
        
        :param techniques: List of technique strings (e.g., ["step_by_step", "role_playing:scientist"])
        """
        # Define which techniques require parameters
        REQUIRED_PARAMS = {
            "role_playing": "profession/role",
            "style_mimicking": "author/style",
            "persona_emulation": "expert name",
            "forbidden_words": "comma-separated words"
        }
        
        # Define which techniques accept optional parameters
        OPTIONAL_PARAMS = {
            "few_shot": "number of examples",
            "reverse_prompting": "number of questions",
            "context_expansion": "context type",
            "comparative_answering": "aspects to compare",
            "tree_of_thought": "number of paths"
        }
        
        # Techniques that don't use parameters
        NO_PARAMS = {
            "step_by_step",
            "chain_of_thought",
            "iterative_refinement",
            "contrarian_perspective",
            "react"
        }

        prompt_techniques = {
            "role_playing": lambda param: (
                f"You are an experienced {param} explaining in a clear and simple way. "
                "Use relatable examples."
            ),
            "step_by_step": lambda _: (
                "Explain your reasoning step-by-step before providing the final answer."
            ),
            "few_shot": lambda param: (
                f"Include {param or 'a few'} examples to demonstrate the pattern before generating your answer."
            ),
            "chain_of_thought": lambda _: (
                "Outline your reasoning in multiple steps before delivering the final result."
            ),
            "persona_emulation": lambda param: (
                f"Adopt the persona of {param} in this field."
            ),
            "context_expansion": lambda param: (
                f"Consider {param or 'additional background'} context and relevant details in your explanation."
            ),
            "reverse_prompting": lambda param: (
                f"First, generate {param or 'key'} questions about this topic before answering."
            ),
            "style_mimicking": lambda param: (
                f"Emulate the writing style of {param} in your response."
            ),
            "iterative_refinement": lambda _: (
                "Iteratively refine your response to improve clarity and detail."
            ),
            "forbidden_words": lambda param: (
                f"Avoid using these words in your response: {param}. "
                "Use more precise alternatives."
            ),
            "comparative_answering": lambda param: (
                f"Compare and contrast {param or 'relevant'} aspects thoroughly in your answer."
            ),
            "contrarian_perspective": lambda _: (
                "Argue a contrarian viewpoint that challenges common beliefs."
            ),
            "tree_of_thought": lambda param: (
                f"Explore {param or 'multiple'} solution paths and evaluate each before concluding."
            ),
            "react": lambda _: (
                "Follow this process: \n"
                "1. Reason about the problem\n"
                "2. Act based on your reasoning\n"
                "3. Observe the results"
            )
        }

        # Process each technique
        for tech in techniques:
            # Split technique and parameter if provided
            parts = tech.split(":", 1)
            tech_name = parts[0]
            tech_param = parts[1] if len(parts) > 1 else None

            # Validate technique exists
            if tech_name not in prompt_techniques:
                raise ValueError(
                    f"Technique '{tech_name}' not recognized.\n"
                    f"Available techniques:\n"
                    f"- Required parameters: {list(REQUIRED_PARAMS.keys())}\n"
                    f"- Optional parameters: {list(OPTIONAL_PARAMS.keys())}\n"
                    f"- No parameters: {list(NO_PARAMS)}"
                )

            # Validate required parameters
            if tech_name in REQUIRED_PARAMS and not tech_param:
                raise ValueError(
                    f"Technique '{tech_name}' requires a {REQUIRED_PARAMS[tech_name]} parameter.\n"
                    f"Use format: {tech_name}:parameter"
                )

            # Warning for unexpected parameters
            if tech_name in NO_PARAMS and tech_param:
                print(f"Warning: Technique '{tech_name}' doesn't use parameters, ignoring '{tech_param}'")

            # Generate technique text with parameter
            technique_text = prompt_techniques[tech_name](tech_param)

            # Apply to all string instructions
            for i, instruction in enumerate(self.instructions):
                if isinstance(instruction, str):
                    self.instructions[i] = instruction.strip() + "\n" + technique_text

# Example usage
if __name__ == "__main__":
    models = [
        {
            "name": "openai/gpt-4",
            "params": {
                "temperature": 0.7,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            }
        },
        {
            "name": "anthropic/claude-3-sonnet-20240229",
            "params": {
                "temperature": 0.3,
                "top_k": 40,
                "metadata": {"user_id": "123"}
            }
        }
    ]
    instructions = [
        "Initial analysis of {input}",
        "Refine and expand the analysis: {input}",
        "Finalize the insights based on: {input}"
    ]

    prompt_chain = PromptChain(
        models=models,
        instructions=instructions,
        full_history=True,
        store_steps=True
    )
    final_result = prompt_chain.process_prompt("How to improve task quality")
    print(f"Final Output: {final_result}")