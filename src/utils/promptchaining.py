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
                 store_steps: bool = False):
        """
        Initialize the PromptChain with optional step storage.

        :param models: List of model names or dicts with model config
                      Format: [{"name": "model_name", "params": {...}}, ...]
        :param instructions: List of instruction templates or callable functions
        :param full_history: Whether to pass full chain history
        :param store_steps: If True, stores step outputs in self.step_outputs without returning full history
        """
        # Extract model names and parameters
        self.models = []
        self.model_params = []
        
        for model in models:
            if isinstance(model, dict):
                self.models.append(model["name"])
                self.model_params.append(model.get("params", {}))
            else:
                self.models.append(model)
                self.model_params.append({})

        # Count non-function instructions to match with models
        model_instruction_count = sum(1 for instr in instructions if not callable(instr))
        
        if len(self.models) != model_instruction_count:
            raise ValueError(
                f"Number of models ({len(self.models)}) must match number of non-function instructions ({model_instruction_count})"
            )
            
        self.instructions = instructions
        self.full_history = full_history
        self.model_index = 0
        self.store_steps = store_steps
        self.step_outputs = {}  # Dictionary to store step outputs

    def is_function(self, instruction: Union[str, Callable]) -> bool:
        """Check if an instruction is actually a function"""
        return callable(instruction)

    def process_prompt(self, initial_input: str):
        """Execute the prompt chain with optional step storage."""
        result = initial_input
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
                result = instruction(content_to_process)
                step_type = "function"
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
                print(f"\nStep {step + 1}: Using model {model}")

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