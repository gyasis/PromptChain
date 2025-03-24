from litellm import completion
import os
from typing import Union, Callable, List, Literal, Dict, Optional
from dotenv import load_dotenv
import inspect
from pydantic import BaseModel, Field, validator
from enum import Enum

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
                 verbose: bool = False,
                 chainbreakers: List[Callable] = None):
        """
        Initialize the PromptChain with optional step storage and verbose output.

        :param models: List of model names or dicts with model config. 
                      If single model provided, it will be used for all instructions.
        :param instructions: List of instruction templates or callable functions
        :param full_history: Whether to pass full chain history
        :param store_steps: If True, stores step outputs in self.step_outputs without returning full history
        :param verbose: If True, prints detailed output for each step with formatting
        :param chainbreakers: List of functions that can break the chain if conditions are met
                             Each function should take (step_number, current_output) and return
                             (should_break: bool, break_reason: str, final_output: Any)
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
        self.store_steps = store_steps
        self.step_outputs = {}
        self.chainbreakers = chainbreakers or []
        self.reset_model_index()

    def reset_model_index(self):
        """Reset the model index counter."""
        self.model_index = 0

    def is_function(self, instruction: Union[str, Callable]) -> bool:
        """Check if an instruction is actually a function"""
        return callable(instruction)

    def process_prompt(self, initial_input: str):
        """Execute the prompt chain with optional step storage and verbose output."""
        # Reset model index at the start of each chain
        self.reset_model_index()
        
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
        
        try:
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

                # Initialize step info dictionary for chainbreakers
                step_info = {
                    "step": step + 1,
                    "input": content_to_process
                }

                # Check if this step is a function or an instruction
                if self.is_function(instruction):
                    if self.verbose:
                        print(f"\n🔧 Executing Function: {instruction.__name__}")
                        print(f"\nInput:\n{content_to_process}")
                    
                    result = instruction(content_to_process)
                    step_type = "function"
                    model_params = None
                    
                    # Add function-specific info to step_info
                    step_info.update({
                        "type": "function",
                        "function_name": instruction.__name__,
                        "function": instruction
                    })
                    
                    if self.verbose:
                        print(f"\nOutput:\n{result}")
                else:
                    # Load instruction if it's a file path
                    if os.path.isfile(str(instruction)):
                        with open(instruction, 'r') as file:
                            instruction = file.read()
                    
                    prompt = instruction.replace("{input}", content_to_process)
                    
                    if self.model_index >= len(self.models):
                        raise IndexError(f"Not enough models provided for instruction at step {step + 1}")
                    
                    model = self.models[self.model_index]
                    model_params = self.model_params[self.model_index]
                    self.model_index += 1
                    
                    result = self.run_model(model, prompt, model_params)
                    step_type = "model"
                    
                    # Add model-specific info to step_info
                    step_info.update({
                        "type": "model",
                        "model": model,
                        "model_params": model_params,
                        "prompt": prompt
                    })
                    
                    if self.verbose:
                        print(f"\n🤖 Using Model: {model}")
                        if model_params:
                            print(f"Parameters: {model_params}")
                        print(f"\nPrompt:\n{instruction.replace('{input}', '...')}")
                        print(f"\nInput:\n{content_to_process}")
                        print(f"\nOutput:\n{result}")

                # Update step_info with output
                step_info["output"] = result

                # Add to chain history
                chain_history.append({
                    "step": step + 1,
                    "input": content_to_process,
                    "output": result,
                    "type": step_type,
                    "model_params": model_params if step_type == "model" else None
                })

                # Store step output if requested
                if self.store_steps:
                    self.step_outputs[f"step_{step + 1}"] = {
                        "type": step_type,
                        "output": result,
                        "model_params": model_params if step_type == "model" else None
                    }
                
                # Check chainbreakers to see if we should stop processing
                for breaker in self.chainbreakers:
                    # Check if the breaker accepts step_info parameter
                    sig = inspect.signature(breaker)
                    if len(sig.parameters) >= 3:
                        # New style breaker with step_info
                        should_break, break_reason, break_output = breaker(step + 1, result, step_info)
                    else:
                        # Legacy breaker without step_info
                        should_break, break_reason, break_output = breaker(step + 1, result)
                    
                    if should_break:
                        if self.verbose:
                            print("\n" + "="*50)
                            print(f"⛔ Chain Broken at Step {step + 1}: {break_reason}")
                            print("="*50)
                            if break_output != result:
                                print("\n📊 Modified Output:")
                                print(f"{break_output}\n")
                        
                        # Update the final result if the breaker provided a different output
                        if break_output is not None:
                            result = break_output
                            
                            # Update the chain history with the modified output
                            chain_history[-1]["output"] = result
                            
                            # Update step output if storing steps
                            if self.store_steps:
                                self.step_outputs[f"step_{step + 1}"]["output"] = result
                        
                        # Return early with the current result
                        return result if not self.full_history else chain_history

        except Exception as e:
            if self.verbose:
                print(f"\n❌ Error in chain processing: {str(e)}")
            raise

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
        try:
            model_params = {
                "model": model_name,
                "messages": [{"content": prompt, "role": "user"}]
            }
            
            # Add any custom parameters
            if params:
                model_params.update(params)
            
            response = completion(**model_params)
            return response['choices'][0]['message']['content']
        except Exception as e:
            raise Exception(f"Error running model {model_name}: {str(e)}")

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

class ChainTechnique(str, Enum):
    NORMAL = "normal"
    HEAD_TAIL = "head-tail"

class ChainInstruction(BaseModel):
    instructions: Union[List[Union[str, Callable]], str]
    technique: ChainTechnique
    
    @validator('instructions')
    def validate_instructions(cls, v, values):
        if values.get('technique') == ChainTechnique.NORMAL:
            if not isinstance(v, list):
                raise ValueError("Normal technique requires a list of instructions")
            if len(v) == 0:
                raise ValueError("Instruction list cannot be empty")
            for instruction in v:
                if not isinstance(instruction, (str, Callable)):
                    raise ValueError("Instructions must be strings or callable functions")
        return v

class DynamicChainBuilder:
    def __init__(self, base_model: Union[str, dict], 
                 base_instruction: str,
                 technique: Literal["normal", "head-tail"] = "normal"):
        """
        Initialize a dynamic chain builder with a base model and instruction.
        
        Args:
            base_model: Base model to use for all dynamic chains
            base_instruction: Base instruction template to build upon
            technique: Chain building technique ("normal" or "head-tail")
        """
        self.base_model = base_model
        self.base_instruction = base_instruction
        self.technique = technique
        self.chain_outputs = {}
        self.chain_registry = {}
        self.execution_groups = {}
        self.memory_bank = {}  # Add memory bank for persistent storage
        
        # Validate base instruction template
        self._validate_template(base_instruction)
        
    def create_chain(self, chain_id: str, 
                    instructions: Union[List[Union[str, Callable]], str],
                    execution_mode: Literal["serial", "parallel", "independent"] = "serial",
                    group: str = "default",
                    dependencies: Optional[List[str]] = None) -> PromptChain:
        """
        Create a new chain with the given instructions.
        
        Args:
            chain_id: Unique identifier for this chain
            instructions: List of instructions for normal technique, or string prompt for head-tail
            execution_mode: How this chain should be executed
            group: Execution group for organizing related chains
            dependencies: List of chain_ids this chain depends on
        
        Returns:
            Configured PromptChain instance
        
        Raises:
            ValueError: If validation fails for instruction format
        """
        # Validate instructions using Pydantic model
        chain_instruction = ChainInstruction(
            instructions=instructions,
            technique=self.technique
        )
        
        # Process validated instructions
        if self.technique == "normal":
            validated_instructions = chain_instruction.instructions
        else:  # head-tail
            validated_instructions = [chain_instruction.instructions] if isinstance(chain_instruction.instructions, str) else chain_instruction.instructions
            
        # Create chain with validated instructions
        chain = PromptChain(
            models=[self.base_model] * len(validated_instructions),
            instructions=validated_instructions,
            store_steps=True
        )
        
        # Register chain with metadata
        self.chain_registry[chain_id] = {
            "chain": chain,
            "execution_mode": execution_mode,
            "group": group,
            "dependencies": dependencies or [],
            "status": "created"
        }
        
        # Add to execution group
        if group not in self.execution_groups:
            self.execution_groups[group] = []
        self.execution_groups[group].append(chain_id)
        
        return chain
    
    @staticmethod
    def _validate_template(template: str) -> None:
        """Validate that template has required placeholders."""
        required = ["{instruction}", "{input}"]
        for req in required:
            if req not in template:
                raise ValueError(f"Template missing required placeholder: {req}")

    def execute_chain(self, chain_id: str, input_data: str) -> str:
        """
        Execute a specific chain and store its output.
        
        :param chain_id: ID of chain to execute
        :param input_data: Input data for the chain
        :return: Chain output
        """
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain {chain_id} not found")
            
        chain_info = self.chain_registry[chain_id]
        
        # Check dependencies only for serial chains
        if chain_info["execution_mode"] == "serial":
            for dep_id in chain_info["dependencies"]:
                if dep_id not in self.chain_outputs:
                    raise ValueError(f"Dependency {dep_id} must be executed before {chain_id}")
        
        # Execute chain
        chain_info["status"] = "running"
        result = chain_info["chain"].process_prompt(input_data)
        chain_info["status"] = "completed"
        
        # Store output
        self.chain_outputs[chain_id] = result
        
        return result
        
    def execute_group(self, group: str, input_data: str, 
                     parallel_executor: Callable = None) -> Dict[str, str]:
        """
        Execute all chains in a group respecting their execution modes.
        
        :param group: Group identifier
        :param input_data: Input data for the chains
        :param parallel_executor: Optional function to handle parallel execution
                              Function should accept list of (chain_id, input_data) tuples
        :return: Dictionary of chain outputs
        """
        if group not in self.execution_groups:
            raise ValueError(f"Group {group} not found")
            
        # Organize chains by execution mode
        serial_chains = []
        parallel_chains = []
        independent_chains = []
        
        for chain_id in self.execution_groups[group]:
            mode = self.chain_registry[chain_id]["execution_mode"]
            if mode == "serial":
                serial_chains.append(chain_id)
            elif mode == "parallel":
                parallel_chains.append(chain_id)
            else:  # independent
                independent_chains.append(chain_id)
                
        # Execute independent chains (can be run anytime)
        for chain_id in independent_chains:
            self.execute_chain(chain_id, input_data)
            
        # Execute serial chains in dependency order
        executed = set()
        while serial_chains:
            for chain_id in serial_chains[:]:
                deps = self.chain_registry[chain_id]["dependencies"]
                if all(dep in executed for dep in deps):
                    self.execute_chain(chain_id, input_data)
                    executed.add(chain_id)
                    serial_chains.remove(chain_id)
                    
        # Execute parallel chains
        if parallel_chains:
            if parallel_executor:
                # Use provided parallel executor
                chain_inputs = [(cid, input_data) for cid in parallel_chains]
                results = parallel_executor(chain_inputs)
                for chain_id, result in results.items():
                    self.chain_outputs[chain_id] = result
                    self.chain_registry[chain_id]["status"] = "completed"
            else:
                # Sequential fallback for parallel chains
                for chain_id in parallel_chains:
                    self.execute_chain(chain_id, input_data)
                    
        return {chain_id: self.chain_outputs[chain_id] 
                for chain_id in self.execution_groups[group]
                if chain_id in self.chain_outputs}
                
    def get_chain_output(self, chain_id: str) -> Union[str, None]:
        """Get the output of a previously executed chain."""
        return self.chain_outputs.get(chain_id)
        
    def get_chain_status(self, chain_id: str) -> Union[str, None]:
        """Get the status of a chain (created, running, completed)."""
        return self.chain_registry.get(chain_id, {}).get("status")
        
    def get_group_status(self, group: str) -> Dict[str, str]:
        """Get the status of all chains in a group."""
        if group not in self.execution_groups:
            raise ValueError(f"Group {group} not found")
            
        return {chain_id: self.get_chain_status(chain_id)
                for chain_id in self.execution_groups[group]}
        
    def insert_chain(self, target_chain_id: str, new_instructions: List[str], 
                    position: int = -1) -> None:
        """
        Insert new instructions into an existing chain.
        
        :param target_chain_id: ID of chain to modify
        :param new_instructions: New instructions to insert
        :param position: Position to insert at (-1 for end)
        """
        if target_chain_id not in self.chain_registry:
            raise ValueError(f"Chain {target_chain_id} not found")
            
        chain_info = self.chain_registry[target_chain_id]
        chain = chain_info["chain"]
        
        # Calculate insert position
        if position < 0:
            position = len(chain.instructions)
        
        # Insert new instructions
        chain.instructions[position:position] = new_instructions
        
        # Add corresponding models
        new_models = [self.base_model] * len(new_instructions)
        chain.models[position:position] = new_models
        chain.model_params[position:position] = [{} for _ in range(len(new_instructions))]
        
    def merge_chains(self, chain_ids: List[str], new_chain_id: str,
                    execution_mode: Literal["serial", "parallel", "independent"] = "serial",
                    group: str = None) -> PromptChain:
        """
        Merge multiple chains into a new chain.
        
        :param chain_ids: List of chain IDs to merge
        :param new_chain_id: ID for the merged chain
        :param execution_mode: Execution mode for the new chain
        :param group: Optional group for the new chain
        :return: New merged PromptChain
        """
        if new_chain_id in self.chain_registry:
            raise ValueError(f"Chain {new_chain_id} already exists")
            
        # Collect all instructions and models
        all_instructions = []
        all_models = []
        
        for chain_id in chain_ids:
            if chain_id not in self.chain_registry:
                raise ValueError(f"Chain {chain_id} not found")
                
            chain = self.chain_registry[chain_id]["chain"]
            all_instructions.extend(chain.instructions)
            all_models.extend(chain.models)
        
        # Create new merged chain
        merged_chain = PromptChain(
            models=all_models,
            instructions=all_instructions,
            store_steps=True
        )
        
        # Register merged chain
        self.chain_registry[new_chain_id] = {
            "chain": merged_chain,
            "execution_mode": execution_mode,
            "group": group or "merged",
            "dependencies": chain_ids if execution_mode == "serial" else [],
            "status": "created"
        }
        
        # Add to group if specified
        if group:
            if group not in self.execution_groups:
                self.execution_groups[group] = []
            self.execution_groups[group].append(new_chain_id)
        
        return merged_chain

    def inject_chain(self, target_chain_id: str, source_chain_id: str, 
                    position: int = -1, adjust_dependencies: bool = True) -> None:
        """
        Inject an entire chain into another chain, adjusting steps and dependencies.
        
        :param target_chain_id: ID of chain to inject into
        :param source_chain_id: ID of chain to inject
        :param position: Position to inject at (-1 for end)
        :param adjust_dependencies: Whether to adjust dependencies of subsequent steps
        """
        if target_chain_id not in self.chain_registry:
            raise ValueError(f"Target chain {target_chain_id} not found")
        if source_chain_id not in self.chain_registry:
            raise ValueError(f"Source chain {source_chain_id} not found")
            
        target_chain = self.chain_registry[target_chain_id]["chain"]
        source_chain = self.chain_registry[source_chain_id]["chain"]
        
        # Calculate injection position
        if position < 0:
            position = len(target_chain.instructions)
        
        # Get number of steps being injected
        injection_size = len(source_chain.instructions)
        
        # Store original steps for dependency adjustment
        original_steps = {
            i: step for i, step in enumerate(target_chain.instructions)
        }
        
        # Insert instructions from source chain
        target_chain.instructions[position:position] = source_chain.instructions
        
        # Insert corresponding models
        target_chain.models[position:position] = source_chain.models
        target_chain.model_params[position:position] = source_chain.model_params
        
        # Adjust step storage if enabled
        if target_chain.store_steps:
            # Shift existing step outputs
            new_outputs = {}
            for step_num in sorted(target_chain.step_outputs.keys(), reverse=True):
                if step_num.startswith("step_"):
                    step_idx = int(step_num.split("_")[1])
                    if step_idx >= position:
                        new_step = f"step_{step_idx + injection_size}"
                        new_outputs[new_step] = target_chain.step_outputs[step_num]
                    else:
                        new_outputs[step_num] = target_chain.step_outputs[step_num]
            target_chain.step_outputs = new_outputs
        
        # Update chain registry metadata
        if adjust_dependencies:
            # Adjust dependencies for all chains that depend on steps in the target chain
            for chain_id, chain_info in self.chain_registry.items():
                if chain_id == target_chain_id:
                    continue
                    
                new_deps = []
                for dep in chain_info["dependencies"]:
                    if dep == target_chain_id:
                        # If depending on the whole chain, no adjustment needed
                        new_deps.append(dep)
                    else:
                        # If depending on specific steps, adjust step numbers
                        try:
                            step_num = int(dep.split("_")[1])
                            if step_num >= position:
                                new_step = f"step_{step_num + injection_size}"
                                new_deps.append(new_step)
                            else:
                                new_deps.append(dep)
                        except (IndexError, ValueError):
                            new_deps.append(dep)
                            
                chain_info["dependencies"] = new_deps
    
    def get_step_dependencies(self, chain_id: str) -> Dict[int, List[str]]:
        """
        Get dependencies for each step in a chain.
        
        :param chain_id: Chain ID to analyze
        :return: Dictionary mapping step numbers to their dependencies
        """
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain {chain_id} not found")
            
        chain_info = self.chain_registry[chain_id]
        chain = chain_info["chain"]
        
        step_deps = {}
        for i in range(len(chain.instructions)):
            deps = []
            # Check explicit dependencies
            for dep_chain_id, dep_info in self.chain_registry.items():
                if dep_chain_id == chain_id:
                    continue
                if f"step_{i}" in dep_info["dependencies"]:
                    deps.append(dep_chain_id)
            step_deps[i] = deps
            
        return step_deps
    
    def validate_injection(self, target_chain_id: str, source_chain_id: str, 
                         position: int = -1) -> bool:
        """
        Validate if a chain injection would create circular dependencies.
        
        :param target_chain_id: ID of chain to inject into
        :param source_chain_id: ID of chain to inject
        :param position: Position to inject at
        :return: True if injection is valid, False otherwise
        """
        if target_chain_id not in self.chain_registry:
            raise ValueError(f"Target chain {target_chain_id} not found")
        if source_chain_id not in self.chain_registry:
            raise ValueError(f"Source chain {source_chain_id} not found")
            
        # Check if source chain depends on target chain
        def has_dependency(chain_id, target_id, visited=None):
            if visited is None:
                visited = set()
            if chain_id in visited:
                return False
            visited.add(chain_id)
            
            chain_info = self.chain_registry[chain_id]
            if target_id in chain_info["dependencies"]:
                return True
            
            for dep in chain_info["dependencies"]:
                if dep in self.chain_registry and has_dependency(dep, target_id, visited):
                    return True
            return False
            
        return not has_dependency(source_chain_id, target_chain_id)

    def reorder_steps(self, chain_id: str) -> None:
        """
        Reorder steps in a chain based on dependencies.
        
        :param chain_id: Chain ID to reorder
        """
        if chain_id not in self.chain_registry:
            raise ValueError(f"Chain {chain_id} not found")
            
        chain_info = self.chain_registry[chain_id]
        chain = chain_info["chain"]
        
        # Get step dependencies
        step_deps = self.get_step_dependencies(chain_id)
        
        # Create dependency graph
        from collections import defaultdict
        graph = defaultdict(list)
        for step, deps in step_deps.items():
            for dep in deps:
                graph[dep].append(step)
                
        # Topologically sort steps
        visited = set()
        temp = set()
        order = []
        
        def visit(step):
            if step in temp:
                raise ValueError("Circular dependency detected")
            if step in visited:
                return
            temp.add(step)
            for neighbor in graph[step]:
                visit(neighbor)
            temp.remove(step)
            visited.add(step)
            order.append(step)
            
        for step in range(len(chain.instructions)):
            if step not in visited:
                visit(step)
                
        # Reorder steps based on topological sort
        chain.instructions = [chain.instructions[i] for i in order]
        chain.models = [chain.models[i] for i in order]
        chain.model_params = [chain.model_params[i] for i in order]
        
        # Update step storage if enabled
        if chain.store_steps:
            new_outputs = {}
            for i, step in enumerate(order):
                if f"step_{step}" in chain.step_outputs:
                    new_outputs[f"step_{i}"] = chain.step_outputs[f"step_{step}"]
            chain.step_outputs = new_outputs

    # Add memory bank methods
    def store_memory(self, key: str, value: any, namespace: str = "default") -> None:
        """
        Store a value in memory bank for later retrieval.
        
        Args:
            key: Unique identifier for this memory item
            value: Any value to store
            namespace: Optional grouping namespace (default: "default")
        """
        if namespace not in self.memory_bank:
            self.memory_bank[namespace] = {}
        self.memory_bank[namespace][key] = value
    
    def retrieve_memory(self, key: str, namespace: str = "default", default: any = None) -> any:
        """
        Retrieve a value from memory bank.
        
        Args:
            key: Identifier for the memory item
            namespace: Namespace to look in (default: "default")
            default: Value to return if key not found
            
        Returns:
            Stored value or default if not found
        """
        if namespace not in self.memory_bank:
            return default
        return self.memory_bank[namespace].get(key, default)
    
    def memory_exists(self, key: str, namespace: str = "default") -> bool:
        """
        Check if a memory item exists.
        
        Args:
            key: Memory item identifier
            namespace: Namespace to check in
            
        Returns:
            True if memory exists, False otherwise
        """
        return namespace in self.memory_bank and key in self.memory_bank[namespace]
    
    def list_memories(self, namespace: str = "default") -> List[str]:
        """
        List all memory keys in a namespace.
        
        Args:
            namespace: Namespace to list keys from
            
        Returns:
            List of memory keys
        """
        if namespace not in self.memory_bank:
            return []
        return list(self.memory_bank[namespace].keys())
    
    def clear_memories(self, namespace: str = None) -> None:
        """
        Clear memories in specified namespace or all if none specified.
        
        Args:
            namespace: Namespace to clear or None for all
        """
        if namespace is None:
            self.memory_bank = {}
        elif namespace in self.memory_bank:
            self.memory_bank[namespace] = {}
    
    def create_memory_function(self, namespace: str = "default") -> callable:
        """
        Creates a specialized memory access function for use in chain steps.
        
        Args:
            namespace: Namespace for this memory function
            
        Returns:
            Function that can be used in chain steps to access memory
        """
        def memory_function(input_text: str) -> str:
            """Parse input to store or retrieve from memory bank"""
            parts = input_text.strip().split("\n")
            
            # Command format: MEMORY [STORE|GET] key=value
            results = []
            
            for part in parts:
                if part.upper().startswith("MEMORY"):
                    try:
                        command_parts = part.split()
                        if len(command_parts) >= 3:
                            action = command_parts[1].upper()
                            key_value = " ".join(command_parts[2:])
                            
                            if action == "STORE" and "=" in key_value:
                                key, value = key_value.split("=", 1)
                                self.store_memory(key.strip(), value.strip(), namespace)
                                results.append(f"Stored '{key.strip()}' in memory")
                            elif action == "GET":
                                key = key_value.strip()
                                value = self.retrieve_memory(key, namespace, "Not found")
                                results.append(f"{key} = {value}")
                            elif action == "LIST":
                                keys = self.list_memories(namespace)
                                results.append(f"Memory keys: {', '.join(keys) if keys else 'none'}")
                    except Exception as e:
                        results.append(f"Memory error: {str(e)}")
                else:
                    results.append(part)
                    
            return "\n".join(results)
        
        return memory_function
    
    def create_memory_chain(self, chain_id: str, namespace: str = "default", 
                           instructions: List[str] = None) -> PromptChain:
        """
        Creates a specialized chain with memory access capabilities.
        
        Args:
            chain_id: Unique identifier for this chain
            namespace: Memory namespace for this chain
            instructions: Optional list of instructions (defaults to memory processing)
            
        Returns:
            Configured PromptChain with memory capabilities
        """
        memory_function = self.create_memory_function(namespace)
        
        default_instructions = [
            "Process the following input and update or retrieve from memory as needed: {input}",
            memory_function
        ]
        
        return self.create_chain(
            chain_id=chain_id,
            instructions=instructions or default_instructions,
            execution_mode="independent",
            group="memory_chains"
        )

# Example usage
if __name__ == "__main__":
    # Create builder with base configuration
    builder = DynamicChainBuilder(
        base_model={
            "name": "openai/gpt-4",
            "params": {"temperature": 0.7}
        },
        base_instruction="Base analysis: {input}"
    )
    
    # Create chains with different execution modes
    builder.create_chain(
        "initial",
        ["Extract key points: {input}"],
        execution_mode="serial",
        group="analysis"
    )
    
    builder.create_chain(
        "sentiment",
        ["Analyze sentiment: {input}"],
        execution_mode="parallel",
        group="analysis"
    )
    
    builder.create_chain(
        "keywords",
        ["Extract keywords: {input}"],
        execution_mode="parallel",
        group="analysis"
    )
    
    builder.create_chain(
        "final",
        ["Synthesize findings: {input}"],
        execution_mode="serial",
        dependencies=["initial"],
        group="analysis"
    )
    
    # Execute all chains in the group
    results = builder.execute_group(
        "analysis",
        "This is a test input",
        parallel_executor=None  # Add your parallel execution function here
    )
    
    print("Results:", results)