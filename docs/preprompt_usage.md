# Using the PrePrompt Class

The `PrePrompt` class (`promptchain.utils.preprompt.PrePrompt`) is a utility designed to simplify loading prompt templates within the PromptChain ecosystem. It handles:

1.  **Loading by ID:** Finding prompt files based on their filename (without extension) within configured directories.
2.  **Strategy Application:** Optionally prepending reusable "strategy" instructions (loaded from a standard location) to a base prompt.
3.  **Custom Locations:** Searching for prompts in user-defined directories in addition to the standard library location.
4.  **Prioritization:** Ensuring prompts found in custom locations override standard prompts if they share the same ID.

## Initialization

You create an instance of the class, optionally providing a list of paths to your custom prompt directories:

```python
from promptchain.utils.preprompt import PrePrompt
import os

# Example directories (assuming they exist)
project_root = "/path/to/your/project" # Replace with actual path
custom_prompts_dir1 = os.path.join(project_root, "my_prompts")
custom_prompts_dir2 = os.path.join(project_root, "shared_prompts")

# 1. Using only the standard library prompts ('promptchain/prompts/')
preprompter_std = PrePrompt()

# 2. Adding one custom directory
preprompter_custom1 = PrePrompt(additional_prompt_dirs=[custom_prompts_dir1])

# 3. Adding multiple custom directories (searched in order)
preprompter_custom2 = PrePrompt(additional_prompt_dirs=[custom_prompts_dir1, custom_prompts_dir2])
```

**Directory Structure Assumption:**

*   **Standard Prompts:** Located inside the installed library at `.../site-packages/promptchain/prompts/`. Files like `example_base.txt` would be loaded by ID `example_base`.
*   **Standard Strategies:** Located at `.../site-packages/promptchain/prompts/strategies/`. Files like `concise.json` would be loaded by ID `concise`. Strategies *must* be JSON files containing a `"prompt"` key with the strategy text.
*   **Custom Prompts:** Can be located anywhere you specify in `additional_prompt_dirs`. For example, a file at `/path/to/your/project/my_prompts/custom_analysis.md` would be loaded by ID `custom_analysis`.

**Supported Prompt Extensions:** `.txt`, `.md`, `.xml`, `.json`

**Prioritization:** When scanning for prompts, `PrePrompt` searches `additional_prompt_dirs` in the order they are provided, followed by the standard library directory. The *first* prompt file found for a given ID (filename without extension) is used. This means prompts in your custom directories will override standard library prompts with the same name.

**Strategies:** Strategies are *always* loaded from the standard library location (`promptchain/prompts/strategies/`) regardless of where the base prompt was found.

## Loading Prompts

Use the `.load()` method with an instruction string:

*   `"promptID"`: Loads the base prompt corresponding to `promptID`.
*   `"promptID:strategyID"`: Loads the base prompt for `promptID` and prepends the strategy prompt found in `strategyID.json`.

```python
# Assume 'example_base.txt' exists in standard prompts
# Assume 'concise.json' exists in standard strategies
# Assume 'custom_analysis.md' exists in custom_prompts_dir1
# Assume 'example_base.md' also exists in custom_prompts_dir1 (overrides standard)

try:
    # Example 1: Load standard prompt by ID
    standard_prompt = preprompter_std.load("example_base")
    print("Loaded standard 'example_base':\n", standard_prompt)

    # Example 2: Load standard prompt with standard strategy
    concise_prompt = preprompter_std.load("example_base:concise")
    print("\nLoaded standard 'example_base:concise':\n", concise_prompt)

    # --- Using preprompter_custom1 (includes custom_prompts_dir1) ---

    # Example 3: Load custom prompt by ID
    custom_prompt = preprompter_custom1.load("custom_analysis")
    print("\nLoaded custom 'custom_analysis':\n", custom_prompt)

    # Example 4: Load custom prompt with standard strategy
    custom_concise = preprompter_custom1.load("custom_analysis:concise")
    print("\nLoaded custom 'custom_analysis:concise':\n", custom_concise)

    # Example 5: Load prompt overridden by custom directory
    # This will load 'example_base.md' from custom_prompts_dir1, not 'example_base.txt'
    overridden_prompt = preprompter_custom1.load("example_base")
    print("\nLoaded overridden 'example_base' (from custom dir):\n", overridden_prompt)

    # Example 6: Load overridden prompt with standard strategy
    overridden_concise = preprompter_custom1.load("example_base:concise")
    print("\nLoaded overridden 'example_base:concise':\n", overridden_concise)

    # Example 7: Non-existent prompt ID
    # non_existent = preprompter_std.load("does_not_exist") # Raises FileNotFoundError

    # Example 8: Non-existent strategy ID
    # bad_strategy = preprompter_std.load("example_base:no_such_strategy") # Raises FileNotFoundError

except FileNotFoundError as e:
    print(f"\nError loading prompt/strategy: {e}")
except ValueError as e:
    print(f"\nError in format: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

```

## Integration with PromptChain

The `PromptChain` class now uses an internal instance of `PrePrompt` to resolve string instructions.

When you initialize `PromptChain`, you can pass your custom prompt directories via the `additional_prompt_dirs` parameter. This configures the internal `PrePrompt` instance used by that chain.

```python
from promptchain import PromptChain
import os

# Assume custom prompts are in './my_prompts'
# Assume standard prompts/strategies exist in the library installation

my_prompts_path = os.path.abspath("./my_prompts") # Example path

def simple_cleaner(text: str) -> str:
    return text.strip().lower()

# Initialize PromptChain, configuring its internal PrePrompt
chain = PromptChain(
    models=["mock-response"] * 3, # Adjust model count as needed
    instructions=[
        "example_base",         # Loaded via PrePrompt (standard or custom)
        "custom_analysis:concise", # Loaded via PrePrompt (custom + standard strategy)
        simple_cleaner,         # Callable function
        "./local_specific_prompt.txt", # Loaded directly as file path
        "Literal template: {input}" # Treated as literal template
    ],
    additional_prompt_dirs=[my_prompts_path],
    verbose=True
)

# When chain.process_prompt() runs:
# - "example_base" will be loaded by the internal PrePrompt,
#   searching my_prompts_path first, then the standard prompts dir.
# - "custom_analysis:concise" will load 'custom_analysis' (from my_prompts_path)
#   and prepend the 'concise' strategy (from standard strategies dir).
# - simple_cleaner is executed directly.
# - "./local_specific_prompt.txt" is recognized as a file path and loaded.
# - "Literal template: {input}" is used as is because PrePrompt won't find an ID
#   and it's not a valid file path.

try:
    result = chain.process_prompt("Some initial data.")
    print("\nFinal Chain Result:", result)
except FileNotFoundError as e:
    print(f"\nError: A required prompt or strategy file was not found: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

```

This integration means you don't typically need to create separate `PrePrompt` instances yourself when using `PromptChain`. Simply pass the necessary `additional_prompt_dirs` during `PromptChain` initialization. 