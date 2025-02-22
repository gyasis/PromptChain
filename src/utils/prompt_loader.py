import os
from typing import Dict, Tuple

def load_prompts(prompts_dir: str = "src/prompts") -> Dict[str, Tuple[str, str]]:
    """
    Automatically load all prompts from the prompts directory structure.
    Returns a dictionary of (category, prompt_text) tuples keyed by uppercase variable names.
    """
    prompts = {}
    
    for root, _, files in os.walk(prompts_dir):
        for file in files:
            if file.endswith('.md'):
                # Get relative path components
                rel_path = os.path.relpath(root, prompts_dir)
                category = rel_path.replace(os.sep, '_')
                
                # Create variable name from filename
                var_name = os.path.splitext(file)[0].upper()
                if category != '.':
                    var_name = f"{category.upper()}_{var_name}"
                
                # Load prompt content
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                
                # Store as tuple of (category, content)
                prompts[var_name] = (category, content)
    
    return prompts

# Example usage in a chain:
def get_prompt_by_name(name: str) -> str:
    """Get prompt content by its variable name"""
    prompts = load_prompts()
    if name in prompts:
        return prompts[name][1]
    raise ValueError(f"Prompt {name} not found") 