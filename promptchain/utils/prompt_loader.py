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

def list_available_prompts() -> dict:
    """
    List all available prompts organized by category.
    Returns a dictionary of categories and their prompts with descriptions.
    """
    prompts = load_prompts()
    organized_prompts = {}
    
    for var_name, (category, content) in prompts.items():
        # Extract description from content (first non-empty line after the title)
        lines = content.split('\n')
        description = "No description available"
        for line in lines[1:]:  # Skip title
            if line.strip() and not line.startswith('#'):
                description = line.strip()
                break
        
        # Organize by category
        if category not in organized_prompts:
            organized_prompts[category] = []
            
        organized_prompts[category].append({
            "name": var_name,
            "description": description,
            "path": f"prompts/{category}/{var_name.lower()}.md"
        })
    
    return organized_prompts

def print_available_prompts():
    """Pretty print all available prompts organized by category."""
    prompts = list_available_prompts()
    
    print("\nAvailable Prompts:")
    print("=================")
    
    for category, prompt_list in prompts.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * (len(category) + 1))
        for prompt in prompt_list:
            print(f"\n  {prompt['name']}:")
            print(f"    Description: {prompt['description']}")
            print(f"    Path: {prompt['path']}") 