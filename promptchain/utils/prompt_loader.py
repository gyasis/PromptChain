import os
from typing import Dict, Tuple

def load_prompts(prompts_dir: str = None) -> Dict[str, Tuple[str, str]]:
    """
    Automatically load all prompts from the prompts directory structure.
    Returns a dictionary of (category, prompt_text) tuples keyed by uppercase variable names.
    """
    if prompts_dir is None:
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts") 