#!/usr/bin/env python3
"""Test script to reproduce PromptChain preprompt warnings"""

import logging
import sys

# Configure logging to show warnings
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

print("Testing PromptChain preprompt warnings...")

try:
    from promptchain import PromptChain
    print("PromptChain imported successfully")
    
    # Try to initialize PromptChain - this might trigger preprompt initialization
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Test prompt: {input}"],
        verbose=False
    )
    print("PromptChain initialized successfully")
    
    # Try to trigger preprompt usage if any
    from promptchain.utils.preprompt import PrePrompt
    print("PrePrompt module imported")
    
    preprompt = PrePrompt()
    print("PrePrompt initialized - warnings should appear above if directories are missing")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()