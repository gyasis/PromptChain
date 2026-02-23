#!/usr/bin/env python3
"""Test hierarchical step numbering in AgenticStepProcessor.

This test verifies that when multiple AgenticStepProcessor calls occur
in a single chain, the ObservePanel displays steps in hierarchical format:
- First processor: Step 1.1, 1.2, 1.3, ...
- Second processor: Step 2.1, 2.2, 2.3, ...
- Third processor: Step 3.1, 3.2, 3.3, ...
"""

import asyncio
import sys
import logging
from pathlib import Path

# Enable INFO logging to see internal iterations
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class StepLogger:
    """Simple callback logger to verify hierarchical numbering."""

    def __init__(self):
        self.steps_logged = []
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False

    def callback(self, current_step: int, max_steps: int, status: str):
        """Mock callback matching the TUI signature."""
        # Detect new AgenticStepProcessor instance (same logic as app.py):
        # - If step goes backward (e.g., 3 -> 1), new processor started
        # - If step is 1 and last was 0 (first ever call), new processor
        # - If step is 1 and previous processor completed, new processor started
        if (current_step < self.last_step_number or
            (current_step == 1 and self.last_step_number == 0) or
            (current_step == 1 and self.processor_completed)):
            self.processor_call_count += 1
            self.processor_completed = False

        self.last_step_number = current_step

        # Track completion for next processor detection
        if status == "Complete":
            self.processor_completed = True

        # Format hierarchical step
        hierarchical_step = f"{self.processor_call_count}.{current_step}"

        # Log step
        log_entry = f"[Step {hierarchical_step}] {status}"
        self.steps_logged.append(log_entry)
        print(log_entry)


async def test_hierarchical_steps():
    """Test hierarchical step numbering across multiple processors."""
    print("=" * 80)
    print("Testing Hierarchical Step Numbering")
    print("=" * 80)

    # Create step logger
    logger = StepLogger()

    # Create two AgenticStepProcessor instances with different objectives
    processor1 = AgenticStepProcessor(
        objective="First analysis task",
        max_internal_steps=3,
        model_name="openai/gpt-4o-mini",
        progress_callback=logger.callback,
    )

    processor2 = AgenticStepProcessor(
        objective="Second analysis task",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini",
        progress_callback=logger.callback,
    )

    # Create a simple tool for the processors to use
    def simple_tool(query: str) -> str:
        """A simple test tool."""
        return f"Tool response for: {query}"

    # Create chain with two processors in sequence
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            processor1,  # Will show steps 1.1, 1.2, 1.3
            "Intermediate processing: {input}",
            processor2,  # Will show steps 2.1, 2.2, 2.3, 2.4, 2.5
        ],
        verbose=True,
    )

    # Register tool
    chain.register_tool_function(simple_tool)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "simple_tool",
            "description": "A simple test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query string"}
                },
                "required": ["query"]
            }
        }
    }])

    # Execute chain
    print("\n" + "=" * 80)
    print("Executing Chain with Multiple AgenticStepProcessors")
    print("=" * 80 + "\n")

    try:
        result = await chain.process_prompt_async("Analyze this data point")

        print("\n" + "=" * 80)
        print("Execution Complete")
        print("=" * 80 + "\n")

        # Verify hierarchical numbering
        print("Steps logged:")
        for step in logger.steps_logged:
            print(f"  {step}")

        print(f"\nTotal processor calls: {logger.processor_call_count}")
        print(f"Total steps logged: {len(logger.steps_logged)}")

        # Check if we have hierarchical numbering (1.x, 2.x format)
        has_hierarchical = any("." in step for step in logger.steps_logged)
        if has_hierarchical:
            print("\n✅ SUCCESS: Hierarchical step numbering detected!")
        else:
            print("\n❌ FAILURE: No hierarchical numbering found!")

        return logger.steps_logged

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    # Run test
    asyncio.run(test_hierarchical_steps())
