#!/usr/bin/env python3
"""Test hierarchical numbering with multiple internal steps per processor.

Verifies format like:
- Processor 1: 1.1, 1.2, 1.3
- Processor 2: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import asyncio
import sys
import logging
from pathlib import Path

# Enable INFO logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class StepLogger:
    """Callback logger matching TUI implementation."""

    def __init__(self):
        self.steps_logged = []
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False

    def callback(self, current_step: int, max_steps: int, status: str):
        """Progress callback with hierarchical numbering logic."""
        # Detect new processor (same logic as app.py)
        if (current_step < self.last_step_number or
            (current_step == 1 and self.last_step_number == 0) or
            (current_step == 1 and self.processor_completed)):
            self.processor_call_count += 1
            self.processor_completed = False

        self.last_step_number = current_step

        if status == "Complete":
            self.processor_completed = True

        # Format hierarchical step
        hierarchical_step = f"{self.processor_call_count}.{current_step}"
        log_entry = f"[Step {hierarchical_step}] {status}"
        self.steps_logged.append(log_entry)
        print(log_entry)


async def test_multi_step_processors():
    """Test with processors that actually use multiple internal steps."""
    print("=" * 80)
    print("Testing Multi-Step Hierarchical Numbering")
    print("=" * 80)

    logger = StepLogger()

    # Create a tool that requires multiple reasoning steps
    def analyze_data(data: str) -> str:
        """Analyze data and return structured result."""
        return f"Analysis complete for: {data}. Found 3 patterns: A, B, C"

    def validate_results(analysis: str) -> str:
        """Validate analysis results."""
        return f"Validation passed: {analysis}"

    # Processor 1: Should take 2-3 steps
    processor1 = AgenticStepProcessor(
        objective="Analyze the input data thoroughly, calling analyze_data tool",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini",
        progress_callback=logger.callback,
    )

    # Processor 2: Should take 3-4 steps
    processor2 = AgenticStepProcessor(
        objective="Validate the analysis results, calling validate_results tool",
        max_internal_steps=5,
        model_name="openai/gpt-4o-mini",
        progress_callback=logger.callback,
    )

    # Create chain with both processors
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            processor1,
            "Process results: {input}",
            processor2,
        ],
        verbose=True,
    )

    # Register tools
    chain.register_tool_function(analyze_data)
    chain.register_tool_function(validate_results)
    chain.add_tools([
        {
            "type": "function",
            "function": {
                "name": "analyze_data",
                "description": "Analyze data and return structured result",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Data to analyze"}
                    },
                    "required": ["data"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "validate_results",
                "description": "Validate analysis results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string", "description": "Analysis to validate"}
                    },
                    "required": ["analysis"]
                }
            }
        }
    ])

    # Execute
    print("\n" + "=" * 80)
    print("Executing Multi-Step Chain")
    print("=" * 80 + "\n")

    try:
        result = await chain.process_prompt_async("Analyze this dataset: [1,2,3,4,5]")

        print("\n" + "=" * 80)
        print("Steps Logged:")
        print("=" * 80)
        for step in logger.steps_logged:
            print(f"  {step}")

        print(f"\nTotal processors: {logger.processor_call_count}")
        print(f"Total steps: {len(logger.steps_logged)}")

        # Verify hierarchical format
        unique_steps = set(step.split("]")[0].replace("[Step ", "") for step in logger.steps_logged)
        print(f"Unique step numbers: {sorted(unique_steps)}")

        # Check for proper hierarchical numbering
        has_multiple_processors = any(step.startswith("2.") for step in unique_steps)
        has_multiple_internal_steps = any(step.endswith(".2") or step.endswith(".3") for step in unique_steps)

        if has_multiple_processors:
            print("\n✅ SUCCESS: Multiple processors detected (1.x, 2.x)")
        else:
            print("\n⚠️  WARNING: Only one processor detected")

        if has_multiple_internal_steps:
            print("✅ SUCCESS: Multiple internal steps detected (x.1, x.2, x.3)")
        else:
            print("⚠️  INFO: Processors completed in single step (not a failure)")

        return logger.steps_logged

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


if __name__ == "__main__":
    asyncio.run(test_multi_step_processors())
