#!/usr/bin/env python3
"""
Test TUI step numbering with TOOL CALLS to see multiple status updates per step

This tests the actual real-world scenario where AgenticStepProcessor:
- Calls progress_callback multiple times PER STEP:
  1. "Reasoning..."
  2. "Calling: tool_name"
  3. "Synthesizing results..."
  4. "Complete"

All showing the SAME step number (e.g., "[Step 1.1]" repeated 4 times).
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


# Simulate TUI's hierarchical step tracking
class TUIStepTracker:
    """Simulates app.py's _reasoning_progress_callback logic"""

    def __init__(self):
        # These match the variables in app.py
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False
        self.step_log = []  # Track all steps for verification

    def reset_for_new_message(self):
        """Called when starting a new user message (like app.py:2989)"""
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False
        self.step_log = []

    def progress_callback(self, current_step: int, max_steps: int, status: str):
        """
        Simulates app.py:_reasoning_progress_callback (lines 1113-1179)

        This is the EXACT logic from the TUI app.
        """
        # Detect new AgenticStepProcessor instance (lines 1135-1143)
        new_processor_detected = False

        # Debug logging (matches lines 1130-1133)
        print(f"[CALLBACK] Step {current_step}/{max_steps}: {status}")

        # Detection conditions (lines 1135-1143)
        if (current_step < self.last_step_number or
            (current_step == 1 and self.last_step_number == 0) or
            (current_step == 1 and self.processor_completed)):
            self.processor_call_count += 1
            self.processor_completed = False
            new_processor_detected = True
            print(f"[DETECT] New processor #{self.processor_call_count}")

        self.last_step_number = current_step

        # Track completion (lines 1147-1150)
        if status == "Complete":
            self.processor_completed = True
            print(f"[COMPLETE] Processor {self.processor_call_count} finished")

        # Format hierarchical step number (line 1153)
        hierarchical_step = f"{self.processor_call_count}.{current_step}"

        # THIS IS WHAT THE USER SEES IN OBSERVEPANEL:
        display_message = f"[Step {hierarchical_step}] {status}"
        print(f"[DISPLAY] {display_message}")

        # Log to our tracking list
        self.step_log.append({
            "hierarchical": hierarchical_step,
            "processor": self.processor_call_count,
            "step": current_step,
            "status": status,
            "display": display_message,
            "new_processor": new_processor_detected
        })


# Simple search tool
def search_files(query: str) -> str:
    """Mock search tool that returns results"""
    return f"Found 3 files matching '{query}': file1.py, file2.py, file3.py"


async def test_with_tools():
    """Test with AgenticStepProcessor that USES TOOLS"""
    print("\n" + "="*80)
    print("TUI Step Numbering Test - WITH TOOL CALLS")
    print("="*80)
    print("\nThis shows how multiple status updates PER STEP create many '[Step 1.1]' entries")
    print("="*80)

    # Create tracker (simulates TUI app state)
    tracker = TUIStepTracker()
    tracker.reset_for_new_message()

    # Create AgenticStepProcessor with tool access
    agentic_step = AgenticStepProcessor(
        objective="Search for Python files and analyze their contents",
        max_internal_steps=2,  # Allow up to 2 steps
        model_name="openai/gpt-4o-mini",
        progress_callback=tracker.progress_callback
    )

    # Create chain with tool
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[agentic_step],
        verbose=True
    )

    # Register tool
    chain.register_tool_function(search_files)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }])

    # Process
    print("\n[TEST] Processing with tool-enabled AgenticStepProcessor...")
    print("-" * 80)
    result = await chain.process_prompt_async("Find Python files related to authentication")

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    # Analyze step log
    print(f"\nTotal callback invocations: {len(tracker.step_log)}")

    # Group by step number
    steps = {}
    for entry in tracker.step_log:
        step_key = entry["hierarchical"]
        if step_key not in steps:
            steps[step_key] = []
        steps[step_key].append(entry)

    print(f"\nUnique steps detected: {len(steps)}")

    for step_key in sorted(steps.keys()):
        entries = steps[step_key]
        print(f"\n{step_key}: {len(entries)} callback invocations")
        for entry in entries:
            print(f"  - {entry['display']}")

    # Show the issue
    print("\n" + "="*80)
    print("ISSUE DEMONSTRATION")
    print("="*80)

    print("\nWhat the user sees in ObservePanel:")
    print("-" * 80)
    for entry in tracker.step_log:
        print(entry['display'])

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Check if we have multiple statuses for the same step
    for step_key, entries in steps.items():
        if len(entries) > 2:  # More than just start and complete
            print(f"\n⚠️  ISSUE: Step {step_key} has {len(entries)} status updates:")
            for entry in entries:
                print(f"    - {entry['status']}")
            print(f"\n   This creates {len(entries)} entries all labeled '[Step {step_key}]'")
            print(f"   Making it appear as if the step number never increments!")

    return tracker.step_log


if __name__ == "__main__":
    step_log = asyncio.run(test_with_tools())

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
The hierarchical numbering LOGIC is correct (1.1, 1.2, 2.1, 2.2).

However, each STEP has multiple status updates:
- "Reasoning..."
- "Calling: tool_name"
- "Synthesizing results..."
- "Complete"

All showing the SAME step number, creating visual confusion.

RECOMMENDATION:
- Only show step number for FIRST callback of each unique step
- Show subsequent status updates as indented sub-items
- OR filter ObservePanel to only show key events, not every status update
""")
    sys.exit(0)
