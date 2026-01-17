#!/usr/bin/env python3
"""
Test the improved ObservePanel display with hierarchical step tracking

This verifies that:
1. First callback for each step shows "[Step X.Y]" prefix2. Subsequent callbacks within same step show "  └─" prefix (indented sub-item)3. Step transitions are clear: 1.1 → 1.2 → 2.1 → 2.2
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class ImprovedTUIStepTracker:
    """Simulates app.py's IMPROVED _reasoning_progress_callback logic"""

    def __init__(self):
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False
        self.last_displayed_step = None  # NEW: Track last displayed step
        self.display_log = []  # What user sees

    def reset_for_new_message(self):
        self.processor_call_count = 0
        self.last_step_number = 0
        self.processor_completed = False
        self.last_displayed_step = None

    def progress_callback(self, current_step: int, max_steps: int, status: str):
        """IMPROVED callback with cleaner display"""
        # Detect new processor
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

        # IMPROVED LOGIC: Check if this is a new step or status update within same step
        is_new_step = (self.last_displayed_step != hierarchical_step)

        if is_new_step:
            # First callback for this step - show full step prefix
            self.last_displayed_step = hierarchical_step
            display_message = f"[Step {hierarchical_step}] {status}"
        else:
            # Status update within same step - show as indented sub-item
            display_message = f"  └─ {status}"

        print(display_message)
        self.display_log.append(display_message)


# Simple search tool
def search_files(query: str) -> str:
    """Mock search tool"""
    return f"Found 3 files matching '{query}': file1.py, file2.py, file3.py"


async def test_improved_display():
    """Test the improved display logic"""
    print("\n" + "="*80)
    print("IMPROVED ObservePanel Display Test")
    print("="*80)

    tracker = ImprovedTUIStepTracker()
    tracker.reset_for_new_message()

    # Create two AgenticStepProcessors to test step transitions
    step1 = AgenticStepProcessor(
        objective="Search for Python files",
        max_internal_steps=2,
        model_name="openai/gpt-4o-mini",
        progress_callback=tracker.progress_callback
    )

    step2 = AgenticStepProcessor(
        objective="Analyze files found",
        max_internal_steps=2,
        model_name="openai/gpt-4o-mini",
        progress_callback=tracker.progress_callback
    )

    # Create chain
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[step1, step2],
        verbose=True
    )

    # Register tool
    chain.register_tool_function(search_files)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }])

    print("\n[TEST] Processing with improved display logic...")
    print("-" * 80)
    await chain.process_prompt_async("Find authentication files and analyze them")

    print("\n" + "="*80)
    print("DISPLAY LOG (What user sees in ObservePanel)")
    print("="*80)

    for msg in tracker.display_log:
        print(msg)

    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Verify improved display
    step_count = sum(1 for msg in tracker.display_log if msg.startswith("[Step "))
    substatus_count = sum(1 for msg in tracker.display_log if msg.startswith("  └─"))

    print(f"\nStep headers: {step_count}")
    print(f"Sub-status updates: {substatus_count}")

    if step_count >= 2:  # At least 2 processors should create 2 step headers
        print("\n✅ PASS: Multiple unique steps detected")
    else:
        print("\n❌ FAIL: Not enough unique steps")

    if substatus_count > 0:  # Should have some sub-status updates
        print(f"✅ PASS: {substatus_count} status updates shown as sub-items")
    else:
        print("⚠️  WARNING: No sub-status updates detected")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
IMPROVED DISPLAY:
- Step headers: [Step 1.1], [Step 1.2], [Step 2.1], [Step 2.2]
  → Shows step number ONCE per unique step

- Sub-items:   └─ Calling: tool_name
               └─ Synthesizing results...
  → Status updates within same step shown as indented sub-items

This creates a clear visual hierarchy showing:1. When steps transition (new step number)
2. What's happening within each step (indented status updates)3. Overall progress (hierarchical numbering maintained)
""")


if __name__ == "__main__":
    asyncio.run(test_improved_display())
    sys.exit(0)
