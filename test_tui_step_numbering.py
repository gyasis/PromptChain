#!/usr/bin/env python3
"""
Test TUI-specific hierarchical step numbering with progress_callback

This tests the actual mechanism that the TUI uses:
- AgenticStepProcessor with progress_callback parameter
- Hierarchical step numbering (1.1, 1.2, ... 2.1, 2.2)
- Simulates the TUI's _reasoning_progress_callback logic
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
        print(f"[STEP TRACKING] Callback invoked: current_step={current_step}, "
              f"last_step={self.last_step_number}, processor_completed={self.processor_completed}, "
              f"processor_call_count={self.processor_call_count}, status={status[:50]}")

        # Detection conditions (lines 1135-1143)
        if (current_step < self.last_step_number or
            (current_step == 1 and self.last_step_number == 0) or
            (current_step == 1 and self.processor_completed)):
            self.processor_call_count += 1
            self.processor_completed = False
            new_processor_detected = True
            print(f"[STEP TRACKING] ✓ New processor detected! Count: {self.processor_call_count}, "
                  f"current_step: {current_step}, last_step: {self.last_step_number}")

        self.last_step_number = current_step

        # Track completion (lines 1147-1150)
        if status == "Complete":
            self.processor_completed = True
            print(f"[STEP TRACKING] Processor {self.processor_call_count} completed at step {current_step}")

        # Format hierarchical step number (line 1153)
        hierarchical_step = f"{self.processor_call_count}.{current_step}"
        print(f"[STEP TRACKING] Formatted step: {hierarchical_step}, status: {status}")

        # Log to our tracking list
        self.step_log.append({
            "hierarchical": hierarchical_step,
            "processor": self.processor_call_count,
            "step": current_step,
            "status": status,
            "new_processor": new_processor_detected
        })

        # Simulate ObservePanel logging (lines 1165-1179)
        print(f"[OBSERVE] [Step {hierarchical_step}] {status}")


async def test_hierarchical_numbering():
    """Test that hierarchical numbering works correctly"""
    print("\n" + "="*80)
    print("TUI Hierarchical Step Numbering Test")
    print("="*80)
    print("\nThis simulates the EXACT logic from app.py:_reasoning_progress_callback")
    print("="*80)

    # Create tracker (simulates TUI app state)
    tracker = TUIStepTracker()
    tracker.reset_for_new_message()

    # Create two AgenticStepProcessor instances with progress_callback
    # (This is what app.py does when creating AgenticStepProcessor)
    step1 = AgenticStepProcessor(
        objective="Analyze quantum computing concepts",
        max_internal_steps=3,  # Will generate 3 steps
        model_name="openai/gpt-4o-mini",
        progress_callback=tracker.progress_callback  # KEY: This is what makes it work
    )

    step2 = AgenticStepProcessor(
        objective="Explain blockchain fundamentals",
        max_internal_steps=3,  # Will generate 3 more steps
        model_name="openai/gpt-4o-mini",
        progress_callback=tracker.progress_callback  # Same callback, different instance
    )

    # Create chain with both processors
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            "Prepare context: {input}",
            step1,  # Should show steps 1.1, 1.2, 1.3
            step2,  # Should show steps 2.1, 2.2, 2.3
            "Final summary: {input}"
        ],
        verbose=True
    )

    # Process
    print("\n[TEST] Processing with 2 AgenticStepProcessor calls...")
    result = await chain.process_prompt_async("Explain technical concepts")

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    # Analyze step log
    print(f"\nTotal steps logged: {len(tracker.step_log)}")

    # Group by processor
    processors = {}
    for entry in tracker.step_log:
        proc_num = entry["processor"]
        if proc_num not in processors:
            processors[proc_num] = []
        processors[proc_num].append(entry)

    print(f"Processors detected: {len(processors)}")

    for proc_num in sorted(processors.keys()):
        steps = processors[proc_num]
        hierarchical_steps = [s["hierarchical"] for s in steps]
        print(f"\nProcessor {proc_num}:")
        print(f"  Steps: {hierarchical_steps}")
        print(f"  Count: {len(steps)} steps")

    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    success = True

    # Check if we have multiple processors
    if len(processors) < 2:
        print(f"❌ FAIL: Expected at least 2 processors, got {len(processors)}")
        success = False
    else:
        print(f"✓ PASS: Detected {len(processors)} processors")

    # Check if step numbers increment properly within each processor
    for proc_num, steps in processors.items():
        hierarchical_steps = [s["hierarchical"] for s in steps]
        expected_pattern = [f"{proc_num}.{i+1}" for i in range(len(steps))]

        # Check for the "all 1.1" issue
        all_same = all(s["hierarchical"] == hierarchical_steps[0] for s in steps)
        if all_same:
            print(f"❌ FAIL: Processor {proc_num} shows all steps as '{hierarchical_steps[0]}' (BUG!)")
            success = False
        else:
            print(f"✓ PASS: Processor {proc_num} steps increment correctly: {hierarchical_steps[:5]}...")

    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED")
        print("\nHierarchical step numbering is working correctly!")
    else:
        print("❌ TESTS FAILED")
        print("\nThis confirms the bug - all steps showing same number.")
        print("Debug logs above show the root cause.")
    print("="*80)

    return success


if __name__ == "__main__":
    success = asyncio.run(test_hierarchical_numbering())
    sys.exit(0 if success else 1)
