#!/usr/bin/env python3
"""
Execution trace demonstrating dual observability system behavior.

This script simulates the exact import and execution sequence that occurs
when the PromptChain CLI starts up and processes an LLM call, showing
where tracking is activated or bypassed.

Run this to see the execution flow:
    python observability_execution_trace.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def trace_import_sequence():
    """Trace the import sequence and show where _ENABLED is set."""
    print("=" * 80)
    print("IMPORT SEQUENCE TRACE")
    print("=" * 80)

    print("\n[STEP 1] CLI imports observability module")
    print("File: promptchain/cli/main.py:19")
    print("Code: from promptchain.observability import track_session, init_mlflow, shutdown_mlflow")

    print("\n[STEP 2] Observability __init__.py imports decorators")
    print("File: promptchain/observability/__init__.py:21-28")
    print("Code: from .decorators import (track_llm_call, track_task, ...)")

    print("\n[STEP 3] Decorators imports ghost pattern")
    print("File: promptchain/observability/decorators.py:44")
    print("Code: from .ghost import conditional_decorator")

    print("\n[STEP 4] Ghost pattern evaluates _ENABLED at IMPORT TIME")
    print("File: promptchain/observability/ghost.py:17")
    print("Code: _ENABLED = is_enabled()")

    # Actual check
    try:
        from promptchain.observability.config import is_enabled, DEFAULT_MLFLOW_ENABLED
        from promptchain.observability.ghost import _ENABLED

        print(f"\n[RESULT] Import-time evaluation:")
        print(f"  DEFAULT_MLFLOW_ENABLED = {DEFAULT_MLFLOW_ENABLED}")
        print(f"  ENV PROMPTCHAIN_MLFLOW_ENABLED = {os.getenv('PROMPTCHAIN_MLFLOW_ENABLED', '(not set)')}")
        print(f"  is_enabled() returned: {is_enabled()}")
        print(f"  _ENABLED cached value: {_ENABLED}")

        if not _ENABLED:
            print("\n  ⚠ MLflow tracking DISABLED - ghost decorators will bypass all tracking")
        else:
            print("\n  ✓ MLflow tracking ENABLED - decorators will track to MLflow")

    except ImportError as e:
        print(f"\n[ERROR] Failed to import observability modules: {e}")

    print("\n[STEP 5] PromptChain imports and applies decorator")
    print("File: promptchain/utils/promptchaining.py:14")
    print("Code: from promptchain.observability import track_llm_call")
    print("\nFile: promptchain/utils/promptchaining.py:1834-1837")
    print("Code:")
    print("    @track_llm_call(model_param='model_name', extract_args=[...])")
    print("    async def run_model_async(self, model_name, messages, ...):")

    print("\n[STEP 6] Decorator application via conditional_decorator")
    print("File: promptchain/observability/decorators.py:226")
    print("Code: return conditional_decorator(decorator)")

    if not _ENABLED:
        print("\nFile: promptchain/observability/ghost.py:68-71")
        print("Code:")
        print("    if _ENABLED:  # False")
        print("        return tracking_decorator")
        print("    else:")
        print("        return make_ghost_decorator()  # ← THIS PATH TAKEN")
        print("\nFile: promptchain/observability/ghost.py:41-42")
        print("Code:")
        print("    def ghost(func):")
        print("        return func  # RETURNS ORIGINAL FUNCTION UNCHANGED")
        print("\n  → @track_llm_call becomes identity decorator (no-op)")
    else:
        print("\n  → @track_llm_call will wrap function with MLflow tracking")


def trace_execution_flow():
    """Trace the execution flow when an LLM call is made."""
    print("\n\n" + "=" * 80)
    print("EXECUTION FLOW TRACE (User sends message)")
    print("=" * 80)

    try:
        from promptchain.observability.ghost import _ENABLED
    except ImportError:
        _ENABLED = False

    print("\n[STEP 1] User sends message in CLI")
    print("Flow: User input → PromptChainApp.on_message_submit")

    print("\n[STEP 2] Message routed to agent")
    print("Flow: → AgentChain.send_message")

    print("\n[STEP 3] Chain processes prompt")
    print("Flow: → PromptChain.process_prompt_async")

    print("\n[STEP 4] LLM call initiated")
    print("Flow: → PromptChain.run_model_async")
    print("Decorator: @track_llm_call")

    if not _ENABLED:
        print("\n[DECORATOR BEHAVIOR]")
        print("  Ghost decorator active: YES")
        print("  MLflow tracking code executed: NO")
        print("  Function wrapper added: NO")
        print("  run_model_async() executes as if undecorated")

        print("\n[EVENT FLOW]")
        print("  Events to MLflow: NONE (decorator bypassed)")
        print("  Events to CallbackManager: POTENTIALLY (if callbacks registered)")
        print("  Actual callbacks registered: NONE (CLI doesn't register any)")

        print("\n[RESULT]")
        print("  MLflow UI: No runs created")
        print("  ObservePanel: 0/0 activities")
        print("  ActivityLogViewer: Empty")
    else:
        print("\n[DECORATOR BEHAVIOR]")
        print("  Ghost decorator active: NO")
        print("  MLflow tracking code executed: YES")
        print("  Function wrapper added: YES")
        print("  Metrics logged to MLflow")

        print("\n[EVENT FLOW]")
        print("  Events to MLflow: Metrics, params, tags logged")
        print("  Events to CallbackManager: POTENTIALLY (if callbacks registered)")

        print("\n[RESULT]")
        print("  MLflow UI: Runs visible with metrics")
        print("  ObservePanel: Shows MLflow runs (if integration exists)")

    print("\n[STEP 5] LiteLLM executes actual API call")
    print("Flow: → litellm.acompletion(**model_params)")

    print("\n[STEP 6] Response returned")
    print("Flow: → Response bubbles back through chain")


def trace_callback_system():
    """Trace the callback system initialization and usage."""
    print("\n\n" + "=" * 80)
    print("CALLBACK SYSTEM TRACE")
    print("=" * 80)

    print("\n[INITIALIZATION]")
    print("File: promptchain/utils/promptchaining.py:234")
    print("Code: self.callback_manager = CallbackManager()")
    print("Result: CallbackManager instance created with empty callbacks list")

    print("\n[REGISTRATION CHECK]")
    print("File: promptchain/cli/main.py")
    print("Expected: promptchain.register_callback(handler_function)")
    print("Actual: NO REGISTRATION CODE EXISTS")

    print("\n[CALLBACK STATE]")
    print("  callback_manager.callbacks = []  # Empty list")
    print("  callback_manager.has_callbacks() = False")

    print("\n[EVENT EMISSION]")
    print("File: promptchain/utils/promptchaining.py (various locations)")
    print("Code: self.callback_manager.emit(event)")
    print("Result: Events fired to empty callback list → silent no-op")

    print("\n[TUI INTEGRATION]")
    print("ObservePanel: Expects to receive callback events")
    print("ActivityLogViewer: Expects to display event history")
    print("Actual data source: None (no callbacks registered)")

    print("\n[IMPACT]")
    print("  ✗ No events reach TUI panels")
    print("  ✗ ObservePanel shows 0/0 activities")
    print("  ✗ ActivityLogViewer remains empty")


def show_fix_options():
    """Show how to fix the observability systems."""
    print("\n\n" + "=" * 80)
    print("FIX OPTIONS")
    print("=" * 80)

    print("\n[OPTION 1: Enable MLflow (Immediate)]")
    print("Steps:")
    print("  1. export PROMPTCHAIN_MLFLOW_ENABLED=true")
    print("  2. mlflow server --host 127.0.0.1 --port 5000")
    print("  3. promptchain")
    print("\nResult:")
    print("  ✓ MLflow UI shows runs at http://localhost:5000")
    print("  ✓ Metrics tracked automatically")
    print("  ✗ TUI panels still empty (need callback bridge)")

    print("\n[OPTION 2: Register Callbacks (Code Change)]")
    print("File: promptchain/cli/main.py")
    print("Add:")
    print("""
    def setup_observability_callbacks(promptchain, activity_logger):
        def bridge_event_to_tui(event):
            activity_logger.log_execution_event(event)

        promptchain.register_callback(bridge_event_to_tui)

    # In _launch_tui():
    setup_observability_callbacks(promptchain_instance, activity_logger)
    """)
    print("\nResult:")
    print("  ✓ TUI panels show activity in real-time")
    print("  ✓ Works without MLflow server")

    print("\n[OPTION 3: Full Observability (Both Systems)]")
    print("Combine Option 1 + Option 2")
    print("\nResult:")
    print("  ✓ MLflow UI for experiment tracking")
    print("  ✓ TUI panels for real-time feedback")
    print("  ✓ Dual observability coverage")


def main():
    """Run full execution trace analysis."""
    import os

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "OBSERVABILITY EXECUTION TRACE" + " " * 29 + "║")
    print("║" + " " * 15 + "Dual System Behavior Analysis" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")

    trace_import_sequence()
    trace_execution_flow()
    trace_callback_system()
    show_fix_options()

    print("\n\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The dual observability systems DO NOT CONFLICT architecturally.
They fail independently due to configuration issues:

1. MLflow System: Disabled via _ENABLED = False (ghost pattern working as designed)
2. Callback System: Enabled but no handlers registered (implementation incomplete)

Both show zero activity because they are not properly activated, not because
they interfere with each other.

Fix: Choose integration strategy from options above.
""")


if __name__ == "__main__":
    main()
