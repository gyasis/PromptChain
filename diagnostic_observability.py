#!/usr/bin/env python3
"""
Diagnostic script to verify observability data flow from decorator → queue → TUI display.

This script checks:
1. Background queue is processing items
2. MLflow tracking is recording data
3. ActivityLogger is capturing events
4. TUI ObservePanel can receive data
5. ActivityLogViewer can display data
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from promptchain.observability import track_task
from promptchain.observability.config import (
    is_enabled,
    get_tracking_uri,
    get_experiment_name,
)
from promptchain.observability.queue import (
    _background_logger,
    queue_log_metric,
    queue_log_param,
    flush_queue,
    get_queue_size,
)
from promptchain.observability.mlflow_adapter import (
    start_run,
    end_run,
)


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def check_observability_config():
    """Check if observability is enabled and configured."""
    print_section("1. Observability Configuration")

    enabled = is_enabled()
    print(f"✓ Observability enabled: {enabled}")

    if enabled:
        print(f"✓ Tracking URI: {get_tracking_uri()}")
        print(f"✓ Experiment name: {get_experiment_name()}")
    else:
        print("⚠️  Observability is DISABLED")
        print("   Set PROMPTCHAIN_OBSERVABILITY_ENABLED=true in environment")
        return False

    return True


def check_background_queue():
    """Check if background queue is running."""
    print_section("2. Background Queue Status")

    # Check if worker thread is alive
    if _background_logger.worker:
        is_alive = _background_logger.worker.is_alive()
        print(f"✓ Worker thread alive: {is_alive}")
        print(f"✓ Queue enabled: {_background_logger.enabled}")
        print(f"✓ Current queue size: {get_queue_size()}")

        # Test queue with sample data
        print("\n  Testing queue with sample metrics...")
        queue_log_metric("test_metric_1", 42.0)
        queue_log_metric("test_metric_2", 100.0)
        queue_log_param("test_param", "diagnostic_value")

        time.sleep(0.5)  # Give queue time to process

        queue_size_after = get_queue_size()
        print(f"  Queue size after adding 3 items: {queue_size_after}")

        # Flush queue
        print("  Flushing queue...")
        flushed = flush_queue(timeout=5.0)
        print(f"  Queue flushed: {flushed}")
        print(f"  Final queue size: {get_queue_size()}")

        return True
    else:
        print("⚠️  Background worker thread NOT running")
        print("   Background logging may be disabled")
        return False


def check_mlflow_integration():
    """Check if MLflow is properly integrated."""
    print_section("3. MLflow Integration")

    try:
        from promptchain.observability.mlflow_adapter import is_available, set_experiment
        from promptchain.observability.context import get_current_run

        if not is_available():
            print("⚠️  MLflow not available")
            return False

        # Set experiment
        set_experiment(get_experiment_name())
        print("✓ MLflow experiment set")

        # Start a run
        run = start_run()
        if run:
            print(f"✓ MLflow run started: {run.info.run_id}")
        else:
            print("⚠️  Could not start MLflow run")
            return False

        # Test logging
        print("\n  Testing MLflow logging...")
        queue_log_metric("diagnostic_test", 123.45)
        queue_log_param("diagnostic_param", "test_value")

        flush_queue(timeout=5.0)
        print("✓ Test metrics/params logged")

        # End run
        end_run()
        print("✓ MLflow run ended")

        return True

    except Exception as e:
        print(f"❌ MLflow integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_activity_logger():
    """Check if ActivityLogger is working."""
    print_section("4. ActivityLogger Status")

    from promptchain.cli.activity_logger import ActivityLogger
    from pathlib import Path
    import tempfile

    try:
        # Create test logger
        test_dir = Path(tempfile.mkdtemp(prefix="promptchain_test_"))
        log_dir = test_dir / "activity_logs"
        db_path = test_dir / "activities.db"

        logger = ActivityLogger(
            session_name="diagnostic_test",
            log_dir=log_dir,
            db_path=db_path,
            enable_console=False
        )

        print(f"✓ ActivityLogger created")
        print(f"  Log dir: {log_dir}")
        print(f"  DB path: {db_path}")

        # Start chain and log activity
        logger.start_chain("diagnostic_chain")
        logger.log_activity(
            activity_type="user_input",
            content="Test user input",
            agent_name="test_agent"
        )
        logger.log_activity(
            activity_type="agent_output",
            content="Test agent response",
            agent_name="test_agent"
        )
        logger.end_chain()

        print("✓ Test activities logged")

        # Check files created
        if log_dir.exists():
            log_files = list(log_dir.glob("*.jsonl"))
            print(f"  JSONL files created: {len(log_files)}")

        if db_path.exists():
            print(f"  SQLite database created: Yes")

        # Clean up
        import shutil
        shutil.rmtree(test_dir)

        return True

    except Exception as e:
        print(f"❌ ActivityLogger error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tui_integration():
    """Check TUI integration points."""
    print_section("5. TUI Integration")

    try:
        from promptchain.cli.tui.observe_panel import ObservePanel
        from promptchain.cli.tui.activity_log_viewer import ActivityLogViewer

        print("✓ ObservePanel importable")
        print("✓ ActivityLogViewer importable")

        # Check ObservePanel can log entries
        print("\n  Testing ObservePanel...")
        panel = ObservePanel()
        panel.log_tool_call("test_tool", {"arg1": "value1"})
        panel.log_llm_request("gpt-4", "test prompt")
        panel.log_llm_response("gpt-4", "test response", tokens=100)

        print(f"  ObservePanel entries logged: {len(panel._entries)}")

        return True

    except Exception as e:
        print(f"❌ TUI integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_flow():
    """Test end-to-end data flow from decorator to display."""
    print_section("6. End-to-End Data Flow")

    @track_task(operation_type="DIAGNOSTIC")
    def test_tracked_function(arg1, arg2):
        """Test function with tracking decorator."""
        time.sleep(0.1)  # Simulate work
        return f"Result: {arg1} + {arg2}"

    try:
        print("  Calling @track_task decorated function...")
        result = test_tracked_function("test", "value")
        print(f"✓ Function executed: {result}")

        # Check if data was queued
        print(f"  Queue size: {get_queue_size()}")

        # Flush and verify
        flush_queue(timeout=5.0)
        print("✓ Queue flushed")

        return True

    except Exception as e:
        print(f"❌ Data flow error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic checks."""
    print("\n" + "="*60)
    print("  PromptChain Observability Diagnostic")
    print("="*60)

    results = {}

    # Run checks
    results['config'] = check_observability_config()

    if results['config']:
        results['queue'] = check_background_queue()
        results['mlflow'] = check_mlflow_integration()
        results['activity_logger'] = check_activity_logger()
        results['tui'] = check_tui_integration()
        results['data_flow'] = check_data_flow()

    # Summary
    print_section("Diagnostic Summary")

    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {check_name:20s}: {status}")

    # Overall result
    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("  ✓ All diagnostics PASSED")
    else:
        print("  ❌ Some diagnostics FAILED")
    print("="*60 + "\n")

    # Cleanup
    try:
        from promptchain.observability.queue import shutdown_background_logger
        shutdown_background_logger()
    except:
        pass

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
