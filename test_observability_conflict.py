#!/usr/bin/env python3
"""
Diagnostic script to identify conflicts between the old callback system
and new MLflow decorator system.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("OBSERVABILITY CONFLICT DIAGNOSTIC")
print("=" * 80)

# 1. Check environment variables
print("\n1. ENVIRONMENT VARIABLES:")
print(f"   PROMPTCHAIN_MLFLOW_ENABLED: {os.getenv('PROMPTCHAIN_MLFLOW_ENABLED', 'NOT SET')}")
print(f"   MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'NOT SET')}")
print(f"   PROMPTCHAIN_MLFLOW_EXPERIMENT: {os.getenv('PROMPTCHAIN_MLFLOW_EXPERIMENT', 'NOT SET')}")

# 2. Check MLflow availability
print("\n2. MLFLOW AVAILABILITY:")
try:
    import mlflow
    print(f"   ✓ MLflow installed: {mlflow.__version__}")
except ImportError:
    print("   ✗ MLflow NOT installed")

# 3. Check ghost pattern status
print("\n3. GHOST PATTERN STATUS:")
try:
    from promptchain.observability.ghost import is_tracking_enabled, is_mlflow_available, _ENABLED
    print(f"   is_tracking_enabled(): {is_tracking_enabled()}")
    print(f"   is_mlflow_available(): {is_mlflow_available()}")
    print(f"   _ENABLED (module-level): {_ENABLED}")
except Exception as e:
    print(f"   ✗ Error checking ghost pattern: {e}")

# 4. Check config
print("\n4. CONFIG MODULE:")
try:
    from promptchain.observability.config import is_enabled, get_tracking_uri, get_experiment_name
    print(f"   is_enabled(): {is_enabled()}")
    print(f"   get_tracking_uri(): {get_tracking_uri()}")
    print(f"   get_experiment_name(): {get_experiment_name()}")
except Exception as e:
    print(f"   ✗ Error checking config: {e}")

# 5. Check if decorators are actually wrapping functions
print("\n5. DECORATOR WRAPPING TEST:")
try:
    from promptchain.utils.promptchaining import PromptChain

    # Check if run_model_async is wrapped
    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Test: {input}"])

    # Check function wrapper
    import inspect
    func = chain.run_model_async
    print(f"   run_model_async.__name__: {func.__name__}")
    print(f"   run_model_async.__wrapped__ exists: {hasattr(func, '__wrapped__')}")

    # Check closure (indicates decorator is active)
    if hasattr(func, '__closure__') and func.__closure__:
        print(f"   run_model_async has closure (decorator active): YES")
        print(f"   Closure cells: {len(func.__closure__)}")
    else:
        print(f"   run_model_async has closure (decorator active): NO (ghost pattern active)")

except Exception as e:
    print(f"   ✗ Error checking decorators: {e}")
    import traceback
    traceback.print_exc()

# 6. Check callback system status
print("\n6. CALLBACK SYSTEM STATUS:")
try:
    from promptchain.utils.promptchaining import PromptChain
    from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

    chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=["Test: {input}"])

    print(f"   CallbackManager exists: {hasattr(chain, 'callback_manager')}")
    print(f"   CallbackManager has_callbacks(): {chain.callback_manager.has_callbacks()}")

    # Register a test callback
    callback_called = []
    def test_callback(event: ExecutionEvent):
        callback_called.append(event.event_type)
        print(f"      -> Callback received: {event.event_type}")

    chain.register_callback(test_callback)
    print(f"   After registering callback, has_callbacks(): {chain.callback_manager.has_callbacks()}")

except Exception as e:
    print(f"   ✗ Error checking callbacks: {e}")
    import traceback
    traceback.print_exc()

# 7. Test actual execution
print("\n7. EXECUTION TEST (with both systems):")

async def test_execution():
    """Test if either system captures events during actual execution."""
    from promptchain.utils.promptchaining import PromptChain
    from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType
    from promptchain.observability import init_mlflow, shutdown_mlflow

    # Initialize MLflow
    init_mlflow()

    # Setup callback tracking
    callback_events = []
    def callback_tracker(event: ExecutionEvent):
        callback_events.append({
            'type': event.event_type,
            'timestamp': event.timestamp
        })
        print(f"   [CALLBACK] {event.event_type.name}")

    # Create chain
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=["Respond with exactly: 'test success'"],
        verbose=True
    )

    # Register callback
    chain.register_callback(callback_tracker)

    try:
        print("\n   Executing chain with simple prompt...")
        result = await chain.process_prompt_async("test")
        print(f"\n   Result: {result[:100] if result else 'None'}...")
        print(f"\n   Callback events captured: {len(callback_events)}")
        for evt in callback_events[:5]:  # Show first 5
            print(f"      - {evt['type'].name}")

    except Exception as e:
        print(f"   ✗ Execution error: {e}")
        import traceback
        traceback.print_exc()

    # Shutdown MLflow
    shutdown_mlflow()

    return callback_events

# Run test
callback_events = asyncio.run(test_execution())

# 8. Check MLflow runs
print("\n8. MLFLOW RUNS CHECK:")
try:
    import mlflow
    from promptchain.observability.config import get_experiment_name

    experiment_name = get_experiment_name()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment:
        print(f"   Experiment '{experiment_name}' exists: YES")
        print(f"   Experiment ID: {experiment.experiment_id}")

        # List recent runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=5)
        print(f"   Recent runs: {len(runs)}")

        if len(runs) > 0:
            latest = runs.iloc[0]
            print(f"   Latest run ID: {latest['run_id'][:8]}...")
            print(f"   Latest run status: {latest['status']}")
        else:
            print("   No runs found in experiment")
    else:
        print(f"   Experiment '{experiment_name}' exists: NO")

except Exception as e:
    print(f"   ✗ Error checking MLflow runs: {e}")

# 9. Summary
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY:")
print("=" * 80)

issues = []

# Check if MLflow is disabled
from promptchain.observability.config import is_enabled
if not is_enabled():
    issues.append("MLflow tracking is DISABLED (PROMPTCHAIN_MLFLOW_ENABLED not set to true)")

# Check if MLflow not installed
try:
    import mlflow
except ImportError:
    issues.append("MLflow is NOT installed (pip install mlflow)")

# Check if both systems are active
if len(callback_events) > 0:
    issues.append(f"Callback system is ACTIVE (captured {len(callback_events)} events)")
else:
    issues.append("Callback system is INACTIVE (no events captured)")

# Check decorator status
from promptchain.observability.ghost import is_tracking_enabled
if is_tracking_enabled():
    issues.append("Decorator system is ACTIVE (ghost pattern disabled)")
else:
    issues.append("Decorator system is INACTIVE (ghost pattern returned identity decorator)")

if issues:
    print("\nISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\n✓ No issues detected")

print("\n" + "=" * 80)
