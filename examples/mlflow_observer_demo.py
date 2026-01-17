"""
MLflow Observer Plugin Demo

This example demonstrates how to use the MLflow observer plugin for automatic
observability without requiring decorators or code modifications.

Prerequisites:
    1. pip install mlflow (optional)
    2. export PROMPTCHAIN_MLFLOW_ENABLED=true
    3. mlflow ui --port 5000 (optional, can use local directory)
"""

import os
from promptchain import PromptChain
from promptchain.observability import MLflowObserver


def demo_basic_observer():
    """Basic observer usage with single chain."""
    print("=== Basic MLflow Observer Demo ===\n")

    # Enable MLflow observer
    os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

    # Create observer
    observer = MLflowObserver(
        experiment_name="promptchain-demo",
        tracking_uri="http://localhost:5000",  # Or use ./mlruns for local
        auto_log_artifacts=True  # Log prompts/responses as files
    )

    # Check if observer is available
    if observer.is_available():
        print("✓ MLflow observer activated")
        print(f"  Experiment: {observer.experiment_name}")
        print(f"  Tracking URI: {observer.tracking_uri}\n")
    else:
        print("✗ MLflow observer not available")
        print("  Install with: pip install mlflow")
        print("  Enable with: export PROMPTCHAIN_MLFLOW_ENABLED=true\n")
        return

    # Create chain
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Analyze the following input: {input}",
            "Provide a summary: {input}"
        ],
        verbose=True
    )

    # Register observer callback
    chain.register_callback(observer.handle_event)
    print("✓ Observer registered with chain\n")

    # Execute chain - events automatically logged to MLflow
    print("Executing chain...")
    result = chain.process_prompt(
        "What are the benefits of using MLflow for observability?"
    )

    print(f"\nResult: {result}\n")

    # Cleanup
    observer.shutdown()
    print("✓ Observer shutdown complete")
    print("\nView results: http://localhost:5000")


def demo_with_tools():
    """Observer usage with tool calls."""
    print("=== MLflow Observer with Tools Demo ===\n")

    os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

    # Create observer
    observer = MLflowObserver(
        experiment_name="promptchain-tools-demo",
        tracking_uri="http://localhost:5000"
    )

    if not observer.is_available():
        print("MLflow observer not available - skipping demo")
        return

    print("✓ MLflow observer activated\n")

    # Define a simple tool
    def search_web(query: str) -> str:
        """Simulate web search tool."""
        return f"Found 5 results for: {query}"

    # Create chain with tools
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Search and analyze: {input}"],
        verbose=True
    )

    # Register tool
    chain.register_tool_function(search_web)
    chain.add_tools([{
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }])

    # Register observer
    chain.register_callback(observer.handle_event)

    print("Executing chain with tools...")
    result = chain.process_prompt("Find information about MLflow observability")

    print(f"\nResult: {result}\n")

    # Cleanup
    observer.shutdown()
    print("✓ Observer shutdown complete")
    print("\nView tool tracking: http://localhost:5000")


def demo_multi_step():
    """Observer usage with multi-step chain."""
    print("=== MLflow Observer Multi-Step Demo ===\n")

    os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"

    # Create observer
    observer = MLflowObserver(
        experiment_name="promptchain-multistep-demo",
        tracking_uri="http://localhost:5000"
    )

    if not observer.is_available():
        print("MLflow observer not available - skipping demo")
        return

    print("✓ MLflow observer activated\n")

    # Create multi-step chain
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=[
            "Step 1 - Extract key points: {input}",
            "Step 2 - Analyze sentiment: {input}",
            "Step 3 - Generate summary: {input}"
        ],
        store_steps=True,  # Store intermediate results
        verbose=True
    )

    # Register observer - will create nested runs for each step
    chain.register_callback(observer.handle_event)

    print("Executing multi-step chain...")
    result = chain.process_prompt(
        "MLflow provides comprehensive tracking for machine learning experiments."
    )

    print(f"\nResult: {result}\n")

    # Cleanup
    observer.shutdown()
    print("✓ Observer shutdown complete")
    print("\nView nested runs: http://localhost:5000")


def demo_local_tracking():
    """Observer usage with local directory (no MLflow server needed)."""
    print("=== MLflow Observer Local Tracking Demo ===\n")

    # Use local directory instead of server
    os.environ["PROMPTCHAIN_MLFLOW_ENABLED"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = "./mlruns"

    print("Using local tracking directory: ./mlruns")
    print("No MLflow server required!\n")

    # Create observer
    observer = MLflowObserver(
        experiment_name="promptchain-local-demo",
        tracking_uri="./mlruns"  # Local directory
    )

    if not observer.is_available():
        print("MLflow observer not available - skipping demo")
        return

    print("✓ MLflow observer activated (local mode)\n")

    # Create simple chain
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Respond to: {input}"],
        verbose=True
    )

    # Register observer
    chain.register_callback(observer.handle_event)

    print("Executing chain...")
    result = chain.process_prompt("Hello, MLflow!")

    print(f"\nResult: {result}\n")

    # Cleanup
    observer.shutdown()
    print("✓ Observer shutdown complete")
    print("\nView results:")
    print("  1. Run: mlflow ui --backend-store-uri ./mlruns")
    print("  2. Open: http://localhost:5000")


def demo_graceful_degradation():
    """Demonstrate graceful degradation when MLflow not available."""
    print("=== MLflow Observer Graceful Degradation Demo ===\n")

    # Disable MLflow
    os.environ.pop("PROMPTCHAIN_MLFLOW_ENABLED", None)

    print("MLflow observer disabled (PROMPTCHAIN_MLFLOW_ENABLED not set)\n")

    # Create observer
    observer = MLflowObserver()

    # Check availability
    if observer.is_available():
        print("✓ Observer is available")
    else:
        print("✗ Observer is not available (expected)")
        print("  Application continues to work normally\n")

    # Create chain
    chain = PromptChain(
        models=["openai/gpt-4"],
        instructions=["Process: {input}"],
        verbose=True
    )

    # Register observer (will be no-op)
    chain.register_callback(observer.handle_event)
    print("✓ Observer registered (no-op mode)\n")

    print("Executing chain (without MLflow tracking)...")
    result = chain.process_prompt("This works without MLflow!")

    print(f"\nResult: {result}\n")
    print("✓ Chain executed successfully")
    print("  No MLflow dependency required")

    # Cleanup (safe to call even when not active)
    observer.shutdown()
    print("✓ Observer shutdown (no-op)")


if __name__ == "__main__":
    import sys

    demos = {
        "1": ("Basic Observer", demo_basic_observer),
        "2": ("With Tools", demo_with_tools),
        "3": ("Multi-Step", demo_multi_step),
        "4": ("Local Tracking", demo_local_tracking),
        "5": ("Graceful Degradation", demo_graceful_degradation),
    }

    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in demos:
            name, func = demos[choice]
            print(f"\nRunning: {name}\n")
            func()
        else:
            print("Invalid choice")
    else:
        print("\nMLflow Observer Plugin Demos")
        print("=" * 50)
        for key, (name, _) in demos.items():
            print(f"{key}. {name}")
        print("\nUsage: python mlflow_observer_demo.py <number>")
        print("Example: python mlflow_observer_demo.py 1")
