#!/usr/bin/env python3
"""
Installation verification script.

Tests that PromptChain works correctly with and without MLflow installed.
"""

import sys
import subprocess


def test_core_imports():
    """Test that core modules can be imported."""
    print("Testing core imports...")
    try:
        from promptchain import PromptChain
        from promptchain.utils.agent_chain import AgentChain
        from promptchain.utils.execution_history_manager import ExecutionHistoryManager
        print("✓ Core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Core import failed: {e}")
        return False


def test_mlflow_graceful_degradation():
    """Test that code works without MLflow installed."""
    print("\nTesting MLflow graceful degradation...")
    try:
        # Try to import MLflow
        import mlflow
        mlflow_installed = True
        print("  MLflow is installed")
    except ImportError:
        mlflow_installed = False
        print("  MLflow is NOT installed (expected for core install)")

    # Try to import observability modules
    try:
        from promptchain.observability.config import is_enabled, get_tracking_uri
        from promptchain.observability.ghost import conditional_decorator
        print("✓ Observability modules import successfully")

        # Test that tracking is disabled when MLflow is not installed
        enabled = is_enabled()

        # MLflow tracking should be disabled by default (env var not set)
        if enabled and not mlflow_installed:
            print("✗ Tracking should be disabled when MLflow is not installed")
            return False

        if mlflow_installed:
            print(f"  MLflow installed, tracking enabled: {enabled}")
        else:
            print(f"  MLflow not installed, tracking disabled: {enabled}")

        print("✓ Graceful degradation working")
        return True
    except ImportError as e:
        print(f"✗ Observability import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic PromptChain functionality."""
    print("\nTesting basic PromptChain functionality...")
    try:
        from promptchain import PromptChain

        # Create a simple chain
        chain = PromptChain(
            models=["openai/gpt-4"],
            instructions=["Echo: {input}"],
            verbose=False
        )
        print("✓ PromptChain instance created successfully")

        # Test that the chain structure is correct
        if len(chain.instructions) != 1:
            print("✗ Chain instructions not set correctly")
            return False

        print("✓ Basic functionality working")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def check_dependencies():
    """Check that all core dependencies are installed."""
    print("\nChecking core dependencies...")
    core_deps = [
        "litellm",
        "dotenv",
        "pydantic",
        "requests",
        "typing_extensions",
        "tiktoken",
        "rich",
        "httpx",
        "yaml",
        "jsonschema",
        "nest_asyncio"
    ]

    all_ok = True
    for dep in core_deps:
        # Map package names to import names
        import_name = {
            "dotenv": "python-dotenv",
            "yaml": "pyyaml",
            "typing_extensions": "typing-extensions"
        }.get(dep, dep)

        try:
            __import__(dep.replace("-", "_"))
            print(f"  ✓ {import_name}")
        except ImportError:
            print(f"  ✗ {import_name} NOT installed")
            all_ok = False

    return all_ok


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PromptChain Installation Verification")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Core Imports", test_core_imports()))
    results.append(("MLflow Graceful Degradation", test_mlflow_graceful_degradation()))
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Core Dependencies", check_dependencies()))

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! Installation is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
