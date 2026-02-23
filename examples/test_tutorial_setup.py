"""Quick validation script for tutorial setup.

Run this before executing the Jupyter notebook to verify all components work.
"""
import sys
sys.path.insert(0, '..')

from tutorial_helpers import *
import mlflow

def test_mock_tools():
    """Test that all mock tools work correctly."""
    print("Testing Mock Tools...")

    # Test search_codebase
    result = search_codebase("authentication")
    assert "Found" in result
    assert "matches" in result
    print("  ✅ search_codebase works")

    # Test read_file
    result = read_file("src/auth/login.py")
    assert "=== Content of" in result
    assert "authenticate_user" in result
    print("  ✅ read_file works")

    # Test analyze_code
    result = analyze_code("sample code")
    assert "Code Analysis Results" in result
    print("  ✅ analyze_code works")

    # Test delete_file
    result = delete_file("dangerous.txt")
    assert "DANGEROUS" in result
    print("  ✅ delete_file works (CoVe blocker)")

    # Test write_report
    result = write_report("Sample report content goes here")
    assert "Report written successfully" in result
    print("  ✅ write_report works")


def test_scenario_builders():
    """Test that scenario builders create valid scenarios."""
    print("\nTesting Scenario Builders...")

    # Test research scenario
    scenario = create_research_scenario()
    assert "objective" in scenario
    assert "tools" in scenario
    assert "tool_functions" in scenario
    assert len(scenario["tools"]) == 3
    print(f"  ✅ research_scenario created with {len(scenario['tools'])} tools")

    # Test production scenario
    scenario = create_production_scenario()
    assert "objective" in scenario
    assert any(tool["function"]["name"] == "delete_file" for tool in scenario["tools"])
    print(f"  ✅ production_scenario created with dangerous operations")

    # Test multi-agent scenario
    scenario = create_multi_agent_scenario()
    assert "tasks" in scenario
    assert len(scenario["tasks"]) == 3
    print(f"  ✅ multi_agent_scenario created with {len(scenario['tasks'])} tasks")


def test_mlflow_helpers():
    """Test MLflow helper functions."""
    print("\nTesting MLflow Helpers...")

    # Test setup (should not error even if MLflow installed)
    setup_mlflow_tracking("test_experiment")
    print("  ✅ setup_mlflow_tracking works")

    # Test visualizations (should not crash)
    mock_snapshots = [
        {"facts_discovered": {"fact1": "value1"}, "observations": ["obs1"], "current_plan": ["step1"]},
        {"facts_discovered": {"fact1": "value1", "fact2": "value2"}, "observations": ["obs1", "obs2"], "current_plan": ["step1", "step2"]}
    ]
    visualize_blackboard_evolution(mock_snapshots)
    print("  ✅ visualize_blackboard_evolution works")

    mock_tao_log = [
        {"phase": "THINK", "data": {"summary": "Thinking about the problem"}},
        {"phase": "ACT", "data": {"tool_name": "search_codebase"}},
        {"phase": "OBSERVE", "data": {"summary": "Observed search results"}}
    ]
    visualize_tao_execution(mock_tao_log)
    print("  ✅ visualize_tao_execution works")


def test_mlflow_availability():
    """Test MLflow installation and configuration."""
    print("\nTesting MLflow Integration...")

    try:
        import mlflow
        print(f"  ✅ MLflow version {mlflow.__version__} installed")

        # Test that we can set tracking URI
        mlflow.set_tracking_uri("file:./test_mlruns")
        print("  ✅ MLflow tracking URI configurable")

        # Check if MLflow UI is accessible
        import requests
        try:
            response = requests.get("http://localhost:5000", timeout=2)
            print("  ✅ MLflow UI running on http://localhost:5000")
        except:
            print("  ⚠️  MLflow UI not running (optional - start with: mlflow ui)")

    except ImportError:
        print("  ❌ MLflow not installed (should not happen)")
        return False

    return True


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Tutorial Setup Validation")
    print("=" * 60)

    try:
        test_mock_tools()
        test_scenario_builders()
        test_mlflow_helpers()
        mlflow_ok = test_mlflow_availability()

        print("\n" + "=" * 60)
        print("✅ All validation tests passed!")
        print("=" * 60)
        print("\nYou're ready to run the Jupyter notebook:")
        print("  1. Launch Jupyter: jupyter notebook")
        print("  2. Open: tutorial_full_stack_with_mlflow.ipynb")
        print("  3. Run cells sequentially")
        print("  4. View results in MLflow UI: http://localhost:5000")
        print("\n" + "=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
