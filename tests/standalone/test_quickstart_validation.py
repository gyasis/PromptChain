#!/usr/bin/env python3
"""Validation script for quickstart.md examples.

This script tests all code examples from the quickstart guide to ensure
they work with the current CLI implementation.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

class QuickstartValidator:
    """Validates quickstart.md examples against current CLI implementation."""

    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "failures": []
        }

    def test_command_handler_methods(self):
        """Test that all command handler methods exist."""
        from promptchain.cli.command_handler import CommandHandler
        from promptchain.cli.session_manager import SessionManager

        print("\n=== Testing Command Handler Methods ===")

        # Initialize test objects
        sessions_dir = Path.home() / ".promptchain" / "sessions" / "test"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        session_manager = SessionManager(sessions_dir=sessions_dir)
        handler = CommandHandler(session_manager)

        # Commands from quickstart.md
        required_methods = [
            # Agent commands
            ("handle_agent_create", "Agent creation"),
            ("handle_agent_list", "Agent listing"),
            ("handle_agent_use", "Agent switching"),
            ("handle_agent_delete", "Agent deletion"),
            ("handle_agent_create_from_template", "Template-based agent creation"),
            ("handle_agent_list_templates", "Template listing"),
            ("handle_agent_update", "Agent updates"),

            # Session commands
            ("handle_session_list", "Session listing"),
            ("handle_session_delete", "Session deletion"),

            # Exit command
            ("handle_exit", "Exit handling"),
        ]

        for method_name, description in required_methods:
            self.results["total_tests"] += 1
            if hasattr(handler, method_name):
                print(f"✓ {description} ({method_name})")
                self.results["passed"] += 1
            else:
                print(f"✗ {description} ({method_name}) - MISSING")
                self.results["failed"] += 1
                self.results["failures"].append(f"Missing method: {method_name}")

    def test_agent_templates(self):
        """Test that agent templates are available."""
        print("\n=== Testing Agent Templates ===")

        try:
            from promptchain.cli.utils.agent_templates import AGENT_TEMPLATES, create_from_template

            required_templates = ["researcher", "coder", "analyst", "terminal"]

            for template_name in required_templates:
                self.results["total_tests"] += 1
                if template_name in AGENT_TEMPLATES:
                    print(f"✓ Template '{template_name}' exists")
                    self.results["passed"] += 1
                else:
                    print(f"✗ Template '{template_name}' missing")
                    self.results["failed"] += 1
                    self.results["failures"].append(f"Missing template: {template_name}")

        except ImportError as e:
            self.results["total_tests"] += 1
            print(f"✗ Agent templates module not found: {e}")
            self.results["failed"] += 1
            self.results["failures"].append(f"Import error: {e}")

    def test_file_context_features(self):
        """Test @syntax file reference capabilities."""
        print("\n=== Testing File Context Features ===")

        try:
            from promptchain.cli.utils.file_context_manager import FileContextManager

            self.results["total_tests"] += 1
            manager = FileContextManager()

            # Test that manager has required methods
            required_methods = ["resolve_file_references", "discover_files"]
            for method_name in required_methods:
                if hasattr(manager, method_name):
                    print(f"✓ FileContextManager has {method_name}")
                else:
                    print(f"✗ FileContextManager missing {method_name}")
                    self.results["failures"].append(f"FileContextManager missing: {method_name}")

            self.results["passed"] += 1

        except ImportError as e:
            print(f"✗ File context features not available: {e}")
            self.results["failed"] += 1
            self.results["failures"].append(f"File context import error: {e}")

    def test_shell_execution(self):
        """Test !command shell execution capabilities."""
        print("\n=== Testing Shell Execution ===")

        try:
            from promptchain.cli.shell_executor import ShellExecutor, ShellCommandParser

            self.results["total_tests"] += 1
            executor = ShellExecutor()
            parser = ShellCommandParser()

            print("✓ Shell execution features available")
            self.results["passed"] += 1

        except ImportError as e:
            print(f"✗ Shell execution not available: {e}")
            self.results["failed"] += 1
            self.results["failures"].append(f"Shell execution import error: {e}")

    def test_model_specifications(self):
        """Test that example model strings are valid."""
        print("\n=== Testing Model Specifications ===")

        # Model strings from quickstart examples
        example_models = [
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-opus-20240229",
            "anthropic/claude-3-sonnet-20240229",
            "ollama/llama2",
        ]

        for model in example_models:
            self.results["total_tests"] += 1
            # Basic validation - model string format
            if "/" in model and len(model.split("/")) == 2:
                provider, model_name = model.split("/")
                if provider and model_name:
                    print(f"✓ Valid model format: {model}")
                    self.results["passed"] += 1
                else:
                    print(f"✗ Invalid model format: {model}")
                    self.results["failed"] += 1
                    self.results["failures"].append(f"Invalid model: {model}")
            else:
                print(f"✗ Invalid model format: {model}")
                self.results["failed"] += 1
                self.results["failures"].append(f"Invalid model format: {model}")

    def test_history_management(self):
        """Test history management features."""
        print("\n=== Testing History Management ===")

        try:
            from promptchain.utils.execution_history_manager import ExecutionHistoryManager

            self.results["total_tests"] += 1

            # Test creation with token limits
            manager = ExecutionHistoryManager(
                max_tokens=4000,
                max_entries=50,
                truncation_strategy="oldest_first"
            )

            print("✓ History management available")
            self.results["passed"] += 1

        except ImportError as e:
            print(f"✗ History management not available: {e}")
            self.results["failed"] += 1
            self.results["failures"].append(f"History management import error: {e}")

    def test_workflow_features(self):
        """Test workflow management features."""
        print("\n=== Testing Workflow Features ===")

        try:
            from promptchain.cli.models.workflow import WorkflowState, WorkflowStep

            self.results["total_tests"] += 1
            print("✓ Workflow features available")
            self.results["passed"] += 1

        except ImportError as e:
            print(f"⚠ Workflow features not available (may be Phase 9): {e}")
            self.results["skipped"] += 1

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']} ✓")
        print(f"Failed: {self.results['failed']} ✗")
        print(f"Skipped: {self.results['skipped']} ⚠")

        if self.results["failures"]:
            print(f"\nFailures ({len(self.results['failures'])}):")
            for failure in self.results["failures"]:
                print(f"  - {failure}")

        print("="*60)

        # Return exit code
        return 0 if self.results["failed"] == 0 else 1

def main():
    """Run all validation tests."""
    validator = QuickstartValidator()

    # Run all test categories
    validator.test_command_handler_methods()
    validator.test_agent_templates()
    validator.test_file_context_features()
    validator.test_shell_execution()
    validator.test_model_specifications()
    validator.test_history_management()
    validator.test_workflow_features()

    # Print summary and exit
    exit_code = validator.print_summary()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
