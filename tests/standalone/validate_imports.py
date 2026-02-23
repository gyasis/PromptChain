#!/usr/bin/env python3
"""Validate all multi-agent communication module imports."""

import sys
sys.path.insert(0, '/home/gyasis/Documents/code/PromptChain')

def test_models_import():
    """Test models module imports."""
    try:
        from promptchain.cli.models import (
            Task, TaskStatus, TaskPriority,
            BlackboardEntry, Blackboard,
            MentalModel, MentalModelManager, SpecializationType
        )
        print("✓ Models import OK")
        print(f"  - Task: {Task}")
        print(f"  - TaskStatus: {TaskStatus}")
        print(f"  - TaskPriority: {TaskPriority}")
        print(f"  - BlackboardEntry: {BlackboardEntry}")
        print(f"  - Blackboard: {Blackboard}")
        print(f"  - MentalModel: {MentalModel}")
        print(f"  - MentalModelManager: {MentalModelManager}")
        print(f"  - SpecializationType: {SpecializationType}")
        return True
    except Exception as e:
        print(f"✗ Models import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tools_import():
    """Test tools library imports."""
    try:
        from promptchain.cli.tools.library import (
            delegate_task_tool,
            write_to_blackboard_tool,
            get_my_capabilities_tool
        )
        print("✓ Tools import OK")
        print(f"  - delegate_task_tool: {delegate_task_tool}")
        print(f"  - write_to_blackboard_tool: {write_to_blackboard_tool}")
        print(f"  - get_my_capabilities_tool: {get_my_capabilities_tool}")
        return True
    except Exception as e:
        print(f"✗ Tools import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_communication_import():
    """Test communication module imports."""
    try:
        from promptchain.cli.communication import (
            MessageBus,
            Message,
            cli_communication_handler
        )
        print("✓ Communication import OK")
        print(f"  - MessageBus: {MessageBus}")
        print(f"  - Message: {Message}")
        print(f"  - cli_communication_handler: {cli_communication_handler}")
        return True
    except Exception as e:
        print(f"✗ Communication import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_registry_import():
    """Test registry extensions."""
    try:
        from promptchain.cli.tools.registry import tool_registry
        caps = tool_registry.list_capabilities()
        print(f"✓ Registry OK - {len(caps)} capabilities found")
        print("  Available capabilities:")
        for cap in caps:
            print(f"    - {cap}")
        return True
    except Exception as e:
        print(f"✗ Registry failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("MULTI-AGENT COMMUNICATION MODULE IMPORT VALIDATION")
    print("=" * 60)
    print()

    results = []

    print("1. Testing models imports...")
    results.append(("Models", test_models_import()))
    print()

    print("2. Testing tools imports...")
    results.append(("Tools", test_tools_import()))
    print()

    print("3. Testing communication imports...")
    results.append(("Communication", test_communication_import()))
    print()

    print("4. Testing registry extensions...")
    results.append(("Registry", test_registry_import()))
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ All imports validated successfully!")
        return 0
    else:
        print("✗ Some imports failed - see details above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
