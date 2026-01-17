#!/usr/bin/env python3
"""
Minimal import test for Research Agent package structure.
Tests that the package can be imported without heavy dependencies.
"""

def test_package_structure():
    """Test that the package structure allows basic imports"""
    
    # Test 1: Basic package import
    try:
        import research_agent
        print("✓ research_agent package import successful")
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False
    
    # Test 2: Core imports that don't require PromptChain
    try:
        from research_agent.core.file_lock import atomic_write, file_lock
        print("✓ file_lock utilities import successful")
    except ImportError as e:
        print(f"❌ file_lock import failed: {e}")
        return False
    
    # Test 3: Model validation utilities (minimal dependencies)
    try:
        from research_agent.core.model_validation import ModelConfigValidator
        print("✓ model_validation import successful")
    except ImportError as e:
        print(f"❌ model_validation import failed: {e}")
        return False
    
    print("🎉 All minimal imports successful!")
    print("   Package structure is correctly set up for isolated environments")
    return True

def test_import_structure_without_circular_deps():
    """Test that imports don't create circular dependencies"""
    
    import sys
    initial_modules = set(sys.modules.keys())
    
    try:
        # This should work without importing heavyweight dependencies
        from research_agent.core.capability_validator import get_capability_validator
        validator = get_capability_validator()
        print("✓ Capability validator creation successful")
        
        # Verify we didn't pull in too many modules
        new_modules = set(sys.modules.keys()) - initial_modules
        research_modules = [m for m in new_modules if 'research_agent' in m]
        
        print(f"✓ Imported {len(research_modules)} research_agent modules")
        if len(research_modules) > 10:
            print(f"⚠️  Warning: Many modules imported ({len(research_modules)})")
        
        return True
        
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Research Agent package structure...")
    print("=" * 50)
    
    success1 = test_package_structure()
    print()
    success2 = test_import_structure_without_circular_deps()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All package structure tests PASSED!")
        print("   The Research Agent is properly configured for import resolution.")
    else:
        print("❌ Some package structure tests FAILED!")
        print("   Review import dependencies and circular import issues.")
        
    exit(0 if (success1 and success2) else 1)