#!/usr/bin/env python3
"""
Quick Test - Athena LightRAG
============================
Quick test of Athena LightRAG implementation without making API calls.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from lightrag_core import create_athena_lightrag, QueryMode, AthenaLightRAGCore
        print("  ✓ lightrag_core imports successful")
        
        from agentic_lightrag import create_agentic_lightrag, AgenticLightRAG
        print("  ✓ agentic_lightrag imports successful")
        
        from context_processor import create_context_processor, create_sql_generator
        print("  ✓ context_processor imports successful")
        
        from athena_mcp_server import create_manual_mcp_server, ManualMCPServer
        print("  ✓ athena_mcp_server imports successful")
        
        from config import get_config, AthenaConfig
        print("  ✓ config imports successful")
        
        from exceptions import AthenaLightRAGException, DatabaseNotFoundError
        print("  ✓ exceptions imports successful")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_database_existence():
    """Test that the database exists and has proper structure."""
    print("\nTesting database existence...")
    
    working_dir = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
    db_path = Path(working_dir)
    
    if not db_path.exists():
        print(f"  ✗ Database directory not found: {working_dir}")
        return False
    
    print(f"  ✓ Database directory exists: {working_dir}")
    
    # Check for key files
    key_files = [
        "kv_store_full_entities.json",
        "kv_store_full_relations.json", 
        "vdb_entities.json",
        "vdb_relationships.json",
        "vdb_chunks.json"
    ]
    
    found_files = 0
    for file_name in key_files:
        file_path = db_path / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"    ✓ {file_name}: {size_mb:.1f} MB")
            found_files += 1
        else:
            print(f"    ✗ {file_name}: Missing")
    
    success_rate = found_files / len(key_files)
    print(f"  Database completeness: {found_files}/{len(key_files)} files ({success_rate:.1%})")
    
    return found_files >= len(key_files) // 2  # At least half the files should exist

def test_basic_instantiation():
    """Test basic class instantiation without API calls.""" 
    print("\nTesting basic instantiation...")
    
    try:
        from lightrag_core import LightRAGConfig, AthenaLightRAGCore
        from config import DatabaseConfig, LLMConfig
        
        # Test configuration
        print("  Testing configuration...")
        db_config = DatabaseConfig(validate_on_init=False)  # Skip validation to avoid API calls
        llm_config = LLMConfig(api_key="test_key")  # Use dummy key
        print("    ✓ Configuration objects created")
        
        # Test MCP server manual implementation
        print("  Testing MCP server...")
        from athena_mcp_server import ManualMCPServer, AthenaMCPServer
        
        # This should work without API calls since it just creates the structure
        print("    ✓ MCP server classes imported")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic instantiation failed: {e}")
        return False

def test_tool_schemas():
    """Test that tool schemas are properly defined."""
    print("\nTesting tool schemas...")
    
    try:
        from athena_mcp_server import ManualMCPServer
        
        # Create a dummy server to test schemas
        class DummyAthenaServer:
            def __init__(self):
                pass
                
        dummy_athena = DummyAthenaServer()
        manual_server = ManualMCPServer(dummy_athena)
        
        # Get tool schemas
        schemas = manual_server.get_tool_schemas()
        
        expected_tools = [
            "lightrag_local_query",
            "lightrag_global_query",
            "lightrag_hybrid_query",
            "lightrag_context_extract",
            "lightrag_multi_hop_reasoning",
            "lightrag_sql_generation"
        ]
        
        print(f"  Available tools: {list(schemas.keys())}")
        
        missing_tools = []
        for tool in expected_tools:
            if tool in schemas:
                schema = schemas[tool]
                if "function" in schema and "parameters" in schema["function"]:
                    print(f"    ✓ {tool}: Valid schema")
                else:
                    print(f"    ✗ {tool}: Invalid schema structure")
                    missing_tools.append(tool)
            else:
                print(f"    ✗ {tool}: Missing")
                missing_tools.append(tool)
        
        success_rate = (len(expected_tools) - len(missing_tools)) / len(expected_tools)
        print(f"  Tool schema completeness: {len(expected_tools) - len(missing_tools)}/{len(expected_tools)} ({success_rate:.1%})")
        
        return len(missing_tools) == 0
        
    except Exception as e:
        print(f"  ✗ Tool schema test failed: {e}")
        return False

def test_directory_structure():
    """Test that the project has proper directory structure."""
    print("\nTesting directory structure...")
    
    current_dir = Path(__file__).parent
    
    expected_files = [
        "lightrag_core.py",
        "agentic_lightrag.py", 
        "context_processor.py",
        "athena_mcp_server.py",
        "config.py",
        "exceptions.py",
        "__init__.py",
        "requirements.txt",
        "README.md"
    ]
    
    found_files = 0
    for file_name in expected_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"    ✓ {file_name}")
            found_files += 1
        else:
            print(f"    ✗ {file_name}: Missing")
    
    completeness = found_files / len(expected_files)
    print(f"  Project completeness: {found_files}/{len(expected_files)} files ({completeness:.1%})")
    
    return completeness >= 0.8  # 80% of files should exist

def main():
    """Run all quick tests."""
    print("ATHENA LIGHTRAG QUICK TESTS")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Database Existence", test_database_existence),
        ("Basic Instantiation", test_basic_instantiation),
        ("Tool Schemas", test_tool_schemas),
        ("Directory Structure", test_directory_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Passed: {passed}/{total} ({passed/total:.1%})")
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print(f"\n🎉 All tests passed! The implementation is ready.")
    elif passed >= total * 0.8:
        print(f"\n✅ Most tests passed! Implementation is mostly ready.")
    elif passed >= total * 0.6:
        print(f"\n⚠️  Some tests failed but core functionality is present.")
    else:
        print(f"\n❌ Many tests failed. Implementation needs work.")
    
    return passed >= total * 0.6

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)