#!/usr/bin/env python3
"""
Environment validation script for Athena LightRAG MCP Server
Tests that all required dependencies can be imported and initialized
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("ATHENA LIGHTRAG MCP SERVER - ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    # Test core Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("❌ ERROR: Python 3.10+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Test LightRAG import
    try:
        import lightrag
        print(f"✅ LightRAG imported successfully (version: {lightrag.__version__ if hasattr(lightrag, '__version__') else 'unknown'})")
    except ImportError as e:
        print(f"❌ Failed to import LightRAG: {e}")
        return False
    
    # Test PromptChain import
    try:
        import promptchain
        from promptchain import PromptChain
        print(f"✅ PromptChain imported successfully (version: {promptchain.__version__ if hasattr(promptchain, '__version__') else 'unknown'})")
    except ImportError as e:
        print(f"❌ Failed to import PromptChain: {e}")
        return False
    
    # Test FastMCP import
    try:
        import fastmcp
        print(f"✅ FastMCP imported successfully (version: {fastmcp.__version__ if hasattr(fastmcp, '__version__') else 'unknown'})")
    except ImportError as e:
        print(f"❌ Failed to import FastMCP: {e}")
        return False
    
    # Test OpenAI import
    try:
        import openai
        print(f"✅ OpenAI imported successfully (version: {openai.__version__ if hasattr(openai, '__version__') else 'unknown'})")
    except ImportError as e:
        print(f"❌ Failed to import OpenAI: {e}")
        return False
    
    # Test other key dependencies
    dependencies = [
        ('python-dotenv', 'dotenv'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('asyncio-throttle', 'asyncio_throttle'),
        ('colorama', 'colorama'),
    ]
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {dep_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {dep_name}: {e}")
            return False
    
    return True

def test_database():
    """Test that the LightRAG database exists and is accessible."""
    print("\n" + "=" * 60)
    print("DATABASE VALIDATION")
    print("=" * 60)
    
    db_path = Path("./athena_lightrag_db")
    if db_path.exists() and db_path.is_dir():
        print(f"✅ Database directory exists: {db_path.absolute()}")
        
        # List contents
        contents = list(db_path.iterdir())
        print(f"Database contains {len(contents)} items:")
        for item in contents[:10]:  # Show first 10 items
            print(f"  - {item.name}")
        if len(contents) > 10:
            print(f"  ... and {len(contents) - 10} more items")
        
        return True
    else:
        print(f"❌ Database directory not found: {db_path.absolute()}")
        return False

def test_basic_functionality():
    """Test basic LightRAG functionality."""
    print("\n" + "=" * 60)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        # Import required modules
        from lightrag import LightRAG, QueryParam
        
        print("✅ LightRAG core classes imported successfully")
        
        # Test basic initialization (without actual API calls)
        working_dir = "./athena_lightrag_db"
        if os.path.exists(working_dir):
            print(f"✅ Working directory verified: {working_dir}")
        else:
            print(f"❌ Working directory not found: {working_dir}")
            return False
        
        # Test QueryParam instantiation
        query_param = QueryParam(mode="naive")
        print("✅ QueryParam instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_promptchain_functionality():
    """Test PromptChain basic functionality."""
    print("\n" + "=" * 60)
    print("PROMPTCHAIN FUNCTIONALITY TEST")  
    print("=" * 60)
    
    try:
        from promptchain import PromptChain
        from promptchain.utils.execution_history_manager import ExecutionHistoryManager
        from promptchain.utils.mcp_helpers import MCPHelper
        
        print("✅ PromptChain core classes imported successfully")
        
        # Test basic PromptChain initialization
        chain = PromptChain(
            models=["openai/gpt-4"],
            instructions=["Test instruction: {input}"],
            verbose=False
        )
        print("✅ PromptChain instance created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ PromptChain functionality test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("Starting environment validation...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Database Tests", test_database),
        ("Basic Functionality Tests", test_basic_functionality),
        ("PromptChain Tests", test_promptchain_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / len(results) * 100
    print(f"\nOverall: {passed}/{len(results)} tests passed ({success_rate:.1f}%)")
    
    if passed == len(results):
        print("\n🎉 Environment validation SUCCESSFUL! Ready for development.")
        return 0
    else:
        print(f"\n⚠️  Environment validation INCOMPLETE. {len(results) - passed} issues need resolution.")
        return 1

if __name__ == "__main__":
    exit(main())