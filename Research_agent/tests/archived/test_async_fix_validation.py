#!/usr/bin/env python3
"""
Test script to validate the async event loop fix for choice 2 (Reasoning mode)
in the LightRAG enhanced demo.

This tests that the ReasoningRAGAgent no longer causes event loop conflicts
when using async tool functions.
"""

import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the examples directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "examples" / "lightrag_demo"))

try:
    from lightrag_enhanced_demo import (
        LightRAG, 
        ReasoningRAGAgent,
        QueryParam,
        CitationTracker
    )
    print("✅ Successfully imported modules from lightrag_enhanced_demo")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

async def test_reasoning_agent_async_fix():
    """Test that the ReasoningRAGAgent works without async event loop conflicts"""
    
    print("\n🧪 Testing ReasoningRAGAgent async fix...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        try:
            # Initialize LightRAG with minimal settings
            rag = LightRAG(
                working_dir=temp_dir,
                llm_model_func=lambda messages, **kwargs: "Mock LightRAG response for testing async fix"
            )
            print("✅ LightRAG instance created successfully")
            
            # Create ReasoningRAGAgent (this triggers setup_reasoning_chain)
            reasoning_agent = ReasoningRAGAgent(rag, name="TestAgent")
            print("✅ ReasoningRAGAgent created successfully")
            
            # Test the core functionality that was causing issues
            test_question = "What are the key findings in machine learning research?"
            
            print(f"🔍 Testing reasoning with question: '{test_question}'")
            
            # This should NOT cause event loop conflicts anymore
            result = await reasoning_agent.reason_with_citations(test_question)
            
            print("✅ Reasoning completed without event loop errors!")
            print(f"📝 Result preview: {result[:200]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print(f"📍 Error type: {type(e).__name__}")
            
            # Check if it's the specific event loop error we were trying to fix
            if "bound to a different event loop" in str(e):
                print("🚨 CRITICAL: The async event loop fix did NOT work!")
                print("   This is the exact error we were trying to fix.")
                return False
            elif "PriorityQueue" in str(e) and "event loop" in str(e):
                print("🚨 CRITICAL: Event loop conflict still present!")
                return False
            else:
                print("ℹ️  Different error - may be related to test setup rather than async fix")
                return False

def test_tool_function_async_signature():
    """Test that tool functions are now properly async"""
    
    print("\n🔍 Testing tool function signatures...")
    
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            rag = LightRAG(
                working_dir=temp_dir,
                llm_model_func=lambda messages, **kwargs: "Mock response"
            )
            
            reasoning_agent = ReasoningRAGAgent(rag)
            
            # Check if tool functions are properly async
            # These are nested functions, so we need to check them differently
            chain = reasoning_agent.chain
            local_functions = chain.local_tool_functions
            
            print(f"📊 Registered tool functions: {list(local_functions.keys())}")
            
            # Check each function
            async_function_count = 0
            for func_name, func in local_functions.items():
                is_async = asyncio.iscoroutinefunction(func)
                status = "✅ ASYNC" if is_async else "❌ SYNC"
                print(f"   {func_name}: {status}")
                if is_async:
                    async_function_count += 1
            
            expected_async_functions = [
                "search_and_cite",
                "analyze_entity_with_citations", 
                "extract_themes_with_sources",
                "find_contradictions"
            ]
            
            if async_function_count >= len(expected_async_functions):
                print("✅ All tool functions are properly async!")
                return True
            else:
                print(f"❌ Only {async_function_count} functions are async, expected {len(expected_async_functions)}")
                return False
                
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("=" * 70)
    print("🎯 ASYNC EVENT LOOP FIX VALIDATION TEST")
    print("=" * 70)
    print("Testing the fix for choice 2 (Reasoning mode) async conflicts")
    print()
    
    # Test 1: Tool function signatures
    signature_test_passed = test_tool_function_async_signature()
    
    # Test 2: Actual async execution
    execution_test_passed = await test_reasoning_agent_async_fix()
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"🔧 Tool Function Signatures: {'✅ PASS' if signature_test_passed else '❌ FAIL'}")
    print(f"⚡ Async Execution Test:     {'✅ PASS' if execution_test_passed else '❌ FAIL'}")
    
    overall_success = signature_test_passed and execution_test_passed
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! The async event loop fix is working correctly.")
        print("   Choice 2 (Reasoning mode) should now work without event loop conflicts.")
    else:
        print("\n⚠️  TESTS FAILED! The async event loop fix needs further work.")
        
    print("=" * 70)
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)