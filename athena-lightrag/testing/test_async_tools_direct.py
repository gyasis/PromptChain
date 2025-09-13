#!/usr/bin/env python3
"""
Direct test of our async tool functions to see if they have hidden blocking issues.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_lightrag import LightRAGToolsProvider
from lightrag_core import create_athena_lightrag

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_async_tools_directly():
    """Test our async tool functions directly without PromptChain."""
    print("🔍 DIRECT ASYNC TOOL TEST")
    print("Testing our async LightRAG tools without PromptChain overhead")
    print("="*60)
    
    # Initialize tools provider
    working_dir = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db"
    lightrag_core = create_athena_lightrag(working_dir=working_dir)
    tools_provider = LightRAGToolsProvider(lightrag_core)
    
    # Test each async tool function directly
    tests = [
        ("lightrag_global_query", tools_provider.lightrag_global_query, {"query": "patient data"}),
        ("lightrag_local_query", tools_provider.lightrag_local_query, {"query": "patient tables", "top_k": 5}),
        ("lightrag_hybrid_query", tools_provider.lightrag_hybrid_query, {"query": "patient info"}),
        ("lightrag_mix_query", tools_provider.lightrag_mix_query, {"query": "patient data", "top_k": 3}),
    ]
    
    total_start = time.time()
    
    for tool_name, tool_func, args in tests:
        print(f"\n🔧 Testing: {tool_name}")
        start_time = time.time()
        
        try:
            # Call the async function directly
            result = await tool_func(**args)
            execution_time = time.time() - start_time
            
            print(f"   ⏱️  Time: {execution_time:.1f}s")
            print(f"   📏 Length: {len(result)} chars")
            print(f"   ✅ SUCCESS")
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ⏱️  Time: {execution_time:.1f}s")
            print(f"   ❌ ERROR: {str(e)}")
    
    total_time = time.time() - total_start
    print(f"\n📊 TOTAL DIRECT TEST TIME: {total_time:.1f}s")
    
    return total_time

async def test_concurrent_async_tools():
    """Test multiple async tools running concurrently."""
    print(f"\n🔄 CONCURRENT ASYNC TOOL TEST")
    print("Testing tools running in parallel to check for blocking")
    print("="*60)
    
    # Initialize tools provider
    working_dir = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db"
    lightrag_core = create_athena_lightrag(working_dir=working_dir)
    tools_provider = LightRAGToolsProvider(lightrag_core)
    
    start_time = time.time()
    
    # Run multiple tools concurrently
    tasks = [
        tools_provider.lightrag_global_query("patient data"),
        tools_provider.lightrag_local_query("patient tables", top_k=3),
        tools_provider.lightrag_hybrid_query("patient info"),
    ]
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = time.time() - start_time
        
        print(f"⏱️  Concurrent execution time: {execution_time:.1f}s")
        
        successes = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"   Task {i+1}: ❌ ERROR - {result}")
            else:
                print(f"   Task {i+1}: ✅ SUCCESS ({len(result)} chars)")
                successes += 1
        
        print(f"🎯 {successes}/3 concurrent tasks succeeded")
        
        if execution_time < 15 and successes >= 2:
            print("✅ CONCURRENT TEST: Good performance, no blocking detected")
            return True
        else:
            print("⚠️  CONCURRENT TEST: May have blocking issues")
            return False
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ CONCURRENT TEST FAILED: {e} (after {execution_time:.1f}s)")
        return False

async def main():
    """Run direct async tool tests."""
    print("🧪 DIRECT ASYNC TOOL ANALYSIS")
    print("Identifying if our async tools have hidden performance issues")
    print("="*80)
    
    # Test 1: Direct sequential calls
    sequential_time = await test_async_tools_directly()
    
    # Test 2: Concurrent calls
    concurrent_success = await test_concurrent_async_tools()
    
    print("\n" + "="*80)
    print("📊 ASYNC TOOL ANALYSIS RESULTS")
    print("="*80)
    
    if sequential_time < 20 and concurrent_success:
        print("✅ CONCLUSION: Async tools perform well directly")
        print("💡 The bottleneck is likely in PromptChain integration or AgenticStepProcessor")
        return True
    else:
        print("❌ CONCLUSION: Async tools themselves have performance issues") 
        print("🔍 The issue is in our LightRAG tool implementations")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)