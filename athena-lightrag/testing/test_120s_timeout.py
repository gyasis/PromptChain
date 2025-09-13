#!/usr/bin/env python3
"""
Test with 120 second timeout to see if tools complete with more time.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena_mcp_server import lightrag_multi_hop_reasoning, lightrag_sql_generation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_120s_timeout():
    """Test with 120s timeout to see completion times."""
    print("⏰ TESTING WITH 120 SECOND TIMEOUT")
    print("Checking if tools complete with more generous timeout")
    print("="*60)
    
    # Test multi-hop reasoning with 120s timeout
    print("\n🔥 TESTING: lightrag_multi_hop_reasoning (120s timeout)")
    
    try:
        start_time = time.time()
        
        result = await lightrag_multi_hop_reasoning(
            query="What are patient data tables?",
            objective="Find patient-related database tables",
            max_steps=2,
            timeout_seconds=120.0  # 2 minutes
        )
        
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.1f}s")
        print(f"🎯 Success: {result.get('success')}")
        
        if result.get("success"):
            print("✅ MULTI-HOP REASONING: COMPLETED")
            print(f"📝 Result length: {len(str(result.get('result', '')))}")
            multihop_success = True
        else:
            print("❌ MULTI-HOP REASONING: FAILED")
            print(f"💥 Error: {result.get('error')}")
            multihop_success = False
            
    except Exception as e:
        print(f"❌ MULTI-HOP REASONING: EXCEPTION - {str(e)}")
        multihop_success = False

    # Test SQL generation
    print(f"\n🔥 TESTING: lightrag_sql_generation (120s timeout)")
    
    try:
        start_time = time.time()
        
        result = await lightrag_sql_generation(
            natural_query="Show me patient appointment tables",
            include_explanation=True,
            timeout_seconds=120.0  # 2 minutes
        )
        
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.1f}s")
        print(f"🎯 Success: {result.get('success')}")
        
        if result.get("success"):
            print("✅ SQL GENERATION: COMPLETED")
            sql_success = True
        else:
            print("❌ SQL GENERATION: FAILED")
            print(f"💥 Error: {result.get('error')}")
            sql_success = False
            
    except Exception as e:
        print(f"❌ SQL GENERATION: EXCEPTION - {str(e)}")
        sql_success = False

    # Results
    print("\n" + "="*60)
    print("📊 120-SECOND TIMEOUT RESULTS:")
    print("="*60)
    
    passed = multihop_success + sql_success
    
    print(f"Multi-hop Reasoning: {'✅ PASS' if multihop_success else '❌ FAIL'}")
    print(f"SQL Generation: {'✅ PASS' if sql_success else '❌ FAIL'}")
    print(f"\nPassed: {passed}/2 tools")
    
    if passed == 2:
        print("\n🎉 SUCCESS: All tools complete with 120s timeout")
        print("💡 Issue may be timing-related rather than fundamental breakage")
        return True
    elif passed == 1:
        print(f"\n⚠️  PARTIAL: {2-passed} tool(s) still have issues even with 120s")
        return False
    else:
        print(f"\n💥 FAILURE: Both tools broken even with 120s timeout")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_120s_timeout())
    sys.exit(0 if success else 1)