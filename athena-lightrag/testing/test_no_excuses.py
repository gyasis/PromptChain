#!/usr/bin/env python3
"""
NO EXCUSES TEST - 60 second timeouts, no assumptions.
Either tools work or they don't. Timeout = FAILURE.
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

async def test_multihop_reasoning():
    """Test multi-hop reasoning with 60 second timeout - no excuses."""
    print("\n🔥 TESTING: lightrag_multi_hop_reasoning (60s timeout)")
    
    try:
        start_time = time.time()
        
        result = await lightrag_multi_hop_reasoning(
            query="What are patient data tables?",
            objective="Find patient-related database tables",
            max_steps=2,  # Reduced steps
            timeout_seconds=60.0  # Full 60 seconds
        )
        
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.1f}s")
        print(f"🎯 Success: {result.get('success')}")
        
        if result.get("success"):
            print("✅ MULTI-HOP REASONING: PASSED")
            print(f"📝 Result length: {len(str(result.get('result', '')))}")
            return True
        else:
            print("❌ MULTI-HOP REASONING: FAILED")
            print(f"💥 Error: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ MULTI-HOP REASONING: EXCEPTION - {str(e)}")
        return False

async def test_sql_generation():
    """Test SQL generation with 60 second timeout - no excuses."""
    print("\n🔥 TESTING: lightrag_sql_generation (60s timeout)")
    
    try:
        start_time = time.time()
        
        result = await lightrag_sql_generation(
            natural_query="Show me patient appointment tables",
            include_explanation=True,
            timeout_seconds=60.0  # Full 60 seconds
        )
        
        execution_time = time.time() - start_time
        
        print(f"⏱️  Execution time: {execution_time:.1f}s")
        print(f"🎯 Success: {result.get('success')}")
        
        if result.get("success"):
            print("✅ SQL GENERATION: PASSED")
            sql = result.get('sql', '')
            print(f"🗄️  SQL length: {len(sql)} chars")
            return True
        else:
            print("❌ SQL GENERATION: FAILED") 
            print(f"💥 Error: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ SQL GENERATION: EXCEPTION - {str(e)}")
        return False

async def main():
    """Run tests with 60s timeouts - no excuses for failures."""
    print("🚫 NO EXCUSES TEST - 60 Second Timeouts")
    print("Timeout = FAILURE. No assumptions. No excuses.")
    print("="*50)
    
    # Verify database exists
    db_path = Path("/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db")
    if not db_path.exists():
        print("❌ Database not found. Test cannot proceed.")
        return False
    
    # Test both tools
    multihop_success = await test_multihop_reasoning()
    sql_success = await test_sql_generation()
    
    # Final verdict - NO EXCUSES
    print("\n" + "="*50)
    print("🏁 FINAL VERDICT - NO EXCUSES")
    print("="*50)
    
    passed = multihop_success + sql_success
    
    print(f"Multi-hop Reasoning: {'✅ PASS' if multihop_success else '❌ FAIL'}")
    print(f"SQL Generation: {'✅ PASS' if sql_success else '❌ FAIL'}")
    print(f"\nPassed: {passed}/2 tools")
    
    if passed == 2:
        print("\n🎉 SUCCESS: All tools working correctly")
        return True
    elif passed == 1:
        print(f"\n⚠️  PARTIAL FAILURE: {2-passed} tool(s) still broken")
        return False
    else:
        print(f"\n💥 COMPLETE FAILURE: Both tools broken")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)