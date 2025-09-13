#!/usr/bin/env python3
"""
Final verification that both basic tools and PromptChain integration work properly.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena_mcp_server import (
    lightrag_local_query,
    lightrag_multi_hop_reasoning,
    lightrag_sql_generation
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def quick_test():
    """Quick test to verify everything works."""
    logger.info("🎯 FINAL VERIFICATION TEST")
    logger.info("="*50)
    
    try:
        # Test 1: Basic tool (should work fast)
        logger.info("1️⃣  Testing basic tool...")
        start = time.time()
        result = await lightrag_local_query(
            query="patient data",
            top_k=3,
            timeout_seconds=10.0
        )
        elapsed = time.time() - start
        
        basic_success = result.get("success", False)
        logger.info(f"   Basic tool: {'✅ PASSED' if basic_success else '❌ FAILED'} ({elapsed:.1f}s)")
        
        # Test 2: Multi-hop reasoning (previously failing)
        logger.info("2️⃣  Testing multi-hop reasoning...")
        start = time.time()
        result = await lightrag_multi_hop_reasoning(
            query="List patient tables",
            max_steps=2,
            timeout_seconds=20.0
        )
        elapsed = time.time() - start
        
        multihop_success = result.get("success", False)
        logger.info(f"   Multi-hop: {'✅ PASSED' if multihop_success else '❌ FAILED'} ({elapsed:.1f}s)")
        
        # Test 3: SQL generation (previously failing) 
        logger.info("3️⃣  Testing SQL generation...")
        start = time.time()
        result = await lightrag_sql_generation(
            natural_query="Show patient tables",
            timeout_seconds=15.0
        )
        elapsed = time.time() - start
        
        sql_success = result.get("success", False)
        logger.info(f"   SQL generation: {'✅ PASSED' if sql_success else '❌ FAILED'} ({elapsed:.1f}s)")
        
        # Results
        logger.info("\n" + "="*50)
        logger.info("📊 FINAL RESULTS:")
        passed = sum([basic_success, multihop_success, sql_success])
        logger.info(f"✅ {passed}/3 tests passed")
        
        if passed == 3:
            logger.info("🎉 ALL SYNC/ASYNC ISSUES RESOLVED!")
            logger.info("🚀 MCP server is ready for production use")
            return True
        else:
            logger.error(f"💥 {3-passed} tests still failing")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(quick_test())
    sys.exit(0 if success else 1)