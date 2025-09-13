#!/usr/bin/env python3
"""
Test to verify that basic MCP tools still work and only PromptChain integration is fixed.
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
    lightrag_global_query, 
    lightrag_hybrid_query,
    lightrag_context_extract,
    lightrag_multi_hop_reasoning
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_tools():
    """Test that basic MCP tools still work."""
    logger.info("Testing basic MCP tools (should work)...")
    
    try:
        # Test local query
        result = await lightrag_local_query(
            query="patient tables",
            top_k=5,
            timeout_seconds=15.0
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        if result.get("success"):
            logger.info("✅ Basic local query test PASSED")
        else:
            logger.error(f"❌ Basic local query failed: {result.get('error')}")
            return False
            
        # Test global query
        result = await lightrag_global_query(
            query="database schema",
            max_relation_tokens=4000,
            timeout_seconds=15.0
        )
        
        if result.get("success"):
            logger.info("✅ Basic global query test PASSED")
            return True
        else:
            logger.error(f"❌ Basic global query failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Basic tools test FAILED: {str(e)}")
        return False

async def test_multi_hop_reasoning():
    """Test the multi-hop reasoning tool with PromptChain integration."""
    logger.info("Testing multi-hop reasoning (PromptChain integration)...")
    
    try:
        start_time = time.time()
        
        result = await lightrag_multi_hop_reasoning(
            query="Find patient data tables",
            objective="Identify patient-related tables",
            max_steps=2,
            timeout_seconds=30.0
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Multi-hop reasoning completed in {execution_time:.2f}s")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        
        if result.get("success"):
            logger.info("✅ Multi-hop reasoning test PASSED")
            return True
        else:
            logger.error(f"❌ Multi-hop reasoning failed: {result.get('error')}")
            return False
            
    except asyncio.TimeoutError:
        logger.error("❌ Multi-hop reasoning test FAILED: Still getting timeout")
        return False
    except Exception as e:
        logger.error(f"❌ Multi-hop reasoning test FAILED: {str(e)}")
        return False

async def main():
    """Run corrected tests."""
    logger.info("🔧 Testing corrected async fixes...")
    logger.info("="*60)
    
    # Verify database exists
    db_path = Path("/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db")
    if not db_path.exists():
        logger.error("❌ LightRAG database not found. Run ingestion first.")
        return False
    
    # Test basic tools first
    basic_success = await test_basic_tools()
    if not basic_success:
        logger.error("💥 Basic tools are broken! Reverting changes was incorrect.")
        return False
    
    logger.info("✅ Basic tools working correctly")
    
    # Test PromptChain integration
    promptchain_success = await test_multi_hop_reasoning()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 CORRECTED TEST RESULTS:")
    logger.info("="*60)
    
    logger.info(f"Basic MCP Tools: {'✅ WORKING' if basic_success else '❌ BROKEN'}")
    logger.info(f"PromptChain Integration: {'✅ FIXED' if promptchain_success else '❌ STILL BROKEN'}")
    
    if basic_success and promptchain_success:
        logger.info("🎉 All issues resolved correctly!")
        return True
    elif basic_success and not promptchain_success:
        logger.error("⚠️  Basic tools work but PromptChain integration still has issues")
        return False
    else:
        logger.error("💥 Major regression - basic tools are now broken")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)