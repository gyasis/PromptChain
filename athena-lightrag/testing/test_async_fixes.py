#!/usr/bin/env python3
"""
Test script to verify sync/async compatibility fixes.
Tests the critical MCP tools that were previously failing with timeout errors.
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multi_hop_reasoning():
    """Test the multi-hop reasoning tool that was previously failing."""
    logger.info("Testing multi-hop reasoning tool...")
    
    try:
        start_time = time.time()
        
        # Test with a simple query and short timeout
        result = await lightrag_multi_hop_reasoning(
            query="Find all tables containing patient data",
            objective="Identify patient-related tables in the database",
            max_steps=3,
            timeout_seconds=30.0  # Short timeout for testing
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Multi-hop reasoning completed in {execution_time:.2f}s")
        
        # Validate result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should contain success field"
        
        if result.get("success"):
            logger.info("✅ Multi-hop reasoning test PASSED")
            return True
        else:
            logger.error(f"❌ Multi-hop reasoning failed: {result.get('error')}")
            return False
            
    except asyncio.TimeoutError:
        logger.error("❌ Multi-hop reasoning test FAILED: Timeout error still occurs")
        return False
    except Exception as e:
        logger.error(f"❌ Multi-hop reasoning test FAILED: {str(e)}")
        return False

async def test_sql_generation():
    """Test the SQL generation tool that was previously failing."""
    logger.info("Testing SQL generation tool...")
    
    try:
        start_time = time.time()
        
        # Test with a simple query and short timeout
        result = await lightrag_sql_generation(
            natural_query="Show all appointment tables",
            include_explanation=True,
            timeout_seconds=20.0  # Short timeout for testing
        )
        
        execution_time = time.time() - start_time
        logger.info(f"SQL generation completed in {execution_time:.2f}s")
        
        # Validate result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "success" in result, "Result should contain success field"
        
        if result.get("success"):
            logger.info("✅ SQL generation test PASSED")
            return True
        else:
            logger.error(f"❌ SQL generation failed: {result.get('error')}")
            return False
            
    except asyncio.TimeoutError:
        logger.error("❌ SQL generation test FAILED: Timeout error still occurs")
        return False
    except Exception as e:
        logger.error(f"❌ SQL generation test FAILED: {str(e)}")
        return False

async def test_nested_asyncio_compatibility():
    """Test that we can run multiple async tools concurrently without deadlocks."""
    logger.info("Testing nested asyncio compatibility...")
    
    try:
        start_time = time.time()
        
        # Run both tools concurrently to test for deadlocks
        results = await asyncio.gather(
            lightrag_multi_hop_reasoning(
                query="List database schemas", 
                max_steps=2, 
                timeout_seconds=25.0
            ),
            lightrag_sql_generation(
                natural_query="Show table count",
                timeout_seconds=15.0
            ),
            return_exceptions=True
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Concurrent execution completed in {execution_time:.2f}s")
        
        # Check if both completed without exceptions
        success_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i+1} failed with exception: {result}")
            elif isinstance(result, dict) and result.get("success"):
                success_count += 1
                logger.info(f"Task {i+1} succeeded")
            else:
                logger.warning(f"Task {i+1} completed but may have failed: {result}")
        
        if success_count >= 1:  # At least one should succeed
            logger.info("✅ Nested asyncio compatibility test PASSED")
            return True
        else:
            logger.error("❌ Nested asyncio compatibility test FAILED: No tasks succeeded")
            return False
            
    except Exception as e:
        logger.error(f"❌ Nested asyncio compatibility test FAILED: {str(e)}")
        return False

async def main():
    """Run all async compatibility tests."""
    logger.info("🧪 Starting async compatibility tests...")
    logger.info("="*60)
    
    # Verify database exists
    db_path = Path("/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db")
    if not db_path.exists():
        logger.error("❌ LightRAG database not found. Run ingestion first.")
        return False
    
    tests = [
        ("Multi-hop Reasoning", test_multi_hop_reasoning),
        ("SQL Generation", test_sql_generation), 
        ("Nested Asyncio Compatibility", test_nested_asyncio_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST RESULTS SUMMARY:")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED! Sync/async issues appear to be resolved.")
        return True
    else:
        logger.error(f"💥 {failed} tests failed. Sync/async issues may still exist.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)