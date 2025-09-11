#!/usr/bin/env python3
"""
Athena LightRAG MCP Server - Test Script
=========================================

Simple test script to validate server functionality before production deployment.

Usage:
    python test_server.py

Author: PromptChain Team
Date: 2025
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from athena_lightrag.core import (
    AthenaLightRAG, 
    query_athena_basic,
    query_athena_multi_hop,
    get_athena_database_info
)


async def test_basic_functionality():
    """Test basic functionality of the Athena LightRAG system."""
    print("🧪 Testing Athena LightRAG MCP Server")
    print("=" * 50)
    
    try:
        # Test 1: Database info
        print("\n1. Testing database status...")
        db_info = await get_athena_database_info()
        print("✅ Database info retrieved:")
        print(db_info)
        
        # Test 2: Basic query
        print("\n2. Testing basic query...")
        basic_result = await query_athena_basic(
            query="What tables are in the database?",
            mode="hybrid",
            top_k=10
        )
        print("✅ Basic query successful:")
        print(f"Result preview: {basic_result[:200]}...")
        
        # Test 3: Multi-hop reasoning (shorter for testing)
        print("\n3. Testing multi-hop reasoning...")
        reasoning_result = await query_athena_multi_hop(
            query="How do patient appointments relate to billing?",
            context_strategy="incremental",
            max_steps=2  # Reduced for testing
        )
        print("✅ Multi-hop reasoning successful:")
        print(f"Result preview: {reasoning_result[:200]}...")
        
        print("\n🎉 All tests passed! Server is ready for deployment.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


async def test_error_handling():
    """Test error handling scenarios."""
    print("\n4. Testing error handling...")
    
    try:
        # Test with empty query
        result = await query_athena_basic(query="", mode="hybrid")
        print(f"Empty query handled: {len(result)} chars returned")
        
        # Test with invalid mode (should default)
        result = await query_athena_basic(query="test", mode="invalid_mode")
        print("Invalid mode handled gracefully")
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Starting Athena LightRAG MCP Server Tests...")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable required for testing")
        sys.exit(1)
    
    # Check database
    working_dir = os.getenv("LIGHTRAG_WORKING_DIR", "./athena_lightrag_db")
    if not Path(working_dir).exists():
        print(f"❌ LightRAG database not found at {working_dir}")
        print("Please copy the database from hybridrag project or run data ingestion")
        sys.exit(1)
    
    # Run async tests
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a new loop
            import nest_asyncio
            nest_asyncio.apply()
        
        success = asyncio.run(test_basic_functionality())
        success = success and asyncio.run(test_error_handling())
        
        if success:
            print("\n🚀 All tests completed successfully!")
            print("The server is ready for MCP client connections.")
        else:
            print("\n💥 Some tests failed. Please check the configuration.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()