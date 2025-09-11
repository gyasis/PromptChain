#!/usr/bin/env python3
"""
Test Schema Discovery for Athena Database
=========================================
Test the multi-hop reasoning tool with the user's specific schema discovery query.
"""

import asyncio
import time
from athena_mcp_server import lightrag_multi_hop_reasoning

async def test_schema_discovery():
    """Test schema discovery query with improved timeout handling."""
    print("🗄️ TESTING ATHENA DATABASE SCHEMA DISCOVERY")
    print("="*60)
    
    # User's specific query
    schema_query = "list the schemas of the athena database"
    objective = "we need to find the map of the athena database and all schemas and definitions"
    
    print(f"📋 Query: {schema_query}")
    print(f"🎯 Objective: {objective}")
    print(f"📊 Max Steps: 8")
    
    try:
        start_time = time.time()
        
        # Test with the exact parameters from the user
        result = await lightrag_multi_hop_reasoning(
            query=schema_query,
            objective=objective,
            max_steps=8
        )
        
        execution_time = time.time() - start_time
        print(f"\n⏱️ Total execution time: {execution_time:.2f} seconds")
        
        if result.get("success"):
            print("✅ SUCCESS: Schema discovery completed")
            print(f"\n📊 RESULTS:")
            print(f"Result length: {len(str(result.get('result', '')))} characters")
            print(f"Reasoning steps: {result.get('reasoning_steps', 'N/A')}")
            print(f"Context sources: {result.get('accumulated_contexts', 'N/A')}")
            print(f"Tokens used: {result.get('total_tokens_used', 'N/A')}")
            
            # Show preview of results
            result_text = result.get('result', '')
            if result_text:
                print(f"\n📝 RESULT PREVIEW:")
                print("-" * 40)
                print(result_text[:500] + ("..." if len(result_text) > 500 else ""))
                print("-" * 40)
            
        else:
            print("❌ FAILURE: Schema discovery failed")
            error = result.get('error', 'Unknown error')
            print(f"❗ Error: {error}")
            
            if "timeout" in error.lower():
                print("\n🔍 TIMEOUT ANALYSIS:")
                print("- Multi-hop reasoning exceeded time limit")
                print("- Consider reducing max_steps parameter")
                print("- Check if LightRAG knowledge graph contains schema information")
                
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n💥 EXCEPTION after {execution_time:.2f} seconds: {e}")
        return {"success": False, "error": str(e), "exception_type": type(e).__name__}

async def test_simpler_schema_query():
    """Test with a simpler schema-related query."""
    print("\n🔍 TESTING SIMPLER SCHEMA QUERY")
    print("="*60)
    
    simple_query = "what schemas exist in the athena database"
    
    try:
        start_time = time.time()
        
        result = await lightrag_multi_hop_reasoning(
            query=simple_query,
            objective="Find available schemas in Athena database",
            max_steps=3  # Reduced steps
        )
        
        execution_time = time.time() - start_time
        print(f"⏱️ Execution time: {execution_time:.2f} seconds")
        
        if result.get("success"):
            print("✅ SUCCESS: Simple schema query completed")
            result_text = result.get('result', '')
            print(f"📝 Result: {result_text[:300]}...")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"💥 EXCEPTION: {e}")

if __name__ == "__main__":
    print("🚀 Starting Schema Discovery Test")
    
    async def main():
        await test_schema_discovery()
        await test_simpler_schema_query()
        print("\n🏁 SCHEMA DISCOVERY TESTING COMPLETE")
    
    asyncio.run(main())