#!/usr/bin/env python3
"""
Test Fixed MCP Tools - SNOMED Procedures Query
==============================================
Test the corrected multi-hop reasoning and SQL generation tools 
with the user's specific query about SNOMED codes.
"""

import asyncio
import time
from athena_mcp_server import (
    lightrag_multi_hop_reasoning, 
    lightrag_sql_generation,
    MultiHopReasoningParams,
    SQLGenerationParams
)

async def test_fixed_tools():
    """Test the corrected MCP tools with SNOMED procedures query."""
    print("🧪 TESTING FIXED MCP TOOLS")
    print("="*60)
    
    # User's specific query about SNOMED codes
    snomed_query = "what tables have snomed codes for procedures and processes?"
    
    # Test 1: Multi-hop reasoning tool
    print("\n🔍 TEST 1: Multi-hop Reasoning Tool")
    print("-" * 40)
    
    try:
        start_time = time.time()
        
        multihop_result = await lightrag_multi_hop_reasoning(
            query=snomed_query,
            objective="Find tables containing SNOMED codes for medical procedures and processes in Athena database",
            max_steps=3
        )
        
        execution_time = time.time() - start_time
        print(f"⏱️ Execution time: {execution_time:.2f} seconds")
        
        if multihop_result.get("success"):
            print("✅ SUCCESS: Multi-hop reasoning completed")
            print(f"📊 Result preview: {str(multihop_result.get('result', ''))[:200]}...")
            print(f"🧠 Reasoning steps: {len(multihop_result.get('reasoning_steps', []))}")
            print(f"📚 Context sources: {len(multihop_result.get('accumulated_contexts', []))}")
        else:
            print("❌ FAILURE: Multi-hop reasoning failed")
            print(f"❗ Error: {multihop_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"💥 EXCEPTION in multi-hop reasoning: {e}")
    
    # Test 2: SQL generation tool  
    print("\n🗄️ TEST 2: SQL Generation Tool")
    print("-" * 40)
    
    try:
        start_time = time.time()
        
        sql_result = await lightrag_sql_generation(
            natural_query=snomed_query,
            include_explanation=True
        )
        
        execution_time = time.time() - start_time
        print(f"⏱️ Execution time: {execution_time:.2f} seconds")
        
        if sql_result.get("success"):
            print("✅ SUCCESS: SQL generation completed")
            print(f"🔍 Generated SQL preview: {str(sql_result.get('sql', ''))[:200]}...")
            print(f"📖 Has explanation: {'yes' if sql_result.get('explanation') else 'no'}")
            print(f"📚 Context used: {'yes' if sql_result.get('context_summary') else 'no'}")
        else:
            print("❌ FAILURE: SQL generation failed")
            print(f"❗ Error: {sql_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"💥 EXCEPTION in SQL generation: {e}")
    
    # Test 3: Quick validation of basic tools
    print("\n🧬 TEST 3: Quick Validation of Core Tools")
    print("-" * 40)
    
    from athena_mcp_server import lightrag_local_query
    
    try:
        start_time = time.time()
        
        local_result = await lightrag_local_query(
            query="SNOMED codes tables",
            top_k=5
        )
        
        execution_time = time.time() - start_time
        print(f"⏱️ Local query execution time: {execution_time:.2f} seconds")
        
        if local_result.get("success"):
            print("✅ SUCCESS: Local query working")
        else:
            print(f"❌ Local query failed: {local_result.get('error', 'Unknown')}")
            
    except Exception as e:
        print(f"💥 EXCEPTION in local query: {e}")
    
    print("\n🏁 TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    print("🚀 Starting Fixed MCP Tools Test")
    asyncio.run(test_fixed_tools())