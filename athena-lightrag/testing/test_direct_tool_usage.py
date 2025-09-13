#!/usr/bin/env python3
"""
Direct Tool Usage Test
=====================
Quick test to verify that the LightRAG tools are working correctly
and can return specific table structure and column information.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_lightrag import AgenticLightRAG

async def test_direct_lightrag_tools():
    """Test LightRAG tools directly to verify they return the expected data"""
    
    print("🔬 Testing LightRAG Tools Directly")
    print("=" * 50)
    
    try:
        # Initialize LightRAG
        print("🚀 Initializing AgenticLightRAG...")
        agentic_rag = AgenticLightRAG(verbose=True)
        
        # Test 1: Local query for patient journey tables
        print("\n📋 TEST 1: Local Query - Patient Journey Tables")
        print("-" * 40)
        
        local_result = await agentic_rag.lightrag_local_query(
            query="patient journey clinical encounters orders bills",
            top_k=10
        )
        
        print(f"✅ Local query completed")
        print(f"Result type: {type(local_result)}")
        if isinstance(local_result, dict):
            print(f"Success: {local_result.get('success', False)}")
            result_text = local_result.get('result', '')
            print(f"Result length: {len(result_text)} characters")
            
            # Check for specific table references
            if 'athena.athenaone' in result_text:
                print("✅ Found athena.athenaone table references")
            else:
                print("❌ No athena.athenaone table references found")
            
            if any(table in result_text.lower() for table in ['appointment', 'patient', 'charge', 'diagnosis']):
                print("✅ Found relevant table names")
            else:
                print("❌ No relevant table names found")
                
            print(f"Result preview: {result_text[:200]}...")
        
        # Test 2: Hybrid query for relationships
        print("\n📋 TEST 2: Hybrid Query - Table Relationships")
        print("-" * 40)
        
        hybrid_result = await agentic_rag.lightrag_hybrid_query(
            query="appointment patient diagnosis billing relationships",
            max_entity_tokens=3000,
            max_relation_tokens=4000
        )
        
        print(f"✅ Hybrid query completed")
        if isinstance(hybrid_result, dict):
            print(f"Success: {hybrid_result.get('success', False)}")
            result_text = hybrid_result.get('result', '')
            print(f"Result length: {len(result_text)} characters")
            
            # Check for relationship information
            if any(rel in result_text.lower() for rel in ['join', 'relationship', 'foreign key', 'primary key']):
                print("✅ Found relationship information")
            else:
                print("❌ No relationship information found")
                
            print(f"Result preview: {result_text[:200]}...")
        
        # Test 3: SQL generation
        print("\n📋 TEST 3: SQL Generation")
        print("-" * 40)
        
        sql_result = await agentic_rag.lightrag_sql_generation(
            natural_query="Show patient appointments with diagnosis and billing information",
            include_explanation=True
        )
        
        print(f"✅ SQL generation completed")
        if isinstance(sql_result, dict):
            print(f"Success: {sql_result.get('success', False)}")
            result_text = sql_result.get('result', '')
            print(f"Result length: {len(result_text)} characters")
            
            # Check for SQL syntax
            if any(sql in result_text.lower() for sql in ['select', 'from', 'join', 'where']):
                print("✅ Found SQL syntax")
            else:
                print("❌ No SQL syntax found")
                
            print(f"Result preview: {result_text[:200]}...")
        
        print("\n" + "=" * 50)
        print("🎯 DIRECT TOOL TEST COMPLETE")
        print("If all tests show ✅, the tools are working correctly")
        print("If any show ❌, there may be issues with the LightRAG setup")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_lightrag_tools())
