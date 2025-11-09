#!/usr/bin/env python3
"""Quick test for hybrid and multi-hop queries"""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_core import create_athena_lightrag
from agentic_lightrag import create_agentic_lightrag

async def test_hybrid_query():
    """Test hybrid query with hardcoded correct path"""
    working_dir = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"

    print(f"✅ Using database path: {working_dir}")
    print(f"✅ Database exists: {Path(working_dir).exists()}")

    print("\n🧪 Testing Hybrid Query...")

    lightrag = create_athena_lightrag(working_dir=working_dir)

    query = "pharmacy orders medication prescriptions drug orders in athena.athenaone - patient information, provider, pharmacy, insurance payer, submission methods"

    result = await lightrag.query_hybrid_async(
        query,
        top_k=200,
        max_entity_tokens=15000,
        max_relation_tokens=20000
    )

    print(f"✅ Hybrid Query Result:")
    print(f"   Mode: {result.mode}")
    print(f"   Success: {result.error is None}")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    if result.error:
        print(f"   ❌ Error: {result.error}")
    else:
        print(f"   Result length: {len(result.result)} chars")
        print(f"   Preview: {result.result[:200]}...")

    return result

async def test_multi_hop_query():
    """Test multi-hop query with hardcoded correct path"""
    working_dir = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"

    print("\n🧪 Testing Multi-hop Query...")

    agentic = create_agentic_lightrag(working_dir=working_dir, max_internal_steps=8)

    query = "pharmacy orders medication prescriptions drug orders in athena.athenaone - patient information, provider, pharmacy, insurance payer, submission methods"
    objective = "Find all tables and relationships for pharmacy medication orders"

    # Use the correct multi-hop method (no max_steps parameter)
    result = await agentic.execute_multi_hop_reasoning(
        query=query,
        objective=objective
    )

    print(f"✅ Multi-hop Query Result:")
    print(f"   Success: {result.get('success', False)}")
    if result.get('success'):
        response_text = result.get('result', result.get('final_response', ''))
        print(f"   Result length: {len(response_text)} chars")
        print(f"   Steps taken: {result.get('steps', 0)}")
        print(f"   Preview: {response_text[:200]}...")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown')}")

    return result

async def main():
    print("=" * 60)
    print("ATHENA LIGHTRAG - HYBRID vs MULTI-HOP COMPARISON")
    print("=" * 60)

    try:
        hybrid_result = await test_hybrid_query()
        multihop_result = await test_multi_hop_query()

        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Hybrid Success: {hybrid_result.error is None}")
        print(f"Multi-hop Success: {multihop_result['success']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
