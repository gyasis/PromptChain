#!/usr/bin/env python3
"""
Test the SPECIFIC tools that were failing with timeout errors.
This will show concrete evidence of success or failure.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_multihop_reasoning():
    """Test the specific multi-hop reasoning tool that was failing."""
    print("\n" + "="*70)
    print("🔍 TESTING: lightrag_multi_hop_reasoning")
    print("📋 This tool was failing with 'MCP error -32001: Request timed out'")
    print("="*70)
    
    try:
        start_time = time.time()
        
        # Call the exact tool that was failing
        result = await lightrag_multi_hop_reasoning(
            query="What are patient data tables?",
            objective="Find patient-related database tables",
            max_steps=3,
            timeout_seconds=30.0
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n📊 EXECUTION RESULTS:")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📤 Result type: {type(result)}")
        print(f"🎯 Success status: {result.get('success', 'Not found')}")
        
        if result.get("success"):
            print(f"✅ RESULT: Multi-hop reasoning tool PASSED")
            print(f"📝 Result preview: {str(result.get('result', ''))[:200]}...")
            if 'reasoning_steps' in result:
                print(f"🧠 Reasoning steps completed: {len(result['reasoning_steps'])}")
            return True
        else:
            print(f"❌ RESULT: Multi-hop reasoning tool FAILED")
            print(f"💥 Error: {result.get('error', 'Unknown error')}")
            return False
            
    except asyncio.TimeoutError:
        print(f"❌ RESULT: Tool still times out (original issue NOT fixed)")
        return False
    except Exception as e:
        print(f"❌ RESULT: Tool failed with exception: {str(e)}")
        return False

async def test_sql_generation():
    """Test the specific SQL generation tool that was failing."""
    print("\n" + "="*70)
    print("🔍 TESTING: lightrag_sql_generation")
    print("📋 This tool was failing with 'MCP error -32001: Request timed out'")
    print("="*70)
    
    try:
        start_time = time.time()
        
        # Call the exact tool that was failing
        result = await lightrag_sql_generation(
            natural_query="Show me all patient appointment tables",
            include_explanation=True,
            timeout_seconds=25.0
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n📊 EXECUTION RESULTS:")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📤 Result type: {type(result)}")
        print(f"🎯 Success status: {result.get('success', 'Not found')}")
        
        if result.get("success"):
            print(f"✅ RESULT: SQL generation tool PASSED")
            sql_result = result.get('sql', 'No SQL found')
            print(f"🗄️  Generated SQL preview: {str(sql_result)[:150]}...")
            if result.get('explanation'):
                print(f"📖 Explanation provided: Yes")
            return True
        else:
            print(f"❌ RESULT: SQL generation tool FAILED")
            print(f"💥 Error: {result.get('error', 'Unknown error')}")
            return False
            
    except asyncio.TimeoutError:
        print(f"❌ RESULT: Tool still times out (original issue NOT fixed)")
        return False
    except Exception as e:
        print(f"❌ RESULT: Tool failed with exception: {str(e)}")
        return False

async def main():
    """Run the specific failing tool tests."""
    print("🧪 SPECIFIC TOOL FAILURE TEST")
    print("Testing the exact tools that were failing with timeout errors")
    print("Original error: 'MCP error -32001: Request timed out'")
    
    # Verify database exists
    db_path = Path("/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db")
    if not db_path.exists():
        print("❌ LightRAG database not found. Cannot run tests.")
        return False
    
    # Test both failing tools
    results = []
    
    # Test 1: Multi-hop reasoning (was failing)
    multihop_success = await test_multihop_reasoning()
    results.append(("Multi-hop Reasoning", multihop_success))
    
    # Test 2: SQL generation (was failing)  
    sql_success = await test_sql_generation()
    results.append(("SQL Generation", sql_success))
    
    # Final summary
    print("\n" + "="*70)
    print("📈 FINAL TEST SUMMARY")
    print("="*70)
    
    passed = 0
    for tool_name, success in results:
        status = "✅ FIXED" if success else "❌ STILL BROKEN"
        print(f"{tool_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 FINAL RESULT: {passed}/2 previously failing tools are now working")
    
    if passed == 2:
        print("🎉 SUCCESS: All timeout issues have been resolved!")
        print("🚀 MCP server is ready for production use")
        return True
    elif passed == 1:
        print("⚠️  PARTIAL: 1 tool fixed, 1 still has issues")
        return False
    else:
        print("💥 FAILURE: Both tools still have timeout issues")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)