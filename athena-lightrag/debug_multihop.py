#!/usr/bin/env python3
"""
Debug Multi-Hop Reasoning Tool Timeouts
========================================
Direct testing of multi-hop reasoning tool to identify timeout causes.
"""

import asyncio
import logging
import time
from athena_mcp_server import MultiHopReasoningParams
from agentic_lightrag import create_agentic_lightrag

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_multihop_reasoning():
    """Debug multi-hop reasoning timeout issues with detailed logging."""
    print("🔍 DEBUG: Multi-Hop Reasoning Tool Timeout Analysis")
    print("="*60)
    
    # Test query from user about SNOMED codes
    test_query = "what tables have snomed codes for procedures and processes?"
    
    try:
        # Step 1: Create agentic lightrag instance
        print("📊 STEP 1: Creating AgenticLightRAG instance...")
        start_time = time.time()
        agentic = create_agentic_lightrag()  # This is sync, not async
        init_time = time.time() - start_time
        print(f"✅ AgenticLightRAG created in {init_time:.2f} seconds")
        
        # Step 2: Create parameters
        print("\n📋 STEP 2: Creating MultiHopReasoningParams...")
        params = MultiHopReasoningParams(
            query=test_query,
            objective='Find tables containing SNOMED codes for medical procedures and processes in Athena database',
            max_steps=3  # Reduced from 8 to avoid timeouts
        )
        print(f"Query: {params.query}")
        print(f"Objective: {params.objective}")
        print(f"Max steps: {params.max_steps}")
        
        # Step 3: Test multi-hop reasoning with timeout monitoring
        print("\n🚀 STEP 3: Executing multi-hop reasoning...")
        reasoning_start = time.time()
        
        # Add timeout wrapper
        try:
            result = await asyncio.wait_for(
                agentic.execute_multi_hop_reasoning(
                    query=params.query,
                    objective=params.objective,
                    reset_context=True
                ),
                timeout=120  # 2 minute timeout
            )
            reasoning_time = time.time() - reasoning_start
            print(f"✅ Multi-hop reasoning completed in {reasoning_time:.2f} seconds")
            
        except asyncio.TimeoutError:
            reasoning_time = time.time() - reasoning_start
            print(f"❌ TIMEOUT: Multi-hop reasoning timed out after {reasoning_time:.2f} seconds")
            print("🔍 TIMEOUT ANALYSIS:")
            print("- Reasoning chain may be stuck in internal loops")
            print("- LLM calls may be taking too long")
            print("- Tool calls may be hanging")
            print("- PromptChain AgenticStepProcessor may need timeout configuration")
            return {
                "success": False,
                "error": "Timeout during multi-hop reasoning",
                "timeout_seconds": reasoning_time,
                "suggested_fixes": [
                    "Reduce max_steps parameter",
                    "Add timeout to AgenticStepProcessor",
                    "Check LLM API response times",
                    "Optimize tool call performance"
                ]
            }
        
        # Step 4: Analyze results
        print("\n📈 STEP 4: Analyzing results...")
        print("Result structure:")
        for key, value in result.items():
            if key == "step_outputs" and isinstance(value, list):
                print(f"  {key}: {len(value)} step outputs")
            elif key == "reasoning_steps" and isinstance(value, list):
                print(f"  {key}: {len(value)} reasoning steps")
            elif key == "accumulated_contexts" and isinstance(value, list):
                print(f"  {key}: {len(value)} accumulated contexts")
            elif len(str(value)) > 100:
                print(f"  {key}: {str(value)[:100]}... (truncated)")
            else:
                print(f"  {key}: {value}")
        
        # Success case
        if result.get("success"):
            print("\n✅ SUCCESS: Multi-hop reasoning completed successfully")
        else:
            print("\n❌ FAILURE: Multi-hop reasoning failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.exception("Multi-hop reasoning debug failed")
        print(f"\n❌ EXCEPTION after {error_time:.2f} seconds: {e}")
        return {
            "success": False,
            "error": str(e),
            "exception_type": type(e).__name__
        }

async def debug_sql_generation():
    """Debug SQL generation tool issues."""
    print("\n🔍 DEBUG: SQL Generation Tool Analysis")
    print("="*60)
    
    from context_processor import SQLGenerator
    from lightrag_core import create_athena_lightrag
    
    try:
        # Create SQL generator
        print("📊 Creating SQL Generator...")
        lightrag_core = create_athena_lightrag()
        sql_generator = SQLGenerator(lightrag_core)
        
        # Test SQL generation
        test_query = "Find tables with SNOMED codes for procedures"
        print(f"Query: {test_query}")
        
        result = await sql_generator.generate_sql_from_query(test_query)
        print("SQL Generation Result:")
        print(f"  success: {result.get('success')}")
        print(f"  sql: {result.get('sql', 'None')}")
        if not result.get('success'):
            print(f"  error: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.exception("SQL generation debug failed")
        print(f"❌ SQL Generation Exception: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Multi-Hop Reasoning and SQL Generation Debug Session")
    
    async def main():
        # Debug multi-hop reasoning
        multihop_result = await debug_multihop_reasoning()
        
        # Debug SQL generation
        sql_result = await debug_sql_generation()
        
        print("\n📊 SUMMARY REPORT")
        print("="*60)
        print(f"Multi-hop reasoning success: {multihop_result.get('success', False)}")
        print(f"SQL generation success: {sql_result.get('success', False)}")
        
        if not multihop_result.get('success'):
            print(f"Multi-hop error: {multihop_result.get('error', 'Unknown')}")
            
        if not sql_result.get('success'):
            print(f"SQL generation error: {sql_result.get('error', 'Unknown')}")
    
    asyncio.run(main())