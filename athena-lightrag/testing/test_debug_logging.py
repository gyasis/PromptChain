#!/usr/bin/env python3
"""
Add debug logging to the multi-hop tool to pinpoint exact bottlenecks.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena_mcp_server import lightrag_multi_hop_reasoning

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_with_debug_timing():
    """Test multi-hop with detailed timing logs."""
    print("🔍 DEBUG TIMING TEST - Multi-hop Reasoning")
    print("Looking for bottlenecks in the execution chain")
    print("="*60)
    
    start_time = time.time()
    
    # Patch the agentic_step_processor to add timing logs
    import importlib
    agentic_module = importlib.import_module('promptchain.utils.agentic_step_processor')
    
    # Store original run_async method
    original_run_async = agentic_module.AgenticStepProcessor.run_async
    
    async def debug_run_async(self, initial_input, available_tools, llm_runner, tool_executor):
        """Wrapped run_async with detailed timing."""
        step_start = time.time()
        logger.critical(f"🟡 AGENTIC STEP START - Tools: {len(available_tools)}")
        
        # Wrap llm_runner to time individual LLM calls
        call_count = 0
        async def timed_llm_runner(messages, tools, tool_choice=None):
            nonlocal call_count
            call_count += 1
            llm_start = time.time()
            logger.critical(f"🔵 LLM CALL #{call_count} START - Messages: {len(messages)}, Tools: {len(tools)}")
            
            result = await llm_runner(messages, tools, tool_choice)
            
            llm_elapsed = time.time() - llm_start
            logger.critical(f"🔵 LLM CALL #{call_count} END - {llm_elapsed:.1f}s")
            
            return result
        
        try:
            result = await original_run_async(self, initial_input, available_tools, timed_llm_runner, tool_executor)
            step_elapsed = time.time() - step_start
            logger.critical(f"🟡 AGENTIC STEP END - {step_elapsed:.1f}s, {call_count} LLM calls")
            return result
        except Exception as e:
            step_elapsed = time.time() - step_start
            logger.critical(f"🔴 AGENTIC STEP FAILED - {step_elapsed:.1f}s, Error: {e}")
            raise
    
    # Monkey patch for debugging
    agentic_module.AgenticStepProcessor.run_async = debug_run_async
    
    try:
        logger.critical("🚀 STARTING MULTI-HOP REASONING TEST")
        
        result = await lightrag_multi_hop_reasoning(
            query="What are patient data tables?",
            objective="Find patient-related database tables",
            max_steps=2,
            timeout_seconds=60.0
        )
        
        total_elapsed = time.time() - start_time
        logger.critical(f"🏁 TOTAL EXECUTION TIME: {total_elapsed:.1f}s")
        
        print(f"\n📊 DEBUG RESULTS:")
        print(f"⏱️  Total time: {total_elapsed:.1f}s")
        print(f"🎯 Success: {result.get('success')}")
        
        if result.get("success"):
            print("✅ Multi-hop completed successfully")
            print("🔍 Check logs above for timing breakdown")
            return True
        else:
            print("❌ Multi-hop failed")
            print(f"💥 Error: {result.get('error')}")
            return False
            
    except Exception as e:
        total_elapsed = time.time() - start_time
        logger.critical(f"🔴 TEST FAILED: {total_elapsed:.1f}s, Error: {e}")
        print(f"❌ Test exception: {e}")
        return False
    finally:
        # Restore original method
        agentic_module.AgenticStepProcessor.run_async = original_run_async

if __name__ == "__main__":
    success = asyncio.run(test_with_debug_timing())
    sys.exit(0 if success else 1)