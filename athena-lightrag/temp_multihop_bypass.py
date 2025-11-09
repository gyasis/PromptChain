#!/usr/bin/env python3
"""
TEMPORARY BYPASS TEST: Multi-hop reasoning using direct OpenAI API calls.
This bypasses PromptChain entirely to isolate if PromptChain is the bottleneck.
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any
from pathlib import Path
import sys

# Add parent directory to path for LightRAG access
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append('/home/gyasis/Documents/code/PromptChain/athena-lightrag')

from lightrag_core import create_athena_lightrag
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BypassMultiHopReasoning:
    """Direct OpenAI API multi-hop reasoning - bypassing PromptChain."""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.lightrag_core = None
        
    async def init_lightrag(self):
        """Initialize LightRAG core."""
        working_dir = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
        self.lightrag_core = create_athena_lightrag(working_dir=working_dir)
        
    async def lightrag_query(self, query: str, mode: str = "global") -> str:
        """Direct LightRAG query without PromptChain."""
        try:
            if mode == "global":
                result = await self.lightrag_core.query_global_async(query)
            elif mode == "local":
                result = await self.lightrag_core.query_local_async(query)
            else:
                result = await self.lightrag_core.query_hybrid_async(query)
                
            if result.error:
                return f"Error: {result.error}"
            return result.result
        except Exception as e:
            return f"Query failed: {str(e)}"
    
    async def openai_reasoning_step(self, query: str, context: str, step: int) -> str:
        """Single reasoning step using direct OpenAI API."""
        try:
            messages = [
                {
                    "role": "system", 
                    "content": f"""You are step {step} in a multi-hop reasoning process about healthcare database analysis.
                    
Use the provided context to reason about the query. You can call tools if needed for more information.
Be concise but thorough. Focus on the specific information requested."""
                },
                {
                    "role": "user",
                    "content": f"""Query: {query}

Context from previous steps:
{context}

Please provide your reasoning for this step."""
                }
            ]
            
            # Direct OpenAI API call with tools
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lightrag_query",
                            "description": "Query the LightRAG knowledge base",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "mode": {"type": "string", "enum": ["local", "global", "hybrid"], "default": "global"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ],
                tool_choice="auto",
                timeout=30
            )
            
            message = response.choices[0].message
            
            # Handle tool calls
            if message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "lightrag_query":
                        import json
                        args = json.loads(tool_call.function.arguments)
                        result = await self.lightrag_query(args["query"], args.get("mode", "global"))
                        tool_results.append(f"Tool result: {result}")
                
                # Get final response with tool results
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
                })
                
                for i, tool_call in enumerate(message.tool_calls):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_results[i] if i < len(tool_results) else "No result"
                    })
                
                final_response = await self.openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    timeout=20
                )
                
                return final_response.choices[0].message.content
            
            return message.content or "No response generated"
            
        except Exception as e:
            return f"Reasoning step failed: {str(e)}"
    
    async def execute_multihop_reasoning(
        self, 
        query: str, 
        max_steps: int = 2,
        timeout_seconds: float = 60.0
    ) -> Dict[str, Any]:
        """Execute multi-hop reasoning using direct OpenAI calls."""
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting BYPASS multi-hop reasoning: {max_steps} steps, {timeout_seconds}s timeout")
            
            # Initialize
            await self.init_lightrag()
            
            # Execute reasoning steps
            context = f"Initial query: {query}"
            steps_completed = []
            
            for step in range(1, max_steps + 1):
                logger.info(f"⚡ Executing bypass step {step}/{max_steps}")
                step_start = time.time()
                
                step_result = await asyncio.wait_for(
                    self.openai_reasoning_step(query, context, step),
                    timeout=timeout_seconds - (time.time() - start_time)
                )
                
                step_time = time.time() - step_start
                logger.info(f"✅ Bypass step {step} completed in {step_time:.1f}s")
                
                steps_completed.append({
                    "step": step,
                    "result": step_result,
                    "execution_time": step_time
                })
                
                # Add to context for next step
                context += f"\n\nStep {step} result: {step_result}"
                
                # Check if we're running out of time
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds * 0.8:  # 80% of timeout used
                    logger.warning(f"⚠️ Approaching timeout at {elapsed:.1f}s, stopping early")
                    break
            
            # Generate final synthesis
            final_result = steps_completed[-1]["result"] if steps_completed else "No steps completed"
            
            total_time = time.time() - start_time
            logger.info(f"🏁 Bypass multi-hop completed in {total_time:.1f}s")
            
            return {
                "success": True,
                "result": final_result,
                "steps_completed": len(steps_completed),
                "reasoning_steps": steps_completed,
                "execution_time": total_time,
                "method": "direct_openai_bypass"
            }
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"❌ Bypass multi-hop TIMEOUT after {elapsed:.1f}s")
            return {
                "success": False,
                "result": "",
                "error": f"TIMEOUT after {elapsed:.1f}s",
                "steps_completed": len(steps_completed) if 'steps_completed' in locals() else 0,
                "execution_time": elapsed,
                "method": "direct_openai_bypass"
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Bypass multi-hop FAILED: {str(e)}")
            return {
                "success": False,
                "result": "",
                "error": str(e),
                "execution_time": elapsed,
                "method": "direct_openai_bypass"
            }

# Test function
async def test_bypass():
    """Test the bypass multi-hop reasoning."""
    print("🧪 TESTING BYPASS MULTI-HOP REASONING")
    print("Using direct OpenAI API calls - NO PromptChain")
    print("="*60)
    
    bypass = BypassMultiHopReasoning()
    
    # Test with the same query that fails in PromptChain
    result = await bypass.execute_multihop_reasoning(
        query="What are patient data tables?",
        max_steps=2,
        timeout_seconds=60.0
    )
    
    print(f"\n📊 BYPASS TEST RESULTS:")
    print(f"⏱️  Execution time: {result['execution_time']:.1f}s")
    print(f"🎯 Success: {result['success']}")
    print(f"🔢 Steps completed: {result.get('steps_completed', 0)}")
    
    if result["success"]:
        print("✅ BYPASS MULTIHOP: PASSED")
        print("🔍 CONCLUSION: PromptChain is likely the bottleneck!")
        print(f"📝 Result preview: {str(result['result'])[:200]}...")
        return True
    else:
        print("❌ BYPASS MULTIHOP: FAILED")  
        print("🤔 CONCLUSION: Issue may be deeper than PromptChain")
        print(f"💥 Error: {result.get('error')}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_bypass())
    sys.exit(0 if success else 1)