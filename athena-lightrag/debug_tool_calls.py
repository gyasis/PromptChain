#!/usr/bin/env python3
"""
Debug Tool Calls Issue
======================
Debug the function name validation error in AgenticStepProcessor tool calls.
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import re
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agentic_lightrag import AgenticLightRAG

def sanitize_function_name(name: str) -> str:
    """Sanitize function name to match OpenAI pattern ^[a-zA-Z0-9_-]+"""
    if not isinstance(name, str):
        name = str(name)
    
    # Remove any invalid characters
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unknown_function"
    
    return sanitized

def debug_function_name(name: str) -> Dict[str, Any]:
    """Debug a function name to identify invalid characters."""
    if not isinstance(name, str):
        return {
            "type": type(name).__name__,
            "string_repr": str(name),
            "is_valid": False,
            "issues": ["Not a string"]
        }
    
    # Check each character
    invalid_chars = []
    char_codes = []
    
    for i, char in enumerate(name):
        char_code = ord(char)
        char_codes.append(char_code)
        
        # Check if character matches pattern [a-zA-Z0-9_-]
        if not re.match(r'[a-zA-Z0-9_-]', char):
            invalid_chars.append({
                "position": i,
                "character": char,
                "code": char_code,
                "description": repr(char)
            })
    
    # Check pattern match
    pattern_match = bool(re.match(r'^[a-zA-Z0-9_-]+$', name))
    
    return {
        "original": name,
        "length": len(name),
        "encoding": name.encode('utf-8').hex() if isinstance(name, str) else "N/A",
        "char_codes": char_codes,
        "invalid_characters": invalid_chars,
        "matches_pattern": pattern_match,
        "sanitized": sanitize_function_name(name),
        "is_valid": pattern_match and len(invalid_chars) == 0
    }

def debug_messages_array(messages: List[Dict]) -> List[Dict[str, Any]]:
    """Debug a messages array to find tool call issues."""
    debug_results = []
    
    for i, message in enumerate(messages):
        message_debug = {
            "index": i,
            "role": message.get("role", "unknown"),
            "has_tool_calls": "tool_calls" in message,
            "tool_calls_debug": []
        }
        
        if "tool_calls" in message and message["tool_calls"]:
            for j, tool_call in enumerate(message["tool_calls"]):
                tool_debug = {
                    "tool_call_index": j,
                    "tool_call_structure": {
                        "has_function": "function" in tool_call,
                        "has_name": "function" in tool_call and "name" in tool_call["function"]
                    }
                }
                
                if "function" in tool_call and "name" in tool_call["function"]:
                    function_name = tool_call["function"]["name"]
                    tool_debug["function_name_debug"] = debug_function_name(function_name)
                
                message_debug["tool_calls_debug"].append(tool_debug)
        
        debug_results.append(message_debug)
    
    return debug_results

def patch_promptchain_for_debugging():
    """Monkey patch PromptChain to add debugging for tool calls."""
    from promptchain.utils.promptchaining import PromptChain
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor
    
    # Save original method
    original_run_model_async = PromptChain.run_model_async
    
    @staticmethod
    async def debug_run_model_async(messages, model_name, tools=None, **kwargs):
        """Debug wrapper for run_model_async to catch tool call issues."""
        
        print(f"\n🔍 DEBUG: About to call {model_name} with {len(messages)} messages and {len(tools) if tools else 0} tools")
        
        # Debug the messages array for tool call issues
        if any("tool_calls" in msg for msg in messages if isinstance(msg, dict)):
            print("🔧 DEBUG: Found tool_calls in message history - analyzing...")
            
            debug_results = debug_messages_array(messages)
            
            for result in debug_results:
                if result["has_tool_calls"]:
                    print(f"  📤 Message {result['index']} ({result['role']}) has tool_calls:")
                    
                    for tool_debug in result["tool_calls_debug"]:
                        if "function_name_debug" in tool_debug:
                            name_debug = tool_debug["function_name_debug"]
                            print(f"    🔧 Tool call {tool_debug['tool_call_index']}:")
                            print(f"       Name: '{name_debug['original']}'")
                            print(f"       Valid: {name_debug['is_valid']}")
                            
                            if not name_debug["is_valid"]:
                                print(f"       ❌ Invalid characters: {name_debug['invalid_characters']}")
                                print(f"       🔧 Sanitized: '{name_debug['sanitized']}'")
                                
                                # Fix the function name in place
                                for msg in messages:
                                    if "tool_calls" in msg:
                                        for tool_call in msg["tool_calls"]:
                                            if ("function" in tool_call and 
                                                "name" in tool_call["function"] and
                                                tool_call["function"]["name"] == name_debug["original"]):
                                                print(f"       🚨 FIXING function name: '{name_debug['original']}' -> '{name_debug['sanitized']}'")
                                                tool_call["function"]["name"] = name_debug["sanitized"]
        
        # Call original method with potentially fixed messages
        return await original_run_model_async(messages, model_name, tools=tools, **kwargs)
    
    # Apply the patch
    PromptChain.run_model_async = debug_run_model_async
    print("✅ Applied debugging patch to PromptChain.run_model_async")

async def test_with_debugging():
    """Test multi-hop reasoning with debugging enabled."""
    print("🐛 Debug Tool Calls Test")
    print("=" * 50)
    
    # Apply debugging patch
    patch_promptchain_for_debugging()
    
    # Create AgenticLightRAG and test
    agentic_rag = AgenticLightRAG(verbose=True)
    
    test_query = "What tables are related to patient appointments?"
    test_objective = """
    DISCOVERY: Find tables related to patient appointments.
    
    Use local mode to explore appointment-related tables and their relationships.
    """
    
    print(f"\n📝 Testing query: {test_query}")
    print(f"🎯 Objective: {test_objective.strip()}")
    
    try:
        result = await agentic_rag.execute_multi_hop_reasoning(
            query=test_query,
            objective=test_objective,
            reset_context=True,
            timeout_seconds=60.0
        )
        
        print(f"\n✅ Test completed!")
        print(f"Success: {result.get('success', False)}")
        print(f"Reasoning steps: {len(result.get('reasoning_steps', []))}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function."""
    print("🔧 Tool Call Debug Utility")
    print("=" * 50)
    print("This tool will debug function name validation issues in AgenticStepProcessor")
    
    try:
        result = asyncio.run(test_with_debugging())
        
        if result:
            print(f"\n🎉 Debug test successful!")
            print("The tool call issues have been identified and potentially fixed.")
        else:
            print(f"\n❌ Debug test failed.")
            print("Check the debug output above for specific issues.")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Debug interrupted by user")
    except Exception as e:
        print(f"\n💥 Debug runner failed: {e}")

if __name__ == "__main__":
    main()