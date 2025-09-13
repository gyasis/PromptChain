#!/usr/bin/env python3
"""
REAL MCP Tool Testing with PromptChain and AgenticStepProcessor
================================================================
This actually tests the MCP tools by calling them through PromptChain,
not just checking if they exist.
"""

import asyncio
import sys
import os
from pathlib import Path
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load project .env file with override to force project keys
load_dotenv(override=True)

# Add parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
promptchain_dir = os.path.dirname(parent_dir)
sys.path.append(promptchain_dir)
sys.path.append(parent_dir)

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

def get_mcp_server_config():
    """Get MCP server configuration."""
    athena_lightrag_path = Path(__file__).parent.parent
    
    return [{
        "id": "athena_lightrag_server",
        "type": "stdio",
        "command": "uv",
        "args": [
            "run",
            "--directory", str(athena_lightrag_path),
            "fastmcp", "run"
        ],
        "env": {
            "MCP_MODE": "stdio",
            "DEBUG": "true"
        },
        "read_timeout_seconds": 120
    }]

async def test_individual_tools():
    """Test each MCP tool individually through PromptChain."""
    
    print("🚀 REAL MCP Tool Testing with PromptChain")
    print("=" * 60)
    
    # Initialize PromptChain with MCP server
    chain = PromptChain(
        models=["openai/gpt-4.1-mini"],
        instructions=["Execute the specific MCP tool as requested: {input}"],
        mcp_servers=get_mcp_server_config(),
        verbose=True
    )
    
    # Connect to MCP server
    print("📡 Connecting to MCP server...")
    await chain.mcp_helper.connect_mcp_async()
    
    # Get list of available tools
    mcp_tools = [t for t in chain.tools if t['function']['name'].startswith('mcp_')]
    print(f"✅ Found {len(mcp_tools)} MCP tools")
    for tool in mcp_tools:
        print(f"  • {tool['function']['name']}")
    
    print("\n" + "-" * 60)
    
    # Test results storage
    test_results = {}
    
    # TEST 1: lightrag_local_query
    print("\n🔍 TEST 1: lightrag_local_query")
    try:
        result = await chain.process_prompt_async(
            "Call the mcp_athena_lightrag_server_lightrag_local_query tool with query='APPOINTMENT table structure' and return the raw result"
        )
        test_results['lightrag_local_query'] = {
            'success': bool(result and 'APPOINTMENT' in result),
            'preview': result[:200] if result else 'No result'
        }
        print(f"Result: {result[:200]}...")
    except Exception as e:
        test_results['lightrag_local_query'] = {'success': False, 'error': str(e)}
        print(f"Error: {e}")
    
    # TEST 2: lightrag_global_query  
    print("\n🌍 TEST 2: lightrag_global_query")
    try:
        result = await chain.process_prompt_async(
            "Call the mcp_athena_lightrag_server_lightrag_global_query tool with query='all appointment workflows' and return the raw result"
        )
        test_results['lightrag_global_query'] = {
            'success': bool(result and 'workflow' in result.lower()),
            'preview': result[:200] if result else 'No result'
        }
        print(f"Result: {result[:200]}...")
    except Exception as e:
        test_results['lightrag_global_query'] = {'success': False, 'error': str(e)}
        print(f"Error: {e}")
    
    # TEST 3: lightrag_hybrid_query
    print("\n🔀 TEST 3: lightrag_hybrid_query")
    try:
        result = await chain.process_prompt_async(
            "Call the mcp_athena_lightrag_server_lightrag_hybrid_query tool with query='patient billing relationships' and return the raw result"
        )
        test_results['lightrag_hybrid_query'] = {
            'success': bool(result and any(word in result.lower() for word in ['patient', 'billing'])),
            'preview': result[:200] if result else 'No result'
        }
        print(f"Result: {result[:200]}...")
    except Exception as e:
        test_results['lightrag_hybrid_query'] = {'success': False, 'error': str(e)}
        print(f"Error: {e}")
    
    # TEST 4: lightrag_context_extract
    print("\n📚 TEST 4: lightrag_context_extract")
    try:
        result = await chain.process_prompt_async(
            "Call the mcp_athena_lightrag_server_lightrag_context_extract tool with query='PATIENT table metadata' and mode='local' and return the raw result"
        )
        test_results['lightrag_context_extract'] = {
            'success': bool(result and 'patient' in result.lower()),
            'preview': result[:200] if result else 'No result'
        }
        print(f"Result: {result[:200]}...")
    except Exception as e:
        test_results['lightrag_context_extract'] = {'success': False, 'error': str(e)}
        print(f"Error: {e}")
    
    # TEST 5: lightrag_sql_generation
    print("\n🗄️ TEST 5: lightrag_sql_generation")
    try:
        result = await chain.process_prompt_async(
            "Call the mcp_athena_lightrag_server_lightrag_sql_generation tool with natural_query='find all appointments today' and return the raw SQL query"
        )
        test_results['lightrag_sql_generation'] = {
            'success': bool(result and 'SELECT' in result.upper()),
            'preview': result[:200] if result else 'No result'
        }
        print(f"Result: {result[:200]}...")
    except Exception as e:
        test_results['lightrag_sql_generation'] = {'success': False, 'error': str(e)}
        print(f"Error: {e}")
    
    # TEST 6: get_server_info
    print("\nℹ️ TEST 6: get_server_info")
    try:
        result = await chain.process_prompt_async(
            "Call the mcp_athena_lightrag_server_get_server_info tool and return the server information"
        )
        test_results['get_server_info'] = {
            'success': bool(result and 'server' in result.lower()),
            'preview': result[:200] if result else 'No result'
        }
        print(f"Result: {result[:200]}...")
    except Exception as e:
        test_results['get_server_info'] = {'success': False, 'error': str(e)}
        print(f"Error: {e}")
    
    # Cleanup
    await chain.mcp_helper.close_mcp_async()
    
    return test_results

async def test_agentic_step_processor():
    """Test multi-hop reasoning with AgenticStepProcessor."""
    
    print("\n" + "=" * 60)
    print("🧠 Testing AgenticStepProcessor with MCP Tools")
    print("-" * 60)
    
    # Create AgenticStepProcessor with MCP tools
    agentic_step = AgenticStepProcessor(
        objective="Analyze patient appointment workflow using MCP tools",
        model_name="openai/gpt-4.1-mini",
        max_internal_steps=3
    )
    
    # Create chain with agentic step
    chain = PromptChain(
        models=["openai/gpt-4.1-mini"],
        instructions=[
            "Analyze the request: {input}",
            agentic_step,
            "Summarize findings: {input}"
        ],
        mcp_servers=get_mcp_server_config(),
        verbose=True
    )
    
    # Connect MCP
    await chain.mcp_helper.connect_mcp_async()
    
    # Test multi-hop reasoning
    try:
        result = await chain.process_prompt_async(
            "Use the lightrag_multi_hop_reasoning tool to trace how patient appointments connect to billing"
        )
        
        print(f"Multi-hop result: {result[:300]}...")
        
        # Cleanup
        await chain.mcp_helper.close_mcp_async()
        
        return {
            'success': bool(result),
            'preview': result[:300] if result else 'No result'
        }
    except Exception as e:
        print(f"Multi-hop error: {e}")
        await chain.mcp_helper.close_mcp_async()
        return {'success': False, 'error': str(e)}

async def main():
    """Run all tests."""
    
    # Test individual tools
    tool_results = await test_individual_tools()
    
    # Test AgenticStepProcessor
    agentic_results = await test_agentic_step_processor()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 60)
    
    # Count successes
    tool_successes = sum(1 for r in tool_results.values() if r.get('success'))
    print(f"Individual Tools: {tool_successes}/{len(tool_results)} passed")
    
    for tool_name, result in tool_results.items():
        status = "✅" if result.get('success') else "❌"
        print(f"  {status} {tool_name}")
        if not result.get('success') and 'error' in result:
            print(f"      Error: {result['error'][:100]}")
    
    print(f"\nAgenticStepProcessor: {'✅' if agentic_results.get('success') else '❌'}")
    
    # Save results
    all_results = {
        'tool_tests': tool_results,
        'agentic_test': agentic_results,
        'summary': {
            'tools_passed': tool_successes,
            'tools_total': len(tool_results),
            'agentic_passed': agentic_results.get('success', False)
        }
    }
    
    with open('testing/real_mcp_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: testing/real_mcp_test_results.json")
    
    # Overall success
    overall_success = tool_successes == len(tool_results) and agentic_results.get('success')
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! MCP tools are working correctly.")
    else:
        print(f"\n⚠️ SOME TESTS FAILED: {tool_successes}/{len(tool_results)} tools, Agentic: {agentic_results.get('success')}")
    
    return overall_success

if __name__ == "__main__":
    # Force use of project .env file
    load_dotenv(override=True)
    
    # Run tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)