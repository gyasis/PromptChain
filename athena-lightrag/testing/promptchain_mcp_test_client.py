#!/usr/bin/env python3
"""
PromptChain MCP Test Client with AgenticStepProcessor
=====================================================
Systematic testing client for Athena LightRAG MCP server using PromptChain
with AgenticStepProcessor to iteratively identify and fix connection issues.

This client:
1. Tests MCP server connection using PromptChain framework
2. Uses AgenticStepProcessor for intelligent error analysis 
3. Provides detailed diagnostics for server issues
4. Supports iterative debugging workflow
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add parent directories to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
promptchain_dir = os.path.dirname(parent_dir)
sys.path.append(promptchain_dir)
sys.path.append(parent_dir)

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('testing/mcp_test_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class AthenaLightRAGMCPTester:
    """
    Comprehensive MCP server testing using PromptChain with AgenticStepProcessor
    """
    
    def __init__(self):
        self.test_results = []
        self.server_config = self._get_mcp_server_config()
        self.chain = None
        
    def _get_mcp_server_config(self) -> List[Dict[str, Any]]:
        """
        Get MCP server configuration using UV isolated environment pattern
        Following FastMCP best practices for UV environment isolation
        """
        athena_lightrag_path = Path(__file__).parent.parent
        
        return [{
            "id": "athena_lightrag_server",
            "type": "stdio",
            # Use FastMCP development server with proper isolation
            "command": "uv",
            "args": [
                "run",
                "--project", str(athena_lightrag_path),  # Specify project directory for uv
                "fastmcp", "run"  # Use FastMCP to run the server properly
            ],
            "env": {
                "MCP_MODE": "stdio",
                "DEBUG": "true"
            },
            "read_timeout_seconds": 60  # Increased timeout for UV environment setup
        }]
    
    async def test_basic_connection(self) -> Dict[str, Any]:
        """
        Test 1: Basic MCP server connection and tool discovery
        """
        logger.info("🔍 TEST 1: Basic MCP Server Connection")
        test_result = {
            "test_name": "basic_connection",
            "success": False,
            "error": None,
            "tools_discovered": [],
            "connection_time": 0
        }
        
        try:
            start_time = time.time()
            
            # Create basic PromptChain with MCP server
            self.chain = PromptChain(
                models=["openai/gpt-4o-mini"],
                instructions=["Test connection: {input}"],
                mcp_servers=self.server_config,
                verbose=True
            )
            
            logger.info("📡 Attempting MCP connection...")
            await self.chain.mcp_helper.connect_mcp_async()
            
            connection_time = time.time() - start_time
            test_result["connection_time"] = connection_time
            
            # Check discovered tools
            tools = [t['function']['name'] for t in self.chain.tools if t['function']['name'].startswith('mcp_')]
            test_result["tools_discovered"] = tools
            
            if tools:
                test_result["success"] = True
                logger.info(f"✅ Connection successful! Discovered {len(tools)} MCP tools in {connection_time:.2f}s")
                for tool in tools:
                    logger.info(f"  • {tool}")
            else:
                test_result["error"] = "No MCP tools discovered"
                logger.error("❌ Connection established but no tools discovered")
                
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Connection failed: {e}")
            
        return test_result
    
    async def test_tool_execution(self) -> Dict[str, Any]:
        """
        Test 2: Execute a simple MCP tool call
        """
        logger.info("🔧 TEST 2: Tool Execution")
        test_result = {
            "test_name": "tool_execution", 
            "success": False,
            "error": None,
            "tool_result": None,
            "execution_time": 0
        }
        
        try:
            if not self.chain:
                raise Exception("No active chain connection")
                
            start_time = time.time()
            
            # Try to execute a basic query using PromptChain
            result = await self.chain.process_prompt_async(
                "Test the lightrag_local_query tool with a simple healthcare query: 'patient appointments'"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["tool_result"] = result[:500] if result else "No result"
            test_result["success"] = bool(result)
            
            if result:
                logger.info(f"✅ Tool execution successful in {execution_time:.2f}s")
                logger.info(f"Result preview: {result[:200]}...")
            else:
                test_result["error"] = "Tool execution returned empty result"
                
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Tool execution failed: {e}")
            
        return test_result
    
    async def test_agentic_step_analysis(self) -> Dict[str, Any]:
        """
        Test 3: Use AgenticStepProcessor for intelligent MCP analysis
        """
        logger.info("🤖 TEST 3: AgenticStepProcessor MCP Analysis")
        test_result = {
            "test_name": "agentic_analysis",
            "success": False,
            "error": None,
            "analysis_result": None,
            "reasoning_steps": 0
        }
        
        try:
            # Create agentic step processor for MCP analysis
            agentic_step = AgenticStepProcessor(
                objective="Analyze the Athena LightRAG MCP server capabilities and test a healthcare database query",
                max_internal_steps=3,
                model_name="openai/gpt-4o-mini"
            )
            
            # Create chain with agentic reasoning
            chain_with_agentic = PromptChain(
                models=["openai/gpt-4o-mini"],
                instructions=[
                    "Initialize MCP analysis: {input}",
                    agentic_step,  # Intelligent multi-step reasoning
                    "Summarize findings: {input}"
                ],
                mcp_servers=self.server_config,
                verbose=True
            )
            
            # Connect MCP
            await chain_with_agentic.mcp_helper.connect_mcp_async()
            
            # Run agentic analysis
            result = await chain_with_agentic.process_prompt_async(
                "Test and analyze the healthcare MCP tools with a query about patient appointment relationships"
            )
            
            test_result["analysis_result"] = result
            test_result["reasoning_steps"] = len(agentic_step.step_outputs) if hasattr(agentic_step, 'step_outputs') else 0
            test_result["success"] = bool(result)
            
            logger.info(f"✅ Agentic analysis completed with {test_result['reasoning_steps']} reasoning steps")
            logger.info(f"Analysis result: {result[:300]}...")
            
            # Cleanup
            await chain_with_agentic.mcp_helper.close_mcp_async()
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Agentic analysis failed: {e}")
            
        return test_result
    
    async def diagnose_server_issues(self) -> Dict[str, Any]:
        """
        Test 4: Use Gemini to diagnose server startup issues
        """
        logger.info("🩺 TEST 4: Server Issue Diagnosis")
        
        # Check if server script exists and is executable
        server_path = Path(__file__).parent.parent / "athena_mcp_server.py"
        python_path = Path(__file__).parent.parent / ".venv" / "bin" / "python3"
        
        diagnostics = {
            "server_script_exists": server_path.exists(),
            "server_script_readable": server_path.is_file() and os.access(server_path, os.R_OK),
            "python_executable_exists": python_path.exists(),
            "python_executable": python_path.is_file() and os.access(python_path, os.X_OK),
            "working_directory": str(Path(__file__).parent.parent),
            "environment_variables": dict(os.environ)
        }
        
        logger.info("📋 Server Diagnostics:")
        for key, value in diagnostics.items():
            if key != "environment_variables":  # Don't log all env vars
                logger.info(f"  {key}: {value}")
        
        return {
            "test_name": "server_diagnostics",
            "diagnostics": diagnostics,
            "success": all([
                diagnostics["server_script_exists"],
                diagnostics["server_script_readable"], 
                diagnostics["python_executable_exists"],
                diagnostics["python_executable"]
            ])
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run all tests in sequence and compile results
        """
        logger.info("🚀 Starting Comprehensive Athena LightRAG MCP Testing")
        logger.info("=" * 60)
        
        all_results = {
            "timestamp": time.time(),
            "test_session": "athena_lightrag_mcp_debug",
            "tests": []
        }
        
        try:
            # Test 1: Basic Connection
            result1 = await self.test_basic_connection()
            all_results["tests"].append(result1)
            
            # Test 2: Tool Execution (only if connection worked)
            if result1["success"]:
                result2 = await self.test_tool_execution()
                all_results["tests"].append(result2)
                
                # Test 3: Agentic Analysis (only if tools work)
                if result2["success"]:
                    result3 = await self.test_agentic_step_analysis()
                    all_results["tests"].append(result3)
            
            # Test 4: Always run diagnostics
            result4 = await self.diagnose_server_issues()
            all_results["tests"].append(result4)
            
            # Calculate overall success
            successful_tests = sum(1 for test in all_results["tests"] if test.get("success", False))
            all_results["overall_success"] = successful_tests == len(all_results["tests"])
            all_results["success_rate"] = successful_tests / len(all_results["tests"])
            
            logger.info("=" * 60)
            logger.info(f"🎯 TEST SUMMARY: {successful_tests}/{len(all_results['tests'])} tests passed ({all_results['success_rate']:.1%})")
            
            # Save detailed results
            with open("testing/mcp_test_results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Test session failed: {e}")
            all_results["session_error"] = str(e)
            return all_results
            
        finally:
            # Cleanup
            if self.chain:
                try:
                    await self.chain.mcp_helper.close_mcp_async()
                    logger.info("🧹 MCP connections closed")
                except Exception as e:
                    logger.warning(f"Cleanup warning: {e}")

async def main():
    """
    Main test execution with Gemini integration for error analysis
    """
    logger.info("🏥 Athena LightRAG MCP Server Testing with PromptChain")
    
    tester = AthenaLightRAGMCPTester()
    results = await tester.run_comprehensive_test()
    
    # Determine next steps based on results
    if results["overall_success"]:
        print("\n🎉 ALL TESTS PASSED! MCP server is working correctly.")
    else:
        print("\n⚠️  ISSUES DETECTED. Preparing for iterative fixes...")
        
        # Print specific issues for Gemini analysis
        print("\n📋 Issues to analyze with Gemini:")
        for test in results["tests"]:
            if not test.get("success", False) and test.get("error"):
                print(f"  • {test['test_name']}: {test['error']}")
        
        print(f"\n💾 Detailed results saved to: testing/mcp_test_results.json")
        print("📋 Debug logs saved to: testing/mcp_test_debug.log")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())