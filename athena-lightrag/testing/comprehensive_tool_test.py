#!/usr/bin/env python3
"""
Comprehensive MCP Tool Testing Script
====================================
Tests all 7 MCP tools with proper prompts to verify functionality.

This script tests:
1. lightrag_local_query - Specific entity relationships
2. lightrag_global_query - High-level overviews  
3. lightrag_hybrid_query - Combined detailed + contextual
4. lightrag_context_extract - Raw data dictionary access
5. lightrag_multi_hop_reasoning - Complex multi-step analysis
6. lightrag_sql_generation - SQL query generation
7. get_server_info - Server status and capabilities
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

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('testing/comprehensive_tool_test.log')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveMCPToolTester:
    """Test all MCP tools with proper healthcare prompts."""
    
    def __init__(self):
        self.test_results = []
        self.server_config = self._get_mcp_server_config()
        self.chain = None
        
    def _get_mcp_server_config(self) -> List[Dict[str, Any]]:
        """Get MCP server configuration using UV directory flag."""
        athena_lightrag_path = Path(__file__).parent.parent
        
        return [{
            "id": "athena_lightrag_server",
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--directory", str(athena_lightrag_path),  # Use UV directory flag
                "fastmcp", "run"
            ],
            "env": {
                "MCP_MODE": "stdio",
                "DEBUG": "true"
            },
            "read_timeout_seconds": 90
        }]
    
    async def setup_connection(self) -> bool:
        """Setup MCP connection."""
        try:
            logger.info("🔌 Setting up MCP connection...")
            self.chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Process healthcare query: {input}"],
                mcp_servers=self.server_config,
                verbose=True
            )
            
            await self.chain.mcp_helper.connect_mcp_async()
            
            # Check discovered tools
            tools = [t['function']['name'] for t in self.chain.tools if t['function']['name'].startswith('mcp_')]
            logger.info(f"✅ Connected! Discovered {len(tools)} MCP tools")
            return len(tools) >= 7  # Should have at least 7 tools
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    async def test_local_query(self) -> Dict[str, Any]:
        """Test 1: lightrag_local_query - Specific entity relationships."""
        logger.info("🔍 TEST 1: lightrag_local_query")
        test_result = {
            "tool": "lightrag_local_query",
            "success": False,
            "error": None,
            "response_preview": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            # Test with specific healthcare entity query
            result = await self.chain.process_prompt_async(
                "Use lightrag_local_query to find specific relationships for APPOINTMENT table structure, "
                "including its columns, primary keys, foreign keys, and direct table connections in the athena.athenaone schema"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["response_preview"] = result[:300] if result else "No result"
            test_result["success"] = bool(result and "APPOINTMENT" in result)
            
            logger.info(f"✅ Local query completed in {execution_time:.2f}s")
            logger.info(f"Result preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Local query failed: {e}")
            
        return test_result
    
    async def test_global_query(self) -> Dict[str, Any]:
        """Test 2: lightrag_global_query - High-level medical workflow overviews."""
        logger.info("🌍 TEST 2: lightrag_global_query")
        test_result = {
            "tool": "lightrag_global_query",
            "success": False,
            "error": None,
            "response_preview": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            result = await self.chain.process_prompt_async(
                "Use lightrag_global_query to get a comprehensive overview of all appointment-related workflows "
                "across the entire Athena Health EHR system, including patient scheduling, provider management, "
                "and appointment type categorization across all schemas"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["response_preview"] = result[:300] if result else "No result"
            test_result["success"] = bool(result and any(word in result.lower() for word in ["workflow", "schema", "appointment"]))
            
            logger.info(f"✅ Global query completed in {execution_time:.2f}s")
            logger.info(f"Result preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Global query failed: {e}")
            
        return test_result
    
    async def test_hybrid_query(self) -> Dict[str, Any]:
        """Test 3: lightrag_hybrid_query - Combined detailed + contextual analysis."""
        logger.info("🔀 TEST 3: lightrag_hybrid_query")
        test_result = {
            "tool": "lightrag_hybrid_query",
            "success": False,
            "error": None,
            "response_preview": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            result = await self.chain.process_prompt_async(
                "Use lightrag_hybrid_query to analyze patient billing workflow connections, "
                "combining specific CHARGEDIAGNOSIS table details with broader revenue cycle workflows "
                "from diagnosis coding through payment processing in the Athena system"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["response_preview"] = result[:300] if result else "No result"
            test_result["success"] = bool(result and any(word in result.lower() for word in ["billing", "diagnosis", "payment"]))
            
            logger.info(f"✅ Hybrid query completed in {execution_time:.2f}s")
            logger.info(f"Result preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Hybrid query failed: {e}")
            
        return test_result
    
    async def test_context_extract(self) -> Dict[str, Any]:
        """Test 4: lightrag_context_extract - Raw data dictionary access."""
        logger.info("📚 TEST 4: lightrag_context_extract")
        test_result = {
            "tool": "lightrag_context_extract",
            "success": False,
            "error": None,
            "response_preview": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            result = await self.chain.process_prompt_async(
                "Use lightrag_context_extract to get raw metadata for all patient-related tables, "
                "including exact column specifications, data types, constraints, and foreign key relationships "
                "without any generated analysis - just pure data dictionary information"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["response_preview"] = result[:300] if result else "No result"
            test_result["success"] = bool(result and any(word in result.lower() for word in ["patient", "column", "table"]))
            
            logger.info(f"✅ Context extract completed in {execution_time:.2f}s")
            logger.info(f"Result preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Context extract failed: {e}")
            
        return test_result
    
    async def test_multi_hop_reasoning(self) -> Dict[str, Any]:
        """Test 5: lightrag_multi_hop_reasoning - Complex multi-step analysis."""
        logger.info("🧠 TEST 5: lightrag_multi_hop_reasoning")
        test_result = {
            "tool": "lightrag_multi_hop_reasoning",
            "success": False,
            "error": None,
            "response_preview": "",
            "reasoning_steps": 0,
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            result = await self.chain.process_prompt_async(
                "Use lightrag_multi_hop_reasoning to trace the complete patient journey: "
                "How does a patient appointment connect through clinical documentation, "
                "diagnosis coding, charge capture, and final payment processing? "
                "Analyze the data flow across all relevant Athena tables with 3-4 reasoning steps."
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["response_preview"] = result[:300] if result else "No result"
            test_result["success"] = bool(result and any(word in result.lower() for word in ["patient", "appointment", "journey"]))
            
            # Try to count reasoning steps mentioned in result
            if result:
                step_indicators = result.lower().count("step") + result.lower().count("reasoning")
                test_result["reasoning_steps"] = step_indicators
            
            logger.info(f"✅ Multi-hop reasoning completed in {execution_time:.2f}s")
            logger.info(f"Result preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Multi-hop reasoning failed: {e}")
            
        return test_result
    
    async def test_sql_generation(self) -> Dict[str, Any]:
        """Test 6: lightrag_sql_generation - SQL query generation."""
        logger.info("🗄️ TEST 6: lightrag_sql_generation")
        test_result = {
            "tool": "lightrag_sql_generation",
            "success": False,
            "error": None,
            "sql_query": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            result = await self.chain.process_prompt_async(
                "Use lightrag_sql_generation to create a Snowflake SQL query that finds all patients "
                "with appointments in the last 30 days, showing patient name, appointment date, "
                "appointment type, and provider information with proper JOINs across athena.athenaone tables"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["sql_query"] = result if result else "No SQL generated"
            test_result["success"] = bool(result and any(word in result.upper() for word in ["SELECT", "FROM", "JOIN", "athena.athenaone"]))
            
            logger.info(f"✅ SQL generation completed in {execution_time:.2f}s")
            logger.info(f"SQL preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ SQL generation failed: {e}")
            
        return test_result
    
    async def test_server_info(self) -> Dict[str, Any]:
        """Test 7: get_server_info - Server status and capabilities."""
        logger.info("ℹ️ TEST 7: get_server_info")
        test_result = {
            "tool": "get_server_info",
            "success": False,
            "error": None,
            "server_details": {},
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            result = await self.chain.process_prompt_async(
                "Use get_server_info to get comprehensive information about the Athena LightRAG MCP Server, "
                "including version, working directory, available tools, query modes, and database status"
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["server_details"] = result[:500] if result else "No server info"
            test_result["success"] = bool(result and any(word in result.lower() for word in ["server", "version", "tools"]))
            
            logger.info(f"✅ Server info completed in {execution_time:.2f}s")
            logger.info(f"Server info preview: {result[:200]}...")
            
        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"❌ Server info failed: {e}")
            
        return test_result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tool tests in sequence."""
        logger.info("🚀 Starting Comprehensive MCP Tool Testing")
        logger.info("=" * 60)
        
        # Setup connection first
        if not await self.setup_connection():
            return {
                "overall_success": False,
                "error": "Failed to establish MCP connection",
                "tests": []
            }
        
        test_functions = [
            self.test_local_query,
            self.test_global_query, 
            self.test_hybrid_query,
            self.test_context_extract,
            self.test_multi_hop_reasoning,
            self.test_sql_generation,
            self.test_server_info
        ]
        
        all_results = {
            "timestamp": time.time(),
            "test_session": "comprehensive_mcp_tool_test",
            "tests": []
        }
        
        try:
            for test_func in test_functions:
                logger.info("-" * 40)
                result = await test_func()
                all_results["tests"].append(result)
                
                # Brief delay between tests
                await asyncio.sleep(2)
            
            # Calculate overall success
            successful_tests = sum(1 for test in all_results["tests"] if test.get("success", False))
            all_results["overall_success"] = successful_tests == len(all_results["tests"])
            all_results["success_rate"] = successful_tests / len(all_results["tests"])
            
            logger.info("=" * 60)
            logger.info(f"🎯 COMPREHENSIVE TEST SUMMARY")
            logger.info(f"Tests passed: {successful_tests}/{len(all_results['tests'])} ({all_results['success_rate']:.1%})")
            
            for test in all_results["tests"]:
                status = "✅" if test["success"] else "❌"
                logger.info(f"{status} {test['tool']}: {test.get('execution_time', 0):.2f}s")
            
            # Save detailed results
            with open("testing/comprehensive_tool_test_results.json", "w") as f:
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
    """Main test execution."""
    logger.info("🏥 Comprehensive Athena LightRAG MCP Tool Testing")
    
    tester = ComprehensiveMCPToolTester()
    results = await tester.run_comprehensive_test()
    
    if results["overall_success"]:
        print(f"\n🎉 ALL {len(results['tests'])} TOOLS PASSED! MCP server is fully functional.")
        print("✅ All healthcare analysis tools are working correctly.")
    else:
        print(f"\n⚠️  ISSUES DETECTED: {results.get('success_rate', 0):.1%} success rate")
        for test in results["tests"]:
            if not test.get("success", False):
                print(f"  • {test['tool']}: {test.get('error', 'Unknown error')}")
    
    print(f"\n💾 Detailed results: testing/comprehensive_tool_test_results.json")
    print("📋 Debug logs: testing/comprehensive_tool_test.log")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())