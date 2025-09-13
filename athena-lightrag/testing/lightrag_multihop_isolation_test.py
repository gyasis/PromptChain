#!/usr/bin/env python3
"""
Interactive LightRAG Multi-Hop Reasoning Test with PromptChain Integration
========================================================================
Enhanced test that integrates PromptChain with MCP configuration for proper
testing of the lightrag_multi_hop_reasoning tool with interactive query loop.

Features:
- PromptChain integration with proper MCP server configuration
- Interactive query loop for continuous testing
- Progressive discovery validation
- JSON serialization testing
- Real-time query input and result display
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import time
from typing import Dict, Any, List, Optional

# Add project root and parent directories to path for imports
project_root = Path(__file__).parent.parent
parent_dir = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(parent_dir))

# Import PromptChain and MCP components
try:
    from promptchain import PromptChain
    from promptchain.utils.agentic_step_processor import AgenticStepProcessor
    print("✅ PromptChain imported successfully")
except ImportError as e:
    print(f"❌ Failed to import PromptChain: {e}")
    print("Make sure PromptChain is installed and in the path")
    sys.exit(1)

# Import the agentic_lightrag module directly
try:
    from agentic_lightrag import AgenticLightRAG, MultiHopContext
    print("✅ AgenticLightRAG imported successfully")
except ImportError as e:
    print(f"❌ Failed to import AgenticLightRAG: {e}")
    print("Make sure you're in the UV environment and dependencies are installed")
    sys.exit(1)

class InteractiveLightRAGTester:
    """
    Interactive LightRAG Multi-Hop Reasoning Tester with PromptChain Integration
    """
    
    def __init__(self):
        self.chain = None
        self.mcp_config = self._get_mcp_server_config()
        self.test_results = []
        
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
            "read_timeout_seconds": 120  # Increased timeout for complex queries
        }]
    
    async def test_direct_lightrag(self, query: str, objective: str) -> Dict[str, Any]:
        """Test LightRAG directly without MCP (isolation test)"""
        print("🔬 Testing LightRAG directly (isolation mode)...")
        print("=" * 70)
        
        test_result = {
            "test_name": "direct_lightrag",
            "success": False,
            "error": None,
            "result": None,
            "execution_time": 0,
            "reasoning_steps": 0,
            "json_serializable": False
        }
        
        try:
            start_time = time.time()
            
            # Initialize AgenticLightRAG directly
            print("🚀 Initializing AgenticLightRAG...")
            agentic_rag = AgenticLightRAG(verbose=True)
            
            # Call the multi-hop reasoning function directly
            print("🧠 Executing multi-hop reasoning...")
            result = await agentic_rag.execute_multi_hop_reasoning(
                query=query,
                objective=objective,
                reset_context=True,
                timeout_seconds=120.0
            )
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["result"] = result
            
            # Analyze result structure
            if isinstance(result, dict):
                test_result["success"] = result.get("success", False)
                test_result["reasoning_steps"] = len(result.get("reasoning_steps", []))
                
                # Test JSON serialization
                try:
                    json.dumps(result)
                    test_result["json_serializable"] = True
                    print("✅ JSON serialization successful")
                except Exception as json_error:
                    print(f"❌ JSON serialization failed: {json_error}")
                    test_result["error"] = f"JSON serialization failed: {json_error}"
            else:
                test_result["error"] = f"Result is not a dictionary: {type(result)}"
                
        except Exception as e:
            test_result["error"] = str(e)
            print(f"❌ Direct LightRAG test failed: {e}")
            
        return test_result
    
    async def test_promptchain_mcp(self, query: str, objective: str) -> Dict[str, Any]:
        """Test LightRAG through PromptChain MCP integration"""
        print("🔗 Testing LightRAG through PromptChain MCP...")
        print("=" * 70)
        
        test_result = {
            "test_name": "promptchain_mcp",
            "success": False,
            "error": None,
            "result": None,
            "execution_time": 0,
            "tools_discovered": []
        }
        
        try:
            start_time = time.time()
            
            # Create PromptChain with MCP server
            print("🚀 Initializing PromptChain with MCP...")
            self.chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=[
                    f"Objective: {objective}",
                    "MANDATORY: You MUST use the available MCP tools to explore the database schema. Do NOT ask for clarification.",
                    "REQUIRED TOOL SEQUENCE:",
                    "1. Call lightrag_local_query with top_k=200 to find tables related to: {input}",
                    "2. Call lightrag_hybrid_query with max_entity_tokens=15000 to understand table relationships", 
                    "3. Call lightrag_sql_generation to create proper SQL queries",
                    "4. Return specific table names (athena.athenaone.TABLENAME) and column information",
                    "Focus on progressive discovery and multi-hop reasoning through tool calls"
                ],
                mcp_servers=self.mcp_config,
                verbose=True
            )
            
            # Connect to MCP server
            print("📡 Connecting to MCP server...")
            await self.chain.mcp_helper.connect_mcp_async()
            
            # Check discovered tools
            tools = [t['function']['name'] for t in self.chain.tools if t['function']['name'].startswith('mcp_')]
            test_result["tools_discovered"] = tools
            print(f"✅ Discovered {len(tools)} MCP tools: {', '.join(tools)}")
            
            # Execute query through PromptChain
            print("🧠 Executing query through PromptChain...")
            result = await self.chain.process_prompt_async(query)
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["result"] = result
            test_result["success"] = bool(result)
            
            print(f"✅ PromptChain MCP test completed in {execution_time:.2f}s")
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"❌ PromptChain MCP test failed: {e}")
            
        return test_result
    
    async def test_agentic_step_processor(self, query: str, objective: str) -> Dict[str, Any]:
        """Test with AgenticStepProcessor for intelligent multi-step reasoning"""
        print("🤖 Testing with AgenticStepProcessor...")
        print("=" * 70)
        
        test_result = {
            "test_name": "agentic_step_processor",
            "success": False,
            "error": None,
            "result": None,
            "execution_time": 0,
            "reasoning_steps": 0
        }
        
        try:
            start_time = time.time()
            
            # Create agentic step processor with directive instructions
            agentic_step = AgenticStepProcessor(
                objective=f"{objective}\n\nMANDATORY ACTIONS:\n1. Use lightrag_local_query with top_k=200 to find tables related to patient journey\n2. Use lightrag_hybrid_query with max_entity_tokens=15000 to understand table relationships\n3. Use lightrag_sql_generation to create SQL queries\n4. Return specific athena.athenaone table names and column information\nDo NOT ask for clarification - use the tools directly.",
                max_internal_steps=5,
                model_name="openai/gpt-4.1-mini"
            )
            
            # Create chain with agentic reasoning
            chain_with_agentic = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=[
                    "Initialize healthcare analysis: {input}",
                    agentic_step,  # Intelligent multi-step reasoning
                    "Summarize findings: {input}"
                ],
                mcp_servers=self.mcp_config,
                verbose=True
            )
            
            # Connect MCP
            await chain_with_agentic.mcp_helper.connect_mcp_async()
            
            # Run agentic analysis
            result = await chain_with_agentic.process_prompt_async(query)
            
            execution_time = time.time() - start_time
            test_result["execution_time"] = execution_time
            test_result["result"] = result
            test_result["reasoning_steps"] = len(agentic_step.step_outputs) if hasattr(agentic_step, 'step_outputs') else 0
            test_result["success"] = bool(result)
            
            print(f"✅ AgenticStepProcessor test completed in {execution_time:.2f}s")
            print(f"   Reasoning steps: {test_result['reasoning_steps']}")
            
            # Cleanup
            await chain_with_agentic.mcp_helper.close_mcp_async()
            
        except Exception as e:
            test_result["error"] = str(e)
            print(f"❌ AgenticStepProcessor test failed: {e}")
            
        return test_result
    
    def validate_tool_usage(self, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that tools were actually used in the test"""
        validation = {
            "tools_used": False,
            "athena_tables_found": False,
            "sql_generated": False,
            "specific_columns_found": False,
            "issues": []
        }
        
        result_text = str(test_result.get('result', '')).lower()
        
        # Check if tools were used (look for tool call patterns)
        if any(pattern in result_text for pattern in ['lightrag_', 'mcp_', 'tool call', 'function call']):
            validation["tools_used"] = True
        else:
            validation["issues"].append("No evidence of tool usage found")
        
        # Check if Athena tables were found
        if any(pattern in result_text for pattern in ['athena.athenaone', 'appointment', 'patient', 'charge', 'diagnosis']):
            validation["athena_tables_found"] = True
        else:
            validation["issues"].append("No Athena table references found")
        
        # Check if SQL was generated
        if any(pattern in result_text for pattern in ['select', 'from', 'join', 'where', 'sql']):
            validation["sql_generated"] = True
        else:
            validation["issues"].append("No SQL queries generated")
        
        # Check if specific columns were mentioned
        if any(pattern in result_text for pattern in ['column', 'field', 'id', 'name', 'date', 'status']):
            validation["specific_columns_found"] = True
        else:
            validation["issues"].append("No specific column information found")
        
        return validation
    
    def display_test_result(self, test_result: Dict[str, Any]):
        """Display formatted test result"""
        print(f"\n📋 {test_result['test_name'].upper()} RESULTS:")
        print("-" * 50)
        print(f"Success: {'✅' if test_result['success'] else '❌'}")
        print(f"Execution Time: {test_result.get('execution_time', 0):.2f}s")
        
        if test_result.get('error'):
            print(f"Error: {test_result['error']}")
        
        if test_result.get('reasoning_steps', 0) > 0:
            print(f"Reasoning Steps: {test_result['reasoning_steps']}")
        
        if test_result.get('tools_discovered'):
            print(f"Tools Discovered: {len(test_result['tools_discovered'])}")
            for tool in test_result['tools_discovered'][:3]:
                print(f"  • {tool}")
        
        # Validate tool usage
        validation = self.validate_tool_usage(test_result)
        print(f"\n🔍 TOOL USAGE VALIDATION:")
        print(f"  Tools Used: {'✅' if validation['tools_used'] else '❌'}")
        print(f"  Athena Tables Found: {'✅' if validation['athena_tables_found'] else '❌'}")
        print(f"  SQL Generated: {'✅' if validation['sql_generated'] else '❌'}")
        print(f"  Column Info Found: {'✅' if validation['specific_columns_found'] else '❌'}")
        
        if validation['issues']:
            print(f"  Issues: {', '.join(validation['issues'])}")
        
        if test_result.get('result'):
            result_preview = str(test_result['result'])[:300] + "..." if len(str(test_result['result'])) > 300 else str(test_result['result'])
            print(f"Result Preview: {result_preview}")
    
    async def run_interactive_session(self):
        """Run interactive testing session with query loop"""
        print("🏥 Interactive LightRAG Multi-Hop Reasoning Test")
        print("=" * 70)
        print("This test will run LightRAG through multiple methods:")
        print("1. Direct LightRAG (isolation)")
        print("2. PromptChain MCP integration")
        print("3. AgenticStepProcessor with MCP")
        print("=" * 70)
        
        # Default test query
        default_query = "Find the patient journey from clinical encounters to orders and bills - show me the table structure and SQL queries"
        default_objective = """
        MANDATORY TASK: Use LightRAG tools to find Athena database tables and generate SQL queries.
        
        REQUIRED TOOL SEQUENCE:
        1. Call lightrag_local_query with top_k=200 and "patient journey clinical encounters orders bills"
        2. Call lightrag_hybrid_query with max_entity_tokens=15000 to understand table relationships
        3. Call lightrag_sql_generation to create proper SQL queries
        4. Return specific athena.athenaone table names and column information
        
        DO NOT ask for clarification - use the tools directly to explore the database.
        Return specific table names like athena.athenaone.APPOINTMENT and column details.
        """
        
        while True:
            print("\n" + "=" * 70)
            print("🔍 QUERY OPTIONS:")
            print("1. Use default healthcare discovery query")
            print("2. Enter custom query")
            print("3. Exit")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "3":
                print("👋 Exiting interactive session...")
                break
            elif choice == "1":
                query = default_query
                objective = default_objective
                print(f"📝 Using default query: {query[:100]}...")
            elif choice == "2":
                query = input("\n🔍 Enter your healthcare query: ").strip()
                if not query:
                    print("❌ Empty query, using default...")
                    query = default_query
                    objective = default_objective
                else:
                    objective = input("🎯 Enter objective (or press Enter for auto): ").strip()
                    if not objective:
                        objective = f"""
                        MANDATORY TASK: Use LightRAG tools to explore the Athena database for: {query}
                        
                        REQUIRED TOOL SEQUENCE:
                        1. Call lightrag_local_query with top_k=200 to find relevant tables
                        2. Call lightrag_hybrid_query with max_entity_tokens=15000 to understand relationships
                        3. Call lightrag_sql_generation to create SQL queries
                        4. Return specific athena.athenaone table names and column information
                        
                        DO NOT ask for clarification - use the tools directly.
                        """
            else:
                print("❌ Invalid choice, using default query...")
                query = default_query
                objective = default_objective
            
            print(f"\n🎯 Testing query: {query[:100]}...")
            print(f"📋 Objective: {objective[:200]}...")
            
            # Run all three test methods
            test_methods = [
                ("Direct LightRAG", self.test_direct_lightrag),
                ("PromptChain MCP", self.test_promptchain_mcp),
                ("AgenticStepProcessor", self.test_agentic_step_processor)
            ]
            
            session_results = []
            
            for method_name, test_method in test_methods:
                print(f"\n🧪 Running {method_name}...")
                try:
                    result = await test_method(query, objective)
                    result["method_name"] = method_name
                    session_results.append(result)
                    self.display_test_result(result)
                except Exception as e:
                    print(f"❌ {method_name} failed with exception: {e}")
                    session_results.append({
                        "method_name": method_name,
                        "success": False,
                        "error": str(e)
                    })
            
            # Summary
            print("\n" + "=" * 70)
            print("📊 SESSION SUMMARY:")
            successful_tests = sum(1 for r in session_results if r.get("success", False))
            print(f"Successful tests: {successful_tests}/{len(session_results)}")
            
            for result in session_results:
                status = "✅" if result.get("success", False) else "❌"
                print(f"  {status} {result['method_name']}")
                if result.get("error"):
                    print(f"    Error: {result['error']}")
            
            # Ask if user wants to continue
            continue_choice = input("\n🔄 Run another test? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                break
        
        # Cleanup
        if self.chain:
            try:
                await self.chain.mcp_helper.close_mcp_async()
                print("🧹 MCP connections closed")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")

async def main():
    """Main execution function for interactive testing"""
    print("🏥 Interactive LightRAG Multi-Hop Reasoning Test")
    print("=" * 70)
    print("This enhanced test integrates PromptChain with MCP configuration")
    print("and provides an interactive query loop for comprehensive testing.")
    print("=" * 70)
    
    # Create tester instance
    tester = InteractiveLightRAGTester()
    
    # Run interactive session
    await tester.run_interactive_session()
    
    print("\n🎉 Testing session completed!")
    print("Check the results above to see which methods worked best.")

if __name__ == "__main__":
    asyncio.run(main())