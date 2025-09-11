#!/usr/bin/env python3
"""
Athena LightRAG Integration Test
===============================
Comprehensive testing of the Athena LightRAG MCP Server implementation.
Tests all query modes, tools, and integration with the existing database.

Author: Athena LightRAG System
Date: 2025-09-08
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, List
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from lightrag_core import create_athena_lightrag, QueryMode
    from agentic_lightrag import create_agentic_lightrag
    from context_processor import create_context_processor, create_sql_generator
    from athena_mcp_server import create_manual_mcp_server
    from config import get_config
    
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AthenaIntegrationTester:
    """Comprehensive integration tester for Athena LightRAG system."""
    
    def __init__(self):
        """Initialize the tester."""
        self.working_dir = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db"
        self.test_queries = [
            "What tables are related to patient appointments?",
            "Show me anesthesia case management structure",
            "How are billing workflows organized?",
            "What collector category tables exist?",
            "Describe allowable schedule categories"
        ]
        self.results = {}
    
    async def test_database_connectivity(self) -> bool:
        """Test basic database connectivity and structure."""
        print("\n=== Testing Database Connectivity ===")
        
        try:
            # Check if database directory exists
            db_path = Path(self.working_dir)
            if not db_path.exists():
                print(f"✗ Database directory not found: {self.working_dir}")
                return False
            
            print(f"✓ Database directory found: {self.working_dir}")
            
            # Check for required files
            required_files = [
                "kv_store_full_entities.json",
                "kv_store_full_relations.json", 
                "kv_store_text_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
                "vdb_chunks.json"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = db_path / file_name
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {file_name}: {size_mb:.2f} MB")
                else:
                    missing_files.append(file_name)
                    print(f"  ✗ {file_name}: Not found")
            
            if missing_files:
                print(f"⚠ Missing files: {missing_files}")
                print("  Database may still work, but some functionality might be limited")
            
            self.results["database_connectivity"] = {
                "success": len(missing_files) < len(required_files) // 2,
                "missing_files": missing_files,
                "working_dir": str(db_path)
            }
            
            return len(missing_files) < len(required_files) // 2
            
        except Exception as e:
            print(f"✗ Database connectivity test failed: {e}")
            self.results["database_connectivity"] = {"success": False, "error": str(e)}
            return False
    
    async def test_lightrag_core(self) -> bool:
        """Test LightRAG core functionality."""
        print("\n=== Testing LightRAG Core ===")
        
        try:
            # Create LightRAG instance
            lightrag = create_athena_lightrag(working_dir=self.working_dir)
            print("✓ LightRAG core created successfully")
            
            # Test database status
            status = lightrag.get_database_status()
            print(f"✓ Database status retrieved: {status['exists']}")
            
            # Test query modes
            test_query = "What tables exist in the database?"
            modes = ["local", "global", "hybrid", "naive", "mix"]
            mode_results = {}
            
            for mode in modes:
                try:
                    print(f"  Testing {mode} mode...")
                    result = await lightrag.query_async(test_query, mode=mode)
                    
                    if result.error:
                        print(f"    ✗ {mode}: {result.error}")
                        mode_results[mode] = {"success": False, "error": result.error}
                    else:
                        result_preview = result.result[:100].replace('\n', ' ')
                        print(f"    ✓ {mode}: {result_preview}... ({result.execution_time:.2f}s)")
                        mode_results[mode] = {
                            "success": True, 
                            "execution_time": result.execution_time,
                            "tokens_used": result.tokens_used,
                            "result_length": len(result.result)
                        }
                
                except Exception as e:
                    print(f"    ✗ {mode}: {str(e)}")
                    mode_results[mode] = {"success": False, "error": str(e)}
            
            # Test context extraction
            try:
                print("  Testing context-only extraction...")
                context = await lightrag.get_context_only_async(test_query, mode="hybrid")
                context_preview = context[:100].replace('\n', ' ')
                print(f"    ✓ Context extraction: {context_preview}...")
                mode_results["context_extract"] = {"success": True, "context_length": len(context)}
            except Exception as e:
                print(f"    ✗ Context extraction: {str(e)}")
                mode_results["context_extract"] = {"success": False, "error": str(e)}
            
            success = sum(1 for r in mode_results.values() if r.get("success", False))
            total = len(mode_results)
            print(f"✓ LightRAG core test: {success}/{total} modes successful")
            
            self.results["lightrag_core"] = {
                "success": success > total // 2,
                "mode_results": mode_results,
                "success_rate": success / total
            }
            
            return success > total // 2
            
        except Exception as e:
            print(f"✗ LightRAG core test failed: {e}")
            self.results["lightrag_core"] = {"success": False, "error": str(e)}
            return False
    
    async def test_agentic_reasoning(self) -> bool:
        """Test agentic reasoning functionality."""
        print("\n=== Testing Agentic Reasoning ===")
        
        try:
            # Create agentic system
            agentic_rag = create_agentic_lightrag(working_dir=self.working_dir, verbose=False)
            print("✓ Agentic LightRAG created successfully")
            
            # Test simple reasoning
            simple_query = "What are the main types of medical tables in this database?"
            print("  Testing simple multi-hop reasoning...")
            
            result = await agentic_rag.execute_multi_hop_reasoning(
                query=simple_query,
                objective="Identify and categorize the main types of medical tables"
            )
            
            if result["success"]:
                print(f"    ✓ Reasoning successful: {len(result['reasoning_steps'])} steps")
                print(f"    ✓ Contexts accumulated: {len(result['accumulated_contexts'])}")
                result_preview = result["result"][:150].replace('\n', ' ')
                print(f"    ✓ Result preview: {result_preview}...")
                
                self.results["agentic_reasoning"] = {
                    "success": True,
                    "reasoning_steps": len(result['reasoning_steps']),
                    "contexts_accumulated": len(result['accumulated_contexts']),
                    "tokens_used": result['total_tokens_used'],
                    "result_length": len(result['result'])
                }
                
                return True
            else:
                print(f"    ✗ Reasoning failed: {result.get('error', 'Unknown error')}")
                self.results["agentic_reasoning"] = {
                    "success": False, 
                    "error": result.get('error', 'Unknown error')
                }
                return False
        
        except Exception as e:
            print(f"✗ Agentic reasoning test failed: {e}")
            self.results["agentic_reasoning"] = {"success": False, "error": str(e)}
            return False
    
    async def test_context_processing(self) -> bool:
        """Test context processing and SQL generation."""
        print("\n=== Testing Context Processing ===")
        
        try:
            # Create context processor and SQL generator
            sql_generator = create_sql_generator(working_dir=self.working_dir)
            print("✓ SQL generator created successfully")
            
            # Test SQL generation
            sql_queries = [
                "Show me all patient appointments",
                "Find completed anesthesia cases",
                "Get billing information for today"
            ]
            
            sql_results = {}
            for i, query in enumerate(sql_queries[:2]):  # Test first 2 to save time
                try:
                    print(f"  Testing SQL generation {i+1}: {query}")
                    
                    result = await sql_generator.generate_sql_from_query(query)
                    
                    if result["success"]:
                        sql_preview = result["sql"][:100] if result["sql"] else "No SQL generated"
                        print(f"    ✓ SQL generated: {sql_preview}...")
                        sql_results[query] = {
                            "success": True,
                            "execution_time": result.get("execution_time", 0),
                            "has_explanation": "explanation" in result,
                            "sql_length": len(result["sql"]) if result["sql"] else 0
                        }
                    else:
                        print(f"    ✗ SQL generation failed: {result.get('error', 'Unknown error')}")
                        sql_results[query] = {
                            "success": False, 
                            "error": result.get('error', 'Unknown error')
                        }
                
                except Exception as e:
                    print(f"    ✗ SQL generation error: {str(e)}")
                    sql_results[query] = {"success": False, "error": str(e)}
            
            success_count = sum(1 for r in sql_results.values() if r.get("success", False))
            total_count = len(sql_results)
            
            print(f"✓ Context processing test: {success_count}/{total_count} SQL generations successful")
            
            self.results["context_processing"] = {
                "success": success_count > 0,
                "sql_results": sql_results,
                "success_rate": success_count / total_count if total_count > 0 else 0
            }
            
            return success_count > 0
            
        except Exception as e:
            print(f"✗ Context processing test failed: {e}")
            self.results["context_processing"] = {"success": False, "error": str(e)}
            return False
    
    async def test_mcp_server(self) -> bool:
        """Test MCP server functionality."""
        print("\n=== Testing MCP Server ===")
        
        try:
            # Create manual MCP server
            mcp_server = create_manual_mcp_server(working_dir=self.working_dir)
            print("✓ MCP server created successfully")
            
            # Get tool schemas
            schemas = mcp_server.get_tool_schemas()
            expected_tools = [
                "lightrag_local_query",
                "lightrag_global_query", 
                "lightrag_hybrid_query",
                "lightrag_context_extract",
                "lightrag_multi_hop_reasoning",
                "lightrag_sql_generation"
            ]
            
            print(f"✓ Tool schemas available: {list(schemas.keys())}")
            
            # Test core tools
            tool_results = {}
            test_query = "medical appointment tables"
            
            # Test the 4 core validated tools
            core_tools = [
                ("lightrag_local_query", {"query": test_query}),
                ("lightrag_global_query", {"query": test_query}),
                ("lightrag_hybrid_query", {"query": test_query}),
                ("lightrag_context_extract", {"query": test_query, "mode": "hybrid"})
            ]
            
            for tool_name, params in core_tools:
                try:
                    print(f"  Testing {tool_name}...")
                    
                    result = await mcp_server.call_tool(tool_name, params)
                    
                    if result.get("success", False):
                        content_key = "result" if "result" in result else "context"
                        content = result.get(content_key, "")
                        preview = str(content)[:80].replace('\n', ' ')
                        print(f"    ✓ {tool_name}: {preview}...")
                        
                        tool_results[tool_name] = {
                            "success": True,
                            "execution_time": result.get("execution_time", 0),
                            "content_length": len(str(content))
                        }
                    else:
                        error = result.get("error", "Unknown error")
                        print(f"    ✗ {tool_name}: {error}")
                        tool_results[tool_name] = {"success": False, "error": error}
                
                except Exception as e:
                    print(f"    ✗ {tool_name}: {str(e)}")
                    tool_results[tool_name] = {"success": False, "error": str(e)}
            
            success_count = sum(1 for r in tool_results.values() if r.get("success", False))
            total_count = len(tool_results)
            
            print(f"✓ MCP server test: {success_count}/{total_count} tools successful")
            
            self.results["mcp_server"] = {
                "success": success_count >= 2,  # At least half should work
                "tool_results": tool_results,
                "available_tools": list(schemas.keys()),
                "success_rate": success_count / total_count
            }
            
            return success_count >= 2
            
        except Exception as e:
            print(f"✗ MCP server test failed: {e}")
            self.results["mcp_server"] = {"success": False, "error": str(e)}
            return False
    
    async def test_comprehensive_queries(self) -> bool:
        """Test with realistic medical database queries."""
        print("\n=== Testing Comprehensive Queries ===")
        
        try:
            lightrag = create_athena_lightrag(working_dir=self.working_dir)
            
            medical_queries = [
                "What tables are related to patient appointments and scheduling?",
                "Show me the structure of anesthesia case management",
                "How are billing and financial workflows organized?",
                "What collector and category tables exist?",
                "Describe the allowable schedule categories structure"
            ]
            
            query_results = {}
            
            for query in medical_queries[:3]:  # Test first 3 to manage time
                try:
                    print(f"  Testing: {query[:50]}...")
                    
                    # Test hybrid mode (most balanced)
                    result = await lightrag.query_hybrid_async(query)
                    
                    if result.error:
                        print(f"    ✗ Failed: {result.error}")
                        query_results[query] = {"success": False, "error": result.error}
                    else:
                        # Check if result contains medical/database terminology
                        medical_terms = ["table", "patient", "appointment", "anesthesia", "billing", "schedule"]
                        result_lower = result.result.lower()
                        term_matches = sum(1 for term in medical_terms if term in result_lower)
                        
                        relevance_score = term_matches / len(medical_terms)
                        result_preview = result.result[:120].replace('\n', ' ')
                        
                        print(f"    ✓ Success: {result_preview}... (relevance: {relevance_score:.2f})")
                        
                        query_results[query] = {
                            "success": True,
                            "execution_time": result.execution_time,
                            "result_length": len(result.result),
                            "relevance_score": relevance_score,
                            "tokens_used": result.tokens_used
                        }
                
                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    query_results[query] = {"success": False, "error": str(e)}
            
            success_count = sum(1 for r in query_results.values() if r.get("success", False))
            total_count = len(query_results)
            
            # Calculate average relevance score
            avg_relevance = 0
            if success_count > 0:
                relevance_scores = [r.get("relevance_score", 0) for r in query_results.values() if r.get("success", False)]
                avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
            
            print(f"✓ Comprehensive queries test: {success_count}/{total_count} successful (avg relevance: {avg_relevance:.2f})")
            
            self.results["comprehensive_queries"] = {
                "success": success_count > 0 and avg_relevance > 0.3,
                "query_results": query_results,
                "success_rate": success_count / total_count,
                "average_relevance": avg_relevance
            }
            
            return success_count > 0 and avg_relevance > 0.3
            
        except Exception as e:
            print(f"✗ Comprehensive queries test failed: {e}")
            self.results["comprehensive_queries"] = {"success": False, "error": str(e)}
            return False
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*80)
        print("ATHENA LIGHTRAG INTEGRATION TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("success", False))
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        print(f"\nTest Results:")
        for test_name, result in self.results.items():
            status = "✓ PASS" if result.get("success", False) else "✗ FAIL"
            print(f"  {status} {test_name}")
            
            if not result.get("success", False) and "error" in result:
                print(f"      Error: {result['error']}")
        
        # Overall assessment
        if passed_tests >= total_tests * 0.8:
            print(f"\n🎉 EXCELLENT: System is working well!")
        elif passed_tests >= total_tests * 0.6:
            print(f"\n✅ GOOD: System is functional with minor issues")
        elif passed_tests >= total_tests * 0.4:
            print(f"\n⚠️  MODERATE: System has significant issues but core functionality works")
        else:
            print(f"\n❌ CRITICAL: System has major issues that need to be addressed")
        
        print("="*80)


async def main():
    """Run comprehensive integration tests."""
    print("ATHENA LIGHTRAG INTEGRATION TESTS")
    print("=" * 80)
    
    tester = AthenaIntegrationTester()
    
    # Run all tests
    tests = [
        ("Database Connectivity", tester.test_database_connectivity),
        ("LightRAG Core", tester.test_lightrag_core),
        ("Agentic Reasoning", tester.test_agentic_reasoning),
        ("Context Processing", tester.test_context_processing),
        ("MCP Server", tester.test_mcp_server),
        ("Comprehensive Queries", tester.test_comprehensive_queries)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🚀 Starting {test_name}...")
            await test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            tester.results[test_name.lower().replace(" ", "_")] = {"success": False, "error": str(e)}
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    asyncio.run(main())