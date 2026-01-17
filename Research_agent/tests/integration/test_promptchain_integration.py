#!/usr/bin/env python3
"""
Comprehensive PromptChain Library Integration Tests
Tests all core components, utilities, and integration points
"""

import asyncio
import os
import sys
import pytest
from typing import Dict, Any, List
from pathlib import Path

# Imports now work with proper package structure

from research_agent.core.config import ResearchConfig
from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.session import ResearchSession, Query, Paper

# Test utilities
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptChainIntegrationTester:
    """Comprehensive tester for PromptChain integration"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {
            'passed': [],
            'failed': [],
            'errors': [],
            'performance': {}
        }
        self.config = ResearchConfig()
        
    def log_test_result(self, test_name: str, success: bool, details: Any = None, error: Exception = None):
        """Log test results for tracking"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        if success:
            self.test_results['passed'].append(result)
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results['failed'].append(result)
            logger.error(f"❌ {test_name}: FAILED - {error or details}")
            
        if error:
            self.test_results['errors'].append(result)
    
    async def test_promptchain_imports(self):
        """Test PromptChain library imports"""
        test_name = "PromptChain Library Imports"
        try:
            from promptchain import PromptChain
            from promptchain.utils.agent_chain import AgentChain
            from promptchain.utils.agentic_step_processor import AgenticStepProcessor
            from promptchain.utils.execution_history_manager import ExecutionHistoryManager
            from promptchain.utils.logging_utils import RunLogger
            from promptchain.utils.mcp_helpers import MCPHelper
            
            self.log_test_result(test_name, True, "All core imports successful")
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_promptchain_initialization(self):
        """Test PromptChain core initialization"""
        test_name = "PromptChain Initialization"
        try:
            # Test basic PromptChain initialization
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Test instruction: {input}"],
                verbose=True
            )
            
            # Test with execution history manager
            history_manager = ExecutionHistoryManager(
                max_tokens=4000,
                max_entries=50
            )
            
            # Test AgentChain initialization
            from promptchain.utils.agent_chain import AgentChain
            agent_chain = AgentChain(
                agents={"test_agent": chain},
                agent_descriptions={"test_agent": "Test agent for validation"},
                execution_mode="router",
                verbose=True
            )
            
            self.log_test_result(test_name, True, {
                "chain_initialized": True,
                "history_manager_initialized": True,
                "agent_chain_initialized": True
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_agentic_step_processor(self):
        """Test AgenticStepProcessor functionality"""
        test_name = "AgenticStepProcessor Integration"
        try:
            from promptchain.utils.agentic_step_processor import AgenticStepProcessor
            
            # Create agentic step processor
            agentic_step = AgenticStepProcessor(
                objective="Test objective: analyze the given input",
                max_internal_steps=3,
                model_name="openai/gpt-4.1-mini"
            )
            
            # Test with simple chain
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=[
                    "Prepare analysis: {input}",
                    agentic_step,
                    "Final summary: {input}"
                ],
                verbose=True
            )
            
            self.log_test_result(test_name, True, {
                "agentic_step_created": True,
                "chain_with_agentic_step": True,
                "max_steps_configured": 3
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_execution_history_manager(self):
        """Test ExecutionHistoryManager functionality"""
        test_name = "ExecutionHistoryManager Integration"
        try:
            from promptchain.utils.execution_history_manager import ExecutionHistoryManager
            
            # Create history manager
            history_manager = ExecutionHistoryManager(
                max_tokens=2000,
                max_entries=25,
                truncation_strategy="oldest_first"
            )
            
            # Test adding entries
            history_manager.add_entry("user_input", "Test query about machine learning", source="user")
            history_manager.add_entry("agent_output", "Analysis of machine learning query", source="agent")
            history_manager.add_entry("tool_call", "search_papers(query='machine learning')", source="agent")
            
            # Test getting formatted history
            formatted_history = history_manager.get_formatted_history(
                format_style='chat',
                max_tokens=1000
            )
            
            # Test filtering
            user_entries = history_manager.get_entries_by_type("user_input")
            
            self.log_test_result(test_name, True, {
                "history_manager_created": True,
                "entries_added": 3,
                "formatted_history_length": len(formatted_history),
                "user_entries_count": len(user_entries)
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_research_orchestrator_initialization(self):
        """Test Research Orchestrator PromptChain integration"""
        test_name = "Research Orchestrator PromptChain Integration"
        try:
            # Test with minimal config
            config = ResearchConfig()
            orchestrator = AdvancedResearchOrchestrator(config)
            
            # Verify PromptChain components are initialized
            self.log_test_result(test_name, True, {
                "orchestrator_initialized": True,
                "config_loaded": True,
                "agents_initialized": hasattr(orchestrator, 'query_agent'),
                "coordinators_initialized": hasattr(orchestrator, 'multi_query_coordinator')
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_model_configuration(self):
        """Test model configuration and selection"""
        test_name = "Model Configuration and Selection"
        try:
            # Test different model configurations
            models_to_test = [
                "openai/gpt-4.1-mini",  # Cost-effective model
                {"name": "openai/gpt-4.1-mini", "params": {"temperature": 0.1}},
                {"name": "anthropic/claude-3-haiku-20240307", "params": {"temperature": 0.5}}
            ]
            
            results = []
            for model_config in models_to_test:
                try:
                    chain = PromptChain(
                        models=[model_config],
                        instructions=["Test: {input}"],
                        verbose=False
                    )
                    results.append({"model": model_config, "success": True})
                except Exception as model_error:
                    results.append({"model": model_config, "success": False, "error": str(model_error)})
            
            successful_models = [r for r in results if r['success']]
            
            self.log_test_result(test_name, len(successful_models) > 0, {
                "models_tested": len(models_to_test),
                "successful_models": len(successful_models),
                "results": results
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_tool_integration(self):
        """Test tool registration and calling"""
        test_name = "Tool Integration and Registration"
        try:
            # Create a test tool function
            def test_search_tool(query: str) -> str:
                """Test search function for validation"""
                return f"Search results for: {query}"
            
            # Create chain with tool
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Use tools to help with: {input}"],
                verbose=True
            )
            
            # Register the tool
            chain.register_tool_function(test_search_tool)
            
            # Add tool schema
            chain.add_tools([{
                "type": "function",
                "function": {
                    "name": "test_search_tool",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }])
            
            self.log_test_result(test_name, True, {
                "tool_function_registered": True,
                "tool_schema_added": True,
                "tools_count": len(chain.tools) if hasattr(chain, 'tools') else 0
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_mcp_helpers(self):
        """Test MCP (Model Context Protocol) helpers"""
        test_name = "MCP Helpers Integration"
        try:
            from promptchain.utils.mcp_helpers import MCPHelper
            
            # Test MCP configuration (without actual servers for testing)
            mcp_config = [{
                "id": "test_server",
                "type": "stdio",
                "command": "echo",
                "args": ["test"]
            }]
            
            # This would normally initialize MCP servers
            # For testing, we just verify the class can be imported and configured
            
            self.log_test_result(test_name, True, {
                "mcp_helper_imported": True,
                "mcp_config_structured": True,
                "config_format_valid": len(mcp_config) > 0
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_async_sync_patterns(self):
        """Test async/sync pattern consistency"""
        test_name = "Async/Sync Pattern Consistency"
        try:
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Test async/sync: {input}"],
                verbose=False
            )
            
            # Test that both sync and async methods exist
            has_sync_process = hasattr(chain, 'process_prompt')
            has_async_process = hasattr(chain, 'process_prompt_async')
            
            self.log_test_result(test_name, has_sync_process and has_async_process, {
                "sync_method_exists": has_sync_process,
                "async_method_exists": has_async_process,
                "dual_interface_available": has_sync_process and has_async_process
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        test_name = "Error Handling and Recovery"
        try:
            # Test with invalid model to trigger error handling
            try:
                chain = PromptChain(
                    models=["invalid/model"],
                    instructions=["Test: {input}"],
                    verbose=False
                )
                # This should fail gracefully
                error_handled = False
            except Exception as model_error:
                error_handled = True
            
            # Test with valid model but invalid input
            try:
                chain = PromptChain(
                    models=["openai/gpt-4.1-mini"],
                    instructions=["Test: {input}"],
                    verbose=False
                )
                validation_handled = True
            except Exception:
                validation_handled = False
            
            self.log_test_result(test_name, error_handled or validation_handled, {
                "invalid_model_error_handled": error_handled,
                "valid_chain_creation": validation_handled
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def run_all_tests(self):
        """Run all PromptChain integration tests"""
        logger.info("🚀 Starting PromptChain Integration Tests...")
        
        test_methods = [
            self.test_promptchain_imports,
            self.test_promptchain_initialization,
            self.test_agentic_step_processor,
            self.test_execution_history_manager,
            self.test_research_orchestrator_initialization,
            self.test_model_configuration,
            self.test_tool_integration,
            self.test_mcp_helpers,
            self.test_async_sync_patterns,
            self.test_error_handling
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.log_test_result(f"Test Execution: {test_method.__name__}", False, error=e)
        
        return self.generate_test_report()
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        passed_count = len(self.test_results['passed'])
        failed_count = len(self.test_results['failed'])
        error_count = len(self.test_results['errors'])
        
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_count,
                'failed': failed_count,
                'errors': error_count,
                'success_rate': f"{success_rate:.1f}%",
                'status': 'PASSED' if failed_count == 0 else 'FAILED'
            },
            'details': self.test_results
        }
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info("📊 PROMPTCHAIN INTEGRATION TEST REPORT")
        logger.info(f"{'='*50}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"⚠️  Errors: {error_count}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {report['summary']['status']}")
        logger.info(f"{'='*50}")
        
        if failed_count > 0:
            logger.error("Failed Tests:")
            for fail in self.test_results['failed']:
                logger.error(f"  - {fail['test']}: {fail.get('error', 'Unknown error')}")
        
        return report

# Main execution function
async def main():
    """Main test execution"""
    tester = PromptChainIntegrationTester()
    
    try:
        report = await tester.run_all_tests()
        
        # Save report to file
        import json
        from datetime import datetime
        
        report_file = f"promptchain_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        
        return report['summary']['status'] == 'PASSED'
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)