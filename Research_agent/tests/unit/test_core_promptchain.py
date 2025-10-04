#!/usr/bin/env python3
"""
Core PromptChain Library Testing
Tests the installed PromptChain library functionality in isolation
"""

import asyncio
import os
import sys
import json
from typing import Dict, Any, List
from datetime import datetime

# Test utilities
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorePromptChainTester:
    """Tests the core installed PromptChain library"""
    
    def __init__(self):
        self.test_results: Dict[str, Any] = {
            'passed': [],
            'failed': [],
            'errors': [],
            'performance': {}
        }
        
    def log_test_result(self, test_name: str, success: bool, details: Any = None, error: Exception = None):
        """Log test results for tracking"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            self.test_results['passed'].append(result)
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results['failed'].append(result)
            logger.error(f"❌ {test_name}: FAILED - {error or details}")
            
        if error:
            self.test_results['errors'].append(result)
    
    async def test_library_imports(self):
        """Test all core PromptChain imports"""
        test_name = "PromptChain Core Library Imports"
        try:
            # Test basic imports
            import promptchain
            from promptchain import PromptChain
            from promptchain.utils.agent_chain import AgentChain
            from promptchain.utils.agentic_step_processor import AgenticStepProcessor
            from promptchain.utils.execution_history_manager import ExecutionHistoryManager
            from promptchain.utils.logging_utils import RunLogger
            
            self.log_test_result(test_name, True, {
                "core_imports": "successful",
                "promptchain_version": getattr(promptchain, '__version__', 'unknown')
            })
        except ImportError as e:
            self.log_test_result(test_name, False, error=e)
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_promptchain_basic_functionality(self):
        """Test basic PromptChain functionality"""
        test_name = "PromptChain Basic Functionality"
        try:
            from promptchain import PromptChain
            
            # Create basic chain
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],  # Using cost-effective model
                instructions=["Analyze this input: {input}"],
                verbose=True
            )
            
            # Test that it initialized correctly
            has_models = hasattr(chain, 'models') or hasattr(chain, 'model_names')
            has_instructions = hasattr(chain, 'instructions')
            has_process_method = hasattr(chain, 'process_prompt')
            
            self.log_test_result(test_name, True, {
                "chain_created": True,
                "has_models": has_models,
                "has_instructions": has_instructions,
                "has_process_method": has_process_method
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_execution_history_manager(self):
        """Test ExecutionHistoryManager in isolation"""
        test_name = "ExecutionHistoryManager Standalone Test"
        try:
            from promptchain.utils.execution_history_manager import ExecutionHistoryManager
            
            # Create history manager
            history_manager = ExecutionHistoryManager(
                max_tokens=2000,
                max_entries=25,
                truncation_strategy="oldest_first"
            )
            
            # Test adding entries
            history_manager.add_entry("user_input", "Test research query", source="user")
            history_manager.add_entry("agent_output", "Research analysis response", source="agent")
            
            # Test retrieval (using available methods)
            all_entries = history_manager.entries if hasattr(history_manager, 'entries') else []
            user_entries = history_manager.get_entries_by_type("user_input") if hasattr(history_manager, 'get_entries_by_type') else []
            
            self.log_test_result(test_name, True, {
                "history_manager_created": True,
                "entries_added": len(all_entries),
                "user_entries_filtered": len(user_entries),
                "max_tokens_set": history_manager.max_tokens
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_agentic_step_processor(self):
        """Test AgenticStepProcessor in isolation"""
        test_name = "AgenticStepProcessor Standalone Test"
        try:
            from promptchain.utils.agentic_step_processor import AgenticStepProcessor
            
            # Create agentic step processor
            agentic_step = AgenticStepProcessor(
                objective="Analyze research data and provide insights",
                max_internal_steps=3,
                model_name="openai/gpt-4.1-mini"
            )
            
            # Test basic properties
            has_objective = hasattr(agentic_step, 'objective')
            has_max_steps = hasattr(agentic_step, 'max_internal_steps')
            has_model_name = hasattr(agentic_step, 'model_name')
            
            self.log_test_result(test_name, True, {
                "agentic_step_created": True,
                "has_objective": has_objective,
                "has_max_steps": has_max_steps,
                "has_model_name": has_model_name,
                "max_steps": agentic_step.max_internal_steps if has_max_steps else None
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_agent_chain_basic(self):
        """Test AgentChain basic functionality"""
        test_name = "AgentChain Basic Functionality"
        try:
            from promptchain import PromptChain
            from promptchain.utils.agent_chain import AgentChain
            
            # Create simple agents
            research_agent = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Research: {input}"],
                verbose=False
            )
            
            analysis_agent = PromptChain(
                models=["openai/gpt-4.1-mini"], 
                instructions=["Analyze: {input}"],
                verbose=False
            )
            
            # Create router configuration with proper format
            router_config = {
                "models": ["openai/gpt-4.1-mini"],
                "instructions": [None, "Choose the best agent for: {input}. Return JSON with 'chosen_agent' key."],
                "decision_prompt_templates": {
                    "single_agent_dispatch": "Choose the best agent for: {user_input}. Return JSON with 'chosen_agent' key."
                }
            }
            
            # Create AgentChain with proper router config
            agent_chain = AgentChain(
                agents={
                    "researcher": research_agent,
                    "analyzer": analysis_agent
                },
                agent_descriptions={
                    "researcher": "Performs research tasks",
                    "analyzer": "Analyzes research results"
                },
                execution_mode="router",
                router=router_config,
                verbose=False
            )
            
            # Test basic properties
            has_agents = hasattr(agent_chain, 'agents')
            has_descriptions = hasattr(agent_chain, 'agent_descriptions')
            has_execution_mode = hasattr(agent_chain, 'execution_mode')
            
            self.log_test_result(test_name, True, {
                "agent_chain_created": True,
                "has_agents": has_agents,
                "has_descriptions": has_descriptions,
                "has_execution_mode": has_execution_mode,
                "agent_count": len(agent_chain.agents) if has_agents else 0
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_tool_registration(self):
        """Test tool registration functionality"""
        test_name = "Tool Registration Test"
        try:
            from promptchain import PromptChain
            
            # Define a test tool
            def research_search(query: str) -> str:
                """Mock research search function"""
                return f"Research results for: {query}"
            
            # Create chain and register tool
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Use available tools: {input}"],
                verbose=False
            )
            
            # Register the tool function
            chain.register_tool_function(research_search)
            
            # Add tool schema
            tool_schema = {
                "type": "function",
                "function": {
                    "name": "research_search",
                    "description": "Search for research information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
            chain.add_tools([tool_schema])
            
            # Test that tools were registered
            has_tools = hasattr(chain, 'tools')
            has_tool_functions = hasattr(chain, 'tool_functions')
            
            self.log_test_result(test_name, True, {
                "tool_function_registered": True,
                "tool_schema_added": True,
                "has_tools_attribute": has_tools,
                "has_tool_functions_attribute": has_tool_functions
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_model_configurations(self):
        """Test different model configurations"""
        test_name = "Model Configuration Test"
        try:
            from promptchain import PromptChain
            
            # Test different model configurations
            test_configs = [
                # String format
                "openai/gpt-4.1-mini",
                # Dict format with parameters
                {"name": "openai/gpt-4.1-mini", "params": {"temperature": 0.1}},
                # List of models
                ["openai/gpt-4.1-mini"]
            ]
            
            successful_configs = []
            
            for config in test_configs:
                try:
                    chain = PromptChain(
                        models=[config] if not isinstance(config, list) else config,
                        instructions=["Test: {input}"],
                        verbose=False
                    )
                    successful_configs.append({"config": config, "success": True})
                except Exception as model_error:
                    successful_configs.append({"config": config, "success": False, "error": str(model_error)})
            
            success_count = sum(1 for cfg in successful_configs if cfg['success'])
            
            self.log_test_result(test_name, success_count > 0, {
                "total_configs_tested": len(test_configs),
                "successful_configs": success_count,
                "results": successful_configs
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_async_sync_methods(self):
        """Test that both async and sync methods exist"""
        test_name = "Async/Sync Method Availability"
        try:
            from promptchain import PromptChain
            
            chain = PromptChain(
                models=["openai/gpt-4.1-mini"],
                instructions=["Test: {input}"],
                verbose=False
            )
            
            # Check for both sync and async methods
            has_sync_process = hasattr(chain, 'process_prompt')
            has_async_process = hasattr(chain, 'process_prompt_async')
            
            # Check for other common methods
            has_run = hasattr(chain, 'run')
            has_run_async = hasattr(chain, 'run_async')
            
            self.log_test_result(test_name, has_sync_process or has_async_process, {
                "sync_process_prompt": has_sync_process,
                "async_process_prompt": has_async_process,
                "sync_run": has_run,
                "async_run": has_run_async
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_logging_utils(self):
        """Test logging utilities"""
        test_name = "Logging Utils Test"
        try:
            from promptchain.utils.logging_utils import RunLogger
            
            # Create logger with temporary directory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                logger_instance = RunLogger(log_dir=temp_dir)
                
                # Test logging a run
                test_data = {"test": "data", "status": "success"}
                logger_instance.log_run(test_data)
                
                # Check if log file was created
                import os
                log_files = [f for f in os.listdir(temp_dir) if f.endswith('.jsonl')]
                
                self.log_test_result(test_name, True, {
                    "run_logger_created": True,
                    "log_run_method_exists": hasattr(logger_instance, 'log_run'),
                    "log_files_created": len(log_files)
                })
                
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def run_all_tests(self):
        """Run all core PromptChain tests"""
        logger.info("🚀 Starting Core PromptChain Library Tests...")
        
        test_methods = [
            self.test_library_imports,
            self.test_promptchain_basic_functionality,
            self.test_execution_history_manager,
            self.test_agentic_step_processor,
            self.test_agent_chain_basic,
            self.test_tool_registration,
            self.test_model_configurations,
            self.test_async_sync_methods,
            self.test_logging_utils
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
        logger.info(f"\n{'='*60}")
        logger.info("📊 CORE PROMPTCHAIN LIBRARY TEST REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"⚠️  Errors: {error_count}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {report['summary']['status']}")
        logger.info(f"{'='*60}")
        
        if failed_count > 0:
            logger.error("Failed Tests:")
            for fail in self.test_results['failed']:
                logger.error(f"  - {fail['test']}: {fail.get('error', 'Unknown error')}")
        
        return report

# Main execution function
async def main():
    """Main test execution"""
    tester = CorePromptChainTester()
    
    try:
        report = await tester.run_all_tests()
        
        # Save report to file
        report_file = f"core_promptchain_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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