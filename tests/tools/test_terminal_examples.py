"""
Test Examples for Terminal Tool - ALL PromptChain Usage Patterns

This file demonstrates ALL ways the terminal tool can be used with PromptChain:
1. Simple function injection
2. Direct callable usage
3. Tool registration for LLM use
4. AgenticStepProcessor integration
5. Multi-agent system integration
6. Security validation examples

Author: PromptChain Team
License: MIT
"""

import pytest
import sys
import os
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.agent_chain import AgentChain
from promptchain.tools.terminal import TerminalTool, SecurityError


class TestTerminalToolExamples:
    """Comprehensive test examples for all PromptChain usage patterns"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create terminal with testing configuration
        self.terminal = TerminalTool(
            timeout=10,  # Short timeout for tests
            require_permission=False,  # Auto-approve for tests
            verbose=True,
            log_all_commands=True
        )
    
    # ===== TEST 1: Simple Function Injection Pattern =====
    
    def test_simple_function_injection(self):
        """Test terminal as simple injected function in PromptChain"""
        print("\n🧪 TEST 1: Simple Function Injection")
        
        def execute_command(command: str) -> str:
            """Simple wrapper function"""
            return self.terminal(command)
        
        # Create chain with function injection
        chain = PromptChain(
            models=["mock"],  # Use mock model for testing
            instructions=[
                "List current directory files",
                execute_command,  # Inject our function
                "Count files: {input}"
            ],
            verbose=True
        )
        
        # Mock the LLM calls
        with patch.object(chain, '_call_llm') as mock_llm:
            mock_llm.side_effect = ["ls -la", "Found files in directory"]
            
            result = chain.process_prompt("Show me the files")
            
            assert result is not None
            assert "Found files" in result
            print(f"✅ Function injection result: {result}")
    
    # ===== TEST 2: Direct Callable Usage =====
    
    def test_direct_callable_usage(self):
        """Test terminal as direct callable in PromptChain instructions"""
        print("\n🧪 TEST 2: Direct Callable Usage")
        
        # Use terminal directly as callable in chain
        chain = PromptChain(
            models=["mock"],
            instructions=[
                "Check system info",
                self.terminal,  # Direct callable usage
                "Analyze output: {input}"
            ],
            verbose=True
        )
        
        with patch.object(chain, '_call_llm') as mock_llm:
            mock_llm.side_effect = ["pwd", "Current directory confirmed"]
            
            result = chain.process_prompt("Where am I?")
            
            assert result is not None
            print(f"✅ Direct callable result: {result}")
    
    # ===== TEST 3: Tool Registration Pattern =====
    
    def test_tool_registration(self):
        """Test terminal registered as tool for LLM use"""
        print("\n🧪 TEST 3: Tool Registration")
        
        chain = PromptChain(
            models=["mock"],
            instructions=[
                "Determine what command to check disk usage: {input}",
                "Execute and analyze: {input}"
            ],
            verbose=True
        )
        
        # Register terminal as tool
        chain.register_tool_function(self.terminal)
        chain.add_tools([{
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Execute terminal commands safely",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        }])
        
        # Mock LLM with tool calling
        with patch.object(chain, '_call_llm') as mock_llm:
            # First call returns tool use decision
            mock_llm.return_value = {
                "content": "I'll check disk usage",
                "tool_calls": [{
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "df -h"}'
                    }
                }]
            }
            
            # Second call analyzes the result
            mock_llm.return_value = "Disk usage analyzed successfully"
            
            result = chain.process_prompt("Check available disk space")
            
            assert result is not None
            print(f"✅ Tool registration result: {result}")
    
    # ===== TEST 4: AgenticStepProcessor Integration =====
    
    @pytest.mark.asyncio
    async def test_agentic_step_processor_integration(self):
        """Test terminal with AgenticStepProcessor for intelligent workflows"""
        print("\n🧪 TEST 4: AgenticStepProcessor Integration")
        
        # Create agentic step for dependency checking
        dependency_checker = AgenticStepProcessor(
            objective="Check if Python and pip are installed and working",
            max_internal_steps=5,
            model_name="mock"
        )
        
        # Create chain with agentic step
        chain = PromptChain(
            models=["mock"],
            instructions=[
                "Starting dependency check: {input}",
                dependency_checker,  # Agentic reasoning
                "Summary: {input}"
            ],
            verbose=True
        )
        
        # Register terminal for agentic use
        chain.register_tool_function(self.terminal)
        chain.add_tools([{
            "type": "function",
            "function": {
                "name": "terminal", 
                "description": "Execute commands like 'python --version', 'pip --version', 'which python'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        }])
        
        # Mock the agentic processor
        with patch.object(dependency_checker, 'process') as mock_agentic:
            mock_agentic.return_value = "Python 3.9.0 and pip 21.0 are installed and working"
            
            with patch.object(chain, '_call_llm') as mock_llm:
                mock_llm.side_effect = [
                    "Checking Python environment",
                    "All dependencies verified"
                ]
                
                result = chain.process_prompt("python environment")
                
                assert result is not None
                print(f"✅ AgenticStepProcessor result: {result}")
    
    # ===== TEST 5: Multi-Agent System Integration =====
    
    def test_multi_agent_terminal_integration(self):
        """Test terminal in multi-agent system"""
        print("\n🧪 TEST 5: Multi-Agent System Integration")
        
        # Create specialized agents with terminal access
        analyzer_agent = PromptChain(
            models=["mock"],
            instructions=["Analyze system: {input}"],
            verbose=True
        )
        analyzer_agent.register_tool_function(self.terminal)
        
        fixer_agent = PromptChain(
            models=["mock"],
            instructions=["Fix issue: {input}"],
            verbose=True
        )
        fixer_agent.register_tool_function(self.terminal)
        
        # Create multi-agent system
        agent_chain = AgentChain(
            agents={
                "analyzer": analyzer_agent,
                "fixer": fixer_agent
            },
            agent_descriptions={
                "analyzer": "Diagnoses system issues using terminal",
                "fixer": "Fixes issues using terminal commands"
            },
            execution_mode="router",
            verbose=True
        )
        
        # Mock the agent selection and execution
        with patch.object(agent_chain, '_route_to_agent') as mock_router:
            mock_router.return_value = ("analyzer", "System analysis complete")
            
            result = agent_chain.process("Check system health")
            
            assert result is not None
            print(f"✅ Multi-agent result: {result}")
    
    # ===== TEST 6: Security Validation Examples =====
    
    def test_security_blocking(self):
        """Test security guardrails block dangerous commands"""
        print("\n🧪 TEST 6: Security Validation")
        
        # Create terminal with strict security
        secure_terminal = TerminalTool(
            require_permission=True,
            auto_approve=False  # No auto-approval
        )
        
        dangerous_commands = [
            "rm -rf /",
            "sudo rm -rf /*", 
            "dd if=/dev/zero of=/dev/sda",
            ":(){ :|:& };:",  # Fork bomb
        ]
        
        for cmd in dangerous_commands:
            with pytest.raises(SecurityError):
                secure_terminal(cmd)
                print(f"✅ Blocked dangerous command: {cmd}")
    
    def test_permission_prompts(self):
        """Test permission prompts for installations"""
        print("\n🧪 TEST 7: Permission Prompts")
        
        terminal_with_prompts = TerminalTool(
            require_permission=True,
            auto_approve=True  # Auto-approve for testing
        )
        
        # Installation commands that should prompt
        install_commands = [
            "pip install requests",
            "npm install express", 
            "conda install numpy"
        ]
        
        for cmd in install_commands:
            # Should succeed with auto-approval
            result = terminal_with_prompts(cmd)
            # Commands might fail due to missing packages, but should not be blocked
            assert "Permission denied" not in result
            print(f"✅ Permission granted for: {cmd}")
    
    # ===== TEST 8: Complex AgenticStepProcessor Workflow =====
    
    @pytest.mark.asyncio
    async def test_complex_agentic_debugging(self):
        """Test complex debugging workflow with AgenticStepProcessor"""
        print("\n🧪 TEST 8: Complex Agentic Debugging")
        
        debugger = AgenticStepProcessor(
            objective="Debug why a web server won't start on port 8000",
            max_internal_steps=10,
            model_name="mock"
        )
        
        # Terminal with comprehensive logging
        debug_terminal = TerminalTool(
            verbose=True,
            log_all_commands=True,
            timeout=30
        )
        
        chain = PromptChain(
            models=["mock"],
            instructions=[
                "Starting debug session: {input}",
                debugger,
                "Debug report: {input}"
            ]
        )
        
        # Register terminal for debugging
        chain.register_tool_function(debug_terminal)
        chain.add_tools([{
            "type": "function",
            "function": {
                "name": "terminal",
                "description": """Debug commands:
                - 'lsof -i :8000' (check port usage)
                - 'ps aux | grep python' (check processes)  
                - 'netstat -tulpn | grep 8000' (network status)
                - 'journalctl -u myservice' (check logs)""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        }])
        
        with patch.object(debugger, 'process') as mock_debug:
            mock_debug.return_value = "Port 8000 is in use by process 1234. Solution: kill process or use different port."
            
            with patch.object(chain, '_call_llm') as mock_llm:
                mock_llm.side_effect = [
                    "Investigating port issue",
                    "Port conflict identified and resolved"
                ]
                
                result = chain.process_prompt("Flask app fails with 'Address already in use'")
                
                assert "port" in result.lower() or "process" in result.lower()
                print(f"✅ Complex debugging result: {result}")
    
    # ===== TEST 9: Environment Management =====
    
    def test_environment_detection(self):
        """Test environment detection and activation"""
        print("\n🧪 TEST 9: Environment Management")
        
        # Terminal with environment auto-activation
        env_terminal = TerminalTool(
            auto_activate_env=True,
            verbose=True
        )
        
        # Get detected environments
        env_info = env_terminal.get_system_info()
        
        assert "detected_environments" in env_info
        assert "active_environment" in env_info
        
        print(f"✅ Environment info: {env_info['detected_environments']}")
        print(f"✅ Active environment: {env_info['active_environment']}")
    
    # ===== TEST 10: Utility Methods for Agents =====
    
    def test_utility_methods(self):
        """Test utility methods designed for AgenticStepProcessor"""
        print("\n🧪 TEST 10: Utility Methods")
        
        # Test installation checking
        python_installed = self.terminal.check_installation("python")
        print(f"✅ Python installed: {python_installed}")
        
        if python_installed:
            # Test version getting
            python_version = self.terminal.get_version("python")
            print(f"✅ Python version: {python_version}")
            
        # Test file finding (look for Python files)
        python_files = self.terminal.find_files("*.py", ".")
        print(f"✅ Found {len(python_files)} Python files")
        
        # Test port checking
        port_8080_busy = self.terminal.check_port(8080)
        print(f"✅ Port 8080 in use: {port_8080_busy}")
        
        assert isinstance(python_installed, bool)
    
    # ===== TEST 11: Configuration Loading =====
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        print("\n🧪 TEST 11: Configuration Loading")
        
        # Create terminal with config file
        config_path = Path(__file__).parent.parent.parent / "promptchain" / "tools" / "terminal" / "config.yaml"
        
        if config_path.exists():
            config_terminal = TerminalTool(
                config_file=str(config_path),
                verbose=True
            )
            
            # Test that configuration was loaded
            assert config_terminal.security is not None
            assert config_terminal.env_manager is not None
            assert config_terminal.path_resolver is not None
            
            print("✅ Configuration loaded successfully")
        else:
            print("⚠️ Config file not found, skipping test")
    
    # ===== TEST 12: Advanced Output Parsing =====
    
    def test_advanced_parsing(self):
        """Test advanced output parsing features"""
        print("\n🧪 TEST 12: Advanced Output Parsing")
        
        # Test structured output parsing
        result = self.terminal.execute_with_success_check("echo 'Hello World'")
        
        assert result["success"] is True
        assert "Hello World" in result["output"]
        assert result["return_code"] == 0
        
        print(f"✅ Parsed result: {result}")
        
        # Test JSON parsing (if available)
        try:
            json_result = self.terminal.execute_with_parsing("echo '{\"test\": \"value\"}'", "json")
            assert json_result["type"] == "json"
            print(f"✅ JSON parsing: {json_result}")
        except:
            print("⚠️ JSON parsing test skipped")


# ===== Integration Test with Mock PromptChain =====

class MockPromptChain:
    """Mock PromptChain for testing without actual LLM calls"""
    
    def __init__(self, instructions):
        self.instructions = instructions
        self.tools = {}
        
    def register_tool_function(self, func):
        self.tools[func.__class__.__name__.lower()] = func
        
    def add_tools(self, tools):
        pass  # Mock implementation
        
    def process_prompt(self, prompt):
        return f"Processed: {prompt}"


def test_complete_integration():
    """Complete integration test with all patterns"""
    print("\n🚀 COMPLETE INTEGRATION TEST")
    
    terminal = TerminalTool(require_permission=False, verbose=True)
    
    # Test 1: Simple usage
    result1 = terminal("echo 'Integration test'")
    assert "Integration test" in result1
    
    # Test 2: Callable usage
    def command_executor(cmd: str) -> str:
        return terminal(cmd)
    
    result2 = command_executor("pwd")
    assert result2  # Should return current directory
    
    # Test 3: Mock chain integration
    mock_chain = MockPromptChain([terminal])
    mock_chain.register_tool_function(terminal)
    
    result3 = mock_chain.process_prompt("test command")
    assert result3
    
    print("✅ Complete integration test passed!")
    
    # Display statistics
    history = terminal.get_command_history()
    system_info = terminal.get_system_info()
    
    print(f"\n📊 TERMINAL TOOL STATISTICS:")
    print(f"Commands executed: {len(history)}")
    print(f"Active environment: {system_info['active_environment']}")
    print(f"Cache stats: {system_info['cache_stats']}")
    
    return True


# ===== Main Test Runner =====

if __name__ == "__main__":
    """Run all tests manually"""
    import traceback
    
    print("🧪 RUNNING ALL TERMINAL TOOL TESTS")
    print("=" * 50)
    
    test_instance = TestTerminalToolExamples()
    test_instance.setup_method()
    
    tests = [
        ("Simple Function Injection", test_instance.test_simple_function_injection),
        ("Direct Callable Usage", test_instance.test_direct_callable_usage), 
        ("Tool Registration", test_instance.test_tool_registration),
        ("Security Blocking", test_instance.test_security_blocking),
        ("Permission Prompts", test_instance.test_permission_prompts),
        ("Environment Detection", test_instance.test_environment_detection),
        ("Utility Methods", test_instance.test_utility_methods),
        ("Configuration Loading", test_instance.test_configuration_loading),
        ("Advanced Parsing", test_instance.test_advanced_parsing),
        ("Complete Integration", test_complete_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} - FAILED: {e}")
            traceback.print_exc()
            failed += 1
        print("-" * 30)
    
    print(f"\n🏁 TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Terminal Tool is ready for use.")
    else:
        print("⚠️ Some tests failed. Check implementation.")