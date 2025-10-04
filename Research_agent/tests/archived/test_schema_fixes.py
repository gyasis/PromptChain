#!/usr/bin/env python3

"""
Test that all agents initialize without tool schema warnings
"""

import logging
import sys
from io import StringIO

def capture_warnings(agent_init_func):
    """Capture warnings during agent initialization"""
    
    # Capture stderr to catch warnings
    old_stderr = sys.stderr
    captured_stderr = StringIO()
    sys.stderr = captured_stderr
    
    # Capture log output
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # Initialize agent
        agent = agent_init_func()
        
        # Get captured output
        stderr_output = captured_stderr.getvalue()
        log_output = log_capture.getvalue()
        
        return agent, stderr_output, log_output
        
    finally:
        # Restore stderr
        sys.stderr = old_stderr
        logging.getLogger().removeHandler(handler)

def test_react_analyzer_schemas():
    """Test ReAct analyzer schema initialization"""
    
    print("Testing ReAct Analyzer tool schemas:")
    
    def init_agent():
        from src.research_agent.agents.react_analyzer import ReActAnalysisAgent
        return ReActAnalysisAgent({'model': 'openai/gpt-4o-mini'})
    
    agent, stderr_output, log_output = capture_warnings(init_agent)
    
    # Check for schema-related warnings
    schema_warnings = []
    all_output = stderr_output + log_output
    
    for line in all_output.split('\n'):
        if ('schema' in line.lower() and 'warning' in line.lower()) or \
           ('tool' in line.lower() and 'warning' in line.lower() and 'schema' in line.lower()):
            schema_warnings.append(line.strip())
    
    if schema_warnings:
        print(f"  ✗ FAILED: Found {len(schema_warnings)} schema warnings:")
        for warning in schema_warnings[:3]:  # Show first 3
            print(f"    - {warning}")
        return False
    else:
        print("  ✓ SUCCESS: No schema warnings")
        return True

def test_search_strategist_schemas():
    """Test Search Strategist schema initialization"""
    
    print("\nTesting Search Strategist tool schemas:")
    
    def init_agent():
        from src.research_agent.agents.search_strategist import SearchStrategistAgent
        return SearchStrategistAgent({'model': 'openai/gpt-4o-mini'})
    
    agent, stderr_output, log_output = capture_warnings(init_agent)
    
    # Check for schema-related warnings
    schema_warnings = []
    all_output = stderr_output + log_output
    
    for line in all_output.split('\n'):
        if ('schema' in line.lower() and 'warning' in line.lower()) or \
           ('tool' in line.lower() and 'warning' in line.lower() and 'schema' in line.lower()):
            schema_warnings.append(line.strip())
    
    if schema_warnings:
        print(f"  ✗ FAILED: Found {len(schema_warnings)} schema warnings:")
        for warning in schema_warnings[:3]:
            print(f"    - {warning}")
        return False
    else:
        print("  ✓ SUCCESS: No schema warnings")
        return True

def test_query_generator_schemas():
    """Test Query Generator schema initialization"""
    
    print("\nTesting Query Generator tool schemas:")
    
    def init_agent():
        from src.research_agent.agents.query_generator import QueryGenerationAgent
        return QueryGenerationAgent({'model': 'openai/gpt-4o-mini'})
    
    agent, stderr_output, log_output = capture_warnings(init_agent)
    
    # Check for schema-related warnings
    schema_warnings = []
    all_output = stderr_output + log_output
    
    for line in all_output.split('\n'):
        if ('schema' in line.lower() and 'warning' in line.lower()) or \
           ('tool' in line.lower() and 'warning' in line.lower() and 'schema' in line.lower()):
            schema_warnings.append(line.strip())
    
    if schema_warnings:
        print(f"  ✗ FAILED: Found {len(schema_warnings)} schema warnings:")
        for warning in schema_warnings[:3]:
            print(f"    - {warning}")
        return False
    else:
        print("  ✓ SUCCESS: No schema warnings")
        return True

def test_all_agents_together():
    """Test initializing all agents together"""
    
    print("\nTesting all agents together:")
    
    try:
        from src.research_agent.agents.react_analyzer import ReActAnalysisAgent
        from src.research_agent.agents.search_strategist import SearchStrategistAgent
        from src.research_agent.agents.query_generator import QueryGenerationAgent
        
        config = {'model': 'openai/gpt-4o-mini'}
        
        react_agent = ReActAnalysisAgent(config)
        strategist_agent = SearchStrategistAgent(config)
        query_agent = QueryGenerationAgent(config)
        
        print("  ✓ SUCCESS: All agents initialized without errors")
        print(f"    - ReAct analyzer tools: {len(react_agent.chain.local_tools)} tools")
        print(f"    - Search strategist tools: {len(strategist_agent.chain.local_tools)} tools")
        print(f"    - Query generator tools: {len(query_agent.chain.local_tools)} tools")
        
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: Agent initialization failed - {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PROMPTCHAIN TOOL SCHEMA FIXES")
    print("=" * 60)
    
    # Run individual tests
    test1_passed = test_react_analyzer_schemas()
    test2_passed = test_search_strategist_schemas()
    test3_passed = test_query_generator_schemas()
    test4_passed = test_all_agents_together()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed and test3_passed and test4_passed:
        print("ALL TESTS PASSED ✓")
        print("PromptChain tool schemas are now properly configured!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Tool schema issues still exist.")
    print("=" * 60)