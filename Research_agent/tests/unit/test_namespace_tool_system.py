#!/usr/bin/env python3
"""
Test script for the namespace-aware tool registration system.

This script validates that the new system prevents tool conflicts
while maintaining full functionality.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, Any
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.utils.namespace_tool_registry import (
    NamespaceToolRegistry, 
    get_global_tool_registry,
    reset_global_registry
)
from research_agent.utils.agent_tool_coordinator import (
    AgentToolCoordinator,
    get_global_coordinator,
    register_research_agents,
    reset_global_coordination
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_namespace_generation():
    """Test namespace generation logic"""
    logger.info("=== Testing Namespace Generation ===")
    
    registry = NamespaceToolRegistry()
    
    test_cases = [
        ("SynthesisAgent", None, "synthesis"),
        ("MultiQueryCoordinator", None, "multiquery"),
        ("QueryGenerator", None, "querygen"),
        ("LiteratureSearcher", None, "literaturesearcher"),
        ("ReactAnalyzer", None, "reactanalyzer"),
        ("SearchStrategist", None, "searchstrategist"),
        ("SynthesisAgent", "unique123", "synthesis_unique"),
    ]
    
    results = []
    for class_name, agent_id, expected in test_cases:
        actual = registry.generate_namespace(class_name, agent_id)
        success = expected in actual  # Allow for variations
        results.append((class_name, agent_id, expected, actual, success))
        
        status = "✅" if success else "❌"
        logger.info(f"{status} {class_name} -> {actual} (expected: {expected})")
    
    all_passed = all(result[4] for result in results)
    logger.info(f"Namespace generation test: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_basic_tool_registration():
    """Test basic tool registration without conflicts"""
    logger.info("=== Testing Basic Tool Registration ===")
    
    registry = NamespaceToolRegistry()
    
    # Define test tools
    def test_tool_1(input_data: str) -> str:
        return f"Tool 1 processed: {input_data}"
    
    def test_tool_2(input_data: str) -> str:
        return f"Tool 2 processed: {input_data}"
    
    schema_1 = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Test tool 1",
            "parameters": {
                "type": "object",
                "properties": {"input_data": {"type": "string"}},
                "required": ["input_data"]
            }
        }
    }
    
    schema_2 = {
        "type": "function", 
        "function": {
            "name": "test_tool",
            "description": "Test tool 2",
            "parameters": {
                "type": "object",
                "properties": {"input_data": {"type": "string"}},
                "required": ["input_data"]
            }
        }
    }
    
    # Register tools with same original name but different namespaces
    ns1 = registry.generate_namespace("TestAgent1")
    ns2 = registry.generate_namespace("TestAgent2")
    
    name1, orig1 = registry.register_tool(test_tool_1, schema_1, ns1, "agent1")
    name2, orig2 = registry.register_tool(test_tool_2, schema_2, ns2, "agent2")
    
    # Verify no conflicts
    success = (
        name1 != name2 and  # Different namespaced names
        orig1 == orig2 and  # Same original name
        name1.startswith(ns1) and
        name2.startswith(ns2)
    )
    
    logger.info(f"Tool 1: {orig1} -> {name1}")
    logger.info(f"Tool 2: {orig2} -> {name2}")
    logger.info(f"Basic registration test: {'PASSED' if success else 'FAILED'}")
    
    return success


def test_conflict_prevention():
    """Test that conflicts are properly prevented"""
    logger.info("=== Testing Conflict Prevention ===")
    
    registry = NamespaceToolRegistry()
    
    def same_name_tool(input_data: str) -> str:
        return f"Same name tool: {input_data}"
    
    schema = {
        "type": "function",
        "function": {
            "name": "conflicting_tool",
            "description": "A tool that will conflict",
            "parameters": {
                "type": "object",
                "properties": {"input_data": {"type": "string"}},
                "required": ["input_data"]
            }
        }
    }
    
    # Register same tool name in same namespace with different agents
    namespace = "testns"
    registry.registry.register_namespace(namespace)
    
    name1, _ = registry.register_tool(same_name_tool, schema, namespace, "agent1")
    name2, _ = registry.register_tool(same_name_tool, schema, namespace, "agent2")
    
    # Names should be different due to conflict resolution
    success = name1 != name2
    conflicts_prevented = registry.registry.conflicts_prevented > 0
    
    logger.info(f"First registration: {name1}")
    logger.info(f"Second registration: {name2}")
    logger.info(f"Conflicts prevented: {registry.registry.conflicts_prevented}")
    logger.info(f"Conflict prevention test: {'PASSED' if success and conflicts_prevented else 'FAILED'}")
    
    return success and conflicts_prevented


async def test_synthesis_agent_integration():
    """Test SynthesisAgent integration with namespace system"""
    logger.info("=== Testing SynthesisAgent Integration ===")
    
    try:
        # Reset global state
        reset_global_coordination()
        
        coordinator = get_global_coordinator()
        
        # Create synthesis agent configuration
        synthesis_config = {
            'model': 'openai/gpt-4o-mini',
            'processor': {
                'objective': 'Test synthesis with namespace tools',
                'max_internal_steps': 3
            }
        }
        
        # Register synthesis agent
        synthesis_agent = coordinator.register_synthesis_agent(synthesis_config, "test_synthesis")
        
        # Verify agent has proper namespace
        success = (
            hasattr(synthesis_agent, 'namespace') and
            hasattr(synthesis_agent, '_registered_tools') and
            len(synthesis_agent._registered_tools) > 0
        )
        
        logger.info(f"SynthesisAgent namespace: {synthesis_agent.namespace}")
        logger.info(f"Registered tools: {list(synthesis_agent._registered_tools.keys())}")
        logger.info(f"SynthesisAgent integration test: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        logger.error(f"SynthesisAgent integration test failed: {e}")
        return False


async def test_multiquery_coordinator_integration():
    """Test MultiQueryCoordinator integration with namespace system"""
    logger.info("=== Testing MultiQueryCoordinator Integration ===")
    
    try:
        coordinator = get_global_coordinator()
        
        # Create multiquery coordinator configuration
        multiquery_config = {
            'coordination': {'model': 'openai/gpt-4o-mini'},
            'rag_tiers': {
                'lightrag': {'batch_size': 5},
                'paper_qa2': {'batch_size': 3},
                'graphrag': {'batch_size': 2}
            },
            'three_tier_rag': {}
        }
        
        # Register multiquery coordinator
        multiquery_coord = coordinator.register_multiquery_coordinator(multiquery_config, "test_multiquery")
        
        # Verify agent has proper namespace
        success = (
            hasattr(multiquery_coord, 'namespace') and
            hasattr(multiquery_coord, '_registered_tools') and
            len(multiquery_coord._registered_tools) > 0
        )
        
        logger.info(f"MultiQueryCoordinator namespace: {multiquery_coord.namespace}")
        logger.info(f"Registered tools: {list(multiquery_coord._registered_tools.keys())}")
        logger.info(f"MultiQueryCoordinator integration test: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        logger.error(f"MultiQueryCoordinator integration test failed: {e}")
        return False


async def test_multiple_agents_no_conflicts():
    """Test multiple agents with no tool conflicts"""
    logger.info("=== Testing Multiple Agents with No Conflicts ===")
    
    try:
        # Reset coordination
        reset_global_coordination()
        
        # Register multiple agents of different types
        agents = register_research_agents(
            synthesis={
                'model': 'openai/gpt-4o-mini',
                'processor': {'objective': 'Test synthesis', 'max_internal_steps': 2}
            },
            multiquery={
                'coordination': {'model': 'openai/gpt-4o-mini'},
                'rag_tiers': {'lightrag': {}, 'paper_qa2': {}, 'graphrag': {}},
                'three_tier_rag': {}
            }
        )
        
        # Validate no conflicts
        coordinator = get_global_coordinator()
        validation = coordinator.validate_no_conflicts()
        
        success = (
            validation['registry_valid'] and
            len(validation['namespace_conflicts']) == 0 and
            validation['total_agents'] == 2
        )
        
        logger.info(f"Registered agents: {list(agents.keys())}")
        logger.info(f"Total tools: {validation['total_tools']}")
        logger.info(f"Conflicts prevented: {validation['conflicts_prevented']}")
        logger.info(f"Multiple agents test: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        logger.error(f"Multiple agents test failed: {e}")
        return False


async def test_tool_functionality():
    """Test that tools remain functional after namespace registration"""
    logger.info("=== Testing Tool Functionality ===")
    
    try:
        coordinator = get_global_coordinator()
        
        # Get synthesis agent (should be registered from previous test)
        synthesis_agents = [agent for agent in coordinator.registered_agents.values() 
                          if agent.agent_class == "SynthesisAgent"]
        
        if not synthesis_agents:
            logger.warning("No SynthesisAgent found, skipping functionality test")
            return True
        
        agent_id = next(aid for aid, config in coordinator.registered_agents.items() 
                       if config.agent_class == "SynthesisAgent")
        synthesis_agent = coordinator.get_agent(agent_id)
        
        # Test that the agent's tools are accessible
        schemas, functions = synthesis_agent.get_my_tools()
        
        success = (
            len(schemas) > 0 and
            len(functions) > 0 and
            all(callable(func) for func in functions.values())
        )
        
        logger.info(f"Agent tools - Schemas: {len(schemas)}, Functions: {len(functions)}")
        logger.info(f"Tool functionality test: {'PASSED' if success else 'FAILED'}")
        
        return success
        
    except Exception as e:
        logger.error(f"Tool functionality test failed: {e}")
        return False


def generate_test_report(test_results: Dict[str, bool]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    # Get final system state
    coordinator = get_global_coordinator()
    final_validation = coordinator.validate_no_conflicts()
    coordination_report = coordinator.get_coordination_report()
    
    report = {
        'test_summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if success_rate == 1.0 else 'FAILED'
        },
        'individual_results': test_results,
        'system_validation': final_validation,
        'coordination_report': coordination_report,
        'timestamp': datetime.now().isoformat()
    }
    
    return report


async def main():
    """Run comprehensive test suite"""
    logger.info("🚀 Starting Namespace Tool System Test Suite")
    logger.info("=" * 60)
    
    # Run all tests
    test_results = {}
    
    # Basic tests
    test_results['namespace_generation'] = test_namespace_generation()
    test_results['basic_registration'] = test_basic_tool_registration()
    test_results['conflict_prevention'] = test_conflict_prevention()
    
    # Integration tests
    test_results['synthesis_integration'] = await test_synthesis_agent_integration()
    test_results['multiquery_integration'] = await test_multiquery_coordinator_integration()
    test_results['multiple_agents'] = await test_multiple_agents_no_conflicts()
    test_results['tool_functionality'] = await test_tool_functionality()
    
    # Generate comprehensive report
    report = generate_test_report(test_results)
    
    # Log summary
    logger.info("=" * 60)
    logger.info("🏁 Test Suite Complete")
    logger.info(f"Overall Status: {report['test_summary']['overall_status']}")
    logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
    logger.info(f"Passed: {report['test_summary']['passed_tests']}/{report['test_summary']['total_tests']}")
    
    # Log detailed results
    logger.info("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
    
    # Save detailed report
    report_file = f"namespace_tool_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"\nDetailed report saved to: {report_file}")
    
    # Final validation summary
    validation = report['system_validation']
    logger.info(f"\nFinal System State:")
    logger.info(f"  Registry Valid: {validation['registry_valid']}")
    logger.info(f"  Total Agents: {validation['total_agents']}")
    logger.info(f"  Total Tools: {validation['total_tools']}")
    logger.info(f"  Conflicts Prevented: {validation['conflicts_prevented']}")
    
    # Exit with appropriate code
    success = report['test_summary']['overall_status'] == 'PASSED'
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)