#!/usr/bin/env python3
"""
Test script to verify JSON parsing fixes for Research Agent system
Tests the specific error patterns that were causing failures.
"""

import sys
import os
import json
import logging
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.utils.robust_json_parser import RobustJSONParser, parse_agent_response
from research_agent.core.orchestrator import AdvancedResearchOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_malformed_responses():
    """Test various malformed response patterns"""
    print("🧪 Testing malformed response patterns...")
    
    parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
    
    # Test cases that were causing failures
    test_cases = [
        # Case 1: Empty response (line 1 column 1 error)
        {
            'name': 'Empty response',
            'response': '',
            'expected_keys': ['primary_queries']
        },
        
        # Case 2: Line-split JSON (from logs)
        {
            'name': 'Line-split JSON',
            'response': '{\n"primary_queries": [],\n"secondary_queries": [],\n"exploratory_queries": []\n}',
            'expected_keys': ['primary_queries', 'secondary_queries', 'exploratory_queries']
        },
        
        # Case 3: Individual lines (actual error from logs)
        {
            'name': 'Individual lines',
            'response': '{\n"primary_queries": [],\n"secondary_queries": [],\n"exploratory_queries": []\n}',
            'expected_keys': ['primary_queries']
        },
        
        # Case 4: Whitespace only
        {
            'name': 'Whitespace only',
            'response': '   \n\t  \n   ',
            'expected_keys': ['search_strategy']
        },
        
        # Case 5: JSON with explanatory text
        {
            'name': 'JSON with text',
            'response': '''Here's the analysis result:
            {
                "primary_queries": [
                    {"text": "What is quantum computing?", "priority": 1.0, "category": "general"}
                ],
                "secondary_queries": [],
                "exploratory_queries": []
            }
            This completes the query generation.''',
            'expected_keys': ['primary_queries']
        },
        
        # Case 6: Malformed quotes
        {
            'name': 'Malformed quotes',
            'response': "{'primary_queries': ['What is AI?'], 'secondary_queries': []}",
            'expected_keys': ['primary_queries']
        },
        
        # Case 7: Completely broken
        {
            'name': 'Completely broken',
            'response': 'This is not JSON at all, just random text',
            'expected_keys': ['primary_queries']
        }
    ]
    
    results = []
    for test_case in test_cases:
        try:
            print(f"\n  Testing: {test_case['name']}")
            print(f"  Input: {repr(test_case['response'][:50])}")
            
            fallback_structure = {}
            for key in test_case['expected_keys']:
                if 'queries' in key:
                    fallback_structure[key] = []
                else:
                    fallback_structure[key] = {}
            
            result = parser.parse(
                test_case['response'],
                expected_keys=test_case['expected_keys'],
                fallback_structure=fallback_structure
            )
            
            print(f"  ✅ Success: {type(result)} with keys: {list(result.keys())}")
            results.append({'name': test_case['name'], 'status': 'PASS', 'result': result})
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({'name': test_case['name'], 'status': 'FAIL', 'error': str(e)})
    
    return results

def test_orchestrator_query_parsing():
    """Test the orchestrator's query parsing specifically"""
    print("\n🧪 Testing orchestrator query parsing...")
    
    try:
        from research_agent.core.config import ResearchConfig
        config = ResearchConfig()
    except Exception:
        # Create minimal config for testing
        config = type('Config', (), {
            'get': lambda self, key, default=None: default,
            'get_agent_config': lambda self, agent: {}
        })()
    
    try:
        orchestrator = AdvancedResearchOrchestrator(config)
        
        # Test cases that caused the line-splitting issue
        test_responses = [
            # Empty response
            '',
            
            # Line-split response from logs
            '{\n"primary_queries": [],\n"secondary_queries": [],\n"exploratory_queries": []\n}',
            
            # Valid JSON with queries
            '''{
                "primary_queries": [
                    {"text": "What is quantum computing?", "priority": 1.0, "category": "general"}
                ],
                "secondary_queries": [
                    {"text": "How does quantum computing work?", "priority": 0.8, "category": "technical"}
                ],
                "exploratory_queries": []
            }''',
            
            # Broken JSON
            'This is not JSON'
        ]
        
        results = []
        for i, response in enumerate(test_responses):
            try:
                print(f"\n  Testing response {i+1}: {repr(response[:50])}")
                queries = orchestrator._parse_query_response(response)
                print(f"  ✅ Parsed {len(queries)} queries")
                
                # Validate query structure
                for j, query in enumerate(queries[:3]):  # Show first 3
                    print(f"    Query {j+1}: {query.get('text', 'NO TEXT')[:50]}")
                
                results.append({'response_id': i+1, 'status': 'PASS', 'query_count': len(queries)})
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append({'response_id': i+1, 'status': 'FAIL', 'error': str(e)})
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return [{'status': 'FAIL', 'error': f"Init error: {e}"}]

def test_agent_response_parsing():
    """Test the parse_agent_response utility function"""
    print("\n🧪 Testing agent response parsing utility...")
    
    test_cases = [
        {
            'name': 'Query Generator Empty',
            'response': '',
            'agent_type': 'query_generator'
        },
        {
            'name': 'Search Strategist Empty', 
            'response': '',
            'agent_type': 'search_strategist'
        },
        {
            'name': 'Query Generator Valid',
            'response': '{"primary_queries": [{"text": "Test query", "priority": 1.0}]}',
            'agent_type': 'query_generator'
        }
    ]
    
    results = []
    for test_case in test_cases:
        try:
            print(f"\n  Testing: {test_case['name']}")
            result = parse_agent_response(
                test_case['response'], 
                test_case['agent_type']
            )
            print(f"  ✅ Success: {list(result.keys())}")
            results.append({'name': test_case['name'], 'status': 'PASS', 'keys': list(result.keys())})
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({'name': test_case['name'], 'status': 'FAIL', 'error': str(e)})
    
    return results

def main():
    """Run all tests"""
    print("🔧 Testing JSON Parsing Fixes for Research Agent System")
    print("=" * 60)
    
    all_results = {}
    
    # Test 1: Malformed responses
    all_results['malformed_responses'] = test_malformed_responses()
    
    # Test 2: Orchestrator query parsing
    all_results['orchestrator_parsing'] = test_orchestrator_query_parsing()
    
    # Test 3: Agent response parsing
    all_results['agent_response_parsing'] = test_agent_response_parsing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    total_passed = 0
    
    for category, results in all_results.items():
        if isinstance(results, list):
            category_passed = sum(1 for r in results if r.get('status') == 'PASS')
            category_total = len(results)
            
            print(f"\n{category.replace('_', ' ').title()}: {category_passed}/{category_total} passed")
            
            for result in results:
                status = "✅" if result.get('status') == 'PASS' else "❌"
                name = result.get('name', result.get('response_id', 'Unknown'))
                print(f"  {status} {name}")
                if result.get('error'):
                    print(f"      Error: {result['error']}")
            
            total_tests += category_total
            total_passed += category_passed
    
    print(f"\n🎯 OVERALL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! JSON parsing fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)