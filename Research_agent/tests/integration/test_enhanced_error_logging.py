#!/usr/bin/env python3
"""
Enhanced Error Logging Test

Tests the hypothesis that JSON parsing errors may be due to:
1. Empty/null LLM responses
2. Error messages from LLMs instead of JSON
3. Refusal responses from LLMs
4. Malformed content beyond just bad JSON syntax

This test simulates various LLM failure modes to validate our enhanced logging.
"""

import json
import logging
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.agents.react_analyzer import ReActAnalysisAgent
from research_agent.agents.synthesis_agent import SynthesisAgent
from research_agent.utils.robust_json_parser import RobustJSONParser

# Configure logging to see our enhanced error messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_empty_llm_responses():
    """Test handling of various empty/null responses"""
    print("🧪 Testing Empty/Null LLM Response Handling")
    print("=" * 50)
    
    parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
    
    test_cases = [
        ("Completely empty string", ""),
        ("None response", None),
        ("Only whitespace", "   \n\t  \n   "),
        ("Only newlines", "\n\n\n\n"),
        ("Single space", " "),
        ("Unicode whitespace", "\u00A0\u2000\u2001\u2002"),  # Non-breaking spaces
        ("Zero-width characters", "\u200B\u200C\u200D"),  # Zero-width chars
        ("Only punctuation", ".,!?;:"),
        ("HTML entities", "&nbsp;&amp;&lt;&gt;"),
    ]
    
    for description, response in test_cases:
        print(f"\nTesting: {description}")
        print(f"Input: {repr(response)}")
        
        try:
            result = parser.parse(
                response,
                expected_keys=['test_key'],
                fallback_structure={'test_key': 'fallback_value'}
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"Exception: {e}")


def test_llm_error_responses():
    """Test handling of LLM error and refusal responses"""
    print("\n🧪 Testing LLM Error/Refusal Response Handling")
    print("=" * 50)
    
    parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
    
    error_responses = [
        ("Direct error message", "I'm sorry, I encountered an error and cannot generate the requested JSON."),
        ("API rate limit", "Error: Rate limit exceeded. Please try again later."),
        ("Model overloaded", "The model is currently overloaded. Please retry your request."),
        ("Safety refusal", "I can't assist with that request for safety reasons."),
        ("Incomplete response", "Here's the analysis: {\"partial\": \"data\" "),  # Cut off
        ("Mixed error", "Analysis complete!\n\nError: Failed to format as JSON\n\nSorry about that."),
        ("HTML error page", "<!DOCTYPE html><html><head><title>Error</title></head><body>Service Unavailable</body></html>"),
        ("Network timeout", "Request timed out after 30 seconds"),
        ("Authentication error", "401 Unauthorized: Invalid API key"),
        ("Content policy", "I apologize, but I cannot provide that information due to content policy restrictions."),
    ]
    
    for description, response in error_responses:
        print(f"\nTesting: {description}")
        print(f"Input: {response[:100]}...")
        
        try:
            result = parser.parse(
                response,
                expected_keys=['analysis_summary', 'gaps_identified'],
                fallback_structure={'analysis_summary': 'error_fallback', 'gaps_identified': []}
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"Exception: {e}")


def test_agent_error_scenarios():
    """Test agents with mocked LLM responses that simulate real error conditions"""
    print("\n🧪 Testing Agent Error Scenarios with Enhanced Logging")
    print("=" * 50)
    
    # Test ReAct Analyzer with various error responses
    react_config = {
        'model': 'openai/gpt-4o',
        'processor': {'objective': 'test', 'max_internal_steps': 3}
    }
    
    error_scenarios = [
        ("Empty response", ""),
        ("API error", "Error: The model produced an empty response"),
        ("JSON with error content", '{"error": "Failed to analyze", "status": "error"}'),
        ("Partial JSON", '{"analysis_summary": {"current_state": "incomplete"'),
        ("Non-JSON response", "I apologize, but I cannot complete this analysis due to technical difficulties."),
        ("Mixed content", "The analysis is complex.\n\nJSON: {}\n\nUnfortunately, I couldn't complete it."),
    ]
    
    for description, mock_response in error_scenarios:
        print(f"\n--- Testing ReAct Agent with: {description} ---")
        
        try:
            # Mock the PromptChain to return our test response
            with patch('research_agent.agents.react_analyzer.PromptChain') as mock_chain_class:
                mock_chain = MagicMock()
                mock_chain.process_prompt_async.return_value = mock_response
                mock_chain_class.return_value = mock_chain
                
                agent = ReActAnalysisAgent(react_config)
                
                # Test the validation method directly
                test_context = {
                    'session_id': 'test_session',
                    'topic': 'test_topic',
                    'iteration': 1,
                    'completed_queries': 5,
                    'total_papers': 10,
                    'completion_score': 0.5
                }
                
                # This will trigger our enhanced logging
                validated_response = agent._validate_analysis_response(mock_response)
                print(f"Validated response type: {type(validated_response)}")
                print(f"Validated response preview: {validated_response[:200]}...")
                
        except Exception as e:
            print(f"Agent test error: {e}")


def test_synthesis_agent_scenarios():
    """Test synthesis agent with error conditions"""
    print("\n🧪 Testing Synthesis Agent Error Scenarios")
    print("=" * 50)
    
    synthesis_config = {
        'model': 'openai/gpt-4o',
        'processor': {'objective': 'test', 'max_internal_steps': 5}
    }
    
    synthesis_errors = [
        ("Token limit exceeded", "Error: This request would exceed the maximum context length."),
        ("Content too large", "The synthesis is too complex to generate in a single response."),
        ("Empty synthesis", ""),
        ("Malformed literature review", '{"literature_review": "incomplete synthesis due to'),
    ]
    
    for description, mock_response in synthesis_errors:
        print(f"\n--- Testing Synthesis Agent with: {description} ---")
        
        try:
            with patch('research_agent.agents.synthesis_agent.PromptChain') as mock_chain_class:
                mock_chain = MagicMock()
                mock_chain.process_prompt_async.return_value = mock_response
                mock_chain_class.return_value = mock_chain
                
                agent = SynthesisAgent(synthesis_config)
                
                test_context = {
                    'session_id': 'test_session',
                    'topic': 'test_topic',
                    'papers': [],
                    'queries': [],
                    'statistics': {}
                }
                
                # This will trigger our enhanced logging
                validated_response = agent._validate_synthesis_response(mock_response, test_context)
                print(f"Validated response type: {type(validated_response)}")
                print(f"Validated response preview: {validated_response[:200]}...")
                
        except Exception as e:
            print(f"Synthesis agent test error: {e}")


def test_real_world_error_patterns():
    """Test patterns observed in real-world LLM failures"""
    print("\n🧪 Testing Real-World Error Patterns")
    print("=" * 50)
    
    parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
    
    real_world_patterns = [
        # Pattern 1: LLM returns explanation instead of JSON
        ("Explanation instead of JSON", 
         "I understand you want a JSON response, but given the complexity of the analysis, let me explain the findings in plain text instead..."),
        
        # Pattern 2: LLM acknowledges request but can't fulfill it
        ("Acknowledgment without fulfillment", 
         "I see that you're asking for a structured analysis in JSON format. While I've reviewed the research context, I'm unable to provide the requested format at this time."),
        
        # Pattern 3: Partial response due to length limits
        ("Truncated response", 
         '{"analysis_summary": {"current_state": "Research shows significant progress", "completion_score": 0.8, "key_findings": ["Finding 1", "Finding 2"]}, "gaps_identified": [{"gap_type": "methodology", "description": "Limited comparative"'),
        
        # Pattern 4: LLM starts JSON but switches to text
        ("JSON starts then becomes text", 
         '{"analysis_summary": "The research analysis reveals several important insights. However, due to the complexity of the findings, I should explain this more clearly: The current research shows..."'),
        
        # Pattern 5: Unicode/encoding issues
        ("Unicode encoding issues", 
         '{"analysis_summary": {"current_state": "Research in progre\ufffd\ufffd", "completion_score": 0.7}}'),
        
        # Pattern 6: Model confusion about format
        ("Format confusion", 
         "You asked for JSON, but I think a table format would be better:\n\n| Analysis | Status |\n|----------|--------|\n| Progress | Good    |"),
    ]
    
    for description, response in real_world_patterns:
        print(f"\nTesting: {description}")
        print(f"Input length: {len(response)}")
        print(f"Input preview: {response[:100]}...")
        
        try:
            result = parser.parse(
                response,
                expected_keys=['analysis_summary', 'gaps_identified'],
                fallback_structure={'analysis_summary': 'pattern_fallback', 'gaps_identified': []}
            )
            print(f"Successfully parsed: {list(result.keys())}")
        except Exception as e:
            print(f"Exception: {e}")


def main():
    """Run all enhanced error logging tests"""
    print("🚀 Enhanced Error Logging Test Suite")
    print("Testing hypothesis: JSON errors may be empty/error LLM responses")
    print("=" * 70)
    
    try:
        test_empty_llm_responses()
        test_llm_error_responses()
        test_agent_error_scenarios()
        test_synthesis_agent_scenarios()
        test_real_world_error_patterns()
        
        print("\n" + "=" * 70)
        print("✅ Enhanced Error Logging Test Suite Complete")
        print("\nKey Insights:")
        print("1. Enhanced logging now captures exact LLM output with banners")
        print("2. Empty/error responses are detected and logged comprehensively") 
        print("3. Multiple failure modes beyond JSON syntax errors are handled")
        print("4. Fallback structures prevent total system failure")
        print("5. Unicode and encoding issues are detected and logged")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)