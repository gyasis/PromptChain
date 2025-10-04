#!/usr/bin/env python3
"""
Production Error Capture Test

This test directly exercises the enhanced logging banners to capture
exact LLM outputs that would cause JSON parsing errors in production.
"""

import asyncio
import logging
import sys
import os
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.agents.react_analyzer import ReActAnalysisAgent
from research_agent.agents.synthesis_agent import SynthesisAgent
from research_agent.utils.robust_json_parser import RobustJSONParser

# Configure logging with file output for human engineers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_error_banners.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)


def create_problematic_llm_responses():
    """Create realistic LLM responses that could cause JSON errors in production"""
    return {
        # Real-world error patterns observed in production
        "empty_response": "",
        "whitespace_only": "   \n\t  \n   ",
        "api_error": "Error: The model is currently overloaded. Please try again later.",
        "context_limit": "I apologize, but your request would exceed the maximum context length. Please try a shorter prompt.",
        "safety_refusal": "I can't assist with generating JSON for this type of analysis due to safety guidelines.",
        "partial_cutoff": '{"analysis_summary": {"current_state": "Research shows significant progress in machine learning", "completion_score": 0.8}, "gaps_identified": [{"gap_type": "methodology", "description": "Limited comparative"',
        "mixed_content": 'Based on my analysis, here are the findings:\n\n{"analysis_summary": "mixed", "gaps": []} \n\nHowever, I should note that this analysis is preliminary.',
        "explanation_instead": "I understand you want a JSON response, but given the complexity of this analysis, let me explain the findings in a more readable format instead of structured JSON.",
        "format_confusion": "You asked for JSON, but I think a table would be better:\n\n| Analysis | Status |\n|----------|--------|\n| Progress | Good   |",
        "unicode_issues": '{"analysis": "Research progre\ufffd\ufffd in AI", "score": 0.7}',
        "malformed_json": '{"analysis_summary": "good progress", "gaps_identified": [missing_bracket, "completion_score": 0.8}',
        "nested_explanation": '{"analysis_summary": {"current_state": "I need to explain that this research is quite complex and involves many factors that are difficult to summarize in a simple JSON structure, so let me provide a more detailed explanation..."}, "gaps_identified": []}',
        "error_in_json": '{"error": "Failed to complete analysis", "status": "error", "message": "Unable to process the research context"}',
        "html_response": "<!DOCTYPE html><html><head><title>Service Unavailable</title></head><body><h1>503 Service Unavailable</h1><p>The server is temporarily unable to service your request.</p></body></html>",
        "non_english": "抱歉，我无法为此分析生成JSON格式的响应。请稍后再试。",
    }


async def test_react_analyzer_error_banners():
    """Test ReAct analyzer with problematic responses to capture error banners"""
    print("🧪 TESTING REACT ANALYZER ERROR BANNERS")
    print("=" * 50)
    
    config = {
        'model': 'openai/gpt-4o',
        'processor': {'objective': 'test', 'max_internal_steps': 3}
    }
    
    problematic_responses = create_problematic_llm_responses()
    
    for description, mock_response in problematic_responses.items():
        print(f"\n--- Testing: {description} ---")
        logger.info(f"🎯 TESTING REACT ANALYZER WITH: {description}")
        
        try:
            with patch('research_agent.agents.react_analyzer.PromptChain') as mock_chain_class:
                mock_chain = MagicMock()
                mock_chain.process_prompt_async.return_value = mock_response
                mock_chain_class.return_value = mock_chain
                
                agent = ReActAnalysisAgent(config)
                
                # This will trigger our enhanced error banners
                context = {
                    'session_id': 'test_session',
                    'topic': 'machine learning',
                    'iteration': 1,
                    'completed_queries': 5,
                    'total_papers': 10,
                    'completion_score': 0.5
                }
                
                result = await agent.analyze_research_progress(context=context)
                print(f"   Result type: {type(result)}")
                print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                
        except Exception as e:
            logger.error(f"🔥 CRITICAL ERROR in ReAct test {description}: {e}")
            print(f"   ❌ Critical error: {e}")


async def test_synthesis_agent_error_banners():
    """Test Synthesis agent with problematic responses to capture error banners"""
    print(f"\n🧪 TESTING SYNTHESIS AGENT ERROR BANNERS")
    print("=" * 50)
    
    config = {
        'model': 'openai/gpt-4o',
        'processor': {'objective': 'test', 'max_internal_steps': 5}
    }
    
    problematic_responses = create_problematic_llm_responses()
    
    # Test subset for synthesis agent
    synthesis_responses = {
        'empty_response': problematic_responses['empty_response'],
        'api_error': problematic_responses['api_error'],
        'partial_cutoff': problematic_responses['partial_cutoff'],
        'unicode_issues': problematic_responses['unicode_issues'],
        'explanation_instead': problematic_responses['explanation_instead'],
    }
    
    for description, mock_response in synthesis_responses.items():
        print(f"\n--- Testing: {description} ---")
        logger.info(f"🎯 TESTING SYNTHESIS AGENT WITH: {description}")
        
        try:
            with patch('research_agent.agents.synthesis_agent.PromptChain') as mock_chain_class:
                mock_chain = MagicMock()
                mock_chain.process_prompt_async.return_value = mock_response
                mock_chain_class.return_value = mock_chain
                
                agent = SynthesisAgent(config)
                
                context = {
                    'session_id': 'test_session',
                    'topic': 'machine learning',
                    'papers': [],
                    'queries': [],
                    'statistics': {}
                }
                
                result = await agent.synthesize_literature_review(context=context)
                print(f"   Result type: {type(result)}")
                print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                
        except Exception as e:
            logger.error(f"🔥 CRITICAL ERROR in Synthesis test {description}: {e}")
            print(f"   ❌ Critical error: {e}")


def test_robust_parser_banners():
    """Test RobustJSONParser directly to capture comprehensive error banners"""
    print(f"\n🧪 TESTING ROBUST JSON PARSER BANNERS")
    print("=" * 50)
    
    parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
    problematic_responses = create_problematic_llm_responses()
    
    for description, response in problematic_responses.items():
        print(f"\n--- Testing: {description} ---")
        logger.info(f"🎯 TESTING ROBUST PARSER WITH: {description}")
        
        try:
            result = parser.parse(
                response,
                expected_keys=['analysis_summary', 'gaps_identified'],
                fallback_structure={'analysis_summary': 'fallback', 'gaps_identified': []}
            )
            print(f"   Parsed successfully: {list(result.keys())}")
            
        except Exception as e:
            logger.error(f"🔥 PARSER ERROR with {description}: {e}")
            print(f"   ❌ Parser error: {e}")


def main():
    """Run production error capture tests"""
    print("🚀 PRODUCTION ERROR BANNER CAPTURE TEST")
    print("Capturing exact LLM outputs that cause JSON parsing errors")
    print("Human engineers can use these banners for debugging")
    print("=" * 70)
    
    try:
        # Test direct parser banners
        test_robust_parser_banners()
        
        # Test agent error banners
        asyncio.run(test_react_analyzer_error_banners())
        asyncio.run(test_synthesis_agent_error_banners())
        
        print(f"\n{'='*70}")
        print("✅ PRODUCTION ERROR CAPTURE COMPLETE")
        print(f"\n📁 CHECK 'production_error_banners.log' FOR:")
        print("  🔥 Error banners with full LLM response details")
        print("  🗝 Empty/null response detection banners")
        print("  === Full response capture sections")
        print("  ⚠️ Warning pattern detection (errors, apologies)")
        print("  📊 Response analysis (length, type, content)")
        print(f"\n🛠️ ENHANCED LOGGING NOW ACTIVE IN PRODUCTION:")
        print("  • react_analyzer.py - lines 408-431 & 482-499")
        print("  • synthesis_agent.py - lines 545-575 & 587-605")
        print("  • robust_json_parser.py - lines 60-75 & 128-153")
        print(f"\n👥 HUMAN ENGINEERS: Use these exact banners to debug JSON errors!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    # Show log file location prominently
    log_file = os.path.abspath('production_error_banners.log')
    print(f"\n📋 DEBUGGING LOG FILE: {log_file}")
    print("This file contains all the error banners for human debugging!")
    
    sys.exit(0 if success else 1)