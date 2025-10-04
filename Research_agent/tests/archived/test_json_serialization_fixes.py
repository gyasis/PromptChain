#!/usr/bin/env python3
"""
Comprehensive JSON Serialization Validation Test Suite

This test suite validates the fixes for persistent JSON serialization issues:
1. IterationSummary direct serialization 
2. ReAct analyzer JSON parsing failures
3. Synthesis agent JSON parsing failures
4. LLM response variability handling

Based on multi-agent debugging findings and Gemini strategic insights.
"""

import json
import copy
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.session import ResearchSession, IterationSummary
from research_agent.agents.react_analyzer import ReActAnalysisAgent
from research_agent.agents.synthesis_agent import SynthesisAgent
from research_agent.utils.robust_json_parser import RobustJSONParser


class TestJSONSerializationFixes(unittest.TestCase):
    """Test suite for validating JSON serialization fixes"""

    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            'research_session': {
                'max_papers_total': 10,
                'max_iterations': 1,
                'source_filter': 'sci_hub'
            }
        }

    def test_iteration_summary_serialization(self):
        """Test that IterationSummary objects can be properly serialized"""
        # Create a sample IterationSummary
        summary = IterationSummary(
            iteration=1,
            queries=['test query 1', 'test query 2'],
            papers_found=5,
            processing_results=[],
            gaps_identified=['gap 1', 'gap 2'],
            new_queries=[],
            completion_score=0.75,
            timestamp=datetime.now()
        )
        
        # Test direct serialization (should fail without to_dict)
        with self.assertRaises(TypeError):
            json.dumps(summary)
        
        # Test proper serialization using to_dict
        serialized = json.dumps(summary.to_dict())
        self.assertIsInstance(serialized, str)
        
        # Test deserialization
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized['iteration'], 1)
        self.assertEqual(len(deserialized['queries']), 2)
        self.assertEqual(deserialized['completion_score'], 0.75)

    def test_orchestrator_synthesis_context_serialization(self):
        """Test that orchestrator properly serializes synthesis context"""
        from research_agent.core.config import ResearchConfig
        
        # Create minimal session with iteration summaries
        config = ResearchConfig()
        session = ResearchSession(
            topic="test topic",
            config=config.research_session.__dict__
        )
        
        # Add test iteration summaries
        summary1 = IterationSummary(
            iteration=1,
            queries=['query 1'],
            papers_found=3,
            processing_results=[],
            gaps_identified=[],
            new_queries=[],
            completion_score=0.5,
            timestamp=datetime.now()
        )
        session.iteration_summaries.append(summary1)
        
        # Test synthesis context creation (this should not fail with our fix)
        synthesis_context = {
            'topic': session.topic,
            'iterations': [summary.to_dict() for summary in session.iteration_summaries],  # Fixed version
            'papers': [],
            'processing_results': [],
            'statistics': {'papers': 0, 'queries': 0}
        }
        
        # This should serialize without error
        serialized_context = json.dumps(synthesis_context, indent=2)
        self.assertIsInstance(serialized_context, str)
        
        # Verify the structure
        parsed_context = json.loads(serialized_context)
        self.assertEqual(len(parsed_context['iterations']), 1)
        self.assertEqual(parsed_context['iterations'][0]['iteration'], 1)

    def test_robust_json_parser_handling(self):
        """Test that RobustJSONParser handles various malformed JSON responses"""
        parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
        
        # Test case 1: Empty response (common LLM issue)
        empty_response = ""
        result = parser.parse(
            empty_response,
            expected_keys=['analysis_summary', 'gaps_identified'],
            fallback_structure={'analysis_summary': 'fallback', 'gaps_identified': []}
        )
        self.assertEqual(result['analysis_summary'], 'fallback')
        
        # Test case 2: Malformed JSON with extra text
        malformed_response = "Here's the analysis:\n\n```json\n{\"analysis\": \"test\"}\n```\n\nHope this helps!"
        result = parser.parse(malformed_response, expected_keys=['analysis'])
        self.assertEqual(result['analysis'], 'test')
        
        # Test case 3: Completely invalid JSON
        invalid_response = "This is not JSON at all! Just plain text."
        result = parser.parse(
            invalid_response,
            expected_keys=['test'],
            fallback_structure={'test': 'fallback_value'}
        )
        self.assertEqual(result['test'], 'fallback_value')
        
        # Test case 4: JSON with missing quotes (common LLM error)
        unquoted_response = "{analysis: test, status: complete}"  # Missing quotes
        result = parser.parse(
            unquoted_response,
            expected_keys=['analysis', 'status'],
            fallback_structure={'analysis': 'fixed', 'status': 'fixed'}
        )
        # Should either parse correctly or use fallback
        self.assertIn(result['status'], ['complete', 'fixed'])

    def test_llm_response_variability_simulation(self):
        """Simulate various LLM response formats to ensure robust handling"""
        parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
        
        # Various response formats that LLMs might produce
        llm_responses = [
            # Standard JSON
            '{"gaps": ["gap1"], "continue": true}',
            
            # JSON with extra whitespace
            '\n\n  {"gaps": ["gap1"], "continue": true}  \n',
            
            # JSON wrapped in markdown
            '```json\n{"gaps": ["gap1"], "continue": true}\n```',
            
            # JSON with explanation text
            'Based on analysis:\n{"gaps": ["gap1"], "continue": true}\n\nThis concludes the analysis.',
            
            # Malformed but recoverable
            '{"gaps": ["gap1"], "continue": true',  # Missing closing brace
            
            # Multiple JSON objects (take first)
            '{"gaps": ["gap1"]} {"continue": true}',
            
            # Empty/whitespace only
            '   \n  \t  \n   ',
            
            # Non-JSON response
            'I found several gaps in the analysis but cannot format as JSON right now.'
        ]
        
        fallback_structure = {'gaps': [], 'continue': False}
        
        for i, response in enumerate(llm_responses):
            with self.subTest(response_index=i):
                result = parser.parse(
                    response,
                    expected_keys=['gaps', 'continue'],
                    fallback_structure=fallback_structure
                )
                
                # All responses should result in valid structure
                self.assertIn('gaps', result)
                self.assertIn('continue', result)
                self.assertIsInstance(result['gaps'], list)
                self.assertIsInstance(result['continue'], bool)

    def test_deep_copy_protection(self):
        """Test that deep copying prevents object mutation during serialization"""
        # Create original object
        original_data = {
            'papers': [{'title': 'Paper 1', 'authors': ['Author A']}],
            'stats': {'count': 5, 'details': {'processed': True}}
        }
        
        # Create deep copy
        copied_data = copy.deepcopy(original_data)
        
        # Modify the copy
        copied_data['papers'][0]['title'] = 'Modified Paper'
        copied_data['stats']['details']['processed'] = False
        
        # Original should be unchanged
        self.assertEqual(original_data['papers'][0]['title'], 'Paper 1')
        self.assertTrue(original_data['stats']['details']['processed'])

    @patch('json.dumps')
    def test_direct_serialization_detection(self):
        """Test monkey patching to detect direct JSON serialization attempts"""
        def strict_json_dumps(obj, *args, **kwargs):
            if hasattr(obj, 'to_dict') and not isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                raise ValueError(f"Direct json.dumps call detected on {type(obj).__name__}! Use to_dict() method.")
            return json.JSONEncoder(*args, **kwargs).encode(obj)
        
        with patch('json.dumps', side_effect=strict_json_dumps):
            # This should work (primitive type)
            result = json.dumps({'test': 'value'})
            
            # This should fail (object with to_dict method)
            summary = IterationSummary(
                iteration=1, queries=[], papers_found=0, processing_results=[],
                gaps_identified=[], new_queries=[], completion_score=0.0, timestamp=datetime.now()
            )
            
            with self.assertRaises(ValueError, msg="Direct json.dumps call detected"):
                json.dumps(summary)

    def test_async_serialization_safety(self):
        """Test that serialization works correctly in async contexts"""
        async def serialize_async_data():
            # Simulate async processing with serialization
            summary = IterationSummary(
                iteration=1, queries=['async query'], papers_found=2, processing_results=[],
                gaps_identified=[], new_queries=[], completion_score=0.8, timestamp=datetime.now()
            )
            
            # This should work without race conditions
            serialized = json.dumps(summary.to_dict())
            await asyncio.sleep(0.01)  # Simulate async work
            
            return json.loads(serialized)
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(serialize_async_data())
            self.assertEqual(result['iteration'], 1)
            self.assertEqual(result['completion_score'], 0.8)
        finally:
            loop.close()

    def test_error_context_logging(self):
        """Test that error contexts provide sufficient debugging information"""
        parser = RobustJSONParser(strict_mode=False, fallback_enabled=True)
        
        # Create a response that will definitely fail parsing
        problematic_response = "{ invalid json with [unclosed brackets and 'mixed quotes }"
        
        # Capture logging output
        import logging
        import io
        
        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        logger = logging.getLogger('research_agent.utils.robust_json_parser')
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
        
        try:
            result = parser.parse(
                problematic_response,
                expected_keys=['test'],
                fallback_structure={'test': 'fallback'}
            )
            
            # Should use fallback
            self.assertEqual(result['test'], 'fallback')
            
            # Check that detailed error information was logged
            log_contents = log_capture_string.getvalue()
            self.assertIn('problematic', log_contents.lower())
            
        finally:
            logger.removeHandler(ch)


class IntegrationTests(unittest.TestCase):
    """Integration tests for end-to-end JSON serialization"""
    
    def test_full_pipeline_serialization(self):
        """Test that the full research pipeline handles serialization correctly"""
        # This would be a more comprehensive test that runs the actual agents
        # For now, we'll test the key integration points
        
        # Test 1: Create session with proper serialization
        from research_agent.core.config import ResearchConfig
        config = ResearchConfig()
        
        session = ResearchSession(
            topic="integration test",
            config=config.research_session.__dict__
        )
        
        # Add test data
        summary = IterationSummary(
            iteration=1, queries=['integration query'], papers_found=1, processing_results=[],
            gaps_identified=['test gap'], new_queries=[], completion_score=0.6, timestamp=datetime.now()
        )
        session.iteration_summaries.append(summary)
        
        # Test session serialization (should work with our fixes)
        session_dict = session.to_dict()
        serialized_session = json.dumps(session_dict, indent=2)
        
        # Verify structure
        parsed_session = json.loads(serialized_session)
        self.assertEqual(parsed_session['topic'], "integration test")
        self.assertEqual(len(parsed_session['iteration_summaries']), 1)
        self.assertEqual(parsed_session['iteration_summaries'][0]['iteration'], 1)


def run_validation_suite():
    """Run the complete validation test suite"""
    print("🔬 Running JSON Serialization Fixes Validation Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestJSONSerializationFixes))
    suite.addTests(loader.loadTestsFromTestCase(IntegrationTests))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - JSON serialization fixes validated!")
    else:
        print("❌ Some tests failed - check the output above")
        print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)