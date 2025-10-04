#!/usr/bin/env python3
"""
Test Data Structure Fixes

Validates that the data structure compatibility issues have been resolved.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add research agent to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_paper_data_validation():
    """Test Paper data structure validation"""
    print("Testing Paper data validation...")
    
    try:
        from research_agent.core.session import ResearchSession, Paper
        from research_agent.utils.data_validation import validate_paper
        
        # Test valid paper data
        valid_paper = {
            'id': 'test_paper_1',
            'title': 'Test Paper',
            'authors': ['Author 1', 'Author 2'],
            'abstract': 'Test abstract',
            'source': 'arxiv',
            'url': 'https://arxiv.org/abs/1234.5678',
            'doi': '10.1234/test',
            'publication_year': 2024
        }
        
        validated = validate_paper(valid_paper, 'test')
        logger.info(f"Valid paper validation passed: {len(validated)} fields")
        
        # Test paper data with metadata fields at top level
        paper_with_journal = {
            'id': 'test_paper_2', 
            'title': 'Paper with Journal',
            'authors': ['Author 1'],
            'abstract': 'Test abstract with journal field',
            'source': 'pubmed',
            'url': 'https://pubmed.ncbi.nlm.nih.gov/12345',
            'journal': 'Nature',  # Should be moved to metadata
            'full_text_available': True
        }
        
        validated_journal = validate_paper(paper_with_journal, 'test_journal')
        assert 'journal' in validated_journal['metadata']
        logger.info("Paper with journal field validation passed")
        
        # Test session paper addition
        session = ResearchSession(topic="Test Topic")
        paper_ids = session.add_papers([valid_paper, paper_with_journal])
        logger.info(f"Session paper addition passed: {len(paper_ids)} papers added")
        
        print("✅ Paper data validation tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Paper data validation tests FAILED: {e}")
        return False

def test_processing_result_structure():
    """Test ProcessingResult data structure"""
    print("Testing ProcessingResult structure...")
    
    try:
        from research_agent.core.session import ProcessingResult, ProcessingStatus
        from research_agent.utils.data_validation import validate_processing_result
        
        # Test correct ProcessingResult structure
        result_data = {
            'tier': 'lightrag',
            'query_id': 'test_query_1',
            'paper_ids': ['paper_1', 'paper_2'],
            'result_data': {'test': 'data'},
            'processing_time': 1.5
        }
        
        validated = validate_processing_result(result_data)
        logger.info("ProcessingResult validation passed")
        
        # Test ProcessingResult creation with correct parameters
        processing_result = ProcessingResult(
            tier='test_tier',
            query_id='test_query',
            paper_ids=['test_paper'],
            result_data={'success': True},
            processing_time=0.5,
            status=ProcessingStatus.COMPLETED,
            timestamp=datetime.now()
        )
        
        assert hasattr(processing_result, 'paper_ids')
        assert isinstance(processing_result.paper_ids, list)
        logger.info("ProcessingResult creation passed")
        
        print("✅ ProcessingResult structure tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ ProcessingResult structure tests FAILED: {e}")
        return False

def test_query_data_handling():
    """Test query data structure handling"""
    print("Testing query data handling...")
    
    try:
        from research_agent.utils.data_validation import validate_query
        
        # Test string query
        string_query = "What are the latest developments in machine learning?"
        validated_string = validate_query(string_query)
        assert 'text' in validated_string
        assert 'priority' in validated_string
        logger.info("String query validation passed")
        
        # Test dict query
        dict_query = {
            'text': 'How do neural networks work?',
            'priority': 0.8,
            'category': 'technical'
        }
        validated_dict = validate_query(dict_query)
        assert validated_dict['text'] == dict_query['text']
        logger.info("Dict query validation passed")
        
        # Test query structure in agent format
        query_response = {
            'primary_queries': [
                {'text': 'Primary query 1', 'priority': 1.0, 'category': 'main'},
                {'text': 'Primary query 2', 'priority': 0.9, 'category': 'main'}
            ],
            'secondary_queries': [
                {'text': 'Secondary query 1', 'priority': 0.7, 'category': 'supporting'}
            ],
            'exploratory_queries': [
                {'text': 'Exploratory query 1', 'priority': 0.5, 'category': 'exploratory'}
            ]
        }
        
        # Flatten queries like orchestrator does
        all_queries = []
        for section in ['primary_queries', 'secondary_queries', 'exploratory_queries']:
            if section in query_response:
                for query in query_response[section]:
                    all_queries.append(validate_query(query))
        
        assert len(all_queries) == 4
        logger.info(f"Query flattening passed: {len(all_queries)} queries")
        
        print("✅ Query data handling tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Query data handling tests FAILED: {e}")
        return False

def test_strategy_validation():
    """Test search strategy validation"""
    print("Testing search strategy validation...")
    
    try:
        from research_agent.utils.data_validation import validate_strategy
        
        # Test complete strategy
        strategy_data = {
            'search_strategy': {
                'primary_keywords': ['machine learning', 'neural networks'],
                'secondary_keywords': ['deep learning', 'AI'],
                'boolean_queries': [
                    {
                        'database': 'sci_hub',
                        'query': 'machine learning AND neural networks',
                        'priority': 1.0
                    }
                ]
            },
            'database_allocation': {
                'sci_hub': {
                    'priority': 1.0,
                    'max_papers': 40,
                    'search_terms': ['neural networks'],
                    'rationale': 'Primary source'
                },
                'arxiv': {
                    'priority': 0.8,
                    'max_papers': 30,
                    'search_terms': ['machine learning'],
                    'rationale': 'Recent research'
                },
                'pubmed': {
                    'priority': 0.6,
                    'max_papers': 20,
                    'search_terms': ['AI applications'],
                    'rationale': 'Medical applications'
                }
            },
            'search_optimization': {
                'iteration_focus': 'Focus on foundational papers',
                'gap_targeting': ['theoretical foundations'],
                'expansion_areas': ['applications'],
                'exclusion_criteria': ['non-English']
            }
        }
        
        validated = validate_strategy(strategy_data)
        assert 'search_strategy' in validated
        assert 'database_allocation' in validated
        assert 'search_optimization' in validated
        logger.info("Complete strategy validation passed")
        
        # Test incomplete strategy
        incomplete_strategy = {
            'search_strategy': {
                'primary_keywords': ['test']
            }
        }
        
        validated_incomplete = validate_strategy(incomplete_strategy)
        assert 'database_allocation' in validated_incomplete
        assert 'search_optimization' in validated_incomplete
        logger.info("Incomplete strategy validation passed")
        
        print("✅ Search strategy validation tests PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Search strategy validation tests FAILED: {e}")
        return False

def main():
    """Run all data structure validation tests"""
    print("🔍 Running Data Structure Fix Validation Tests\n")
    
    test_results = [
        test_paper_data_validation(),
        test_processing_result_structure(),
        test_query_data_handling(),
        test_strategy_validation()
    ]
    
    total_tests = len(test_results)
    passed_tests = sum(test_results)
    
    print(f"\n📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All data structure compatibility fixes are working correctly!")
        return True
    else:
        print("⚠️  Some tests failed. Data structure issues may still exist.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)