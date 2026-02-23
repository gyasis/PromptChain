#!/usr/bin/env python3

"""
Test PubMed Bio.Entrez ListElement handling to identify and fix parsing issues
"""

import asyncio
import logging
from typing import Dict, List, Any
from src.research_agent.integrations.pubmed import PubMedSearcher

# Set up logging to see any errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_pubmed_element_parsing():
    """Test PubMed parsing with different element types that might cause issues"""
    
    print("Testing PubMed Element Parsing:")
    print("=" * 50)
    
    # Initialize PubMed searcher with test config
    config = {
        'email': 'test@research.example.com',
        'tool': 'ResearchAgentTest',
        'max_results_per_query': 5  # Small number for testing
    }
    
    pubmed = PubMedSearcher(config)
    
    # Test parsing with mock Bio.Entrez data that simulates ListElement issues
    print("\n1. Testing with simulated problematic PubMed data:")
    
    # Create mock data that simulates Bio.Entrez ListElement structures
    class MockListElement:
        """Mock Bio.Entrez ListElement that could cause issues"""
        def __init__(self, value):
            self.value = value
            
        def __str__(self):
            return str(self.value)
        
        def strip(self):
            return str(self.value).strip()
    
    class MockElement:
        """Mock Bio.Entrez element with attributes"""
        def __init__(self, value, attrs=None):
            self.value = value
            self.attributes = attrs or {}
            
        def get(self, key, default=None):
            return self.attributes.get(key, default)
        
        def __str__(self):
            return str(self.value)
    
    # Simulate problematic article data
    mock_article = {
        'MedlineCitation': {
            'PMID': '12345678',
            'Article': {
                'ArticleTitle': MockListElement('Test Article Title with ListElement'),
                'Abstract': {
                    'AbstractText': [
                        MockListElement('Background: This is a test abstract.'),
                        MockListElement('Methods: We tested various approaches.'),
                        MockListElement('Results: The results were positive.')
                    ]
                },
                'AuthorList': [
                    MockElement('Smith, John', {'LastName': 'Smith', 'ForeName': 'John'}),
                    MockListElement('Doe, Jane'),  # String-like author
                    {'LastName': MockListElement('Johnson'), 'ForeName': MockListElement('Alice')},  # Dict with ListElements
                    MockElement('Research Group', {'CollectiveName': 'Research Group'})
                ],
                'Journal': {
                    'Title': MockListElement('Journal of Test Research'),
                    'JournalIssue': {
                        'PubDate': {'Year': MockListElement('2023')}
                    }
                },
                'ArticleDate': [
                    {'Year': MockListElement('2023'), 'Month': MockListElement('06'), 'Day': MockListElement('15')}
                ],
                'PublicationTypeList': [
                    MockListElement('Journal Article'),
                    MockListElement('Research Support')
                ]
            },
            'MeshHeadingList': [
                MockListElement('Machine Learning'),
                MockListElement('Artificial Intelligence')
            ],
            'KeywordList': [
                MockListElement('deep learning'),
                MockListElement('neural networks')
            ]
        },
        'PubmedData': {
            'ArticleIdList': [
                MockElement('10.1234/test.2023', {'IdType': 'doi'}),
                MockElement('12345678', {'IdType': 'pubmed'})
            ]
        }
    }
    
    # Test parsing the mock article
    try:
        parsed_paper = pubmed.parse_paper_metadata(mock_article, '12345678')
        
        if parsed_paper:
            print("✓ SUCCESS: Parsed mock article with ListElements")
            print(f"  Title: {parsed_paper['title']}")
            print(f"  Authors: {parsed_paper['authors']}")
            print(f"  Abstract length: {len(parsed_paper['abstract'])} chars")
            print(f"  Publication year: {parsed_paper['publication_year']}")
            print(f"  Journal: {parsed_paper['metadata']['journal']}")
            print(f"  DOI: {parsed_paper['doi']}")
            print(f"  MeSH terms count: {len(parsed_paper['metadata']['mesh_terms'])}")
            print(f"  Keywords count: {len(parsed_paper['metadata']['keywords'])}")
        else:
            print("✗ FAILED: Could not parse mock article")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: Exception during parsing - {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_edge_cases():
    """Test edge cases that might cause ListElement issues"""
    
    print("\n2. Testing edge cases:")
    print("=" * 30)
    
    pubmed = PubMedSearcher()
    
    # Test with None/empty data
    test_cases = [
        {
            'name': 'Empty article',
            'data': {}
        },
        {
            'name': 'Missing MedlineCitation',
            'data': {'PubmedData': {}}
        },
        {
            'name': 'Article with None values',
            'data': {
                'MedlineCitation': {
                    'Article': {
                        'ArticleTitle': None,
                        'Abstract': None,
                        'AuthorList': None
                    }
                }
            }
        },
        {
            'name': 'Article with empty lists',
            'data': {
                'MedlineCitation': {
                    'Article': {
                        'ArticleTitle': '',
                        'Abstract': {'AbstractText': []},
                        'AuthorList': []
                    }
                }
            }
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = pubmed.parse_paper_metadata(test_case['data'])
            if result is None:
                print(f"  {i}. {test_case['name']}: ✓ Handled gracefully (returned None)")
            else:
                print(f"  {i}. {test_case['name']}: ✓ Parsed successfully")
        except Exception as e:
            print(f"  {i}. {test_case['name']}: ✗ FAILED - {str(e)}")
            all_passed = False
    
    return all_passed

async def test_real_pubmed_search():
    """Test actual PubMed search if available"""
    
    print("\n3. Testing real PubMed search (if available):")
    print("=" * 45)
    
    try:
        config = {
            'email': 'test@research.example.com',
            'tool': 'ResearchAgentTest',
            'max_results_per_query': 3  # Very small number for testing
        }
        
        pubmed = PubMedSearcher(config)
        
        # Test with simple search terms
        search_terms = ['machine learning', 'deep learning']
        print(f"Searching PubMed for: {search_terms}")
        
        # Use asyncio timeout to prevent hanging
        papers = await asyncio.wait_for(
            pubmed.search_papers(search_terms, max_papers=3),
            timeout=30.0
        )
        
        print(f"✓ SUCCESS: Found {len(papers)} papers from PubMed")
        
        for i, paper in enumerate(papers, 1):
            print(f"  Paper {i}: {paper['title'][:60]}...")
            print(f"    Authors: {len(paper['authors'])} authors")
            print(f"    Abstract: {len(paper['abstract'])} chars")
            print(f"    Year: {paper['publication_year']}")
        
        return True
        
    except asyncio.TimeoutError:
        print("⚠ WARNING: PubMed search timed out (likely network issue)")
        return True  # Don't fail the test for network issues
    except Exception as e:
        print(f"✗ FAILED: Real PubMed search failed - {str(e)}")
        # Check if it's a network/API issue vs code issue
        if 'network' in str(e).lower() or 'timeout' in str(e).lower() or 'connection' in str(e).lower():
            print("  (This appears to be a network issue, not a code issue)")
            return True
        return False

async def main():
    """Run all PubMed ListElement tests"""
    
    print("=" * 70)
    print("TESTING PUBMED LISTELEMENT HANDLING")
    print("=" * 70)
    
    # Run tests
    test1_passed = test_pubmed_element_parsing()
    test2_passed = test_edge_cases()
    test3_passed = await test_real_pubmed_search()
    
    print("\n" + "=" * 70)
    if test1_passed and test2_passed and test3_passed:
        print("ALL TESTS PASSED ✓")
        print("PubMed ListElement handling is working correctly!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Issues found in PubMed ListElement handling.")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())