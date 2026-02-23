#!/usr/bin/env python3
"""
Test the literature search fix with the improved query generation
"""

import asyncio
import sys
sys.path.append('/home/gyasis/Documents/code/PromptChain/Research_agent/src')

import logging
logging.basicConfig(level=logging.INFO)

async def test_fixed_literature_search():
    """Test the fixed literature search implementation"""
    
    # Test configuration
    config = {
        'model': 'openai/gpt-4o-mini',
        'pubmed': {
            'email': 'test@example.com',
            'tool': 'debug-test'
        },
        'processor': {
            'max_internal_steps': 3
        }
    }
    
    # Import the fixed agent
    from research_agent.agents.literature_searcher import LiteratureSearchAgent
    agent = LiteratureSearchAgent(config)
    
    # Test queries - including the original problematic one
    test_queries = [
        "early neurilogical diseae detection with gait analysis",  # Original with typos
        "machine learning for medical diagnosis",
        "parkinson disease gait analysis",
        "neurological disease detection"
    ]
    
    print("Testing Fixed Literature Search Agent")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Test ArXiv only first
            print("Testing ArXiv...")
            arxiv_papers = await agent.search_papers(
                strategy=query,
                max_papers=10,
                source_filter='arxiv'
            )
            print(f"  ArXiv: {len(arxiv_papers)} papers found")
            if arxiv_papers:
                print(f"    Sample: {arxiv_papers[0].get('title', 'No title')[:70]}...")
            
            # Test PubMed
            print("Testing PubMed...")
            pubmed_papers = await agent.search_papers(
                strategy=query,
                max_papers=10,
                source_filter='pubmed'
            )
            print(f"  PubMed: {len(pubmed_papers)} papers found")
            if pubmed_papers:
                print(f"    Sample: {pubmed_papers[0].get('title', 'No title')[:70]}...")
            
            # Test combined search
            print("Testing combined search...")
            all_papers = await agent.search_papers(
                strategy=query,
                max_papers=15
            )
            print(f"  Combined: {len(all_papers)} papers found")
            
            # Summary
            total_found = len(all_papers)
            if total_found > 0:
                print(f"  ✓ SUCCESS: Found {total_found} papers")
            else:
                print(f"  ✗ FAILED: No papers found")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        
        print()
    
    print("=" * 50)
    return True

async def test_query_preprocessing():
    """Test the query preprocessing specifically"""
    print("\nTesting Query Preprocessing")
    print("-" * 30)
    
    from research_agent.agents.literature_searcher import LiteratureSearchAgent
    agent = LiteratureSearchAgent({'model': 'openai/gpt-4o-mini'})
    
    # Test typo correction
    test_inputs = [
        "early neurilogical diseae detection",
        "machien learnign algorithim",
        "artifical intelligance diagosis"
    ]
    
    for original in test_inputs:
        corrected = agent._preprocess_query(original)
        print(f"  '{original}' → '{corrected}'")
    
    # Test key term extraction
    print("\nTesting Key Term Extraction")
    print("-" * 30)
    
    test_text = "early neurological disease detection with gait analysis using machine learning"
    key_terms = agent._extract_key_terms(test_text)
    print(f"  Input: {test_text}")
    print(f"  Key terms: {key_terms}")

if __name__ == "__main__":
    asyncio.run(test_query_preprocessing())
    asyncio.run(test_fixed_literature_search())