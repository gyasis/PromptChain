#!/usr/bin/env python3
"""
Debug script to isolate literature search issues
Tests each component individually to identify the root cause
"""

import asyncio
import logging
import sys
import os
sys.path.append('/home/gyasis/Documents/code/PromptChain/Research_agent/src')

from datetime import datetime

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test 1: Basic imports"""
    print("=== TEST 1: Basic Imports ===")
    try:
        import arxiv
        print("✓ arxiv imported successfully")
        
        from Bio import Entrez
        print("✓ Bio.Entrez imported successfully")
        
        from research_agent.agents.literature_searcher import LiteratureSearchAgent
        print("✓ LiteratureSearchAgent imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config_loading():
    """Test 2: Configuration loading"""
    print("\n=== TEST 2: Configuration Loading ===")
    try:
        import yaml
        config_path = "/home/gyasis/Documents/code/PromptChain/Research_agent/config/research_config.yaml"
        
        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Configuration loaded successfully")
        print(f"  - Literature search sources: {config.get('literature_search', {}).get('enabled_sources', [])}")
        return config
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return None

async def test_arxiv_direct():
    """Test 3: Direct ArXiv API test"""
    print("\n=== TEST 3: Direct ArXiv API Test ===")
    try:
        import arxiv
        
        # Test with a simple, known working query
        print("Testing with simple query: 'machine learning'")
        search = arxiv.Search(
            query='machine learning',
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            papers.append({
                'title': result.title,
                'authors': [str(author) for author in result.authors],
                'published': result.published.year
            })
            
        print(f"✓ ArXiv API working: {len(papers)} papers found")
        for i, paper in enumerate(papers[:3]):
            print(f"  {i+1}. {paper['title'][:60]}...")
        return True
        
    except Exception as e:
        print(f"✗ ArXiv API failed: {e}")
        return False

async def test_pubmed_direct():
    """Test 4: Direct PubMed API test"""
    print("\n=== TEST 4: Direct PubMed API Test ===")
    try:
        from Bio import Entrez
        from Bio.Entrez import esearch, efetch
        
        # Set required email
        Entrez.email = "test@example.com"
        Entrez.tool = "test-script"
        
        # Test with a simple query
        print("Testing with simple query: 'machine learning'")
        search_handle = esearch(
            db='pubmed',
            term='machine learning',
            retmax=5,
            sort='relevance'
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        pmids = search_results.get('IdList', [])
        print(f"✓ PubMed search working: {len(pmids)} papers found")
        
        if pmids:
            print(f"  - Found PMIDs: {pmids[:3]}...")
        return True
        
    except Exception as e:
        print(f"✗ PubMed API failed: {e}")
        return False

async def test_search_with_typo_query():
    """Test 5: Test the actual problematic query"""
    print("\n=== TEST 5: Test Problematic Query ===")
    
    # The actual query with typos from the log
    problematic_query = "early neurilogical diseae detection with gait analysis"
    corrected_query = "early neurological disease detection with gait analysis"
    
    print(f"Original query: '{problematic_query}'")
    print(f"Corrected query: '{corrected_query}'")
    
    try:
        import arxiv
        
        # Test both queries
        for query_type, query in [("typo", problematic_query), ("corrected", corrected_query)]:
            print(f"\nTesting {query_type} query: '{query}'")
            
            search = arxiv.Search(
                query=f'all:"{query}"',
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = list(search.results())
            print(f"  - ArXiv results: {len(papers)} papers")
            
            # Also test with broader search
            search_broad = arxiv.Search(
                query='neurological disease gait analysis',
                max_results=10,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers_broad = list(search_broad.results())
            print(f"  - ArXiv broader search: {len(papers_broad)} papers")
        
        return True
        
    except Exception as e:
        print(f"✗ Query testing failed: {e}")
        return False

async def test_literature_agent_initialization():
    """Test 6: Literature Agent initialization"""
    print("\n=== TEST 6: Literature Agent Initialization ===")
    try:
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
        
        from research_agent.agents.literature_searcher import LiteratureSearchAgent
        agent = LiteratureSearchAgent(config)
        print("✓ LiteratureSearchAgent initialized successfully")
        
        return agent
        
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        return None

async def test_agent_search():
    """Test 7: Full agent search test"""
    print("\n=== TEST 7: Full Agent Search Test ===")
    
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
    
    try:
        from research_agent.agents.literature_searcher import LiteratureSearchAgent
        agent = LiteratureSearchAgent(config)
        
        # Test with corrected query
        corrected_topic = "neurological disease detection with gait analysis"
        print(f"Testing with corrected topic: '{corrected_topic}'")
        
        # Test search with source filter to isolate which source fails
        for source in ['arxiv', 'pubmed']:
            print(f"\nTesting {source} only...")
            try:
                papers = await agent.search_papers(
                    strategy=corrected_topic,
                    max_papers=5,
                    source_filter=source
                )
                print(f"  ✓ {source}: {len(papers)} papers found")
                if papers:
                    print(f"    Sample: {papers[0].get('title', 'No title')[:60]}...")
            except Exception as source_error:
                print(f"  ✗ {source} failed: {source_error}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent search test failed: {e}")
        return False

async def test_network_connectivity():
    """Test 8: Network connectivity"""
    print("\n=== TEST 8: Network Connectivity ===")
    try:
        import aiohttp
        
        # Test connectivity to key APIs
        urls_to_test = [
            ("ArXiv API", "http://export.arxiv.org/api/query?search_query=machine+learning&max_results=1"),
            ("PubMed API", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=test&retmax=1"),
        ]
        
        async with aiohttp.ClientSession() as session:
            for name, url in urls_to_test:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            print(f"✓ {name}: Connection successful (status {response.status})")
                        else:
                            print(f"⚠ {name}: Connection issues (status {response.status})")
                except Exception as url_error:
                    print(f"✗ {name}: Connection failed - {url_error}")
        
        return True
        
    except Exception as e:
        print(f"✗ Network connectivity test failed: {e}")
        return False

async def main():
    """Run all diagnostic tests"""
    print("Literature Search Diagnostic Test Suite")
    print("=" * 50)
    
    # Run tests in sequence
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Config loading
    config = test_config_loading()
    if config:
        tests_passed += 1
    
    # Test 3: ArXiv direct
    if await test_arxiv_direct():
        tests_passed += 1
    
    # Test 4: PubMed direct
    if await test_pubmed_direct():
        tests_passed += 1
    
    # Test 5: Query testing
    if await test_search_with_typo_query():
        tests_passed += 1
    
    # Test 6: Agent initialization
    agent = await test_literature_agent_initialization()
    if agent:
        tests_passed += 1
    
    # Test 7: Full agent search
    if await test_agent_search():
        tests_passed += 1
    
    # Test 8: Network connectivity
    if await test_network_connectivity():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"DIAGNOSTIC SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed - literature search should be working")
    elif tests_passed >= 6:
        print("⚠ Most tests passed - minor configuration issues")
    else:
        print("✗ Major issues detected - requires investigation")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    asyncio.run(main())