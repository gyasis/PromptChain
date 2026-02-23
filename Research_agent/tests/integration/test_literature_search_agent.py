#!/usr/bin/env python3
"""
Simple Test Script for Literature Search Agent

This script demonstrates how to use the literature search agent system:
1. Initialize the SearchStrategistAgent and LiteratureSearchAgent
2. Create a search strategy for a research topic
3. Execute literature search across multiple databases
4. Display and analyze results

Usage:
    python test_literature_search_agent.py
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, List, Any

# Add the src directory to the path so we can import the agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.agents.search_strategist import SearchStrategistAgent
from research_agent.agents.literature_searcher import LiteratureSearchAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def create_test_config() -> Dict[str, Any]:
    """Create a test configuration for the agents"""
    return {
        # Model configuration
        'model': 'openai/gpt-4o-mini',
        
        # PubMed configuration (required for PubMed searches)
        'pubmed': {
            'email': 'test@example.com',  # Replace with your email
            'tool': 'ResearchAgent-Test'
        },
        
        # Agentic processor configuration
        'processor': {
            'max_internal_steps': 3,
            'objective': 'Test literature search functionality'
        },
        
        # Search limits for testing
        'max_papers': 10,
        'timeout': 30
    }


async def test_search_strategy_creation(config: Dict[str, Any], topic: str) -> str:
    """
    Test the SearchStrategistAgent to create a search strategy
    
    Args:
        config: Agent configuration
        topic: Research topic to search for
        
    Returns:
        JSON string containing the search strategy
    """
    logger.info(f"Testing SearchStrategistAgent with topic: {topic}")
    
    try:
        # Initialize the search strategist agent
        strategist = SearchStrategistAgent(config)
        
        # Create a simple search strategy
        # In a real scenario, this would use the agentic processor
        # For testing, we'll create a basic strategy manually
        strategy = {
            "search_strategy": {
                "primary_keywords": [topic, f"{topic} research", f"{topic} methods"],
                "secondary_keywords": [f"{topic} applications", f"{topic} analysis"],
                "boolean_queries": [
                    {
                        "database": "sci_hub",
                        "query": f'"{topic}" OR "{topic} research"',
                        "priority": 1.0
                    }
                ],
                "filters": {
                    "publication_years": [2020, 2024],
                    "paper_types": ["research", "review"],
                    "languages": ["en"]
                }
            },
            "database_allocation": {
                "sci_hub": {
                    "priority": 1.0,
                    "max_papers": 5,
                    "search_terms": [topic, f"{topic} research"],
                    "rationale": "Best for getting full papers"
                },
                "arxiv": {
                    "priority": 0.8,
                    "max_papers": 3,
                    "search_terms": [topic, f"{topic} methods"],
                    "rationale": "Latest research before publication"
                },
                "pubmed": {
                    "priority": 0.7,
                    "max_papers": 2,
                    "search_terms": [topic, f"{topic} applications"],
                    "rationale": "Medical domain expertise"
                }
            },
            "search_optimization": {
                "iteration_focus": "Initial broad search for foundational papers",
                "gap_targeting": ["Recent advances", "Methodological improvements"],
                "expansion_areas": ["Cross-disciplinary applications"],
                "exclusion_criteria": ["Non-English papers", "Very old papers (pre-2020)"]
            }
        }
        
        logger.info("Search strategy created successfully")
        return json.dumps(strategy)
        
    except Exception as e:
        logger.error(f"Failed to create search strategy: {e}")
        raise


async def test_literature_search(config: Dict[str, Any], strategy: str, max_papers: int = 10) -> List[Dict[str, Any]]:
    """
    Test the LiteratureSearchAgent to execute literature search
    
    Args:
        config: Agent configuration
        strategy: JSON string containing search strategy
        max_papers: Maximum number of papers to retrieve
        
    Returns:
        List of paper dictionaries
    """
    logger.info(f"Testing LiteratureSearchAgent with max {max_papers} papers")
    
    try:
        # Initialize the literature search agent
        searcher = LiteratureSearchAgent(config)
        
        # Execute the search
        papers = await searcher.search_papers(
            strategy=strategy,
            max_papers=max_papers
        )
        
        logger.info(f"Literature search completed: {len(papers)} papers found")
        return papers
        
    except Exception as e:
        logger.error(f"Failed to execute literature search: {e}")
        raise


def display_papers(papers: List[Dict[str, Any]], max_display: int = 5):
    """
    Display the found papers in a readable format
    
    Args:
        papers: List of paper dictionaries
        max_display: Maximum number of papers to display
    """
    if not papers:
        print("\n❌ No papers found!")
        return
    
    print(f"\n📚 Found {len(papers)} papers:")
    print("=" * 80)
    
    for i, paper in enumerate(papers[:max_display]):
        print(f"\n📄 Paper {i+1}:")
        print(f"   Title: {paper.get('title', 'No title')}")
        print(f"   Authors: {', '.join(paper.get('authors', ['Unknown']))[:100]}...")
        print(f"   Source: {paper.get('source', 'Unknown')}")
        print(f"   Year: {paper.get('publication_year', 'Unknown')}")
        print(f"   DOI: {paper.get('doi', 'No DOI')}")
        
        # Show abstract preview
        abstract = paper.get('abstract', 'No abstract')
        if len(abstract) > 150:
            abstract = abstract[:150] + "..."
        print(f"   Abstract: {abstract}")
        
        # Show metadata
        metadata = paper.get('metadata', {})
        if metadata:
            print(f"   Journal: {metadata.get('journal', 'Unknown')}")
            print(f"   Database Priority: {metadata.get('database_priority', 'Unknown')}")
    
    if len(papers) > max_display:
        print(f"\n... and {len(papers) - max_display} more papers")


def analyze_results(papers: List[Dict[str, Any]]):
    """
    Analyze the search results and provide statistics
    
    Args:
        papers: List of paper dictionaries
    """
    if not papers:
        print("\n📊 No papers to analyze")
        return
    
    print("\n📊 Search Results Analysis:")
    print("=" * 40)
    
    # Count by source
    sources = {}
    years = {}
    has_doi = 0
    has_full_text = 0
    
    for paper in papers:
        source = paper.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
        
        year = paper.get('publication_year', 'unknown')
        years[year] = years.get(year, 0) + 1
        
        if paper.get('doi'):
            has_doi += 1
        
        if paper.get('full_text_available'):
            has_full_text += 1
    
    print(f"📈 Total Papers: {len(papers)}")
    print(f"📚 Papers by Source:")
    for source, count in sources.items():
        print(f"   {source.upper()}: {count}")
    
    print(f"\n📅 Papers by Year:")
    for year in sorted(years.keys(), reverse=True):
        print(f"   {year}: {years[year]}")
    
    print(f"\n🔗 Papers with DOI: {has_doi}/{len(papers)} ({has_doi/len(papers)*100:.1f}%)")
    print(f"📄 Papers with Full Text: {has_full_text}/{len(papers)} ({has_full_text/len(papers)*100:.1f}%)")


async def test_individual_sources(config: Dict[str, Any], topic: str):
    """
    Test searching individual sources separately
    
    Args:
        config: Agent configuration
        topic: Research topic
    """
    logger.info(f"Testing individual source searches for topic: {topic}")
    
    # Create a simple strategy for individual source testing
    simple_strategy = {
        "database_allocation": {
            "arxiv": {
                "max_papers": 3,
                "search_terms": [topic]
            },
            "pubmed": {
                "max_papers": 2,
                "search_terms": [topic]
            },
            "sci_hub": {
                "max_papers": 3,
                "search_terms": [topic]
            }
        },
        "search_strategy": {
            "primary_keywords": [topic],
            "secondary_keywords": []
        }
    }
    
    searcher = LiteratureSearchAgent(config)
    
    # Test each source individually
    sources = ['arxiv', 'pubmed', 'sci_hub']
    
    for source in sources:
        print(f"\n🔍 Testing {source.upper()} search:")
        try:
            papers = await searcher.search_papers(
                strategy=json.dumps(simple_strategy),
                max_papers=3,
                source_filter=source
            )
            print(f"   ✅ {source.upper()}: {len(papers)} papers found")
            if papers:
                print(f"   📄 Sample: {papers[0].get('title', 'No title')[:60]}...")
        except Exception as e:
            print(f"   ❌ {source.upper()}: Error - {e}")


async def main():
    """Main test function"""
    print("🧪 Literature Search Agent Test Script")
    print("=" * 50)
    
    # Configuration
    config = create_test_config()
    
    # Test topic
    topic = "machine learning"
    
    try:
        # Test 1: Search Strategy Creation
        print(f"\n1️⃣ Testing Search Strategy Creation")
        strategy = await test_search_strategy_creation(config, topic)
        print("   ✅ Search strategy created successfully")
        
        # Test 2: Full Literature Search
        print(f"\n2️⃣ Testing Full Literature Search")
        papers = await test_literature_search(config, strategy, max_papers=10)
        
        # Display results
        display_papers(papers, max_display=3)
        
        # Analyze results
        analyze_results(papers)
        
        # Test 3: Individual Source Testing
        print(f"\n3️⃣ Testing Individual Sources")
        await test_individual_sources(config, topic)
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 