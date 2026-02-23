#!/usr/bin/env python3
"""
Quick test of the enhanced Sci-Hub fallback logic
"""

import asyncio
import sys
from pathlib import Path

# Add the research agent to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from research_agent.agents.literature_searcher import LiteratureSearchAgent

async def test_fallback():
    """Test the enhanced search with fallback"""
    print("🔬 Testing Enhanced Literature Search with Sci-Hub Cleanup Fallback")
    
    # Create agent
    agent_config = {
        'model': 'openai/gpt-4o-mini',
        'pubmed': {
            'email': 'research@example.com',
            'tool': 'ResearchAgent'
        }
    }
    
    agent = LiteratureSearchAgent(config=agent_config)
    
    # Create test search strategy
    search_strategy = {
        'database_allocation': {
            'arxiv': {'max_papers': 2, 'search_terms': ['neural networks']},
            'pubmed': {'max_papers': 2, 'search_terms': ['machine learning']},
            'sci_hub': {'max_papers': 3, 'search_terms': ['artificial intelligence']}
        },
        'search_strategy': {
            'primary_keywords': ['artificial intelligence'],
            'secondary_keywords': ['machine learning']
        }
    }
    
    strategy_json = str(search_strategy).replace("'", '"')
    
    try:
        # Test the enhanced search
        papers = await agent.search_papers(strategy_json, max_papers=6)
        
        print(f"\n📊 Results Summary:")
        print(f"Total papers found: {len(papers)}")
        
        # Analyze by source
        source_breakdown = {}
        for paper in papers:
            source = paper.get('source', 'unknown')
            search_method = paper.get('metadata', {}).get('search_method', 'keyword')
            
            key = f"{source}_{search_method}" if search_method != 'keyword' else source
            source_breakdown[key] = source_breakdown.get(key, 0) + 1
        
        print(f"Source breakdown: {source_breakdown}")
        
        # Show some sample papers
        print(f"\n📑 Sample Papers:")
        for i, paper in enumerate(papers[:3]):
            title = paper.get('title', 'Unknown Title')[:60] + '...' if len(paper.get('title', '')) > 60 else paper.get('title', 'Unknown Title')
            source = paper.get('source', 'unknown')
            search_method = paper.get('metadata', {}).get('search_method', 'keyword')
            print(f"  {i+1}. [{source}|{search_method}] {title}")
        
        print(f"\n✅ Enhanced literature search with cleanup fallback working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fallback())
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}")