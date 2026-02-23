#!/usr/bin/env python3
"""
Test Literature Search Agent with Real Sci-Hub Integration
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.agents.literature_searcher import LiteratureSearchAgent

async def test_literature_search_with_real_scihub():
    """Test the literature search agent with real Sci-Hub data"""
    print("=== Testing Literature Search Agent with Real Sci-Hub ===\n")
    
    # Configuration
    config = {
        'model': 'openai/gpt-4o-mini',
        'pubmed': {
            'email': 'test@example.com',
            'tool': 'ResearchAgent'
        },
        'processor': {
            'max_internal_steps': 2
        },
        'scihub': {
            'enabled': True,
            'max_papers': 3
        },
        'arxiv': {
            'enabled': True,
            'max_papers': 3
        },
        'pubmed': {
            'enabled': True,
            'max_papers': 3
        }
    }
    
    # Initialize agent
    agent = LiteratureSearchAgent(config)
    
    # Initialize and connect MCP client
    from research_agent.integrations.mcp_client import MCPClient
    mcp_client = MCPClient()
    await mcp_client.connect()
    agent.set_mcp_client(mcp_client)
    
    try:
        # Test search
        query = "machine learning medical diagnosis"
        print(f"Search Query: {query}")
        print("-" * 50)
        
        # Execute search
        print("1. Starting multi-source literature search...")
        results = await agent.search_papers(
            strategy=query,
            max_papers=9  # 3 per source
        )
        
        print(f"\nSearch completed. Total results: {len(results)}")
        
        # Analyze results by source
        source_counts = {}
        scihub_papers = []
        arxiv_papers = []
        pubmed_papers = []
        
        for paper in results:
            source = paper.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            
            if 'scihub' in source.lower() or 'sci-hub' in source.lower():
                scihub_papers.append(paper)
            elif 'arxiv' in source.lower():
                arxiv_papers.append(paper)
            elif 'pubmed' in source.lower():
                pubmed_papers.append(paper)
        
        print("\n2. Results by source:")
        for source, count in source_counts.items():
            print(f"  - {source}: {count} papers")
        
        # Check if Sci-Hub returned real data
        print("\n3. Sci-Hub Integration Analysis:")
        if scihub_papers:
            print(f"✅ Found {len(scihub_papers)} Sci-Hub papers")
            
            # Check first Sci-Hub paper for real vs mock data
            first_scihub = scihub_papers[0]
            print(f"First Sci-Hub paper:")
            print(f"  ID: {first_scihub.get('id', 'N/A')}")
            print(f"  Title: {first_scihub.get('title', 'N/A')}")
            print(f"  Authors: {first_scihub.get('authors', 'N/A')}")
            print(f"  DOI: {first_scihub.get('doi', 'N/A')}")
            print(f"  Year: {first_scihub.get('year', 'N/A')}")
            print(f"  Source: {first_scihub.get('source', 'N/A')}")
            
            # Check if this is real data
            authors_str = str(first_scihub.get('authors', ''))
            id_str = str(first_scihub.get('id', ''))
            
            if ('mock' in id_str.lower() or 
                'Author 1' in authors_str or 
                'Co-Author 1' in authors_str):
                print("❌ Still getting mock Sci-Hub data!")
            else:
                print("✅ Getting REAL Sci-Hub data!")
                
            # Show all Sci-Hub papers
            print(f"\nAll {len(scihub_papers)} Sci-Hub papers:")
            for i, paper in enumerate(scihub_papers, 1):
                print(f"  {i}. {paper.get('title', 'N/A')}")
                if paper.get('doi'):
                    print(f"     DOI: {paper.get('doi')}")
                    
        else:
            print("❌ No Sci-Hub papers found")
        
        # Show sample papers from each source
        print("\n4. Sample papers from each source:")
        
        if arxiv_papers:
            print(f"\nArXiv Sample (showing 1 of {len(arxiv_papers)}):")
            sample = arxiv_papers[0]
            print(f"  Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"  Authors: {sample.get('authors', 'N/A')}")
        
        if pubmed_papers:
            print(f"\nPubMed Sample (showing 1 of {len(pubmed_papers)}):")
            sample = pubmed_papers[0]
            print(f"  Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"  Authors: {sample.get('authors', 'N/A')}")
        
        if scihub_papers:
            print(f"\nSci-Hub Sample (showing 1 of {len(scihub_papers)}):")
            sample = scihub_papers[0]
            print(f"  Title: {sample.get('title', 'N/A')[:80]}...")
            print(f"  Authors: {sample.get('authors', 'N/A')}")
            print(f"  DOI: {sample.get('doi', 'N/A')}")
        
        # Final Assessment
        print("\n=== Final Assessment ===")
        
        total_sources = len(source_counts)
        has_real_scihub = (len(scihub_papers) > 0 and 
                          'mock' not in str(scihub_papers[0].get('id', '')).lower() and
                          'Author 1' not in str(scihub_papers[0].get('authors', '')))
        has_multiple_sources = total_sources >= 2
        
        print(f"✅ Total sources active: {total_sources}")
        print(f"✅ Real Sci-Hub data: {'Yes' if has_real_scihub else 'No'}")
        print(f"✅ Multiple sources integrated: {'Yes' if has_multiple_sources else 'No'}")
        print(f"✅ Total papers found: {len(results)}")
        
        if has_real_scihub and has_multiple_sources and len(results) > 5:
            print("\n🎉 SUCCESS: Literature search with real Sci-Hub integration is working!")
        elif has_multiple_sources and len(results) > 5:
            print("\n✅ PARTIAL SUCCESS: Multi-source search working, Sci-Hub may have connection issues")
        else:
            print("\n⚠️ Issues detected - check source integrations")
        
        # Save results for inspection
        output_file = "literature_search_real_scihub_results.json"
        with open(output_file, 'w') as f:
            # Convert to JSON serializable format
            serializable_results = []
            for paper in results:
                serializable_paper = {}
                for key, value in paper.items():
                    try:
                        json.dumps(value)  # Test if serializable
                        serializable_paper[key] = value
                    except:
                        serializable_paper[key] = str(value)
                serializable_results.append(serializable_paper)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up MCP client
        if agent.mcp_client:
            await agent.mcp_client.disconnect()

if __name__ == "__main__":
    asyncio.run(test_literature_search_with_real_scihub())