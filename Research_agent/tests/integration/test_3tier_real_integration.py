#!/usr/bin/env python3
"""
Test 3-Tier RAG with Real Sci-Hub Integration
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.integrations.three_tier_rag import ThreeTierRAG

async def test_3tier_with_real_scihub():
    """Test the 3-tier search with real Sci-Hub data"""
    print("=== Testing 3-Tier RAG with Real Sci-Hub Integration ===\n")
    
    # Initialize 3-tier RAG system
    rag = ThreeTierRAG()
    
    try:
        # Test query
        query = "machine learning applications in medical diagnosis"
        print(f"Search Query: {query}")
        print("-" * 50)
        
        # Execute 3-tier search
        print("1. Starting 3-tier search...")
        results = await rag.process_query(query)
        
        print(f"\nSearch completed. Total results: {len(results.get('combined_results', []))}")
        
        # Analyze results by source
        combined_results = results.get('combined_results', [])
        
        # Count papers by source
        source_counts = {}
        scihub_papers = []
        arxiv_papers = []
        pubmed_papers = []
        
        for paper in combined_results:
            source = paper.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            
            if 'scihub' in source.lower():
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
            
            # Check if this is real data
            if ('mock' in str(first_scihub.get('id', '')).lower() or 
                'Author 1' in str(first_scihub.get('authors', ''))):
                print("❌ Still getting mock Sci-Hub data!")
            else:
                print("✅ Getting REAL Sci-Hub data via CrossRef!")
        else:
            print("❌ No Sci-Hub papers found")
        
        # Show sample papers from each source
        print("\n4. Sample papers from each source:")
        
        if arxiv_papers:
            print(f"\nArXiv Sample (showing 1 of {len(arxiv_papers)}):")
            sample = arxiv_papers[0]
            print(f"  Title: {sample.get('title', 'N/A')}")
            print(f"  Authors: {sample.get('authors', 'N/A')}")
            print(f"  Year: {sample.get('year', 'N/A')}")
        
        if pubmed_papers:
            print(f"\nPubMed Sample (showing 1 of {len(pubmed_papers)}):")
            sample = pubmed_papers[0]
            print(f"  Title: {sample.get('title', 'N/A')}")
            print(f"  Authors: {sample.get('authors', 'N/A')}")
            print(f"  Year: {sample.get('year', 'N/A')}")
        
        if scihub_papers:
            print(f"\nSci-Hub Sample (showing 1 of {len(scihub_papers)}):")
            sample = scihub_papers[0]
            print(f"  Title: {sample.get('title', 'N/A')}")
            print(f"  Authors: {sample.get('authors', 'N/A')}")
            print(f"  Year: {sample.get('year', 'N/A')}")
            print(f"  DOI: {sample.get('doi', 'N/A')}")
        
        # Check for duplicate removal
        print("\n5. Duplicate Analysis:")
        unique_dois = set()
        unique_titles = set()
        duplicates_by_doi = 0
        duplicates_by_title = 0
        
        for paper in combined_results:
            doi = paper.get('doi', '')
            title = paper.get('title', '').lower().strip()
            
            if doi and doi in unique_dois:
                duplicates_by_doi += 1
            elif doi:
                unique_dois.add(doi)
            
            if title and title in unique_titles:
                duplicates_by_title += 1
            elif title:
                unique_titles.add(title)
        
        print(f"  Total papers: {len(combined_results)}")
        print(f"  Unique DOIs: {len(unique_dois)}")
        print(f"  Unique titles: {len(unique_titles)}")
        print(f"  DOI duplicates found: {duplicates_by_doi}")
        print(f"  Title duplicates found: {duplicates_by_title}")
        
        if duplicates_by_doi == 0 and duplicates_by_title == 0:
            print("✅ Duplicate removal working correctly!")
        else:
            print("⚠️ Some duplicates detected")
        
        print("\n=== Final Assessment ===")
        
        # Overall assessment
        total_sources = len(source_counts)
        has_real_scihub = len(scihub_papers) > 0 and 'mock' not in str(scihub_papers[0].get('id', '')).lower()
        has_multiple_sources = total_sources >= 2
        
        print(f"✅ Total sources active: {total_sources}")
        print(f"✅ Real Sci-Hub data: {'Yes' if has_real_scihub else 'No'}")
        print(f"✅ Multiple sources integrated: {'Yes' if has_multiple_sources else 'No'}")
        print(f"✅ Total papers found: {len(combined_results)}")
        
        if has_real_scihub and has_multiple_sources and len(combined_results) > 5:
            print("\n🎉 SUCCESS: 3-Tier RAG with real Sci-Hub integration is working!")
        else:
            print("\n⚠️ Partial success - some issues detected")
        
        # Save results for inspection
        output_file = "3tier_real_scihub_test_results.json"
        with open(output_file, 'w') as f:
            # Convert to JSON serializable format
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_3tier_with_real_scihub())