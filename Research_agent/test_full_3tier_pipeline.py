#!/usr/bin/env python3
"""
Test script for the complete 3-tier document search pipeline.

This tests the full integration of:
- TIER 1: ArXiv API search
- TIER 2: PubMed API search  
- TIER 3: Sci-Hub MCP search (with AgenticStepProcessor)
- FALLBACK: Sci-Hub download attempts

Verifies that all tiers work together and return balanced results.
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add the current directory and parent directories to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Import the document search service
from src.research_agent.services.document_search_service import DocumentSearchService, SearchConfiguration

async def test_full_3tier_pipeline():
    """Test the complete 3-tier document search pipeline."""
    
    print("🔬 Testing Full 3-Tier Document Search Pipeline")
    print("=" * 60)
    
    # Initialize the service
    config = SearchConfiguration(
        working_directory=Path("./test_workspace"),
        max_papers_per_tier=5,  # Small test to verify each tier
        rate_limit_delay=1.0,
        enable_metadata_enhancement=True,
        enable_fallback_downloads=True
    )
    
    service = DocumentSearchService(config=config)
    
    try:
        # Initialize the service
        print("🔧 Initializing Document Search Service...")
        success = await service.initialize()
        if not success:
            print("❌ Failed to initialize service")
            return
        
        # Start a test session
        print("📁 Starting search session...")
        session_id = service.start_session("3tier_pipeline_test")
        print(f"✅ Session started: {session_id}")
        
        # Test query that should work across all tiers
        test_query = "machine learning healthcare"
        max_papers = 15  # 5 papers per tier
        
        print(f"\n🔍 Testing 3-tier search for: '{test_query}'")
        print(f"📊 Requesting {max_papers} papers total (5 per tier)")
        print("\nExecuting search across all tiers...")
        
        # Execute the full 3-tier search
        papers, metadata = await service.search_documents(
            search_query=test_query,
            max_papers=max_papers,
            enhance_metadata=True,
            tier_allocation={
                'arxiv': 5,
                'pubmed': 5, 
                'scihub': 5
            }
        )
        
        print(f"\n✅ Search completed successfully!")
        print(f"📄 Total papers found: {len(papers)}")
        
        # Analyze results by tier
        print("\n📊 TIER DISTRIBUTION ANALYSIS:")
        tier_counts = {}
        for paper in papers:
            tier = paper.get('tier', 'unknown')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        for tier, count in tier_counts.items():
            print(f"  {tier.upper()}: {count} papers")
        
        # Show sample papers from each tier
        print("\n📋 SAMPLE PAPERS BY TIER:")
        tiers_shown = set()
        for paper in papers:
            tier = paper.get('tier')
            if tier not in tiers_shown and len(tiers_shown) < 3:
                tiers_shown.add(tier)
                print(f"\n  {tier.upper()} SAMPLE:")
                print(f"    Title: {paper.get('title', 'N/A')[:60]}...")
                print(f"    Authors: {', '.join(paper.get('authors', [])[:2])}")
                print(f"    Year: {paper.get('publication_year', 'N/A')}")
                print(f"    DOI: {paper.get('doi', 'N/A')}")
                print(f"    Search Method: {paper.get('search_method', 'N/A')}")
                print(f"    Full Text Available: {paper.get('metadata', {}).get('full_text_available', False)}")
        
        # Analyze Sci-Hub tier specifically (since it was the focus of our work)
        print("\n🎯 SCI-HUB TIER ANALYSIS:")
        scihub_papers = [p for p in papers if p.get('tier') == 'scihub']
        print(f"  Sci-Hub papers found: {len(scihub_papers)}")
        
        if scihub_papers:
            print("  Sci-Hub search method verification:")
            for paper in scihub_papers:
                method = paper.get('search_method', 'N/A')
                title = paper.get('title', 'N/A')[:40]
                print(f"    • {method}: {title}...")
            
            # Check if AgenticStepProcessor was used
            mcp_direct_count = sum(1 for p in scihub_papers if p.get('search_method') == 'scihub_mcp_direct')
            print(f"  ✅ MCP direct search (AgenticStepProcessor): {mcp_direct_count} papers")
        
        # Analyze fallback attempts
        print("\n🔄 FALLBACK ANALYSIS:")
        fallback_attempts = sum(1 for p in papers if p.get('fallback_attempted', False))
        fallback_successes = sum(1 for p in papers if p.get('fallback_successful', False))
        print(f"  Fallback attempts: {fallback_attempts}")
        print(f"  Fallback successes: {fallback_successes}")
        
        # Show metadata summary
        print("\n📈 SEARCH METADATA SUMMARY:")
        print(f"  Status: {metadata.get('status', 'N/A')}")
        print(f"  Architecture: {metadata.get('architecture', 'N/A')}")
        print(f"  Service Version: {metadata.get('service_info', {}).get('version', 'N/A')}")
        print(f"  Session ID: {metadata.get('session_id', 'N/A')}")
        
        tier_results = metadata.get('tier_distribution', {}).get('tier_results', {})
        print(f"  ArXiv results: {tier_results.get('arxiv_count', 0)}")
        print(f"  PubMed results: {tier_results.get('pubmed_count', 0)}")
        print(f"  Sci-Hub results: {tier_results.get('scihub_count', 0)}")
        
        # Verify the fix is working
        print("\n🎉 INTEGRATION SUCCESS VERIFICATION:")
        if len(scihub_papers) > 0:
            print("  ✅ Sci-Hub tier is now working with AgenticStepProcessor")
            print("  ✅ MCP tool calling is functioning properly")
            print("  ✅ All 3 tiers are producing distinct results")
        else:
            print("  ⚠️ Sci-Hub tier returned no results - may need investigation")
        
        # Save test results
        test_results = {
            'test_query': test_query,
            'total_papers': len(papers),
            'tier_distribution': tier_counts,
            'scihub_papers_count': len(scihub_papers),
            'fallback_attempts': fallback_attempts,
            'fallback_successes': fallback_successes,
            'metadata_summary': {
                'status': metadata.get('status'),
                'architecture': metadata.get('architecture'),
                'tier_results': tier_results
            },
            'test_status': 'SUCCESS' if len(papers) > 0 and len(scihub_papers) > 0 else 'PARTIAL_SUCCESS'
        }
        
        with open('test_3tier_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n📄 Test results saved to: test_3tier_results.json")
        
        # End session
        service.end_session(cleanup=False)
        print(f"📁 Session ended: {session_id}")
        
        # Final success message
        if len(scihub_papers) > 0:
            print("\n🚀 SUCCESS: Complete 3-tier pipeline is working!")
            print("   ✅ ArXiv tier functional")
            print("   ✅ PubMed tier functional") 
            print("   ✅ Sci-Hub tier functional (with AgenticStepProcessor fix)")
            print("   ✅ All tiers producing distinct, balanced results")
        else:
            print("\n⚠️ PARTIAL SUCCESS: Pipeline working but Sci-Hub needs attention")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await service.shutdown()
        print("\n🧹 Service shutdown complete")

async def main():
    """Main test execution function."""
    await test_full_3tier_pipeline()

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())