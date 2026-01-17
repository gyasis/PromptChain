#!/usr/bin/env python3
"""
Test Enhanced Sci-Hub MCP Integration

Tests the new static prompt chain functionality for iterative Sci-Hub title searches
and PDF downloading with source balancing.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.config import ResearchConfig


async def test_enhanced_scihub_integration():
    """Test the enhanced Sci-Hub integration with static prompt chains"""
    
    print("🧪 Testing Enhanced Sci-Hub MCP Integration")
    print("=" * 60)
    
    try:
        # Initialize Research Config with enhanced settings
        config = ResearchConfig()
        
        # Configure for testing
        config.research_session.max_papers_total = 15
        config.research_session.max_iterations = 1
        
        # Add enhanced configuration
        config_dict = config.__dict__
        config_dict['pdf_storage'] = './test_downloads/enhanced_scihub'
        config_dict['source_balance'] = {
            'arxiv': 0.4,  # Target 40% ArXiv
            'pubmed': 0.4,  # Target 40% PubMed  
            'sci_hub': 0.2  # Target 20% Sci-Hub exclusive
        }
        
        # Test with a neurological disease query (known to have PubMed papers)
        test_query = "neurological gait analysis Parkinson disease detection"
        
        print(f"🔍 Test Query: '{test_query}'")
        print(f"📊 Target Distribution: ArXiv 40%, PubMed 40%, Sci-Hub 20%")
        print()
        
        # Initialize orchestrator
        print("🚀 Initializing Enhanced Research Orchestrator...")
        orchestrator = AdvancedResearchOrchestrator(config)
        
        # Initialize MCP client
        print("🔗 Connecting to Sci-Hub MCP server...")
        await orchestrator.initialize_mcp_client()
        
        if orchestrator.mcp_client and orchestrator.mcp_client.connected:
            tools = orchestrator.mcp_client.get_available_tools()
            print(f"✅ MCP Connection Status: CONNECTED")
            print(f"   Available tools: {tools}")
        else:
            print("⚠️  MCP Connection Status: FALLBACK MODE")
        
        print()
        
        # Get enhanced literature searcher
        searcher = orchestrator.literature_searcher
        print(f"📁 PDF Storage Path: {searcher.pdf_storage_path}")
        print(f"🎯 Source Balance Targets: {searcher.source_targets}")
        print()
        
        # Test the enhanced search with metrics tracking
        print("🔍 Executing Enhanced Literature Search...")
        print("-" * 40)
        
        # Reset metrics
        searcher.reset_metrics()
        
        start_time = datetime.now()
        papers = await searcher.search_papers(
            test_query,
            max_papers=15
        )
        end_time = datetime.now()
        
        print(f"⏱️  Search completed in {(end_time - start_time).total_seconds():.1f} seconds")
        print()
        
        # Analyze results
        if papers:
            print(f"📊 SEARCH RESULTS ANALYSIS")
            print("=" * 40)
            
            # Source distribution analysis
            source_counts = {}
            pdf_available_count = 0
            scihub_enhanced_count = 0
            
            for paper in papers:
                source = paper.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
                
                if paper.get('full_text_available'):
                    pdf_available_count += 1
                
                if paper.get('metadata', {}).get('scihub_enhanced'):
                    scihub_enhanced_count += 1
            
            # Display source distribution
            total_papers = len(papers)
            print(f"Total Papers Found: {total_papers}")
            print()
            print("Source Distribution:")
            for source, count in source_counts.items():
                percentage = (count / total_papers) * 100
                print(f"  {source.upper()}: {count} papers ({percentage:.1f}%)")
            print()
            
            # Display PDF availability
            pdf_rate = (pdf_available_count / total_papers) * 100
            print(f"PDF Availability: {pdf_available_count}/{total_papers} ({pdf_rate:.1f}%)")
            print(f"Sci-Hub Enhanced: {scihub_enhanced_count} papers")
            print()
            
            # Display comprehensive metrics
            metrics = searcher.get_search_metrics()
            print("📈 COMPREHENSIVE METRICS")
            print("=" * 30)
            print(json.dumps(metrics, indent=2))
            print()
            
            # Show sample papers with details
            print("📄 SAMPLE PAPERS (First 5)")
            print("=" * 40)
            for i, paper in enumerate(papers[:5], 1):
                title = paper.get('title', 'No title')[:60]
                source = paper.get('source', 'Unknown').upper()
                year = paper.get('publication_year', 'Unknown')
                has_pdf = "📄 PDF" if paper.get('full_text_available') else "📋 Abstract"
                enhanced = "🔬 Enhanced" if paper.get('metadata', {}).get('scihub_enhanced') else ""
                
                print(f"{i:2d}. {title}...")
                print(f"    📍 {source} | {year} | {has_pdf} {enhanced}")
                
                # Show PDF path if available
                pdf_path = paper.get('pdf_path')
                if pdf_path:
                    print(f"    💾 PDF: {pdf_path}")
                
                # Show Sci-Hub search details if available
                metadata = paper.get('metadata', {})
                if metadata.get('scihub_search_attempted'):
                    method = metadata.get('scihub_search_method', 'unknown')
                    success = "✅ Success" if metadata.get('scihub_enhanced') else "❌ Failed"
                    print(f"    🔬 Sci-Hub Search: {method} - {success}")
                
                print()
            
            # Test source balancing effectiveness
            print("⚖️  SOURCE BALANCING ANALYSIS")
            print("=" * 35)
            arxiv_ratio = source_counts.get('arxiv', 0) / total_papers
            pubmed_ratio = source_counts.get('pubmed', 0) / total_papers
            scihub_ratio = source_counts.get('sci_hub', 0) / total_papers
            
            target_arxiv = searcher.source_targets.get('arxiv', 0.4)
            target_pubmed = searcher.source_targets.get('pubmed', 0.35)
            target_scihub = searcher.source_targets.get('sci_hub', 0.25)
            
            print(f"ArXiv:  {arxiv_ratio:.2f} (target: {target_arxiv:.2f}) {'✅' if abs(arxiv_ratio - target_arxiv) < 0.15 else '⚠️'}")
            print(f"PubMed: {pubmed_ratio:.2f} (target: {target_pubmed:.2f}) {'✅' if abs(pubmed_ratio - target_pubmed) < 0.15 else '⚠️'}")
            print(f"Sci-Hub: {scihub_ratio:.2f} (target: {target_scihub:.2f}) {'✅' if abs(scihub_ratio - target_scihub) < 0.15 else '⚠️'}")
            
            # Check for ArXiv bias
            if arxiv_ratio > 0.6:
                print("🚨 WARNING: ArXiv bias detected! Consider adjusting search strategy.")
            else:
                print("✅ Source distribution appears balanced")
            
        else:
            print("❌ No papers found!")
            print("💡 This could indicate:")
            print("   • MCP connection issues")
            print("   • API rate limiting")
            print("   • Query too restrictive")
        
        # Test enhanced functionality directly
        print("\n🧪 TESTING ENHANCED FEATURES")
        print("=" * 40)
        
        # Test static prompt chain creation
        if hasattr(searcher, 'scihub_chain'):
            print("✅ Static Prompt Chain: Initialized")
        else:
            print("❌ Static Prompt Chain: Missing")
        
        # Test PDF download directory
        if os.path.exists(searcher.pdf_storage_path):
            pdf_files = [f for f in os.listdir(searcher.pdf_storage_path) if f.endswith('.pdf')]
            print(f"✅ PDF Storage: {len(pdf_files)} files in {searcher.pdf_storage_path}")
        else:
            print(f"⚠️  PDF Storage: Directory {searcher.pdf_storage_path} not found")
        
        # Test metrics tracking
        if hasattr(searcher, 'metrics'):
            print("✅ Metrics Tracking: Active")
            print(f"   📊 Papers identified: {sum(searcher.metrics.papers_identified.values())}")
            print(f"   📄 PDFs retrieved: {searcher.metrics.pdfs_retrieved}")
            print(f"   📈 Success rate: {searcher.metrics.calculate_download_rate():.2f}")
        else:
            print("❌ Metrics Tracking: Missing")
        
        print("\n🔄 Cleaning up...")
        await orchestrator.shutdown()
        
        print("\n✅ Enhanced Sci-Hub Integration Test Complete!")
        
    except Exception as e:
        import traceback
        print(f"\n❌ Test failed with error: {e}")
        print(f"🐛 Full traceback:\n{traceback.format_exc()}")
        
        # Cleanup on error
        try:
            if 'orchestrator' in locals():
                await orchestrator.shutdown()
        except:
            pass


async def test_simple_workflow():
    """Test with the simple search_papers.py workflow"""
    print("\n" + "=" * 60)
    print("🔬 Testing Simple Workflow Integration")
    print("=" * 60)
    
    # Import the search function from search_papers.py
    try:
        # Import search_papers module
        import search_papers
        
        print("📋 Running search_papers with enhanced integration...")
        await search_papers.search_papers("machine learning gait analysis", 10)
        print("✅ Simple workflow test completed")
        
    except Exception as e:
        print(f"❌ Simple workflow test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting Enhanced Sci-Hub MCP Integration Tests")
    print("=" * 60)
    
    # Run main test
    asyncio.run(test_enhanced_scihub_integration())
    
    # Run simple workflow test
    asyncio.run(test_simple_workflow())
    
    print("\n🎉 All tests completed!")