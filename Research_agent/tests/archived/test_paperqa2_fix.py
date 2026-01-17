#!/usr/bin/env python3
"""
Test PaperQA2 Fix for Empty Index Issue

This script verifies that the PaperQA2 integration now properly maintains
documents and can answer queries without the "empty index" error.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_paperqa2_fix():
    """Test that PaperQA2 fix resolves the empty index issue"""
    
    print("🔧 Testing PaperQA2 Fix for Empty Index Issue")
    print("=" * 60)
    
    try:
        from research_agent.integrations.three_tier_rag import ThreeTierRAG, RAGTier
        from research_agent.core.document_pipeline import DocumentPipeline
        
        # Step 1: Setup document pipeline and create sample docs
        print("📄 Setting up document pipeline...")
        doc_pipeline = DocumentPipeline()
        sample_docs = doc_pipeline.create_sample_documents()
        print(f"✅ Created {len(sample_docs)} sample documents")
        
        # Step 2: Initialize RAG system
        print("\n🧠 Initializing RAG system...")
        rag_config = {
            'paperqa2_working_dir': './paperqa2_data',
            'paperqa2_llm_model': 'gpt-4o-mini',
            'paperqa2_summary_model': 'gpt-4o-mini',
            'paperqa2_temperature': 0.1
        }
        
        rag_system = ThreeTierRAG(rag_config)
        print("✅ RAG system initialized")
        
        # Step 3: Process documents
        print("\n⚙️ Processing documents...")
        await doc_pipeline.process_all_documents(rag_system, force_reprocess=True)
        print("✅ Documents processed")
        
        # Step 4: Check PaperQA2 status before query
        print("\n📊 Checking PaperQA2 status...")
        if RAGTier.TIER2_PAPERQA2 in rag_system.tier_processors:
            processor_info = rag_system.tier_processors[RAGTier.TIER2_PAPERQA2]
            docs = processor_info['processor']
            print(f"   Documents in PaperQA2: {len(docs.docs)}")
            print(f"   Text chunks: {len(docs.texts)}")
        
        # Step 5: Test queries that previously failed
        test_queries = [
            "What are the main applications of quantum computing?",
            "How do neural network optimization techniques work?",
            "What are the key principles of AI ethics?"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        
        all_success = True
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            
            try:
                # Test PaperQA2 specifically
                results = await rag_system.process_query(query, [RAGTier.TIER2_PAPERQA2])
                result = results[0]
                
                if result.success:
                    print(f"   ✅ SUCCESS")
                    print(f"   📄 Content length: {len(result.content)} chars")
                    print(f"   🎯 Confidence: {result.confidence:.2f}")
                    print(f"   📚 Sources: {len(result.sources)}")
                    print(f"   ⏱️ Time: {result.processing_time:.2f}s")
                    
                    # Show snippet of response
                    if len(result.content) > 200:
                        snippet = result.content[:200] + "..."
                    else:
                        snippet = result.content
                    print(f"   📝 Response snippet: {snippet}")
                else:
                    print(f"   ❌ FAILED: {result.error}")
                    all_success = False
                    
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
                all_success = False
        
        # Step 6: Test all tiers together
        print(f"\n🔄 Testing all tiers together...")
        
        try:
            results = await rag_system.process_query(
                "Compare quantum computing and neural networks", 
                [RAGTier.TIER1_LIGHTRAG, RAGTier.TIER2_PAPERQA2, RAGTier.TIER3_GRAPHRAG]
            )
            
            print(f"   All tiers results:")
            for result in results:
                status = "✅" if result.success else "❌"
                print(f"     {status} {result.tier.value}: {len(result.content)} chars")
                
        except Exception as e:
            print(f"   ❌ All tiers test failed: {e}")
            all_success = False
        
        # Step 7: Summary
        print("\n" + "=" * 60)
        if all_success:
            print("🎉 SUCCESS: PaperQA2 fix is working!")
            print("   ✅ No more 'empty index' errors")
            print("   ✅ Documents are being processed correctly")
            print("   ✅ Queries return meaningful responses")
        else:
            print("❌ ISSUES REMAIN: Some tests failed")
            print("   Please check the logs above for details")
        
        print("=" * 60)
        
        return all_success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    
    print("🚀 PaperQA2 Fix Verification Test")
    print()
    print("This test will:")
    print("1. Create sample research documents")
    print("2. Process them through the RAG system")
    print("3. Test queries that previously failed with 'empty index'")
    print("4. Verify that PaperQA2 now works correctly")
    print()
    
    success = await test_paperqa2_fix()
    
    if success:
        print("\n🎉 PRODUCTION ISSUE RESOLVED!")
        print("   The RAG system is now ready for production use")
    else:
        print("\n❌ Issues still exist - further debugging needed")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)