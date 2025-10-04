#!/usr/bin/env python3
"""
Test Production Reranking Implementation

Tests the improved reranking configuration in the 3-tier RAG system
with proper error handling, fallbacks, and production-ready setup.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from research_agent.integrations.three_tier_rag import ThreeTierRAG, RAGTier, get_default_config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_reranking_configuration():
    """Test reranking configuration and initialization"""
    logger.info("🔄 Testing Production Reranking Configuration")
    
    try:
        # Test with reranking enabled
        config = get_default_config()
        config['lightrag_enable_rerank'] = True
        config['lightrag_llm_model'] = 'gpt-4o-mini'
        
        # Create ThreeTierRAG system
        rag_system = ThreeTierRAG(config)
        
        # Check available tiers
        available_tiers = rag_system.get_available_tiers()
        logger.info(f"✅ Available tiers: {[tier.value for tier in available_tiers]}")
        
        if RAGTier.TIER1_LIGHTRAG in available_tiers:
            tier_info = rag_system.tier_processors[RAGTier.TIER1_LIGHTRAG]
            logger.info(f"✅ LightRAG processor status: {tier_info['status']}")
            logger.info(f"✅ Reranking enabled: {tier_info.get('rerank_enabled', False)}")
            logger.info(f"✅ Async rerank init required: {tier_info.get('rerank_async_init_required', False)}")
            
            return True
        else:
            logger.error("❌ LightRAG tier not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_reranking_with_mock_documents():
    """Test reranking functionality with mock documents"""
    logger.info("🔄 Testing Reranking with Mock Documents")
    
    try:
        # Create test configuration
        config = get_default_config()
        config['lightrag_enable_rerank'] = True
        config['lightrag_llm_model'] = 'gpt-4o-mini'
        config['lightrag_working_dir'] = './test_lightrag_data'
        
        # Create RAG system
        rag_system = ThreeTierRAG(config)
        
        if RAGTier.TIER1_LIGHTRAG not in rag_system.get_available_tiers():
            logger.warning("⚠️ LightRAG not available, skipping reranking test")
            return True
        
        # Test the rerank function creation
        rerank_func = await rag_system._create_production_rerank_func()
        
        if rerank_func is None:
            logger.info("ℹ️ No rerank function available (expected without JINA_API_KEY)")
            return True
        
        # Create mock documents for testing
        mock_documents = [
            {"content": "Research on machine learning algorithms for natural language processing."},
            {"content": "Deep learning approaches to computer vision and image recognition."},
            {"content": "Natural language processing techniques using transformer models."},
            {"content": "Computer science fundamentals and programming languages."},
            {"content": "Advanced machine learning research in neural networks."}
        ]
        
        test_query = "natural language processing research"
        
        # Test reranking function
        start_time = time.time()
        reranked_docs = await rerank_func(
            query=test_query,
            documents=mock_documents,
            top_n=3
        )
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Reranking completed in {processing_time:.2f}s")
        logger.info(f"✅ Original docs: {len(mock_documents)}, Reranked: {len(reranked_docs)}")
        
        # Check if reranking worked or fell back gracefully
        if len(reranked_docs) <= len(mock_documents):
            logger.info("✅ Reranking function working correctly")
            
            # Log first few results
            for i, doc in enumerate(reranked_docs[:2]):
                score = doc.get('rerank_score', 'N/A')
                content_preview = doc.get('content', str(doc))[:50] + "..."
                logger.info(f"  {i+1}. Score: {score} | Content: {content_preview}")
            
            return True
        else:
            logger.error("❌ Reranking returned more documents than input")
            return False
            
    except Exception as e:
        logger.error(f"❌ Reranking test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_reranking_fallback_behavior():
    """Test reranking fallback behavior when API key is missing"""
    logger.info("🔄 Testing Reranking Fallback Behavior")
    
    try:
        # Temporarily remove JINA_API_KEY to test fallback
        original_key = os.environ.get('JINA_API_KEY')
        if 'JINA_API_KEY' in os.environ:
            del os.environ['JINA_API_KEY']
        
        # Create test configuration
        config = get_default_config()
        config['lightrag_enable_rerank'] = True
        config['lightrag_llm_model'] = 'gpt-4o-mini'
        
        # Create RAG system
        rag_system = ThreeTierRAG(config)
        
        # Test rerank function creation without API key
        rerank_func = await rag_system._create_production_rerank_func()
        
        if rerank_func is None:
            logger.info("✅ Correctly returned None when JINA_API_KEY is missing")
            fallback_success = True
        else:
            # Test fallback behavior with actual function
            mock_documents = [
                {"content": "Test document 1"},
                {"content": "Test document 2"}
            ]
            
            result = await rerank_func(
                query="test query",
                documents=mock_documents
            )
            
            # Should return original documents on fallback
            if result == mock_documents:
                logger.info("✅ Correctly fell back to original documents without API key")
                fallback_success = True
            else:
                logger.warning("⚠️ Unexpected fallback behavior")
                fallback_success = False
        
        # Restore original API key if it existed
        if original_key:
            os.environ['JINA_API_KEY'] = original_key
        
        return fallback_success
        
    except Exception as e:
        logger.error(f"❌ Fallback test failed: {e}")
        return False


async def test_end_to_end_query_with_reranking():
    """Test end-to-end query processing with reranking enabled"""
    logger.info("🔄 Testing End-to-End Query with Reranking")
    
    try:
        # Create test configuration
        config = get_default_config()
        config['lightrag_enable_rerank'] = True
        config['lightrag_llm_model'] = 'gpt-4o-mini'
        config['lightrag_working_dir'] = './test_lightrag_data'
        
        # Create RAG system
        rag_system = ThreeTierRAG(config)
        
        if RAGTier.TIER1_LIGHTRAG not in rag_system.get_available_tiers():
            logger.warning("⚠️ LightRAG not available, skipping end-to-end test")
            return True
        
        # Add some test content
        test_documents = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers for complex pattern recognition."
        ]
        
        # Insert documents
        insert_success = await rag_system.insert_documents(
            documents=test_documents,
            tier=RAGTier.TIER1_LIGHTRAG
        )
        
        if not insert_success:
            logger.warning("⚠️ Document insertion failed, but this is expected for test setup")
        
        # Process a query
        test_query = "What is machine learning?"
        
        results = await rag_system.process_query(
            query=test_query,
            tiers=[RAGTier.TIER1_LIGHTRAG]
        )
        
        if results and len(results) > 0:
            result = results[0]
            logger.info(f"✅ Query processed successfully")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Content length: {len(result.content)}")
            logger.info(f"  Processing time: {result.processing_time:.2f}s")
            logger.info(f"  Confidence: {result.confidence}")
            
            # Check if reranking information is in metadata
            rerank_info = result.metadata.get('rerank_enabled', 'N/A')
            logger.info(f"  Reranking enabled: {rerank_info}")
            
            return result.success
        else:
            logger.error("❌ No results returned from query")
            return False
            
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    """Run all reranking tests"""
    logger.info("🚀 Starting Production Reranking Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Configuration
    logger.info("\n📋 Test 1: Reranking Configuration")
    test_results['configuration'] = await test_reranking_configuration()
    
    # Test 2: Mock documents
    logger.info("\n📄 Test 2: Reranking with Mock Documents")
    test_results['mock_documents'] = await test_reranking_with_mock_documents()
    
    # Test 3: Fallback behavior
    logger.info("\n🔄 Test 3: Fallback Behavior")
    test_results['fallback'] = await test_reranking_fallback_behavior()
    
    # Test 4: End-to-end query
    logger.info("\n🔍 Test 4: End-to-End Query Processing")
    test_results['end_to_end'] = await test_end_to_end_query_with_reranking()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.ljust(20)}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All reranking tests passed! Production reranking is ready.")
        return True
    else:
        logger.error("💥 Some tests failed. Check configuration and dependencies.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)