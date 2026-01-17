#!/usr/bin/env python3
"""
Test LightRAG Implementation
Validate the real LightRAG integration in the 3-tier system
"""

import asyncio
import sys
import json
import logging
from pathlib import Path

# Add the research agent to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_lightrag_integration():
    """Test the real LightRAG integration"""
    logger.info("🚀 Testing LightRAG Real Implementation")
    
    try:
        from research_agent.integrations.three_tier_rag import ThreeTierRAG, RAGTier
        
        # Initialize with test configuration
        config = {
            'lightrag_working_dir': './test_lightrag_data',
            'lightrag_llm_model': 'gpt-4o-mini',
            'embedding_dim': 1536,
            'max_token_size': 8192,
            'lightrag_query_param': 'naive'
        }
        
        # Create 3-tier RAG system
        rag_system = ThreeTierRAG(config)
        logger.info(f"✅ Created 3-tier RAG system")
        
        # Check tier availability
        available_tiers = rag_system.get_available_tiers()
        logger.info(f"Available tiers: {[tier.value for tier in available_tiers]}")
        
        # Check if LightRAG is available
        if RAGTier.TIER1_LIGHTRAG not in available_tiers:
            logger.error("❌ LightRAG tier not available")
            return False
        
        # Get tier status
        tier_status = rag_system.get_tier_status()
        lightrag_status = tier_status[RAGTier.TIER1_LIGHTRAG]
        logger.info(f"LightRAG status: {lightrag_status}")
        
        # Insert some test documents
        test_documents = [
            "Machine learning is a branch of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.",
            "Deep learning is a subset of machine learning based on artificial neural networks with representation learning.",
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
        ]
        
        logger.info(f"📝 Inserting {len(test_documents)} test documents...")
        insert_success = await rag_system.insert_documents(test_documents, RAGTier.TIER1_LIGHTRAG)
        
        if not insert_success:
            logger.error("❌ Failed to insert documents into LightRAG")
            return False
        
        logger.info("✅ Documents inserted successfully")
        
        # Test query processing
        test_query = "What is machine learning?"
        logger.info(f"🔍 Testing query: '{test_query}'")
        
        results = await rag_system.process_query(test_query, [RAGTier.TIER1_LIGHTRAG])
        
        if not results:
            logger.error("❌ No results returned from query")
            return False
        
        result = results[0]
        logger.info(f"✅ Query processed successfully")
        logger.info(f"Result success: {result.success}")
        logger.info(f"Result content length: {len(result.content)}")
        logger.info(f"Processing time: {result.processing_time:.3f}s")
        logger.info(f"Confidence: {result.confidence}")
        logger.info(f"Metadata: {result.metadata}")
        
        if result.success and result.metadata.get('processor') == 'actual_lightrag':
            logger.info("🎉 LightRAG integration test PASSED!")
            return True
        else:
            logger.error("❌ LightRAG integration test FAILED!")
            logger.error(f"Result: {result}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test execution"""
    success = await test_lightrag_integration()
    
    if success:
        print("\n✅ LightRAG Implementation Test: PASSED")
    else:
        print("\n❌ LightRAG Implementation Test: FAILED")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)