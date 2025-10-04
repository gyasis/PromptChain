#!/usr/bin/env python3
"""
Final Integration Verification Test

This script performs a direct test to verify that the MultiQueryCoordinator 
is using the real ThreeTierRAG system instead of placeholder implementations.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from research_agent.integrations.multi_query_coordinator import MultiQueryCoordinator
from research_agent.integrations.three_tier_rag import RAGTier, ThreeTierRAG
from research_agent.core.session import Query, Paper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_real_integration():
    """Direct test to verify real RAG integration"""
    logger.info("🔍 Direct Real Integration Verification Test")
    
    # Initialize coordinator with real RAG
    config = {
        'coordination': {'model': 'openai/gpt-4o-mini'},
        'three_tier_rag': {
            'paperqa2_working_dir': './verification_paperqa2',
            'graphrag_working_dir': './verification_graphrag'
        }
    }
    
    coordinator = MultiQueryCoordinator(config)
    await coordinator.initialize_tiers()
    
    # Check that we have real three_tier_rag instance
    logger.info(f"ThreeTierRAG instance present: {coordinator.three_tier_rag is not None}")
    logger.info(f"ThreeTierRAG type: {type(coordinator.three_tier_rag)}")
    
    # Check available tiers
    available_tiers = coordinator._get_available_tiers()
    logger.info(f"Available tiers: {[tier.value for tier in available_tiers]}")
    
    # Test direct processing with real system
    if coordinator.three_tier_rag and available_tiers:
        logger.info("Testing direct real RAG processing...")
        
        # Process a simple query directly through real RAG
        test_query = "What is artificial intelligence?"
        rag_results = await coordinator.three_tier_rag.process_query(
            test_query, available_tiers[:1]  # Use first available tier
        )
        
        for result in rag_results:
            logger.info(f"✅ Real RAG Result:")
            logger.info(f"  Tier: {result.tier.value}")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Processor: {result.metadata.get('processor', 'unknown')}")
            logger.info(f"  Content length: {len(result.content)}")
            
            # Verify this is real processing
            if result.metadata.get('processor', '').startswith('actual_'):
                logger.info(f"✅ CONFIRMED: Real processing detected - {result.metadata['processor']}")
            else:
                logger.warning(f"⚠️ Processing type unclear: {result.metadata.get('processor')}")
    
    # Test through coordinator processing
    logger.info("\nTesting through MultiQueryCoordinator...")
    
    test_papers = [Paper(
        id="verification_paper",
        title="AI Verification Test",
        abstract="Test paper for verification.",
        authors=["Test Author"],
        source="test",
        url="http://test.com/verification"
    )]
    
    test_queries = [Query(
        id="verification_query",
        text="What is machine learning?",
        priority=1.0,
        iteration=1
    )]
    
    coordinator_results = await coordinator.process_papers_with_queries(test_papers, test_queries)
    
    logger.info(f"\nCoordinator Processing Results: {len(coordinator_results)}")
    
    # Analyze results for real processing indicators
    real_processing_count = 0
    for result in coordinator_results:
        result_data = result.result_data
        metadata = result_data.get('metadata', {})
        processor = metadata.get('processor', 'unknown')
        
        logger.info(f"Result - Tier: {result.tier}, Success: {result_data.get('success', False)}, Processor: {processor}")
        
        if processor.startswith('actual_'):
            real_processing_count += 1
            logger.info(f"✅ Real processor detected: {processor}")
    
    # Final verification
    logger.info(f"\n🎯 FINAL VERIFICATION RESULTS:")
    logger.info(f"  Real ThreeTierRAG integrated: {coordinator.three_tier_rag is not None}")
    logger.info(f"  Available tiers: {len(available_tiers)}")
    logger.info(f"  Results with real processing: {real_processing_count}/{len(coordinator_results)}")
    
    # Check for placeholder elimination
    placeholder_indicators = ['placeholder', 'simulate', 'mock', 'sleep']
    has_placeholders = False
    
    for result in coordinator_results:
        result_str = str(result.result_data)
        if any(indicator in result_str.lower() for indicator in placeholder_indicators):
            has_placeholders = True
            break
    
    logger.info(f"  Placeholder processing eliminated: {not has_placeholders}")
    
    # Integration status
    if coordinator.three_tier_rag and len(available_tiers) > 0 and not has_placeholders:
        status = "✅ FULLY INTEGRATED"
    elif coordinator.three_tier_rag and len(available_tiers) > 0:
        status = "⚠️ MOSTLY INTEGRATED" 
    else:
        status = "❌ INTEGRATION INCOMPLETE"
    
    logger.info(f"  Integration Status: {status}")
    
    return {
        'real_rag_integrated': coordinator.three_tier_rag is not None,
        'available_tiers': len(available_tiers),
        'real_processing_results': real_processing_count,
        'total_results': len(coordinator_results),
        'placeholders_eliminated': not has_placeholders,
        'integration_status': status
    }


if __name__ == "__main__":
    results = asyncio.run(test_real_integration())
    print(f"\n🏁 VERIFICATION COMPLETE: {json.dumps(results, indent=2)}")