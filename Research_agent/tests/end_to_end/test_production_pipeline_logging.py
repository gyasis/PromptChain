#!/usr/bin/env python3
"""
Production Pipeline Error Logging Test

This test runs the actual Research Agent pipeline with enhanced logging
to capture real LLM responses that cause JSON parsing errors in production.
The goal is to provide human engineers with exact output banners when issues occur.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from research_agent.core.orchestrator import AdvancedResearchOrchestrator
from research_agent.core.config import ResearchConfig

# Configure logging to capture our enhanced banners
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('production_error_logging_test.log')
    ]
)

logger = logging.getLogger(__name__)


async def test_production_pipeline_with_enhanced_logging():
    """Test the actual production pipeline to capture real error scenarios"""
    print("🚀 PRODUCTION PIPELINE ENHANCED LOGGING TEST")
    print("=" * 70)
    print("Testing real Research Agent pipeline with enhanced error capture")
    print("This will help identify the root cause of persistent JSON errors")
    print()
    
    try:
        # Create research configuration with minimal resources to increase likelihood of errors
        config = ResearchConfig()
        config.research_session.max_papers_total = 5  # Small to reduce complexity
        config.research_session.max_iterations = 1    # Single iteration
        config.research_session.source_filter = 'sci_hub'  # Test source isolation
        
        # Initialize orchestrator
        print("🔧 Initializing Research Orchestrator...")
        orchestrator = AdvancedResearchOrchestrator(config)
        
        # Test topics that might cause different types of LLM responses
        test_topics = [
            # Simple topic - should work
            "machine learning optimization",
            
            # Complex topic - might cause token limit issues
            "quantum computational complexity theory with applications in cryptographic protocol verification and distributed consensus mechanisms",
            
            # Ambiguous topic - might cause LLM confusion
            "AI ethics",
            
            # Very specific topic - might yield few results
            "LSTM gradient vanishing solutions in transformer architectures",
        ]
        
        for i, topic in enumerate(test_topics, 1):
            print(f"\n{'='*50}")
            print(f"TEST {i}: {topic[:50]}{'...' if len(topic) > 50 else ''}")
            print(f"{'='*50}")
            
            try:
                # Add extra logging context
                logger.info(f"🎯 STARTING PRODUCTION TEST {i}")
                logger.info(f"📝 TOPIC: {topic}")
                logger.info(f"⚙️ CONFIG: max_papers={config.research_session.max_papers_total}, iterations={config.research_session.max_iterations}")
                
                # Run the actual research session
                session = await orchestrator.conduct_research_session(
                    research_topic=topic,
                    session_id=f"production_test_{i}_{int(datetime.now().timestamp())}"
                )
                
                print(f"✅ Test {i} completed successfully")
                print(f"   - Session ID: {session.session_id}")
                print(f"   - Status: {session.status}")
                print(f"   - Papers found: {len(session.papers)}")
                print(f"   - Queries processed: {len(session.queries)}")
                
                logger.info(f"✅ PRODUCTION TEST {i} SUCCESS")
                logger.info(f"📊 RESULTS: papers={len(session.papers)}, queries={len(session.queries)}")
                
            except Exception as e:
                print(f"❌ Test {i} failed: {e}")
                logger.error(f"❌ PRODUCTION TEST {i} FAILED")
                logger.error(f"🔥 ERROR TYPE: {type(e).__name__}")
                logger.error(f"🔥 ERROR MESSAGE: {str(e)}")
                
                # Continue with next test
                continue
        
        print(f"\n{'='*70}")
        print("📋 PRODUCTION PIPELINE TEST SUMMARY")
        print("Check the log file 'production_error_logging_test.log' for detailed banners")
        print("Look for these markers in the logs:")
        print("  🔥 = Error conditions with full response capture")
        print("  🗝 = Empty/null response detection")
        print("  === = Full response capture banners")
        print("  ⚠️ = Warning patterns (apologies, errors, etc.)")
        
    except Exception as e:
        logger.error(f"🔥 CRITICAL PIPELINE FAILURE: {e}")
        print(f"❌ Critical pipeline failure: {e}")
        import traceback
        traceback.print_exc()


async def test_specific_agent_error_conditions():
    """Test specific agents that commonly cause JSON errors"""
    print(f"\n{'='*50}")
    print("🧪 TESTING SPECIFIC AGENT ERROR CONDITIONS")
    print(f"{'='*50}")
    
    # Test the agents individually with potentially problematic inputs
    from research_agent.agents.query_generator import QueryGenerationAgent
    from research_agent.agents.react_analyzer import ReActAnalysisAgent
    from research_agent.agents.synthesis_agent import SynthesisAgent
    
    config = ResearchConfig()
    
    # Test Query Generator with edge cases
    print("\n1. Testing Query Generator with edge cases...")
    query_agent = QueryGenerationAgent(config.get_agent_config('query_generator'))
    
    edge_case_topics = [
        "",  # Empty topic
        "a" * 1000,  # Very long topic
        "What is AI? How does ML work? Why use DL?",  # Multiple questions
        "研究机器学习优化",  # Non-English topic
        "AI/ML & (DL) optimization [methods]",  # Special characters
    ]
    
    for topic in edge_case_topics[:2]:  # Test first 2 to avoid too much output
        try:
            print(f"   Testing topic: {repr(topic[:50])}")
            result = await query_agent.generate_queries(topic=topic, context={'max_queries': 3})
            print(f"   ✅ Success: {len(result) if result else 0} chars")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.error(f"🔥 QUERY AGENT ERROR with topic {repr(topic[:50])}: {e}")
    
    # Test ReAct Analyzer with problematic contexts
    print("\n2. Testing ReAct Analyzer with edge cases...")
    react_agent = ReActAnalysisAgent(config.get_agent_config('react_analyzer'))
    
    problematic_contexts = [
        {},  # Empty context
        {'topic': None},  # Null topic
        {'topic': 'test', 'processing_results': [None] * 100},  # Many null results
    ]
    
    for ctx in problematic_contexts[:1]:  # Test first one
        try:
            print(f"   Testing context: {str(ctx)[:50]}...")
            result = await react_agent.analyze_research_progress(context=ctx)
            print(f"   ✅ Success: {len(str(result)) if result else 0} chars")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.error(f"🔥 REACT AGENT ERROR with context {str(ctx)[:50]}: {e}")


def main():
    """Run production pipeline enhanced logging test"""
    print("🎯 PRODUCTION PIPELINE ENHANCED LOGGING TEST")
    print("Objective: Capture exact LLM outputs that cause JSON parsing errors")
    print("This will provide human engineers with debugging banners")
    print()
    
    try:
        # Run the tests
        asyncio.run(test_production_pipeline_with_enhanced_logging())
        asyncio.run(test_specific_agent_error_conditions())
        
        print(f"\n{'='*70}")
        print("✅ PRODUCTION ENHANCED LOGGING TEST COMPLETE")
        print()
        print("📁 Check 'production_error_logging_test.log' for:")
        print("  • Full LLM response captures with === banners")
        print("  • Empty/error response detection with 🗝 markers") 
        print("  • JSON parsing failure analysis")
        print("  • Response type, length, and content details")
        print()
        print("🔧 This enhanced logging is now active in production agents:")
        print("  • ReAct Analyzer: captures analysis response errors")
        print("  • Synthesis Agent: captures literature review errors")
        print("  • RobustJSONParser: comprehensive failure analysis")
        print()
        print("👥 Human engineers can now pair program using these detailed logs!")
        
    except Exception as e:
        print(f"\n❌ Production test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)