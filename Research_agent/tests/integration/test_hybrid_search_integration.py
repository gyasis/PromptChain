#!/usr/bin/env python3
"""
Test Hybrid Search Agent Integration
Verifies the intelligent hybrid search functionality in the LightRAG demo
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, 'src')
sys.path.insert(0, 'examples/lightrag_demo')

def test_hybrid_search_agent_imports():
    """Test that all hybrid search components can be imported"""
    print("=== Testing Hybrid Search Agent Imports ===")
    
    try:
        from research_agent.agents.hybrid_search_agent import (
            HybridSearchAgent, CorpusAnalyzer, WebSearchDecisionMaker, 
            QueryGenerator, SourceSynthesizer, SearchResult, 
            CorpusAnalysis, WebSearchDecision
        )
        print("✅ All hybrid search components imported successfully")
        
        # Test dataclass instantiation
        test_result = SearchResult(
            content="test", source_type="test", confidence=0.5,
            citations=["test"], metadata={}
        )
        print("✅ SearchResult dataclass working")
        
        test_analysis = CorpusAnalysis(
            completeness_score=0.8, knowledge_gaps=[], temporal_coverage="current",
            recommendation="sufficient", reasoning="test"
        )
        print("✅ CorpusAnalysis dataclass working")
        
        test_decision = WebSearchDecision(
            should_search=True, search_queries=["test"], reasoning="test", confidence=0.8
        )
        print("✅ WebSearchDecision dataclass working")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_lightrag_demo_integration():
    """Test that the LightRAG demo properly integrates hybrid search"""
    print("\n=== Testing LightRAG Demo Integration ===")
    
    try:
        import lightrag_enhanced_demo
        
        # Check if HybridSearchAgent is available
        if hasattr(lightrag_enhanced_demo, 'HybridSearchAgent'):
            print("✅ HybridSearchAgent imported in demo")
        else:
            print("❌ HybridSearchAgent not available in demo")
            return False
        
        # Check if the demo has the hybrid search query method
        demo_system = lightrag_enhanced_demo.EnhancedLightRAGSystem("./test_data")
        
        if hasattr(demo_system, 'hybrid_search_query'):
            print("✅ hybrid_search_query method available")
        else:
            print("❌ hybrid_search_query method not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Demo integration test failed: {e}")
        return False

async def test_corpus_analyzer():
    """Test the CorpusAnalyzer component"""
    print("\n=== Testing CorpusAnalyzer ===")
    
    try:
        from research_agent.agents.hybrid_search_agent import CorpusAnalyzer
        
        analyzer = CorpusAnalyzer()
        print("✅ CorpusAnalyzer initialized")
        
        # Test analysis with sample data
        test_question = "What are recent developments in gait analysis?"
        test_corpus_result = "Gait analysis is used for disease detection. Several papers discuss methodologies from 2020-2022."
        
        analysis = analyzer.analyze_corpus_results(test_question, test_corpus_result)
        print(f"✅ Corpus analysis completed")
        print(f"   Completeness score: {analysis.completeness_score}")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Knowledge gaps: {len(analysis.knowledge_gaps)} identified")
        
        return True
        
    except Exception as e:
        print(f"❌ CorpusAnalyzer test failed: {e}")
        return False

async def test_web_search_decision_maker():
    """Test the WebSearchDecisionMaker component"""
    print("\n=== Testing WebSearchDecisionMaker ===")
    
    try:
        from research_agent.agents.hybrid_search_agent import WebSearchDecisionMaker, CorpusAnalysis
        
        decision_maker = WebSearchDecisionMaker()
        print("✅ WebSearchDecisionMaker initialized")
        
        # Create test corpus analysis
        test_analysis = CorpusAnalysis(
            completeness_score=0.4,  # Low completeness should trigger web search
            knowledge_gaps=["recent developments", "current technology"],
            temporal_coverage="outdated",
            recommendation="needs_web_search",
            reasoning="Research corpus lacks recent information"
        )
        
        test_question = "What recent technology did Apple announce for health monitoring?"
        
        decision = await decision_maker.decide_web_search(test_question, test_analysis)
        print(f"✅ Web search decision completed")
        print(f"   Should search: {decision.should_search}")
        print(f"   Confidence: {decision.confidence}")
        print(f"   Generated queries: {len(decision.search_queries)}")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSearchDecisionMaker test failed: {e}")
        return False

def test_query_generator():
    """Test the QueryGenerator component"""
    print("\n=== Testing QueryGenerator ===")
    
    try:
        from research_agent.agents.hybrid_search_agent import QueryGenerator, CorpusAnalysis
        
        generator = QueryGenerator()
        print("✅ QueryGenerator initialized")
        
        # Create test corpus analysis
        test_analysis = CorpusAnalysis(
            completeness_score=0.5,
            knowledge_gaps=["recent Apple health technology", "2024 announcements"],
            temporal_coverage="mixed",
            recommendation="needs_web_search",
            reasoning="Missing recent technology announcements"
        )
        
        test_question = "What recent health technology did Apple announce?"
        
        queries = generator.generate_queries(test_question, test_analysis)
        print(f"✅ Query generation completed")
        print(f"   Generated {len(queries)} queries")
        for i, query in enumerate(queries, 1):
            print(f"   Query {i}: {query}")
        
        return True
        
    except Exception as e:
        print(f"❌ QueryGenerator test failed: {e}")
        return False

def test_source_synthesizer():
    """Test the SourceSynthesizer component"""
    print("\n=== Testing SourceSynthesizer ===")
    
    try:
        from research_agent.agents.hybrid_search_agent import SourceSynthesizer, CorpusAnalysis
        
        synthesizer = SourceSynthesizer()
        print("✅ SourceSynthesizer initialized")
        
        # Test synthesis with sample data
        test_question = "How does gait analysis help with disease detection?"
        test_corpus = "Research shows gait analysis can detect Parkinson's disease with 85% accuracy."
        test_web = "Recent 2024 studies show Apple Watch can now detect gait patterns with new sensors."
        
        test_analysis = CorpusAnalysis(
            completeness_score=0.7,
            knowledge_gaps=["recent technology"],
            temporal_coverage="mixed",
            recommendation="sufficient",
            reasoning="Good corpus coverage with web enhancement"
        )
        
        result = synthesizer.synthesize_results(test_question, test_corpus, test_web, test_analysis)
        print(f"✅ Source synthesis completed")
        print(f"   Result type: {result.source_type}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Citations: {result.citations}")
        print(f"   Content length: {len(result.content)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ SourceSynthesizer test failed: {e}")
        return False

def test_environment_configuration():
    """Test environment setup for hybrid search"""
    print("\n=== Testing Environment Configuration ===")
    
    # Check for required dependencies
    dependencies = {
        'openai': False,
        'lightrag': False,
        'promptchain': False,
        'research_agent.tools': False,
        'research_agent.agents.hybrid_search_agent': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
    
    # Check environment variables
    env_vars = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'SERPER_API_KEY': os.getenv('SERPER_API_KEY')
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            print(f"✅ {var_name}: Configured ({len(var_value)} characters)")
        else:
            print(f"⚠️ {var_name}: Not configured")
    
    all_deps_available = all(dependencies.values())
    openai_configured = bool(env_vars['OPENAI_API_KEY'])
    
    if all_deps_available and openai_configured:
        print("✅ Environment ready for hybrid search")
        return True
    else:
        print("⚠️ Environment partially configured")
        return False

async def main():
    """Run all hybrid search integration tests"""
    print("🧠 Hybrid Search Agent Integration Test")
    print("=" * 70)
    
    # Track test results
    tests = [
        ("Hybrid Search Agent Imports", test_hybrid_search_agent_imports),
        ("LightRAG Demo Integration", test_lightrag_demo_integration),
        ("CorpusAnalyzer", test_corpus_analyzer),
        ("WebSearchDecisionMaker", test_web_search_decision_maker),
        ("QueryGenerator", test_query_generator),
        ("SourceSynthesizer", test_source_synthesizer),
        ("Environment Configuration", test_environment_configuration),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
            if result:
                passed += 1
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"{status_icon} {test_name}: {result}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Generate test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "hybrid_search_agent_integration",
        "test_results": results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": f"{passed/total*100:.1f}%"
        }
    }
    
    report_file = f"hybrid_search_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved: {report_file}")
    
    if passed == total:
        print("\n🎉 All tests passed! Hybrid search agent integration is ready.")
        print("\n🚀 Ready to use intelligent hybrid search with:")
        print("   • Corpus completeness analysis")
        print("   • Autonomous web search decisions")
        print("   • ReACT-style reasoning")
        print("   • Source synthesis and attribution")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the issues above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)