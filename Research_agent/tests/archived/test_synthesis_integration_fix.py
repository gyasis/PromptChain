#!/usr/bin/env python3

"""
Test synthesis pipeline integration with real 3-tier RAG system
This tests the complete flow from RAG processing to literature review synthesis.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from research_agent.core.config import ResearchConfig
from research_agent.core.session import ResearchSession, Query, Paper
from research_agent.integrations.multi_query_coordinator import MultiQueryCoordinator
from research_agent.agents.synthesis_agent import SynthesisAgent
from research_agent.core.orchestrator import AdvancedResearchOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('synthesis_integration_test.log')
    ]
)

logger = logging.getLogger(__name__)


class SynthesisIntegrationTester:
    """Test synthesis pipeline integration with real RAG system"""
    
    def __init__(self):
        self.config = self._create_test_config()
        self.test_results = {}
        
    def _create_test_config(self) -> ResearchConfig:
        """Create test configuration"""
        # Use default config and access its internal data
        config = ResearchConfig()
        
        # Override with test-specific settings directly in config_data
        test_overrides = {
            'agents': {
                'synthesis_agent': {
                    'model': 'openai/gpt-4o',
                    'processor': {
                        'objective': 'Test synthesis of real RAG results',
                        'max_internal_steps': 5
                    }
                }
            }
        }
        
        # Merge with existing config data
        if 'agents' not in config.config_data:
            config.config_data['agents'] = {}
        config.config_data['agents'].update(test_overrides['agents'])
        
        return config
    
    async def test_complete_synthesis_pipeline(self):
        """Test the complete synthesis pipeline integration"""
        logger.info("🔬 TESTING COMPLETE SYNTHESIS PIPELINE INTEGRATION")
        
        try:
            # Initialize components
            coordinator = MultiQueryCoordinator(self.config)
            synthesis_agent = SynthesisAgent(self.config.get_agent_config('synthesis_agent'))
            
            # Create test data
            test_papers = self._create_test_papers()
            test_queries = self._create_test_queries()
            
            # Step 1: Process papers through real RAG tiers
            logger.info("Step 1: Processing papers through real RAG tiers...")
            processing_results = await coordinator.process_papers_with_queries(
                papers=test_papers,
                queries=test_queries
            )
            
            # Analyze processing results
            self._analyze_processing_results(processing_results)
            
            # Step 2: Prepare synthesis context from real results
            logger.info("Step 2: Preparing synthesis context from real RAG results...")
            synthesis_context = self._prepare_synthesis_context(
                papers=test_papers,
                queries=test_queries,
                processing_results=processing_results
            )
            
            # Step 3: Generate literature review synthesis
            logger.info("Step 3: Generating literature review synthesis...")
            literature_review = await synthesis_agent.synthesize_literature_review(synthesis_context)
            
            # Step 4: Validate synthesis quality
            logger.info("Step 4: Validating synthesis quality...")
            validation_results = self._validate_synthesis_output(literature_review)
            
            # Step 5: Test orchestrator integration
            logger.info("Step 5: Testing orchestrator integration...")
            orchestrator_results = await self._test_orchestrator_integration()
            
            # Compile test results
            self.test_results = {
                'processing_results': {
                    'total_results': len(processing_results),
                    'successful_results': len([r for r in processing_results if r.result_data.get('success', False)]),
                    'tier_distribution': self._analyze_tier_distribution(processing_results)
                },
                'synthesis_results': {
                    'synthesis_successful': bool(literature_review),
                    'validation_score': validation_results,
                    'sections_generated': len(literature_review.get('literature_review', {}).get('sections', {}))
                },
                'orchestrator_integration': orchestrator_results,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ SYNTHESIS INTEGRATION TEST COMPLETED SUCCESSFULLY")
            return True
            
        except Exception as e:
            logger.error(f"❌ SYNTHESIS INTEGRATION TEST FAILED: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _create_test_papers(self) -> List[Paper]:
        """Create test papers for processing"""
        return [
            Paper(
                id='paper_1',
                title='Deep Learning Approaches to Natural Language Processing',
                abstract='This paper reviews recent advances in deep learning for NLP tasks.',
                authors=['Smith, J.', 'Johnson, A.'],
                source='arxiv',
                url='https://arxiv.org/abs/2301.00001',
                publication_year=2023
            ),
            Paper(
                id='paper_2', 
                title='Transformer Models: A Comprehensive Survey',
                abstract='A comprehensive survey of transformer architectures and applications.',
                authors=['Brown, K.', 'Davis, M.'],
                source='journals',
                url='https://example.com/journal/transformers',
                publication_year=2022
            )
        ]
    
    def _create_test_queries(self) -> List[Query]:
        """Create test queries for processing"""
        return [
            Query(
                id='query_1',
                text='What are the main approaches to natural language processing?',
                priority=1.0,
                iteration=1
            ),
            Query(
                id='query_2',
                text='How do transformer models work?',
                priority=0.9,
                iteration=1
            ),
            Query(
                id='query_3',
                text='What are the applications of deep learning in NLP?',
                priority=0.8,
                iteration=1
            )
        ]
    
    def _analyze_processing_results(self, results: List[Any]):
        """Analyze processing results from RAG tiers"""
        logger.info(f"📊 PROCESSING RESULTS ANALYSIS:")
        logger.info(f"  Total results: {len(results)}")
        
        successful_results = [r for r in results if r.result_data.get('success', False)]
        failed_results = [r for r in results if not r.result_data.get('success', False)]
        
        logger.info(f"  Successful: {len(successful_results)}")
        logger.info(f"  Failed: {len(failed_results)}")
        
        # Analyze by tier
        tier_stats = {}
        for result in results:
            tier = result.tier
            if tier not in tier_stats:
                tier_stats[tier] = {'total': 0, 'successful': 0}
            tier_stats[tier]['total'] += 1
            if result.result_data.get('success', False):
                tier_stats[tier]['successful'] += 1
        
        for tier, stats in tier_stats.items():
            success_rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
            logger.info(f"  {tier}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Sample successful result content
        if successful_results:
            sample = successful_results[0]
            logger.info(f"  Sample successful result: {sample.result_data.get('content', '')[:100]}...")
    
    def _prepare_synthesis_context(self, papers: List[Paper], queries: List[Query], processing_results: List[Any]) -> Dict[str, Any]:
        """Prepare synthesis context from real processing results"""
        
        # Convert papers to serializable format
        papers_data = []
        for paper in papers:
            papers_data.append({
                'id': paper.id,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'publication_year': paper.publication_year,
                'source': paper.source
            })
        
        # Convert processing results to serializable format
        results_data = []
        for result in processing_results:
            results_data.append({
                'tier': result.tier,
                'query_id': result.query_id,
                'success': result.result_data.get('success', False),
                'content': result.result_data.get('content', ''),
                'confidence': result.result_data.get('confidence', 0.0),
                'sources': result.result_data.get('sources', []),
                'metadata': result.result_data.get('metadata', {})
            })
        
        return {
            'session_id': 'test_synthesis_session',
            'topic': 'Natural Language Processing and Deep Learning',
            'queries': [q.text for q in queries],
            'papers': papers_data,
            'processing_results': results_data,
            'statistics': {
                'total_papers': len(papers),
                'total_queries': len(queries),
                'total_results': len(processing_results)
            }
        }
    
    def _analyze_tier_distribution(self, results: List[Any]) -> Dict[str, int]:
        """Analyze distribution of results across tiers"""
        distribution = {}
        for result in results:
            tier = result.tier
            if tier not in distribution:
                distribution[tier] = 0
            distribution[tier] += 1
        return distribution
    
    def _validate_synthesis_output(self, literature_review: Dict[str, Any]) -> Dict[str, Any]:
        """Validate literature review synthesis output"""
        validation = {
            'has_literature_review': 'literature_review' in literature_review,
            'has_statistics': 'statistics' in literature_review,
            'has_citations': 'citations' in literature_review,
            'has_recommendations': 'recommendations' in literature_review,
            'has_executive_summary': False,
            'has_sections': False,
            'section_count': 0
        }
        
        if literature_review.get('literature_review'):
            lit_review = literature_review['literature_review']
            validation['has_executive_summary'] = 'executive_summary' in lit_review
            validation['has_sections'] = 'sections' in lit_review
            if 'sections' in lit_review:
                validation['section_count'] = len(lit_review['sections'])
        
        score = sum(1 for v in validation.values() if isinstance(v, bool) and v)
        validation['overall_score'] = score / 6  # 6 boolean checks
        
        return validation
    
    async def _test_orchestrator_integration(self) -> Dict[str, Any]:
        """Test integration with orchestrator"""
        try:
            orchestrator = AdvancedResearchOrchestrator(self.config)
            
            # Test basic initialization
            return {
                'initialization_successful': True,
                'synthesis_agent_available': hasattr(orchestrator, 'synthesis_agent'),
                'multi_query_coordinator_available': hasattr(orchestrator, 'multi_query_coordinator'),
                'config_loaded': orchestrator.config is not None
            }
        except Exception as e:
            return {
                'initialization_successful': False,
                'error': str(e)
            }
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report = f"""
# Synthesis Integration Test Report
Generated: {datetime.now().isoformat()}

## Test Results Summary
- **Processing Results**: {self.test_results.get('processing_results', {}).get('total_results', 0)} total
- **Successful Processing**: {self.test_results.get('processing_results', {}).get('successful_results', 0)}
- **Synthesis Success**: {self.test_results.get('synthesis_results', {}).get('synthesis_successful', False)}
- **Validation Score**: {self.test_results.get('synthesis_results', {}).get('validation_score', {}).get('overall_score', 0):.2%}

## Processing Analysis
{json.dumps(self.test_results.get('processing_results', {}), indent=2)}

## Synthesis Analysis  
{json.dumps(self.test_results.get('synthesis_results', {}), indent=2)}

## Orchestrator Integration
{json.dumps(self.test_results.get('orchestrator_integration', {}), indent=2)}

## Recommendations
- {"✅ Synthesis pipeline working correctly" if self.test_results.get('synthesis_results', {}).get('synthesis_successful', False) else "❌ Synthesis pipeline needs attention"}
- {"✅ RAG processing successful" if self.test_results.get('processing_results', {}).get('successful_results', 0) > 0 else "❌ RAG processing needs debugging"}
- {"✅ Orchestrator integration ready" if self.test_results.get('orchestrator_integration', {}).get('initialization_successful', False) else "❌ Orchestrator integration needs fixes"}
        """
        
        return report.strip()


async def main():
    """Run synthesis integration test"""
    tester = SynthesisIntegrationTester()
    
    # Run the complete integration test
    success = await tester.test_complete_synthesis_pipeline()
    
    # Generate and display report
    report = tester.generate_test_report()
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"synthesis_integration_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(tester.test_results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {report_file}")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)