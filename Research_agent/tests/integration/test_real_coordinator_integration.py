#!/usr/bin/env python3
"""
Test Script: Real Three-Tier RAG Integration with Multi-Query Coordinator

This script validates the integration between the real ThreeTierRAG system 
and the MultiQueryCoordinator, ensuring all placeholder processing has been replaced.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from research_agent.integrations.multi_query_coordinator import MultiQueryCoordinator
from research_agent.integrations.three_tier_rag import RAGTier
from research_agent.core.session import Query, Paper, ProcessingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('real_coordinator_integration_test.log')
    ]
)
logger = logging.getLogger(__name__)


class RealCoordinatorIntegrationTester:
    """Test real integration between coordinator and 3-tier RAG"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {},
            'integration_status': 'unknown'
        }
    
    async def run_all_tests(self):
        """Run comprehensive integration tests"""
        logger.info("🚀 Starting Real Coordinator Integration Tests")
        
        try:
            # Test 1: Coordinator Initialization with Real RAG
            await self._test_coordinator_initialization()
            
            # Test 2: Real RAG System Availability
            await self._test_rag_system_availability()
            
            # Test 3: Document Processing Pipeline
            await self._test_document_processing_pipeline()
            
            # Test 4: Multi-Query Processing with Real Tiers
            await self._test_multi_query_processing()
            
            # Test 5: Success/Failure Logic Integration
            await self._test_success_failure_integration()
            
            # Test 6: Performance Metrics with Real Processing
            await self._test_performance_metrics()
            
            # Generate summary
            self._generate_test_summary()
            
            logger.info("✅ Real Coordinator Integration Tests Completed")
            
        except Exception as e:
            logger.error(f"❌ Integration test suite failed: {e}")
            self.test_results['integration_status'] = 'failed'
            self.test_results['error'] = str(e)
    
    async def _test_coordinator_initialization(self):
        """Test coordinator initialization with real RAG system"""
        test_name = "coordinator_initialization"
        logger.info("📋 Testing Coordinator Initialization with Real RAG")
        
        try:
            # Test configuration
            config = {
                'coordination': {'model': 'openai/gpt-4o-mini'},
                'three_tier_rag': {
                    'lightrag_working_dir': './test_lightrag',
                    'paperqa2_working_dir': './test_paperqa2',
                    'graphrag_working_dir': './test_graphrag'
                }
            }
            
            # Initialize coordinator
            coordinator = MultiQueryCoordinator(config)
            await coordinator.initialize_tiers()
            
            # Check real system initialization
            stats = coordinator.get_processing_stats()
            
            self.test_results['tests'][test_name] = {
                'status': 'passed',
                'three_tier_rag_available': stats.get('three_tier_rag_available', False),
                'available_tiers': stats.get('available_tiers', []),
                'tier_count': stats.get('tier_count', 0),
                'systems_status': stats.get('systems_status', {}),
                'message': 'Coordinator initialized with real RAG system'
            }
            
            # Store coordinator for other tests
            self.coordinator = coordinator
            
            logger.info(f"✅ Coordinator initialized with {stats.get('tier_count', 0)} real tiers")
            
        except Exception as e:
            logger.error(f"❌ Coordinator initialization failed: {e}")
            self.test_results['tests'][test_name] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Failed to initialize coordinator with real RAG'
            }
    
    async def _test_rag_system_availability(self):
        """Test availability of real RAG systems"""
        test_name = "rag_system_availability"
        logger.info("📋 Testing Real RAG System Availability")
        
        try:
            if not hasattr(self, 'coordinator'):
                raise Exception("Coordinator not available from previous test")
            
            # Check available tiers
            available_tiers = self.coordinator._get_available_tiers()
            tier_status = self.coordinator._get_tier_status()
            
            # Check for real implementations
            has_real_processors = False
            for tier_name, status in tier_status.items():
                if status.get('available', False):
                    processor_info = status.get('processor', {})
                    if processor_info.get('status') == 'initialized':
                        has_real_processors = True
                        logger.info(f"✅ Real processor available: {tier_name}")
            
            self.test_results['tests'][test_name] = {
                'status': 'passed' if has_real_processors else 'warning',
                'available_tiers': [tier.value for tier in available_tiers],
                'tier_status': tier_status,
                'has_real_processors': has_real_processors,
                'message': f'Found {len(available_tiers)} available tiers with real processors'
            }
            
            if has_real_processors:
                logger.info(f"✅ Real RAG systems available: {len(available_tiers)} tiers")
            else:
                logger.warning("⚠️ No real RAG processors available")
            
        except Exception as e:
            logger.error(f"❌ RAG system availability check failed: {e}")
            self.test_results['tests'][test_name] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Failed to check RAG system availability'
            }
    
    async def _test_document_processing_pipeline(self):
        """Test document processing through real RAG pipeline"""
        test_name = "document_processing_pipeline"
        logger.info("📋 Testing Document Processing Pipeline")
        
        try:
            if not hasattr(self, 'coordinator'):
                raise Exception("Coordinator not available from previous test")
            
            # Create test documents as temporary files
            import tempfile
            test_documents = []
            
            # Create temporary text file with test content
            test_content = (
                "Artificial intelligence and machine learning are transforming research methodologies. "
                "Modern AI systems can process vast amounts of scientific literature and extract key insights. "
                "This enables researchers to identify patterns and make connections across different domains."
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                test_documents.append(f.name)
                temp_file_path = f.name
            
            # Try adding documents to available tiers
            available_tiers = self.coordinator._get_available_tiers()
            document_addition_results = {}
            
            for tier in available_tiers:
                try:
                    success = await self.coordinator.add_documents_to_rag(test_documents, tier)
                    document_addition_results[tier.value] = {
                        'success': success,
                        'document_count': len(test_documents)
                    }
                    logger.info(f"✅ Document addition to {tier.value}: {'Success' if success else 'Failed'}")
                except Exception as e:
                    document_addition_results[tier.value] = {
                        'success': False,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ Document addition to {tier.value} failed: {e}")
            
            self.test_results['tests'][test_name] = {
                'status': 'passed' if any(r.get('success', False) for r in document_addition_results.values()) else 'warning',
                'available_tiers': [tier.value for tier in available_tiers],
                'document_addition_results': document_addition_results,
                'message': f'Document processing tested on {len(available_tiers)} tiers'
            }
            
            logger.info(f"✅ Document processing pipeline tested on {len(available_tiers)} tiers")
            
            # Cleanup temporary file
            import os
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {e}")
            
        except Exception as e:
            logger.error(f"❌ Document processing pipeline test failed: {e}")
            self.test_results['tests'][test_name] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Failed to test document processing pipeline'
            }
    
    async def _test_multi_query_processing(self):
        """Test multi-query processing with real RAG tiers"""
        test_name = "multi_query_processing"
        logger.info("📋 Testing Multi-Query Processing with Real RAG")
        
        try:
            if not hasattr(self, 'coordinator'):
                raise Exception("Coordinator not available from previous test")
            
            # Create test papers and queries
            test_papers = [
                Paper(
                    id="test_paper_1",
                    title="Machine Learning in Research",
                    abstract="This paper explores machine learning applications in research.",
                    authors=["Test Author"],
                    source="test",
                    doi="10.1000/test",
                    url="http://test.com/paper1"
                )
            ]
            
            test_queries = [
                Query(
                    id="test_query_1",
                    text="What are the applications of machine learning in research?",
                    priority=1.0,
                    iteration=1
                ),
                Query(
                    id="test_query_2", 
                    text="How does AI transform research methodologies?",
                    priority=0.8,
                    iteration=1
                )
            ]
            
            # Process papers with queries through real system
            start_time = datetime.now()
            processing_results = await self.coordinator.process_papers_with_queries(
                test_papers, test_queries
            )
            processing_duration = (datetime.now() - start_time).total_seconds()
            
            # Analyze results
            result_analysis = {
                'total_results': len(processing_results),
                'successful_results': len([r for r in processing_results if r.result_data.get('success', False)]),
                'failed_results': len([r for r in processing_results if not r.result_data.get('success', True)]),
                'processing_duration': processing_duration,
                'tiers_used': list(set(r.tier for r in processing_results)),
                'average_confidence': 0.0,
                'has_real_processing': False
            }
            
            # Check for real processing indicators
            real_processing_indicators = 0
            confidence_scores = []
            
            for result in processing_results:
                result_data = result.result_data
                metadata = result_data.get('metadata', {})
                
                # Check for real processor indicators
                if metadata.get('processor', '').startswith('actual_'):
                    real_processing_indicators += 1
                
                # Collect confidence scores
                if result_data.get('success', False):
                    confidence = result_data.get('confidence', 0.0)
                    if confidence > 0:
                        confidence_scores.append(confidence)
            
            result_analysis['has_real_processing'] = real_processing_indicators > 0
            result_analysis['real_processing_count'] = real_processing_indicators
            
            if confidence_scores:
                result_analysis['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
            
            self.test_results['tests'][test_name] = {
                'status': 'passed' if result_analysis['has_real_processing'] else 'warning',
                'result_analysis': result_analysis,
                'processing_results_summary': [
                    {
                        'tier': r.tier,
                        'success': r.result_data.get('success', False),
                        'confidence': r.result_data.get('confidence', 0.0),
                        'processor': r.result_data.get('metadata', {}).get('processor', 'unknown'),
                        'processing_time': r.processing_time
                    }
                    for r in processing_results[:10]  # Limit to first 10 for readability
                ],
                'message': f'Processed {len(test_queries)} queries through real RAG system'
            }
            
            logger.info(f"✅ Multi-query processing: {result_analysis['total_results']} results, "
                       f"{result_analysis['real_processing_count']} with real processing")
            
        except Exception as e:
            logger.error(f"❌ Multi-query processing test failed: {e}")
            self.test_results['tests'][test_name] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Failed to test multi-query processing'
            }
    
    async def _test_success_failure_integration(self):
        """Test success/failure logic integration with real processing"""
        test_name = "success_failure_integration"
        logger.info("📋 Testing Success/Failure Logic Integration")
        
        try:
            if not hasattr(self, 'coordinator'):
                raise Exception("Coordinator not available from previous test")
            
            # Test with valid and invalid queries
            test_papers = [
                Paper(
                    id="integration_test_paper",
                    title="Integration Test Paper", 
                    abstract="Test paper for integration validation.",
                    authors=["Test Author"],
                    source="test",
                    doi="10.1000/integration",
                    url="http://test.com/integration"
                )
            ]
            
            # Mix of valid and challenging queries
            test_queries = [
                Query(
                    id="valid_query",
                    text="What is machine learning?",
                    priority=1.0,
                    iteration=1
                ),
                Query(
                    id="empty_query",
                    text="",  # Empty query to test error handling
                    priority=0.5,
                    iteration=1
                ),
                Query(
                    id="complex_query",
                    text="Compare and contrast the methodological implications of deep learning architectures in multi-domain research applications",
                    priority=0.8,
                    iteration=1
                )
            ]
            
            # Process and analyze success/failure patterns
            processing_results = await self.coordinator.process_papers_with_queries(
                test_papers, test_queries
            )
            
            # Analyze success/failure metrics
            success_failure_analysis = {
                'total_results': len(processing_results),
                'success_count': 0,
                'failure_count': 0,
                'confidence_scores': [],
                'error_types': {},
                'real_vs_placeholder': {
                    'real_processing': 0,
                    'placeholder_processing': 0
                }
            }
            
            for result in processing_results:
                result_data = result.result_data
                
                if result_data.get('success', False):
                    success_failure_analysis['success_count'] += 1
                    confidence = result_data.get('confidence', 0.0)
                    if confidence > 0:
                        success_failure_analysis['confidence_scores'].append(confidence)
                else:
                    success_failure_analysis['failure_count'] += 1
                    error = result_data.get('error', 'Unknown error')
                    success_failure_analysis['error_types'][error] = success_failure_analysis['error_types'].get(error, 0) + 1
                
                # Check for real vs placeholder processing
                metadata = result_data.get('metadata', {})
                processor = metadata.get('processor', 'unknown')
                
                if processor.startswith('actual_'):
                    success_failure_analysis['real_vs_placeholder']['real_processing'] += 1
                elif processor == 'placeholder' or 'placeholder' in str(result_data):
                    success_failure_analysis['real_vs_placeholder']['placeholder_processing'] += 1
            
            # Calculate success rate
            success_rate = success_failure_analysis['success_count'] / max(success_failure_analysis['total_results'], 1)
            average_confidence = sum(success_failure_analysis['confidence_scores']) / max(len(success_failure_analysis['confidence_scores']), 1)
            
            success_failure_analysis['success_rate'] = success_rate
            success_failure_analysis['average_confidence'] = average_confidence
            
            # Determine test status
            test_status = 'passed'
            if success_failure_analysis['real_vs_placeholder']['placeholder_processing'] > 0:
                test_status = 'warning'  # Still has placeholders
            if success_rate < 0.5:
                test_status = 'failed'  # Low success rate
            
            self.test_results['tests'][test_name] = {
                'status': test_status,
                'success_failure_analysis': success_failure_analysis,
                'message': f'Success rate: {success_rate:.2%}, Real processing: {success_failure_analysis["real_vs_placeholder"]["real_processing"]}/{success_failure_analysis["total_results"]}'
            }
            
            logger.info(f"✅ Success/Failure Integration: {success_rate:.2%} success rate, "
                       f"{success_failure_analysis['real_vs_placeholder']['real_processing']} real processing results")
            
        except Exception as e:
            logger.error(f"❌ Success/failure integration test failed: {e}")
            self.test_results['tests'][test_name] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Failed to test success/failure integration'
            }
    
    async def _test_performance_metrics(self):
        """Test performance metrics with real processing data"""
        test_name = "performance_metrics"
        logger.info("📋 Testing Performance Metrics with Real Processing")
        
        try:
            if not hasattr(self, 'coordinator'):
                raise Exception("Coordinator not available from previous test")
            
            # Get processing statistics
            processing_stats = self.coordinator.get_processing_stats()
            
            # Performance analysis
            performance_analysis = {
                'cache_utilization': processing_stats.get('cache_size', 0),
                'tier_availability': processing_stats.get('tier_count', 0),
                'system_health': 'unknown',
                'real_system_integration': processing_stats.get('three_tier_rag_available', False),
                'available_tiers': processing_stats.get('available_tiers', [])
            }
            
            # Determine system health
            if performance_analysis['real_system_integration'] and performance_analysis['tier_availability'] > 0:
                performance_analysis['system_health'] = 'healthy'
            elif performance_analysis['tier_availability'] > 0:
                performance_analysis['system_health'] = 'degraded'
            else:
                performance_analysis['system_health'] = 'critical'
            
            # Check for health check capability
            if hasattr(self.coordinator.three_tier_rag, 'health_check'):
                try:
                    health_check = await self.coordinator.three_tier_rag.health_check()
                    performance_analysis['detailed_health'] = health_check
                except Exception as e:
                    performance_analysis['health_check_error'] = str(e)
            
            self.test_results['tests'][test_name] = {
                'status': 'passed' if performance_analysis['system_health'] == 'healthy' else 'warning',
                'performance_analysis': performance_analysis,
                'processing_stats': processing_stats,
                'message': f'System health: {performance_analysis["system_health"]}, {performance_analysis["tier_availability"]} tiers available'
            }
            
            logger.info(f"✅ Performance Metrics: {performance_analysis['system_health']} system with "
                       f"{performance_analysis['tier_availability']} tiers")
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            self.test_results['tests'][test_name] = {
                'status': 'failed',
                'error': str(e),
                'message': 'Failed to test performance metrics'
            }
    
    def _generate_test_summary(self):
        """Generate comprehensive test summary"""
        logger.info("📊 Generating Test Summary")
        
        # Count test results
        total_tests = len(self.test_results['tests'])
        passed_tests = len([t for t in self.test_results['tests'].values() if t['status'] == 'passed'])
        warning_tests = len([t for t in self.test_results['tests'].values() if t['status'] == 'warning'])
        failed_tests = len([t for t in self.test_results['tests'].values() if t['status'] == 'failed'])
        
        # Determine overall integration status
        if failed_tests == 0 and warning_tests == 0:
            integration_status = 'fully_integrated'
        elif failed_tests == 0:
            integration_status = 'mostly_integrated'
        elif passed_tests > failed_tests:
            integration_status = 'partially_integrated'
        else:
            integration_status = 'integration_failed'
        
        # Generate summary
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'warning_tests': warning_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / max(total_tests, 1),
            'integration_status': integration_status,
            'key_findings': []
        }
        
        # Key findings based on test results
        if 'coordinator_initialization' in self.test_results['tests']:
            init_result = self.test_results['tests']['coordinator_initialization']
            if init_result.get('three_tier_rag_available', False):
                summary['key_findings'].append('✅ Real ThreeTierRAG system successfully integrated')
            else:
                summary['key_findings'].append('❌ ThreeTierRAG system not available')
        
        if 'multi_query_processing' in self.test_results['tests']:
            processing_result = self.test_results['tests']['multi_query_processing']
            if processing_result.get('result_analysis', {}).get('has_real_processing', False):
                summary['key_findings'].append('✅ Real processing detected in multi-query results')
            else:
                summary['key_findings'].append('⚠️ No real processing detected in results')
        
        if 'success_failure_integration' in self.test_results['tests']:
            success_result = self.test_results['tests']['success_failure_integration']
            analysis = success_result.get('success_failure_analysis', {})
            real_processing = analysis.get('real_vs_placeholder', {}).get('real_processing', 0)
            placeholder_processing = analysis.get('real_vs_placeholder', {}).get('placeholder_processing', 0)
            
            if placeholder_processing == 0:
                summary['key_findings'].append('✅ All placeholder processing eliminated')
            elif real_processing > placeholder_processing:
                summary['key_findings'].append('⚠️ Some placeholder processing remains')
            else:
                summary['key_findings'].append('❌ Placeholder processing still dominant')
        
        self.test_results['summary'] = summary
        self.test_results['integration_status'] = integration_status
        
        # Log summary
        logger.info(f"📊 Integration Test Summary:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}, Warnings: {warning_tests}, Failed: {failed_tests}")
        logger.info(f"   Success Rate: {summary['success_rate']:.2%}")
        logger.info(f"   Integration Status: {integration_status}")
        
        for finding in summary['key_findings']:
            logger.info(f"   {finding}")
    
    def save_test_report(self, filename: str = None):
        """Save comprehensive test report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_coordinator_integration_test_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"📄 Test report saved: {filename}")
            
            # Also create a summary report
            summary_filename = filename.replace('.json', '_summary.txt')
            with open(summary_filename, 'w') as f:
                f.write("Real Coordinator Integration Test Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {self.test_results['timestamp']}\n")
                f.write(f"Integration Status: {self.test_results['integration_status']}\n\n")
                
                summary = self.test_results.get('summary', {})
                f.write(f"Test Summary:\n")
                f.write(f"- Total Tests: {summary.get('total_tests', 0)}\n")
                f.write(f"- Passed: {summary.get('passed_tests', 0)}\n")
                f.write(f"- Warnings: {summary.get('warning_tests', 0)}\n")
                f.write(f"- Failed: {summary.get('failed_tests', 0)}\n")
                f.write(f"- Success Rate: {summary.get('success_rate', 0):.2%}\n\n")
                
                f.write("Key Findings:\n")
                for finding in summary.get('key_findings', []):
                    f.write(f"  {finding}\n")
                
                f.write("\nTest Details:\n")
                for test_name, test_result in self.test_results.get('tests', {}).items():
                    f.write(f"\n{test_name.upper()}:\n")
                    f.write(f"  Status: {test_result.get('status', 'unknown')}\n")
                    f.write(f"  Message: {test_result.get('message', 'No message')}\n")
                    if test_result.get('status') == 'failed' and 'error' in test_result:
                        f.write(f"  Error: {test_result['error']}\n")
            
            logger.info(f"📄 Summary report saved: {summary_filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save test report: {e}")


async def main():
    """Run the real coordinator integration test suite"""
    logger.info("🔬 Real Coordinator Integration Test Suite")
    logger.info("=" * 60)
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("⚠️ OPENAI_API_KEY not set - some tests may fail")
    
    # Create and run tester
    tester = RealCoordinatorIntegrationTester()
    
    try:
        await tester.run_all_tests()
        
        # Save results
        tester.save_test_report()
        
        # Final status
        integration_status = tester.test_results.get('integration_status', 'unknown')
        summary = tester.test_results.get('summary', {})
        
        logger.info("=" * 60)
        logger.info(f"🎯 FINAL INTEGRATION STATUS: {integration_status.upper()}")
        logger.info(f"📊 Success Rate: {summary.get('success_rate', 0):.2%}")
        logger.info("=" * 60)
        
        return tester.test_results
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return {'error': str(e), 'integration_status': 'failed'}


if __name__ == "__main__":
    # Run the test suite
    results = asyncio.run(main())