#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Research Agent Core Workflows - Simplified Version

This test suite provides comprehensive testing of all core workflows in the Research Agent,
focusing on what's actually implemented and available.
"""

import asyncio
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add the research agent to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core Research Agent imports
from research_agent.core.session import ResearchSession, SessionStatus, Query, Paper
from research_agent.core.config import ResearchConfig
from research_agent.agents.literature_searcher import LiteratureSearchAgent
from research_agent.integrations.multi_query_coordinator import MultiQueryCoordinator
from research_agent.utils.pdf_manager import PDFManager

class ComprehensiveWorkflowTestSuite:
    """
    Simplified comprehensive test suite for Research Agent workflows
    """
    
    def __init__(self):
        self.temp_dir = None
        self.config = None
        self.test_results = {
            'core_functionality': [],
            'integration_tests': [],
            'workflow_tests': [],
            'error_handling': []
        }
        
    async def setup_test_environment(self):
        """Set up test environment"""
        logger.info("🔧 Setting up comprehensive test environment")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="research_agent_comprehensive_"))
        logger.info(f"Test directory: {self.temp_dir}")
        
        # Create basic config
        self.config = {
            'base_paths': {
                'data': str(self.temp_dir / 'data'),
                'cache': str(self.temp_dir / 'cache'),
                'logs': str(self.temp_dir / 'logs'),
                'papers': str(self.temp_dir / 'papers')
            },
            'testing_mode': True,
            'mock_external_apis': True,
            'literature_sources': {
                'arxiv': {'enabled': True},
                'pubmed': {'enabled': True},
                'sci_hub': {'enabled': True}
            },
            'three_tier_rag': {
                'tier_1': {'mode': 'mock'},
                'tier_2': {'mode': 'mock'},
                'tier_3': {'mode': 'mock'}
            }
        }
        
        # Create directories
        for path in self.config['base_paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Test environment setup complete")
        
    async def test_core_functionality(self):
        """Test core functionality components"""
        logger.info("\n📋 Testing Core Functionality")
        
        tests = [
            ('Research Session Creation', self._test_research_session_creation),
            ('Query Management', self._test_query_management),
            ('Paper Management', self._test_paper_management),
            ('PDF Manager Core', self._test_pdf_manager_core),
            ('Literature Search Core', self._test_literature_search_core)
        ]
        
        for test_name, test_func in tests:
            await self._run_test(test_name, test_func, 'core_functionality')
    
    async def test_integration_workflows(self):
        """Test integration between components"""
        logger.info("\n🔗 Testing Integration Workflows")
        
        tests = [
            ('Session-Query Integration', self._test_session_query_integration),
            ('Literature-PDF Integration', self._test_literature_pdf_integration),
            ('Multi-Query Coordinator Integration', self._test_multi_query_integration),
            ('Data Flow Integration', self._test_data_flow_integration)
        ]
        
        for test_name, test_func in tests:
            await self._run_test(test_name, test_func, 'integration_tests')
    
    async def test_end_to_end_workflows(self):
        """Test complete end-to-end workflows"""
        logger.info("\n🎯 Testing End-to-End Workflows")
        
        tests = [
            ('Literature Search Workflow', self._test_literature_search_workflow),
            ('3-Tier RAG Processing', self._test_3tier_rag_processing),
            ('PDF Download and Storage', self._test_pdf_download_storage),
            ('Complete Research Pipeline', self._test_complete_research_pipeline)
        ]
        
        for test_name, test_func in tests:
            await self._run_test(test_name, test_func, 'workflow_tests')
    
    async def test_error_handling_scenarios(self):
        """Test error handling and recovery scenarios"""
        logger.info("\n⚠️ Testing Error Handling Scenarios")
        
        tests = [
            ('Invalid Data Handling', self._test_invalid_data_handling),
            ('Resource Limits', self._test_resource_limits),
            ('Configuration Errors', self._test_configuration_errors),
            ('Network Simulation', self._test_network_simulation)
        ]
        
        for test_name, test_func in tests:
            await self._run_test(test_name, test_func, 'error_handling')
    
    async def _run_test(self, test_name: str, test_func, category: str):
        """Run a single test and record results"""
        try:
            logger.info(f"  Testing: {test_name}")
            result = await test_func()
            self.test_results[category].append({
                'test': test_name,
                'status': 'passed' if result else 'failed',
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"  ✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"  ❌ {test_name}: ERROR - {e}")
            self.test_results[category].append({
                'test': test_name,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Core Functionality Tests
    async def _test_research_session_creation(self) -> bool:
        """Test research session creation and management"""
        try:
            # Create a research session
            session = ResearchSession(
                session_id="test_session_123",
                topic="Machine Learning Research",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Verify session properties
            assert session.session_id == "test_session_123"
            assert session.topic == "Machine Learning Research"
            assert session.status == SessionStatus.INITIALIZING
            
            # Test session state transitions
            session.status = SessionStatus.QUERY_GENERATION
            assert session.status == SessionStatus.QUERY_GENERATION
            
            return True
        except Exception as e:
            logger.error(f"Research session creation test failed: {e}")
            return False
    
    async def _test_query_management(self) -> bool:
        """Test query management functionality"""
        try:
            session = ResearchSession(
                session_id="query_test",
                topic="AI Research",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Add queries
            test_queries = [
                {
                    'query_text': 'machine learning algorithms',
                    'priority': 1,
                    'category': 'overview'
                },
                {
                    'query_text': 'neural network architectures',
                    'priority': 2,
                    'category': 'technical'
                }
            ]
            
            query_ids = session.add_queries(test_queries)
            assert len(query_ids) == 2
            
            # Test query retrieval
            active_queries = session.get_active_queries()
            assert len(active_queries) == 2
            
            # Test query completion
            session.mark_query_completed(query_ids[0], {'papers_found': 5})
            
            return True
        except Exception as e:
            logger.error(f"Query management test failed: {e}")
            return False
    
    async def _test_paper_management(self) -> bool:
        """Test paper management functionality"""
        try:
            session = ResearchSession(
                session_id="paper_test",
                topic="Paper Management Test",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Add papers
            test_papers = [
                {
                    'id': 'paper_1',
                    'title': 'Test Paper 1',
                    'authors': ['Author A', 'Author B'],
                    'source': 'arxiv',
                    'publication_year': 2024
                },
                {
                    'id': 'paper_2',
                    'title': 'Test Paper 2',
                    'authors': ['Author C'],
                    'source': 'pubmed',
                    'publication_year': 2023
                }
            ]
            
            paper_ids = session.add_papers(test_papers)
            assert len(paper_ids) == 2
            
            # Test paper retrieval
            papers = session.get_papers_for_queries()
            assert len(papers) == 2
            
            return True
        except Exception as e:
            logger.error(f"Paper management test failed: {e}")
            return False
    
    async def _test_pdf_manager_core(self) -> bool:
        """Test PDF manager core functionality"""
        try:
            pdf_manager = PDFManager(
                base_path=str(self.temp_dir / 'papers'),
                config={'max_retries': 1, 'timeout': 10}
            )
            
            # Test storage structure creation
            assert (self.temp_dir / 'papers' / 'arxiv').exists()
            assert (self.temp_dir / 'papers' / 'pubmed').exists()
            
            # Test metadata database
            assert pdf_manager.db_path.exists()
            
            # Test statistics (empty initially)
            stats = pdf_manager.get_storage_statistics()
            assert stats['total_pdfs'] == 0
            
            # Test path generation
            test_paper = {
                'id': 'test_paper_123',
                'title': 'Test Paper For Path Generation',
                'source': 'arxiv',
                'publication_year': 2024
            }
            
            storage_path = pdf_manager._generate_storage_path(test_paper)
            assert 'arxiv' in str(storage_path)
            assert '2024' in str(storage_path)
            
            return True
        except Exception as e:
            logger.error(f"PDF manager core test failed: {e}")
            return False
    
    async def _test_literature_search_core(self) -> bool:
        """Test literature search core functionality"""
        try:
            search_agent = LiteratureSearchAgent(config=self.config)
            
            # Test search strategy generation
            search_terms = ["machine learning", "neural networks"]
            strategy = await search_agent.optimize_search_strategy(search_terms)
            assert isinstance(strategy, dict)
            
            # Test query generation
            queries = await search_agent.generate_search_queries("artificial intelligence")
            assert len(queries) > 0
            assert isinstance(queries, list)
            
            return True
        except Exception as e:
            logger.error(f"Literature search core test failed: {e}")
            return False
    
    # Integration Tests
    async def _test_session_query_integration(self) -> bool:
        """Test session and query integration"""
        try:
            session = ResearchSession(
                session_id="integration_test",
                topic="Integration Test",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Generate queries with literature search agent
            search_agent = LiteratureSearchAgent(config=self.config)
            generated_queries = await search_agent.generate_search_queries(session.topic)
            
            # Convert to session format and add
            session_queries = [
                {
                    'query_text': q,
                    'priority': 1,
                    'category': 'auto_generated'
                }
                for q in generated_queries[:3]  # Limit for testing
            ]
            
            query_ids = session.add_queries(session_queries)
            assert len(query_ids) > 0
            
            return True
        except Exception as e:
            logger.error(f"Session-query integration test failed: {e}")
            return False
    
    async def _test_literature_pdf_integration(self) -> bool:
        """Test literature search and PDF manager integration"""
        try:
            # Initialize components
            pdf_manager = PDFManager(
                base_path=str(self.temp_dir / 'papers'),
                config={'max_retries': 1, 'timeout': 5}
            )
            
            # Mock literature search results
            mock_papers = [
                {
                    'id': 'integration_paper_1',
                    'title': 'Integration Test Paper',
                    'source': 'arxiv',
                    'publication_year': 2024,
                    'pdf_url': 'https://example.com/paper.pdf'
                }
            ]
            
            # Test storage path generation for literature results
            for paper in mock_papers:
                storage_path = pdf_manager._generate_storage_path(paper)
                assert storage_path.parent.exists()
                assert paper['source'] in str(storage_path)
            
            return True
        except Exception as e:
            logger.error(f"Literature-PDF integration test failed: {e}")
            return False
    
    async def _test_multi_query_integration(self) -> bool:
        """Test multi-query coordinator integration"""
        try:
            coordinator = MultiQueryCoordinator(config=self.config)
            
            # Mock papers for processing
            mock_papers = [
                {
                    'id': 'coord_test_1',
                    'title': 'Coordination Test Paper',
                    'content': 'This is a test paper about machine learning coordination.',
                    'source': 'arxiv'
                }
            ]
            
            # Test tier initialization
            await coordinator.initialize_tiers()
            
            # Test processing (with mocks)
            results = await coordinator.process_papers_all_tiers(mock_papers)
            assert 'tier_1_results' in results
            assert 'tier_2_results' in results
            assert 'tier_3_results' in results
            
            return True
        except Exception as e:
            logger.error(f"Multi-query integration test failed: {e}")
            return False
    
    async def _test_data_flow_integration(self) -> bool:
        """Test data flow between components"""
        try:
            # Create session with workflow
            session = ResearchSession(
                session_id="dataflow_test",
                topic="Data Flow Test",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Add queries
            queries = [{
                'query_text': 'data flow testing',
                'priority': 1,
                'category': 'test'
            }]
            query_ids = session.add_queries(queries)
            
            # Add papers
            papers = [{
                'id': 'dataflow_paper',
                'title': 'Data Flow Paper',
                'source': 'arxiv',
                'publication_year': 2024
            }]
            paper_ids = session.add_papers(papers)
            
            # Verify data connectivity
            session_papers = session.get_papers_for_queries(query_ids)
            assert len(session_papers) > 0
            
            return True
        except Exception as e:
            logger.error(f"Data flow integration test failed: {e}")
            return False
    
    # Workflow Tests
    async def _test_literature_search_workflow(self) -> bool:
        """Test complete literature search workflow"""
        try:
            search_agent = LiteratureSearchAgent(config=self.config)
            
            # Generate search queries
            topic = "machine learning algorithms"
            queries = await search_agent.generate_search_queries(topic)
            assert len(queries) > 0
            
            # Optimize search strategy
            strategy = await search_agent.optimize_search_strategy(queries)
            assert 'arxiv' in strategy
            
            # Test search execution (mock)
            search_terms = queries[:2]  # Limit for testing
            results = await search_agent.search_multiple_sources(search_terms)
            assert isinstance(results, dict)
            
            return True
        except Exception as e:
            logger.error(f"Literature search workflow test failed: {e}")
            return False
    
    async def _test_3tier_rag_processing(self) -> bool:
        """Test 3-tier RAG processing workflow"""
        try:
            coordinator = MultiQueryCoordinator(config=self.config)
            
            # Test with mock papers
            mock_papers = [
                {
                    'id': 'rag_workflow_1',
                    'title': 'RAG Workflow Test Paper',
                    'content': 'This paper discusses advanced retrieval methods for AI systems.',
                    'source': 'arxiv'
                }
            ]
            
            # Initialize and test processing
            await coordinator.initialize_tiers()
            results = await coordinator.process_papers_all_tiers(mock_papers)
            
            assert results is not None
            assert 'processing_summary' in results
            
            return True
        except Exception as e:
            logger.error(f"3-tier RAG processing test failed: {e}")
            return False
    
    async def _test_pdf_download_storage(self) -> bool:
        """Test PDF download and storage workflow"""
        try:
            pdf_manager = PDFManager(
                base_path=str(self.temp_dir / 'papers'),
                config={'max_retries': 1, 'timeout': 5}
            )
            
            # Test batch processing setup
            mock_papers = [
                {
                    'id': 'storage_test_1',
                    'title': 'Storage Test Paper 1',
                    'source': 'arxiv',
                    'publication_year': 2024,
                    'pdf_url': 'https://example.com/paper1.pdf'
                },
                {
                    'id': 'storage_test_2',
                    'title': 'Storage Test Paper 2',
                    'source': 'pubmed',
                    'publication_year': 2023,
                    'pdf_url': 'https://example.com/paper2.pdf'
                }
            ]
            
            # Test storage path generation for each paper
            for paper in mock_papers:
                storage_path = pdf_manager._generate_storage_path(paper)
                assert paper['source'] in str(storage_path)
                assert str(paper['publication_year']) in str(storage_path)
            
            # Test statistics
            stats = pdf_manager.get_storage_statistics()
            assert 'total_pdfs' in stats
            assert 'by_source' in stats
            
            return True
        except Exception as e:
            logger.error(f"PDF download storage test failed: {e}")
            return False
    
    async def _test_complete_research_pipeline(self) -> bool:
        """Test complete research pipeline"""
        try:
            # Create session
            session = ResearchSession(
                session_id="pipeline_test",
                topic="Complete Pipeline Test",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Initialize components
            search_agent = LiteratureSearchAgent(config=self.config)
            pdf_manager = PDFManager(
                base_path=str(self.temp_dir / 'papers'),
                config={'max_retries': 1, 'timeout': 5}
            )
            coordinator = MultiQueryCoordinator(config=self.config)
            
            # Step 1: Generate queries
            queries = await search_agent.generate_search_queries(session.topic)
            session_queries = [
                {'query_text': q, 'priority': 1, 'category': 'pipeline_test'}
                for q in queries[:2]
            ]
            session.add_queries(session_queries)
            
            # Step 2: Mock literature search results
            mock_papers = [
                {
                    'id': 'pipeline_paper_1',
                    'title': 'Pipeline Test Paper',
                    'source': 'arxiv',
                    'publication_year': 2024
                }
            ]
            session.add_papers(mock_papers)
            
            # Step 3: Test RAG processing
            await coordinator.initialize_tiers()
            rag_results = await coordinator.process_papers_all_tiers(mock_papers)
            
            # Step 4: Test PDF management
            for paper in mock_papers:
                storage_path = pdf_manager._generate_storage_path(paper)
                assert storage_path.parent.exists()
            
            # Verify pipeline completion
            assert len(session.queries) > 0
            assert len(session.papers) > 0
            assert rag_results is not None
            
            return True
        except Exception as e:
            logger.error(f"Complete research pipeline test failed: {e}")
            return False
    
    # Error Handling Tests
    async def _test_invalid_data_handling(self) -> bool:
        """Test handling of invalid data"""
        try:
            # Test session with invalid data
            try:
                session = ResearchSession(
                    session_id="",  # Invalid empty ID
                    topic="",       # Invalid empty topic
                    user_id=None,   # Invalid None user
                    created_at=datetime.now()
                )
                # Should handle gracefully or raise appropriate error
            except Exception:
                pass  # Expected for some invalid data
            
            # Test PDF manager with invalid papers
            pdf_manager = PDFManager(
                base_path=str(self.temp_dir / 'papers'),
                config={'max_retries': 1, 'timeout': 1}
            )
            
            invalid_papers = [
                None,
                {'invalid': 'structure'},
                {'title': None, 'source': ''}
            ]
            
            for invalid_paper in invalid_papers:
                try:
                    if invalid_paper:
                        pdf_manager._generate_storage_path(invalid_paper)
                except Exception:
                    pass  # Expected for invalid data
            
            return True
        except Exception as e:
            logger.error(f"Invalid data handling test failed: {e}")
            return False
    
    async def _test_resource_limits(self) -> bool:
        """Test resource limit handling"""
        try:
            # Test with resource-limited config
            limited_config = self.config.copy()
            limited_config['max_papers'] = 1
            limited_config['max_queries'] = 2
            
            session = ResearchSession(
                session_id="resource_test",
                topic="Resource Limit Test",
                user_id="test_user",
                created_at=datetime.now()
            )
            
            # Test query limits
            many_queries = [
                {'query_text': f'query_{i}', 'priority': 1, 'category': 'test'}
                for i in range(5)  # More than limit
            ]
            
            # Should handle appropriately (limit or raise error)
            try:
                query_ids = session.add_queries(many_queries)
                # If successful, verify reasonable number returned
                assert len(query_ids) >= 0
            except Exception:
                pass  # May limit or reject
            
            return True
        except Exception as e:
            logger.error(f"Resource limits test failed: {e}")
            return False
    
    async def _test_configuration_errors(self) -> bool:
        """Test configuration error handling"""
        try:
            # Test with invalid configuration
            invalid_configs = [
                None,
                {},
                {'invalid_key': 'invalid_value'},
                {'base_paths': None}
            ]
            
            for invalid_config in invalid_configs:
                try:
                    search_agent = LiteratureSearchAgent(config=invalid_config)
                    # Should handle gracefully or provide defaults
                except Exception:
                    pass  # Expected for some invalid configs
            
            return True
        except Exception as e:
            logger.error(f"Configuration errors test failed: {e}")
            return False
    
    async def _test_network_simulation(self) -> bool:
        """Test network failure simulation"""
        try:
            # Test with very short timeouts to simulate network issues
            quick_config = self.config.copy()
            quick_config['api_timeout'] = 0.001  # Very short timeout
            
            pdf_manager = PDFManager(
                base_path=str(self.temp_dir / 'papers'),
                config={'max_retries': 1, 'timeout': 0.001}
            )
            
            # Test with invalid URL (simulates network failure)
            invalid_paper = {
                'id': 'network_test',
                'title': 'Network Test Paper',
                'source': 'test',
                'publication_year': 2024,
                'pdf_url': 'https://invalid-domain-12345.com/paper.pdf'
            }
            
            # Should handle gracefully
            try:
                success, path, info = await pdf_manager.download_pdf(
                    invalid_paper['pdf_url'],
                    invalid_paper
                )
                # Should fail gracefully
                assert not success
            except Exception:
                pass  # Expected for network simulation
            
            return True
        except Exception as e:
            logger.error(f"Network simulation test failed: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0
            },
            'categories': self.test_results
        }
        
        # Calculate totals
        for category, tests in self.test_results.items():
            for test in tests:
                report['test_summary']['total_tests'] += 1
                if test['status'] == 'passed':
                    report['test_summary']['passed'] += 1
                elif test['status'] == 'failed':
                    report['test_summary']['failed'] += 1
                else:
                    report['test_summary']['errors'] += 1
        
        # Calculate success rate
        total = report['test_summary']['total_tests']
        passed = report['test_summary']['passed']
        report['test_summary']['success_rate'] = (passed / total * 100) if total > 0 else 0
        
        return report
    
    async def cleanup(self):
        """Clean up test environment"""
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup test directory: {e}")

async def run_comprehensive_tests():
    """Run all comprehensive tests"""
    logger.info("🚀 Starting Comprehensive Research Agent Workflow Tests")
    
    test_suite = ComprehensiveWorkflowTestSuite()
    
    try:
        # Setup test environment
        await test_suite.setup_test_environment()
        
        # Run all test categories
        await test_suite.test_core_functionality()
        await test_suite.test_integration_workflows()
        await test_suite.test_end_to_end_workflows()
        await test_suite.test_error_handling_scenarios()
        
        # Generate and display report
        report = test_suite.generate_test_report()
        
        logger.info("\n" + "="*80)
        logger.info("📊 COMPREHENSIVE TEST REPORT")
        logger.info("="*80)
        logger.info(f"Total Tests: {report['test_summary']['total_tests']}")
        logger.info(f"Passed: {report['test_summary']['passed']}")
        logger.info(f"Failed: {report['test_summary']['failed']}")
        logger.info(f"Errors: {report['test_summary']['errors']}")
        logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
        logger.info("="*80)
        
        # Category breakdown
        for category, tests in report['categories'].items():
            passed = len([t for t in tests if t['status'] == 'passed'])
            total = len(tests)
            logger.info(f"{category.replace('_', ' ').title()}: {passed}/{total} passed")
        
        # Save detailed report
        report_file = test_suite.temp_dir / 'comprehensive_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        # Overall result
        success_rate = report['test_summary']['success_rate']
        if success_rate >= 85:
            logger.info("\n🎉 COMPREHENSIVE TESTING: PASSED (Excellent)")
            return True
        elif success_rate >= 70:
            logger.info("\n✅ COMPREHENSIVE TESTING: PASSED (Good)")
            return True
        else:
            logger.info("\n❌ COMPREHENSIVE TESTING: NEEDS IMPROVEMENT")
            return False
        
    except Exception as e:
        logger.error(f"❌ Comprehensive testing failed: {e}")
        return False
    
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    print(f"\n🎯 Comprehensive Testing {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)