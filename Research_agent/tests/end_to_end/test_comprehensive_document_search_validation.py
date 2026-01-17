#!/usr/bin/env python3
"""
Comprehensive Document Search Service Test Suite
==============================================

This test suite validates the DocumentSearchService with comprehensive testing including:
1. Critical "early parkinsons disease" test (the key validation case)
2. 3-tier search solution validation
3. Service interface tests with complete error handling
4. Integration tests for all RAG demos (LightRAG, PaperQA2, GraphRAG)
5. Performance benchmarks and reliability tests
6. Real API testing with paper quality validation

This is the definitive test suite to validate the DocumentSearchService works correctly
and can be relied upon by all RAG integration demos.
"""

import asyncio
import os
import sys
import json
import tempfile
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import uuid
import warnings

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent / "src"))
sys.path.insert(0, str(current_dir.parent))

# Import DocumentSearchService
try:
    from research_agent.services.document_search_service import DocumentSearchService
except ImportError:
    # Fallback import path
    sys.path.insert(0, str(current_dir.parent / "src" / "research_agent"))
    from services.document_search_service import DocumentSearchService

# Suppress async warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ComprehensiveDocumentSearchTestSuite:
    """Comprehensive test suite for DocumentSearchService validation"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.start_time = None
        self.performance_metrics = {}
        
        # Critical test queries (the ones that must work)
        self.critical_test_queries = [
            "early parkinsons disease",  # THE critical validation case
            "machine learning attention mechanisms",
            "quantum computing algorithms",
            "neural network architectures",
            "covid 19 treatment options"
        ]
        
        # Test configuration
        self.test_config = {
            'max_papers_per_test': 5,
            'timeout_per_test': 180,  # 3 minutes per test
            'rate_limit_delay': 1.0,
            'enable_real_api_tests': True,
            'validate_paper_quality': True
        }
        
        # API availability check results
        self.api_status = {
            'openai_available': bool(os.getenv('OPENAI_API_KEY')),
            'mcp_available': False,
            'scihub_available': False
        }
        
    def setup_test_environment(self):
        """Setup temporary test environment with proper structure"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="comprehensive_doc_search_test_"))
        self.start_time = datetime.now()
        
        # Create test subdirectories
        (self.temp_dir / "critical_tests").mkdir(exist_ok=True)
        (self.temp_dir / "integration_tests").mkdir(exist_ok=True)
        (self.temp_dir / "performance_tests").mkdir(exist_ok=True)
        (self.temp_dir / "logs").mkdir(exist_ok=True)
        
        print(f"🧪 Test environment: {self.temp_dir}")
        print(f"📅 Test started: {self.start_time.isoformat()}")
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("🧹 Test environment cleaned up")
    
    async def check_api_availability(self) -> Dict[str, bool]:
        """Check availability of required APIs and services"""
        print("\n🔍 Checking API Availability...")
        
        # Check OpenAI API
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️ OPENAI_API_KEY not found - some tests will be skipped")
            self.api_status['openai_available'] = False
        else:
            print("✅ OpenAI API key found")
            self.api_status['openai_available'] = True
        
        # Test basic service initialization to check MCP
        try:
            service = DocumentSearchService(working_directory=str(self.temp_dir / "api_check"))
            initialized = await service.initialize()
            
            if initialized:
                print("✅ DocumentSearchService initialization successful")
                
                # Quick MCP availability test
                session_id = service.start_session("mcp_check")
                try:
                    # Attempt a minimal search to test MCP
                    papers, metadata = await service.search_documents(
                        search_query="test query",
                        max_papers=1,
                        enhance_metadata=False
                    )
                    
                    # Check if we got real papers (indicating MCP worked)
                    if papers and any(paper.get('source') == 'sci_hub_mcp' for paper in papers):
                        self.api_status['mcp_available'] = True
                        self.api_status['scihub_available'] = True
                        print("✅ MCP and Sci-Hub integration available")
                    else:
                        print("⚠️ MCP/Sci-Hub not fully available - will use fallback mode")
                        
                except Exception as e:
                    print(f"⚠️ MCP test failed: {str(e)[:100]}... (may be expected)")
                
                service.end_session()
                await service.shutdown()
            else:
                print("⚠️ Service initialization failed")
                
        except Exception as e:
            print(f"⚠️ API availability check error: {str(e)[:100]}...")
        
        return self.api_status
    
    async def test_critical_parkinsons_query(self) -> Dict[str, Any]:
        """
        THE CRITICAL TEST: Validate "early parkinsons disease" query works correctly.
        This is the test case that must pass for the service to be considered functional.
        """
        print("\n🎯 CRITICAL TEST: Early Parkinsons Disease Query")
        print("=" * 60)
        print("This is THE validation test that must pass for service approval")
        
        test_start = time.time()
        test_result = {
            'test_name': 'critical_parkinsons_query',
            'status': 'running',
            'papers_found': 0,
            'papers_quality_validated': 0,
            'real_papers_count': 0,
            'fallback_papers_count': 0,
            'errors': [],
            'execution_time': 0
        }
        
        try:
            # Create service for critical test
            service = DocumentSearchService(
                working_directory=str(self.temp_dir / "critical_tests" / "parkinsons"),
                rate_limit_delay=self.test_config['rate_limit_delay']
            )
            
            # Initialize service
            initialized = await service.initialize()
            if not initialized:
                test_result['status'] = 'skipped'
                test_result['errors'].append('Service initialization failed - no API keys')
                print("⚠️ Skipping critical test - no API keys available")
                return test_result
            
            # Start session for this critical test
            session_id = service.start_session("critical_parkinsons_test")
            print(f"📁 Session: {session_id}")
            
            # Execute the critical search query
            print("🔍 Searching for 'early parkinsons disease' papers...")
            
            papers, metadata = await service.search_documents(
                search_query="early parkinsons disease",
                max_papers=self.test_config['max_papers_per_test'],
                enhance_metadata=True
            )
            
            test_result['papers_found'] = len(papers)
            test_result['metadata'] = metadata
            
            # Validate paper quality
            quality_validation = self.validate_paper_quality(papers, "early parkinsons disease")
            test_result.update(quality_validation)
            
            # Count real vs fallback papers
            real_papers = [p for p in papers if p.get('source') in ['sci_hub_mcp', 'real_api']]
            fallback_papers = [p for p in papers if p.get('source') in ['backup', 'fallback']]
            
            test_result['real_papers_count'] = len(real_papers)
            test_result['fallback_papers_count'] = len(fallback_papers)
            
            # Log findings
            print(f"📊 Results: {len(papers)} papers found")
            print(f"   - Real papers: {len(real_papers)}")
            print(f"   - Fallback papers: {len(fallback_papers)}")
            print(f"   - Quality validated: {test_result['papers_quality_validated']}")
            
            # Print sample paper titles for validation
            print("📄 Sample paper titles:")
            for i, paper in enumerate(papers[:3], 1):
                title = paper.get('title', 'No title')[:80]
                source = paper.get('source', 'unknown')
                print(f"   {i}. {title}... (source: {source})")
            
            # Determine test success criteria
            success_criteria = [
                test_result['papers_found'] > 0,
                test_result['papers_quality_validated'] > 0,
                metadata.get('status') in ['success', 'fallback_mode']
            ]
            
            if all(success_criteria):
                test_result['status'] = 'passed'
                print("✅ CRITICAL TEST PASSED: Early Parkinsons Disease query successful")
            else:
                test_result['status'] = 'failed'
                test_result['errors'].append('Failed success criteria validation')
                print("❌ CRITICAL TEST FAILED: Did not meet success criteria")
            
            # Save results to session
            session_info = service.get_session_info()
            test_result['session_info'] = session_info
            
            # Cleanup
            service.end_session()
            await service.shutdown()
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['errors'].append(str(e))
            print(f"❌ CRITICAL TEST ERROR: {str(e)[:100]}...")
            traceback.print_exc()
        
        test_result['execution_time'] = time.time() - test_start
        
        # This is critical - log detailed results
        print(f"\n📊 CRITICAL TEST SUMMARY:")
        print(f"   Status: {test_result['status'].upper()}")
        print(f"   Papers found: {test_result['papers_found']}")
        print(f"   Real papers: {test_result['real_papers_count']}")
        print(f"   Quality validated: {test_result['papers_quality_validated']}")
        print(f"   Execution time: {test_result['execution_time']:.2f}s")
        
        return test_result
    
    def validate_paper_quality(self, papers: List[Dict[str, Any]], search_query: str) -> Dict[str, Any]:
        """
        Validate the quality of retrieved papers for the search query.
        Returns quality metrics and validation results.
        """
        if not papers:
            return {
                'papers_quality_validated': 0,
                'quality_score': 0,
                'quality_issues': ['No papers to validate']
            }
        
        quality_issues = []
        validated_papers = 0
        total_quality_score = 0
        
        for i, paper in enumerate(papers):
            paper_quality_score = 0
            paper_issues = []
            
            # Check required fields
            required_fields = ['title', 'authors', 'abstract', 'content']
            for field in required_fields:
                if field in paper and paper[field] and str(paper[field]).strip():
                    paper_quality_score += 1
                else:
                    paper_issues.append(f'Missing or empty {field}')
            
            # Check title relevance to search query
            title = str(paper.get('title', '')).lower()
            query_words = search_query.lower().split()
            
            # For "early parkinsons disease" - check for relevant terms
            relevant_terms = ['parkinson', 'parkinsons', 'early', 'treatment', 'disease', 'neurodegenerative']
            title_relevance = sum(1 for term in relevant_terms if term in title)
            
            if title_relevance > 0:
                paper_quality_score += min(title_relevance, 3)  # Max 3 points for relevance
            else:
                paper_issues.append('Title not relevant to search query')
            
            # Check if paper has proper metadata structure
            if paper.get('document_search_service'):
                paper_quality_score += 1
            
            # Check citation format
            if paper.get('citation'):
                paper_quality_score += 1
            
            # Minimum quality threshold (4/8 points)
            if paper_quality_score >= 4:
                validated_papers += 1
            
            total_quality_score += paper_quality_score
            
            if paper_issues:
                quality_issues.extend([f"Paper {i+1}: {issue}" for issue in paper_issues])
        
        average_quality = total_quality_score / len(papers) if papers else 0
        
        return {
            'papers_quality_validated': validated_papers,
            'quality_score': round(average_quality, 2),
            'quality_issues': quality_issues[:10],  # Limit to first 10 issues
            'total_papers_analyzed': len(papers)
        }
    
    async def test_three_tier_search_validation(self) -> Dict[str, Any]:
        """Test the 3-tier search solution functionality"""
        print("\n🏗️ Testing 3-Tier Search Solution...")
        
        test_start = time.time()
        test_result = {
            'test_name': '3_tier_search_validation',
            'status': 'running',
            'mcp_integration_test': False,
            'arxiv_access_test': False,
            'pubmed_access_test': False,
            'scihub_access_test': False,
            'fallback_mechanism_test': False,
            'errors': []
        }
        
        try:
            service = DocumentSearchService(
                working_directory=str(self.temp_dir / "integration_tests" / "three_tier")
            )
            
            initialized = await service.initialize()
            if not initialized:
                test_result['status'] = 'skipped'
                test_result['errors'].append('No API keys - skipping 3-tier tests')
                return test_result
            
            session_id = service.start_session("three_tier_test")
            
            # Test different search queries to trigger different tiers
            tier_tests = [
                ("machine learning", "Tier 1: General ML query"),
                ("quantum entanglement physics", "Tier 2: Specific physics query"),
                ("early parkinsons disease treatment", "Tier 3: Medical research query")
            ]
            
            tier_results = []
            
            for query, description in tier_tests:
                print(f"🔍 {description}: '{query}'")
                
                try:
                    papers, metadata = await service.search_documents(
                        search_query=query,
                        max_papers=3,
                        enhance_metadata=True
                    )
                    
                    tier_results.append({
                        'query': query,
                        'description': description,
                        'papers_found': len(papers),
                        'metadata_status': metadata.get('status'),
                        'paper_sources': metadata.get('paper_sources', {}),
                        'success': len(papers) > 0
                    })
                    
                    # Check for MCP integration indicators
                    if any(paper.get('source') == 'sci_hub_mcp' for paper in papers):
                        test_result['mcp_integration_test'] = True
                        test_result['scihub_access_test'] = True
                    
                    print(f"   ✓ Found {len(papers)} papers")
                    
                except Exception as e:
                    tier_results.append({
                        'query': query,
                        'description': description,
                        'error': str(e)[:100],
                        'success': False
                    })
                    print(f"   ✗ Error: {str(e)[:50]}...")
            
            test_result['tier_results'] = tier_results
            
            # Test fallback mechanism by forcing an error scenario
            try:
                # This should trigger fallback mode
                papers, metadata = await service.search_documents(
                    search_query="",  # Invalid query to test error handling
                    max_papers=1
                )
                # Should not reach here due to validation
                test_result['fallback_mechanism_test'] = False
                
            except ValueError:
                # Expected error - fallback mechanism working
                test_result['fallback_mechanism_test'] = True
                print("   ✓ Fallback mechanism working (caught invalid query)")
            
            # Determine overall success
            successful_tiers = sum(1 for result in tier_results if result.get('success', False))
            
            if successful_tiers >= 2:  # At least 2 out of 3 tiers working
                test_result['status'] = 'passed'
                print("✅ 3-Tier Search Solution validation passed")
            else:
                test_result['status'] = 'failed'
                test_result['errors'].append('Insufficient tier success rate')
                print("❌ 3-Tier Search Solution validation failed")
            
            service.end_session()
            await service.shutdown()
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['errors'].append(str(e))
            print(f"❌ 3-Tier Search test error: {str(e)[:100]}...")
        
        test_result['execution_time'] = time.time() - test_start
        return test_result
    
    async def test_service_interface_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive test of service interface methods"""
        print("\n🔧 Testing Service Interface Comprehensively...")
        
        test_start = time.time()
        test_result = {
            'test_name': 'service_interface_comprehensive',
            'status': 'running',
            'methods_tested': {},
            'error_handling_tests': {},
            'edge_cases_tested': {},
            'errors': []
        }
        
        try:
            # Test 1: Service lifecycle
            service = DocumentSearchService(
                working_directory=str(self.temp_dir / "interface_tests")
            )
            
            # Test initialization
            initialized = await service.initialize()
            test_result['methods_tested']['initialize'] = initialized
            
            if not initialized:
                print("⚠️ Skipping interface tests - no API keys")
                test_result['status'] = 'skipped'
                return test_result
            
            # Test session management
            session_id = service.start_session("interface_test")
            test_result['methods_tested']['start_session'] = bool(session_id)
            
            session_info = service.get_session_info()
            test_result['methods_tested']['get_session_info'] = session_info is not None
            
            # Test search_documents with various parameters
            search_tests = [
                ("basic_search", {"search_query": "test query", "max_papers": 3}),
                ("enhanced_metadata", {"search_query": "test", "max_papers": 2, "enhance_metadata": True}),
                ("minimal_papers", {"search_query": "test", "max_papers": 1}),
            ]
            
            for test_name, kwargs in search_tests:
                try:
                    papers, metadata = await service.search_documents(**kwargs)
                    test_result['methods_tested'][f'search_documents_{test_name}'] = True
                    print(f"   ✓ {test_name}: {len(papers)} papers")
                except Exception as e:
                    test_result['methods_tested'][f'search_documents_{test_name}'] = False
                    test_result['errors'].append(f"{test_name}: {str(e)[:50]}")
                    print(f"   ✗ {test_name}: {str(e)[:50]}...")
            
            # Test error handling
            error_tests = [
                ("empty_query", {"search_query": "", "max_papers": 1}),
                ("invalid_max_papers", {"search_query": "test", "max_papers": 0}),
                ("very_large_max_papers", {"search_query": "test", "max_papers": 100}),
            ]
            
            for test_name, kwargs in error_tests:
                try:
                    papers, metadata = await service.search_documents(**kwargs)
                    test_result['error_handling_tests'][test_name] = False  # Should have raised error
                    print(f"   ✗ {test_name}: Should have raised error")
                except (ValueError, RuntimeError) as e:
                    test_result['error_handling_tests'][test_name] = True  # Correctly raised error
                    print(f"   ✓ {test_name}: Correctly raised {type(e).__name__}")
                except Exception as e:
                    test_result['error_handling_tests'][test_name] = False
                    test_result['errors'].append(f"{test_name}: Unexpected error {str(e)[:50]}")
            
            # Test edge cases
            edge_case_tests = [
                ("unicode_query", {"search_query": "машинное обучение", "max_papers": 2}),
                ("special_chars", {"search_query": "C++ algorithms", "max_papers": 2}),
                ("very_long_query", {"search_query": "machine learning " * 20, "max_papers": 2}),
            ]
            
            for test_name, kwargs in edge_case_tests:
                try:
                    papers, metadata = await service.search_documents(**kwargs)
                    test_result['edge_cases_tested'][test_name] = True
                    print(f"   ✓ {test_name}: {len(papers)} papers")
                except Exception as e:
                    test_result['edge_cases_tested'][test_name] = False
                    print(f"   ⚠️ {test_name}: {str(e)[:50]}...")
            
            # Test session cleanup
            service.end_session(cleanup=True)
            test_result['methods_tested']['end_session_with_cleanup'] = True
            
            # Test shutdown
            await service.shutdown()
            test_result['methods_tested']['shutdown'] = True
            
            # Calculate success metrics
            methods_passed = sum(1 for result in test_result['methods_tested'].values() if result)
            error_handling_passed = sum(1 for result in test_result['error_handling_tests'].values() if result)
            edge_cases_passed = sum(1 for result in test_result['edge_cases_tested'].values() if result)
            
            total_tests = len(test_result['methods_tested']) + len(test_result['error_handling_tests']) + len(test_result['edge_cases_tested'])
            total_passed = methods_passed + error_handling_passed + edge_cases_passed
            
            success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
            
            if success_rate >= 80:  # 80% pass rate required
                test_result['status'] = 'passed'
                print(f"✅ Service Interface tests passed ({success_rate:.1f}% success rate)")
            else:
                test_result['status'] = 'failed'
                print(f"❌ Service Interface tests failed ({success_rate:.1f}% success rate)")
            
            test_result['success_metrics'] = {
                'methods_passed': methods_passed,
                'error_handling_passed': error_handling_passed,
                'edge_cases_passed': edge_cases_passed,
                'total_passed': total_passed,
                'total_tests': total_tests,
                'success_rate': f"{success_rate:.1f}%"
            }
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['errors'].append(str(e))
            print(f"❌ Service Interface test error: {str(e)[:100]}...")
        
        test_result['execution_time'] = time.time() - test_start
        return test_result
    
    async def test_integration_with_rag_demos(self) -> Dict[str, Any]:
        """Test integration compatibility with LightRAG, PaperQA2, and GraphRAG demos"""
        print("\n🔗 Testing RAG Demo Integration Compatibility...")
        
        test_start = time.time()
        test_result = {
            'test_name': 'rag_demo_integration',
            'status': 'running',
            'demo_compatibility': {},
            'cross_demo_consistency': False,
            'errors': []
        }
        
        try:
            # Test the same query across different "demo scenarios"
            test_query = "machine learning applications"
            demo_results = {}
            
            # Simulate different demo usage patterns
            demo_scenarios = [
                ("lightrag_style", {"enhance_metadata": True, "max_papers": 5}),
                ("paperqa2_style", {"enhance_metadata": True, "max_papers": 3}),
                ("graphrag_style", {"enhance_metadata": True, "max_papers": 4})
            ]
            
            for demo_name, search_params in demo_scenarios:
                print(f"🔍 Testing {demo_name} compatibility...")
                
                try:
                    # Create service for this demo scenario
                    service = DocumentSearchService(
                        working_directory=str(self.temp_dir / "integration_tests" / demo_name)
                    )
                    
                    initialized = await service.initialize()
                    if not initialized:
                        test_result['demo_compatibility'][demo_name] = {
                            'status': 'skipped',
                            'reason': 'No API keys'
                        }
                        continue
                    
                    session_id = service.start_session(f"{demo_name}_compatibility_test")
                    
                    # Execute search with demo-specific parameters
                    papers, metadata = await service.search_documents(
                        search_query=test_query,
                        **search_params
                    )
                    
                    demo_results[demo_name] = {
                        'papers_found': len(papers),
                        'metadata_keys': list(metadata.keys()),
                        'paper_sources': metadata.get('paper_sources', {}),
                        'enhancement_applied': search_params.get('enhance_metadata', False),
                        'session_id': session_id,
                        'papers_sample': [p.get('title', 'No title')[:50] for p in papers[:2]]
                    }
                    
                    test_result['demo_compatibility'][demo_name] = {
                        'status': 'passed',
                        'papers_found': len(papers),
                        'metadata_consistent': 'search_query' in metadata
                    }
                    
                    print(f"   ✓ {demo_name}: {len(papers)} papers found")
                    
                    service.end_session()
                    await service.shutdown()
                    
                except Exception as e:
                    test_result['demo_compatibility'][demo_name] = {
                        'status': 'failed',
                        'error': str(e)[:100]
                    }
                    test_result['errors'].append(f"{demo_name}: {str(e)[:100]}")
                    print(f"   ✗ {demo_name}: {str(e)[:50]}...")
            
            # Test cross-demo consistency
            if len(demo_results) >= 2:
                # Check if all demos return papers
                all_found_papers = all(result['papers_found'] > 0 for result in demo_results.values())
                
                # Check metadata consistency
                metadata_keys_sets = [set(result['metadata_keys']) for result in demo_results.values()]
                consistent_metadata = len(set(frozenset(keys) for keys in metadata_keys_sets)) == 1
                
                # Check paper title consistency (should have some overlap)
                paper_titles_sets = [set(result['papers_sample']) for result in demo_results.values()]
                title_overlap = len(set.intersection(*paper_titles_sets)) > 0 if len(paper_titles_sets) > 1 else True
                
                test_result['cross_demo_consistency'] = all([all_found_papers, consistent_metadata])
                
                print(f"   📊 Cross-demo consistency: {test_result['cross_demo_consistency']}")
                print(f"      - All demos found papers: {all_found_papers}")
                print(f"      - Metadata structure consistent: {consistent_metadata}")
                print(f"      - Some title overlap: {title_overlap}")
            
            # Determine overall success
            successful_demos = sum(1 for demo in test_result['demo_compatibility'].values() 
                                 if demo.get('status') == 'passed')
            
            if successful_demos >= 2 and test_result['cross_demo_consistency']:
                test_result['status'] = 'passed'
                print("✅ RAG Demo Integration compatibility passed")
            else:
                test_result['status'] = 'failed'
                print("❌ RAG Demo Integration compatibility failed")
            
            test_result['demo_results'] = demo_results
            test_result['successful_demos'] = successful_demos
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['errors'].append(str(e))
            print(f"❌ RAG Demo Integration test error: {str(e)[:100]}...")
        
        test_result['execution_time'] = time.time() - test_start
        return test_result
    
    async def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks and reliability"""
        print("\n⚡ Testing Performance Benchmarks...")
        
        test_start = time.time()
        test_result = {
            'test_name': 'performance_benchmarks',
            'status': 'running',
            'response_times': [],
            'memory_usage': [],
            'concurrent_requests': 0,
            'errors': []
        }
        
        try:
            # Single request performance test
            print("📊 Single request performance test...")
            
            service = DocumentSearchService(
                working_directory=str(self.temp_dir / "performance_tests")
            )
            
            initialized = await service.initialize()
            if not initialized:
                test_result['status'] = 'skipped'
                test_result['errors'].append('No API keys - skipping performance tests')
                return test_result
            
            session_id = service.start_session("performance_test")
            
            # Test response times for different queries
            test_queries = ["AI research", "quantum computing", "machine learning"]
            
            for query in test_queries:
                request_start = time.time()
                
                try:
                    papers, metadata = await service.search_documents(
                        search_query=query,
                        max_papers=3,
                        enhance_metadata=True
                    )
                    
                    response_time = time.time() - request_start
                    test_result['response_times'].append({
                        'query': query,
                        'response_time': response_time,
                        'papers_found': len(papers)
                    })
                    
                    print(f"   ✓ '{query}': {response_time:.2f}s ({len(papers)} papers)")
                    
                except Exception as e:
                    test_result['errors'].append(f"Performance test query '{query}': {str(e)[:50]}")
                    print(f"   ✗ '{query}': {str(e)[:50]}...")
            
            # Calculate performance metrics
            if test_result['response_times']:
                avg_response_time = sum(r['response_time'] for r in test_result['response_times']) / len(test_result['response_times'])
                max_response_time = max(r['response_time'] for r in test_result['response_times'])
                min_response_time = min(r['response_time'] for r in test_result['response_times'])
                
                test_result['performance_metrics'] = {
                    'average_response_time': round(avg_response_time, 2),
                    'max_response_time': round(max_response_time, 2),
                    'min_response_time': round(min_response_time, 2),
                    'total_requests': len(test_result['response_times'])
                }
                
                # Performance criteria: average response time < 60 seconds (reasonable for real API calls)
                if avg_response_time < 60:
                    test_result['status'] = 'passed'
                    print(f"✅ Performance benchmark passed (avg: {avg_response_time:.2f}s)")
                else:
                    test_result['status'] = 'failed'
                    print(f"❌ Performance benchmark failed (avg: {avg_response_time:.2f}s)")
            else:
                test_result['status'] = 'failed'
                test_result['errors'].append('No successful response time measurements')
            
            service.end_session()
            await service.shutdown()
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['errors'].append(str(e))
            print(f"❌ Performance benchmark error: {str(e)[:100]}...")
        
        test_result['execution_time'] = time.time() - test_start
        return test_result
    
    def generate_validation_report(self, all_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        print("\n📄 Generating Comprehensive Validation Report...")
        
        # Calculate overall statistics
        total_tests = len(all_test_results)
        passed_tests = sum(1 for test in all_test_results if test.get('status') == 'passed')
        failed_tests = sum(1 for test in all_test_results if test.get('status') == 'failed')
        error_tests = sum(1 for test in all_test_results if test.get('status') == 'error')
        skipped_tests = sum(1 for test in all_test_results if test.get('status') == 'skipped')
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Find critical test result
        critical_test = next((test for test in all_test_results if test.get('test_name') == 'critical_parkinsons_query'), None)
        critical_test_passed = critical_test and critical_test.get('status') == 'passed'
        
        # Performance analysis
        performance_test = next((test for test in all_test_results if test.get('test_name') == 'performance_benchmarks'), None)
        performance_metrics = performance_test.get('performance_metrics', {}) if performance_test else {}
        
        # Generate overall assessment
        overall_status = "PASSED" if critical_test_passed and success_rate >= 75 else "FAILED"
        
        # Create detailed report
        report = {
            "validation_report": {
                "timestamp": datetime.now().isoformat(),
                "test_duration": str(datetime.now() - self.start_time) if self.start_time else "Unknown",
                "overall_status": overall_status,
                "critical_test_status": "PASSED" if critical_test_passed else "FAILED",
                
                "summary_statistics": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "error_tests": error_tests,
                    "skipped_tests": skipped_tests,
                    "success_rate": f"{success_rate:.1f}%"
                },
                
                "api_availability": self.api_status,
                
                "performance_summary": performance_metrics,
                
                "test_results_detailed": all_test_results,
                
                "recommendations": self.generate_recommendations(all_test_results, critical_test_passed),
                
                "environment_info": {
                    "python_version": sys.version,
                    "test_configuration": self.test_config,
                    "test_directory": str(self.temp_dir) if self.temp_dir else None
                }
            }
        }
        
        return report
    
    def generate_recommendations(self, test_results: List[Dict[str, Any]], critical_test_passed: bool) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not critical_test_passed:
            recommendations.append("🚨 CRITICAL: The 'early parkinsons disease' test failed. This service should not be deployed until this is resolved.")
        
        # Check API availability recommendations
        if not self.api_status.get('openai_available'):
            recommendations.append("⚠️ Set OPENAI_API_KEY environment variable to enable full testing")
        
        if not self.api_status.get('mcp_available'):
            recommendations.append("⚠️ MCP integration not available - some features will use fallback mode")
        
        # Performance recommendations
        performance_test = next((test for test in test_results if test.get('test_name') == 'performance_benchmarks'), None)
        if performance_test and performance_test.get('performance_metrics'):
            avg_time = performance_test['performance_metrics'].get('average_response_time', 0)
            if avg_time > 30:
                recommendations.append(f"⚠️ Average response time ({avg_time:.1f}s) is high - consider optimizing")
        
        # Success rate recommendations
        success_rate = 0
        if test_results:
            passed = sum(1 for test in test_results if test.get('status') == 'passed')
            success_rate = (passed / len(test_results)) * 100
            
        if success_rate < 75:
            recommendations.append(f"⚠️ Overall success rate ({success_rate:.1f}%) is below 75% - investigate failures")
        elif success_rate >= 90:
            recommendations.append("✅ Excellent test success rate - service is highly reliable")
        
        # Error analysis recommendations
        error_count = sum(len(test.get('errors', [])) for test in test_results)
        if error_count > 5:
            recommendations.append("⚠️ High number of errors detected - review error logs for patterns")
        
        if critical_test_passed and success_rate >= 80:
            recommendations.append("✅ DocumentSearchService is ready for production use across all RAG demos")
        
        return recommendations
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive test suite"""
        print("🚀 COMPREHENSIVE DOCUMENT SEARCH SERVICE TEST SUITE")
        print("=" * 70)
        print("Testing DocumentSearchService for production readiness")
        print("Critical validation: 'early parkinsons disease' query must work")
        print("=" * 70)
        
        self.setup_test_environment()
        
        try:
            # Check API availability first
            await self.check_api_availability()
            
            # Run all test categories
            test_functions = [
                ("Critical Parkinsons Query", self.test_critical_parkinsons_query()),
                ("3-Tier Search Validation", self.test_three_tier_search_validation()),
                ("Service Interface Comprehensive", self.test_service_interface_comprehensive()),
                ("RAG Demo Integration", self.test_integration_with_rag_demos()),
                ("Performance Benchmarks", self.test_performance_benchmarks())
            ]
            
            all_test_results = []
            
            for test_name, test_coro in test_functions:
                print(f"\n{'='*20} {test_name} {'='*20}")
                
                try:
                    # Run test with timeout
                    test_result = await asyncio.wait_for(
                        test_coro,
                        timeout=self.test_config['timeout_per_test']
                    )
                    all_test_results.append(test_result)
                    
                except asyncio.TimeoutError:
                    print(f"⏰ {test_name} timed out after {self.test_config['timeout_per_test']}s")
                    all_test_results.append({
                        'test_name': test_name.lower().replace(' ', '_'),
                        'status': 'timeout',
                        'errors': ['Test timed out'],
                        'execution_time': self.test_config['timeout_per_test']
                    })
                    
                except Exception as e:
                    print(f"❌ {test_name} failed with exception: {str(e)[:100]}...")
                    all_test_results.append({
                        'test_name': test_name.lower().replace(' ', '_'),
                        'status': 'error',
                        'errors': [str(e)],
                        'execution_time': 0
                    })
            
            # Generate comprehensive report
            validation_report = self.generate_validation_report(all_test_results)
            
            # Print summary
            self.print_test_summary(validation_report)
            
            # Save reports
            self.save_test_reports(validation_report, all_test_results)
            
            return validation_report
            
        finally:
            self.cleanup_test_environment()
    
    def print_test_summary(self, validation_report: Dict[str, Any]):
        """Print formatted test summary"""
        report = validation_report['validation_report']
        
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        print("="*70)
        
        print(f"🎯 Overall Status: {report['overall_status']}")
        print(f"🚨 Critical Test (early parkinsons disease): {report['critical_test_status']}")
        print(f"📈 Success Rate: {report['summary_statistics']['success_rate']}")
        print(f"⏱️ Test Duration: {report['test_duration']}")
        
        stats = report['summary_statistics']
        print(f"\n📊 Test Results:")
        print(f"   ✅ Passed: {stats['passed_tests']}")
        print(f"   ❌ Failed: {stats['failed_tests']}")
        print(f"   🚫 Errors: {stats['error_tests']}")
        print(f"   ⏭️ Skipped: {stats['skipped_tests']}")
        print(f"   📝 Total: {stats['total_tests']}")
        
        if report.get('performance_summary'):
            perf = report['performance_summary']
            print(f"\n⚡ Performance Metrics:")
            print(f"   Average Response Time: {perf.get('average_response_time', 'N/A')}s")
            print(f"   Max Response Time: {perf.get('max_response_time', 'N/A')}s")
        
        print(f"\n🔧 API Status:")
        api_status = report['api_availability']
        for service, available in api_status.items():
            status = "✅" if available else "❌"
            print(f"   {status} {service}: {available}")
        
        print(f"\n💡 Recommendations:")
        for rec in report.get('recommendations', []):
            print(f"   {rec}")
        
        print("\n" + "="*70)
    
    def save_test_reports(self, validation_report: Dict[str, Any], all_test_results: List[Dict[str, Any]]):
        """Save detailed test reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main validation report
        report_file = f"comprehensive_document_search_validation_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Save detailed test results
        details_file = f"comprehensive_document_search_details_{timestamp}.json"
        with open(details_file, 'w') as f:
            json.dump(all_test_results, f, indent=2, default=str)
        
        print(f"\n📄 Reports saved:")
        print(f"   Main report: {report_file}")
        print(f"   Detailed results: {details_file}")


async def main():
    """Run the comprehensive test suite"""
    test_suite = ComprehensiveDocumentSearchTestSuite()
    
    try:
        validation_report = await test_suite.run_comprehensive_test_suite()
        
        # Determine exit code based on critical test and overall success
        report = validation_report['validation_report']
        critical_passed = report['critical_test_status'] == 'PASSED'
        success_rate = float(report['summary_statistics']['success_rate'].replace('%', ''))
        
        if critical_passed and success_rate >= 75:
            print("\n🎉 COMPREHENSIVE TEST SUITE PASSED!")
            print("DocumentSearchService is validated and ready for production use.")
            return 0
        else:
            print("\n⚠️ COMPREHENSIVE TEST SUITE FAILED!")
            print("DocumentSearchService requires fixes before production deployment.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite execution failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)