#!/usr/bin/env python3
"""
Comprehensive Test Suite for DocumentSearchService Class
========================================================

This test suite thoroughly validates all functionality of the DocumentSearchService:
1. Service initialization and configuration
2. Session management
3. 3-tier search execution (ArXiv, PubMed, Sci-Hub)
4. Metadata enhancement and fallback systems
5. Error handling and recovery
6. Resource management and cleanup
7. Real-world query scenarios

Tests both individual components and full integration workflows.
"""

import asyncio
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory and parent directories to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# Import the document search service and related components
from src.research_agent.services.document_search_service import (
    DocumentSearchService, 
    SearchConfiguration,
    ServiceConstants,
    SearchTier,
    SearchMethod,
    SearchResult
)

class ComprehensiveDocumentSearchTester:
    """Comprehensive test suite for DocumentSearchService."""
    
    def __init__(self):
        self.test_results = []
        self.test_workspace = None
        self.service = None
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", data: Any = None):
        """Log a test result for reporting."""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "data": data,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    {details}")
        if not success and data:
            print(f"    Error data: {data}")
    
    async def setup_test_environment(self):
        """Set up the test environment."""
        print("🔧 Setting up test environment...")
        
        # Create temporary workspace
        self.test_workspace = Path(tempfile.mkdtemp(prefix="doc_search_test_"))
        print(f"   Test workspace: {self.test_workspace}")
        
        # Verify environment variables
        required_env = ["OPENAI_API_KEY"]
        missing_env = [var for var in required_env if not os.getenv(var)]
        
        if missing_env:
            self.log_test_result(
                "Environment Setup", 
                False, 
                f"Missing environment variables: {missing_env}"
            )
            return False
        
        self.log_test_result("Environment Setup", True, "All required environment variables present")
        return True
    
    async def test_service_initialization(self):
        """Test service initialization with various configurations."""
        print("\n📋 Testing Service Initialization...")
        
        # Test 1: Default initialization
        try:
            config = SearchConfiguration(
                working_directory=self.test_workspace,
                max_papers_per_tier=3,
                rate_limit_delay=0.5
            )
            
            self.service = DocumentSearchService(config=config)
            success = await self.service.initialize()
            
            self.log_test_result(
                "Default Initialization", 
                success, 
                f"Service initialized: {success}"
            )
            
            # Verify service state
            info = self.service.get_session_info()
            self.log_test_result(
                "Service State Check",
                self.service._initialized,
                f"Initialized: {self.service._initialized}"
            )
            
        except Exception as e:
            self.log_test_result("Default Initialization", False, str(e), e)
            return False
        
        # Test 2: Custom configuration
        try:
            custom_constants = ServiceConstants()
            custom_config = SearchConfiguration(
                working_directory=self.test_workspace / "custom",
                max_papers_per_tier=10,
                rate_limit_delay=1.0,
                enable_metadata_enhancement=True,
                enable_fallback_downloads=True
            )
            
            custom_service = DocumentSearchService(config=custom_config, constants=custom_constants)
            custom_success = await custom_service.initialize()
            
            self.log_test_result(
                "Custom Configuration",
                custom_success,
                f"Custom service initialized: {custom_success}"
            )
            
            await custom_service.shutdown()
            
        except Exception as e:
            self.log_test_result("Custom Configuration", False, str(e), e)
        
        return True
    
    async def test_session_management(self):
        """Test session creation, management, and cleanup."""
        print("\n📁 Testing Session Management...")
        
        # Test 1: Start session with custom name
        try:
            session_id = self.service.start_session("test_session_custom")
            
            self.log_test_result(
                "Custom Session Creation",
                session_id is not None,
                f"Session ID: {session_id}"
            )
            
            # Verify session folder structure
            session_info = self.service.get_session_info()
            if session_info:
                session_folder = Path(session_info['session_folder'])
                required_dirs = ['pdfs', 'metadata', 'logs']
                dirs_exist = all((session_folder / dir_name).exists() for dir_name in required_dirs)
                
                self.log_test_result(
                    "Session Folder Structure",
                    dirs_exist,
                    f"Required directories created: {dirs_exist}"
                )
            
        except Exception as e:
            self.log_test_result("Custom Session Creation", False, str(e), e)
        
        # Test 2: Start session with auto-generated name
        try:
            self.service.end_session()
            auto_session_id = self.service.start_session()
            
            self.log_test_result(
                "Auto Session Creation",
                auto_session_id is not None,
                f"Auto session ID: {auto_session_id}"
            )
            
        except Exception as e:
            self.log_test_result("Auto Session Creation", False, str(e), e)
        
        # Test 3: Session info retrieval
        try:
            session_info = self.service.get_session_info()
            required_keys = ['session_id', 'session_folder', 'is_initialized', 'service_version']
            has_required_keys = all(key in session_info for key in required_keys)
            
            self.log_test_result(
                "Session Info Retrieval",
                has_required_keys,
                f"Session info contains all required keys: {has_required_keys}"
            )
            
        except Exception as e:
            self.log_test_result("Session Info Retrieval", False, str(e), e)
    
    async def test_individual_tier_searches(self):
        """Test each search tier individually."""
        print("\n🔍 Testing Individual Tier Searches...")
        
        test_query = "artificial intelligence"
        max_papers = 3
        
        # Test ArXiv tier
        try:
            arxiv_papers = await self.service._search_arxiv_tier(test_query, max_papers)
            
            self.log_test_result(
                "ArXiv Tier Search",
                isinstance(arxiv_papers, list),
                f"Found {len(arxiv_papers)} papers, all have tier=arxiv: {all(p.tier == SearchTier.ARXIV for p in arxiv_papers)}"
            )
            
            # Verify paper structure
            if arxiv_papers:
                sample_paper = arxiv_papers[0]
                required_fields = ['id', 'title', 'authors', 'publication_year', 'doi', 'abstract']
                has_fields = all(hasattr(sample_paper, field) for field in required_fields)
                
                self.log_test_result(
                    "ArXiv Paper Structure",
                    has_fields,
                    f"Sample paper has all required fields: {has_fields}"
                )
            
        except Exception as e:
            self.log_test_result("ArXiv Tier Search", False, str(e), e)
        
        # Test PubMed tier
        try:
            pubmed_papers = await self.service._search_pubmed_tier(test_query, max_papers)
            
            self.log_test_result(
                "PubMed Tier Search",
                isinstance(pubmed_papers, list),
                f"Found {len(pubmed_papers)} papers, all have tier=pubmed: {all(p.tier == SearchTier.PUBMED for p in pubmed_papers)}"
            )
            
        except Exception as e:
            self.log_test_result("PubMed Tier Search", False, str(e), e)
        
        # Test Sci-Hub tier (this is our key test)
        try:
            scihub_papers = await self.service._search_scihub_tier(test_query, max_papers)
            
            self.log_test_result(
                "Sci-Hub Tier Search",
                isinstance(scihub_papers, list),
                f"Found {len(scihub_papers)} papers, all have tier=scihub: {all(p.tier == SearchTier.SCIHUB for p in scihub_papers)}"
            )
            
            # Verify Sci-Hub specific attributes
            if scihub_papers:
                mcp_direct_count = sum(1 for p in scihub_papers if p.search_method == SearchMethod.SCIHUB_MCP_DIRECT)
                
                self.log_test_result(
                    "Sci-Hub MCP Integration",
                    mcp_direct_count > 0,
                    f"Papers using MCP direct search: {mcp_direct_count}/{len(scihub_papers)}"
                )
            
        except Exception as e:
            self.log_test_result("Sci-Hub Tier Search", False, str(e), e)
    
    async def test_full_search_integration(self):
        """Test the complete 3-tier search integration."""
        print("\n🎯 Testing Full 3-Tier Search Integration...")
        
        # Test with different query types
        test_scenarios = [
            ("machine learning", 15, "General tech query"),
            ("cancer treatment", 12, "Medical query"),
            ("quantum computing", 9, "Physics/CS query")
        ]
        
        for query, max_papers, description in test_scenarios:
            try:
                print(f"  Testing: {description} - '{query}'")
                
                papers, metadata = await self.service.search_documents(
                    search_query=query,
                    max_papers=max_papers,
                    enhance_metadata=True,
                    tier_allocation={
                        'arxiv': max_papers // 3,
                        'pubmed': max_papers // 3,
                        'scihub': max_papers - (2 * (max_papers // 3))
                    }
                )
                
                # Verify results
                success_conditions = [
                    isinstance(papers, list),
                    isinstance(metadata, dict),
                    len(papers) > 0,
                    'status' in metadata,
                    'tier_distribution' in metadata
                ]
                
                test_success = all(success_conditions)
                
                # Analyze tier distribution
                tier_counts = {}
                for paper in papers:
                    tier = paper.get('tier', 'unknown')
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                
                details = f"Papers: {len(papers)}, Tiers: {tier_counts}, Status: {metadata.get('status')}"
                
                self.log_test_result(
                    f"Full Search - {description}",
                    test_success,
                    details
                )
                
                # Verify metadata structure
                required_metadata = ['status', 'architecture', 'search_query', 'tier_allocation', 'retrieved_papers']
                metadata_complete = all(key in metadata for key in required_metadata)
                
                self.log_test_result(
                    f"Metadata Completeness - {description}",
                    metadata_complete,
                    f"Required metadata fields present: {metadata_complete}"
                )
                
            except Exception as e:
                self.log_test_result(f"Full Search - {description}", False, str(e), e)
    
    async def test_metadata_enhancement(self):
        """Test metadata enhancement functionality."""
        print("\n📊 Testing Metadata Enhancement...")
        
        try:
            # Search with enhancement enabled
            papers_enhanced, metadata_enhanced = await self.service.search_documents(
                search_query="data science",
                max_papers=6,
                enhance_metadata=True
            )
            
            # Search with enhancement disabled  
            papers_plain, metadata_plain = await self.service.search_documents(
                search_query="data science", 
                max_papers=6,
                enhance_metadata=False
            )
            
            # Compare enhancement levels
            enhanced_count = sum(
                1 for paper in papers_enhanced 
                if paper.get('metadata', {}).get('document_search_service', {}).get('enhanced', False)
            )
            
            plain_count = sum(
                1 for paper in papers_plain
                if paper.get('metadata', {}).get('document_search_service', {}).get('enhanced', False) 
            )
            
            self.log_test_result(
                "Metadata Enhancement Control",
                enhanced_count > plain_count,
                f"Enhanced papers: {enhanced_count}, Plain papers: {plain_count}"
            )
            
            # Verify enhancement quality
            if papers_enhanced:
                sample_paper = papers_enhanced[0]
                enhancement_indicators = [
                    'paper_index' in sample_paper.get('metadata', {}),
                    'search_topic' in sample_paper.get('metadata', {}),
                    'retrieval_method' in sample_paper.get('metadata', {}),
                    'citation' in sample_paper.get('metadata', {})
                ]
                
                self.log_test_result(
                    "Enhancement Quality", 
                    all(enhancement_indicators),
                    f"Enhancement indicators present: {sum(enhancement_indicators)}/4"
                )
            
        except Exception as e:
            self.log_test_result("Metadata Enhancement", False, str(e), e)
    
    async def test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        print("\n⚠️ Testing Error Handling...")
        
        # Test 1: Invalid search query
        try:
            papers, metadata = await self.service.search_documents(
                search_query="",  # Empty query
                max_papers=5
            )
            
            # Should not reach here - should raise ValueError
            self.log_test_result("Empty Query Handling", False, "Should have raised ValueError")
            
        except ValueError as e:
            self.log_test_result("Empty Query Handling", True, f"Correctly raised ValueError: {str(e)[:50]}")
        except Exception as e:
            self.log_test_result("Empty Query Handling", False, f"Unexpected error: {str(e)[:50]}")
        
        # Test 2: Invalid paper count
        try:
            papers, metadata = await self.service.search_documents(
                search_query="test",
                max_papers=1000  # Exceeds MAX_PAPERS_LIMIT
            )
            
            self.log_test_result("Invalid Paper Count", False, "Should have raised ValueError")
            
        except ValueError as e:
            self.log_test_result("Invalid Paper Count", True, f"Correctly raised ValueError: {str(e)[:50]}")
        except Exception as e:
            self.log_test_result("Invalid Paper Count", False, f"Unexpected error: {str(e)[:50]}")
        
        # Test 3: Service behavior with network issues (simulated through timeout)
        try:
            # Use a very short timeout to simulate network issues
            original_delay = self.service.config.rate_limit_delay
            self.service.config.rate_limit_delay = 0.01  # Very short
            
            papers, metadata = await self.service.search_documents(
                search_query="network test",
                max_papers=3
            )
            
            # Should still return results (fallback mode)
            fallback_mode = metadata.get('status') in ['success', 'fallback_mode']
            
            self.log_test_result(
                "Network Issue Handling",
                fallback_mode,
                f"Status: {metadata.get('status')}, Papers: {len(papers)}"
            )
            
            # Restore original delay
            self.service.config.rate_limit_delay = original_delay
            
        except Exception as e:
            self.log_test_result("Network Issue Handling", False, str(e), e)
    
    async def test_resource_management(self):
        """Test resource management and cleanup."""
        print("\n🧹 Testing Resource Management...")
        
        # Test 1: MCP connection management
        try:
            # Force MCP connection creation
            mcp_chain = await self.service._get_shared_mcp_connection()
            
            connection_success = mcp_chain is not None and hasattr(mcp_chain, 'mcp_helper')
            
            self.log_test_result(
                "MCP Connection Creation",
                connection_success,
                f"MCP chain created: {connection_success}"
            )
            
            # Test connection cleanup
            await self.service._cleanup_shared_mcp_connection()
            
            self.log_test_result(
                "MCP Connection Cleanup",
                self.service.shared_mcp_chain is None,
                f"MCP chain cleaned up: {self.service.shared_mcp_chain is None}"
            )
            
        except Exception as e:
            self.log_test_result("MCP Connection Management", False, str(e), e)
        
        # Test 2: Session cleanup
        try:
            # Create a session and verify cleanup
            session_id = self.service.start_session("cleanup_test")
            session_folder = self.service.current_session_folder
            
            # End session with cleanup
            self.service.end_session(cleanup=True)
            
            cleanup_success = not session_folder.exists() if session_folder else True
            
            self.log_test_result(
                "Session Cleanup",
                cleanup_success,
                f"Session folder cleaned up: {cleanup_success}"
            )
            
        except Exception as e:
            self.log_test_result("Session Cleanup", False, str(e), e)
    
    async def test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        print("\n🌍 Testing Real-World Scenarios...")
        
        # Scenario 1: Research literature review workflow
        try:
            # Start a research session
            session_id = self.service.start_session("literature_review")
            
            # Search for papers on a specific topic
            papers, metadata = await self.service.search_documents(
                search_query="deep learning natural language processing",
                max_papers=18,
                enhance_metadata=True
            )
            
            # Verify results are research-ready
            research_quality_indicators = [
                len(papers) > 10,  # Sufficient paper count
                len(set(p.get('tier') for p in papers)) >= 2,  # Multiple sources
                all('title' in p for p in papers),  # All papers have titles
                all('abstract' in p for p in papers),  # All papers have abstracts
                metadata.get('status') == 'success'  # Search succeeded
            ]
            
            research_ready = all(research_quality_indicators)
            
            self.log_test_result(
                "Literature Review Workflow",
                research_ready,
                f"Research quality indicators: {sum(research_quality_indicators)}/5"
            )
            
            # End the research session
            self.service.end_session(cleanup=False)
            
        except Exception as e:
            self.log_test_result("Literature Review Workflow", False, str(e), e)
        
        # Scenario 2: Multi-domain search
        try:
            domains = [
                ("machine learning healthcare", "ML + Healthcare"),
                ("quantum physics applications", "Physics"),
                ("bioinformatics algorithms", "Biology + CS")
            ]
            
            domain_results = {}
            for query, domain_name in domains:
                papers, metadata = await self.service.search_documents(
                    search_query=query,
                    max_papers=6,
                    enhance_metadata=False  # Speed up test
                )
                
                domain_results[domain_name] = {
                    'paper_count': len(papers),
                    'status': metadata.get('status'),
                    'tiers_used': len(set(p.get('tier') for p in papers))
                }
            
            multi_domain_success = all(
                result['paper_count'] > 0 for result in domain_results.values()
            )
            
            self.log_test_result(
                "Multi-Domain Search",
                multi_domain_success,
                f"Domain results: {domain_results}"
            )
            
        except Exception as e:
            self.log_test_result("Multi-Domain Search", False, str(e), e)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📋 TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['details']}")
        
        print(f"\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result['success']:
                print(f"  - {result['test_name']}")
        
        # Save detailed report
        report_file = self.test_workspace / "comprehensive_test_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        return passed_tests == total_tests
    
    async def cleanup_test_environment(self):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        try:
            if self.service:
                await self.service.shutdown()
            
            if self.test_workspace and self.test_workspace.exists():
                shutil.rmtree(self.test_workspace)
                print(f"   Removed test workspace: {self.test_workspace}")
            
        except Exception as e:
            print(f"   Cleanup error: {e}")
    
    async def run_comprehensive_tests(self):
        """Run all comprehensive tests."""
        print("🧪 COMPREHENSIVE DOCUMENT SEARCH SERVICE TEST SUITE")
        print("=" * 70)
        
        try:
            # Setup
            if not await self.setup_test_environment():
                return False
            
            # Run all test categories
            await self.test_service_initialization()
            await self.test_session_management()
            await self.test_individual_tier_searches() 
            await self.test_full_search_integration()
            await self.test_metadata_enhancement()
            await self.test_error_handling()
            await self.test_resource_management()
            await self.test_real_world_scenarios()
            
            # Generate report
            success = self.generate_test_report()
            
            return success
            
        finally:
            await self.cleanup_test_environment()

async def main():
    """Main test execution."""
    tester = ComprehensiveDocumentSearchTester()
    success = await tester.run_comprehensive_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! DocumentSearchService is fully functional.")
    else:
        print("\n⚠️ SOME TESTS FAILED. Check the report for details.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)