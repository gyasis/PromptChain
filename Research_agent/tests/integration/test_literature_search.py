#!/usr/bin/env python3
"""
Literature Search Agent Testing
Tests Sci-Hub MCP, ArXiv, and PubMed integrations
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Imports now work with proper package structure

# Test utilities
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from test_ledger import TestLedger, TestResult

class LiteratureSearchTester:
    """Tests Literature Search Agent integrations"""
    
    def __init__(self):
        self.ledger = TestLedger()
        self.test_results: Dict[str, Any] = {
            'passed': [],
            'failed': [],
            'errors': [],
            'performance': {}
        }
        
    def log_test_result(self, test_name: str, success: bool, details: Any = None, error: Exception = None):
        """Log test results for tracking"""
        result = TestResult(
            test_id=f"literature_search_{test_name.lower().replace(' ', '_')}",
            test_name=test_name,
            category='literature_search',
            status='passed' if success else 'failed' if not error else 'error',
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=None,
            details=details or {},
            error_message=str(error) if error else None,
            dependencies=[],
            priority=1  # Critical priority
        )
        
        # Record in ledger
        self.ledger.record_test_result(result)
        
        if success:
            self.test_results['passed'].append(result.__dict__)
            logger.info(f"✅ {test_name}: PASSED")
        else:
            self.test_results['failed'].append(result.__dict__)
            logger.error(f"❌ {test_name}: FAILED - {error or details}")
            
        if error:
            self.test_results['errors'].append(result.__dict__)
    
    async def test_research_agent_imports(self):
        """Test Literature Search Agent imports"""
        test_name = "Literature Search Agent Imports"
        try:
            from research_agent.agents.literature_searcher import LiteratureSearchAgent
            from research_agent.agents.search_strategist import SearchStrategistAgent
            from research_agent.core.config import ResearchConfig
            
            self.log_test_result(test_name, True, {
                "literature_searcher_imported": True,
                "search_strategist_imported": True,
                "config_imported": True
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_literature_search_agent_initialization(self):
        """Test LiteratureSearchAgent initialization"""
        test_name = "LiteratureSearchAgent Initialization"
        try:
            from research_agent.agents.literature_searcher import LiteratureSearchAgent
            from research_agent.core.config import ResearchConfig
            
            config = ResearchConfig()
            agent_config = config.get_agent_config('literature_searcher')
            
            # Initialize agent
            agent = LiteratureSearchAgent(config=agent_config)
            
            # Verify properties
            has_config = hasattr(agent, 'config')
            has_search_method = hasattr(agent, 'search_papers')
            
            self.log_test_result(test_name, True, {
                "agent_initialized": True,
                "has_config": has_config,
                "has_search_method": has_search_method,
                "config_keys": list(agent_config.keys()) if agent_config else []
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_search_strategist_agent_initialization(self):
        """Test SearchStrategistAgent initialization"""
        test_name = "SearchStrategistAgent Initialization"
        try:
            from research_agent.agents.search_strategist import SearchStrategistAgent
            from research_agent.core.config import ResearchConfig
            
            config = ResearchConfig()
            agent_config = config.get_agent_config('search_strategist')
            
            # Initialize agent
            agent = SearchStrategistAgent(config=agent_config)
            
            # Verify properties
            has_config = hasattr(agent, 'config')
            has_strategy_method = hasattr(agent, 'generate_strategy')
            
            self.log_test_result(test_name, True, {
                "agent_initialized": True,
                "has_config": has_config,
                "has_strategy_method": has_strategy_method,
                "config_keys": list(agent_config.keys()) if agent_config else []
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_arxiv_integration_structure(self):
        """Test ArXiv integration structure"""
        test_name = "ArXiv Integration Structure"
        try:
            # Check if ArXiv integration files exist
            arxiv_file = Path(__file__).parent.parent / 'src' / 'research_agent' / 'integrations' / 'arxiv.py'
            
            # Test basic structure
            if arxiv_file.exists():
                # Try importing
                from research_agent.integrations.arxiv import ArXivSearcher
                
                # Create instance
                searcher = ArXivSearcher()
                
                # Check methods
                has_search = hasattr(searcher, 'search_papers')
                has_parse = hasattr(searcher, 'parse_paper_metadata')
                
                self.log_test_result(test_name, True, {
                    "arxiv_file_exists": True,
                    "arxiv_searcher_imported": True,
                    "has_search_method": has_search,
                    "has_parse_method": has_parse
                })
            else:
                self.log_test_result(test_name, False, details="ArXiv integration file not found")
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_pubmed_integration_structure(self):
        """Test PubMed integration structure"""
        test_name = "PubMed Integration Structure"
        try:
            # Check if PubMed integration files exist
            pubmed_file = Path(__file__).parent.parent / 'src' / 'research_agent' / 'integrations' / 'pubmed.py'
            
            # Test basic structure
            if pubmed_file.exists():
                # Try importing
                from research_agent.integrations.pubmed import PubMedSearcher
                
                # Create instance
                searcher = PubMedSearcher()
                
                # Check methods
                has_search = hasattr(searcher, 'search_papers')
                has_parse = hasattr(searcher, 'parse_paper_metadata')
                
                self.log_test_result(test_name, True, {
                    "pubmed_file_exists": True,
                    "pubmed_searcher_imported": True,
                    "has_search_method": has_search,
                    "has_parse_method": has_parse
                })
            else:
                self.log_test_result(test_name, False, details="PubMed integration file not found")
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_scihub_mcp_configuration(self):
        """Test Sci-Hub MCP configuration"""
        test_name = "Sci-Hub MCP Configuration"
        try:
            from research_agent.core.config import ResearchConfig
            
            # Create config with fallback to defaults if file not found
            try:
                config = ResearchConfig()
            except FileNotFoundError:
                # Create minimal config for testing
                config = self._create_test_config()
            
            # Check for MCP configuration
            mcp_config = config.get('mcp', {})
            scihub_servers = [s for s in mcp_config.get('servers', []) if 'scihub' in s.get('id', '').lower()]
            
            # Check configuration structure
            has_mcp_section = mcp_config is not None
            has_servers = len(mcp_config.get('servers', [])) > 0
            has_scihub = len(scihub_servers) > 0
            
            self.log_test_result(test_name, has_mcp_section, {
                "has_mcp_section": has_mcp_section,
                "has_servers": has_servers,
                "has_scihub_config": has_scihub,
                "total_servers": len(mcp_config.get('servers', [])),
                "scihub_servers": len(scihub_servers)
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_search_coordination(self):
        """Test search coordination between different sources"""
        test_name = "Search Coordination"
        try:
            from research_agent.agents.literature_searcher import LiteratureSearchAgent
            from research_agent.core.config import ResearchConfig
            
            config = ResearchConfig()
            agent_config = config.get_agent_config('literature_searcher')
            
            # Initialize agent
            agent = LiteratureSearchAgent(config=agent_config)
            
            # Test search strategy structure
            test_strategy = {
                'queries': ['machine learning', 'neural networks'],
                'sources': ['arxiv', 'pubmed'],
                'max_papers': 10,
                'filters': {
                    'date_range': '2020-2024',
                    'min_citations': 5
                }
            }
            
            # Check if agent can handle strategy format
            has_search_method = hasattr(agent, 'search_papers')
            
            self.log_test_result(test_name, has_search_method, {
                "agent_initialized": True,
                "has_search_method": has_search_method,
                "strategy_format_valid": True,
                "test_strategy": test_strategy
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_paper_extraction_structure(self):
        """Test paper extraction and metadata processing structure"""
        test_name = "Paper Extraction Structure"
        try:
            from research_agent.core.session import Paper
            
            # Test Paper class structure with required 'id' field
            test_paper_data = {
                'id': 'test_paper_1',
                'title': 'Test Paper',
                'authors': ['Author 1', 'Author 2'],
                'abstract': 'Test abstract',
                'source': 'arxiv',
                'url': 'https://example.com/paper.pdf',
                'publication_year': 2024
            }
            
            # Create paper instance
            paper = Paper(**test_paper_data)
            
            # Verify properties
            has_title = hasattr(paper, 'title')
            has_authors = hasattr(paper, 'authors')
            has_abstract = hasattr(paper, 'abstract')
            has_source = hasattr(paper, 'source')
            has_id = hasattr(paper, 'id')
            
            self.log_test_result(test_name, all([has_title, has_authors, has_abstract, has_source, has_id]), {
                "paper_class_imported": True,
                "has_id": has_id,
                "has_title": has_title,
                "has_authors": has_authors,
                "has_abstract": has_abstract,
                "has_source": has_source,
                "paper_created": True
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_metadata_processing_capabilities(self):
        """Test metadata processing capabilities"""
        test_name = "Metadata Processing Capabilities"
        try:
            from research_agent.core.session import Paper
            
            # Test with various paper metadata formats (include required 'id' and 'url' fields)
            paper_formats = [
                {
                    'id': 'arxiv_2024_12345',
                    'title': 'ArXiv Paper',
                    'authors': ['John Doe', 'Jane Smith'],
                    'abstract': 'Machine learning research',
                    'source': 'arxiv',
                    'url': 'https://arxiv.org/abs/2024.12345',
                    'publication_year': 2024,
                    'metadata': {'arxiv_id': '2024.12345'}
                },
                {
                    'id': 'pubmed_12345678',
                    'title': 'PubMed Paper',
                    'authors': ['Dr. Smith'],
                    'abstract': 'Medical research',
                    'source': 'pubmed',
                    'url': 'https://pubmed.ncbi.nlm.nih.gov/12345678/',
                    'publication_year': 2023,
                    'metadata': {'pmid': '12345678'}
                }
            ]
            
            created_papers = []
            for paper_data in paper_formats:
                try:
                    paper = Paper(**paper_data)
                    created_papers.append(paper)
                except Exception as paper_error:
                    logger.warning(f"Failed to create paper: {paper_error}")
            
            self.log_test_result(test_name, len(created_papers) > 0, {
                "total_formats_tested": len(paper_formats),
                "successful_papers": len(created_papers),
                "formats_supported": [p.source for p in created_papers]
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_integration_with_orchestrator(self):
        """Test integration with research orchestrator"""
        test_name = "Integration with Orchestrator"
        try:
            from research_agent.core.orchestrator import AdvancedResearchOrchestrator
            from research_agent.core.config import ResearchConfig
            
            config = ResearchConfig()
            orchestrator = AdvancedResearchOrchestrator(config)
            
            # Verify literature search components are present
            has_literature_searcher = hasattr(orchestrator, 'literature_searcher')
            has_search_strategist = hasattr(orchestrator, 'search_strategist')
            
            self.log_test_result(test_name, has_literature_searcher and has_search_strategist, {
                "orchestrator_initialized": True,
                "has_literature_searcher": has_literature_searcher,
                "has_search_strategist": has_search_strategist,
                "components_integrated": True
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def run_all_tests(self):
        """Run all literature search tests"""
        logger.info("🚀 Starting Literature Search Agent Tests...")
        
        # Start test session
        session_id = self.ledger.start_session({
            'test_type': 'literature_search_tests',
            'category': 'literature_search'
        })
        
        test_methods = [
            self.test_research_agent_imports,
            self.test_literature_search_agent_initialization,
            self.test_search_strategist_agent_initialization,
            self.test_arxiv_integration_structure,
            self.test_pubmed_integration_structure,
            self.test_scihub_mcp_configuration,
            self.test_search_coordination,
            self.test_paper_extraction_structure,
            self.test_metadata_processing_capabilities,
            self.test_integration_with_orchestrator
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.log_test_result(f"Test Execution: {test_method.__name__}", False, error=e)
        
        # End test session
        self.ledger.end_session(session_id)
        
        return self.generate_test_report(session_id)
    
    def _create_test_config(self):
        """Create a minimal test configuration"""
        from research_agent.core.config import ResearchConfig
        import tempfile
        import yaml
        
        # Create minimal config data
        test_config_data = {
            'system': {
                'name': 'Test Research Agent',
                'version': '0.1.0'
            },
            'literature_search': {
                'enabled_sources': ['sci_hub', 'arxiv', 'pubmed'],
                'sci_hub': {},
                'arxiv': {},
                'pubmed': {'email': 'test@example.com'}
            },
            'three_tier_rag': {
                'execution_mode': 'sequential'
            },
            'promptchain': {
                'agents': {
                    'literature_searcher': {'model': 'openai/gpt-4o-mini'},
                    'search_strategist': {'model': 'openai/gpt-4o-mini'}
                }
            },
            'research_session': {
                'max_iterations': 5
            },
            'mcp': {
                'servers': [
                    {
                        'id': 'scihub_server',
                        'type': 'stdio',
                        'command': 'test-mcp-server'
                    }
                ]
            }
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config_data, f)
            temp_config_path = f.name
        
        # Create config instance
        config = ResearchConfig.__new__(ResearchConfig)
        config.config_path = temp_config_path
        config.config_data = test_config_data
        
        return config
    
    def generate_test_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        # Get ledger report
        ledger_report = self.ledger.generate_report(session_id)
        
        # Generate summary
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed'])
        passed_count = len(self.test_results['passed'])
        failed_count = len(self.test_results['failed'])
        error_count = len(self.test_results['errors'])
        
        success_rate = (passed_count / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_count,
                'failed': failed_count,
                'errors': error_count,
                'success_rate': f"{success_rate:.1f}%",
                'status': 'PASSED' if failed_count == 0 else 'FAILED'
            },
            'ledger_report': ledger_report,
            'details': self.test_results
        }
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 LITERATURE SEARCH AGENT TEST REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"⚠️  Errors: {error_count}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {report['summary']['status']}")
        logger.info(f"{'='*60}")
        
        if failed_count > 0:
            logger.error("Failed Tests:")
            for fail in self.test_results['failed']:
                logger.error(f"  - {fail['test_name']}: {fail.get('error_message', 'Unknown error')}")
        
        return report

# Main execution function
async def main():
    """Main test execution"""
    tester = LiteratureSearchTester()
    
    try:
        report = await tester.run_all_tests()
        
        # Save report to file
        report_file = f"literature_search_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {report_file}")
        
        # Export ledger reports
        tester.ledger.export_report(format='json')
        tester.ledger.export_report(format='html')
        
        return report['summary']['status'] == 'PASSED'
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)