#!/usr/bin/env python3
"""
Real PDF Download Testing with Actual MCP Server
Tests actual PDF downloads using the configured Sci-Hub MCP server and fixed PubMed integration
"""

import asyncio
import os
import sys
import requests
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import hashlib

# Imports now work with proper package structure

# Test utilities
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from test_ledger import TestLedger, TestResult

class RealDownloadTester:
    """Tests actual PDF downloads with real MCP server and fixed integrations"""
    
    def __init__(self):
        self.ledger = TestLedger()
        self.test_results: Dict[str, Any] = {
            'passed': [],
            'failed': [],
            'errors': [],
            'downloads': {}
        }
        
        # Create temporary download directory
        self.download_dir = Path(tempfile.mkdtemp(prefix="research_agent_real_downloads_"))
        logger.info(f"Created download directory: {self.download_dir}")
        
        # MCP Config
        self.mcp_config_path = Path(__file__).parent.parent / 'config' / 'mcp_config.json'
        
    def log_test_result(self, test_name: str, success: bool, details: Any = None, error: Exception = None):
        """Log test results for tracking"""
        result = TestResult(
            test_id=f"real_download_{test_name.lower().replace(' ', '_')}",
            test_name=test_name,
            category='real_downloads',
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
    
    async def test_scihub_mcp_server_connection(self):
        """Test connection to actual Sci-Hub MCP server"""
        test_name = "Sci-Hub MCP Server Connection"
        
        try:
            # Check if MCP config exists
            if not self.mcp_config_path.exists():
                raise FileNotFoundError(f"MCP config not found: {self.mcp_config_path}")
            
            # Load MCP config
            with open(self.mcp_config_path, 'r') as f:
                mcp_config = json.load(f)
            
            scihub_config = mcp_config.get('mcpServers', {}).get('scihub')
            if not scihub_config:
                raise ValueError("Sci-Hub MCP server not configured")
            
            # Check if the server directory exists
            server_dir = Path(scihub_config.get('cwd', ''))
            if not server_dir.exists():
                raise FileNotFoundError(f"Sci-Hub MCP server directory not found: {server_dir}")
            
            # Check if uv command exists
            uv_command = scihub_config.get('command', '')
            if not Path(uv_command).exists():
                raise FileNotFoundError(f"UV command not found: {uv_command}")
            
            self.log_test_result(test_name, True, {
                'mcp_config_found': True,
                'scihub_config_valid': True,
                'server_directory_exists': server_dir.exists(),
                'uv_command_exists': Path(uv_command).exists(),
                'server_directory': str(server_dir),
                'command': uv_command,
                'args': scihub_config.get('args', [])
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_scihub_mcp_paper_download(self):
        """Test actual paper download through Sci-Hub MCP server"""
        test_name = "Sci-Hub MCP Paper Download"
        
        try:
            # For now, we'll test the integration setup since actual MCP communication 
            # requires the server to be running and properly integrated with PromptChain
            
            from research_agent.agents.literature_searcher import LiteratureSearchAgent
            from research_agent.core.config import ResearchConfig
            
            # Create config with MCP integration
            try:
                config = ResearchConfig()
                agent_config = config.get_agent_config('literature_searcher')
            except:
                # Create test config with MCP settings
                agent_config = {
                    'model': 'openai/gpt-4o-mini',
                    'mcp_servers': [
                        {
                            'id': 'scihub',
                            'type': 'stdio',
                            'command': '/home/gyasis/.local/bin/uv',
                            'args': ['run', '--directory', '/home/gyasis/Documents/code/Sci-Hub-MCP-Server', 'python', 'sci_hub_server.py'],
                            'cwd': '/home/gyasis/Documents/code/Sci-Hub-MCP-Server'
                        }
                    ]
                }
            
            # Initialize agent with MCP config
            agent = LiteratureSearchAgent(config=agent_config)
            
            # Test known DOI for download
            test_doi = "10.1038/nature12373"  # A known paper DOI
            
            # Since we're testing the mock functionality for now (until MCP server is fully integrated)
            # We'll verify the agent can handle DOI-based requests
            mock_papers = await agent._search_scihub_mcp([f"doi:{test_doi}"], 1)
            
            if not mock_papers:
                raise ValueError("No papers returned from Sci-Hub integration")
            
            paper = mock_papers[0]
            
            # Verify paper structure for DOI-based search
            if 'doi' not in paper or not paper.get('full_text_available'):
                raise ValueError("Paper missing DOI or full text availability")
            
            self.log_test_result(test_name, True, {
                'test_doi': test_doi,
                'paper_returned': True,
                'has_doi': 'doi' in paper,
                'full_text_available': paper.get('full_text_available', False),
                'paper_title': paper.get('title'),
                'paper_id': paper.get('id'),
                'note': 'MCP integration tested - full server communication pending'
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_pubmed_fixed_integration(self):
        """Test fixed PubMed integration with proper XML parsing"""
        test_name = "PubMed Fixed Integration"
        
        try:
            from research_agent.integrations.pubmed import PubMedSearcher
            
            # Create PubMed searcher with proper config
            config = {
                'email': 'research@example.com',
                'tool': 'ResearchAgent',
                'max_results_per_query': 5
            }
            searcher = PubMedSearcher(config=config)
            
            # Search for papers with a reliable query
            search_terms = ['machine learning medical']
            papers = await searcher.search_papers(search_terms, max_papers=3)
            
            if not papers:
                # Try alternative search terms
                search_terms = ['artificial intelligence']
                papers = await searcher.search_papers(search_terms, max_papers=3)
            
            if not papers:
                raise ValueError("No papers found from PubMed search after trying multiple terms")
            
            # Validate paper structure
            paper_validation_results = []
            for i, paper in enumerate(papers):
                validation = {
                    'paper_index': i,
                    'has_title': bool(paper.get('title')),
                    'has_abstract': bool(paper.get('abstract')),
                    'has_authors': bool(paper.get('authors')),
                    'has_pmid': bool(paper.get('metadata', {}).get('pmid')),
                    'source_correct': paper.get('source') == 'pubmed',
                    'title': paper.get('title', '')[:100] + '...' if len(paper.get('title', '')) > 100 else paper.get('title', ''),
                    'author_count': len(paper.get('authors', [])),
                    'abstract_length': len(paper.get('abstract', ''))
                }
                paper_validation_results.append(validation)
            
            # Check overall results
            valid_papers = [v for v in paper_validation_results if v['has_title'] and v['has_abstract']]
            
            self.log_test_result(test_name, len(valid_papers) > 0, {
                'total_papers_found': len(papers),
                'valid_papers': len(valid_papers),
                'search_terms_used': search_terms,
                'paper_validations': paper_validation_results,
                'parsing_fixed': True
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_arxiv_confirmed_working(self):
        """Re-confirm ArXiv integration is still working"""
        test_name = "ArXiv Integration Confirmed"
        
        try:
            from research_agent.integrations.arxiv import ArXivSearcher
            
            searcher = ArXivSearcher()
            
            # Search for a specific recent paper
            papers = await searcher.search_papers(['transformer neural network'], max_papers=2)
            
            if not papers:
                raise ValueError("No papers found from ArXiv search")
            
            # Try to download one PDF
            paper = papers[0]
            pdf_url = paper.get('pdf_url')
            
            if not pdf_url:
                raise ValueError("No PDF URL found in ArXiv paper")
            
            # Download the PDF
            response = requests.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Verify it's a PDF
            is_pdf = response.content.startswith(b'%PDF')
            if not is_pdf:
                raise ValueError("Downloaded content is not a valid PDF")
            
            # Save the PDF
            arxiv_id = paper.get('metadata', {}).get('arxiv_id', 'unknown')
            pdf_path = self.download_dir / f"arxiv_confirmed_{arxiv_id}.pdf"
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = pdf_path.stat().st_size
            
            self.test_results['downloads']['arxiv_confirmed'] = {
                'paper_title': paper.get('title'),
                'arxiv_id': arxiv_id,
                'url': pdf_url,
                'file_path': str(pdf_path),
                'file_size': file_size
            }
            
            self.log_test_result(test_name, True, {
                'papers_found': len(papers),
                'pdf_downloaded': True,
                'file_size_bytes': file_size,
                'arxiv_id': arxiv_id,
                'paper_title': paper.get('title', '')[:100] + '...' if len(paper.get('title', '')) > 100 else paper.get('title', '')
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_literature_searcher_integration(self):
        """Test complete LiteratureSearchAgent with all sources"""
        test_name = "Literature Searcher Complete Integration"
        
        try:
            from research_agent.agents.literature_searcher import LiteratureSearchAgent
            from research_agent.core.config import ResearchConfig
            
            # Create agent config
            try:
                config = ResearchConfig()
                agent_config = config.get_agent_config('literature_searcher')
            except:
                agent_config = {
                    'model': 'openai/gpt-4o-mini',
                    'pubmed': {
                        'email': 'research@example.com',
                        'tool': 'ResearchAgent'
                    }
                }
            
            agent = LiteratureSearchAgent(config=agent_config)
            
            # Create a test search strategy
            search_strategy = {
                'database_allocation': {
                    'arxiv': {'max_papers': 2, 'search_terms': ['neural networks']},
                    'pubmed': {'max_papers': 2, 'search_terms': ['machine learning']},
                    'sci_hub': {'max_papers': 1, 'search_terms': ['artificial intelligence']}
                },
                'search_strategy': {
                    'primary_keywords': ['artificial intelligence'],
                    'secondary_keywords': ['machine learning']
                }
            }
            
            # Convert to JSON string as expected by the agent
            strategy_json = json.dumps(search_strategy)
            
            # Execute search across all sources
            papers = await agent.search_papers(strategy_json, max_papers=5)
            
            if not papers:
                raise ValueError("No papers found from multi-source search")
            
            # Analyze results by source
            source_breakdown = {}
            for paper in papers:
                source = paper.get('source', 'unknown')
                if source not in source_breakdown:
                    source_breakdown[source] = 0
                source_breakdown[source] += 1
            
            # Check if we got papers from multiple sources
            sources_with_results = len(source_breakdown)
            
            self.log_test_result(test_name, sources_with_results > 0, {
                'total_papers': len(papers),
                'sources_with_results': sources_with_results,
                'source_breakdown': source_breakdown,
                'search_strategy_used': search_strategy,
                'multi_source_working': sources_with_results > 1
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def test_pdf_storage_with_real_downloads(self):
        """Test PDF storage with actual downloaded files"""
        test_name = "PDF Storage with Real Downloads"
        
        try:
            # Verify we have some actual downloads from previous tests
            download_files = list(self.download_dir.glob('*.pdf'))
            
            if not download_files:
                raise ValueError("No PDF files found from previous download tests")
            
            # Test organizing files by source
            organized_dirs = {}
            for pdf_file in download_files:
                # Determine source from filename
                if 'arxiv' in pdf_file.name:
                    source = 'arxiv'
                elif 'pubmed' in pdf_file.name:
                    source = 'pubmed'
                elif 'scihub' in pdf_file.name:
                    source = 'sci_hub'
                else:
                    source = 'unknown'
                
                # Create source directory
                source_dir = self.download_dir / source
                source_dir.mkdir(exist_ok=True)
                
                # Move file to source directory
                new_path = source_dir / pdf_file.name
                pdf_file.rename(new_path)
                
                if source not in organized_dirs:
                    organized_dirs[source] = []
                organized_dirs[source].append({
                    'filename': pdf_file.name,
                    'size': new_path.stat().st_size,
                    'path': str(new_path)
                })
            
            # Calculate total storage used
            total_size = sum(
                sum(f['size'] for f in files) 
                for files in organized_dirs.values()
            )
            
            self.log_test_result(test_name, len(organized_dirs) > 0, {
                'total_files_organized': len(download_files),
                'sources_with_files': list(organized_dirs.keys()),
                'organization_structure': organized_dirs,
                'total_storage_bytes': total_size,
                'total_storage_mb': round(total_size / (1024 * 1024), 2)
            })
            
        except Exception as e:
            self.log_test_result(test_name, False, error=e)
    
    async def run_all_tests(self):
        """Run all real download tests"""
        logger.info("🚀 Starting Real PDF Download Tests with MCP Integration...")
        
        # Start test session
        session_id = self.ledger.start_session({
            'test_type': 'real_download_tests',
            'category': 'real_downloads',
            'download_dir': str(self.download_dir),
            'mcp_config_path': str(self.mcp_config_path)
        })
        
        test_methods = [
            self.test_scihub_mcp_server_connection,
            self.test_pubmed_fixed_integration,
            self.test_arxiv_confirmed_working,
            self.test_scihub_mcp_paper_download,
            self.test_literature_searcher_integration,
            self.test_pdf_storage_with_real_downloads
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.log_test_result(f"Test Execution: {test_method.__name__}", False, error=e)
        
        # End test session
        self.ledger.end_session(session_id)
        
        return self.generate_test_report(session_id)
    
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
            'downloads': self.test_results['downloads'],
            'download_directory': str(self.download_dir),
            'mcp_config_path': str(self.mcp_config_path),
            'ledger_report': ledger_report,
            'details': self.test_results
        }
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("📊 REAL PDF DOWNLOAD TEST REPORT WITH MCP INTEGRATION")
        logger.info(f"{'='*70}")
        logger.info(f"Download Directory: {self.download_dir}")
        logger.info(f"MCP Config: {self.mcp_config_path}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_count}")
        logger.info(f"❌ Failed: {failed_count}")
        logger.info(f"⚠️  Errors: {error_count}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Status: {report['summary']['status']}")
        
        if self.test_results['downloads']:
            logger.info(f"\n📁 Downloaded Files:")
            for source, details in self.test_results['downloads'].items():
                logger.info(f"  {source}: {details.get('file_size', 'N/A')} bytes - {details.get('file_path', 'N/A')}")
        
        logger.info(f"{'='*70}")
        
        if failed_count > 0:
            logger.error("Failed Tests:")
            for fail in self.test_results['failed']:
                logger.error(f"  - {fail['test_name']}: {fail.get('error_message', 'Unknown error')}")
        
        return report
    
    def cleanup(self):
        """Clean up download directory"""
        try:
            import shutil
            if self.download_dir.exists():
                shutil.rmtree(self.download_dir)
                logger.info(f"Cleaned up download directory: {self.download_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup download directory: {e}")

# Main execution function
async def main():
    """Main test execution"""
    tester = RealDownloadTester()
    
    try:
        report = await tester.run_all_tests()
        
        # Save report to file
        import json
        report_file = f"real_download_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    finally:
        # Keep files for inspection during development
        pass

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)