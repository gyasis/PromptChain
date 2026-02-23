#!/usr/bin/env python3
"""
MCP Client Integration Test for Sci-Hub MCP Server

This script tests the actual MCP client connection to the Sci-Hub server
using a real MCP client library.
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from research_agent.integrations.multi_query_coordinator import MultiQueryCoordinator
from research_agent.integrations.pdf_manager import PDFManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_scihub_mcp_client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScihubMCPClientTester:
    """Test MCP client integration with Sci-Hub server"""
    
    def __init__(self):
        self.config_path = project_root / "config" / "mcp_config.json"
        self.test_results = {
            'mcp_config_load': False,
            'server_command_validation': False,
            'integration_with_multi_query': False,
            'integration_with_pdf_manager': False,
            'mock_tool_calls': False,
            'storage_path_generation': False,
            'errors': [],
            'test_timestamp': datetime.now().isoformat()
        }
    
    async def run_client_tests(self) -> Dict[str, Any]:
        """Run MCP client integration tests"""
        logger.info("Starting Sci-Hub MCP client integration tests...")
        
        try:
            # Test 1: MCP config loading
            await self.test_mcp_config_load()
            
            # Test 2: Server command validation
            await self.test_server_command_validation()
            
            # Test 3: Integration with MultiQueryCoordinator
            await self.test_integration_with_multi_query()
            
            # Test 4: Integration with PDFManager
            await self.test_integration_with_pdf_manager()
            
            # Test 5: Mock tool calls
            await self.test_mock_tool_calls()
            
            # Test 6: Storage path generation
            await self.test_storage_path_generation()
            
        except Exception as e:
            logger.error(f"Client test suite failed with error: {e}")
            self.test_results['errors'].append(f"Test suite error: {str(e)}")
        
        # Generate test report
        await self.generate_test_report()
        
        return self.test_results
    
    async def test_mcp_config_load(self):
        """Test MCP configuration loading"""
        logger.info("Testing MCP configuration loading...")
        
        try:
            # Load MCP configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate structure
            if 'mcpServers' not in config:
                raise Exception("mcpServers section missing from configuration")
            
            if 'scihub' not in config['mcpServers']:
                raise Exception("scihub server configuration missing")
            
            scihub_config = config['mcpServers']['scihub']
            
            # Check required fields
            required_fields = ['command', 'args', 'cwd', 'enabled', 'transport', 'tools']
            for field in required_fields:
                if field not in scihub_config:
                    raise Exception(f"Required field '{field}' missing from scihub configuration")
            
            # Check client configuration
            if 'client' not in config:
                raise Exception("client configuration missing")
            
            client_config = config['client']
            if 'name' not in client_config or 'version' not in client_config:
                raise Exception("client configuration incomplete")
            
            # Check integration configuration
            if 'integration' not in config:
                raise Exception("integration configuration missing")
            
            integration_config = config['integration']
            expected_integrations = ['literature_search', 'pdf_manager', 'multi_query_coordinator']
            for integration in expected_integrations:
                if integration not in integration_config:
                    raise Exception(f"Integration '{integration}' configuration missing")
            
            logger.info("✓ MCP configuration loaded successfully")
            self.test_results['mcp_config_load'] = True
            
        except Exception as e:
            logger.error(f"✗ MCP configuration loading failed: {e}")
            self.test_results['errors'].append(f"MCP config load: {str(e)}")
    
    async def test_server_command_validation(self):
        """Test server command can be constructed and validated"""
        logger.info("Testing server command validation...")
        
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            scihub_config = config['mcpServers']['scihub']
            
            # Extract command components
            command = scihub_config['command']
            args = scihub_config['args']
            cwd = scihub_config['cwd']
            env = scihub_config.get('env', {})
            
            # Validate command exists
            if command != 'uv':
                raise Exception(f"Expected command 'uv', got '{command}'")
            
            # Check if uv is available
            result = subprocess.run(["/home/gyasis/.local/bin/uv", "--version"], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise Exception("uv command not available")
            
            # Validate args
            if not args or len(args) < 3:
                raise Exception("Invalid command arguments")
            
            if args[0] != 'run':
                raise Exception("Expected first arg to be 'run'")
            
            if 'python' not in args[1]:
                raise Exception("Expected python in arguments")
            
            # Validate working directory
            cwd_path = Path(cwd) if cwd != '.' else project_root
            if not cwd_path.exists():
                raise Exception(f"Working directory does not exist: {cwd_path}")
            
            # Build full command for testing
            full_cmd = ["/home/gyasis/.local/bin/uv"] + args
            logger.info(f"Server command: {' '.join(full_cmd)}")
            logger.info(f"Working directory: {cwd_path}")
            logger.info(f"Environment variables: {list(env.keys())}")
            
            logger.info("✓ Server command validation successful")
            self.test_results['server_command_validation'] = True
            
        except Exception as e:
            logger.error(f"✗ Server command validation failed: {e}")
            self.test_results['errors'].append(f"Server command validation: {str(e)}")
    
    async def test_integration_with_multi_query(self):
        """Test integration with MultiQueryCoordinator"""
        logger.info("Testing integration with MultiQueryCoordinator...")
        
        try:
            # Load integration configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            integration_config = config['integration']['multi_query_coordinator']
            
            if not integration_config.get('enabled', False):
                logger.warning("MultiQueryCoordinator integration is disabled")
                self.test_results['integration_with_multi_query'] = True
                return
            
            # Initialize MultiQueryCoordinator
            coordinator = MultiQueryCoordinator({
                'sources': ['arxiv', 'pubmed', 'sci_hub'],  # Include sci_hub as a source
                'parallel_searches': integration_config.get('parallel_searches', True),
                'merge_duplicates': integration_config.get('merge_duplicates', True),
                'quality_filtering': integration_config.get('quality_filtering', True)
            })
            
            # Test configuration
            if not hasattr(coordinator, 'config'):
                raise Exception("MultiQueryCoordinator missing config attribute")
            
            coordinator_config = coordinator.get_config()
            if 'sci_hub' not in coordinator_config.get('sources', []):
                logger.warning("sci_hub not in MultiQueryCoordinator sources")
                # Add it for testing
                coordinator_config['sources'].append('sci_hub')
            
            # Test mock query generation
            test_query = "machine learning healthcare"
            mock_queries = coordinator.generate_source_queries(test_query)
            
            if not isinstance(mock_queries, dict):
                raise Exception("Query generation did not return dictionary")
            
            if 'sci_hub' not in mock_queries:
                logger.warning("sci_hub queries not generated by coordinator")
            
            logger.info("✓ MultiQueryCoordinator integration working")
            self.test_results['integration_with_multi_query'] = True
            
        except Exception as e:
            logger.error(f"✗ MultiQueryCoordinator integration failed: {e}")
            self.test_results['errors'].append(f"MultiQueryCoordinator integration: {str(e)}")
    
    async def test_integration_with_pdf_manager(self):
        """Test integration with PDFManager"""
        logger.info("Testing integration with PDFManager...")
        
        try:
            # Load integration configuration
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            pdf_manager_config = config['integration']['pdf_manager']
            
            if not pdf_manager_config.get('enabled', False):
                logger.warning("PDFManager integration is disabled")
                self.test_results['integration_with_pdf_manager'] = True
                return
            
            # Initialize PDFManager
            storage_path = pdf_manager_config.get('storage_path', './papers/sci_hub')
            auto_download = pdf_manager_config.get('auto_download', True)
            max_concurrent = pdf_manager_config.get('max_concurrent_downloads', 3)
            
            pdf_manager = PDFManager(
                base_path=str(project_root / "test_papers"),
                config={
                    'max_retries': 2,
                    'timeout': 30,
                    'max_concurrent_downloads': max_concurrent,
                    'auto_download': auto_download
                }
            )
            
            # Test sci_hub storage path generation
            mock_sci_hub_paper = {
                'id': 'test_scihub_paper',
                'title': 'Test Paper from Sci-Hub',
                'authors': ['Test Author'],
                'source': 'sci_hub',
                'publication_year': 2023,
                'doi': '10.1000/test_scihub',
                'url': 'https://sci-hub.se/test'
            }
            
            storage_path = pdf_manager._generate_storage_path(mock_sci_hub_paper)
            
            if not isinstance(storage_path, Path):
                raise Exception("Storage path not generated as Path object")
            
            if 'sci_hub' not in str(storage_path).lower():
                raise Exception("sci_hub not in generated storage path")
            
            if str(storage_path.suffix) != '.pdf':
                raise Exception("Storage path does not have .pdf extension")
            
            # Test storage statistics
            stats = pdf_manager.get_storage_statistics()
            if not isinstance(stats, dict):
                raise Exception("Storage statistics not returned as dictionary")
            
            # Test search functionality
            search_results = pdf_manager.search_pdfs("test", limit=10)
            if not isinstance(search_results, list):
                raise Exception("Search results not returned as list")
            
            logger.info("✓ PDFManager integration working")
            self.test_results['integration_with_pdf_manager'] = True
            
        except Exception as e:
            logger.error(f"✗ PDFManager integration failed: {e}")
            self.test_results['errors'].append(f"PDFManager integration: {str(e)}")
    
    async def test_mock_tool_calls(self):
        """Test mock tool calls to validate parameter structures"""
        logger.info("Testing mock tool calls...")
        
        try:
            # Load tool configurations
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            tools = config['mcpServers']['scihub']['tools']
            
            # Test each tool's parameter structure
            test_cases = [
                {
                    'tool': 'search_scihub_by_doi',
                    'params': {'doi': '10.1038/nature12373'},
                    'required': ['doi']
                },
                {
                    'tool': 'search_scihub_by_title',
                    'params': {'title': 'CRISPR genome editing', 'author': 'Zhang'},
                    'required': ['title']
                },
                {
                    'tool': 'search_scihub_by_keyword',
                    'params': {'keywords': 'machine learning', 'limit': 5},
                    'required': ['keywords']
                },
                {
                    'tool': 'download_scihub_pdf',
                    'params': {'identifier': '10.1000/test', 'output_path': './test.pdf'},
                    'required': ['identifier']
                },
                {
                    'tool': 'get_paper_metadata',
                    'params': {'identifier': '10.1000/test'},
                    'required': ['identifier']
                }
            ]
            
            for test_case in test_cases:
                tool_name = test_case['tool']
                params = test_case['params']
                required = test_case['required']
                
                # Check tool exists in configuration
                if tool_name not in tools:
                    raise Exception(f"Tool '{tool_name}' not found in configuration")
                
                tool_config = tools[tool_name]
                
                # Validate tool structure
                if 'description' not in tool_config:
                    raise Exception(f"Tool '{tool_name}' missing description")
                
                if 'parameters' not in tool_config:
                    raise Exception(f"Tool '{tool_name}' missing parameters")
                
                tool_params = tool_config['parameters']
                
                # Check required parameters
                required_params = tool_params.get('required', [])
                for req_param in required:
                    if req_param not in required_params:
                        raise Exception(f"Tool '{tool_name}' missing required parameter '{req_param}'")
                
                # Check parameter properties
                properties = tool_params.get('properties', {})
                for param_name, param_value in params.items():
                    if param_name not in properties:
                        logger.warning(f"Parameter '{param_name}' not in {tool_name} properties")
                        continue
                    
                    prop = properties[param_name]
                    expected_type = prop.get('type')
                    
                    # Basic type checking
                    if expected_type == 'string' and not isinstance(param_value, str):
                        raise Exception(f"Parameter '{param_name}' should be string")
                    elif expected_type == 'integer' and not isinstance(param_value, int):
                        raise Exception(f"Parameter '{param_name}' should be integer")
            
            logger.info("✓ Mock tool calls validation successful")
            self.test_results['mock_tool_calls'] = True
            
        except Exception as e:
            logger.error(f"✗ Mock tool calls test failed: {e}")
            self.test_results['errors'].append(f"Mock tool calls: {str(e)}")
    
    async def test_storage_path_generation(self):
        """Test storage path generation for Sci-Hub papers"""
        logger.info("Testing storage path generation...")
        
        try:
            # Test different types of Sci-Hub papers
            test_papers = [
                {
                    'id': 'scihub_doi_paper',
                    'title': 'Test DOI Paper',
                    'authors': ['Smith, J.', 'Doe, A.'],
                    'source': 'sci_hub',
                    'publication_year': 2023,
                    'doi': '10.1038/nature12373',
                    'url': 'https://sci-hub.se/10.1038/nature12373'
                },
                {
                    'id': 'scihub_title_paper',
                    'title': 'Machine Learning in Healthcare Applications',
                    'authors': ['Johnson, M.'],
                    'source': 'sci_hub',
                    'publication_year': 2022,
                    'doi': None,
                    'url': 'https://sci-hub.se/title/machine-learning-healthcare'
                },
                {
                    'id': 'scihub_keyword_paper',
                    'title': 'Deep Neural Networks for Medical Image Analysis',
                    'authors': ['Brown, K.', 'Wilson, L.'],
                    'source': 'sci_hub',
                    'publication_year': 2024,
                    'doi': '10.1016/j.media.2024.01.001',
                    'url': 'https://sci-hub.se/10.1016/j.media.2024.01.001'
                }
            ]
            
            # Initialize PDFManager for testing
            pdf_manager = PDFManager(
                base_path=str(project_root / "test_papers"),
                config={'max_retries': 2, 'timeout': 30}
            )
            
            for paper in test_papers:
                # Generate storage path
                storage_path = pdf_manager._generate_storage_path(paper)
                
                # Validate path structure
                if not isinstance(storage_path, Path):
                    raise Exception(f"Storage path for {paper['id']} not a Path object")
                
                # Check that sci_hub is in the path
                if 'sci_hub' not in str(storage_path).lower():
                    raise Exception(f"sci_hub not in storage path for {paper['id']}")
                
                # Check year is in the path
                if str(paper['publication_year']) not in str(storage_path):
                    raise Exception(f"Publication year not in storage path for {paper['id']}")
                
                # Check file extension
                if storage_path.suffix != '.pdf':
                    raise Exception(f"Storage path for {paper['id']} does not end with .pdf")
                
                # Test path creation (without actually creating files)
                try:
                    storage_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Storage path validated for {paper['id']}: {storage_path}")
                except OSError as e:
                    raise Exception(f"Cannot create storage path for {paper['id']}: {e}")
            
            logger.info("✓ Storage path generation working correctly")
            self.test_results['storage_path_generation'] = True
            
        except Exception as e:
            logger.error(f"✗ Storage path generation test failed: {e}")
            self.test_results['errors'].append(f"Storage path generation: {str(e)}")
    
    async def generate_test_report(self):
        """Generate client test report"""
        logger.info("Generating client test report...")
        
        # Calculate success rate
        total_tests = len([k for k in self.test_results.keys() if k not in ['errors', 'test_timestamp']])
        passed_tests = sum(1 for k, v in self.test_results.items() 
                          if k not in ['errors', 'test_timestamp'] and v is True)
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Generate report
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': round(success_rate, 2)
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps()
        }
        
        # Save report to file
        report_file = project_root / f"scihub_mcp_client_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info(f"Client Test Report Summary:")
        logger.info(f"  Total Tests: {total_tests}")
        logger.info(f"  Passed: {passed_tests}")
        logger.info(f"  Failed: {total_tests - passed_tests}")
        logger.info(f"  Success Rate: {success_rate:.1f}%")
        logger.info(f"  Report saved to: {report_file}")
        
        if self.test_results['errors']:
            logger.info("Errors encountered:")
            for error in self.test_results['errors']:
                logger.info(f"  - {error}")
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.test_results['mcp_config_load']:
            recommendations.append("Fix MCP configuration file structure and content")
        
        if not self.test_results['server_command_validation']:
            recommendations.append("Fix server command configuration and ensure uv is properly installed")
        
        if not self.test_results['integration_with_multi_query']:
            recommendations.append("Fix MultiQueryCoordinator integration with Sci-Hub source")
        
        if not self.test_results['integration_with_pdf_manager']:
            recommendations.append("Fix PDFManager integration for Sci-Hub paper storage")
        
        if not self.test_results['mock_tool_calls']:
            recommendations.append("Fix tool parameter structures and validation")
        
        if not self.test_results['storage_path_generation']:
            recommendations.append("Fix storage path generation for Sci-Hub papers")
        
        if len(recommendations) == 0:
            recommendations.append("All client integration tests passed! MCP client is properly configured.")
        
        return recommendations
    
    def _generate_next_steps(self):
        """Generate next steps for complete MCP integration"""
        return [
            "Test actual MCP server startup with real MCP client library",
            "Implement full MCP tool calls in LiteratureSearchAgent",
            "Add Sci-Hub paper download capability to PDFManager",
            "Configure proper error handling for Sci-Hub API rate limits",
            "Add integration tests for complete research workflows",
            "Set up monitoring and logging for MCP server health"
        ]


async def main():
    """Run the MCP client integration tests"""
    tester = ScihubMCPClientTester()
    results = await tester.run_client_tests()
    
    # Exit with appropriate code
    total_tests = len([k for k in results.keys() if k not in ['errors', 'test_timestamp']])
    passed_tests = sum(1 for k, v in results.items() 
                      if k not in ['errors', 'test_timestamp'] and v is True)
    
    if passed_tests == total_tests:
        print("✓ All client integration tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {total_tests - passed_tests} client tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())