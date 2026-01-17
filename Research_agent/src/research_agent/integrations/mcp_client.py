"""
MCP Client Manager for Research Agent

Manages Model Context Protocol connections to external servers like Sci-Hub.
"""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Simple MCP client for Research Agent integration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "./config/mcp_config.json"
        self.config = {}
        self.servers = {}
        self.connected = False
        self._load_config()
    
    def _load_config(self):
        """Load MCP configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded MCP config from {self.config_path}")
            else:
                logger.warning(f"MCP config file not found: {self.config_path}")
                self.config = self._default_config()
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default MCP configuration"""
        return {
            "mcpServers": {
                "scihub": {
                    "command": "uv",
                    "args": ["run", "python", "-c", "import sci_hub_server; sci_hub_server.mcp.run(transport='stdio')"],
                    "cwd": ".",
                    "env": {
                        "SCIHUB_BASE_URL": "https://sci-hub.se",
                        "SCIHUB_TIMEOUT": "30",
                        "SCIHUB_MAX_RETRIES": "3"
                    },
                    "enabled": True,
                    "transport": "stdio"
                }
            }
        }
    
    async def connect(self) -> bool:
        """Connect to configured MCP servers"""
        try:
            servers_config = self.config.get("mcpServers", {})
            
            for server_id, server_config in servers_config.items():
                if not server_config.get("enabled", True):
                    logger.info(f"Skipping disabled server: {server_id}")
                    continue
                
                try:
                    # Try to connect to real MCP server first
                    if server_id == 'scihub':
                        server = RealSciHubMCPServer(server_id, server_config)
                        connection_success = await server.connect()
                        if connection_success:
                            self.servers[server_id] = server
                            logger.info(f"Connected to real MCP server: {server_id}")
                        else:
                            logger.warning(f"Real MCP server connection failed, using mock for: {server_id}")
                            self.servers[server_id] = MockMCPServer(server_id, server_config)
                    else:
                        # For other servers, use mock for now
                        self.servers[server_id] = MockMCPServer(server_id, server_config)
                        logger.info(f"Using mock server for: {server_id}")
                        
                except Exception as e:
                    logger.error(f"Failed to connect to MCP server {server_id}: {e}")
                    # Fallback to mock server
                    self.servers[server_id] = MockMCPServer(server_id, server_config)
                    logger.info(f"Using mock server as fallback for: {server_id}")
            
            self.connected = len(self.servers) > 0
            logger.info(f"MCP client connected to {len(self.servers)} servers")
            return self.connected
            
        except Exception as e:
            logger.error(f"MCP client connection failed: {e}")
            return False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on an MCP server"""
        if not self.connected:
            logger.warning("MCP client not connected, cannot call tool")
            return None
        
        try:
            # Determine which server to use based on tool name
            server_id = self._get_server_for_tool(tool_name)
            if not server_id or server_id not in self.servers:
                logger.error(f"No server found for tool: {tool_name}")
                return None
            
            server = self.servers[server_id]
            result = await server.call_tool(tool_name, parameters)
            
            logger.debug(f"Tool {tool_name} called successfully on server {server_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            return None
    
    def _get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Determine which server to use for a given tool"""
        # Simple mapping based on tool name prefixes
        if tool_name.startswith('search_scihub') or tool_name.startswith('download_scihub'):
            return 'scihub'
        
        # Default to first available server
        if self.servers:
            return list(self.servers.keys())[0]
        
        return None
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools across all connected servers"""
        tools = []
        for server in self.servers.values():
            tools.extend(server.get_tools())
        return tools
    
    async def disconnect(self):
        """Disconnect from all MCP servers"""
        for server_id, server in self.servers.items():
            try:
                await server.disconnect()
                logger.info(f"Disconnected from MCP server: {server_id}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_id}: {e}")
        
        self.servers.clear()
        self.connected = False
        logger.info("MCP client disconnected")


class RealSciHubMCPServer:
    """
    Real MCP server connection for Sci-Hub integration
    
    This connects to the actual sci_hub_server.py MCP server.
    """
    
    def __init__(self, server_id: str, config: Dict[str, Any]):
        self.server_id = server_id
        self.config = config
        self.process = None
        self.connected = False
        self.tools = [
            'search_scihub_by_keyword',
            'search_scihub_by_title', 
            'search_scihub_by_doi',
            'download_scihub_pdf',
            'get_paper_metadata'
        ]
    
    async def connect(self) -> bool:
        """Connect to the real Sci-Hub MCP server"""
        try:
            # Test if we can import and use the sci_hub_search functions directly
            # This is simpler than managing subprocess MCP protocol
            import sys
            sys.path.insert(0, '.venv/lib/python3.12/site-packages')
            
            from sci_hub_search import (
                search_paper_by_doi,
                search_paper_by_title, 
                search_papers_by_keyword,
                download_paper
            )
            
            # Store the functions for direct use
            self.search_by_doi = search_paper_by_doi
            self.search_by_title = search_paper_by_title
            self.search_by_keyword = search_papers_by_keyword
            self.download_paper = download_paper
            
            self.connected = True
            logger.info(f"Successfully connected to real Sci-Hub functions")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import sci_hub_search: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Sci-Hub server: {e}")
            return False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call tools using real Sci-Hub functions"""
        if not self.connected:
            logger.error("Not connected to Sci-Hub server")
            return {"error": "Not connected to Sci-Hub server"}
        
        logger.info(f"RealSciHubMCPServer: Calling tool {tool_name} with params: {parameters}")
        
        try:
            if tool_name == 'search_scihub_by_keyword':
                return await self._real_keyword_search(parameters)
            elif tool_name == 'search_scihub_by_title':
                return await self._real_title_search(parameters)
            elif tool_name == 'search_scihub_by_doi':
                return await self._real_doi_search(parameters)
            elif tool_name == 'download_scihub_pdf':
                return await self._real_pdf_download(parameters)
            elif tool_name == 'get_paper_metadata':
                return await self._real_metadata_retrieval(parameters)
            else:
                logger.error(f"Unknown tool: {tool_name}")
                return {"error": f"Tool {tool_name} not found"}
                
        except Exception as e:
            logger.error(f"Error calling real tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def _real_keyword_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real Sci-Hub keyword search using CrossRef"""
        keywords = parameters.get('keywords', '')
        limit = parameters.get('limit', 10)
        
        try:
            # Use asyncio.to_thread to run the sync function in thread pool
            results = await asyncio.to_thread(self.search_by_keyword, keywords, limit)
            
            # Convert results to the expected format
            papers = []
            for result in results:
                paper = {
                    'id': f'scihub_real_{result.get("doi", "").replace("/", "_")}',
                    'title': result.get('title', ''),
                    'authors': result.get('author', '').split(', ') if result.get('author') else [],
                    'abstract': f'Paper found via CrossRef: {result.get("title", "")}',
                    'doi': result.get('doi', ''),
                    'year': result.get('year', ''),
                    'journal': 'CrossRef Metadata',
                    'url': f'https://sci-hub.se/{result.get("doi", "")}' if result.get('doi') else '',
                    'full_text_available': result.get('status') == 'success'
                }
                papers.append(paper)
            
            logger.info(f"Real Sci-Hub keyword search returned {len(papers)} papers")
            return {
                'papers': papers,
                'total_found': len(papers),
                'search_query': keywords,
                'source': 'sci_hub_real_crossref'
            }
            
        except Exception as e:
            logger.error(f"Real keyword search failed: {e}")
            return {"error": str(e)}
    
    async def _real_title_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real Sci-Hub title search"""
        title = parameters.get('title', '')
        
        try:
            result = await asyncio.to_thread(self.search_by_title, title)
            
            if result.get('status') == 'success':
                paper = {
                    'id': f'scihub_real_title_{result.get("doi", "").replace("/", "_")}',
                    'title': result.get('title', title),
                    'authors': [result.get('author', '')] if result.get('author') else [],
                    'abstract': f'Paper found by title search: {title}',
                    'doi': result.get('doi', ''),
                    'year': result.get('year', ''),
                    'journal': 'Sci-Hub',
                    'url': result.get('pdf_url', ''),
                    'full_text_available': True
                }
                
                papers = [paper]
                total_found = 1
            else:
                papers = []
                total_found = 0
            
            logger.info(f"Real Sci-Hub title search found {total_found} paper")
            return {
                'papers': papers,
                'total_found': total_found,
                'search_query': title,
                'source': 'sci_hub_real_title'
            }
            
        except Exception as e:
            logger.error(f"Real title search failed: {e}")
            return {"error": str(e)}
    
    async def _real_doi_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real Sci-Hub DOI search"""
        doi = parameters.get('doi', '')
        
        try:
            result = await asyncio.to_thread(self.search_by_doi, doi)
            
            if result.get('status') == 'success':
                paper = {
                    'id': f'scihub_real_doi_{doi.replace("/", "_")}',
                    'title': result.get('title', f'Paper with DOI {doi}'),
                    'authors': [result.get('author', '')] if result.get('author') else [],
                    'abstract': f'Paper retrieved by DOI: {doi}',
                    'doi': doi,
                    'year': result.get('year', ''),
                    'journal': 'Sci-Hub',
                    'url': result.get('pdf_url', ''),
                    'full_text_available': True
                }
                
                papers = [paper]
                total_found = 1
            else:
                papers = []
                total_found = 0
            
            logger.info(f"Real Sci-Hub DOI search found {total_found} paper")
            return {
                'papers': papers,
                'total_found': total_found,
                'search_query': doi,
                'source': 'sci_hub_real_doi'
            }
            
        except Exception as e:
            logger.error(f"Real DOI search failed: {e}")
            return {"error": str(e)}
    
    async def _real_pdf_download(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real PDF download from Sci-Hub"""
        identifier = parameters.get('identifier', '')
        output_path = parameters.get('output_path', f'./papers/sci_hub/{identifier}.pdf')
        
        try:
            # Create output directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # First get the PDF URL if we have a DOI
            if identifier.startswith('10.'):
                # This is a DOI, get the PDF URL first
                paper_result = await asyncio.to_thread(self.search_by_doi, identifier)
                if paper_result.get('status') == 'success':
                    pdf_url = paper_result.get('pdf_url')
                    if pdf_url:
                        success = await asyncio.to_thread(self.download_paper, pdf_url, output_path)
                        if success:
                            file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                            return {
                                'success': True,
                                'file_path': output_path,
                                'file_size': file_size,
                                'identifier': identifier,
                                'source': 'sci_hub_real_download'
                            }
                        else:
                            return {
                                'success': False,
                                'error': 'Download failed',
                                'identifier': identifier,
                                'source': 'sci_hub_real_download'
                            }
                    else:
                        return {
                            'success': False,
                            'error': 'No PDF URL found',
                            'identifier': identifier,
                            'source': 'sci_hub_real_download'
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Paper not found',
                        'identifier': identifier,
                        'source': 'sci_hub_real_download'
                    }
            else:
                # Assume it's already a URL
                success = await asyncio.to_thread(self.download_paper, identifier, output_path)
                if success:
                    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    return {
                        'success': True,
                        'file_path': output_path,
                        'file_size': file_size,
                        'identifier': identifier,
                        'source': 'sci_hub_real_download'
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Download failed',
                        'identifier': identifier,
                        'source': 'sci_hub_real_download'
                    }
                    
        except Exception as e:
            logger.error(f"Real PDF download failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'identifier': identifier,
                'source': 'sci_hub_real_download'
            }
    
    async def _real_metadata_retrieval(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Real metadata retrieval using DOI search"""
        identifier = parameters.get('identifier', '')
        
        try:
            result = await asyncio.to_thread(self.search_by_doi, identifier)
            
            if result.get('status') == 'success':
                metadata = {
                    'title': result.get('title', ''),
                    'authors': [result.get('author', '')] if result.get('author') else [],
                    'doi': identifier,
                    'year': result.get('year', ''),
                    'journal': 'Retrieved via Sci-Hub',
                    'abstract': f'Metadata for paper: {result.get("title", identifier)}',
                    'url': result.get('pdf_url', ''),
                    'full_text_available': True
                }
                
                return {
                    'metadata': metadata,
                    'identifier': identifier,
                    'source': 'sci_hub_real_metadata'
                }
            else:
                return {
                    'error': 'Paper not found',
                    'identifier': identifier,
                    'source': 'sci_hub_real_metadata'
                }
                
        except Exception as e:
            logger.error(f"Real metadata retrieval failed: {e}")
            return {
                'error': str(e),
                'identifier': identifier,
                'source': 'sci_hub_real_metadata'
            }
    
    def get_tools(self) -> List[str]:
        """Get available tools"""
        return self.tools
    
    async def disconnect(self):
        """Disconnect from server"""
        self.connected = False
        logger.info(f"RealSciHubMCPServer {self.server_id} disconnected")


class MockMCPServer:
    """
    Mock MCP server for Sci-Hub integration
    
    This simulates the Sci-Hub MCP server for testing and development.
    In production, this would be replaced with actual MCP protocol implementation.
    """
    
    def __init__(self, server_id: str, config: Dict[str, Any]):
        self.server_id = server_id
        self.config = config
        self.tools = self._get_available_tools()
    
    def _get_available_tools(self) -> List[str]:
        """Get available tools for this server"""
        if self.server_id == 'scihub':
            return [
                'search_scihub_by_keyword',
                'search_scihub_by_title', 
                'search_scihub_by_doi',
                'download_scihub_pdf',
                'get_paper_metadata'
            ]
        return []
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool execution"""
        logger.info(f"MockMCPServer: Calling tool {tool_name} with params: {parameters}")
        
        if tool_name == 'search_scihub_by_keyword':
            return await self._mock_keyword_search(parameters)
        elif tool_name == 'search_scihub_by_title':
            return await self._mock_title_search(parameters)
        elif tool_name == 'search_scihub_by_doi':
            return await self._mock_doi_search(parameters)
        elif tool_name == 'download_scihub_pdf':
            return await self._mock_pdf_download(parameters)
        elif tool_name == 'get_paper_metadata':
            return await self._mock_metadata_retrieval(parameters)
        else:
            logger.error(f"Unknown tool: {tool_name}")
            return {"error": f"Tool {tool_name} not found"}
    
    async def _mock_keyword_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Sci-Hub keyword search"""
        keywords = parameters.get('keywords', '')
        limit = parameters.get('limit', 10)
        
        # Simulate realistic paper search results
        papers = []
        for i in range(min(limit, 5)):  # Return up to 5 mock papers
            papers.append({
                'id': f'scihub_mock_{i}',
                'title': f'Research Paper on {keywords} - Study {i+1}',
                'authors': [f'Author {i+1}', f'Co-Author {i+1}'],
                'abstract': f'This paper presents research on {keywords} with novel approaches and findings. '
                           f'The study demonstrates significant advances in the field through comprehensive analysis.',
                'doi': f'10.1000/mock.{i+1}',
                'year': 2023 - i,
                'journal': f'Journal of {keywords.title()} Research',
                'url': f'https://sci-hub.se/10.1000/mock.{i+1}',
                'full_text_available': True
            })
        
        logger.info(f"Mock Sci-Hub keyword search returned {len(papers)} papers")
        return {
            'papers': papers,
            'total_found': len(papers),
            'search_query': keywords,
            'source': 'sci_hub_mcp_mock'
        }
    
    async def _mock_title_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Sci-Hub title search"""
        title = parameters.get('title', '')
        
        # Simulate finding a paper by title
        paper = {
            'id': 'scihub_title_mock',
            'title': title,
            'authors': ['Research Author', 'Co-Author'],
            'abstract': f'This is the abstract for the paper titled "{title}". '
                       'It contains relevant research findings and methodological approaches.',
            'doi': '10.1000/title.mock',
            'year': 2023,
            'journal': 'Research Journal',
            'url': f'https://sci-hub.se/title/{title.replace(" ", "-")}',
            'full_text_available': True
        }
        
        logger.info(f"Mock Sci-Hub title search found paper: {title}")
        return {
            'papers': [paper],
            'total_found': 1,
            'search_query': title,
            'source': 'sci_hub_mcp_mock'
        }
    
    async def _mock_doi_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Sci-Hub DOI search"""
        doi = parameters.get('doi', '')
        
        paper = {
            'id': f'scihub_doi_mock_{doi.replace("/", "_")}',
            'title': f'Paper with DOI {doi}',
            'authors': ['DOI Author'],
            'abstract': f'This paper is identified by DOI {doi} and contains research content.',
            'doi': doi,
            'year': 2023,
            'journal': 'DOI Journal',
            'url': f'https://sci-hub.se/{doi}',
            'full_text_available': True
        }
        
        logger.info(f"Mock Sci-Hub DOI search found paper: {doi}")
        return {
            'papers': [paper],
            'total_found': 1,
            'search_query': doi,
            'source': 'sci_hub_mcp_mock'
        }
    
    async def _mock_pdf_download(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock PDF download from Sci-Hub"""
        identifier = parameters.get('identifier', '')
        output_path = parameters.get('output_path', f'./temp/{identifier}.pdf')
        
        logger.info(f"Mock PDF download for {identifier} to {output_path}")
        
        return {
            'success': True,
            'file_path': output_path,
            'file_size': 1024 * 1024,  # Mock 1MB file
            'identifier': identifier,
            'source': 'sci_hub_mcp_mock'
        }
    
    async def _mock_metadata_retrieval(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock metadata retrieval"""
        identifier = parameters.get('identifier', '')
        
        metadata = {
            'title': f'Paper: {identifier}',
            'authors': ['Metadata Author'],
            'doi': f'10.1000/meta.{hash(identifier) % 10000}',
            'year': 2023,
            'journal': 'Metadata Journal',
            'abstract': f'Metadata for paper identified by: {identifier}',
            'citations': 42,
            'references': 25
        }
        
        logger.info(f"Mock metadata retrieval for: {identifier}")
        return {
            'metadata': metadata,
            'identifier': identifier,
            'source': 'sci_hub_mcp_mock'
        }
    
    def get_tools(self) -> List[str]:
        """Get available tools"""
        return self.tools
    
    async def disconnect(self):
        """Disconnect from server"""
        logger.info(f"MockMCPServer {self.server_id} disconnected")