"""
Literature Search Agent

Coordinates search across Sci-Hub MCP, ArXiv, and PubMed for comprehensive paper retrieval.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, date
import re

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Academic API clients
import arxiv
from Bio import Entrez
from Bio.Entrez import efetch, esearch

logger = logging.getLogger(__name__)


class LiteratureSearchAgent:
    """
    Agent that searches literature using Sci-Hub MCP, ArXiv, and PubMed APIs
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.search_results_cache: Dict[str, List[Dict]] = {}
        
        # Initialize PubMed settings
        Entrez.email = config.get('pubmed', {}).get('email', 'research@example.com')
        Entrez.tool = config.get('pubmed', {}).get('tool', 'ResearchAgent')
        
        # Initialize PromptChain for search optimization
        self.chain = PromptChain(
            models=[config.get('model', 'openai/gpt-4o-mini')],
            instructions=[
                "You are a literature search specialist. Optimize search queries for: {databases}",
                AgenticStepProcessor(
                    objective="Optimize search queries and coordinate literature retrieval",
                    max_internal_steps=config.get('processor', {}).get('max_internal_steps', 3)
                ),
                "Provide search optimization recommendations and paper filtering guidance"
            ],
            verbose=True
        )
        
        # Register search optimization tools
        self._register_tools()
        
        logger.info("LiteratureSearchAgent initialized")
    
    def _register_tools(self):
        """Register tools for search optimization and coordination"""
        
        def search_query_optimizer(query_context: str) -> str:
            """Optimize search queries for different academic databases"""
            optimization_guide = """
            Database-Specific Query Optimization:
            
            ARXIV:
            - Use 'ti:' for title search, 'abs:' for abstract
            - Categories: 'cat:cs.AI', 'cat:cs.LG', 'cat:q-bio'
            - Boolean: AND, OR, ANDNOT operators
            - Date: 'submittedDate:[20200101 TO 20241231]'
            
            PUBMED:
            - MeSH terms: "machine learning"[MeSH Terms]
            - Field tags: [Title/Abstract], [Author], [Journal]
            - Date: ("2020/01/01"[Date - Publication] : "2024/12/31"[Date - Publication])
            - Boolean: AND, OR, NOT operators
            
            SCI-HUB MCP:
            - DOI-based retrieval preferred
            - Title + author fallback search
            - Journal + year filtering
            """
            return optimization_guide
        
        def paper_quality_filter(paper_metadata: str) -> str:
            """Provide guidance on filtering papers by quality metrics"""
            quality_metrics = """
            Paper Quality Filtering Criteria:
            
            HIGH PRIORITY:
            - Peer-reviewed journal articles
            - Conference papers from top venues
            - High citation count (varies by field/age)
            - Recent publications (2020+)
            - Full text availability
            
            MEDIUM PRIORITY:
            - Preprints from reputable sources
            - Workshop papers with novel insights
            - Technical reports from institutions
            - Review papers for background
            
            FILTERING RULES:
            - Exclude non-English papers (unless specified)
            - Exclude papers without abstracts
            - Prioritize open access content
            - Filter duplicates across databases
            """
            return quality_metrics
        
        def search_result_deduplication(papers_list: str) -> str:
            """Guide deduplication of papers across multiple sources"""
            dedup_strategy = """
            Paper Deduplication Strategy:
            
            MATCHING CRITERIA:
            1. Exact DOI match (highest confidence)
            2. Title similarity > 90% + author overlap
            3. ArXiv ID matching across sources
            4. Similar title + publication year + first author
            
            PRIORITY ORDER:
            1. Journal version over preprint
            2. More recent version
            3. Version with full text access
            4. Higher citation count
            
            MERGE STRATEGY:
            - Combine metadata from all sources
            - Keep best abstract (longest/most complete)
            - Track all source URLs/identifiers
            """
            return dedup_strategy
        
        # Register tools
        self.chain.register_tool_function(search_query_optimizer)
        self.chain.register_tool_function(paper_quality_filter)
        self.chain.register_tool_function(search_result_deduplication)
        
        # Add tool schemas
        self.chain.add_tools([
            {
                "type": "function",
                "function": {
                    "name": "search_query_optimizer",
                    "description": "Get database-specific query optimization guidance",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_context": {"type": "string", "description": "Search context and target databases"}
                        },
                        "required": ["query_context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "paper_quality_filter",
                    "description": "Get guidance on paper quality filtering",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paper_metadata": {"type": "string", "description": "Paper metadata for quality assessment"}
                        },
                        "required": ["paper_metadata"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_result_deduplication",
                    "description": "Get deduplication strategy for search results",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "papers_list": {"type": "string", "description": "List of papers for deduplication"}
                        },
                        "required": ["papers_list"]
                    }
                }
            }
        ])
    
    async def search_papers(
        self,
        strategy: str,
        max_papers: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search papers using the provided strategy across all configured databases
        
        Args:
            strategy: JSON string containing search strategy from SearchStrategistAgent
            max_papers: Maximum total papers to retrieve
            
        Returns:
            List of paper dictionaries with metadata and content
        """
        logger.info(f"Starting literature search with max {max_papers} papers")
        
        try:
            # Parse search strategy
            strategy_data = json.loads(strategy)
            db_allocation = strategy_data.get('database_allocation', {})
            search_terms = self._extract_search_terms(strategy_data)
            
            # Execute searches in parallel
            search_tasks = []
            
            # Sci-Hub MCP search
            if 'sci_hub' in db_allocation:
                sci_hub_config = db_allocation['sci_hub']
                sci_hub_limit = min(sci_hub_config.get('max_papers', 40), max_papers)
                search_tasks.append(self._search_scihub_mcp(
                    search_terms.get('sci_hub', []),
                    sci_hub_limit
                ))
            
            # ArXiv search
            if 'arxiv' in db_allocation:
                arxiv_config = db_allocation['arxiv']
                arxiv_limit = min(arxiv_config.get('max_papers', 30), max_papers)
                search_tasks.append(self._search_arxiv(
                    search_terms.get('arxiv', []),
                    arxiv_limit
                ))
            
            # PubMed search
            if 'pubmed' in db_allocation:
                pubmed_config = db_allocation['pubmed']
                pubmed_limit = min(pubmed_config.get('max_papers', 20), max_papers)
                search_tasks.append(self._search_pubmed(
                    search_terms.get('pubmed', []),
                    pubmed_limit
                ))
            
            # Execute all searches concurrently
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine and process results
            all_papers = []
            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.error(f"Search {i} failed: {result}")
                    continue
                
                if isinstance(result, list):
                    all_papers.extend(result)
            
            # Deduplicate and filter papers
            deduplicated_papers = self._deduplicate_papers(all_papers)
            
            # Apply quality filtering
            filtered_papers = self._filter_papers_by_quality(deduplicated_papers)
            
            # Limit total results
            final_papers = filtered_papers[:max_papers]
            
            logger.info(f"Literature search completed: {len(final_papers)} papers from {len(all_papers)} initial results")
            
            return final_papers
            
        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            return []
    
    def _extract_search_terms(self, strategy_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract database-specific search terms from strategy"""
        search_terms = {}
        
        db_allocation = strategy_data.get('database_allocation', {})
        primary_keywords = strategy_data.get('search_strategy', {}).get('primary_keywords', [])
        secondary_keywords = strategy_data.get('search_strategy', {}).get('secondary_keywords', [])
        
        for db, config in db_allocation.items():
            terms = config.get('search_terms', [])
            
            # Add primary keywords
            terms.extend(primary_keywords[:3])  # Top 3 primary keywords
            
            # Add some secondary keywords 
            terms.extend(secondary_keywords[:2])  # Top 2 secondary keywords
            
            # Remove duplicates and empty terms
            search_terms[db] = list(set([t for t in terms if t and t.strip()]))
        
        return search_terms
    
    async def _search_scihub_mcp(self, search_terms: List[str], max_papers: int) -> List[Dict[str, Any]]:
        """Search using Sci-Hub MCP server"""
        try:
            # This would integrate with the actual Sci-Hub MCP server
            # For now, we'll create a placeholder that simulates the integration
            
            papers = []
            
            # NOTE: In real implementation, this would use MCP client to call sci-hub tools
            # The MCP integration would be something like:
            # mcp_client = self.get_mcp_client('sci_hub')
            # results = await mcp_client.call_tool('search_papers', {
            #     'query': ' OR '.join(search_terms),
            #     'max_results': max_papers
            # })
            
            # Placeholder implementation
            for i, term in enumerate(search_terms[:max_papers//len(search_terms) if search_terms else 1]):
                # Simulate sci-hub paper results
                paper = {
                    'id': f'scihub_{i}_{hash(term) % 10000}',
                    'title': f'Research on {term}: A Comprehensive Study',
                    'authors': ['Author, First', 'Researcher, Second'],
                    'abstract': f'This paper presents comprehensive research on {term}, exploring various methodologies and applications in the field.',
                    'doi': f'10.1000/example.{i}.{hash(term) % 1000}',
                    'publication_year': 2023 - (i % 4),
                    'journal': 'Journal of Advanced Research',
                    'source': 'sci_hub',
                    'url': f'https://sci-hub.se/{hash(term) % 10000}',
                    'full_text_available': True,
                    'metadata': {
                        'search_term': term,
                        'database_priority': 1.0,
                        'retrieved_at': datetime.now().isoformat()
                    }
                }
                papers.append(paper)
                
                if len(papers) >= max_papers:
                    break
            
            logger.info(f"Sci-Hub MCP search: {len(papers)} papers retrieved")
            return papers
            
        except Exception as e:
            logger.error(f"Sci-Hub MCP search failed: {e}")
            return []
    
    async def _search_arxiv(self, search_terms: List[str], max_papers: int) -> List[Dict[str, Any]]:
        """Search ArXiv using arxiv-py client"""
        try:
            papers = []
            
            # Build ArXiv query
            if not search_terms:
                return papers
            
            # Combine search terms with OR logic
            query_parts = []
            for term in search_terms[:5]:  # Limit to 5 terms to avoid overly complex queries
                # Search in title and abstract
                query_parts.append(f'ti:"{term}" OR abs:"{term}"')
            
            query = ' OR '.join(f'({part})' for part in query_parts)
            
            # Add date filter for recent papers
            current_year = datetime.now().year
            date_filter = f' AND submittedDate:[{current_year-4}0101 TO {current_year}1231]'
            query += date_filter
            
            logger.info(f"ArXiv query: {query}")
            
            # Execute search
            search = arxiv.Search(
                query=query,
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance,
                sort_order=arxiv.SortOrder.Descending
            )
            
            for i, result in enumerate(search.results()):
                # Extract paper metadata
                paper = {
                    'id': f'arxiv_{result.entry_id.split("/")[-1]}',
                    'title': result.title.strip(),
                    'authors': [str(author) for author in result.authors],
                    'abstract': result.summary.strip(),
                    'doi': result.doi,
                    'publication_year': result.published.year,
                    'journal': result.journal_ref or 'arXiv preprint',
                    'source': 'arxiv',
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'full_text_available': True,
                    'metadata': {
                        'arxiv_id': result.entry_id.split('/')[-1],
                        'categories': [str(cat) for cat in result.categories],
                        'updated': result.updated.isoformat(),
                        'database_priority': 0.8,
                        'retrieved_at': datetime.now().isoformat()
                    }
                }
                papers.append(paper)
            
            logger.info(f"ArXiv search: {len(papers)} papers retrieved")
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
            return []
    
    async def _search_pubmed(self, search_terms: List[str], max_papers: int) -> List[Dict[str, Any]]:
        """Search PubMed using Biopython"""
        try:
            papers = []
            
            if not search_terms:
                return papers
            
            # Build PubMed query
            query_parts = []
            for term in search_terms[:5]:  # Limit complexity
                # Search in title/abstract field
                query_parts.append(f'"{term}"[Title/Abstract]')
            
            query = ' OR '.join(query_parts)
            
            # Add date filter for recent papers
            current_year = datetime.now().year
            date_filter = f' AND ("2020/01/01"[Date - Publication] : "{current_year}/12/31"[Date - Publication])'
            query += date_filter
            
            logger.info(f"PubMed query: {query}")
            
            # Execute search
            search_handle = esearch(
                db='pubmed',
                term=query,
                retmax=max_papers,
                sort='relevance'
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            pmids = search_results.get('IdList', [])
            
            if not pmids:
                logger.info("No PubMed results found")
                return papers
            
            # Fetch detailed information
            fetch_handle = efetch(
                db='pubmed',
                id=','.join(pmids),
                rettype='medline',
                retmode='xml'
            )
            
            fetch_results = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            # Process results
            for i, article in enumerate(fetch_results['PubmedArticle']):
                try:
                    medline = article.get('MedlineCitation', {})
                    article_data = medline.get('Article', {})
                    
                    # Extract basic info
                    title = article_data.get('ArticleTitle', 'No title')
                    abstract_list = article_data.get('Abstract', {}).get('AbstractText', [])
                    
                    # Handle abstract
                    if isinstance(abstract_list, list):
                        abstract = ' '.join([str(a) for a in abstract_list])
                    else:
                        abstract = str(abstract_list)
                    
                    # Extract authors
                    authors = []
                    author_list = article_data.get('AuthorList', [])
                    for author in author_list:
                        if 'LastName' in author and 'ForeName' in author:
                            authors.append(f"{author['LastName']}, {author['ForeName']}")
                    
                    # Extract publication year
                    pub_date = article_data.get('ArticleDate', [])
                    if pub_date and len(pub_date) > 0:
                        pub_year = int(pub_date[0].get('Year', datetime.now().year))
                    else:
                        # Fallback to journal issue date
                        journal_issue = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
                        pub_year = int(journal_issue.get('Year', datetime.now().year))
                    
                    # Extract journal
                    journal = article_data.get('Journal', {}).get('Title', 'Unknown Journal')
                    
                    # Extract DOI
                    doi = None
                    article_ids = article.get('PubmedData', {}).get('ArticleIdList', [])
                    for aid in article_ids:
                        if aid.get('IdType') == 'doi':
                            doi = str(aid)
                            break
                    
                    pmid = medline.get('PMID', pmids[i])
                    
                    paper = {
                        'id': f'pubmed_{pmid}',
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'doi': doi,
                        'publication_year': pub_year,
                        'journal': journal,
                        'source': 'pubmed',
                        'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
                        'full_text_available': False,  # PubMed typically only has abstracts
                        'metadata': {
                            'pmid': str(pmid),
                            'database_priority': 0.6,
                            'mesh_terms': medline.get('MeshHeadingList', []),
                            'retrieved_at': datetime.now().isoformat()
                        }
                    }
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse PubMed article {i}: {e}")
                    continue
            
            logger.info(f"PubMed search: {len(papers)} papers retrieved")
            return papers
            
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers using multiple matching criteria"""
        if not papers:
            return papers
        
        # Track seen papers by different criteria
        seen_dois: Set[str] = set()
        seen_titles: Set[str] = set()
        seen_arxiv_ids: Set[str] = set()
        
        deduplicated = []
        duplicate_count = 0
        
        for paper in papers:
            is_duplicate = False
            
            # Check DOI duplicates (highest confidence)
            if paper.get('doi'):
                doi = paper['doi'].lower().strip()
                if doi in seen_dois:
                    duplicate_count += 1
                    is_duplicate = True
                else:
                    seen_dois.add(doi)
            
            # Check ArXiv ID duplicates
            arxiv_id = paper.get('metadata', {}).get('arxiv_id')
            if arxiv_id and not is_duplicate:
                if arxiv_id in seen_arxiv_ids:
                    duplicate_count += 1
                    is_duplicate = True
                else:
                    seen_arxiv_ids.add(arxiv_id)
            
            # Check title similarity (basic approach)
            if not is_duplicate:
                title = paper.get('title', '').lower().strip()
                # Simple title normalization
                normalized_title = re.sub(r'[^\w\s]', '', title)
                normalized_title = re.sub(r'\s+', ' ', normalized_title).strip()
                
                if normalized_title in seen_titles:
                    duplicate_count += 1
                    is_duplicate = True
                else:
                    seen_titles.add(normalized_title)
            
            if not is_duplicate:
                deduplicated.append(paper)
        
        logger.info(f"Deduplication: removed {duplicate_count} duplicates, {len(deduplicated)} papers remaining")
        return deduplicated
    
    def _filter_papers_by_quality(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter papers by quality metrics and relevance"""
        if not papers:
            return papers
        
        filtered = []
        filtered_count = 0
        
        for paper in papers:
            # Basic quality filters
            should_include = True
            
            # Must have title and abstract
            if not paper.get('title') or not paper.get('abstract'):
                filtered_count += 1
                should_include = False
            
            # Filter very short abstracts (likely incomplete)
            elif len(paper.get('abstract', '').strip()) < 50:
                filtered_count += 1
                should_include = False
            
            # Filter very old papers (unless it's a foundational work)
            elif paper.get('publication_year', 2024) < 2015:
                # Allow some older papers but deprioritize heavily
                paper['metadata'] = paper.get('metadata', {})
                paper['metadata']['quality_score'] = 0.3
            
            # Prioritize papers with full text access
            if should_include and paper.get('full_text_available'):
                paper['metadata'] = paper.get('metadata', {})
                paper['metadata']['quality_score'] = paper['metadata'].get('quality_score', 1.0) + 0.2
            
            if should_include:
                filtered.append(paper)
        
        # Sort by quality score (if available) and database priority
        def sort_key(paper):
            metadata = paper.get('metadata', {})
            quality_score = metadata.get('quality_score', 1.0)
            db_priority = metadata.get('database_priority', 0.5)
            return -(quality_score * db_priority)  # Negative for descending sort
        
        filtered.sort(key=sort_key)
        
        logger.info(f"Quality filtering: removed {filtered_count} low-quality papers, {len(filtered)} papers remaining")
        return filtered
    
    async def download_paper_content(self, paper: Dict[str, Any]) -> Optional[str]:
        """Download full paper content if available"""
        try:
            if paper.get('source') == 'sci_hub' and paper.get('full_text_available'):
                # Would use Sci-Hub MCP to download full text
                # For now, return placeholder
                return f"Full text content for: {paper.get('title', 'Unknown')}"
            
            elif paper.get('source') == 'arxiv' and paper.get('pdf_url'):
                # Could download ArXiv PDF and extract text
                # For now, return abstract as content
                return paper.get('abstract', '')
            
            else:
                # Return abstract for papers without full text
                return paper.get('abstract', '')
            
        except Exception as e:
            logger.error(f"Failed to download paper content: {e}")
            return paper.get('abstract', '')
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Update agent configuration"""
        self.config.update(config)
        
        # Update PubMed settings if changed
        if 'pubmed' in config:
            Entrez.email = config['pubmed'].get('email', Entrez.email)
            Entrez.tool = config['pubmed'].get('tool', Entrez.tool)
        
        logger.info("LiteratureSearchAgent configuration updated")