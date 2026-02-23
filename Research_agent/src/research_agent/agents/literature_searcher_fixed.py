#!/usr/bin/env python3
"""
Fixed Literature Search Agent - Properly iterates through papers individually
"""

async def enhance_papers_with_scihub_pdfs(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhanced method: Iterate through each paper INDIVIDUALLY to find PDFs
    """
    logger.info("Starting enhanced Sci-Hub PDF retrieval process")
    
    # Build list of papers needing PDFs
    papers_needing_pdfs = []
    papers_with_fulltext = []
    
    for paper in papers:
        # Check if paper needs PDF (PubMed usually doesn't have full text)
        if paper.get('source') == 'pubmed' or not paper.get('full_text_available', False):
            # Extract key info safely
            paper_info = {
                'title': paper.get('title', ''),
                'doi': paper.get('doi', ''),
                'authors': paper.get('authors', []),
                'source': paper.get('source', ''),
                'original_paper': paper  # Keep reference to original
            }
            papers_needing_pdfs.append(paper_info)
        else:
            papers_with_fulltext.append(paper)
    
    logger.info(f"Found {len(papers_needing_pdfs)} papers needing PDF retrieval")
    
    if not papers_needing_pdfs or not self.mcp_client:
        return papers  # Return original if nothing to do
    
    # Track download success
    download_stats = {
        'attempted': 0,
        'successful': 0,
        'failed': []
    }
    
    # ITERATE THROUGH EACH PAPER INDIVIDUALLY
    enhanced_papers = []
    for paper_info in papers_needing_pdfs:
        download_stats['attempted'] += 1
        
        # Try to find and download PDF for THIS SPECIFIC PAPER
        enhanced_paper = await self._process_single_paper_for_pdf(paper_info, download_stats)
        enhanced_papers.append(enhanced_paper)
    
    # Combine results
    final_results = papers_with_fulltext + enhanced_papers
    
    # Log statistics
    logger.info(f"PDF Download Statistics:")
    logger.info(f"  Attempted: {download_stats['attempted']}")
    logger.info(f"  Successful: {download_stats['successful']}")
    logger.info(f"  Success Rate: {download_stats['successful']/download_stats['attempted']*100:.1f}%")
    
    if download_stats['failed']:
        logger.info(f"  Failed papers: {len(download_stats['failed'])}")
        for failed in download_stats['failed'][:5]:  # Show first 5 failures
            logger.debug(f"    - {failed['title'][:60]}... ({failed['reason']})")
    
    return final_results


async def _process_single_paper_for_pdf(self, paper_info: Dict[str, Any], stats: Dict) -> Dict[str, Any]:
    """
    Process a SINGLE paper to find and download its PDF
    Uses multiple search strategies in order of accuracy
    """
    title = paper_info.get('title', '').strip() if paper_info.get('title') else ''
    doi = paper_info.get('doi', '').strip() if paper_info.get('doi') else ''
    authors = paper_info.get('authors', [])
    original_paper = paper_info['original_paper']
    
    # Skip if no searchable info
    if not title and not doi:
        stats['failed'].append({'title': 'Unknown', 'reason': 'No title or DOI'})
        return original_paper
    
    # Strategy 1: Search by DOI (most accurate)
    if doi and self.mcp_client:
        try:
            logger.debug(f"Searching Sci-Hub by DOI: {doi}")
            result = await self.mcp_client.call_tool('search_scihub_by_doi', {'doi': doi})
            
            if result and result.get('success'):
                # Try to download PDF
                pdf_result = await self.mcp_client.call_tool('download_scihub_pdf', {
                    'identifier': doi,
                    'output_path': f"./pdfs/{self._safe_filename(title)}.pdf"
                })
                
                if pdf_result and pdf_result.get('success'):
                    logger.info(f"✅ Downloaded PDF via DOI: {title[:50]}...")
                    stats['successful'] += 1
                    
                    # Update paper metadata
                    enhanced_paper = original_paper.copy()
                    enhanced_paper['full_text_available'] = True
                    enhanced_paper['pdf_path'] = pdf_result.get('path')
                    enhanced_paper['pdf_source'] = 'sci_hub_doi'
                    return enhanced_paper
                    
        except Exception as e:
            logger.debug(f"DOI search failed for {doi}: {e}")
    
    # Strategy 2: Search by exact title
    if title and self.mcp_client:
        try:
            logger.debug(f"Searching Sci-Hub by title: {title[:50]}...")
            result = await self.mcp_client.call_tool('search_scihub_by_title', {'title': title})
            
            if result and result.get('success') and result.get('papers'):
                # Found paper, try to download
                found_paper = result['papers'][0]  # Take first match
                identifier = found_paper.get('doi') or title
                
                pdf_result = await self.mcp_client.call_tool('download_scihub_pdf', {
                    'identifier': identifier,
                    'output_path': f"./pdfs/{self._safe_filename(title)}.pdf"
                })
                
                if pdf_result and pdf_result.get('success'):
                    logger.info(f"✅ Downloaded PDF via title: {title[:50]}...")
                    stats['successful'] += 1
                    
                    # Update paper metadata
                    enhanced_paper = original_paper.copy()
                    enhanced_paper['full_text_available'] = True
                    enhanced_paper['pdf_path'] = pdf_result.get('path')
                    enhanced_paper['pdf_source'] = 'sci_hub_title'
                    return enhanced_paper
                    
        except Exception as e:
            logger.debug(f"Title search failed for {title}: {e}")
    
    # Strategy 3: Search by title + first author
    if title and authors and self.mcp_client:
        try:
            first_author = authors[0] if isinstance(authors[0], str) else str(authors[0])
            search_query = f"{title} {first_author}"
            logger.debug(f"Searching Sci-Hub by title+author: {search_query[:50]}...")
            
            result = await self.mcp_client.call_tool('search_scihub_by_title', {'title': search_query})
            
            if result and result.get('success') and result.get('papers'):
                # Found paper, try to download
                found_paper = result['papers'][0]
                identifier = found_paper.get('doi') or title
                
                pdf_result = await self.mcp_client.call_tool('download_scihub_pdf', {
                    'identifier': identifier,
                    'output_path': f"./pdfs/{self._safe_filename(title)}.pdf"
                })
                
                if pdf_result and pdf_result.get('success'):
                    logger.info(f"✅ Downloaded PDF via title+author: {title[:50]}...")
                    stats['successful'] += 1
                    
                    # Update paper metadata
                    enhanced_paper = original_paper.copy()
                    enhanced_paper['full_text_available'] = True
                    enhanced_paper['pdf_path'] = pdf_result.get('path')
                    enhanced_paper['pdf_source'] = 'sci_hub_title_author'
                    return enhanced_paper
                    
        except Exception as e:
            logger.debug(f"Title+author search failed: {e}")
    
    # If all strategies failed, record failure and return original
    stats['failed'].append({
        'title': title[:60] if title else 'Unknown',
        'reason': 'Not found in Sci-Hub'
    })
    
    # Mark that we tried
    original_paper['metadata'] = original_paper.get('metadata', {})
    original_paper['metadata']['scihub_attempted'] = True
    original_paper['metadata']['scihub_found'] = False
    
    return original_paper


def _safe_filename(self, title: str) -> str:
    """Create a safe filename from paper title"""
    import re
    # Remove special characters and limit length
    safe = re.sub(r'[^\w\s-]', '', title).strip()
    safe = re.sub(r'[-\s]+', '-', safe)
    return safe[:60]  # Limit to 60 chars