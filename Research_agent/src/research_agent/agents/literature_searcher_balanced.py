#!/usr/bin/env python3
"""
Balanced Literature Search Agent - Ensures fair representation from all sources
"""

def balance_sources_fairly(papers: List[Dict[str, Any]], max_papers: int = 15) -> List[Dict[str, Any]]:
    """
    Balance papers from different sources to ensure diversity
    
    Strategy:
    - Minimum 2 papers per active source (if available)
    - Remaining slots filled by quality score
    """
    # Group papers by source
    papers_by_source = {}
    for paper in papers:
        source = paper.get('source', 'unknown')
        if source not in papers_by_source:
            papers_by_source[source] = []
        papers_by_source[source].append(paper)
    
    # Sort each source's papers by quality
    for source in papers_by_source:
        papers_by_source[source].sort(
            key=lambda p: -(p.get('metadata', {}).get('quality_score', 1.0))
        )
    
    balanced_results = []
    
    # Phase 1: Ensure minimum representation (2 papers per source)
    min_per_source = 2
    for source, source_papers in papers_by_source.items():
        papers_to_take = min(min_per_source, len(source_papers))
        balanced_results.extend(source_papers[:papers_to_take])
    
    # Phase 2: Fill remaining slots with best papers regardless of source
    remaining_quota = max_papers - len(balanced_results)
    if remaining_quota > 0:
        # Get all remaining papers
        all_remaining = []
        for source, source_papers in papers_by_source.items():
            papers_already_taken = min(min_per_source, len(source_papers))
            all_remaining.extend(source_papers[papers_already_taken:])
        
        # Sort by combined score
        all_remaining.sort(
            key=lambda p: -(
                p.get('metadata', {}).get('quality_score', 1.0) * 
                p.get('metadata', {}).get('database_priority', 0.5)
            )
        )
        
        # Add best remaining papers
        balanced_results.extend(all_remaining[:remaining_quota])
    
    return balanced_results[:max_papers]


def improve_scihub_search_relevance(search_terms: List[str]) -> List[str]:
    """
    Improve Sci-Hub search terms for better relevance
    
    Instead of: "early neurological OR analysis OR early"
    Generate: "neurological disease gait analysis"
    """
    # Remove duplicate words and common terms
    stop_words = {'or', 'and', 'the', 'of', 'in', 'to', 'for', 'with', 'on', 'at'}
    
    # Extract meaningful terms
    meaningful_terms = []
    for term in search_terms:
        words = term.lower().split()
        filtered = [w for w in words if w not in stop_words and len(w) > 2]
        meaningful_terms.extend(filtered)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in meaningful_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    # Create focused search queries
    improved_queries = []
    
    # Strategy 1: Combine related medical/technical terms
    if len(unique_terms) >= 3:
        # Take most specific terms (usually longer words)
        specific_terms = sorted(unique_terms, key=len, reverse=True)[:3]
        improved_queries.append(' '.join(specific_terms))
    
    # Strategy 2: Create phrase search for key concepts
    if len(unique_terms) >= 2:
        improved_queries.append(f'"{unique_terms[0]} {unique_terms[1]}"')
    
    # Strategy 3: Single focused term
    if unique_terms:
        improved_queries.append(unique_terms[0])
    
    return improved_queries[:3]  # Limit to 3 focused queries


def adjust_database_priorities_for_balance():
    """
    Adjust database priorities to be more balanced
    
    Current (biased):
    - Sci-Hub: 1.0
    - ArXiv: 0.8  
    - PubMed: 0.6
    
    Balanced:
    - Sci-Hub: 0.9 (still high for full-text)
    - ArXiv: 0.85 (preprints valuable)
    - PubMed: 0.8 (peer-reviewed important)
    """
    return {
        'sci_hub': 0.9,
        'arxiv': 0.85,
        'pubmed': 0.8
    }