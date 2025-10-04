#!/usr/bin/env python3
"""
Test query optimization and demonstrate the fix
"""

import asyncio
import sys
sys.path.append('/home/gyasis/Documents/code/PromptChain/Research_agent/src')

async def test_query_strategies():
    """Test different query strategies to find effective approaches"""
    import arxiv
    from Bio import Entrez
    from Bio.Entrez import esearch
    
    # Set up PubMed
    Entrez.email = "test@example.com"
    Entrez.tool = "debug-test"
    
    # Original problematic query
    original_query = "early neurilogical diseae detection with gait analysis"
    
    # Test various query strategies
    test_queries = {
        "exact_phrase_arxiv": f'all:"{original_query}"',
        "exact_phrase_corrected": f'all:"early neurological disease detection with gait analysis"',
        "broken_down_terms": 'gait analysis neurological disease',
        "individual_terms": 'gait AND analysis AND neurological AND disease',
        "broader_gait": 'gait analysis',
        "broader_neuro": 'neurological disease',
        "machine_learning_gait": 'gait analysis machine learning',
        "parkinson_gait": 'parkinson gait analysis',
        "title_only": 'ti:"gait analysis neurological"',
        "abstract_only": 'abs:"gait analysis disease"'
    }
    
    print("=== ARXIV QUERY STRATEGY TESTING ===")
    for strategy, query in test_queries.items():
        try:
            print(f"\nStrategy: {strategy}")
            print(f"Query: {query}")
            
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = list(search.results())
            print(f"Results: {len(papers)} papers")
            
            if papers:
                print(f"  Sample: {papers[0].title[:80]}...")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test PubMed strategies
    print("\n\n=== PUBMED QUERY STRATEGY TESTING ===")
    pubmed_queries = {
        "exact_phrase": '"early neurological disease detection with gait analysis"',
        "title_abstract": '"gait analysis"[Title/Abstract] AND "neurological disease"[Title/Abstract]',
        "broad_terms": 'gait analysis neurological disease',
        "gait_only": 'gait analysis',
        "parkinson_specific": 'parkinson gait analysis',
        "mesh_terms": '"Gait"[Mesh] AND "Neurologic Manifestations"[Mesh]'
    }
    
    for strategy, query in pubmed_queries.items():
        try:
            print(f"\nStrategy: {strategy}")
            print(f"Query: {query}")
            
            search_handle = esearch(
                db='pubmed',
                term=query,
                retmax=5,
                sort='relevance'
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            pmids = search_results.get('IdList', [])
            print(f"Results: {len(pmids)} papers")
            
            if pmids:
                print(f"  PMIDs: {pmids[:3]}...")
                
        except Exception as e:
            print(f"  Error: {e}")

async def demonstrate_fix():
    """Demonstrate the fix with better query generation"""
    print("\n\n=== DEMONSTRATION OF OPTIMIZED QUERY GENERATION ===")
    
    # Original problematic input
    original_input = "early neurilogical diseae detection with gait analysis"
    print(f"Original input: '{original_input}'")
    
    # Step 1: Query preprocessing (fix typos, normalize)
    def preprocess_query(query: str) -> str:
        """Basic query preprocessing"""
        # Fix common typos
        corrections = {
            'neurilogical': 'neurological',
            'diseae': 'disease',
            'analysys': 'analysis',
            'machien': 'machine',
            'learnign': 'learning'
        }
        
        corrected = query
        for typo, fix in corrections.items():
            corrected = corrected.replace(typo, fix)
        
        return corrected.strip()
    
    corrected_query = preprocess_query(original_input)
    print(f"After preprocessing: '{corrected_query}'")
    
    # Step 2: Generate effective search variations
    def generate_search_variations(base_query: str) -> dict:
        """Generate multiple search variations for better coverage"""
        
        # Extract key terms
        terms = base_query.split()
        key_terms = [term for term in terms if len(term) > 3]  # Filter short words
        
        variations = {
            'broad_combination': ' '.join(key_terms[:4]),  # Combine main terms
            'paired_terms': f'"{key_terms[0]} {key_terms[1]}" AND "{key_terms[2]} {key_terms[3]}"' if len(key_terms) >= 4 else base_query,
            'individual_concepts': ' AND '.join(key_terms[:3]),
            'domain_specific': None  # Will be filled based on domain
        }
        
        # Add domain-specific queries
        if any(term in base_query.lower() for term in ['gait', 'walking', 'mobility']):
            variations['gait_specific'] = 'gait analysis OR walking analysis OR mobility assessment'
            
        if any(term in base_query.lower() for term in ['neurological', 'neuro', 'brain']):
            variations['neuro_specific'] = 'neurological disorders OR neurodegenerative disease OR brain disorders'
            
        if any(term in base_query.lower() for term in ['detection', 'diagnosis', 'screening']):
            variations['diagnostic'] = 'early detection OR diagnosis OR screening OR assessment'
        
        return {k: v for k, v in variations.items() if v}
    
    variations = generate_search_variations(corrected_query)
    
    print(f"\nGenerated search variations:")
    for name, query in variations.items():
        print(f"  {name}: {query}")
    
    # Step 3: Test the most promising variation
    print(f"\nTesting most promising variation...")
    
    import arxiv
    promising_query = variations.get('broad_combination', corrected_query)
    print(f"Testing: {promising_query}")
    
    try:
        search = arxiv.Search(
            query=promising_query,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = list(search.results())
        print(f"✓ Success! Found {len(papers)} papers")
        
        if papers:
            for i, paper in enumerate(papers[:3]):
                print(f"  {i+1}. {paper.title}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

async def main():
    print("Literature Search Query Optimization Analysis")
    print("=" * 60)
    
    # Test query strategies
    await test_query_strategies()
    
    # Demonstrate the fix
    success = await demonstrate_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ SOLUTION IDENTIFIED: Query optimization and preprocessing needed")
        print("\nRecommendations:")
        print("1. Add query preprocessing to fix typos")
        print("2. Generate multiple search variations instead of exact phrases")
        print("3. Use broader terms when exact phrases fail")
        print("4. Implement fallback strategies with individual terms")
    else:
        print("✗ Further investigation needed")

if __name__ == "__main__":
    asyncio.run(main())