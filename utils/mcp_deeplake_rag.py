"""
RAG (Retrieval Augmented Generation) utilities for the Chain of Draft system.
This module provides the core RAG functionality using DeepLake for vector storage and retrieval.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from .customdeeplake import VectorSearchV4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize vector search with default dataset
DATASET_PATH = os.getenv('DEEPLAKE_DATASET_PATH', '/media/gyasis/Drive 2/Deeplake_Storage/memory_lane_v4')
vector_search = None

def init_vector_search(dataset_path: Optional[str] = None) -> None:
    """Initialize the vector search instance."""
    global vector_search
    try:
        vector_search = VectorSearchV4(dataset_path or DATASET_PATH)
        logger.info(f"Initialized vector search with dataset: {dataset_path or DATASET_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize vector search: {str(e)}")
        raise

def get_vector_search() -> VectorSearchV4:
    """Get or initialize the vector search instance."""
    global vector_search
    if vector_search is None:
        init_vector_search()
    return vector_search

def retrieve_context(query: str, n_results: int = 5) -> Dict:
    """
    Retrieve relevant context for a query using DeepLake RAG.
    
    Args:
        query (str): The search query
        n_results (int, optional): Number of results to return. Defaults to 5.
    
    Returns:
        dict: Full formatted results including documents, scores, and metadata.
    """
    try:
        vs = get_vector_search()
        results = vs.search(query, n_results=n_results, return_text_only=False)
        
        return {
            'documents': [r['text'] for r in results],
            'metadata': [r['metadata'] for r in results],
            'ids': [r['id'] for r in results]
        }
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return {'documents': [], 'metadata': [], 'ids': []}

def get_summary(query: str, n_results: int = 3) -> str:
    """
    Get just the text content from the top matching documents.
    
    Args:
        query (str): The search query
        n_results (int, optional): Number of results to return. Defaults to 3.
    
    Returns:
        str: Concatenated text from the top matching documents.
    """
    try:
        vs = get_vector_search()
        return vs.search(query, n_results=n_results, return_text_only=True)
    except Exception as e:
        logger.error(f"Error getting summary: {str(e)}")
        return ""

if __name__ == "__main__":
    # Example usage
    query = "What is the role of mitochondria in cellular respiration?"
    print("\nTesting retrieve_context:")
    results = retrieve_context(query)
    print(f"Found {len(results['documents'])} documents")
    
    print("\nTesting get_summary:")
    summary = get_summary(query)
    print(f"Summary length: {len(summary)} characters") 