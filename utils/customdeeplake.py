"""
VectorSearchV4: A modern wrapper class for DeepLake vector search operations with OpenAI embeddings.

This module provides a clean interface for searching DeepLake vector stores,
optimized for version 4+ of DeepLake.
"""

import os
import logging
import deeplake
from typing import List, Dict, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/gyasis/Documents/code/hello-World/.env",override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorSearchV4:
    def __init__(self, dataset_path: str, client: Optional[OpenAI] = None):
        """
        Initialize the VectorSearchV4 class.

        Parameters:
        - dataset_path: Path to the V4 dataset
        - client: OpenAI client for generating embeddings. If None, will create new client.
        """
        self.dataset_path = dataset_path
        self.client = client if client else OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        logger.info(f"Opening dataset at path: {dataset_path}")
        try:
            self.ds = deeplake.open(dataset_path)
            logger.info("Dataset opened successfully")
        except Exception as e:
            logger.error(f"Error opening dataset: {str(e)}")
            raise

    def embedding_function(self, texts: Union[str, List[str]], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Generate embeddings for the given texts using the specified model.
        
        Parameters:
        - texts: Single string or list of strings to embed
        - model: OpenAI model to use for embeddings
        
        Returns:
        - List of embeddings
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            texts = [t.replace("\n", " ") for t in texts]
            
            embeddings = [
                data.embedding
                for data in self.client.embeddings.create(input=texts, model=model).data
            ]
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        return_text_only: bool = False,
    ) -> Union[str, List[Dict]]:
        """
        Perform a vector similarity search on the V4 dataset.

        Parameters:
        - query: The text to search for similar entries
        - n_results: Number of results to return
        - return_text_only: If True, return only the text of the results

        Returns:
        - If return_text_only: A compiled string of texts
        - Otherwise: List of dicts with full result information
        """
        try:
            logger.info(f"Searching for: {query}")
            
            # Generate the embedding for the query text
            query_embedding = self.embedding_function(query)[0]
            
            # Convert embedding to string format for query
            text_vector = ','.join(str(x) for x in query_embedding)

            # Perform the vector similarity search
            similar = self.ds.query(f"""
                SELECT *
                ORDER BY COSINE_SIMILARITY(embedding, ARRAY[{text_vector}]) DESC
                LIMIT {n_results}
            """)
            
            # Process results based on return type
            if return_text_only:
                compiled_text = "\n\n---\n\n".join(item["text"] for item in similar)
                return compiled_text
            else:
                # Get all relevant fields
                ids = similar["id"][:]
                texts = similar["text"][:]
                embeddings = similar["embedding"][:]
                metadata = similar["metadata"][:]

                # Combine results into a list of dictionaries
                results = []
                for id_, text, embedding, meta in zip(ids, texts, embeddings, metadata):
                    results.append({
                        "id": id_,
                        "text": text,
                        "embedding": embedding,
                        "metadata": meta,
                    })
                
                logger.info(f"Found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    try:
        dataset_path = "/media/gyasis/Drive 2/Deeplake_Storage/Medicine_v4"
        vector_search = VectorSearchV4(dataset_path)
        
        while True:
            query = input("\nEnter your search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                print("Exiting search...")
                break
                
            results = vector_search.search(query, n_results=5, return_text_only=False)
            
            print("\nSearch Results:")
            print("-" * 50)
            for result in results:
                print(f"\nID: {result['id']}")
                print(f"Text: {result['text']}")
                print(f"Metadata: {result['metadata']}")
                print("-" * 50)
                
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")