#!/usr/bin/env python3
"""
Athena LightRAG Core Module
===========================
Function-based LightRAG interface using validated patterns from Context7 documentation.
Implements QueryParam-based query system with all supported modes.

Author: Athena LightRAG System  
Date: 2025-09-08
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Literal, Union, Any
from dataclasses import dataclass
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
# CRITICAL: Load environment variables FIRST before any other imports
from dotenv import load_dotenv
import os

load_dotenv(override=True)  # Override system env vars with project .env

# Force the correct API key to prevent contamination from imported libraries
project_api_key = os.getenv('OPENAI_API_KEY')
if project_api_key:
    os.environ['OPENAI_API_KEY'] = project_api_key

from lightrag.utils import EmbeddingFunc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions based on validated patterns
QueryMode = Literal["local", "global", "hybrid", "naive", "mix", "bypass"]
ResponseType = Literal["Multiple Paragraphs", "Single Paragraph", "Bullet Points"]

@dataclass
class LightRAGConfig:
    """Configuration for LightRAG instance."""
    working_dir: str = "./athena_lightrag_db"
    api_key: Optional[str] = None
    model_name: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    max_async: int = 4
    enable_cache: bool = True

@dataclass
class QueryResult:
    """Structured result from LightRAG query."""
    result: str
    mode: QueryMode
    context_only: bool
    tokens_used: Dict[str, int]
    execution_time: float
    error: Optional[str] = None

class AthenaLightRAGCore:
    """
    Core LightRAG interface using function-based architecture.
    Implements validated QueryParam patterns from Context7 documentation.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None):
        """
        Initialize Athena LightRAG Core.
        
        Args:
            config: LightRAG configuration object
        """
        self.config = config or LightRAGConfig()
        self._setup_api_key()
        self._validate_database()
        self._init_lightrag()
        self.rag_initialized = False
        self._initialization_in_progress = False
        self.context_cache: Dict[str, str] = {}
    
    def _setup_api_key(self):
        """Setup OpenAI API key from config or environment."""
        self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide it in LightRAGConfig."
            )
    
    def _validate_database(self):
        """Validate that LightRAG database exists."""
        if not Path(self.config.working_dir).exists():
            raise FileNotFoundError(
                f"LightRAG database not found at {self.config.working_dir}. "
                "Run ingestion first with deeplake_to_lightrag.py"
            )
        logger.info(f"Validated LightRAG database at {self.config.working_dir}")
    
    def _init_lightrag(self):
        """Initialize LightRAG instance with validated patterns."""
        logger.info(f"Initializing LightRAG with working directory: {self.config.working_dir}")
        
        # LLM model function using validated openai_complete_if_cache pattern
        def llm_model_func(
            prompt: str, 
            system_prompt: Optional[str] = None, 
            history_messages: List[Dict[str, str]] = None,
            **kwargs
        ) -> str:
            return openai_complete_if_cache(
                model=self.config.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.api_key,
                **kwargs
            )
        
        # Embedding function using validated openai_embed pattern
        def embedding_func(texts: List[str]) -> List[List[float]]:
            return openai_embed(
                texts=texts,
                model=self.config.embedding_model,
                api_key=self.api_key
            )
        
        # Initialize LightRAG with validated configuration
        self.rag = LightRAG(
            working_dir=self.config.working_dir,
            llm_model_func=llm_model_func,
            llm_model_max_async=self.config.max_async,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.config.embedding_dim,
                func=embedding_func
            ),
        )
        logger.info("LightRAG initialized successfully")
    
    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized (validated pattern)."""
        if not self.rag_initialized:
            logger.info("Initializing LightRAG storages for first query...")
            await self.rag.initialize_storages()
            self.rag_initialized = True
            logger.info("LightRAG storages initialized successfully")
    
    def _create_query_param(
        self,
        mode: QueryMode = "hybrid",
        only_need_context: bool = False,
        only_need_prompt: bool = False,
        response_type: ResponseType = "Multiple Paragraphs",
        top_k: int = 60,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000,
        max_total_tokens: int = 30000,
        **kwargs
    ) -> QueryParam:
        """
        Create QueryParam using validated patterns from Context7.
        
        Args:
            mode: Query mode (local, global, hybrid, naive, mix, bypass)
            only_need_context: If True, only return context without generating response
            only_need_prompt: If True, only return the generated prompt
            response_type: Format of the response
            top_k: Number of top items to retrieve
            max_entity_tokens: Maximum tokens for entity context
            max_relation_tokens: Maximum tokens for relationship context
            max_total_tokens: Maximum total tokens budget
            **kwargs: Additional QueryParam parameters
            
        Returns:
            Configured QueryParam object
        """
        return QueryParam(
            mode=mode,
            only_need_context=only_need_context,
            only_need_prompt=only_need_prompt,
            response_type=response_type,
            top_k=top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            max_total_tokens=max_total_tokens,
            **kwargs
        )
    
    async def query_async(
        self,
        query_text: str,
        mode: QueryMode = "hybrid",
        only_need_context: bool = False,
        **query_params
    ) -> QueryResult:
        """
        Execute async query using validated LightRAG patterns.
        
        Args:
            query_text: The query string
            mode: Query mode (local, global, hybrid, naive, mix, bypass)
            only_need_context: If True, only return context without response
            **query_params: Additional QueryParam parameters
            
        Returns:
            QueryResult with structured response
        """
        import time
        start_time = time.time()
        
        try:
            # Ensure storages are initialized
            await self._ensure_initialized()
            
            # Create QueryParam using validated pattern
            query_param = self._create_query_param(
                mode=mode,
                only_need_context=only_need_context,
                **query_params
            )
            
            # Execute query using validated rag.aquery pattern
            result = await self.rag.aquery(query_text, param=query_param)
            
            # Cache context if context-only mode
            if only_need_context:
                self.context_cache[f"{mode}_{query_text[:50]}"] = result
            
            execution_time = time.time() - start_time
            
            return QueryResult(
                result=result,
                mode=mode,
                context_only=only_need_context,
                tokens_used={"estimated": len(str(result)) // 4},  # Rough estimate
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query failed: {e}")
            return QueryResult(
                result="",
                mode=mode,
                context_only=only_need_context,
                tokens_used={"estimated": 0},
                execution_time=execution_time,
                error=str(e)
            )
    
    def query(
        self,
        query_text: str,
        mode: QueryMode = "hybrid",
        only_need_context: bool = False,
        **query_params
    ) -> QueryResult:
        """
        Execute synchronous query (wrapper around async method).
        
        Args:
            query_text: The query string
            mode: Query mode
            only_need_context: If True, only return context
            **query_params: Additional QueryParam parameters
            
        Returns:
            QueryResult with structured response
        """
        return asyncio.run(self.query_async(
            query_text=query_text,
            mode=mode,
            only_need_context=only_need_context,
            **query_params
        ))
    
    # Function-based query methods for each mode (validated patterns)
    async def query_local_async(self, query_text: str, **kwargs) -> QueryResult:
        """Query in local mode (context-dependent information)."""
        return await self.query_async(query_text, mode="local", **kwargs)
    
    async def query_global_async(self, query_text: str, **kwargs) -> QueryResult:
        """Query in global mode (high-level overview and summaries)."""
        return await self.query_async(query_text, mode="global", **kwargs)
    
    async def query_hybrid_async(self, query_text: str, **kwargs) -> QueryResult:
        """Query in hybrid mode (combination of local and global)."""
        return await self.query_async(query_text, mode="hybrid", **kwargs)
    
    async def query_naive_async(self, query_text: str, **kwargs) -> QueryResult:
        """Query in naive mode (basic search without advanced techniques)."""
        return await self.query_async(query_text, mode="naive", **kwargs)
    
    async def query_mix_async(self, query_text: str, **kwargs) -> QueryResult:
        """Query in mix mode (integrates knowledge graph and vector retrieval)."""
        return await self.query_async(query_text, mode="mix", **kwargs)
    
    async def get_context_only_async(
        self, 
        query_text: str, 
        mode: QueryMode = "hybrid",
        **kwargs
    ) -> str:
        """
        Get only context without generating response (validated pattern).
        
        Args:
            query_text: The query string
            mode: Query mode for context extraction
            **kwargs: Additional QueryParam parameters
            
        Returns:
            Context string
        """
        result = await self.query_async(
            query_text=query_text,
            mode=mode,
            only_need_context=True,
            **kwargs
        )
        return result.result
    
    # Synchronous wrappers
    def query_local(self, query_text: str, **kwargs) -> QueryResult:
        """Synchronous local mode query."""
        return asyncio.run(self.query_local_async(query_text, **kwargs))
    
    def query_global(self, query_text: str, **kwargs) -> QueryResult:
        """Synchronous global mode query."""
        return asyncio.run(self.query_global_async(query_text, **kwargs))
    
    def query_hybrid(self, query_text: str, **kwargs) -> QueryResult:
        """Synchronous hybrid mode query."""
        return asyncio.run(self.query_hybrid_async(query_text, **kwargs))
    
    def query_naive(self, query_text: str, **kwargs) -> QueryResult:
        """Synchronous naive mode query."""
        return asyncio.run(self.query_naive_async(query_text, **kwargs))
    
    def query_mix(self, query_text: str, **kwargs) -> QueryResult:
        """Synchronous mix mode query."""
        return asyncio.run(self.query_mix_async(query_text, **kwargs))
    
    def get_context_only(
        self, 
        query_text: str, 
        mode: QueryMode = "hybrid",
        **kwargs
    ) -> str:
        """Synchronous context-only extraction."""
        return asyncio.run(self.get_context_only_async(query_text, mode, **kwargs))
    
    def get_available_modes(self) -> List[QueryMode]:
        """Get list of available query modes."""
        return ["local", "global", "hybrid", "naive", "mix", "bypass"]
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get database status information."""
        db_path = Path(self.config.working_dir)
        
        status = {
            "database_path": str(db_path),
            "exists": db_path.exists(),
            "initialized": self.rag_initialized,
            "initialization_in_progress": self._initialization_in_progress,
            "thread_safety": "enabled",
            "available_modes": self.get_available_modes(),
            "config": {
                "model_name": self.config.model_name,
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "max_async": self.config.max_async
            }
        }
        
        if db_path.exists():
            # Get file sizes for storage components
            storage_files = [
                "kv_store_full_entities.json",
                "kv_store_full_relations.json", 
                "kv_store_text_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
                "vdb_chunks.json"
            ]
            
            storage_info = {}
            for file_name in storage_files:
                file_path = db_path / file_name
                if file_path.exists():
                    storage_info[file_name] = {
                        "size_bytes": file_path.stat().st_size,
                        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
                    }
            
            status["storage_info"] = storage_info
        
        return status


# Factory function for easy instantiation
def create_athena_lightrag(
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db",
    **config_kwargs
) -> AthenaLightRAGCore:
    """
    Factory function to create AthenaLightRAGCore instance.
    
    Args:
        working_dir: Path to LightRAG database
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured AthenaLightRAGCore instance
    """
    config = LightRAGConfig(working_dir=working_dir, **config_kwargs)
    return AthenaLightRAGCore(config)


# Example usage and testing functions
async def main():
    """Example usage of the function-based LightRAG interface."""
    try:
        # Create LightRAG instance
        lightrag = create_athena_lightrag()
        
        # Test database status
        status = lightrag.get_database_status()
        print(f"Database Status: {status}")
        
        # Test query in different modes
        test_query = "What tables are related to patient appointments?"
        
        print(f"\n=== Testing Query: {test_query} ===")
        
        # Test hybrid mode (default)
        print("\n--- Hybrid Mode ---")
        result = await lightrag.query_hybrid_async(test_query)
        print(f"Result: {result.result[:200]}...")
        print(f"Execution time: {result.execution_time:.2f}s")
        
        # Test context-only extraction
        print("\n--- Context Only ---")
        context = await lightrag.get_context_only_async(test_query, mode="local")
        print(f"Context: {context[:200]}...")
        
        # Test local mode
        print("\n--- Local Mode ---")
        local_result = await lightrag.query_local_async(test_query)
        print(f"Local Result: {local_result.result[:200]}...")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())