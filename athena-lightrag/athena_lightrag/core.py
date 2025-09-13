#!/usr/bin/env python3
"""
Athena LightRAG Core Functions
==============================
Function-based interface for LightRAG queries with multi-hop reasoning support.

This module transforms the interactive CLI interface into reusable functions
suitable for MCP server integration.

Author: PromptChain Team
Date: 2025
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv

# Import PromptChain components for multi-hop reasoning
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass 
class QueryResult:
    """Structured result from LightRAG query."""
    result: str
    query_mode: str
    context_only: bool
    reasoning_steps: Optional[List[str]] = None
    accumulated_context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AthenaLightRAG:
    """
    Function-based LightRAG interface with multi-hop reasoning capabilities.
    
    This class provides the core functionality that was previously in the 
    interactive CLI, but structured as reusable async functions suitable
    for MCP server integration.
    """
    
    def __init__(
        self, 
        working_dir: str = "./athena_lightrag_db",
        api_key: Optional[str] = None,
        reasoning_model: str = "gpt-4.1-mini",
        max_reasoning_steps: int = 5
    ):
        """
        Initialize the Athena LightRAG interface.
        
        Args:
            working_dir: Path to LightRAG database
            api_key: OpenAI API key (defaults to env var)
            reasoning_model: Model for multi-hop reasoning
            max_reasoning_steps: Maximum steps for reasoning chains
        """
        self.working_dir = working_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.reasoning_model = reasoning_model
        self.max_reasoning_steps = max_reasoning_steps
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Check if database exists
        if not Path(working_dir).exists():
            raise FileNotFoundError(f"LightRAG database not found at {working_dir}. Run ingestion first.")
        
        self._init_lightrag()
        self.rag_initialized = False
    
    def _init_lightrag(self):
        """Initialize LightRAG instance."""
        logger.info(f"Loading LightRAG database from {self.working_dir}")
        
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: 
                openai_complete_if_cache(
                    "gpt-4.1-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    **kwargs
                ),
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-ada-002",
                    api_key=self.api_key
                ),
            ),
        )
        logger.info("LightRAG loaded successfully")
    
    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized."""
        if not self.rag_initialized:
            logger.info("Initializing LightRAG storages for first query...")
            await self.rag.initialize_storages()
            self.rag_initialized = True
            logger.info("LightRAG storages initialized successfully")
    
    async def basic_query(
        self, 
        query_text: str, 
        mode: str = "hybrid",
        only_need_context: bool = False,
        top_k: int = 60,
        max_entity_tokens: int = 6000,
        max_relation_tokens: int = 8000
    ) -> QueryResult:
        """
        Execute a basic query against the LightRAG database.
        
        Args:
            query_text: The query string
            mode: Query mode (local, global, hybrid, naive)
            only_need_context: If True, only return context without generating response
            top_k: Number of top items to retrieve
            max_entity_tokens: Maximum tokens for entity context
            max_relation_tokens: Maximum tokens for relationship context
            
        Returns:
            QueryResult object with structured results
        """
        await self._ensure_initialized()
        
        # Validate and normalize mode
        valid_modes = ["local", "global", "hybrid", "naive"]
        if mode not in valid_modes:
            logger.warning(f"Invalid mode '{mode}', defaulting to 'hybrid'")
            mode = "hybrid"
        
        # Create QueryParam object
        query_param = QueryParam(
            mode=mode,
            only_need_context=only_need_context,
            top_k=top_k,
            max_entity_tokens=max_entity_tokens,
            max_relation_tokens=max_relation_tokens,
            response_type="Multiple Paragraphs"
        )
        
        try:
            result = await self.rag.aquery(query_text, param=query_param)
            
            return QueryResult(
                result=result or "",
                query_mode=mode,
                context_only=only_need_context,
                metadata={
                    "top_k": top_k,
                    "max_entity_tokens": max_entity_tokens,
                    "max_relation_tokens": max_relation_tokens,
                    "query_text": query_text
                }
            )
        except Exception as e:
            logger.error(f"Basic query failed: {e}")
            raise
    
    async def multi_hop_reasoning_query(
        self,
        initial_query: str,
        context_accumulation_strategy: str = "incremental",
        reasoning_objective: Optional[str] = None,
        mode: str = "hybrid",
        top_k: int = 60
    ) -> QueryResult:
        """
        Execute a multi-hop reasoning query using AgenticStepProcessor.
        
        This method implements intelligent reasoning chains that can:
        1. Break down complex questions into sub-queries
        2. Accumulate context from multiple retrieval steps
        3. Synthesize final comprehensive answers
        
        Args:
            initial_query: The initial complex query
            context_accumulation_strategy: How to accumulate context ('incremental', 'comprehensive', 'focused')
            reasoning_objective: Specific objective for the reasoning process
            mode: LightRAG query mode
            top_k: Number of retrieval results per step
            
        Returns:
            QueryResult with reasoning steps and accumulated context
        """
        await self._ensure_initialized()
        
        # Set up the reasoning objective
        if reasoning_objective is None:
            reasoning_objective = f"""
            Answer the complex query: "{initial_query}"
            
            Use multi-hop reasoning to:
            1. Break down the question into logical sub-queries
            2. Query the LightRAG database for each sub-component
            3. Accumulate and synthesize information across retrievals
            4. Provide a comprehensive final answer
            
            Context accumulation strategy: {context_accumulation_strategy}
            """
        
        # Create the agentic step processor for multi-hop reasoning
        agentic_processor = AgenticStepProcessor(
            objective=reasoning_objective,
            max_internal_steps=self.max_reasoning_steps,
            model_name=self.reasoning_model
        )
        
        # Create a PromptChain for the reasoning workflow
        reasoning_chain = PromptChain(
            models=[self.reasoning_model],
            instructions=[
                f"Initialize multi-hop reasoning for query: {initial_query}",
                agentic_processor,  # This handles the complex reasoning
                "Synthesize final comprehensive answer: {input}"
            ],
            store_steps=True,
            verbose=False  # Set to False for MCP protocol compliance
        )
        
        # Create advanced tool function with real LightRAG integration
        def query_lightrag_tool(query: str) -> str:
            """
            Advanced tool function to query LightRAG database with real integration.
            This performs actual queries and provides enhanced context for reasoning.
            """
            try:
                # Perform real LightRAG query using sync wrapper
                def _sync_query():
                    async def _async_query():
                        await self._ensure_initialized()
                        
                        query_param = QueryParam(
                            mode=mode,
                            only_need_context=True,  # Get context for reasoning
                            top_k=top_k,
                            max_entity_tokens=4000,
                            max_relation_tokens=6000,
                            response_type="Multiple Paragraphs"
                        )
                        
                        result = await self.rag.aquery(query, param=query_param)
                        return result or "No relevant context found."
                    
                    # Execute in new event loop to avoid conflicts
                    return asyncio.run(_async_query())
                
                # Execute the query
                context_result = _sync_query()
                
                logger.info(f"LightRAG tool query successful: {len(context_result)} characters retrieved")
                
                return f"""
[LightRAG Query Result]
Query: {query}
Mode: {mode}
Context Retrieved ({len(context_result)} chars):
{context_result[:1000]}{'...' if len(context_result) > 1000 else ''}

[Real database context provided for multi-hop reasoning]
                """
            except Exception as e:
                logger.error(f"LightRAG tool query failed: {e}")
                return f"LightRAG query failed: {str(e)}. Using fallback context retrieval."
        
        # Register the query function as a tool for the reasoning chain
        reasoning_chain.register_tool_function(query_lightrag_tool)
        
        # Add the query tool schema
        reasoning_chain.add_tools([{
            "type": "function",
            "function": {
                "name": "query_lightrag",
                "description": f"Query the Athena medical database using LightRAG. Mode: {mode}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query to execute against the database"}
                    },
                    "required": ["query"]
                }
            }
        }])
        
        try:
            # Execute the multi-hop reasoning using async method
            logger.info(f"Starting multi-hop reasoning for query: {initial_query}")
            final_result = await reasoning_chain.process_prompt_async(initial_query)
            
            # Extract reasoning steps from the agentic processor
            reasoning_steps = []
            accumulated_context = ""
            
            if reasoning_chain.step_outputs:
                reasoning_steps = [step for step in reasoning_chain.step_outputs if step]
                accumulated_context = " ".join(reasoning_steps)
            
            return QueryResult(
                result=final_result,
                query_mode=mode,
                context_only=False,
                reasoning_steps=reasoning_steps,
                accumulated_context=accumulated_context,
                metadata={
                    "initial_query": initial_query,
                    "context_accumulation_strategy": context_accumulation_strategy,
                    "reasoning_model": self.reasoning_model,
                    "max_reasoning_steps": self.max_reasoning_steps,
                    "top_k": top_k,
                    "total_reasoning_steps": len(reasoning_steps)
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the LightRAG database.
        
        Returns:
            Dictionary with database metadata and statistics
        """
        await self._ensure_initialized()
        
        db_path = Path(self.working_dir)
        
        info = {
            "database_path": str(db_path),
            "database_exists": db_path.exists(),
            "initialized": self.rag_initialized
        }
        
        if db_path.exists():
            # Get basic file system info
            info["database_files"] = [f.name for f in db_path.iterdir() if f.is_file()]
            
            # Try to get some basic stats if possible
            try:
                # This would depend on LightRAG's internal structure
                # For now, just provide basic filesystem info
                total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
                info["total_size_bytes"] = total_size
                info["total_files"] = len(list(db_path.rglob('*')))
            except Exception as e:
                info["stats_error"] = str(e)
        
        return info


# Global instance for MCP server
_athena_instance: Optional[AthenaLightRAG] = None


def get_athena_instance() -> AthenaLightRAG:
    """Get or create the global Athena LightRAG instance."""
    global _athena_instance
    if _athena_instance is None:
        _athena_instance = AthenaLightRAG()
    return _athena_instance


# Convenience functions for MCP server integration
async def query_athena_basic(
    query: str,
    mode: str = "hybrid",
    context_only: bool = False,
    top_k: int = 60
) -> str:
    """
    Convenience function for basic Athena queries.
    
    Args:
        query: The query string
        mode: Query mode (local, global, hybrid, naive)  
        context_only: If True, only return context
        top_k: Number of results to retrieve
        
    Returns:
        Query result as string
    """
    athena = get_athena_instance()
    result = await athena.basic_query(
        query_text=query,
        mode=mode,
        only_need_context=context_only,
        top_k=top_k
    )
    return result.result


async def query_athena_multi_hop(
    query: str,
    context_strategy: str = "incremental",
    mode: str = "hybrid",
    max_steps: int = 5
) -> str:
    """
    Convenience function for multi-hop reasoning queries.
    
    Args:
        query: The complex query requiring multi-hop reasoning
        context_strategy: Context accumulation strategy
        mode: LightRAG query mode
        max_steps: Maximum reasoning steps
        
    Returns:
        Final reasoning result as string
    """
    athena = get_athena_instance()
    athena.max_reasoning_steps = max_steps
    
    result = await athena.multi_hop_reasoning_query(
        initial_query=query,
        context_accumulation_strategy=context_strategy,
        mode=mode
    )
    return result.result


async def get_athena_database_info() -> str:
    """
    Get Athena database information.
    
    Returns:
        Database info as formatted string
    """
    athena = get_athena_instance()
    info = await athena.get_database_info()
    
    # Format the info nicely
    formatted_info = []
    formatted_info.append(f"Database Path: {info['database_path']}")
    formatted_info.append(f"Database Exists: {info['database_exists']}")
    formatted_info.append(f"Initialized: {info['initialized']}")
    
    if info.get('total_size_bytes'):
        size_mb = info['total_size_bytes'] / (1024 * 1024)
        formatted_info.append(f"Total Size: {size_mb:.1f} MB")
        formatted_info.append(f"Total Files: {info['total_files']}")
    
    if info.get('database_files'):
        formatted_info.append(f"Database Files: {', '.join(info['database_files'])}")
    
    return "\n".join(formatted_info)