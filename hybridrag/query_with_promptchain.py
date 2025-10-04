#!/usr/bin/env python3
"""
HybridRAG Query Interface with PromptChain
==========================================
Query interface for ingested .specstory documents using PromptChain with LightRAG retrieval tools.
Based on athena-lightrag implementation pattern.

Author: HybridRAG System
Date: 2025-10-02
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# CRITICAL FIX: Disable LiteLLM's caching to prevent event loop conflicts
import litellm
litellm.disable_cache()

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpecStoryRAG:
    """LightRAG wrapper for querying .specstory documents."""

    def __init__(self, working_dir: str = "./specstory_lightrag_db"):
        """Initialize LightRAG for .specstory documents."""
        self.working_dir = working_dir

        if not Path(working_dir).exists():
            raise FileNotFoundError(
                f"LightRAG database not found at {working_dir}. "
                "Run folder_to_lightrag.py first to ingest documents."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Initialize LightRAG
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs:
                openai_complete_if_cache(
                    "gpt-4o-mini",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    **kwargs
                ),
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-ada-002",
                    api_key=api_key
                ),
            ),
        )

        logger.info(f"Initialized SpecStoryRAG with database: {working_dir}")

    async def initialize(self):
        """Initialize LightRAG storages."""
        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    # LightRAG retrieval functions - ASYNC for proper event loop integration with PromptChain
    # PromptChain detects async functions and calls them directly without thread wrapping

    async def query_local(self, query: str, top_k: int = 60) -> str:
        """
        Query for specific entity relationships and details in the PromptChain project history.

        Best for: Finding specific implementations, code patterns, debugging sessions,
        or detailed discussions about particular features.

        Args:
            query: Search query (e.g., "How was MCP integration implemented?")
            top_k: Number of top results (default: 60)

        Returns:
            Context-focused answer with specific details
        """
        try:
            await self.initialize()
            query_param = QueryParam(mode="local")
            result = await self.rag.aquery(query, param=query_param)
            return result if result else "No relevant information found."
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Query error: {str(e)}"

    async def query_global(self, query: str, max_relation_tokens: int = 8000) -> str:
        """
        Query for high-level overviews and comprehensive summaries of the PromptChain project.

        Best for: Understanding project evolution, major architectural decisions,
        broad patterns across development history.

        Args:
            query: Search query (e.g., "What are the main components of PromptChain?")
            max_relation_tokens: Max tokens for relationships (default: 8000)

        Returns:
            High-level overview with comprehensive context
        """
        try:
            await self.initialize()
            query_param = QueryParam(mode="global")
            result = await self.rag.aquery(query, param=query_param)
            return result if result else "No relevant information found."
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Query error: {str(e)}"

    async def query_hybrid(self, query: str, max_entity_tokens: int = 6000, max_relation_tokens: int = 8000) -> str:
        """
        Query combining specific details with broader context about the PromptChain project.

        Best for: Complex questions requiring both implementation details AND
        understanding of how they fit into the larger architecture.

        Args:
            query: Search query (e.g., "Explain AgenticStepProcessor and how it integrates with PromptChain")
            max_entity_tokens: Max tokens for entities (default: 6000)
            max_relation_tokens: Max tokens for relationships (default: 8000)

        Returns:
            Balanced answer with details and context
        """
        try:
            await self.initialize()
            query_param = QueryParam(mode="hybrid")
            result = await self.rag.aquery(query, param=query_param)
            return result if result else "No relevant information found."
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Query error: {str(e)}"

    async def get_context_only(self, query: str, mode: str = "hybrid") -> str:
        """
        Retrieve raw context from the knowledge graph without LLM processing.

        Best for: Getting raw chunks of related documentation for further analysis,
        or when you want to see the actual source material.

        Args:
            query: Search query
            mode: Retrieval mode (local/global/hybrid)

        Returns:
            Raw context chunks from the knowledge graph
        """
        try:
            await self.initialize()
            query_param = QueryParam(mode=mode, only_need_context=True)
            context = await self.rag.aquery(query, param=query_param)
            return context if context else "No context found."
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return f"Context retrieval error: {str(e)}"


def create_promptchain_with_rag():
    """Create PromptChain with LightRAG retrieval tools."""

    # Initialize SpecStory RAG
    spec_rag = SpecStoryRAG()

    # Create AgenticStepProcessor for complex reasoning using ReACT method
    agentic_step = AgenticStepProcessor(
        objective="""
        Answer questions about the PromptChain project development history
        using EXHAUSTIVE multi-hop reasoning with LightRAG retrieval tools.

        CRITICAL SEARCH PRINCIPLES:
        - DO NOT give up after initial failures
        - Try AT LEAST 5-7 different search variations before concluding "no information"
        - Break compound terms into individual components
        - Use alternative phrasings, synonyms, and related concepts
        - When queries return empty results, use get_context_only to see raw documentation

        ReACT METHODOLOGY:
        1. Break down complex queries into focused sub-questions
        2. Try multiple search variations for each concept:
           - Search the exact term as asked
           - Break compound terms into individual words and search separately
           - Try alternative phrasings and synonyms
           - Search for broader category terms
           - Search for related concepts and adjacent topics
        3. Use different query modes strategically:
           - query_local: For specific entity relationships and implementation details
           - query_global: For high-level overviews and architectural patterns
           - query_hybrid: For balanced analysis combining details and context
           - get_context_only: For gathering raw documentation without LLM generation (USE THIS when other queries fail!)
        4. Accumulate contexts from multiple tool calls
        5. Look for patterns and connections across different contexts
        6. Synthesize findings into a comprehensive response

        EXHAUSTIVE SEARCH LOGIC:
        - If first search is empty, DON'T stop - the information may exist under different terminology
        - Break technical terms into their component words
        - Think of synonyms and alternative industry terms
        - Search broader categories when specific terms fail
        - Use get_context_only to inspect raw chunks
        - Be persistent - information often exists in unexpected forms

        REQUIRED MINIMUM EFFORT:
        - Original exact term query
        - Individual component word searches
        - Synonym/alternative phrasing searches
        - Broader category searches
        - Raw context inspection (get_context_only)
        TOTAL: At least 5-7 searches before concluding "no information found"
        """,
        max_internal_steps=5,  # Balanced for thorough multi-hop reasoning without excessive iterations
        model_name="openai/gpt-4o-mini",
        history_mode="progressive",  # ✨ NEW: Accumulate context across tool calls for true multi-hop reasoning
        max_context_tokens=128000 # Monitor token usage
    )

    # Create PromptChain with RAG tools
    chain = PromptChain(
        models=["openai/gpt-4o-mini"],
        instructions=[
            """
            You are an expert assistant for analyzing the PromptChain project development history.

            The user will ask questions about PromptChain development, implementations,
            architecture, debugging sessions, and project evolution.

            Use the available LightRAG retrieval tools to gather accurate information
            from the project's .specstory documentation history.

            User question: {input}
            """,
            agentic_step,
            """
            Based on the information gathered, provide a comprehensive answer to: {input}

            Include:
            - Direct answer to the question
            - Relevant context from the project history
            - Specific examples or code snippets if mentioned
            - Related discussions or decisions if relevant
            """
        ],
        verbose=True
    )

    # Register RAG tools (ASYNC - PromptChain automatically detects async functions)
    chain.register_tool_function(spec_rag.query_local)
    chain.register_tool_function(spec_rag.query_global)
    chain.register_tool_function(spec_rag.query_hybrid)
    chain.register_tool_function(spec_rag.get_context_only)

    # Add tool schemas with parameters for ReACT method
    chain.add_tools([
        {
            "type": "function",
            "function": {
                "name": "query_local",
                "description": "Query for specific entity relationships and implementation details in the PromptChain project history. Best for finding specific code patterns, debugging sessions, or detailed feature discussions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for specific details (e.g., 'How was MCP integration implemented?')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to retrieve (default: 60)",
                            "default": 60
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_global",
                "description": "Query for high-level overviews and comprehensive summaries of the PromptChain project. Best for understanding project evolution and major architectural decisions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for overviews (e.g., 'What are the main components of PromptChain?')"
                        },
                        "max_relation_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens for relationship context (default: 8000)",
                            "default": 8000
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_hybrid",
                "description": "Query combining specific details with broader context about the PromptChain project. Best for complex questions requiring both implementation details AND architectural understanding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Complex search query (e.g., 'Explain AgenticStepProcessor and how it integrates')"
                        },
                        "max_entity_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens for entity context (default: 6000)",
                            "default": 6000
                        },
                        "max_relation_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens for relationship context (default: 8000)",
                            "default": 8000
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_context_only",
                "description": "Retrieve raw context chunks from the knowledge graph without LLM processing. Best for getting actual source material for further analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for context retrieval"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["local", "global", "hybrid"],
                            "description": "Retrieval mode (default: hybrid)",
                            "default": "hybrid"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ])

    return chain


async def run_single_query(query: str):
    """Run a single query and exit."""
    print("=" * 70)
    print("📚 PromptChain Project History Query")
    print("=" * 70)
    print(f"\n🔍 Query: {query}\n")
    print("🤖 Processing with PromptChain + LightRAG...\n")

    chain = create_promptchain_with_rag()

    try:
        result = await chain.process_prompt_async(query)

        print("\n" + "=" * 70)
        print("📖 Answer:")
        print("=" * 70)
        print(result)
        print("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)


async def main():
    """Main interactive query loop."""

    print("=" * 70)
    print("📚 PromptChain Project History Query Interface")
    print("=" * 70)
    print("\nQuerying .specstory documentation using PromptChain + LightRAG")
    print("\nCommands:")
    print("  - Ask any question about PromptChain development")
    print("  - Type 'exit' to quit")
    print("=" * 70)

    # Create PromptChain with RAG tools
    chain = create_promptchain_with_rag()

    while True:
        print("\n" + "=" * 70)
        user_query = input("\n🔍 Your question: ").strip()

        if user_query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break

        if not user_query:
            continue

        try:
            print("\n🤖 Processing with PromptChain + LightRAG...\n")

            result = await chain.process_prompt_async(user_query)

            print("\n" + "=" * 70)
            print("📖 Answer:")
            print("=" * 70)
            print(result)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Check if query was provided as command-line argument
    if len(sys.argv) > 1:
        # Single query mode: python query_with_promptchain.py 'your question here'
        query = ' '.join(sys.argv[1:])
        asyncio.run(run_single_query(query))
    else:
        # Interactive mode
        asyncio.run(main())
