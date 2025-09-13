#!/usr/bin/env python3
"""
Agentic LightRAG Integration
============================
AgenticStepProcessor integration with LightRAG tools for multi-hop reasoning.
Implements validated patterns for complex reasoning workflows.

Author: Athena LightRAG System
Date: 2025-09-08
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import sys
import os
from contextlib import asynccontextmanager
import time
from functools import wraps

# Add parent directories to path for imports
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append('/home/gyasis/Documents/code/PromptChain/athena-lightrag')

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain import PromptChain
from lightrag_core import AthenaLightRAGCore, QueryResult, QueryMode, create_athena_lightrag

logger = logging.getLogger(__name__)

@dataclass
class MultiHopContext:
    """Context accumulation for multi-hop reasoning."""
    initial_query: str
    contexts: List[Dict[str, Any]]
    reasoning_steps: List[str]
    final_synthesis: Optional[str] = None
    total_tokens_used: int = 0
    execution_time: float = 0.0

class LightRAGToolsProvider:
    """Provides LightRAG tools for AgenticStepProcessor."""
    
    def __init__(self, lightrag_core: AthenaLightRAGCore):
        """
        Initialize with LightRAG core instance.
        
        Args:
            lightrag_core: AthenaLightRAGCore instance
        """
        self.lightrag = lightrag_core
        self.context_accumulator = MultiHopContext("", [], [])
    
    def create_lightrag_tools(self) -> List[Dict[str, Any]]:
        """
        Create LightRAG tool definitions for AgenticStepProcessor.
        Follows validated patterns from Context7 documentation.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "lightrag_local_query",
                    "description": "Query LightRAG in local mode for context-dependent information focusing on specific entity relationships",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for local context"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top entities to retrieve (default: 60)",
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
                    "name": "lightrag_global_query",
                    "description": "Query LightRAG in global mode for high-level overviews and summaries across the entire knowledge graph",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for global overview"
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
                    "name": "lightrag_hybrid_query", 
                    "description": "Query LightRAG in hybrid mode combining local entity focus with global relationship patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for hybrid analysis"
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
                    "name": "lightrag_context_extract",
                    "description": "Extract only context from LightRAG without generating a response, useful for gathering information for further processing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for context extraction"
                            },
                            "mode": {
                                "type": "string", 
                                "enum": ["local", "global", "hybrid", "naive", "mix"],
                                "description": "Query mode for context extraction (default: hybrid)",
                                "default": "hybrid"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lightrag_mix_query",
                    "description": "Query LightRAG in mix mode integrating knowledge graph and vector retrieval for comprehensive analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for mixed retrieval"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top items to retrieve (default: 60)",
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
                    "name": "accumulate_context",
                    "description": "Add information to the multi-hop context accumulator for building comprehensive understanding",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "context_type": {
                                "type": "string",
                                "enum": ["local", "global", "hybrid", "mix", "synthesis"],
                                "description": "Type of context being accumulated"
                            },
                            "context_data": {
                                "type": "string",
                                "description": "The context data to accumulate"
                            },
                            "reasoning_step": {
                                "type": "string",
                                "description": "Description of the reasoning step"
                            }
                        },
                        "required": ["context_type", "context_data", "reasoning_step"]
                    }
                }
            }
        ]
    
    def register_tool_functions(self, prompt_chain: PromptChain):
        """
        Register LightRAG tool functions with PromptChain.
        
        Args:
            prompt_chain: PromptChain instance to register tools with
        """
        # Register each tool function (now async-safe to avoid nested asyncio.run() deadlocks)
        prompt_chain.register_tool_function(self.lightrag_local_query)
        prompt_chain.register_tool_function(self.lightrag_global_query)
        prompt_chain.register_tool_function(self.lightrag_hybrid_query)
        prompt_chain.register_tool_function(self.lightrag_context_extract)
        prompt_chain.register_tool_function(self.lightrag_mix_query)
        prompt_chain.register_tool_function(self.accumulate_context)
    
    # Tool function implementations - Async-safe to avoid nested asyncio.run() deadlocks
    async def lightrag_local_query(self, query: str, top_k: int = 60) -> str:
        """Execute local mode query focusing on specific entity relationships."""
        try:
            # Use async version directly to avoid nested asyncio.run() deadlocks
            result = await self.lightrag.query_local_async(query, top_k=top_k)
            if result.error:
                return f"Error in local query: {result.error}"
            return result.result
        except Exception as e:
            logger.error(f"Local query failed: {e}")
            return f"Local query failed: {str(e)}"
    
    async def lightrag_global_query(self, query: str, max_relation_tokens: int = 8000) -> str:
        """Execute global mode query for high-level overviews."""
        try:
            # Use async version directly to avoid nested asyncio.run() deadlocks
            result = await self.lightrag.query_global_async(
                query, 
                max_relation_tokens=max_relation_tokens
            )
            if result.error:
                return f"Error in global query: {result.error}"
            return result.result
        except Exception as e:
            logger.error(f"Global query failed: {e}")
            return f"Global query failed: {str(e)}"
    
    async def lightrag_hybrid_query(
        self, 
        query: str, 
        max_entity_tokens: int = 6000, 
        max_relation_tokens: int = 8000
    ) -> str:
        """Execute hybrid mode query combining local and global approaches."""
        try:
            # Use async version directly to avoid nested asyncio.run() deadlocks
            result = await self.lightrag.query_hybrid_async(
                query,
                max_entity_tokens=max_entity_tokens,
                max_relation_tokens=max_relation_tokens
            )
            if result.error:
                return f"Error in hybrid query: {result.error}"
            return result.result
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            return f"Hybrid query failed: {str(e)}"
    
    async def lightrag_context_extract(self, query: str, mode: str = "hybrid") -> str:
        """Extract only context without generating response."""
        try:
            # Validate mode
            valid_modes = ["local", "global", "hybrid", "naive", "mix"]
            if mode not in valid_modes:
                return f"Invalid mode '{mode}'. Must be one of: {valid_modes}"
            
            # Use async version directly to avoid nested asyncio.run() deadlocks
            context = await self.lightrag.get_context_only_async(query, mode=mode)
            return context
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            return f"Context extraction failed: {str(e)}"
    
    async def lightrag_mix_query(self, query: str, top_k: int = 60) -> str:
        """Execute mix mode query integrating knowledge graph and vector retrieval."""
        try:
            # Use async version directly to avoid nested asyncio.run() deadlocks
            result = await self.lightrag.query_mix_async(query, top_k=top_k)
            if result.error:
                return f"Error in mix query: {result.error}"
            return result.result
        except Exception as e:
            logger.error(f"Mix query failed: {e}")
            return f"Mix query failed: {str(e)}"
    
    def accumulate_context(
        self, 
        context_type: str, 
        context_data: str, 
        reasoning_step: str
    ) -> str:
        """Accumulate context for multi-hop reasoning."""
        try:
            context_entry = {
                "type": context_type,
                "data": context_data,
                "step": reasoning_step,
                "tokens": len(context_data) // 4  # Rough estimate
            }
            
            self.context_accumulator.contexts.append(context_entry)
            self.context_accumulator.reasoning_steps.append(reasoning_step)
            self.context_accumulator.total_tokens_used += context_entry["tokens"]
            
            return f"Accumulated {context_type} context ({context_entry['tokens']} tokens). Total contexts: {len(self.context_accumulator.contexts)}"
        
        except Exception as e:
            logger.error(f"Context accumulation failed: {e}")
            return f"Context accumulation failed: {str(e)}"
    
    def get_accumulated_context(self) -> MultiHopContext:
        """Get the current accumulated context."""
        return self.context_accumulator
    
    def reset_context_accumulator(self, initial_query: str = ""):
        """Reset the context accumulator for a new reasoning session."""
        self.context_accumulator = MultiHopContext(initial_query, [], [])

class AgenticLightRAG:
    """
    Main class combining AgenticStepProcessor with LightRAG tools for multi-hop reasoning.
    """
    
    def __init__(
        self,
        lightrag_core: Optional[AthenaLightRAGCore] = None,
        model_name: str = "openai/gpt-4.1-mini",
        max_internal_steps: int = 8,
        verbose: bool = False  # Set to False for MCP protocol compliance
    ):
        """
        Initialize AgenticLightRAG.
        
        Args:
            lightrag_core: LightRAG core instance
            model_name: Model for AgenticStepProcessor 
            max_internal_steps: Maximum internal reasoning steps
            verbose: Enable verbose logging
        """
        self.lightrag = lightrag_core or create_athena_lightrag()
        self.tools_provider = LightRAGToolsProvider(self.lightrag)
        self.model_name = model_name
        self.max_internal_steps = max_internal_steps
        self.verbose = verbose
    
    def create_multi_hop_processor(
        self,
        objective: str,
        custom_instructions: Optional[List[str]] = None
    ) -> AgenticStepProcessor:
        """
        Create AgenticStepProcessor configured for multi-hop LightRAG reasoning.
        
        Args:
            objective: The reasoning objective
            custom_instructions: Optional custom processing instructions
            
        Returns:
            Configured AgenticStepProcessor
        """
        # Default instructions for multi-hop reasoning
        default_instructions = [
            "Break down complex queries into smaller, focused sub-questions",
            "Use different LightRAG query modes strategically:",
            "  - local mode for specific entity relationships", 
            "  - global mode for high-level overviews",
            "  - hybrid mode for balanced analysis",
            "  - mix mode for comprehensive retrieval",
            "  - context extraction for gathering information without generation",
            "Accumulate contexts from multiple queries before synthesis",
            "Look for patterns and connections across different contexts",
            "Synthesize findings into a comprehensive response"
        ]
        
        instructions = custom_instructions or default_instructions
        
        return AgenticStepProcessor(
            objective=objective,
            max_internal_steps=self.max_internal_steps,
            model_name=self.model_name
        )
    
    def create_reasoning_chain(
        self,
        objective: str,
        pre_processing_steps: Optional[List[str]] = None,
        post_processing_steps: Optional[List[str]] = None
    ) -> PromptChain:
        """
        Create a PromptChain with LightRAG tools for multi-hop reasoning.
        
        Args:
            objective: The reasoning objective
            pre_processing_steps: Optional pre-processing instructions
            post_processing_steps: Optional post-processing instructions
            
        Returns:
            Configured PromptChain with LightRAG tools
        """
        # Create agentic step processor
        agentic_step = self.create_multi_hop_processor(objective)
        
        # Build instruction sequence
        instructions = []
        
        # Add pre-processing steps
        if pre_processing_steps:
            instructions.extend(pre_processing_steps)
        
        # Add the main agentic reasoning step
        instructions.append(agentic_step)
        
        # Add post-processing steps
        if post_processing_steps:
            instructions.extend(post_processing_steps)
        else:
            # Default synthesis step
            instructions.append(
                "Synthesize all accumulated contexts and reasoning into a comprehensive final response: {input}"
            )
        
        # Create PromptChain
        chain = PromptChain(
            models=[self.model_name],
            instructions=instructions,
            verbose=self.verbose,
            store_steps=True
        )
        
        # Register LightRAG tools
        self.tools_provider.register_tool_functions(chain)
        chain.add_tools(self.tools_provider.create_lightrag_tools())
        
        return chain
    
    async def execute_multi_hop_reasoning(
        self,
        query: str,
        objective: Optional[str] = None,
        reset_context: bool = True,
        timeout_seconds: float = 300.0,  # 5 minute timeout
        circuit_breaker_failures: int = 3
    ) -> Dict[str, Any]:
        """
        Execute multi-hop reasoning using LightRAG tools with timeout and circuit breaker.
        
        Args:
            query: The input query
            objective: Optional custom objective
            reset_context: Whether to reset context accumulator
            timeout_seconds: Maximum execution time in seconds
            circuit_breaker_failures: Max failures before circuit breaker triggers
            
        Returns:
            Dictionary with reasoning results and context
        """
        start_time = time.time()
        
        if reset_context:
            self.tools_provider.reset_context_accumulator(query)
        
        # Create objective if not provided
        if not objective:
            objective = f"Analyze the query '{query}' using multi-hop reasoning across the Athena medical database knowledge graph"
        
        # Create reasoning chain with circuit breaker
        chain = self.create_reasoning_chain(objective)
        
        # Execute reasoning with timeout protection
        try:
            # Use asyncio.wait_for for timeout protection
            result = await asyncio.wait_for(
                self._execute_with_circuit_breaker(chain, query, circuit_breaker_failures),
                timeout=timeout_seconds
            )
            
            # Get accumulated context
            context = self.tools_provider.get_accumulated_context()
            context.final_synthesis = result
            context.execution_time = time.time() - start_time
            
            return {
                "result": result,
                "reasoning_steps": context.reasoning_steps,
                "accumulated_contexts": context.contexts,
                "total_tokens_used": context.total_tokens_used,
                "execution_time": context.execution_time,
                "step_outputs": chain.step_outputs if hasattr(chain, 'step_outputs') else [],
                "success": True
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Multi-hop reasoning timed out after {timeout_seconds} seconds")
            return {
                "result": f"Multi-hop reasoning timed out after {timeout_seconds} seconds. Consider breaking down your query into smaller parts.",
                "reasoning_steps": self.tools_provider.get_accumulated_context().reasoning_steps,
                "accumulated_contexts": self.tools_provider.get_accumulated_context().contexts,
                "total_tokens_used": self.tools_provider.get_accumulated_context().total_tokens_used,
                "execution_time": time.time() - start_time,
                "step_outputs": [],
                "success": False,
                "error": "TIMEOUT"
            }
        except Exception as e:
            logger.error(f"Multi-hop reasoning failed: {e}")
            return {
                "result": f"Multi-hop reasoning failed: {str(e)}",
                "reasoning_steps": [],
                "accumulated_contexts": [],
                "total_tokens_used": 0,
                "execution_time": time.time() - start_time,
                "step_outputs": [],
                "success": False,
                "error": str(e)
            }
    
    async def _execute_with_circuit_breaker(
        self, 
        chain: 'PromptChain', 
        query: str, 
        max_failures: int
    ) -> str:
        """
        Execute chain with circuit breaker pattern to prevent infinite loops.
        
        Args:
            chain: PromptChain instance
            query: Query to process
            max_failures: Maximum failures before circuit breaker opens
            
        Returns:
            Processing result
        """
        failures = 0
        last_error = None
        
        while failures < max_failures:
            try:
                # Execute with individual step timeout
                result = await chain.process_prompt_async(query)
                return result
            except Exception as e:
                failures += 1
                last_error = e
                logger.warning(f"Circuit breaker attempt {failures}/{max_failures} failed: {e}")
                
                if failures < max_failures:
                    # Brief backoff before retry
                    await asyncio.sleep(min(2 ** failures, 10))  # Exponential backoff, max 10s
        
        # Circuit breaker opened - all retries failed
        raise Exception(f"Circuit breaker opened after {max_failures} failures. Last error: {last_error}")
    
    def execute_multi_hop_reasoning_sync(
        self,
        query: str,
        objective: Optional[str] = None,
        reset_context: bool = True,
        timeout_seconds: float = 300.0
    ) -> Dict[str, Any]:
        """Synchronous wrapper for multi-hop reasoning with timeout."""
        return asyncio.run(self.execute_multi_hop_reasoning(
            query=query,
            objective=objective, 
            reset_context=reset_context,
            timeout_seconds=timeout_seconds
        ))


# Factory function for easy instantiation
def create_agentic_lightrag(
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db",
    model_name: str = "openai/gpt-4.1-mini",
    max_internal_steps: int = 8,
    verbose: bool = True,
    **lightrag_config
) -> AgenticLightRAG:
    """
    Factory function to create AgenticLightRAG instance.
    
    Args:
        working_dir: Path to LightRAG database
        model_name: Model for reasoning
        max_internal_steps: Max reasoning steps
        verbose: Enable verbose mode
        **lightrag_config: Additional LightRAG configuration
        
    Returns:
        Configured AgenticLightRAG instance
    """
    lightrag_core = create_athena_lightrag(working_dir=working_dir, **lightrag_config)
    return AgenticLightRAG(
        lightrag_core=lightrag_core,
        model_name=model_name,
        max_internal_steps=max_internal_steps,
        verbose=verbose
    )


# Example usage and testing
async def main():
    """Example usage of AgenticLightRAG for multi-hop reasoning."""
    try:
        # Create AgenticLightRAG instance
        agentic_rag = create_agentic_lightrag(verbose=True)
        
        # Test multi-hop reasoning
        complex_query = "What is the relationship between patient appointment tables and anesthesia case management? How do they connect to billing and financial workflows?"
        
        logger.info(f"=== Multi-Hop Reasoning Test ===")
        logger.info(f"Query: {complex_query}")
        logger.info("=" * 80)
        
        result = await agentic_rag.execute_multi_hop_reasoning(
            query=complex_query,
            objective="Analyze the complex relationships between patient appointments, anesthesia management, and billing workflows in the Athena medical database"
        )
        
        if result["success"]:
            logger.info(f"\n=== Final Result ===")
            logger.info(result["result"])
            
            logger.info(f"\n=== Reasoning Steps ===")
            for i, step in enumerate(result["reasoning_steps"], 1):
                logger.info(f"{i}. {step}")
            
            logger.info(f"\n=== Context Summary ===")
            logger.info(f"Total contexts accumulated: {len(result['accumulated_contexts'])}")
            logger.info(f"Total tokens used: {result['total_tokens_used']}")
            
            for i, context in enumerate(result["accumulated_contexts"], 1):
                logger.info(f"\nContext {i} ({context['type']}): {context['step']}")
                logger.info(f"Data preview: {context['data'][:100]}...")
        
        else:
            logger.error(f"Reasoning failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        logger.error(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())