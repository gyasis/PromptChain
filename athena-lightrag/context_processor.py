#!/usr/bin/env python3
"""
LightRAG Context Processing and SQL Generation
==============================================
Context accumulation system and SQL generation pipeline using LightRAG context extraction.
Implements validated QueryParam configurations for intelligent data processing.

Author: Athena LightRAG System
Date: 2025-09-08
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
import os
import time
from functools import wraps

# Add parent directories to path for imports
sys.path.append('/home/gyasis/Documents/code/PromptChain')
sys.path.append('/home/gyasis/Documents/code/PromptChain/athena-lightrag')

from lightrag_core import AthenaLightRAGCore, QueryResult, QueryMode, create_athena_lightrag

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of contexts that can be accumulated."""
    SCHEMA = "schema"
    RELATIONSHIPS = "relationships"
    BUSINESS_LOGIC = "business_logic"
    DATA_PATTERNS = "data_patterns"
    CONSTRAINTS = "constraints"
    EXAMPLES = "examples"

@dataclass
class ContextFragment:
    """Individual context fragment with metadata."""
    content: str
    context_type: ContextType
    source_query: str
    query_mode: QueryMode
    confidence_score: float
    tokens_estimated: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccumulatedContext:
    """Container for accumulated context across multiple queries."""
    fragments: List[ContextFragment] = field(default_factory=list)
    total_tokens: int = 0
    query_history: List[str] = field(default_factory=list)
    synthesis_cache: Dict[str, str] = field(default_factory=dict)
    
    def add_fragment(self, fragment: ContextFragment):
        """Add a context fragment to the accumulator."""
        self.fragments.append(fragment)
        self.total_tokens += fragment.tokens_estimated
        if fragment.source_query not in self.query_history:
            self.query_history.append(fragment.source_query)
    
    def get_fragments_by_type(self, context_type: ContextType) -> List[ContextFragment]:
        """Get all fragments of a specific type."""
        return [f for f in self.fragments if f.context_type == context_type]
    
    def get_synthesis_key(self, query: str, context_types: List[ContextType]) -> str:
        """Generate a cache key for synthesis results."""
        type_names = [ct.value for ct in context_types]
        return f"{query}_{'-'.join(sorted(type_names))}"

class ContextProcessor:
    """
    Processes and accumulates context from LightRAG using QueryParam configurations.
    """
    
    def __init__(self, lightrag_core: AthenaLightRAGCore):
        """
        Initialize context processor with LightRAG core.
        
        Args:
            lightrag_core: AthenaLightRAGCore instance
        """
        self.lightrag = lightrag_core
        self.accumulated_context = AccumulatedContext()
        self.context_extraction_strategies = self._init_extraction_strategies()
    
    def _init_extraction_strategies(self) -> Dict[ContextType, Dict[str, Any]]:
        """Initialize context extraction strategies for different types."""
        return {
            ContextType.SCHEMA: {
                "query_mode": "local",
                "keywords": ["table", "column", "field", "structure", "schema"],
                "max_entity_tokens": 8000,
                "top_k": 80
            },
            ContextType.RELATIONSHIPS: {
                "query_mode": "global", 
                "keywords": ["relationship", "connection", "foreign key", "join", "related"],
                "max_relation_tokens": 10000,
                "top_k": 60
            },
            ContextType.BUSINESS_LOGIC: {
                "query_mode": "hybrid",
                "keywords": ["business", "process", "workflow", "logic", "rule"],
                "max_entity_tokens": 6000,
                "max_relation_tokens": 8000,
                "top_k": 70
            },
            ContextType.DATA_PATTERNS: {
                "query_mode": "mix",
                "keywords": ["pattern", "data", "values", "distribution", "count"],
                "top_k": 100
            },
            ContextType.CONSTRAINTS: {
                "query_mode": "local",
                "keywords": ["constraint", "validation", "required", "unique", "primary"],
                "max_entity_tokens": 5000,
                "top_k": 50
            },
            ContextType.EXAMPLES: {
                "query_mode": "naive",
                "keywords": ["example", "sample", "instance", "case"],
                "top_k": 40
            }
        }
    
    async def extract_context_by_type(
        self,
        query: str,
        context_type: ContextType,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> ContextFragment:
        """
        Extract context of a specific type using optimized QueryParam configuration.
        
        Args:
            query: The context extraction query
            context_type: Type of context to extract
            custom_params: Optional custom query parameters
            
        Returns:
            ContextFragment with extracted context
        """
        strategy = self.context_extraction_strategies[context_type]
        
        # Build query parameters
        query_params = {
            "top_k": strategy.get("top_k", 60),
            "max_entity_tokens": strategy.get("max_entity_tokens", 6000),
            "max_relation_tokens": strategy.get("max_relation_tokens", 8000)
        }
        
        # Override with custom parameters if provided
        if custom_params:
            query_params.update(custom_params)
        
        try:
            # Extract context using only_need_context=True
            context_content = await self.lightrag.get_context_only_async(
                query_text=query,
                mode=strategy["query_mode"],
                **query_params
            )
            
            # Estimate tokens and confidence
            tokens_estimated = len(context_content) // 4
            confidence_score = self._calculate_confidence(context_content, strategy["keywords"])
            
            fragment = ContextFragment(
                content=context_content,
                context_type=context_type,
                source_query=query,
                query_mode=strategy["query_mode"],
                confidence_score=confidence_score,
                tokens_estimated=tokens_estimated,
                metadata={
                    "strategy": strategy,
                    "query_params": query_params
                }
            )
            
            return fragment
            
        except Exception as e:
            logger.error(f"Context extraction failed for {context_type}: {e}")
            return ContextFragment(
                content=f"Context extraction failed: {str(e)}",
                context_type=context_type,
                source_query=query,
                query_mode=strategy["query_mode"],
                confidence_score=0.0,
                tokens_estimated=0,
                metadata={"error": str(e)}
            )
    
    def _calculate_confidence(self, content: str, keywords: List[str]) -> float:
        """Calculate confidence score based on keyword presence and content quality."""
        if not content or len(content) < 10:
            return 0.0
        
        # Keyword matching score
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content.lower())
        keyword_score = min(keyword_matches / len(keywords), 1.0)
        
        # Content quality score (based on length, structure)
        length_score = min(len(content) / 1000, 1.0)
        structure_score = 1.0 if any(marker in content for marker in [":", "\n", "-", "•"]) else 0.5
        
        # Combined confidence
        confidence = (keyword_score * 0.4 + length_score * 0.3 + structure_score * 0.3)
        return min(confidence, 1.0)
    
    async def accumulate_comprehensive_context(
        self, 
        base_query: str,
        context_types: Optional[List[ContextType]] = None,
        max_parallel: int = 3
    ) -> AccumulatedContext:
        """
        Accumulate comprehensive context across multiple types and queries.
        
        Args:
            base_query: Base query to expand from
            context_types: Types of context to extract (default: all)
            max_parallel: Maximum parallel extractions
            
        Returns:
            AccumulatedContext with all extracted fragments
        """
        if context_types is None:
            context_types = list(ContextType)
        
        # Generate specialized queries for each context type
        specialized_queries = await self._generate_specialized_queries(base_query, context_types)
        
        # Extract context fragments in parallel (limited concurrency)
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def extract_with_semaphore(query: str, ctx_type: ContextType) -> ContextFragment:
            async with semaphore:
                return await self.extract_context_by_type(query, ctx_type)
        
        tasks = []
        for ctx_type in context_types:
            query = specialized_queries.get(ctx_type, base_query)
            tasks.append(extract_with_semaphore(query, ctx_type))
        
        # Execute all extractions
        fragments = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and update accumulated context
        for fragment in fragments:
            if isinstance(fragment, ContextFragment):
                self.accumulated_context.add_fragment(fragment)
            else:
                logger.error(f"Context extraction failed: {fragment}")
        
        return self.accumulated_context
    
    async def _generate_specialized_queries(
        self,
        base_query: str,
        context_types: List[ContextType]
    ) -> Dict[ContextType, str]:
        """Generate specialized queries for different context types."""
        specialized_queries = {}
        
        # Query templates for different context types
        templates = {
            ContextType.SCHEMA: f"What are the table structures and column definitions related to: {base_query}",
            ContextType.RELATIONSHIPS: f"What are the relationships and connections between tables for: {base_query}",
            ContextType.BUSINESS_LOGIC: f"What business processes and workflows are involved in: {base_query}",
            ContextType.DATA_PATTERNS: f"What data patterns and distributions exist for: {base_query}",
            ContextType.CONSTRAINTS: f"What constraints and validation rules apply to: {base_query}",
            ContextType.EXAMPLES: f"What are examples and sample data for: {base_query}"
        }
        
        for ctx_type in context_types:
            specialized_queries[ctx_type] = templates.get(ctx_type, base_query)
        
        return specialized_queries
    
    def synthesize_context(
        self,
        target_query: str,
        context_types: Optional[List[ContextType]] = None,
        use_cache: bool = True
    ) -> str:
        """
        Synthesize accumulated context into a coherent response.
        
        Args:
            target_query: Query to synthesize context for
            context_types: Specific context types to include
            use_cache: Whether to use synthesis cache
            
        Returns:
            Synthesized context string
        """
        if context_types is None:
            context_types = list(ContextType)
        
        # Check cache first
        if use_cache:
            cache_key = self.accumulated_context.get_synthesis_key(target_query, context_types)
            if cache_key in self.accumulated_context.synthesis_cache:
                return self.accumulated_context.synthesis_cache[cache_key]
        
        # Group fragments by type
        synthesis_parts = []
        
        for ctx_type in context_types:
            fragments = self.accumulated_context.get_fragments_by_type(ctx_type)
            if not fragments:
                continue
            
            # Sort by confidence score
            fragments.sort(key=lambda f: f.confidence_score, reverse=True)
            
            # Combine high-confidence fragments
            type_content = []
            for fragment in fragments:
                if fragment.confidence_score > 0.3:  # Filter low-confidence content
                    type_content.append(fragment.content)
            
            if type_content:
                synthesis_parts.append(f"\n=== {ctx_type.value.upper()} ===\n")
                synthesis_parts.append("\n---\n".join(type_content))
        
        synthesized = "\n".join(synthesis_parts)
        
        # Cache the result
        if use_cache:
            self.accumulated_context.synthesis_cache[cache_key] = synthesized
        
        return synthesized

class SQLGenerator:
    """
    Generates SQL queries using accumulated LightRAG context.
    """
    
    def __init__(self, context_processor: ContextProcessor):
        """
        Initialize SQL generator with context processor.
        
        Args:
            context_processor: ContextProcessor instance
        """
        self.context_processor = context_processor
        self.sql_templates = self._init_sql_templates()
    
    def _init_sql_templates(self) -> Dict[str, str]:
        """Initialize SQL generation templates."""
        return {
            "select": """
SELECT {columns}
FROM {tables}
{joins}
{where}
{group_by}
{having}
{order_by}
{limit}
""".strip(),
            
            "join": "JOIN {table} ON {condition}",
            "left_join": "LEFT JOIN {table} ON {condition}",
            "where": "WHERE {conditions}",
            "group_by": "GROUP BY {columns}",
            "order_by": "ORDER BY {columns}",
            "limit": "LIMIT {count}"
        }
    
    async def generate_sql_from_query(
        self,
        natural_query: str,
        context_types: Optional[List[ContextType]] = None,
        include_explanation: bool = True,
        timeout_seconds: float = 120.0,  # 2 minute timeout
        max_context_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language using accumulated context with timeout protection.
        
        Args:
            natural_query: Natural language query
            context_types: Context types to use for generation
            include_explanation: Whether to include explanation
            timeout_seconds: Maximum execution time in seconds
            max_context_retries: Maximum retries for context accumulation
            
        Returns:
            Dictionary with SQL query and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure we have comprehensive context with timeout protection
            if not self.context_processor.accumulated_context.fragments:
                logger.info("Accumulating context for SQL generation...")
                
                # Use timeout and retry logic for context accumulation
                context_retries = 0
                context_accumulated = False
                
                while context_retries < max_context_retries and not context_accumulated:
                    try:
                        await asyncio.wait_for(
                            self.context_processor.accumulate_comprehensive_context(natural_query),
                            timeout=60.0  # 1 minute for context accumulation
                        )
                        context_accumulated = True
                        logger.info(f"Context accumulated successfully on attempt {context_retries + 1}")
                    except asyncio.TimeoutError:
                        context_retries += 1
                        logger.warning(f"Context accumulation timeout (attempt {context_retries}/{max_context_retries})")
                        if context_retries >= max_context_retries:
                            return {
                                "success": False,
                                "sql": None,
                                "error": "Context accumulation timed out after multiple retries",
                                "natural_query": natural_query,
                                "execution_time": time.time() - start_time,
                                "metadata": {"timeout_reason": "context_accumulation"}
                            }
                    except Exception as e:
                        context_retries += 1
                        logger.error(f"Context accumulation failed (attempt {context_retries}): {e}")
                        if context_retries >= max_context_retries:
                            return {
                                "success": False,
                                "sql": None,
                                "error": f"Context accumulation failed after {max_context_retries} retries: {str(e)}",
                                "natural_query": natural_query,
                                "execution_time": time.time() - start_time,
                                "metadata": {"failure_reason": "context_accumulation"}
                            }
            
            # Synthesize relevant context with error handling
            context_types = context_types or [
                ContextType.SCHEMA, 
                ContextType.RELATIONSHIPS,
                ContextType.BUSINESS_LOGIC
            ]
            
            try:
                relevant_context = await asyncio.wait_for(
                    self.context_processor.synthesize_context_async(
                        natural_query, 
                        context_types
                    ) if hasattr(self.context_processor, 'synthesize_context_async') 
                    else asyncio.to_thread(
                        self.context_processor.synthesize_context,
                        natural_query,
                        context_types
                    ),
                    timeout=30.0  # 30 seconds for context synthesis
                )
            except asyncio.TimeoutError:
                logger.error("Context synthesis timed out")
                return {
                    "success": False,
                    "sql": None,
                    "error": "Context synthesis timed out",
                    "natural_query": natural_query,
                    "execution_time": time.time() - start_time,
                    "metadata": {"timeout_reason": "context_synthesis"}
                }
            except Exception as e:
                logger.error(f"Context synthesis failed: {e}")
                return {
                    "success": False,
                    "sql": None,
                    "error": f"Context synthesis failed: {str(e)}",
                    "natural_query": natural_query,
                    "execution_time": time.time() - start_time,
                    "metadata": {"failure_reason": "context_synthesis"}
                }
            
            # Generate SQL using LightRAG with context and timeout protection
            sql_generation_query = f"""
Given this database context:

{relevant_context[:8000]}  # Truncate to prevent token overflow

Generate a SQL query to answer: {natural_query}

Requirements:
- Use proper table and column names from the context
- Include appropriate JOINs for relationships  
- Add WHERE clauses for filtering
- Consider performance optimization
- Return only valid SQL without explanation unless requested

SQL Query:
"""
            
            # Use hybrid mode for SQL generation
            result = await self.context_processor.lightrag.query_hybrid_async(
                sql_generation_query,
                max_entity_tokens=8000,
                max_relation_tokens=10000
            )
            
            if result.error:
                return {
                    "sql": None,
                    "error": result.error,
                    "success": False
                }
            
            # Extract SQL from result
            sql_query = self._extract_sql_from_response(result.result)
            
            response = {
                "sql": sql_query,
                "natural_query": natural_query,
                "context_used": context_types,
                "execution_time": result.execution_time,
                "success": True,
                "metadata": {
                    "tokens_estimated": result.tokens_used,
                    "context_fragments": len(self.context_processor.accumulated_context.fragments)
                }
            }
            
            if include_explanation:
                response["explanation"] = self._generate_explanation(sql_query, natural_query, relevant_context)
            
            return response
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return {
                "sql": None,
                "natural_query": natural_query,
                "error": str(e),
                "success": False
            }
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LightRAG response."""
        # Look for SQL patterns
        sql_patterns = [
            r'```sql\n(.*?)```',
            r'```\n(SELECT.*?)```',
            r'(SELECT.*?;)',
            r'(SELECT.*?)(?:\n\n|\n$)'
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                return sql
        
        # If no pattern matches, look for lines starting with SQL keywords
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if re.match(r'^(SELECT|WITH|INSERT|UPDATE|DELETE)', line, re.IGNORECASE):
                in_sql = True
                sql_lines.append(line)
            elif in_sql and line and not line.startswith('--'):
                sql_lines.append(line)
            elif in_sql and (not line or line.startswith('--')):
                break
        
        return '\n'.join(sql_lines) if sql_lines else response
    
    def _generate_explanation(self, sql: str, natural_query: str, context: str) -> str:
        """Generate explanation for the SQL query."""
        explanation_parts = [
            f"This SQL query answers: '{natural_query}'",
            "",
            "Query breakdown:",
        ]
        
        # Simple SQL analysis
        if "SELECT" in sql.upper():
            explanation_parts.append("- Retrieves data using SELECT statement")
        if "JOIN" in sql.upper():
            explanation_parts.append("- Combines data from multiple tables using JOINs")
        if "WHERE" in sql.upper():
            explanation_parts.append("- Filters results with WHERE conditions")
        if "GROUP BY" in sql.upper():
            explanation_parts.append("- Groups results for aggregation")
        if "ORDER BY" in sql.upper():
            explanation_parts.append("- Sorts results by specified columns")
        
        return "\n".join(explanation_parts)


# Factory function for easy instantiation
def create_context_processor(
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db",
    **lightrag_config
) -> ContextProcessor:
    """
    Factory function to create ContextProcessor instance.
    
    Args:
        working_dir: Path to LightRAG database
        **lightrag_config: Additional LightRAG configuration
        
    Returns:
        Configured ContextProcessor instance
    """
    lightrag_core = create_athena_lightrag(working_dir=working_dir, **lightrag_config)
    return ContextProcessor(lightrag_core)

def create_sql_generator(
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db",
    **lightrag_config
) -> SQLGenerator:
    """
    Factory function to create SQLGenerator instance.
    
    Args:
        working_dir: Path to LightRAG database
        **lightrag_config: Additional LightRAG configuration
        
    Returns:
        Configured SQLGenerator instance
    """
    context_processor = create_context_processor(working_dir=working_dir, **lightrag_config)
    return SQLGenerator(context_processor)


# Example usage and testing
async def main():
    """Example usage of context processing and SQL generation."""
    try:
        # Create context processor and SQL generator
        sql_generator = create_sql_generator()
        
        # Test context accumulation and SQL generation
        test_queries = [
            "Show me all patient appointments for today",
            "What are the billing amounts for completed anesthesia cases?",
            "Find patients with multiple appointments this month"
        ]
        
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Processing: {query}")
            print(f"{'='*80}")
            
            # Generate SQL
            result = await sql_generator.generate_sql_from_query(query)
            
            if result["success"]:
                print(f"\nGenerated SQL:")
                print(result["sql"])
                
                if "explanation" in result:
                    print(f"\nExplanation:")
                    print(result["explanation"])
                
                print(f"\nMetadata:")
                print(f"- Execution time: {result['execution_time']:.2f}s")
                print(f"- Context fragments used: {result['metadata']['context_fragments']}")
            else:
                print(f"SQL generation failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        print(f"Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())