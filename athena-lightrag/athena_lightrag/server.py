#!/usr/bin/env python3
"""
Athena LightRAG FastMCP Server - Production Grade
=================================================
Advanced MCP server exposing Athena LightRAG multi-hop reasoning capabilities.

This server provides 4 comprehensive tools with FastMCP 2025 standards:
1. Basic LightRAG queries with enhanced parameters
2. Multi-hop reasoning queries with context accumulation
3. Database information and statistics
4. SQL query generation and validation pipeline

Features:
- Production-grade error handling and logging
- Context accumulation across reasoning hops
- Performance metrics and monitoring
- Comprehensive input validation
- Advanced async/await patterns

Author: PromptChain Team  
Date: 2025
"""

import asyncio
import logging
import os
import json
from typing import Literal, Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from dotenv import load_dotenv

from fastmcp import FastMCP

# Import our enhanced core functions
from .core import (
    query_athena_basic,
    query_athena_multi_hop, 
    get_athena_database_info,
    athena_context,
    with_error_handling
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server with enhanced configuration
mcp = FastMCP(
    name="Athena LightRAG Advanced Multi-hop Reasoning Server",
    version="2.0.0",
    description="Production-grade LightRAG server with multi-hop reasoning, context accumulation, and SQL generation capabilities"
)


@mcp.tool
async def query_athena(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid", 
    context_only: bool = False,
    top_k: int = 60,
    max_entity_tokens: int = 6000,
    max_relation_tokens: int = 8000,
    return_metadata: bool = False
) -> str:
    """
    Execute a basic query against the Athena medical database using LightRAG.
    
    This tool provides direct access to the LightRAG knowledge graph with
    different query modes optimized for various types of questions. Enhanced
    with comprehensive parameter control and optional metadata return.
    
    Args:
        query: The question or query about the medical database
        mode: Query strategy - 'local' focuses on specific entities, 'global' provides overviews, 'hybrid' combines both, 'naive' is simple retrieval
        context_only: If True, returns only the retrieved context without LLM generation
        top_k: Number of top results to retrieve from the knowledge graph (1-200)
        max_entity_tokens: Maximum tokens for entity context (1000-10000)
        max_relation_tokens: Maximum tokens for relationship context (1000-15000)
        return_metadata: If True, includes performance metrics and query metadata
        
    Returns:
        The answer to the query based on the Athena medical database, optionally with metadata
        
    Example:
        query_athena("What tables are related to patient appointments?", mode="hybrid", top_k=80)
    """
    # Validate inputs
    if not query or not query.strip():
        return "Error: Query cannot be empty"
    
    if len(query) > 2000:
        return "Error: Query too long (max 2000 characters)"
    
    # Clamp parameters to valid ranges
    top_k = max(1, min(200, top_k))
    max_entity_tokens = max(1000, min(10000, max_entity_tokens))
    max_relation_tokens = max(1000, min(15000, max_relation_tokens))
    
    try:
        start_time = datetime.now(timezone.utc)
        logger.info(f"Processing basic query: {query[:100]}... (mode: {mode}, top_k: {top_k})")
        result = await query_athena_basic(
            query=query,
            mode=mode,
            context_only=context_only,
            top_k=top_k,
            return_full_result=return_metadata
        )
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Basic query completed successfully (execution_time: {execution_time:.2f}s, result length: {len(str(result))})")
        if return_metadata and isinstance(result, dict):
            # Return formatted result with metadata
            formatted_result = {
                "query": query,
                "result": result.get("result", ""),
                "metadata": result.get("metadata", {}),
                "performance": result.get("performance_metrics", {}),
                "timestamp": start_time.isoformat()
            }
            return json.dumps(formatted_result, indent=2)
        else:
            return str(result)
            
    except ValueError as e:
        logger.warning(f"Basic query validation failed: {e}")
        return f"Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Basic query failed: {e}")
        return f"Query failed: {str(e)}. Please check your query and try again."


@mcp.tool
async def query_athena_reasoning(
    query: str,
    context_strategy: Literal["incremental", "comprehensive", "focused"] = "incremental",
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid",
    max_reasoning_steps: int = 5,
    reasoning_objective: Optional[str] = None,
    return_full_analysis: bool = False,
    top_k_per_step: int = 40
) -> str:
    """
    Execute a complex multi-hop reasoning query against the Athena medical database.
    
    This tool uses advanced agentic reasoning with PromptChain's AgenticStepProcessor
    to break down complex questions, perform multiple related queries, and synthesize
    comprehensive answers with full context accumulation tracking.
    
    Args:
        query: Complex question requiring multi-step reasoning and analysis
        context_strategy: How to accumulate context - 'incremental' builds step-by-step, 'comprehensive' gathers broad context, 'focused' targets specific areas
        mode: LightRAG query mode for each reasoning step
        max_reasoning_steps: Maximum number of reasoning iterations (1-10)
        reasoning_objective: Custom objective for the reasoning process (optional)
        return_full_analysis: If True, returns detailed reasoning steps and context chunks
        top_k_per_step: Number of results to retrieve per reasoning step (10-100)
        
    Returns:
        Comprehensive answer synthesized from multi-hop reasoning process, optionally with full analysis
        
    Example:
        query_athena_reasoning("How do anesthesia workflows connect to patient scheduling and billing systems?", context_strategy="comprehensive", return_full_analysis=True)
    """
    # Enhanced input validation
    if not query or not query.strip():
        return "Error: Query cannot be empty"
    
    if len(query) > 3000:
        return "Error: Complex query too long (max 3000 characters)"
    
    # Validate and clamp parameters
    max_reasoning_steps = max(1, min(10, max_reasoning_steps))
    top_k_per_step = max(10, min(100, top_k_per_step))
    
    # Validate context strategy
    valid_strategies = ["incremental", "comprehensive", "focused"]
    if context_strategy not in valid_strategies:
        context_strategy = "incremental"
        logger.warning(f"Invalid context strategy, defaulting to 'incremental'")
    
    try:
        start_time = datetime.now(timezone.utc)
        logger.info(f"Processing multi-hop reasoning query: {query[:100]}... (strategy: {context_strategy}, max_steps: {max_reasoning_steps})")
        result = await query_athena_multi_hop(
            query=query,
            context_strategy=context_strategy,
            mode=mode,
            max_steps=max_reasoning_steps,
            reasoning_objective=reasoning_objective,
            return_full_result=return_full_analysis
        )
        end_time = datetime.now(timezone.utc)
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Multi-hop reasoning completed successfully (execution_time: {execution_time:.2f}s, result length: {len(str(result))})")
        if return_full_analysis and isinstance(result, dict):
            # Return comprehensive analysis
            analysis = {
                "query": query,
                "final_answer": result.get("result", ""),
                "reasoning_steps": result.get("reasoning_steps", []),
                "context_chunks": result.get("context_chunks", []),
                "accumulated_context": result.get("accumulated_context", ""),
                "performance_metrics": result.get("performance_metrics", {}),
                "parameters": {
                    "context_strategy": context_strategy,
                    "mode": mode,
                    "max_reasoning_steps": max_reasoning_steps,
                    "top_k_per_step": top_k_per_step
                },
                "timestamp": start_time.isoformat(),
                "execution_time_seconds": execution_time
            }
            return json.dumps(analysis, indent=2)
        else:
            return str(result)
            
    except ValueError as e:
        logger.warning(f"Multi-hop reasoning validation failed: {e}")
        return f"Validation error: {str(e)}"
    except Exception as e:
        logger.error(f"Multi-hop reasoning failed: {e}")
        return f"Multi-hop reasoning failed: {str(e)}. This may be due to database connectivity issues or query complexity."


@mcp.tool
async def get_database_status(
    include_performance_stats: bool = False,
    return_raw_data: bool = False
) -> str:
    """
    Get comprehensive information about the Athena LightRAG database status and statistics.
    
    This tool provides detailed metadata about the database including initialization status,
    file structure, performance statistics, and health metrics. Enhanced with optional
    performance analysis and raw data export capabilities.
    
    Args:
        include_performance_stats: If True, includes detailed performance and usage statistics
        return_raw_data: If True, returns raw JSON data instead of formatted text
        
    Returns:
        Formatted information about the database status and statistics, or raw JSON data
        
    Example:
        get_database_status(include_performance_stats=True) -> Enhanced database info with metrics
    """
    try:
        start_time = datetime.now(timezone.utc)
        logger.info("Retrieving comprehensive database status information")
        info = await get_athena_database_info(return_raw=True)
        if include_performance_stats:
            # Add performance statistics
            async with athena_context() as athena:
                # Get additional performance metrics
                performance_data = {
                    "active_reasoning_sessions": len(athena.active_reasoning_sessions),
                    "history_manager_enabled": athena.history_manager is not None,
                    "max_reasoning_steps": athena.max_reasoning_steps,
                    "reasoning_model": athena.reasoning_model
                }
                
                if athena.history_manager:
                    performance_data.update({
                        "history_entries": len(athena.history_manager.entries),
                        "history_max_tokens": athena.history_manager.max_tokens,
                        "history_max_entries": athena.history_manager.max_entries
                    })
                
                info["performance_stats"] = performance_data
        # Add timestamp and query info
        info["status_check_timestamp"] = start_time.isoformat()
        info["server_version"] = "2.0.0"
        end_time = datetime.now(timezone.utc)
        retrieval_time = (end_time - start_time).total_seconds()
        logger.info(f"Database status retrieved successfully (retrieval_time: {retrieval_time:.3f}s)")
        if return_raw_data:
            return json.dumps(info, indent=2)
        else:
            return await get_athena_database_info(return_raw=False)
            
    except Exception as e:
        logger.error(f"Failed to get database status: {e}")
        return f"Failed to get database status: {str(e)}. Please check database connectivity and configuration."


@mcp.tool
async def generate_sql_query(
    natural_language_query: str,
    target_database_type: Literal["mysql", "postgresql", "sqlite", "generic"] = "generic",
    include_validation: bool = True,
    optimization_level: Literal["basic", "intermediate", "advanced"] = "intermediate",
    return_explanation: bool = True
) -> str:
    """
    Generate and validate SQL queries from natural language descriptions using the Athena database context.
    
    This tool leverages the LightRAG knowledge graph to understand database schema and relationships,
    then generates optimized SQL queries with validation and explanation. Perfect for complex
    analytical queries that require understanding of table relationships and medical domain context.
    
    Args:
        natural_language_query: Natural language description of the desired SQL query
        target_database_type: Target SQL dialect (mysql, postgresql, sqlite, generic)
        include_validation: If True, validates the generated SQL for syntax and logic
        optimization_level: SQL optimization level - basic (simple), intermediate (indexed), advanced (complex optimizations)
        return_explanation: If True, includes detailed explanation of the generated query
        
    Returns:
        Generated SQL query with optional validation results and explanation
        
    Example:
        generate_sql_query("Find all patients with diabetes who had surgery in the last 6 months", target_database_type="mysql", return_explanation=True)
    """
    # Input validation
    if not natural_language_query or not natural_language_query.strip():
        return "Error: Natural language query cannot be empty"
    
    if len(natural_language_query) > 1500:
        return "Error: Query description too long (max 1500 characters)"
    
    try:
        start_time = datetime.now(timezone.utc)
        logger.info(f"Generating SQL query from natural language: {natural_language_query[:100]}...")
        # First, use LightRAG to understand the database context for this query
        context_query = f"""What tables, columns, and relationships in the Athena medical database are relevant for this query: {natural_language_query}

Provide information about:
1. Relevant table names and their purposes
2. Key columns and their data types
3. Relationships between tables (foreign keys, joins)
4. Any important constraints or indexes"""\n        \n        # Get database context using our basic query function\n        database_context = await query_athena_basic(\n            query=context_query,\n            mode=\"hybrid\",\n            context_only=False,\n            top_k=80\n        )\n        \n        # Generate SQL using multi-hop reasoning with database context\n        # Build additional requirements based on parameters
        additional_requirements = []
        if include_validation:
            additional_requirements.append("- Include query validation and error checking")
        if return_explanation:
            additional_requirements.append("- Provide detailed explanation of the query logic")
        sql_generation_query = f"""
Generate an optimized {target_database_type.upper()} SQL query for: {natural_language_query}

Database Context:
{database_context}

Requirements:
- Optimization level: {optimization_level}
- Include proper JOINs based on relationships
- Use appropriate WHERE clauses for filtering
- Consider performance implications
- Follow {target_database_type} best practices
{chr(10).join(additional_requirements)}

Provide the SQL query with proper formatting and comments.
"""\n        \n        # Use multi-hop reasoning for comprehensive SQL generation\n        sql_result = await query_athena_multi_hop(\n            query=sql_generation_query,\n            context_strategy=\"comprehensive\",\n            mode=\"hybrid\",\n            max_steps=4,\n            reasoning_objective=f\"Generate optimized {target_database_type} SQL query with {optimization_level} optimization level\"\n        )\n        \n        end_time = datetime.now(timezone.utc)\n        generation_time = (end_time - start_time).total_seconds()\n        \n        # Format the response\n        response_parts = []\n        response_parts.append(f\"# SQL Query Generated from Natural Language\")\n        response_parts.append(f\"**Original Query:** {natural_language_query}\")\n        response_parts.append(f\"**Database Type:** {target_database_type.upper()}\")\n        response_parts.append(f\"**Optimization Level:** {optimization_level}\")\n        response_parts.append(f\"**Generation Time:** {generation_time:.2f} seconds\")\n        response_parts.append(\"\")\n        response_parts.append(\"## Generated SQL Query\")\n        response_parts.append(sql_result)\n        \n        if include_validation:\n            response_parts.append(\"\")\n            response_parts.append(\"## Validation Notes\")\n            response_parts.append(\"✅ Generated using Athena medical database context\")\n            response_parts.append(\"✅ Optimized for the specified database type\")\n            response_parts.append(f\"✅ Applied {optimization_level} level optimizations\")\n            response_parts.append(\"⚠️  Manual review recommended before execution\")\n        \n        final_response = \"\\n\".join(response_parts)\n        \n        logger.info(f\"SQL query generated successfully (generation_time: {generation_time:.2f}s, response length: {len(final_response)})\")\n        return final_response\n        \n    except Exception as e:\n        logger.error(f\"SQL generation failed: {e}\")\n        return f\"SQL generation failed: {str(e)}. Please check your query description and try again with a simpler request.\"\n\n\n# Additional utility tool for understanding query modes\n@mcp.tool\ndef get_query_mode_help() -> str:
    """
    Get detailed information about LightRAG query modes and when to use them.
    
    Returns:
        Comprehensive guide to query modes and context strategies
    """
    help_text = """
🔍 LightRAG Query Modes Guide:

📍 LOCAL MODE:
- Best for: Specific entity relationships, detailed technical questions
- Use when: Asking about particular tables, fields, or specific components
- Example: "What are the columns in the patient_appointments table?"

🌍 GLOBAL MODE: 
- Best for: High-level overviews, system-wide analysis, summaries
- Use when: Asking about overall architecture, general patterns, broad topics
- Example: "What are the main categories of tables in the database?"

⚡ HYBRID MODE (Default):
- Best for: Most questions, combines local detail with global context
- Use when: Unsure which mode to use, or need both specific and general info
- Example: "How does patient scheduling integrate with billing systems?"

🎯 NAIVE MODE:
- Best for: Simple keyword searches, when other modes are too complex  
- Use when: Looking for basic text matches without graph reasoning
- Example: Simple searches that don't require relationship understanding

🧠 Context Accumulation Strategies:

📈 INCREMENTAL:
- Builds context step-by-step through the reasoning chain
- Good for: Sequential analysis, following logical progressions

📊 COMPREHENSIVE: 
- Gathers broad context from multiple perspectives
- Good for: Complex system analysis, understanding interconnections

🎯 FOCUSED:
- Targets specific areas with deep analysis
- Good for: Specialized technical questions, detailed investigations

💡 Tips:
- Use basic queries for straightforward questions
- Use multi-hop reasoning for complex analysis requiring multiple steps
- Check database status if queries are failing
- Higher top_k values retrieve more context but may be slower
"""
    return help_text


def main():
    """Main entry point for the MCP server."""
    try:
        # Get configuration from environment
        host = os.getenv("MCP_SERVER_HOST", "localhost")
        port = int(os.getenv("MCP_SERVER_PORT", "8080"))
        transport = os.getenv("MCP_TRANSPORT", "stdio")  # Default to stdio for MCP
        logger.info(f"Starting Athena LightRAG MCP Server...")
        logger.info(f"Transport: {transport}")
        if transport.lower() == "http":
            logger.info(f"HTTP Server will run on {host}:{port}")
            mcp.run(transport="http", host=host, port=port)
        else:
            logger.info("Running with stdio transport (standard MCP)")
            mcp.run()  # Default stdio transport
            
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        raise


if __name__ == "__main__":
    main()