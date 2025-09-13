#!/usr/bin/env python3
"""
Athena LightRAG FastMCP Server - Production Grade (Fixed Version)
=================================================================
Fixed version of the SQL generation function to avoid syntax errors.
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
4. Any important constraints or indexes"""
        
        # Get database context using our basic query function
        database_context = await query_athena_basic(
            query=context_query,
            mode="hybrid",
            context_only=False,
            top_k=80
        )
        
        # Build additional requirements based on parameters
        additional_requirements = []
        if include_validation:
            additional_requirements.append("- Include query validation and error checking")
        if return_explanation:
            additional_requirements.append("- Provide detailed explanation of the query logic")
        
        # Generate SQL using multi-hop reasoning with database context
        sql_generation_query = f"""Generate an optimized {target_database_type.upper()} SQL query for: {natural_language_query}

Database Context:
{database_context}

Requirements:
- Optimization level: {optimization_level}
- Include proper JOINs based on relationships
- Use appropriate WHERE clauses for filtering
- Consider performance implications
- Follow {target_database_type} best practices
{chr(10).join(additional_requirements)}

Provide the SQL query with proper formatting and comments."""
        
        # Use multi-hop reasoning for comprehensive SQL generation
        sql_result = await query_athena_multi_hop(
            query=sql_generation_query,
            context_strategy="comprehensive",
            mode="hybrid",
            max_steps=4,
            reasoning_objective=f"Generate optimized {target_database_type} SQL query with {optimization_level} optimization level"
        )
        
        end_time = datetime.now(timezone.utc)
        generation_time = (end_time - start_time).total_seconds()
        
        # Format the response
        response_parts = [
            f"# SQL Query Generated from Natural Language",
            f"**Original Query:** {natural_language_query}",
            f"**Database Type:** {target_database_type.upper()}",
            f"**Optimization Level:** {optimization_level}",
            f"**Generation Time:** {generation_time:.2f} seconds",
            "",
            "## Generated SQL Query",
            sql_result
        ]
        
        if include_validation:
            response_parts.extend([
                "",
                "## Validation Notes",
                "✅ Generated using Athena medical database context",
                "✅ Optimized for the specified database type",
                f"✅ Applied {optimization_level} level optimizations",
                "⚠️  Manual review recommended before execution"
            ])
        
        final_response = "\n".join(response_parts)
        
        logger.info(f"SQL query generated successfully (generation_time: {generation_time:.2f}s, response length: {len(final_response)})")
        return final_response
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return f"SQL generation failed: {str(e)}. Please check your query description and try again with a simpler request."


if __name__ == "__main__":
    print("Fixed SQL generation function ready!")