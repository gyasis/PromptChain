#!/usr/bin/env python3
"""
Athena LightRAG MCP Server
=========================
FastMCP 2.0 compliant server for Athena Health EHR database analysis and SQL generation.

🏥 ATHENA HEALTH EHR DATABASE CONTEXT:
This MCP server provides intelligent access to comprehensive medical database schemas,
table relationships, and healthcare workflow patterns from the Athena Health EHR system.

❄️ SNOWFLAKE DATABASE STRUCTURE:
• Database: `athena`
• Multiple Schemas: `athenaone`, `athenaclinical`, `athenafinancial`, `athenaoperational`, etc.
• Schema Discovery: Automatic detection of all available schemas in the database
• Fully Qualified Tables: `athena.{schema}.TABLENAME` format (e.g., `athena.athenaone.APPOINTMENT`)
• All SQL generation uses proper Snowflake syntax with full database.schema.table references

📊 DATABASE COVERAGE:
• 100+ medical tables across multiple schemas with detailed metadata
• Multi-Schema Architecture: Tables organized by functional domain (clinical, financial, operational)
• Categories: Collector tables (appointments, patients, billing, clinical data) across all schemas
• Snowflake table format: `athena.{schema}.{table}` naming convention
• Complete data dictionary with schema-aware descriptions, constraints, and relationships

🔍 CORE MEDICAL DATA AREAS:
• Patient Management: PATIENT, APPOINTMENT, APPOINTMENTTYPE, APPOINTMENTNOTE
• Clinical Workflow: ANESTHESIACASE, CHARGEDIAGNOSIS, clinical documentation
• Billing & Revenue: CAPPAYMENT, CHARGEEXTRAFIELDS, insurance, payments
• Scheduling: APPOINTMENTTICKLER, ALLOWABLESCHEDULECATEGORY, scheduling rules
• Operational: Provider management, department structure, system configuration

🛠️ MCP TOOLS PROVIDED:
1. lightrag_local_query - Focused entity/table relationship discovery across all schemas
2. lightrag_global_query - Comprehensive medical workflow overviews spanning multiple schemas
3. lightrag_hybrid_query - Combined detailed + contextual analysis with schema awareness
4. lightrag_context_extract - Raw data dictionary metadata extraction from all schemas
5. lightrag_multi_hop_reasoning - Complex medical data relationship analysis across schema boundaries
6. lightrag_sql_generation - Validated SQL generation for multi-schema Athena queries
7. lightrag_schema_discovery - Schema discovery and listing functionality

🎯 USE CASES:
• Healthcare data discovery and documentation
• SQL query development and validation
• Medical workflow analysis and optimization
• Database schema exploration for developers
• Data integration and ETL planning
• Clinical data analysis and reporting

Author: Athena LightRAG System
Date: 2025-09-09
"""

# CRITICAL: Load environment variables with override BEFORE any other imports
from dotenv import load_dotenv
import os

load_dotenv(override=True)  # Force project .env to override system environment variables

# Check if we should enable debug logging (for development) - DEFINE EARLY
MCP_DEBUG_MODE = os.getenv('MCP_DEBUG_MODE', 'false').lower() == 'true'

# Back up original stdout and stderr BEFORE any imports that might suppress output
import sys
stderr_backup = sys.stderr
stdout_backup = sys.stdout

# Force the correct API key from project .env file to prevent contamination
project_api_key = os.getenv('OPENAI_API_KEY')
if project_api_key:
    os.environ['OPENAI_API_KEY'] = project_api_key  # Explicitly override any defaults

# Configure early logging suppression BEFORE any PromptChain imports
import logging
import warnings
if not MCP_DEBUG_MODE:
    # Suppress PromptChain logging early
    logging.getLogger('promptchain').setLevel(logging.ERROR)
    logging.getLogger('promptchain.utils').setLevel(logging.ERROR)
    
    # Suppress Pydantic deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
    warnings.filterwarnings("ignore", message=".*Pydantic V1 style.*")

# Import custom JSON encoder for MCP protocol compliance
from json_encoder import prepare_mcp_response, MCPJSONEncoder

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
import sys
import os
from functools import lru_cache
import threading
import asyncio
from contextlib import contextmanager, redirect_stderr
import io

# Use relative imports - no hardcoded paths
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# FastMCP 2.0 imports - required for proper server operation
from fastmcp import FastMCP
from pydantic import BaseModel, Field

from lightrag_core import AthenaLightRAGCore, QueryResult, QueryMode, create_athena_lightrag
from agentic_lightrag import AgenticLightRAG, create_agentic_lightrag
from context_processor import ContextProcessor, SQLGenerator, create_context_processor, create_sql_generator
# Debug logger removed - using standard logging instead

# Restore stdout after imports are loaded (but keep stderr suppressed)
if not MCP_DEBUG_MODE:
    sys.stdout = stdout_backup

# Configure logging to suppress PromptChain output unless in debug mode
log_file = "/tmp/athena_lightrag_mcp.log"

# Additional specific logger suppression after imports
if not MCP_DEBUG_MODE:
    logging.getLogger('promptchain.utils.model_management').setLevel(logging.ERROR)
    logging.getLogger('promptchain.utils.ollama_model_manager').setLevel(logging.ERROR)

# CRITICAL: Redirect stderr to prevent PromptChain contamination of MCP protocol

# NOTE: We suppress stderr only temporarily during PromptChain operations, not permanently
# The MCP server needs stderr for initialization and error reporting

# Context manager to temporarily suppress all output during tool execution
@contextmanager
def suppress_output():
    """Context manager to completely suppress all output during tool execution."""
    if MCP_DEBUG_MODE:
        # In debug mode, don't suppress output
        yield
        return
        
    original_stderr = sys.stderr
    original_stdout = sys.stdout
    try:
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        if sys.stderr != original_stderr:
            sys.stderr.close()
        sys.stderr = original_stderr
        if sys.stdout != original_stdout:
            sys.stdout.close()
        sys.stdout = original_stdout

# First, disable all existing handlers to prevent stdout contamination
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure only file logging with CRITICAL level for maximum suppression
logging.basicConfig(
    level=logging.CRITICAL,  # Only CRITICAL messages to minimize noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='a',
    force=True  # Force reconfiguration to override any existing loggers
)

# Apply maximum suppression for MCP protocol compliance
# Set all loggers to CRITICAL level to prevent any stdout contamination
logging.getLogger().setLevel(logging.CRITICAL)  # Root logger - maximum suppression
logging.getLogger('LiteLLM').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

# ENHANCED: Comprehensive PromptChain logging suppression
logging.getLogger('promptchain').setLevel(logging.CRITICAL)
logging.getLogger("promptchain.utils.promptchaining").setLevel(logging.CRITICAL) 
logging.getLogger("promptchain.utils.agentic_step_processor").setLevel(logging.CRITICAL)
logging.getLogger("promptchain.agents").setLevel(logging.CRITICAL)
logging.getLogger("promptchain.core").setLevel(logging.CRITICAL)
logging.getLogger("promptchain.utils").setLevel(logging.CRITICAL)
logging.getLogger("promptchain.utils.dynamic_parameter_resolver").setLevel(logging.CRITICAL)
logging.getLogger("promptchain.utils.llm_tools").setLevel(logging.CRITICAL)
logging.getLogger("promptchain.utils.parameter_validation").setLevel(logging.CRITICAL)

# Suppress LLM libraries
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("anthropic").setLevel(logging.CRITICAL)

# Suppress LightRAG debug logs completely
logging.getLogger("lightrag").setLevel(logging.CRITICAL)
logging.getLogger("nano-vectordb").setLevel(logging.CRITICAL)
logging.getLogger("lightrag_core").setLevel(logging.CRITICAL)
logging.getLogger("agentic_lightrag").setLevel(logging.CRITICAL)
logging.getLogger("context_processor").setLevel(logging.CRITICAL)

# Disable any other common noisy loggers
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("h11").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Initialize FastMCP 2.0 server - module-level instance for auto-detection
mcp = FastMCP("Athena LightRAG MCP Server")

# Pydantic models for tool parameters
class LocalQueryParams(BaseModel):
    """Parameters for local query tool focused on Athena Health EHR Snowflake database entities."""
    query: str = Field(..., description="Search query for Athena medical data entities across all Snowflake schemas. Examples: 'APPOINTMENT table relationships', 'patient ID columns across schemas', 'diagnosis code fields', 'billing workflow connections', 'all schemas containing patient data'")
    top_k: int = Field(60, description="Number of top Athena Snowflake entities to retrieve - medical tables, columns, or relationships (default: 60)")
    max_entity_tokens: int = Field(6000, description="Maximum tokens for Athena Snowflake entity context - controls detail level of table/column descriptions (default: 6000)")

class GlobalQueryParams(BaseModel):
    """Parameters for global query tool for Athena Health EHR Snowflake database overviews."""
    query: str = Field(..., description="Search query for Athena medical data overviews across all Snowflake schemas. Examples: 'all appointment-related tables across schemas', 'Collector category tables by schema', 'patient data workflow spanning multiple schemas', 'billing system structure across financial schemas'")
    max_relation_tokens: int = Field(8000, description="Maximum tokens for Athena Snowflake relationship context - controls breadth of medical workflow descriptions (default: 8000)")
    top_k: int = Field(60, description="Number of top Athena table relationships to retrieve across the Snowflake medical database (default: 60)")
    timeout_seconds: float = Field(90.0, description="Maximum execution time in seconds (default: 90 seconds)")

class HybridQueryParams(BaseModel):
    """Parameters for hybrid query tool combining Athena entity details with broader medical workflows."""
    query: str = Field(..., description="Search query combining specific Athena entities with workflow context. Examples: 'APPOINTMENT table structure and patient journey', 'charge diagnosis relationships and billing flow'")
    max_entity_tokens: int = Field(6000, description="Maximum tokens for specific Athena entity details (tables, columns, constraints) (default: 6000)")
    max_relation_tokens: int = Field(8000, description="Maximum tokens for Athena workflow relationships across medical processes (default: 8000)")
    top_k: int = Field(60, description="Number of top Athena entities and relationships to retrieve for comprehensive analysis (default: 60)")
    timeout_seconds: float = Field(120.0, description="Maximum execution time in seconds (default: 120 seconds)")

class ContextExtractParams(BaseModel):
    """Parameters for extracting raw Athena Health EHR Snowflake database context without processing."""
    query: str = Field(..., description="Search query for raw Athena Snowflake data dictionary extraction from all schemas. Examples: 'APPOINTMENT table schema details', 'patient ID column details across schemas', 'billing table metadata by schema', 'list all available schemas'")
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = Field(
        "hybrid", 
        description="Extraction mode: 'local' for specific Snowflake tables/columns, 'global' for category overviews, 'hybrid' for balanced coverage"
    )
    max_entity_tokens: int = Field(6000, description="Maximum tokens for Athena Snowflake entity metadata (table descriptions, column specs)")
    max_relation_tokens: int = Field(8000, description="Maximum tokens for Athena Snowflake relationship metadata (foreign keys, workflow connections)")
    top_k: int = Field(60, description="Number of top Athena Snowflake metadata items to extract (tables, columns, relationships)")

class MultiHopReasoningParams(BaseModel):
    """Parameters for sophisticated multi-hop reasoning across Athena Health EHR database."""
    query: str = Field(..., description="Complex Athena medical data question requiring multi-step analysis. Examples: 'How do patient appointments connect to billing and outcomes?', 'Trace anesthesia case workflow from scheduling to payment'")
    objective: Optional[str] = Field(None, description="Specific reasoning goal for Athena analysis (e.g., 'Map patient revenue cycle', 'Identify data quality issues')")
    max_steps: int = Field(8, description="Maximum reasoning steps for complex Athena medical data analysis (2-15 recommended)")
    timeout_seconds: float = Field(300.0, description="Maximum execution time in seconds (default: 300 seconds / 5 minutes)")

class SQLGenerationParams(BaseModel):
    """Parameters for generating validated Snowflake SQL queries against Athena Health EHR database."""
    natural_query: str = Field(..., description="Natural language description of desired Athena medical data query for Snowflake. Examples: 'Find all patients with diabetes appointments', 'Show revenue by department last quarter', 'List overdue appointment ticklers', 'Show me schemas containing financial data'. Will generate proper athena.{schema}.TABLE_NAME references with automatic schema detection.")
    include_explanation: bool = Field(True, description="Whether to include Snowflake SQL explanation with Athena table relationships, JOIN logic, fully qualified table names, and medical data context")

# Singleton pattern for resource management - prevents memory bloat
working_dir = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
_instances_lock = asyncio.Lock()  # FIXED: Use async lock instead of threading.Lock
_instances = {}

async def get_singleton_instance(instance_type: str, working_dir: str = None):
    """
    Thread-safe singleton factory for LightRAG instances.
    Prevents memory bloat from duplicate instance creation.
    
    Args:
        instance_type: Type of instance ('lightrag_core', 'agentic_lightrag', 'context_processor', 'sql_generator')
        working_dir: Working directory path
        
    Returns:
        Singleton instance of requested type
    """
    wd = working_dir or globals()['working_dir']
    cache_key = f"{instance_type}_{wd}"
    
    if cache_key not in _instances:
        async with _instances_lock:
            # Double-check pattern for async safety
            if cache_key not in _instances:
                logger.info(f"Creating singleton instance: {instance_type}")
                
                if instance_type == 'lightrag_core':
                    _instances[cache_key] = create_athena_lightrag(working_dir=wd)
                elif instance_type == 'agentic_lightrag':
                    # Reuse lightrag_core instance to prevent duplication
                    core_key = f"lightrag_core_{wd}"
                    if core_key not in _instances:
                        _instances[core_key] = create_athena_lightrag(working_dir=wd)
                    _instances[cache_key] = create_agentic_lightrag(working_dir=wd)
                elif instance_type == 'context_processor':
                    _instances[cache_key] = create_context_processor(working_dir=wd)
                elif instance_type == 'sql_generator':
                    _instances[cache_key] = create_sql_generator(working_dir=wd)
                else:
                    raise ValueError(f"Unknown instance type: {instance_type}")
    
    return _instances[cache_key]

# Lazy initialization using singleton pattern
async def get_lightrag_core():
    return await get_singleton_instance('lightrag_core', working_dir)

async def get_agentic_lightrag():
    return await get_singleton_instance('agentic_lightrag', working_dir)

async def get_context_processor():
    return await get_singleton_instance('context_processor', working_dir)

async def get_sql_generator():
    return await get_singleton_instance('sql_generator', working_dir)
    
# FastMCP 2.0 Tool Definitions using @mcp.tool() decorator

@mcp.tool()
async def lightrag_local_query(
    query: str,
    top_k: int = 200,
    max_entity_tokens: int = 15000,
    timeout_seconds: float = 60.0
) -> Dict[str, Any]:
    """Query Athena Health EHR database in local mode for context-dependent information. 
    
    Specializes in finding specific entity relationships between tables, columns, and medical workflows. 
    Perfect for discovering table connections like APPOINTMENT → PATIENT relationships, specific column 
    descriptions, or focused schema exploration within the Athena medical database containing 100+ tables 
    with detailed metadata.
    
    Args:
        query: Search query for Athena medical data entities in Snowflake. Examples: 'athena.athenaone.APPOINTMENT table relationships', 'patient ID columns', 'diagnosis code fields', 'billing workflow connections'
        top_k: Number of top Athena Snowflake entities to retrieve - medical tables, columns, or relationships (default: 60)
        max_entity_tokens: Maximum tokens for Athena Snowflake entity context - controls detail level of table/column descriptions (default: 6000)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Input validation
    if not query or not query.strip():
        return {
            "success": False,
            "result": "",
            "error": "Query cannot be empty",
            "mode": "local"
        }
    
    if top_k <= 0 or top_k > 200:
        return {
            "success": False,
            "result": "",
            "error": "top_k must be between 1 and 200",
            "mode": "local"
        }
    
    try:
        core = await get_lightrag_core()
        
        # Execute with timeout protection
        result = await asyncio.wait_for(
            core.query_local_async(
                query_text=query.strip()[:1000],  # Truncate extremely long queries
                top_k=min(top_k, 200),  # Cap to reasonable limit
                max_entity_tokens=min(max_entity_tokens, 15000)  # Cap token limit
            ),
            timeout=timeout_seconds
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        if not result or not hasattr(result, 'result'):
            return {
                "success": False,
                "result": "",
                "error": "Invalid result from LightRAG core",
                "mode": "local",
                "execution_time": execution_time
            }
        
        return {
            "success": True,
            "result": result.result[:50000] if result.result else "",  # Truncate huge responses
            "mode": "local",
            "execution_time": execution_time,
            "tokens_used": getattr(result, 'tokens_used', {}),
            "error": getattr(result, 'error', None),
            "query_length": len(query),
            "result_length": len(result.result) if result.result else 0
        }
    
    except asyncio.TimeoutError:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Local query timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "result": "",
            "error": f"Query timed out after {timeout_seconds} seconds. Try simplifying your query.",
            "mode": "local",
            "execution_time": execution_time,
            "timeout": True
        }
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Local query failed: {e}")
        return {
            "success": False,
            "result": "",
            "error": f"Local query failed: {str(e)[:500]}...",  # Truncate error messages
            "mode": "local",
            "execution_time": execution_time,
            "exception_type": type(e).__name__
        }

@mcp.tool()
async def lightrag_global_query(
    query: str,
    max_relation_tokens: int = 8000,
    top_k: int = 60,
    timeout_seconds: float = 90.0
) -> Dict[str, Any]:
    """Query Athena Health EHR database in global mode for high-level overviews and comprehensive summaries. 
    
    Excellent for understanding broad medical data patterns, getting complete category overviews (like all 
    Collector tables), discovering major workflow relationships across the entire Athena schema, or finding 
    tables by functional areas (appointments, billing, clinical data, anesthesia cases, etc.).
    
    Args:
        query: Search query for Athena medical data overviews in Snowflake. Examples: 'all appointment-related tables in athena.athenaone', 'Collector category tables', 'patient data workflow', 'billing system structure'
        max_relation_tokens: Maximum tokens for Athena Snowflake relationship context - controls breadth of medical workflow descriptions (default: 8000)
        top_k: Number of top Athena table relationships to retrieve across the Snowflake medical database (default: 60)
        timeout_seconds: Maximum execution time in seconds (default: 90 seconds)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Input validation
    if not query or not query.strip():
        return {
            "success": False,
            "result": "",
            "error": "Query cannot be empty",
            "mode": "global"
        }
    
    try:
        core = await get_lightrag_core()
        
        # Execute with timeout protection
        result = await asyncio.wait_for(
            core.query_global_async(
                query_text=query.strip()[:1000],
                max_relation_tokens=min(max_relation_tokens, 15000),
                top_k=min(top_k, 100)
            ),
            timeout=timeout_seconds
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "success": True,
            "result": result.result[:50000] if result.result else "",
            "mode": "global", 
            "execution_time": execution_time,
            "tokens_used": getattr(result, 'tokens_used', {}),
            "error": getattr(result, 'error', None)
        }
    
    except asyncio.TimeoutError:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Global query timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "result": "",
            "error": f"Global query timed out after {timeout_seconds} seconds. Try simplifying your query.",
            "mode": "global",
            "execution_time": execution_time,
            "timeout": True
        }
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Global query failed: {e}")
        return {
            "success": False,
            "result": "",
            "error": f"Global query failed: {str(e)[:500]}...",
            "mode": "global",
            "execution_time": execution_time,
            "exception_type": type(e).__name__
        }

@mcp.tool()
async def lightrag_hybrid_query(
    query: str,
    max_entity_tokens: int = 15000,
    max_relation_tokens: int = 20000,
    top_k: int = 200,
    timeout_seconds: float = 120.0
) -> Dict[str, Any]:
    """Query Athena Health EHR database in hybrid mode, combining detailed entity information with broader relationship patterns. 
    
    Ideal for complex medical data discovery requiring both specific table/column details AND their broader 
    context within Athena workflows. Perfect for SQL development, data validation, and understanding how 
    specific medical entities (patients, appointments, charges, diagnoses) connect across the Athena EHR system.
    
    Args:
        query: Search query combining specific Athena entities with workflow context. Examples: 'APPOINTMENT table structure and patient journey', 'charge diagnosis relationships and billing flow'
        max_entity_tokens: Maximum tokens for specific Athena entity details (tables, columns, constraints) (default: 6000)
        max_relation_tokens: Maximum tokens for Athena workflow relationships across medical processes (default: 8000)
        top_k: Number of top Athena entities and relationships to retrieve for comprehensive analysis (default: 60)
        timeout_seconds: Maximum execution time in seconds (default: 120 seconds)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Input validation
    if not query or not query.strip():
        return {
            "success": False,
            "result": "",
            "error": "Query cannot be empty",
            "mode": "hybrid"
        }
    
    try:
        core = await get_lightrag_core()
        
        # Execute with timeout protection
        result = await asyncio.wait_for(
            core.query_hybrid_async(
                query_text=query.strip()[:1000],
                max_entity_tokens=min(max_entity_tokens, 15000),
                max_relation_tokens=min(max_relation_tokens, 20000),
                top_k=min(top_k, 200)
            ),
            timeout=timeout_seconds
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "success": True,
            "result": result.result[:50000] if result.result else "",
            "mode": "hybrid",
            "execution_time": execution_time,
            "tokens_used": getattr(result, 'tokens_used', {}),
            "error": getattr(result, 'error', None)
        }
    
    except asyncio.TimeoutError:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Hybrid query timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "result": "",
            "error": f"Hybrid query timed out after {timeout_seconds} seconds. Try simplifying your query.",
            "mode": "hybrid",
            "execution_time": execution_time,
            "timeout": True
        }
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Hybrid query failed: {e}")
        return {
            "success": False,
            "result": "",
            "error": f"Hybrid query failed: {str(e)[:500]}...",
            "mode": "hybrid",
            "execution_time": execution_time,
            "exception_type": type(e).__name__
        }

@mcp.tool()
async def lightrag_context_extract(
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "hybrid",
    max_entity_tokens: int = 6000,
    max_relation_tokens: int = 8000,
    top_k: int = 60,
    timeout_seconds: float = 90.0
) -> Dict[str, Any]:
    """Extract raw context from Athena Health EHR Snowflake database without generating a response. 
    
    Returns pure data dictionary information, table descriptions, column details, and schema metadata 
    for the athena.athenaone schema. Essential for programmatic access to Snowflake database structure, 
    building data dictionaries, validating table/column existence, and gathering precise metadata for 
    Snowflake SQL query construction and database documentation.
    
    Args:
        query: Search query for raw Athena Snowflake data dictionary extraction. Examples: 'athena.athenaone.APPOINTMENT schema', 'patient ID column details', 'billing table metadata'
        mode: Extraction mode: 'local' for specific Snowflake tables/columns, 'global' for category overviews, 'hybrid' for balanced coverage
        max_entity_tokens: Maximum tokens for Athena Snowflake entity metadata (table descriptions, column specs)
        max_relation_tokens: Maximum tokens for Athena Snowflake relationship metadata (foreign keys, workflow connections)
        top_k: Number of top Athena Snowflake metadata items to extract (tables, columns, relationships)
        timeout_seconds: Maximum execution time in seconds (default: 90 seconds)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Input validation
    if not query or not query.strip():
        return {
            "success": False,
            "context": "",
            "error": "Query cannot be empty",
            "mode": mode
        }
    
    try:
        core = await get_lightrag_core()
        
        # Execute with timeout protection
        context = await asyncio.wait_for(
            core.get_context_only_async(
                query_text=query.strip()[:1000],
                mode=mode,
                max_entity_tokens=min(max_entity_tokens, 10000),
                max_relation_tokens=min(max_relation_tokens, 15000),
                top_k=min(top_k, 100)
            ),
            timeout=timeout_seconds
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "success": True,
            "context": context[:50000] if context else "",  # Truncate large context
            "mode": mode,
            "tokens_estimated": len(context) // 4,
            "query": query,
            "execution_time": execution_time
        }
    
    except asyncio.TimeoutError:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Context extraction timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "context": "",
            "error": f"Context extraction timed out after {timeout_seconds} seconds. Try simplifying your query.",
            "mode": mode,
            "execution_time": execution_time,
            "timeout": True
        }
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Context extraction failed: {e}")
        return {
            "success": False,
            "context": "",
            "error": f"Context extraction failed: {str(e)[:500]}...",
            "mode": mode,
            "execution_time": execution_time,
            "exception_type": type(e).__name__
        }

@mcp.tool()
async def lightrag_multi_hop_reasoning(
    query: str,
    objective: Optional[str] = None,
    max_steps: int = 8,
    timeout_seconds: float = 300.0  # 5 minutes default timeout
) -> Dict[str, Any]:
    """Execute sophisticated multi-hop reasoning across Athena Health EHR database using PromptChain's AgenticStepProcessor. 
    
    Performs complex analysis requiring multiple reasoning steps, such as tracing patient journeys across 
    appointments→diagnoses→charges→payments, discovering workflow dependencies, or analyzing complex medical 
    data relationships. Uses advanced AI reasoning to connect disparate Athena tables and provide comprehensive 
    insights for healthcare data analysis.
    
    Args:
        query: Complex Athena medical data question requiring multi-step analysis. Examples: 'How do patient appointments connect to billing and outcomes?', 'Trace anesthesia case workflow from scheduling to payment', 'List all schemas in the athena database', 'Find all tables containing patient data across schemas'
        objective: Specific reasoning goal for Athena analysis (e.g., 'Map patient revenue cycle', 'Identify data quality issues')
        max_steps: Maximum reasoning steps for complex Athena medical data analysis (2-10 recommended)
        timeout_seconds: Maximum execution time in seconds (default: 180 seconds / 3 minutes)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Input validation and sanitization
    if not query or not query.strip():
        return {
            "success": False,
            "result": "",
            "error": "Query cannot be empty",
            "reasoning_steps": [],
            "execution_time": 0
        }
    
    # Cap max_steps to prevent runaway execution
    max_steps = max(1, min(max_steps, 15))  # Limit between 1-15 steps
    
    # Sanitize inputs
    # query = query.strip()[:2000]  # Truncate very long queries
    # if objective:
    #     objective = objective.strip()[:500]  # Truncate long objectives
    
    # Dynamic timeout based on max_steps (30-35 seconds per step + base time)
    # Each step may involve multiple LightRAG queries (7-11s each), so allocate more time
    adaptive_timeout = min(timeout_seconds, max(60, max_steps * 30 + 60))
    
    logger.info(f"Starting multi-hop reasoning: steps={max_steps}, timeout={adaptive_timeout}s")
    
    retry_count = 0
    max_retries = 2
    
    while retry_count <= max_retries:
        try:
            # Wrap the entire execution in output suppression to prevent MCP protocol contamination
            with suppress_output():
                agentic = await get_agentic_lightrag()
                
                # Execute with timeout protection and circuit breaker
                result = await asyncio.wait_for(
                    agentic.execute_multi_hop_reasoning(
                        query=query,
                        objective=objective,
                        timeout_seconds=adaptive_timeout * 0.9,  # Leave buffer for outer timeout
                        circuit_breaker_failures=max(2, max_steps // 3)  # Dynamic failure threshold
                    ),
                    timeout=adaptive_timeout
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Multi-hop reasoning completed in {execution_time:.2f}s")
            
            # Debug: Log result structure before serialization
            logger.debug(f"Result type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            
            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Invalid result format from agentic reasoning")
            
            # Truncate large results to prevent MCP response overflow
            result_text = result.get("result", "")
            if len(result_text) > 25000:  # 25KB limit
                result_text = result_text[:25000] + "\n\n[TRUNCATED - Result too large]"
                result["result"] = result_text
            
            # Prepare raw response preserving tool outputs with safe JSON serialization
            accumulated_contexts = result.get("accumulated_contexts", [])
            
            # Safely serialize accumulated contexts ensuring JSON compatibility
            serializable_contexts = []
            for ctx in accumulated_contexts:
                try:
                    if isinstance(ctx, dict):
                        # Safely serialize each field
                        safe_ctx = {
                            "type": str(ctx.get("type", "unknown")),
                            "data": str(ctx.get("data", ""))[:5000],  # Keep more data but safely as string
                            "step": str(ctx.get("step", ""))[:1000],  # Keep more step info
                            "tokens": int(ctx.get("tokens", 0)) if isinstance(ctx.get("tokens"), (int, float)) else 0
                        }
                        # Test JSON serialization of this context
                        json.dumps(safe_ctx)
                        serializable_contexts.append(safe_ctx)
                    else:
                        # Handle non-dict contexts
                        safe_ctx = {
                            "type": "raw_output",
                            "data": str(ctx)[:5000],
                            "step": "",
                            "tokens": len(str(ctx)) // 4
                        }
                        # Test JSON serialization
                        json.dumps(safe_ctx)
                        serializable_contexts.append(safe_ctx)
                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    # If serialization fails, create safe fallback
                    logger.warning(f"Context serialization failed: {e}")
                    serializable_contexts.append({
                        "type": "serialization_error",
                        "data": f"Context could not be serialized: {type(ctx).__name__}",
                        "step": "",
                        "tokens": 0
                    })
            
            raw_response = {
                "success": result.get("success", True),
                "result": result_text,
                "reasoning_steps": result.get("reasoning_steps", []),
                "accumulated_contexts": serializable_contexts,  # Include full contexts, not just preview
                "accumulated_contexts_count": len(accumulated_contexts),
                "total_tokens_used": result.get("total_tokens_used", 0),
                "execution_time": execution_time,
                "timeout_used": adaptive_timeout,
                "max_steps_requested": max_steps,
                "retry_count": retry_count,
                "step_outputs_count": len(result.get("step_outputs", [])),
                "error": result.get("error")
            }
            
            # Use safe JSON serialization for MCP protocol compliance
            return prepare_mcp_response(raw_response)
        
        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            retry_count += 1
            
            logger.error(f"Multi-hop reasoning timed out after {adaptive_timeout}s (attempt {retry_count}/{max_retries + 1})")
            
            if retry_count <= max_retries:
                # Implement exponential backoff and reduce complexity for retry
                max_steps = max(2, max_steps // 2)  # Reduce steps for retry
                adaptive_timeout = min(adaptive_timeout * 0.8, 120)  # Reduce timeout slightly
                
                logger.info(f"Retrying with reduced complexity: steps={max_steps}, timeout={adaptive_timeout}s")
                await asyncio.sleep(1)  # Brief delay before retry
                continue
            else:
                return {
                    "success": False,
                    "result": f"Multi-hop reasoning timed out after {adaptive_timeout} seconds across {max_retries + 1} attempts. The query '{query[:100]}...' may be too complex. Try reducing max_steps or simplifying the query.",
                    "error": f"Timeout after {adaptive_timeout}s with {max_retries + 1} attempts",
                    "reasoning_steps": [],
                    "execution_time": execution_time,
                    "timeout_used": adaptive_timeout,
                    "max_steps_requested": max_steps,
                    "retry_count": retry_count,
                    "timeout": True,
                    "circuit_breaker_triggered": True
                }
        
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = str(e)
            
            logger.error(f"Multi-hop reasoning failed: {error_msg}")
            
            # Check for specific error types that shouldn't be retried
            if any(err_type in error_msg.lower() for err_type in ['connection', 'network', 'api key', 'authentication']):
                retry_count = max_retries + 1  # Skip retries for these errors
            else:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"Retrying after error (attempt {retry_count}/{max_retries + 1})")
                    await asyncio.sleep(2)  # Brief delay before retry
                    continue
            
            return {
                "success": False,
                "result": "",
                "error": f"Multi-hop reasoning failed: {error_msg[:500]}{'...' if len(error_msg) > 500 else ''}",
                "reasoning_steps": [],
                "execution_time": execution_time,
                "timeout_used": adaptive_timeout,
                "max_steps_requested": max_steps,
                "retry_count": retry_count,
                "exception_type": type(e).__name__
            }
    
    # This should never be reached, but just in case
    return {
        "success": False,
        "result": "",
        "error": "Unexpected error in retry logic",
        "reasoning_steps": [],
        "execution_time": asyncio.get_event_loop().time() - start_time
    }

@mcp.tool()
async def lightrag_sql_generation(
    natural_query: str,
    include_explanation: bool = True,
    timeout_seconds: float = 60.0
) -> Dict[str, Any]:
    """Generate validated Snowflake SQL queries for Athena Health EHR database from natural language descriptions. 
    
    Uses comprehensive knowledge of Athena table schemas, column relationships, and medical data patterns to 
    create accurate Snowflake SQL. Generates proper fully qualified table names (athena.athenaone.TABLE_NAME), 
    correct JOIN logic for medical workflows, and ensures queries follow Snowflake syntax and healthcare data 
    best practices. Essential for healthcare analysts and developers working with Athena medical data in Snowflake.
    
    Args:
        natural_query: Natural language description of desired Athena medical data query for Snowflake. Examples: 'Find all patients with diabetes appointments', 'Show revenue by department last quarter', 'List overdue appointment ticklers'. Will generate proper athena.athenaone.TABLE_NAME references.
        include_explanation: Whether to include Snowflake SQL explanation with Athena table relationships, JOIN logic, fully qualified table names, and medical data context
    """
    start_time = asyncio.get_event_loop().time()
    
    # Input validation
    if not natural_query or not natural_query.strip():
        return {
            "success": False,
            "sql": None,
            "error": "Natural query cannot be empty"
        }
    
    try:
        generator = await get_sql_generator()
        
        # Add timeout protection for SQL generation
        result = await asyncio.wait_for(
            generator.generate_sql_from_query(
                natural_query=natural_query.strip()[:1000],  # Truncate long queries
                include_explanation=include_explanation
            ),
            timeout=timeout_seconds
        )
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return {
            "success": result["success"],
            "sql": result.get("sql"),
            "explanation": result.get("explanation"),
            "natural_query": result["natural_query"],
            "execution_time": execution_time,
            "error": result.get("error"),
            "metadata": result.get("metadata", {})
        }
    
    except asyncio.TimeoutError:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"SQL generation timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "sql": None,
            "explanation": None,
            "natural_query": natural_query,
            "execution_time": execution_time,
            "timeout": True,
            "error": f"SQL generation timed out after {timeout_seconds} seconds. Try simplifying your query."
        }
    
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"SQL generation failed: {e}")
        return {
            "success": False,
            "sql": None,
            "error": str(e),
            "natural_query": natural_query,
            "execution_time": execution_time
        }
    
# Legacy support functions (keeping for backward compatibility)
class AthenaMCPServer:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, working_dir: str = None, server_name: str = None, server_version: str = None):
        self.working_dir = working_dir or "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
        self.server_name = server_name or "athena-lightrag"
        self.server_version = server_version or "1.0.0"
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and status."""
        try:
            core = await get_lightrag_core()
            db_status = core.get_database_status()
            
            return {
                "server_name": self.server_name,
                "server_version": self.server_version,
                "working_dir": self.working_dir,
                "database_status": db_status,
                "available_tools": [
                    "lightrag_local_query",
                    "lightrag_global_query", 
                    "lightrag_hybrid_query",
                    "lightrag_context_extract",
                    "lightrag_multi_hop_reasoning",
                    "lightrag_sql_generation"
                ],
                "query_modes": core.get_available_modes()
            }
        
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {
                "server_name": self.server_name,
                "error": str(e)
            }
    
    async def run_server(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server."""
        logger.info(f"Starting Athena LightRAG MCP Server on {host}:{port}")
        await mcp.run(host=host, port=port)


# Manual MCP implementation (fallback)
class ManualMCPServer:
    """Manual MCP server implementation when FastMCP is not available."""
    
    def __init__(self, athena_server: AthenaMCPServer):
        """Initialize manual MCP server."""
        self.athena = athena_server
        self.tools = {
            "lightrag_local_query": lightrag_local_query,
            "lightrag_global_query": lightrag_global_query,
            "lightrag_hybrid_query": lightrag_hybrid_query,
            "lightrag_context_extract": lightrag_context_extract,
            "lightrag_multi_hop_reasoning": lightrag_multi_hop_reasoning,
            "lightrag_sql_generation": lightrag_sql_generation
        }
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool manually."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            }
        
        try:
            # Convert parameters to appropriate model
            if tool_name == "lightrag_local_query":
                params = LocalQueryParams(**parameters)
            elif tool_name == "lightrag_global_query":
                params = GlobalQueryParams(**parameters)
            elif tool_name == "lightrag_hybrid_query":
                params = HybridQueryParams(**parameters)
            elif tool_name == "lightrag_context_extract":
                params = ContextExtractParams(**parameters)
            elif tool_name == "lightrag_multi_hop_reasoning":
                params = MultiHopReasoningParams(**parameters)
            elif tool_name == "lightrag_sql_generation":
                params = SQLGenerationParams(**parameters)
            else:
                params = parameters
            
            # Call the tool handler with unpacked parameters
            if hasattr(params, 'dict'):  # Pydantic model
                result = await self.tools[tool_name](**params.dict())
            else:  # Regular dict
                result = await self.tools[tool_name](**params)
            return result
        
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get tool schemas in OpenAI format."""
        return {
            "lightrag_local_query": {
                "type": "function",
                "function": {
                    "name": "lightrag_local_query",
                    "description": "Query LightRAG in local mode for context-dependent information focusing on specific entity relationships",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query for local context"},
                            "top_k": {"type": "integer", "description": "Number of top entities to retrieve (default: 200)", "default": 200},
                            "max_entity_tokens": {"type": "integer", "description": "Maximum tokens for entity context (default: 15000)", "default": 15000}
                        },
                        "required": ["query"]
                    }
                }
            },
            "lightrag_global_query": {
                "type": "function",
                "function": {
                    "name": "lightrag_global_query",
                    "description": "Query LightRAG in global mode for high-level overviews and summaries",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query for global overview"},
                            "max_relation_tokens": {"type": "integer", "description": "Maximum tokens for relationship context (default: 8000)", "default": 8000},
                            "top_k": {"type": "integer", "description": "Number of top relationships to retrieve (default: 60)", "default": 60},
                            "timeout_seconds": {"type": "number", "description": "Maximum execution time in seconds", "default": 90.0}
                        },
                        "required": ["query"]
                    }
                }
            },
            "lightrag_hybrid_query": {
                "type": "function",
                "function": {
                    "name": "lightrag_hybrid_query",
                    "description": "Query LightRAG in hybrid mode combining local and global approaches",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query for hybrid analysis"},
                            "max_entity_tokens": {"type": "integer", "description": "Maximum tokens for entity context (default: 15000)", "default": 15000},
                            "max_relation_tokens": {"type": "integer", "description": "Maximum tokens for relationship context (default: 20000)", "default": 20000},
                            "top_k": {"type": "integer", "description": "Number of top items to retrieve (default: 200)", "default": 200},
                            "timeout_seconds": {"type": "number", "description": "Maximum execution time in seconds", "default": 120.0}
                        },
                        "required": ["query"]
                    }
                }
            },
            "lightrag_context_extract": {
                "type": "function",
                "function": {
                    "name": "lightrag_context_extract",
                    "description": "Extract only context from LightRAG without generating a response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query for context extraction"},
                            "mode": {"type": "string", "enum": ["local", "global", "hybrid", "naive", "mix"], "description": "Query mode for context extraction", "default": "hybrid"},
                            "max_entity_tokens": {"type": "integer", "description": "Maximum tokens for entity context", "default": 6000},
                            "max_relation_tokens": {"type": "integer", "description": "Maximum tokens for relationship context", "default": 8000},
                            "top_k": {"type": "integer", "description": "Number of top items to retrieve", "default": 60}
                        },
                        "required": ["query"]
                    }
                }
            },
            "lightrag_multi_hop_reasoning": {
                "type": "function",
                "function": {
                    "name": "lightrag_multi_hop_reasoning",
                    "description": "Execute multi-hop reasoning using LightRAG with AgenticStepProcessor",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Complex query requiring multi-hop reasoning"},
                            "objective": {"type": "string", "description": "Custom reasoning objective (optional)"},
                            "max_steps": {"type": "integer", "description": "Maximum internal reasoning steps (2-15)", "default": 8},
                            "timeout_seconds": {"type": "number", "description": "Maximum execution time in seconds", "default": 180.0}
                        },
                        "required": ["query"]
                    }
                }
            },
            "lightrag_sql_generation": {
                "type": "function",
                "function": {
                    "name": "lightrag_sql_generation",
                    "description": "Generate SQL queries from natural language using LightRAG context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "natural_query": {"type": "string", "description": "Natural language query to convert to SQL"},
                            "include_explanation": {"type": "boolean", "description": "Whether to include query explanation", "default": True}
                        },
                        "required": ["natural_query"]
                    }
                }
            }
        }


# Factory functions for backward compatibility
def create_athena_mcp_server(
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db",
    server_name: str = "athena-lightrag",
    server_version: str = "1.0.0"
) -> AthenaMCPServer:
    """Create Athena MCP Server instance."""
    return AthenaMCPServer(
        working_dir=working_dir,
        server_name=server_name,
        server_version=server_version
    )

def create_manual_mcp_server(
    working_dir: str = "/home/gyasis/Documents/code/PromptChain/athena-lightrag/athena_lightrag_db"
) -> ManualMCPServer:
    """Create manual MCP server instance."""
    athena_server = create_athena_mcp_server(working_dir=working_dir)
    return ManualMCPServer(athena_server)

# FastMCP 2.0 Server Information Tool
@mcp.tool()
async def get_server_info() -> Dict[str, Any]:
    """Get Athena LightRAG MCP Server information and status.
    
    Returns comprehensive server status including database connectivity, available tools,
    working directory, and query modes supported by the Athena Health EHR system.
    """
    try:
        core = await get_lightrag_core()
        db_status = core.get_database_status()
        
        return {
            "server_name": "Athena LightRAG MCP Server",
            "server_version": "2.0.0",
            "working_dir": working_dir,
            "database_status": db_status,
            "available_tools": [
                "lightrag_local_query",
                "lightrag_global_query", 
                "lightrag_hybrid_query",
                "lightrag_context_extract",
                "lightrag_multi_hop_reasoning",
                "lightrag_sql_generation",
                "get_server_info"
            ],
            "query_modes": (await get_lightrag_core()).get_available_modes(),
            "fastmcp_version": "2.0",
            "compliance_status": "FastMCP 2.0 Compliant"
        }
    
    except Exception as e:
        logger.error(f"Failed to get server info: {e}")
        return {
            "server_name": "Athena LightRAG MCP Server",
            "error": str(e),
            "compliance_status": "FastMCP 2.0 Compliant"
        }


