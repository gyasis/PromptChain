#!/usr/bin/env python3
"""
Athena LightRAG Package
======================
Core Athena LightRAG MCP Server implementation with validated patterns.

Author: Athena LightRAG System
Date: 2025-09-08
"""

from .lightrag_core import (
    AthenaLightRAGCore,
    QueryResult,
    QueryMode,
    LightRAGConfig,
    create_athena_lightrag
)

from .agentic_lightrag import (
    AgenticLightRAG,
    LightRAGToolsProvider,
    MultiHopContext,
    create_agentic_lightrag
)

from .context_processor import (
    ContextProcessor,
    SQLGenerator,
    ContextType,
    ContextFragment,
    AccumulatedContext,
    create_context_processor,
    create_sql_generator
)

from .athena_mcp_server import (
    AthenaMCPServer,
    ManualMCPServer,
    create_athena_mcp_server,
    create_manual_mcp_server
)

__version__ = "1.0.0"
__author__ = "Athena LightRAG System"

__all__ = [
    # Core components
    "AthenaLightRAGCore",
    "QueryResult", 
    "QueryMode",
    "LightRAGConfig",
    "create_athena_lightrag",
    
    # Agentic components
    "AgenticLightRAG",
    "LightRAGToolsProvider", 
    "MultiHopContext",
    "create_agentic_lightrag",
    
    # Context processing
    "ContextProcessor",
    "SQLGenerator",
    "ContextType",
    "ContextFragment",
    "AccumulatedContext",
    "create_context_processor",
    "create_sql_generator",
    
    # MCP server
    "AthenaMCPServer",
    "ManualMCPServer",
    "create_athena_mcp_server",
    "create_manual_mcp_server"
]