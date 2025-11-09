# Project Brief: Athena LightRAG

## Project Overview
Sophisticated MCP server for healthcare database analysis using LightRAG knowledge graphs and PromptChain multi-hop reasoning capabilities. The system provides intelligent access to comprehensive medical database schemas and healthcare workflow patterns from the Athena Health EHR system.

## Core Mission
Create an advanced MCP server that combines LightRAG's knowledge graph capabilities with PromptChain's multi-hop reasoning to enable sophisticated healthcare database analysis and SQL generation for the Athena Health EHR Snowflake database.

## Key Components
- **FastMCP 2.0 Compliant Server**: Module-level architecture with @mcp.tool decorators
- **LightRAG Integration**: Knowledge graph construction and querying for medical data
- **PromptChain Integration**: Multi-hop reasoning via AgenticStepProcessor
- **Snowflake Database Context**: Full Athena Health EHR schema with athena.athenaone structure
- **Healthcare Analysis Tools**: 6 specialized tools for medical data exploration

## Technical Foundation
- **Language**: Python 3.12
- **Architecture**: FastMCP 2.0 module-level structure
- **Database**: Snowflake (athena.athenaone schema)
- **Knowledge Engine**: LightRAG-HKU
- **Reasoning Engine**: PromptChain AgenticStepProcessor
- **Package Manager**: UV for dependency management

## Success Metrics
- :white_check_mark: FastMCP 2.0 compliance achieved
- :white_check_mark: All 6 healthcare analysis tools operational
- :white_check_mark: Snowflake database integration functional
- :white_check_mark: MCP inspector compatibility verified
- :hourglass_flowing_sand: Healthcare workflow optimization in progress
- :white_large_square: Production deployment planning

## Current Status: MAJOR MILESTONE ACHIEVED
**FastMCP 2.0 Compliance Successfully Implemented** - Full architectural transformation from class-based to module-level structure completed with all tools operational and MCP inspector integration working.