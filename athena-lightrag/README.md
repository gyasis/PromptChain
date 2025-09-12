# Athena LightRAG MCP Server

Sophisticated MCP server for healthcare database analysis using LightRAG knowledge graphs and PromptChain multi-hop reasoning.

## Quick Start

```bash
# Activate environment
source activate_env.sh

# Run MCP server
python athena_mcp_server.py
```

## Core Files

- `athena_mcp_server.py` - Main MCP server with 6 sophisticated healthcare analysis tools
- `agentic_lightrag.py` - AgenticStepProcessor integration for multi-hop reasoning
- `lightrag_core.py` - Core LightRAG functionality
- `config.py` - Configuration management
- `pyproject.toml` - Project metadata and dependencies

## Directory Structure

```
├── athena_mcp_server.py    # Main MCP server
├── docs/                   # Documentation and reports
├── examples/               # Query examples and demos
├── prd/                    # Product requirements documents
├── testing/                # Test files and validation scripts
├── src/                    # Source code packages
└── .env.example            # Environment configuration template
```

## Tools Available

1. **lightrag_local_query** - Focused entity relationships
2. **lightrag_global_query** - Comprehensive category analysis
3. **lightrag_hybrid_query** - Combined local + global context
4. **lightrag_context_extract** - Raw metadata extraction
5. **lightrag_multi_hop_reasoning** - Complex multi-step analysis
6. **lightrag_sql_generation** - Snowflake SQL generation

## Database Context

Athena Health EHR Snowflake database structure:
- Database: `athena`
- Schema: `athenaone`
- Fully qualified tables: `athena.athenaone.TABLE_NAME`

## Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys
```

For detailed documentation, see `docs/` folder.