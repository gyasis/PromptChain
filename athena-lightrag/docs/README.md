# Athena LightRAG MCP Server

A production-ready MCP server implementation using validated LightRAG patterns for the Athena medical database. Provides function-based query interface, multi-hop reasoning, context accumulation, and SQL generation capabilities.

## Features

### Core Components

- **LightRAG Core** (`lightrag_core.py`): Function-based query interface with all validated modes
- **Agentic Integration** (`agentic_lightrag.py`): Multi-hop reasoning using AgenticStepProcessor  
- **Context Processing** (`context_processor.py`): Context accumulation and SQL generation
- **MCP Server** (`athena_mcp_server.py`): FastMCP 2025 server with 6 validated tools

### Validated LightRAG Patterns

- **QueryParam Class**: Supports modes "local", "global", "hybrid", "naive", "mix", "bypass"
- **Function-based Interface**: `rag.query(query_text, param=QueryParam())`
- **Context Control**: `only_need_context=True` for context extraction
- **Token Management**: `max_entity_tokens`, `max_relation_tokens`, `max_total_tokens`
- **Async Initialization**: `await rag.initialize_storages()`
- **OpenAI Integration**: `openai_complete_if_cache`, `openai_embed` functions

## Installation

1. **Prerequisites**:
   ```bash
   # Ensure you have the Athena LightRAG database
   ls /home/gyasis/Documents/code/PromptChain/hybridrag/athena_lightrag_db/
   
   # Should contain: vdb_*.json, kv_store_*.json files
   ```

2. **Install Dependencies**:
   ```bash
   cd /home/gyasis/Documents/code/PromptChain/athena-lightrag
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   ```bash
   # Create .env file with API keys
   echo "OPENAI_API_KEY=your_key_here" > .env
   
   # Optional environment variables
   export ATHENA_WORKING_DIR="/path/to/athena_lightrag_db"
   export ATHENA_MODEL_NAME="gpt-4o-mini"
   export ATHENA_SERVER_PORT=8000
   ```

## Quick Start

### 1. Basic LightRAG Queries

```python
from lightrag_core import create_athena_lightrag

# Create LightRAG instance
lightrag = create_athena_lightrag()

# Query in different modes
result = lightrag.query_hybrid("What tables are related to patient appointments?")
print(result.result)

# Context-only extraction
context = lightrag.get_context_only("anesthesia case management", mode="local")
print(context)

# Async usage
result = await lightrag.query_local_async("billing workflows")
```

### 2. Multi-Hop Reasoning

```python
from agentic_lightrag import create_agentic_lightrag

# Create agentic reasoning system
agentic_rag = create_agentic_lightrag()

# Execute complex multi-hop reasoning
result = await agentic_rag.execute_multi_hop_reasoning(
    query="How do patient appointments connect to anesthesia billing?",
    objective="Analyze complex relationships in medical workflows"
)

print(f"Result: {result['result']}")
print(f"Reasoning steps: {len(result['reasoning_steps'])}")
```

### 3. Context Processing and SQL Generation

```python
from context_processor import create_sql_generator

# Create SQL generator
sql_gen = create_sql_generator()

# Generate SQL from natural language
result = await sql_gen.generate_sql_from_query(
    "Show me all patient appointments for today"
)

if result["success"]:
    print(f"Generated SQL: {result['sql']}")
    print(f"Explanation: {result['explanation']}")
```

### 4. MCP Server

```python
from athena_mcp_server import create_athena_mcp_server

# Create and run MCP server
server = create_athena_mcp_server()

# Get server info
info = await server.get_server_info()
print(f"Available tools: {info['available_tools']}")

# Run server (if FastMCP is available)
await server.run_server(host="localhost", port=8000)
```

### 5. Manual MCP Usage (without FastMCP)

```python
from athena_mcp_server import create_manual_mcp_server

# Create manual MCP server
manual_server = create_manual_mcp_server()

# Call tools directly
result = await manual_server.call_tool(
    "lightrag_hybrid_query", 
    {"query": "patient appointment workflows"}
)

print(f"Success: {result['success']}")
print(f"Result: {result['result'][:200]}...")

# Get tool schemas for integration
schemas = manual_server.get_tool_schemas()
print(f"Available tools: {list(schemas.keys())}")
```

## MCP Tools

The server provides 6 validated MCP tools:

### Core Tools (Context7 Validated)

1. **`lightrag_local_query`**
   - Local mode queries for entity relationships
   - Parameters: `query`, `top_k`, `max_entity_tokens`

2. **`lightrag_global_query`**  
   - Global mode queries for overviews
   - Parameters: `query`, `max_relation_tokens`, `top_k`

3. **`lightrag_hybrid_query`**
   - Hybrid mode combining local and global
   - Parameters: `query`, `max_entity_tokens`, `max_relation_tokens`, `top_k`

4. **`lightrag_context_extract`**
   - Context-only extraction for information gathering
   - Parameters: `query`, `mode`, `max_entity_tokens`, `max_relation_tokens`, `top_k`

### Advanced Tools

5. **`lightrag_multi_hop_reasoning`**
   - Multi-hop reasoning with AgenticStepProcessor
   - Parameters: `query`, `objective`, `max_steps`

6. **`lightrag_sql_generation`**
   - SQL generation from natural language
   - Parameters: `natural_query`, `include_explanation`

## Configuration

### Environment Variables

```bash
# Database
export ATHENA_WORKING_DIR="/path/to/athena_lightrag_db"

# LLM Settings  
export OPENAI_API_KEY="your_key"
export ATHENA_MODEL_NAME="gpt-4o-mini"
export ATHENA_EMBEDDING_MODEL="text-embedding-ada-002"
export ATHENA_MAX_ASYNC=4

# Query Defaults
export ATHENA_DEFAULT_MODE="hybrid"

# Server Settings
export ATHENA_SERVER_PORT=8000
export ATHENA_LOG_LEVEL="INFO"
```

### Programmatic Configuration

```python
from config import AthenaConfig, DatabaseConfig, LLMConfig

# Create custom configuration
config = AthenaConfig(
    database=DatabaseConfig(working_dir="/custom/path"),
    llm=LLMConfig(model_name="gpt-4", max_async=8),
)

# Validate and use
config.validate()
```

## Architecture

### Validated Integration Pattern

```python
# Function-based LightRAG query (validated from Context7)
async def lightrag_function_query(query: str, mode: str = "hybrid") -> str:
    query_param = QueryParam(
        mode=mode,
        only_need_context=False,  # or True for context only
        top_k=60,
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        response_type="Multiple Paragraphs"
    )
    result = await rag.aquery(query, param=query_param)
    return result

# AgenticStepProcessor integration with LightRAG tools
def create_lightrag_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "lightrag_local_query",
                "description": "Query LightRAG in local mode...",
                "parameters": {
                    "type": "object", 
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        }
        # Additional tools for global, hybrid, context-only modes
    ]
```

### Component Integration

1. **LightRAG Core** → Function-based query interface
2. **AgenticStepProcessor** → Multi-hop reasoning with LightRAG tools
3. **Context Processor** → Accumulates contexts across query modes  
4. **SQL Generator** → Uses accumulated context for SQL generation
5. **MCP Server** → Exposes all capabilities as standardized tools

## Error Handling

Comprehensive error handling with custom exceptions:

```python
from exceptions import (
    DatabaseNotFoundError,
    QueryExecutionError, 
    ContextExtractionError,
    SQLGenerationError,
    AgenticReasoningError,
    MCPServerError
)

try:
    result = lightrag.query_hybrid("test query")
except QueryExecutionError as e:
    print(f"Query failed: {e.message}")
    print(f"Details: {e.details}")
```

## Testing

### Basic Functionality Test

```python
# Test database connection
lightrag = create_athena_lightrag()
status = lightrag.get_database_status()
print(f"Database OK: {status['exists'] and status['initialized']}")

# Test query modes
modes = ["local", "global", "hybrid", "naive", "mix"]
for mode in modes:
    result = await lightrag.query_async("test query", mode=mode)
    print(f"{mode}: {result.error is None}")
```

### Integration Test

```python
async def test_full_pipeline():
    # Create all components
    agentic_rag = create_agentic_lightrag()
    sql_gen = create_sql_generator() 
    mcp_server = create_manual_mcp_server()
    
    # Test multi-hop reasoning
    reasoning_result = await agentic_rag.execute_multi_hop_reasoning(
        "Complex medical workflow analysis"
    )
    
    # Test SQL generation
    sql_result = await sql_gen.generate_sql_from_query(
        "Patient appointments today"
    )
    
    # Test MCP tools
    mcp_result = await mcp_server.call_tool(
        "lightrag_hybrid_query",
        {"query": "billing workflows"}
    )
    
    print(f"All tests passed: {all([
        reasoning_result['success'],
        sql_result['success'], 
        mcp_result['success']
    ])}")

asyncio.run(test_full_pipeline())
```

## Production Usage

### Performance Optimization

- **Token Management**: Configure `max_entity_tokens` and `max_relation_tokens`
- **Concurrency Control**: Set `max_async` based on your system
- **Caching**: Enable LLM response caching with `enable_cache=True`
- **Context Reuse**: Use `ContextProcessor` to accumulate and reuse contexts

### Monitoring

- **Structured Logging**: Enable with `enable_structured_logging=True`
- **Execution Metrics**: Track `execution_time` and `tokens_used` 
- **Error Tracking**: Monitor custom exception types
- **Database Health**: Check `get_database_status()` regularly

### Scaling

- **Horizontal**: Run multiple MCP server instances
- **Vertical**: Increase `max_async` and system resources
- **Database**: Use persistent storage backends for large-scale deployment

## License

MIT License - see LICENSE file for details.

## Support

For issues, questions, or contributions:
1. Check existing documentation and examples
2. Review error messages and logs
3. Test with simple queries first
4. Validate database and configuration setup