# Athena LightRAG MCP Server - API Documentation

> **Complete API reference for the Athena LightRAG Multi-hop Reasoning MCP Server**

## 📋 Table of Contents

1. [Overview](#overview)
2. [MCP Tools](#mcp-tools)
3. [Core Functions](#core-functions)
4. [Data Types](#data-types)
5. [Error Handling](#error-handling)
6. [Usage Examples](#usage-examples)
7. [Performance Guidelines](#performance-guidelines)
8. [Integration Patterns](#integration-patterns)

---

## Overview

The Athena LightRAG MCP Server provides intelligent access to medical database knowledge through three main interfaces:

- **MCP Tools**: FastMCP-compatible tools for client integration
- **Core Functions**: Direct async function access
- **Server Interface**: HTTP/stdio transport support

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │────│   FastMCP       │────│   Core Module   │
│                 │    │   Server        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │   LightRAG      │
                                               │   Database      │
                                               └─────────────────┘
                                                        │
                                               ┌─────────────────┐
                                               │  PromptChain    │
                                               │  AgenticStep    │
                                               └─────────────────┘
```

---

## MCP Tools

### `query_athena`

Execute basic queries against the Athena medical database.

**Signature:**
```typescript
query_athena(
  query: string,
  mode?: "local" | "global" | "hybrid" | "naive" = "hybrid",
  context_only?: boolean = false,
  top_k?: number = 60
): Promise<string>
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Your question about the medical database |
| `mode` | string | ❌ | "hybrid" | Query strategy (see [Query Modes](#query-modes)) |
| `context_only` | boolean | ❌ | false | Return only context without LLM generation |
| `top_k` | number | ❌ | 60 | Number of top results to retrieve |

**Returns:** `Promise<string>` - The answer to your query

**Example:**
```json
{
  "tool": "query_athena",
  "arguments": {
    "query": "What tables are related to patient appointments?",
    "mode": "hybrid",
    "top_k": 30
  }
}
```

**Response:**
```
Based on the database structure, several tables are related to patient appointments:

1. **appointment_slots** - Contains available appointment time slots
2. **patient_schedules** - Links patients to their scheduled appointments  
3. **provider_availability** - Manages healthcare provider availability
4. **appointment_types** - Defines different types of appointments
...
```

---

### `query_athena_reasoning`

Execute complex multi-hop reasoning queries with intelligent decomposition.

**Signature:**
```typescript
query_athena_reasoning(
  query: string,
  context_strategy?: "incremental" | "comprehensive" | "focused" = "incremental",
  mode?: "local" | "global" | "hybrid" | "naive" = "hybrid",
  max_reasoning_steps?: number = 5
): Promise<string>
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Complex question requiring multi-step analysis |
| `context_strategy` | string | ❌ | "incremental" | How to accumulate context (see [Context Strategies](#context-strategies)) |
| `mode` | string | ❌ | "hybrid" | LightRAG query mode for each step |
| `max_reasoning_steps` | number | ❌ | 5 | Maximum reasoning iterations (1-10) |

**Returns:** `Promise<string>` - Comprehensive answer from multi-hop reasoning

**Example:**
```json
{
  "tool": "query_athena_reasoning",
  "arguments": {
    "query": "How do anesthesia workflows connect to patient scheduling and billing systems?",
    "context_strategy": "comprehensive",
    "max_reasoning_steps": 7
  }
}
```

**Response:**
```
The relationship between anesthesia workflows, patient scheduling, and billing systems involves several interconnected processes:

### 1. Pre-operative Scheduling Integration
- Anesthesia consultations are scheduled through the patient scheduling system
- Pre-op assessments determine anesthesia requirements and complexity
- Scheduling system allocates appropriate time slots based on procedure complexity

### 2. Workflow Coordination
- Anesthesia teams coordinate with surgical schedules
- Equipment and medication preparation is synchronized with case scheduling
- Provider assignments are managed through the scheduling system

### 3. Documentation and Billing Integration
- Anesthesia services are documented using specific CPT codes
- Time-based billing calculations are integrated with case duration
- Billing system captures anesthesia units, base units, and qualifying circumstances
...
```

---

### `get_database_status`

Retrieve database information and health status.

**Signature:**
```typescript
get_database_status(): Promise<string>
```

**Parameters:** None

**Returns:** `Promise<string>` - Formatted database status information

**Example:**
```json
{
  "tool": "get_database_status",
  "arguments": {}
}
```

**Response:**
```
Database Path: athena_lightrag_db
Database Exists: True
Initialized: True
Total Size: 117.0 MB
Total Files: 14
Database Files: kv_store_llm_response_cache.json, kv_store_full_entities.json, ...
```

---

### `get_query_mode_help`

Get detailed guidance on query modes and strategies.

**Signature:**
```typescript
get_query_mode_help(): string
```

**Parameters:** None

**Returns:** `string` - Comprehensive guide to query modes and strategies

**Example:**
```json
{
  "tool": "get_query_mode_help", 
  "arguments": {}
}
```

---

## Core Functions

### `query_athena_basic`

Direct access to basic query functionality.

**Signature:**
```python
async def query_athena_basic(
    query: str,
    mode: str = "hybrid",
    context_only: bool = False,
    top_k: int = 60
) -> str
```

**Usage:**
```python
from athena_lightrag.core import query_athena_basic

result = await query_athena_basic(
    query="What is the patient appointment workflow?",
    mode="hybrid",
    top_k=50
)
```

### `query_athena_multi_hop`

Direct access to multi-hop reasoning functionality.

**Signature:**
```python
async def query_athena_multi_hop(
    query: str,
    context_strategy: str = "incremental",
    mode: str = "hybrid",
    max_steps: int = 5
) -> str
```

**Usage:**
```python
from athena_lightrag.core import query_athena_multi_hop

result = await query_athena_multi_hop(
    query="Analyze the complete patient care workflow from admission to discharge",
    context_strategy="comprehensive",
    max_steps=8
)
```

### `get_athena_database_info`

Direct access to database information.

**Signature:**
```python
async def get_athena_database_info() -> str
```

**Usage:**
```python
from athena_lightrag.core import get_athena_database_info

info = await get_athena_database_info()
```

---

## Data Types

### QueryResult

Internal data structure for query results.

```python
@dataclass
class QueryResult:
    result: str
    query_mode: str
    context_only: bool
    reasoning_steps: Optional[List[str]] = None
    accumulated_context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### AthenaLightRAG

Main interface class for direct integration.

```python
class AthenaLightRAG:
    def __init__(
        self,
        working_dir: str = "./athena_lightrag_db",
        api_key: Optional[str] = None,
        reasoning_model: str = "gpt-4o-mini",
        max_reasoning_steps: int = 5
    )
    
    async def basic_query(...) -> QueryResult
    async def multi_hop_reasoning_query(...) -> QueryResult
    async def get_database_info() -> Dict[str, Any]
```

---

## Query Modes

### Local Mode
- **Best for:** Specific entity relationships, detailed technical questions
- **Use when:** Asking about particular tables, fields, or specific components
- **Example:** "What are the columns in the patient_appointments table?"

### Global Mode
- **Best for:** High-level overviews, system-wide analysis, summaries
- **Use when:** Asking about overall architecture, general patterns, broad topics
- **Example:** "What are the main categories of tables in the database?"

### Hybrid Mode (Recommended)
- **Best for:** Most questions, combines local detail with global context
- **Use when:** Unsure which mode to use, or need both specific and general info
- **Example:** "How does patient scheduling integrate with billing systems?"

### Naive Mode
- **Best for:** Simple keyword searches, when other modes are too complex
- **Use when:** Looking for basic text matches without graph reasoning
- **Example:** Simple searches that don't require relationship understanding

---

## Context Strategies

### Incremental Strategy
- **Behavior:** Builds context step-by-step through the reasoning chain
- **Best for:** Sequential analysis, following logical progressions
- **Performance:** Moderate resource usage
- **Example:** Understanding workflow stages in order

### Comprehensive Strategy
- **Behavior:** Gathers broad context from multiple perspectives
- **Best for:** Complex system analysis, understanding interconnections
- **Performance:** Higher resource usage, longer execution time
- **Example:** Complete system integration analysis

### Focused Strategy
- **Behavior:** Targets specific areas with deep analysis
- **Best for:** Specialized technical questions, detailed investigations
- **Performance:** Moderate resource usage, focused execution
- **Example:** Deep dive into specific subsystem functionality

---

## Error Handling

### Common Error Scenarios

#### Invalid Parameters
```json
{
  "error": "Invalid mode 'invalid_mode', defaulting to 'hybrid'",
  "result": "..."
}
```

#### Database Connection Issues
```json
{
  "error": "LightRAG database not found at ./athena_lightrag_db"
}
```

#### API Key Issues
```json
{
  "error": "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
}
```

#### Tool Function Errors
```json
{
  "error": "Tool function 'query_lightrag' is not available."
}
```

### Error Recovery

The system implements graceful error recovery:

1. **Parameter Validation:** Invalid parameters are corrected with defaults
2. **Graceful Degradation:** Multi-hop reasoning falls back to general responses
3. **Retry Logic:** Temporary failures are handled with appropriate retries
4. **Logging:** All errors are logged for debugging

---

## Performance Guidelines

### Query Optimization

| Query Type | Expected Time | Memory Usage | Best Practices |
|------------|---------------|---------------|----------------|
| Basic Query | 1-5 seconds | ~50MB | Use appropriate `top_k` values |
| Multi-hop Reasoning | 10-30 seconds | ~100MB | Limit `max_reasoning_steps` |
| Database Status | <1 second | ~10MB | Cache results when possible |

### Resource Management

```python
# Good: Use context managers for multiple queries
async with AthenaLightRAG() as athena:
    result1 = await athena.basic_query("Query 1")
    result2 = await athena.basic_query("Query 2")

# Avoid: Creating multiple instances
athena1 = AthenaLightRAG()
athena2 = AthenaLightRAG()  # Wasteful
```

### Batch Processing

```python
# Efficient batch processing
queries = ["Query 1", "Query 2", "Query 3"]
tasks = [query_athena_basic(q) for q in queries]
results = await asyncio.gather(*tasks)
```

---

## Usage Examples

### Basic Integration

```python
import asyncio
from athena_lightrag.core import query_athena_basic

async def main():
    result = await query_athena_basic(
        "What are the main patient management tables?",
        mode="hybrid"
    )
    print(result)

asyncio.run(main())
```

### MCP Client Integration

```javascript
// Using MCP client
const client = new MCPClient(serverConfig);

await client.initialize();

const result = await client.callTool("query_athena", {
  query: "How do appointments relate to billing?",
  mode: "hybrid"
});

console.log(result);
```

### Advanced Multi-hop Analysis

```python
from athena_lightrag.core import query_athena_multi_hop

async def analyze_workflow():
    analysis = await query_athena_multi_hop(
        query="Analyze the complete revenue cycle from patient registration to payment processing",
        context_strategy="comprehensive",
        max_steps=10
    )
    return analysis

result = asyncio.run(analyze_workflow())
```

### Custom Server Integration

```python
from athena_lightrag.core import AthenaLightRAG
import uvicorn
from fastapi import FastAPI

app = FastAPI()
athena = AthenaLightRAG()

@app.post("/query")
async def custom_query(query: str, mode: str = "hybrid"):
    result = await athena.basic_query(query, mode=mode)
    return {"result": result.result, "mode": result.query_mode}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Integration Patterns

### Claude Desktop Integration

```json
{
  "mcpServers": {
    "athena-lightrag": {
      "command": "python",
      "args": ["/path/to/athena-lightrag/main.py"],
      "env": {
        "OPENAI_API_KEY": "your_key_here"
      }
    }
  }
}
```

### HTTP Client Integration

```python
import requests

response = requests.post("http://localhost:8080/mcp/tools/call", json={
    "tool": "query_athena",
    "arguments": {
        "query": "What tables handle patient demographics?",
        "mode": "local"
    }
})

result = response.json()
```

### Async Client Pattern

```python
import aiohttp
import asyncio

async def query_server(query: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8080/mcp/tools/call",
            json={
                "tool": "query_athena_reasoning",
                "arguments": {"query": query}
            }
        ) as response:
            return await response.json()

# Usage
result = await query_server("Analyze patient care workflows")
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Tool not found | MCP server not registered | Check server configuration |
| Empty results | Invalid query or database | Verify database and query syntax |
| Timeout errors | Complex multi-hop query | Reduce `max_reasoning_steps` |
| Memory errors | Large result sets | Reduce `top_k` parameter |
| API key errors | Missing environment variable | Set `OPENAI_API_KEY` |

### Debug Mode

Enable detailed logging:
```bash
python main.py --log-level DEBUG
```

### Health Check

```python
from athena_lightrag.core import get_athena_database_info

async def health_check():
    try:
        info = await get_athena_database_info()
        return "healthy" in info.lower()
    except Exception:
        return False
```

---

## Version Information

- **API Version:** 1.0.0
- **MCP Compatibility:** 2025.1
- **FastMCP Version:** 2.12.2+
- **LightRAG Version:** 1.4.7+
- **PromptChain Version:** 0.2.4+

---

*This documentation covers all public APIs and integration patterns for the Athena LightRAG MCP Server. For additional support, see the main README.md file.*