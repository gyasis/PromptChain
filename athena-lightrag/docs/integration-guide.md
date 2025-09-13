# Athena LightRAG MCP Server - Integration Guide

## Overview

This guide covers how to integrate the Athena LightRAG MCP Server with various MCP clients, including Claude Desktop, custom applications, and testing environments.

## Prerequisites

### System Requirements
- Python 3.10 or higher
- OpenAI API key
- LightRAG database (created through data ingestion)
- FastMCP and PromptChain dependencies

### Environment Setup

1. **Install Dependencies:**
   ```bash
   cd athena-lightrag
   pip install -e .
   ```

2. **Environment Variables:**
   Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   LIGHTRAG_WORKING_DIR=./athena_lightrag_db
   LOG_LEVEL=INFO
   ```

3. **Validate Environment:**
   ```bash
   python main.py --validate-only
   ```

## Claude Desktop Integration

### Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "athena-lightrag": {
      "command": "python",
      "args": ["/absolute/path/to/athena-lightrag/main.py"],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here",
        "LIGHTRAG_WORKING_DIR": "/absolute/path/to/athena_lightrag_db"
      }
    }
  }
}
```

### UV Configuration Pattern (Recommended)

For projects using UV (Python package and project manager), use the correct UV configuration pattern:

```json
{
  "mcpServers": {
    "athena-lightrag": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/absolute/path/to/athena-lightrag",
        "fastmcp", "run"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here",
        "LIGHTRAG_WORKING_DIR": "/absolute/path/to/athena_lightrag_db",
        "DEBUG": "true"
      }
    }
  }
}
```

**Why This UV Configuration Works:**
- `--directory` flag sets UV's working directory AND finds pyproject.toml automatically
- FastMCP can locate its fastmcp.json config file in the correct directory
- Proper environment isolation is maintained without complex path manipulation
- No need for bash wrappers or shell scripts

**Common UV Mistakes to Avoid:**
- ❌ Don't use `--project` without `--directory` - this can cause path resolution issues
- ❌ Don't use bash wrappers when UV can handle execution natively
- ❌ Don't hardcode config file paths in your application when UV handles this automatically
- ❌ Don't mix `cd` commands with UV execution - use `--directory` instead

**UV vs Direct Python Execution:**
- **UV Method**: Better for development environments, handles dependencies automatically, ensures correct Python version
- **Direct Python**: Simpler for production deployments where dependencies are pre-installed

### Claude Desktop Configuration Locations

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
%APPDATA%\Claude\claude_desktop_config.json  
```

**Linux:**
```bash
~/.config/Claude/claude_desktop_config.json
```

### Testing Claude Desktop Integration

1. **Restart Claude Desktop** after configuration changes
2. **Verify Tools Available:** In Claude Desktop, type:
   ```
   What MCP tools are available?
   ```
3. **Test Basic Query:**
   ```
   Use the query_athena tool to ask: "What tables store patient information?"
   ```
4. **Test Multi-hop Reasoning:**
   ```
   Use query_athena_reasoning to analyze: "How do scheduling and billing workflows connect?"
   ```

## UV MCP Configuration Best Practices

### UV Project Structure Requirements

For UV-based MCP servers to work correctly, ensure your project has the proper structure:

```
your-project/
├── pyproject.toml          # UV finds this via --directory
├── fastmcp.json           # FastMCP config in project root
├── src/
│   └── your_module/
└── main.py                # Entry point
```

### Comprehensive UV Configuration Examples

**1. Basic UV Configuration:**
```json
{
  "mcpServers": {
    "your-server": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/project", "fastmcp", "run"],
      "env": {
        "OPENAI_API_KEY": "your-key",
        "DEBUG": "true"
      }
    }
  }
}
```

**2. UV with Custom Python Version:**
```json
{
  "mcpServers": {
    "your-server": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/project",
        "--python", "3.11",
        "fastmcp", "run"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

**3. UV with Additional Dependencies:**
```json
{
  "mcpServers": {
    "your-server": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/project", 
        "--with", "additional-package",
        "fastmcp", "run"
      ],
      "env": {
        "OPENAI_API_KEY": "your-key"
      }
    }
  }
}
```

### UV Configuration Validation

**Test Your UV Configuration:**
```bash
# Navigate to your project directory
cd /path/to/your-project

# Test UV can find pyproject.toml
uv info

# Test UV can run your MCP server
uv run --directory . fastmcp run --help

# Test with actual server execution
uv run --directory . fastmcp run --validate-only
```

**Debug UV Issues:**
```bash
# Verbose UV output
uv run --directory /path/to/project --verbose fastmcp run

# Check UV environment
uv run --directory /path/to/project python -c "import sys; print(sys.path)"

# Verify working directory
uv run --directory /path/to/project python -c "import os; print(os.getcwd())"
```

### UV vs Other Package Managers

| Manager | Command Example | Working Directory | Dependency Handling |
|---------|-----------------|-------------------|-------------------|
| **UV** | `uv run --directory /path fastmcp run` | Set by `--directory` | Automatic from pyproject.toml |
| **Poetry** | `poetry -C /path run fastmcp run` | Set by `-C` flag | From pyproject.toml |
| **Pipenv** | `cd /path && pipenv run fastmcp run` | Must cd first | From Pipfile |
| **Conda** | `conda run -p /path fastmcp run` | Current directory | From environment.yml |
| **Pip** | `python /path/main.py` | Current directory | Manual installation |

### Troubleshooting UV MCP Integration

**Common Issues and Solutions:**

1. **"Cannot find pyproject.toml"**
   ```bash
   # ❌ Wrong: Missing --directory
   uv run fastmcp run
   
   # ✅ Correct: Specify project directory
   uv run --directory /path/to/project fastmcp run
   ```

2. **"FastMCP config file not found"**
   ```bash
   # Check file exists in project root
   ls -la /path/to/project/fastmcp.json
   
   # Verify UV is using correct directory
   uv run --directory /path/to/project pwd
   ```

3. **"Module not found" errors**
   ```bash
   # Verify dependencies are installed
   uv sync --directory /path/to/project
   
   # Check if package is in editable mode
   uv pip list --directory /path/to/project
   ```

4. **Environment variable issues**
   ```json
   {
     "command": "uv",
     "args": ["run", "--directory", "/path", "fastmcp", "run"],
     "env": {
       "PYTHONPATH": "/path/to/project/src",
       "DEBUG": "true"
     }
   }
   ```

### Production UV Configuration

**For Production Deployments:**
```json
{
  "mcpServers": {
    "production-server": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/opt/your-project",
        "--no-dev",
        "--frozen",
        "fastmcp", "run"
      ],
      "env": {
        "OPENAI_API_KEY": "your-production-key",
        "LOG_LEVEL": "INFO",
        "ENVIRONMENT": "production"
      }
    }
  }
}
```

**Production Flags Explained:**
- `--no-dev`: Skip development dependencies
- `--frozen`: Use exact versions from uv.lock
- `--directory`: Explicit project path for reliability

## MCP Client SDK Integration

### Python MCP Client

```python
import asyncio
import json
from mcp_client import Client, StdioServerParameters

async def integrate_athena_client():
    # Configure server parameters - UV method (recommended)
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run", "--directory", "/path/to/athena-lightrag",
            "fastmcp", "run"
        ],
        env={"OPENAI_API_KEY": "your-key"}
    )
    
    # Alternative: Direct Python method
    # server_params = StdioServerParameters(
    #     command="python",
    #     args=["/path/to/athena-lightrag/main.py"]
    # )
    
    # Create client
    async with Client(server_params) as client:
        # Initialize session
        await client.initialize()
        
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools.tools]}")
        
        # Execute basic query
        result = await client.call_tool(
            name="query_athena",
            arguments={
                "query": "What tables handle patient appointments?",
                "mode": "hybrid",
                "top_k": 20
            }
        )
        print(f"Basic query result: {result.content}")
        
        # Execute multi-hop reasoning
        reasoning_result = await client.call_tool(
            name="query_athena_reasoning",
            arguments={
                "query": "Analyze the relationship between patient scheduling, clinical workflows, and billing systems",
                "context_strategy": "comprehensive",
                "max_reasoning_steps": 4
            }
        )
        print(f"Reasoning result: {reasoning_result.content}")
        
        # Get database status
        status_result = await client.call_tool(
            name="get_database_status",
            arguments={}
        )
        print(f"Database status: {status_result.content}")

# Run the integration
asyncio.run(integrate_athena_client())
```

### Node.js MCP Client

```javascript
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');

async function integrateAthenaClient() {
    // Create transport - UV method (recommended)
    const transport = new StdioClientTransport({
        command: 'uv',
        args: [
            'run', '--directory', '/path/to/athena-lightrag',
            'fastmcp', 'run'
        ],
        env: { OPENAI_API_KEY: 'your-key' }
    });
    
    // Alternative: Direct Python method
    // const transport = new StdioClientTransport({
    //     command: 'python',
    //     args: ['/path/to/athena-lightrag/main.py']
    // });
    
    // Create client
    const client = new Client(
        {
            name: 'athena-client',
            version: '1.0.0'
        },
        {
            capabilities: {}
        }
    );
    
    try {
        // Connect
        await client.connect(transport);
        
        // List tools
        const tools = await client.listTools();
        console.log('Available tools:', tools.tools.map(t => t.name));
        
        // Execute basic query
        const queryResult = await client.callTool({
            name: 'query_athena',
            arguments: {
                query: 'What are the main database table categories?',
                mode: 'global'
            }
        });
        console.log('Query result:', queryResult.content);
        
        // Execute multi-hop reasoning
        const reasoningResult = await client.callTool({
            name: 'query_athena_reasoning', 
            arguments: {
                query: 'How do anesthesia workflows integrate with billing and scheduling?',
                context_strategy: 'incremental',
                max_reasoning_steps: 3
            }
        });
        console.log('Reasoning result:', reasoningResult.content);
        
    } finally {
        await client.close();
    }
}

integrateAthenaClient().catch(console.error);
```

## HTTP Testing Integration

### Starting HTTP Server

```bash
# Start HTTP server on default port 8080
python main.py --http

# Start on custom port
python main.py --http --port 9090

# Start with debug logging
python main.py --http --log-level DEBUG
```

### HTTP API Testing with curl

```bash
# Basic query
curl -X POST http://localhost:8080/tools/query_athena \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What tables store patient demographic data?",
    "mode": "local",
    "top_k": 15
  }'

# Multi-hop reasoning
curl -X POST http://localhost:8080/tools/query_athena_reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze data flow from patient registration through billing",
    "context_strategy": "comprehensive",
    "max_reasoning_steps": 4
  }'

# Database status
curl -X POST http://localhost:8080/tools/get_database_status \
  -H "Content-Type: application/json" \
  -d '{}'

# Query mode help
curl -X POST http://localhost:8080/tools/get_query_mode_help \
  -H "Content-Type: application/json" \
  -d '{}'
```

### HTTP API Testing with Python requests

```python
import requests
import json

base_url = "http://localhost:8080/tools"

def test_athena_api():
    # Basic query test
    response = requests.post(f"{base_url}/query_athena", json={
        "query": "What tables are involved in patient scheduling?",
        "mode": "hybrid",
        "top_k": 25
    })
    
    print("Basic Query Result:")
    print(json.dumps(response.json(), indent=2))
    
    # Multi-hop reasoning test
    response = requests.post(f"{base_url}/query_athena_reasoning", json={
        "query": "How do clinical workflows connect to financial reporting?",
        "context_strategy": "incremental",
        "max_reasoning_steps": 3
    })
    
    print("\\nMulti-hop Reasoning Result:")
    print(json.dumps(response.json(), indent=2))
    
    # Database status test
    response = requests.post(f"{base_url}/get_database_status", json={})
    
    print("\\nDatabase Status:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_athena_api()
```

## Postman Collection

### Collection Configuration

Create a Postman collection with the following requests:

1. **Basic Query Request**
   - Method: POST
   - URL: `http://localhost:8080/tools/query_athena`
   - Body (JSON):
   ```json
   {
     "query": "{{query}}",
     "mode": "{{mode}}",
     "top_k": {{top_k}}
   }
   ```
   
2. **Multi-hop Reasoning Request**
   - Method: POST
   - URL: `http://localhost:8080/tools/query_athena_reasoning`
   - Body (JSON):
   ```json
   {
     "query": "{{reasoning_query}}",
     "context_strategy": "{{strategy}}",
     "max_reasoning_steps": {{max_steps}}
   }
   ```

3. **Database Status Request**
   - Method: POST
   - URL: `http://localhost:8080/tools/get_database_status`
   - Body (JSON): `{}`

### Environment Variables

Set up Postman environment variables:
```json
{
  "base_url": "http://localhost:8080/tools",
  "query": "What tables handle patient data?",
  "mode": "hybrid",
  "top_k": 30,
  "reasoning_query": "How do workflows connect across systems?",
  "strategy": "incremental",
  "max_steps": 4
}
```

## Custom Application Integration

### Flask Web Application

```python
from flask import Flask, request, jsonify
import asyncio
from mcp_client import Client, StdioServerParameters

app = Flask(__name__)

class AthenaClient:
    def __init__(self):
        # UV method (recommended for development)
        self.server_params = StdioServerParameters(
            command="uv",
            args=[
                "run", "--directory", "/path/to/athena-lightrag",
                "fastmcp", "run"
            ],
            env={"OPENAI_API_KEY": "your-key"}
        )
        
        # Alternative: Direct Python method
        # self.server_params = StdioServerParameters(
        #     command="python",
        #     args=["/path/to/athena-lightrag/main.py"]
        # )
    
    async def query(self, query, mode="hybrid", top_k=60):
        async with Client(self.server_params) as client:
            await client.initialize()
            result = await client.call_tool(
                name="query_athena",
                arguments={
                    "query": query,
                    "mode": mode,
                    "top_k": top_k
                }
            )
            return result.content[0].text if result.content else ""
    
    async def reasoning_query(self, query, context_strategy="incremental", max_steps=5):
        async with Client(self.server_params) as client:
            await client.initialize()
            result = await client.call_tool(
                name="query_athena_reasoning",
                arguments={
                    "query": query,
                    "context_strategy": context_strategy,
                    "max_reasoning_steps": max_steps
                }
            )
            return result.content[0].text if result.content else ""

athena_client = AthenaClient()

@app.route('/api/query', methods=['POST'])
def basic_query():
    data = request.get_json()
    query = data.get('query')
    mode = data.get('mode', 'hybrid')
    top_k = data.get('top_k', 60)
    
    try:
        result = asyncio.run(athena_client.query(query, mode, top_k))
        return jsonify({"result": result, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/reasoning', methods=['POST'])
def reasoning_query():
    data = request.get_json()
    query = data.get('query')
    strategy = data.get('context_strategy', 'incremental')
    max_steps = data.get('max_steps', 5)
    
    try:
        result = asyncio.run(athena_client.reasoning_query(query, strategy, max_steps))
        return jsonify({"result": result, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### React Frontend Integration

```javascript
// athenaApi.js
const BASE_URL = 'http://localhost:5000/api';

export class AthenaAPI {
    async basicQuery(query, mode = 'hybrid', topK = 60) {
        const response = await fetch(`${BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                mode,
                top_k: topK
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async reasoningQuery(query, contextStrategy = 'incremental', maxSteps = 5) {
        const response = await fetch(`${BASE_URL}/reasoning`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                context_strategy: contextStrategy,
                max_steps: maxSteps
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    }
}

// React Component
import React, { useState } from 'react';
import { AthenaAPI } from './athenaApi';

const AthenaQueryInterface = () => {
    const [query, setQuery] = useState('');
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);
    const [queryType, setQueryType] = useState('basic');
    
    const athenaApi = new AthenaAPI();
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        
        try {
            let response;
            if (queryType === 'basic') {
                response = await athenaApi.basicQuery(query, 'hybrid');
            } else {
                response = await athenaApi.reasoningQuery(query, 'incremental');
            }
            
            setResult(response.result);
        } catch (error) {
            setResult(`Error: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="athena-query-interface">
            <form onSubmit={handleSubmit}>
                <div>
                    <label>
                        <input
                            type="radio"
                            value="basic"
                            checked={queryType === 'basic'}
                            onChange={(e) => setQueryType(e.target.value)}
                        />
                        Basic Query
                    </label>
                    <label>
                        <input
                            type="radio"
                            value="reasoning"
                            checked={queryType === 'reasoning'}
                            onChange={(e) => setQueryType(e.target.value)}
                        />
                        Multi-hop Reasoning
                    </label>
                </div>
                
                <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your question about the medical database..."
                    rows={4}
                    cols={80}
                />
                
                <button type="submit" disabled={loading || !query.trim()}>
                    {loading ? 'Processing...' : 'Submit Query'}
                </button>
            </form>
            
            {result && (
                <div className="result">
                    <h3>Result:</h3>
                    <pre>{result}</pre>
                </div>
            )}
        </div>
    );
};

export default AthenaQueryInterface;
```

## Error Handling and Troubleshooting

### Common Integration Issues

1. **MCP Connection Failures**
   ```bash
   # Validate environment first
   python main.py --validate-only
   
   # Test HTTP mode
   python main.py --http --log-level DEBUG
   ```

2. **API Key Issues**
   ```bash
   # Check environment variable
   echo $OPENAI_API_KEY
   
   # Test API connectivity
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

3. **Database Access Issues**
   ```bash
   # Verify database path
   ls -la ./athena_lightrag_db/
   
   # Check permissions
   python -c "from pathlib import Path; print(Path('./athena_lightrag_db').exists())"
   ```

### Integration Testing

```python
import asyncio
import json
from mcp_client import Client, StdioServerParameters

async def integration_health_check():
    """Comprehensive integration health check."""
    
    # UV configuration (recommended)
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run", "--directory", "/path/to/athena-lightrag",
            "fastmcp", "run"
        ],
        env={"OPENAI_API_KEY": "your-key"}
    )
    
    # Alternative configurations:
    # Direct Python:
    # server_params = StdioServerParameters(
    #     command="python",
    #     args=["/path/to/athena-lightrag/main.py"]
    # )
    
    # Poetry:
    # server_params = StdioServerParameters(
    #     command="poetry",
    #     args=["-C", "/path/to/athena-lightrag", "run", "fastmcp", "run"]
    # )
    
    try:
        async with Client(server_params) as client:
            await client.initialize()
            
            # Test 1: List tools
            tools = await client.list_tools()
            expected_tools = ['query_athena', 'query_athena_reasoning', 
                            'get_database_status', 'get_query_mode_help']
            
            available_tools = [tool.name for tool in tools.tools]
            missing_tools = set(expected_tools) - set(available_tools)
            
            if missing_tools:
                print(f"❌ Missing tools: {missing_tools}")
                return False
            else:
                print(f"✅ All tools available: {available_tools}")
            
            # Test 2: Database status
            status_result = await client.call_tool(
                name="get_database_status",
                arguments={}
            )
            print(f"✅ Database status: {status_result.content[0].text[:100]}...")
            
            # Test 3: Basic query
            query_result = await client.call_tool(
                name="query_athena",
                arguments={
                    "query": "Test query",
                    "mode": "hybrid",
                    "top_k": 5
                }
            )
            print(f"✅ Basic query working: {len(query_result.content[0].text)} characters")
            
            # Test 4: Multi-hop reasoning
            reasoning_result = await client.call_tool(
                name="query_athena_reasoning",
                arguments={
                    "query": "Test reasoning query",
                    "max_reasoning_steps": 2
                }
            )
            print(f"✅ Reasoning query working: {len(reasoning_result.content[0].text)} characters")
            
            print("✅ All integration tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

# Run health check
if __name__ == "__main__":
    success = asyncio.run(integration_health_check())
    exit(0 if success else 1)
```

## Performance Optimization

### Connection Pooling

```python
import asyncio
from contextlib import asynccontextmanager
from mcp_client import Client, StdioServerParameters

class AthenaConnectionPool:
    def __init__(self, max_connections=3):
        self.max_connections = max_connections
        self.connections = asyncio.Queue(maxsize=max_connections)
        # UV configuration (recommended)
        self.server_params = StdioServerParameters(
            command="uv",
            args=[
                "run", "--directory", "/path/to/athena-lightrag",
                "fastmcp", "run"
            ],
            env={"OPENAI_API_KEY": "your-key"}
        )
        
        # Alternative: Direct Python
        # self.server_params = StdioServerParameters(
        #     command="python",
        #     args=["/path/to/athena-lightrag/main.py"]
        # )
    
    async def initialize(self):
        """Initialize connection pool."""
        for _ in range(self.max_connections):
            client = Client(self.server_params)
            await self.connections.put(client)
    
    @asynccontextmanager
    async def get_client(self):
        """Get a client from the pool."""
        client = await self.connections.get()
        try:
            if not client.session:
                await client.connect()
            yield client
        finally:
            await self.connections.put(client)
    
    async def close_all(self):
        """Close all connections in pool."""
        while not self.connections.empty():
            client = await self.connections.get()
            await client.close()

# Usage example
pool = AthenaConnectionPool(max_connections=3)

async def optimized_queries():
    await pool.initialize()
    
    try:
        # Execute multiple queries efficiently
        tasks = []
        queries = [
            "What tables store patient data?",
            "How do scheduling workflows work?", 
            "What are the billing table relationships?"
        ]
        
        async def execute_query(query):
            async with pool.get_client() as client:
                result = await client.call_tool(
                    name="query_athena",
                    arguments={"query": query, "mode": "hybrid"}
                )
                return result.content[0].text
        
        # Execute queries concurrently
        for query in queries:
            tasks.append(execute_query(query))
        
        results = await asyncio.gather(*tasks)
        
        for query, result in zip(queries, results):
            print(f"Query: {query}")
            print(f"Result: {result[:100]}...")
            print("-" * 50)
            
    finally:
        await pool.close_all()

# Run optimized queries
asyncio.run(optimized_queries())
```

### Caching Layer

```python
import hashlib
import json
import time
from typing import Dict, Any, Optional

class AthenaQueryCache:
    def __init__(self, ttl_seconds=3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
    
    def _get_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate cache key from tool name and arguments."""
        cache_data = {"tool": tool_name, "args": arguments}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(tool_name, arguments)
        
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            if time.time() - cached_entry["timestamp"] < self.ttl_seconds:
                return cached_entry["result"]
            else:
                # Remove expired entry
                del self.cache[cache_key]
        
        return None
    
    def set(self, tool_name: str, arguments: Dict[str, Any], result: str):
        """Store result in cache."""
        cache_key = self._get_cache_key(tool_name, arguments)
        self.cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()

# Usage with caching
class CachedAthenaClient:
    def __init__(self):
        self.cache = AthenaQueryCache(ttl_seconds=1800)  # 30 minutes
        # UV configuration (recommended)
        self.server_params = StdioServerParameters(
            command="uv",
            args=[
                "run", "--directory", "/path/to/athena-lightrag",
                "fastmcp", "run"
            ],
            env={"OPENAI_API_KEY": "your-key"}
        )
        
        # Alternative: Direct Python
        # self.server_params = StdioServerParameters(
        #     command="python", 
        #     args=["/path/to/athena-lightrag/main.py"]
        # )
    
    async def query_with_cache(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        # Check cache first
        cached_result = self.cache.get(tool_name, arguments)
        if cached_result:
            return cached_result
        
        # Execute query if not cached
        async with Client(self.server_params) as client:
            await client.initialize()
            result = await client.call_tool(name=tool_name, arguments=arguments)
            result_text = result.content[0].text
            
            # Store in cache
            self.cache.set(tool_name, arguments, result_text)
            
            return result_text
```

This comprehensive integration guide covers all major use cases for integrating the Athena LightRAG MCP Server with different client applications, testing environments, and custom implementations. The examples provide both basic integration patterns and advanced optimization techniques for production deployments.