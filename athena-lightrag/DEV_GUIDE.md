# Athena LightRAG MCP Server - Development Guide

🏥 **Healthcare Database Analysis with FastMCP 2.0 & LightRAG**

---

## 🚀 Quick Start

### **Primary Development Command**
```bash
# REQUIRED: Source venv environment first
source .venv/bin/activate

# Then launch dev server with MCP Inspector
uv run fastmcp dev
```

### **Alternative Commands**
```bash
# REQUIRED: Source venv environment first for all commands
source .venv/bin/activate

# Specify server file directly
uv run fastmcp dev athena_mcp_server.py

# Production server
uv run fastmcp run athena_mcp_server.py

# HTTP transport testing
uv run fastmcp run athena_mcp_server.py --transport http
```

---

## 📋 Project Overview

**Database Context:** Snowflake database with `athena.athenaone` schema
**Framework:** FastMCP 2.0 for MCP server implementation  
**AI Engine:** LightRAG knowledge graphs + PromptChain multi-hop reasoning
**Target Domain:** Athena Health EHR database analysis and SQL generation

### **6 Sophisticated MCP Tools:**
1. **`lightrag_local_query`** - Focused entity/table relationship discovery
2. **`lightrag_global_query`** - Comprehensive medical workflow overviews  
3. **`lightrag_hybrid_query`** - Combined detailed + contextual analysis
4. **`lightrag_context_extract`** - Raw data dictionary metadata extraction
5. **`lightrag_multi_hop_reasoning`** - Complex medical data relationship analysis
6. **`lightrag_sql_generation`** - Validated Snowflake SQL generation

---

## 🛠️ Development Environment

### **Prerequisites**
- Python 3.10+
- UV package manager
- API keys for LLM providers (OpenAI, Anthropic)

### **Environment Setup**
```bash
# 1. Clone and navigate
cd athena-lightrag

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Activate environment (if needed)
source activate_env.sh
```

### **Environment Variables**
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional Configuration
LIGHTRAG_ENABLE_RERANK=false
LIGHTRAG_RERANK_TOP_N=10
DEBUG=true
```

---

## 🧪 Development Workflow

### **FastMCP 2.0 Development Commands**

#### **Primary Development**
```bash
# Launch dev server + MCP Inspector (RECOMMENDED)
uv run fastmcp dev

# Launch with specific configuration
uv run fastmcp dev fastmcp.json

# Add dependencies on the fly
uv run fastmcp dev athena_mcp_server.py --with pandas --with matplotlib

# Specify Python version
uv run fastmcp dev athena_mcp_server.py --python 3.12
```

#### **Testing & Validation**
```bash
# Run full test suite
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run pre-commit hooks (linting, formatting, type checking)
uv run pre-commit run --all-files

# Quick tool test
uv run python test_mcp_tools.py
```

#### **Production & Inspection**
```bash
# Production server
uv run fastmcp run athena_mcp_server.py

# Inspect server configuration
uv run fastmcp inspect athena_mcp_server.py

# HTTP transport for network testing
uv run fastmcp run athena_mcp_server.py --transport http
```

### **MCP Inspector Integration**
FastMCP 2.0 automatically integrates with MCP Inspector:
- **`uv run fastmcp dev`** launches both server and Inspector UI
- Interactive tool testing with parameter validation
- Real-time request/response debugging
- Schema validation and documentation

---

## 🏗️ Architecture Overview

### **Core Components**
```
athena_mcp_server.py        # Main MCP server with 6 tools
├── agentic_lightrag.py     # Multi-hop reasoning integration
├── lightrag_core.py        # Core LightRAG functionality  
├── config.py               # Configuration management
└── context_processor.py    # Context processing utilities
```

### **FastMCP 2.0 Configuration**
**`fastmcp.json`** - Development configuration:
```json
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    "path": "athena_mcp_server.py",
    "entrypoint": "create_athena_mcp_server"
  },
  "environment": {
    "type": "uv",
    "python": "3.12",
    "dependencies": ["fastmcp", "lightrag-hku", "promptchain @ git+https://github.com/gyasis/promptchain.git"]
  },
  "deployment": {
    "transport": "stdio",
    "log_level": "DEBUG"
  }
}
```

### **Directory Structure**
```
├── athena_mcp_server.py    # Main MCP server
├── fastmcp.json           # FastMCP configuration
├── docs/                  # Documentation and reports
├── examples/              # Query examples and demos
├── prd/                   # Product requirements
├── testing/               # Test files and validation
└── .env.example           # Environment template
```

---

## 🔧 Tool Development

### **Tool Registration Pattern**
```python
from fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("Athena LightRAG Server")

class QueryParams(BaseModel):
    query: str = Field(..., description="Healthcare database query")
    top_k: int = Field(10, description="Number of results")

@mcp.tool
async def lightrag_local_query(params: QueryParams) -> dict:
    """Focused entity relationship discovery for Athena tables."""
    # Implementation here
    return {"results": "..."}
```

### **Testing Tools**
```python
# In-memory testing (recommended)
from fastmcp import Client

async with Client(mcp) as client:
    result = await client.call_tool("lightrag_local_query", {
        "query": "athena.athenaone.PATIENT table relationships",
        "top_k": 5
    })
```

---

## 📊 Database Context

### **Snowflake Structure**
- **Database:** `athena`
- **Schema:** `athenaone`
- **Tables:** `athena.athenaone.APPOINTMENT`, `athena.athenaone.PATIENT`, etc.

### **Key Medical Data Areas**
- **Patient Management:** PATIENT, APPOINTMENT, APPOINTMENTTYPE
- **Clinical Workflow:** ANESTHESIACASE, CHARGEDIAGNOSIS
- **Billing & Revenue:** CAPPAYMENT, CHARGEEXTRAFIELDS
- **Scheduling:** APPOINTMENTTICKLER, scheduling rules
- **Operational:** Provider management, departments

### **Example Queries**
```json
{
  "tool": "lightrag_sql_generation",
  "params": {
    "natural_query": "Find all appointments for patients with diabetes in the last 3 months, include patient name and appointment type"
  }
}
```

---

## 🧪 Testing Guide

### **Testing Hierarchy**
1. **Unit Tests** - Individual tool validation
2. **Integration Tests** - MCP protocol compliance
3. **End-to-End Tests** - Full healthcare workflow scenarios
4. **Performance Tests** - LightRAG knowledge graph performance

### **Test Commands**
```bash
# Run all tests
uv run pytest

# Specific test categories
uv run pytest -m "not integration"           # Unit tests only
uv run pytest -m "integration"               # Integration tests
uv run pytest testing/test_server.py -v     # Specific test file

# Coverage reporting
uv run pytest --cov=src --cov-report=html
```

### **Manual Testing**
```bash
# Quick tool validation
uv run python test_mcp_tools.py

# Interactive server testing
uv run python athena_mcp_server.py

# MCP Inspector (visual testing)
uv run fastmcp dev
```

---

## 🚀 Deployment

### **Development Deployment**
```bash
# Development server with hot reload
uv run fastmcp dev

# HTTP server for network testing
uv run fastmcp run athena_mcp_server.py --transport http --port 8000
```

### **Production Deployment**
```bash
# Production server
uv run fastmcp run athena_mcp_server.py

# With custom configuration
uv run fastmcp run prod.fastmcp.json
```

### **MCP Client Integration**

**Claude Desktop Configuration** - Add to `~/.config/claude-desktop/config.json`:
```json
{
  "mcpServers": {
    "athena-lightrag": {
      "command": "uv",
      "args": ["run", "fastmcp", "run", "athena_mcp_server.py"],
      "cwd": "/path/to/athena-lightrag",
      "env": {
        "PATH": "/path/to/athena-lightrag/.venv/bin:/usr/bin:/bin"
      }
    }
  }
}
```

**Alternative Configuration (with shell activation):**
```json
{
  "mcpServers": {
    "athena-lightrag": {
      "command": "bash",
      "args": ["-c", "source .venv/bin/activate && uv run fastmcp run athena_mcp_server.py"],
      "cwd": "/path/to/athena-lightrag"
    }
  }
}
```

---

## 🔍 Debugging & Troubleshooting

### **Common Issues**

**1. FastMCP Import Errors**
```bash
# Reinstall FastMCP
uv sync
# or
uv add fastmcp
```

**2. LightRAG Initialization Fails**
```bash
# Check environment variables
cat .env
# Verify API keys are set
```

**3. Tool Execution Timeouts**
```bash
# Check knowledge graph size
ls -la athena_lightrag_db/
# Consider reducing top_k parameters
```

**4. Pydantic Validation Errors**
- Check parameter types in tool calls
- Verify required fields are provided
- Review tool schemas with `uv run fastmcp inspect`

### **Debug Commands**
```bash
# Verbose logging
DEBUG=true uv run fastmcp dev

# Inspect server configuration
uv run fastmcp inspect athena_mcp_server.py

# Check tool schemas
uv run python -c "
from athena_mcp_server import create_manual_mcp_server
server = create_manual_mcp_server()
schemas = server.get_tool_schemas()
print(list(schemas.keys()))
"
```

### **Performance Monitoring**
```bash
# Profile tool execution
uv run python -m cProfile athena_mcp_server.py

# Memory usage monitoring
uv run python -m tracemalloc test_mcp_tools.py
```

---

## 📚 Resources

### **Documentation**
- **FastMCP 2.0 Docs:** [https://gofastmcp.com/](https://gofastmcp.com/)
- **LightRAG Docs:** [https://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- **PromptChain Docs:** [https://github.com/gyasis/promptchain](https://github.com/gyasis/promptchain)

### **Examples & Queries**
- **Query Examples:** `examples/ATHENA_QUERY_EXAMPLES.md`
- **Postman Collection:** `examples/postman-collection.json`
- **Test Cases:** `testing/` directory

### **Configuration**
- **Environment Template:** `.env.example`
- **FastMCP Config:** `fastmcp.json`
- **Project Config:** `pyproject.toml`

---

## 🤝 Contributing

### **Development Workflow**
1. **Setup:** `uv sync`
2. **Develop:** `uv run fastmcp dev`
3. **Test:** `uv run pytest`
4. **Validate:** `uv run pre-commit run --all-files`
5. **Deploy:** `uv run fastmcp run`

### **Code Standards**
- **Type Hints:** All functions must have type hints
- **Pydantic Models:** Use for parameter validation
- **Async/Await:** Prefer async implementations
- **Documentation:** Every tool needs comprehensive docstrings
- **Testing:** Unit tests required for new tools

### **Pull Request Checklist**
- [ ] Tests pass: `uv run pytest`
- [ ] Linting passes: `uv run pre-commit run --all-files`  
- [ ] Tools validated with MCP Inspector: `uv run fastmcp dev`
- [ ] Documentation updated
- [ ] Healthcare context preserved

---

**🎯 Happy Healthcare Data Analysis with FastMCP 2.0!** 

*Last Updated: 2025-09-09*