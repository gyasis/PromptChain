# Technical Context: Athena LightRAG

## Technology Stack

### Core Technologies
- **Python 3.12**: Primary language with modern async/await support
- **FastMCP 2.0**: Latest MCP protocol implementation with module-level architecture
- **UV Package Manager**: Modern Python dependency management and project tooling
- **LightRAG-HKU**: Advanced knowledge graph construction and querying
- **PromptChain**: Multi-hop reasoning via AgenticStepProcessor integration

### Database & Storage
- **Snowflake**: Primary database platform
- **Database Context**: `athena` database, `athenaone` schema
- **Table Structure**: Fully qualified references (`athena.athenaone.TABLE_NAME`)
- **Medical Domain**: 100+ healthcare tables with comprehensive metadata

### Development Environment

#### Package Management
```bash
# UV-based dependency management
uv add fastmcp
uv add lightrag-hku  
uv add pydantic
uv add "promptchain @ git+https://github.com/gyasis/promptchain.git"
```

#### Development Commands
```bash
# Activate environment
source activate_env.sh

# Run FastMCP development server
uv run fastmcp dev

# Test MCP tools
python test_mcp_tools.py
```

### Configuration Architecture

#### FastMCP Configuration (`fastmcp.json`)
```json
{
  "$schema": "https://gofastmcp.com/public/schemas/fastmcp.json/v1.json",
  "source": {
    "path": "athena_mcp_server.py",
    "entrypoint": "mcp"
  },
  "environment": {
    "type": "uv",
    "python": "3.12",
    "dependencies": [...]
  },
  "metadata": {
    "name": "Athena LightRAG MCP Server",
    "version": "2.0.0"
  }
}
```

#### Environment Configuration (`.env`)
```bash
# Database credentials
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password

# LLM API keys  
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Dependencies & Constraints

#### Core Dependencies
- **fastmcp**: MCP protocol implementation (>=2.0.0)
- **lightrag-hku**: Knowledge graph engine
- **pydantic**: Data validation and serialization
- **promptchain**: Multi-hop reasoning (git dependency)

#### System Requirements
- **Python**: >=3.12 for modern async support
- **Memory**: Minimum 4GB RAM for knowledge graph operations
- **Storage**: SSD recommended for LightRAG database operations
- **Network**: Stable connection for Snowflake and LLM API access

### Architecture Constraints

#### FastMCP 2.0 Requirements
- Module-level server definition (not class-based)
- @mcp.tool decorators for all tools
- Proper JSON schema compliance
- UV environment support

#### Healthcare Domain Constraints
- **Medical Context**: All responses must maintain healthcare context
- **Data Privacy**: No medical data logging or persistence
- **Query Validation**: All SQL must use proper athena.athenaone references
- **Error Handling**: Healthcare-friendly error messages

### Performance Characteristics

#### Knowledge Graph Performance
- **Initialization**: ~2-5 seconds for medical metadata loading
- **Query Response**: <1 second for most healthcare queries
- **Memory Usage**: ~500MB-2GB depending on graph size
- **Concurrent Queries**: Up to 10 simultaneous healthcare analyses

#### Database Performance
- **Connection Time**: ~1-3 seconds for Snowflake connection
- **Query Execution**: Variable based on medical data complexity
- **Result Processing**: ~100ms-1s for typical healthcare queries
- **Connection Pooling**: Optimized for healthcare workflow patterns

### Development Patterns

#### Code Organization
```
athena-lightrag/
├── athena_mcp_server.py     # Main FastMCP 2.0 server
├── lightrag_core.py         # Knowledge graph operations
├── agentic_lightrag.py      # PromptChain integration  
├── config.py               # Configuration management
├── fastmcp.json           # FastMCP 2.0 configuration
└── memory-bank/           # Project documentation
```

#### Tool Development Pattern
```python
@mcp.tool()
def lightrag_healthcare_tool(
    query: str,
    context: str = "medical"
) -> str:
    """Healthcare-specific analysis tool"""
    # Input validation
    # Medical context processing  
    # LightRAG knowledge graph query
    # Healthcare response formatting
    return medical_response
```

### Testing Strategy

#### Unit Testing
- Individual tool functionality validation
- Medical query processing accuracy
- Error handling robustness
- Response format compliance

#### Integration Testing
- FastMCP 2.0 server functionality
- MCP inspector compatibility
- Database connection reliability
- Knowledge graph consistency

#### Healthcare Domain Testing
- Medical workflow accuracy
- SQL generation validation
- Clinical context preservation
- Healthcare data privacy compliance

### Deployment Considerations

#### Local Development
- UV environment isolation
- FastMCP development server (`uv run fastmcp dev`)
- Real-time code reloading
- MCP inspector integration

#### Production Requirements
- **Security**: Secure credential management
- **Scalability**: Connection pooling and caching
- **Monitoring**: Healthcare query logging and analytics
- **Compliance**: HIPAA and medical data handling requirements

### Known Technical Limitations

#### LightRAG Constraints
- Knowledge graph size limitations for large medical datasets
- Query complexity bounds for multi-hop reasoning
- Memory usage scaling with medical entity count

#### FastMCP 2.0 Constraints
- Module-level architecture requirements
- Tool parameter type restrictions
- Error handling standardization

#### Healthcare Domain Constraints
- Medical data privacy requirements
- Snowflake query performance limitations
- Healthcare workflow complexity bounds

#### CRITICAL: Sync/Async Compatibility Issues
- **Timeout Failures**: `lightrag_multi_hop_reasoning` and `lightrag_sql_generation` tools failing with "Request timed out" errors
- **PromptChain Integration**: AgenticStepProcessor requires fully async tool execution chains
- **Blocking Operations**: Synchronous operations in async contexts causing request timeouts
- **Investigation Status**: Active analysis of sync/async boundaries and blocking operations

### Future Technical Directions
- Advanced medical NLP integration
- Real-time healthcare event processing
- Enhanced medical workflow optimization
- Distributed knowledge graph architectures
- Healthcare AI model integration