# Technical Context: Athena LightRAG MCP Server

## Technology Stack

### Core Technologies
- **Python**: 3.11+ (specified in `.python-version`)
- **Package Manager**: UV (modern Python package management)
- **Dependency Management**: `pyproject.toml` with comprehensive dependency specifications
- **Environment**: Virtual environment (`.venv/`) with full isolation

### Primary Dependencies

#### LLM and Reasoning Framework
- **PromptChain**: GitHub integration (`git+https://github.com/gyasis/PromptChain.git`)
  - AgenticStepProcessor for multi-hop reasoning
  - Tool calling and function integration
  - Context management and conversation history
- **LightRAG**: Knowledge graph queries and reasoning
- **LiteLLM**: Unified LLM API access (OpenAI, Anthropic, etc.)

#### MCP Server Framework
- **FastMCP**: 2.12.2 with full dependency tree
  - stdio transport for MCP standard compliance
  - HTTP transport for web application integration
  - Tool registration and schema management
  - Async/await support for performance

#### Supporting Libraries
- **asyncio**: Async programming support
- **pydantic**: Data validation and serialization (via dependencies)
- **dotenv**: Environment variable management
- **tiktoken**: Token counting for context management

### Database Technology
- **LightRAG Database**: Pre-built medical knowledge graph
  - Size: 117MB compressed medical knowledge
  - Entities: 1,865 medical entities
  - Relationships: 3,035 medical relationships
  - Location: `./athena_lightrag_db/`

## Development Environment

### Setup Requirements
```bash
# Python version management
pyproject.toml specifies python = ">=3.11"
.python-version contains "3.11"

# Package installation
uv pip install -e .  # Development mode
uv pip install -r requirements.txt  # Alternative method

# Environment variables
.env file with OPENAI_API_KEY required
```

### Project Structure
```
athena-lightrag/
├── athena_lightrag/           # Main package
│   ├── __init__.py           # Package initialization
│   ├── core.py               # Core LightRAG functions
│   └── server.py             # FastMCP server
├── athena_lightrag_db/        # LightRAG database (117MB)
├── main.py                   # CLI entry point
├── test_server.py            # Basic functionality tests
├── integration_test.py       # Comprehensive testing
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Compatibility requirements
└── .env.example              # Configuration template
```

## API Integrations

### LLM Provider Integration
- **Primary**: OpenAI GPT-4o-mini (configurable)
- **Framework**: LiteLLM for provider abstraction
- **Configuration**: Environment variable based (`OPENAI_API_KEY`)
- **Fallback**: Support for multiple providers through PromptChain

### MCP Protocol Integration
- **Version**: MCP 2025 compliant
- **Transports**: 
  - stdio: Standard MCP transport for Claude Desktop
  - HTTP: REST API for web applications and testing
- **Tool Schema**: OpenAI-compatible function definitions
- **Error Handling**: Structured ToolResult responses

## Performance Characteristics

### Initialization Performance
- **Database Load**: 2-3 seconds for 117MB knowledge graph
- **Memory Usage**: ~200MB with full database loaded
- **Cold Start**: Complete server initialization in <5 seconds

### Query Performance
- **Basic Queries**: 1-5 seconds (local/global/hybrid/naive modes)
- **Multi-hop Reasoning**: 10-30 seconds (depends on complexity and steps)
- **Concurrent Queries**: 90%+ success rate with 3 simultaneous queries
- **Memory Growth**: +50-100MB per active reasoning session

### Scalability Constraints
- **Memory**: Single instance recommended for <10 concurrent complex queries
- **Token Limits**: Context window managed by PromptChain execution history
- **API Limits**: Bound by OpenAI API rate limits and quotas

## Configuration Management

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_key_here

# Optional with defaults
DATABASE_PATH=./athena_lightrag_db
MODEL_NAME=openai/gpt-4o-mini
MAX_REASONING_STEPS=10
```

### CLI Configuration
```bash
# Validation mode
python main.py --validate-only

# HTTP server mode  
python main.py --http --port 8000

# Standard MCP stdio mode
python main.py
```

### Runtime Configuration
- **Query Modes**: local, global, hybrid, naive (intelligent defaults)
- **Context Strategies**: incremental, comprehensive, focused
- **Reasoning Parameters**: max_steps, context_accumulation, strategy selection

## Security Considerations

### Input Validation
- **Parameter Validation**: All MCP tool inputs validated and sanitized
- **Query Mode Validation**: Only valid modes accepted with safe defaults
- **Context Size Limits**: Prevented through PromptChain execution history management

### API Security
- **No Direct Database Access**: LightRAG abstraction prevents SQL injection
- **Error Message Sanitization**: No sensitive information in error responses
- **Resource Limits**: Timeout handling and memory constraints

### Deployment Security
- **Environment Isolation**: Virtual environment prevents dependency conflicts
- **API Key Management**: Environment variable based, not hardcoded
- **Transport Security**: stdio transport inherits Claude Desktop security model

## Testing Infrastructure

### Test Categories
1. **Unit Tests** (`test_server.py`): Core function validation
2. **Integration Tests** (`integration_test.py`): End-to-end workflows
3. **Adversarial Tests**: Edge cases, security boundaries, performance limits
4. **Performance Tests**: Load testing and resource validation

### Test Environment
- **Mock Patterns**: Database mocking for unit tests
- **Real Database Tests**: Integration tests use actual LightRAG database
- **Error Simulation**: Comprehensive error condition testing
- **Performance Validation**: Memory and timing measurements

## Deployment Options

### MCP Client Integration
- **Transport**: stdio (standard MCP)
- **Target**: Claude Desktop native integration
- **Configuration**: MCP server configuration in Claude settings

### HTTP API Deployment
- **Transport**: HTTP REST API
- **Target**: Web applications and custom integrations
- **Configuration**: `--http --port 8000` CLI arguments

### Direct Python Integration
- **Import**: `from athena_lightrag import AthenaLightRAG`
- **Target**: Custom Python applications
- **Configuration**: Direct instantiation with parameters

### Container Deployment (Future)
- **Docker Ready**: Project structure supports containerization
- **Dependencies**: All dependencies specified in pyproject.toml
- **Configuration**: Environment variable based configuration

## Known Technical Constraints

### LightRAG Limitations
- **Database Size**: 117MB may be large for some deployment scenarios
- **Query Complexity**: Very complex queries may exceed LLM context windows
- **Update Frequency**: Static knowledge graph requires periodic updates

### PromptChain Integration
- **Async Complexity**: Proper async/await handling required throughout
- **Token Management**: Context window management crucial for long reasoning chains
- **Tool Integration**: MCP tool naming must avoid conflicts with PromptChain tools

### Performance Constraints  
- **Single Threaded**: LightRAG database access is not thread-safe
- **Memory Usage**: Large knowledge graph requires significant memory
- **API Dependencies**: Performance bound by external LLM API response times

## Future Technical Enhancements

### Immediate Opportunities
1. **Caching Layer**: Query result caching for improved performance
2. **Streaming Responses**: Real-time response delivery for long operations
3. **Connection Pooling**: Database access optimization

### Advanced Enhancements
1. **Multi-Database Support**: Support for multiple knowledge graphs
2. **Custom Reasoning Strategies**: User-defined context accumulation patterns
3. **High Availability**: Clustering and load balancing support
4. **Monitoring Integration**: Production observability and metrics collection