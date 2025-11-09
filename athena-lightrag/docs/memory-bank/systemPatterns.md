# System Patterns: Athena LightRAG MCP Server

## Architectural Patterns

### Core Architecture Pattern: Function-Based Transformation
**Pattern**: Transform interactive CLI to function-based architecture  
**Implementation**: `athena_lightrag/core.py`

```python
class AthenaLightRAG:
    """Main LightRAG wrapper with multi-hop reasoning capabilities"""
    
    def __init__(self, database_path: str, model_name: str = "openai/gpt-4o-mini"):
        # Singleton-like pattern for LightRAG instance
        # Lazy initialization for performance
        
    async def query_with_reasoning(self, query: str, strategy: str = "comprehensive"):
        # Multi-hop reasoning pattern using AgenticStepProcessor
        # Context accumulation strategies
```

### Server Architecture Pattern: FastMCP Integration
**Pattern**: Modern MCP server with dual transport support  
**Implementation**: `athena_lightrag/server.py`

```python
# Tool registration pattern
server.add_tool(query_athena_tool)
server.add_tool(query_athena_reasoning_tool) 
server.add_tool(get_database_status_tool)
server.add_tool(get_query_mode_help_tool)

# Transport abstraction pattern
if "--http" in sys.argv:
    # HTTP transport for web integration
else:
    # stdio transport for MCP standard
```

## Design Patterns

### 1. Wrapper Pattern
**Usage**: AthenaLightRAG wraps LightRAG for enhanced functionality
**Benefit**: Clean abstraction layer, easy testing, future extensibility

### 2. Strategy Pattern  
**Usage**: Context accumulation strategies (incremental, comprehensive, focused)
**Implementation**: Dynamic strategy selection in `query_with_reasoning`

### 3. Factory Pattern
**Usage**: QueryResult creation with consistent structure
**Benefit**: Type safety and consistent response format

### 4. Singleton Pattern (Modified)
**Usage**: LightRAG instance management within AthenaLightRAG
**Benefit**: Resource efficiency and consistent state

## Integration Patterns

### MCP Tool Registration Pattern
```python
@server.tool()
async def tool_name(parameter: type) -> ToolResult:
    """Tool description for MCP schema"""
    try:
        result = await core_function(parameter)
        return ToolResult(content=[TextContent(text=result)])
    except Exception as e:
        return ToolResult(content=[TextContent(text=f"Error: {str(e)}")], isError=True)
```

### Error Handling Pattern
**Consistent Pattern**: Try/catch with meaningful error messages
**Implementation**: All tools follow same error handling structure
**Benefit**: Predictable error responses for client integration

### Async/Await Pattern
**Usage**: All core functions are async-first
**Fallback**: Sync wrappers provided where needed
**Benefit**: Performance optimization and future-proofing

## Data Flow Patterns

### Query Processing Flow
1. **Input Validation**: Parameter checking and sanitization
2. **LightRAG Initialization**: Lazy loading with database verification
3. **Query Execution**: Mode-specific query processing
4. **Result Formatting**: Consistent QueryResult structure
5. **Error Handling**: Graceful degradation with informative messages

### Multi-Hop Reasoning Flow
1. **Initial Query**: Base query to establish context
2. **Context Accumulation**: Strategy-based context building
3. **Iterative Reasoning**: AgenticStepProcessor loop
4. **Result Synthesis**: Combine reasoning steps into final response
5. **Context Preservation**: Save all intermediate results

## Performance Patterns

### Lazy Initialization Pattern
- LightRAG database loaded only when needed
- Reduces startup time and memory footprint
- Graceful error handling for database issues

### Memory Management Pattern
- QueryResult objects with controlled memory footprint
- Context accumulation with configurable limits
- Database connection reuse for efficiency

### Caching Opportunities (Future Enhancement)
- Query result caching based on parameters
- Database metadata caching
- Reasoning path caching for similar queries

## Security Patterns

### Input Validation Pattern
```python
def validate_mode(mode: str) -> str:
    """Validate and normalize query mode"""
    valid_modes = ["local", "global", "hybrid", "naive"]
    if mode.lower() not in valid_modes:
        return "hybrid"  # Safe default
    return mode.lower()
```

### Safe Execution Pattern
- All external inputs validated before processing
- Database operations wrapped in error handling
- No direct SQL exposure to prevent injection

### Resource Management Pattern
- Proper async context management
- Timeout handling for long operations
- Memory limits for large result sets

## Testing Patterns

### Comprehensive Test Strategy
1. **Unit Tests**: Individual function validation
2. **Integration Tests**: End-to-end workflow testing
3. **Adversarial Tests**: Edge cases and security boundaries
4. **Performance Tests**: Load and stress testing

### Mock Pattern Usage
- Database mocking for unit tests
- API response mocking for integration tests
- Error condition simulation for robustness testing

## Configuration Patterns

### Environment-Based Configuration
```python
# .env pattern for sensitive configuration
OPENAI_API_KEY=your_key_here
DATABASE_PATH=./athena_lightrag_db
MODEL_NAME=openai/gpt-4o-mini
```

### CLI Argument Pattern
```python
# Flexible runtime configuration
python main.py --validate-only
python main.py --http --port 8000
```

### Default Value Pattern
- Intelligent defaults for all parameters
- Graceful degradation when optional config missing
- Clear documentation of all configuration options

## Future Pattern Considerations

### Extension Points
1. **Plugin Architecture**: For additional reasoning strategies
2. **Multi-Database Pattern**: For supporting multiple knowledge graphs
3. **Streaming Pattern**: For real-time response delivery
4. **Authentication Pattern**: For production security requirements

### Scalability Patterns
1. **Connection Pooling**: For database access optimization
2. **Load Balancing**: For high-availability deployments
3. **Caching Layer**: For frequently accessed queries
4. **Monitoring Pattern**: For production observability