# System Patterns: Athena LightRAG

## Architecture Overview

### FastMCP 2.0 Module-Level Architecture
The system follows FastMCP 2.0 compliant module-level architecture pattern:

```python
# Module-level FastMCP instance
mcp = FastMCP()

# Tool definitions with decorators
@mcp.tool()
def lightrag_local_query(query: str) -> str:
    """Healthcare entity relationship analysis"""
    # Implementation
```

### Core Architectural Patterns

#### 1. Healthcare Knowledge Graph Pattern
- **LightRAG Core**: Knowledge graph construction from medical database metadata
- **Entity Extraction**: Automatic identification of medical entities and relationships
- **Graph Querying**: Local, global, and hybrid query strategies for different analysis needs

#### 2. Multi-Hop Reasoning Pattern
- **AgenticStepProcessor Integration**: Complex healthcare workflow analysis
- **Step-by-step Reasoning**: Breaking down complex medical queries into manageable steps
- **Context Preservation**: Maintaining medical context across reasoning chains

#### 3. Database Abstraction Pattern
- **Snowflake Integration**: Native support for athena.athenaone schema structure
- **Fully Qualified References**: All SQL uses database.schema.table format
- **Medical Domain Context**: Healthcare-specific query optimization and validation

### Tool Architecture Patterns

#### Tool Categories
1. **Query Tools**: Local, global, hybrid query capabilities
2. **Extraction Tools**: Context and metadata extraction
3. **Reasoning Tools**: Multi-hop analysis and SQL generation

#### Tool Implementation Pattern
```python
@mcp.tool()
def tool_name(parameter: str) -> str:
    """Tool description with medical context"""
    try:
        # Healthcare-specific processing
        result = process_medical_query(parameter)
        return format_medical_response(result)
    except Exception as e:
        return f"Healthcare analysis error: {str(e)}"
```

### Data Flow Architecture

#### Request Processing Flow
```
MCP Client → FastMCP Server → Tool Decorator → LightRAG Core → Snowflake Context → Response
```

#### Knowledge Graph Flow
```
Medical Metadata → LightRAG Processing → Knowledge Graph → Query Engine → Healthcare Insights
```

### Configuration Patterns

#### FastMCP 2.0 Configuration
- **Source Definition**: Module path and entrypoint specification
- **Environment Setup**: UV-based dependency management
- **Metadata Standards**: Healthcare-specific server description and versioning

#### Environment Configuration
- **API Key Management**: Secure handling of database and LLM credentials
- **Database Context**: Athena Health EHR connection parameters
- **LightRAG Settings**: Knowledge graph construction parameters

### Error Handling Patterns

#### Healthcare-Specific Error Recovery
- **Medical Context Preservation**: Errors maintain healthcare domain context
- **Graceful Degradation**: Fallback to simpler analysis when complex reasoning fails
- **User-Friendly Medical Responses**: Error messages include healthcare context

#### Critical Issue: Sync/Async Compatibility Failures
- **Timeout Errors**: "Request timed out" failures in multi-hop reasoning and SQL generation tools
- **PromptChain Integration**: AgenticStepProcessor expects fully async tool execution chains
- **Blocking Operations**: Synchronous operations in async contexts cause request timeouts
- **Error Pattern**: `lightrag_multi_hop_reasoning` and `lightrag_sql_generation` tools failing consistently

### Performance Patterns

#### Knowledge Graph Optimization
- **Lazy Loading**: Load medical entities and relationships on demand
- **Caching Strategy**: Cache frequently accessed healthcare workflows
- **Batch Processing**: Efficient handling of multiple medical queries

#### Database Query Optimization  
- **Snowflake-Specific**: Optimized for athena.athenaone schema patterns
- **Medical Workflow Awareness**: Queries optimized for common healthcare use cases
- **Connection Pooling**: Efficient database resource management

### Integration Patterns

#### MCP Ecosystem Integration
- **Standard Tool Schemas**: OpenAI-compatible function definitions
- **Inspector Compatibility**: Full support for MCP development tools  
- **Client Agnostic**: Works with Claude Code, other MCP clients

#### Healthcare System Integration
- **EHR Compatibility**: Designed for Athena Health EHR workflows
- **Medical Standards**: Follows healthcare data standards and practices
- **Compliance Ready**: Architecture supports HIPAA and medical data requirements

#### PromptChain Integration Challenges
- **AgenticStepProcessor Requirements**: Expects fully async tool execution chains
- **Async Tool Executor Pattern**: `Callable[[Any], Awaitable[str]]` signature requirement
- **Multi-Step Reasoning**: Complex workflows require non-blocking operation chains
- **Timeout Sensitivity**: Blocking operations cause request timeout failures

### Critical Investigation Patterns

#### Sync/Async Boundary Analysis
- **Identification Process**: Map all sync/async transition points in the system
- **Blocking Operation Detection**: Locate synchronous operations in async contexts
- **PromptChain Compatibility**: Validate integration with AgenticStepProcessor requirements
- **Performance Impact**: Assess blocking operations on overall system responsiveness

#### Investigation Methodology
- **Code Audit**: Comprehensive review of async/await implementation patterns
- **Timeout Analysis**: Determine appropriate timeout values for complex healthcare queries
- **Integration Testing**: Validate resolved implementation with PromptChain library
- **Performance Validation**: Ensure async operations meet healthcare workflow requirements

### Scalability Patterns

#### Horizontal Scaling
- **Stateless Design**: Tools can be distributed across multiple instances
- **Database Connection Management**: Efficient resource utilization
- **Knowledge Graph Partitioning**: Scalable medical entity management

#### Vertical Scaling
- **Memory Optimization**: Efficient knowledge graph storage
- **CPU Optimization**: Optimized reasoning algorithms
- **I/O Optimization**: Efficient database and file system access

### Security Patterns

#### Healthcare Data Security
- **Credential Management**: Secure API key and database credential handling
- **Data Isolation**: Proper separation of medical data contexts
- **Audit Trail**: Comprehensive logging for healthcare compliance

#### MCP Security
- **Input Validation**: Robust parameter validation for all tools
- **Output Sanitization**: Safe response formatting
- **Error Information Control**: Controlled error information exposure