# Interactive LightRAG Multi-Hop Reasoning Test

## Overview

This enhanced test suite provides comprehensive testing of the LightRAG multi-hop reasoning functionality through multiple integration methods:

1. **Direct LightRAG** - Tests the core functionality in isolation
2. **PromptChain MCP** - Tests through PromptChain with MCP server integration
3. **AgenticStepProcessor** - Tests with intelligent multi-step reasoning

## Features

- 🔄 **Interactive Query Loop** - Enter custom queries or use defaults
- 🔗 **PromptChain Integration** - Proper MCP server configuration
- 🧠 **Multi-Method Testing** - Compare different approaches
- 📊 **Detailed Results** - Execution time, success rates, error analysis
- 🎯 **Progressive Discovery** - Validates multi-hop reasoning steps
- 🔧 **JSON Serialization** - Tests MCP communication compatibility

## Quick Start

### Option 1: Use the Runner Script
```bash
cd testing/
./run_lightrag_test.sh
```

### Option 2: Run Directly
```bash
# From project root
uv run python testing/lightrag_multihop_isolation_test.py
```

### Option 3: Manual Setup
```bash
# Activate environment
source activate_env.sh

# Run test
cd testing/
python lightrag_multihop_isolation_test.py
```

## Test Methods

### 1. Direct LightRAG (Isolation)
- Tests the core `AgenticLightRAG` class directly
- Bypasses MCP server communication
- Validates JSON serialization
- Checks progressive discovery steps

### 2. PromptChain MCP Integration
- Uses PromptChain with proper MCP server configuration
- Tests tool discovery and execution
- Validates MCP communication protocol
- Uses UV environment isolation

### 3. AgenticStepProcessor
- Intelligent multi-step reasoning
- Advanced error analysis
- Context-aware query processing
- Healthcare domain expertise

## Interactive Features

### Query Options
1. **Default Query** - Pre-configured healthcare discovery query
2. **Custom Query** - Enter your own healthcare questions
3. **Auto Objective** - Automatically generate discovery objectives

### Example Queries
- "How does patient data flow from appointment scheduling through diagnosis to billing?"
- "What are the relationships between APPOINTMENT, PATIENT, and CHARGEDIAGNOSIS tables?"
- "Trace the complete patient journey from scheduling to payment"
- "Find all tables related to anesthesia cases and their connections"

### Session Management
- **Continuous Testing** - Run multiple queries in one session
- **Result Comparison** - See which methods work best
- **Error Analysis** - Detailed error reporting for debugging
- **Progress Tracking** - Real-time execution monitoring

## Configuration

### MCP Server Config
The test uses FastMCP 2.0 compliant configuration:
```python
{
    "id": "athena_lightrag_server",
    "type": "stdio",
    "command": "uv",
    "args": ["run", "--project", "/path/to/athena-lightrag", "fastmcp", "run"],
    "env": {"MCP_MODE": "stdio", "DEBUG": "true"},
    "read_timeout_seconds": 120
}
```

### Environment Requirements
- Python 3.12+
- UV package manager
- PromptChain framework
- AgenticLightRAG module
- OpenAI API key (for LLM calls)

## Output Analysis

### Success Indicators
- ✅ **Progressive Discovery** - Multiple reasoning steps
- ✅ **JSON Serialization** - MCP communication ready
- ✅ **Tool Discovery** - MCP tools found and accessible
- ✅ **Execution Success** - Queries complete without errors

### Common Issues
- ❌ **No Tools Discovered** - MCP server not running
- ❌ **JSON Serialization Failed** - Data structure issues
- ❌ **Single Reasoning Step** - Multi-hop not working
- ❌ **Connection Timeout** - Server startup issues

## Debugging

### Check MCP Server
```bash
# Test MCP server directly
uv run fastmcp dev
```

### Check Dependencies
```bash
# Verify UV environment
uv pip list

# Check PromptChain
python -c "import promptchain; print('PromptChain OK')"
```

### Check Database
```bash
# Verify LightRAG database files
ls -la athena_lightrag_db/
```

## Advanced Usage

### Custom Test Scenarios
```python
# Create custom tester
tester = InteractiveLightRAGTester()

# Run specific test method
result = await tester.test_direct_lightrag(query, objective)
result = await tester.test_promptchain_mcp(query, objective)
result = await tester.test_agentic_step_processor(query, objective)
```

### Batch Testing
```python
# Run multiple queries programmatically
queries = [
    "Patient appointment workflow",
    "Billing system relationships", 
    "Clinical data connections"
]

for query in queries:
    result = await tester.test_direct_lightrag(query, auto_objective)
    print(f"Query: {query} - Success: {result['success']}")
```

## Troubleshooting

### Common Solutions

1. **MCP Connection Issues**
   - Check if `uv run fastmcp dev` works
   - Verify MCP server configuration
   - Check timeout settings

2. **Import Errors**
   - Ensure you're in the UV environment
   - Check Python path configuration
   - Verify all dependencies installed

3. **JSON Serialization Issues**
   - Check result data structure
   - Look for non-serializable objects
   - Validate MCP tool responses

4. **Progressive Discovery Not Working**
   - Check LightRAG database files
   - Verify query complexity
   - Review objective configuration

## Contributing

To add new test methods or improve existing ones:

1. Add new test method to `InteractiveLightRAGTester` class
2. Update the test methods list in `run_interactive_session()`
3. Add appropriate result analysis in `display_test_result()`
4. Update this README with new features

## Support

For issues or questions:
- Check the debug logs in `testing/mcp_test_debug.log`
- Review test results in `testing/mcp_test_results.json`
- Examine the MCP server logs
- Verify environment setup with `validate_environment.py`
