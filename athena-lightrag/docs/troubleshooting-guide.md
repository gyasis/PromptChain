# Athena LightRAG MCP Server - Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide covers common issues, diagnostic procedures, and solutions for the Athena LightRAG MCP Server. Use this guide to quickly identify and resolve problems in development, testing, and production environments.

## Quick Diagnostic Checklist

### Pre-troubleshooting Steps

1. **Environment Validation:**
   ```bash
   python main.py --validate-only
   ```

2. **Basic Connectivity Test:**
   ```bash
   python main.py --http --port 8080
   curl -X POST http://localhost:8080/tools/get_database_status -d '{}'
   ```

3. **Integration Test:**
   ```bash
   python integration_test.py
   ```

## Common Issues and Solutions

### 1. Database Issues

#### Issue: Database Not Found
```
❌ Error: "LightRAG database not found at ./athena_lightrag_db. Run ingestion first."
```

**Diagnostic Steps:**
```bash
# Check if database directory exists
ls -la ./athena_lightrag_db/

# Check environment variable
echo $LIGHTRAG_WORKING_DIR

# Verify database files
find ./athena_lightrag_db -name "*.json" -type f
```

**Solutions:**

1. **Create Database Directory:**
   ```bash
   mkdir -p ./athena_lightrag_db
   # Run your data ingestion process here
   ```

2. **Set Correct Path:**
   ```bash
   export LIGHTRAG_WORKING_DIR="/absolute/path/to/athena_lightrag_db"
   ```

3. **Verify Database Structure:**
   ```bash
   # Database should contain these files:
   # entities.json, relationships.json, communities.json
   ls -la ./athena_lightrag_db/
   ```

#### Issue: Database Initialization Failure
```
❌ Error: "Failed to initialize LightRAG storages"
```

**Diagnostic Steps:**
```bash
# Check file permissions
ls -la ./athena_lightrag_db/
stat ./athena_lightrag_db/

# Check available disk space
df -h ./athena_lightrag_db/

# Verify file integrity
python -c "
import json
try:
    with open('./athena_lightrag_db/entities.json', 'r') as f:
        json.load(f)
    print('✅ entities.json is valid')
except Exception as e:
    print(f'❌ entities.json error: {e}')
"
```

**Solutions:**

1. **Fix Permissions:**
   ```bash
   chmod -R 755 ./athena_lightrag_db/
   chown -R $USER:$USER ./athena_lightrag_db/
   ```

2. **Recreate Database:**
   ```bash
   # Backup existing database
   mv ./athena_lightrag_db ./athena_lightrag_db.backup
   
   # Re-run data ingestion process
   # (Specific to your data ingestion workflow)
   ```

3. **Check Disk Space:**
   ```bash
   # Ensure at least 1GB free space
   df -h ./athena_lightrag_db/
   ```

#### Issue: Database Corruption
```
❌ Error: "JSON decode error" or "Malformed database file"
```

**Diagnostic Steps:**
```bash
# Check JSON file validity
for file in ./athena_lightrag_db/*.json; do
    echo "Checking $file..."
    python -c "
import json
try:
    with open('$file', 'r') as f:
        json.load(f)
    print('✅ Valid JSON')
except Exception as e:
    print(f'❌ JSON error: {e}')
"
done

# Check file sizes
du -sh ./athena_lightrag_db/*
```

**Solutions:**

1. **Restore from Backup:**
   ```bash
   cp -r ./athena_lightrag_db.backup ./athena_lightrag_db
   ```

2. **Partial Recovery:**
   ```bash
   # Identify corrupted files and replace with empty JSON
   echo '{}' > ./athena_lightrag_db/corrupted_file.json
   ```

3. **Complete Rebuild:**
   ```bash
   rm -rf ./athena_lightrag_db
   # Re-run complete data ingestion
   ```

### 2. API Key Issues

#### Issue: Missing API Key
```
❌ Error: "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
```

**Diagnostic Steps:**
```bash
# Check environment variable
echo $OPENAI_API_KEY

# Check .env file
cat .env | grep OPENAI_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

**Solutions:**

1. **Set Environment Variable:**
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

2. **Create .env File:**
   ```bash
   echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
   ```

3. **Verify API Key Format:**
   ```bash
   # OpenAI keys start with 'sk-' and are 51 characters long
   echo $OPENAI_API_KEY | wc -c  # Should be 52 (including newline)
   ```

#### Issue: API Key Invalid or Expired
```
❌ Error: "OpenAI API error: Invalid API key"
```

**Diagnostic Steps:**
```bash
# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.openai.com/v1/models

# Check API quota
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/dashboard/billing/usage
```

**Solutions:**

1. **Generate New API Key:**
   - Visit https://platform.openai.com/api-keys
   - Generate new key
   - Update environment variable

2. **Check Account Status:**
   - Verify account is active
   - Check billing and usage limits
   - Ensure sufficient credits

3. **Test with Different Model:**
   ```python
   # Test if specific model access is the issue
   import openai
   openai.api_key = "your-key"
   models = openai.models.list()
   print([m.id for m in models if 'gpt' in m.id])
   ```

### 3. MCP Connection Issues

#### Issue: MCP Client Connection Failure
```
❌ Error: "Failed to establish MCP connection" or "Stdio transport error"
```

**Diagnostic Steps:**
```bash
# Test server startup
python main.py --validate-only

# Test HTTP mode
python main.py --http --port 8080
curl http://localhost:8080/health  # If health endpoint exists

# Check Python path and imports
python -c "
import sys
sys.path.insert(0, '.')
try:
    from athena_lightrag.server import main
    print('✅ Import successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Verify FastMCP installation
python -c "
try:
    from fastmcp import FastMCP
    print('✅ FastMCP available')
except ImportError:
    print('❌ FastMCP not installed')
"
```

**Solutions:**

1. **Fix Python Path:**
   ```bash
   # Add current directory to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   
   # Or use absolute paths in MCP config
   "command": "python",
   "args": ["/absolute/path/to/athena-lightrag/main.py"]
   ```

2. **Install Missing Dependencies:**
   ```bash
   pip install fastmcp
   pip install -e .  # Install in development mode
   ```

3. **Test Stdio Transport:**
   ```bash
   # Create simple test script
   echo '{"method": "ping"}' | python main.py
   ```

#### Issue: Claude Desktop Integration Failure
```
❌ Error: "Tool not available" or "MCP server not responding"
```

**Diagnostic Steps:**
```bash
# Check Claude Desktop config location
cat ~/.config/Claude/claude_desktop_config.json

# Validate JSON syntax
python -c "
import json
with open('~/.config/Claude/claude_desktop_config.json') as f:
    config = json.load(f)
    print('✅ Valid JSON configuration')
"

# Test server independently
python main.py --http --port 8080
# In separate terminal:
curl -X POST http://localhost:8080/tools/get_database_status -d '{}'
```

**Solutions:**

1. **Fix Configuration Path:**
   ```json
   {
     "mcpServers": {
       "athena-lightrag": {
         "command": "python",
         "args": ["/absolute/path/to/main.py"],
         "env": {
           "OPENAI_API_KEY": "your-key",
           "LIGHTRAG_WORKING_DIR": "/absolute/path/to/db"
         }
       }
     }
   }
   ```

2. **Restart Claude Desktop:**
   ```bash
   # Kill Claude Desktop process
   pkill -f "Claude"
   
   # Restart Claude Desktop
   open -a "Claude"  # macOS
   # Or restart manually on other platforms
   ```

3. **Check Logs:**
   ```bash
   # Enable debug logging
   export LOG_LEVEL=DEBUG
   python main.py --http --log-level DEBUG
   ```

### 4. Performance Issues

#### Issue: Slow Query Response
```
❌ Issue: Queries taking >60 seconds or timing out
```

**Diagnostic Steps:**
```bash
# Monitor resource usage
top -p $(pgrep -f "python.*main.py")

# Check database size
du -sh ./athena_lightrag_db/

# Test with simplified parameters
curl -X POST http://localhost:8080/tools/query_athena \
  -H "Content-Type: application/json" \
  -d '{"query": "simple test", "mode": "naive", "top_k": 5}'

# Monitor memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

**Solutions:**

1. **Optimize Query Parameters:**
   ```json
   {
     "query": "your question",
     "mode": "naive",        // Fastest mode
     "top_k": 10,           // Reduce from default 60
     "max_reasoning_steps": 2  // Reduce for reasoning queries
   }
   ```

2. **Resource Optimization:**
   ```bash
   # Increase available memory
   export PYTHONMALLOC=malloc
   ulimit -v 4194304  # 4GB virtual memory limit
   
   # Use more efficient mode
   export TOKENIZERS_PARALLELISM=false
   ```

3. **Database Optimization:**
   ```bash
   # Check for database fragmentation
   ls -la ./athena_lightrag_db/
   
   # Consider rebuilding if database is very large
   # (>500MB may indicate inefficient storage)
   ```

#### Issue: Memory Usage Too High
```
❌ Issue: Process consuming >2GB RAM or getting killed
```

**Diagnostic Steps:**
```bash
# Monitor memory over time
watch -n 5 'ps aux | grep "python.*main.py"'

# Check for memory leaks
python -c "
import psutil, time
process = psutil.Process()
for i in range(10):
    print(f'Iteration {i}: {process.memory_info().rss / 1024 / 1024:.1f} MB')
    time.sleep(2)
"

# Profile memory usage
pip install memory_profiler
python -m memory_profiler main.py --validate-only
```

**Solutions:**

1. **Implement Memory Limits:**
   ```python
   # Add to core.py initialization
   import resource
   resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, -1))  # 2GB limit
   ```

2. **Optimize Context Management:**
   ```json
   {
     "top_k": 20,              // Reduce context size
     "max_entity_tokens": 3000, // Lower token limits
     "max_relation_tokens": 4000
   }
   ```

3. **Garbage Collection:**
   ```python
   # Add periodic cleanup
   import gc
   gc.collect()  # Force garbage collection
   ```

### 5. Multi-hop Reasoning Issues

#### Issue: Reasoning Queries Failing
```
❌ Error: "Multi-hop reasoning failed" or "AgenticStepProcessor error"
```

**Diagnostic Steps:**
```bash
# Test PromptChain installation
python -c "
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
print('✅ PromptChain imports successful')
"

# Test with minimal reasoning
curl -X POST http://localhost:8080/tools/query_athena_reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "query": "simple test", 
    "max_reasoning_steps": 1
  }'

# Check model availability
python -c "
import openai
client = openai.OpenAI()
models = client.models.list()
gpt_models = [m.id for m in models.data if 'gpt' in m.id]
print(f'Available GPT models: {gpt_models}')
"
```

**Solutions:**

1. **Reduce Complexity:**
   ```json
   {
     "query": "simplified question",
     "context_strategy": "focused",
     "max_reasoning_steps": 2,
     "mode": "naive"
   }
   ```

2. **Check Model Access:**
   ```python
   # Test specific model
   from openai import OpenAI
   client = OpenAI()
   
   response = client.chat.completions.create(
     model="gpt-4o-mini",  # Default reasoning model
     messages=[{"role": "user", "content": "test"}]
   )
   print("✅ Model access working")
   ```

3. **Fallback Configuration:**
   ```python
   # Modify core.py to use fallback model
   reasoning_model = "gpt-3.5-turbo"  # Instead of gpt-4o-mini
   ```

#### Issue: Reasoning Steps Timeout
```
❌ Issue: "Reasoning timeout after 120 seconds"
```

**Diagnostic Steps:**
```bash
# Test individual reasoning components
python -c "
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

processor = AgenticStepProcessor(
    objective='simple test',
    max_internal_steps=1,
    model_name='gpt-4o-mini'
)
print('✅ AgenticStepProcessor created successfully')
"

# Monitor processing time
time curl -X POST http://localhost:8080/tools/query_athena_reasoning \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "max_reasoning_steps": 1}'
```

**Solutions:**

1. **Reduce Reasoning Complexity:**
   ```json
   {
     "max_reasoning_steps": 2,    // Reduce from default 5
     "context_strategy": "focused", // More targeted analysis
     "mode": "local"              // Faster retrieval
   }
   ```

2. **Optimize Model Selection:**
   ```python
   # Use faster model for reasoning
   reasoning_model = "gpt-4o-mini"  # Faster than gpt-4
   ```

3. **Implement Timeouts:**
   ```python
   # Add timeout handling in core.py
   import asyncio
   
   try:
       result = await asyncio.wait_for(
           reasoning_chain.process_prompt_async(query),
           timeout=90  # 90 second timeout
       )
   except asyncio.TimeoutError:
       return "Reasoning timeout - try simpler query"
   ```

### 6. HTTP Transport Issues

#### Issue: HTTP Server Won't Start
```
❌ Error: "Failed to start MCP server" or "Port already in use"
```

**Diagnostic Steps:**
```bash
# Check port availability
netstat -tlnp | grep 8080
lsof -i :8080

# Test alternative port
python main.py --http --port 8081

# Check firewall rules
iptables -L | grep 8080  # Linux
pfctl -sr | grep 8080    # macOS
```

**Solutions:**

1. **Use Different Port:**
   ```bash
   python main.py --http --port 9090
   ```

2. **Kill Existing Process:**
   ```bash
   # Find process using port
   lsof -ti:8080 | xargs kill -9
   
   # Or restart system service
   sudo systemctl restart networking
   ```

3. **Check Host Binding:**
   ```bash
   # Try binding to all interfaces
   python main.py --http --host 0.0.0.0 --port 8080
   ```

#### Issue: HTTP Requests Failing
```
❌ Error: "Connection refused" or "Request timeout"
```

**Diagnostic Steps:**
```bash
# Test server health
curl -v http://localhost:8080/

# Check request format
curl -X POST http://localhost:8080/tools/query_athena \
  -H "Content-Type: application/json" \
  -v \
  -d '{"query": "test"}'

# Monitor server logs
python main.py --http --log-level DEBUG
```

**Solutions:**

1. **Fix Request Format:**
   ```bash
   # Ensure proper JSON and headers
   curl -X POST http://localhost:8080/tools/query_athena \
     -H "Content-Type: application/json" \
     -H "Accept: application/json" \
     -d '{"query": "test query", "mode": "hybrid"}'
   ```

2. **Check Content-Type:**
   ```python
   # Ensure proper content type in requests
   headers = {'Content-Type': 'application/json'}
   response = requests.post(url, json=data, headers=headers)
   ```

3. **Test with Minimal Request:**
   ```bash
   # Test with simplest possible request
   curl -X POST http://localhost:8080/tools/get_database_status \
     -H "Content-Type: application/json" \
     -d '{}'
   ```

## Advanced Troubleshooting

### Debug Mode Setup

1. **Enable Comprehensive Logging:**
   ```bash
   export LOG_LEVEL=DEBUG
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   python main.py --log-level DEBUG
   ```

2. **Create Debug Configuration:**
   ```python
   # debug_config.py
   import logging
   
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('athena_debug.log'),
           logging.StreamHandler()
       ]
   )
   ```

3. **Profile Performance:**
   ```bash
   pip install cProfile
   python -m cProfile -o profile_output.prof main.py --validate-only
   python -m pstats profile_output.prof
   ```

### Integration Testing Suite

```python
#!/usr/bin/env python3
"""
Comprehensive troubleshooting test suite
"""

import asyncio
import json
import sys
import traceback
from pathlib import Path

async def run_diagnostic_tests():
    """Run comprehensive diagnostic tests."""
    
    tests = [
        ("Environment Check", test_environment),
        ("Database Access", test_database), 
        ("API Connectivity", test_api_key),
        ("MCP Tools", test_mcp_tools),
        ("Basic Queries", test_basic_queries),
        ("Multi-hop Reasoning", test_reasoning),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"🔍 Running {test_name}...")
        try:
            result = await test_func()
            results[test_name] = {"status": "PASS", "details": result}
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            results[test_name] = {"status": "FAIL", "error": str(e)}
            print(f"❌ {test_name}: FAILED - {e}")
            traceback.print_exc()
    
    # Generate report
    print("\n" + "="*50)
    print("DIAGNOSTIC REPORT")
    print("="*50)
    
    for test_name, result in results.items():
        status = result["status"]
        print(f"{test_name}: {status}")
        if status == "FAIL":
            print(f"  Error: {result['error']}")
    
    # Save detailed report
    with open("diagnostic_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: diagnostic_report.json")
    
    failed_tests = [name for name, result in results.items() if result["status"] == "FAIL"]
    return len(failed_tests) == 0

async def test_environment():
    """Test environment setup."""
    import os
    from pathlib import Path
    
    checks = []
    
    # Check required environment variables
    if os.getenv("OPENAI_API_KEY"):
        checks.append("✅ OPENAI_API_KEY set")
    else:
        checks.append("❌ OPENAI_API_KEY missing")
    
    # Check database directory
    db_path = Path(os.getenv("LIGHTRAG_WORKING_DIR", "./athena_lightrag_db"))
    if db_path.exists():
        checks.append(f"✅ Database directory exists: {db_path}")
    else:
        checks.append(f"❌ Database directory missing: {db_path}")
    
    # Check Python imports
    try:
        from athena_lightrag.core import AthenaLightRAG
        checks.append("✅ Core imports successful")
    except ImportError as e:
        checks.append(f"❌ Import error: {e}")
    
    return "\n".join(checks)

async def test_database():
    """Test database access."""
    from athena_lightrag.core import get_athena_instance
    
    instance = get_athena_instance()
    await instance._ensure_initialized()
    
    db_info = await instance.get_database_info()
    
    return f"Database initialized: {db_info['initialized']}"

async def test_api_key():
    """Test OpenAI API connectivity."""
    import openai
    
    client = openai.OpenAI()
    models = client.models.list()
    
    available_models = [m.id for m in models.data if 'gpt' in m.id]
    
    return f"Available models: {available_models[:3]}..."

async def test_mcp_tools():
    """Test MCP tool registration."""
    from athena_lightrag.server import mcp
    
    tools = mcp.get_tools()
    tool_names = list(tools.keys())
    
    expected = ["query_athena", "query_athena_reasoning", 
               "get_database_status", "get_query_mode_help"]
    
    missing = set(expected) - set(tool_names)
    
    if missing:
        raise Exception(f"Missing tools: {missing}")
    
    return f"All {len(tool_names)} tools registered"

async def test_basic_queries():
    """Test basic query functionality."""
    from athena_lightrag.core import query_athena_basic
    
    result = await query_athena_basic(
        query="test query",
        mode="naive",
        top_k=5
    )
    
    return f"Query successful, result length: {len(result)}"

async def test_reasoning():
    """Test multi-hop reasoning."""
    from athena_lightrag.core import query_athena_multi_hop
    
    result = await query_athena_multi_hop(
        query="simple reasoning test",
        max_steps=1
    )
    
    return f"Reasoning successful, result length: {len(result)}"

async def test_performance():
    """Test performance characteristics."""
    import time
    from athena_lightrag.core import query_athena_basic
    
    start_time = time.time()
    
    await query_athena_basic(
        query="performance test",
        mode="naive", 
        top_k=5
    )
    
    duration = time.time() - start_time
    
    if duration > 30:
        raise Exception(f"Query too slow: {duration:.2f}s")
    
    return f"Performance acceptable: {duration:.2f}s"

if __name__ == "__main__":
    success = asyncio.run(run_diagnostic_tests())
    sys.exit(0 if success else 1)
```

### Log Analysis Tools

```bash
#!/bin/bash
# log_analyzer.sh - Analyze Athena LightRAG logs

echo "🔍 Athena LightRAG Log Analysis"
echo "================================"

LOG_FILE="athena_debug.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo "Run with --log-level DEBUG to generate logs"
    exit 1
fi

echo "📊 Log Statistics:"
echo "- Total lines: $(wc -l < $LOG_FILE)"
echo "- Error count: $(grep -c "ERROR" $LOG_FILE)"
echo "- Warning count: $(grep -c "WARNING" $LOG_FILE)"
echo "- Info count: $(grep -c "INFO" $LOG_FILE)"

echo -e "\n🚨 Recent Errors:"
grep "ERROR" $LOG_FILE | tail -5

echo -e "\n⚠️  Recent Warnings:"
grep "WARNING" $LOG_FILE | tail -5

echo -e "\n📈 Performance Metrics:"
grep "Query completed" $LOG_FILE | tail -5

echo -e "\n🔧 Common Issues:"
if grep -q "API key" $LOG_FILE; then
    echo "- API key issues detected"
fi

if grep -q "Database not found" $LOG_FILE; then
    echo "- Database access issues detected"
fi

if grep -q "timeout" $LOG_FILE; then
    echo "- Timeout issues detected"
fi

if grep -q "Memory" $LOG_FILE; then
    echo "- Memory issues detected"
fi
```

## Recovery Procedures

### Complete System Recovery

1. **Full Environment Reset:**
   ```bash
   # Backup current state
   cp -r ./athena_lightrag_db ./backup_$(date +%Y%m%d)
   
   # Clean environment
   unset OPENAI_API_KEY
   unset LIGHTRAG_WORKING_DIR
   
   # Reinstall dependencies
   pip uninstall athena-lightrag -y
   pip install -e .
   
   # Restore environment
   source .env
   
   # Validate
   python main.py --validate-only
   ```

2. **Database Recovery:**
   ```bash
   # If database is corrupted, restore from backup
   rm -rf ./athena_lightrag_db
   cp -r ./backup_20250108 ./athena_lightrag_db
   
   # Or rebuild from scratch
   # (Run your data ingestion process)
   ```

3. **Configuration Recovery:**
   ```bash
   # Reset MCP configuration
   cp claude_desktop_config.json.backup ~/.config/Claude/claude_desktop_config.json
   
   # Restart services
   pkill -f "Claude"
   # Restart Claude Desktop manually
   ```

This troubleshooting guide should help you quickly identify and resolve most common issues with the Athena LightRAG MCP Server. For persistent problems not covered here, enable debug logging and run the diagnostic test suite to gather detailed information for further analysis.