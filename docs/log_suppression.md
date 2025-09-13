# Log Suppression in PromptChain

## Overview

This document provides comprehensive guidance on suppressing logs in the PromptChain project, particularly focusing on LiteLLM and PromptChain internal logging. Proper log suppression is essential for clean output in production environments and when you want to focus on the actual chain results rather than verbose logging information.

## Why Suppress Logs?

- **Cleaner Output**: Focus on chain results without noise
- **Production Readiness**: Reduce log verbosity in deployed systems
- **Performance**: Reduce I/O overhead from excessive logging
- **User Experience**: Provide cleaner interfaces for end users
- **Debugging Control**: Choose what level of detail you need

## Common Log Sources

### 1. **LiteLLM Logs**
- API call details
- Model response information
- Rate limiting and retry attempts
- Provider-specific logging

### 2. **HTTPX Logs**
- HTTP request/response details
- Connection pooling information
- Retry attempts and timeouts

### 3. **PromptChain Internal Logs**
- Chain execution steps
- Model switching information
- Tool execution details
- History management

### 4. **Other Package Logs**
- Various dependencies
- System-level information
- Environment warnings

## Basic Log Suppression Patterns

### **Pattern 1: Minimal Suppression (Recommended)**
```python
import logging

# Suppress only the most verbose loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
```

### **Pattern 2: Comprehensive Suppression**
```python
import logging

# Suppress all major noise sources
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("promptchain").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)  # Suppress all root logs
```

### **Pattern 3: Selective Debug Logging**
```python
import logging

# Enable debug for specific components while suppressing others
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger('litellm').setLevel(logging.DEBUG)
logging.getLogger('promptchain').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)  # Keep HTTP logs quiet
```

## Complete Examples from the Codebase

### **Example 1: Multimodal Input Test** (`tests/tests_multimodal_input.py`)
```python
import os
from datetime import datetime

def setup_logging():
    """Configure logging to both file and console."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a log file with timestamp
    log_file = os.path.join(logs_dir, f'multimodal_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will print to console
        ]
    )
    
    # Suppress specific loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('LiteLLM').setLevel(logging.WARNING)
```

### **Example 2: Agentic Step Calculator** (`tests/test_agentic_step_calculator.py`)
```python
# Set up more verbose logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Make sure LiteLLM and PromptChain loggers are at DEBUG level
logging.getLogger('litellm').setLevel(logging.DEBUG)
logging.getLogger('promptchain').setLevel(logging.DEBUG)
```

### **Example 3: Simple Agent Chat** (`examples/simple_agent_user_chat.py`)
```python
# Suppress LiteLLM and HTTPX logs
# logging.getLogger("LiteLLM").setLevel(logging.WARNING)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("promptchain").setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.WARNING)  # Optionally suppress all root logs
```

## Log Levels Reference

| Level | Numeric Value | Description | Use Case |
|-------|---------------|-------------|----------|
| `CRITICAL` | 50 | Only critical errors | Production with minimal logging |
| `ERROR` | 40 | Errors and critical issues | Production with error tracking |
| `WARNING` | 30 | Warnings and errors | **Recommended for suppression** |
| `INFO` | 20 | General information | Development and debugging |
| `DEBUG` | 10 | Detailed debugging info | Deep debugging |
| `NOTSET` | 0 | Inherit from parent | Default behavior |

## Best Practices

### **1. Set Logging Levels Early**
```python
# Do this BEFORE importing PromptChain or LiteLLM
import logging
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Then import your packages
from promptchain import PromptChain
import litellm
```

### **2. Use Specific Logger Names**
```python
# Good: Target specific loggers
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Avoid: Suppressing everything unless necessary
logging.getLogger().setLevel(logging.WARNING)  # Use sparingly
```

### **3. Consider File Logging for Debug**
```python
import logging
import os

# Set up file logging for debugging while keeping console clean
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # Console output
    ]
)

# Suppress console noise
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
```

### **4. Environment-Based Configuration**
```python
import logging
import os

# Adjust logging based on environment
if os.getenv('ENVIRONMENT') == 'production':
    # Production: minimal logging
    logging.getLogger('LiteLLM').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('promptchain').setLevel(logging.WARNING)
else:
    # Development: more verbose
    logging.getLogger('LiteLLM').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
```

## Common Use Cases

### **Production Deployment**
```python
import logging

# Minimal logging for production
logging.getLogger('LiteLLM').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('promptchain').setLevel(logging.WARNING)
```

### **Development and Testing**
```python
import logging

# Moderate logging for development
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('promptchain').setLevel(logging.INFO)
```

### **Deep Debugging**
```python
import logging

# Detailed logging for debugging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('LiteLLM').setLevel(logging.DEBUG)
logging.getLogger('promptchain').setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)  # Keep HTTP quiet
```

### **Clean User Interface**
```python
import logging

# Clean output for user-facing applications
logging.getLogger('LiteLLM').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('promptchain').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)  # Suppress everything
```

## Troubleshooting

### **Logs Still Appearing?**
1. **Check import order**: Set logging levels before importing packages
2. **Restart kernel**: In Jupyter/IPython, restart the kernel
3. **Check logger names**: Verify the exact logger names being used
4. **Environment variables**: Some packages respect environment-based logging

### **Too Quiet?**
1. **Increase levels**: Use `logging.INFO` or `logging.DEBUG`
2. **Target specific loggers**: Only suppress what you need
3. **Use file logging**: Log to files while keeping console clean

### **Performance Issues?**
1. **Use WARNING level**: Balance between noise and information
2. **Avoid DEBUG in production**: DEBUG level can be expensive
3. **Consider async logging**: For high-performance applications

## Integration with PromptChain Configuration

### **Using PromptChain's Built-in Verbosity**
```python
from promptchain import PromptChain

# PromptChain has its own verbose control
chain = PromptChain(
    models=["gpt-4"],
    instructions=["Your prompt: {input}"],
    verbose=False  # This reduces PromptChain's internal logging
)
```

### **Combining with Log Suppression**
```python
import logging
from promptchain import PromptChain

# Suppress external package logs
logging.getLogger('LiteLLM').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Create chain with minimal internal logging
chain = PromptChain(
    models=["gpt-4"],
    instructions=["Your prompt: {input}"],
    verbose=False  # Minimal PromptChain logging
)
```

## Summary

Effective log suppression in PromptChain involves:

1. **Targeting specific loggers** (`LiteLLM`, `httpx`, `promptchain`)
2. **Setting appropriate levels** (usually `WARNING` for suppression)
3. **Configuring early** (before imports)
4. **Using PromptChain's built-in controls** (`verbose=False`)
5. **Considering your use case** (production vs. development vs. debugging)

The examples in the codebase demonstrate these patterns effectively, providing a solid foundation for implementing log suppression in your own PromptChain applications.

## Related Documentation

- [MCP Tool Hijacker Documentation](mcp_tool_hijacker.md)
- [PromptChain Core Usage](simple_agent_user_chat.md)
- [Model Management](model_management.md)
- [Testing and Validation](../tests/README.md)
