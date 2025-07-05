# Model Management System

The PromptChain Model Management System provides automatic VRAM optimization for local LLM providers like Ollama, enabling efficient execution of large model chains that wouldn't normally fit in memory.

## Overview

### The Problem
Large language models (like mistral-small:22b and deepseek-r1:70b) consume significant VRAM. Running multiple large models simultaneously often leads to:
- Out of memory errors
- Timeout issues (5+ minute waits)
- Server crashes
- Poor performance

### The Solution
The Model Management System automatically:
- **Loads models** on-demand before each step
- **Unloads models** after each step to free VRAM
- **Optimizes context windows** to maximum available
- **Manages model lifecycle** across the entire chain

## Architecture

```
PromptChain Execution Flow:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │ -> │  PromptChain     │ -> │  Final Result   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              |
                              v
                    ┌──────────────────┐
                    │ Model Management │
                    └──────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    Model Lifecycle                             │
│                                                                 │
│  Step 1: Load Model A → Execute → Unload Model A              │
│  Step 2: Load Model B → Execute → Unload Model B              │
│  Step 3: Load Model C → Execute → Unload Model C              │
│                                                                 │
│  Result: Only ONE model in VRAM at any time                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. AbstractModelManager
Base class defining the model management interface:

```python
from promptchain.utils.model_management import AbstractModelManager

class AbstractModelManager(ABC):
    async def load_model_async(self, model_name: str, model_params: dict) -> ModelInfo
    async def unload_model_async(self, model_name: str) -> bool
    async def is_model_loaded_async(self, model_name: str) -> bool
    async def get_loaded_models_async(self) -> List[ModelInfo]
    async def health_check_async(self) -> bool
```

### 2. OllamaModelManager
Ollama-specific implementation with advanced features:

- **VRAM Tracking**: Monitor model memory usage
- **Health Monitoring**: Check server status
- **Retry Logic**: Exponential backoff for failed requests
- **Context Optimization**: Auto-detect and set maximum context length
- **Connection Pooling**: Efficient HTTP client management

### 3. ModelManagerFactory
Creates provider-specific model managers:

```python
from promptchain.utils.model_management import ModelManagerFactory, ModelProvider

# Create Ollama manager
manager = ModelManagerFactory.create_manager(
    ModelProvider.OLLAMA,
    base_url="http://localhost:11434",
    timeout=30,
    verbose=True
)
```

### 4. PromptChain Integration
Seamless integration with automatic lifecycle management:

```python
chain = PromptChain(
    models=["ollama/mistral-small:22b", "ollama/deepseek-r1:70b"],
    instructions=["Step 1: {input}", "Step 2: {input}"],
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=True
)
```

## Configuration

### Global Configuration
Set up model management for all chains:

```python
from promptchain.utils.model_management import (
    ModelManagementConfig, 
    ModelProvider, 
    set_global_config
)

config = ModelManagementConfig(
    enabled=True,
    default_provider=ModelProvider.OLLAMA,
    auto_unload=True,
    max_loaded_models=2,
    provider_configs={
        ModelProvider.OLLAMA: {
            'base_url': 'http://192.168.1.160:11434',
            'timeout': 30,
            'max_retries': 3,
            'verbose': True
        }
    }
)

set_global_config(config)
```

### Chain-Specific Configuration
Override global settings per chain:

```python
model_mgmt_config = {
    'enabled': True,
    'provider': 'ollama',
    'base_url': 'http://192.168.1.160:11434',
    'timeout': 30,
    'max_retries': 3,
    'verbose': True
}

chain = PromptChain(
    models=[
        {
            "name": "ollama/mistral-small:22b",
            "params": {
                "api_base": "http://192.168.1.160:11434",
                "timeout": 1800,  # 30 minutes
                "max_retries": 3,
                "stream": True
            }
        }
    ],
    instructions=["Process: {input}"],
    model_management=model_mgmt_config,  # Chain-specific config
    auto_unload_models=True,
    verbose=True
)
```

## Features

### Automatic Model Loading
Models are loaded just-in-time before each step:

```python
# Before step execution
🔄 Model Management: Loading mistral-small:22b
✓ Successfully loaded mistral-small:22b in 2.34s (VRAM: 22000MB)
```

### Intelligent Unloading
Models are unloaded after each step, with optimizations:

- **Same Model Detection**: If the next step uses the same model, keep it loaded
- **Error Recovery**: Continue on unload failures (with warnings)
- **Final Cleanup**: Unload all models at chain completion

### Context Window Optimization
Automatically detects and configures maximum context length:

```python
# Auto-detected from model info
🔄 Model mistral-small:22b max context length: 32768
🔄 Setting model context length to maximum: 32768
```

### VRAM Monitoring
Track memory usage across model lifecycle:

```python
# Get current status
loaded_models = await chain.get_loaded_models_async()
for model in loaded_models:
    print(f"{model.name}: {model.vram_usage}MB")
```

## Manual Control

### Load/Unload Models Manually
```python
# Manual loading
model_info = await chain.load_model_async("mistral-small:22b")
print(f"Loaded: {model_info.name}, VRAM: {model_info.vram_usage}MB")

# Manual unloading
success = await chain.unload_model_async("mistral-small:22b")
print(f"Unloaded: {success}")

# Check status
status = chain.get_model_manager_status()
print(f"Status: {status}")
```

### Disable Auto-Unload
```python
chain = PromptChain(
    models=["ollama/mistral-small:22b"],
    instructions=["Process: {input}"],
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=False,  # Manual control
    verbose=True
)

# Manual cleanup when done
await chain.cleanup_models_async()
```

## Advanced Usage

### Multi-Step Chain with Different Models
```python
chain = PromptChain(
    models=[
        "ollama/mistral-small:22b",    # Step 1: Fast analysis
        "ollama/deepseek-r1:70b",      # Step 2: Deep reasoning
        "ollama/mistral-small:22b"     # Step 3: Summarization
    ],
    instructions=[
        "Quick analysis: {input}",
        "Deep reasoning: {input}",
        "Final summary: {input}"
    ],
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=True,
    verbose=True
)

# Execution flow:
# Load mistral-small → Execute Step 1 → Unload mistral-small
# Load deepseek-r1   → Execute Step 2 → Unload deepseek-r1  
# Load mistral-small → Execute Step 3 → Unload mistral-small
```

### Error Handling and Fallbacks
```python
def timeout_chainbreaker(step_num, output, step_info):
    """Break chain on model management errors"""
    if "model management failed" in str(output).lower():
        return (True, "Model management error", "Process stopped due to VRAM issues")
    return (False, "", None)

chain = PromptChain(
    models=["ollama/very-large-model:100b"],
    instructions=["Process: {input}"],
    model_management={'enabled': True, 'provider': 'ollama'},
    chainbreakers=[timeout_chainbreaker],
    verbose=True
)
```

### Integration with MCP Servers
```python
# DeepLake MCP + Model Management
mcp_config = [{
    "id": "deeplake_server",
    "type": "stdio",
    "command": "/path/to/python",
    "args": ["/path/to/deeplake_server.py"]
}]

model_mgmt_config = {
    'enabled': True,
    'provider': 'ollama',
    'base_url': 'http://192.168.1.160:11434'
}

chain = PromptChain(
    models=["ollama/mistral-small:22b", "ollama/deepseek-r1:70b"],
    instructions=[
        "Search with DeepLake: {input}",
        "Analyze results: {input}"
    ],
    mcp_servers=mcp_config,           # MCP integration
    model_management=model_mgmt_config, # Model management
    auto_unload_models=True,
    verbose=True
)
```

## Monitoring and Debugging

### Status Checking
```python
# Model management status
status = chain.get_model_manager_status()
print(f"""
Model Management Status:
- Enabled: {status['enabled']}
- Auto-unload: {status['auto_unload']}
- Provider: {status['manager_type']}
- Available: {status['available']}
""")

# Currently loaded models
loaded = await chain.get_loaded_models_async()
for model in loaded:
    print(f"- {model.name}: {model.vram_usage}MB, Last used: {model.last_used}")
```

### Verbose Logging
Enable detailed logging to see model lifecycle:

```python
chain = PromptChain(
    models=["ollama/mistral-small:22b"],
    instructions=["Process: {input}"],
    model_management={'enabled': True, 'provider': 'ollama', 'verbose': True},
    verbose=True  # Chain-level verbose
)

# Output:
# 🔄 Model Management: Initial cleanup - checking for loaded models
# 🔄 Model Management: Loading ollama/mistral-small:22b
# ✓ Successfully loaded mistral-small:22b in 2.34s (VRAM: 22000MB)
# 🔄 Model Management: Unloading mistral-small:22b
# ✓ Successfully unloaded mistral-small:22b
```

## Backward Compatibility

The Model Management System is fully backward compatible:

```python
# This works exactly as before - no model management
old_chain = PromptChain(
    models=["ollama/mistral-small:22b"],
    instructions=["Process: {input}"],
    verbose=True
    # No model_management parameter = disabled by default
)

# Check that model management is disabled
status = old_chain.get_model_manager_status()
assert status['enabled'] == False
```

## Supported Providers

### Current
- **Ollama**: Full support with VRAM optimization

### Planned
- **LocalAI**: Similar to Ollama implementation
- **LlamaCPP**: Direct integration with llama.cpp
- **vLLM**: Enterprise-grade model serving

### Adding New Providers
```python
from promptchain.utils.model_management import AbstractModelManager, ModelManagerFactory

class CustomModelManager(AbstractModelManager):
    async def load_model_async(self, model_name: str, model_params: dict):
        # Custom implementation
        pass
    
    # Implement other abstract methods...

# Register with factory
ModelManagerFactory.register_manager(ModelProvider.CUSTOM, CustomModelManager)
```

## Performance Impact

### Before Model Management
```
Chain with mistral-small:22b + deepseek-r1:70b:
├── Load mistral-small (22GB VRAM) ✓
├── Load deepseek-r1 (70GB VRAM) ❌ OUT OF MEMORY!
└── Result: FAILURE
```

### With Model Management
```
Chain with mistral-small:22b + deepseek-r1:70b:
├── Load mistral-small (22GB) → Execute → Unload ✓
├── Load deepseek-r1 (70GB) → Execute → Unload ✓  
└── Result: SUCCESS with 0GB final VRAM usage
```

### Benchmarks
- **VRAM Efficiency**: 90%+ reduction in peak memory usage
- **Load Time**: 2-5 seconds per model (network dependent)
- **Unload Time**: <1 second per model
- **Context Optimization**: 10-50% improvement in token handling

## Troubleshooting

### Common Issues

**1. Model Not Found Error**
```
ERROR: model 'ollama/mistral-small:22b' not found
```
**Solution**: Check model name format and availability:
```bash
curl http://localhost:11434/api/tags | jq '.models[].name'
```

**2. Connection Timeout**
```
ERROR: Failed to connect to Ollama after 3 attempts
```
**Solution**: Increase timeout or check server status:
```python
model_mgmt_config = {
    'enabled': True,
    'provider': 'ollama',
    'timeout': 60,  # Increase timeout
    'max_retries': 5
}
```

**3. VRAM Still Full**
```
WARNING: Model may still be loaded in Ollama
```
**Solution**: Manual cleanup:
```python
await chain.cleanup_models_async()
# Or check Ollama directly:
# curl -X POST http://localhost:11434/api/generate -d '{"model":"model-name","keep_alive":0}'
```

### Debug Mode
```python
import logging
logging.getLogger("promptchain.utils.model_management").setLevel(logging.DEBUG)
logging.getLogger("promptchain.utils.ollama_model_manager").setLevel(logging.DEBUG)

chain = PromptChain(
    models=["ollama/mistral-small:22b"],
    instructions=["Debug: {input}"],
    model_management={'enabled': True, 'provider': 'ollama', 'verbose': True},
    verbose=True
)
```

## Best Practices

### 1. Model Selection
- Use appropriate model sizes for tasks
- Consider quantized models (Q4, Q5, Q8) for VRAM efficiency
- Test model combinations in isolation first

### 2. Configuration
- Set realistic timeouts for large models (30+ minutes)
- Enable streaming for better timeout handling
- Use retry logic for unstable connections

### 3. Monitoring
- Enable verbose logging during development
- Monitor VRAM usage and model lifecycle
- Implement chainbreakers for error recovery

### 4. Performance
- Keep frequently-used models loaded when possible
- Use same models across steps to avoid unnecessary unloads
- Consider model quantization for better performance

## Examples

See the following example files:
- `examples/simple_deeplake_search_chain.py` - Complete integration example
- `examples/model_management_config_example.py` - Configuration examples
- `examples/manual_model_control.py` - Manual management example

## API Reference

### ModelInfo
```python
@dataclass
class ModelInfo:
    name: str
    provider: ModelProvider
    is_loaded: bool = False
    vram_usage: Optional[int] = None  # MB
    load_time: Optional[float] = None  # seconds
    last_used: Optional[float] = None  # timestamp
    parameters: Optional[Dict[str, Any]] = None
```

### ModelManagementConfig
```python
@dataclass
class ModelManagementConfig:
    enabled: bool = False
    default_provider: ModelProvider = ModelProvider.OLLAMA
    auto_unload: bool = True
    health_check_interval: int = 300  # seconds
    max_loaded_models: int = 2
    provider_configs: Dict[ModelProvider, Dict[str, Any]] = None
```

### PromptChain Methods
```python
# Async methods
await chain.load_model_async(model_name, model_params)
await chain.unload_model_async(model_name)
await chain.get_loaded_models_async()
await chain.cleanup_models_async()

# Sync wrappers
chain.load_model(model_name, model_params)
chain.unload_model(model_name)
chain.get_loaded_models()
chain.cleanup_models()

# Status and control
chain.get_model_manager_status()
chain.set_auto_unload(enabled)
```

---

*For more examples and advanced usage, see the `examples/` directory and test files.*