# Model Management Quick Reference

## 🚀 Quick Start

### Basic Setup
```python
from promptchain import PromptChain

chain = PromptChain(
    models=["ollama/mistral-small:22b", "ollama/deepseek-r1:70b"],
    instructions=["Step 1: {input}", "Step 2: {input}"],
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=True,  # 🔥 Auto VRAM management
    verbose=True
)

result = await chain.process_prompt_async("Your task here")
```

## ⚡ Configuration Options

### Chain-Level Config
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
    models=["ollama/model1", "ollama/model2"],
    instructions=["Task 1: {input}", "Task 2: {input}"],
    model_management=model_mgmt_config,
    auto_unload_models=True
)
```

### Global Config
```python
from promptchain.utils.model_management import ModelManagementConfig, set_global_config

config = ModelManagementConfig(
    enabled=True,
    auto_unload=True,
    provider_configs={
        'ollama': {
            'base_url': 'http://localhost:11434',
            'timeout': 30
        }
    }
)
set_global_config(config)
```

## 🎛️ Manual Control

### Load/Unload Models
```python
# Load model manually
model_info = await chain.load_model_async("mistral-small:22b")
print(f"Loaded: {model_info.name}, VRAM: {model_info.vram_usage}MB")

# Unload model
await chain.unload_model_async("mistral-small:22b")

# Check status
status = chain.get_model_manager_status()
loaded = await chain.get_loaded_models_async()
```

### Disable Auto-Unload
```python
chain = PromptChain(
    models=["ollama/mistral-small:22b"],
    instructions=["Process: {input}"],
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=False,  # Manual control
)

# Process multiple tasks without reloading
result1 = await chain.process_prompt_async("Task 1")
result2 = await chain.process_prompt_async("Task 2")

# Cleanup when done
await chain.cleanup_models_async()
```

## 🔧 Common Patterns

### Large Model Chain
```python
# Chain large models that don't fit together in VRAM
chain = PromptChain(
    models=[
        "ollama/mistral-small:22b",   # 22GB model
        "ollama/deepseek-r1:70b",     # 70GB model  
    ],
    instructions=[
        "Quick analysis: {input}",
        "Deep reasoning: {input}"
    ],
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=True,  # Essential for large models
    verbose=True
)
```

### Remote Ollama Server
```python
chain = PromptChain(
    models=[{
        "name": "ollama/mistral-small:22b",
        "params": {
            "api_base": "http://192.168.1.160:11434",
            "timeout": 1800,  # 30 minutes
            "max_retries": 3,
            "stream": True
        }
    }],
    instructions=["Process: {input}"],
    model_management={
        'enabled': True,
        'provider': 'ollama',
        'base_url': 'http://192.168.1.160:11434'
    },
    auto_unload_models=True
)
```

### MCP + Model Management
```python
# DeepLake MCP with automatic model management
mcp_config = [{"id": "deeplake", "type": "stdio", ...}]

chain = PromptChain(
    models=["ollama/mistral-small:22b", "ollama/deepseek-r1:70b"],
    instructions=[
        "Search with DeepLake: {input}",
        "Analyze results: {input}"
    ],
    mcp_servers=mcp_config,
    model_management={'enabled': True, 'provider': 'ollama'},
    auto_unload_models=True
)
```

## 📊 Monitoring

### Status Check
```python
# Get detailed status
status = chain.get_model_manager_status()
print(f"""
Model Management: {status['enabled']}
Auto-unload: {status['auto_unload']}
Provider: {status['manager_type']}
""")

# Check loaded models
loaded = await chain.get_loaded_models_async()
for model in loaded:
    print(f"- {model.name}: {model.vram_usage}MB")
```

### Interactive Commands
```python
# In your script's main loop
while True:
    task = input("Enter task (or 'models'/'cleanup'/'status'): ")
    
    if task == 'models':
        loaded = await chain.get_loaded_models_async()
        print(f"Loaded: {[m.name for m in loaded]}")
    elif task == 'cleanup':
        await chain.cleanup_models_async()
        print("All models unloaded")
    elif task == 'status':
        print(chain.get_model_manager_status())
    else:
        result = await chain.process_prompt_async(task)
        print(result)
```

## 🐛 Troubleshooting

### Model Not Found
```bash
# Check available models
curl http://localhost:11434/api/tags | jq '.models[].name'
```

### Connection Issues
```python
# Increase timeout and retries
model_mgmt_config = {
    'enabled': True,
    'provider': 'ollama',
    'timeout': 60,        # Longer timeout
    'max_retries': 5      # More retries
}
```

### VRAM Still Full
```python
# Force cleanup
await chain.cleanup_models_async()

# Check Ollama directly
import httpx
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:11434/api/generate",
        json={"model": "model-name", "keep_alive": 0}
    )
```

## ⚠️ Important Notes

1. **Model Names**: Use exact Ollama model names (e.g., `mistral-small:22b`)
2. **Prefixes**: Both `ollama/model-name` and `model-name` work
3. **Context Length**: Automatically optimized to maximum available
4. **Backward Compatible**: Works with existing chains (disabled by default)
5. **Provider Support**: Currently supports Ollama, more providers coming

## 📝 Example Files

- `examples/simple_deeplake_search_chain.py` - Complete MCP + Model Management
- `examples/model_management_config_example.py` - Configuration examples
- `docs/model_management.md` - Full documentation

## 🔗 Related

- [Model Management Full Documentation](model_management.md)
- [MCP Integration Guide](MCP_server_use.md)
- [PromptChain Quick Start](promptchain_quickstart_multistep.md)