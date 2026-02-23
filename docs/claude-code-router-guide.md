# Claude Code Router: Complete Guide

## Overview

**Claude Code Router** is a powerful middleware/proxy tool that acts as an intelligent proxy between Claude Code (Anthropic's command-line coding assistant) and various AI model providers. It enables you to use Claude Code's interface while routing requests to alternative AI providers, eliminating the need for an Anthropic account.

### Key Capabilities

- **Multi-Provider Support**: Route to OpenRouter, DeepSeek, Ollama, Gemini, Azure OpenAI, VolcEngine, SiliconFlow, ModelScope, DashScope, and more
- **Intelligent Routing**: Automatically selects models based on task context (default, background, reasoning, long-context)
- **Dynamic Model Switching**: Change models mid-session using `/model provider,model_name` commands
- **Cost Optimization**: Route simple tasks to cheaper models, complex tasks to powerful models
- **Request Transformation**: Handles API format differences between providers automatically

---

## Installation

### Prerequisites
- Node.js 18+ installed
- npm package manager

### Step 1: Install Claude Code
```bash
npm install -g @anthropic-ai/claude-code
```

### Step 2: Install Claude Code Router
```bash
npm install -g @musistudio/claude-code-router
```

---

## Configuration

### Configuration File Location

**macOS/Linux:**
```bash
mkdir -p ~/.claude-code-router
nano ~/.claude-code-router/config.json
```

**Windows (PowerShell):**
```bash
mkdir "$env:USERPROFILE\.claude-code-router" -Force
notepad "$env:USERPROFILE\.claude-code-router\config.json"
```

### Basic Configuration Structure

```json
{
  "LOG": true,
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "API_TIMEOUT_MS": "600000",
  "Providers": [
    {
      "name": "provider_name",
      "api_base_url": "https://api.provider.com/v1/chat/completions",
      "api_key": "YOUR_API_KEY_HERE",
      "models": ["model-1", "model-2"]
    }
  ],
  "Router": {
    "default": "provider,model",
    "background": "provider,model",
    "think": "provider,model",
    "longContext": "provider,model",
    "longContextThreshold": 60000
  }
}
```

---

## Model Configuration Examples

### 1. Google Gemini Integration

#### Via OpenRouter (Recommended)
```json
{
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "sk-or-v1-your-openrouter-key",
      "models": [
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-flash-preview",
        "google/gemini-pro-1.5"
      ]
    }
  ],
  "Router": {
    "default": "openrouter,google/gemini-2.5-pro-preview",
    "background": "openrouter,google/gemini-2.5-flash-preview",
    "think": "openrouter,google/gemini-2.5-pro-preview",
    "longContext": "openrouter,google/gemini-pro-1.5"
  }
}
```

#### Direct Google Gemini API
```json
{
  "Providers": [
    {
      "name": "gemini",
      "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models",
      "api_key": "YOUR_GOOGLE_API_KEY",
      "models": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro"
      ]
    }
  ],
  "Router": {
    "default": "gemini,gemini-2.5-pro",
    "background": "gemini,gemini-2.5-flash"
  }
}
```

### 2. Azure OpenAI (GPT-5, GPT-4, etc.)

```json
{
  "Providers": [
    {
      "name": "azure",
      "api_base_url": "https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments",
      "api_key": "YOUR_AZURE_OPENAI_API_KEY",
      "api_version": "2024-02-15-preview",
      "models": [
        "gpt-5",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-35-turbo"
      ],
      "transformer": {
        "use": []
      }
    }
  ],
  "Router": {
    "default": "azure,gpt-5",
    "background": "azure,gpt-35-turbo",
    "think": "azure,gpt-5",
    "longContext": "azure,gpt-4-turbo"
  }
}
```

**Azure Configuration Notes:**
- Replace `YOUR_RESOURCE_NAME` with your Azure OpenAI resource name
- API key format: `YOUR_AZURE_OPENAI_API_KEY`
- For GPT-5, ensure your Azure subscription has access to the model
- API version may need to be updated based on Azure OpenAI service version

### 3. GPT-5 via OpenRouter

```json
{
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "sk-or-v1-your-key",
      "models": [
        "openai/gpt-5",
        "openai/gpt-4-turbo",
        "openai/gpt-4"
      ]
    }
  ],
  "Router": {
    "default": "openrouter,openai/gpt-5",
    "background": "openrouter,openai/gpt-4"
  }
}
```

### 4. Multi-Provider Configuration (Gemini + GPT-5 + Azure)

```json
{
  "LOG": true,
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "API_TIMEOUT_MS": "600000",
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "sk-or-v1-your-openrouter-key",
      "models": [
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-flash-preview",
        "openai/gpt-5",
        "openai/gpt-4-turbo"
      ]
    },
    {
      "name": "azure",
      "api_base_url": "https://YOUR_RESOURCE.openai.azure.com/openai/deployments",
      "api_key": "YOUR_AZURE_KEY",
      "api_version": "2024-02-15-preview",
      "models": [
        "gpt-5",
        "gpt-4-turbo"
      ]
    },
    {
      "name": "gemini-direct",
      "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models",
      "api_key": "YOUR_GOOGLE_API_KEY",
      "models": [
        "gemini-2.5-pro",
        "gemini-2.5-flash"
      ]
    }
  ],
  "Router": {
    "default": "openrouter,google/gemini-2.5-pro-preview",
    "background": "openrouter,google/gemini-2.5-flash-preview",
    "think": "azure,gpt-5",
    "longContext": "openrouter,google/gemini-2.5-pro-preview",
    "longContextThreshold": 60000
  }
}
```

### 5. DeepSeek (Specialized Coding Models)

```json
{
  "Providers": [
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "sk-your-deepseek-key",
      "models": [
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-coder"
      ]
    }
  ],
  "Router": {
    "default": "deepseek,deepseek-chat",
    "background": "deepseek,deepseek-chat",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "deepseek,deepseek-chat"
  }
}
```

### 6. Ollama (Local Models)

```json
{
  "Providers": [
    {
      "name": "ollama",
      "api_base_url": "http://localhost:11434/v1/chat/completions",
      "api_key": "ollama",
      "models": [
        "llama3",
        "mistral",
        "codellama",
        "qwen2.5-coder"
      ]
    }
  ],
  "Router": {
    "default": "ollama,llama3",
    "background": "ollama,mistral",
    "think": "ollama,llama3"
  }
}
```

---

## Usage

### Starting Claude Code with Router

```bash
# Start the router service
ccr restart

# Launch Claude Code through router
ccr code
```

### Dynamic Model Switching

During a Claude Code session, switch models dynamically:

```bash
# Switch to Gemini
/model openrouter,google/gemini-2.5-pro-preview

# Switch to GPT-5 on Azure
/model azure,gpt-5

# Switch to local Ollama model
/model ollama,qwen2.5-coder:latest

# Switch to DeepSeek
/model deepseek,deepseek-reasoner
```

### Router Status

```bash
# Check router status
ccr status

# Restart router
ccr restart

# Stop router
ccr stop
```

---

## Router Configuration Explained

### Router Types

The `Router` section defines which model to use for different task types:

- **`default`**: Standard coding tasks and general queries
- **`background`**: Background operations, less critical tasks (use cheaper/faster models)
- **`think`**: Complex reasoning tasks requiring deep analysis (use powerful models)
- **`longContext`**: Tasks requiring large context windows (use models with extended context)
- **`longContextThreshold`**: Token count threshold to trigger long-context routing (default: 60000)

### Example: Cost-Optimized Routing

```json
{
  "Router": {
    "default": "openrouter,google/gemini-2.5-flash-preview",  // Fast, cheap
    "background": "ollama,mistral",                          // Free, local
    "think": "azure,gpt-5",                                  // Powerful, expensive
    "longContext": "openrouter,google/gemini-2.5-pro-preview" // Long context
  }
}
```

This setup:
- Uses fast/cheap models for routine tasks
- Uses local models for background work
- Reserves expensive models for complex reasoning
- Uses long-context models when needed

---

## Advanced Features

### Custom Transformers

Transform requests/responses for provider compatibility:

```json
{
  "Providers": [
    {
      "name": "custom-provider",
      "api_base_url": "https://api.example.com/v1",
      "api_key": "your-key",
      "models": ["model-1"],
      "transformer": {
        "use": ["request", "response"]
      }
    }
  ]
}
```

### Environment Variables

You can also use environment variables for sensitive data:

```bash
export CLAUDE_CODE_ROUTER_API_KEY_OPENROUTER="sk-or-v1-your-key"
export CLAUDE_CODE_ROUTER_API_KEY_AZURE="your-azure-key"
```

Then reference in config:
```json
{
  "api_key": "${CLAUDE_CODE_ROUTER_API_KEY_OPENROUTER}"
}
```

---

## Troubleshooting

### Common Issues

#### Error: Connection Refused
- **Cause**: Router service not running
- **Solution**: Run `ccr restart` before `ccr code`

#### Error: Invalid API Key
- **Cause**: Incorrect API key in configuration
- **Solution**: Verify API key in provider dashboard

#### Error: Model Not Found
- **Cause**: Model name doesn't match provider's model list
- **Solution**: Verify exact model names using provider's API

#### Error: Timeout
- **Cause**: Network issues or slow provider response
- **Solution**: Increase `API_TIMEOUT_MS` in config

### Verifying Model Availability

**OpenRouter:**
```bash
curl -X GET "https://openrouter.ai/api/v1/models" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | \
  grep -i "gemini\|gpt-5"
```

**Azure OpenAI:**
```bash
curl -X GET "https://YOUR_RESOURCE.openai.azure.com/openai/models?api-version=2024-02-15-preview" \
  -H "api-key: YOUR_AZURE_KEY"
```

**Google Gemini:**
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_API_KEY"
```

---

## Security Considerations

1. **API Key Security**: Never commit API keys to version control
2. **Local Storage**: Store config file with restricted permissions (`chmod 600`)
3. **Network Security**: Use HTTPS endpoints only
4. **Cost Monitoring**: Set usage limits in provider dashboards
5. **Auto-Updates**: Be aware that router can auto-update; review updates before applying

---

## Cost Optimization Strategies

### Strategy 1: Tiered Model Usage
- Simple tasks → Fast/cheap models (Gemini Flash, GPT-3.5)
- Complex tasks → Powerful models (GPT-5, Gemini Pro)
- Background tasks → Local models (Ollama)

### Strategy 2: Context-Aware Routing
- Short queries → Fast models
- Long context → Models optimized for long context
- Reasoning tasks → Models with strong reasoning capabilities

### Strategy 3: Hybrid Approach
```json
{
  "Router": {
    "default": "openrouter,google/gemini-2.5-flash-preview",  // $0.075/1M tokens
    "think": "azure,gpt-5",                                    // $5/1M tokens
    "background": "ollama,llama3"                              // Free
  }
}
```

---

## Integration with Other Tools

### Claude Code Proxy Alternative

For Python-based proxy setup:

```bash
git clone https://github.com/fuergaosi233/claude-code-proxy
cd claude-code-proxy
```

Create `.env`:
```ini
ANTHROPIC_API_KEY=sk-ant-fake-key
OPENROUTER_API_KEY=sk-or-v1-real-key
TARGET_MODEL=google/gemini-2.5-pro-preview
PROXY_PORT=8080
```

Set environment variables:
```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=sk-ant-fake-key
```

### LiteLLM Integration

Use LiteLLM as a unified proxy:

```yaml
# config.yaml
model_list:
  - model_name: gemini-2.5-pro
    litellm_params:
      model: gemini/gemini-2.5-pro
      api_key: os.environ/GOOGLE_API_KEY
  
  - model_name: gpt-5-azure
    litellm_params:
      model: azure/gpt-5
      api_base: https://YOUR_RESOURCE.openai.azure.com
      api_key: os.environ/AZURE_OPENAI_API_KEY
```

---

## Best Practices

1. **Start Simple**: Begin with one provider, expand gradually
2. **Monitor Costs**: Track usage across providers
3. **Test Models**: Verify model capabilities before production use
4. **Use Environment Variables**: Keep API keys out of config files
5. **Version Control**: Exclude config files from git (add to `.gitignore`)
6. **Documentation**: Document your routing strategy for team members
7. **Fallback Models**: Configure fallback models for reliability

---

## Research Insights

Based on multi-hop research and analysis:

### Model Performance Characteristics

- **Gemini 2.5 Pro**: Excellent for long-context tasks, competitive pricing
- **Gemini 2.5 Flash**: Fast responses, cost-effective for routine tasks
- **GPT-5**: Strong reasoning, best for complex problem-solving
- **DeepSeek**: Specialized for coding tasks, cost-effective
- **Ollama Models**: Free, local, good for privacy-sensitive tasks

### Routing Strategy Recommendations

1. **Development Workflow**: Use fast models (Gemini Flash) for quick iterations, powerful models (GPT-5) for architecture decisions
2. **Code Review**: Use reasoning models (DeepSeek Reasoner, GPT-5)
3. **Documentation**: Use long-context models (Gemini Pro)
4. **Testing**: Use local models (Ollama) for privacy-sensitive code

---

## Additional Resources

- **Official Documentation**: [Claude Code Router Docs](https://claudelog.com/claude-code-mcps/claude-code-router/)
- **OpenRouter Models**: [OpenRouter Model List](https://openrouter.ai/models)
- **Azure OpenAI**: [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- **Google Gemini**: [Gemini API Documentation](https://ai.google.dev/docs)
- **DeepSeek**: [DeepSeek API Docs](https://api.deepseek.com/)

---

## Example: Complete Multi-Provider Setup

```json
{
  "LOG": true,
  "HOST": "127.0.0.1",
  "PORT": 3456,
  "API_TIMEOUT_MS": "600000",
  "Providers": [
    {
      "name": "openrouter",
      "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
      "api_key": "${OPENROUTER_API_KEY}",
      "models": [
        "google/gemini-2.5-pro-preview",
        "google/gemini-2.5-flash-preview",
        "openai/gpt-5",
        "openai/gpt-4-turbo"
      ]
    },
    {
      "name": "azure",
      "api_base_url": "https://my-resource.openai.azure.com/openai/deployments",
      "api_key": "${AZURE_OPENAI_API_KEY}",
      "api_version": "2024-02-15-preview",
      "models": ["gpt-5", "gpt-4-turbo"]
    },
    {
      "name": "deepseek",
      "api_base_url": "https://api.deepseek.com/chat/completions",
      "api_key": "${DEEPSEEK_API_KEY}",
      "models": ["deepseek-chat", "deepseek-reasoner"]
    },
    {
      "name": "ollama",
      "api_base_url": "http://localhost:11434/v1/chat/completions",
      "api_key": "ollama",
      "models": ["llama3", "qwen2.5-coder"]
    }
  ],
  "Router": {
    "default": "openrouter,google/gemini-2.5-flash-preview",
    "background": "ollama,llama3",
    "think": "azure,gpt-5",
    "longContext": "openrouter,google/gemini-2.5-pro-preview",
    "longContextThreshold": 60000
  }
}
```

This configuration provides:
- ✅ Gemini integration (via OpenRouter)
- ✅ GPT-5 support (Azure + OpenRouter)
- ✅ Cost optimization (tiered routing)
- ✅ Local fallback (Ollama)
- ✅ Specialized coding models (DeepSeek)

---

## Conclusion

Claude Code Router enables flexible, cost-effective AI coding assistance by:
- Supporting multiple providers (Gemini, GPT-5, Azure, etc.)
- Intelligent routing based on task complexity
- Dynamic model switching
- Cost optimization through strategic model selection

By following this guide, you can set up a powerful, multi-model coding assistant that adapts to your needs and budget.

