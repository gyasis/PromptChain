# Using Hugging Face Models with PromptChain

PromptChain provides comprehensive support for Hugging Face models through LiteLLM integration. This guide explains how to configure and use different types of Hugging Face models in your PromptChain applications.

## Table of Contents
- [Basic Configuration](#basic-configuration)
- [Model Types and Endpoints](#model-types-and-endpoints)
  - [Default Hugging Face Endpoint](#default-hugging-face-endpoint)
  - [Public Inference Endpoints](#public-inference-endpoints)
  - [Private Inference Endpoints](#private-inference-endpoints)
  - [Provider-Specific Models](#provider-specific-models)
- [Complete Example](#complete-example)
- [Authentication](#authentication)
- [Best Practices](#best-practices)

## Basic Configuration

To use Hugging Face models with PromptChain, you need to specify the model in the format `huggingface/model-name` or provide a configuration dictionary with additional parameters.

```python
from promptchain import PromptChain

chain = PromptChain(
    models=["huggingface/deepset/deberta-v3-large-squad2"],
    instructions=["Your instruction here"]
)
```

## Model Types and Endpoints

### Default Hugging Face Endpoint

For models available through the default Hugging Face inference API:

```python
chain = PromptChain(
    models=["huggingface/deepset/deberta-v3-large-squad2"],
    instructions=["Your instruction here"]
)
```

### Public Inference Endpoints

For models deployed to public Hugging Face endpoints:

```python
chain = PromptChain(
    models=[{
        "model": "huggingface/meta-llama/Llama-2-7b-hf",
        "api_base": "https://your-public-endpoint.huggingface.cloud",
        "temperature": 0.7,
        "max_tokens": 1000
    }],
    instructions=["Your instruction here"]
)
```

### Private Inference Endpoints

For models deployed to private Hugging Face endpoints:

```python
import os
# Option 1: Set token in environment
os.environ["HF_TOKEN"] = "hf_..."

chain = PromptChain(
    models=[{
        "model": "huggingface/meta-llama/Llama-2-7b-chat-hf",
        "api_base": "https://your-private-endpoint.huggingface.cloud",
        "api_key": "hf_...",  # Option 2: Pass token directly in config
        "temperature": 0.7
    }],
    instructions=["Your instruction here"]
)
```

### Provider-Specific Models

For models hosted by specific providers (e.g., Together AI, Sambanova):

```python
chain = PromptChain(
    models=["huggingface/together/deepseek-ai/DeepSeek-R1"],
    instructions=["Your instruction here"]
)
```

## Complete Example

Here's a comprehensive example showing how to use different types of Hugging Face models in a single chain:

```python
import os
from promptchain import PromptChain

# Set HF token in environment (optional)
os.environ["HF_TOKEN"] = "hf_..."

# Create a chain with multiple HF models
chain = PromptChain(
    models=[
        # Default HF endpoint
        "huggingface/deepset/deberta-v3-large-squad2",
        
        # Public endpoint
        {
            "model": "huggingface/meta-llama/Llama-2-7b-hf",
            "api_base": "https://public-endpoint.huggingface.cloud",
            "temperature": 0.7
        },
        
        # Private endpoint
        {
            "model": "huggingface/meta-llama/Llama-2-7b-chat-hf",
            "api_base": "https://private-endpoint.huggingface.cloud",
            "api_key": "hf_...",
            "max_tokens": 2000
        },
        
        # Provider-specific model
        "huggingface/together/deepseek-ai/DeepSeek-R1"
    ],
    instructions=[
        "Analyze the input",
        "Generate summary",
        "Provide recommendations",
        "Create final report"
    ]
)

# Use the chain
result = chain.process_prompt("Your input text here")
```

## Authentication

There are multiple ways to handle authentication for Hugging Face models:

1. **Environment Variable**:
   ```python
   os.environ["HF_TOKEN"] = "hf_..."
   ```

2. **Direct Configuration**:
   ```python
   models=[{
       "model": "huggingface/model-name",
       "api_key": "hf_..."
   }]
   ```

3. **Configuration File**:
   Store your token in a `.env` file:
   ```
   HF_TOKEN=hf_...
   ```

## Best Practices

1. **API Base URL**:
   - Always use HTTPS for endpoint URLs
   - Verify the endpoint URL format matches your Hugging Face deployment

2. **Authentication**:
   - Use environment variables for tokens in production
   - Never commit tokens to version control
   - Rotate tokens periodically for security

3. **Model Configuration**:
   - Set appropriate `max_tokens` for your use case
   - Adjust `temperature` based on desired output randomness
   - Consider using model-specific parameters when available

4. **Error Handling**:
   - Implement proper error handling for API calls
   - Handle token expiration and renewal
   - Monitor rate limits and usage

5. **Performance**:
   - Use appropriate batch sizes for your use case
   - Consider model loading time in your application design
   - Monitor memory usage with larger models 
noteId: "0a1f08c02db211f0a342718cff550985"
tags: []

---

 