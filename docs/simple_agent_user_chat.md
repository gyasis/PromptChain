# Simple Agent User Chat — Documentation & Scaffold Guide

## Overview
This document explains the structure and usage of `simple_agent_user_chat.py`, a minimal but powerful baseline for building conversational agents with PromptChain, LiteLLM, and AgentChain. Use this as a reference for expanding to multi-agent, multi-model, or cloud-integrated workflows.

---

## Script Purpose
- Demonstrates a robust pattern for a conversational agent.
- Designed for easy expansion to more complex, multi-agent, or multi-model scenarios.

---

## Key Components

### 1. Imports and Setup
```python
import asyncio
from promptchain.utils.promptchaining import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
import logging
import litellm
```
- **PromptChain**: Orchestrates LLM calls and prompt logic.
- **AgentChain**: Manages one or more agents, allowing for flexible routing and orchestration.
- **ExecutionHistoryManager**: Handles chat history and truncation.
- **litellm**: Unified LLM interface (local/cloud).

---

### 2. Model Endpoint Routing
```python
remote_litellm_url = "http://192.168.1.159:4000/v1"  # (Proxy example, not used directly here)
```
- This is the address of a LiteLLM proxy server (for OpenAI-compatible API calls).
- **Not used directly in this script**; see below for routing options.

---

### 3. Chainbreaker Example
```python
def break_on_bye(step, output):
    if "bye" in output.lower():
        return (True, "User said bye", "Goodbye!")
    return (False, "", None)
```
- Optional: Shows how to break the chain on a specific user input.

---

### 4. Agent Definition (PromptChain)
```python
qa_agent = PromptChain(
    models=[{
        "name": "ollama/deepseek-r1:70b",
        "params": {
            "api_base": "http://192.168.1.159:11434"
        }
    }],
    instructions=["You are a helpful assistant. Here is the conversation so far:\n{history}\nUser: {input}\nAssistant:"],
    full_history=False,
    store_steps=False,
    verbose=False
)
```
- **models**: List of model configs.
  - `"name"`: Model identifier (e.g., `"ollama/deepseek-r1:70b"`).
  - `"params"`: Dict of model-specific parameters.
    - **`api_base`**: The endpoint for the model.
      - Local Ollama: `"http://localhost:11434"`
      - Remote Ollama: `"http://<remote-ip>:11434"`
      - Cloud (OpenAI, etc.): Use the provider's endpoint.
- **instructions**: Prompt template for the agent.
- **full_history/store_steps/verbose**: Control history and logging.

---

### 5. AgentChain Setup
```python
agent_dict = {"qa_agent": qa_agent}
agent_descriptions_dict = {"qa_agent": "Handles all user questions and provides helpful answers."}

agent_orchestrator = AgentChain(
    agents=agent_dict,
    agent_descriptions=agent_descriptions_dict,
    execution_mode="pipeline",  # Linear: user -> agent -> answer
    verbose=True
)
```
- **AgentChain** allows you to:
  - Route between multiple agents (for multi-agent setups).
  - Use different execution modes (pipeline, router, etc.).
  - Add descriptions for documentation and UI.

---

### 6. Main Chat Loop
```python
async def main():
    history_manager = ExecutionHistoryManager(max_tokens=2048)
    print("Simple AgentChain Q&A with managed history. Type 'bye' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Exiting chat.")
            break
        history_manager.add_entry("user_input", user_input)
        formatted_history = history_manager.get_formatted_history(format_style='chat')
        agent_input = f"You are a helpful assistant. Here is the conversation so far:\n{formatted_history}\nUser: {user_input}\nAssistant:"
        response = await agent_orchestrator.process_input(agent_input)
        print(f"AI: {response}\n")
        history_manager.add_entry("agent_output", response)
```
- **History management**: Keeps conversation context within token limits.
- **Agent input formatting**: Ensures the agent sees the full chat context.
- **AgentChain orchestration**: Handles the actual LLM call and response.

---

## How to Scaffold for Multiple Agents and Models

### Adding More Agents
- Define additional `PromptChain` instances with different models, instructions, or parameters.
- Add them to `agent_dict` and `agent_descriptions_dict`.
- Example:
  ```python
  agent_dict = {
      "qa_agent": qa_agent,
      "math_agent": math_agent,
      "code_agent": code_agent
  }
  ```

### Routing Options
- **pipeline**: Linear flow (default, as in this script).
- **router**: Route input to different agents based on input type or classification.
- **custom**: Implement your own routing logic.

### Flexible Model Access with `api_base`
- Each agent can use a different model and endpoint:
  - Local Ollama: `"api_base": "http://localhost:11434"`
  - Remote Ollama: `"api_base": "http://<remote-ip>:11434"`
  - LiteLLM Proxy: `"api_base": "http://<proxy-ip>:4000/v1"`
  - Cloud (OpenAI, Anthropic, etc.): Use their respective endpoints and API keys.
- **You can mix and match local, remote, and cloud models in a single script.**

---

## Model Routing Table

| Agent Name   | Model Name                | api_base                        | Use Case                |
|--------------|--------------------------|----------------------------------|-------------------------|
| qa_agent     | ollama/deepseek-r1:70b   | http://192.168.1.159:11434      | Local/remote Ollama     |
| math_agent   | openai/gpt-4o            | https://api.openai.com/v1       | OpenAI cloud            |
| code_agent   | ollama/codellama         | http://localhost:11434          | Local Ollama            |
| ...          | ...                      | ...                              | ...                     |

---

## Best Practices
- **Use `api_base`** to control where each model call is routed.
- **Keep agent logic modular**: Each agent can have its own prompt, model, and parameters.
- **Leverage AgentChain** for orchestration, routing, and multi-agent workflows.
- **History management**: Use `ExecutionHistoryManager` to keep context concise and within token limits.
- **Verbose logging**: Enable for debugging, especially when adding new models or endpoints.

---

## How This Baseline Helps You Expand
- **Single agent → Multi-agent**: Just add more agents and update the routing logic.
- **Local → Cloud**: Change the `api_base` and model name; no other code changes needed.
- **Simple Q&A → Complex workflows**: Use AgentChain's advanced modes and add custom functions or tools.

---

**This script is your foundation for building scalable, flexible, and powerful agent-based applications with PromptChain and LiteLLM.**

---

*Reference this file when scaffolding new projects or expanding to multi-agent, multi-model, or cloud-integrated workflows.* 