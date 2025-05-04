---
noteId: "1f604ca0055111f0b67657686c686f9a"
tags: []

---

# Technical Context: PromptChain

## Technologies Used

### Primary Technologies

1. **Python 3.8+**: Core implementation language
2. **LiteLLM**: For abstracting calls to various LLM providers
3. **Pydantic**: For data validation and settings management
4. **Python-dotenv**: For environment variable management
5. **PromptEngineer**: For advanced prompt improvement and optimization
6. **AsyncIO**: For asynchronous operations
7. **MCP**: For Model Context Protocol server integration
8. **WebSockets**: For real-time chat applications

### Core Components

1. **PromptChain**: Main chain execution class
2. **ChainStep**: Data model for step information
3. **AgenticStepProcessor**: Internal agentic loop processor for complex steps
4. **Memory Bank**: Persistent storage for chain state
5. **MCP Client Manager**: Handles MCP server connections and tool execution

### LLM Provider Support

PromptChain supports multiple LLM providers through LiteLLM, including:

1. **OpenAI**: GPT-3.5, GPT-4 models
2. **Anthropic**: Claude models (Claude-3-Opus, Claude-3-Sonnet, etc.)
3. **Other providers**: Any provider supported by LiteLLM

### MCP Server Support

PromptChain integrates with Model Context Protocol (MCP) servers:

1. **Server Types**:
   - stdio: Standard I/O based servers
   - (Future) WebSocket servers
   - (Future) gRPC servers

2. **Configuration Options**:
   ```python
   mcp_servers = [{
       "id": "tools_server",
       "type": "stdio",
       "command": "python",
       "args": ["server.py"],
       "env": {"CUSTOM_VAR": "value"}
   }]
   ```

## Development Setup

### Environment Setup

1. **Python Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   
   # Install dependencies
   pip install -e .  # Development installation
   # or
   pip install -r requirements.txt
   ```

2. **API Keys Configuration**:
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   # Add other provider keys as needed
   ```

3. **MCP Server Setup**:
   ```bash
   # Install MCP library
   pip install mcp-python

   # Set up server environment
   export MCP_SERVER_ENV_VAR=value
   ```

### Project Structure

```
promptchain/
├── __init__.py           # Package initialization
├── utils/               # Utility modules
│   ├── __init__.py
│   ├── promptchaining.py # Core PromptChain implementation
│   ├── logging_utils.py  # Logging utilities
│   ├── prompt_loader.py  # Functions for loading prompt templates
│   └── mcp_utils.py     # MCP server utilities
├── examples/            # Example implementations
│   ├── async_examples/  # Async usage examples
│   └── mcp_examples/   # MCP integration examples
├── tests/              # Test cases
└── experiments/        # Experimental features
```

## Technical Constraints

### API Limitations

1. **Rate Limits**:
   - OpenAI: Varies by tier, typically 3-60 RPM for GPT-4
   - Anthropic: Varies by tier, check current limits
   - Handle rate limiting with appropriate retry mechanisms
   - Consider async operations for better throughput

2. **Token Limitations**:
   - GPT-4: ~8K-32K token context windows (model dependent)
   - Claude: ~100K token context windows
   - Chain design should consider these limitations
   - MCP tools may have their own limits

3. **Cost Considerations**:
   - API calls incur costs based on token usage
   - Multi-step chains increase total token consumption
   - Function steps can reduce token usage for certain operations
   - MCP tools may have separate usage costs

### Performance Considerations

1. **Latency**:
   - LLM API calls typically have latency of 1-10 seconds
   - Chain execution time scales with the number of model steps
   - Async processing can help manage multiple operations
   - MCP server communication adds some overhead

2. **Scalability**:
   - Function steps can become bottlenecks if computationally intensive
   - Consider pooling or parallel processing for high-volume applications
   - Async operations help with concurrent processing
   - MCP servers may have their own scaling limits

3. **Async Operations**:
   - Event loop management is critical
   - Connection lifecycle must be properly handled
   - Error handling needs special attention
   - Resource cleanup is important

## Dependencies

### Core Dependencies

```
# From requirements.txt
litellm>=1.2.0
python-dotenv>=1.0.0
pydantic>=2.0.0
mcp-python>=0.1.0  # For MCP support
```

### Optional Dependencies

```
# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0  # For async tests

# Development
black>=23.0.0
isort>=5.10.0
flake8>=6.0.0
```

## Integration Patterns

### Agentic Step Integration

1. **Basic Usage**:
   ```python
   from promptchain import PromptChain
   from promptchain.utils.agentic_step_processor import AgenticStepProcessor

   # Define a calculator tool
   calculator_schema = {
       "type": "function",
       "function": {
           "name": "simple_calculator",
           "description": "Evaluates a simple mathematical expression.",
           "parameters": {
               "type": "object",
               "properties": {
                   "expression": {"type": "string", "description": "The expression to evaluate."}
               },
               "required": ["expression"]
           }
       }
   }

   def simple_calculator(expression: str) -> str:
       """Evaluates a mathematical expression safely"""
       try:
           # Create a safe evaluation environment with limited scope
           allowed_names = {"abs": abs, "pow": pow, "round": round}
           result = eval(expression, {"__builtins__": None}, allowed_names)
           return str(result)
       except Exception as e:
           return f"Error: {str(e)}"

   # Create the PromptChain with an agentic step
   chain = PromptChain(
       models=[],  # No models needed for AgenticStepProcessor
       instructions=[
           AgenticStepProcessor(
               objective="You are a math assistant. Use the calculator tool when needed.",
               max_internal_steps=3,
               model_name="openai/gpt-4o",  # Set model directly on processor
               model_params={"tool_choice": "auto"}
           )
       ],
       verbose=True
   )

   # Register the tool
   chain.add_tools([calculator_schema])
   chain.register_tool_function(simple_calculator)

   # Run the chain
   result = chain.process_prompt("What is the square root of 144?")
   ```

2. **Mixed Instructions**:
   ```python
   chain = PromptChain(
       models=["openai/gpt-3.5-turbo"],  # Model for regular prompt steps
       instructions=[
           "Analyze the following request: {input}",
           AgenticStepProcessor(
               objective="You are a research assistant. Extract key information.",
               max_internal_steps=5,
               model_name="openai/gpt-4o",  # Different model for agentic step
               model_params={"temperature": 0.2}
           ),
           "Summarize the findings from the previous step."
       ]
   )
   ```

3. **With MCP Tools**:
   ```python
   chain = PromptChain(
       models=[],
       instructions=[
           AgenticStepProcessor(
               objective="You are a coding assistant. Use tools to help answer questions.",
               max_internal_steps=10,
               model_name="anthropic/claude-3-opus"
           )
       ],
       mcp_servers=[{
           "id": "tools_server",
           "type": "stdio",
           "command": "python",
           "args": ["mcp_tools_server.py"]
       }]
   )

   await chain.connect_mcp_async()
   result = await chain.process_prompt_async("How do I implement a binary tree in Python?")
   await chain.close_mcp_async()
   ```

### Async Usage Patterns

1. **Basic Async Chain**:
   ```python
   async def process_chain():
       chain = PromptChain(
           models=["gpt-4"],
           instructions=["Process: {input}"]
       )
       result = await chain.process_prompt_async("Hello")
       return result
   ```

2. **MCP Integration**:
   ```python
   async def use_mcp_tools():
       chain = PromptChain(
           models=["gpt-4"],
           instructions=["Use tools: {input}"],
           mcp_servers=[{
               "id": "tools",
               "type": "stdio",
               "command": "python",
               "args": ["server.py"]
           }]
       )
       try:
           await chain.connect_mcp_async()
           result = await chain.process_prompt_async("Task")
           return result
       finally:
           await chain.close_mcp_async()
   ```

### Asynchronous vs. Synchronous Usage (Best Practices)

**Background:** `PromptChain` provides both synchronous (`process_prompt`, `run_model`) and asynchronous (`process_prompt_async`, `run_model_async`) methods for core operations. The synchronous methods act as wrappers, using `asyncio.run()` internally to execute their async counterparts. This design aims for backward compatibility but requires careful usage.

**Critical Issue Resolved:** A `RuntimeError: Cannot run the event loop while another loop is running` was encountered when synchronous `PromptChain` methods were called from within an existing async context (like a WebSocket server or FastAPI route). This occurs because `asyncio.run()` cannot be nested or called when an event loop is already active.

**Best Practices:**

1.  **Identify Your Context:** Determine if your calling code is synchronous or asynchronous.
    *   **Asynchronous Context:** Code within `async def` functions, `asyncio.run()`, WebSocket handlers, `FastAPI` endpoints, `aiohttp` servers, etc.
    *   **Synchronous Context:** Standard Python scripts, `Flask` routes (without async extensions), basic functions not using `async/await`.
2.  **Use Matching Methods:**
    *   **In Async Context:** **MUST** use the `async` methods: `await chain.process_prompt_async(...)`, `await PromptChain.run_model_async(...)`.
    *   **In Sync Context:** Can safely use the synchronous wrappers: `chain.process_prompt(...)`, `PromptChain.run_model(...)`.
3.  **Never Mix Contexts Incorrectly:** **DO NOT** call synchronous `PromptChain` wrappers (`process_prompt`, `run_model`) from an asynchronous context. This is the primary cause of the runtime error.
4.  **`asyncio.to_thread` Limitation:** Using `asyncio.to_thread` to call a synchronous `PromptChain` wrapper from async code **DOES NOT** solve the underlying issue, as `asyncio.run()` within the wrapper will still conflict with the main event loop.
5.  **Consistency:** Strive for consistency. If your application is primarily asynchronous (e.g., using FastAPI), use the `async` methods throughout your interaction with `PromptChain`.

By adhering to these practices, you can avoid runtime errors related to event loop conflicts and leverage the async capabilities of `PromptChain` effectively.

### Agent Framework Integration

PromptChain can integrate with various agent frameworks:

1. **AutoGen**:
   ```python
   # Example AutoGen integration
   from autogen import AssistantAgent
   
   prompt = chain.process_prompt("Task description")
   agent = AssistantAgent(
       name="researcher",
       system_message=prompt,
       llm_config={"config_list": [...]}
   )
   ```

2. **LangChain**:
   ```python
   # Example LangChain integration
   from langchain.agents import initialize_agent
   
   prompt = chain.process_prompt("Task description")
   agent = initialize_agent(
       agent="zero-shot-react-description",
       tools=[...],
       llm=ChatOpenAI(),
       agent_kwargs={"system_message": prompt}
   )
   ```

### Chat Integration

1. **WebSocket Server Setup**:
   ```python
   # Setup WebSocket server with Memory Bank integration
   async def handle_websocket(websocket):
       session_id = generate_session_id()
       chat_chain = PromptChain(
           models=["gpt-4"],
           instructions=["Process chat message: {input}"]
       )
       
       # Initialize session in Memory Bank
       chat_chain.store_memory("history", [], namespace=f"chat:{session_id}")
       
       try:
           async for message in websocket:
               # Retrieve conversation history
               history = chat_chain.retrieve_memory("history", namespace=f"chat:{session_id}", default=[])
               
               # Add user message to history
               history.append({"role": "user", "content": message})
               
               # Process with history context
               result = await chat_chain.process_prompt_async(f"History: {str(history)}\nNew message: {message}")
               
               # Add assistant response to history
               history.append({"role": "assistant", "content": result})
               
               # Update history in memory
               chat_chain.store_memory("history", history, namespace=f"chat:{session_id}")
               
               # Send response
               await websocket.send(result)
       except Exception as e:
           logging.error(f"WebSocket error: {str(e)}")
   ```

2. **Memory-Based Chat Management**:
   ```python
   # Create dedicated chat memory manager
   def create_chat_manager(chain, session_id):
       namespace = f"chat:{session_id}"
       
       def get_history():
           return chain.retrieve_memory("history", namespace=namespace, default=[])
           
       def add_message(role, content):
           history = get_history()
           history.append({"role": role, "content": content})
           chain.store_memory("history", history, namespace=namespace)
           
       def clear_history():
           chain.clear_memories(namespace=namespace)
           
       def get_user_preferences():
           return chain.retrieve_memory("preferences", namespace=namespace, default={})
           
       def set_user_preference(key, value):
           prefs = get_user_preferences()
           prefs[key] = value
           chain.store_memory("preferences", prefs, namespace=namespace)
           
       return {
           "get_history": get_history,
           "add_message": add_message,
           "clear_history": clear_history,
           "get_user_preferences": get_user_preferences,
           "set_user_preference": set_user_preference
       }
   ```

### Tool Integration Best Practices

1. **Tool Definition**:
   - Define tools using compatible JSON schemas
   - Include clear descriptions and parameter definitions
   - Follow function-calling conventions for your model provider

2. **Tool Registration**:
   ```python
   # Add tool schema
   chain.add_tools([tool_schema])
   
   # Register the implementation function
   chain.register_tool_function(tool_function)
   ```

3. **Function Name Extraction**:
   - Different model providers return tool calls in various formats
   - Use the built-in `get_function_name_from_tool_call` function for robust extraction
   - Handle all potential formats (dict, object with attributes, nested structures)

4. **Error Handling**:
   - Tools should always return strings (JSON strings for structured data)
   - Implement proper error handling within tool functions
   - Return meaningful error messages rather than throwing exceptions
   - Set appropriate `max_internal_steps` to prevent infinite loops

5. **Security Considerations**:
   - Validate inputs thoroughly, especially for tools with system access
   - Implement proper sandboxing for code execution tools
   - Consider authentication/authorization for sensitive operations
   - Restrict scope and capabilities of tools to necessary functionality

## Deployment Considerations

1. **Environment Variables**:
   - Secure management of API keys in production
   - Consider using a secrets manager for production deployments
   - Protect MCP server credentials

2. **Logging and Monitoring**:
   - Enable appropriate logging for production use
   - Monitor API usage to control costs
   - Track chain performance for optimization
   - Monitor MCP server health
   - Track async operation metrics

3. **Error Handling**:
   - Implement proper error handling for API failures
   - Consider fallback models or retry strategies
   - Cache intermediate results for long chains when possible
   - Handle MCP server failures gracefully
   - Manage async errors properly

4. **Async Deployment**:
   - Configure event loop policies appropriately
   - Handle connection pooling effectively
   - Implement proper cleanup on shutdown
   - Monitor resource usage

5. **MCP Server Management**:
   - Implement health checks
   - Handle server restarts gracefully
   - Monitor tool availability
   - Manage server resources effectively

## Testing Strategy

1. **Unit Tests**:
   ```python
   import pytest
   
   @pytest.mark.asyncio
   async def test_async_chain():
       chain = PromptChain(...)
       result = await chain.process_prompt_async("Test")
       assert result
   
   @pytest.mark.asyncio
   async def test_mcp_integration():
       chain = PromptChain(
           mcp_servers=[{"id": "test", ...}]
       )
       await chain.connect_mcp_async()
       try:
           result = await chain.process_prompt_async("Test")
           assert result
       finally:
           await chain.close_mcp_async()
   ```

2. **Integration Tests**:
   - Test async operations end-to-end
   - Verify MCP server interactions
   - Check error handling
   - Validate cleanup procedures

3. **Performance Tests**:
   - Measure async operation throughput
   - Test concurrent processing
   - Monitor resource usage
   - Evaluate MCP server performance

## PromptEngineer Configuration

### Command Line Parameters

## Multi-Client Protocol (MCP) Integration

PromptChain integrates with external tools via MCP using `litellm.experimental_mcp_client`.

- **Configuration:** Defined in `mcp_servers` during `PromptChain` initialization (currently supports `stdio`). Requires server `id`, `type`, `command`, `args` (absolute paths recommended), and optional `env`.
- **Connection:** `connect_mcp_async()` starts servers and discovers tools.
- **Tool Discovery:** Tools are discovered via `experimental_mcp_client.load_mcp_tools`. Discovered tool names are prefixed (`mcp_<server_id>_<original_name>`) and added to `chain.tools`.
- **Execution:** `process_prompt_async` routes tool calls to local functions or the appropriate MCP server based on the (prefixed) name.
- **`chain.tools` Structure:** When MCP tools are registered, `chain.tools` becomes a **list** of tool schema dictionaries. To access tool names (e.g., for logging), iterate through the list and use `tool['function']['name']`.
- **Sequential MCP Tool Calls:** For tasks requiring multiple MCP tools in sequence (e.g., resolve ID then get docs), use multiple distinct instruction steps. Each step should guide the LLM for that specific tool call or synthesis task, leveraging the implicit conversation history passed between steps for context. Avoid non-standard context management like `store_memory` for this pattern.