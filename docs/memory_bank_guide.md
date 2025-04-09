---
noteId: "10242440150111f0a459a9c802cb6e4a"
tags: []

---

# Memory Bank Guide

The Memory Bank feature in PromptChain provides persistent storage capabilities across chain executions, allowing chains to maintain state and share information.

## Basic Usage

```python
from promptchain.utils.promptchaining import PromptChain

chain = PromptChain(
    models=["gpt-4"],
    instructions=["Process and store: {input}"]
)

# Store a value
chain.store_memory("user_preference", "dark_mode", namespace="settings")

# Retrieve a value
preference = chain.retrieve_memory("user_preference", namespace="settings")

# Check if memory exists
if chain.memory_exists("user_preference", namespace="settings"):
    print("User preference is set")

# List all memories in a namespace
all_settings = chain.list_memories(namespace="settings")

# Clear memories
chain.clear_memories(namespace="settings")  # Clear specific namespace
chain.clear_memories()  # Clear all memories
```

## Memory Functions in Chains

You can create specialized memory access functions for use in your chains:

```python
chain = PromptChain(
    models=["gpt-4"],
    instructions=[
        "Store user preferences: {input}",
        chain.create_memory_function(namespace="user_prefs"),
        "Retrieve and use preferences: {input}"
    ]
)

# The memory function understands commands like:
# MEMORY STORE key=value
# MEMORY GET key
# MEMORY LIST
```

## Creating Memory-Aware Chains

Use the `create_memory_chain` helper to create chains with built-in memory capabilities:

```python
memory_chain = chain.create_memory_chain(
    chain_id="preferences_manager",
    namespace="user_settings",
    instructions=[
        "Process user preferences: {input}",
        "Store processed preferences in memory"
    ]
)
```

## Memory Bank in Async Contexts

The memory bank works seamlessly in both synchronous and asynchronous contexts:

```python
async def main():
    chain = PromptChain(
        models=["gpt-4"],
        instructions=["Process with memory: {input}"]
    )
    
    # Memory operations are synchronous and safe to use in async context
    chain.store_memory("status", "processing")
    
    # Process with async chain
    result = await chain.process_prompt_async("Use stored status")
    
    # Update memory with result
    chain.store_memory("status", "completed")
    chain.store_memory("last_result", result)

asyncio.run(main())
```

## Best Practices

1. **Use Namespaces**
   - Organize related memories into namespaces
   - Prevents naming conflicts
   - Makes clearing specific memory sets easier

2. **Memory Lifecycle**
   - Clear unnecessary memories to prevent memory leaks
   - Use `clear_memories()` when appropriate
   - Consider implementing automatic cleanup for temporary memories

3. **Error Handling**
   - Always provide default values when retrieving memories
   - Check if memories exist before critical operations
   - Handle missing memories gracefully

4. **Memory Security**
   - Don't store sensitive information in memory
   - Clear sensitive memories immediately after use
   - Use separate namespaces for different security contexts

## Advanced Usage

### Memory Bank with Dynamic Chain Builder

```python
from promptchain.utils.promptchaining import DynamicChainBuilder

builder = DynamicChainBuilder(base_model="gpt-4")

# Create a chain with memory capabilities
chain = builder.create_chain(
    chain_id="memory_chain",
    instructions=[
        "Store this information: {input}",
        builder.create_memory_function(namespace="dynamic_data"),
        "Process using stored data: {input}"
    ]
)

# Execute with memory operations
builder.store_memory("chain_status", "initialized", namespace="status")
result = builder.execute_chain("memory_chain", "Test input")
builder.store_memory("chain_status", "completed", namespace="status")
```

### Memory Bank with MCP Integration

The memory bank can be used alongside MCP servers to maintain state across tool calls:

```python
chain = PromptChain(
    models=["gpt-4"],
    instructions=["Process with tools and memory: {input}"],
    mcp_servers=[{
        "id": "tools_server",
        "type": "stdio",
        "command": "python",
        "args": ["tools_server.py"]
    }]
)

async def main():
    # Store initial state
    chain.store_memory("session_id", "12345")
    
    # Process with MCP tools
    result = await chain.process_prompt_async("Use tools and check session")
    
    # Update state based on tool results
    chain.store_memory("last_tool_result", result)
    
    # Clean up
    await chain.close_mcp_async()
    chain.clear_memories(namespace="session")

asyncio.run(main())
```

## Implementation Details

The Memory Bank is implemented using a dictionary-based storage system with namespace support:

```python
self.memory_bank = {
    "namespace1": {
        "key1": "value1",
        "key2": "value2"
    },
    "namespace2": {
        "key3": "value3"
    }
}
```

This structure provides:
- Fast access and updates (O(1))
- Namespace isolation
- Easy serialization if needed
- Efficient memory management

## Future Enhancements

Planned features for the Memory Bank include:
- Persistent storage backends (Redis, SQLite)
- Memory expiration and TTL
- Memory access logging
- Memory compression for large values
- Memory sharing between chains
- Memory backup and restore 