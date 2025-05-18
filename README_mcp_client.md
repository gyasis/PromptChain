# PromptChain MCP Client Examples

This directory contains examples of using PromptChain with MCP (Model Context Protocol) servers.

## Prerequisites

1. Install required packages:
   ```bash
   pip install promptchain mcp litellm python-dotenv
   ```

2. Make sure you have Python 3 installed (the examples use `python3` explicitly)

3. Set up your API keys in a `.env` file (for models like GPT-4)

## Examples

### Basic MCP Client

The `mcp_client.py` file demonstrates how to:
- Create a PromptChain that connects to an MCP server
- Process a prompt using the server's functions
- Clean up connections when done

To run:
```bash
python3 mcp_client.py
```

### Memory-Enhanced MCP Client

The `mcp_client_with_memory.py` file demonstrates:
- Using PromptChain's Memory Bank with MCP integration
- Storing and retrieving data from memory
- Processing prompts that use memory context
- Working with memory namespaces

To run:
```bash
python3 mcp_client_with_memory.py
```

## Configuration

Both examples connect to the MCP server in the `project_prompts` directory. You can modify:

- The model used (default: "gpt-4")
- The instruction templates
- The task descriptions in the process_prompt_async calls
- Memory operations and namespaces

## Troubleshooting

- If you get a "Command 'python' not found" error, make sure to use `python3` explicitly
- If the MCP server fails to start, check that the path to run_server.py is correct
- If you get import errors, ensure all required packages are installed

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│  PromptChain    │────▶│   MCP Server     │────▶│   LLM Model     │
│  (Client)       │◀────│   (Function      │◀────│   (Processing)  │
│                 │     │    Provider)     │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                                                 │
        │                                                 │
        ▼                                                 ▼
┌─────────────────┐                             ┌─────────────────┐
│                 │                             │                 │
│  Memory Bank    │                             │  Task Results   │
│  (State)        │                             │  (Output)       │
│                 │                             │                 │
└─────────────────┘                             └─────────────────┘
```

## Further Reading

- See the PromptChain documentation for more details on chain configuration
- Refer to the Memory Bank guide for advanced memory operations
- Check the MCP documentation for developing custom MCP servers 