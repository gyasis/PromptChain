# PromptChain MCP Tool Integration

This document explains how the `PromptChain` class integrates with external tools exposed via the Multi-Client Protocol (MCP), leveraging the `litellm.experimental_mcp_client` library. This allows a `PromptChain` to utilize tools running in separate processes or written in different languages, alongside standard local Python functions.

## Prerequisites

-   The `mcp-client` library must be installed (`pip install mcp-client`).
-   The `litellm` library must be installed (`pip install litellm`), including the necessary extras for MCP if any.

## Configuration

To use MCP tools, you need to configure the MCP servers when initializing the `PromptChain`. This is done via the `mcp_servers` parameter, which accepts a list of dictionaries. `PromptChain` uses an internal `MCPHelper` instance to manage these connections.

Each dictionary in the list defines one MCP server connection. Currently, only the `stdio` type is supported.

**Required `stdio` Server Configuration Fields:**

-   `id` (str): A unique identifier for this server connection (e.g., "context7_server"). This ID is used internally for mapping tools.
-   `type` (str): Must be `"stdio"`.
-   `command` (str): The command to execute to start the server process (e.g., `"/path/to/npx"` or `"python3"`). **Use absolute paths for reliability.**
-   `args` (List[str]): A list of arguments to pass to the command (e.g., `["-y", "@upstash/context7-mcp@latest"]` or `["/path/to/your/mcp_server.py"]`).
-   `env` (Optional[Dict[str, str]]): An optional dictionary of environment variables to set for the server process.
-   `read_timeout_seconds` (Optional[int]): Optional timeout for reading from the server process (defaults may vary, increase if needed).

**Example Initialization:**

```python
from promptchain import PromptChain # Correct import path

mcp_server_config = [
    {
        "id": "context7_server",
        "type": "stdio",
        "command": "/home/user/.nvm/versions/node/vXX.X.X/bin/npx", # Example absolute path
        "args": ["-y", "@upstash/context7-mcp@latest"],
        "env": {"DEBUG": "1"},
        "read_timeout_seconds": 120
    }
    # Add other server configs here
]

chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=["Analyze this: {input}"],
    mcp_servers=mcp_server_config,
    verbose=True
)

# --> Connection happens later using chain.mcp_helper <--
```

## Connection and Tool Discovery (`mcp_helper.connect_mcp_async`)

**Crucially, MCP servers are NOT connected automatically on initialization.** You *must* explicitly connect them before running the chain if any steps require MCP tools.

Connection and discovery are handled by the internal `MCPHelper` instance, accessible via `chain.mcp_helper`.

**Call this *before* `process_prompt_async`:**

```python
# Connect to MCP and discover tools
await chain.mcp_helper.connect_mcp_async()
print("Discovered tools:", [t['function']['name'] for t in chain.tools])
```

When `await chain.mcp_helper.connect_mcp_async()` is called:

1.  **Connection:** The `MCPHelper` iterates through the `mcp_servers` list and attempts to start and connect to each server using the provided configuration.
2.  **Tool Discovery:** For each successfully connected server, the helper calls `await experimental_mcp_client.load_mcp_tools(session=session, format="openai")` to get the list of tools provided by that server.
3.  **Tool Registration & Prefixing:**
    *   For each tool discovered on a server (e.g., a tool named `resolve-library-id` on the server with `id="context7_server"`):
    *   A **prefixed tool name** is generated: `mcp_<server_id>_<original_tool_name>` (e.g., `mcp_context7_server_resolve-library-id`). This prevents naming collisions.
    *   A *copy* of the original tool schema is made, but the `function.name` is updated to the prefixed name.
    *   This **prefixed schema is added to `chain.tools`**. This property (`chain.tools`) returns a combined list of all registered local tools and all discovered (prefixed) MCP tools.
    *   The LLM will see and request tools using these prefixed names (e.g., `mcp_context7_server_resolve-library-id`).
    *   The `MCPHelper` stores an internal mapping (`mcp_tools_map`) to remember the original name and server ID associated with each prefixed name for later execution.

## Execution Flow (`process_prompt_async`)

1.  **LLM Request:** The LLM receives the available tools (including prefixed MCP names) and might request one (e.g., `mcp_context7_server_resolve-library-id`).
2.  **Tool Call Routing:** `PromptChain` checks if the requested name belongs to a local function or is in the `MCPHelper`'s map.
3.  **MCP Tool Execution:** If it's an MCP tool:
    *   The `MCPHelper` uses its internal map to find the `server_id` and original tool name (`resolve-library-id`).
    *   It retrieves the correct `ClientSession`.
    *   It calls `await experimental_mcp_client.call_openai_tool(...)` sending the **original tool name** and arguments to the correct server via the session.
4.  **Result Handling:** The tool result (or error) is received from the MCP server and formatted into a `role: "tool"` message using the **prefixed name** requested by the LLM.
5.  **Follow-up LLM Call:** The history (including the assistant's request and the tool result) is sent back to the LLM.

## Handling Sequential MCP Tool Calls

As discovered, the most reliable way to handle sequences like *resolve ID -> get docs -> synthesize* is using **separate `PromptChain` steps**.

1.  **Define Separate Instructions:** Each step performs one primary action (call tool, parse result, call next tool, synthesize).
2.  **Use `{input}`:** Pass the output of one step as the input to the next using the `{input}` placeholder.
3.  **Instruct LLM to Parse/Extract:** If a tool returns verbose output but the next step needs only a specific piece (like an ID), instruct the LLM in the *first* step's prompt to parse the tool result and output *only* the required piece.

**Example Structure (Resolve ID -> Get Docs -> Synthesize):**

```python
chain = PromptChain(
    models=["openai/gpt-4o-mini"] * 3, # Model for each step
    instructions=[
        # Step 1: Resolve ID, parse result, output ONLY the first ID string
        "Use the 'mcp_context7_server_resolve-library-id' tool with the library name provided as input: '{input}'. "
        "The tool will return text describing matching libraries. Parse this text carefully, find the line starting with '- Library ID:' for the *very first* library listed, extract the ID value (e.g., '/org/repo'), and output *only* that single ID string.",

        # Step 2: Get Docs using the ID from Step 1
        "Use the 'mcp_context7_server_get-library-docs' tool with the Context7 ID provided as input ('{input}') to retrieve its documentation. Output *only* the raw documentation text.",

        # Step 3: Synthesize the final answer using history and docs from Step 2
        "Review the full conversation history to find the original user query (the library name). Based on that original query and the documentation provided as input ('{input}') from the previous step, provide a comprehensive answer. Include an overview of the library, its main features, and relevant code examples from the documentation."
    ],
    mcp_servers=mcp_server_config,
    verbose=True
)

# --- Connect and Run ---
try:
    await chain.mcp_helper.connect_mcp_async()
    library_name = input("Enter library name: ")
    final_output = await chain.process_prompt_async(library_name) # Initial input is just the name
    print(final_output)
finally:
    await chain.mcp_helper.close_mcp_async()

```

## Interaction with `AgenticStepProcessor`

-   MCP tools discovered via `chain.mcp_helper.connect_mcp_async()` are automatically added to the `chain.tools` list.
-   When an `AgenticStepProcessor` runs as part of the chain, it automatically uses the `chain.tools` available *at that time*.
-   Therefore, **no special steps are needed to pass MCP tools to an `AgenticStepProcessor`**, as long as `connect_mcp_async()` was called *before* the chain (and thus the agentic step) runs. The agentic step's internal LLM calls will receive the prefixed MCP tool schemas just like a regular step.

## Shutdown and Cleanup (`mcp_helper.close_mcp_async`)

-   It's crucial to close the MCP connections and terminate the server processes cleanly.
-   Call `await chain.mcp_helper.close_mcp_async()` in a `finally` block after your chain processing is complete.
-   **Note on `asyncio.run()`:** Using `asyncio.run()` directly can sometimes conflict with the cleanup mechanisms of libraries like `mcp-client` (which uses `anyio`), potentially causing `RuntimeError: Attempted to exit cancel scope...`. If you encounter this, consider managing the asyncio event loop manually as shown in the `test_context7_mcp_single_step.py` example (using `loop.run_until_complete` and explicit loop closing) to ensure cleanup happens in the correct order.

## Summary

MCP integration allows extending `PromptChain` with external tools. Key points are: explicit connection via `mcp_helper`, understanding tool name prefixing, structuring sequential calls across separate steps, and proper cleanup via `close_mcp_async`. MCP tools are automatically available to `AgenticStepProcessor` if connected beforehand.
