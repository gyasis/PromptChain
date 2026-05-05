---
id: recipe-mcp-tool
title: External MCP server tool integration
when: You need the LLM to use tools from an external MCP server (filesystem, browser, custom service).
api_version: 0.6.1+
---

# MCP Tool

External MCP servers are wired in via the `mcp_servers` constructor argument on `PromptChain`. Tools from server `<id>` are auto-prefixed `mcp_<id>_<tool_name>` to avoid colliding with local tools.

```python
import asyncio
from promptchain import PromptChain


async def main():
    chain = PromptChain(
        models=["openai/gpt-4o"],
        instructions=["Use the filesystem tool to read /tmp/example.txt and summarise it."],
        mcp_servers=[
            {
                "id": "filesystem",                 # becomes the prefix
                "type": "stdio",                    # only "stdio" is currently supported
                "command": "mcp-server-filesystem", # the CLI to spawn
                "args": ["/tmp"],                    # args passed to the server
            },
        ],
        verbose=True,
    )
    # Tools from this server are now exposed to the LLM as mcp_filesystem_<tool>
    result = await chain.process_prompt_async("start")
    print(result)


asyncio.run(main())
```

## Mixing local + MCP tools

```python
def my_local_tool(x: str) -> str:
    """Local tool docstring."""
    return f"local: {x}"

chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=["…"],
    mcp_servers=[{"id": "fs", "type": "stdio", "command": "mcp-server-filesystem", "args": ["/tmp"]}],
)
chain.register_tool_function(my_local_tool)
# LLM sees: "my_local_tool" + "mcp_fs_read_file", "mcp_fs_write_file", etc.
```

## Direct execution without LLM round-trip (advanced)

```python
chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=["…"],
    mcp_servers=[...],
    enable_mcp_hijacker=True,
    hijacker_config={"connection_timeout": 5.0, "max_retries": 2, "parameter_validation": True},
)
```

## Common failures + fix

- **MCP tool not in the schema sent to the LLM** — `MCPHelper` is async; verify the chain initialised properly. Increase `verbose=True` to see connection logs.
- **Tool name collision** — local tool with the same bare name as `mcp_<id>_<name>` after stripping prefix → local wins, MCP is masked. Rename one.
- **`mcp` package not installed** — `pip install mcp`. PromptChain's MCP integration is optional; absence is silent.
- **Server fails to spawn** — `command` not on PATH. Test with `which mcp-server-filesystem` first.
