---
id: recipe-tool-calling-local
title: Local tool calling — multiple Python tools, the LLM loop, schema generation
when: You want the LLM to call several local Python functions to accomplish a task. The LLM picks which tool to call when.
api_version: 0.6.1+
---

# Local Tool Calling

`PromptChain` (and `AgenticStepProcessor`) lets the LLM call locally-registered Python functions. The function's signature + docstring become the OpenAI-format tool schema sent to the model.

## Minimal — register multiple tools

```python
import asyncio
import json
from promptchain import PromptChain


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Real impl would hit an API
    return json.dumps({"city": city, "temp_c": 22, "condition": "sunny"})


def list_cities(country: str) -> str:
    """List major cities in a country."""
    return json.dumps({"FR": ["Paris", "Lyon", "Marseille"]}.get(country, []))


async def main():
    chain = PromptChain(
        models=["openai/gpt-4o"],
        instructions=["Tell me the weather in two French cities."],
        verbose=True,
    )

    # Register both tools — the LLM sees both in the schema and picks
    chain.register_tool_function(get_weather)
    chain.register_tool_function(list_cities)

    # The LLM will:
    # 1. Call list_cities("FR") → ["Paris", "Lyon", "Marseille"]
    # 2. Call get_weather("Paris"), get_weather("Lyon") — possibly in parallel
    # 3. Synthesise a final answer
    print(await chain.process_prompt_async("start"))


asyncio.run(main())
```

## How the schema is generated

For each registered function, PromptChain auto-creates:

```json
{
  "type": "function",
  "function": {
    "name": "<func.__name__>",
    "description": "<first line of docstring>",
    "parameters": {
      "type": "object",
      "properties": {
        "<arg>": {"type": "<inferred from annotation>", "description": "..."},
        ...
      },
      "required": [<args without defaults>]
    }
  }
}
```

So:
- **The function MUST have a docstring** — it becomes the description the LLM uses to pick the tool.
- **The function MUST have type hints** — they become the schema types. No hint → default `string`.
- **Default values mark args as optional**.

## Multi-turn tool loop

The framework handles the loop for you:
1. LLM emits one or more `tool_calls`.
2. Framework dispatches each to the registered function (PARALLEL — see v0.6.1 fix below).
3. Tool results are appended to the conversation as `tool` messages.
4. LLM is called again with the augmented conversation; loop until no more `tool_calls`.

No manual loop required.

## ⚠️ v0.6.1 critical fix — parallel tool calls now actually run

**Before v0.6.1**, when an LLM emitted multiple `tool_calls` in a single turn, only the LAST one was executed (loop indentation bug in `agentic_step_processor.py`). If you're on `<0.6.1`, upgrade. If you have weird behaviour with parallel tool calls, check your version: `pip show promptchain`.

## v0.6.1 — schema/function name validation

Both `chain.add_tools(schemas)` and `chain.register_tool_function(func)` now validate that every schema name has a matching registered function. Mismatch raises `ValueError` at registration time — no more cryptic `Missing parameter 'tool_call_id'` errors from OpenAI later.

```python
chain.add_tools([{"type": "function", "function": {"name": "my_tool", ...}}])
def my_tool_func(...): ...                       # ← name mismatch!
chain.register_tool_function(my_tool_func)       # ← raises ValueError now
```

Fix: rename the function to match (`def my_tool(...)`) or rename the schema.

## Common failures + fix

- **LLM never calls the tool** — docstring is missing or vague. The model picks tools based on the description; "Get weather" beats `"helper function"`.
- **Tool returns `dict`, LLM sees garbage** — the framework stringifies non-string returns via `repr()`. Always `return json.dumps(...)` for structured data, or a plain string.
- **`ValueError: schema name 'X' has no registered function'`** (v0.6.1+) — schema and function names are out of sync. Match them.
- **Tool raises an exception** — propagates and stops the LLM turn. Wrap in try/except and return an error string if you want graceful degradation.
- **Hint-less arg becomes required string** — add `: int = 0` (typed + default) for optional integer args.

## Mixing local + MCP tools

See `recipe-mcp-tool.md`. MCP tool names get prefixed `mcp_<server_id>_<name>`; locals do not. A local named `read_file` and an MCP tool `mcp_fs_read_file` coexist.
