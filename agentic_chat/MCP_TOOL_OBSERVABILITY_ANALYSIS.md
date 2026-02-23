# MCP Tool Observability Analysis

**Date**: 2025-10-04
**Issue**: MCP tool calls not visible in --dev mode

---

## Problem Statement

When running with `--dev` flag, MCP tool calls (like `gemini_research`) are NOT shown with their parameters. User only sees generic LiteLLM logs but not:
- Tool name: `gemini_research` or `ask_gemini`
- Tool arguments: `{'topic': 'what is zig programming language'}`

---

## Root Cause Analysis

### 1. Where MCP Tools Are Called

MCP tools are called inside **AgenticStepProcessor**, not directly by PromptChain. The flow is:

```
User Query
  → AgentChain (router)
  → Research Agent (PromptChain)
  → AgenticStepProcessor (agentic step)
  → AgenticStepProcessor.run_async()
    → Internal reasoning loop
    → Calls MCP tools via MCPHelper.execute_mcp_tool()
```

### 2. Event Emission Investigation

**MCPHelper.execute_mcp_tool()** (`/promptchain/utils/mcp_helpers.py`):
- Lines 344-493: Executes MCP tool calls
- **ONLY emits**: `MCP_ERROR` events (on errors)
- **DOES NOT emit**: `TOOL_CALL_START` or `TOOL_CALL_END` events

**Why**: MCPHelper was designed to be a low-level utility, and event emission for tool calls was expected to happen at a higher level (PromptChain or AgenticStepProcessor).

### 3. Current Event Types

According to PromptChain v0.4.1i observability docs:

**MCP-specific events**:
- `MCP_CONNECT_START`, `MCP_CONNECT_END`
- `MCP_DISCONNECT_START`, `MCP_DISCONNECT_END`
- `MCP_TOOL_DISCOVERED`
- `MCP_ERROR`

**Tool call events**:
- `TOOL_CALL_START`, `TOOL_CALL_END` - for regular (non-MCP) tools

**Agentic events**:
- `AGENTIC_STEP_START`
- `AGENTIC_INTERNAL_STEP` - **should contain `tool_calls` metadata**
- `AGENTIC_STEP_END`

---

## Fix Implementation

### Current Approach (Lines 1136-1163)

I added handling for `AGENTIC_INTERNAL_STEP` events to extract and display tool calls:

```python
# Agentic internal steps - track reasoning and tool calls
elif event.event_type == ExecutionEventType.AGENTIC_INTERNAL_STEP:
    step_number = event.metadata.get("step_number", 0)
    reasoning = event.metadata.get("reasoning", "")
    tool_calls = event.metadata.get("tool_calls", [])

    # --DEV MODE: Show tool calls from agentic steps (this is where MCP tools appear!)
    if tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})

            # Determine if it's an MCP tool (prefixed with mcp__)
            is_mcp = tool_name.startswith("mcp__") or "_" in tool_name
            tool_type = "mcp" if is_mcp else "local"

            dev_print(f"🔧 Tool Call: {tool_name} ({tool_type})", "")
            if tool_args:
                args_str = str(tool_args)
                args_preview = args_str[:150] + "..." if len(args_str) > 150 else args_str
                dev_print(f"   Args: {args_preview}", "")
```

### Expected Output in --dev Mode

```
🎯 Orchestrator Decision:
   Agent chosen: research
   Reasoning: Zig is a newer language requiring web research

🔧 Tool Call: mcp__gemini__gemini_research (mcp)
   Args: {'topic': 'what is zig programming language'}

📤 Agent Responding: research
────────────────────────────────────────────────────────────
[Zig documentation and examples...]
```

---

## Testing Required

Need to verify that `AGENTIC_INTERNAL_STEP` events:
1. Are actually emitted by AgenticStepProcessor
2. Contain `tool_calls` array in metadata
3. Tool calls have `name` and `args` keys

If not, we need a library-level fix.

---

## Alternative Fix (If Current Approach Fails)

### Option 1: Add Events to MCPHelper

Modify `/promptchain/utils/mcp_helpers.py` to emit proper events:

```python
async def execute_mcp_tool(self, tool_call: Any) -> str:
    function_name = get_function_name_from_tool_call(tool_call)
    # ... extract args ...

    # EMIT TOOL_CALL_START
    await self._emit_event(
        ExecutionEventType.TOOL_CALL_START,
        {
            "tool_name": function_name,
            "tool_args": json.loads(function_args_str),
            "is_mcp_tool": True
        }
    )

    try:
        # ... execute tool ...
        result = await experimental_mcp_client.call_openai_tool(...)

        # EMIT TOOL_CALL_END
        await self._emit_event(
            ExecutionEventType.TOOL_CALL_END,
            {
                "tool_name": function_name,
                "result": tool_output_str[:200],
                "execution_time_ms": ...,
                "success": True
            }
        )

        return tool_output_str
    except Exception as e:
        # EMIT TOOL_CALL_ERROR
        await self._emit_event(
            ExecutionEventType.TOOL_CALL_ERROR,
            {
                "tool_name": function_name,
                "error": str(e)
            }
        )
        raise
```

### Option 2: AgenticStepProcessor Enhancement

Ensure `AGENTIC_INTERNAL_STEP` events properly populate `tool_calls` metadata with full details.

---

## Recommendation

1. **Test current fix** - Run `--dev` mode and see if AGENTIC_INTERNAL_STEP shows tool calls
2. **If it fails** - Implement library-level fix in MCPHelper.execute_mcp_tool()
3. **Long-term** - Ensure all PromptChain tool calls (MCP and local) emit consistent TOOL_CALL_* events

---

## Files Modified

1. `/agentic_chat/agentic_team_chat.py` - Lines 1136-1163
   - Added AGENTIC_INTERNAL_STEP handling to show MCP tool calls

## Files That May Need Modification (If Current Fix Fails)

1. `/promptchain/utils/mcp_helpers.py` - execute_mcp_tool method
2. `/promptchain/utils/agentic_step_processor.py` - AGENTIC_INTERNAL_STEP emission
