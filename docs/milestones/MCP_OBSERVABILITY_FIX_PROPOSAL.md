# MCP Observability Fix - Library Level

**File**: `/promptchain/utils/mcp_helpers.py`
**Method**: `execute_mcp_tool()` (lines 344-493)

---

## Current Issue

MCPHelper.execute_mcp_tool() does NOT emit:
- `TOOL_CALL_START` (with tool name and arguments)
- `TOOL_CALL_END` (with results and timing)
- `TOOL_CALL_ERROR` (with error details)

This means MCP tool calls are invisible to the event system and can't be tracked in --dev mode or via callbacks.

---

## Proposed Fix

Add event emissions at key points in execute_mcp_tool():

### 1. Start Event (After argument parsing)

```python
async def execute_mcp_tool(self, tool_call: Any) -> str:
    # ... existing code to extract function_name, function_args_str, tool_call_id ...

    # Parse arguments to dict for event metadata
    try:
        tool_args_dict = json.loads(function_args_str) if function_args_str else {}
    except json.JSONDecodeError:
        tool_args_dict = {"raw_args": function_args_str}

    # 🔧 EMIT TOOL_CALL_START EVENT
    start_time = time.time()
    await self._emit_event(
        ExecutionEventType.TOOL_CALL_START,
        {
            "tool_name": function_name,
            "tool_args": tool_args_dict,
            "tool_call_id": tool_call_id,
            "is_mcp_tool": True
        }
    )

    # ... rest of the function ...
```

### 2. Success Event (After successful execution)

```python
    # After line 472 (successful result)
    execution_time_ms = (time.time() - start_time) * 1000

    # 🔧 EMIT TOOL_CALL_END EVENT
    await self._emit_event(
        ExecutionEventType.TOOL_CALL_END,
        {
            "tool_name": function_name,
            "tool_call_id": tool_call_id,
            "result": tool_output_str[:500],  # Preview
            "result_length": len(tool_output_str),
            "execution_time_ms": execution_time_ms,
            "success": True,
            "is_mcp_tool": True
        }
    )
```

### 3. Error Event (In exception handler)

```python
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000

        # 🔧 EMIT TOOL_CALL_ERROR EVENT
        await self._emit_event(
            ExecutionEventType.TOOL_CALL_ERROR,
            {
                "tool_name": function_name,
                "tool_call_id": tool_call_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": execution_time_ms,
                "is_mcp_tool": True
            }
        )

        # ... existing error handling ...
```

---

## Complete Modified Method

```python
import time
import json
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

async def execute_mcp_tool(self, tool_call: Any) -> str:
    """
    Executes an MCP tool based on the tool_call object using the LiteLLM MCP client.
    Returns the result as a string (plain or JSON).
    Now emits proper TOOL_CALL_START/END/ERROR events for observability.
    """
    from promptchain.utils.agentic_step_processor import get_function_name_from_tool_call

    # Extract function name
    function_name = get_function_name_from_tool_call(tool_call)

    # Extract arguments string
    if isinstance(tool_call, dict):
        function_obj = tool_call.get('function', {})
        function_args_str = function_obj.get('arguments', '{}') if isinstance(function_obj, dict) else '{}'
        tool_call_id = tool_call.get('id', 'N/A')
    else:
        function_obj = getattr(tool_call, 'function', None)
        if function_obj:
            if hasattr(function_obj, 'arguments'):
                function_args_str = function_obj.arguments
            elif isinstance(function_obj, dict) and 'arguments' in function_obj:
                function_args_str = function_obj['arguments']
            else:
                function_args_str = '{}'
        else:
            function_args_str = '{}'
        tool_call_id = getattr(tool_call, 'id', 'N/A')

    # Parse arguments for event metadata
    try:
        tool_args_dict = json.loads(function_args_str) if function_args_str else {}
    except json.JSONDecodeError:
        tool_args_dict = {"raw_args": function_args_str}

    # ✅ EMIT TOOL_CALL_START
    start_time = time.time()
    await self._emit_event(
        ExecutionEventType.TOOL_CALL_START,
        {
            "tool_name": function_name,
            "tool_args": tool_args_dict,
            "tool_call_id": tool_call_id,
            "is_mcp_tool": True
        }
    )

    self.logger.debug(f"[MCP Helper] Processing MCP tool call: {function_name} (ID: {tool_call_id})")
    self.logger.debug(f"[MCP Helper] Tool call argument string: {function_args_str}")

    tool_output_str = json.dumps({"error": f"MCP tool '{function_name}' execution failed."})

    if not function_name or function_name not in self.mcp_tools_map:
        error_msg = f"MCP tool function '{function_name}' not found in map."
        self.logger.error(f"[MCP Helper] Tool function '{function_name}' (ID: {tool_call_id}) not found in map.")
        self.logger.debug(f"[MCP Helper] Available MCP tools: {list(self.mcp_tools_map.keys())}")

        # ✅ EMIT ERROR EVENT
        await self._emit_event(
            ExecutionEventType.MCP_ERROR,
            {
                "error": error_msg,
                "tool_name": function_name,
                "tool_call_id": tool_call_id,
                "phase": "tool_execution",
                "available_tools": list(self.mcp_tools_map.keys())
            }
        )
        return json.dumps({"error": error_msg})

    if not MCP_AVAILABLE or not experimental_mcp_client:
        error_msg = f"MCP library not available to call tool '{function_name}'."
        self.logger.error(f"[MCP Helper] Tool '{function_name}' (ID: {tool_call_id}) called, but MCP library/client not available.")

        # ✅ EMIT ERROR EVENT
        await self._emit_event(
            ExecutionEventType.MCP_ERROR,
            {
                "error": error_msg,
                "tool_name": function_name,
                "tool_call_id": tool_call_id,
                "phase": "tool_execution",
                "reason": "MCP library not available"
            }
        )
        return json.dumps({"error": error_msg})

    try:
        mcp_info = self.mcp_tools_map[function_name]
        server_id = mcp_info['server_id']
        session = self.mcp_sessions.get(server_id)

        if not session:
            error_msg = f"MCP session '{server_id}' unavailable for tool '{function_name}'."
            self.logger.error(f"[MCP Helper] Session '{server_id}' not found for tool '{function_name}' (ID: {tool_call_id}).")

            # ✅ EMIT ERROR EVENT
            await self._emit_event(
                ExecutionEventType.MCP_ERROR,
                {
                    "error": error_msg,
                    "server_id": server_id,
                    "tool_name": function_name,
                    "tool_call_id": tool_call_id,
                    "phase": "tool_execution",
                    "reason": "Session not found"
                }
            )
            return json.dumps({"error": error_msg})

        original_schema = mcp_info['original_schema']
        original_tool_name = original_schema['function']['name']

        openai_tool_for_mcp = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": original_tool_name,
                "arguments": function_args_str
            }
        }

        if self.verbose:
            self.logger.debug(f"  [MCP Helper] Calling Tool: {original_tool_name} on {server_id} (Prefixed: {function_name}, ID: {tool_call_id})")
            self.logger.debug(f"  [MCP Helper] Arguments: {function_args_str}")

        call_result = await experimental_mcp_client.call_openai_tool(
            session=session,
            openai_tool=openai_tool_for_mcp
        )

        # Process result
        if call_result and hasattr(call_result, 'content') and call_result.content and hasattr(call_result.content[0], 'text'):
             tool_output_str = str(call_result.content[0].text)
        elif call_result:
             self.logger.debug(f"[MCP Helper] Tool {original_tool_name} returned unexpected structure. Converting result to string. Result: {call_result}")
             try: tool_output_str = json.dumps(call_result) if not isinstance(call_result, str) else call_result
             except TypeError: tool_output_str = str(call_result)
        else:
             self.logger.warning(f"[MCP Helper] Tool {original_tool_name} (ID: {tool_call_id}) executed but returned no structured content.")
             tool_output_str = json.dumps({"warning": f"Tool {original_tool_name} executed but returned no structured content."})

        if self.verbose: self.logger.debug(f"  [MCP Helper] Result (ID: {tool_call_id}): {tool_output_str[:150]}...")

        # ✅ EMIT TOOL_CALL_END (SUCCESS)
        execution_time_ms = (time.time() - start_time) * 1000
        await self._emit_event(
            ExecutionEventType.TOOL_CALL_END,
            {
                "tool_name": function_name,
                "original_tool_name": original_tool_name,
                "tool_call_id": tool_call_id,
                "result": tool_output_str[:500],  # Preview for logging
                "result_length": len(tool_output_str),
                "execution_time_ms": execution_time_ms,
                "success": True,
                "is_mcp_tool": True,
                "server_id": server_id
            }
        )

    except Exception as e:
        original_tool_name_err = mcp_info.get('original_schema',{}).get('function',{}).get('name','?') if 'mcp_info' in locals() else '?'
        server_id_err = server_id if 'server_id' in locals() else '?'
        self.logger.error(f"[MCP Helper] Error executing tool {function_name} (Original: {original_tool_name_err}, ID: {tool_call_id}) on {server_id_err}: {e}", exc_info=self.verbose)

        # ✅ EMIT TOOL_CALL_ERROR
        execution_time_ms = (time.time() - start_time) * 1000
        await self._emit_event(
            ExecutionEventType.TOOL_CALL_ERROR,
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "server_id": server_id_err,
                "tool_name": function_name,
                "original_tool_name": original_tool_name_err,
                "tool_call_id": tool_call_id,
                "execution_time_ms": execution_time_ms,
                "phase": "tool_execution",
                "is_mcp_tool": True
            }
        )

        tool_output_str = json.dumps({"error": str(e)})

    return tool_output_str
```

---

## Benefits

1. **Complete MCP Tool Visibility**: All MCP tool calls now emit proper events
2. **Consistent Event System**: MCP tools use same events as local tools
3. **Timing Information**: Execution time tracked for performance monitoring
4. **Error Tracking**: Detailed error events for debugging
5. **Backward Compatible**: Only adds events, doesn't change existing behavior

---

## Testing

After implementing this fix:

```bash
python agentic_chat/agentic_team_chat.py --dev

# Should now see:
🔧 Tool Call: mcp__gemini__gemini_research (mcp)
   Args: {'topic': 'what is zig programming language'}
```

The event callback in agentic_team_chat.py will automatically catch these events!
