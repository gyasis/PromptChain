# PromptChain Debug Logging Usage Guide

## 🎯 Overview

The PromptChain debug logging system allows you to monitor AgenticStepProcessor execution in real-time without interfering with MCP JSON-RPC communication.

## 🚀 Quick Start

### 1. Enable Debug Logging
```bash
export PROMPTCHAIN_DEBUG=true
```

### 2. Start MCP Server
```bash
uv run fastmcp run athena_mcp_server.py
```

### 3. Monitor Debug Logs in Real-Time
```bash
# In a separate terminal
tail -f promptchain_debug.log
```

### 4. Analyze Structured Logs
```bash
# Pretty print JSON logs
cat promptchain_debug.log | jq '.'

# Filter specific events
cat promptchain_debug.log | grep "llm_call" | jq '.'

# Filter by tool calls
cat promptchain_debug.log | grep "tool_call" | jq '.'
```

## 📊 Log Event Types

### Objective Events
- `objective_start` - When AgenticStepProcessor begins
- `objective_complete` - When the entire objective is finished

### LLM Events  
- `llm_call` - Before calling the LLM
- `llm_response` - After LLM responds
- `llm_call_timing` - LLM call duration

### Tool Events
- `tool_call` - Tool execution (status: executing/success/error)
- `tool_call_timing` - Tool execution duration

### Error Events
- `error` - Any errors during execution

## 🔧 Example Log Analysis

### Find All Tool Calls
```bash
grep "tool_call" promptchain_debug.log | jq '.tool_name'
```

### Monitor Execution Times
```bash
grep "timing" promptchain_debug.log | jq '{event, duration: .call_duration_seconds}'
```

### Track Step Progression
```bash
grep "step_number" promptchain_debug.log | jq '{step: .step_number, event}'
```

## 🎛️ Environment Controls

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PROMPTCHAIN_DEBUG` | `false` | Enable/disable debug logging |
| `MCP_DEBUG_MODE` | `false` | Enable MCP server debug output |

## 📁 Log File Location

- **File**: `promptchain_debug.log` (in athena-lightrag directory)
- **Format**: Timestamped JSON entries
- **Rotation**: Manual (consider adding logrotate for production)

## 🎯 Production Usage

### For Development
```bash
PROMPTCHAIN_DEBUG=true uv run fastmcp dev athena_mcp_server.py
```

### For Production (Debug Off)
```bash
uv run fastmcp run athena_mcp_server.py
```

## 🛠️ Troubleshooting

### No Debug Logs Appearing
1. Check environment variable: `echo $PROMPTCHAIN_DEBUG`
2. Verify file permissions on `promptchain_debug.log`
3. Ensure AgenticStepProcessor is being used (multi-hop reasoning tools)

### Log File Too Large
```bash
# Rotate logs manually
mv promptchain_debug.log promptchain_debug.log.$(date +%Y%m%d)
touch promptchain_debug.log
```

### Filter Noise
```bash
# Only show errors and completions
grep -E "(error|complete)" promptchain_debug.log | jq '.'
```

## 🎉 Success Verification

You should see log entries like this when debug logging is working:

```json
{
  "event": "objective_start",
  "objective": "Find patient appointment patterns",
  "max_internal_steps": 5,
  "available_tools": ["lightrag_local_query", "lightrag_global_query"]
}
```

## 💡 Pro Tips

1. **Use `jq` for JSON parsing**: `cat promptchain_debug.log | jq '.'`
2. **Monitor specific tool performance**: `grep "tool_name.*your_tool" promptchain_debug.log`
3. **Track reasoning steps**: `grep "step_number" promptchain_debug.log | jq '.step_number' | sort -u`
4. **Find slow operations**: `grep "duration_seconds" promptchain_debug.log | jq 'select(.call_duration_seconds > 1)'`

The debug logging system is now ready for production use! 🚀