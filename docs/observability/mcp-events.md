# MCP Events Guide

This guide covers Model Context Protocol (MCP) specific events introduced in v0.4.1f.

## Overview

MCPHelper now emits events for connection lifecycle, tool discovery, and errors. This enables monitoring of external MCP server interactions.

### MCP Event Types

```python
class ExecutionEventType(Enum):
    # Connection lifecycle
    MCP_CONNECT_START = auto()      # Connection attempt started
    MCP_CONNECT_END = auto()        # Connection completed
    MCP_DISCONNECT_START = auto()   # Disconnection started
    MCP_DISCONNECT_END = auto()     # Disconnection completed

    # Tool discovery
    MCP_TOOL_DISCOVERED = auto()    # New MCP tool discovered

    # Error handling
    MCP_ERROR = auto()              # MCP-specific error occurred
```

## Connection Events

### MCP_CONNECT_START

Fired when attempting to connect to an MCP server.

**Metadata**:
```python
{
    "server_id": "filesystem",
    "command": "mcp-server-filesystem",
    "args": ["--root", "/path/to/project"],
    "transport": "stdio"  # or "sse"
}
```

**Example callback**:
```python
def on_mcp_connect_start(event: ExecutionEvent):
    server_id = event.metadata["server_id"]
    print(f"Connecting to MCP server: {server_id}")
    print(f"Command: {event.metadata['command']}")
```

### MCP_CONNECT_END

Fired when MCP server connection completes (success or failure).

**Metadata**:
```python
{
    "server_id": "filesystem",
    "status": "connected",  # or "failed"
    "execution_time_ms": 234.5,
    "error": None  # or error message if failed
}
```

**Example callback**:
```python
def on_mcp_connect_end(event: ExecutionEvent):
    server_id = event.metadata["server_id"]
    status = event.metadata["status"]

    if status == "connected":
        print(f"✓ Connected to {server_id}")
    else:
        error = event.metadata.get("error", "Unknown error")
        print(f"✗ Failed to connect to {server_id}: {error}")
```

## Tool Discovery Events

### MCP_TOOL_DISCOVERED

Fired for each tool discovered from an MCP server.

**Metadata**:
```python
{
    "server_id": "filesystem",
    "tool_name": "read_file",
    "description": "Read contents of a file",
    "schema": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        }
    },
    "mcp_prefixed_name": "mcp_filesystem_read_file"
}
```

**Example callback**:
```python
def on_tool_discovered(event: ExecutionEvent):
    tool_name = event.metadata["tool_name"]
    server_id = event.metadata["server_id"]
    prefixed_name = event.metadata["mcp_prefixed_name"]

    print(f"Discovered tool: {tool_name}")
    print(f"  From server: {server_id}")
    print(f"  Registered as: {prefixed_name}")
```

## Disconnection Events

### MCP_DISCONNECT_START

Fired when MCP server disconnection begins.

**Metadata**:
```python
{
    "server_id": "filesystem",
    "reason": "manual"  # or "cleanup", "error"
}
```

### MCP_DISCONNECT_END

Fired when MCP server disconnection completes.

**Metadata**:
```python
{
    "server_id": "filesystem",
    "status": "disconnected",  # or "failed"
    "execution_time_ms": 45.2,
    "error": None  # or error message if failed
}
```

## Error Events

### MCP_ERROR

Fired for MCP-specific errors.

**Metadata**:
```python
{
    "server_id": "filesystem",
    "error": "Connection timeout",
    "error_type": "ConnectionError",
    "phase": "connection",  # or "tool_discovery", "tool_execution", "disconnection"
    "retry_attempt": 0,
    "context": {
        # Additional context about the error
        "tool_name": "read_file",  # if during tool execution
        "args": {...}  # if during tool execution
    }
}
```

**Example callback**:
```python
def on_mcp_error(event: ExecutionEvent):
    server_id = event.metadata.get("server_id", "unknown")
    error = event.metadata["error"]
    phase = event.metadata.get("phase", "unknown")

    print(f"MCP Error [{server_id}] during {phase}: {error}")

    # Check if it's a tool execution error
    if "tool_name" in event.metadata.get("context", {}):
        tool = event.metadata["context"]["tool_name"]
        print(f"  Tool: {tool}")
```

## Complete MCP Monitoring Example

```python
from promptchain import PromptChain
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

class MCPMonitor:
    """Monitor all MCP operations."""

    def __init__(self):
        self.servers = {}
        self.tools_discovered = []
        self.errors = []

    def __call__(self, event: ExecutionEvent):
        """Handle all MCP events."""
        if event.event_type == ExecutionEventType.MCP_CONNECT_START:
            self.on_connect_start(event)
        elif event.event_type == ExecutionEventType.MCP_CONNECT_END:
            self.on_connect_end(event)
        elif event.event_type == ExecutionEventType.MCP_TOOL_DISCOVERED:
            self.on_tool_discovered(event)
        elif event.event_type == ExecutionEventType.MCP_DISCONNECT_START:
            self.on_disconnect_start(event)
        elif event.event_type == ExecutionEventType.MCP_DISCONNECT_END:
            self.on_disconnect_end(event)
        elif event.event_type == ExecutionEventType.MCP_ERROR:
            self.on_error(event)

    def on_connect_start(self, event: ExecutionEvent):
        server_id = event.metadata["server_id"]
        self.servers[server_id] = {
            "status": "connecting",
            "start_time": event.timestamp,
            "command": event.metadata["command"]
        }
        print(f"→ Connecting to {server_id}...")

    def on_connect_end(self, event: ExecutionEvent):
        server_id = event.metadata["server_id"]
        status = event.metadata["status"]
        self.servers[server_id]["status"] = status
        self.servers[server_id]["connection_time_ms"] = event.metadata["execution_time_ms"]

        if status == "connected":
            print(f"✓ {server_id} connected ({event.metadata['execution_time_ms']}ms)")
        else:
            print(f"✗ {server_id} failed: {event.metadata.get('error')}")

    def on_tool_discovered(self, event: ExecutionEvent):
        tool_info = {
            "server_id": event.metadata["server_id"],
            "tool_name": event.metadata["tool_name"],
            "prefixed_name": event.metadata["mcp_prefixed_name"]
        }
        self.tools_discovered.append(tool_info)
        print(f"  + Tool discovered: {tool_info['tool_name']} → {tool_info['prefixed_name']}")

    def on_disconnect_start(self, event: ExecutionEvent):
        server_id = event.metadata["server_id"]
        print(f"← Disconnecting from {server_id}...")

    def on_disconnect_end(self, event: ExecutionEvent):
        server_id = event.metadata["server_id"]
        status = event.metadata["status"]
        if status == "disconnected":
            print(f"✓ {server_id} disconnected")
        else:
            print(f"✗ {server_id} disconnect failed")

    def on_error(self, event: ExecutionEvent):
        error_info = {
            "server_id": event.metadata.get("server_id"),
            "error": event.metadata["error"],
            "phase": event.metadata.get("phase")
        }
        self.errors.append(error_info)
        print(f"⚠ MCP Error: {error_info['error']}")

    def get_summary(self):
        """Get summary of MCP operations."""
        return {
            "servers_connected": sum(
                1 for s in self.servers.values() if s["status"] == "connected"
            ),
            "total_servers": len(self.servers),
            "tools_discovered": len(self.tools_discovered),
            "errors": len(self.errors),
            "server_details": self.servers,
            "discovered_tools": self.tools_discovered
        }

# Usage
mcp_config = [
    {
        "id": "filesystem",
        "type": "stdio",
        "command": "mcp-server-filesystem",
        "args": ["--root", "./project"]
    },
    {
        "id": "github",
        "type": "stdio",
        "command": "mcp-server-github",
        "args": ["--token", "ghp_xxxxx"]
    }
]

chain = PromptChain(
    models=["openai/gpt-4"],
    instructions=["Analyze files: {input}"],
    mcp_servers=mcp_config
)

# Monitor MCP operations
monitor = MCPMonitor()
chain.register_callback(
    monitor,
    event_filter=[
        ExecutionEventType.MCP_CONNECT_START,
        ExecutionEventType.MCP_CONNECT_END,
        ExecutionEventType.MCP_TOOL_DISCOVERED,
        ExecutionEventType.MCP_DISCONNECT_START,
        ExecutionEventType.MCP_DISCONNECT_END,
        ExecutionEventType.MCP_ERROR
    ]
)

# Run chain
result = chain.process_prompt("Read and analyze config.json")

# Get MCP summary
summary = monitor.get_summary()
print(f"\nMCP Summary:")
print(f"  Servers connected: {summary['servers_connected']}/{summary['total_servers']}")
print(f"  Tools discovered: {summary['tools_discovered']}")
print(f"  Errors: {summary['errors']}")
```

## MCP Event Flow

```
User Code
    ↓
PromptChain initialization with mcp_servers
    ↓
MCPHelper.connect_mcp_async()
    ↓
For each MCP server config:
    ↓
    [MCP_CONNECT_START] ──→ Callbacks
        metadata: {server_id, command, transport}
    ↓
    Try connection:
        ↓
        Success:
            [MCP_CONNECT_END] ──→ Callbacks
                metadata: {server_id, status: "connected", time_ms}
            ↓
            List tools from server
            ↓
            For each tool:
                [MCP_TOOL_DISCOVERED] ──→ Callbacks
                    metadata: {server_id, tool_name, schema, prefixed_name}
        ↓
        Failure:
            [MCP_CONNECT_END] ──→ Callbacks
                metadata: {server_id, status: "failed", error}
            ↓
            [MCP_ERROR] ──→ Callbacks
                metadata: {server_id, error, phase: "connection"}
    ↓
Chain execution (tools available)
    ↓
Chain cleanup
    ↓
MCPHelper.disconnect_mcp_async()
    ↓
For each connected server:
    ↓
    [MCP_DISCONNECT_START] ──→ Callbacks
        metadata: {server_id, reason}
    ↓
    Try disconnection:
        ↓
        Success:
            [MCP_DISCONNECT_END] ──→ Callbacks
                metadata: {server_id, status: "disconnected"}
        ↓
        Failure:
            [MCP_DISCONNECT_END] ──→ Callbacks
                metadata: {server_id, status: "failed", error}
            ↓
            [MCP_ERROR] ──→ Callbacks
                metadata: {server_id, error, phase: "disconnection"}
```

## Tool Execution Events

When MCP tools are called during chain execution, they fire regular tool events (not MCP-specific):

```python
# MCP tool call
[TOOL_CALL_START] ──→ Callbacks
    metadata: {
        "tool_name": "mcp_filesystem_read_file",
        "tool_args": {"path": "config.json"},
        "is_mcp_tool": True,
        "server_id": "filesystem"
    }
    ↓
[TOOL_CALL_END] ──→ Callbacks
    metadata: {
        "execution_time_ms": 45.2,
        "result_length": 1500,
        "success": True
    }
```

If there's an error:
```python
[TOOL_CALL_ERROR] ──→ Callbacks
    metadata: {
        "error": "File not found",
        "tool_name": "mcp_filesystem_read_file"
    }
    ↓
[MCP_ERROR] ──→ Callbacks (if MCP-related)
    metadata: {
        "server_id": "filesystem",
        "error": "File not found",
        "phase": "tool_execution",
        "context": {
            "tool_name": "read_file",
            "args": {"path": "config.json"}
        }
    }
```

## Best Practices

### 1. Track Connection Health

```python
def connection_health_check(event: ExecutionEvent):
    if event.event_type == ExecutionEventType.MCP_CONNECT_END:
        if event.metadata["status"] == "failed":
            # Alert or retry logic
            alert_ops_team(f"MCP server {event.metadata['server_id']} failed to connect")

chain.register_callback(
    connection_health_check,
    event_filter=ExecutionEventType.MCP_CONNECT_END
)
```

### 2. Monitor Tool Availability

```python
class ToolInventory:
    def __init__(self):
        self.tools_by_server = {}

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MCP_TOOL_DISCOVERED:
            server_id = event.metadata["server_id"]
            if server_id not in self.tools_by_server:
                self.tools_by_server[server_id] = []
            self.tools_by_server[server_id].append(event.metadata["tool_name"])

    def get_available_tools(self, server_id: str):
        return self.tools_by_server.get(server_id, [])

inventory = ToolInventory()
chain.register_callback(inventory, event_filter=ExecutionEventType.MCP_TOOL_DISCOVERED)
```

### 3. Error Recovery

```python
class MCPErrorHandler:
    def __init__(self, chain):
        self.chain = chain
        self.retry_count = {}

    def __call__(self, event: ExecutionEvent):
        if event.event_type == ExecutionEventType.MCP_ERROR:
            server_id = event.metadata.get("server_id")
            error_phase = event.metadata.get("phase")

            if error_phase == "connection":
                # Retry connection
                retry_count = self.retry_count.get(server_id, 0)
                if retry_count < 3:
                    print(f"Retrying connection to {server_id} (attempt {retry_count + 1})")
                    self.retry_count[server_id] = retry_count + 1
                    # Implement retry logic here

handler = MCPErrorHandler(chain)
chain.register_callback(handler, event_filter=ExecutionEventType.MCP_ERROR)
```

## Performance Impact

- **Connection/disconnection events**: Negligible overhead (only at start/end)
- **Tool discovery events**: Minimal overhead (only during initialization)
- **Tool execution events**: Use regular TOOL_CALL events (same overhead as non-MCP tools)

## Next Steps

- Review [Event System Guide](event-system.md) for general event patterns
- Check [examples/observability/](../../examples/observability/) for working examples
- Read [Best Practices](best-practices.md) for production recommendations
