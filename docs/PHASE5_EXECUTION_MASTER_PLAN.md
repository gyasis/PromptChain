# Phase 5: MCP Integration - Execution Master Plan

**Feature**: CLI Orchestration Integration (002-cli-orchestration)
**Phase**: MCP Integration (T056-T069)
**Strategy**: Wave-based parallel execution with dependency synchronization
**Estimated Total Time**: 3.5-4 hours
**Token Savings**: ~65% vs sequential execution

---

## Executive Summary

**Phase 5 Goal**: Integrate Model Context Protocol (MCP) ecosystem into CLI for external tool discovery, registration, and execution.

**Parallelization Approach**: 4 waves combining independent parallel execution with synchronization gates at dependency boundaries.

**Key Dependencies**:
- **T061 (MCP Manager)**: Foundation for all MCP functionality - MUST complete before tool operations
- **T062 (Auto-connect)**: Depends on T061 connection lifecycle
- **T063-T064 (Tool Discovery/Prefixing)**: Depends on T061 connection manager
- **T065-T067 (Tool Commands)**: Depends on T063 (tool discovery)
- **T068-T068a (Graceful Failure)**: Depends on T061 connection logic
- **T069 (TUI Status)**: Depends on T061 state tracking

**Existing Infrastructure**:
- ✅ `promptchain/cli/utils/mcp_manager.py` - ALREADY IMPLEMENTED (T061 foundation)
- ✅ `promptchain/cli/models/mcp_config.py` - MCPServerConfig with state tracking
- ✅ `promptchain/cli/session_manager.py` - MCP server persistence (T062 save/load)
- ✅ `promptchain/utils/mcp_helpers.py` - MCPHelper for actual MCP protocol
- ✅ Tool prefixing pattern established (mcp_{server_id}_{tool_name})

---

## Wave 1: Foundation Tests + Contract Validation (30-35 min) - PARALLEL

**Goal**: Establish testing foundation and validate MCP server configuration schema.

**Tasks**: T056, T057, T060 (3 test tasks)

**Synchronization Point**: All tests pass before Wave 2 implementation begins.

### T056: Contract Test - MCP Server Config Schema [P]

**Agent**: `test-automator`

**File**: `tests/cli/contract/test_mcp_server_contract.py` ✅ ALREADY EXISTS

**Objective**: Validate MCPServerConfig schema against contract requirements.

**Contract Validation**:
```python
# Test MCPServerConfig schema matches data-model.md contract:
# 1. id: str (1-50 chars, alphanumeric + dash/underscore)
# 2. type: Literal["stdio", "http"]
# 3. command: Optional[str] (required if type=stdio)
# 4. args: List[str]
# 5. url: Optional[str] (required if type=http)
# 6. auto_connect: bool (default False)
# 7. state: Literal["disconnected", "connected", "error"]
# 8. discovered_tools: List[str]
# 9. error_message: Optional[str]
# 10. connected_at: Optional[float]

# Test state transitions:
# disconnected → connected (mark_connected with tools)
# disconnected → error (mark_error with message)
# connected → disconnected (mark_disconnected)
# error → disconnected (mark_disconnected)

# Test validation errors:
# - Empty id raises ValueError
# - id > 50 chars raises ValueError
# - stdio without command raises ValueError
# - http without url raises ValueError
# - Invalid type raises ValueError
```

**Test Cases**:
1. Valid stdio config with command
2. Valid http config with URL
3. State transition validation (disconnected → connected → disconnected)
4. Error state with message
5. Validation errors (missing command, missing URL, invalid type)
6. Tool discovery tracking
7. JSON serialization/deserialization (to_dict, from_dict)

**Expected Output**: Contract test file with 15-20 test cases covering all schema requirements.

**Integration Point**: Uses `promptchain/cli/models/mcp_config.py` (already exists).

---

### T057: Integration Test - MCP Server Lifecycle [P]

**Agent**: `test-automator`

**File**: `tests/cli/integration/test_mcp_integration.py` ✅ ALREADY EXISTS

**Objective**: Test connect/disconnect/reconnect lifecycle using MCPManager.

**Test Scenarios**:
```python
# Lifecycle test flow:
# 1. Create session with MCP server config
# 2. Connect to server (verify state=connected, tools discovered)
# 3. Disconnect from server (verify state=disconnected, tools cleared)
# 4. Reconnect to server (verify state=connected, tools rediscovered)
# 5. Simulate connection failure (verify state=error, error_message set)
# 6. Test graceful shutdown (disconnect all servers)

# Mock MCPHelper to simulate:
# - Successful connection with tool discovery
# - Connection failure with error
# - Disconnection cleanup
```

**Test Cases**:
1. Successful connection with tool discovery
2. Connection failure handling (state=error)
3. Disconnection clears state
4. Reconnection after failure
5. Multiple server management
6. Auto-connect on session load
7. Graceful shutdown disconnects all servers

**Expected Output**: Integration test file with 10-12 lifecycle test cases.

**Integration Point**: Uses `promptchain/cli/utils/mcp_manager.py` (already exists).

---

### T060: Unit Test - MCP Tool Name Prefixing [P]

**Agent**: `test-automator`

**File**: `tests/cli/unit/test_mcp_tool_prefixing.py` ✅ ALREADY EXISTS

**Objective**: Test tool name prefix logic (mcp_{server_id}_{tool_name}).

**Test Scenarios**:
```python
# Prefixing logic tests:
# 1. Prefix MCP tool: "read_file" → "mcp_filesystem_read_file"
# 2. Multiple servers: same tool gets different prefixes
#    - "filesystem_1.read" → "mcp_filesystem_1_read"
#    - "filesystem_2.read" → "mcp_filesystem_2_read"
# 3. Conflict detection: local tool "mcp_filesystem_read" exists
#    → MCP tool registration fails with warning
# 4. No prefix for local tools (remain unchanged)
# 5. Long server IDs handled correctly
# 6. Special characters in tool names sanitized

# Use promptchain/cli/utils/mcp_tool_prefixing.py helper
```

**Test Cases**:
1. Basic prefixing (tool → mcp_server_tool)
2. Multiple servers with same tool name
3. Conflict with local tool (reject MCP tool)
4. Long server ID handling
5. Special character sanitization
6. Prefix removal (unprefixing utility)
7. Prefix validation (correct format check)

**Expected Output**: Unit test file with 10-15 prefixing test cases.

**Integration Point**: Uses `promptchain/cli/utils/mcp_tool_prefixing.py` (may need creation).

---

**Wave 1 Completion Criteria**:
- ✅ All 3 test files pass (T056, T057, T060)
- ✅ Contract tests validate MCPServerConfig schema
- ✅ Lifecycle tests demonstrate connect/disconnect/error handling
- ✅ Prefixing tests demonstrate conflict resolution

**Token Optimization**: 3 agents running in parallel saves ~60 minutes vs sequential.

---

## Wave 2: Core MCP Infrastructure Implementation (40-45 min) - SEQUENTIAL

**Goal**: Implement core MCP connection management and tool discovery.

**Tasks**: T061, T062, T063, T064 (4 implementation tasks)

**Synchronization Point**: T061 must complete before T062, T063, T064 (dependency tree).

**Why Sequential**: T061 is the foundation that T062-T064 build upon. However, T063 and T064 can run in parallel after T061 completes.

### T061: MCP Connection Manager Implementation [P]

**Agent**: `backend-architect`

**File**: `promptchain/cli/utils/mcp_manager.py` ✅ ALREADY IMPLEMENTED

**Objective**: Verify and enhance existing MCPManager implementation.

**Current Status**: File EXISTS with complete implementation:
- ✅ `connect_server(server_id)` - Uses real MCPHelper
- ✅ `disconnect_server(server_id)` - Cleanup with tool unregistration
- ✅ `register_tools_with_agent(server_id, agent_name)` - T063 support
- ✅ `unregister_tools_from_agent(server_id, agent_name)` - T067 support
- ✅ `connect_all_auto_servers()` - T062 support
- ✅ `disconnect_all_servers()` - Graceful shutdown
- ✅ `get_connected_servers()` - Status queries
- ✅ `get_all_discovered_tools()` - Tool inventory

**Implementation Review Checklist**:
- [x] MCPHelper integration for real MCP protocol
- [x] State synchronization with MCPServerConfig
- [x] Error handling with state transitions (connected/error/disconnected)
- [x] Tool discovery storage in server.discovered_tools
- [x] Logging for debugging

**Enhancement Opportunities** (if needed):
- Add retry logic for transient connection failures?
- Add connection timeout configuration?
- Add health check pings for long-running connections?

**Expected Outcome**: Code review confirming implementation meets all T061 requirements OR minor enhancements if gaps found.

**Integration Point**: Uses `promptchain/utils/mcp_helpers.py` and `promptchain/cli/models/mcp_config.py`.

---

### T062: Auto-Connect MCP Servers at Session Init

**Agent**: `backend-architect`

**File**: `promptchain/cli/session_manager.py` (already has auto-connect logic)

**Objective**: Verify auto-connect implementation in create_session() and load_session().

**Current Status**: File ALREADY HAS auto-connect logic at lines 229-257 (create_session) and 476-504 (load_session):

```python
# From session_manager.py:
if mcp_servers:
    mcp_manager = MCPManager(session)
    async def auto_connect_servers():
        for server in mcp_servers:
            if server.auto_connect:
                try:
                    await mcp_manager.connect_server(server.id)
                except Exception as e:
                    pass  # Graceful degradation

    # Run synchronously using asyncio
    loop.run_until_complete(auto_connect_servers())
```

**Implementation Review Checklist**:
- [x] Auto-connect in create_session() for new sessions
- [x] Auto-connect in load_session() for restored sessions
- [x] Graceful failure handling (continue if server fails)
- [x] Async execution using nest_asyncio for compatibility
- [x] Error logging (uses server.mark_error internally)

**Enhancement Opportunities** (if needed):
- Add connection summary logging ("Connected 2/3 servers, 1 failed")?
- Add retry delay for failed servers?
- Add user notification for failed auto-connects?

**Expected Outcome**: Code review confirming auto-connect works correctly OR minor enhancements.

**Integration Point**: Uses `promptchain/cli/utils/mcp_manager.py` (T061).

---

### T063: Tool Discovery and Registration (PARALLEL with T064 after T061)

**Agent**: `backend-architect`

**File**: `promptchain/cli/utils/mcp_manager.py` (already has register_tools_with_agent)

**Objective**: Verify tool discovery and agent registration implementation.

**Current Status**: File ALREADY HAS implementation at lines 137-181:

```python
async def register_tools_with_agent(
    self, server_id: str, agent_name: str
) -> bool:
    # Finds server, verifies connected state
    # Adds discovered_tools to agent.tools list
    # Returns True on success
```

**Implementation Review Checklist**:
- [x] Verify server is connected before registration
- [x] Check agent exists in session
- [x] Add tools to agent.tools list (avoid duplicates)
- [x] Return success/failure status
- [x] Error logging

**Test Integration**:
```python
# Test in tests/cli/integration/test_mcp_tool_registration.py:
# 1. Connect server with tools ["read", "write"]
# 2. Register tools with agent "default"
# 3. Verify agent.tools contains ["mcp_filesystem_read", "mcp_filesystem_write"]
# 4. Test registration with disconnected server (should fail)
# 5. Test registration with non-existent agent (should fail)
```

**Expected Outcome**: Code review + integration test confirming registration works.

**Integration Point**: Uses `agent.tools` list from Agent model.

---

### T064: MCP Tool Name Prefixing (PARALLEL with T063 after T061)

**Agent**: `backend-architect`

**File**: `promptchain/cli/utils/mcp_tool_prefixing.py` (may need creation)

**Objective**: Implement tool name prefixing utility functions.

**Current Status**: MCPManager already applies prefixing via MCPHelper (which uses mcp_{server_id}_{tool_name} format automatically).

**Utility Functions Needed**:
```python
def prefix_tool_name(server_id: str, tool_name: str) -> str:
    """Add MCP prefix: read → mcp_filesystem_read"""
    return f"mcp_{server_id}_{tool_name}"

def unprefix_tool_name(prefixed_name: str) -> tuple[str, str]:
    """Extract server_id and tool_name from prefix"""
    # "mcp_filesystem_read" → ("filesystem", "read")
    if not prefixed_name.startswith("mcp_"):
        raise ValueError("Not a prefixed tool name")
    parts = prefixed_name[4:].split("_", 1)
    if len(parts) != 2:
        raise ValueError("Invalid prefix format")
    return parts[0], parts[1]

def is_prefixed_tool(tool_name: str) -> bool:
    """Check if tool name has MCP prefix"""
    return tool_name.startswith("mcp_")

def check_tool_conflict(tool_name: str, existing_tools: List[str]) -> bool:
    """Check if tool name conflicts with existing tools"""
    return tool_name in existing_tools
```

**Implementation**:
- Create utility file with 4-5 helper functions
- Add validation for server_id (alphanumeric + dash/underscore)
- Add sanitization for special characters

**Test Coverage**: Unit tests in T060 cover these functions.

**Expected Outcome**: Utility file with prefixing helpers + unit tests.

**Integration Point**: Used by MCPManager during tool registration.

---

**Wave 2 Completion Criteria**:
- ✅ T061: MCPManager verified/enhanced (connection lifecycle works)
- ✅ T062: Auto-connect verified in session manager
- ✅ T063: Tool registration verified
- ✅ T064: Prefixing utilities created and tested

**Parallelization Note**: T063 and T064 can run in parallel AFTER T061 completes (both depend on MCPManager existing).

**Token Optimization**: T063+T064 in parallel saves ~20 minutes vs sequential.

---

## Wave 3: MCP Tool Commands + TUI Integration (35-40 min) - PARALLEL

**Goal**: Implement user-facing commands for MCP tool management and TUI status display.

**Tasks**: T065, T066, T067, T069 (4 implementation tasks)

**Synchronization Point**: All tasks depend on Wave 2 (T063 tool discovery), but can run in parallel with each other.

### T065: `/tools list` Command

**Agent**: `backend-architect`

**File**: `promptchain/cli/command_handler.py` ✅ ALREADY HAS handle_tools_list()

**Objective**: Verify `/tools list` implementation for displaying available MCP tools.

**Current Status**: File ALREADY HAS implementation at lines 897-985:

```python
async def handle_tools_list(self, session) -> CommandResult:
    # Gets connected servers
    # Collects discovered tools from each server
    # Shows registration status per agent
    # Groups tools by server
```

**Implementation Review Checklist**:
- [x] List tools from all connected servers
- [x] Group tools by server ID
- [x] Show registration status (which agents have each tool)
- [x] Handle no connected servers gracefully
- [x] Handle servers with no tools
- [x] Format output clearly

**Output Format**:
```
Available MCP Tools (5 total from 2 servers):

Server: filesystem (3 tools)
  - mcp_filesystem_read (registered with: coder, analyst)
  - mcp_filesystem_write (registered with: coder)
  - mcp_filesystem_list (not registered)

Server: web_search (2 tools)
  - mcp_web_search_google (registered with: researcher)
  - mcp_web_search_brave (not registered)
```

**Expected Outcome**: Code review confirming command works correctly.

**Integration Point**: Uses `session.mcp_servers` and `agent.tools`.

---

### T066: `/tools add` Command (PARALLEL with T065, T067, T069)

**Agent**: `backend-architect`

**File**: `promptchain/cli/command_handler.py` ✅ ALREADY HAS handle_tools_add()

**Objective**: Verify `/tools add` implementation for registering server tools with agent.

**Current Status**: File ALREADY HAS implementation at lines 987-1071:

```python
async def handle_tools_add(self, session, server_id: str) -> CommandResult:
    # Checks current agent exists
    # Finds server by ID
    # Verifies server is connected
    # Calls MCPManager.register_tools_with_agent()
```

**Implementation Review Checklist**:
- [x] Check current agent set
- [x] Find server by ID
- [x] Verify server is connected
- [x] Call MCPManager.register_tools_with_agent()
- [x] Return success message with tool count
- [x] Error handling for all failure modes

**Command Syntax**: `/tools add <server_id>`

**Success Output**: `Successfully registered 3 tools from 'filesystem' with agent 'default'.`

**Error Cases**:
- No current agent set
- Server not found
- Server not connected
- Registration failure

**Expected Outcome**: Code review confirming command works correctly.

**Integration Point**: Uses `MCPManager.register_tools_with_agent()` (T063).

---

### T067: `/tools remove` Command (PARALLEL with T065, T066, T069)

**Agent**: `backend-architect`

**File**: `promptchain/cli/command_handler.py` ✅ ALREADY HAS handle_tools_remove()

**Objective**: Verify `/tools remove` implementation for unregistering server tools.

**Current Status**: File ALREADY HAS implementation at lines 1073-1149:

```python
async def handle_tools_remove(self, session, server_id: str) -> CommandResult:
    # Checks current agent exists
    # Finds server by ID
    # Calls MCPManager.unregister_tools_from_agent()
```

**Implementation Review Checklist**:
- [x] Check current agent set
- [x] Find server by ID
- [x] Call MCPManager.unregister_tools_from_agent()
- [x] Return success message with tool count
- [x] Error handling

**Command Syntax**: `/tools remove <server_id>`

**Success Output**: `Successfully unregistered 3 tools from 'filesystem' with agent 'default'.`

**Expected Outcome**: Code review confirming command works correctly.

**Integration Point**: Uses `MCPManager.unregister_tools_from_agent()`.

---

### T069: MCP Status Display in TUI (PARALLEL with T065, T066, T067)

**Agent**: `frontend-developer`

**File**: `promptchain/cli/tui/status_bar.py` ✅ ALREADY HAS MCP status display

**Objective**: Verify MCP server status display in TUI status bar.

**Current Status**: File ALREADY HAS implementation at lines 32-107:

```python
class StatusBar(Static):
    mcp_servers: reactive[List[Dict[str, str]]] = reactive([])  # T069

    def render(self) -> str:
        # MCP server status display (T069) at lines 88-106
        if self.mcp_servers:
            for server in self.mcp_servers:
                server_id = server.get("id")
                state = server.get("state")

                # Emoji indicators:
                # ✓ = connected (green)
                # ✗ = error (red)
                # ○ = disconnected (dim)

                mcp_status_parts.append(f"{emoji}{server_id}")

            parts.append(f"MCP: {' '.join(mcp_status_parts)}")
```

**Implementation Review Checklist**:
- [x] Add mcp_servers reactive field
- [x] Render MCP status in status bar
- [x] Use emoji indicators (✓/✗/○)
- [x] Show server IDs with state
- [x] Update on server state changes

**Status Bar Format**: `MCP: ✓filesystem ✗web_search ○code_exec`

**Update Trigger**: `status_bar.update_session_info(mcp_servers=[...])`

**Expected Outcome**: Code review confirming TUI integration works correctly.

**Integration Point**: Called from TUI app when session changes.

---

**Wave 3 Completion Criteria**:
- ✅ T065: `/tools list` command verified
- ✅ T066: `/tools add` command verified
- ✅ T067: `/tools remove` command verified
- ✅ T069: TUI status display verified

**Token Optimization**: 4 agents in parallel saves ~30-40 minutes vs sequential.

---

## Wave 4: Graceful Failure Handling + Integration Tests (30-35 min) - PARALLEL

**Goal**: Implement graceful degradation for MCP failures and verify end-to-end integration.

**Tasks**: T058, T059, T068, T068a (4 tasks: 2 tests + 2 implementation)

**Synchronization Point**: All previous waves must complete before testing full integration.

### T058: Integration Test - Tool Discovery and Registration (PARALLEL)

**Agent**: `test-automator`

**File**: `tests/cli/integration/test_mcp_tool_discovery.py` ✅ ALREADY EXISTS

**Objective**: Test end-to-end tool discovery from MCP server connection.

**Test Scenarios**:
```python
# Full discovery flow:
# 1. Create session with MCP server config
# 2. Connect to server (auto or manual)
# 3. Verify tools discovered and stored in server.discovered_tools
# 4. Verify tool schemas include proper prefixes
# 5. Test discovery with multiple servers
# 6. Test discovery with server restart (rediscovery)

# Mock MCPHelper.connect_mcp_async() to return tool schemas:
# [
#   {"function": {"name": "mcp_filesystem_read", ...}},
#   {"function": {"name": "mcp_filesystem_write", ...}}
# ]
```

**Test Cases**:
1. Single server discovery
2. Multiple server discovery
3. Rediscovery after reconnect
4. Discovery with empty tool list
5. Discovery failure handling
6. Tool schema validation

**Expected Output**: Integration test with 8-10 discovery test cases.

**Integration Point**: Uses MCPManager.connect_server() with mocked MCPHelper.

---

### T059: Integration Test - Tool Execution in Conversation (PARALLEL)

**Agent**: `test-automator`

**File**: `tests/cli/integration/test_mcp_tool_execution.py` ✅ ALREADY EXISTS

**Objective**: Test MCP tool execution within agent conversation flow.

**Test Scenarios**:
```python
# Execution flow:
# 1. Create session with agent
# 2. Connect MCP server and discover tools
# 3. Register tools with agent
# 4. Send user message triggering tool call
# 5. Verify AgentChain executes MCP tool via MCPHelper
# 6. Verify tool result returned in conversation

# Mock execution stack:
# - User: "Read the file README.md"
# - Agent calls: mcp_filesystem_read(path="README.md")
# - MCPHelper executes tool
# - Agent receives result and responds
```

**Test Cases**:
1. Single tool execution
2. Multiple tool calls in sequence
3. Tool execution with arguments
4. Tool execution failure handling
5. Tool not registered error
6. Conversation context preserved after tool call

**Expected Output**: Integration test with 8-10 execution test cases.

**Integration Point**: Uses AgentChain with MCP tools registered.

---

### T068: Graceful MCP Failure Handling (PARALLEL)

**Agent**: `backend-architect`

**File**: `promptchain/cli/utils/mcp_manager.py` (enhance error handling)

**Objective**: Implement graceful degradation for MCP connection failures.

**Current Status**: Basic error handling EXISTS (server.mark_error), but may need enhancement.

**Graceful Failure Scenarios**:
```python
# Scenario 1: Server unreachable at session start
# - Attempt connection
# - Mark server as error state with message
# - Continue session initialization with available servers
# - Log warning: "MCP server 'web_search' failed to connect: Connection refused"
# - User notification: "Session started with 2/3 MCP servers connected"

# Scenario 2: Tool discovery timeout
# - Attempt connection with 5-second timeout
# - If timeout, mark error and continue
# - Log: "MCP server 'code_exec' discovery timeout"

# Scenario 3: Mid-session server disconnect
# - Detect connection loss during tool call
# - Mark server as error
# - Unregister tools from affected agents
# - Notify user: "MCP server 'filesystem' disconnected, tools unavailable"

# Scenario 4: Partial server availability
# - Connect to available servers
# - Skip unavailable servers
# - Session continues with partial tool access
```

**Implementation Enhancements**:
- Add connection timeout (default 10 seconds)
- Add retry logic with exponential backoff (3 attempts)
- Add health check pings for long-running connections
- Add graceful disconnect notification
- Add error summary in session load

**Expected Outcome**: Enhanced MCPManager with production-ready error handling.

**Integration Point**: Used during session initialization and tool operations.

---

### T068a: Test Graceful Degradation (PARALLEL)

**Agent**: `test-automator`

**File**: `tests/cli/integration/test_mcp_graceful_degradation.py` ✅ ALREADY EXISTS

**Objective**: Test graceful failure handling in various scenarios.

**Test Scenarios**:
```python
# Test graceful failure modes:
# 1. Server unreachable at session start (verify session continues)
# 2. Server timeout during connection (verify timeout handling)
# 3. Tool discovery fails (verify error state but session works)
# 4. Mid-session disconnect (verify tools unregistered)
# 5. Partial server availability (verify working servers used)
# 6. All servers fail (verify session works without MCP tools)
# 7. Server recovery after failure (verify reconnection works)
```

**Test Cases**:
1. Session start with 1/3 servers failing
2. Connection timeout handling
3. Discovery failure with error state
4. Mid-session disconnect recovery
5. All servers unavailable (no MCP tools)
6. Server reconnection after failure
7. User notification for failures

**Expected Output**: Integration test with 10-12 failure test cases.

**Integration Point**: Uses MCPManager with simulated failures.

---

**Wave 4 Completion Criteria**:
- ✅ T058: Tool discovery integration test passes
- ✅ T059: Tool execution integration test passes
- ✅ T068: Graceful failure handling implemented
- ✅ T068a: Graceful degradation test passes

**Token Optimization**: 4 agents in parallel saves ~25-30 minutes vs sequential.

---

## Summary of Parallelization Strategy

### Wave Breakdown

| Wave | Tasks | Time | Agents | Parallelization |
|------|-------|------|--------|-----------------|
| Wave 1 | T056, T057, T060 (Tests) | 30-35 min | 3 test-automator | PARALLEL |
| Wave 2 | T061, T062, T063, T064 (Core) | 40-45 min | 1-2 backend-architect | SEQUENTIAL + PARTIAL PARALLEL |
| Wave 3 | T065, T066, T067, T069 (Commands) | 35-40 min | 3 backend + 1 frontend | PARALLEL |
| Wave 4 | T058, T059, T068, T068a (Integration) | 30-35 min | 2 test + 2 backend | PARALLEL |

**Total Sequential Time**: ~6-7 hours
**Total Parallel Time**: ~3.5-4 hours
**Time Savings**: ~65% reduction

---

### Dependency Graph

```
Wave 1 (PARALLEL - No dependencies)
├── T056 (Contract test)
├── T057 (Lifecycle test)
└── T060 (Prefixing test)
        ↓
Wave 2 (SEQUENTIAL + PARTIAL PARALLEL)
├── T061 (MCP Manager) ← FOUNDATION (must complete first)
├── T062 (Auto-connect) ← Depends on T061
├── T063 (Tool Discovery) ← Depends on T061
└── T064 (Prefixing) ← Depends on T061
        ↓
Wave 3 (PARALLEL - All depend on Wave 2)
├── T065 (/tools list) ← Depends on T063
├── T066 (/tools add) ← Depends on T063
├── T067 (/tools remove) ← Depends on T063
└── T069 (TUI status) ← Depends on T061
        ↓
Wave 4 (PARALLEL - All depend on Wave 2+3)
├── T058 (Discovery test) ← Depends on T063
├── T059 (Execution test) ← Depends on T063
├── T068 (Graceful failure) ← Depends on T061
└── T068a (Failure test) ← Depends on T068
```

---

### Token Savings Calculation

**Sequential Execution** (baseline):
- Wave 1: 30 min × 3 tasks = 90 min
- Wave 2: 40 min × 4 tasks = 160 min
- Wave 3: 35 min × 4 tasks = 140 min
- Wave 4: 30 min × 4 tasks = 120 min
- **Total**: 510 minutes (~8.5 hours)

**Parallel Execution** (optimized):
- Wave 1: 35 min (3 agents in parallel)
- Wave 2: 65 min (T061 sequential, then T063+T064 parallel)
- Wave 3: 40 min (4 agents in parallel)
- Wave 4: 35 min (4 agents in parallel)
- **Total**: 175 minutes (~2.9 hours)

**Token Savings**: (510 - 175) / 510 = 65.7% reduction

**Estimated Tokens Saved**:
- Sequential: ~510,000 tokens (1000 tokens/min average)
- Parallel: ~175,000 tokens
- **Savings**: ~335,000 tokens (~$0.33 at $1/1M tokens)

---

## Execution Instructions

### Wave 1: Foundation Tests (START HERE)

Spawn 3 test-automator agents in parallel:

**Agent 1 Prompt**:
```
Task: T056 - Implement contract test for MCPServerConfig schema validation

File: tests/cli/contract/test_mcp_server_contract.py (already exists - review and enhance)

Requirements:
1. Test MCPServerConfig schema matches data-model.md contract
2. Validate all required fields (id, type, command/url, state, etc.)
3. Test state transitions (disconnected → connected → error)
4. Test validation errors (missing command, invalid type)
5. Test JSON serialization/deserialization

Use: promptchain/cli/models/mcp_config.py

Expected: 15-20 test cases covering all schema requirements
```

**Agent 2 Prompt**:
```
Task: T057 - Implement integration test for MCP server lifecycle

File: tests/cli/integration/test_mcp_integration.py (already exists - review and enhance)

Requirements:
1. Test connect/disconnect/reconnect lifecycle
2. Test connection failure handling (error state)
3. Test tool discovery during connection
4. Test graceful shutdown (disconnect all servers)
5. Mock MCPHelper to simulate connection scenarios

Use: promptchain/cli/utils/mcp_manager.py

Expected: 10-12 lifecycle test cases
```

**Agent 3 Prompt**:
```
Task: T060 - Implement unit test for MCP tool name prefixing

File: tests/cli/unit/test_mcp_tool_prefixing.py (already exists - review and enhance)

Requirements:
1. Test prefix logic: "read" → "mcp_filesystem_read"
2. Test multiple servers with same tool name
3. Test conflict detection with local tools
4. Test special character handling
5. Test prefix removal utility

Use: promptchain/cli/utils/mcp_tool_prefixing.py (may need creation)

Expected: 10-15 prefixing test cases
```

**Synchronization**: Wait for all 3 agents to complete and tests to pass before proceeding to Wave 2.

---

### Wave 2: Core Infrastructure (AFTER Wave 1)

**Sub-Wave 2a: Foundation (SEQUENTIAL)**

Spawn 1 backend-architect agent:

**Agent Prompt**:
```
Task: T061 - Review and enhance MCP connection manager

File: promptchain/cli/utils/mcp_manager.py (ALREADY IMPLEMENTED - review and enhance)

Current Status:
- connect_server() - IMPLEMENTED
- disconnect_server() - IMPLEMENTED
- register_tools_with_agent() - IMPLEMENTED
- Auto-connect logic - IMPLEMENTED

Review Checklist:
1. Verify MCPHelper integration for real MCP protocol
2. Check error handling and state transitions
3. Add connection timeout if missing
4. Add retry logic if missing
5. Confirm logging is adequate

Enhancement Opportunities:
- Connection timeout (default 10 seconds)
- Retry with exponential backoff
- Health check for long-running connections

Expected: Code review report + any enhancements needed
```

**Sub-Wave 2b: Dependent Implementation (PARALLEL after T061)**

Spawn 3 backend-architect agents in parallel:

**Agent 1 Prompt**:
```
Task: T062 - Review auto-connect implementation in session manager

File: promptchain/cli/session_manager.py (ALREADY HAS auto-connect logic)

Current Status:
- Lines 229-257: Auto-connect in create_session()
- Lines 476-504: Auto-connect in load_session()

Review Checklist:
1. Verify auto-connect runs for servers with auto_connect=True
2. Check graceful failure handling (continue if server fails)
3. Confirm async execution using nest_asyncio

Enhancement Opportunities:
- Add connection summary logging
- Add retry delay for failed servers
- Add user notification

Expected: Code review report + any enhancements
```

**Agent 2 Prompt**:
```
Task: T063 - Review tool discovery and registration

File: promptchain/cli/utils/mcp_manager.py (ALREADY HAS register_tools_with_agent)

Current Status:
- Lines 137-181: register_tools_with_agent() implemented

Review Checklist:
1. Verify server connection check
2. Check agent existence validation
3. Confirm tools added to agent.tools list
4. Check duplicate prevention

Test Integration:
- Create integration test in tests/cli/integration/test_mcp_tool_registration.py

Expected: Code review + integration test
```

**Agent 3 Prompt**:
```
Task: T064 - Create MCP tool name prefixing utilities

File: promptchain/cli/utils/mcp_tool_prefixing.py (NEW FILE)

Requirements:
1. prefix_tool_name(server_id, tool_name) → "mcp_{server}_{tool}"
2. unprefix_tool_name(prefixed) → (server_id, tool_name)
3. is_prefixed_tool(name) → bool
4. check_tool_conflict(name, existing_tools) → bool

Implementation:
- Add validation for server_id format
- Add special character sanitization
- Add comprehensive error handling

Test Coverage: Unit tests in T060 cover these functions

Expected: Utility file with 4-5 helper functions
```

**Synchronization**: Wait for all sub-wave 2b agents to complete before proceeding to Wave 3.

---

### Wave 3: Commands + TUI (AFTER Wave 2)

Spawn 4 agents in parallel (3 backend + 1 frontend):

**Agent 1 Prompt**:
```
Task: T065 - Review /tools list command implementation

File: promptchain/cli/command_handler.py (ALREADY HAS handle_tools_list)

Current Status:
- Lines 897-985: handle_tools_list() implemented

Review Checklist:
1. Verify listing tools from all connected servers
2. Check grouping by server ID
3. Confirm registration status display (which agents have each tool)
4. Test error handling (no servers, no tools)

Expected Output Format:
Available MCP Tools (5 total from 2 servers):

Server: filesystem (3 tools)
  - mcp_filesystem_read (registered with: coder, analyst)
  - mcp_filesystem_write (not registered)

Expected: Code review report
```

**Agent 2 Prompt**:
```
Task: T066 - Review /tools add command implementation

File: promptchain/cli/command_handler.py (ALREADY HAS handle_tools_add)

Current Status:
- Lines 987-1071: handle_tools_add() implemented

Review Checklist:
1. Check current agent validation
2. Verify server ID lookup
3. Confirm server connection check
4. Test MCPManager.register_tools_with_agent() call
5. Check error handling for all failure modes

Command: /tools add <server_id>
Success: "Successfully registered 3 tools from 'filesystem' with agent 'default'."

Expected: Code review report
```

**Agent 3 Prompt**:
```
Task: T067 - Review /tools remove command implementation

File: promptchain/cli/command_handler.py (ALREADY HAS handle_tools_remove)

Current Status:
- Lines 1073-1149: handle_tools_remove() implemented

Review Checklist:
1. Check current agent validation
2. Verify server ID lookup
3. Test MCPManager.unregister_tools_from_agent() call
4. Check error handling

Command: /tools remove <server_id>
Success: "Successfully unregistered 3 tools from 'filesystem' with agent 'default'."

Expected: Code review report
```

**Agent 4 Prompt** (frontend-developer):
```
Task: T069 - Review MCP status display in TUI status bar

File: promptchain/cli/tui/status_bar.py (ALREADY HAS MCP status display)

Current Status:
- Line 32: mcp_servers reactive field
- Lines 88-106: MCP status rendering

Review Checklist:
1. Verify mcp_servers reactive field
2. Check emoji indicators (✓/✗/○)
3. Confirm server ID display with state
4. Test update trigger: update_session_info(mcp_servers=[...])

Status Format: MCP: ✓filesystem ✗web_search ○code_exec

Expected: Code review report + TUI integration test
```

**Synchronization**: Wait for all Wave 3 agents to complete before proceeding to Wave 4.

---

### Wave 4: Integration Tests + Graceful Failure (AFTER Wave 3)

Spawn 4 agents in parallel (2 test + 2 backend):

**Agent 1 Prompt** (test-automator):
```
Task: T058 - Implement integration test for tool discovery

File: tests/cli/integration/test_mcp_tool_discovery.py (already exists - review and enhance)

Requirements:
1. Test end-to-end tool discovery from MCP connection
2. Test single server and multiple server discovery
3. Test rediscovery after reconnect
4. Test discovery with empty tool list
5. Mock MCPHelper.connect_mcp_async() to return tool schemas

Test Cases:
- Single server discovery
- Multiple server discovery
- Rediscovery after reconnect
- Discovery failure handling

Expected: 8-10 discovery test cases
```

**Agent 2 Prompt** (test-automator):
```
Task: T059 - Implement integration test for tool execution

File: tests/cli/integration/test_mcp_tool_execution.py (already exists - review and enhance)

Requirements:
1. Test MCP tool execution within agent conversation
2. Test single tool call and multiple tool calls
3. Test tool execution with arguments
4. Test tool execution failure handling
5. Mock AgentChain execution with MCP tools

Test Flow:
- User: "Read file README.md"
- Agent calls: mcp_filesystem_read(path="README.md")
- MCPHelper executes tool
- Agent receives result

Expected: 8-10 execution test cases
```

**Agent 3 Prompt** (backend-architect):
```
Task: T068 - Enhance graceful MCP failure handling

File: promptchain/cli/utils/mcp_manager.py (enhance error handling)

Requirements:
1. Add connection timeout (default 10 seconds)
2. Add retry logic with exponential backoff (3 attempts)
3. Add health check pings for long-running connections
4. Add graceful disconnect notification
5. Add error summary in session load

Graceful Failure Scenarios:
- Server unreachable at session start (continue with available)
- Tool discovery timeout (mark error, continue)
- Mid-session disconnect (unregister tools, notify user)
- Partial server availability (use working servers)

Expected: Enhanced MCPManager with production-ready error handling
```

**Agent 4 Prompt** (test-automator):
```
Task: T068a - Implement graceful degradation tests

File: tests/cli/integration/test_mcp_graceful_degradation.py (already exists - review and enhance)

Requirements:
1. Test session start with failing servers
2. Test connection timeout handling
3. Test mid-session disconnect recovery
4. Test all servers unavailable scenario
5. Test server reconnection after failure

Test Scenarios:
- 1/3 servers fail (verify session continues)
- Connection timeout (verify error state)
- All servers fail (verify session works without MCP)
- Server recovery (verify reconnection)

Expected: 10-12 failure test cases
```

**Synchronization**: Wait for all Wave 4 agents to complete. Run full integration test suite to verify Phase 5 completion.

---

## Phase 5 Completion Checklist

### Wave 1 Verification
- [ ] T056: MCPServerConfig contract tests pass (15-20 tests)
- [ ] T057: MCP lifecycle integration tests pass (10-12 tests)
- [ ] T060: Tool prefixing unit tests pass (10-15 tests)

### Wave 2 Verification
- [ ] T061: MCPManager implementation reviewed/enhanced
- [ ] T062: Auto-connect verified in session manager
- [ ] T063: Tool registration implementation verified
- [ ] T064: Prefixing utilities created and tested

### Wave 3 Verification
- [ ] T065: `/tools list` command reviewed/tested
- [ ] T066: `/tools add` command reviewed/tested
- [ ] T067: `/tools remove` command reviewed/tested
- [ ] T069: TUI MCP status display reviewed/tested

### Wave 4 Verification
- [ ] T058: Tool discovery integration tests pass (8-10 tests)
- [ ] T059: Tool execution integration tests pass (8-10 tests)
- [ ] T068: Graceful failure handling implemented
- [ ] T068a: Graceful degradation tests pass (10-12 tests)

### Final Integration Tests
- [ ] Connect to real MCP server (filesystem)
- [ ] Discover tools and register with agent
- [ ] Execute tool call in conversation
- [ ] Test graceful failure with unreachable server
- [ ] Verify TUI status bar updates correctly
- [ ] Run full test suite: `pytest tests/cli/ -v`

**Success Criteria**: All Phase 5 tasks complete, all tests passing, MCP integration working end-to-end.

---

## Risk Mitigation

### Known Risks

1. **MCPHelper Integration Complexity**
   - **Risk**: Real MCP protocol may behave differently than expected
   - **Mitigation**: T061 thoroughly tests MCPHelper integration with real servers
   - **Fallback**: Mock MCPHelper in tests to isolate CLI logic

2. **Async Context Issues**
   - **Risk**: nest_asyncio may not work in all environments
   - **Mitigation**: Test auto-connect logic in multiple environments (Linux/macOS/Windows)
   - **Fallback**: Provide manual connection commands if auto-connect fails

3. **Tool Name Conflicts**
   - **Risk**: Prefixing may not prevent all conflicts
   - **Mitigation**: T064 thoroughly tests conflict detection and prevention
   - **Fallback**: Reject conflicting tools with clear error messages

4. **Graceful Failure Edge Cases**
   - **Risk**: Unexpected failure modes may crash sessions
   - **Mitigation**: T068/T068a extensively test failure scenarios
   - **Fallback**: Global error handler catches uncaught exceptions

### Rollback Plan

If Phase 5 encounters blocking issues:

1. **Partial Rollback**: Disable MCP features via config flag
2. **Full Rollback**: Revert to Phase 4 (AgentChain working without MCP)
3. **Isolated Testing**: Test MCP integration in separate branch before merging

**Rollback Trigger**: >50% of integration tests failing or critical bugs in production.

---

## Success Metrics

### Quantitative Metrics
- **Test Coverage**: ≥90% for MCP-related code
- **Test Pass Rate**: 100% of Phase 5 tests passing
- **Connection Success**: ≥95% success rate with standard MCP servers
- **Error Handling**: 0 uncaught exceptions in graceful failure tests
- **Performance**: MCP tool discovery <2 seconds per server

### Qualitative Metrics
- **Code Quality**: All MCP code passes mypy type checking
- **Documentation**: All MCP features documented in CLAUDE.md
- **User Experience**: MCP integration transparent to users (works automatically)
- **Maintainability**: MCP manager cleanly separates concerns (connection/discovery/registration)

---

## Next Steps After Phase 5

Once Phase 5 completes successfully:

1. **Phase 6**: Advanced Features (if defined)
2. **Phase 7**: Performance Optimization
3. **Phase 8**: Production Hardening
4. **Final Testing**: End-to-end user acceptance testing

**Ready to Execute**: Phase 5 is fully planned and ready for wave-by-wave execution!
