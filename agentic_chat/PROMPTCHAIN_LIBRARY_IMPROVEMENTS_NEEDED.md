# PromptChain Library Improvements Needed for Better Observability

## Date: 2025-10-04

## Executive Summary

While building the agentic chat system with comprehensive logging, we've identified multiple areas where the **PromptChain library itself** needs to expose more information for proper observability, debugging, and production monitoring. Currently, we're using workarounds (accessing private attributes, parsing logs with regex, hasattr checks) that are fragile and incomplete.

This document catalogs all the limitations and proposes library-level improvements.

---

## Current Workarounds & Their Fragility

### 1. **Private Attribute Access** ❌

**Current Code**:
```python
# Accessing private attributes (lines 1162, 1173, 1193, 1263, 1310, 1333, 1363)
current_size = history_manager._current_token_count  # ❌ Private attribute
total_entries = len(history_manager._history)        # ❌ Private attribute
```

**Problem**:
- Private attributes can change without warning in library updates
- No guarantee these attributes will exist
- Breaking changes with no migration path

**Needed**: Public API for history statistics

---

### 2. **AgentChain: No Execution Metadata** ❌

**Current Code**:
```python
# Line 1320-1339: After agent execution, we have NO IDEA which agent ran
response = await agent_chain.process_input(user_input)

# Note: This is best-effort - AgentChain doesn't expose which agent ran
# The orchestrator decision log already has this info
logging.info(f"✅ Agent execution completed | Response length: {len(response)} chars")
```

**What We Know**:
- Orchestrator chose an agent (from our logging callback)
- Got a response string

**What We DON'T Know**:
- Did that agent actually execute? (Could have been error/fallback)
- How long did execution take?
- How many tokens were used?
- Which model was called?
- Were there any internal errors/retries?
- Was cache hit or miss?

**Needed**: Execution metadata return object

---

### 3. **AgenticStepProcessor: No Step Metadata** ❌

**Current Code**:
```python
# Line 820: Hacky hasattr check with fallback
step_count = len(orchestrator_chain.step_outputs) if hasattr(orchestrator_chain, 'step_outputs') else 0
```

**What We Know**:
- Final string response

**What We DON'T Know**:
- How many actual steps were taken (max_internal_steps=5, but did it use all 5?)
- Which tools were called in each step
- Tool call arguments and results
- Whether objective was achieved or max steps reached
- Token usage per step
- Clarification attempts count
- History mode used and its effect

**Needed**: Structured step execution metadata

---

### 4. **Tool Call Information: Regex Parsing** ❌

**Current Code**:
```python
# Lines 1040-1107: Parsing log messages with regex - VERY FRAGILE
if "gemini_research" in msg.lower():
    tool_name = "gemini_research"
    query_match = re.search(r'(?:query|topic|prompt)["\']?\s*:\s*["\']([^"\']+)["\']', msg, re.IGNORECASE)
    if query_match:
        tool_args = query_match.group(1)
```

**Problems**:
- Log message format can change
- Regex breaks if format changes
- No structured data
- Can't reliably extract all tool arguments
- No tool results
- No timing information

**Needed**: Structured tool call events with callbacks/hooks

---

### 5. **PromptChain: No Execution Visibility** ❌

**Current Issues**:
- `process_prompt_async()` returns only string
- No access to:
  - Which instructions were executed
  - Which models were called
  - Token usage per instruction
  - Execution time per instruction
  - Chain breaker triggers
  - Errors/retries

**Needed**: Execution metadata and callbacks

---

## Proposed Library Improvements

### 🎯 **1. ExecutionHistoryManager: Public API for Statistics**

**Location**: `promptchain/utils/execution_history_manager.py`

**Add Public Methods**:
```python
class ExecutionHistoryManager:
    # ... existing code ...

    @property
    def current_token_count(self) -> int:
        """Public getter for current token count"""
        return self._current_token_count

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Public getter for history entries (read-only copy)"""
        return self._history.copy()

    @property
    def history_size(self) -> int:
        """Get number of entries in history"""
        return len(self._history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive history statistics

        Returns:
            {
                "total_tokens": int,
                "total_entries": int,
                "max_tokens": int,
                "utilization": float,  # percentage
                "entry_types": {"type": count},
                "truncations": int
            }
        """
        ...
```

**Benefits**:
- No more private attribute access
- Stable API
- Better encapsulation
- Can add computed statistics

---

### 🎯 **2. AgentChain: Execution Metadata Return**

**Location**: `promptchain/utils/agent_chain.py`

**Add Execution Result Class**:
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

@dataclass
class AgentExecutionResult:
    """Complete execution metadata from AgentChain"""

    # Response data
    response: str
    agent_name: str

    # Execution metadata
    execution_time_ms: float
    start_time: datetime
    end_time: datetime

    # Routing information
    router_decision: Optional[Dict[str, Any]]  # Orchestrator's decision
    router_steps: int
    fallback_used: bool

    # Agent execution details
    agent_execution_metadata: Optional[Dict[str, Any]]
    tools_called: List[Dict[str, Any]]

    # Token usage
    total_tokens: Optional[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]

    # Cache information
    cache_hit: bool
    cache_key: Optional[str]

    # Errors (if any)
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        ...
```

**Modified Method Signature**:
```python
class AgentChain:
    async def process_input(
        self,
        user_input: str,
        override_include_history: Optional[bool] = None,
        return_metadata: bool = False  # NEW parameter
    ) -> Union[str, AgentExecutionResult]:
        """
        Process user input through agent chain

        Args:
            user_input: User's query
            override_include_history: Override history inclusion
            return_metadata: If True, return AgentExecutionResult instead of str

        Returns:
            - str: Just the response (default, backward compatible)
            - AgentExecutionResult: Full metadata (if return_metadata=True)
        """
        start_time = datetime.now()

        # ... existing execution logic ...

        if return_metadata:
            return AgentExecutionResult(
                response=final_response,
                agent_name=executed_agent_name,  # Track this!
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                start_time=start_time,
                end_time=datetime.now(),
                router_decision=router_decision,
                router_steps=router_internal_steps,
                # ... populate other fields ...
            )
        else:
            return final_response  # Backward compatible
```

**Benefits**:
- Backward compatible (default returns string)
- Opt-in metadata for advanced use cases
- Structured data for logging
- No more guessing which agent ran

---

### 🎯 **3. AgenticStepProcessor: Step Execution Metadata**

**Location**: `promptchain/utils/agentic_step_processor.py`

**Add Step Result Class**:
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class StepExecutionMetadata:
    """Metadata for a single agentic step"""
    step_number: int
    tool_calls: List[Dict[str, Any]]  # [{name, args, result, time_ms}]
    tokens_used: int
    execution_time_ms: float
    clarification_attempts: int
    error: Optional[str]

@dataclass
class AgenticStepResult:
    """Complete execution result from AgenticStepProcessor"""

    # Final output
    final_answer: str

    # Execution summary
    total_steps: int
    max_steps_reached: bool
    objective_achieved: bool

    # Step-by-step details
    steps: List[StepExecutionMetadata]

    # Overall statistics
    total_tools_called: int
    total_tokens_used: int
    total_execution_time_ms: float

    # Configuration used
    history_mode: str
    max_internal_steps: int
    model_name: str

    # Errors/warnings
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        ...
```

**Modified Run Method**:
```python
class AgenticStepProcessor:
    async def run_async(
        self,
        current_input: str,
        available_tools: List[Dict[str, Any]],
        return_metadata: bool = False  # NEW parameter
    ) -> Union[str, AgenticStepResult]:
        """
        Execute agentic step with optional metadata return

        Args:
            current_input: Input for this step
            available_tools: Tools the agent can use
            return_metadata: If True, return AgenticStepResult instead of str

        Returns:
            - str: Just final answer (default, backward compatible)
            - AgenticStepResult: Full execution metadata
        """
        start_time = datetime.now()
        steps_metadata = []
        total_tools = 0

        # Track metadata during execution
        for step_num in range(1, self.max_internal_steps + 1):
            step_start = datetime.now()
            step_tools = []

            # ... existing execution logic ...
            # When tool is called:
            step_tools.append({
                "name": tool_name,
                "args": tool_args,
                "result": tool_result,
                "time_ms": tool_execution_time
            })

            # At end of step:
            steps_metadata.append(StepExecutionMetadata(
                step_number=step_num,
                tool_calls=step_tools,
                tokens_used=step_tokens,
                execution_time_ms=(datetime.now() - step_start).total_seconds() * 1000,
                clarification_attempts=clarification_attempts,
                error=None
            ))
            total_tools += len(step_tools)

        if return_metadata:
            return AgenticStepResult(
                final_answer=final_answer,
                total_steps=step_num,
                max_steps_reached=(step_num >= self.max_internal_steps),
                objective_achieved=(final_answer is not None),
                steps=steps_metadata,
                total_tools_called=total_tools,
                total_tokens_used=total_tokens,
                total_execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                history_mode=self.history_mode,
                max_internal_steps=self.max_internal_steps,
                model_name=self.model_name,
                errors=[],
                warnings=[]
            )
        else:
            return final_answer  # Backward compatible
```

**Benefits**:
- Backward compatible
- Complete visibility into step execution
- Structured tool call data
- No more hasattr checks
- Production-grade observability

---

### 🎯 **4. PromptChain: Execution Hooks & Callbacks**

**Location**: `promptchain/utils/promptchaining.py`

**Add Callback System**:
```python
from typing import Protocol, Optional, Dict, Any
from enum import Enum

class ExecutionEvent(Enum):
    """Events during PromptChain execution"""
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    INSTRUCTION_START = "instruction_start"
    INSTRUCTION_END = "instruction_end"
    MODEL_CALL_START = "model_call_start"
    MODEL_CALL_END = "model_call_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    CHAIN_BREAKER_TRIGGERED = "chain_breaker_triggered"
    ERROR = "error"

class ExecutionCallback(Protocol):
    """Protocol for execution callbacks"""
    def on_event(self, event: ExecutionEvent, data: Dict[str, Any]) -> None:
        """Called when execution event occurs"""
        ...

class PromptChain:
    def __init__(
        self,
        # ... existing args ...
        execution_callbacks: Optional[List[ExecutionCallback]] = None
    ):
        # ... existing init ...
        self.execution_callbacks = execution_callbacks or []

    def register_callback(self, callback: ExecutionCallback):
        """Register an execution callback"""
        self.execution_callbacks.append(callback)

    def _emit_event(self, event: ExecutionEvent, data: Dict[str, Any]):
        """Emit event to all callbacks"""
        for callback in self.execution_callbacks:
            try:
                callback.on_event(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def process_prompt_async(self, initial_input: Optional[str] = None):
        # Emit chain start
        self._emit_event(ExecutionEvent.CHAIN_START, {
            "input": initial_input,
            "timestamp": datetime.now()
        })

        # ... existing logic ...

        # When calling a tool:
        self._emit_event(ExecutionEvent.TOOL_CALL_START, {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "timestamp": datetime.now()
        })

        # After tool call:
        self._emit_event(ExecutionEvent.TOOL_CALL_END, {
            "tool_name": tool_name,
            "tool_result": tool_result,
            "execution_time_ms": elapsed_ms,
            "timestamp": datetime.now()
        })

        # When calling LLM:
        self._emit_event(ExecutionEvent.MODEL_CALL_START, {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "timestamp": datetime.now()
        })

        # After LLM call:
        self._emit_event(ExecutionEvent.MODEL_CALL_END, {
            "model": model_name,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "execution_time_ms": elapsed_ms,
            "timestamp": datetime.now()
        })

        # At chain end:
        self._emit_event(ExecutionEvent.CHAIN_END, {
            "output": final_output,
            "timestamp": datetime.now(),
            "total_time_ms": total_elapsed_ms
        })

        return final_output
```

**Usage in Our Code**:
```python
# Create callback for logging
class LoggingCallback:
    def __init__(self, log_event_fn):
        self.log_event = log_event_fn

    def on_event(self, event: ExecutionEvent, data: Dict[str, Any]):
        if event == ExecutionEvent.TOOL_CALL_END:
            self.log_event("tool_call", {
                "tool_name": data["tool_name"],
                "tool_args": data["tool_args"],  # STRUCTURED, not regex!
                "tool_result": data["tool_result"],
                "execution_time_ms": data["execution_time_ms"]
            })
        elif event == ExecutionEvent.MODEL_CALL_END:
            self.log_event("llm_call", {
                "model": data["model"],
                "prompt_tokens": data["prompt_tokens"],
                "completion_tokens": data["completion_tokens"],
                "execution_time_ms": data["execution_time_ms"]
            })

# Register callback with agents
research_agent = create_research_agent(mcp_config)
research_agent.register_callback(LoggingCallback(log_event))
```

**Benefits**:
- No more regex parsing of logs
- Structured, reliable data
- Real-time event stream
- Can add multiple callbacks (logging, metrics, debugging)
- Production observability

---

### 🎯 **5. MCPHelper: Connection & Tool Call Events**

**Location**: `promptchain/utils/mcp_helpers.py`

**Add Event Callbacks**:
```python
class MCPEvent(Enum):
    CONNECTION_START = "connection_start"
    CONNECTION_SUCCESS = "connection_success"
    CONNECTION_FAILED = "connection_failed"
    TOOL_DISCOVERED = "tool_discovered"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_ERROR = "tool_call_error"

class MCPHelper:
    def __init__(
        self,
        # ... existing args ...
        event_callback: Optional[Callable[[MCPEvent, Dict[str, Any]], None]] = None
    ):
        # ... existing init ...
        self.event_callback = event_callback

    def _emit_event(self, event: MCPEvent, data: Dict[str, Any]):
        if self.event_callback:
            try:
                self.event_callback(event, data)
            except Exception as e:
                logger.error(f"MCP callback error: {e}")

    async def connect(self):
        self._emit_event(MCPEvent.CONNECTION_START, {
            "server_id": self.server_id,
            "timestamp": datetime.now()
        })

        try:
            # ... existing connection logic ...

            self._emit_event(MCPEvent.CONNECTION_SUCCESS, {
                "server_id": self.server_id,
                "tools_discovered": len(self.tools),
                "timestamp": datetime.now()
            })
        except Exception as e:
            self._emit_event(MCPEvent.CONNECTION_FAILED, {
                "server_id": self.server_id,
                "error": str(e),
                "timestamp": datetime.now()
            })
            raise

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        self._emit_event(MCPEvent.TOOL_CALL_START, {
            "tool_name": tool_name,
            "arguments": arguments,  # STRUCTURED!
            "timestamp": datetime.now()
        })

        start_time = datetime.now()
        try:
            result = await self._actual_tool_call(tool_name, arguments)

            self._emit_event(MCPEvent.TOOL_CALL_END, {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "timestamp": datetime.now()
            })

            return result
        except Exception as e:
            self._emit_event(MCPEvent.TOOL_CALL_ERROR, {
                "tool_name": tool_name,
                "arguments": arguments,
                "error": str(e),
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "timestamp": datetime.now()
            })
            raise
```

**Benefits**:
- Structured MCP events
- No subprocess log parsing
- Connection status visibility
- Tool call timing and errors
- Production monitoring

---

## Implementation Priority

### Phase 1: High Priority (Immediate Impact)
1. ✅ **ExecutionHistoryManager Public API** - Stop accessing private attributes
2. ✅ **AgentChain Execution Metadata** - Know which agent ran
3. ✅ **AgenticStepProcessor Step Metadata** - Complete step visibility

### Phase 2: Medium Priority (Enhanced Observability)
4. ✅ **PromptChain Execution Callbacks** - Stop regex parsing logs
5. ✅ **MCPHelper Event Callbacks** - MCP tool call visibility

### Phase 3: Nice to Have (Advanced Features)
6. ⚪ **Token Usage Tracking** - Per-agent, per-step token counts
7. ⚪ **Performance Metrics** - Built-in latency tracking
8. ⚪ **Error Aggregation** - Structured error collection
9. ⚪ **Cost Tracking** - API cost estimation per execution

---

## Backward Compatibility Strategy

All proposed changes maintain **100% backward compatibility**:

1. **Optional Parameters**: `return_metadata=False` (default)
   - Default behavior: Return string (existing)
   - Opt-in: Return metadata object (new)

2. **Callback Systems**: Optional, not required
   - No callbacks registered: Works as before
   - Callbacks registered: Enhanced functionality

3. **Property Decorators**: Add public properties alongside private attributes
   - Private attributes remain (deprecated)
   - Public properties recommended (stable API)

4. **Event Systems**: Opt-in, zero impact if not used

---

## Benefits Summary

### For Users
- ✅ **Better Debugging**: See exactly what's happening
- ✅ **Production Monitoring**: Real-time observability
- ✅ **Cost Tracking**: Know token usage and API costs
- ✅ **Performance Optimization**: Identify bottlenecks

### For Library
- ✅ **Better API Design**: Public contracts instead of private access
- ✅ **Extensibility**: Callback system allows customization
- ✅ **Testing**: Easier to test with structured data
- ✅ **Documentation**: Clear interfaces to document

---

## Example: Before vs After

### Before (Current - Fragile)
```python
# Access private attributes
tokens = history_manager._current_token_count  # ❌

# Parse logs with regex
if "gemini_research" in msg.lower():  # ❌
    query_match = re.search(r'query.*?["\']([^"\']+)', msg)

# Check hasattr for step_outputs
steps = len(chain.step_outputs) if hasattr(chain, 'step_outputs') else 0  # ❌

# No idea which agent ran
response = await agent_chain.process_input(user_input)  # ❌
# Which agent was it? No clue!
```

### After (Proposed - Robust)
```python
# Public API
tokens = history_manager.current_token_count  # ✅

# Structured callbacks (no regex!)
def on_tool_call(event, data):
    log_event("tool_call", {
        "tool": data["tool_name"],
        "args": data["arguments"],  # ✅ Structured!
        "result": data["result"]
    })

# Metadata return
result = await agent_chain.process_input(user_input, return_metadata=True)  # ✅
print(f"Agent: {result.agent_name}")  # ✅
print(f"Steps: {result.router_steps}")  # ✅
print(f"Time: {result.execution_time_ms}ms")  # ✅
print(f"Tokens: {result.total_tokens}")  # ✅
```

---

## Conclusion

These library improvements would transform PromptChain from a "works but hard to debug" library into a **production-ready framework with enterprise-grade observability**.

**Current Status**: We're making it work with workarounds ✅
**With Improvements**: It would work beautifully by design 🎯

**Estimated Development Time**: 2-3 weeks for Phase 1+2
**User Impact**: Massive improvement in DX and production monitoring
**Breaking Changes**: Zero (100% backward compatible)

---

**Next Steps**:
1. Review this proposal with PromptChain maintainers
2. Create GitHub issues for each improvement
3. Implement Phase 1 (high priority items)
4. Update documentation with new APIs
5. Deprecate private attribute access patterns
