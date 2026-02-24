# API Contract: Bug Fixes (FR-001 to FR-006)
**Branch**: `006-promptchain-improvements`

---

## FR-001: Gemini MCP Tool Parameter Corrections

**File**: `promptchain/utils/enhanced_agentic_step_processor.py`

### Before (broken)
```python
# gemini_debug — BUG-017
arguments={"error_context": error_msg, "code": code}

# gemini_brainstorm — BUG-018
arguments={"topic": topic, "num_ideas": 5}

# ask_gemini — BUG-019
arguments={"question": query}
```

### After (contract)
```python
# gemini_debug
arguments={"error_message": error_msg, "code": code}

# gemini_brainstorm
arguments={"topic": topic}   # num_ideas removed

# ask_gemini
arguments={"prompt": query}
```

**Validation**: Integration test calls each tool with mocked MCP server
returning success; asserts no `KeyError` or MCP parameter validation error.

---

## FR-002: Event Loop Safety Contract

**File**: `promptchain/cli/utils/event_loop_manager.py` (existing)
**Usage sites**: All TUI pattern command handlers

### Contract
```python
# In CLI context (no running loop):
result = run_async_in_context(my_async_coro())  # uses asyncio.run()

# In TUI context (Textual running loop):
result = await my_async_coro()  # direct await — never call run_async_in_context
```

**Rule**: Pattern command handlers MUST check `is_event_loop_running()` and
branch accordingly. `asyncio.run()` MUST NOT be called when a loop is running.

---

## FR-003: JSON Parser Graceful Fallback

**File**: `promptchain/utils/json_output_parser.py`

### Contract
```python
def extract(data, key_path, default=None) -> Any:
    """
    Returns default on any parse failure.
    NEVER raises an unhandled exception.
    Logs warning with raw string on failure.
    """
```

**Failure modes handled**:
- `json.JSONDecodeError` — logs warning, returns `default`
- `KeyError` / `IndexError` on path traversal — returns `default`
- Any other `Exception` — logs error with full traceback, returns `default`

---

## FR-004: MLflow Queue Bounded Shutdown

**File**: `promptchain/observability/queue.py`

### Contract
```python
def shutdown(self, timeout: float = 5.0) -> None:
    """
    Signals shutdown, calls flush(timeout=timeout), then joins worker.
    ALWAYS completes within timeout seconds.
    Does NOT block indefinitely.
    """
```

**Guarantee**: If `flush(timeout)` returns `False` (timed out), shutdown
proceeds anyway. Worker thread is set to daemon=True as a backstop.

---

## FR-005: Observability Config Cache

**File**: `promptchain/observability/config.py`

### Contract
```python
def get_observability_config() -> ObservabilityConfig:
    """
    Returns cached config if file mtime is unchanged.
    Only re-reads file when mtime changes.
    Thread-safe via module-level lock.
    """
```

**Performance guarantee**: Two consecutive calls with no file change produce
zero disk reads on the second call.

---

## FR-006: Verification Result Deep Copy

**File**: `promptchain/utils/enhanced_agentic_step_processor.py`

### Contract
```python
def verify_logic(self, ...) -> VerificationResult:
    """
    Returns deep copy from cache.
    Modifying the returned object NEVER affects the cache.
    """
    cached = self.verification_cache.get(cache_key)
    if cached:
        return copy.deepcopy(cached)   # <-- required
```
