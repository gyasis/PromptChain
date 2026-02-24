# API Contract: Context & Memory (FR-007 to FR-010)
**Branch**: `006-promptchain-improvements`

---

## FR-007: Auto Context Distillation

**File**: `promptchain/utils/execution_history_manager.py`
**Class**: `ContextDistiller` (line 519 — wiring into step processor)

### Contract: ContextDistiller.should_distill()
```python
def should_distill(self, history_manager: ExecutionHistoryManager) -> bool:
    """
    Returns True when current token usage >= distillation_threshold (default 0.7).
    Returns False if no max_tokens configured.
    Pure function — no side effects.
    """
```

### Contract: ContextDistiller.distill()
```python
async def distill(self, history_manager: ExecutionHistoryManager) -> str:
    """
    Generates "Current State of Knowledge" summary via LLM call.
    Replaces older messages in history_manager with distilled summary.
    Returns the generated summary text.
    On LLM call failure: logs error, leaves history_manager unchanged.
    """
```

### Integration Contract (AgenticStepProcessor)
```python
# Called at start of each thought cycle:
if self.context_distiller and self.context_distiller.should_distill(self.history_manager):
    await self.context_distiller.distill(self.history_manager)
```

---

## FR-008 / FR-009: MemoStore Public API

**File**: `promptchain/utils/memo_store.py` (existing — wiring needed)

### Contract: MemoStore.store_memo()
```python
def store_memo(
    task_description: str,
    solution: str,
    outcome: str = "success",
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Persists memo to SQLite. Returns memo_id.
    Thread-safe (SQLite WAL mode).
    Enforces max_memos limit (removes oldest on overflow).
    """
```

### Contract: MemoStore.retrieve_relevant_memos()
```python
def retrieve_relevant_memos(
    task_description: str,
    top_k: int = 3,
    outcome_filter: Optional[str] = None
) -> List[Memo]:
    """
    Returns memos with cosine similarity >= similarity_threshold.
    Sorted by relevance descending.
    Returns empty list if no memos meet threshold — never raises.
    Each returned Memo has relevance_score populated.
    """
```

### Integration Contract (AgenticStepProcessor)
```python
# Before building system prompt:
if self.memo_store:
    context = inject_relevant_memos(self.memo_store, task_description, context)

# After successful task completion:
if self.memo_store:
    self.memo_store.store_memo(task_description, solution_summary, "success")
```

---

## FR-010: Background Compression (JanitorAgent)

**File**: `promptchain/utils/janitor_agent.py` (new)

### Contract: JanitorAgent
```python
class JanitorAgent:
    async def start(self) -> None:
        """Launches background asyncio.Task. Idempotent — safe to call multiple times."""

    async def stop(self) -> None:
        """Cancels background task. Waits for clean shutdown. Max 5s timeout."""

    async def _monitor_loop(self) -> None:
        """
        Runs every check_interval seconds.
        Checks token usage against compression_threshold.
        If exceeded: triggers ContextDistiller.distill().
        Does NOT block primary agent execution.
        """
```

**Non-blocking guarantee**: `_monitor_loop` runs as a separate `asyncio.Task`,
never calling `await` on the primary agent's LLM operations.
