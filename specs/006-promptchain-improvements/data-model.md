# Data Model: PromptChain Comprehensive Improvements
**Branch**: `006-promptchain-improvements` | **Phase**: 1 — Design
**Date**: 2026-02-24

---

## Entities

### 1. Interrupt (existing — `promptchain/utils/interrupt_queue.py`)

Already implemented. No schema changes required.

```python
@dataclass
class Interrupt:
    interrupt_id: str           # "int_{counter}_{unix_ts}"
    interrupt_type: InterruptType  # STEERING | CORRECTION | CLARIFICATION | ABORT | PAUSE | RESUME
    message: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None   # unix epoch
    processed: bool = False
```

**Validation rules**: `interrupt_id` must be non-empty. `message` must be
non-empty. `timestamp` auto-populated in `__post_init__`.

---

### 2. InterruptQueue (existing — wiring needed)

```python
class InterruptQueue:
    _queue: queue.Queue          # maxsize=100
    _interrupt_counter: int
    _counter_lock: threading.Lock
    _interrupt_history: List[Interrupt]
    _paused: threading.Event
```

**Integration addition**: `AgenticStepProcessor` gains:
```python
self.interrupt_queue: Optional[InterruptQueue] = None
self.interrupt_handler: Optional[InterruptHandler] = None
```

---

### 3. Memo (existing — `promptchain/utils/memo_store.py`)

Already implemented. No schema changes.

```python
@dataclass
class Memo:
    memo_id: int                          # AUTOINCREMENT from SQLite
    task_description: str
    solution: str
    outcome: str                          # "success" | "failure" | "partial"
    embedding: Optional[np.ndarray]       # 100-dim vector (bag-of-words fallback)
    timestamp: Optional[float]
    metadata: Optional[Dict[str, Any]]
    relevance_score: Optional[float]      # Populated during semantic search
```

**SQLite schema** (already created in `~/.promptchain/memos.db`):
```sql
CREATE TABLE memos (
    memo_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    task_description TEXT NOT NULL,
    solution  TEXT NOT NULL,
    outcome   TEXT NOT NULL,
    embedding BLOB,
    timestamp REAL NOT NULL,
    metadata  TEXT   -- JSON-encoded Dict
);
CREATE INDEX idx_timestamp ON memos(timestamp DESC);
```

**State transitions**: `outcome` transitions `pending → success | failure | partial`.

---

### 4. MicroCheckpoint (new — wiring into CheckpointManager)

```python
@dataclass
class MicroCheckpoint:
    checkpoint_id: str        # "{agent_id}_{step}_{unix_ts}"
    agent_id: str
    step_number: int
    tool_call_index: int      # Index of tool call that triggered this checkpoint
    conversation_snapshot: List[Dict[str, Any]]   # Copy of messages at this point
    tool_results_snapshot: List[Any]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
```

**Storage**: In-memory `Dict[str, MicroCheckpoint]` keyed by `checkpoint_id`
inside `AgenticStepProcessor`. Not persisted to disk (per spec assumption).

**Retention**: Last 10 checkpoints per agent session; oldest evicted on overflow.

---

### 5. PubSubBus (new — extends `promptchain/cli/communication/message_bus.py`)

```python
class PubSubBus:
    _subscribers: Dict[str, List[Callable]]   # topic → list of async callbacks
    _lock: asyncio.Lock

    async def publish(topic: str, payload: Any) -> None
    async def subscribe(topic: str, callback: Callable[[str, Any], Awaitable[None]]) -> None
    async def unsubscribe(topic: str, callback: Callable) -> None
    def publish_sync(topic: str, payload: Any) -> None   # sync wrapper
```

**Topic naming convention**: `"{domain}.{event}"` e.g., `"agent.tool_complete"`,
`"agent.interrupt"`, `"session.context_distilled"`.

**Fan-out**: Uses `asyncio.gather(*[cb(topic, payload) for cb in subscribers])`
for concurrent delivery. Individual subscriber errors are caught and logged
without propagating to publisher.

---

### 6. AsyncAgentInbox (new — `promptchain/utils/async_agent_inbox.py`)

```python
@dataclass
class InboxMessage:
    priority: int          # 0=interrupt, 1=normal, 2=background
    topic: str
    payload: Any
    sender_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):   # Required for PriorityQueue
        return self.priority < other.priority

class AsyncAgentInbox:
    agent_id: str
    _queue: asyncio.PriorityQueue

    async def send(message: InboxMessage) -> None
    async def receive() -> InboxMessage          # blocks until message available
    async def try_receive() -> Optional[InboxMessage]   # non-blocking
    def qsize() -> int
    def empty() -> bool
```

---

### 7. ContextDistiller (existing — `promptchain/utils/execution_history_manager.py`)

Already implemented at line 519. Integration required.

**Wiring**: `AgenticStepProcessor` gains:
```python
self.context_distiller: Optional[ContextDistiller] = None
```
Called before each LLM call:
```python
if self.context_distiller and self.context_distiller.should_distill(self.history_manager):
    await self.context_distiller.distill(self.history_manager)
```

---

### 8. JanitorAgent (new — `promptchain/utils/janitor_agent.py`)

```python
class JanitorAgent:
    history_manager: ExecutionHistoryManager
    context_distiller: ContextDistiller
    check_interval: float       # seconds between checks (default: 30.0)
    compression_threshold: float  # token % to trigger (default: 0.8)
    _task: Optional[asyncio.Task]

    async def start() -> None    # launches background asyncio Task
    async def stop() -> None     # cancels background task
    async def _monitor_loop() -> None   # internal: checks + compresses
```

---

## Relationship Diagram

```
AgenticStepProcessor
├── interrupt_queue: InterruptQueue (optional)
│   └── Interrupt[]
├── interrupt_handler: InterruptHandler
├── context_distiller: ContextDistiller (optional)
│   └── monitors ExecutionHistoryManager.token_count
├── memo_store: MemoStore (optional)
│   └── Memo[] (SQLite)
├── inbox: AsyncAgentInbox (optional)
│   └── InboxMessage[] (priority queue)
└── _micro_checkpoints: Dict[str, MicroCheckpoint] (in-memory, last 10)

PubSubBus (singleton or injected)
└── _subscribers: Dict[str, List[Callable]]

JanitorAgent
├── history_manager: ExecutionHistoryManager
└── context_distiller: ContextDistiller
```

---

## Validation Rules

| Entity | Field | Rule |
|--------|-------|------|
| InboxMessage | priority | Must be 0, 1, or 2 |
| InboxMessage | topic | Non-empty string |
| MicroCheckpoint | conversation_snapshot | Deep copy, not reference |
| PubSubBus | topic | Non-empty; convention `"domain.event"` |
| MemoStore | similarity_threshold | Float in [0.0, 1.0] |
| MemoStore | max_memos | Integer > 0 |
| JanitorAgent | check_interval | Float > 0.0 |
| JanitorAgent | compression_threshold | Float in (0.0, 1.0] |
