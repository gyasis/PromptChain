# Research Agent API Reference

## AdvancedResearchOrchestrator

The `AdvancedResearchOrchestrator` provides multiple API interfaces to accommodate different integration patterns and usage scenarios.

### Class Definition

```python
from research_agent.core.orchestrator import AdvancedResearchOrchestrator

orchestrator = AdvancedResearchOrchestrator(config=None)
```

### API Methods

#### 1. Simplified API - `start_research()` (Async)

**Primary method for external integrations and simple use cases.**

```python
async def start_research(
    topic: str,
    session_id: Optional[str] = None,
    max_papers: Optional[int] = None,
    max_iterations: Optional[int] = None
) -> ResearchSession
```

**Parameters:**
- `topic` (str): Research topic to investigate
- `session_id` (Optional[str]): Custom session ID (auto-generated if not provided)
- `max_papers` (Optional[int]): Override config max papers limit
- `max_iterations` (Optional[int]): Override config max iterations limit

**Returns:** `ResearchSession` - Completed research session with all results

**Usage:**
```python
# Basic usage
session = await orchestrator.start_research("machine learning in healthcare")

# With parameters
session = await orchestrator.start_research(
    topic="artificial intelligence ethics",
    max_papers=50,
    max_iterations=3
)
```

#### 2. Synchronous Wrapper - `start_research_sync()`

**For backward compatibility with synchronous code.**

```python
def start_research_sync(
    topic: str,
    session_id: Optional[str] = None,
    max_papers: Optional[int] = None,
    max_iterations: Optional[int] = None
) -> ResearchSession
```

**Usage:**
```python
# Synchronous execution
session = orchestrator.start_research_sync("quantum computing")
```

#### 3. Advanced API - `conduct_research_session()` (Async)

**For advanced use cases requiring callbacks and detailed control.**

```python
async def conduct_research_session(
    research_topic: str,
    session_id: Optional[str] = None,
    callbacks: Optional[List[Callable]] = None
) -> ResearchSession
```

**Parameters:**
- `research_topic` (str): Research topic to investigate
- `session_id` (Optional[str]): Custom session ID
- `callbacks` (Optional[List[Callable]]): Progress callback functions

**Usage:**
```python
def progress_callback(message: str):
    print(f"Progress: {message}")

session = await orchestrator.conduct_research_session(
    research_topic="neural networks",
    callbacks=[progress_callback]
)
```

#### 4. Session Management Methods

```python
async def initialize_session(session_id: str) -> bool
async def continue_session(session_id: str) -> Optional[ResearchSession]
async def get_chat_interface(session_id: str) -> Optional[InteractiveChatInterface]
def get_session_statistics() -> Dict[str, Any]
async def shutdown()
```

**Session Initialization:**
```python
# Initialize orchestrator for a specific session
success = await orchestrator.initialize_session("session_123")
if success:
    # Now ready to start research with this session ID
    session = await orchestrator.start_research(
        topic="AI research",
        session_id="session_123"
    )
```

### API Design Principles

#### 1. **Layered API Architecture**

```
Simple API (start_research)
    ↓
Advanced API (conduct_research_session)  
    ↓
Internal Implementation (phase methods)
```

#### 2. **Backward Compatibility**

- `start_research()` - New simplified interface
- `start_research_sync()` - Synchronous compatibility wrapper
- `conduct_research_session()` - Original advanced interface (preserved)

#### 3. **Parameter Flexibility**

```python
# Minimal parameters
await orchestrator.start_research("AI ethics")

# With configuration overrides
await orchestrator.start_research(
    topic="AI ethics",
    max_papers=100,
    max_iterations=5
)

# Advanced usage with callbacks
await orchestrator.conduct_research_session(
    research_topic="AI ethics",
    callbacks=[progress_callback]
)
```

#### 4. **Error Handling**

All methods maintain consistent error handling:
- Configuration errors are raised during initialization
- Research errors are logged and session status updated to ERROR
- Fallback mechanisms ensure graceful degradation

#### 5. **Return Value Consistency**

All research methods return a `ResearchSession` object containing:
```python
class ResearchSession:
    session_id: str
    topic: str
    status: SessionStatus
    queries: Dict[str, Query]
    papers: Dict[str, Paper]
    processing_results: List[ProcessingResult]
    literature_review: Dict[str, Any]
    iteration_summaries: List[IterationSummary]
    chat_history: List[ChatMessage]
    # ... additional fields
```

### Integration Examples

#### Test Integration
```python
# Expected by test files
await orchestrator.start_research("artificial intelligence")
```

#### CLI Integration  
```python
# Current CLI usage (preserved)
session = await orchestrator.conduct_research_session(
    research_topic=topic,
    callbacks=[progress_callback]
)
```

#### Web API Integration
```python
# Backend API usage
session = await orchestrator.start_research(
    topic=request.topic,
    max_papers=request.max_papers,
    session_id=generated_session_id
)
```

#### Frontend Integration
```python
# JavaScript/TypeScript can call via API
const response = await fetch('/api/research/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        topic: "machine learning",
        max_papers: 50
    })
});
```

### Migration Guide

#### From Missing Methods
If you were getting `AttributeError: 'AdvancedResearchOrchestrator' object has no attribute 'start_research'`:

**Before (broken):**
```python
# This would fail
await orchestrator.start_research("topic")
```

**After (fixed):**
```python
# Now works with full API
session = await orchestrator.start_research("topic")
```

#### API Evolution
```python
# Level 1: Simple usage
session = await orchestrator.start_research("topic")

# Level 2: With parameters
session = await orchestrator.start_research(
    topic="topic", 
    max_papers=50
)

# Level 3: Advanced features
session = await orchestrator.conduct_research_session(
    research_topic="topic",
    callbacks=[callback]
)
```

### Performance Considerations

1. **Async by Default**: All primary methods are async for optimal performance
2. **Sync Wrappers**: Provided only for backward compatibility
3. **Resource Management**: Proper cleanup handled in `shutdown()` method
4. **Memory Efficiency**: Session data structured for efficient access patterns

### Architecture Benefits

1. **Consistent Interface**: All methods return the same `ResearchSession` structure
2. **Progressive Enhancement**: Simple → Advanced → Internal implementation layers
3. **Backward Compatibility**: Existing code continues to work
4. **Future Extensibility**: Clear patterns for adding new API methods
5. **Error Recovery**: Robust error handling at each layer

This API design resolves the immediate `AttributeError` while establishing a solid foundation for future development and integration needs.