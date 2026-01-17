# Internal Tool Enhancement PRD: Advanced Token Management & History Compression

**Document Type:** Product Requirements Document
**Target:** PromptChain CLI
**Version:** 1.0
**Status:** Draft
**Date:** 2025-01-18
**Author:** Engineering Team

---

## Executive Summary

This PRD outlines critical enhancements to the PromptChain CLI to address token management, history compression, and real-time status visibility. These improvements will enable users to conduct longer conversations without hitting token limits while maintaining full awareness of resource usage and system state.

## Background

### Current State
- PromptChain CLI provides interactive TUI for LLM conversations
- Long conversations can exceed token limits without warning
- Users have no visibility into token consumption or remaining capacity
- No automatic history compression/summarization
- Status bar shows basic info (session, agent, model) but lacks critical metrics

### Problem Statement
1. **Token Limit Overflow:** Users hit token limits unexpectedly, causing conversation failures
2. **Lack of Visibility:** No indication of token usage, consumption rate, or remaining capacity
3. **Manual History Management:** Users must manually start new sessions when limits approach
4. **Library-CLI Disconnect:** CLI updates don't automatically inherit library improvements
5. **Limited Status Context:** Status bar doesn't show processing state or resource metrics

## PromptChain's Existing Token Management Infrastructure

### ExecutionHistoryManager (Already Implemented)

PromptChain already has robust token management infrastructure in `promptchain/utils/execution_history_manager.py`:

**Token Counting:**
- Uses tiktoken library for accurate token counting per model
- `_count_tokens(text)` method using cl100k_base encoding
- Tracks `_current_token_count` efficiently across all history entries
- Character-based estimation fallback if tiktoken unavailable

**Automatic Truncation:**
- `max_tokens` parameter sets token limit for history
- `truncation_strategy` supports "oldest_first" (removes oldest entries when limit exceeded)
- Automatic truncation in `_apply_truncation()` method
- Emits HISTORY_TRUNCATED events with metadata (entries_removed, tokens_removed, etc.)

**Public API for Token Tracking:**
```python
# Get current token count
current_tokens = history_manager.current_token_count

# Get comprehensive statistics
stats = history_manager.get_statistics()
# Returns: total_tokens, total_entries, max_tokens, utilization_pct, entry_types, etc.
```

**Formatted History with Token Limits:**
```python
# Get history that fits within token budget
formatted = history_manager.get_formatted_history(
    max_tokens=2000,  # Only return entries that fit within 2000 tokens
    format_style='chat'
)
```

### LiteLLM Token Tracking

LiteLLM automatically returns token usage in responses:

```python
response = litellm.completion(model="gpt-4", messages=[...])

# Token metrics available in response object
response.usage.prompt_tokens      # Tokens in input
response.usage.completion_tokens  # Tokens in output
response.usage.total_tokens       # Total tokens consumed
```

**Available in:**
- `AgentExecutionResult` class has `total_tokens`, `prompt_tokens`, `completion_tokens` fields
- `AgenticStepResult` class tracks `total_tokens_used`
- Can be aggregated across multi-agent conversations

### CLI Integration Strategy

The CLI should leverage this existing infrastructure rather than reinventing it:

1. **Use ExecutionHistoryManager directly** for conversation tracking
2. **Extract token metrics from LiteLLM responses** after each agent call
3. **Display metrics in status bar** using history_manager.get_statistics()
4. **Auto-compress using existing truncation** when approaching limits

## Research Findings

### Claude's History Compression Approach
Based on research, Claude employs several sophisticated techniques:

**Compaction Strategy:**
- Distills context window contents in high-fidelity manner
- Preserves architectural decisions, unresolved bugs, implementation details
- Discards redundant tool outputs and messages
- Enables agents to continue with minimal performance degradation

**Structured Summarization:**
- Meta-summarization for very long documents (chunk → summarize → combine)
- Structured summaries using XML headers for consistent patterns
- Handoff summaries for model switching with all critical details

**Memory Tools:**
- File-based system for storing information outside context window
- Just-in-time approach with lightweight identifiers (file paths, stored queries)
- Structured note-taking persisted to memory and pulled back when needed

### Gemini CLI Token Tracking Patterns
**Status Display:**
- `/stats` command showing session token usage and savings
- Real-time event streaming with `--output-format stream-json`
- Token caching to reduce costs by reusing previous context

**Monitoring:**
- OpenTelemetry integration for comprehensive tracking
- Token count tracking, model usage, tool usage metrics
- Google Cloud Monitoring dashboards for enterprise visibility

### Goose AI Context Management
**Two-Tiered Approach:**
1. **Auto-Compaction:** Proactively summarizes conversation at 80% token threshold
   - Configurable via `GOOSE_AUTO_COMPACT_THRESHOLD` environment variable
   - Can be disabled by setting to 0.0

2. **Context Limit Strategies:** Backup strategies when limit exceeded
   - `summarize`: Condense conversation history
   - `truncate`: Remove oldest messages
   - `clear`: Start fresh context
   - `prompt`: Ask user for action

**Visual Token Display:**
- Colored circle next to model name
- Hover shows: token count, percentage used, total available, progress bar
- Persistent display in terminal UI for real-time feedback

**Context Engineering Principles:**
- **Offloading:** Move context to external systems (filesystem)
- **Reducing:** Minimize context size per turn
- **Isolating:** Separate context windows for individual tasks

## Objectives

### Primary Goals
1. **Real-time Token Awareness:** Users always know token status (used/total/remaining)
2. **Automatic History Management:** System proactively compresses history before limits
3. **Seamless Library Integration:** CLI inherits library improvements automatically
4. **Enhanced Status Visibility:** Show agent state, model, and processing indicators

### Success Metrics
- Zero unexpected token limit failures
- 90%+ user satisfaction with token visibility
- <200ms status bar update latency
- Successful compression preserves 95%+ conversation quality
- Library updates propagate to CLI within 1 release cycle

## Requirements

### 1. Token Tracking & Display

#### R1.1: Status Bar Token Metrics (P0)
**Description:** Display comprehensive token usage in status bar

**Specifications:**
```
Status Bar Format:
[*] Session: my-session | Agent: analyst (gpt-4) | Tokens: 3,247/8,000 (40%) [●●●●○○○○○○] | Messages: 42
```

**Components:**
- `Tokens Used`: Current conversation token count
- `Total Available`: Model's context window size
- `Percentage`: Visual indicator (0-100%)
- `Progress Bar`: 10-segment visual bar (●/○ characters)
- `Compression Indicator`: Show when history compressed (🗜️ icon)

**Token Counting:**
- Use tiktoken for accurate token counting by model
- Count tokens for: user messages, assistant responses, system prompts, tool calls
- Update in real-time after each message

#### R1.2: Token Usage Details (P1)
**Description:** Detailed token breakdown on demand

**Specifications:**
- `/tokens` slash command shows:
  - Input tokens (all user messages)
  - Output tokens (all assistant responses)
  - System tokens (prompts, tool schemas)
  - Cached tokens (if using caching)
  - Compression savings (tokens reclaimed)

**Example Output:**
```
Token Usage Details:
─────────────────────────────────
Input Tokens:       1,247 (15.6%)
Output Tokens:      1,823 (22.8%)
System Tokens:        177 (2.2%)
Total Used:         3,247 (40.6%)
Remaining:          4,753 (59.4%)
─────────────────────────────────
Compression: 2 times, saved 1,430 tokens
Cache Hits: 156 tokens
```

### 2. Automatic History Compression

#### R2.1: Threshold-Based Auto-Compression (P0)
**Description:** Automatically compress conversation history at configurable threshold

**Specifications:**
- **Default Threshold:** 75% of token limit
- **Configurable:** Via config.json `compression_threshold` (0.0-1.0)
- **Disable:** Set threshold to 0.0

**Compression Trigger:**
```python
if current_tokens / max_tokens >= compression_threshold:
    compressed_history = compress_conversation_history(history)
    replace_history_with_compressed(compressed_history)
```

**Configuration:**
```json
{
  "performance": {
    "history_compression": {
      "enabled": true,
      "threshold": 0.75,
      "strategy": "compaction",
      "preserve_recent_messages": 10
    }
  }
}
```

#### R2.2: Compression Strategies (P0)
**Description:** Multiple compression approaches with fallback

**Strategies (Priority Order):**

1. **Compaction** (Default):
   - Summarize conversation into structured format
   - Preserve: architectural decisions, unresolved issues, key context
   - Discard: redundant outputs, repeated tool calls, verbose responses
   - Use LLM to generate high-fidelity summary

2. **Truncation** (Fallback):
   - Keep most recent N messages (configurable)
   - Discard oldest messages first
   - Preserve conversation continuity

3. **Prompt** (Interactive Mode):
   - Ask user to choose: summarize, truncate, clear, continue anyway
   - Show token warning dialog with options

**Compaction Prompt Template:**
```
Analyze the conversation history and create a compact summary preserving:
- Key decisions and architectural choices
- Unresolved bugs or issues
- Important implementation details
- Current task context and progress

Discard:
- Redundant tool outputs
- Verbose explanations already acted upon
- Repeated information

Format as structured summary with sections:
- **Context**: Current task and objective
- **Decisions**: Key choices made
- **Issues**: Unresolved problems
- **Progress**: What has been completed

Original History: {history}
```

#### R2.3: Compression Indicators (P1)
**Description:** Visual feedback when compression occurs

**Specifications:**
- Status bar shows 🗜️ icon when history compressed
- Hover tooltip: "History compressed 2 times, saved 1,430 tokens"
- Chat message: "📋 [dim]Conversation history compressed to fit token limits[/dim]"
- `/history` command shows compression events

**Example:**
```
[*] Session: my-session | Agent: analyst | Tokens: 4,127/8,000 (51%) 🗜️ | Messages: 42 → 28
                                                                         ^^
                                                         Compression indicator
```

### 3. Enhanced Status Bar

#### R3.1: Agent & Model Display (P0)
**Description:** Show active agent name and model clearly

**Current:**
```
[*] Session: my-session | Agent: default | Model: gpt-4
```

**Enhanced:**
```
[*] Session: my-session | Agent: analyst (gpt-4-turbo) | Tokens: 3,247/8,000 (40%) | Messages: 42
```

**Specifications:**
- Agent name in cyan color
- Model name in parentheses (dim)
- Show agent switching transitions

#### R3.2: Processing/Thinking Indicator (P0)
**Description:** Animated spinner during agent processing

**Specifications:**
- Add spinner animation to status bar during LLM calls
- Position: Before agent name
- Animation: Rotating characters from AnimatedIndicator (◰ ◳ ◲ ◱)
- Color: Orange (#ffaa00) during processing

**Example:**
```
[◳] Session: my-session | Agent: analyst (thinking...) | Tokens: 3,247/8,000 (40%)
 ^^
Thinking spinner
```

**Implementation:**
```python
# In app.py handle_user_message:
status_bar.show_thinking_indicator(agent_name)  # Start spinner
response = await agent_chain.process_message(content)
status_bar.hide_thinking_indicator()  # Stop spinner
```

#### R3.3: Dynamic Token Color Coding (P1)
**Description:** Color-code token percentage for at-a-glance awareness

**Color Rules:**
- 0-60%: Green (#00ff00) - Safe
- 60-75%: Yellow (#ffaa00) - Warning
- 75-90%: Orange (#ff8800) - Approaching limit
- 90-100%: Red (#ff0000) - Critical

**Example:**
```
Tokens: 3,247/8,000 (40%)    # Green
Tokens: 6,100/8,000 (76%)    # Orange
Tokens: 7,800/8,000 (97%)    # Red - triggers compression
```

### 4. Library Integration

#### R4.1: Shared ExecutionHistoryManager (P0)
**Description:** CLI uses PromptChain library's ExecutionHistoryManager directly

**Current State:**
- CLI maintains its own message list
- No integration with library's history management

**Target State:**
- CLI uses `ExecutionHistoryManager` for all conversation tracking
- Inherits token-aware truncation automatically
- Compression strategies defined in library

**Implementation:**
```python
from promptchain.utils.execution_history_manager import ExecutionHistoryManager

class PromptChainApp:
    def __init__(self, ...):
        # Use library's history manager
        self.history_manager = ExecutionHistoryManager(
            max_tokens=self.config.performance.history_max_tokens,
            max_entries=self.config.ui.max_displayed_messages,
            truncation_strategy="compaction"
        )

    async def handle_user_message(self, content: str):
        # Add to library's history
        self.history_manager.add_entry("user_input", content, source="user")

        # Get formatted history for agent
        formatted_history = self.history_manager.get_formatted_history(
            format_style='chat',
            max_tokens=self.get_remaining_tokens()
        )
```

#### R4.2: Session State Synchronization (P0)
**Description:** CLI session storage uses library's caching mechanisms

**Target:**
- Leverage `AgentChain.cache_config` for session persistence
- SQLite cache stores conversation state
- CLI loads/saves sessions through library APIs

**Example:**
```python
# Create agent chain with cache
self.agent_chain = AgentChain(
    agents=self.agents,
    cache_config={
        "name": self.session_name,
        "path": str(self.sessions_dir)
    }
)

# Auto-saves conversation to SQLite
# CLI just queries current state
conversation = self.agent_chain.get_conversation_history()
```

#### R4.3: Feature Flag System (P1)
**Description:** CLI automatically enables library features via config

**Mechanism:**
```json
{
  "library_features": {
    "history_compression": true,
    "token_caching": true,
    "mcp_integration": true,
    "agentic_steps": true
  }
}
```

**Behavior:**
- CLI checks library version and available features
- Automatically enables compatible features
- Shows warning if library version too old

### 5. User Experience

#### R5.1: Token Warning Dialog (P1)
**Description:** Proactive warning before hitting limits

**Trigger:** 90% token usage (before auto-compression at 75%)

**Dialog:**
```
┌─ Token Limit Warning ─────────────────────────┐
│                                               │
│  You're using 7,200/8,000 tokens (90%)       │
│                                               │
│  Options:                                     │
│  [C] Compress history automatically           │
│  [N] Start new session (save current)         │
│  [I] Continue anyway (may hit limit)          │
│  [V] View token breakdown                     │
│                                               │
└───────────────────────────────────────────────┘
```

#### R5.2: Compression Quality Feedback (P2)
**Description:** Show compression results to user

**After Compression:**
```
📋 History Compressed
   Before: 42 messages, 7,200 tokens
   After:  28 messages, 4,100 tokens
   Saved:  3,100 tokens (43%)
   Quality: High (preserved all key context)
```

## Technical Specifications

### Architecture

#### Token Counting Integration
```python
import tiktoken

class TokenCounter:
    """Accurate token counting per model."""

    def __init__(self, model_name: str):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def count_messages(self, messages: List[Message]) -> int:
        """Count tokens in message list."""
        total = 0
        for msg in messages:
            # Count role, content, and formatting tokens
            total += len(self.encoding.encode(msg.role))
            total += len(self.encoding.encode(msg.content))
            total += 4  # Message overhead
        return total

    def count_text(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
```

#### Compression Pipeline
```python
class HistoryCompressor:
    """Manages conversation history compression."""

    def __init__(self, strategy: str = "compaction"):
        self.strategy = strategy
        self.compression_events = []

    async def compress(
        self,
        history: List[Message],
        preserve_recent: int = 10,
        target_tokens: Optional[int] = None
    ) -> List[Message]:
        """Compress conversation history."""

        if self.strategy == "compaction":
            return await self._compaction_strategy(
                history, preserve_recent, target_tokens
            )
        elif self.strategy == "truncation":
            return self._truncation_strategy(history, preserve_recent)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _compaction_strategy(
        self, history: List[Message], preserve_recent: int, target_tokens: Optional[int]
    ) -> List[Message]:
        """Use LLM to create high-fidelity summary."""

        # Keep recent messages intact
        recent = history[-preserve_recent:]
        to_compress = history[:-preserve_recent]

        # Generate summary using LLM
        summary_prompt = self._build_compaction_prompt(to_compress)
        summary = await self._get_llm_summary(summary_prompt)

        # Create compressed message list
        compressed = [
            Message(
                role="system",
                content=f"[Compressed History]\n{summary}",
                metadata={"compressed": True, "original_count": len(to_compress)}
            )
        ] + recent

        # Log compression event
        self.compression_events.append({
            "timestamp": datetime.now(),
            "original_messages": len(history),
            "compressed_messages": len(compressed),
            "tokens_saved": self._calculate_savings(history, compressed)
        })

        return compressed
```

#### Status Bar Update Logic
```python
class StatusBar(Static):
    """Enhanced status bar with token tracking."""

    # Reactive properties
    tokens_used: reactive[int] = reactive(0)
    tokens_total: reactive[int] = reactive(8000)
    is_thinking: reactive[bool] = reactive(False)
    compression_count: reactive[int] = reactive(0)

    def render(self) -> str:
        """Render status bar with all metrics."""

        # Thinking indicator
        if self.is_thinking:
            indicator = self._get_thinking_frame()  # Animated spinner
            agent_display = f"{indicator} {self.active_agent} (thinking...)"
        else:
            indicator = "*" if self.session_state == "Active" else "-"
            agent_display = f"{self.active_agent} ({self.model_name})"

        # Token display with color coding
        percentage = (self.tokens_used / self.tokens_total) * 100
        token_color = self._get_token_color(percentage)
        progress_bar = self._render_progress_bar(percentage)

        compression_icon = " 🗜️" if self.compression_count > 0 else ""

        parts = [
            f"[{self.session_state_color}]{indicator}[/] Session: [bold]{self.session_name}[/bold]",
            f"Agent: [cyan]{agent_display}[/cyan]",
            f"Tokens: [{token_color}]{self.tokens_used:,}/{self.tokens_total:,} ({percentage:.0f}%)[/{token_color}]",
            progress_bar + compression_icon,
            f"Messages: {self.message_count}"
        ]

        return " | ".join(parts)

    def _get_token_color(self, percentage: float) -> str:
        """Get color based on token usage."""
        if percentage < 60:
            return "#00ff00"  # Green
        elif percentage < 75:
            return "#ffaa00"  # Yellow
        elif percentage < 90:
            return "#ff8800"  # Orange
        else:
            return "#ff0000"  # Red

    def _render_progress_bar(self, percentage: float) -> str:
        """Render 10-segment progress bar."""
        filled = int(percentage / 10)
        bar = "●" * filled + "○" * (10 - filled)
        return f"[{self._get_token_color(percentage)}]{bar}[/]"
```

### Data Models

#### Configuration Extensions
```python
@dataclass
class CompressionConfig:
    """History compression settings."""
    enabled: bool = True
    threshold: float = 0.75  # 75% token usage
    strategy: str = "compaction"  # compaction, truncation, prompt
    preserve_recent_messages: int = 10
    min_compression_gain: int = 500  # Minimum tokens to save

    def validate(self):
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be 0.0-1.0")
        if self.strategy not in ["compaction", "truncation", "prompt"]:
            raise ValueError(f"Invalid strategy: {self.strategy}")

@dataclass
class TokenTrackingConfig:
    """Token tracking settings."""
    enabled: bool = True
    show_in_status_bar: bool = True
    warning_threshold: float = 0.90  # Warn at 90%
    color_coding: bool = True
    show_progress_bar: bool = True

@dataclass
class PerformanceConfig:
    """Performance tuning settings."""
    lazy_load_agents: bool = True
    history_max_tokens: int = 6000
    cache_enabled: bool = True

    # New fields
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    token_tracking: TokenTrackingConfig = field(default_factory=TokenTrackingConfig)
```

### Migration Path

#### Phase 1: Token Tracking Foundation
1. Integrate tiktoken for accurate token counting
2. Add TokenCounter utility class
3. Update StatusBar with token metrics
4. Implement real-time token updates

#### Phase 2: Basic Compression
1. Implement truncation strategy (simple)
2. Add compression threshold monitoring
3. Create compression warning dialog
4. Add `/tokens` command

#### Phase 3: Advanced Compression
1. Implement compaction strategy (LLM-based)
2. Add compression quality metrics
3. Integrate with ExecutionHistoryManager
4. Add compression indicators in UI

#### Phase 4: Library Integration
1. Refactor to use library's ExecutionHistoryManager
2. Sync session state with AgentChain cache
3. Add feature flag system
4. Update documentation

## Dependencies

### New Dependencies
- `tiktoken>=0.5.0` - Accurate token counting by model
- No other new dependencies (use existing Rich, Textual, LiteLLM)

### Library Version Requirements
- `promptchain>=0.4.2` - Requires ExecutionHistoryManager with token support

## Testing Strategy

### Unit Tests
```python
# Test token counting accuracy
def test_token_counter_accuracy():
    counter = TokenCounter("gpt-4")
    messages = [
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you!")
    ]
    tokens = counter.count_messages(messages)
    assert tokens > 0
    assert tokens < 100  # Reasonable range

# Test compression threshold trigger
def test_compression_trigger():
    compressor = HistoryCompressor(strategy="truncation")
    history = [Message(role="user", content="test")] * 100

    # Should trigger compression at 75% threshold
    should_compress = compressor.should_compress(
        current_tokens=6000,
        max_tokens=8000,
        threshold=0.75
    )
    assert should_compress is True

# Test compression strategies
async def test_compaction_strategy():
    compressor = HistoryCompressor(strategy="compaction")
    history = create_long_conversation(50)  # 50 messages

    compressed = await compressor.compress(
        history,
        preserve_recent=10,
        target_tokens=2000
    )

    assert len(compressed) < len(history)
    assert any(msg.metadata.get("compressed") for msg in compressed)
```

### Integration Tests
```python
# Test end-to-end compression flow
async def test_compression_flow():
    app = create_test_app()

    # Simulate conversation approaching limit
    for i in range(40):
        await app.send_message(f"Test message {i}")

    # Verify compression triggered
    assert app.history_manager.compression_count > 0
    assert app.status_bar.compression_count > 0
    assert app.status_bar.tokens_used < app.config.compression.threshold * app.status_bar.tokens_total

# Test status bar updates
def test_status_bar_token_display():
    status_bar = StatusBar()
    status_bar.tokens_used = 6000
    status_bar.tokens_total = 8000

    rendered = status_bar.render()
    assert "6,000/8,000" in rendered
    assert "(75%)" in rendered
    assert "●●●●●●●○○○" in rendered  # Progress bar
```

### User Acceptance Tests
1. **Token Visibility:** User can always see current token usage
2. **Compression Trigger:** Compression activates automatically at 75% threshold
3. **Quality Preservation:** Compressed conversations retain key context
4. **No Unexpected Failures:** Zero token limit errors during normal use
5. **Performance:** Status bar updates <200ms after each message

## Success Criteria

### MVP (Minimum Viable Product)
- ✅ Token count displayed in status bar (used/total/percentage)
- ✅ Auto-compression at 75% threshold with truncation strategy
- ✅ Compression indicator in status bar
- ✅ `/tokens` command for detailed breakdown
- ✅ Color-coded token percentage

### V1.0 Release Criteria
- ✅ All MVP features
- ✅ Compaction strategy using LLM summarization
- ✅ Thinking indicator animation in status bar
- ✅ Token warning dialog at 90% threshold
- ✅ Integration with ExecutionHistoryManager
- ✅ Comprehensive documentation and user guide

### Future Enhancements (V1.1+)
- Token usage analytics and trends
- Per-agent token budgets
- Conversation cost estimation
- Export compressed history
- Custom compression prompts per domain

## Timeline Estimate

- **Phase 1:** Token Tracking Foundation - 1 week
- **Phase 2:** Basic Compression - 1 week
- **Phase 3:** Advanced Compression - 2 weeks
- **Phase 4:** Library Integration - 1 week
- **Testing & Documentation:** 1 week

**Total:** ~6 weeks for full implementation

## Open Questions

1. Should compression be opt-in or opt-out by default?
   - **Recommendation:** Opt-out (enabled by default) with easy disable

2. What compression ratio indicates "good" vs "poor" compression?
   - **Recommendation:** >30% savings = good, <15% = poor, prompt user to manually summarize

3. Should we cache compression results to avoid re-compressing?
   - **Recommendation:** Yes, cache compressed summaries in session storage

4. How to handle multi-agent conversations with different token limits?
   - **Recommendation:** Track tokens per agent, use smallest limit for global threshold

## References

- Claude AI: History compression techniques (compaction, meta-summarization, memory tools)
- Gemini CLI: Token tracking with `/stats` command and OpenTelemetry
- Goose AI: Two-tiered compression (auto-compaction + fallback strategies)
- PromptChain: ExecutionHistoryManager documentation
- LiteLLM: Token counting and usage tracking APIs

---

**Approval Required From:**
- Product Lead: _________________
- Engineering Lead: _________________
- Design Lead: _________________
- QA Lead: _________________

**Next Steps:**
1. Review and approve PRD
2. Create implementation tickets in task tracker
3. Assign engineering resources
4. Schedule kickoff meeting
5. Begin Phase 1 development
