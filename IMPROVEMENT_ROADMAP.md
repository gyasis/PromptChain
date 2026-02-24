# PromptChain Improvement Roadmap
**Generated:** 2026-02-23
**Branch:** 005-mlflow-observability-clean

## Executive Summary

This document consolidates findings from:
1. Adversarial bug hunting (23 bugs identified)
2. Competitive analysis of AG2, Claude Code, and Gemini CLI
3. Architectural improvement brainstorming

The roadmap is organized into three tiers: **Critical Fixes**, **High-Impact Improvements**, and **Strategic Enhancements**.

---

## Part 1: Critical Bug Fixes (Immediate Action Required)

### P0 Critical Bugs (Breaks Functionality)

#### BUG-017, BUG-018, BUG-019: MCP Tool Parameter Mismatches
- **Location:** `promptchain/utils/enhanced_agentic_step_processor.py`
- **Impact:** All Gemini augmentation functionality is broken
- **Issues:**
  - Line 645: Uses `error_context` instead of `error_message` for `gemini_debug`
  - Line 564: Uses unsupported `num_ideas` parameter for `gemini_brainstorm`
  - Line 575: Uses `question` instead of `prompt` for `ask_gemini`
- **Fix:** Update parameter names to match MCP tool signatures
- **Priority:** P0 - Fix immediately

#### BUG-001: Event Loop Race Conditions
- **Location:** Multiple files (patterns.py, app.py)
- **Impact:** LightRAG pattern commands fail when invoked from TUI
- **Issue:** Inconsistent async event loop handling
- **Fix:** Standardize event loop creation/management
- **Priority:** P0 - Prevents crashes

#### BUG-003: JSON Parsing Crash Without Error Handling
- **Location:** `promptchain/utils/json_output_parser.py:52`
- **Impact:** Data loss when LLM returns malformed JSON
- **Fix:** Add try-except with graceful fallback
- **Priority:** P0 - Data integrity

### P1 High Priority Bugs

#### BUG-008: Flush Timeout Ignored in Queue
- **Location:** `promptchain/observability/queue.py:134`
- **Impact:** Shutdown can hang indefinitely if MLflow server unresponsive
- **Fix:** Implement proper timeout handling for `queue.join()`
- **Priority:** P1 - Production stability

#### BUG-007: Config File Read on Every Access
- **Location:** `promptchain/observability/config.py:72`
- **Impact:** Performance degradation (disk I/O on every config access)
- **Fix:** Implement config caching with file modification time checking
- **Priority:** P1 - Performance

#### BUG-009: Verification Result Mutation
- **Location:** `promptchain/utils/enhanced_agentic_step_processor.py:781`
- **Impact:** Cache corruption in verification results
- **Fix:** Use deep copy before modifying cached results
- **Priority:** P1 - Data integrity

See `BUG_HUNTING_REPORT.md` for complete list of 23 bugs with detailed analysis.

---

## Part 2: Architectural Evolution (Inspired by AG2, Claude Code, Gemini CLI)

### Research Findings: Key Patterns from Industry Leaders

#### AutoGen 2 (AG2) - 2026
- **Event-driven actor model**: Agents as independent actors with async message passing
- **Non-blocking patterns**: `async/await` enables I/O operations without halting the team
- **Pub/Sub pipelines**: Agents publish to topics, subscribers trigger automatically
- **Teachability & Memos**: Vector DB storage for long-term memory retrieval
- **Stateful checkpointing**: Event-sourced architecture enables crash recovery

#### Claude Code (Anthropic)
- **Single-threaded master loop ("nO")**: Disciplined single reasoning thread
- **h2A async dual-buffer queue**: Non-blocking user steering during execution
- **Real-time steering**: User can inject instructions mid-thought
- **wU2 context compressor**: Automatic context compression at 92% capacity
- **Ralph Loop**: Autonomous persistence until goal met

#### Gemini CLI (Google)
- **A2A protocol**: Standardized agent-to-agent communication
- **Event-driven messaging**: Terminal remains responsive during long operations
- **Session persistence**: SQLite-backed automatic session saving
- **Lifecycle hooks**: JSON-defined scripts attached to agent events
- **Thought signatures**: Cryptographic state tokens prevent reasoning drift

### Current PromptChain Architecture Gaps

| Gap | Current State | Industry Pattern | Impact |
|-----|---------------|------------------|---------|
| **Blocking execution** | Agents block during LLM calls | AG2: Async actors, Claude: h2A queue | Poor UX, no multi-tasking |
| **Limited longevity** | Token truncation only | AG2: Memos, Claude: wU2 compression | Loses context over time |
| **Basic pipelines** | Sequential only | AG2: Pub/Sub topics | Limited flexibility |
| **No user steering** | Can't interrupt agents | Claude: Real-time steering | No course correction |
| **Simple context mgmt** | Truncation-based | Claude: Context distillation | Inefficient token usage |

---

## Part 3: Improvement Roadmap

### Phase 1: Non-Blocking Agent Flows (High Priority)

#### 1.1 Async Actor Pattern
**Inspired by:** AG2 event-driven actor model
**Effort:** Medium
**Impact:** Enables true multi-agent concurrency

**Implementation:**
- Refactor `AgenticStepProcessor` to `AsyncAgenticStepProcessor`
- Add `PriorityQueue` inbox for each agent
- Implement yielding execution instead of blocking waits
- Add `PendingState` for agents waiting on dependencies

**Files to modify:**
- `promptchain/utils/enhanced_agentic_step_processor.py`
- `promptchain/utils/agent_chain.py`

#### 1.2 Pub/Sub Pipeline System
**Inspired by:** AG2 topic-based message bus
**Effort:** High
**Impact:** Flexible, event-driven pipelines

**Implementation:**
- Extend existing `MessageBus` with topic subscriptions
- Replace `PipelineMode` with `TopicBasedBus`
- Add concurrent triggering for multiple subscribers
- Implement "Virtual Waiting Rooms" for dependency management

**Files to create:**
- `promptchain/cli/communication/pubsub_pipeline.py`

### Phase 2: Conversational Longevity (Critical for UX)

#### 2.1 Context Distillation (wU2 Pattern)
**Inspired by:** Claude Code's wU2 compressor
**Effort:** Low
**Impact:** Better token efficiency, maintained context

**Implementation:**
- Add `ContextDistiller` to `ExecutionHistoryManager`
- Trigger at 70% token capacity
- Generate "Current State of Knowledge" summary
- Replace old messages with distilled state

**Files to modify:**
- `promptchain/utils/execution_history_manager.py` (if exists, else create)

#### 2.2 Semantic Memo Store
**Inspired by:** AG2 Teachability
**Effort:** Low
**Impact:** Long-term learning across sessions

**Implementation:**
- Add `MemoStore` with SQLite backend
- Store "Lessons Learned" as JSON blobs
- Vector search integration for memo retrieval
- Inject relevant memos into system prompts

**Files to create:**
- `promptchain/utils/memo_store.py`

#### 2.3 Recursive Compression with Janitor Agent
**Inspired by:** AG2 + Claude Code patterns
**Effort:** Medium
**Impact:** Automatic memory management

**Implementation:**
- Create "Janitor Agent" running in background task
- Monitor `ExecutionHistoryManager` for token usage
- Perform lossy compression (summarization) at thresholds
- Extract facts/entities to Blackboard for lossless storage

**Files to create:**
- `promptchain/utils/janitor_agent.py`

### Phase 3: Real-Time User Steering (Game Changer)

#### 3.1 Interrupt Queue (h2A Pattern)
**Inspired by:** Claude Code's h2A dual-buffer queue
**Effort:** Medium
**Impact:** User control during execution

**Implementation:**
- Add `InterruptQueue` to `AgenticStepProcessor`
- Check queue at start of each thought cycle
- Integrate with Textual TUI input handler
- Support "Stop and summarize" type commands

**Files to modify:**
- `promptchain/utils/enhanced_agentic_step_processor.py`
- `promptchain/cli/tui/app.py`

#### 3.2 Hot-Swapping Prompts
**Inspired by:** Claude Code's mid-stream pivoting
**Effort:** Medium
**Impact:** Adaptive agent behavior

**Implementation:**
- Add `GlobalOverrideSignal` to MessageBus
- Implement prompt replacement mid-execution
- Save micro-checkpoints after each tool call
- Enable rewind to last valid checkpoint

**Files to modify:**
- `promptchain/cli/communication/message_bus.py`

### Phase 4: Advanced Agentic Patterns (Strategic)

#### 4.1 Active Blackboard with Observables
**Inspired by:** Reactive programming + AG2
**Effort:** High
**Impact:** Automatic agent triggering

**Implementation:**
- Make Blackboard an `Observable` object
- Add `watch(key)` registration for agents
- Trigger agent instantiation on key writes
- Implement condition-action rules

**Files to modify:**
- `promptchain/cli/models/blackboard.py`

#### 4.2 A2A Protocol Integration
**Inspired by:** Gemini CLI A2A protocol
**Effort:** High
**Impact:** Interoperability with external agents

**Implementation:**
- Define standard Handshake/Handoff schemas
- Add A2A message serialization/deserialization
- Implement protocol negotiation
- Support external agent connections

**Files to create:**
- `promptchain/protocols/a2a.py`

#### 4.3 Shadow Reasoning (Critic Loop)
**Inspired by:** Quality assurance patterns
**Effort:** Medium
**Impact:** Error detection before user sees output

**Implementation:**
- Spawn low-cost "Critic Agent" in parallel with main agent
- Check for hallucinations, logic errors
- Send `NACK` signals to trigger retries
- Implement before user sees output

**Files to create:**
- `promptchain/utils/critic_agent.py`

### Phase 5: Innovative Features (Differentiators)

#### 5.1 Time-Traveler Debugger
**Novel idea from brainstorming**
**Effort:** High
**Impact:** Unique debugging/development experience

**Implementation:**
- Build TUI timeline view of all agent thoughts
- Enable pause/rewind/edit of agent memory
- Support branching execution (create timelines)
- SQLite backend for timeline persistence

**Files to create:**
- `promptchain/cli/tui/time_traveler.py`

---

## Implementation Priority Matrix

### Quick Wins (High Impact, Low/Medium Effort)

| Feature | Effort | Impact | Dependencies |
|---------|--------|--------|--------------|
| Context Distiller | Low | High | None |
| Semantic Memo Store | Low | High | SQLite |
| Bug Fixes (P0) | Low | Critical | None |
| Interrupt Queue | Medium | High | TUI integration |
| Async Step Processor | Medium | High | Refactoring |

### Strategic Investments (High Impact, High Effort)

| Feature | Effort | Impact | Dependencies |
|---------|--------|--------|--------------|
| Pub/Sub Pipeline | High | High | MessageBus refactor |
| Active Blackboard | High | Medium | Observer pattern |
| A2A Protocol | High | Medium | Protocol design |
| Time-Traveler Debugger | High | High | TUI + SQLite |

### Deferred (Low Priority)

| Feature | Reason |
|---------|--------|
| Shadow Reasoning | Complex, can be added post-MVP |
| Full A2A Protocol | Requires ecosystem maturity |

---

## Next Steps

### Immediate Actions (This Sprint)

1. **Fix P0 Bugs**
   - [ ] Fix MCP tool parameter mismatches (BUG-017, 018, 019)
   - [ ] Standardize event loop handling (BUG-001)
   - [ ] Add JSON parsing error handling (BUG-003)

2. **Implement Quick Wins**
   - [ ] Add Context Distiller to ExecutionHistoryManager
   - [ ] Create Semantic Memo Store with SQLite backend
   - [ ] Add Interrupt Queue to AgenticStepProcessor

3. **Prototype Non-Blocking Flow**
   - [ ] Create AsyncAgenticStepProcessor prototype
   - [ ] Test with simple multi-agent scenario
   - [ ] Measure performance improvement

### Medium-Term (Next Quarter)

1. Implement Pub/Sub Pipeline system
2. Build Active Blackboard with observers
3. Add Real-Time User Steering to TUI
4. Create comprehensive documentation for new patterns

### Long-Term (6-12 Months)

1. A2A Protocol integration for ecosystem interoperability
2. Time-Traveler Debugger for agent development
3. Shadow Reasoning for quality assurance
4. Full migration to event-driven architecture

---

## Success Metrics

### Technical Metrics
- **Non-blocking rate**: % of agent operations that don't block event loop
- **Context retention**: Effective context window after compression/distillation
- **Steering latency**: Time from user interrupt to agent response
- **Memory efficiency**: Tokens saved through compression vs truncation

### User Experience Metrics
- **Perceived responsiveness**: User feedback on TUI responsiveness
- **Conversation length**: Average session duration before context loss
- **Error recovery**: % of failed operations recovered via steering
- **Development velocity**: Time to implement new agentic patterns

---

## References

- **AG2 Research**: Event-driven actor model, Pub/Sub, Teachability & Memos
- **Claude Code Research**: h2A queue, wU2 compressor, Ralph Loop
- **Gemini CLI Research**: A2A protocol, lifecycle hooks, session persistence
- **Bug Report**: `/home/gyasis/Documents/code/PromptChain/BUG_HUNTING_REPORT.md`

---

*This roadmap is a living document. Update after each sprint with progress, learnings, and revised priorities.*
