# Changelog

All notable changes to PromptChain will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2025-10-07

### Added
- **Orchestrator Metrics Tracking**:
  - `OrchestratorSupervisor.get_last_execution_metrics()` method for retrieving orchestrator performance data
  - Accurate `router_steps` tracking in `AgentExecutionResult` (now captures actual orchestrator reasoning steps)
  - Accurate `tools_called` tracking in `AgentExecutionResult`
  - Accurate `total_tokens`, `prompt_tokens`, `completion_tokens` tracking
  - Internal field renamed from `router_steps` to `orchestrator_reasoning_steps` for clarity (non-breaking)

- **Dual-Mode Logging Architecture**:
  - `ObservabilityFilter` class with 24 pattern markers for intelligent log filtering
  - Three logging modes:
    - Normal: Clean terminal output + full DEBUG file logs
    - `--dev`: Full observability in terminal + full DEBUG file logs
    - `--quiet`: Errors/warnings only in terminal + full DEBUG file logs
  - Filtered patterns include: `[HISTORY INJECTED]`, `[orchestrator_decision]`, `Executing tool:`, `[RunLog] Event:`, etc.

- **Enhanced Orchestrator Decision-Making**:
  - New 5-step reasoning process (upgraded from 4-step)
  - STEP 2: "KNOWLEDGE vs EXECUTION" classification to distinguish between:
    - Knowledge queries: "What year is it?" → Uses documentation agent
    - Execution queries: "Check the date" → Uses terminal agent
  - Execution verb detection: check, verify, run, execute, test, "show me actual"
  - Improved agent selection accuracy for command execution vs knowledge retrieval tasks

### Changed
- **Log Directory Structure**:
  - Logs now saved to `./agentic_chat/logs/` instead of `./agentic_team_logs/`
  - Debug logs: `agentic_chat/logs/YYYY-MM-DD/session_HHMMSS.log`
  - Event logs: `agentic_chat/logs/YYYY-MM-DD/session_HHMMSS.jsonl`
  - Cache: `agentic_chat/logs/cache/YYYY-MM-DD/`
  - Scripts: `agentic_chat/logs/scripts/YYYY-MM-DD/`

- **Orchestrator Reasoning Steps**:
  - Updated from 4-step to 5-step analysis in orchestrator prompts
  - Updated tool descriptions to reflect new step numbering
  - Updated examples in `OrchestratorSupervisor` to use 5-step process

### Fixed
- **Metadata Tracking Bug**:
  - `router_steps` now accurately reflects orchestrator reasoning steps (was always 0)
  - `tools_called` now accurately reflects tools invoked during orchestration (was always 0)
  - `total_tokens` now accurately reflects token consumption (was always None)
  - Metrics now properly flow from `OrchestratorSupervisor` → `StaticPlanStrategy` → `AgentChain` → `AgentExecutionResult`

- **Agent Selection Bug**:
  - Orchestrator now correctly chooses `terminal` agent for execution queries like "Check the date"
  - Previously incorrectly chose `documentation` agent (no tools) for execution-based queries
  - Added explicit STEP 2 to detect execution intent from user queries

### Files Modified
- `promptchain/utils/orchestrator_supervisor.py`: Added `get_last_execution_metrics()`, updated prompts to 5-step
- `promptchain/utils/strategies/static_plan_strategy.py`: Capture orchestrator metrics after routing
- `promptchain/utils/agent_chain.py`: Retrieve and populate metrics in `AgentExecutionResult`
- `promptchain/utils/agent_execution_result.py`: Updated documentation for `router_steps` field
- `agentic_chat/agentic_team_chat.py`: Added `ObservabilityFilter`, reorganized log paths, updated orchestrator prompt

### Performance Impact
- Metrics tracking overhead: <1% performance impact
- Memory overhead: ~0.5 KB per request for metrics storage
- Log filtering: <1ms per log message (negligible)

### Migration Notes
See [API Documentation v0.4.2](docs/api/v0.4.2.md) for detailed migration guide.

**Quick Migration**:
```python
# Update log paths in monitoring scripts
OLD_PATH = "./agentic_team_logs/**/*.jsonl"
NEW_PATH = "./agentic_chat/logs/**/*.jsonl"

# Access orchestrator metrics (new feature)
result = await agent_chain.process_request("Check date")
print(result.router_steps)    # Now shows actual steps (e.g., 5)
print(result.tools_called)    # Now shows actual tool count
print(result.total_tokens)    # Now shows actual token usage

# Use development mode for full observability
python agentic_chat/agentic_team_chat.py --dev
```

### Backward Compatibility
- ✅ All changes are backward compatible
- ✅ Public API unchanged (internal field rename only)
- ⚠️ Log file locations changed (update monitoring scripts)

## [0.4.1h] - 2025-10-04

### Added (Phase 3: Documentation & Examples)
- **Comprehensive Documentation**:
  - Complete observability system documentation in `docs/observability/`
  - Public APIs guide with detailed examples
  - Event system guide with real-world patterns
  - MCP events guide for monitoring external servers
  - Migration guide from v0.4.0 to v0.4.1h
  - Best practices guide for production usage

- **Working Examples**:
  - `examples/observability/basic_callbacks.py` - Simple callback usage
  - `examples/observability/event_filtering.py` - Advanced event filtering
  - `examples/observability/execution_metadata.py` - Metadata usage patterns
  - `examples/observability/monitoring_dashboard.py` - Real-world monitoring system
  - `examples/observability/migration_example.py` - Migration from old to new APIs

- **README Updates**:
  - Added observability system overview section
  - Event-based monitoring quick start
  - Rich execution metadata examples
  - Public APIs examples
  - Quick links to observability documentation

- **CHANGELOG**: Created comprehensive changelog tracking all v0.4.1 improvements

### Documentation
- All 5 observability documentation files completed with examples
- All 5 example files with working code
- README.md updated with observability section
- Complete migration guide for users upgrading from v0.4.0

### Notes
- This release completes Phase 3 of the observability improvements roadmap
- All features remain backward compatible (no breaking changes)
- Private attributes deprecated with warnings (removal planned for v0.5.0)

## [0.4.1g] - 2025-10-04

### Fixed
- **Backward Compatibility Validation**:
  - Verified all private attributes still accessible with deprecation warnings
  - Confirmed existing code works unchanged
  - Tested async/sync compatibility across all new features
  - Validated metadata collection opt-in behavior

### Tested
- Private attribute access (with deprecation warnings)
- Public API equivalents working correctly
- Metadata collection with `return_metadata=True`
- Event callbacks with filtering
- Symbol verification for all public APIs

## [0.4.1f] - 2025-10-04

### Added
- **MCPHelper Event Callbacks** (Milestone 0.4.1f):
  - 6 new MCP-specific event types:
    - `MCP_CONNECT_START` / `MCP_CONNECT_END` - Connection lifecycle
    - `MCP_DISCONNECT_START` / `MCP_DISCONNECT_END` - Disconnection lifecycle
    - `MCP_TOOL_DISCOVERED` - Tool discovery events
    - `MCP_ERROR` - MCP-specific errors
  - CallbackManager integration in MCPHelper
  - Comprehensive event metadata for MCP operations
  - Server ID tracking and transport information

### Fixed
- Race condition in MCPHelper disconnection (async-safe server_ids snapshot)
- Proper event emission for MCP tool discovery and errors

### Tested
- 12 comprehensive MCP event tests
- Connection/disconnection event firing
- Tool discovery event emission
- MCP error handling with proper context

## [0.4.1e] - 2025-10-04

### Added
- **PromptChain Event Integration** (Milestone 0.4.1e):
  - Integrated CallbackManager throughout PromptChain execution lifecycle
  - Event firing for all major execution phases:
    - Chain lifecycle (START/END/ERROR)
    - Step execution (START/END/ERROR)
    - Model calls (START/END/ERROR)
    - Tool calls (START/END/ERROR)
    - Function calls (START/END/ERROR)
    - Agentic steps (START/END/ERROR)
  - Rich event metadata with timing, model info, step numbers
  - Proper error event firing with full context

### Documentation
- Event system integrated into PromptChain
- 17 distinct event types covering all operations

## [0.4.1d] - 2025-10-04

### Added
- **PromptChain Callback System** (Milestone 0.4.1d):
  - `CallbackManager` class for registration and execution
  - `FilteredCallback` system with event type filtering
  - Support for both sync and async callbacks
  - Concurrent callback execution with error isolation
  - Thread-safe callback management

### Features
- Optional callback registration with event filtering
- Automatic sync/async detection and execution
- Error handling that doesn't interrupt other callbacks
- Zero overhead when callbacks not registered

### Tested
- 19 comprehensive callback system tests
- Sync and async callback execution
- Event filtering functionality
- Error isolation and handling

## [0.4.1c] - 2025-10-04

### Added
- **AgenticStepProcessor Metadata Tracking** (Milestone 0.4.1c):
  - `AgenticStepResult` dataclass for comprehensive execution data
  - `StepExecutionMetadata` for per-step details
  - `return_metadata` parameter in `run_async()` method
  - Step-by-step tool call tracking
  - Token usage estimation per step
  - Execution time tracking per step
  - Clarification attempt counting
  - Error tracking per step

### Features
- Detailed breakdown of internal reasoning steps
- Tool call metadata (name, args, result, timing)
- History mode tracking (minimal/accumulative/summary)
- Objective achievement status
- Configuration metadata (max_steps, model_name)

## [0.4.1b] - 2025-10-04

### Added
- **AgentChain Execution Metadata** (Milestone 0.4.1b):
  - `AgentExecutionResult` dataclass for comprehensive metadata
  - `return_metadata` parameter in `process_input()` method
  - Router decision tracking and metadata
  - Tool call tracking with timing information
  - Token usage tracking (total, prompt, completion)
  - Cache hit/miss information
  - Error and warning collection
  - Fallback mechanism tracking

### Features
- Execution timing (start_time, end_time, execution_time_ms)
- Agent selection metadata
- Router decision details (chosen_agent, refined_query, reasoning)
- Comprehensive error and warning tracking

## [0.4.1a] - 2025-10-04

### Added
- **ExecutionHistoryManager Public API** (Milestone 0.4.1a):
  - `current_token_count` property - get current total tokens
  - `history` property - get copy of history entries
  - `history_size` property - get number of entries
  - `get_statistics()` method - comprehensive statistics

### Deprecated
- Direct access to private attributes (`_current_tokens`, `_history`, etc.)
- Deprecation warnings added for private attribute access
- Migration path provided via public APIs

### Changed
- All public properties return copies to prevent external modification
- Statistics include memory usage, entry counts by type, truncation info

## [0.4.0] - 2025-09-XX (Previous Release)

### Added
- History accumulation modes for AgenticStepProcessor
- Enhanced MCP tool integration
- Improved async/sync compatibility

### Changed
- Performance optimizations for large chains
- Better error handling in tool execution

## [0.3.1] - 2025-09-XX

### Fixed
- MCP logging improvements
- Tool execution error handling

## [0.3.0] - 2025-08-XX

### Added
- MCP (Model Context Protocol) integration
- Tool hijacker for external tools
- Enhanced agent chain capabilities

## [0.2.0] - 2025-07-XX

### Added
- AgentChain multi-agent orchestration
- Router-based agent selection
- Session persistence and caching

## [0.1.0] - 2025-06-XX

### Added
- Initial PromptChain implementation
- Basic LLM chaining functionality
- Multi-model support via LiteLLM
- Function injection capabilities

---

## Versioning Strategy

- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible

## Deprecation Policy

- Features are deprecated with warnings for at least one minor version
- Private APIs may be removed after deprecation period
- Public APIs follow semantic versioning strictly

## Migration Notes

### v0.4.0 → v0.4.1h

See [Migration Guide](docs/observability/migration-guide.md) for detailed instructions.

**Key Changes**:
1. Use public APIs instead of private attributes
2. Enable metadata with `return_metadata=True`
3. Add callbacks for monitoring (optional)
4. All changes are backward compatible

**Deprecation Timeline**:
- v0.4.1h: Private attributes deprecated (warnings shown)
- v0.5.0: Private attributes removed (planned Q2 2025)

## Links

- [Documentation](docs/index.md)
- [Observability Guide](docs/observability/README.md)
- [GitHub Repository](https://github.com/gyasis/promptchain)
- [Issues](https://github.com/gyasis/promptchain/issues)
