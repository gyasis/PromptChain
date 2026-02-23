# PromptChain Constitution

<!--
SYNC IMPACT REPORT - 2025-11-16
================================

Version Change: Initial → 1.0.0

CHANGES SUMMARY:
- ADDED: I. Library-First Architecture (new principle)
- ADDED: II. Observable Systems (new principle emphasizing event system and metadata)
- ADDED: III. Test-First Development (TDD mandatory for all changes)
- ADDED: IV. Integration Testing (contract testing and inter-component validation)
- ADDED: V. Token Economy & Performance (critical for LLM applications)
- ADDED: VI. Async-First Design (dual async/sync interface pattern)
- ADDED: VII. Simplicity & Maintainability (YAGNI, clear documentation)
- ADDED: Development Workflow section (PR requirements, review process)
- ADDED: Quality Gates section (test coverage, observability, performance standards)
- ADDED: Governance section (amendment process, versioning, compliance)

TEMPLATE PROPAGATION STATUS:
✅ plan-template.md - Reviewed: Constitution Check section aligns with principles
✅ spec-template.md - Reviewed: User scenarios and requirements align with testing principles
✅ tasks-template.md - Reviewed: Task structure supports TDD and independent testing
✅ agent-file-template.md - Not modified: Generic template, no constitution-specific content
✅ checklist-template.md - Not reviewed: Not present in .specify/templates/

FOLLOW-UP TODOS: None

RATIONALE FOR VERSION 1.0.0:
- Initial constitution establishment for PromptChain project
- Captures existing architectural patterns from codebase analysis
- Establishes core principles for future development
- Major version as this is the founding constitution
-->

## Core Principles

### I. Library-First Architecture

Every feature must be designed as a standalone, reusable library component before integration into the larger system. Libraries must be:

- **Self-contained**: Minimal external dependencies, clear boundaries
- **Independently testable**: Can be tested in isolation without complex setup
- **Well-documented**: Clear purpose, usage examples, API contracts
- **Purposeful**: Solves a specific problem; organizational-only libraries are prohibited

**Rationale**: This ensures modularity, reusability, and prevents tight coupling that makes systems brittle and difficult to maintain.

### II. Observable Systems

All system components MUST emit structured events and provide rich execution metadata for monitoring, debugging, and performance analysis.

**Requirements**:
- Event-based monitoring with 30+ lifecycle event types
- Execution metadata including timing, token usage, tool calls, and errors
- Public APIs for accessing system state (no reliance on private attributes)
- JSONL file logging for post-execution analysis
- Dual-mode logging: clean terminal output with full debug logs to file

**Rationale**: Observability is critical for production LLM applications where token costs, latency, and complex multi-step reasoning must be monitored and optimized. The v0.4.1h observability system establishes this as a core architectural pattern.

### III. Test-First Development (NON-NEGOTIABLE)

Test-Driven Development (TDD) is mandatory for all code changes:

1. **Tests written FIRST**: Before any implementation code
2. **User approval**: Tests must be approved by stakeholder/user before implementation
3. **Tests must FAIL**: Verify tests fail initially (red)
4. **Implementation**: Write code to make tests pass (green)
5. **Refactor**: Clean up code while keeping tests green

**Red-Green-Refactor cycle strictly enforced. No exceptions.**

**Rationale**: TDD ensures requirements are understood before coding begins, prevents over-engineering, and creates a safety net for refactoring. The "tests fail first" rule catches poorly designed tests.

### IV. Integration Testing

Focus integration testing on critical interaction points:

**Required Integration Tests**:
- New library contract tests (verify public API behavior)
- Contract changes (ensure backward compatibility or document breaking changes)
- Inter-component communication (AgentChain ↔ PromptChain ↔ MCPHelper)
- Shared data structures (ExecutionHistoryManager, AgentExecutionResult)
- Tool integration (local tools, MCP tools, tool schema standardization)

**Rationale**: While unit tests validate individual components, integration tests catch issues at system boundaries where most production bugs occur, especially in multi-agent orchestration and tool calling.

### V. Token Economy & Performance

LLM applications must optimize for token usage and execution time:

**Token Management**:
- ExecutionHistoryManager with token-aware truncation (using tiktoken)
- Per-agent history configuration to optimize context size
- Memory management to prevent context overflow
- Accurate token tracking in execution metadata

**Performance Standards**:
- Async/await support for concurrent LLM operations
- Execution timing tracked at all levels (chain, step, model, tool)
- Performance metrics exposed via public APIs
- Clean terminal output to reduce I/O overhead

**Rationale**: Token costs and latency directly impact user experience and operational costs. The per-agent history system (v0.4.2) demonstrates 30-60% token savings for execution-only agents.

### VI. Async-First Design

All core APIs must provide both synchronous and asynchronous interfaces:

- **Async as primary**: Internal implementations use async/await
- **Sync wrapper**: Synchronous methods wrap async with `asyncio.run()`
- **Context managers**: Proper resource cleanup (e.g., MCP connections)
- **Backward compatibility**: Sync APIs for simple use cases

**Rationale**: Modern LLM applications often require concurrent operations (multi-agent coordination, parallel tool calls). Async-first design enables high-performance applications while maintaining simplicity for basic use cases.

### VII. Simplicity & Maintainability

Start simple and add complexity only when justified:

- **YAGNI principle**: "You Aren't Gonna Need It" - don't build features speculatively
- **Text I/O ensures debuggability**: stdin/stdout protocols for observability
- **Clear documentation**: README, CLAUDE.md, comprehensive docs/ directory
- **Explicit over implicit**: No magic, clear component interactions
- **Modular architecture**: Separate concerns (PromptChain, AgentChain, AgenticStepProcessor, ExecutionHistoryManager, MCPHelper)

**Rationale**: Complex systems are hard to understand, maintain, and extend. The library-first architecture with clear component boundaries reduces cognitive load and enables confident changes.

## Development Workflow

### Pull Request Requirements

All changes MUST:
1. **Pass Constitution Check**: Verify compliance with all seven principles
2. **Include tests**: TDD process followed, tests written first and pass
3. **Demonstrate observability**: Events emitted, metadata captured
4. **Document token impact**: For LLM-related changes, document token usage implications
5. **Maintain async/sync parity**: Both interfaces work correctly
6. **Update relevant docs**: README, CLAUDE.md, or docs/ as needed

### Code Review Process

Reviewers MUST verify:
- **Principle adherence**: Each of the seven core principles checked
- **Test quality**: Tests are meaningful, cover edge cases, and fail appropriately before implementation
- **Integration points**: Inter-component contracts respected
- **Performance implications**: Token usage, latency, memory consumption
- **Documentation clarity**: Changes are understandable to future maintainers

## Quality Gates

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Critical paths**: 100% coverage for AgentChain routing, PromptChain execution, tool integration
- **Test types**: Unit tests (component isolation), integration tests (component interaction), contract tests (API stability)

### Observability Standards

- **Event emission**: All state changes emit appropriate events
- **Metadata completeness**: Execution metadata includes timing, tokens, tools, errors
- **Logging levels**: Support for dev/quiet modes
- **Public APIs**: No reliance on private attributes for monitoring

### Performance Benchmarks

- **Token efficiency**: Per-agent history optimization where appropriate
- **Latency tracking**: All LLM calls timed and reported
- **Memory management**: History truncation prevents unbounded growth
- **Async efficiency**: Concurrent operations where possible

## Governance

### Amendment Process

Constitution changes require:
1. **Documentation**: Proposed amendment with rationale and impact analysis
2. **Review**: Team/maintainer approval
3. **Migration plan**: For changes affecting existing code
4. **Version update**: Semantic versioning applied to constitution

### Versioning Policy

Constitution versions follow semantic versioning:
- **MAJOR**: Backward incompatible governance changes, principle removals or redefinitions
- **MINOR**: New principles, materially expanded guidance, new sections
- **PATCH**: Clarifications, wording improvements, non-semantic refinements

### Compliance Review

All pull requests and code reviews MUST verify compliance with this constitution. Violations must be justified in the "Complexity Tracking" section of the implementation plan with:
- Why the violation is necessary
- What simpler alternatives were considered and rejected
- Mitigation strategies for the added complexity

**Runtime Development Guidance**: For day-to-day development patterns and agent-specific instructions, consult CLAUDE.md.

**Version**: 1.0.0 | **Ratified**: 2025-11-16 | **Last Amended**: 2025-11-16
