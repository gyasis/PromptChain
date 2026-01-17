# Phase 6 Test Execution Plan
## Token-Efficient History Management Tests (T070-T074)

**Branch**: `002-cli-orchestration`
**Phase**: Phase 6 - User Story 4 Tests
**Date**: 2025-11-23
**Implementation Status**: T075-T081 COMPLETE (7 tasks) ✅
**Test Status**: 3/5 COMPLETE, 2/5 PENDING

---

## Executive Summary

**Goal**: Validate per-agent history configurations achieve 30-60% token usage reduction.

**Current Status**:
- ✅ T070: Contract test - **COMPLETE** (25 tests passing)
- ✅ T071: Integration test (agent configs) - **COMPLETE** (14 tests passing)
- ❌ T072: Integration test (token truncation) - **PENDING**
- ✅ T073: Unit test (history defaults) - **COMPLETE** (14 tests passing)
- ❌ T074: Performance test (token optimization) - **PENDING**

**Implementation Complete**: All 7 implementation tasks (T075-T081) finished.

---

## Discovery Phase Results

### Existing Files Inventory

| Task | File | Status | Tests | Result |
|------|------|--------|-------|--------|
| T070 | `tests/cli/contract/test_history_config_contract.py` | ✅ EXISTS | 25 | **PASSING** |
| T071 | `tests/cli/integration/test_agent_history_configs.py` | ✅ EXISTS | 14 | **PASSING** |
| T072 | `tests/cli/integration/test_history_truncation.py` | ❌ MISSING | 0 | **N/A** |
| T073 | `tests/cli/unit/test_history_defaults.py` | ✅ EXISTS | 14 | **PASSING** |
| T074 | `tests/cli/integration/test_token_optimization.py` | ❌ MISSING | 0 | **N/A** |

**Summary**: 3 of 5 test files exist and pass (53 total tests). 2 files need creation.

---

## Wave Structure

### Wave 1: Token Truncation Tests (T072) - PARALLEL ELIGIBLE

**Agent Type**: `test-automator` (integration specialist)
**File**: `tests/cli/integration/test_history_truncation.py`
**Estimated Time**: 25-30 minutes
**Dependencies**: None (independent integration test)

**Test Requirements**:
1. Test ExecutionHistoryManager token counting accuracy
2. Test oldest_first truncation strategy
3. Test keep_last truncation strategy
4. Test token limit enforcement (max_tokens boundary)
5. Test entry limit enforcement (max_entries boundary)
6. Test combined token + entry limits
7. Test truncation with filtered history (include_types, exclude_sources)
8. Test truncation preserves most recent entries when using keep_last
9. Test truncation removes oldest entries when using oldest_first
10. Test token counting matches tiktoken behavior

**Expected Outcome**: 10-12 tests validating ExecutionHistoryManager truncation logic

**Implementation Reference**:
- `promptchain/utils/execution_history_manager.py` - Truncation logic
- `promptchain/cli/session_manager.py` - Integration point (T078 implementation)
- Existing tests: `test_history_config_contract.py` (truncation strategy validation)

---

### Wave 2: Performance Benchmarking (T074) - SEQUENTIAL AFTER WAVE 1

**Agent Type**: `performance-engineer` (specialized benchmarking agent)
**File**: `tests/cli/integration/test_token_optimization.py`
**Estimated Time**: 35-40 minutes
**Dependencies**: T072 (requires truncation logic verified)

**Test Requirements**:

**Baseline Scenarios (no optimization)**:
1. Multi-agent conversation (6 agents, full history) - measure baseline tokens
2. Long conversation (50 turns, 8000 tokens/agent) - measure baseline tokens
3. Terminal-heavy workflow (4 terminal agents, 2 researchers) - measure baseline tokens

**Optimized Scenarios (per-agent configs)**:
4. Same multi-agent conversation with history disabled for terminal agents
5. Same long conversation with token limits (terminal: 0, coder: 4000, researcher: 8000)
6. Same terminal-heavy workflow with terminal history disabled

**Performance Assertions**:
- Terminal agent with disabled history: **60% token savings** vs baseline
- Coder agent with 4000 limit: **40% token savings** vs full history
- Researcher agent with 8000 limit: **20% token savings** vs unlimited
- Multi-agent system (6 agents, 2 terminal, 2 coder, 2 researcher): **30-60% overall savings**
- Token-heavy conversation (50 turns): **50%+ savings** with optimized configs

**Expected Outcome**: 8-10 performance tests with quantified token savings

**Implementation Reference**:
- `promptchain/cli/utils/agent_templates.py` - Default configs (T076)
- `promptchain/cli/tui/status_bar.py` - Token tracking (T079)
- `promptchain/cli/command_handler.py` - `/history stats` command (T081)

---

## Execution Plan

### Phase 1: Wave 1 Execution (Independent)

**Timeline**: 25-30 minutes

```bash
# Task: T072 - Token Truncation Tests
# Agent: test-automator (integration specialist)
# File: tests/cli/integration/test_history_truncation.py

# Spawn specialized test-automator agent
promptchain agent spawn test-automator \
  --model "openai/gpt-4" \
  --objective "Create test_history_truncation.py with 10-12 tests validating ExecutionHistoryManager truncation logic" \
  --context "docs/PHASE6_TEST_EXECUTION_PLAN.md" \
  --reference-files "promptchain/utils/execution_history_manager.py,promptchain/cli/session_manager.py,tests/cli/contract/test_history_config_contract.py"
```

**Agent Instructions**:
1. Read ExecutionHistoryManager implementation (execution_history_manager.py)
2. Read integration point (session_manager.py - T078 implementation)
3. Study existing contract tests for truncation strategies
4. Create `test_history_truncation.py` with:
   - TestHistoryTruncationStrategies class (oldest_first, keep_last)
   - TestTokenCountingAccuracy class (tiktoken validation)
   - TestTruncationBoundaries class (max_tokens, max_entries)
   - TestFilteredTruncation class (include_types, exclude_sources)
5. Run tests: `pytest tests/cli/integration/test_history_truncation.py -v`
6. Verify all tests pass
7. Report results

**Success Criteria**:
- [ ] 10-12 tests created
- [ ] All tests passing
- [ ] Token counting validated against tiktoken
- [ ] Both truncation strategies tested
- [ ] Edge cases covered (empty history, single entry, etc.)

---

### Phase 2: Wave 2 Execution (After Wave 1)

**Timeline**: 35-40 minutes
**Dependency**: T072 passing (truncation logic verified)

```bash
# Task: T074 - Performance Benchmarking
# Agent: performance-engineer (specialized benchmarking)
# File: tests/cli/integration/test_token_optimization.py

# Spawn specialized performance-engineer agent
promptchain agent spawn performance-engineer \
  --model "openai/gpt-4" \
  --objective "Create test_token_optimization.py with 8-10 performance tests measuring 30-60% token savings" \
  --context "docs/PHASE6_TEST_EXECUTION_PLAN.md" \
  --reference-files "promptchain/cli/utils/agent_templates.py,promptchain/cli/tui/status_bar.py,promptchain/cli/command_handler.py,tests/cli/integration/test_history_truncation.py"
```

**Agent Instructions**:
1. Read default config implementation (agent_templates.py - T076)
2. Read token tracking implementation (status_bar.py - T079)
3. Read history stats implementation (command_handler.py - T081)
4. Study truncation tests (test_history_truncation.py - T072)
5. Create `test_token_optimization.py` with:
   - TestBaselineTokenUsage class (no optimization scenarios)
   - TestOptimizedTokenUsage class (per-agent config scenarios)
   - TestTokenSavingsCalculations class (30-60% validation)
   - TestMultiAgentOptimization class (6-agent system benchmarks)
6. Implement token counting utilities (baseline vs optimized)
7. Run tests: `pytest tests/cli/integration/test_token_optimization.py -v`
8. Verify 30-60% savings achieved
9. Generate performance report

**Success Criteria**:
- [ ] 8-10 performance tests created
- [ ] All tests passing
- [ ] Terminal agent: 60% savings validated
- [ ] Coder agent: 40% savings validated
- [ ] Researcher agent: 20% savings validated
- [ ] Multi-agent system: 30-60% overall savings validated
- [ ] Performance report generated

---

## Token Savings Estimate

### Agent Orchestration Efficiency

**Wave 1: Single Agent (test-automator)**
- Estimated context: ~8,000 tokens (ExecutionHistoryManager + session_manager + contract tests)
- Estimated generations: 3-4 iterations (test creation + refinement)
- Total tokens: ~32,000 tokens

**Wave 2: Single Agent (performance-engineer)**
- Estimated context: ~12,000 tokens (agent_templates + status_bar + command_handler + T072 tests)
- Estimated generations: 4-5 iterations (benchmark creation + validation)
- Total tokens: ~60,000 tokens

**Total Estimated**: ~92,000 tokens (2 specialized agents)

**Parallelization Benefit**: Wave 1 and Wave 2 are sequential (T074 depends on T072), so no parallel execution savings. However, using specialized agents prevents context pollution in main orchestrator session.

---

## Success Criteria

### Technical Validation

1. **Test Coverage**:
   - [ ] 5/5 test files exist
   - [ ] All tests passing (53 existing + 18-22 new = 71-75 total)
   - [ ] No regressions in existing Phase 6 tests

2. **Token Optimization Validation**:
   - [ ] Terminal agents: 60% token savings vs baseline
   - [ ] Coder agents: 40% token savings vs baseline
   - [ ] Researcher agents: 20% token savings vs baseline
   - [ ] Multi-agent systems: 30-60% overall savings

3. **Integration Validation**:
   - [ ] ExecutionHistoryManager truncation working correctly
   - [ ] Per-agent configs applied during AgentChain execution
   - [ ] Token tracking displayed in TUI status bar
   - [ ] `/history stats` command shows accurate breakdown

### Quality Gates

1. **Code Quality**:
   - [ ] All tests follow pytest conventions
   - [ ] Proper fixtures and test utilities
   - [ ] Clear test names and docstrings
   - [ ] No hardcoded values (use constants/fixtures)

2. **Documentation**:
   - [ ] Test files include module-level docstrings
   - [ ] Each test has clear docstring explaining what it validates
   - [ ] Performance benchmarks documented with expected ranges

3. **Performance**:
   - [ ] All tests run in < 5 seconds (except performance benchmarks)
   - [ ] Performance benchmarks complete in < 30 seconds
   - [ ] No memory leaks during test execution

---

## Risk Mitigation

### Known Risks

1. **Risk**: Performance tests may show < 30% savings
   - **Mitigation**: Use realistic conversation scenarios (50+ turns, multi-agent)
   - **Fallback**: Adjust agent type defaults if savings insufficient

2. **Risk**: Token counting may not match tiktoken exactly
   - **Mitigation**: Allow 5% margin of error in assertions
   - **Fallback**: Use tiktoken directly in ExecutionHistoryManager

3. **Risk**: Truncation tests may fail due to edge cases
   - **Mitigation**: Test empty history, single entry, boundary conditions
   - **Fallback**: Fix ExecutionHistoryManager implementation if bugs found

---

## Verification Checklist

After both waves complete:

```bash
# Run all Phase 6 tests
pytest tests/cli/contract/test_history_config_contract.py -v
pytest tests/cli/integration/test_agent_history_configs.py -v
pytest tests/cli/integration/test_history_truncation.py -v
pytest tests/cli/unit/test_history_defaults.py -v
pytest tests/cli/integration/test_token_optimization.py -v

# Run full test suite (ensure no regressions)
pytest tests/cli/ -v

# Check test count
pytest tests/cli/ --collect-only | grep "test session starts"
# Expected: 71-75 total tests (53 existing + 18-22 new)

# Validate token savings
pytest tests/cli/integration/test_token_optimization.py -v -k "savings"
# Expected: All savings assertions pass (30-60% range)
```

---

## Timeline Summary

| Phase | Agent | File | Duration | Dependencies |
|-------|-------|------|----------|--------------|
| Wave 1 | test-automator | test_history_truncation.py | 25-30 min | None |
| Wave 2 | performance-engineer | test_token_optimization.py | 35-40 min | Wave 1 complete |
| **Total** | **2 agents** | **2 files** | **60-70 min** | **Sequential** |

---

## Next Steps

1. **Execute Wave 1**: Spawn test-automator agent for T072
2. **Validate Wave 1**: Run `pytest tests/cli/integration/test_history_truncation.py -v`
3. **Execute Wave 2**: Spawn performance-engineer agent for T074
4. **Validate Wave 2**: Run `pytest tests/cli/integration/test_token_optimization.py -v`
5. **Final Validation**: Run full Phase 6 test suite
6. **Mark Complete**: Update tasks.md with ✅ for T070-T074

---

**Status**: READY FOR EXECUTION
**Blocker**: None
**Estimated Completion**: 60-70 minutes (2 sequential waves)
