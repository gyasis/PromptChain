# Phase 6 Agent Spawning Templates

**Purpose**: Ready-to-use agent spawning commands for Phase 6 test execution

---

## Wave 1: T072 - Token Truncation Tests

### Agent Spawn Command

```bash
promptchain agent spawn test-automator-t072 \
  --model "openai/gpt-4" \
  --role "integration-test-specialist" \
  --objective "Create comprehensive integration tests for ExecutionHistoryManager token truncation"
```

### Initial Prompt (Copy-Paste into Chat)

```
I need you to create test_history_truncation.py with 10-12 integration tests validating ExecutionHistoryManager truncation logic.

CONTEXT:
- Implementation: T075-T081 complete (all working)
- Existing tests: 53 tests passing across 3 files
- Dependencies: None (independent integration test)

REQUIREMENTS:
Read the following files to understand the implementation:
1. @promptchain/utils/execution_history_manager.py - Core truncation logic
2. @promptchain/cli/session_manager.py - Integration point (T078)
3. @tests/cli/contract/test_history_config_contract.py - Contract test patterns

TEST COVERAGE NEEDED:
1. Token counting accuracy (validate against tiktoken)
2. oldest_first truncation strategy
3. keep_last truncation strategy
4. max_tokens boundary enforcement
5. max_entries boundary enforcement
6. Combined token + entry limits
7. Filtered truncation (include_types, exclude_sources)
8. Edge cases (empty history, single entry, exact limit)

OUTPUT:
Create tests/cli/integration/test_history_truncation.py with:
- Clear test class organization
- Pytest fixtures for history manager setup
- Proper assertions with error messages
- Edge case coverage
- All tests passing

VALIDATION:
Run: pytest tests/cli/integration/test_history_truncation.py -v
Expected: 10-12 tests passing

See detailed requirements in @docs/PHASE6_TEST_EXECUTION_PLAN.md (Wave 1 section)
```

### Verification Commands

```bash
# Check file created
ls -la tests/cli/integration/test_history_truncation.py

# Run tests
pytest tests/cli/integration/test_history_truncation.py -v

# Check test count
pytest tests/cli/integration/test_history_truncation.py --collect-only

# Run with coverage
pytest tests/cli/integration/test_history_truncation.py -v --cov=promptchain.utils.execution_history_manager
```

---

## Wave 2: T074 - Performance Benchmarking

### Agent Spawn Command

```bash
promptchain agent spawn performance-engineer-t074 \
  --model "openai/gpt-4" \
  --role "performance-benchmarking-specialist" \
  --objective "Create performance tests validating 30-60% token savings from per-agent history configs"
```

### Initial Prompt (Copy-Paste into Chat)

```
I need you to create test_token_optimization.py with 8-10 performance tests measuring token savings from per-agent history configurations.

CONTEXT:
- Implementation: T075-T081 complete (all working)
- Existing tests: 63-65 tests passing (including T072)
- Dependencies: T072 passing (truncation logic verified)

TARGET SAVINGS:
- Terminal agents: 60% token reduction
- Coder agents: 40% token reduction
- Researcher agents: 20% token reduction
- Multi-agent systems: 30-60% overall reduction

REQUIREMENTS:
Read the following files to understand the implementation:
1. @promptchain/cli/utils/agent_templates.py - Default configs (T076)
2. @promptchain/cli/tui/status_bar.py - Token tracking (T079)
3. @promptchain/cli/command_handler.py - /history stats (T081)
4. @tests/cli/integration/test_history_truncation.py - Truncation validation (T072)

TEST SCENARIOS:

BASELINE (no optimization):
1. Multi-agent conversation (6 agents, full history) - measure tokens
2. Long conversation (50 turns, 8000 tokens/agent) - measure tokens
3. Terminal-heavy workflow (4 terminal, 2 researcher) - measure tokens

OPTIMIZED (per-agent configs):
4. Same multi-agent with terminal history disabled
5. Same long conversation with limits (terminal: 0, coder: 4000, researcher: 8000)
6. Same terminal-heavy with terminal history disabled

PERFORMANCE ASSERTIONS:
- Terminal disabled: 60% savings vs baseline
- Coder 4000 limit: 40% savings vs full
- Researcher 8000 limit: 20% savings vs unlimited
- Multi-agent (6 agents): 30-60% overall savings
- Long conversation (50 turns): 50%+ savings

OUTPUT:
Create tests/cli/integration/test_token_optimization.py with:
- TestBaselineTokenUsage class
- TestOptimizedTokenUsage class
- TestTokenSavingsCalculations class
- TestMultiAgentOptimization class
- Token counting utilities
- Performance report generation
- All assertions passing

VALIDATION:
Run: pytest tests/cli/integration/test_token_optimization.py -v
Expected: 8-10 tests passing, all savings targets met

See detailed requirements in @docs/PHASE6_TEST_EXECUTION_PLAN.md (Wave 2 section)
```

### Verification Commands

```bash
# Check file created
ls -la tests/cli/integration/test_token_optimization.py

# Run tests
pytest tests/cli/integration/test_token_optimization.py -v

# Run only savings tests
pytest tests/cli/integration/test_token_optimization.py -v -k "savings"

# Run with detailed output
pytest tests/cli/integration/test_token_optimization.py -v -s

# Check test count
pytest tests/cli/integration/test_token_optimization.py --collect-only
```

---

## Full Phase 6 Validation

### After Both Waves Complete

```bash
# Run all Phase 6 tests sequentially
pytest tests/cli/contract/test_history_config_contract.py -v
pytest tests/cli/integration/test_agent_history_configs.py -v
pytest tests/cli/integration/test_history_truncation.py -v
pytest tests/cli/unit/test_history_defaults.py -v
pytest tests/cli/integration/test_token_optimization.py -v

# Run all Phase 6 tests together
pytest tests/cli/contract/test_history_config_contract.py \
       tests/cli/integration/test_agent_history_configs.py \
       tests/cli/integration/test_history_truncation.py \
       tests/cli/unit/test_history_defaults.py \
       tests/cli/integration/test_token_optimization.py \
       -v

# Check total test count
pytest tests/cli/ --collect-only | grep "tests collected"
# Expected: 71-75 tests

# Run full CLI test suite (check regressions)
pytest tests/cli/ -v

# Generate coverage report
pytest tests/cli/ --cov=promptchain.cli --cov-report=html
```

---

## Agent Configuration Recommendations

### test-automator-t072 Configuration

```yaml
model: openai/gpt-4
temperature: 0.1  # Low temperature for precise test creation
max_tokens: 4000
role: integration-test-specialist

capabilities:
  - Read implementation code
  - Understand test patterns
  - Create pytest tests
  - Validate test coverage

history_config:
  enabled: true
  max_tokens: 8000  # Full history for comprehensive understanding
  truncation_strategy: keep_last
```

### performance-engineer-t074 Configuration

```yaml
model: openai/gpt-4
temperature: 0.2  # Slightly higher for creative benchmarking
max_tokens: 4000
role: performance-benchmarking-specialist

capabilities:
  - Design performance benchmarks
  - Measure token usage
  - Calculate savings percentages
  - Generate performance reports

history_config:
  enabled: true
  max_tokens: 8000  # Full history for benchmark continuity
  truncation_strategy: keep_last
```

---

## Troubleshooting Guide

### T072 Tests Fail

**Symptom**: Truncation tests fail with assertion errors

**Diagnosis**:
```bash
pytest tests/cli/integration/test_history_truncation.py -v -s
```

**Common Fixes**:
1. Check ExecutionHistoryManager.truncate() implementation
2. Verify tiktoken integration (token counting accuracy)
3. Review truncation_strategy logic in session_manager.py
4. Check edge cases (empty history, single entry)

**Reference Files**:
- `promptchain/utils/execution_history_manager.py`
- `promptchain/cli/session_manager.py`

---

### T074 Shows < 30% Savings

**Symptom**: Performance tests fail savings assertions

**Diagnosis**:
```bash
pytest tests/cli/integration/test_token_optimization.py -v -s -k "savings"
```

**Common Fixes**:
1. Review agent type defaults in agent_templates.py
2. Check token tracking in status_bar.py
3. Verify history configs applied correctly
4. Use longer conversation scenarios (50+ turns for statistical significance)
5. Ensure baseline uses unlimited history (no configs)

**Reference Files**:
- `promptchain/cli/utils/agent_templates.py`
- `promptchain/cli/tui/status_bar.py`
- `promptchain/cli/command_handler.py`

---

### Existing Tests Break

**Symptom**: Previously passing tests now fail

**Diagnosis**:
```bash
pytest tests/cli/ -v | grep FAILED
```

**Common Fixes**:
1. Check for shared state between tests (use isolated fixtures)
2. Verify no global state pollution
3. Review test execution order (use pytest-order if needed)
4. Check fixture scope (function vs module vs session)

**Prevention**:
- Use `@pytest.fixture(scope="function")` for test isolation
- Clean up state in fixture teardown
- Avoid modifying global objects in tests

---

## Success Criteria Checklist

### Wave 1 (T072)
- [ ] File created: `tests/cli/integration/test_history_truncation.py`
- [ ] 10-12 tests passing
- [ ] Token counting validated against tiktoken
- [ ] Both truncation strategies tested
- [ ] Edge cases covered (empty, single, boundary)
- [ ] No regressions in existing tests

### Wave 2 (T074)
- [ ] File created: `tests/cli/integration/test_token_optimization.py`
- [ ] 8-10 performance tests passing
- [ ] Terminal agents: 60% savings validated
- [ ] Coder agents: 40% savings validated
- [ ] Researcher agents: 20% savings validated
- [ ] Multi-agent: 30-60% overall savings validated
- [ ] Performance report generated

### Final Validation
- [ ] All 5 Phase 6 test files exist
- [ ] 71-75 total tests passing
- [ ] No regressions in existing 344 passing tests
- [ ] All token savings targets met
- [ ] Documentation updated (tasks.md marked complete)

---

## Documentation References

- **Full Execution Plan**: `/home/gyasis/Documents/code/PromptChain/docs/PHASE6_TEST_EXECUTION_PLAN.md`
- **Quick Start Guide**: `/home/gyasis/Documents/code/PromptChain/docs/PHASE6_QUICK_START.md`
- **Status Summary**: `/home/gyasis/Documents/code/PromptChain/PHASE6_TEST_STATUS.md`
- **Tasks Definition**: `/home/gyasis/Documents/code/PromptChain/specs/002-cli-orchestration/tasks.md`

---

**Ready to execute? Copy the appropriate agent spawn command and initial prompt for each wave.**
