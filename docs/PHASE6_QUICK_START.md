# Phase 6 Test Quick Start Guide

**Status**: 3/5 Complete | 2 Tests Pending (T072, T074)

---

## Current Status

✅ **T070** - Contract test (25 tests passing)
✅ **T071** - Integration test - agent configs (14 tests passing)
❌ **T072** - Integration test - token truncation (PENDING)
✅ **T073** - Unit test - history defaults (14 tests passing)
❌ **T074** - Performance test - token optimization (PENDING)

**Total**: 53 tests passing, 18-22 tests pending

---

## Quick Execution Commands

### Wave 1: T072 - Token Truncation Tests (25-30 min)

```bash
# Create the test file
touch tests/cli/integration/test_history_truncation.py

# Spawn test-automator agent
promptchain agent spawn test-automator \
  --model "openai/gpt-4" \
  --objective "Create test_history_truncation.py with 10-12 tests for ExecutionHistoryManager truncation"

# Provide context in chat
@docs/PHASE6_TEST_EXECUTION_PLAN.md Read Wave 1 requirements

# Reference implementation files
@promptchain/utils/execution_history_manager.py
@promptchain/cli/session_manager.py
@tests/cli/contract/test_history_config_contract.py

# Verify tests pass
pytest tests/cli/integration/test_history_truncation.py -v
```

---

### Wave 2: T074 - Performance Benchmarking (35-40 min)

**Dependency**: T072 must pass first!

```bash
# Create the test file
touch tests/cli/integration/test_token_optimization.py

# Spawn performance-engineer agent
promptchain agent spawn performance-engineer \
  --model "openai/gpt-4" \
  --objective "Create test_token_optimization.py with 8-10 performance tests validating 30-60% token savings"

# Provide context in chat
@docs/PHASE6_TEST_EXECUTION_PLAN.md Read Wave 2 requirements

# Reference implementation files
@promptchain/cli/utils/agent_templates.py
@promptchain/cli/tui/status_bar.py
@promptchain/cli/command_handler.py
@tests/cli/integration/test_history_truncation.py

# Verify tests pass and savings achieved
pytest tests/cli/integration/test_token_optimization.py -v
```

---

## Validation Commands

```bash
# Run all Phase 6 tests
pytest tests/cli/contract/test_history_config_contract.py -v
pytest tests/cli/integration/test_agent_history_configs.py -v
pytest tests/cli/integration/test_history_truncation.py -v
pytest tests/cli/unit/test_history_defaults.py -v
pytest tests/cli/integration/test_token_optimization.py -v

# Check test count
pytest tests/cli/ --collect-only | grep "tests collected"

# Run full CLI test suite (check for regressions)
pytest tests/cli/ -v

# Validate token savings specifically
pytest tests/cli/integration/test_token_optimization.py -v -k "savings"
```

---

## Success Criteria Checklist

### Wave 1 (T072)
- [ ] `test_history_truncation.py` created
- [ ] 10-12 tests passing
- [ ] Token counting validated
- [ ] Both truncation strategies tested (oldest_first, keep_last)
- [ ] Edge cases covered

### Wave 2 (T074)
- [ ] `test_token_optimization.py` created
- [ ] 8-10 performance tests passing
- [ ] Terminal agents: 60% savings validated
- [ ] Coder agents: 40% savings validated
- [ ] Researcher agents: 20% savings validated
- [ ] Multi-agent: 30-60% overall savings validated

### Final Validation
- [ ] All 5 Phase 6 test files exist
- [ ] 71-75 total tests passing (53 existing + 18-22 new)
- [ ] No regressions in existing tests
- [ ] Performance report generated

---

## Troubleshooting

### If T072 tests fail:
1. Check ExecutionHistoryManager implementation
2. Verify tiktoken integration
3. Review truncation strategy logic in session_manager.py
4. Check edge cases (empty history, single entry)

### If T074 shows < 30% savings:
1. Review agent type defaults in agent_templates.py
2. Check token tracking in status_bar.py
3. Verify history configs applied correctly
4. Use longer conversation scenarios (50+ turns)

### If existing tests break:
1. Run `pytest tests/cli/ -v` to identify failures
2. Check if new tests conflict with existing fixtures
3. Verify no shared state between tests
4. Review test isolation (each test should be independent)

---

## Estimated Timeline

| Wave | Duration | Dependency |
|------|----------|------------|
| Wave 1 (T072) | 25-30 min | None |
| Wave 2 (T074) | 35-40 min | Wave 1 complete |
| **Total** | **60-70 min** | **Sequential** |

---

**Ready to execute? See full plan in `docs/PHASE6_TEST_EXECUTION_PLAN.md`**
