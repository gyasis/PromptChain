# Phase 4: Multi-Hop Reasoning - Complete Execution Plan

## Executive Overview

**Phase**: User Story 2 - Multi-Hop Reasoning Integration
**Tasks**: T045-T055 (11 tasks total)
**Approach**: 5-wave parallel execution strategy
**Timeline**: 2-3 hours total
**Token Budget**: ~180K tokens (vs ~400K sequential, **55% savings**)

---

## Wave Structure

### Wave 1: Foundation Tests (20-25 min, ~40K tokens)
**Parallel Execution**: 2 agents simultaneously

| Task | Agent | Focus | Deliverable |
|------|-------|-------|-------------|
| T045 | test-dev-specialist | Contract test for instruction chain | `tests/cli/contract/test_agentic_instruction_chain.py` |
| T048 | test-dev-specialist | Unit test for YAML config translation | `tests/cli/unit/test_yaml_agentic_step_config.py` |

**Spawn Commands**:
```bash
promptchain spawn test-dev-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave1_t045_prompt.md \
  --session phase4-t045-contract-test \
  --background

promptchain spawn test-dev-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave1_t048_prompt.md \
  --session phase4-t048-yaml-config-test \
  --background

# Wait for completion
wait-agents phase4-t045-contract-test phase4-t048-yaml-config-test

# Verify
pytest tests/cli/contract/test_agentic_instruction_chain.py -v
pytest tests/cli/unit/test_yaml_agentic_step_config.py -v
```

---

### Wave 2: Core Backend (30-35 min, ~50K tokens)
**Sequential Execution**: Dependencies require ordering

| Task | Agent | Focus | Deliverable |
|------|-------|-------|-------------|
| T050 | backend-implementation-specialist | Create AgenticStepProcessor from YAML | Extended `yaml_translator.py` |
| T049 | backend-implementation-specialist | Instruction chain processing | Modified `promptchaining.py` |

**Spawn Commands**:
```bash
promptchain spawn backend-implementation-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave2_t050_prompt.md \
  --session phase4-t050-yaml-processor \
  --wait  # Block until complete

pytest tests/cli/unit/test_yaml_agentic_step_config.py -v

promptchain spawn backend-implementation-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave2_t049_prompt.md \
  --session phase4-t049-instruction-chain \
  --wait  # Block until complete

pytest tests/cli/contract/test_agentic_instruction_chain.py -v
```

**Why Sequential**: T049 depends on T050's YAML translation capability.

---

### Wave 3: Integration Tests + UI Foundation (25-30 min, ~45K tokens)
**Hybrid Execution**: Tests sequential, UI parallel

| Task | Agent | Focus | Deliverable |
|------|-------|-------|-------------|
| T046 | test-dev-specialist | AgenticStepProcessor in AgentChain | `tests/cli/integration/test_agentchain_agentic_processor.py` |
| T047 | test-dev-specialist | Multi-hop reasoning with tool calls | `tests/cli/integration/test_multihop_reasoning_tools.py` |
| T051 | ui-specialist | Reasoning progress widget | `promptchain/cli/tui/widgets/reasoning_progress.py` |
| T055 | backend-implementation-specialist | Error handling | Error classes + TUI integration |

**Spawn Commands**:
```bash
# Group 3A: Integration Tests (Sequential)
promptchain spawn test-dev-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave3_t046_prompt.md \
  --session phase4-t046-agentchain-integration \
  --wait

pytest tests/cli/integration/test_agentchain_agentic_processor.py -v

promptchain spawn test-dev-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave3_t047_prompt.md \
  --session phase4-t047-multihop-tools \
  --wait

pytest tests/cli/integration/test_multihop_reasoning_tools.py -v

# Group 3B: UI Foundation (Parallel - can overlap with tests)
promptchain spawn ui-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave3_t051_prompt.md \
  --session phase4-t051-progress-widget \
  --background

promptchain spawn backend-implementation-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave3_t055_prompt.md \
  --session phase4-t055-error-handling \
  --background

wait-agents phase4-t051-progress-widget phase4-t055-error-handling
```

---

### Wave 4: UI Integration (20-25 min, ~30K tokens)
**Sequential Execution**: Each builds on previous

| Task | Agent | Focus | Deliverable |
|------|-------|-------|-------------|
| T052 | ui-integration-specialist | Streaming output | Progress callback integration |
| T053 | backend-implementation-specialist | History logging | Extended `ExecutionHistoryManager` |
| T054 | ui-integration-specialist | Completion display | Synthesis formatting + TUI display |

**Spawn Commands**:
```bash
promptchain spawn ui-integration-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave4_t052_prompt.md \
  --session phase4-t052-streaming \
  --wait

promptchain spawn backend-implementation-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave4_t053_prompt.md \
  --session phase4-t053-history-logging \
  --wait

pytest tests/cli/unit/test_reasoning_history_logging.py -v

promptchain spawn ui-integration-specialist \
  --prompt-file /home/gyasis/Documents/code/PromptChain/docs/phase4_wave4_t054_prompt.md \
  --session phase4-t054-completion-display \
  --wait

pytest tests/cli/unit/test_completion_detection.py -v
```

**Why Sequential**: T052 → T053 → T054 form integrated pipeline.

---

### Wave 5: Final Verification (10-15 min, ~15K tokens)
**Comprehensive Testing**: All components together

**Test Suite**:
```bash
# All Phase 4 tests
pytest tests/cli/ -k "agentic" -v --tb=short

# Contract tests
pytest tests/cli/contract/test_agentic_instruction_chain.py -v

# Unit tests
pytest tests/cli/unit/test_yaml_agentic_step_config.py -v
pytest tests/cli/unit/test_agentic_error_handling.py -v
pytest tests/cli/unit/test_reasoning_history_logging.py -v
pytest tests/cli/unit/test_completion_detection.py -v

# Integration tests
pytest tests/cli/integration/test_agentchain_agentic_processor.py -v
pytest tests/cli/integration/test_multihop_reasoning_tools.py -v

# Expected: >= 15 tests passing
```

**End-to-End TUI Test**:
```bash
# Create test config
cat > /tmp/test_agentic_config.yml <<EOF
agents:
  researcher:
    model: "openai/gpt-4o-mini"
    description: "Research agent with multi-hop reasoning"
    instructions:
      - "Prepare research strategy: {input}"
      - agentic_step:
          objective: "Conduct thorough research and analysis"
          max_internal_steps: 6
          history_mode: "progressive"
      - "Synthesize findings: {input}"

router:
  type: "single_agent_dispatch"
  model: "openai/gpt-4o-mini"

execution_mode: "router"
EOF

# Launch TUI
promptchain --config /tmp/test_agentic_config.yml

# Test query
# > Find and analyze authentication patterns in this codebase
```

**Verification Checklist**:
- [ ] All 11 tasks (T045-T055) complete
- [ ] >= 15 new tests passing
- [ ] YAML config loads without errors
- [ ] TUI progress widget displays correctly
- [ ] Streaming updates work smoothly
- [ ] Error handling graceful
- [ ] History logging functional
- [ ] Completion synthesis displays
- [ ] Code quality clean (mypy, flake8)
- [ ] No regressions

---

## Agent Specializations

### Test Development Specialist
**Expertise**: Test design, contract validation, edge case coverage
**Responsibilities**: T045, T046, T047, T048
**Output**: Comprehensive test suites with >90% coverage

### Backend Implementation Specialist
**Expertise**: Core logic, data structures, async patterns
**Responsibilities**: T049, T050, T053, T055
**Output**: Production-ready backend code

### UI Specialist
**Expertise**: Textual widgets, Rich formatting, user experience
**Responsibilities**: T051
**Output**: Polished TUI components

### UI Integration Specialist
**Expertise**: Event handling, callbacks, thread safety
**Responsibilities**: T052, T054
**Output**: Seamless UI integration

---

## Synchronization Points

### Checkpoint 1: After Wave 1
**Verification**: Both contract tests pass
**Block**: Wave 2 cannot start until Wave 1 complete

### Checkpoint 2: After Wave 2
**Verification**: YAML translation works, instruction chain processes AgenticStepProcessor
**Block**: Wave 3 cannot start until T049 complete

### Checkpoint 3: After Wave 3
**Verification**: Integration tests pass, UI widgets ready
**Block**: Wave 4 cannot start until all Wave 3 complete

### Checkpoint 4: After Wave 4
**Verification**: Streaming, history, completion all functional
**Block**: Wave 5 verification can begin

### Checkpoint 5: Final Sign-Off
**Verification**: All 11 tasks complete, all tests pass, E2E demo works
**Result**: Phase 4 complete, ready for Phase 5

---

## Token Budget Breakdown

| Wave | Tasks | Estimated Tokens | Parallel Savings |
|------|-------|------------------|------------------|
| Wave 1 | T045, T048 | 40K | 50% (parallel) |
| Wave 2 | T049, T050 | 50K | 0% (sequential) |
| Wave 3 | T046, T047, T051, T055 | 45K | 40% (hybrid) |
| Wave 4 | T052, T053, T054 | 30K | 0% (sequential) |
| Wave 5 | Verification | 15K | N/A |
| **Total** | **11 tasks** | **~180K** | **~55% overall** |

**Comparison**:
- Sequential approach: ~400K tokens
- Parallel approach: ~180K tokens
- **Savings**: 220K tokens (55%)

---

## Risk Mitigation

### Risk 1: Integration Failures
**Mitigation**: Comprehensive tests at each wave, early integration in Wave 3

### Risk 2: UI Thread Safety Issues
**Mitigation**: Use Textual's `@work` decorator, test with background agents

### Risk 3: Token Budget Overrun
**Mitigation**: Use fast models (gpt-4o-mini), clear prompts, focused scopes

### Risk 4: Agent Coordination Delays
**Mitigation**: Clear synchronization points, explicit wait commands

### Risk 5: Test Flakiness
**Mitigation**: Mock external dependencies, deterministic test data

---

## Success Metrics

### Quantitative
- All 11 tasks complete ✓
- >= 15 new tests passing ✓
- 0 regressions in existing tests ✓
- Token budget within 180K ± 10% ✓
- Timeline within 3 hours ± 30 min ✓

### Qualitative
- Professional TUI appearance ✓
- Smooth user experience ✓
- Clear error messages ✓
- Comprehensive documentation ✓
- Clean, maintainable code ✓

---

## Deliverables Inventory

### Code Files
- `promptchain/utils/agentic_step_processor.py` (extended)
- `promptchain/utils/promptchaining.py` (extended)
- `promptchain/utils/execution_history_manager.py` (extended)
- `promptchain/cli/config/yaml_translator.py` (extended)
- `promptchain/cli/tui/app.py` (extended)
- `promptchain/cli/tui/widgets/reasoning_progress.py` (new)
- `promptchain/cli/error_handler.py` (extended)
- `promptchain/cli/utils/synthesis_formatter.py` (new)

### Test Files
- `tests/cli/contract/test_agentic_instruction_chain.py` (new)
- `tests/cli/unit/test_yaml_agentic_step_config.py` (new)
- `tests/cli/unit/test_agentic_error_handling.py` (new)
- `tests/cli/unit/test_reasoning_history_logging.py` (new)
- `tests/cli/unit/test_completion_detection.py` (new)
- `tests/cli/integration/test_agentchain_agentic_processor.py` (new)
- `tests/cli/integration/test_multihop_reasoning_tools.py` (new)

### Documentation
- 11 task prompt files (`docs/phase4_wave*_t*.md`)
- Verification plan (`docs/phase4_wave5_verification.md`)
- This master plan (`docs/PHASE4_EXECUTION_MASTER_PLAN.md`)
- Completion summary (generated in Wave 5)

---

## Next Steps After Phase 4

### Phase 5: Advanced MCP Tool Integration
**Prerequisites**: Phase 4 complete
**Focus**: Multi-hop reasoning with complex MCP tool workflows

### Phase 6: Complex Workflow Orchestration
**Prerequisites**: Phases 4 & 5 complete
**Focus**: Multi-agent collaboration patterns with reasoning

### Phase 7: Production Optimization
**Prerequisites**: Phases 4, 5, 6 complete
**Focus**: Performance tuning, monitoring, production hardening

---

## Quick Start Commands

**Execute Entire Phase 4**:
```bash
# Clone this master plan
cd /home/gyasis/Documents/code/PromptChain

# Execute waves sequentially (with built-in checkpoints)
./scripts/execute_phase4.sh  # (create this script from wave commands above)

# Or manual execution:
# Wave 1
bash -c "$(grep -A 15 'Wave 1.*Spawn Commands' docs/PHASE4_EXECUTION_MASTER_PLAN.md | tail -n +2)"

# Wave 2
bash -c "$(grep -A 12 'Wave 2.*Spawn Commands' docs/PHASE4_EXECUTION_MASTER_PLAN.md | tail -n +2)"

# Wave 3
bash -c "$(grep -A 20 'Wave 3.*Spawn Commands' docs/PHASE4_EXECUTION_MASTER_PLAN.md | tail -n +2)"

# Wave 4
bash -c "$(grep -A 12 'Wave 4.*Spawn Commands' docs/PHASE4_EXECUTION_MASTER_PLAN.md | tail -n +2)"

# Wave 5
bash -c "$(grep -A 15 'Wave 5.*Test Suite' docs/PHASE4_EXECUTION_MASTER_PLAN.md | tail -n +2)"
```

---

## Contact & Support

**Phase Owner**: Team Orchestrator Agent
**Execution Period**: Phase 4 (Multi-Hop Reasoning)
**Dependencies**: Phases 1-3 must be complete
**Status Tracking**: `docs/phase4_completion.md` (generated in Wave 5)

---

*Generated: 2025-11-22*
*Plan Version: 1.0*
*PromptChain CLI - Phase 4: Multi-Hop Reasoning Integration*
