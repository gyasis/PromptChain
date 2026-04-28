---

description: "Task list for implementing 011-agentic-prompt-builder"
---

# Tasks: Agentic Prompt Builder Decoupling

**Input**: Design documents from /specs/011-agentic-prompt-builder/
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/prompt_builder_protocol.md, quickstart.md

**Tests**: MANDATORY. Constitution principle III requires tests written and failing before implementation. Test-writing tasks precede their implementation tasks within every phase.

**Organization**: Tasks grouped by user story. Each task is a single line in speckit-canonical form for parser compatibility with both dev-kid orchestrate and dev-kid sentinel-run. Rich context lives in the Task Annotations section below.

## Format: `- [ ] T### [P?] [US?] <description with file path>`

- **[P]**: Logically independent, can run in parallel
- **[US?]**: User story (US1/US2/US3/US4)
- One line per task, no bold markup, no parenthetical asides between ID and description
- Dependencies in the Dependencies section below
- Extra context (blocks-annotations, design notes) in the Task Annotations section below

## Path Conventions

Single Python library project. New module at promptchain/prompts/, new subclass at promptchain/cli/tui_processor.py, tests under tests/. See plan.md for the full layout.

---

## Phase 1: Setup

- [x] T001 Capture legacy prompt snapshot from promptchain/utils/agentic_step_processor.py lines 909-1013 and write to tests/fixtures/legacy_tui_prompt.snapshot.txt
- [x] T002 [P] Create package marker promptchain/prompts/__init__.py as an empty file
- [x] T003 [P] Create the tests/fixtures/ directory if it does not exist

---

## Phase 2: Foundational

- [x] T004 Write skeleton tests/test_prompt_builders.py with test_protocol_has_required_methods and test_protocol_structural_type_check asserting BasePromptBuilder has generate and get_token_estimate
- [x] T005 Implement promptchain/prompts/base.py with BasePromptBuilder Protocol importing only from typing stdlib
- [x] T006 Add BasePromptBuilder re-export to promptchain/prompts/__init__.py

---

## Phase 3: User Story 1 — Library consumer truthful default (Priority P1, MVP)

- [x] T007 [P] [US1] Add test_dynamic_renders_registered_tools to tests/test_prompt_builders.py verifying all 4 custom tool names appear under AVAILABLE TOOLS header
- [x] T008 [P] [US1] Add test_dynamic_does_not_advertise_tui_only_tools to tests/test_prompt_builders.py verifying no TUI-only tool names appear
- [x] T009 [P] [US1] Add test_dynamic_empty_tools_renders_sentinel to tests/test_prompt_builders.py verifying empty tools list produces sentinel line
- [x] T010 [P] [US1] Add test_dynamic_standard_mode_omits_react_block to tests/test_prompt_builders.py verifying standard mode has no ReAct scaffold
- [x] T011 [P] [US1] Add test_dynamic_react_with_tasklist_tool_renders_scaffold to tests/test_prompt_builders.py verifying ReAct mode with task-list tool renders full scaffold
- [ ] T012 [P] [US1] Add test_dynamic_react_without_tasklist_warns_and_falls_back to tests/test_prompt_builders.py verifying warning plus minimal scaffold fallback
- [ ] T013 [P] [US1] Add test_dynamic_standard_mode_line_count_under_cap to tests/test_prompt_builders.py verifying 3-tool standard output stays under 15 lines
- [ ] T014 [P] [US1] Add test_dynamic_token_estimate_positive_and_monotonic to tests/test_prompt_builders.py verifying non-negative monotonic estimate
- [ ] T015 [P] [US1] Add test_dynamic_context_block_renders_when_provided to tests/test_prompt_builders.py verifying PRIOR CONTEXT header appears when context argument passed
- [ ] T016 [P] [US1] Add test_dispatch_default_is_dynamic_standard_mode to tests/test_prompt_builders.py verifying no-kwarg processor gets DynamicPromptGenerator in standard mode
- [ ] T017 [P] [US1] Add test_dynamic_tool_description_updates_reflected_on_regenerate to tests/test_prompt_builders.py verifying no caching at construction
- [x] T018 [US1] Implement DynamicPromptGenerator in promptchain/prompts/dynamic.py with extra_instructions, workflow_pattern, include_response_format_hint constructor params and the full render logic from contracts
- [x] T019 [US1] Re-export DynamicPromptGenerator from promptchain/prompts/__init__.py
- [x] T020 [US1] Modify promptchain/utils/agentic_step_processor.py __init__ to add prompt_builder and workflow_pattern kwargs with default-branch dispatch constructing DynamicPromptGenerator
- [x] T021 [US1] Modify promptchain/utils/agentic_step_processor.py run() replacing the hardcoded system-prompt f-string with a self.prompt_builder.generate call
- [x] T022 [P] [US1] Write tests/integration/test_011_library_consumer_flow.py mirroring the PRD reproduction with 4 stub tools and a mocked LLM
- [ ] T023 [US1] Run the US1 pytest subset and confirm all pass

---

## Phase 4: User Story 2 — TUI preservation and swappable processor class (Priority P1)

- [ ] T024 [P] [US2] Add test_legacy_snapshot_byte_identical to tests/test_prompt_builders.py verifying LegacyTUIPromptGenerator output byte-equals fixture
- [ ] T025 [P] [US2] Add test_legacy_ignores_tools_for_content to tests/test_prompt_builders.py verifying tool-list argument does not alter output
- [ ] T026 [P] [US2] Add test_legacy_drift_warning_on_unexpected_tools to tests/test_prompt_builders.py verifying logger.warning on tool-name set mismatch
- [ ] T027 [P] [US2] Add test_tui_subclass_bakes_legacy_builder to tests/test_prompt_builders.py verifying TUIAgenticStepProcessor prompt_builder is LegacyTUIPromptGenerator instance
- [ ] T028 [P] [US2] Add test_tui_subclass_rejects_prompt_builder_kwarg to tests/test_prompt_builders.py verifying TypeError
- [ ] T029 [P] [US2] Add test_tui_subclass_rejects_instructions_kwarg to tests/test_prompt_builders.py verifying TypeError
- [ ] T030 [P] [US2] Add test_subclass_inheritance_enhanced_sees_custom_tools to tests/test_prompt_builders.py verifying EnhancedAgenticStepProcessor propagates the fix
- [ ] T031 [P] [US2] Add test_subclass_inheritance_state_agent_sees_custom_tools to tests/test_prompt_builders.py verifying StateAgent propagates the fix
- [ ] T032 [P] [US2] Add test_callsite_compliance_tui_grep to tests/test_prompt_builders.py verifying no bare AgenticStepProcessor calls remain in promptchain/cli/ or agentic_chat/
- [x] T033 [US2] Implement LegacyTUIPromptGenerator in promptchain/prompts/legacy_tui.py returning the frozen v0.5.0 prompt with objective substituted
- [x] T034 [US2] Re-export LegacyTUIPromptGenerator from promptchain/prompts/__init__.py
- [x] T035 [US2] Implement TUIAgenticStepProcessor subclass in promptchain/cli/tui_processor.py baking LegacyTUIPromptGenerator and rejecting prompt_builder or instructions kwargs
- [x] T036 [P] [US2] Migrate both default-path call sites in promptchain/cli/tui/app.py to TUIAgenticStepProcessor
- [x] T037 [P] [US2] Migrate promptchain/cli/config/yaml_translator.py line 315 to TUIAgenticStepProcessor
- [x] T038 [P] [US2] Migrate the 7 call sites in agentic_chat/agentic_team_chat.py default-path sites to TUIAgenticStepProcessor and annotate any sub-agent spawners with the whitelist comment
- [ ] T039 [US2] Run the US2 pytest subset and confirm all pass including the grep-compliance check

---

## Phase 5: User Story 3 — instructions= deprecation shim (Priority P2)

- [ ] T040 [P] [US3] Add test_dispatch_mutually_exclusive_raises_value_error to tests/test_prompt_builders.py verifying ValueError when both kwargs supplied
- [ ] T041 [P] [US3] Add test_dispatch_instructions_emits_single_deprecation_warning to tests/test_prompt_builders.py verifying one DeprecationWarning naming the replacement API
- [ ] T042 [P] [US3] Add test_dispatch_instructions_renders_as_extra_instructions_block to tests/test_prompt_builders.py verifying supplied instructions appear under ADDITIONAL INSTRUCTIONS header
- [x] T043 [US3] Extend promptchain/utils/agentic_step_processor.py __init__ with the instructions kwarg handling the mutual-exclusion ValueError and the DeprecationWarning shim path
- [x] T044 [P] [US3] Update tests/test_tao_loop.py replacing old-prompt substring assertions with dynamic-render or legacy-snapshot assertions per FR-019
- [x] T045 [P] [US3] Update tests/test_verification_integration.py with the same migration per FR-019
- [x] T046 [P] [US3] Update tests/cli/integration/test_agentic_reasoning.py with the same migration per FR-019
- [ ] T047 [US3] Run the US3 pytest subset and confirm all pass

---

## Phase 6: User Story 4 — Custom prompt-builder injection (Priority P3)

- [ ] T048 [P] [US4] Add test_dispatch_accepts_custom_builder to tests/test_prompt_builders.py verifying processor delegates to the injected builder
- [ ] T049 [P] [US4] Add test_dispatch_custom_builder_with_non_default_workflow_hint_warns to tests/test_prompt_builders.py verifying logger.warning on the ignored hint
- [x] T050 [US4] Extend promptchain/utils/agentic_step_processor.py __init__ prompt_builder-alone branch with the workflow_pattern-ignored warning

---

## Phase 7: Polish

- [x] T051 [P] Update CHANGELOG.md with the v0.6.0 section naming Added, Restored, and Changed BREAKING entries and referencing issue 2
- [x] T052 [P] Add module docstring to promptchain/prompts/dynamic.py documenting the ReAct task-list-tool heuristic allowlist
- [ ] T053 [P] Add cross-reference note at top of promptchain/utils/agentic_step_processor.py describing the prompt_builder delegation contract for subclass authors
- [ ] T054 Run full pytest regression and verify zero pre-existing tests regress beyond the three files migrated in T044-T046
- [ ] T055 [P] Run mypy over promptchain/prompts/ promptchain/utils/agentic_step_processor.py promptchain/cli/tui_processor.py
- [x] T056 [P] Run black and isort over the new and modified files
- [x] T057 Run quickstart.md Part 1 library-consumer recipe manually end-to-end against a throwaway chain
- [ ] T058 Run quickstart.md Part 2 TUI maintainer recipe manually including a sub-agent spawn
- [x] T059 Bump version string in pyproject.toml and setup.py to 0.6.0

---

## Phase 8: Coverage-gap closure

- [ ] T060 [P] [US1] Add test_dynamic_extra_instructions_renders_block to tests/test_prompt_builders.py verifying FR-005 direct-path behavior
- [ ] T061 [P] [US1] Add test_dynamic_final_answer_block_conditional_on_flag to tests/test_prompt_builders.py verifying FR-007 both True and False cases
- [ ] T062 [P] Add test_no_process_global_state_in_prompts_module to tests/test_prompt_builders.py performing AST static check for FR-017 and SC-008 verification
- [ ] T063 [P] Add test_base_module_has_no_internal_imports to tests/test_prompt_builders.py performing AST import audit for FR-018 verification
- [x] T064 [P] [US1] Parameterize test_dynamic_renders_registered_tools over N in 0 1 4 10 for SC-001 full verification
- [ ] T065 [P] [US2] Add test_tui_sub_agent_spawn_uses_custom_instructions to tests/test_prompt_builders.py using DynamicPromptGenerator directly to avoid the instructions shim dependency

---

## Task Annotations

Rich context moved out of task lines to keep each task parseable by strict sentinel-run regex. The orchestrator reads this section too.

### T001

- Blocks T021: snapshot must be captured before the f-string is removed from source
- Fallback: if T021 has already landed, extract snapshot from git history via `git show HEAD~N:promptchain/utils/agentic_step_processor.py | sed -n '909,1013p'`
- Covers FR-020 and SC-002 reference fixture
- Write content to `tests/fixtures/legacy_tui_prompt.snapshot.txt` preserving `{objective}` as a literal placeholder for later substitution

### T004

- Tests fail at this point because BasePromptBuilder does not yet exist
- Covers FR-002 and FR-003 Protocol contract verification

### T005

- Imports only from typing stdlib so the module has zero internal dependencies and cannot create circular import cycles
- Methods: `generate(objective, tools, context=None) -> str` and `get_token_estimate(objective, tools) -> int`
- Covers FR-018 zero-internal-imports constraint

### T018

- Constructor parameters: `extra_instructions: Optional[List[str]] = None`, `workflow_pattern: Literal["standard", "react"] = "standard"`, `include_response_format_hint: bool = True`
- `generate()` renders objective plus AVAILABLE TOOLS block plus optional workflow/extra-instructions/context/final-answer blocks per contracts/prompt_builder_protocol.md section 2
- `get_token_estimate()` uses tiktoken when importable, falls back to `len(rendered) // 4`
- Must conform to BasePromptBuilder Protocol structurally

### T020

- Adds kwargs `prompt_builder: Optional[BasePromptBuilder] = None` and `workflow_pattern: Literal["standard", "react"] = "standard"`
- The instructions= kwarg and mutual-exclusion branch are deferred to T043 in Phase 5
- Import BasePromptBuilder and DynamicPromptGenerator from promptchain.prompts
- This is the single serialization point extended by T043 and T050

### T021

- Replaces the hardcoded f-string at the former lines 909-1013 with this exact snippet:
  ```python
  system_prompt = self.prompt_builder.generate(
      objective=self.objective,
      tools=available_tools,
      context=(scratchpad_text if scratchpad_text else None),
  )
  ```
- Delete the dead f-string after replacement. Everything else in run() stays untouched.
- Requires T001 snapshot capture to have happened first

### T022

- Chain with 4 mock retrieval tools (stub functions returning deterministic strings)
- AgenticStepProcessor constructed with no prompt config
- Mocked LLM client records the system prompt sent
- Asserts: (a) prompt contains all 4 tool names, (b) prompt does NOT contain TUI-only names, (c) final result references at least one stub return value
- Uses pytest.mark.integration
- Note on proxy strength: the mocked-LLM grounding assertion is weaker than SC-003's live-path assertion. The full live-LLM check belongs in an optional companion test marked pytest.mark.requires_api_key gated on environment variables, added as a follow-up PR rather than gating v0.6.0 CI on live-API credentials.

### T033

- No-arg constructor
- `generate(objective, tools, context=None)` returns the frozen snapshot with `{objective}` substituted via str.replace
- When context is provided, appends `\n\nPRIOR CONTEXT:\n{context}` after the frozen body
- When `len(tools) > 0` and `set(tool_names) != _EXPECTED_TUI_DEFAULT_TOOL_NAMES`, logs a single warning naming the unexpected tools
- `get_token_estimate` matches the dynamic estimator implementation
- Module docstring must point at `tests/fixtures/legacy_tui_prompt.snapshot.txt`

### T035

- Subclass of AgenticStepProcessor
- Constructor rejects `prompt_builder=` and `instructions=` kwargs with TypeError pointing callers at the base class
- Calls `super().__init__(prompt_builder=LegacyTUIPromptGenerator(), **kwargs)`
- Imports from promptchain.prompts

### T038

- Call sites at current lines 60, 150, 210, 526, 566, 647, 1249 in agentic_chat/agentic_team_chat.py
- Default-path sites (research_step, analysis_step, terminal_step, documentation_step, synthesis_step, coding_step) convert to TUIAgenticStepProcessor
- Sub-agent spawner sites (orchestrator_step at 1249 is the likely candidate) keep vanilla AgenticStepProcessor with a `# sub-agent: ...` whitelist comment explaining why

### T043

- Full dispatch state machine from contracts/prompt_builder_protocol.md section 4:
  - when both instructions and prompt_builder are non-None raise ValueError with the mutual-exclusion message
  - when instructions alone is non-None emit warnings.warn with DeprecationWarning and stacklevel=2 then construct DynamicPromptGenerator with extra_instructions and workflow_pattern
  - prompt_builder-alone and default branches remain from T020

### T054

- Fix any pre-existing test fallout in the file where it surfaces, not in a global patch commit

### T062

- AST static check parses promptchain/prompts/base.py, dynamic.py, legacy_tui.py
- Asserts no ClassVar annotations on mutable types, no os.environ references, no tool-name substring branching outside the documented ReAct task-list heuristic allowlist
- Also inspects promptchain/utils/agentic_step_processor.py __init__ for no env-var reads influencing dispatch

### T063

- AST import audit parses promptchain/prompts/base.py
- Walks ast.Import and ast.ImportFrom nodes
- Asserts every imported module starts with typing or is a stdlib module and no import starts with promptchain.

### T065

- Logical home Phase 4 tests, self-contained with no cross-phase dependency
- Construct vanilla AgenticStepProcessor with prompt_builder=DynamicPromptGenerator(extra_instructions=[...]) directly to avoid the instructions= shim dependency on T043
- Capture prompt via processor.prompt_builder.generate and assert it contains the supplied instructions and does NOT contain legacy scaffold fragments
- Design note: do NOT rewrite this test to use instructions=[...] even though it would also work. The shim path adds an unnecessary cross-phase dependency that was deliberately eliminated during the post-analyze remediation pass.

### Phase 8 logical homes

- T060 logical home Phase 3 tests
- T061 logical home Phase 3 tests
- T062 logical home Phase 7 polish
- T063 logical home Phase 2 foundational tests
- T064 logical home Phase 3 tests extending T007
- T065 logical home Phase 4 tests

---

## Dependencies

### Task dependency graph (machine-readable)

Orchestrator reads these as `Tcurrent requires Tprereq1, Tprereq2, ...`.

**Phase 2 — Foundational**:

- T004 requires T002, T003
- T005 requires T002
- T006 requires T005

**Phase 3 — US1**:

- T007 requires T004, T005
- T008 requires T004, T005
- T009 requires T004, T005
- T010 requires T004, T005
- T011 requires T004, T005
- T012 requires T004, T005
- T013 requires T004, T005
- T014 requires T004, T005
- T015 requires T004, T005
- T016 requires T004, T005
- T017 requires T004, T005
- T018 requires T005
- T019 requires T018, T006
- T020 requires T005, T018
- T021 requires T020, T001
- T022 requires T021, T020, T018, T019
- T023 requires T007, T008, T009, T010, T011, T012, T013, T014, T015, T016, T017, T018, T019, T020, T021, T022

**Phase 4 — US2**:

- T024 requires T004, T001
- T025 requires T004
- T026 requires T004
- T027 requires T004
- T028 requires T004
- T029 requires T004
- T030 requires T004, T005
- T031 requires T004, T005
- T032 requires T004
- T033 requires T005, T001
- T034 requires T033, T019
- T035 requires T033, T020
- T036 requires T035
- T037 requires T035
- T038 requires T035
- T039 requires T024, T025, T026, T027, T028, T029, T030, T031, T032, T033, T034, T035, T036, T037, T038

**Phase 5 — US3**:

- T040 requires T004
- T041 requires T004
- T042 requires T004
- T043 requires T020, T018
- T044 requires T043
- T045 requires T043
- T046 requires T043
- T047 requires T040, T041, T042, T043, T044, T045, T046

**Phase 6 — US4**:

- T048 requires T004, T005
- T049 requires T004, T005
- T050 requires T020

**Phase 7 — Polish**:

- T052 requires T018
- T053 requires T021
- T054 requires T023, T039, T047, T050
- T055 requires T005, T018, T033, T035, T043
- T056 requires T018, T033, T035
- T057 requires T018, T020, T043
- T058 requires T035, T036, T037, T038

**Phase 8 — Coverage-gap closure**:

- T060 requires T018, T004
- T061 requires T018, T004
- T062 requires T005, T018, T033, T020, T043
- T063 requires T005, T004
- T064 requires T018, T004
- T065 requires T020, T018

### Phase-level narrative summary

- Phase 1: no dependencies
- Phase 2: depends on Phase 1, blocks all user stories
- Phase 3 US1: depends on Phase 2, produces dispatch + run() integration prerequisites for US3 and US4
- Phase 4 US2: depends on Phase 2 and T020, produces LegacyTUIPromptGenerator + TUIAgenticStepProcessor + call-site migrations
- Phase 5 US3: depends on Phase 3, can run in parallel with Phase 4 once T020 lands
- Phase 6 US4: depends on Phase 3, can run in parallel with Phases 4 and 5
- Phase 7: depends on all user story phases complete
- Phase 8: tasks logically belong in earlier phases per their annotations, but dependency edges above route them correctly

### Within-phase TDD pattern

- Tests within any user story phase are written first and fail, implementation tasks turn them green
- Tests appending new def test_xxx functions to tests/test_prompt_builders.py are marked [P] because pytest does not care about order
- T020 is the single serialization point shared by T043 and T050, merges must be sequential

### Parallel opportunities

- All Phase 1 [P] tasks can run together
- All test-writing [P] tasks within a user story can run together
- Once Phase 2 plus T020 are done, US2 US3 US4 can be worked in parallel by three developers
- Within Phase 4 implementation, T036 T037 T038 are in different files and are [P]
- Within Phase 7, polish tasks T051 T052 T053 T055 T056 are [P]

---

## Implementation Strategy

### MVP First (US1 only)

1. Complete Phase 1 and Phase 2 (T001 to T006)
2. Complete Phase 3 US1 (T007 to T023)
3. STOP and validate: library consumers now receive truthful prompts
4. Optionally tag v0.6.0-alpha for downstream consumer migration

### Incremental delivery

1. Setup and Foundational, foundation ready
2. Add US1, library consumers fixed, alpha tag
3. Add US2, TUI preservation confirmed, beta tag
4. Add US3, instructions= compat restored, rc1
5. Add US4, custom-builder injection, Polish, v0.6.0 final

### Parallel team strategy

Once Phase 2 plus T020 plus T021 are in main:

- Developer A: Phase 4 US2
- Developer B: Phase 5 US3
- Developer C: Phase 6 US4

Serialization points T020, T043, T050 all extend the same __init__ method in agentic_step_processor.py. Merge with care or land sequentially.

---

## Notes

- [P] within a single-file test append means logically independent test function
- Every user story phase ends with a verification task (T023, T039, T047) confirming tests pass
- T001 snapshot capture must happen before T021 removes the hardcoded f-string. If out of order, extract from git history per T001 annotation.
- After US2 lands, any new AgenticStepProcessor call site in promptchain/cli/ or agentic_chat/ must use TUIAgenticStepProcessor or carry a `# sub-agent:` whitelist comment or T032 grep-compliance test will fail
