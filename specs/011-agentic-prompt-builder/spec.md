# Feature Specification: Agentic Prompt Builder Decoupling

**Feature Branch**: `011-agentic-prompt-builder`
**Created**: 2026-04-17
**Status**: Draft
**Input**: User description: "Decouple AgenticStepProcessor from its hardcoded TUI system prompt and tool inventory using the Strategy pattern. Full PRD at prd/agentic_step_processor_prompt_decoupling_prd.md. High-level: library consumers currently get agents whose system prompt lies about what tools are available (hardcoded at agentic_step_processor.py:909-1013). Fix: introduce promptchain/prompts/ module with BasePromptBuilder Protocol, DynamicPromptGenerator (new default, renders chain.tools at runtime), and LegacyTUIPromptGenerator (preserves current behavior, used explicitly by TUI). Restore the removed instructions= parameter as a DeprecationWarning-emitting shim. Target v0.6.0. Tracking issue gyasis/PromptChain#2."
**Source PRD**: `prd/agentic_step_processor_prompt_decoupling_prd.md`
**Target Release**: v0.6.0
**Tracking Issue**: gyasis/PromptChain#2

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Library consumer gets a truthful system prompt by default (Priority: P1)

A developer building a domain-specific agent (for example a RAG service, a healthcare EHR analysis server, or a custom research agent) installs PromptChain, registers their own custom tools on a chain, and instantiates the default agentic step processor with nothing more than an objective. They expect the agent's underlying system prompt to describe the tools they actually registered — not an inventory of tools that only exist inside PromptChain's shipped terminal UI.

**Why this priority**: This is the root bug. Every non-TUI consumer of the library today receives agents that hallucinate, free-associate from training priors, or skip tool use entirely because the system prompt misrepresents the tool inventory. Fixing this is the entire purpose of the release. Without it, the library is structurally misleading for every consumer outside of the shipped terminal UI.

**Independent Test**: Instantiate the default agentic step processor on a chain with four custom tools and zero additional configuration. Capture the system prompt that would be sent to the language model. Verify it names exactly the four registered tools and contains no reference to any tool name that only belongs to the shipped terminal UI's default toolset. This can be validated by snapshot/string-match tests without running any live model call.

**Acceptance Scenarios**:

1. **Given** a chain with four custom tools registered and no prompt configuration passed, **When** the default agentic step processor is constructed and prepares its system prompt, **Then** the prompt contains a block listing each of the four registered tool names with their descriptions and contains none of the terminal-UI-specific tool names (task list writer, ripgrep search, generic file read/write/edit, terminal executor, sandbox operations, or the Gemini MCP tools).
2. **Given** a chain with zero tools registered, **When** the default agentic step processor prepares its system prompt, **Then** the prompt clearly indicates that no tools are available rather than advertising nonexistent tools.
3. **Given** a chain with a tool whose description has been updated between constructions, **When** the prompt is regenerated, **Then** the new description is reflected (the prompt is derived from live tool state at generation time, not cached at construction time).

---

### User Story 2 - Terminal UI behavior is unchanged and processor class is swappable (Priority: P1)

A user who runs the PromptChain shipped terminal UI before and after upgrading to the target release sees identical default-agent system-level behavior. The exact wording, structure, and tool inventory of the default terminal-UI agent's system prompt must not change across this release. In addition, the terminal UI is not locked into a single processor class — the default path uses a dedicated terminal-UI processor subclass (which bakes in the legacy prompt), but the terminal UI is free to construct vanilla processor instances, enhanced processor instances, or state-driven processor instances on demand for specialized workflows such as sub-agent spawning.

**Why this priority**: The terminal UI is a shipped user-facing product. The PRD explicitly constrains the release to preserve its default behavior. Any regression here would be a user-visible breaking change for the headline product of the library and would block the release. The second half of the story (swappable processor class) is equally P1 because without it the terminal UI cannot launch sub-agents whose system prompts match their own instructions — defeating much of the library-consumer fix for the TUI's own orchestrator.

**Independent Test**: Run the terminal UI before the release and capture the default agent's system prompt. Run it after the release and capture the same prompt. Compare the two — they must match modulo the dynamic objective substitution. Separately, confirm that terminal-UI code paths that spawn sub-agents can construct a vanilla processor with their own per-call instructions and receive a prompt that reflects those instructions rather than the legacy frozen prompt. Both checks can be automated inside the project test suite.

**Acceptance Scenarios**:

1. **Given** the terminal UI launches its default agent after the release, **When** the agent prepares its first system prompt, **Then** the prompt is byte-identical to the pre-release prompt after substituting the user-supplied objective.
2. **Given** every default-path terminal-UI call site that constructs the agentic step processor, **When** the code is inspected, **Then** each call site uses the dedicated terminal-UI processor subclass. No default-path call site relies on the base class's new dynamic default.
3. **Given** a terminal UI orchestrator agent spawns a sub-agent with custom per-call instructions, **When** the sub-agent's system prompt is prepared, **Then** the prompt reflects those per-call instructions (via the dynamic builder or the compat-shim path) and NOT the legacy frozen terminal-UI prompt.
4. **Given** a terminal UI code path needs the enhanced multi-hop variant or the state-driven variant of the processor for a specialized task, **When** that code path constructs the variant directly, **Then** the variant's behavior is unchanged by this release (it still inherits the reasoning loop) and its system prompt correctly reflects the currently-registered tools.
5. **Given** a terminal UI plugin or extension that registers extra tools beyond the default toolset alongside a default-path agent, **When** the legacy prompt is prepared, **Then** a warning is surfaced to the plugin author (via logs) making it clear that the frozen legacy prompt does not advertise their extra tools, so they can switch to the dynamic builder or construct a vanilla processor if they want those tools surfaced.

---

### User Story 3 - Pre-v0.4.2 user code is un-broken with a clear upgrade path (Priority: P2)

A developer whose code pre-dates PromptChain v0.4.2 passes a list of custom agent instructions to the processor constructor. This call used to work, was silently disabled in v0.4.2, and now works again. The developer sees a deprecation warning that points them at the new supported way to achieve the same thing.

**Why this priority**: Silently breaking compatibility once is forgivable; leaving it broken indefinitely is not. The PRD lists this as a named regression class in the field (with a confirming comment in a downstream consumer's code) and resolves it explicitly. This is P2 rather than P1 because affected users already adapted with workarounds; restoring this path unblocks the cleanest forward migration.

**Independent Test**: Construct the agentic step processor with a non-empty list of custom instructions and nothing else. Verify that (a) construction succeeds, (b) a single deprecation warning is emitted that names the replacement approach, and (c) the resulting system prompt contains the supplied instructions as an additional-instructions block. No live language-model call is required.

**Acceptance Scenarios**:

1. **Given** a processor is constructed with only a list of custom instructions (no explicit prompt builder), **When** the processor is built, **Then** a single deprecation warning is emitted naming the new supported approach, construction succeeds, and the generated system prompt contains the supplied instructions verbatim inside an additional-instructions block.
2. **Given** a processor is constructed with both a custom instructions list and an explicit prompt builder, **When** the processor is built, **Then** a clear value error is raised stating that the two configurations are mutually exclusive.
3. **Given** a processor is constructed with only an explicit prompt builder and no custom instructions, **When** the processor is built, **Then** construction succeeds with no deprecation warning.

---

### User Story 4 - Advanced user injects a fully custom prompt strategy (Priority: P3)

A developer with a specialized prompting need (for example a domain-specific instruction scaffold, a non-English prompt, a ReAct-style think/plan/act/observe scaffold, or a research scaffold with different final-answer requirements) passes their own prompt builder to the processor constructor. The processor uses their builder to generate the system prompt and ignores all default builder logic.

**Why this priority**: This is the point of the Strategy pattern — to enable future extension without further library changes. It unlocks the follow-up work listed in the PRD (groundedness checks, per-domain builders, terminal-UI modernization) without requiring another release. It is P3 because the two shipped builders (dynamic default, legacy frozen) already cover the immediate needs.

**Independent Test**: Provide a trivial custom prompt builder that returns a known fixed string for any input. Construct the processor with that builder. Capture the generated system prompt and verify it matches the known fixed string. This validates that the dispatch path to a custom builder is honored end-to-end.

**Acceptance Scenarios**:

1. **Given** a caller provides their own prompt builder, **When** the processor prepares a system prompt, **Then** the output is exactly what the custom builder returns and none of the shipped builders' output is mixed in.
2. **Given** a caller provides both a custom prompt builder and a workflow-pattern hint, **When** the processor is built, **Then** the caller is warned that the workflow-pattern hint is ignored because a custom builder was supplied (the hint only affects the shipped dynamic builder).

---

### Edge Cases

- **Empty tool inventory**: A chain with zero registered tools must produce a prompt that clearly states no tools are available, not a prompt that silently omits the tool block or lists phantom tools.
- **Very large tool inventory**: A chain with many registered tools (for example 50+) could produce a prompt that exceeds small-context model limits. The builder surface must expose a token-estimate method so callers can check and fail fast before making a model call.
- **Tool-schema changes at runtime**: If a tool's description is updated between two prompt generations on the same processor instance, the second prompt must reflect the new description. Prompts are generated from live state at call time.
- **Terminal-UI plugin registers extra tools**: The frozen legacy prompt deliberately ignores the tools list. Plugin authors whose extra tools silently do not appear must receive a runtime warning so they can migrate to the dynamic builder.
- **Downstream subclasses**: Existing subclasses of the agentic step processor (the enhanced multi-hop variant, and a state-driven agent variant) both call the parent constructor and must pick up the fix automatically, without needing their own code changes.
- **Conflicting configuration at construction**: Supplying both the legacy compat-shim instruction list and a new explicit prompt builder is ambiguous and must be rejected with a clear error — not silently resolved with implicit precedence.
- **Multiple processor instances in the same process**: Switching between library-consumer and terminal-UI usage inside the same Python process (for example in a notebook) must not leak state between them. No process-global default mutation is acceptable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The library MUST expose a new public module for prompt builders that contains (a) a builder protocol, (b) a dynamic builder that renders actually-registered tools, and (c) a legacy builder that reproduces the pre-release terminal-UI prompt verbatim.
- **FR-002**: The builder protocol MUST define a method that accepts an objective, a tool-schema list, and an optional prior-context string and returns the final system-prompt string.
- **FR-003**: The builder protocol MUST define a method that accepts an objective and a tool-schema list and returns an estimated token count for the prompt, so callers can fail fast on oversize prompts.
- **FR-004**: The dynamic builder MUST render the system prompt from the tools actually present on the chain at call time, including each tool's name and description. It MUST NOT embed or advertise any tool name that is not present in the supplied tool-schema list.
- **FR-005**: The dynamic builder MUST support an optional extra-instructions list that, when provided, is rendered as a dedicated additional-instructions block in the generated prompt.
- **FR-006**: The dynamic builder MUST support a workflow-pattern hint with at least two named modes (a standard minimal mode and a ReAct-style think/plan/act/observe mode). In ReAct mode when no task-list-writer tool is registered, the builder MUST emit a runtime warning and fall back to a safe alternative block rather than silently producing an inconsistent prompt.
- **FR-007**: The dynamic builder's output MUST end with a final-answer-requirements block that instructs the model to include full tool-result content in its final answer rather than summarizing it away.
- **FR-008**: The legacy builder MUST return the v0.5.0 hardcoded terminal-UI prompt verbatim (with only the objective substituted) regardless of the tool-schema list supplied to it. When the tool-schema list differs from the expected default terminal-UI inventory, the legacy builder MUST surface a runtime warning naming the discrepancy.
- **FR-009**: The agentic step processor constructor MUST accept a new optional argument that takes an instance of any object conforming to the builder protocol.
- **FR-010**: The agentic step processor constructor MUST restore the previously removed custom-instructions argument as a backwards-compatibility shim. When supplied, it MUST emit a single deprecation warning naming the replacement approach and internally construct a dynamic builder pre-configured with those instructions.
- **FR-011**: When both the compat-shim custom-instructions argument and the new explicit builder argument are supplied, the constructor MUST raise a clear value error stating they are mutually exclusive.
- **FR-012**: When only the explicit builder argument is supplied, the constructor MUST use that builder for all subsequent prompt generation and MUST warn the caller if they also passed a workflow-pattern hint (since that hint only affects the shipped dynamic builder).
- **FR-013**: When neither argument is supplied, the constructor MUST default to a dynamic builder — not to the legacy frozen prompt.
- **FR-014**: The agentic step processor's reasoning-loop code MUST generate its system prompt by calling the selected builder's generate method with the current objective, the currently-registered tool schemas, and any prior-step scratchpad text. The hardcoded f-string at `agentic_step_processor.py:909-1013` MUST be removed.
- **FR-015**: Every existing default-path call site inside the shipped terminal UI and its agentic chat components that instantiates the agentic step processor MUST be updated to use a dedicated terminal-UI processor subclass that bakes in the legacy prompt. No terminal-UI default-path call site may rely on the base class's new dynamic default.
- **FR-015a**: The terminal UI MUST retain the ability to construct vanilla agentic step processors and other processor variants (enhanced multi-hop, state-driven, third-party custom subclasses) on demand — for example when an orchestrator agent spawns a sub-agent whose system prompt must reflect the sub-agent's per-call instructions rather than the frozen terminal-UI default. The terminal UI's processor class is configurable per call site, not hardcoded globally.
- **FR-016**: The release MUST introduce a dedicated subclass (for example `TUIAgenticStepProcessor`) that bakes in the legacy prompt builder and forwards all other constructor arguments. This subclass is the terminal UI's default-path convenience and centralizes the opt-in so future terminal-UI code cannot accidentally drift off the legacy prompt.
- **FR-017**: The release MUST NOT introduce any process-global default for prompt selection — no class-level mutable defaults, no environment-variable switches, and no auto-detection heuristics based on tool-name pattern matching.
- **FR-018**: The module boundaries MUST avoid circular imports: the builder protocol module must depend on nothing else inside the library.
- **FR-019**: Tests that previously string-matched against fragments of the old hardcoded prompt MUST be updated to (a) assert against the frozen legacy snapshot when validating terminal-UI behavior and (b) assert against the dynamic-render behavior when validating library-consumer behavior.
- **FR-020**: A snapshot test MUST protect the legacy builder's output against incidental drift. The snapshot represents the frozen v0.5.0 prompt; any change to the snapshot must be a conscious, reviewed change.
- **FR-021**: Tests that exercise the compat-shim custom-instructions path MUST assert that a deprecation warning is emitted exactly once per construction.
- **FR-022**: The project changelog MUST document the release with a breaking-change entry naming the default-prompt behavior change and naming the legacy opt-in for consumers who need the old default.
- **FR-023**: The two existing subclasses of the agentic step processor (the enhanced multi-hop variant and the state-driven agent variant) MUST continue to work unmodified and MUST pick up the fix automatically by virtue of calling the updated parent constructor.
- **FR-024**: The reproduction failure described in the PRD (a downstream consumer registering its tools and receiving an ungrounded off-topic response with zero tool calls) MUST be resolved: running the same reproduction against the fixed release MUST produce a response that contains content grounded in the registered tools and a non-zero number of retrieval or tool calls.

### Key Entities *(include if feature involves data)*

- **Prompt Builder**: A strategy object that knows how to generate the system prompt for an agentic step processor. Attributes: a generate method, a token-estimate method. Relationships: used by the agentic step processor during each reasoning step; receives the live tool-schema list from the chain at call time.
- **Dynamic Builder**: The new default implementation of the prompt-builder strategy. Renders the live tool-schema list as the advertised tool inventory. Optional configuration: extra instructions list, workflow-pattern hint, final-answer-hint flag.
- **Legacy Builder**: The frozen pre-release implementation of the prompt-builder strategy. Returns the v0.5.0 hardcoded terminal-UI prompt verbatim. Used by every terminal-UI call site to preserve shipped behavior.
- **Tool Schema**: A single entry in the tool-schema list. Represents one tool registered on the chain. Required attributes visible to the builder: a name and a human-readable description.
- **Agentic Step Processor**: The existing reasoning-loop orchestrator. After this release, it holds a reference to a prompt builder and delegates all system-prompt generation to it. Its internal reasoning-loop behavior is otherwise unchanged.
- **Compat Shim Parameter**: The restored custom-instructions argument on the processor constructor. Internally maps to constructing a dynamic builder pre-configured with those instructions and emits a deprecation warning.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a library consumer registering N custom tools on a chain and instantiating the default agentic step processor with no prompt configuration, the generated system prompt lists exactly those N tools and zero terminal-UI-specific tool names. Verified by static inspection of the generated string for N in {0, 1, 4, 10}.
- **SC-002**: The v0.5.0 terminal-UI system prompt is reproduced byte-for-byte (modulo the dynamic objective substitution) by the legacy builder, and every terminal-UI call site in the shipped code selects the legacy builder explicitly. Verified by a frozen-snapshot test and by a grep across the terminal-UI source tree showing zero implicit-default call sites.
- **SC-003**: Re-running the PRD's named reproduction (a downstream consumer with four registered tools asking a multi-hop domain question that currently returns a response with zero retrieval calls) against the fixed release produces a response with at least one retrieval or tool call and with content grounded in the registered tools' outputs. Verified by the accumulated-contexts count being greater than zero and by a human or automated domain-relevance check.
- **SC-004**: A developer whose pre-v0.4.2 code passed a custom instructions list can install the release and run their code unchanged. Construction succeeds, a single deprecation warning is emitted, and the resulting system prompt contains their instructions verbatim. Verified by an integration test.
- **SC-005**: All pre-existing project tests continue to pass after the release. Tests that string-matched on the old hardcoded prompt are updated to match the new behavior and remain green. Verified by a clean CI run on the release branch.
- **SC-006**: The two existing subclasses of the agentic step processor receive the fix with zero source changes. Verified by instantiating each subclass with a chain of custom tools and confirming those tools appear in the generated prompt.
- **SC-007**: Token cost of the dynamic builder's output for a small workload (objective plus three tools, standard workflow mode) is no more than roughly 15 lines of prompt text, keeping the default path friendly to small-context models. Verified by measuring line count of the generated prompt against a fixed fixture.
- **SC-008**: The release introduces zero process-global mutable state for prompt selection. Verified by inspection of the new module and of the processor constructor — no class-level mutable defaults, no environment-variable reads for prompt routing, no tool-name heuristic code path.

## Assumptions

- The existing OpenAI-format tool-schema shape (a list of dicts each carrying at minimum a name and description) remains the interchange format between chain and builder; no schema redesign is in scope.
- The reasoning loop's non-prompt behavior (step counting, tool dispatch, success reporting) remains unchanged in this release. Groundedness checks and success-reporting changes are explicitly out of scope and tracked as separate follow-up work.
- The terminal UI's default tool inventory is stable enough that the legacy builder can freeze it as a literal string. Any new terminal-UI-side tools registered before this release are already represented in the frozen inventory.
- Consumers who pass their own builder accept that the shipped dynamic builder's configuration hints (for example the workflow-pattern hint) do not apply to their custom builder; the constructor surfaces this as a warning.
- A single deprecation cycle (one release emitting a warning, no hard removal yet) is the right pacing for the compat-shim parameter; actual removal is a future decision tracked separately.
