# Quickstart — Agentic Prompt Builder Decoupling

**Feature**: 011-agentic-prompt-builder
**Audience**: (1) library consumers migrating to v0.6.0, (2) PromptChain TUI maintainers adding or auditing `AgenticStepProcessor` call sites.

This quickstart is deliberately short and recipe-oriented. The authoritative details live in `spec.md`, `data-model.md`, and `contracts/prompt_builder_protocol.md`.

---

## Part 1 — Library consumer recipe

You are building a RAG service, a custom research agent, a domain-specific assistant, etc. You import PromptChain, register your tools on a chain, and construct an agent.

### Before v0.6.0 (broken-by-default state)

```python
from promptchain import ChainBuilder
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

chain = ChainBuilder().model("openai/gpt-4o-mini").build()
chain.add_tools([my_retrieval_tool, my_ranking_tool, my_summarize_tool, my_write_tool])

step = AgenticStepProcessor(objective="Answer the user's multi-hop question.")
result = chain.run(step)   # ← silently ships a prompt advertising tools you never registered
```

### v0.6.0 and later (fixed-by-default)

**No code change required.** The same code now produces a prompt that lists your four tools and nothing else. That is the fix.

### If you want ReAct scaffolding

```python
from promptchain.prompts import DynamicPromptGenerator
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

step = AgenticStepProcessor(
    objective="Answer the user's multi-hop question.",
    prompt_builder=DynamicPromptGenerator(workflow_pattern="react"),
)
```

If your chain has no task-list-writer tool but you picked `workflow_pattern="react"`, the builder logs a warning and falls back to a minimal thought/action scaffold.

### If your pre-v0.4.2 code used `instructions=`

```python
step = AgenticStepProcessor(
    objective="...",
    instructions=["Always cite sources.", "Prefer structured output."],
)
```

This works again in v0.6.0. A single `DeprecationWarning` is emitted per construction pointing you at the migration:

```python
from promptchain.prompts import DynamicPromptGenerator

step = AgenticStepProcessor(
    objective="...",
    prompt_builder=DynamicPromptGenerator(
        extra_instructions=["Always cite sources.", "Prefer structured output."],
    ),
)
```

### If you want a token estimate before calling the model

```python
builder = DynamicPromptGenerator()
estimate = builder.get_token_estimate(
    objective="...",
    tools=chain.tools,   # the OpenAI-format tool schema list
)
if estimate > 6000:
    raise RuntimeError(f"Prompt is {estimate} tokens — too large for this model.")
```

---

## Part 2 — TUI maintainer recipe

You are adding a new call site to the PromptChain terminal UI, or auditing an existing one.

### The default path — use `TUIAgenticStepProcessor`

```python
from promptchain.cli.tui_processor import TUIAgenticStepProcessor

step = TUIAgenticStepProcessor(
    objective="…",
    # …all existing kwargs work — max_internal_steps, model_config, etc.
)
```

This bakes in `LegacyTUIPromptGenerator()` so the TUI's agent continues to ship the v0.5.0 prompt byte-for-byte. **Do not** pass `prompt_builder=` or `instructions=` to `TUIAgenticStepProcessor` — it rejects both with a `TypeError` pointing you at the base class instead.

### The sub-agent path — use vanilla `AgenticStepProcessor`

When an orchestrator agent inside the TUI launches a sub-agent whose system prompt must reflect its own per-call instructions, skip the TUI subclass and use the base class directly:

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

sub_agent = AgenticStepProcessor(
    objective=sub_task.description,
    instructions=sub_task.custom_instructions,   # via the deprecation shim — still works
)
# OR, preferred for new code:
from promptchain.prompts import DynamicPromptGenerator

sub_agent = AgenticStepProcessor(
    objective=sub_task.description,
    prompt_builder=DynamicPromptGenerator(
        extra_instructions=sub_task.custom_instructions,
        workflow_pattern=sub_task.workflow,
    ),
)
```

The sub-agent gets a prompt reflecting its own instructions and the chain's currently-registered tools — not the frozen TUI default.

### The specialized-variant path — use enhanced / state-driven / custom subclass

```python
from promptchain.utils.enhanced_agentic_step_processor import EnhancedAgenticStepProcessor
from promptchain.utils.strategies.state_agent import StateAgent

enhanced = EnhancedAgenticStepProcessor(objective="…", enable_rag_verification=True)
state_driven = StateAgent(objective="…", states=[...])
```

Both inherit the fix automatically via `super().__init__` — their system prompts reflect the registered tools.

### Call-site compliance check

Before opening a PR that adds a TUI call site, run:

```bash
grep -rnE "AgenticStepProcessor\s*\(" promptchain/cli/ agentic_chat/ \
  | grep -v "TUIAgenticStepProcessor" \
  | grep -v "# sub-agent: "       # whitelist marker for intentional base-class use
```

The output should either be empty OR every line should have the `# sub-agent:` marker comment explaining why the vanilla base class is correct for that site.

---

## Part 3 — Test-first implementation order

Implementation proceeds red → green → refactor. The order is:

1. **Write test 1** — `tests/test_prompt_builders.py::test_legacy_snapshot_byte_identical`. Fails (module doesn't exist yet).
2. **Freeze the snapshot** — copy the pre-change `agentic_step_processor.py:909-1013` literal into `tests/fixtures/legacy_tui_prompt.snapshot.txt`.
3. **Implement** `promptchain/prompts/base.py` (Protocol) and `promptchain/prompts/legacy_tui.py`. Test 1 turns green.
4. **Write tests 5, 6, 7** (dynamic render / empty tools / ReAct warning).
5. **Implement** `promptchain/prompts/dynamic.py`. Tests 5-7 turn green.
6. **Write tests 8, 9, 10** (dispatch behavior on `AgenticStepProcessor.__init__`).
7. **Modify** `AgenticStepProcessor.__init__` and `run()` per the contract. Tests 8-10 turn green. Existing processor tests should still pass.
8. **Write tests 11, 12** (TUI subclass rejection of `prompt_builder=` / `instructions=`).
9. **Implement** `promptchain/cli/tui_processor.py`. Tests 11-12 turn green.
10. **Write tests 13, 14** (subclass inheritance — enhanced, state-agent).
11. **Verify** enhanced and state-agent subclasses pass with zero source changes. Tests 13-14 turn green.
12. **Migrate TUI call sites** — 10 call sites in total switch from `AgenticStepProcessor(...)` to `TUIAgenticStepProcessor(...)` except for any that are specifically sub-agent-spawning sites (add `# sub-agent:` whitelist comment on those).
13. **Write test 15** (call-site compliance grep).
14. **Write test 16** (library-consumer e2e PRD reproduction).
15. **Update three existing tests** that string-matched the old hardcoded prompt (`tests/test_tao_loop.py`, `tests/test_verification_integration.py`, `tests/cli/integration/test_agentic_reasoning.py`).
16. **Update CHANGELOG.md** with the v0.6.0 entry (Added / Restored / Changed BREAKING).

---

## Part 4 — FAQ

### Why does `TUIAgenticStepProcessor` reject `prompt_builder=`?

Because its entire purpose is to bake in `LegacyTUIPromptGenerator()`. If a caller needs a different builder, they should reach for the base class directly — silently overriding the baked-in builder would defeat the subclass's whole reason to exist.

### My TUI sub-agent spawning site now shows up in the grep check. Do I whitelist it?

Yes, with a short comment. Example:

```python
# sub-agent: orchestrator spawns sub-agents with per-call instructions; vanilla processor is intentional
sub_agent = AgenticStepProcessor(objective=task.desc, instructions=task.instructions)
```

The comment tells future maintainers (and the compliance grep) why the base class is correct there.

### Can I mix an `EnhancedAgenticStepProcessor` with `LegacyTUIPromptGenerator`?

Yes — pass it explicitly: `EnhancedAgenticStepProcessor(objective=..., prompt_builder=LegacyTUIPromptGenerator())`. The constructor dispatch honors the `prompt_builder=` argument regardless of which subclass you are constructing.

### When does the `DeprecationWarning` for `instructions=` fire?

Once per `AgenticStepProcessor.__init__` call that supplies `instructions=`. If you construct 100 processors in a loop, you get 100 warnings by default. Suppress in tests with `pytest.warns(DeprecationWarning)` or in production with `warnings.simplefilter("ignore", DeprecationWarning)` scoped tightly.

### Will `instructions=` be removed in v0.7.0?

No commitment in v0.6.0. The warning is the signal; the removal timing is a separate future decision.
