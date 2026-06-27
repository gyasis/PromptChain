# Design Constraints — Token Economy + Heterogeneous-Model Architecture

Two user-set constraints (2026-06-27) that govern the whole foundation-prompt + architecture design.
These sit ABOVE the 20 foundations.

## Constraint A — Token economy ("kitchen sink, not the SIZE of the kitchen sink")

PromptChain is NOT Claude — no 1M-token context. A 4–8K-token static system prompt crashes a small/
local model after a couple of turns. So **token economy = Foundation #0**; capture all 20 foundations'
VALUE at a tiny COST.

1. **Hard budget on the always-on core:** target ~600–1,200 tokens (validate w/ compression research).
2. **Dense encoding:** one terse imperative line per rule; XML/markdown; no prose padding; **no examples
   in the always-on base** (examples = optional module).
3. **Modularity = the compression lever:** core = universal foundations only; persona/domain/verbose-
   guidance/examples/loop = **on-demand modules** attached only when the turn needs them.
4. **Measure + cap:** use generators' `get_token_estimate()` to budget the assembled prompt; drop optional
   modules if over cap. Enforced, not hoped.
5. **Separate the two bloats:** system prompt stays lean; conversation/tool-output bloat handled by the
   EXISTING `ContextDistiller` (@70%) + `HistorySummarizer`. Don't conflate.
6. **Dense structure resolves the weak-model tension:** weak models need structure (costs tokens) but
   have small budgets → use high-signal-per-token XML + one-line rules, tiered/modular.

## Constraint B — Heterogeneous models ("different models for different things" + a "big brother")

PromptChain's differentiator: a model is chosen **per agent / per step** (`AgentChain` agents each carry
`model_name`; `AgenticStepProcessor` has `enable_two_tier_routing` + `fallback_model`). Most coding agents
are single-model — this is genuinely unique. Lean into it:

1. **Prompt tiering by model strength.** The foundation generator becomes **model-aware**:
   `generate(objective, tools, model=...)` → STRONG model gets the **extended** foundation (richer, more
   examples/modules); WEAK/small model gets the **core** (lean) foundation. Native, because PromptChain
   knows each agent's model.
2. **Big-brother / planner-executor split.** A STRONG "big brother" model (orchestrator/planner) gets the
   FULL rich prompt and does the expensive cognition — **planning, task breakdown, review**. It delegates
   **atomic, scoped tasks** to WEAK/cheap executor sub-agents that get **tiny task-scoped prompts** (just
   the atomic task + core foundations) — narrow, structured execution where small models are reliable.
3. **This RESOLVES the token tension:** the big prompt only ever goes on the model that can afford it
   (strong, usually bigger context). Weak models never see a big prompt — only tiny task slices.
4. **Reuse:** subagency (`OrchestratorSupervisor` → workers), two-tier routing (`AgenticStepProcessor`),
   Amp's "Oracle" pattern (strong advisor for planning/review), per-agent `model_name`. No nested spawning.

## Constraint C — Cross-model normalization ("don't base everything on one model's idiom")

The TUI runs MANY model families (Anthropic, OpenAI incl. reasoning, Gemini, local Qwen/Gemma/DeepSeek/
Devstral via ollama + ollama-cloud). Models behave DIFFERENTLY on the same prompt: formatting prefs
(XML vs markdown vs JSON), system-vs-user role placement, few-shot sensitivity, tool-call format,
reasoning vs non-reasoning. **Risk:** if we author the foundation in Claude-Sonnet-4.6's idiom (heavy
XML, Anthropic reminder tags), other models may underperform or break.

1. **LiteLLM normalizes TRANSPORT, not CONTENT.** It translates message roles, tool-call schemas, and
   `response_format` across providers — but it does NOT rewrite our prompt's STYLE. Content portability
   is on us.
2. **Write the base model-agnostically.** Widely-supported structure (clear headers, one-line rules);
   avoid family-specific tags/idioms in the always-on core.
3. **Per-family ADAPTER module (dynamic).** A small, model-keyed module adjusts FORMAT only — e.g. XML
   for Claude, markdown for GPT/Gemini, explicit reasoning-trigger for reasoning models, tool-call style
   hints — attached at `generate()` based on the agent's model. Just another dynamic add-on.
4. **DSPy is the portability paradigm.** One Signature compiles to model-specific prompts via Adapters;
   our base = the intent, the adapter = the model-specific rendering. Reinforces the modular design.

## Constraint D — Weak models run STANDALONE, with a turn budget + "Document & Clear" reset

Weak ≠ tiny — think a capable **12B (gemma-12b class)**, just not top-of-the-line. We must NOT always
require a strong/foundation model: a weak model should run a **standalone** session for a scoped task.
But weak models degrade from context accumulation and fail after **~6–8 turns**. Goal: keep a weak model
usable for **≥10 task-turns**, then compress/reset. (Grounded by compression research, 61 sources.)

1. **Standalone is viable.** The "Pi" agent proves a **<1,000-token base + 4 tools** drives autonomous
   coding. A weak model with a lean core prompt is first-class, not a fallback.
2. **Turn longevity via "Document & Clear"** (the user's "route loop"): at **~60% context**, the agent
   dumps plan/decisions/progress to disk (`PROGRESS.md` / `todo.md`), the session **clears**, and resumes
   from the doc — swapping ~25k noisy tokens for a ~500-token state doc, resetting reasoning fidelity.
   This is the **Ralph fresh-context loop** applied to weak-model longevity. ≥10 turns, then reset.
3. **Document-&-Clear is primary; lossy auto-compaction is the fallback** — summarization erases "why X
   failed" → infinite-loop relapse. Disk-state is durable + auditable.
4. **Externalize structure to the harness** — don't spend a weak model's tokens forcing it to be a state
   machine; `ExternalLoop` + format adapters own control flow; the model does free-form reasoning + simple
   outputs. ("Free-form reasoning for safe tasks, structured execution for long-horizon pipelines.")
5. **Escalation is OPTIONAL, not mandatory** — only if the weak model stalls (no progress in N turns) does
   it escalate to the big-brother planner. Default: the weak model finishes its scoped task itself.

### Concrete token budgets (32k local model — from research)
- **Static base prompt: 300–1,000 tokens** (hard cap ~1,500; Pi ≈ 300 words).
- tools ~2k · workspace code ~15k · retrieval ~5k · history ~5k · output headroom ~4k.
- Compress the base with **LLMLingua-2** / terse-imperative / DSPy token-penalty; **XML boundaries +
  markdown lists** (avoid JSON "brace tax"); one-rule-per-line; reference-don't-repeat (progressive
  disclosure); **lazy tool-loading** (metadata only, inject schema on demand).

## Net architectural implications (feeds Phase 4 + 5)

- The foundation generator is **model-aware (strength + family) + budget-capped**: `(objective, tools,
  model, budget) →` pick TIER (strength, Constraint B) + ADAPTER (family, Constraint C) → attach only
  needed modules → assemble → measure → **trim to a 300–1,000-token base** (Constraint A). Three
  model-aware axes.
- The **loop** owns weak-model longevity: `ExternalLoop` runs the turn, watches context %, triggers
  **Document & Clear** at ~60% (state→disk, fresh context, resume), targets ≥10 turns, and escalates to
  big-brother only on stall (Constraint D). Reuse `ContextDistiller` (retune ~60%) + a disk state file.
- A weak model is a **first-class standalone executor** with a lean core prompt — not always a delegate.
- The "big brother" planner runs the rich prompt + loop-until-completion; it breaks the goal into atomic
  tasks (the loop's worklist) and dispatches them to lean executor sub-agents.
- Token economy + heterogeneous models are two sides of one idea: **right-size the prompt to the model,
  and right-size the model to the task.**
