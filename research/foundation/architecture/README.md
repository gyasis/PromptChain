# North Star — Invert the harness/model relationship

> **Every other coding harness makes the MODEL fit the HARNESS.
> PromptChain makes the HARNESS fit the MODEL.**

## The inversion

- **Today (everyone else):** the harness is built around ONE model (e.g. Claude Code around Claude).
  There is one fixed system prompt, one loop, one structure — and every model must conform to it. Plug in
  a weaker/local model and it breaks: the 8K-token Claude-idiom prompt poisons its context, the loop
  outlives its turn budget, the XML/tool style it can't follow makes it fail.
- **PromptChain (this redesign):** the harness **measures the model and reshapes itself around it** —
  prompt content, token budget, format dialect, loop/compression cadence, and delegation all adapt to the
  specific model in the slot. The model never has to fit us; we fit the model.

## The two layers — a shared base + a per-model split

- **The BASE follows everyone else's plan.** A static, model-AGNOSTIC foundation built from the
  industry's proven best practices (the 20 foundations). This is our **floor** — it makes PromptChain at
  least as good as any single-model harness. Written portably (no Claude-only idioms) so *any* model can
  run it.
- **The DYNAMIC layer splits around the model.** On top of that shared base, our dynamic prompting adapts
  to the specific model — tier · family-adapter · token-budget · on-demand modules · loop & compression
  cadence (all from the Model Profiler). This is our **edge** — the part nobody else has, because nobody
  else is multi-model by design.

> **Base = parity with the field. Dynamic split = the inversion (the harness fits the model).**

## Why PromptChain *can* do this (the unique primitives)

A model is chosen **per agent / per step** (`AgentChain.model_name`, two-tier routing); the system prompt
is produced by a **pluggable builder** (`BasePromptBuilder`); loops are a **first-class object**
(`ExternalLoop`); delegation is built in (`OrchestratorSupervisor`, `delegate_task`); transport is
normalized by LiteLLM. Most harnesses are single-model and hardcoded — PromptChain is multi-model and
composable by design. The inversion is *natural* here, not a bolt-on.

## How every piece serves the thesis

| Component | How it makes the harness fit the model |
|---|---|
| **Model Profiler** (E) | Measures the model (10-cast probe + telemetry) → emits the config the harness fits to |
| **Token budget** (A) | Sizes the prompt to the model's *real* context — 300–1k for small, more for big |
| **Prompt tiering** (B) | Picks prompt richness to the model's capability (core vs extended) |
| **Family adapter** (C) | Speaks the model's *dialect* (XML for Claude, markdown for others, reasoning-trigger…) |
| **Weak-model longevity** (D) | Document-&-Clear cadence + standalone + escalation tuned to the model's degradation point |
| **Modular dynamic prompting** | Assembles only what *this model + task* needs — nothing wasted |
| **Subagency / big-brother** | Routes each task to the right-sized model (strong plans, weak executes) |
| **Grounded tool inventory** | Tells the model what *it* actually has — never a tool it can't call |
| **The 20 foundations** | The model-agnostic *content* every model gets, dialect-adapted and budget-trimmed |

## The tagline

**Right-size the prompt to the model · right-size the model to the task · let the harness adapt, not the
model.** Everything in `00-design-constraints.md` + `01-model-profiler.md` + the foundation prompt is in
service of this one inversion.
