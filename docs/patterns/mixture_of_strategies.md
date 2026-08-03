# Mixture of Strategies — exploring a STATIC "poor man's Mixture of Experts"

> **Scope:** an EXPERIMENT in the *static* implementation of a few specific strategies (Tool-Scribe, task
> decomposition). PromptChain does both static AND agentic — both first-class; this doc is only about the
> static exploration of these strategies, not a stance on PromptChain overall.

> Framing coined 2026-06-30. Companion: PRD `tool_scribe_format_routing_2026-06-30`, lab at
> `/media/gyasis/Drive 2/workbench/` (TOOL-SCRIBE.md, TOOL-SCRIBE-FINDINGS.md, GOVERNOR-USAGE.md).

## The idea

PromptChain is a **"poor man's Mixture of Experts"**: compose small (or surgically-targeted strong)
models — each doing **ONE thing well** — to reach foundational-model capability from **weak/local models**
(Gemma 4, GPT-OSS, ornith, ministral) without the foundational model.

PromptChain is **not "just an LLM caller."** It is also a **pipeline**, a **machine-learning tool**, and a
**function/tool on its own** (it nests as a step in another chain, or is exposed as a tool/MCP).

## Scope: this experiment explores the STATIC form of these strategies

PromptChain does **two** things — **static chains AND agentic chains** — and **both are first-class.** This
is NOT a claim that static is better or that agentic is "off the table"; agentic chains are the right tool
for plenty of work. **This particular experiment** asks one narrow question:

> *Can the poor-man's-MoE strategies below (Tool-Scribe, task decomposition) be implemented as **STATIC**
> chains — a fixed pipeline with a few deterministic forks — instead of agentically?*

That's it. We're *testing whether* these specific strategies can be done statically. The two modes:

| | **AGENTIC** (a valid PromptChain mode; not what THIS experiment uses) | **STATIC** (what THIS experiment tries) |
|---|---|---|
| Flow control | an **LLM decides** what to do next | a **fixed, developer-wired pipeline**, left-to-right |
| Mechanism | `AgentChain` router · `AgenticStepProcessor` · router-driven `utils/strategies/` (`single_dispatch` / `static_plan` / `dynamic_decomposition`) | `PromptChain(models=[...], instructions=[...])` — ordered string/function steps |
| Decisions | the agent reasons about the path (non-deterministic) | a FEW **deterministic function FORKS** — **boolean / yes-no / 0-1 / one-of-N** |
| Determinism | non-deterministic | deterministic shape, predictable, cheap |

**For this experiment** we borrow the agentic strategies only as high-level *concepts* (dispatch,
decomposition, planning) and try to translate them into a static chain — to see if the static form holds up.

## The fork decision-makers (this is where constrained decoding comes in)

A static chain branches via a few **deterministic forks**. A fork is either:
1. **a pure Python function** — e.g. `is_json_valid(call) -> bool`; or
2. **a CONSTRAINED LLM call** that can ONLY return a bounded label — `yes`/`no`, `0`/`1`, or one-of-N —
   via `guidance.select()` / Ollama `format` enum / vLLM `guided_choice`. **One token, bounded, no reasoning.**

**This is exactly why we researched Guidance / constrained decoding** (see TOOL-SCRIBE-FINDINGS F-7):
the static fork needs a *boolean / multiple-choice* output, GUARANTEED — not an agent's free-form decision.
A fork is wired with a `chainbreaker` (TERMINATES the chain — for a final decision) or a nested
function-step / sub-PromptChain (CONTINUES the chain — skip-but-continue, mid-pipeline) — see F-12.

## The two guiding principles (optimize for BOTH)

1. **Accuracy** — composed reliability approaching a foundational model.
2. **Token savings & cost** — bounded, targeted input per expert; **boolean forks instead of agentic reasoning.**

Static + bounded forks = deterministic (accuracy) + minimal tokens (cost). Every strategy is judged on both.

## The experts (each a static step, gated by boolean forks)

- **Tool-Scribe** — small tool-call-reliable model emits/repairs a step's tool-call JSON (free/local).
- **Reverse Power** — strong model repairs a weak driver's call on **targeted bounded input** (~280 tok:
  malformed call + schema, not the driver's context). Cheap: format repair is a context-free local transform.
- **Fixer spectrum:** deterministic `repair()` → Tool-Scribe → Reverse Power, escalated by a boolean fork
  (`json_valid?`).
- **Decomposition (planner) — STATIC form:** translate `static_plan`/`dynamic_decomposition` CONCEPTS into a
  fixed planner step + boolean forks (e.g. `more_steps_needed? yes/no`), NOT the agentic router.
- **Orchestration-load fork:** a cheap classifier (WideMLP/fastText <4ms, or a constrained one-of-N call)
  that forks the pipeline (format-light vs needs-decomposition) — a static fork, not an agent.

## Canonical shape (the repair-only Tool-Scribe, fully static)
```
PromptChain(
  models=[PRIMARY, SCRIBE],
  instructions=[ primary_emit_tool_call, scribe_repair ],
  chainbreakers=[ json_gate ],   # boolean fork: valid -> break(pass call) ; invalid -> fall through to scribe
)
```
No agent decides anything — a fixed 2-step pipeline with one boolean fork. That is the whole pattern.
