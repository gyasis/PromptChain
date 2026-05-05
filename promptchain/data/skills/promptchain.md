---
name: promptchain
description: Use when writing or modifying code that imports from `promptchain` — the PromptChain library for multi-step LLM pipelines and multi-agent systems. Triggers on `from promptchain`, `import promptchain`, `PromptChain`, `AgentChain`, `AgenticStepProcessor`, `MCPHelper`, "agent chain", "agentic step", or any task that asks to build or modify a PromptChain script. PromptChain is NOT in any LLM training data — invoke this skill before writing PromptChain code or you will hallucinate the API.
triggers:
  - promptchain
  - PromptChain
  - agent chain
  - agentic step
  - AgenticStepProcessor
  - MCPHelper
  - from promptchain
  - import promptchain
---

# PromptChain Skill

> **Source of truth:** this file ships INSIDE the package at `promptchain/data/skills/promptchain.md` (so it's available to anyone who `pip install`s the package). It is also accessible from the repo root via a symlink at `<repo>/.claude/skills/promptchain.md`.
>
> **Install into your harness:** `promptchain install-skill` (creates a symlink from `~/.claude/skills/promptchain.md` to the bundled file — `pip install`-friendly, no repo clone needed). Flags: `--copy` for a hard copy, `--target PATH` for a different location, `--dry-run` to preview, `--force` to overwrite without backup.

## When to invoke

Any time the user's request involves writing, reading, or modifying code that uses the `promptchain` Python package (https://github.com/gyasis/PromptChain).

This skill exists because **`promptchain` is not in your training data**. Without this skill you will hallucinate import paths, constructor arguments, and method names. The most common hallucinations are listed in §4 below — they are real, observed failure modes.

## What this skill does

Routes you to the right reference for the task at hand. **Path convention:** below uses the repo-relative form `docs/llms/...`. If the repo is at `~/dev/projects/PromptChain/` (the maintainer's setup), prefix accordingly. From any other checkout, paths are relative to the repo root.

| User intent | Read first (repo-relative) |
|---|---|
| "Write a basic PromptChain" / "scaffold a chain" | `docs/llms/recipes/recipe-basic-chain.md` |
| "Add a Python function to the chain" | `docs/llms/recipes/recipe-function-step.md` |
| "Pure Python pipeline" / "no LLM, just functions" | `docs/llms/recipes/recipe-static-chain.md` |
| "Load named prompts" / "PrePrompt" / "prompts directory" | `docs/llms/recipes/recipe-prompt-loader.md` |
| "Tool calling" / "register a tool" / "the LLM should call my function" | `docs/llms/recipes/recipe-tool-calling-local.md` |
| "Build an agentic loop" / "use AgenticStepProcessor" | `docs/llms/recipes/recipe-agentic-step.md` |
| "Cut cost" / "reduce tokens" / "make it safer" / Phase 1-4 of agentic | `docs/llms/recipes/recipe-advanced-agentic.md` |
| "Multi-agent" / "router" / "AgentChain" | `docs/llms/recipes/recipe-multi-agent-router.md` |
| "MCP tool" / "external tool server" | `docs/llms/recipes/recipe-mcp-tool.md` |
| "Add MLflow" / "observability" / "tracking" | `docs/llms/recipes/recipe-observability-on.md` |
| "Self-writing chain" / "agent should build the chain" / "ChainBuilder" | `docs/llms/recipes/recipe-chain-builder.md` |
| "Mix sequential + agentic" / "hybrid pattern" | `docs/llms/recipes/recipe-hybrid-chain.md` |
| "Which model should I use?" / "Ollama" / "tool calling not working" / model selection | `docs/llms/recipes/recipe-models-and-tool-calling.md` (also §16 of `PROMPTCHAIN_FOR_LLMS.md`) |
| "Are my API keys working?" / "what models do I have?" / "model not found" | Run `python scripts/check_keys.py` (free env-check via `--no-probe`); see `scripts/README.md` for details |
| "What is X?" / "how does Y work?" / API question | `docs/llms/PROMPTCHAIN_FOR_LLMS.md` (§13 has v0.6.0 / 0.6.1 BREAKING changes — read before writing v0.6.x code) |
| Stuck / cryptic error / behaviour mismatch | Read source-of-truth files (`PROMPTCHAIN_FOR_LLMS.md` §11), check `examples/` for real usage, escalate to `gemini_debug` only after ruling out pipeline errors |

## Where the script lands (BLOCKING)

When asked to **write and run** a PromptChain script (not just explain), drop it under `scripts/runs/<YYYY-MM-DD>_<short-name>/` (relative to repo root):

- `run.py` — the script. First two non-import lines MUST be `from promptchain.observability import init_mlflow; init_mlflow()`.
- `README.md` — intent + expected output + what triggered the script (session, ad-hoc, etc.).

Tell the user to run it via `bash scripts/observe.sh runs/<folder>` (auto-loads `.env`, sets MLflow, tees to `output.log`). See `scripts/README.md`.

For ephemeral throwaways, use `scripts/scratch/` (gitignored).

## Workflow

1. **Read** `docs/llms/PROMPTCHAIN_FOR_LLMS.md` once at session start (§1-§4 minimum — §5-§16 on demand).
2. **Pick** the matching recipe and adapt it. Recipes are tested patterns; do NOT improvise the import paths or method names.
3. **Verify** any constructor argument or method name against the source-of-truth files listed in `PROMPTCHAIN_FOR_LLMS.md §11` before you write the call.
4. **If you write code that fails** with an `ImportError`, `AttributeError`, or `TypeError` from inside `promptchain` — STOP. Do not retry the same approach. **Apply the meta-rule** (`~/.claude/rules/domains/library-vs-pipeline-bugs.md` if present, else mentally): rule out pipeline errors in YOUR code first by reading source contract → checking canonical examples → checking verbose logs → checking call order. Most "library bugs" are pipeline errors.

## §4 — The most common LLM hallucinations to avoid

(Mirrors `docs/llms/PROMPTCHAIN_FOR_LLMS.md` §9 — keep these in working memory.)

1. `from promptchain import AgentChain` → **wrong**. Use `from promptchain.utils.agent_chain import AgentChain`.
2. `from promptchain import AgenticStepProcessor` → **wrong**. Use `from promptchain.utils.agentic_step_processor import AgenticStepProcessor`.
3. `models=["gpt-4o"]` → **wrong**. Use `"openai/gpt-4o"`. PromptChain delegates to `litellm` which requires the `provider/model` prefix.
4. Counting `models` against ALL instructions → **wrong**. Only string instructions consume model slots. Functions and `AgenticStepProcessor` instances do NOT.
5. `agent_chain.process_prompt_async(...)` → **wrong**. `AgentChain` uses `process_input(...)`. Only `PromptChain` has `process_prompt_async`.
6. `execution_mode="broadcast"` without `synthesizer_config` → `ValueError`.
7. Local tool function returns a dict → LLM sees stringified `repr()`. Return a string or `json.dumps(...)`.
8. `from promptchain import track_llm_call` → **wrong**. Use `from promptchain.observability import track_llm_call`.
13. **`register_tool_function(func)` WITHOUT `add_tools([schema])`** → LLM sees zero tools, agentic step shortcuts to stub answer. Register all functions FIRST, then `add_tools()` ONCE at the end (v0.6.1 validates symmetrically).
14. **Tool-weak specialist model (e.g. `medgemma:4b`) in a chain step that has chain-scoped tools registered** → specialist hallucinates a fake tool call. Workaround: implement that step as a Python function calling `litellm.acompletion(...)` directly without a `tools=` parameter. Real fix pending: per-step tool scoping API (see `docs/llms/issues/PROPOSAL_per_step_tool_scoping.md`).

## Closed-loop feedback

If a mistake you make in this session is **not** in §4 above, the user will run `sio scan` to surface it. The fix lands in:
- `PROMPTCHAIN_FOR_LLMS.md §9` if it's a new anti-pattern
- A recipe edit if the recipe was wrong
- A `FEEDBACK_LOG.md` row in either case
- A real `gyasis/PromptChain` issue if the package itself needs to change

## Files (paths)

All paths repo-relative. The maintainer's repo is at `~/dev/projects/PromptChain/`; substitute your local clone path.

- Long-form reference: `docs/llms/PROMPTCHAIN_FOR_LLMS.md`
- Recipes: `docs/llms/recipes/`
- Feedback audit trail: `docs/llms/FEEDBACK_LOG.md`
- llms.txt index: `docs/llms/llms.txt`
- Issue drafts: `docs/llms/issues/`
- Source of truth: `promptchain/`
- Skill source-of-truth (this file): `.claude/skills/promptchain.md`
