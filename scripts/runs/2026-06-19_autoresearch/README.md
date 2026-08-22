# 2026-06-19_autoresearch — autoresearch as a PromptChain MVP

**Intent.** A runnable, single-process port of the 7-agent **autoresearch** doer/critic pipeline
(designed at `~/Documents/code/dataengineer/autoresearch/`). Proves the orchestration: a
deterministic **stage state-machine** (the `while`-loop in `run_brief`) dispatches each stage to its
agent and routes on the returned verdict — reproducing the doer⇄critic bounce-backs that a *linear*
PromptChain can't. Showcases all three PromptChain modes: an **`AgenticStepProcessor`** (miner, with
tools), **sequential `PromptChain`** steps (taste/reviewer/reviser/implementer/code-reviewer), and a
**Python `Callable`** (runner).

**Triggered by:** the session building autoresearch (2026-06-19) — "build the PromptChain version."

**MVP scope (what's mocked):**
- grounding tools (`deeplake_retrieve` / `web_search` / `github_readme`) return canned JSON, so it
  runs with no external deps;
- job state is in-memory (not the file-bus);
- the `--hitl` gate auto-approves (set `AR_HITL=ask` to pause for input);
- `run_job` (the runner) mocks a sandboxed build+check → PASS.
Swap these for the real deeplake/web/github tools + a real docker runner to productionize.

**Run it:**
```bash
bash scripts/observe.sh runs/2026-06-19_autoresearch      # auto-loads .env, MLflow, tees output.log
# or directly:
python scripts/runs/2026-06-19_autoresearch/run.py
```
Env knobs: `AR_MODEL` (default `openai/gpt-4o-mini`), `AR_HITL=ask`, `AR_MAX_TICKS`.

**Expected output:** a stage-by-stage trace, e.g.
```
[queue           ] miner          filed approach
[needs-taste     ] taste          ACCEPT
[needs-review    ] reviewer       INITIAL_DRAFT
[needs-revision  ] reviser        authored plan
[needs-review    ] reviewer       APPROVE
[needs-code      ] implementer    wrote code
[needs-code-review] code-reviewer ACCEPT
[needs-human     ] --hitl         auto-approved
[ready-to-run    ] runner         PASS: (mock) ...
=== FINISHED: stage=done result=PASS bounces={...} ===
```
(Exact verdicts vary — they're real LLM judgments. Bounces route back per the caps of 3.)

**Cost:** real OpenAI calls (gpt-4o-mini) — a handful per run; the miner's agentic step makes a few
tool-calling round-trips. Cheap, but not free.
