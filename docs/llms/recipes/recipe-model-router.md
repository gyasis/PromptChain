---
id: recipe-model-router
title: Model router - pick a model per task from difficulty x ability (experimental)
when: You run a cheap model and a strong model, and you want each task to start on the cheapest model that is likely to pass. Read the validation section before you trust it.
api_version: 0.20.0+
---

# Model router (experimental)

> **Status: unvalidated; `link` must be fit on outcome data; there is no default link.**
> No measurement shows that routing by difficulty beats "always use the same model". The
> module is a tool to run that experiment and, if it passes, to route. Until then, treat a
> routing decision as a hypothesis.

`ModelRouter` takes a task, asks a **difficulty source** how hard it is (a number in [0, 1]),
and applies a **link** `P(pass | model ability, difficulty)` that you fit from past outcomes.
It picks the first model whose `P(pass)` reaches your `threshold`. It reads your models
cheapest first. It also returns the rest of the list as an escalation order.

Source: `promptchain/utils/model_router.py`. Tests: `tests/test_model_router.py`.

## What it does NOT do
- It does **not** ship a link. `link=None` makes every plan fall back to the most able model.
- It does **not** know costs or latency. "Cheapest" means the order of your `models` list.
- It does **not** check the answer after the call. Use a verifier or a cascade for that.
- It does **not** import `cascade`, and it does **not** change `AgenticStepProcessor`, the
  profiler or `decision_step`.
- Abilities from the profiler's default item bank are hand-set (`ItemBank.synthetic`). Do not
  treat them as measured. Prefer `abilities=` values you fitted from your own outcomes.
- The profiler can probe models on the governor's llamacpp lane. A cohort entry `llamacpp:<alias>` runs the latency and tool-call probes through `/llm/llamacpp/v1/chat/completions` with thinking off (`promptchain/profiler/governor.py`, `LlamacppGovernorClient`). Listing aliases loads nothing. The first call per alias can wait for the sidecar's load bound.

## Offline example (no model, no network)
```python
import asyncio, random
from promptchain import PromptChain
from promptchain.profiler.irt import prob_3pl
from promptchain.utils.model_router import (
    Difficulty, DifficultyLink, ModelRouter, RoutingOutcome, route_model,
    write_outcomes, read_outcomes,
)

MODELS = ["small", "big"]                    # cheapest first, most able last
ABILITY = {"small": 1.0, "big": 3.0}         # abilities you fitted yourself

# 1. Label data. In production these come from the exploration slice (below).
rnd = random.Random(0)
rows = []
for i in range(2000):
    m = rnd.choice(MODELS); d = rnd.random()
    passed = rnd.random() < prob_3pl(ABILITY[m], 1.8, 0.1 + 2.0 * d, 0.0)   # stand-in for a real grader
    rows.append(RoutingOutcome(f"t{i}", "task text", d, "demo", m, ABILITY[m], passed, "explore"))
write_outcomes(rows, "/tmp/outcomes.jsonl", append=False)

# 2. Fit the link on a TRAIN split only. Keep the rest for evaluation.
train = read_outcomes("/tmp/outcomes.jsonl")[:1500]
link = DifficultyLink.fit(train)             # raises on one outcome class or a <= 0
print(link)

# 3. Any async callable str -> Difficulty is a difficulty source.
async def my_difficulty(task: str) -> Difficulty:
    return Difficulty(min(1.0, len(task) / 200), source="length-demo")

router = ModelRouter(MODELS, threshold=0.7, timeout_s=2.0,
                     difficulty=my_difficulty, link=link, abilities=ABILITY)

async def run(model: str, task: str) -> str:   # your real call goes here
    return f"[{model}] answered: {task}"

async def main():
    easy, hard = "hi", "analyse this contract clause by clause " * 6
    for task in (easy, hard):
        plan = await router.plan(task)
        print(plan.chosen, plan.order, round(plan.difficulty.value, 2), plan.reason)
    chain = PromptChain(models=[], instructions=[route_model(router, run)])
    print(await chain.process_prompt_async(easy))   # runs on "small"

asyncio.run(main())
```

## Difficulty sources
`difficulty` is any `async (task: str) -> Difficulty`. `Difficulty.value` must be a finite
number in [0, 1]; 0 is easiest.

### tev1 (a decision backend, one `score` question)
```python
from promptchain.utils.decision_step import SystemOneBackend
from promptchain.utils.model_router import ModelRouter, decision_difficulty

backend = SystemOneBackend(model="tev1:4b", timeout=5.0)   # SHORT timeout; default is 300 s
difficulty = decision_difficulty(
    backend,
    instructions="How hard is this task for a small language model?",
    levels=["trivial", "easy", "medium", "hard", "expert"],  # easiest first
    prepare_state=lambda t: t[:4000],                        # tev1 has a 2,050-token limit
)
router = ModelRouter(MODELS, threshold=0.7, timeout_s=6.0, difficulty=difficulty, link=link,
                     abilities=ABILITY)
```
- The value is the expected level index divided by `len(levels) - 1`. The router reads the
  answer's `probabilities` (keyed by level label or index) when present, else the numeric
  `score` as a level index. Verified against a real tev1 call: the reply is
  `{"type": "score", "score": 2.924, "legend": {"0": "trivial", ..., "4": "expert"},
  "probabilities": {"0": ..., "4": ...}, "confidence": 0.269}`. The score is the 0-based
  expected index, the probabilities use the keys `"0"` to `"4"`, and `decision_difficulty` read
  the value 0.7311 from it.
- The levels are **ordinal**, not interval. The link absorbs the spacing only if you fit it on
  this exact source. Change the levels or the model and you must refit.
- `asyncio.wait_for` cannot kill the executor thread inside `SystemOneBackend`. Set the
  backend `timeout` at or below the router's `timeout_s`. The router logs a warning at
  construction when it is longer. See "Executor starvation" below.
- Pick `prepare_state` so long tasks are cut to a prefix or summary that tev1 can read.

### NanoJev (or any custom callable)
There is no difficulty head on NanoJev today (`nanojev-duly` is a fixed router with 9 labels
and 4 flags). If you train one, wrap it as a plain callable:
```python
from promptchain.utils.model_router import Difficulty

async def nanojev_difficulty(task: str) -> Difficulty:
    score = await my_nanojev_client.difficulty(task)        # your code; return a float in [0, 1]
    return Difficulty(score, source="nanojev-difficulty-v1", model="nanojev")
```
Use a distinct `source` string per version. The link belongs to one source.

## Failure behaviour (never blocks a chain)
`plan()` never raises. Suppose the difficulty call times out, raises (including
`DecisionError`) or returns a non-`Difficulty`. Then the router logs **one** warning for that
call and returns the last model with `degraded` and `reason` set. The reason says "most able
model" only when the abilities support it; otherwise it says "last model in the list".
When `difficulty` is `None` or `link` is `None`, the plan degrades the same way without a
warning, because that is configuration and not a failed call. With `link=None` the router does
not call the difficulty source at all: the difficulty cannot change the choice without a link.
To collect difficulty before a link exists, call your difficulty function yourself when you
record outcomes.

`asyncio.CancelledError` from your own task cancellation propagates. The router does not
swallow it.

The router also logs two checks once. One fires when abilities do not rise in list order, which
means your list may be misordered. The other fires when a profile store exists but holds no
readable profiles. `reason` names any model with a missing or non-finite ability. The router
never chooses that model.

### Executor starvation (read this before you use tev1 in a loop)
`SystemOneBackend` runs its HTTP call in asyncio's default executor, which has about 16 threads.
`asyncio.wait_for` stops waiting after `timeout_s`, but the thread keeps running until the
backend's own `timeout` (default 300 s). Two effects follow when the server hangs:
- **Starvation.** After about 16 hung calls the executor is full. Every later call that needs
  it, including `loop.getaddrinfo` (DNS), waits behind them.
- **Exit hang.** `asyncio.run` joins the default executor when it exits. A sync
  `chain.process_prompt(...)` uses `asyncio.run`, so the process can sit at exit for up to the
  backend timeout after your answer is already printed.

The fix is to set the backend `timeout` at or below the router's `timeout_s`. Then a hung thread
frees itself on the same schedule the router uses. The router logs a warning at construction
when the backend timeout is longer. The check is a warning and not an error, because a slow
backend with a matching long `timeout_s` is a valid choice; the router only warns about the
mismatch it can see.

Also plan for **warning volume**: a down backend logs one warning per task. In a loop, add your
own rate limit, or check `plan.degraded` and stop.

## Abilities
`abilities={"small": -0.5, ...}` wins. Otherwise the router reads `store_path` (a
`ModelProfiler` JSON store) lazily: `skills[dimension].theta` when you pass `dimension`, else
the mean theta over the capability dimensions. If any skill used is `low_precision`, the plan
`reason` says so for that plan only. The router reads the store once per plan. It skips models
with no ability and never chooses one on a guess. **One link must not span several dimensions**; fit one link per `dimension` and per task family.

## Compose with other pieces
- **`route_on`:** a branch is any callable, so `route_model(router, run)` can be one branch.
- **Two-tier `AgenticStepProcessor`:** call `await router.plan(task)` before you build the
  step. Pass `plan.chosen` as its model and the next model in `plan.order` as
  `fallback_model`. The step's per-call `model_override` is not wired; do not rely on it.
- **Cascade (verify-then-escalate):** `plan.order` is the tier list: start at `plan.chosen`,
  escalate to the next. The router does not import the cascade module; pass the list yourself.
- **Escalation cost:** a long task that escalates pays for every tier it touches. Log
  `plan.to_dict()` (use `on_plan=`) and watch the escalation rate.

## Learning loop and the exploration slice
Every run produces a label: did model `m` pass task `t` with difficulty `d`?
`RoutingOutcome` is the record; `write_outcomes` / `read_outcomes` are the JSONL I/O.
```json
{"task_id": "t17", "task": "...", "difficulty": 0.42, "difficulty_source": "decision:systemone",
 "model": "small", "theta": -0.5, "passed": false, "source": "explore", "reason": ""}
```
**You MUST run an exploration slice.** Sample a fixed fraction of tasks (for example 5%) and
run them on **every** model, ignoring the router. Label them `source="explore"`.
Data from routed runs alone is selection-biased: the router sent easy tasks to the cheap
model, so the cheap model looks better than it is, and the fit learns the router's own choices.
Fit on exploration rows; use routed rows only for monitoring.

Fit needs at least 30 outcomes with both passes and fails. It raises on perfectly
separable data, on no convergence, and on degenerate parameters (`a < 1e-6`, or `|alpha|` or
`|beta|` above 1e3), and it logs a warning when `beta <= 0`. Read those errors as "collect more
mixed outcomes", not as bugs. `read_outcomes` checks field types (`passed` must be a real
boolean) and `write_outcomes` rejects NaN.

Library scope: `DifficultyLink.fit` and the JSONL helpers. Training a NanoJev difficulty head
stays in `harness-lab`.

## Validation experiment (the gate)
Do not trust the router before this passes. It is not built in this package.
1. **Data.** RouterBench (`withmartian/routerbench`; check the licence and file shape first):
   per-prompt correctness for about 11 models. RouteLLM arena data is a second source.
2. **Stage 0, go or no-go.** Spearman correlation between your difficulty score and the
   empirical failure rate across models, on the train split. Near zero: stop here.
3. **Fit.** Fit abilities and the link on train only.
4. **Report on held-out data:** AUROC and ECE of `P(pass)`, and accuracy-versus-cost against
   always-cheap, always-strong, random-matched-cost, a theta-only constant-difficulty router,
   and the oracle. Split by prompt, and also leave one benchmark out.
5. **Repeat on your own fleet** with prompts that have pulled gold answers (not self-written).
Never let the eval split into `fit`. `DifficultyLink.auroc` is in-sample.

## Common mistakes
- Routing with an unfitted link, or with abilities from the synthetic item bank.
- Reusing one link across difficulty sources, dimensions or models.
- Fitting on routed data only (selection bias); skipping the exploration slice.
- Using one global `threshold`. Set it per task family from the held-out reliability table.
- Treating the ordinal tev1 score as a calibrated probability.
- Leaving the backend `timeout` at its 300 s default.
- Reading `plan.chosen` without checking `plan.degraded`.
