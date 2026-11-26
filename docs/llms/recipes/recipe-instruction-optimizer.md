---
id: recipe-instruction-optimizer
title: Optimise one step's instruction against a labelled dataset
when: You have one chain step whose instruction you want to improve, and a few labelled examples to measure it on.
api_version: 0.6.1+
---

# Instruction optimiser

`promptchain.utils.instruction_optimizer` runs the baseline instruction, your hand-written variants and
(optionally) proposer-written variants over a dataset, scores each, and returns the winner with validity
checks. It imports no `rich`, `argparse` or `dspy` of its own.

```python
from promptchain.utils.instruction_optimizer import (
    optimize_instruction,        # async
    optimize_instruction_sync,   # blocking; safe inside a running event loop
    llm_proposer,                # proposer backed by one model
    OptimizationResult, VariantScore,
)
```

## Offline example (no model, runs as-is)

```python
from promptchain.utils.instruction_optimizer import optimize_instruction_sync

# Dataset: (step_input, expected) tuples, or {"input": ..., "expected": ...} dicts.
dataset = [(f"{n}+{n}", str(2 * n)) for n in range(1, 9)]  # 8 items; fewer than ~6 cannot pass items_above_noise

# Metric: (output, expected) -> float in [0, 1]. Anything else raises ValueError.
def metric(output, expected):
    return 1.0 if output.strip() == expected else 0.0

# run: (instruction, step_input) -> output. Stand-in for a model call.
async def run(instruction, step_input):
    if "just the number" in instruction:
        return str(eval(step_input))
    return f"The answer is {eval(step_input)}"

result = optimize_instruction_sync(
    "Solve: {input}",
    dataset,
    metric,
    run=run,
    variants=["Solve: {input}. Reply with just the number."],
    repeats=3,
)
print(result.best.instruction, result.best.mean, result.improved)
for check in result.checks:
    print(check)
```

Expected output: the variant wins (`1.0 True`) and every check prints `PASS`. The stand-in `run` is
deterministic, so the gain is certain. scipy may print a `RuntimeWarning: Precision loss` because both arms
have zero variance; it is harmless (silence it with `warnings.filterwarnings("ignore", category=RuntimeWarning)`).

## Read `checks` before trusting a win

`improved` is `best is not baseline and best.mean > baseline.mean`. **It ignores `checks`.** A coin-flip
runner reports `improved=True` in about three of four runs. Trust a win only when every check passes and
`repeats >= 3`. Each check measures one way a number can lie:

| Check | What it measures | Fails when |
|---|---|---|
| `technique_fired` | Share of best outputs that are byte-identical to the baseline's. | 95% or more are identical: the new instruction changed nothing. |
| `above_noise` | Repeat means (baseline vs best), Welch test when scipy is present, else delta vs spread. | The gain sits inside run-to-run variance, or `repeats < 3` (always fails then). A deterministic runner has no variance, so any gain passes: use the next check too. |
| `items_above_noise` | Paired exact sign test over per-item mean scores. The test skips ties. | The best does not win significantly more items than it loses (p >= 0.05). A gain on 1 of 30 items fails. Needs about 6 discordant items to pass. |
| `no_errors` | Run exceptions in the baseline and the best. | Any run raised. An error scores 0.0, so a flaky baseline inflates the best's win. |

When the baseline is the best, the first three checks report `n/a, baseline kept` and pass.

Also read `errors` and `error_samples` on each variant (the first exception reprs), and `dropped`: proposed
or hand-written variants rejected for losing or adding a `{placeholder}`, each with its reason. Each failed
run also logs a WARNING from `promptchain.utils.instruction_optimizer`. If the baseline raises on every
call, the function raises `RuntimeError`: the runner is broken.

## Hold out a split

The winner scores best on the dataset you gave it, so it fits that data.
Split first, optimise on one part, confirm on the other:

```python
train, held_out = dataset[: len(dataset) // 2], dataset[len(dataset) // 2 :]
tuned = optimize_instruction_sync(base, train, metric, run=run, variants=candidates, repeats=3)
check = optimize_instruction_sync(tuned.best.instruction, held_out, metric, run=run)  # score only
```

Compare `check.best.mean` with the same measurement of the baseline on `held_out`. A tiny dataset (under
about 20 items) overfits easily and cannot pass `items_above_noise`; treat its winner as a lead, not a result.

## Contract

| Item | Rule |
|---|---|
| `instruction` | Baseline text. It is variant 0 and wins ties. |
| `dataset` | Non-empty list of `(input, expected)` pairs or `{"input", "expected"}` dicts. Any other row raises `ValueError`. |
| `metric(output, expected)` | Returns a float in [0, 1]; a bool, NaN, an out-of-range or a non-numeric value raises `ValueError`. |
| `model` / `run` | Pass exactly one. `model` runs `PromptChain(models=[model], instructions=[variant])`; if the baseline has no `{input}`, the optimiser appends `\n{input}` to it, because PromptChain renders the step input only through that placeholder. `run(instruction, step_input)` is an async callable. |
| `variants` | Extra instructions to score. The optimiser removes duplicates. Each must keep exactly the baseline's `{placeholder}` set, or it goes to `dropped`. |
| `proposer`, `n_proposals` | `n_proposals > 0` calls `proposer(instruction, n)` once. The optimiser reads at most `n` proposals. Proposals that lose or add a `{placeholder}`, or are not strings, go to `dropped`. |
| `repeats` | Full passes over the dataset per variant. Use 3 or more: below 3, `above_noise` always fails. |
| Result | `best`, `baseline`, `variants`, `improved` (`best is not baseline` and strictly higher mean), `dropped`, `checks`, `to_dict()`. |

With `model=`, set a fixed low temperature on your model config if you want `repeats` to measure the
instruction instead of sampling noise.

## Cost

Model calls = `N x |D| x repeats`. `N` counts the scored candidates: the baseline, each distinct valid
variant, and each distinct valid proposal (at most `n_proposals`). `|D|` is the dataset size. A proposer
adds one call. Example: baseline + 4 variants, 30 items, 3 repeats = 5 x 30 x 3 = 450 calls. Estimate
before running (R-SPEND4) and pilot on a few items first.

## Plug in a proposer

A proposer is `async (instruction: str, n: int) -> list[str]`.

```python
import asyncio
from promptchain.utils.instruction_optimizer import optimize_instruction, llm_proposer

async def main():
    return await optimize_instruction(base, train, metric, model="ollama_chat/glm-5.3-flash:cloud",
                                      proposer=llm_proposer("ollama_chat/glm-5.3:cloud"), n_proposals=4, repeats=3)

result = asyncio.run(main())
```

`llm_proposer` wraps the instruction in marker lines with a random nonce and asks for a JSON array. It reads
the first JSON array of strings in the reply (code fences allowed); else it keeps numbered or bulleted lines. Your own proposer can be anything, including DSPy, imported by you (the module never
imports it):

```python
async def dspy_proposer(instruction, n):
    import dspy                      # your dependency, configured with dspy.configure(lm=...)
    propose = dspy.Predict("instruction, n -> rewrites")
    out = propose(instruction=instruction, n=str(n)).rewrites
    return [line.strip() for line in out.splitlines() if line.strip()][:n]
```

A proposer sees the instruction text, so an instruction that holds untrusted text can steer it. Keep
instructions you optimise under your control.

## Common failures + fix

- **Sync wrapper inside a running loop** — `optimize_instruction_sync` does not raise there. It blocks the loop and runs your `run` on a fresh loop in a worker thread. A `run` that uses a client, lock or future from the outer loop fails on every call, and the baseline then raises `RuntimeError`. Await `optimize_instruction` from async code instead.
- **`ValueError: pass exactly one of model= or run=`** — give one runner.
- **`ValueError: metric must return a number in [0, 1]`** — normalise your metric; do not return bools or percentages.
- **Every variant scores 0.0** — check `errors`; the runner is probably raising.
- **`improved False`** — the baseline held or tied. That is a valid result.
- **`dropped` is full** — the proposer removes `{input}`; tighten its prompt or supply `variants` by hand.
