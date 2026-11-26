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
dataset = [("2+2", "4"), ("3+3", "6"), ("5+5", "10")]

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

Expected output: the variant wins (`1.0 True`), and both checks print `PASS`. The stand-in `run` is
deterministic, so the gain is certain; a sampled model gives noisy `per_repeat_means`, and then `above_noise`
can fail.

## Read `checks` before trusting a win

`result.checks` always holds `technique_fired(baseline.outputs, best.outputs)`: it fails when the best
variant's outputs are byte-identical to the baseline's, so the "better" instruction changed nothing. When
`repeats >= 3` it also holds `above_noise(baseline.per_repeat_means, best.per_repeat_means)`: it fails when
the gain sits inside run-to-run variance. Report a win only if the checks you care about pass. With
`repeats < 3` no noise check exists, and a single run proves nothing about a sampled model.

Also read two more fields. `errors` counts runs that raised. Each scores 0.0, so a broken variant looks
like a bad one, not like a crash. `dropped` lists proposed variants rejected for losing a `{placeholder}`,
each with its reason.

## Hold out a split

The winner scores best on the dataset you gave it, so it fits that data.
Split first, optimise on one part, confirm on the other:

```python
train, held_out = dataset[: len(dataset) // 2], dataset[len(dataset) // 2 :]
tuned = optimize_instruction_sync(base, train, metric, run=run, variants=candidates, repeats=3)
check = optimize_instruction_sync(tuned.best.instruction, held_out, metric, run=run)  # score only
```

Compare `check.best.mean` with the same measurement of the baseline on `held_out`. A tiny dataset (under
about 20 items) overfits easily; treat its winner as a lead, not a result.

## Contract

| Item | Rule |
|---|---|
| `instruction` | Baseline text. It is variant 0 and wins ties. |
| `dataset` | Non-empty list of `(input, expected)` or `{"input", "expected"}`. |
| `metric(output, expected)` | Returns a float in [0, 1]; NaN, out-of-range or non-numeric raises `ValueError`. |
| `model` / `run` | Pass exactly one. `model` runs `PromptChain(models=[model], instructions=[variant])`. `run(instruction, step_input)` is an async callable. |
| `variants` | Extra instructions to score. The optimiser removes duplicates. |
| `proposer`, `n_proposals` | `n_proposals > 0` calls `proposer(instruction, n)` once. Proposals that lose a `{placeholder}` the baseline has go to `dropped`. |
| `repeats` | Full passes over the dataset per variant. Use 3 or more to get `above_noise`. |
| Result | `best`, `baseline`, `variants`, `improved` (`best is not baseline` and strictly higher mean), `dropped`, `checks`, `to_dict()`. |

With `model=`, set a fixed low temperature on your model config if you want `repeats` to measure the
instruction instead of sampling noise.

## Cost

Calls to the model = `N x |D| x repeats`, where `N = 1 + len(variants) + (accepted proposals)` and
`|D|` is the dataset size. A proposer adds one call. Example: baseline + 4 variants, 30 items, 3 repeats =
5 x 30 x 3 = 450 calls. Estimate before running (R-SPEND4) and pilot on a few items first.

## Plug in a proposer

A proposer is `async (instruction: str, n: int) -> list[str]`.

```python
from promptchain.utils.instruction_optimizer import optimize_instruction, llm_proposer

result = await optimize_instruction(base, train, metric, model="openai/gpt-4o-mini",
                                    proposer=llm_proposer("openai/gpt-4o"), n_proposals=4, repeats=3)
```

`llm_proposer` wraps the instruction in marker lines and asks for a JSON array; it falls back to one
proposal per line. Your own proposer can be anything, including DSPy, imported by you (the module never
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

- **`ValueError: pass exactly one of model= or run=`** — give one runner.
- **`ValueError: metric must return a number in [0, 1]`** — normalise your metric; do not return bools or percentages.
- **Every variant scores 0.0** — check `errors`; the runner is probably raising.
- **`improved False`** — the baseline held or tied. That is a valid result.
- **`dropped` is full** — the proposer removes `{input}`; tighten its prompt or supply `variants` by hand.
