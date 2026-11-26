"""Offline tests for promptchain.utils.instruction_optimizer (no model or network calls)."""
import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import pytest

from promptchain.utils.instruction_optimizer import (
    llm_proposer,
    optimize_instruction,
    optimize_instruction_sync,
)

DATA = [("a", "A"), ("b", "B"), ("c", "C")]


def metric(out, expected):
    return 1.0 if out == expected else 0.0


async def run_by_instruction(instruction, x):
    if "upper" in instruction:
        return x.upper()
    return x


def test_picks_known_best():
    res = optimize_instruction_sync(
        "echo {input}", DATA, metric, run=run_by_instruction, variants=["upper {input}", "other {input}"]
    )
    assert res.best.instruction == "upper {input}"
    assert res.improved and res.best.mean == 1.0 and res.baseline.mean == 0.0
    assert res.checks[0].name == "technique_fired" and res.checks[0].passed


def test_tie_keeps_baseline_and_dedupes():
    res = optimize_instruction_sync(
        "echo {input}", DATA, metric, run=run_by_instruction, variants=["echo {input}", "echo2 {input}"]
    )
    assert res.best is res.baseline and not res.improved
    assert [v.instruction for v in res.variants] == ["echo {input}", "echo2 {input}"]


def test_errors_counted_as_zero():
    async def run(instruction, x):
        if "bad" in instruction:
            raise RuntimeError("boom")
        return x.upper()

    res = optimize_instruction_sync("ok {input}", DATA, metric, run=run, variants=["bad {input}"])
    bad = res.variants[1]
    assert bad.errors == 3 and bad.mean == 0.0 and res.baseline.errors == 0


@pytest.mark.parametrize("value", [1.5, -0.1, float("nan"), "x"])
def test_out_of_range_metric_raises(value):
    with pytest.raises(ValueError):
        optimize_instruction_sync("i {input}", DATA, lambda o, e: value, run=run_by_instruction)


def test_placeholder_drop():
    async def proposer(instruction, n):
        return ["keeps {input}", "lost the slot"]

    res = optimize_instruction_sync(
        "i {input}", DATA, metric, run=run_by_instruction, proposer=proposer, n_proposals=2
    )
    assert [d[0] for d in res.dropped] == ["lost the slot"]
    assert "{input}" in res.dropped[0][1]
    assert [v.instruction for v in res.variants] == ["i {input}", "keeps {input}"]


def test_above_noise_only_at_three_repeats():
    r1 = optimize_instruction_sync("echo {input}", DATA, metric, run=run_by_instruction,
                                   variants=["upper {input}"], repeats=2)
    r3 = optimize_instruction_sync("echo {input}", DATA, metric, run=run_by_instruction,
                                   variants=["upper {input}"], repeats=3)
    assert [c.name for c in r1.checks] == ["technique_fired"]
    assert [c.name for c in r3.checks] == ["technique_fired", "above_noise"]
    assert len(r3.best.per_repeat_means) == 3 and len(r3.best.scores) == 9


def test_model_xor_run_and_dict_rows():
    with pytest.raises(ValueError):
        optimize_instruction_sync("i", DATA, metric)
    with pytest.raises(ValueError):
        optimize_instruction_sync("i", DATA, metric, run=run_by_instruction, model="m")
    res = optimize_instruction_sync("echo {input}", [{"input": "a", "expected": "a"}], metric,
                                    run=run_by_instruction)
    assert res.best.mean == 1.0


def test_model_path_uses_promptchain():
    with patch("promptchain.utils.promptchaining.PromptChain.process_prompt_async",
               new=AsyncMock(side_effect=lambda x: x.upper())):
        res = optimize_instruction_sync("i {input}", DATA, metric, model="fake/model")
    assert res.baseline.mean == 1.0


def test_llm_proposer_parsing():
    reply = 'sure: ["one {input}", "two {input}", 3]'
    with patch("promptchain.utils.promptchaining.PromptChain.process_prompt_async",
               new=AsyncMock(return_value=reply)) as m:
        out = asyncio.run(llm_proposer("fake/model")("i {input}", 5))
    assert out == ["one {input}", "two {input}"]
    assert "i {input}" in m.call_args.args[0]
    with patch("promptchain.utils.promptchaining.PromptChain.process_prompt_async",
               new=AsyncMock(return_value="1. first {input}\n- second {input}\n")):
        out = asyncio.run(llm_proposer("fake/model")("i {input}", 1))
    assert out == ["first {input}"]


def test_sync_wrapper_inside_running_loop():
    async def main():
        return optimize_instruction_sync("echo {input}", DATA, metric, run=run_by_instruction,
                                         variants=["upper {input}"])

    assert asyncio.run(main()).improved


def test_to_dict_is_json_safe():
    import json

    res = optimize_instruction_sync("echo {input}", DATA, metric, run=run_by_instruction)
    json.dumps(res.to_dict())


def test_import_pulls_no_heavy_modules():
    # `import promptchain` itself loads rich/argparse, so measure what this module adds on top.
    code = (
        "import sys, promptchain\n"
        "before={m for m in ('rich','argparse','dspy') if m in sys.modules}\n"
        "import promptchain.utils.instruction_optimizer\n"
        "after={m for m in ('rich','argparse','dspy') if m in sys.modules}\n"
        "print(sorted(after-before), 'dspy' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[] False", out.stdout + out.stderr
