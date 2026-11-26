"""Safety behaviour of the loop tools: CoVe replies, verdict parsing, signatures, sandbox cleanup."""
import asyncio
import json
import logging
import shutil
import subprocess
import warnings
from unittest import mock

import pytest

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.autoresearch import AutoResearch
from promptchain.utils.docker_executor import DockerExecutor
from promptchain.utils.ralph_chain import RalphChain, _norm_sig
from promptchain.utils.test_loop_chain import MicroPromptChain, is_approval
from promptchain.utils.verification import CoVeVerifier

TOOLS = [
    {"type": "function", "function": {"name": n, "description": n,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
    for n in ("read_file", "delete_file")
]
FENCE = "`" * 3


def _call(call_id, name):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps({"path": "/x"})}}


def _verdict(execute, confidence):
    return json.dumps({"should_execute": execute, "confidence": confidence, "assumptions": [],
                       "risks": [], "reasoning": "r", "suggested_modifications": None})


def _verifier(reply):
    async def runner(messages, model=None, **kw):
        return {"content": reply}
    return CoVeVerifier(runner, model_name="m")


def _verify(reply):
    return asyncio.run(_verifier(reply).verify_tool_call("delete_file", {"path": "/x"}, "ctx", TOOLS))


# -- 1 + 7: skipped CoVe calls get a tool reply; the log reports real counts -------------------
def test_cove_skipped_call_gets_tool_reply_and_log_counts(caplog):
    turns = [{"content": "go", "tool_calls": [_call("c1", "read_file"), _call("c2", "delete_file")]},
             {"content": "done", "tool_calls": []}]
    second_request = []

    async def runner(messages, model=None, tools=None, tool_choice=None, **kw):
        if tools is None:
            risky = "Name: delete_file" in messages[0]["content"]
            return {"content": _verdict(not risky, 0.2 if risky else 0.95)}
        if len(turns) == 1:
            second_request.extend(messages)
        return turns.pop(0)

    async def executor(call):
        return "ok"

    step = AgenticStepProcessor(objective="o", model_name="m", max_internal_steps=3, enable_cove=True)
    with caplog.at_level(logging.INFO):
        asyncio.run(step.run_async("start", TOOLS, runner, executor))
    answered = {m["tool_call_id"]: m["content"] for m in second_request if m.get("role") == "tool"}
    assert set(answered) == {"c1", "c2"}
    assert answered["c2"].startswith("skipped by CoVe:")
    assert "[CoVe] Verified 1/2 tool calls" in caplog.text


# -- 2: the parser fails closed on a malformed verdict or confidence ---------------------------
@pytest.mark.parametrize("reply", [
    '{"should_execute": "false", "confidence": 0.99}',
    '{"confidence": 0.99}',
    '{"should_execute": true, "confidence": NaN}',
    '{"should_execute": true, "confidence": Infinity}',
    '[1, 2]',
])
def test_cove_parser_rejects_malformed_reply(reply):
    result = _verify(reply)
    assert result.should_execute is False
    assert result.confidence == 0.0


def test_cove_parser_still_accepts_a_well_formed_reply():
    result = _verify(_verdict(True, 0.9))
    assert result.should_execute is True and result.confidence == 0.9


# -- 3: fallback_model reaches the verifier through a PromptChain ------------------------------
def test_fallback_model_reaches_verifier_inside_promptchain():
    seen = []
    turns = [{"role": "assistant", "content": "", "tool_calls": [_call("c1", "read_file")]},
             {"role": "assistant", "content": "done"}]

    async def fake(self, model_name, messages, params=None, tools=None, tool_choice=None, **kw):
        seen.append((model_name, bool(tools)))
        return turns.pop(0) if tools else {"role": "assistant", "content": _verdict(True, 0.9)}

    step = AgenticStepProcessor(objective="o", model_name="openai/MAIN", fallback_model="openai/CHEAP",
                                enable_cove=True, max_internal_steps=3)
    chain = PromptChain(models=[], instructions=[step], verbose=False)
    def read_file(path: str) -> str:
        return "x"

    chain.register_tool_function(read_file)
    chain.add_tools([TOOLS[0]])
    with mock.patch.object(PromptChain, "run_model_async", fake):
        asyncio.run(chain.process_prompt_async("go"))
    verifier_calls = [m for m, with_tools in seen if not with_tools]
    assert verifier_calls == ["openai/CHEAP"]


# -- 4: the stagnation signature comes from the failure, not the banner -------------------------
BANNER = "=" * 30 + " test session starts " + "=" * 30 + "\nplatform linux -- Python 3.12\ncollected 1 item\n" * 3


def test_norm_sig_distinguishes_failures_with_a_shared_banner():
    a = BANNER + "E   assert 5 == 6\n"
    b = BANNER + "E   NameError: name 'x' is not defined\n"
    assert _norm_sig(a) != _norm_sig(b)


def test_norm_sig_matches_the_same_failure_with_different_numbers():
    assert _norm_sig(BANNER + "E   assert 5 == 6\n") == _norm_sig(BANNER + "E   assert 7 == 8\n")


def test_norm_sig_without_e_lines_uses_the_tail():
    assert _norm_sig(BANNER + "tail one") != _norm_sig(BANNER + "tail two")


def test_ralph_does_not_stagnate_on_different_failures():
    n = {"i": 0}

    async def lib(p):
        return "brief"

    async def art(p):
        n["i"] += 1
        word = ("alpha", "beta", "gamma", "delta")[n["i"] - 1]
        body = "def f():\n    return '%s'\n" % word
        return f"{FENCE}python\n{body}{FENCE}"

    tests = "from solution import f\n\ndef test_f():\n    assert f() == 99\n"
    cmd = "python -c \"import sys; print('=' * 400); print('E   got different value %s' % open('solution.py').read().split()[-1].strip(chr(39))); sys.exit(1)\""
    res = RalphChain(librarian=lib, artisan=art, run_critic=False, use_docker=False,
                     max_iterations=4, entropy_threshold=3).run_sync(
        "o", test_command=cmd, deps={"test_solution.py": tests})
    assert res.result != "STAGNATED"


# -- 5: the unsandboxed fallback warns every time ----------------------------------------------
def test_unsandboxed_fallback_warns_on_every_run():
    async def gen(p):
        return f"{FENCE}python\nx = 1\n{FENCE}"

    with mock.patch.object(DockerExecutor, "available", staticmethod(lambda *a, **k: False)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                MicroPromptChain(generate=gen, max_iterations=1).run_sync("o", "s.py", test_command="true")
    assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 3


# -- 6: verdict parsing -------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("APPROVE", True), ("approved", True), ("**APPROVE**", True), ("APPROVE, no failures found", True),
    ("APPROVE\nbut not safe", True),
    ("DO NOT APPROVE", False), ("Not approved", False), ("Unacceptable: bad", False),
    ("APPROVE but I cannot confirm", False), ("REJECT", False), ("FAIL", False), ("", False),
    ("PASS", False),
])
def test_is_approval(text, expected):
    assert is_approval(text) is expected


def test_autoresearch_rejects_a_negated_verdict():
    async def gen(p):
        return f"{FENCE}python\nx = 1\n{FENCE}"

    async def critique(brief, build):
        return "DO NOT APPROVE"

    res = AutoResearch(generate=gen, critique=critique, use_docker=False, max_iterations=1).run_sync(
        "b", test_command="true")
    assert res.build.result == "PASS" and res.verified is False


# -- 8: a Docker timeout kills the named container ----------------------------------------------
def test_docker_timeout_kills_the_container():
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, 1)
        return subprocess.CompletedProcess(argv, 0, "", "")

    ex = DockerExecutor(timeout=1)
    with mock.patch("promptchain.utils.docker_executor.subprocess.run", fake_run):
        res = ex.run("sleep 60")
    assert res.timed_out
    name = calls[0][calls[0].index("--name") + 1]
    assert ["docker", "kill", name] in calls


@pytest.mark.skipif(not DockerExecutor.available() or shutil.which("docker") is None, reason="no docker daemon")
def test_docker_timeout_leaves_no_running_container():
    img = "python:3.12-slim"
    if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode != 0:
        pytest.skip(f"{img} is not pulled locally")
    ex = DockerExecutor(image=img, network="none", timeout=3)
    res = ex.run("sleep 60")
    assert res.timed_out
    left = subprocess.run(["docker", "ps", "-q", "--filter", "name=pc-exec-"],
                          capture_output=True, text=True).stdout.strip()
    assert left == ""
