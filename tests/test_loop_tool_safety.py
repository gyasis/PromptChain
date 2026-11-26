"""Safety behaviour of the loop tools: CoVe replies, verdict parsing, signatures, sandbox cleanup."""
import asyncio
import json
import logging
import os
import shutil
import time
import subprocess
import warnings
from unittest import mock

import pytest

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.autoresearch import AutoResearch
from promptchain.utils.docker_executor import DockerExecutor
from promptchain.utils.ralph_chain import RalphChain, _norm_sig
from promptchain.utils.test_loop_chain import LocalExecutor, MicroPromptChain, is_approval
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
    ("APPROVE", True), ("approved", True), ("**APPROVE**", True), ("VERDICT: APPROVE", True),
    ("**Verdict:** APPROVED", True), ("<think>hmm, not sure</think>\nAPPROVE", True),
    ("APPROVE\nbut not safe", True),
    ("APPROVE, no failures found", False), ("APPROVE - it didn't break", False),
    ("APPROVE (tests aren't exhaustive)", False), ("APPROVE. No issues.", False),
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
def test_docker_timeout_kills_the_named_container_it_started():
    img = "python:3.12-slim"
    if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode != 0:
        pytest.skip(f"{img} is not pulled locally")
    captured = {}
    real_run = subprocess.run

    def spy(argv, **kw):
        if len(argv) > 1 and argv[1] == "run":
            captured["name"] = argv[argv.index("--name") + 1]
        return real_run(argv, **kw)

    ex = DockerExecutor(image=img, network="none", timeout=3)
    with mock.patch("promptchain.utils.docker_executor.subprocess.run", spy):
        res = ex.run("sleep 60")
    assert res.timed_out and captured["name"].startswith("pc-exec-")
    left = subprocess.run(["docker", "ps", "-q", "--filter", f"name=^/{captured['name']}$"],
                          capture_output=True, text=True).stdout.strip()
    assert left == ""


def test_docker_name_equals_form_is_respected():
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, 1)
        return subprocess.CompletedProcess(argv, 0, "", "")

    ex = DockerExecutor(timeout=1, extra_run_args=["--name=mine"])
    with mock.patch("promptchain.utils.docker_executor.subprocess.run", fake_run):
        ex.run("sleep 60")
    assert sum(1 for a in calls[0] if str(a).startswith("--name")) == 1
    assert not any(c[1] in ("kill", "rm") for c in calls[1:])


# -- 4: a timeout whose captured output is bytes ----------------------------------------------
def test_docker_timeout_with_bytes_output_does_not_raise():
    def fake_run(argv, **kw):
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, 1, output=b"partial \xff out", stderr=b"err")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with mock.patch("promptchain.utils.docker_executor.subprocess.run", fake_run):
        res = DockerExecutor(timeout=1).run("x")
    assert res.timed_out and "partial" in res.output and "err" in res.output


def test_local_executor_timeout_kills_the_process_group_and_keeps_output():
    with LocalExecutor(timeout=1) as ex:
        res = ex.run("echo started; sleep 30 & echo $! > child.pid; wait")
        pid = int(open(os.path.join(ex.work_dir, "child.pid")).read())
    assert res.timed_out and "started" in res.output
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        pytest.fail("the shell's child survived the timeout")


# -- 1: fallback_model on another provider ------------------------------------------------------
def _run_cove_chain(step_kw, chain_models=()):
    seen = []
    turns = [{"role": "assistant", "content": "", "tool_calls": [_call("c1", "read_file")]},
             {"role": "assistant", "content": "done"}]

    async def fake(self, model_name, messages, params=None, tools=None, tool_choice=None, **kw):
        seen.append((model_name, dict(params or {}), bool(tools)))
        return turns.pop(0) if tools else {"role": "assistant", "content": _verdict(True, 0.9)}

    step = AgenticStepProcessor(objective="o", enable_cove=True, max_internal_steps=3, **step_kw)
    chain = PromptChain(models=list(chain_models), instructions=[step], verbose=False)

    def read_file(path: str = "") -> str:
        return "x"

    chain.register_tool_function(read_file)
    chain.add_tools([TOOLS[0]])
    with mock.patch.object(PromptChain, "run_model_async", fake):
        asyncio.run(chain.process_prompt_async("go"))
    return seen


MAIN_PARAMS = {"api_base": "https://ollama.com", "api_key": "K", "temperature": 0}


def test_fallback_on_another_provider_drops_connection_params(caplog):
    with caplog.at_level(logging.WARNING):
        seen = _run_cove_chain(dict(model_name="ollama_chat/qwen3:8b", model_params=dict(MAIN_PARAMS),
                                    fallback_model="openai/gpt-4o-mini"))
    verifier = [(m, p) for m, p, with_tools in seen if not with_tools]
    assert verifier == [("openai/gpt-4o-mini", {"temperature": 0})]
    assert "different provider" in caplog.text


def test_fallback_on_the_same_provider_keeps_params():
    seen = _run_cove_chain(dict(model_name="ollama_chat/qwen3:8b", model_params=dict(MAIN_PARAMS),
                                fallback_model="ollama_chat/qwen3:1b"))
    verifier = [(m, p) for m, p, with_tools in seen if not with_tools]
    assert verifier == [("ollama_chat/qwen3:1b", MAIN_PARAMS)]


def test_fallback_dict_config_uses_its_own_params():
    fb = {"name": "openai/gpt-4o-mini", "params": {"temperature": 0.1}}
    seen = _run_cove_chain(dict(model_name="ollama_chat/qwen3:8b", model_params=dict(MAIN_PARAMS),
                                fallback_model=fb))
    verifier = [(m, p) for m, p, with_tools in seen if not with_tools]
    assert verifier == [("openai/gpt-4o-mini", {"temperature": 0.1})]
    main = [(m, p) for m, p, with_tools in seen if with_tools]
    assert all(m == "ollama_chat/qwen3:8b" and p == MAIN_PARAMS for m, p in main)


# -- 2 + 3: parsing real replies ----------------------------------------------------------------
JSON_OK = ('{"should_execute": true, "confidence": 0.9, "assumptions": [], "risks": [], '
           '"reasoning": "ok"}')


@pytest.mark.parametrize("reply", [
    FENCE + "json\n" + JSON_OK + "\n" + FENCE,
    FENCE + "\n" + JSON_OK + "\n" + FENCE,
    "Here is my verification:\n" + FENCE + "json\n" + JSON_OK + "\n" + FENCE + "\nLet me know.",
    "Sure. " + JSON_OK + " Done.",
    "<think>It might be {risky} but fine</think>\n" + JSON_OK,
    "<think>should_execute: false?</think>" + JSON_OK,
])
def test_cove_parser_reads_realistic_replies(reply):
    result = _verify(reply)
    assert result.should_execute is True and result.confidence == 0.9


@pytest.mark.parametrize("reply", [
    '{"should_execute": true, "confidence": "high"}',
    '{"should_execute": true, "confidence": null}',
    '{"should_execute": true}',
    '{"should_execute": false, "confidence": "0.9"}',
    '{"should_execute": true, "confidence": true}',
    "<think>ok</think>" + '{"should_execute": "true", "confidence": 0.9}',
])
def test_cove_parser_fails_closed_on_bad_confidence_or_verdict(reply):
    result = _verify(reply)
    assert result.should_execute is False and result.confidence == 0.0


def test_cove_parser_unexpected_error_fails_closed():
    verifier = _verifier(JSON_OK)
    with mock.patch.object(CoVeVerifier, "_extract_json_object", side_effect=RuntimeError("boom")):
        result = verifier._parse_verification_response(JSON_OK)
    assert result.should_execute is False and result.confidence == 0.0


# -- 5: non-dict suggested_modifications --------------------------------------------------------
@pytest.mark.parametrize("mods", [["x"], "use /safe"])
def test_non_dict_suggested_modifications_do_not_crash_the_step(mods):
    turns = [{"role": "assistant", "content": "", "tool_calls": [_call("c1", "read_file")]}]
    executed = []

    async def runner(messages, model=None, tools=None, **kw):
        if tools is None:
            return {"content": json.dumps({"should_execute": True, "confidence": 0.9,
                                           "suggested_modifications": mods})}
        return turns.pop(0) if turns else {"role": "assistant", "content": "final"}

    async def ex(call):
        executed.append(call["function"]["arguments"])
        return "ok"

    step = AgenticStepProcessor(objective="o", model_name="m", enable_cove=True, max_internal_steps=2)
    out = asyncio.run(step.run_async("go", TOOLS, runner, ex))
    assert out == "final" and executed == ['{"path": "/x"}']


# -- 7: the skip reply is also in internal_history -----------------------------------------------
def test_skipped_call_reply_is_in_internal_history():
    turns = [{"content": "go", "tool_calls": [_call("c1", "delete_file")]},
             {"content": "done", "tool_calls": []}]

    async def runner(messages, model=None, tools=None, **kw):
        if tools is None:
            return {"content": _verdict(False, 0.1)}
        return turns.pop(0)

    async def ex(call):
        return "ok"

    step = AgenticStepProcessor(objective="o", model_name="m", enable_cove=True, max_internal_steps=3)
    asyncio.run(step.run_async("go", TOOLS, runner, ex))
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1"
               and m["content"].startswith("skipped by CoVe:") for m in step.internal_history)
