---
id: recipe-cove-verifier
title: Chain of Verification for tool calls (CoVe)
when: An AgenticStepProcessor may call tools with side effects (delete, write, send) and each call should be vetted by a model before it runs.
api_version: 0.10.0+
---

# Chain of Verification (CoVe)

CoVe asks a model to review **a tool call** before it executes: what it assumes, what can go
wrong, how confident it is. Low-confidence calls do not run.

## What CoVe is not

CoVe checks **tool calls only** (tool name, arguments, schema, a little context). It does not
look at final answers, text output or code. To check that generated code works, run its
tests (`recipe-test-loop`). To gate a text answer on a measured probability, use
`decision()` (`recipe-decision-step`).

## Which one?

| Need | Use |
|---|---|
| Vet every tool call an agentic step makes | `AgenticStepProcessor(enable_cove=True, ...)` |
| Vet a tool call in your own loop or executor | `CoVeVerifier(...).verify_tool_call(...)` |
| Vet generated code | Not CoVe. `MicroPromptChain` or `RalphChain` (`recipe-test-loop`) |
| Vet a text answer | Not CoVe. `decision()` + `route_on()` (`recipe-decision-step`) |

## Safety: what happens when the verifier fails

`CoVeVerifier` never raises for runner or parse errors. A `None` context or tools list, or
tool arguments that `json.dumps` cannot serialise, still raise `TypeError`.

| Case | `should_execute` | `confidence` |
|---|---|---|
| `llm_runner` raises | `True` | `0.5` |
| Reply has no JSON object; no risk words in it | `True` | `0.4` |
| Reply has no JSON object; contains "do not execute", "should not", "dangerous", "risky" or "abort" | `False` | `0.4` |
| Malformed verdict (below), or an unexpected parse error | `False` | `0.0` |

A reply is well formed when it holds a JSON object with a real boolean `should_execute` and a
finite numeric `confidence`. The parser finds the object in plain JSON, in a fenced block, after
prose, or after a `<think>` block. It rejects a string verdict such as `"false"`, a missing or
non-numeric `confidence` (`"high"`, `null`, a string), `NaN`, `Infinity` and a non-object reply.

Inside `AgenticStepProcessor` a call runs only if `should_execute` is true **and**
`confidence >= cove_confidence_threshold`. With the default threshold `0.7`, the step skips every
row above, so it fails closed. Standalone, the two low-confidence `True` rows (a runner
error, a prose reply with no risk words) approve unless you compare `confidence` too. Lower
thresholds reopen the step:

- At `0.5` or below, a verifier that raises lets calls through.
- At `0.4` or below, prose replies with no risk words do too.

Standalone, you apply the threshold yourself. If you check only `should_execute`, an outage approves everything.
The step merges the verifier model's `suggested_modifications` over the arguments of an approved
call with no validation, so a verifier reply can change what runs. Treat CoVe as a risk reducer, not a
security boundary. Put hard limits (allow-lists, read-only credentials) in the tool itself.

## Offline example

```python
import asyncio, json
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.verification import CoVeVerifier

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "delete_file", "description": "Delete a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
]

def tool_call(call_id, name, path):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps({"path": path})}}

agent_turns = [
    {"content": "Checking both files.", "tool_calls": [
        tool_call("c1", "read_file", "/etc/app.json"),
        tool_call("c2", "delete_file", "/data/backup.db")]},
    {"content": "Done.", "tool_calls": []},
]

async def llm_runner(messages, model=None, tools=None, tool_choice=None, **kw):
    # One runner serves the agent and the verifier. Agent calls pass tools=; verifier calls do not.
    if tools is not None:
        return agent_turns.pop(0)
    prompt = messages[0]["content"]
    risky = "Name: delete_file" in prompt
    return {"content": json.dumps({
        "should_execute": not risky, "confidence": 0.2 if risky else 0.95,
        "assumptions": [], "risks": ["irreversible"] if risky else [],
        "reasoning": "destructive" if risky else "read-only",
        "suggested_modifications": None})}

executed = []

async def tool_executor(call):
    executed.append(call["function"]["name"])
    return "ok"

async def main():
    step = AgenticStepProcessor(
        objective="Inspect the config and clean up the backup",
        model_name="ollama_chat/glm-5.3-flash:cloud",
        max_internal_steps=3,
        enable_cove=True,
        cove_confidence_threshold=0.7,       # skip calls below this
        verifier_context="read_file is safe; every tool is audited.",   # added to the verifier prompt
    )
    await step.run_async("start", TOOLS, llm_runner, tool_executor)
    print(executed)                          # ['read_file']  (delete_file was skipped)

    # Standalone: you decide what to do with the result.
    verifier = CoVeVerifier(llm_runner, model_name="ollama_chat/glm-5.3-flash:cloud")
    result = await verifier.verify_tool_call(
        tool_name="delete_file", tool_args={"path": "/data/backup.db"},
        context="User asked to clean up temp files", available_tools=TOOLS)
    print(result.should_execute, result.confidence, result.risks)   # False 0.2 ['irreversible']

    # Fail-open check: a runner that raises still returns a result.
    async def broken(messages, model=None, **kw):
        raise RuntimeError("provider down")
    bad = await CoVeVerifier(broken, model_name="x").verify_tool_call(
        "delete_file", {"path": "/x"}, "ctx", TOOLS)
    print(bad.should_execute, bad.confidence)                        # True 0.5

asyncio.run(main())
```

## Contract

| Item | Value |
|---|---|
| `AgenticStepProcessor(..., enable_cove=False, cove_confidence_threshold=0.7, verifier_context=None, fallback_model=None)`; `fallback_model` is a string or `{"name": ..., "params": {...}}` | CoVe is off by default. `verifier_context` has no effect when `enable_cove=False`. |
| Verifier model | The step builds the verifier on first use with `fallback_model` if set, else the step's `model_name`, and passes it as `model=` to the step's `llm_runner`. A `PromptChain` runner honours `model=`, so a cheaper `fallback_model` serves the verifier. A bare string keeps the step's `model_params` only when its provider prefix matches the step's model. On another provider the chain logs a warning and drops `api_base`, `api_key` and the other connection keys. To use a model on another provider, give it its own settings: `fallback_model={"name": "ollama_chat/glm-5.3-flash:cloud", "params": {"temperature": 0}}` (add `api_base` / `api_key` there when it needs them). The dict form also applies to two-tier routing. |
| `CoVeVerifier(llm_runner, model_name, extra_context=None)` | `llm_runner` is an async callable invoked as `llm_runner(messages=[...], model=model_name)`. It returns a dict with `"content"` or an object with `.content`. |
| `await verifier.verify_tool_call(tool_name, tool_args, context, available_tools)` | All four args need a value. `available_tools` is the list of tool schemas; for a missing tool name the verifier logs a warning and uses an empty schema. It cuts context to 2000 chars. |
| Returns `VerificationResult` | `should_execute: bool`, `confidence: float` (clamped 0..1), `assumptions: list`, `risks: list`, `verification_reasoning: str`, `suggested_modifications: dict or None`. |
| Reply format the verifier model must give | JSON with `should_execute`, `confidence`, `assumptions`, `risks`, `reasoning`, `suggested_modifications`. The object may sit in a fenced block, after prose or after a `<think>` block. |
| `suggested_modifications` | Used only when it is a JSON object; a list or string becomes `None`. In an agentic step, an approved call has the object merged over its arguments. |
| `verifier_context` / `extra_context` | Text added to the verifier prompt under `## Step Context`. Use it to say why a curated tool list is safe; without it the verifier over-rejects hand-vetted calls. |

The step drops a skipped call from the round and, when `enable_blackboard=True`, notes it on the
blackboard as "Skipped <tool>: ...". The step answers the skipped call with a tool
message, `skipped by CoVe: <reason>`, so every `tool_call_id` has a reply and the model sees why
the call did not run. The step logs `[CoVe] Verified N/M tool calls` with the real counts.

## Common mistakes

- **Using CoVe to check an answer or generated code.** It sees tool calls only.
- **A threshold at or below the fallback confidence.** `cove_confidence_threshold=0.5` approves calls when the verifier is down (see "what happens when the verifier fails").
- **Checking only `should_execute` standalone.** Compare `confidence` to your threshold too.
- **A `llm_runner` that cannot take `messages=` and `model=` keywords.** The verifier calls it with exactly those. The agent's own runner usually already does; a custom one needs `**kwargs`.
- **Expecting the agent to see the full verdict.** It sees only `skipped by CoVe: <reason>`. The log and the blackboard hold the rest.
- **No `verifier_context` on a curated tool list.** The verifier sees a tool name and args, not the objective, and rejects calls that are in fact safe.
- **A bare `fallback_model` string on another provider.** The chain drops the step's `api_base` and `api_key`. The call may then fail, and CoVe skips every tool call. Pass the dict form with its own `params`.
- **A `fallback_model` that cannot judge tool calls.** The verifier uses it first, so pick a model that can follow the JSON reply format.
