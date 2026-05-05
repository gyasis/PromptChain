# PromptChain for LLMs

> **Audience:** an LLM coding agent (Claude Code, Cursor, etc.) that has never seen this package.
> **Purpose:** write working PromptChain code on the first try.
> **Source-of-truth:** the code in `promptchain/` v0.6.1+. When this doc and the code disagree, the code wins. File a fix in `docs/llms/FEEDBACK_LOG.md`.

---

## 1. Identity (one paragraph)

PromptChain is a Python library for building **multi-step LLM pipelines** and **multi-agent systems**. A pipeline is a `PromptChain` — an ordered list of *instructions* (string templates, Python functions, or self-contained agentic loops) executed left-to-right with the previous step's output as the next step's input. Multi-agent orchestration is a `AgentChain` — a router/pipeline/round-robin/broadcast wrapper around several `PromptChain` instances. The package wraps `litellm` for model access (any OpenAI/Anthropic/Gemini/etc. model name works as `"<provider>/<model>"`), supports MCP tool servers, has a structured event/callback system, and ships an MLflow observability layer that's a no-op when MLflow isn't installed.

---

## 2. The Public API — what `import promptchain` gives you

`promptchain/__init__.py` exports **9** names:

```python
from promptchain import (
    PromptChain,        # core pipeline class
    PromptEngineer,     # prompt design / variation utility
    load_prompts,       # bulk-load prompt templates from disk
    get_prompt_by_name, # fetch one prompt by name
    AsyncAgentInbox,    # async message inbox (multi-agent)
    PubSubBus,          # pub/sub message bus (multi-agent)
    JanitorAgent,       # background cleanup agent
    MemoStore,          # cross-agent memo / scratchpad store
    InterruptQueue,     # human-in-the-loop interrupt queue
)
```

### **Critical** — what is NOT in the top-level export

These are the **most-used** classes in real PromptChain code, but they are **not** re-exported from `promptchain`. You must import them from their submodules:

```python
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.execution_history_manager import ExecutionHistoryManager
from promptchain.utils.mcp_helpers import MCPHelper
from promptchain.observability import (
    track_llm_call, track_task, track_routing, track_session,
    init_mlflow, shutdown_mlflow, MLflowObserver,
)
from promptchain.cli.tools import (
    ToolRegistry, ToolExecutor, ToolCategory, SafetyValidator,
)
```

**Common mistake:** writing `from promptchain import AgentChain`. It will fail. Use the submodule path.

---

## 3. Core class: `PromptChain`

**Location:** `promptchain/utils/promptchaining.py:219`

### Minimal constructor

```python
from promptchain import PromptChain

chain = PromptChain(
    models=["openai/gpt-4o-mini"],   # one model used for all string instructions
    instructions=["Summarise this in 3 bullets:\n\n{input}"],
    verbose=True,
)
output = chain.process_prompt("Long article text here…")  # sync
# or
output = await chain.process_prompt_async("Long article text here…")  # async
```

### Full constructor signature (the parameters that matter)

| param | type | purpose |
|---|---|---|
| `models` | `List[str \| dict]` | One model per non-callable, non-agentic instruction. Pass a single-element list to reuse it for all. Dict form: `{"name": "openai/gpt-4o", "params": {"temperature": 0.5}}` |
| `instructions` | `List[str \| Callable \| AgenticStepProcessor \| ChainCall]` | Ordered steps. See §4 for the dispatch contract |
| `full_history` | `bool` | If True, each step receives the full chain history string instead of just the prior output |
| `store_steps` | `bool` | If True, populates `chain.step_outputs` (dict) without returning the full history |
| `verbose` | `bool` | Verbose stdout logging |
| `chainbreakers` | `List[Callable]` | Each `(step_no, current_output) -> (should_break, reason, final_output)`. Allows early termination |
| `mcp_servers` | `List[Dict]` | MCP server configs. Each dict needs `id`, `type` (`"stdio"`), and command/args |
| `additional_prompt_dirs` | `List[str]` | Where `PrePrompt` looks for named templates referenced by `instruction_id:strategy` strings |
| `enable_mcp_hijacker` | `bool` | Direct tool execution without LLM round-trip |

### Entry methods

- `process_prompt(initial_input: Optional[str] = None) -> str | List` — sync. Returns the final step's output (or full history list if `full_history=True`).
- `process_prompt_async(initial_input=None) -> str | List` — async. **Always prefer this in agent code** — most LLM calls are async under the hood.
- `register_tool_function(func: Callable)` — register a local Python function as a callable tool. The function's docstring + signature become the tool schema sent to the LLM.

### **Common mistakes**

- Passing `models=["gpt-4o"]` instead of `models=["openai/gpt-4o"]`. PromptChain delegates to `litellm` which **requires** the `provider/model` prefix.
- `models` length not matching the count of *non-callable, non-agentic* instructions. Either pass exactly that many models, or pass a single-element list (it auto-expands). Functions and `AgenticStepProcessor` instances do **not** consume a model slot.
- Forgetting `await` on `process_prompt_async`. There is no auto-coercion.

---

## 4. The Three Instruction Types

`PromptChain.instructions` is a heterogeneous list. The dispatcher (in `promptchaining.py`) decides what to do per element:

### (a) `str` — template instruction
The string is treated as a prompt template. `{input}` is substituted with the previous step's output. Sent to the model at the corresponding index in `self.models`.

```python
instructions = ["Translate to French:\n\n{input}"]
```

You can also use a *named prompt* via `PrePrompt`:
```python
instructions = ["my_prompt_id"]                          # loads template from prompt dirs
instructions = ["my_prompt_id:cot"]                      # template id : strategy modifier
```

### (b) `Callable` — Python function
Must accept a single string argument and return a string. **Does not consume a model slot.**

```python
def deduplicate(s: str) -> str:
    return "\n".join(dict.fromkeys(s.splitlines()))

chain = PromptChain(
    models=["openai/gpt-4o-mini"],   # one model for one string instruction
    instructions=[
        "List 10 ideas about {input}",   # uses the model
        deduplicate,                     # pure Python, no model
    ],
)
```

### (c) `AgenticStepProcessor` — internal agentic loop
A self-contained reasoning loop with its own LLM calls and tool access. **Does not consume a model slot in the parent chain** (it carries its own `model_name`).

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

agentic = AgenticStepProcessor(
    objective="Read the input, identify all named entities, return a JSON list",
    max_internal_steps=8,
    model_name="openai/gpt-4o",
)
chain = PromptChain(models=[], instructions=[agentic])  # no model needed
```

> Internal history of an `AgenticStepProcessor` is **isolated** — it never leaks to other agents in a multi-agent flow. Only the `final_answer` is exposed. See `agentic_step_processor.py:135-199` for the rationale (token-explosion prevention).

---

## 5. `AgenticStepProcessor` — the agentic loop

**Location:** `promptchain/utils/agentic_step_processor.py:135`

### Minimal use

```python
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain

agentic = AgenticStepProcessor(
    objective="Find all .py files, count their lines, return a markdown table",
    max_internal_steps=10,
    model_name="openai/gpt-4o-mini",
)
chain = PromptChain(models=["openai/gpt-4o-mini"], instructions=[agentic])
result = await chain.process_prompt_async("start")
```

### Important constructor knobs

| param | default | what it does |
|---|---|---|
| `objective` | required | The goal the loop tries to satisfy |
| `max_internal_steps` | `5` | Hard cap on internal LLM calls (prevents runaway) |
| `model_name` | inherits | Specific model for this loop |
| `history_mode` | `"minimal"` | `"minimal"` / `"progressive"` / `"kitchen_sink"`. **Progressive is recommended** for multi-hop reasoning |
| `step_timeout` | `120.0` | Seconds per LLM call |
| `enable_two_tier_routing` | `False` | Phase 1: route simple sub-tasks to `fallback_model` (cost saving) |
| `fallback_model` | `None` | The cheap model for two-tier routing, e.g. `"openai/gpt-4o-mini"` |
| `enable_blackboard` | `False` | Phase 2: structured state instead of growing chat history (~72% token reduction) |
| `enable_cove` | `False` | Phase 3: Chain-of-Verification — pre-execution tool-call validation |
| `cove_confidence_threshold` | `0.7` | Min confidence to execute a tool call when CoVe is on |
| `enable_checkpointing` | `False` | Phase 3: stuck-state detection + rollback |
| `enable_tao_loop` | `False` | Phase 4: explicit Think-Act-Observe phases |
| `enable_dry_run` | `False` | Phase 4: predict tool output before executing |
| `prompt_builder` | `None` | Spec 011: pluggable `BasePromptBuilder` strategy |
| `workflow_pattern` | `"standard"` | `"standard"` or `"react"` |

### History modes

- `"minimal"` — last assistant message + tool results only. Cheapest. **May be deprecated.**
- `"progressive"` — accumulate assistant messages + tool results progressively. **Recommended default for new code.**
- `"kitchen_sink"` — keep everything (debugging only).

### **Common mistakes**

- Passing `objective` as the *prompt* instead of the *goal*. The objective is "Find X and return Y", not "Hello, please find X."
- Not setting `max_internal_steps` and getting runaway loops.
- Enabling `enable_blackboard=True` without realising it changes the tool-result format the LLM sees — re-test prompts after toggling.

---

## 6. `AgentChain` — multi-agent orchestration

**Location:** `promptchain/utils/agent_chain.py:36`

### Four execution modes

| mode | behaviour |
|---|---|
| `"router"` | A router (LLM, dict config, or custom callable) picks ONE agent per turn. Agents may emit `[REROUTE] next_input` to chain to another agent (capped by `max_internal_steps`, default 3). |
| `"pipeline"` | All agents run sequentially, output of N feeds N+1 |
| `"round_robin"` | Cycles through agents, one per turn |
| `"broadcast"` | All agents run in parallel; results synthesized via `synthesizer_config` (REQUIRED for this mode) |

### Minimal router setup

```python
from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain

researcher = PromptChain(models=["openai/gpt-4o"], instructions=["Research: {input}"])
writer    = PromptChain(models=["openai/gpt-4o"], instructions=["Write a draft: {input}"])

chain = AgentChain(
    agents={"researcher": researcher, "writer": writer},
    agent_descriptions={
        "researcher": "Use when the user wants facts gathered.",
        "writer":     "Use when the user wants a draft written.",
    },
    router={
        "models": ["openai/gpt-4o-mini"],
        "instructions": [None, "Choose one of: researcher, writer. User: {input}"],
        "decision_prompt_template": "Pick the agent for: {input}. Reply only with the agent name.",
    },
    execution_mode="router",
    verbose=True,
)
result = await chain.process_input("I need a summary of climate change papers")
```

### Per-agent history config (v0.4.2+)

Pass `agent_history_configs` via `**kwargs`:

```python
agent_history_configs = {
    "terminal_agent": {"enabled": False},      # save 30-60% tokens
    "writer":         {"enabled": True, "max_tokens": 8000, "truncation_strategy": "keep_last"},
}
```

### **Common mistakes**

- Setting `execution_mode="broadcast"` without `synthesizer_config` — raises `ValueError`.
- Passing only `agents` without `agent_descriptions` (both required, keys must match exactly).
- Using `process_prompt_async` (that's `PromptChain`'s method). `AgentChain` uses `process_input(...)`.

---

## 7. Tool Registration

### Local tool functions
The simplest way: `chain.register_tool_function(my_func)`. The function's docstring + type hints become the tool schema. Used by both `PromptChain` and `AgenticStepProcessor`.

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}"

chain = PromptChain(models=["openai/gpt-4o"], instructions=["What's the weather in Paris?"])
chain.register_tool_function(get_weather)
```

### MCP tools (external servers)
Pass `mcp_servers` to the `PromptChain` constructor:

```python
chain = PromptChain(
    models=["openai/gpt-4o"],
    instructions=["…"],
    mcp_servers=[
        {"id": "filesystem", "type": "stdio", "command": "mcp-server-filesystem", "args": ["/tmp"]},
    ],
)
```

External MCP tools are auto-prefixed: a tool `read_file` from server `filesystem` becomes **`mcp_filesystem_read_file`** to prevent collisions with local tools.

### Heavy-duty: `ToolRegistry` + `ToolExecutor`
For CLI-quality tool execution with parameter validation, type coercion (LLM strings → typed args), safety validation (path traversal prevention), and performance metrics — see `tool_executor_demo.py`. Use this when building the *PromptChain CLI*; for normal chains the simpler `register_tool_function` is enough.

### **Common mistakes**

- Local tool function returns a non-string (dict, int, etc.). The LLM will receive it stringified by `repr()`. Either return a string or JSON-serialise inside the function.
- Naming a local tool the same as an MCP tool **without** the prefix — local wins, MCP is masked.

---

## 8. Observability (MLflow-backed, optional)

**Location:** `promptchain/observability/`

### What's exported

```python
from promptchain.observability import (
    track_llm_call, track_task, track_routing, track_session,
    init_mlflow, shutdown_mlflow, MLflowObserver,
)
```

### Behaviour

- **MLflow not installed → all decorators are no-ops.** Zero-overhead by design (the "ghost pattern").
- **MLflow installed → decorators auto-log** model, tokens, execution time, errors, custom params.
- Decorators use `ContextVars` for nested run tracking and a background queue for non-blocking writes.

### Enable

```python
from promptchain.observability import init_mlflow
init_mlflow()  # picks up .promptchain.yml or env vars (MLFLOW_TRACKING_URI etc.)
```

### Env knobs
- `PROMPTCHAIN_MLFLOW_BACKGROUND=1` — non-blocking queue
- `MLFLOW_TRACKING_URI` — where to send runs

### Decorate your own functions
```python
from promptchain.observability import track_llm_call

@track_llm_call(model_param="model")
async def my_call(prompt: str, model: str): ...
```

---

## 9. What NOT to Do (anti-pattern catalog)

This list is the closed-loop output: every Claude Code mistake observed via SIO becomes an entry here. Update via PR + a `FEEDBACK_LOG.md` row.

1. **`from promptchain import AgentChain`** — not exported. Use `from promptchain.utils.agent_chain import AgentChain`.
2. **`from promptchain import AgenticStepProcessor`** — not exported. Use `from promptchain.utils.agentic_step_processor import AgenticStepProcessor`.
3. **`models=["gpt-4o"]`** — missing provider prefix. Use `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet-20241022"`, `"gemini/gemini-2.5-pro"`, `"openai/gpt-4o-mini"`.
4. **`models` length mismatch.** Count only string instructions (functions and `AgenticStepProcessor` don't consume model slots). Or pass a single-element `models` list to auto-expand.
5. **Calling `chain.process_prompt_async(...)` without `await`.** The result is a coroutine, not a string.
6. **Calling `agent_chain.process_prompt_async(...)`** — `AgentChain` uses `process_input(...)`.
7. **`broadcast` mode without `synthesizer_config`** — raises `ValueError`.
8. **Tool function returns dict** — LLM sees stringified `repr()`. Return a string or `json.dumps(...)`.
9. **Local tool name collides with MCP tool** — local wins, MCP is masked silently.
10. **`AgenticStepProcessor` with `objective="hello, please do X"`** — the objective is a *goal* statement, not a conversational prompt. Write `"Find X and return Y as JSON"`.
11. **Setting `enable_blackboard=True` mid-development** — changes the tool-result format the LLM sees; re-validate prompts.
12. **Forgetting `await` on `MCPHelper.connect_mcp_async()`** — silent no-op, tools won't appear in the schema.

---

## 10. Worked End-to-End Example

```python
"""
Three-step PromptChain: agentic research → Python dedup → LLM polish.
Demonstrates all 3 instruction types in one chain.
"""
import asyncio
from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


def deduplicate(text: str) -> str:
    """Remove duplicate lines, preserve order."""
    return "\n".join(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


async def main():
    research = AgenticStepProcessor(
        objective="Find 10 distinct facts about the input topic. Return one fact per line.",
        max_internal_steps=8,
        model_name="openai/gpt-4o-mini",
        history_mode="progressive",
    )

    chain = PromptChain(
        models=["openai/gpt-4o"],     # one model for the one string instruction
        instructions=[
            research,                  # agentic step (no model slot)
            deduplicate,               # python function (no model slot)
            "Polish into 5 numbered bullet points:\n\n{input}",  # string (uses the model)
        ],
        verbose=True,
    )

    output = await chain.process_prompt_async("the history of the printing press")
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 11. Where to Look in the Code

When you need ground truth (this doc may have drifted), read in this order:

1. `promptchain/__init__.py` — what's actually exported
2. `promptchain/utils/promptchaining.py:219` — `PromptChain.__init__` signature
3. `promptchain/utils/agent_chain.py:36` — `AgentChain.__init__` signature
4. `promptchain/utils/agentic_step_processor.py:135` — `AgenticStepProcessor.__init__` and the **history-isolation contract** (lines 142-199 — read this; it explains why multi-agent chains don't blow up token budgets)
5. `promptchain/observability/__init__.py` — exact decorator names and stub fallbacks
6. `examples/two_tier_routing_demo.py` — the cleanest end-to-end example using all 4 phases
7. `tests/integration/test_011_library_consumer_flow.py` — the canonical "library consumer" pattern, kept current by CI

## 12. `ChainBuilder` — the agent-facing self-writing-chains API

**Source:** `promptchain/utils/chain_builder.py:40` (titled "Agent-Facing API for Self-Writing Chains").

This is the bridge to the watertight north-star: the API designed so that **the LLM itself** can construct, validate, version, and persist `ChainDefinition` objects via tool calls.

### Two surfaces

- **Fluent builder** — for human / direct code use: `ChainBuilder("name").add_prompt(...).add_chain(...).build()`
- **Static tool methods** — for LLM tool-call use: `ChainBuilder.create_chain(name, steps, ...)`, `.modify_chain(...)`, `.clone_chain(...)`. They return `dict` with `success`/`error` (no exceptions to recover from).

### Auto-registerable tool surface

```python
from promptchain.utils.chain_builder import (
    get_chain_builder_tools,        # OpenAI function-schema list
    get_chain_builder_functions,    # name → callable mapping
)
# Hand these to PromptChain.add_tools(...) + register_tool_function(...) and the LLM can self-author chains.
```

### Step types — only 4

`prompt` / `chain` / `function` / `agentic`. The `agentic` step REQUIRES `mode="hybrid"` (auto-switched by the builder if you forget).

### Versioning + validation

`modify_chain` auto-increments the patch version and re-validates via `ChainFactory.validate()` before save. New version is persisted; the old version stays intact.

For a copy-paste recipe see `recipes/recipe-chain-builder.md`.

---

## 13. Recent changes (mine before writing v0.6.x code)

When this doc was written the package was at **v0.6.1 (2026-04-28)**. The most recent two minor versions changed behaviour in ways that *will* trip up an agent if it works from older mental models. Read these before generating PromptChain code.

### v0.6.1 (2026-04-28) — library-consumer blocker fixes

1. **`import promptchain` no longer pulls in `textual`.** TUI users must `pip install "promptchain[tui]"` and explicitly import `promptchain.cli.tui.app.PromptChainApp` (or use the `promptchain` console-script). Plain library consumers do `pip install promptchain` (no textual).
2. **Schema/function name validation at registration.** Both `chain.add_tools()` and `chain.register_tool_function()` now raise `ValueError` immediately if a registered tool name has no matching function (or vice-versa). Previously this surfaced later as an opaque `Missing parameter 'tool_call_id'` error from OpenAI.
3. **Tool-dispatch loop bug fix (CRITICAL).** In `agentic_step_processor.py`, the per-tool execution block was wrongly indented to the same column as `for tool_call in tool_calls:` and ran ONCE per ACT phase regardless of how many `tool_calls` the LLM emitted. **If you're on `<0.6.1` and parallel tool calls behave weirdly, upgrade.** The fix also normalises `tool_call_id` extraction for both dict-style (OpenAI) and object-style (LiteLLM) responses.

### v0.6.0 (2026-04-19) — spec 011 prompt-builder decoupling (BREAKING)

1. **Default prompt is now dynamically built from the agent's actually-registered tools.** Previously `AgenticStepProcessor` shipped a hardcoded ReAct/TUI scaffold that advertised tools (`ripgrep_search`, `file_read`, `terminal_execute`) regardless of what was registered. Library consumers got a dishonest prompt. v0.6.0 ships `DynamicPromptGenerator` as the default; consumers see a prompt that references only their tools.
2. **Legacy ReAct scaffold preserved as opt-in.** Use `LegacyTUIPromptGenerator()` (`promptchain.prompts.legacy_tui`) for the old behaviour, OR use `TUIAgenticStepProcessor` if you want the old behaviour with no kwargs.
3. **New kwargs on `AgenticStepProcessor`:** `prompt_builder` (any `BasePromptBuilder` Protocol implementer) and `workflow_pattern` (`"standard"` or `"react"`). Library consumers who relied on the legacy scaffold MUST switch to one of the opt-in paths above.
4. **`instructions=` kwarg on `AgenticStepProcessor`** — for consumers who want to pass an already-rendered prompt instead of a builder.

### Earlier highlights

- **v0.4.2 (2025-10-07)** — orchestrator metrics (`OrchestratorSupervisor.get_last_execution_metrics()`); accurate `router_steps`/`tools_called`/token counts in `AgentExecutionResult`. Internal field rename: `router_steps` → `orchestrator_reasoning_steps` (non-breaking).

For the full log, read `CHANGELOG.md`.

---

## 14. Recipes

For copy-paste working snippets, see `docs/llms/recipes/`. Each recipe is a single-file pattern that has been tested against the current API.

| Recipe | When to use |
|---|---|
| `recipe-basic-chain.md` | One LLM step, string in / string out |
| `recipe-function-step.md` | Mix Python functions into a chain |
| `recipe-static-chain.md` | Pure-Python chain, zero LLM calls |
| `recipe-prompt-loader.md` | Load named prompts from disk via `PrePrompt` |
| `recipe-tool-calling-local.md` | Multiple local tools, the LLM loop, schema generation |
| `recipe-agentic-step.md` | Self-contained agentic loop with internal tools |
| `recipe-advanced-agentic.md` | Phase 1-4 cost/token/safety/transparency knobs |
| `recipe-multi-agent-router.md` | Multiple specialist agents, dynamic routing |
| `recipe-mcp-tool.md` | External MCP server tool integration |
| `recipe-observability-on.md` | Turn on MLflow tracking |
| `recipe-chain-builder.md` | `ChainBuilder` fluent + tool API for self-writing chains |

---

## 15. Where agent-authored scripts go

When the user asks the agent to **write and run** a PromptChain script, the script lands in `scripts/runs/<YYYY-MM-DD>_<short-name>/run.py` with a sibling `README.md`. See `scripts/README.md` for the convention. Run via `bash scripts/observe.sh runs/<folder>` to get MLflow tracking automatically.
