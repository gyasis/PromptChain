# `promptchain.patterns.agent_comms`

Every way agents talk to each other — dyad, pipeline, group, broadcast, routing,
FSM, nesting. **Agentic participants, static orchestration:** each *agent* is a real
**`AgenticStepProcessor`** (its own internal reason-act loop, can call tools), while a
plain **Callable** — never a model — decides who-speaks-next and when-to-stop. That is
the correct split: agentic where it belongs (the agents), static where it belongs (the
control flow). Note `feedback-static-not-agentic-promptchain`: "static" is scoped to one
experiment — *PromptChain does BOTH static and agentic, both first-class.*

> This is the "graduated" home of the patterns prototyped in `twicedata_intra`'s
> `secretary_mesh`. There, the workers were raw `urllib` callers; here they are real
> `PromptChain` agents and the loop is the real `ExternalLoop`.

## The primitive

```python
from promptchain.patterns.agent_comms import agent, group, round_robin, quorum

A = agent("A", "a cautious engineer")          # an agent = ONE line
B = agent("B", "a ship-it product lead")
C = agent("C", "an evidence-driven researcher")

t = group([A, B, C], round_robin(), term=[quorum(3)]).run("Ship Friday?")
for m in t:
    print(m)                                    # A: … / B: … / C: …
```

## Taxonomy → construct

| G | Pattern | Construct | One-liner |
|---|---|---|---|
| G1 | Dyad (1:1) | `dyad(a, b)` | `dyad(A, B, "Ship Friday?")` |
| G2 | Sequential pipeline + carryover | `pipeline(agents)` | `pipeline([A,B,C]).run("idea…")` |
| G3 | Moderated group | `group(...)` | `group([A,B,C], round_robin()).run("goal")` |
| G4 | Speaker-selection menu | `round_robin` · `random_pick` · `manual` · `llm_auto` · `custom` | `group(agents, llm_auto(judge))` |
| G5 | FSM transition graph | `fsm(graph, inner)` | `fsm({"A":["B"],"B":["C"],"C":["A"]}, round_robin())` |
| G6 | Broadcast (fan-out/in) | `broadcast(agents, synth)` | `broadcast([A,B,C], S).run("payload")` |
| G7 | Handoff / condition router | `router(rules, fallback)` + `keyword_rule` / `cond_rule` / `llm_rule` / `sel_from_router` | `router([keyword_rule(["data"], C)], B)` |
| G8 | Nested encapsulation | `as_agent(name, inner_run, summarizer)` | wraps a whole group as ONE agent |
| G9 | Shared blackboard | `MeshContext` | `ctx.vars["days_left"] = 2` |
| G10 | Step-clock loop | `ExternalLoop` (drives `Group`) | bounded, un-disableable guard |
| G11 | Accessibility gate | `AccessibilityGate` | `gate.can_reach("A", "C")` |
| G12 | Termination / loop-break | `max_turns` · `quorum` · `stop_when` · `jaccard_repeat` | `term=[quorum(3)]` |

## Selector menu (G4) — who speaks next

```python
from promptchain.patterns.agent_comms import round_robin, random_pick, manual, llm_auto, custom

round_robin()               # strict cyclic order
random_pick(seed=0)         # random, excluding the current speaker
manual(["C", "A", "B"])     # explicit order; ends the loop when exhausted
by_capability()             # STATIC: route to the agent whose declared capabilities fit the topic
llm_auto(judge)             # a judge agent NAMES the next speaker  (only model-driven one)
custom(fn)                  # any fn(last, agents, transcript, ctx) -> agent | None
```

Only `llm_auto` calls a model, and it is **async** (the group engine awaits it inside
the `ExternalLoop`). All others are pure and free.

## Routing (G7)

```python
from promptchain.patterns.agent_comms import router, keyword_rule, cond_rule, sel_from_router

route = router([keyword_rule(["data"], C), keyword_rule(["risk"], A)], fallback=B)
route("what does the data say?")            # -> C   (standalone handoff)
group(agents, sel_from_router(route))       # -> use the router AS a group selector
```

Rules are tried in priority order. `keyword_rule` / `cond_rule` are model-free (safe
inside a group loop); `llm_rule` calls a model synchronously — standalone use only.

## Termination (G12)

```python
term = [quorum(3)]                          # stop once all 3 have spoken
term = [max_turns(6), stop_when("AGREE")]   # …or after 6 turns, or on a token
term = [jaccard_repeat(0.85)]               # …or when an agent repeats itself
group(agents, round_robin(), term=term, max_turns=20)   # max_turns = hard ExternalLoop cap
```

## Nested encapsulation (G8)

```python
from promptchain.patterns.agent_comms import as_agent, group, round_robin

summ = agent("Σ", "summarizes a sub-team into one recommendation")
team = as_agent("Team", lambda: group([A, B, C], round_robin(), max_turns=3).run("x"), summ)
# `team` composes like any agent — outsiders see only its one-line summary.
```

## Orchestrators — static vs agentic (match to your agents)

The orchestrator (the "manager" that runs a group) is a tunable choice — the same
static-vs-agentic axis, one level up. **Spend the intelligence where the participants
don't have it:**

| Participants | Orchestrator |
|---|---|
| capable agents (strong models) | static `group()` — or `orchestrator(authority="select")` |
| mixed / a few bounded tasks | `orchestrator(authority="steer")` |
| dumb agents (small / 1B) | smart: strong `model`, `authority="full"`, higher `max_internal_steps` |

Both orchestrators route by **capability**: give agents `capabilities=[...]` and the
static `by_capability()` selector picks by skill-match, while the agentic manager knows
each agent's capabilities and — when a turn **raises a question but addresses no one** —
forwards that open question to whoever can best answer it.

```python
from promptchain.patterns.agent_comms import group, by_capability, quorum, orchestrator, captain

A = agent("A", "security eng", capabilities=["security", "risk"])
C = agent("C", "researcher", capabilities=["data", "churn"], tools=[lookup_metric])

# STATIC orchestrator — capability-aware, deterministic, no model in the loop:
group([A, B, C], by_capability(), term=[quorum(3)]).run("We found a vulnerability")

# AGENTIC orchestrator — a manager AGENT (ASP) runs the group (AG2 GroupChatManager):
boss = orchestrator("Boss", model="openai/gpt-4o", authority="steer")  # smart, for dumb agents
t = boss.run_group([A, B, C], "what's our churn, and is Friday risky?")  # open Q -> routed to C
#   authority: "select" (pick next speaker) | "steer" (+ per-turn directive) | "full" (+ synthesis)

# CAPTAIN — a manager that runs a sub-team and presents as ONE agent (AG2 CaptainAgent, G8):
team_lead = captain("Legal", [counsel_a, counsel_b], goal="Assess the contract risk.")
group([team_lead, product, eng], round_robin()).run("Approve the deal?")
```

The manager is itself a true `AgenticStepProcessor`; its `model` + `max_internal_steps`
ARE the "orchestrator intelligence" knob. `Orchestrator.run_group_async` exists for
nesting inside another loop (a `captain` used in an outer agentic group).

## Testing without a model

`EchoAgent` is a deterministic, no-LLM agent — use it to test control flow for free:

```python
from promptchain.patterns.agent_comms import EchoAgent, group, round_robin
A, B, C = EchoAgent("A"), EchoAgent("B"), EchoAgent("C")
t = group([A, B, C], round_robin(), max_turns=4).run("go")
assert [m.role for m in t if m.role != "Facilitator"] == ["A", "B", "C", "A"]
```

See `tests/test_agent_comms.py` (13 offline tests) and `examples/agent_comms_demo.py`
(live, real `PromptChain` workers on `gpt-4o-mini`).

## Design mandate

Static, non-agentic — no `AgenticStepProcessor` for control flow. Selectors,
routers, FSM edges, and stop predicates are Callables / boolean forks; the only model
calls are (a) an agent generating its line and (b) the optional `llm_auto` selector.
"""
