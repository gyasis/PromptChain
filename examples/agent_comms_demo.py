#!/usr/bin/env python3
"""Live demo of promptchain.patterns.agent_comms with real PromptChain workers.

Run: OPENAI_API_KEY=... python3 examples/agent_comms_demo.py
Every "agent" is one line; every pattern is a few lines. A Callable decides flow.
"""
from promptchain.patterns.agent_comms import (
    Msg, agent, broadcast, dyad, fsm, group, keyword_rule, llm_auto, pipeline,
    quorum, round_robin, router, sel_from_router,
)

def lookup_metric(metric: str) -> str:
    """Return the current value of a named business metric (churn, mrr)."""
    return {"churn": "7.3%", "mrr": "$42k"}.get(metric.strip().lower(), "unknown")


A = agent("A", "a cautious risk-focused engineer")
B = agent("B", "an optimistic ship-it product lead")
C = agent("C", "an evidence-driven researcher", tools=[lookup_metric])  # a TRUE agent w/ a tool


def show(title, transcript):
    print(f"\n━━━ {title} ━━━")
    for m in transcript:
        print("  ", m)


# G3+G4 — moderated group, round-robin, stop when all 3 weigh in
show("G3 group · round_robin · quorum(3)",
     group([A, B, C], round_robin(), term=[quorum(3)], max_turns=12).run("Ship Friday? Decide."))

# G4 — llm_auto: a judge names who speaks next
judge = agent("M", "a neutral facilitator")
show("G4 group · llm_auto",
     group([A, B, C], llm_auto(judge), term=[quorum(3)], max_turns=6).run("Ship Friday? Decide."))

# G5 — FSM: A->B->C->A only
show("G5 group · fsm(A->B->C->A)",
     group([A, B, C], fsm({"A": ["B"], "B": ["C"], "C": ["A"]}, round_robin()), max_turns=4).run("Ship Friday?"))

# G1 — dyad
show("G1 dyad(A,B)", dyad(A, B, "Ship Friday, yes or no?", max_turns=3))

# G7 — router picks ONE agent for a query, that agent answers
route = router([keyword_rule(["data", "evidence"], C), keyword_rule(["risk", "bug"], A)], fallback=B)
picked = route("What does the data say about churn?")
t = [Msg("Facilitator", "What does the data say about churn?")]
picked.say(t)
show("G7 router · 'data' -> C", t)

# G2 — pipeline (carryover)
print("\n━━━ G2 pipeline(A->B->C) ━━━")
print("  ", pipeline([A, B, C]).run("Idea: an app that reminds you to drink water."))

# G6 — broadcast + synthesizer
synth = agent("S", "a synthesizer who merges views into one verdict")
replies, verdict = broadcast([A, B, C], synth).run("In one line, is Friday realistic?")
show("G6 broadcast + synth", replies + [Msg("S", verdict)])

print("\ndone.")
