"""Offline control-flow tests for promptchain.patterns.agent_comms.

These use EchoAgent (no LLM) so the STATIC control flow — selection, FSM legality,
termination, gating, routing — is proven deterministically, for free. The model is
never exercised here; that's the point (control flow is not model-driven).
"""
from promptchain.patterns.agent_comms import (
    AccessibilityGate, EchoAgent, Orchestrator, as_agent, broadcast, by_capability,
    dyad, fsm, group, keyword_rule, manual, max_turns, pipeline, quorum,
    round_robin, router, sel_from_router, speakers, stop_when,
)


def _abc():
    return EchoAgent("A"), EchoAgent("B"), EchoAgent("C")


def test_round_robin_order():
    A, B, C = _abc()
    t = group([A, B, C], round_robin(), max_turns=4).run("go")
    assert [m.role for m in t if m.role != "Facilitator"] == ["A", "B", "C", "A"]


def test_quorum_stops_after_all_spoke():
    A, B, C = _abc()
    t = group([A, B, C], round_robin(), term=[quorum(3)], max_turns=20).run("go")
    # stops the moment the 3rd distinct speaker lands — exactly 3 turns, not 20
    assert len([m for m in t if m.role != "Facilitator"]) == 3
    assert speakers(t) == {"A", "B", "C"}


def test_max_turns_hard_cap():
    A, B, C = _abc()
    t = group([A, B, C], round_robin(), max_turns=2).run("go")
    assert len([m for m in t if m.role != "Facilitator"]) == 2


def test_fsm_only_legal_successors():
    A, B, C = _abc()
    # A->B->C->A only; round-robin within the (single) legal set == the cycle
    t = group([A, B, C], fsm({"A": ["B"], "B": ["C"], "C": ["A"]}, round_robin()),
              max_turns=4).run("go")
    assert [m.role for m in t if m.role != "Facilitator"] == ["A", "B", "C", "A"]


def test_manual_order_and_exhaustion():
    A, B, C = _abc()
    t = group([A, B, C], manual(["C", "A"]), max_turns=10).run("go")
    # manual list exhausts after 2 picks -> selector returns None -> loop ends
    assert [m.role for m in t if m.role != "Facilitator"] == ["C", "A"]


def test_stop_when_token():
    A = EchoAgent("A", script=["thinking", "we AGREE now"])
    B = EchoAgent("B", script=["hmm", "still unsure"])
    t = group([A, B], round_robin(), term=[stop_when("agree")], max_turns=10).run("go")
    assert "agree" in t[-1].content.lower()


def test_dyad_alternates():
    A, B, _ = _abc()
    t = dyad(A, B, max_turns=4)
    assert [m.role for m in t if m.role != "Facilitator"] == ["A", "B", "A", "B"]


def test_pipeline_carryover():
    A, B, C = _abc()
    out = pipeline([A, B, C]).run("seed")
    # each stage wraps the prior output -> nesting proves order A then B then C
    assert out.startswith("C(") and "A" not in out[:2]  # last transform is C's


def test_broadcast_fanout():
    A, B, C = _abc()
    replies, verdict = broadcast([A, B, C]).run("payload")
    assert [m.role for m in replies] == ["A", "B", "C"]
    assert verdict is None  # no synthesizer supplied


def test_nested_encapsulation_hides_inner():
    A, B, C = _abc()
    summ = EchoAgent("Sum")
    team = as_agent("Team", lambda: group([A, B, C], round_robin(), max_turns=3).run("x"), summ)
    outer = []
    team.say(outer)
    assert len(outer) == 1 and outer[0].role == "Team"   # only ONE line escapes


def test_router_keyword():
    A, B, C = _abc()
    route = router([keyword_rule(["data"], C), keyword_rule(["risk"], A)], fallback=B)
    assert route("what does the data say?") is C
    assert route("is there a risk?") is A
    assert route("hello") is B


def test_sel_from_router_in_group():
    A, B, C = _abc()
    route = router([keyword_rule(["A-0"], A)], fallback=B)  # matches A's first echo line
    # first speaker is agents[0]=A (echoes "A-0"); router then routes on that -> A again
    t = group([A, B, C], sel_from_router(route), max_turns=2).run("go")
    assert [m.role for m in t if m.role != "Facilitator"][0] in {"A", "B"}


def test_accessibility_gate():
    g = AccessibilityGate({"A": {"B"}, "B": {"A", "C"}, "C": {"B"}})
    assert g.can_reach("A", "B") is True
    assert g.can_reach("A", "C") is False


def test_orchestrator_full_authority():
    # agentic orchestrator: a scripted "manager" picks speakers, stops on DONE, synthesizes
    A, B, C = _abc()
    mgr = EchoAgent("MGR", ["A :: think", "B :: think", "DONE"])
    t = Orchestrator("Boss", manager=mgr, authority="full", max_rounds=10).run_group([A, B, C], "decide")
    assert [m.role for m in t if m.role in {"A", "B", "C"}] == ["A", "B"]  # manager chose A then B
    assert t[-1].role == "Boss"                                            # then wrote the synthesis


def test_orchestrator_select_authority_no_synthesis():
    A, B, C = _abc()
    mgr = EchoAgent("MGR", ["A", "DONE"])
    t = Orchestrator("Boss", manager=mgr, authority="select", max_rounds=10).run_group([A, B, C], "decide")
    assert [m.role for m in t if m.role in {"A", "B", "C"}] == ["A"]
    assert t[-1].role == "A"                                               # select => no synthesis line


def test_by_capability_static_routing():
    # static orchestrator picks the agent whose capabilities match the current topic
    A = EchoAgent("A", capabilities=["risk", "security"])
    B = EchoAgent("B", capabilities=["shipping", "launch"])
    C = EchoAgent("C", capabilities=["data", "metrics"])
    agents = [A, B, C]

    def first(goal):
        t = group(agents, by_capability(), max_turns=1).run(goal)
        return [m.role for m in t if m.role in {"A", "B", "C"}][0]

    assert first("We found a security vulnerability") == "A"
    assert first("What does the data show?") == "C"
    assert first("Time to plan the launch") == "B"
    assert first("Hello everyone") == "A"            # no match -> round-robin fallback (first)


def test_orchestrator_on_turn_hook():
    # on_turn fires once per completed subagent turn (used to stream to a UI / TUI dock)
    A, B, C = _abc()
    mgr = EchoAgent("MGR", ["A :: x", "B :: x", "DONE"])
    seen = []
    Orchestrator("Boss", manager=mgr, authority="steer", max_rounds=10,
                 on_turn=lambda msg, ctx: seen.append(msg.role)).run_group([A, B, C], "go")
    assert seen == ["A", "B"]


def test_group_run_async():
    # run_async runs on the current event loop (for TUIs); same result as sync run
    import asyncio
    A, B, C = _abc()
    t = asyncio.run(group([A, B, C], round_robin(), max_turns=3).run_async("go"))
    assert [m.role for m in t if m.role in {"A", "B", "C"}] == ["A", "B", "C"]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
