"""Agent-communication patterns for PromptChain — agentic participants, static mesh.

Every way agents talk to each other. Each *agent* is a real
:class:`~promptchain.utils.agentic_step_processor.AgenticStepProcessor` (its own
internal reason-act loop, can call tools); each *pattern* is a few lines over
``say``/``respond``, with a **Callable** — never a model — deciding who-speaks-next and
when-to-stop. R-PC1/R-PC2 governs the mesh, not the agents; PromptChain does BOTH static
and agentic, both first-class.

Taxonomy → construct
--------------------
======  =========================  =================================================
G       pattern                    construct here
======  =========================  =================================================
G1      dyad (1:1)                 :func:`dyad`  (= a group over two agents)
G2      sequential pipeline        :func:`pipeline`
G3      moderated group            :func:`group` / :class:`Group`
G4      speaker-selection menu     :func:`round_robin` :func:`random_pick`
                                    :func:`manual` :func:`llm_auto` :func:`custom`
G5      FSM transition graph       :func:`fsm`
G6      broadcast (fan-out/in)     :func:`broadcast`
G7      handoff / condition router :func:`router` (+ :func:`keyword_rule`,
                                    :func:`cond_rule`, :func:`llm_rule`,
                                    :func:`sel_from_router`)
G8      nested encapsulation       :func:`as_agent`
G9      shared blackboard          :class:`MeshContext`
G10     step-clock loop            :class:`ExternalLoop` (drives :class:`Group`)
G11     accessibility gate         :class:`AccessibilityGate`
G12     termination / loop-break   :func:`max_turns` :func:`quorum`
                                    :func:`stop_when` :func:`jaccard_repeat`
======  =========================  =================================================

Quick start
-----------
    from promptchain.patterns.agent_comms import agent, group, round_robin, quorum

    A = agent("A", "a cautious engineer")
    B = agent("B", "a ship-it product lead")
    C = agent("C", "an evidence-driven researcher")

    t = group([A, B, C], round_robin(), term=[quorum(3)]).run("Ship Friday?")
    for m in t: print(m)
"""
from promptchain.utils.external_loop import ExternalLoop  # G10 (re-exported)

from .agents import DEFAULT_MODEL, EchoAgent, LLMAgent, agent
from .orchestrator import Orchestrator, captain, orchestrator
from .patterns import Group, as_agent, broadcast, dyad, group, pipeline
from .routing import cond_rule, keyword_rule, llm_rule, router, sel_from_router
from .selectors import (by_capability, custom, fsm, llm_auto, manual,
                        random_pick, round_robin)
from .termination import jaccard_repeat, max_turns, quorum, stop_when
from .types import AccessibilityGate, MeshContext, Msg, render, speakers

__all__ = [
    # agents
    "agent", "LLMAgent", "EchoAgent", "DEFAULT_MODEL",
    # patterns
    "group", "Group", "dyad", "pipeline", "broadcast", "as_agent",
    # orchestrators (static group() above; agentic manager agent here)
    "orchestrator", "Orchestrator", "captain",
    # selectors (G4) + fsm (G5) + capability-based (static routing)
    "round_robin", "random_pick", "manual", "llm_auto", "custom", "fsm", "by_capability",
    # routing (G7)
    "router", "keyword_rule", "cond_rule", "llm_rule", "sel_from_router",
    # termination (G12)
    "max_turns", "quorum", "stop_when", "jaccard_repeat",
    # types (G9 / G11) + loop (G10)
    "Msg", "MeshContext", "AccessibilityGate", "render", "speakers", "ExternalLoop",
]
