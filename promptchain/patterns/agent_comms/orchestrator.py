"""Agentic orchestrators — a MANAGER AGENT that runs a group.

Opt-in alternative to the static :func:`~promptchain.patterns.agent_comms.group`. The
orchestrator is itself a true agent (an ``AgenticStepProcessor``) whose model and
reasoning depth ARE the "orchestrator intelligence" knob.

**Match orchestrator intelligence to participant capability:**
    * dumb agents (small/1B models) → a SMART orchestrator: strong ``model``, higher
      ``max_internal_steps``, ``authority="steer"``/``"full"`` (it directs every turn).
    * capable agents → a SIMPLE orchestrator: cheap model, ``authority="select"`` — or
      skip the agentic orchestrator entirely and use the static ``group()``.

Authority levels (how much the manager does):
    * ``"select"`` — only picks the next speaker (+ detects DONE). Lightest.
    * ``"steer"``  — select + injects a per-turn directive telling the speaker what to
      focus on (this is how a smart orchestrator carries dumb agents).
    * ``"full"``   — steer + writes the final synthesis. Most authority.
"""
from __future__ import annotations

from typing import List, Optional

from promptchain.utils.external_loop import ExternalLoop

from .agents import DEFAULT_MODEL, LLMAgent
from .types import MeshContext, Msg, Transcript, render

_AUTHORITY = ("select", "steer", "full")
_DEFAULT_PERSONA = "a decisive facilitator who keeps the group on task and knows when it is done"


class Orchestrator:
    """A manager agent that runs a group (AG2 ``GroupChatManager('auto')``)."""

    def __init__(self, name: str, persona: str = _DEFAULT_PERSONA,
                 model: str = DEFAULT_MODEL, max_internal_steps: int = 3,
                 authority: str = "full", max_rounds: int = 8, manager=None,
                 on_turn=None):
        if authority not in _AUTHORITY:
            raise ValueError(f"authority must be one of {_AUTHORITY}")
        self.name = name
        self.authority = authority
        self.max_rounds = max_rounds
        # Optional per-completed-turn callback `on_turn(msg, ctx)` (mirrors Group) — used
        # to stream each subagent's turn to a UI live (e.g. the TUI /task dock).
        self.on_turn = on_turn
        # The manager is a true agent with an ORCHESTRATOR objective (it runs the team,
        # it does NOT do their work). `model`/`max_internal_steps` = its intelligence.
        self.manager = manager or LLMAgent(
            name, persona, model=model, max_internal_steps=max_internal_steps,
            one_sentence=False,
            objective=(f"You are {name}, {persona}. You ORCHESTRATE a team of agents "
                       f"toward a goal — you do NOT do their work yourself. Follow the "
                       f"specific instruction in each input exactly and return ONLY what "
                       f"it asks (a name, DONE, a directive, or a synthesis)."))

    # — shared setup: build the transcript + the per-round async step —
    def _prep(self, agents, goal, ctx):
        ctx = ctx or MeshContext(participants=[a.name for a in agents])
        by_name = {a.name: a for a in agents}
        # roster with capabilities → the manager routes by what each agent can DO
        roster = "; ".join(
            f"{a.name} ({', '.join(getattr(a, 'capabilities', []) or ['general'])})"
            for a in agents)
        t: Transcript = [Msg("Facilitator", goal)]
        steer = self.authority in ("steer", "full")

        async def step(it, st) -> bool:
            fmt = ("Reply EXACTLY as `NAME :: directive` (directive = the specific "
                   "question/task for that agent to address), OR `DONE`."
                   if steer else "Reply with the NAME of who speaks next, OR `DONE`.")
            decision = (await self.manager.respond_async(
                f"Team (name → capabilities): {roster}\nGoal: {goal}\n"
                f"Conversation so far:\n{render(t)}\n\n"
                f"Choose who speaks next:\n"
                f"- If the LAST message directly addresses a specific agent, pick that agent.\n"
                f"- If it raises a question or concern but leaves it UNADDRESSED (names no one), "
                f"pick the agent whose capabilities best fit ANSWERING that open question.\n"
                f"- If the goal is resolved, reply DONE.\n{fmt}")).strip()
            if "done" in decision.lower():
                return False
            name_part, _, directive = decision.partition("::")
            nxt = next((by_name[n] for n in by_name if n.lower() in name_part.lower()),
                       agents[0])
            if steer and directive.strip():
                t.append(Msg(f"{self.name}→{nxt.name}", directive.strip()))
            await nxt.say_async(t, ctx)
            if self.on_turn:
                self.on_turn(t[-1], ctx)
            return True

        return t, step

    def _synth_prompt(self, goal, t) -> str:
        return (f"Goal: {goal}\nDiscussion:\n{render(t)}\n\n"
                f"Write the final decision/synthesis in 1-2 sentences.")

    def run_group(self, agents, goal: str, ctx=None) -> Transcript:
        """Run the group synchronously (top-level use)."""
        t, step = self._prep(agents, goal, ctx)
        ExternalLoop(max_iters=self.max_rounds).run_sync(step, {})
        if self.authority == "full":
            t.append(Msg(self.name, self.manager.respond(self._synth_prompt(goal, t))))
        return t

    async def run_group_async(self, agents, goal: str, ctx=None) -> Transcript:
        """Async variant — use when the orchestrator runs INSIDE another loop (e.g. a
        :func:`captain` nested in an outer group), to avoid nesting event loops."""
        t, step = self._prep(agents, goal, ctx)
        await ExternalLoop(max_iters=self.max_rounds).run(step, {})
        if self.authority == "full":
            t.append(Msg(self.name, await self.manager.respond_async(self._synth_prompt(goal, t))))
        return t


def orchestrator(name: str, persona: str = _DEFAULT_PERSONA, model: str = DEFAULT_MODEL,
                 max_internal_steps: int = 3, authority: str = "full",
                 max_rounds: int = 8, manager=None, on_turn=None) -> Orchestrator:
    """An agentic orchestrator (manager agent). See :class:`Orchestrator`.

    >>> # dumb agents -> smart orchestrator that steers every turn:
    >>> orchestrator("M", model="openai/gpt-4o", max_internal_steps=6, authority="full")
    >>> # capable agents -> simple orchestrator:
    >>> orchestrator("M", authority="select")
    >>> # stream each turn to a UI:
    >>> orchestrator("M", on_turn=lambda msg, ctx: dock.add(f"{msg.role}: {msg.content}"))
    """
    return Orchestrator(name, persona, model, max_internal_steps, authority, max_rounds,
                        manager, on_turn)


def captain(name: str, team: List, goal: str,
            persona: str = "a captain who runs a sub-team and reports ONE recommendation",
            model: str = DEFAULT_MODEL):
    """G8 nested orchestrator (AG2 ``CaptainAgent``): a manager that RUNS a sub-team and
    presents to the outside world as ONE agent — only its synthesis escapes. Composes
    like any agent (has ``name`` / ``say`` / ``say_async``)."""
    orch = Orchestrator(name, persona, model=model, authority="full")

    class _Captain:
        def __init__(self):
            self.name = name

        def say(self, transcript: Transcript, ctx=None) -> str:
            s = orch.run_group(team, goal, ctx)[-1].content
            transcript.append(Msg(self.name, s))
            return s

        async def say_async(self, transcript: Transcript, ctx=None) -> str:
            s = (await orch.run_group_async(team, goal, ctx))[-1].content
            transcript.append(Msg(self.name, s))
            return s

    return _Captain()
