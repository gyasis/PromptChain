"""Agents — each participant is a TRUE agent (an ``AgenticStepProcessor``).

An agent is NOT a single-shot LLM call. Each turn runs the agent's own internal
reason-act loop (``AgenticStepProcessor``, up to ``max_internal_steps`` LLM calls,
with tools) and returns only its final answer — the ASP's history is internally
isolated, so it never bloats the conversation.

This is the correct split (house note ``feedback-static-not-agentic-promptchain``:
static IS scoped to one experiment; *"PromptChain does BOTH static and agentic, both
first-class"*):
    * the MESH / control flow (who-speaks-next, when-to-stop) is STATIC — pure
      Callables in selectors.py / routing.py / termination.py. A model never drives it.
    * each PARTICIPANT is AGENTIC — a real agent per turn. That is what makes this a
      multi-AGENT system rather than a multi-prompt one.

Give an agent ``tools=[plain_func]`` and it genuinely acts: the schema is auto-derived
and the ASP can call the function inside its loop.

``EchoAgent`` is a deterministic, no-LLM stand-in so control flow is testable for free.
"""
from __future__ import annotations

import inspect
from typing import Callable, List, Optional

from promptchain import PromptChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

from .types import Msg, Transcript, render

DEFAULT_MODEL = "openai/gpt-4o-mini"


def _tool_schema(func: Callable) -> dict:
    """Auto-derive an OpenAI/litellm tool schema from a plain function's signature +
    docstring. Params are typed as strings (good enough for simple tools) — override by
    passing a full schema via ``add_tools`` yourself if you need richer types."""
    props, required = {}, []
    for pname, p in inspect.signature(func).parameters.items():
        props[pname] = {"type": "string", "description": pname}
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    desc = (func.__doc__ or func.__name__).strip().split("\n")[0]
    return {"type": "function", "function": {
        "name": func.__name__, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


class LLMAgent:
    """A true agent: an ``AgenticStepProcessor`` wrapped as a discussion participant."""

    def __init__(self, name: str, persona: str, model: str = DEFAULT_MODEL,
                 tools: Optional[List[Callable]] = None, max_internal_steps: int = 4,
                 one_sentence: bool = True, objective: Optional[str] = None,
                 capabilities: Optional[List[str]] = None):
        self.name = name
        self.persona = persona
        # Declared skills — used by capability-based selection (by_capability) and by an
        # agentic orchestrator to route an open question to whoever can best answer it.
        self.capabilities = list(capabilities or [])
        turn = ("Contribute ONE short in-character sentence as your turn."
                if one_sentence else "Contribute your turn.")
        caps = f" Your areas: {', '.join(self.capabilities)}." if self.capabilities else ""
        # A goal definition (not a conversational message) — per recipe-agentic-step.
        # `objective` overrides the default participant role (used to build a manager/
        # orchestrator agent, whose role is to run the group, not to participate).
        self.objective = objective or (
            f"You are {name}, {persona}.{caps} You are ONE participant in a multi-agent "
            f"discussion. Read the conversation provided as your input, reason "
            f"step-by-step internally (call a tool if it helps), then {turn} "
            f"Return ONLY your turn text — no name prefix, no narration, no quotes.")
        self.asp = AgenticStepProcessor(
            objective=self.objective,
            model_name=model,
            max_internal_steps=max_internal_steps,
            history_mode="progressive",   # better multi-hop reasoning than "minimal"
        )
        # ASP carries its own model → the parent chain needs no model slot.
        self.chain = PromptChain(models=[], instructions=[self.asp])
        self.tools = list(tools or [])
        for fn in self.tools:
            self.chain.add_tools([_tool_schema(fn)])   # schema → the model SEES the tool
            self.chain.register_tool_function(fn)       # implementation the loop calls

    def _clean(self, raw: str) -> str:
        s = raw.strip().replace("\n", " ")
        prefix = f"{self.name}:"                         # strip a self-echoed "Name:" prefix
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
        return s

    # — conversational turn (reads transcript, runs the agent, appends its answer) —
    def say(self, transcript: Transcript, ctx=None) -> str:
        reply = self._clean(self.chain.process_prompt(render(transcript)))
        transcript.append(Msg(self.name, reply))
        return reply

    async def say_async(self, transcript: Transcript, ctx=None) -> str:
        reply = self._clean(await self.chain.process_prompt_async(render(transcript)))
        transcript.append(Msg(self.name, reply))
        return reply

    # — single-shot response to a raw string (pipeline / broadcast) —
    def respond(self, text: str) -> str:
        return self._clean(self.chain.process_prompt(text))

    async def respond_async(self, text: str) -> str:
        return self._clean(await self.chain.process_prompt_async(text))


def agent(name: str, persona: str, model: str = DEFAULT_MODEL,
          tools: Optional[List[Callable]] = None, max_internal_steps: int = 4,
          capabilities: Optional[List[str]] = None, **kw) -> LLMAgent:
    """A true agent in one line::

        A = agent("A", "a cautious engineer", capabilities=["risk", "security"])
        C = agent("C", "a researcher", tools=[lookup_metric])   # give it real tools
    """
    return LLMAgent(name, persona, model=model, tools=tools,
                    max_internal_steps=max_internal_steps, capabilities=capabilities, **kw)


class EchoAgent:
    """Deterministic, no-LLM agent for testing control flow.

    ``script`` cycles canned replies per turn; default ``"<name>-N"``. Same
    say/say_async/respond surface as :class:`LLMAgent`.
    """

    def __init__(self, name: str, script: Optional[List[str]] = None,
                 capabilities: Optional[List[str]] = None):
        self.name = name
        self.capabilities = list(capabilities or [])
        self._script = list(script or [])
        self._i = 0

    def _next(self) -> str:
        v = self._script[self._i % len(self._script)] if self._script else f"{self.name}-{self._i}"
        self._i += 1
        return v

    def say(self, transcript: Transcript, ctx=None) -> str:
        v = self._next()
        transcript.append(Msg(self.name, v))
        return v

    async def say_async(self, transcript: Transcript, ctx=None) -> str:
        return self.say(transcript, ctx)

    def respond(self, text: str) -> str:
        return f"{self.name}({self._next()})"

    async def respond_async(self, text: str) -> str:
        return self.respond(text)
