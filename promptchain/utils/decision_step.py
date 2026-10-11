"""Typed decision steps — a closed question answered with a MEASURED probability.

A decision step takes the chain's current text as *state*, asks typed questions about it
(`choice` / `boolean` / `score`), and gets back a real distribution plus a confidence from a
decision backend — not a confidence the model wrote about itself. `route_on` then branches the
chain on that confidence (accept / review / escalate) or on the chosen label.

Both are plain Callable steps, so they compose with zero framework change and consume no
model slot (R-PC2)::

    from promptchain.utils.decision_step import decision, route_on, SystemOneBackend

    triage = decision(
        {"dept": {"type": "choice", "instructions": "Which team owns this ticket?",
                  "criteria": {"billing": "charges, refunds", "technical": "bugs, errors"}}},
        backend=SystemOneBackend(model="tev1:4b"),
        name="triage",
    )
    route = route_on("dept", accept=0.9, review=0.6,
                     on_accept={"billing": billing_chain, "technical": tech_chain},
                     on_review=human_queue, on_escalate=escalate)
    chain = PromptChain(models=[], instructions=[triage, route])

Backends share one interface (`DecisionBackend.decide`) and one reply shape, so a chain swaps
backend by config alone:

- `SystemOneBackend` — Ollama >= 0.35 `/v1/systemone` (the Jev API served locally; real
  classifier heads such as `tev1:4b`). The measured default.
- `hunch_backend()` — the `hunch` CLI (next-token logprobs from an ordinary LM). Wrapped as a
  subprocess on purpose: improvements to `hunch` reach chains without a copy here.
- `CliBackend` — any CLI with the `<tool> - -q <json> --raw` contract.

The model is REQUIRED on every backend (R-PC4): there is no environment-variable default, so a
step can never silently run on a model nobody chose. Thresholds on `route_on` are required too:
calibration differs by task, so one global 0.9 does not give stable coverage.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Union

logger = logging.getLogger(__name__)

# The governed lane on the Mac Studio. Never raw :11434 — admission control lives here.
GOVERNED_OLLAMA = "http://192.168.0.159:8799/llm/ollama"

QUESTION_TYPES = ("choice", "boolean", "score")


class DecisionError(RuntimeError):
    """A decision could not be made or its answer is invalid. Never swallowed silently."""


class DecisionBackend(Protocol):
    name: str
    model: str

    async def decide(self, state: str, questions: Dict[str, dict]) -> dict:
        """Return ``{"model": str, "answers": {question: answer}}``."""
        ...


# --------------------------------------------------------------------------- validation


def validate_questions(questions: Mapping[str, dict]) -> None:
    """Reject a malformed question set before any call is made."""
    if not questions:
        raise DecisionError("decision needs at least one question")
    for qname, q in questions.items():
        kind = q.get("type")
        if kind not in QUESTION_TYPES:
            raise DecisionError(
                f"question {qname!r}: type {kind!r} is not one of {QUESTION_TYPES}"
            )
        crit = q.get("criteria")
        if kind == "choice" and not (isinstance(crit, dict) and len(crit) >= 2):
            raise DecisionError(
                f"question {qname!r}: a choice needs criteria as a dict of >= 2 labels"
            )
        if kind == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            raise DecisionError(
                f"question {qname!r}: a score needs criteria as a list of >= 2 levels"
            )


def _entropy_confidence(probs: List[float]) -> float:
    """1 - H(p)/log(n): 1.0 = all mass on one option, 0.0 = uniform. Same as hunch."""
    nz = [p for p in probs if p > 0]
    if len(probs) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in nz)
    return max(0.0, min(1.0, 1 - h / math.log(len(probs))))


def normalize_answer(qname: str, q: dict, ans: dict) -> dict:
    """Validate one backend answer against its question and fill the common fields.

    Every normalized answer carries ``type``, ``value`` (the chosen label / bool / score) and
    ``confidence``. An answer outside the question's option set is an error, not a guess.
    """
    if not isinstance(ans, dict):
        raise DecisionError(f"question {qname!r}: backend returned no answer")
    if "error" in ans:
        raise DecisionError(f"question {qname!r}: backend error: {ans['error']}")
    kind = q["type"]
    out = dict(ans)
    out["type"] = kind
    if kind == "choice":
        value = ans.get("choice")
        if value not in q["criteria"]:
            raise DecisionError(
                f"question {qname!r}: answer {value!r} is not one of "
                f"{list(q['criteria'])}"
            )
    elif kind == "boolean":
        value = ans.get("boolean")
        p = ans.get("probability", ans.get("noul"))
        if value is None and isinstance(p, (int, float)):
            value = p >= 0.5
        if not isinstance(value, bool):
            raise DecisionError(f"question {qname!r}: no boolean answer in {ans}")
        if "probabilities" not in out and isinstance(p, (int, float)):
            out["probabilities"] = {"True": p, "False": 1 - p}
    else:  # score
        value = ans.get("score")
        if not isinstance(value, (int, float)):
            raise DecisionError(f"question {qname!r}: no numeric score in {ans}")
    out["value"] = value
    if not isinstance(out.get("confidence"), (int, float)):
        probs = list((out.get("probabilities") or {}).values())
        if not probs:
            raise DecisionError(
                f"question {qname!r}: answer has neither confidence nor probabilities"
            )
        out["confidence"] = round(_entropy_confidence(probs), 4)
    return out


# --------------------------------------------------------------------------- backends


class SystemOneBackend:
    """Ollama ``/v1/systemone`` — purpose-trained decision models (tev1, nimble, clef)."""

    name = "systemone"

    def __init__(self, model: str, base_url: str = GOVERNED_OLLAMA, timeout: float = 300.0):
        if not model:
            raise DecisionError("SystemOneBackend needs an explicit model (R-PC4)")
        self.model = model
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.timeout = timeout

    def build_request(self, state: str, questions: Dict[str, dict]) -> dict:
        """The exact JSON body sent. Public so tests can assert what goes on the wire."""
        qs = json.loads(json.dumps(questions))
        for q in qs.values():
            # This endpoint spells the yes/no primitive `noul`; the Vercel gateway says `boolean`.
            if q.get("type") == "boolean":
                q["type"] = "noul"
        return {"model": self.model, "state": state, "questions": qs}

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.url, json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise DecisionError(f"systemone HTTP {e.code} ({self.model}): {detail}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise DecisionError(
                f"systemone ({self.model}): no answer from {self.url} within "
                f"{self.timeout:.0f}s ({type(e).__name__}); needs ollama >= 0.35"
            ) from e

    async def decide(self, state: str, questions: Dict[str, dict]) -> dict:
        body = self.build_request(state, questions)
        d = await asyncio.get_running_loop().run_in_executor(None, self._post, body)
        answers = {}
        for k, v in (d.get("answers") or {}).items():
            v = dict(v)
            if v.get("type") == "noul":
                p = v.get("noul")
                v["type"] = "boolean"
                if isinstance(p, (int, float)):
                    v["probability"] = p
                    v["boolean"] = p >= 0.5
            answers[k] = v
        return {"model": d.get("model", self.model), "answers": answers, "usage": d.get("usage")}


class CliBackend:
    """Any decision CLI with the shared contract: ``<cmd> - -q '<json>' --raw``, state on stdin,
    model selected through ``model_env``, JSON ``{"model", "answers"}`` on stdout."""

    def __init__(
        self,
        command: List[str],
        model: str,
        model_env: str,
        name: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
        timeout: float = 300.0,
    ):
        if not model:
            raise DecisionError(f"{command[0]} backend needs an explicit model (R-PC4)")
        self.command = list(command)
        self.model = model
        self.model_env = model_env
        self.name = name or os.path.basename(command[0])
        self.extra_env = dict(extra_env or {})
        self.timeout = timeout

    def build_invocation(self, questions: Dict[str, dict]) -> tuple:
        """(argv, env overrides) for one call. Public so tests can assert them."""
        argv = self.command + ["-", "-q", json.dumps(questions), "--raw",
                               "--timeout", str(self.timeout)]
        return argv, {self.model_env: self.model, **self.extra_env}

    async def decide(self, state: str, questions: Dict[str, dict]) -> dict:
        argv, overrides = self.build_invocation(questions)
        if shutil.which(argv[0]) is None:
            raise DecisionError(f"{argv[0]!r} is not on PATH")
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **overrides},
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(state.encode()), timeout=self.timeout + 30
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            raise DecisionError(f"{self.name} ({self.model}) timed out") from e
        if proc.returncode != 0:
            raise DecisionError(
                f"{self.name} ({self.model}) exited {proc.returncode}: "
                f"{err.decode(errors='replace').strip()[-300:]}"
            )
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            raise DecisionError(f"{self.name}: stdout was not JSON: {out[:200]!r}") from e


def hunch_backend(
    model: str, vote: int = 0, debias: bool = False, constrained: bool = False,
    timeout: float = 300.0,
) -> CliBackend:
    """The ``hunch`` CLI. Measured advice: if you threshold on it, use ``vote`` (e.g. 6) —
    plain hunch on small models reports confidence ~1.0 on nearly everything."""
    extra = {}
    if vote:
        extra["HUNCH_VOTE"] = str(vote)
    if debias:
        extra["HUNCH_DEBIAS"] = "1"
    if constrained:
        extra["HUNCH_CONSTRAINED"] = "1"
    mode = "-vote" if vote else "-debias" if debias else ""
    return CliBackend(["hunch"], model=model, model_env="HUNCH_MODEL",
                      name=f"hunch{mode}", extra_env=extra, timeout=timeout)


# --------------------------------------------------------------------------- steps


def decision(questions: Dict[str, dict], backend: DecisionBackend, name: str) -> Callable:
    """A Callable chain step: ask ``questions`` about the step input.

    Returns a JSON record (a string, as chain steps must) that keeps the original ``state`` so
    a following ``route_on`` can hand it to the chosen branch::

        {"decision": name, "backend": ..., "model": ..., "elapsed_s": ...,
         "state": <step input>, "answers": {q: {"type", "value", "confidence", ...}}}

    Any backend failure or out-of-set answer raises ``DecisionError`` and stops the chain.
    """
    validate_questions(questions)
    if backend is None:
        raise DecisionError("decision needs an explicit backend (R-PC4)")

    async def _decision_step(state: str) -> str:
        t0 = time.time()
        reply = await backend.decide(state, questions)
        raw = reply.get("answers") or {}
        missing = [q for q in questions if q not in raw]
        if missing:
            raise DecisionError(f"decision {name!r}: backend gave no answer for {missing}")
        answers = {q: normalize_answer(q, questions[q], raw[q]) for q in questions}
        record = {
            "decision": name,
            "backend": backend.name,
            "model": reply.get("model", backend.model),
            "elapsed_s": round(time.time() - t0, 3),
            "state": state,
            "answers": answers,
        }
        logger.info(
            "decision %s via %s/%s: %s",
            name, record["backend"], record["model"],
            {q: (a["value"], a["confidence"]) for q, a in answers.items()},
        )
        return json.dumps(record)

    _decision_step.__name__ = f"decision_{name}"
    return _decision_step


Branch = Union[Callable[..., Any], Any]  # a Callable step or a PromptChain


async def _run_branch(branch: Branch, state: str) -> str:
    if hasattr(branch, "process_prompt_async"):  # a nested PromptChain
        return str(await branch.process_prompt_async(state))
    out = branch(state)
    if inspect.isawaitable(out):
        out = await out
    return str(out)


def band(confidence: float, accept: float, review: float) -> str:
    """accept (>= accept) / review (>= review) / escalate (below review)."""
    if confidence >= accept:
        return "accept"
    if confidence >= review:
        return "review"
    return "escalate"


def route_on(
    question: str,
    *,
    accept: float,
    review: float,
    on_accept: Union[Branch, Mapping[Any, Branch]],
    on_review: Branch,
    on_escalate: Branch,
) -> Callable:
    """A Callable chain step that follows a ``decision`` step and branches on its answer.

    The confidence of ``question`` picks the band. In the accept band ``on_accept`` runs — or,
    if it is a mapping, the branch keyed by the chosen value (a router). Every branch receives
    the original state string. Thresholds are required: set them per task from that task's
    eval reliability table, never one global number.
    """
    if not 0 <= review <= accept <= 1:
        raise DecisionError(f"need 0 <= review ({review}) <= accept ({accept}) <= 1")

    async def _route_step(record_json: str) -> str:
        try:
            record = json.loads(record_json)
            ans = record["answers"][question]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise DecisionError(
                f"route_on({question!r}) must follow a decision step that asks it"
            ) from e
        which = band(ans["confidence"], accept, review)
        if which == "accept":
            if isinstance(on_accept, Mapping):
                key = ans["value"]
                if key not in on_accept:
                    raise DecisionError(
                        f"route_on({question!r}): no accept branch for {key!r}"
                    )
                branch = on_accept[key]
            else:
                branch = on_accept
        else:
            branch = on_review if which == "review" else on_escalate
        logger.info(
            "route_on %s: value=%r confidence=%s -> %s",
            question, ans["value"], ans["confidence"], which,
        )
        return await _run_branch(branch, record["state"])

    _route_step.__name__ = f"route_on_{question}"
    return _route_step
