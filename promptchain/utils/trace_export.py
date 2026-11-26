"""TRL-shaped training-data export from PromptChain runs.

``TraceRecorder`` is an execution callback that pairs each MODEL_CALL_START
event with its MODEL_CALL_END event and keeps one record per model call:

    {"prompt": [messages], "completion": [assistant message],
     "tools": [...], "model": str, "step": int}

``records_from_agentic`` turns the full internal history of an
``AgenticStepProcessor`` into one conversational ``messages`` record, because
agentic steps emit no MODEL_CALL events. ``write_jsonl`` writes records one
JSON object per line, which ``datasets.load_dataset("json", ...)`` reads.

The recorder reads the ``prompt_messages``, ``tools``, ``response_message`` and
``trace_call_id`` metadata keys. ``PromptChain`` fills them only while a
callback is registered.
"""

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

logger = logging.getLogger(__name__)

RewardFn = Callable[[dict], Optional[float]]

REDACTED = "***REDACTED***"

# Message keys that survive export. Everything else (usage, refusal,
# reasoning_content, ...) is dropped, and so is any key whose value is None.
_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")

# Span patterns: only the matched text is replaced.
_SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{10,}"),
    re.compile(r"Bearer\s+\S{4,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abp]-[A-Za-z0-9-]+"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
]

# Dict keys whose whole value is replaced. Matched exactly after lower-casing
# and mapping "-" to "_", so "max_tokens" and "token_count" are kept.
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
    }
)

# Keys that carry identifiers: never redacted.
_ID_KEYS = frozenset({"id", "tool_call_id"})


def redact_spans(value: Any) -> Tuple[Any, int]:
    """Return ``(redacted copy, number of replacements)`` for *value*.

    Replaces matched secret spans inside strings, and the whole value of a dict
    key named exactly like a secret field. Identifier keys are left alone.
    """
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        total = 0
        for k, v in value.items():
            if isinstance(k, str) and k in _ID_KEYS:
                out[k] = v
            elif (
                isinstance(k, str)
                and k.lower().replace("-", "_") in _SECRET_KEYS
                and v not in (None, "")
            ):
                out[k] = REDACTED
                total += 1
            else:
                out[k], n = redact_spans(v)
                total += n
        return out, total
    if isinstance(value, (list, tuple)):
        items = []
        total = 0
        for item in value:
            new, n = redact_spans(item)
            items.append(new)
            total += n
        return items, total
    if isinstance(value, str):
        total = 0
        for pattern in _SECRET_PATTERNS:
            value, n = pattern.subn(REDACTED, value)
            total += n
        return value, total
    return value, 0


def _as_dict(obj: Any) -> Any:
    """Return a plain dict for pydantic-style objects, else *obj* unchanged."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _normalize_tool_calls(message: dict) -> None:
    """Parse ``tool_calls[].function.arguments`` JSON strings into dicts, in place."""
    calls = message.get("tool_calls")
    if not calls:
        return
    normalized = []
    for call in calls:
        call = _as_dict(call)
        if isinstance(call, dict):
            function = _as_dict(call.get("function"))
            if isinstance(function, dict):
                call["function"] = function
                args = function.get("arguments")
                if isinstance(args, str):
                    try:
                        function["arguments"] = json.loads(args)
                    except ValueError:
                        logger.warning(
                            "tool call %r: arguments are not valid JSON; kept as string",
                            function.get("name"),
                        )
        normalized.append(call)
    message["tool_calls"] = normalized


def _clean_message(msg: Any) -> Optional[dict]:
    """Return a deep-copied message limited to the allowed keys, or None."""
    msg = _as_dict(copy.deepcopy(msg))
    if not isinstance(msg, dict):
        logger.warning("dropping non-dict message of type %s", type(msg).__name__)
        return None
    cleaned = {k: msg[k] for k in _MESSAGE_KEYS if msg.get(k) is not None}
    _normalize_tool_calls(cleaned)
    return cleaned


def _clean_messages(messages: Iterable[Any]) -> List[dict]:
    out = [_clean_message(m) for m in messages]
    return [m for m in out if m is not None]


def _clean_completion(response_message: Any) -> dict:
    """Return the assistant message limited to the allowed keys."""
    cleaned = _clean_message(response_message) or {}
    cleaned.setdefault("role", "assistant")
    return cleaned


def _is_empty_response(msg: Any) -> bool:
    msg = _as_dict(msg)
    if not isinstance(msg, dict):
        return True
    content = msg.get("content")
    has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
    return not has_content and not msg.get("tool_calls")


class TraceRecorder:
    """Collect one prompt/completion record per model call.

    Register with ``chain.register_callback(recorder.handle_event)``.

    START and END events pair on the ``trace_call_id`` metadata key that
    ``PromptChain`` adds while a callback is registered, so one recorder can
    serve concurrent runs. Events without that id (from another emitter)
    pair on ``(step_number, is_followup)``, which is **not safe under
    concurrency**: two overlapping calls on the same key would swap
    completions. When a START replaces an unfinished START on such a key, the
    recorder drops that key's next END and the replacing START, and logs it.

    A START without a matching END (the call errored) is dropped. END events
    whose response has a ``warning`` key, or has neither content nor tool
    calls, are skipped and logged.

    With ``redact_secrets`` the recorder replaces secret spans in prompts and
    completions and stores the replacement count in a ``redactions`` column.
    ``drop_redacted_completions=True`` drops records whose completion needed
    a replacement, which keeps a trained model from learning to emit
    ``***REDACTED***``.
    """

    def __init__(
        self,
        *,
        redact_secrets: bool = True,
        drop_redacted_completions: bool = False,
    ) -> None:
        self.redact_secrets = redact_secrets
        self.drop_redacted_completions = drop_redacted_completions
        self._pending: Dict[tuple, dict] = {}
        self._poisoned: Set[tuple] = set()
        self._records: List[dict] = []

    @staticmethod
    def _key(event: ExecutionEvent) -> tuple:
        md = event.metadata or {}
        call_id = md.get("trace_call_id")
        if call_id:
            return ("id", call_id)
        return ("step", event.step_number, bool(md.get("is_followup")))

    async def handle_event(self, event: ExecutionEvent) -> None:
        etype = event.event_type
        if etype not in (
            ExecutionEventType.MODEL_CALL_START,
            ExecutionEventType.MODEL_CALL_END,
            ExecutionEventType.MODEL_CALL_ERROR,
        ):
            return
        md = event.metadata or {}
        key = self._key(event)

        if etype == ExecutionEventType.MODEL_CALL_START:
            if "prompt_messages" not in md:
                logger.debug("MODEL_CALL_START without prompt_messages; skipped")
                return
            if key in self._pending and key[0] == "step":
                logger.warning(
                    "MODEL_CALL_START %s replaced an unfinished call without a "
                    "trace_call_id; dropping the next END for this key",
                    key,
                )
                self._poisoned.add(key)
            self._pending[key] = {
                "prompt": _clean_messages(md["prompt_messages"]),
                "tools": copy.deepcopy(md.get("tools")),
                "model": event.model_name,
                "step": event.step_number,
            }
        elif etype == ExecutionEventType.MODEL_CALL_ERROR:
            if self._pending.pop(key, None) is not None:
                logger.warning(
                    "dropped model call %s: %s", key, md.get("error", "unknown error")
                )
        else:
            start = self._pending.pop(key, None)
            if key in self._poisoned:
                self._poisoned.discard(key)
                logger.warning("MODEL_CALL_END %s is ambiguous after overlap; dropped", key)
                return
            if start is None:
                logger.warning("MODEL_CALL_END %s has no matching START; skipped", key)
                return
            response = md.get("response_message")
            if response is None:
                logger.warning("MODEL_CALL_END %s has no response_message; skipped", key)
                return
            if isinstance(response, dict) and "warning" in response:
                logger.warning(
                    "MODEL_CALL_END %s carries a warning (%s); skipped",
                    key,
                    response["warning"],
                )
                return
            if _is_empty_response(response):
                logger.warning("MODEL_CALL_END %s has an empty response; skipped", key)
                return
            prompt = start["prompt"]
            completion = [_clean_completion(response)]
            redactions = 0
            if self.redact_secrets:
                prompt, n_prompt = redact_spans(prompt)
                completion, n_completion = redact_spans(completion)
                redactions = n_prompt + n_completion
                if n_completion and self.drop_redacted_completions:
                    logger.warning(
                        "MODEL_CALL_END %s completion needed redaction; dropped", key
                    )
                    return
            record: Dict[str, Any] = {"prompt": prompt, "completion": completion}
            if start["tools"]:
                record["tools"] = start["tools"]
            record["model"] = start["model"]
            record["step"] = start["step"]
            if self.redact_secrets:
                record["redactions"] = redactions
            self._records.append(record)

    def records(self, reward: Optional[RewardFn] = None) -> List[dict]:
        """Return copies of the collected records.

        *reward* is called with each record and returns a float or ``None``;
        a float is stored in the ``reward`` column.
        """
        out = copy.deepcopy(self._records)
        if reward is not None:
            for rec in out:
                value = reward(rec)
                if value is not None:
                    rec["reward"] = float(value)
        return out

    def clear(self) -> None:
        """Forget collected records and unfinished calls."""
        self._pending.clear()
        self._poisoned.clear()
        self._records.clear()


def records_from_agentic(
    history_or_processor: Union[Iterable[dict], Any],
    *,
    tools: Optional[List[dict]] = None,
    reward: Optional[float] = None,
    redact_secrets: bool = True,
    drop_redacted: bool = False,
) -> List[dict]:
    """Build a conversational ``messages`` record from an agentic step's history.

    Accepts a message list or an object with an ``internal_history`` attribute
    (``AgenticStepProcessor``). Returns a one-element list, or an empty list
    when ``drop_redacted`` is set and any span was replaced. Raises
    ``ValueError`` for an empty history.

    The record is the whole internal conversation. With
    ``history_mode="minimal"`` the model saw only the system message, the
    objective and the latest tool exchange on each step, so the record is the
    trajectory, not the exact prompt the model saw at each step.
    """
    history = getattr(history_or_processor, "internal_history", history_or_processor)
    messages = _clean_messages(history)
    if not messages:
        raise ValueError(
            "empty history: the agentic step has not run, or it failed before "
            "recording any message"
        )
    redactions = 0
    if redact_secrets:
        messages, redactions = redact_spans(messages)
        if redactions and drop_redacted:
            logger.warning("agentic record needed %d redactions; dropped", redactions)
            return []
    record: Dict[str, Any] = {"messages": messages}
    if tools:
        record["tools"] = copy.deepcopy(tools)
    if reward is not None:
        record["reward"] = float(reward)
    if redact_secrets:
        record["redactions"] = redactions
    return [record]


def write_jsonl(records: Iterable[dict], path: Union[str, Path], *, append: bool = False) -> int:
    """Write *records* one JSON object per line and return the line count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count
