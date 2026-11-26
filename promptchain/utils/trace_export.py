"""TRL-shaped training-data export from PromptChain runs.

``TraceRecorder`` is an execution callback that pairs each MODEL_CALL_START
event with its MODEL_CALL_END event and keeps one record per model call:

    {"prompt": [messages], "completion": [assistant message],
     "tools": [...], "model": str, "step": int}

``records_from_agentic`` turns the full internal history of an
``AgenticStepProcessor`` into one conversational ``messages`` record, because
agentic steps emit no MODEL_CALL events. ``write_jsonl`` writes records one
JSON object per line, which ``datasets.load_dataset("json", ...)`` reads.

The recorder reads the ``prompt_messages``, ``tools`` and ``response_message``
metadata keys. ``PromptChain`` fills them only while a callback is registered.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from promptchain.observability._transcript_redaction import redact
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

logger = logging.getLogger(__name__)

RewardFn = Callable[[dict], Optional[float]]

_ASSISTANT_KEYS = ("role", "content", "tool_calls")


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


def _clean_messages(messages: Iterable[Any]) -> List[dict]:
    """Deep-copy *messages* into plain dicts with parsed tool-call arguments."""
    out: List[dict] = []
    for msg in copy.deepcopy(list(messages)):
        msg = _as_dict(msg)
        if not isinstance(msg, dict):
            logger.warning("dropping non-dict message of type %s", type(msg).__name__)
            continue
        _normalize_tool_calls(msg)
        out.append(msg)
    return out


def _clean_completion(response_message: Any) -> dict:
    """Return the assistant message holding only role, content and tool_calls."""
    msg = _as_dict(copy.deepcopy(response_message))
    cleaned = {k: msg[k] for k in _ASSISTANT_KEYS if k in msg and msg[k] is not None}
    cleaned.setdefault("role", "assistant")
    _normalize_tool_calls(cleaned)
    return cleaned


class TraceRecorder:
    """Collect one prompt/completion record per model call.

    Register with ``chain.register_callback(recorder.handle_event)``. START and
    END events are paired by ``(step_number, is_followup)``, so use one recorder
    per run: concurrent runs would collide on those keys.

    A START event without a matching END (the call errored) is dropped.
    """

    def __init__(self, *, redact_secrets: bool = True) -> None:
        self.redact_secrets = redact_secrets
        self._pending: Dict[Tuple[Optional[int], bool], dict] = {}
        self._records: List[dict] = []

    async def handle_event(self, event: ExecutionEvent) -> None:
        etype = event.event_type
        if etype not in (
            ExecutionEventType.MODEL_CALL_START,
            ExecutionEventType.MODEL_CALL_END,
            ExecutionEventType.MODEL_CALL_ERROR,
        ):
            return
        md = event.metadata or {}
        key = (event.step_number, bool(md.get("is_followup")))

        if etype == ExecutionEventType.MODEL_CALL_START:
            if "prompt_messages" not in md:
                logger.debug("MODEL_CALL_START without prompt_messages; skipped")
                return
            if key in self._pending:
                logger.warning("MODEL_CALL_START %s replaced an unfinished call", key)
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
            if start is None:
                logger.warning("MODEL_CALL_END %s has no matching START; skipped", key)
                return
            if "response_message" not in md:
                logger.warning("MODEL_CALL_END %s has no response_message; skipped", key)
                return
            record = {
                "prompt": start["prompt"],
                "completion": [_clean_completion(md["response_message"])],
            }
            if start["tools"]:
                record["tools"] = start["tools"]
            record["model"] = start["model"]
            record["step"] = start["step"]
            if self.redact_secrets:
                tools = record.pop("tools", None)
                record = redact(record)
                if tools:
                    record["tools"] = tools
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
        self._records.clear()


def records_from_agentic(
    history_or_processor: Union[Iterable[dict], Any],
    *,
    tools: Optional[List[dict]] = None,
    reward: Optional[float] = None,
    redact_secrets: bool = True,
) -> List[dict]:
    """Build a conversational ``messages`` record from an agentic step's history.

    Accepts a message list or an object with an ``internal_history`` attribute
    (``AgenticStepProcessor``). Returns a one-element list.
    """
    history = getattr(history_or_processor, "internal_history", history_or_processor)
    messages = _clean_messages(history)
    if redact_secrets:
        messages = redact(messages)
    record: Dict[str, Any] = {"messages": messages}
    if tools:
        record["tools"] = copy.deepcopy(tools)
    if reward is not None:
        record["reward"] = float(reward)
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
