"""Guard the ExecutionEvent field contract that observability consumers read.

Origin: the TUI's observability_callback read `event.data.get(...)` in ELEVEN
places. ExecutionEvent has no `.data` attribute, so every branch raised
AttributeError -- and the callback's own `except Exception` swallowed it. The
ObservePanel, logged at startup as "primary observability", rendered nothing
and quietly logged an error per event. Nothing failed loudly, and no test
covered the callback, so it stayed broken.

These tests are cheap and cover the whole bug class rather than the ten
specific lines: the field contract itself, and a source scan for the wrong
accessor. A future consumer that reaches for `.data` fails here instead of
silently going dark in production.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _event() -> ExecutionEvent:
    return ExecutionEvent(
        event_type=ExecutionEventType.STEP_START,
        step_number=3,
        model_name="test-model",
        metadata={"tool_name": "t", "tool_args": {}, "tokens_used": {}},
    )


def test_execution_event_exposes_the_documented_fields():
    """The fields observability consumers are entitled to rely on."""
    e = _event()
    for field in ("event_type", "timestamp", "step_number",
                  "step_instruction", "model_name", "metadata"):
        assert hasattr(e, field), f"ExecutionEvent lost its {field!r} field"
    assert e.step_number == 3
    assert e.model_name == "test-model"
    assert isinstance(e.metadata, dict)


def test_execution_event_has_no_data_attribute():
    """`.data` never existed. Consumers must use `.metadata` or a top-level field.

    If a future change ADDS a `.data` alias, this test should be updated
    deliberately -- silently growing a second accessor is how the two diverge.
    """
    with pytest.raises(AttributeError):
        _ = _event().data


@pytest.mark.parametrize("rel", ["promptchain/cli/tui/app.py"])
def test_observability_consumers_do_not_read_event_data(rel: str):
    """Source guard: the accessor that was silently dead must not come back.

    A behavioural test would need a full TUI instance; this catches the same
    regression for the cost of a regex, which is why it is worth having.
    """
    src = (REPO_ROOT / rel).read_text(encoding="utf-8")
    hits = [
        f"{rel}:{i}"
        for i, line in enumerate(src.splitlines(), 1)
        if re.search(r"""\bevent\s*\.\s*data\b|getattr\(\s*event\s*,\s*['"]data['"]""", line)
    ]
    assert not hits, (
        "ExecutionEvent has no `.data`; these reads raise AttributeError and "
        "get swallowed by the surrounding except -> " + ", ".join(hits)
    )


# ---------------------------------------------------------------------------
# Key contract.
#
# The attribute guard above catches `.data` coming back, but NOT the more
# dangerous regression: reading a key nobody emits. That fails silently -- the
# panel renders a default and looks fine, which is exactly the class of bug
# this file exists for. These assert the metadata key names the TUI depends on
# still exist at their emit sites, so a rename fails here instead of going dark.

EMITTED_KEYS = {
    "promptchain/utils/promptchaining.py": [
        "tokens_used",     # MODEL_CALL_END token counts
        "tool_name",       # TOOL_CALL_START / _END
        "tool_args",       # TOOL_CALL_START args (NOT "arguments")
        "result_length",   # local-tool TOOL_CALL_END size
        "message_count",   # MODEL_CALL_START prompt size
    ],
    "promptchain/utils/mcp_helpers.py": [
        "tool_name",
        "result",          # MCP TOOL_CALL_END preview
    ],
}


@pytest.mark.parametrize(
    "rel,key",
    [(rel, k) for rel, keys in EMITTED_KEYS.items() for k in keys],
)
def test_metadata_keys_the_tui_reads_are_still_emitted(rel: str, key: str):
    src = (REPO_ROOT / rel).read_text(encoding="utf-8")
    assert f'"{key}"' in src, (
        f"{rel} no longer emits metadata key {key!r}. The TUI's "
        "observability_callback reads it; a rename makes the panel show a "
        "default instead of the value, with no error."
    )


def test_prompt_and_completion_tokens_are_the_summed_keys():
    """The panel sums these because no `total_tokens` key is emitted."""
    src = (REPO_ROOT / "promptchain/utils/promptchaining.py").read_text(encoding="utf-8")
    assert '"prompt_tokens"' in src and '"completion_tokens"' in src
    assert '"total_tokens"' not in src, (
        "A total_tokens key now exists; the TUI sums prompt+completion instead "
        "and should be updated to prefer the emitted total."
    )


# ---------------------------------------------------------------------------
# Field -> key pairing.
#
# The presence tests above prove a key is still emitted SOMEWHERE. They do not
# prove the panel reads the RIGHT key for a given field. Demonstrated in
# adversarial review: swapping args_preview's read from "tool_args" to
# "tool_name" -- both of which are emitted -- left every other test in this
# file green while the panel silently showed a tool's name where its arguments
# belong. A wrong-but-existing key is the successor to the `.data` bug: no
# exception, no log line, just quietly wrong output.
#
# Each row pins ONE displayed field to the ONE key it must come from.

FIELD_KEY_PAIRS = [
    ("message_count", 'event.metadata.get("message_count"'),
    ("token usage",   'event.metadata.get("tokens_used")'),
    ("tool name",     'tool_name = event.metadata.get("tool_name"'),
    ("tool args",     'args_preview = str(event.metadata.get("tool_args"'),
]


@pytest.mark.parametrize("field,expected_read", FIELD_KEY_PAIRS)
def test_tui_reads_each_field_from_its_own_key(field: str, expected_read: str):
    src = (REPO_ROOT / "promptchain/cli/tui/app.py").read_text(encoding="utf-8")
    assert expected_read in src, (
        f"The TUI no longer reads {field!r} via {expected_read!r}. If the key "
        "changed, update the emitter, this test, and the panel together -- a "
        "mismatched-but-existing key renders wrong data with no error."
    )


def test_top_level_fields_are_not_read_from_metadata():
    """model_name and step_number are ExecutionEvent FIELDS, not metadata keys.

    Reading them from metadata returns the default forever -- "unknown" models
    and "?" step numbers -- which is the failure this whole file exists for.
    """
    src = (REPO_ROOT / "promptchain/cli/tui/app.py").read_text(encoding="utf-8")
    for wrong in ('metadata.get("model_name"', 'metadata.get("step_number"',
                  'metadata.get("step_index"'):
        assert wrong not in src, (
            f"{wrong!r} reads a top-level ExecutionEvent field out of metadata; "
            "it will silently return the default."
        )
