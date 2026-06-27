"""TEST-FIRST (Constitution III) failing tests for F3 US2 — the additive-optional
``Jacket.tool_mode`` field.

US2 adds an OPTIONAL ``tool_mode`` to the F2 ``Jacket`` dataclass. The change is
purely additive: ``tool_mode`` defaults to ``None`` when omitted, round-trips through
``to_dict``/``from_dict``, and a legacy dict WITHOUT a ``tool_mode`` key still loads
(no schema bump). These FAIL now because ``Jacket`` has no ``tool_mode`` field yet.

Refs: specs/014-dynamic-prompt-layer/spec.md (US2, FR-008/009/010),
contracts/generator-api.md (toolshim section).
"""
from __future__ import annotations

from promptchain.profiler.jacket import Jacket

# The existing required Jacket fields (read from promptchain/profiler/jacket.py).
_BASE_KWARGS = dict(
    tier="extended",
    budget_tokens=4000,
    mode="heavy-loop",
    spawn_temp=0.15,
    compress_at=0.60,
    max_turns=20,
    role="both",
)


def test_tool_mode_defaults_to_none_when_omitted():
    # Constructing without tool_mode must work and default to None (additive + optional).
    jk = Jacket(**_BASE_KWARGS)
    assert jk.tool_mode is None


def test_tool_mode_round_trips_through_dict():
    jk = Jacket(**_BASE_KWARGS, tool_mode="shim_prompt")
    # Serializes into to_dict()...
    assert jk.to_dict()["tool_mode"] == "shim_prompt"
    # ...and survives a from_dict(to_dict(...)) round-trip.
    assert Jacket.from_dict(jk.to_dict()).tool_mode == "shim_prompt"


def test_from_dict_without_tool_mode_key_is_backward_compatible():
    # A legacy on-disk dict has NO "tool_mode" key — must still load → None (no schema bump).
    d = Jacket(**_BASE_KWARGS).to_dict()
    d.pop("tool_mode", None)  # ensure the key is absent regardless of impl
    assert "tool_mode" not in d
    loaded = Jacket.from_dict(d)
    assert loaded.tool_mode is None
