"""F3 Dynamic Prompt Layer — toolshim (US2).

For models that cannot call tools natively, the shim re-expresses tool use as a
JSON-in-text protocol: the model emits one JSON object the harness parses and
executes. This module provides the three pure helpers the generator wires in:

- ``resolve_tool_mode`` — jacket.tool_mode or "native" (D4 / FR-010).
- ``render_tools_block`` — the ``<tools>`` JSON-in-text block (FR-008).
- ``serialize_history_plaintext`` — re-serialize native tool history as plain text (FR-009).

See ``specs/014-dynamic-prompt-layer/contracts/prompt-layout.md`` (the ``<tools>``
shape) + ``contracts/generator-api.md`` (toolshim section).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def resolve_tool_mode(jacket: Any) -> str:
    """Return the jacket's tool_mode, defaulting to "native" (D4).

    A ``None`` jacket OR a jacket with no ``tool_mode`` set → "native" (FR-010).
    """
    if jacket is not None:
        mode = getattr(jacket, "tool_mode", None)
        if mode:
            return mode
    return "native"


def _tool_name_desc(tool: Dict[str, Any]) -> tuple[Optional[str], str]:
    """Extract (name, description) from an OpenAI-format schema.

    Tolerates the wrapped ``{"function": {"name", "description"}}`` shape and the
    already-flattened ``{"name", "description"}`` shape.
    """
    if not isinstance(tool, dict):
        return None, ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        desc = fn.get("description") or ""
    else:
        name = tool.get("name")
        desc = tool.get("description") or ""
    name = name if isinstance(name, str) else None
    desc = desc if isinstance(desc, str) else ""
    return name, desc.strip()


def render_tools_block(tools: List[Dict[str, Any]]) -> str:
    """Build the ``<tools>`` JSON-in-text block for shim modes (FR-008).

    The block tells the model it cannot call tools natively and must emit one
    JSON object ``{"tool": "<name>", "arguments": { ... }}``, then enumerates each
    tool as ``- <name>: <description>``. The output is part of the never-dropped
    parity floor.
    """
    lines: List[str] = [
        "<tools>",
        "You cannot call tools natively. To use a tool, emit EXACTLY one JSON object:",
        '{"tool": "<name>", "arguments": { ... }}',
        "Available tools:",
    ]
    for tool in tools:
        name, desc = _tool_name_desc(tool)
        if not name:
            continue
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    lines.append("</tools>")
    return "\n".join(lines)


def serialize_history_plaintext(history: List[Dict[str, Any]]) -> str:
    """Re-serialize a native tool-call message history as readable plain text (FR-009).

    Native ``tool_calls`` on assistant messages and ``role:"tool"`` results become
    plain lines (``ASSISTANT called <name>(<args>)`` / ``TOOL <name> returned: <content>``)
    so a shim model never sees raw native tool-call objects.
    """
    out: List[str] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            calls = msg.get("tool_calls") or []
            for call in calls:
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name", "") if isinstance(fn, dict) else ""
                args = fn.get("arguments", "") if isinstance(fn, dict) else ""
                out.append(f"ASSISTANT called {name}({args})")
            text = msg.get("content")
            if text:
                out.append(f"ASSISTANT: {text}")
        elif role == "tool":
            name = msg.get("name", "")
            content = msg.get("content", "")
            out.append(f"TOOL {name} returned: {content}")
        else:
            content = msg.get("content")
            if content:
                label = (role or "message").upper()
                out.append(f"{label}: {content}")
    return "\n".join(out)
