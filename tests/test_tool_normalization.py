"""Tests for the portable tool-call normalization layer (issue #29)."""
from promptchain.utils.tool_normalization import render_tools, parse_tool_calls

TOOLS = [{"type": "function", "function": {
    "name": "send_email",
    "description": "Send an email.",
    "parameters": {"type": "object",
                   "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                   "required": ["to", "body"]}}}]


def test_render_tools_injects_format_and_names():
    p = render_tools(TOOLS)
    assert "<tool_call>" in p and "send_email" in p


def test_parse_clean_tool_call():
    calls, content = parse_tool_calls(
        'Sure!\n<tool_call>{"name":"send_email","arguments":{"to":"a@b.com","body":"hi"}}</tool_call>', TOOLS)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "send_email"
    assert content == "Sure!"


def test_parse_malformed_lenient_repairs():
    # trailing comma + single quotes -> lenient mode repairs to a valid call
    calls, _ = parse_tool_calls(
        "<tool_call>{'name':'send_email','arguments':{'to':'a','body':'b',}}</tool_call>", TOOLS, lenient=True)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "send_email"


def test_parse_malformed_strict_surfaces_for_repair():
    # broken JSON -> strict mode surfaces it (does not silently drop) so a helper can fix it
    calls, _ = parse_tool_calls('<tool_call>{"name":"send_email","arguments":{"to": }}</tool_call>', TOOLS, lenient=False)
    assert calls and calls[0]["function"].get("_malformed") is True


def test_parse_no_tool_call_is_plain_text():
    calls, content = parse_tool_calls("I can't help with that.", TOOLS)
    assert calls == [] and content


def test_parse_json_fence_fallback():
    calls, _ = parse_tool_calls('```json\n{"name":"send_email","arguments":{"to":"x","body":"y"}}\n```', TOOLS)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "send_email"
