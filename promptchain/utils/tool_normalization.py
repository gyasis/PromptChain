"""Portable tool-call normalization for PromptChain — PromptChain's own version of what ollama does
internally, so structured tool-calling works on ANY endpoint (raw llama.cpp, MLX, base HTTP, a "naked"
model), not only runtimes that ship a tools-param + parser.

Two halves (mirror ollama):
  • render_tools(tools)                — INPUT side: inject OpenAI-format tool schemas into the prompt in a
                                         generic, parseable format (for a model called WITHOUT a tools param).
  • parse_tool_calls(text, lenient=)   — OUTPUT side: extract structured tool_calls from the model's raw text.
                                         The `lenient` flag is the LENIENCY DIAL: True auto-repairs minor JSON
                                         damage (ollama-like); False surfaces the raw malformed call so a
                                         strong-model helper can fix it (the poor-man's-MoE test needs this).

Design: no dependencies beyond stdlib; emits/consumes the same OpenAI tool_call shape the rest of the
pipeline uses ({"function": {"name", "arguments"(json str)}}).
"""
import json, re

_TOOL_INSTRUCTION = (
    "You can call tools. To call ONE, output exactly this on its own line:\n"
    "<tool_call>{{\"name\": \"<tool_name>\", \"arguments\": {{<json args>}}}}</tool_call>\n"
    "Call at most one tool per step, only when the task needs it. If no tool is needed, reply in plain "
    "text with NO <tool_call>. Available tools:\n{tools}"
)


def _fn(t):
    return t.get("function") if isinstance(t, dict) and isinstance(t.get("function"), dict) else t


def render_tools(tools) -> str:
    """INPUT side: render OpenAI-format tool schemas into a prompt block for a naked (no tools-param) model."""
    lines = []
    for t in tools or []:
        fn = _fn(t) or {}
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')} | schema: {json.dumps(fn.get('parameters', {}))}")
    return _TOOL_INSTRUCTION.format(tools="\n".join(lines))


def _iter_balanced(s):
    """Yield top-level balanced {...} objects, string-aware (won't split on braces inside strings)."""
    depth = 0; start = -1; instr = False; esc = False
    for i, ch in enumerate(s or ""):
        if instr:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': instr = False
            continue
        if ch == '"': instr = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield s[start:i + 1]; start = -1


def _repair_json(raw):
    """Light, SAFE JSON repair (ollama-like leniency): trailing commas, single quotes, unquoted → best effort."""
    r = re.sub(r",\s*([}\]])", r"\1", raw)          # trailing commas
    r2 = r.replace("'", '"')                          # single → double quotes
    for cand in (raw, r, r2):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _extract_raw_calls(text):
    """Find candidate call payloads in priority order: <tool_call> markers, ```json fences, bare {..} with a name."""
    if not text:
        return []
    raws = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if raws:
        return raws
    fenced = re.findall(r"```(?:json|tool_call)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced
    return [o for o in _iter_balanced(text) if ('"name"' in o or "'name'" in o)]


def parse_tool_calls(text, tools=None, lenient=True):
    """OUTPUT side: parse raw model text into (tool_calls, leftover_content).

    tool_calls are OpenAI-shaped: {"function": {"name": str, "arguments": <json str>}}.
    lenient=True  → repair minor JSON damage, drop unparseable (ollama-like: valid-or-nothing).
    lenient=False → surface malformed calls as {"function": {"name": name_or_None, "arguments": <raw>, "_malformed": True}}
                    so a downstream gate marks them malformed and a strong helper can repair them.
    """
    raws = _extract_raw_calls(text)
    calls = []
    for raw in raws:
        obj = _repair_json(raw) if lenient else (json.loads(raw) if _is_json(raw) else None)
        if isinstance(obj, dict) and obj.get("name"):
            args = obj.get("arguments", {})
            args = args if isinstance(args, str) else json.dumps(args)
            calls.append({"function": {"name": obj["name"], "arguments": args}})
        elif not lenient:
            # strict mode — keep the damage visible for the repair helper
            nm = None
            m = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)', raw)
            if m: nm = m.group(1)
            calls.append({"function": {"name": nm, "arguments": raw, "_malformed": True}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text or "", flags=re.DOTALL).strip()
    return calls, content


def _is_json(raw):
    try:
        json.loads(raw); return True
    except Exception:
        return False
