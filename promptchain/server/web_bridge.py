"""web_bridge.py — headless SSE bridge: drive PromptChain's agentic engine from a web overlay (Lookout).

PromptChain's engine is headless; this wraps a single agentic chain and streams its events over SSE so
a web front-end (Lookout's Tauri overlay) can BE the interface — replacing the textual terminal.

  POST /chat/turn   {message, model?, context?}  -> text/event-stream of {type, content} events
                    context = Lookout's watched screen region / detected card (text), injected as grounding
  GET  /health

Run:  BRIDGE_MODEL=openai/gpt-4o-mini  uvicorn promptchain.server.web_bridge:app --port 7788
Events: start · thinking · tool_call · tool_result · answer_delta · tokens · error · answer · done
"""
import asyncio
import json
import os
import urllib.request
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.promptchaining import PromptChain
from promptchain.cli.tools import registry

DEFAULT_MODEL = os.environ.get("BRIDGE_MODEL", "openai/gpt-4o-mini")
_TOOLS = [registry.get(t) for t in registry.list_tools()]
_SCHEMAS = registry.get_openai_schemas()

# name -> (ToolMeta, openai-schema) so a turn can register an arbitrary SUBSET of the registry.
_TOOL_FN = {t: registry.get(t) for t in registry.list_tools()}
def _schema_name(s):
    f = s.get("function") if isinstance(s, dict) else None
    src = f if isinstance(f, dict) else (s if isinstance(s, dict) else {})
    return src.get("name")
_SCHEMA_BY_NAME = {n: s for s in _SCHEMAS if (n := _schema_name(s))}

# grouping for the Lookout tool-picker (names not in any group fall to "Other")
_GROUPS = [
    ("files", "📄 Files", {"file_read", "file_write", "file_append", "file_edit", "file_delete",
        "read_file_range", "replace_lines", "insert_at_line", "insert_after_pattern", "insert_before_pattern",
        "create_directory", "list_directory", "find_paths", "path_info", "resolve_path", "get_cwd"}),
    ("search", "🔎 Search", {"ripgrep_search"}),
    ("exec", "⚙ Execution", {"terminal_execute", "sandbox_execute", "sandbox_provision_docker",
        "sandbox_provision_uv", "sandbox_list", "sandbox_cleanup"}),
    ("orchestration", "🤝 Orchestration", {"delegate_task", "request_help_tool", "get_pending_tasks",
        "update_task_status", "task_list_write_tool"}),
    ("memory", "🧠 Blackboard", {"write_to_blackboard", "read_from_blackboard", "list_blackboard_keys",
        "delete_blackboard_entry"}),
]

# Capture token usage across ALL the agentic processor's litellm calls (it owns the calls, not us).
# litellm calls CustomLogger.(async_)log_success_event with response_obj.usage on every completion.
import litellm
from litellm.integrations.custom_logger import CustomLogger
_USAGE = {"prompt": 0, "completion": 0}

class _UsageLogger(CustomLogger):
    def _add(self, response_obj):
        try:
            u = getattr(response_obj, "usage", None)
            if u:
                _USAGE["prompt"] += int(getattr(u, "prompt_tokens", 0) or 0)
                _USAGE["completion"] += int(getattr(u, "completion_tokens", 0) or 0)
        except Exception:
            pass
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._add(response_obj)
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._add(response_obj)

litellm.callbacks = [_UsageLogger()]

app = FastAPI(title="PromptChain ↔ Lookout bridge")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Turn(BaseModel):
    message: str
    model: Optional[str] = None
    context: Optional[str] = None          # Lookout's screen region / detected card (text)
    tools: Optional[list] = None           # subset of registry tool names (None = all)
    images: Optional[list] = None          # data-URI screen images for the vision model (native multimodal)


# Build EXACTLY like the real PromptChain TUI: pull the genuine foundation prompt from a real
# session's default agent, and use TUIAgenticStepProcessor (the LegacyTUIPromptGenerator system prompt).
import tempfile, pathlib
from promptchain.cli.tui_processor import TUIAgenticStepProcessor
from promptchain.cli.session_manager import SessionManager
from promptchain.utils.agent_chain import AgentChain
try:
    _SESSION = SessionManager(sessions_dir=tempfile.mkdtemp(prefix="lookout-pc-")).create_session(
        name="lookout", working_directory=pathlib.Path.cwd())
    _FOUNDATION = _SESSION.agents["default"].instruction_chain[0]   # the real foundation/execution prompt
    print(f"[bridge] foundation prompt from real session ({len(_FOUNDATION)} chars)")
except Exception as e:
    print(f"[bridge] session foundation unavailable ({e}); using fallback objective")
    _FOUNDATION = ("You are an EXECUTION agent, not an explanation agent. COMPLETE tasks by USING TOOLS "
                   "(file_read/file_write/file_edit, terminal_execute, sandbox_*, ripgrep_search). Use multi-hop "
                   "reasoning. ONLY respond with results after completing the task with tools.")

# ===== FAITHFUL TUI ENGINE PORT =====
# The real TUI keeps ONE AgentChain alive for the session and calls run_chat_turn_async() per turn;
# history accumulates in AgentChain._conversation_history and is auto-fed (auto_include_history=True)
# via _format_chat_history() before each agent call (agent_chain.py:2522/2534/2550). We mirror that:
# build the AgentChain ONCE, reuse it across turns. Rebuild only on model/tool change — carrying the
# conversation history forward (the TUI behaviour: switching model keeps the conversation).
_CB = {"fn": None}   # the current turn's event sink; the persistent processor's progress_callback routes here

def _emit(ev_type, content):
    fn = _CB["fn"]
    if fn:
        fn(ev_type, str(content))

async def _route_default(user_input, history, agent_descriptions):
    """Single-agent dispatch — no routing LLM call, instant. Returns the faithful router contract."""
    return json.dumps({"chosen_agent": "default"})

def _build_agent_chain(model: str, tool_names):
    """One AgentChain wrapping the real TUIAgenticStepProcessor agent — the TUI's exact assembly."""
    proc = TUIAgenticStepProcessor(
        objective=_FOUNDATION,
        max_internal_steps=6, model_name=model, history_mode="progressive",
        progress_callback=lambda cur, mx, status="": _emit("loop", json.dumps({"cur": cur, "max": mx, "agent": model})),
    )
    chain = PromptChain(models=[{"name": model, "params": {"max_completion_tokens": 1024}}],
                        instructions=[proc], verbose=False)
    names = list(tool_names) if tool_names else list(_TOOL_FN.keys())
    for n in names:
        tm = _TOOL_FN.get(n)
        if tm is not None and getattr(tm, "function", None) is not None:
            chain.register_tool_function(tm.function)
    chain.add_tools([_SCHEMA_BY_NAME[n] for n in names if n in _SCHEMA_BY_NAME])
    return AgentChain(
        agents={"default": chain},
        agent_descriptions={"default": "Lookout execution agent (PromptChain TUI engine)"},
        execution_mode="router", router=_route_default, default_agent="default",
        auto_include_history=True, verbose=False,
    )

_AC = {"chain": None, "key": None}   # persistent AgentChain + its (model, tools) key

def _ensure_chain(model: str, tool_names):
    key = (model, tuple(sorted(tool_names)) if tool_names else None)
    if _AC["chain"] is not None and _AC["key"] == key:
        return _AC["chain"]
    prev = _AC["chain"]
    ac = _build_agent_chain(model, tool_names)
    if prev is not None:                          # carry the conversation across model/tool changes
        try:
            ac._conversation_history = prev._conversation_history
        except Exception:
            pass
    _AC["chain"] = ac; _AC["key"] = key
    return ac


def _catalog():
    """Models the bridge can drive via litellm, filtered to the API keys actually present.
    Model switching is PER-TURN (passed to /chat/turn) — no bridge restart needed."""
    out = []
    # Ollama (Mac Studio / local) — PromptChain's own host, live from OLLAMA_HOST (needs OLLAMA_API_BASE set)
    host = os.environ.get("OLLAMA_HOST", "http://192.168.0.159:11434")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=3) as r:
            for m in json.load(r).get("models", []):
                n = m["name"]
                out.append({"id": f"ollama/{n}", "label": n, "provider": "ollama · Mac Studio", "note": "local"})
    except Exception:
        pass
    # OpenAI (litellm, verified)
    if os.environ.get("OPENAI_API_KEY"):
        out += [{"id": "openai/gpt-4o-mini", "label": "GPT-4o mini", "provider": "openai", "note": "cloud · fast"},
                {"id": "openai/gpt-4.1-mini", "label": "GPT-4.1 mini", "provider": "openai", "note": "cloud · balanced"},
                {"id": "openai/gpt-4o", "label": "GPT-4o", "provider": "openai", "note": "cloud · strong"}]
    # Gemini is reachable as TOOLS via the gemini MCP (ask_gemini/gemini_research), not a reasoning model here.
    return out


def _hist_len():
    ac = _AC["chain"]
    return len(getattr(ac, "_conversation_history", [])) if ac is not None else 0


@app.get("/health")
def health():
    return {"ok": True, "model": DEFAULT_MODEL, "tools": len(_SCHEMAS), "turns": _hist_len() // 2}


@app.get("/models")
def models():
    return {"default": DEFAULT_MODEL, "models": _catalog()}


@app.get("/tools")
def tools_catalog():
    """The registry tools, grouped, for Lookout's multiselect picker."""
    avail = set(_TOOL_FN.keys()); groups = []; seen = set()
    for gid, glabel, names in _GROUPS:
        items = []
        for n in sorted(names & avail):
            seen.add(n)
            s = _SCHEMA_BY_NAME.get(n); f = (s or {}).get("function", s or {})
            note = (f.get("description", "") or "").split("\n")[0][:70]
            items.append({"id": n, "label": n, "note": note})
        if items:
            groups.append({"id": gid, "label": glabel, "tools": items})
    rest = sorted(avail - seen)
    if rest:
        groups.append({"id": "other", "label": "• Other", "tools": [{"id": n, "label": n, "note": ""} for n in rest]})
    return {"groups": groups, "count": len(avail)}


import re
_VISION_RE = re.compile(
    r"(gpt-4o|gpt-4\.1|o[134]|gemini|claude-3|claude-4|sonnet|opus|haiku|[-_]?vl\b|vision|"
    r"llava|gemma-?3|gemma-?4|minicpm-?v|qwen.*vl|pixtral|moondream|internvl|kimi.*vision)", re.I)

def _is_vision_model(model: str) -> bool:
    """Best-effort: trust litellm's capability map, fall back to a name pattern (covers local *-vl/gemma3
    that litellm may not know). Used only to WARN, never to block."""
    m = model or ""
    try:
        if litellm.supports_vision(model=m):
            return True
    except Exception:
        pass
    return bool(_VISION_RE.search(m))


def _proc_of(ac):
    """The persistent TUIAgenticStepProcessor inside the default agent (holds its OWN history)."""
    try:
        agent = ac.agents["default"]
        for attr in ("instructions", "_instructions", "steps", "instruction_chain"):
            seq = getattr(agent, attr, None) or []
            for it in seq:
                if hasattr(it, "conversation_history"):
                    return it
    except Exception:
        pass
    return None


@app.post("/session/new")
def session_new():
    """Discard ALL conversation memory — BOTH stores: AgentChain._conversation_history AND the
    persistent processor's own conversation_history (else the agent keeps remembering)."""
    ac = _AC["chain"]
    if ac is not None:
        try:
            ac._conversation_history.clear()
        except Exception:
            ac._conversation_history = []
        proc = _proc_of(ac)
        if proc is not None:
            try:
                proc.conversation_history.clear()
            except Exception:
                try:
                    proc.conversation_history = []
                except Exception:
                    pass
    return {"ok": True, "turns": 0}


class WorkDir(BaseModel):
    path: str

_WORKDIR = {"path": str(pathlib.Path.cwd())}

@app.get("/session/workdir")
def get_workdir():
    return {"path": _WORKDIR["path"]}

@app.post("/session/workdir")
def set_workdir(w: WorkDir):
    """Root the agent's file tools (they use os.getcwd()) in the chosen folder — process-wide chdir."""
    p = os.path.expanduser(w.path or "")
    if p and os.path.isdir(p):
        os.chdir(p); _WORKDIR["path"] = os.getcwd()
        return {"ok": True, "path": _WORKDIR["path"]}
    return {"ok": False, "path": _WORKDIR["path"], "error": "not a directory"}


@app.get("/session/state")
def session_state():
    ac = _AC["chain"]; proc = _proc_of(ac) if ac else None
    return {"turns": _hist_len() // 2, "messages": _hist_len(),
            "proc_messages": len(getattr(proc, "conversation_history", []) or []) if proc else 0,
            "model": (_AC["key"][0] if _AC["key"] else None)}


@app.post("/chat/turn")
async def chat_turn(t: Turn):
    model = t.model or DEFAULT_MODEL
    _USAGE["prompt"] = 0; _USAGE["completion"] = 0     # accumulate this turn's tokens
    ac = _ensure_chain(model, t.tools)                 # persistent AgentChain — history carries across turns
    proc = _proc_of(ac)                                # attach this turn's screen image(s) for the vision model
    imgs = t.images or []
    warn_msg = None
    if imgs and not _is_vision_model(model):           # WARN, don't block: run text-only this turn
        imgs = []
        warn_msg = (f"vision language model needed — '{model}' isn't a recognized vision model, so the screen "
                    f"image was skipped this turn (text still works). Pick a vision model (gpt-4o, gemini, or a "
                    f"local *-vl / gemma3) to let it see the screen.")
    if proc is not None:
        proc.pending_images = imgs
    q: asyncio.Queue = asyncio.Queue()
    def cb(event_type, content):
        q.put_nowait({"type": event_type, "content": str(content)})
    _CB["fn"] = cb                                     # route the persistent processor's loop events to this turn
    prompt = t.message
    if t.context:
        prompt = f"[Developer's screen — what they're looking at]\n{t.context}\n\n[Question]\n{t.message}"

    async def gen():
        yield f"data: {json.dumps({'type': 'start', 'content': model})}\n\n"
        if warn_msg:
            yield f"data: {json.dumps({'type': 'warning', 'content': warn_msg})}\n\n"
        # THE REAL TUI TURN: run_chat_turn_async(message, streaming_callback=cb) — appends to
        # _conversation_history and auto-includes prior turns (agent_chain.py:2516/2522/2550).
        task = asyncio.create_task(ac.run_chat_turn_async(prompt, streaming_callback=cb))
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.1)
                yield f"data: {json.dumps(ev)}\n\n"
            except asyncio.TimeoutError:
                if task.done():
                    break
        while not q.empty():                # drain any trailing events
            yield f"data: {json.dumps(q.get_nowait())}\n\n"
        try:
            resp = await task
            resp = resp if isinstance(resp, str) else getattr(resp, "response", str(resp))
        except Exception as e:
            resp = f"(engine error: {e})"
        _CB["fn"] = None
        yield f"data: {json.dumps({'type': 'answer', 'content': str(resp)})}\n\n"
        yield f"data: {json.dumps({'type': 'tokens', 'content': json.dumps({'prompt_tokens': _USAGE['prompt'], 'completion_tokens': _USAGE['completion'], 'turns': _hist_len() // 2})})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
