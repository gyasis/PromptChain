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


# Build EXACTLY like the real PromptChain TUI: pull the genuine foundation prompt from a real
# session's default agent, and use TUIAgenticStepProcessor (the LegacyTUIPromptGenerator system prompt).
import tempfile, pathlib
from promptchain.cli.tui_processor import TUIAgenticStepProcessor
from promptchain.cli.session_manager import SessionManager
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

def build_chain(model: str, cb):
    """The REAL PromptChain agent: TUIAgenticStepProcessor (legacy TUI prompt) + foundation + full tools."""
    proc = TUIAgenticStepProcessor(
        objective=_FOUNDATION,
        max_internal_steps=6, model_name=model, history_mode="progressive", streaming_callback=cb,
        progress_callback=lambda cur, mx, status="": cb("loop", json.dumps({"cur": cur, "max": mx, "agent": model})),
    )
    chain = PromptChain(models=[{"name": model, "params": {"max_completion_tokens": 1024}}],
                        instructions=[proc], verbose=False)
    for tm in _TOOLS:
        if tm is not None:
            chain.register_tool_function(tm.function)
    chain.add_tools(_SCHEMAS)
    return chain


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


@app.get("/health")
def health():
    return {"ok": True, "model": DEFAULT_MODEL, "tools": len(_SCHEMAS)}


@app.get("/models")
def models():
    return {"default": DEFAULT_MODEL, "models": _catalog()}


@app.post("/chat/turn")
async def chat_turn(t: Turn):
    model = t.model or DEFAULT_MODEL
    _USAGE["prompt"] = 0; _USAGE["completion"] = 0     # accumulate this turn's tokens
    q: asyncio.Queue = asyncio.Queue()
    def cb(event_type, content):
        q.put_nowait({"type": event_type, "content": str(content)})
    chain = build_chain(model, cb)
    prompt = t.message
    if t.context:
        prompt = f"[Developer's screen — what they're looking at]\n{t.context}\n\n[Question]\n{t.message}"

    async def gen():
        yield f"data: {json.dumps({'type': 'start', 'content': model})}\n\n"
        task = asyncio.create_task(chain.process_prompt_async(prompt))
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
        except Exception as e:
            resp = f"(engine error: {e})"
        yield f"data: {json.dumps({'type': 'answer', 'content': str(resp)})}\n\n"
        yield f"data: {json.dumps({'type': 'tokens', 'content': json.dumps({'prompt_tokens': _USAGE['prompt'], 'completion_tokens': _USAGE['completion']})})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
