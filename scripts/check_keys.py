#!/usr/bin/env python3
"""
check_keys.py — Probe API keys + model availability for PromptChain.

Reads env vars (does NOT print key values — masks to last 4 chars), pings each
configured provider with a minimal call to verify the key works, optionally
lists available models. Designed to be run before starting a PromptChain
session so you know which models you can actually pass to `PromptChain(models=...)`.

Usage:
    python scripts/check_keys.py
    python scripts/check_keys.py --json                 # machine-readable
    python scripts/check_keys.py --no-probe              # env-presence check only (free, no API calls)
    python scripts/check_keys.py --providers openai,gemini

Output:
    Stdout: human-readable table.
    File:   scripts/scratch/api-status.md (gitignored — overwrite each run).

Exit code:
    0 — at least one provider works
    1 — every configured provider failed (or none configured)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH = REPO_ROOT / "scripts" / "scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)


def mask(value: Optional[str]) -> str:
    """Mask a key — show only the last 4 chars."""
    if not value:
        return "(unset)"
    if len(value) <= 4:
        return "****"
    return f"****{value[-4:]}"


@dataclass
class ProviderStatus:
    name: str
    env_var: str
    env_set: bool
    masked_key: str
    probe_ok: Optional[bool] = None  # None = not probed
    probe_error: Optional[str] = None
    sample_models: list[str] = field(default_factory=list)
    notes: str = ""


def load_dotenv_into_os(env_path: Path) -> None:
    """Tiny dotenv parser — avoids requiring python-dotenv."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        # Don't overwrite if already exported
        os.environ.setdefault(key, val)


def check_openai(probe: bool) -> ProviderStatus:
    key = os.getenv("OPENAI_API_KEY")
    s = ProviderStatus(name="OpenAI", env_var="OPENAI_API_KEY", env_set=bool(key), masked_key=mask(key))
    if not probe or not key:
        return s
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        models = client.models.list()
        s.probe_ok = True
        # Widened filter: include all chat-capable model families
        keep = ("gpt-", "o1", "o3", "o4", "chatgpt-")
        skip = ("audio", "tts", "whisper", "embedding", "image", "moderation", "instruct")
        s.sample_models = sorted({
            m.id for m in models.data
            if any(k in m.id for k in keep) and not any(b in m.id for b in skip)
        })[:20]
    except ImportError:
        s.probe_ok = None
        s.notes = "openai package not installed — pip install openai"
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
    return s


def check_anthropic(probe: bool) -> ProviderStatus:
    key = os.getenv("ANTHROPIC_API_KEY")
    s = ProviderStatus(name="Anthropic", env_var="ANTHROPIC_API_KEY", env_set=bool(key), masked_key=mask(key))
    if not probe or not key:
        return s
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key)
        # Anthropic has no list-models endpoint; do a minimal completion
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        s.probe_ok = True
        s.sample_models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-opus-* (verify in console)"]
        s.notes = f"Probe call succeeded; usage={resp.usage.input_tokens}/{resp.usage.output_tokens}"
    except ImportError:
        s.probe_ok = None
        s.notes = "anthropic package not installed — pip install anthropic"
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
    return s


def check_gemini(probe: bool) -> ProviderStatus:
    # LiteLLM expects GOOGLE_API_KEY; some setups use GEMINI_API_KEY
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    used_var = "GOOGLE_API_KEY" if os.getenv("GOOGLE_API_KEY") else ("GEMINI_API_KEY" if os.getenv("GEMINI_API_KEY") else "GOOGLE_API_KEY")
    s = ProviderStatus(name="Gemini", env_var=used_var, env_set=bool(key), masked_key=mask(key))
    if not key and os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        s.notes = "GEMINI_API_KEY set but LiteLLM expects GOOGLE_API_KEY — `export GOOGLE_API_KEY=$GEMINI_API_KEY`"
    if not probe or not key:
        return s
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        models = list(genai.list_models())
        s.probe_ok = True
        # Show only generation-capable Gemini models
        s.sample_models = sorted({
            m.name.replace("models/", "") for m in models
            if "generateContent" in m.supported_generation_methods and "gemini" in m.name
        })[:10]
    except ImportError:
        s.probe_ok = None
        s.notes = (s.notes + "; " if s.notes else "") + "google-generativeai not installed — pip install google-generativeai"
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
    return s


def check_ollama(probe: bool) -> ProviderStatus:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    s = ProviderStatus(name="Ollama", env_var="OLLAMA_HOST (optional)", env_set=True, masked_key=host)
    if not probe:
        return s
    try:
        import urllib.request

        url = host.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
        s.probe_ok = True
        s.sample_models = sorted([m["name"] for m in data.get("models", [])])
        if not s.sample_models:
            s.notes = "Ollama running but no models pulled. Try: `ollama pull llama3.1`"
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
        s.notes = "Ollama server not reachable — start with `ollama serve` (or skip if unused)"
    return s


@dataclass
class OllamaModelTest:
    """Per-model load + output + tool-calling probe."""
    model: str
    generate_ok: Optional[bool] = None
    generate_latency_s: Optional[float] = None
    generate_sample: Optional[str] = None
    generate_error: Optional[str] = None
    tools_ok: Optional[bool] = None              # True = emitted a structured tool_call
    tools_error: Optional[str] = None
    tools_response_excerpt: Optional[str] = None


def probe_ollama_model(model: str, host: str, test_tools: bool, timeout: float = 90.0) -> OllamaModelTest:
    """Hit /api/generate with a 1-token prompt. Optionally also test /api/chat with tools."""
    import time
    import urllib.error
    import urllib.request

    out = OllamaModelTest(model=model)

    # 1) Minimal generate call — tests load + inference
    payload = json.dumps({
        "model": model,
        "prompt": "Reply with the single word: OK",
        "stream": False,
        "options": {"num_predict": 8, "temperature": 0.0},
    }).encode()
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(
            host.rstrip("/") + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        out.generate_ok = True
        out.generate_latency_s = round(time.monotonic() - t0, 2)
        out.generate_sample = (data.get("response") or "").strip()[:80]
    except Exception as e:
        out.generate_ok = False
        out.generate_latency_s = round(time.monotonic() - t0, 2)
        out.generate_error = type(e).__name__ + ": " + str(e)[:160]
        return out

    # 2) Tool-calling probe — sends a tools=[...] payload and checks if the model emits tool_calls
    if test_tools:
        tools_payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "user", "content": "What is the weather in Paris? Use the tool."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        }).encode()
        try:
            req = urllib.request.Request(
                host.rstrip("/") + "/api/chat",
                data=tools_payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            msg = data.get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            out.tools_ok = bool(tool_calls)
            content = msg.get("content") or ""
            if tool_calls:
                first = tool_calls[0]
                fn = (first.get("function") or {}).get("name")
                args = (first.get("function") or {}).get("arguments")
                out.tools_response_excerpt = f"name={fn} args={json.dumps(args)[:80]}"
            else:
                out.tools_response_excerpt = (content or "")[:120]
        except urllib.error.HTTPError as e:
            out.tools_ok = False
            try:
                body = e.read().decode()[:200]
            except Exception:
                body = ""
            out.tools_error = f"HTTP {e.code}: {body}"
        except Exception as e:
            out.tools_ok = False
            out.tools_error = type(e).__name__ + ": " + str(e)[:160]

    return out


def check_openrouter(probe: bool) -> ProviderStatus:
    key = os.getenv("OPENROUTER_API_KEY")
    s = ProviderStatus(name="OpenRouter", env_var="OPENROUTER_API_KEY", env_set=bool(key), masked_key=mask(key))
    if not probe or not key:
        return s
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        s.probe_ok = True
        s.sample_models = sorted([m["id"] for m in data.get("data", [])])[:10]
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
    return s


def check_groq(probe: bool) -> ProviderStatus:
    key = os.getenv("GROQ_API_KEY")
    s = ProviderStatus(name="Groq", env_var="GROQ_API_KEY", env_set=bool(key), masked_key=mask(key))
    if not probe or not key:
        return s
    try:
        import urllib.request

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        s.probe_ok = True
        s.sample_models = sorted([m["id"] for m in data.get("data", [])])[:8]
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
    return s


CHECKS = {
    "openai": check_openai,
    "anthropic": check_anthropic,
    "gemini": check_gemini,
    "ollama": check_ollama,
    "openrouter": check_openrouter,
    "groq": check_groq,
}


def render_ollama_section(tests: list[OllamaModelTest]) -> str:
    if not tests:
        return ""
    lines: list[str] = []
    lines.append("## Ollama — per-model load + tool-calling probe")
    lines.append("")
    lines.append("| Model | Load+gen | Latency (s) | Tool-calling? | Notes |")
    lines.append("|---|---|---|---|---|")
    for t in tests:
        gen = "✅" if t.generate_ok else "❌"
        if t.tools_ok is True:
            tools = "✅ tool_call emitted"
        elif t.tools_ok is False and t.tools_error:
            tools = f"❌ {t.tools_error[:60]}"
        elif t.tools_ok is False:
            tools = f"❌ plain text: {(t.tools_response_excerpt or '')[:60]}"
        else:
            tools = "—"
        notes = (t.generate_error or t.tools_response_excerpt or t.generate_sample or "").replace("|", "/")[:80]
        lines.append(f"| `ollama/{t.model}` | {gen} | {t.generate_latency_s or '?'} | {tools} | {notes} |")
    lines.append("")
    return "\n".join(lines)


def render_human(statuses: list[ProviderStatus], ollama_tests: Optional[list[OllamaModelTest]] = None) -> str:
    lines: list[str] = []
    lines.append(f"# PromptChain API Key & Model Status")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("| Provider | Env var | Set? | Key | Probe | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for s in statuses:
        if s.probe_ok is True:
            probe = "✅"
        elif s.probe_ok is False:
            probe = "❌"
        else:
            probe = "—"
        notes = (s.notes or "") + (f" {s.probe_error}" if s.probe_error else "")
        lines.append(f"| {s.name} | `{s.env_var}` | {'✅' if s.env_set else '❌'} | `{s.masked_key}` | {probe} | {notes.strip()} |")
    lines.append("")
    for s in statuses:
        if s.sample_models:
            lines.append(f"## {s.name} — sample available models")
            lines.append("")
            for m in s.sample_models:
                lines.append(f"- `{s.name.lower()}/{m}`" if "/" not in m else f"- `{m}`")
            lines.append("")
    if ollama_tests:
        lines.append(render_ollama_section(ollama_tests))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output to stdout")
    p.add_argument("--no-probe", action="store_true", help="Env-presence only — no API calls (free, no quota use)")
    p.add_argument("--providers", default=",".join(CHECKS.keys()), help="Comma-list of providers")
    p.add_argument("--probe-ollama-models", action="store_true",
                   help="For each Ollama model, run a load+generate test AND a tool-calling probe (slow — each model is loaded once)")
    p.add_argument("--ollama-smoke", action="store_true",
                   help="Smoke test only: probe ONE Ollama model (the smallest installed by name suffix, e.g. :4b before :32b) and stop. Implies --probe-ollama-models.")
    p.add_argument("--ollama-test-tools", action="store_true", default=True,
                   help="When --probe-ollama-models, also test tool-calling capability (default: on)")
    p.add_argument("--ollama-models", default="",
                   help="Comma-list to limit which Ollama models to probe (default: all installed)")
    p.add_argument("--ollama-timeout", type=float, default=90.0, help="Per-model timeout (seconds)")
    args = p.parse_args()

    # Load .env from repo root if present (does NOT overwrite already-exported vars)
    load_dotenv_into_os(REPO_ROOT / ".env")

    providers = [p.strip().lower() for p in args.providers.split(",") if p.strip()]
    statuses: list[ProviderStatus] = []
    for name in providers:
        if name not in CHECKS:
            print(f"Unknown provider: {name} (valid: {', '.join(CHECKS)})", file=sys.stderr)
            continue
        statuses.append(CHECKS[name](probe=not args.no_probe))

    # Optional per-Ollama-model probe (slow — each model is loaded into VRAM once)
    ollama_tests: list[OllamaModelTest] = []
    if (args.probe_ollama_models or args.ollama_smoke) and not args.no_probe:
        ollama_status = next((s for s in statuses if s.name == "Ollama"), None)
        if ollama_status and ollama_status.probe_ok and ollama_status.sample_models:
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            if args.ollama_models:
                requested = [m.strip() for m in args.ollama_models.split(",") if m.strip()]
            elif args.ollama_smoke:
                # Pick the smallest by size suffix heuristic (:4b before :7b before :13b before :32b)
                import re
                def size_key(name: str) -> int:
                    m = re.search(r":(\d+)([bB])", name)
                    return int(m.group(1)) if m else 999
                requested = [sorted(ollama_status.sample_models, key=size_key)[0]]
            else:
                requested = ollama_status.sample_models
            print(f"\n[check_keys.py] Probing {len(requested)} Ollama model(s) "
                  f"(load+gen, then tool-calling). This loads each model into VRAM once.", file=sys.stderr)
            for i, m in enumerate(requested, 1):
                print(f"  [{i}/{len(requested)}] {m} ...", file=sys.stderr, end=" ", flush=True)
                t = probe_ollama_model(m, host, test_tools=args.ollama_test_tools, timeout=args.ollama_timeout)
                ollama_tests.append(t)
                if t.generate_ok:
                    tag = "tools=✅" if t.tools_ok else ("tools=❌" if t.tools_ok is False else "tools=—")
                    print(f"gen={t.generate_latency_s}s {tag}", file=sys.stderr)
                else:
                    print(f"FAILED: {t.generate_error}", file=sys.stderr)

    if args.json:
        out = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "providers": [asdict(s) for s in statuses],
            "ollama_model_tests": [asdict(t) for t in ollama_tests],
        }
        print(json.dumps(out, indent=2))
    else:
        report = render_human(statuses, ollama_tests=ollama_tests or None)
        print(report)
        out_path = SCRATCH / "api-status.md"
        out_path.write_text(report + "\n")
        print(f"\n[check_keys.py] Wrote {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    any_ok = any(s.probe_ok is True for s in statuses)
    return 0 if (any_ok or args.no_probe) else 1


if __name__ == "__main__":
    raise SystemExit(main())
