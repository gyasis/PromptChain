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
        s.sample_models = sorted([m.id for m in models.data if "gpt-" in m.id or "o1" in m.id or "o4" in m.id])[:8]
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
        s.sample_models = sorted([m["name"] for m in data.get("models", [])])[:15]
        if not s.sample_models:
            s.notes = "Ollama running but no models pulled. Try: `ollama pull llama3.1`"
    except Exception as e:
        s.probe_ok = False
        s.probe_error = type(e).__name__ + ": " + str(e)[:200]
        s.notes = "Ollama server not reachable — start with `ollama serve` (or skip if unused)"
    return s


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


def render_human(statuses: list[ProviderStatus]) -> str:
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
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Machine-readable JSON output to stdout")
    p.add_argument("--no-probe", action="store_true", help="Env-presence only — no API calls (free, no quota use)")
    p.add_argument("--providers", default=",".join(CHECKS.keys()), help="Comma-list of providers")
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

    if args.json:
        out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "providers": [asdict(s) for s in statuses]}
        print(json.dumps(out, indent=2))
    else:
        report = render_human(statuses)
        print(report)
        out_path = SCRATCH / "api-status.md"
        out_path.write_text(report + "\n")
        print(f"\n[check_keys.py] Wrote {out_path.relative_to(REPO_ROOT)}", file=sys.stderr)

    any_ok = any(s.probe_ok is True for s in statuses)
    return 0 if (any_ok or args.no_probe) else 1


if __name__ == "__main__":
    raise SystemExit(main())
