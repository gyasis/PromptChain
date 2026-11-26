"""Live model catalog for the PromptChain TUI.

Aggregates selectable models from several providers into a single catalog.
Each entry carries BOTH:

  * its own routing  — the litellm model string + ``api_base`` + which env var
    holds the api key (so Mac-Studio-local Ollama, Ollama Cloud, OpenAI and
    Gemini can coexist and be switched freely), and
  * a best-effort reasoning/output PROFILE — how that model exposes its
    "thinking" (separate field vs ``<think>`` tags vs ``reasoning_content``)
    and how to turn it on.

Design principle — **profile-driven WITH graceful fallback**:
we try to detect a model's specific reasoning profile to capture its features,
but detection is heuristic and never load-bearing. An unknown model still
produces a fully functional entry (``supports=False`` → the extractor treats
the whole response as the answer). Nothing breaks on a model we don't know.

Sources (each is best-effort — if one is unreachable it's skipped, the rest
still build):
  1. Mac Studio local Ollama  — GET <host>/api/tags                (no auth)
  2. Ollama Cloud             — GET ollama.com/v1/models  (Bearer OLLAMA_API_KEY)
  3. Generated catalog files   — models.json / local-models.json (incl. the
                                 llamacpp_lane) in the catalog directory

The catalog directory is ``$PROMPTCHAIN_MODEL_CATALOG_DIR`` (default
``~/.config/ollama-cloud``). The library names no model: when the files are
missing the chooser says so and the user types any provider/model string.

Zero third-party deps — stdlib ``urllib`` only.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Mac Studio Ollama endpoint (see ~/.claude/rules/domains/mac-studio.md). Honour
# OLLAMA_HOST if set, else fall back to the known LAN address.
DEFAULT_MAC_HOST = os.environ.get("OLLAMA_HOST") or "http://192.168.0.159:11434"
OLLAMA_CLOUD_BASE = "https://ollama.com/v1"

# Models that are embeddings / rerankers / vision-only helpers — not chat
# targets, so we keep them out of the chooser.
_NON_CHAT_HINTS = ("embed", "bge-", "minilm", "nomic", "rerank", "moondream")


@dataclass
class ModelEntry:
    """One selectable model: routing + reasoning profile."""

    label: str
    model_name: str  # litellm string, e.g. "ollama_chat/<name>"
    source: str  # openai | gemini | ollama-local | ollama-cloud
    api_base: Optional[str] = None
    api_key_env: Optional[str] = None
    reasoning: Dict = field(default_factory=lambda: {"supports": False, "extract": "none"})
    size_gb: Optional[float] = None
    capabilities: List[str] = field(default_factory=list)
    context_length: Optional[int] = None

    def params(self) -> Dict:
        """litellm call params this entry needs (api_base / api_key / enable)."""
        p: Dict = {}
        if self.api_base:
            p["api_base"] = self.api_base
        if self.api_key_env and os.environ.get(self.api_key_env):
            p["api_key"] = os.environ[self.api_key_env]
        enable = self.reasoning.get("enable")
        if enable:
            p.update(enable)
        return p


# --------------------------------------------------------------------------- #
# Reasoning-profile heuristic (graceful: unknown → supports=False)
# --------------------------------------------------------------------------- #
def infer_reasoning_profile(model_name: str, source: str) -> Dict:
    """Best-effort guess of how a model exposes its reasoning.

    extract ∈ {field, think_tag, reasoning_content, none}. Always returns a
    valid profile; an unknown model yields ``supports=False`` so the extractor
    falls back to treating the whole response as the answer.
    """
    n = model_name.lower()
    ollama = source.startswith("ollama")

    # OpenAI o-series reasoners → litellm surfaces reasoning_content
    if any(t in n for t in ("o1", "o3", "o4-mini")) and "gpt-" not in n:
        return {"supports": True, "extract": "reasoning_content", "tags": None,
                "enable": {"reasoning_effort": "medium"}}

    # deepseek-r1 family — ollama returns a separate message.thinking field when
    # think=true; through other paths it tends to surface as reasoning_content.
    if "deepseek-r1" in n or "deepseek-reasoner" in n or ":r1" in n:
        return {"supports": True,
                "extract": "field" if source == "ollama-local" else "reasoning_content",
                "tags": None, "enable": {"think": True} if ollama else None}

    # gpt-oss (OpenAI open-weights, harmony format) — surfaces reasoning_content
    # via the OpenAI-compat shim, NOT <think> tags (verified live on Ollama Cloud).
    if "gpt-oss" in n:
        return {"supports": True, "extract": "reasoning_content", "tags": None,
                "enable": {"think": True} if ollama else None}

    # qwq / qwen3 / magistral / explicit "think/reason" distills → <think> tags
    if any(t in n for t in ("qwq", "qwen3", "magistral", "thinking", "-think", "reason")):
        return {"supports": True, "extract": "think_tag", "tags": ["<think>", "</think>"],
                "enable": {"think": True} if ollama else None}

    # Ornith-1.0 (new DeepReinforce model) — reasoning behaviour unconfirmed;
    # assume tag-based but soft, and rely on the extractor's graceful fallback.
    if "ornith" in n:
        return {"supports": True, "extract": "think_tag", "tags": ["<think>", "</think>"], "enable": None}

    # Default: no special reasoning → generic fallback at extract time.
    return {"supports": False, "extract": "none", "tags": None, "enable": None}


# --------------------------------------------------------------------------- #
# HTTP helper
# --------------------------------------------------------------------------- #
def _get_json(url: str, headers: Optional[Dict] = None, timeout: float = 6.0) -> Dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _is_chat_model(name: str) -> bool:
    n = name.lower()
    return not any(h in n for h in _NON_CHAT_HINTS)


# --------------------------------------------------------------------------- #
# Source fetchers — each returns [] on any failure (graceful)
# --------------------------------------------------------------------------- #
def fetch_ollama_local(host: Optional[str] = None, timeout: float = 6.0) -> List[ModelEntry]:
    host = host or DEFAULT_MAC_HOST
    if not host.startswith("http"):
        host = "http://" + host
    host = host.rstrip("/")
    try:
        data = _get_json(f"{host}/api/tags", timeout=timeout)
    except Exception:
        return []
    out: List[ModelEntry] = []
    for m in data.get("models", []):
        name = m.get("name")
        if not name or not _is_chat_model(name):
            continue
        size = m.get("size")
        out.append(ModelEntry(
            label=f"Mac Studio · {name}",
            model_name=f"ollama_chat/{name}",
            source="ollama-local",
            api_base=host,
            api_key_env=None,
            reasoning=infer_reasoning_profile(name, "ollama-local"),
            size_gb=round(size / 1e9, 1) if size else None,
        ))
    return out


def fetch_ollama_cloud(timeout: float = 8.0) -> List[ModelEntry]:
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        return []
    try:
        data = _get_json(f"{OLLAMA_CLOUD_BASE}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except Exception:
        return []
    out: List[ModelEntry] = []
    for m in data.get("data", []):
        mid = m.get("id")
        if not mid or not _is_chat_model(mid):
            continue
        out.append(ModelEntry(
            label=f"Ollama Cloud · {mid}",
            model_name=f"openai/{mid}",  # addressed via the OpenAI-compat shim
            source="ollama-cloud",
            api_base=OLLAMA_CLOUD_BASE,
            api_key_env="OLLAMA_API_KEY",
            reasoning=infer_reasoning_profile(mid, "ollama-cloud"),
        ))
    return out


CATALOG_DIR_ENV = "PROMPTCHAIN_MODEL_CATALOG_DIR"
DEFAULT_CATALOG_DIR = "~/.config/ollama-cloud"
CLOUD_CATALOG_FILE = "models.json"
LOCAL_CATALOG_FILE = "local-models.json"


def catalog_dir() -> Path:
    """Directory holding the generated catalog files (env override, else default)."""
    return Path(os.environ.get(CATALOG_DIR_ENV) or DEFAULT_CATALOG_DIR).expanduser()


def no_catalog_message(path: Optional[Path] = None) -> str:
    """Message shown when no catalog entries could be loaded."""
    path = path or catalog_dir()
    return (f"no model catalog found at {path}; type a provider/model string "
            f"(set {CATALOG_DIR_ENV} to point at a catalog directory)")


def _read_json(path: Path, problems: List[str]) -> Optional[Dict]:
    """Parse a catalog file; record a problem and return None on any failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"{path} not found")
        return None
    except (OSError, ValueError) as exc:
        problems.append(f"{path} unreadable: {exc}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{path} unreadable: top level is not an object")
        return None
    return data


def _catalog_reasoning(name: str, source: str, caps: List[str]) -> Dict:
    """Reasoning profile: name heuristic first, else the catalog's own capability."""
    prof = infer_reasoning_profile(name, source)
    if not prof["supports"] and "thinking" in caps:
        return {"supports": True, "extract": "reasoning_content", "tags": None,
                "enable": {"think": True}}
    return prof


def _ollama_entries(data: Dict, source: str, host: str, label_prefix: str) -> List[ModelEntry]:
    out: List[ModelEntry] = []
    models = data.get("models")
    for m in models if isinstance(models, list) else []:
        name = m.get("name") if isinstance(m, dict) else None
        caps = [c for c in (m.get("capabilities") or []) if isinstance(c, str)] if name else []
        if not name or "embedding" in caps or not _is_chat_model(name):
            continue
        out.append(ModelEntry(
            label=f"{label_prefix} · {name}",
            model_name=f"ollama_chat/{name}",
            source=source,
            api_base=host,
            reasoning=_catalog_reasoning(name, source, caps),
            capabilities=caps,
            context_length=m.get("context_length") if isinstance(m.get("context_length"), int) else None,
        ))
    return out


def _lane_entries(lane: Dict) -> List[ModelEntry]:
    """llamacpp_lane aliases → OpenAI-compatible governor endpoint entries."""
    src = lane.get("source")
    if not isinstance(src, str) or not src:
        return []
    base = src.rstrip("/")
    if base.endswith("/models"):
        base = base[: -len("/models")]
    base += "/v1"
    out: List[ModelEntry] = []
    models = lane.get("models")
    for m in models if isinstance(models, list) else []:
        alias = m.get("alias") if isinstance(m, dict) else None
        if not alias:
            continue
        caps = ["completion"] + (["vision"] if m.get("vision") else [])
        out.append(ModelEntry(
            label=f"llama.cpp lane · {alias}",
            model_name=f"openai/{alias}",
            source="llamacpp",
            api_base=base,
            reasoning=infer_reasoning_profile(alias, "llamacpp"),
            capabilities=caps,
            context_length=m.get("ctx") if isinstance(m.get("ctx"), int) else None,
        ))
    return out


def load_catalog_files(directory: Optional[Path] = None) -> tuple[List[ModelEntry], List[str]]:
    """Read the generated catalog files → (entries, problems). Never raises.

    Cloud tags and local tags are called as ``ollama_chat/<tag>`` against an
    Ollama endpoint (cloud: ``$OLLAMA_HOST`` or the local daemon, which proxies
    cloud tags; local: the ``host`` recorded in the file). Lane aliases are
    called as ``openai/<alias>`` against the governor's OpenAI-compatible route.
    """
    directory = Path(directory) if directory else catalog_dir()
    problems: List[str] = []
    out: List[ModelEntry] = []
    cloud = _read_json(directory / CLOUD_CATALOG_FILE, problems)
    if cloud is not None:
        host = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if not host.startswith("http"):
            host = "http://" + host
        out += _ollama_entries(cloud, "ollama-cloud", host.rstrip("/"), "Ollama Cloud")
    local = _read_json(directory / LOCAL_CATALOG_FILE, problems)
    if local is not None:
        host = local.get("host")
        if isinstance(host, str) and host:
            out += _ollama_entries(local, "ollama-local", host.rstrip("/"), "Local")
        else:
            problems.append(f"{directory / LOCAL_CATALOG_FILE} has no host")
        lane = local.get("llamacpp_lane")
        if isinstance(lane, dict):
            out += _lane_entries(lane)
    return out, problems


# --------------------------------------------------------------------------- #
# Catalog assembly + resolution
# --------------------------------------------------------------------------- #
def build_catalog(*, include_files: bool = True, include_local: bool = True,
                  include_cloud: bool = True, mac_host: Optional[str] = None) -> List[ModelEntry]:
    """Build the full catalog from all reachable sources (each best-effort)."""
    cat: List[ModelEntry] = []
    if include_files:
        cat += load_catalog_files()[0]
    if include_local:
        cat += fetch_ollama_local(mac_host)
    if include_cloud:
        cat += fetch_ollama_cloud()
    seen: set = set()
    unique: List[ModelEntry] = []
    for e in cat:  # file entries come first and carry capabilities, so they win
        if e.model_name not in seen:
            seen.add(e.model_name)
            unique.append(e)
    return unique


def resolve(model_or_label: str, catalog: Optional[List[ModelEntry]] = None) -> tuple[str, Dict]:
    """Resolve a label OR raw model string → (litellm model_name, call params).

    Graceful: if nothing in the catalog matches, the input is treated as a raw
    litellm model string with no extra params (still functional for anything
    litellm can resolve from env alone, e.g. ``openai/<name>``).
    """
    if catalog:
        for e in catalog:
            if model_or_label in (e.label, e.model_name):
                return e.model_name, e.params()
    return model_or_label, {}


if __name__ == "__main__":  # quick live check: `python -m promptchain.cli.model_catalog`
    cat = build_catalog()
    if not cat:
        print(no_catalog_message())
    by_source: Dict[str, int] = {}
    for e in cat:
        by_source[e.source] = by_source.get(e.source, 0) + 1
    print(f"catalog: {len(cat)} models  ({by_source})\n")
    for e in cat:
        r = e.reasoning
        rtag = f"think:{r['extract']}" if r.get("supports") else "—"
        sz = f"{e.size_gb}GB" if e.size_gb else ""
        print(f"  [{e.source:13s}] {e.model_name:40s} {rtag:22s} {sz}")
