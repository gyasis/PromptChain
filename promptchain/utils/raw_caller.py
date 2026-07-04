"""RawCaller — a litellm-free, passthrough HTTP client for PromptChain (issue #31).

Why: litellm is a hidden normalization layer. For tool calls it can inject tool-use instructions
into the system prompt, force ollama `format=json`, parse raw output into `tool_calls` itself, rewrite
roles/params, and `drop_params=True` silently deletes params. That contaminates a "naked model"
experiment and couples PromptChain to litellm's provider conventions.

RawCaller is the opposite: a STRICT PASSTHROUGH. It POSTs exactly the messages + explicit params you
give it — no system-prompt injection, no role rewriting, no tool-prompt injection, no output parsing,
no silent defaults — and returns the raw provider JSON (plus thin `.text`/`.usage` accessors). Tool
calling is the caller's job via `promptchain.utils.tool_normalization` (render + parse), so WE own the
normalization and its leniency dial.

Stdlib only (urllib) on purpose: the whole point is a caller with no hidden behavior to audit.
litellm stays PromptChain's default; RawCaller is OPT-IN for experiments, naked models, and
provider-agnostic deployments (ollama /api/chat, OpenAI-compatible /v1, llama.cpp, MLX, base HTTP).
"""
import json
import urllib.request
import urllib.error


class RawResponse:
    """Thin wrapper over the raw provider JSON. `.raw` is the untouched dict; accessors normalize
    only the two fields callers usually need, without hiding anything."""

    def __init__(self, raw: dict, api_style: str):
        self.raw = raw
        self.api_style = api_style

    @property
    def text(self) -> str:
        if self.api_style == "ollama":
            return ((self.raw.get("message") or {}).get("content") or "")
        ch = (self.raw.get("choices") or [{}])[0]
        return ((ch.get("message") or {}).get("content") or "")

    @property
    def usage(self) -> dict:
        if self.api_style == "ollama":
            return {"prompt_tokens": int(self.raw.get("prompt_eval_count", 0) or 0),
                    "completion_tokens": int(self.raw.get("eval_count", 0) or 0)}
        return self.raw.get("usage") or {}


class RawCaller:
    """Passthrough caller. `api_style` selects the wire format:
      - "openai": POST {base_url}/chat/completions with {model, messages, **params}
      - "ollama": POST {base_url}/api/chat        with {model, messages, stream:false, options:params}
    `base_url` should include the /v1 for openai-style endpoints (e.g. http://host:8771/v1)."""

    def __init__(self, base_url: str, model: str | None = None, api_key: str | None = None,
                 api_style: str = "openai", default_params: dict | None = None,
                 headers: dict | None = None, timeout: int = 600):
        if api_style not in ("openai", "ollama"):
            raise ValueError("api_style must be 'openai' or 'ollama'")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.api_style = api_style
        self.default_params = dict(default_params or {})
        self.headers = dict(headers or {})
        self.timeout = timeout

    def build_payload(self, messages, model=None, params=None) -> dict:
        """The EXACT dict that will be POSTed. Exposed so passthrough purity is testable/auditable.
        No injection: `messages` is passed through verbatim; only explicit params are merged."""
        model = model or self.model
        p = {**self.default_params, **(params or {})}
        if self.api_style == "ollama":
            payload = {"model": model, "messages": messages, "stream": False}
            if p:
                payload["options"] = p
            return payload
        payload = {"model": model, "messages": messages}
        payload.update(p)
        return payload

    def _endpoint(self) -> str:
        return self.base_url + ("/api/chat" if self.api_style == "ollama" else "/chat/completions")

    def complete(self, messages, model=None, params=None) -> RawResponse:
        payload = self.build_payload(messages, model=model, params=params)
        data = json.dumps(payload).encode("utf-8")
        hdr = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            hdr["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self._endpoint(), data=data, headers=hdr, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        return RawResponse(raw, self.api_style)
