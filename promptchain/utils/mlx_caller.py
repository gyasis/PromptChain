"""MLXCaller — in-process Apple-Silicon (MLX) caller for PromptChain (issue #35).

WHY (tracked): experiment models must load on the Mac Studio (Apple Silicon, Metal) — the orchestrating
Linux box can't afford them. LlamaCppCaller (#33/#34) covers crushed GGUF quants in-process on the Mac;
MLXCaller is the sibling for ORIGINAL/un-quantized HF weights, where MLX (`mlx-lm`) is the best
Apple-Silicon runtime (native, unified-memory-efficient, first-class Metal).

Naked by design: no litellm, no ollama — so nothing injects tool prompts, forces `format=json`, or
parses output. The ONE normalization is the chat template, applied EXPLICITLY here (we control it, or
skip it). Tool-calling is the caller's job via promptchain.utils.tool_normalization (render + parse).
`mlx-lm` is a lazy, OPTIONAL dependency (Apple Silicon only); importing this module never requires it.
"""


class MLXResponse:
    """Wrapper over the generated text (mlx_lm.generate returns text). `.text` is the raw output;
    `.usage` is populated only if the caller passes token counts in (mlx_lm generate is text-only)."""

    def __init__(self, text: str, usage: dict | None = None):
        self.text = text or ""
        self.usage = usage or {}


class MLXCaller:
    """Load HF weights with mlx_lm and generate in-process on Apple Silicon.

    use_chat_template=True → apply the tokenizer's chat template (roles/special tokens the model
    expects) — the only normalization, and it's explicit/ours. False → raw concatenation of contents.
    """

    def __init__(self, model_path, default_params=None, use_chat_template=True):
        self.model_path = model_path
        self.default_params = dict(default_params or {})
        self.use_chat_template = use_chat_template
        self._model = None
        self._tok = None

    def _ensure_loaded(self):
        if self._model is None:
            try:
                from mlx_lm import load
            except ImportError as e:  # pragma: no cover - env-dependent (Apple Silicon only)
                raise ImportError(
                    "MLXCaller requires the optional dependency 'mlx-lm' (Apple Silicon only). "
                    "Install on the Mac: pip install mlx-lm"
                ) from e
            self._model, self._tok = load(self.model_path)
        return self._model, self._tok

    def build_prompt(self, messages, tokenizer=None) -> str:
        """Turn messages into the prompt string. With use_chat_template, apply the tokenizer's chat
        template (add generation prompt, no tokenize); else raw-concat the message contents.
        `tokenizer` is injectable so this is unit-testable without loading a model."""
        tok = tokenizer if tokenizer is not None else self._tok
        if self.use_chat_template and tok is not None and hasattr(tok, "apply_chat_template"):
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join((m.get("content") or "") for m in messages)

    def resolve_params(self, params=None) -> dict:
        """Merge default + per-call params (call overrides). Offline/unit-testable core."""
        return {**self.default_params, **(params or {})}

    def complete(self, messages, params=None) -> MLXResponse:
        model, tok = self._ensure_loaded()
        from mlx_lm import generate
        prompt = self.build_prompt(messages, tokenizer=tok)
        merged = self.resolve_params(params)
        text = generate(model, tok, prompt=prompt, **merged)
        return MLXResponse(text)
