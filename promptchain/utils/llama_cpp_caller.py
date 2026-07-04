"""LlamaCppCaller — in-process llama.cpp (llama-cpp-python) caller for PromptChain (issue #33).

WHY (tracked): litellm AND ollama both normalize tool calls — injecting tool-use prompts, forcing
`format=json`, and parsing raw output into `tool_calls`. So a naked-model experiment routed through
them can never surface a malformed call, and we can't test whether a strong corrector repairs a
weak/quantized model's mistakes. Structural validity is already a solved problem (llama.cpp GBNF
grammars / vLLM guided decoding guarantee valid JSON); the unsolved gap is semantic/judgment
correctness, which is where quantized models degrade.

LlamaCppCaller runs GGUF weights DIRECTLY (no litellm, no ollama) with a GRAMMAR TOGGLE:
  • grammar=None       -> RAW, malformable output (repair experiments on crushed models)
  • grammar=<GBNF str> -> structurally-constrained valid output (the production / solved-syntax path)

Tool-calling is the caller's job via promptchain.utils.tool_normalization (render + parse, leniency
dial). litellm stays PromptChain's default; this is OPT-IN for experiments and local Metal inference.
`llama-cpp-python` is a lazy, OPTIONAL dependency — importing this module never requires it.
"""

_UNSET = object()  # distinguishes "grammar not passed" (use instance default) from "grammar=None" (raw)


class LlamaCppResponse:
    """Wrapper over llama-cpp-python's OpenAI-shaped chat-completion dict. `.raw` is untouched."""

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def text(self) -> str:
        ch = (self.raw.get("choices") or [{}])[0]
        return ((ch.get("message") or {}).get("content") or "")

    @property
    def usage(self) -> dict:
        return self.raw.get("usage") or {}


class LlamaCppCaller:
    """Load a GGUF model with llama-cpp-python and run chat completions in-process.

    grammar (instance-level default) and the per-call `grammar` arg control decoding:
      unset (default)    -> use the instance default
      None               -> RAW output (no grammar; malformation survives)
      "<GBNF string>"    -> constrained decoding (guaranteed to match the grammar)
    """

    def __init__(self, model_path, n_ctx=8192, n_gpu_layers=-1, chat_format=None,
                 default_params=None, grammar=None, verbose=False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers          # -1 = offload all layers to Metal/GPU
        self.chat_format = chat_format            # None = use the GGUF's built-in template
        self.default_params = dict(default_params or {})
        self.grammar_default = grammar            # GBNF string or None (None = raw)
        self.verbose = verbose
        self._llm = None                          # lazy-loaded Llama instance

    def _ensure_loaded(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as e:  # pragma: no cover - env-dependent
                raise ImportError(
                    "LlamaCppCaller requires the optional dependency 'llama-cpp-python'. "
                    "Install: pip install llama-cpp-python  "
                    "(macOS/Metal: CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python)"
                ) from e
            self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx,
                              n_gpu_layers=self.n_gpu_layers, chat_format=self.chat_format,
                              verbose=self.verbose)
        return self._llm

    def resolve(self, params=None, grammar=_UNSET):
        """Merge params and resolve the grammar to use (per-call arg overrides the instance default).
        Returns (merged_params, grammar_gbnf_or_None). Pure/offline — the unit-testable core."""
        merged = {**self.default_params, **(params or {})}
        gbnf = self.grammar_default if grammar is _UNSET else grammar
        return merged, gbnf

    def complete(self, messages, params=None, grammar=_UNSET) -> LlamaCppResponse:
        llm = self._ensure_loaded()
        merged, gbnf = self.resolve(params, grammar)
        gram = None
        if gbnf:
            from llama_cpp import LlamaGrammar
            gram = LlamaGrammar.from_string(gbnf)
        raw = llm.create_chat_completion(messages=messages, grammar=gram, **merged)
        return LlamaCppResponse(raw)
