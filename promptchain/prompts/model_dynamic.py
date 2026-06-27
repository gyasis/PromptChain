"""F3 Dynamic Prompt Layer — the per-model generator (the integration seam).

``DynamicModelPromptGenerator`` is a ``BasePromptBuilder`` drop-in that assembles
a per-model system prompt from the F2 capability profile/jacket, layered
*additively* over the intact static base (``DynamicTUIPromptGenerator`` /
``TUI_FOUNDATION_PROMPT``).

Assembly (see ``contracts/generator-api.md`` + ``contracts/prompt-layout.md``):

1. reject an empty objective (base contract).
2. resolve the model (explicit kwarg → construction-time → None=default CORE).
3. load the profile from the store (tolerant — any failure → None).
4. select the prompt tier (None profile → CORE).
5. render the static base + live tool inventory VERBATIM (the parity floor).
6. gather the tier's optional modules.
7. resolve the effective budget (jacket → profile → default 1000).
8. fit base + modules to budget (base never dropped/truncated).
9. frame with the family FORMAT preamble (default/unknown → identity).

Deterministic for fixed ``(objective, tools, model, profile)``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from promptchain.prompts.budget import fit_to_budget, measure
from promptchain.prompts.family import adapt_format, family_of
from promptchain.prompts.tiers import modules_for_tier, select_tier
from promptchain.prompts.toolshim import render_tools_block, resolve_tool_mode
from promptchain.prompts.tui_dynamic import DynamicTUIPromptGenerator

_TARGET_CAP = 1000
_HARD_CAP = 1500
_DEFAULT_BUDGET = 1000


class DynamicModelPromptGenerator:
    """Per-model ``BasePromptBuilder`` over the static TUI foundation."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        store=None,
        base_generator=None,
        store_path=None,
    ) -> None:
        self._model = model
        self._base = base_generator or DynamicTUIPromptGenerator()
        self._store = store
        self._store_path = store_path
        self._store_loaded = store is not None

    def _get_store(self):
        """Return the profile store, lazily building a default F2 store if absent.

        Any failure (missing file, import error) is swallowed — the generator
        then behaves as "no profile" (default CORE).
        """
        if self._store is not None or self._store_loaded:
            return self._store
        self._store_loaded = True
        try:
            from promptchain.profiler.probe import ModelProfiler

            self._store = ModelProfiler(store_path=self._store_path)
        except Exception:
            self._store = None
        return self._store

    def _load_profile(self, model: Optional[str]):
        if not model:
            return None
        store = self._get_store()
        if store is None:
            return None
        try:
            return store.get_profile(model)
        except Exception:
            return None

    def generate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
        context: Optional[str] = None,
        *,
        model: Optional[str] = None,
    ) -> str:
        if objective is None or not objective.strip():
            raise ValueError("objective must be a non-empty string")

        resolved_model = model or self._model
        profile = self._load_profile(resolved_model)
        tier = select_tier(profile)

        jacket = profile.jacket if profile is not None else None
        mode = resolve_tool_mode(jacket)
        if mode == "native":
            # Native tools embedded by the base (AVAILABLE TOOLS / MCP TOOLS).
            # The static base is emitted VERBATIM (SC-003) — never mutated.
            base_text = self._base.generate(objective, tools, context)
        else:
            # Shim mode: static base (VERBATIM) WITHOUT the native tool inventory,
            # then append the <tools> JSON-in-text protocol block. The shim block
            # is part of the never-dropped parity floor (passed to fit_to_budget as
            # ``base``). Native vs shim is discriminated by the shim block's own
            # content, NOT by rewriting the foundation's tags.
            base_text = self._base.generate(objective, [], context)
            base_text = base_text + "\n\n" + render_tools_block(tools)

        modules = modules_for_tier(tier)

        if jacket is not None:
            effective_budget = jacket.budget_tokens
        elif profile is not None:
            effective_budget = profile.budget_tokens
        else:
            effective_budget = _DEFAULT_BUDGET

        target_max = min(effective_budget, _TARGET_CAP)
        assembled_core, _dropped = fit_to_budget(
            base_text, modules, target_max=target_max, hard_cap=_HARD_CAP
        )

        family = family_of(resolved_model or "")
        framed = adapt_format([assembled_core], family)
        return "\n\n".join(framed)

    def get_token_estimate(
        self,
        objective: str,
        tools: List[Dict[str, Any]],
    ) -> int:
        return measure(self.generate(objective, tools))
