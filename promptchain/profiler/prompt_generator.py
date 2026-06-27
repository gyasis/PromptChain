"""prompt_generator — the per-model DSPy jacket generator (F2, US3).

Maps a per-model :class:`CapabilityProfile` to a :class:`Jacket`, optionally attaching a
DSPy-compiled per-model ``system_prompt``. Distinct from SIO's suggestion_generator (which
mines cross-session error patterns) — this is profile → jacket for a single model.

When no DSPy LM is configured (or any error occurs in the DSPy path) it DEGRADES GRACEFULLY:
it returns the profile's existing jacket, or derives one from the profile. It never raises.
"""
from __future__ import annotations

import dataclasses

from promptchain.profiler import composite
from promptchain.profiler.jacket import CapabilityProfile, Jacket


class ModelPromptGenerator:
    """Generate a per-model Jacket from a CapabilityProfile (DSPy path optional)."""

    def __init__(self, lm=None) -> None:
        self.lm = lm

    def generate(self, profile: CapabilityProfile) -> Jacket:
        base = self._base_jacket(profile)
        try:
            system_prompt = self._compile_system_prompt(profile)
        except Exception:
            # Any failure in the DSPy path → degrade to the base jacket unchanged.
            return base
        if not system_prompt:
            return base
        return dataclasses.replace(base, system_prompt=system_prompt)

    # ------------------------------------------------------------------ #
    def _base_jacket(self, profile: CapabilityProfile) -> Jacket:
        """The probe-derived jacket, or one derived on the fly from the profile."""
        if profile.jacket is not None:
            return profile.jacket
        return composite.derive_jacket(
            omega=(profile.omega if profile.omega is not None else 0.0),
            theta=0.0,
            se=1.0,
            capability=profile.capability,
        )

    def _compile_system_prompt(self, profile: CapabilityProfile):
        """Use DSPy to compile a short per-model system prompt. Returns None if unavailable."""
        try:
            import dspy
        except Exception:
            return None

        # Only attempt if an LM is configured (explicitly or globally on dspy.settings).
        configured = self.lm is not None or getattr(dspy.settings, "lm", None) is not None
        if not configured:
            return None

        class _JacketPrompt(dspy.Signature):
            """Write a concise system prompt tuned to a model's measured capability."""

            model_id = dspy.InputField()
            capability = dspy.InputField()
            recommended_tier = dspy.InputField()
            system_prompt = dspy.OutputField(desc="a short system prompt for this model")

        predictor = dspy.Predict(_JacketPrompt)
        kwargs = {}
        if self.lm is not None:
            kwargs["lm"] = self.lm
        out = predictor(
            model_id=profile.model_id,
            capability=f"{profile.capability:.3f}",
            recommended_tier=profile.recommended_tier,
            **kwargs,
        )
        prompt = getattr(out, "system_prompt", None)
        return prompt.strip() if isinstance(prompt, str) and prompt.strip() else None
