"""promptchain.profiler — Model Profiler (F2).

Public API is populated across the F2 build (US1 probe/profile, US2 composite/jacket,
US3 refine/fit). See specs/013-model-profiler/contracts/profiler-api.md.
"""
from promptchain.profiler.item_bank import ItemBank, ProbeItem
from promptchain.profiler.jacket import (
    CapabilityProfile,
    Jacket,
    ProbeResponse,
    SkillEstimate,
)
from promptchain.profiler.probe import ModelProfiler
from promptchain.profiler.prompt_generator import ModelPromptGenerator


def run_probe(model_id, **kw) -> CapabilityProfile:
    """Convenience: probe *model_id* with a default ModelProfiler and return its profile."""
    return ModelProfiler().run_probe(model_id, **kw)


__all__ = [
    "ModelProfiler",
    "ModelPromptGenerator",
    "run_probe",
    "CapabilityProfile",
    "Jacket",
    "SkillEstimate",
    "ProbeResponse",
    "ProbeItem",
    "ItemBank",
]
