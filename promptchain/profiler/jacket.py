"""Data structures for the Model Profiler (F2): SkillEstimate, ProbeResponse, Jacket,
CapabilityProfile.

These are plain dataclasses with ``to_dict``/``from_dict`` for JSON persistence. The
on-disk shape is the contract in specs/013-model-profiler/contracts/profile-schema.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SkillEstimate:
    """Per-dimension ability result from the CAT probe."""

    dimension: str
    theta: float
    se: float
    n_items: int = 0
    capability: float = 0.0  # C = sigma(theta)
    low_precision: bool = False  # True if SE never reached tau within the item cap

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "theta": self.theta,
            "se": self.se,
            "n_items": self.n_items,
            "capability": self.capability,
            "low_precision": self.low_precision,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillEstimate":
        return cls(
            dimension=d["dimension"],
            theta=d["theta"],
            se=d["se"],
            n_items=d.get("n_items", 0),
            capability=d.get("capability", 0.0),
            low_precision=d.get("low_precision", False),
        )


@dataclass
class ProbeResponse:
    """One administered item against the target model (the IRT response + raw values)."""

    item_id: str
    correct: bool
    raw: dict = field(default_factory=dict)
    transcript_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Jacket:
    """The per-model derived configuration (the F3-facing artifact)."""

    tier: str
    budget_tokens: int
    mode: str  # single-shot | single-shot+retry | heavy-loop | escalate
    spawn_temp: float  # 1 - C
    compress_at: float  # ctx fraction
    max_turns: int
    role: str  # planner | executor | both
    escalate: bool = False
    system_prompt: Optional[str] = None  # optional GEPA-compiled per-model prompt (US3)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "budget_tokens": self.budget_tokens,
            "mode": self.mode,
            "spawn_temp": self.spawn_temp,
            "compress_at": self.compress_at,
            "max_turns": self.max_turns,
            "role": self.role,
            "escalate": self.escalate,
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Jacket":
        return cls(
            tier=d["tier"],
            budget_tokens=d["budget_tokens"],
            mode=d["mode"],
            spawn_temp=d["spawn_temp"],
            compress_at=d["compress_at"],
            max_turns=d["max_turns"],
            role=d["role"],
            escalate=d.get("escalate", False),
            system_prompt=d.get("system_prompt"),
        )


@dataclass
class CapabilityProfile:
    """The persisted per-model record (the unit in model_profiles.json)."""

    model_id: str
    skills: Dict[str, SkillEstimate] = field(default_factory=dict)
    capability: float = 0.0
    recommended_tier: str = "standard"
    budget_tokens: int = 4000
    effective_context: Optional[int] = None
    degradation_turn: Optional[int] = None
    omega: Optional[float] = None
    calibration_k: Optional[float] = None
    cost_penalty_f: Optional[float] = None
    jacket: Optional[Jacket] = None
    n_observations: int = 0
    created_ts: str = ""
    updated_ts: str = ""
    schema_version: int = 1

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "skills": {k: v.to_dict() for k, v in self.skills.items()},
            "capability": self.capability,
            "recommended_tier": self.recommended_tier,
            "budget_tokens": self.budget_tokens,
            "effective_context": self.effective_context,
            "degradation_turn": self.degradation_turn,
            "omega": self.omega,
            "calibration_k": self.calibration_k,
            "cost_penalty_f": self.cost_penalty_f,
            "jacket": self.jacket.to_dict() if self.jacket is not None else None,
            "n_observations": self.n_observations,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CapabilityProfile":
        jacket = d.get("jacket")
        return cls(
            model_id=d["model_id"],
            skills={k: SkillEstimate.from_dict(v) for k, v in d.get("skills", {}).items()},
            capability=d.get("capability", 0.0),
            recommended_tier=d.get("recommended_tier", "standard"),
            budget_tokens=d.get("budget_tokens", 4000),
            effective_context=d.get("effective_context"),
            degradation_turn=d.get("degradation_turn"),
            omega=d.get("omega"),
            calibration_k=d.get("calibration_k"),
            cost_penalty_f=d.get("cost_penalty_f"),
            jacket=Jacket.from_dict(jacket) if jacket is not None else None,
            n_observations=d.get("n_observations", 0),
            created_ts=d.get("created_ts", ""),
            updated_ts=d.get("updated_ts", ""),
            schema_version=d.get("schema_version", 1),
        )
