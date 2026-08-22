"""Probe item bank for the Model Profiler (F2).

A ``ProbeItem`` is one calibrated probe task with 3PL item parameters (a, b, c).
An ``ItemBank`` is the calibrated collection the CAT loop draws from. The bank can be
generated synthetically with KNOWN parameters (for the math tests + MVP) and filtered
for calibration quality (variance / accuracy / point-biserial).

See specs/013-model-profiler/data-model.md and research.md (D1, D2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

# The locked skill dimensions (data-model.md).
DIMENSIONS = (
    "instruction_following",
    "tool_call_reliability",
    "reasoning_depth",
    "degradation_turn",
    "effective_context",
    "format_sensitivity",
    "latency_throughput",
)

# Dimensions that contribute to the IRT capability score C (the others contribute raw
# values to the jacket: max_turns, budget, F).
CAPABILITY_DIMENSIONS = (
    "instruction_following",
    "tool_call_reliability",
    "reasoning_depth",
    "format_sensitivity",
)


@dataclass
class ProbeItem:
    """One calibrated probe item (3PL)."""

    item_id: str
    dimension: str
    a: float  # discrimination (> 0)
    b: float  # difficulty (on the theta scale)
    c: float = 0.0  # guessing (0 <= c < 1)
    prompt: str = ""
    scorer: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError(f"ProbeItem.a (discrimination) must be > 0, got {self.a}")
        if not (0.0 <= self.c < 1.0):
            raise ValueError(f"ProbeItem.c (guessing) must be in [0, 1), got {self.c}")
        if self.dimension not in DIMENSIONS:
            raise ValueError(
                f"ProbeItem.dimension {self.dimension!r} not in known DIMENSIONS {DIMENSIONS}"
            )

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "dimension": self.dimension,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "prompt": self.prompt,
            "scorer": self.scorer,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeItem":
        return cls(
            item_id=d["item_id"],
            dimension=d["dimension"],
            a=d["a"],
            b=d["b"],
            c=d.get("c", 0.0),
            prompt=d.get("prompt", ""),
            scorer=d.get("scorer", ""),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class ItemBank:
    """A calibrated collection of probe items."""

    items: List[ProbeItem]
    calibrated: bool = True

    def __post_init__(self) -> None:
        # An empty bank cannot be calibrated; capability scoring must refuse it.
        if not self.items:
            self.calibrated = False

    def by_dimension(self, dimension: str) -> List[ProbeItem]:
        return [it for it in self.items if it.dimension == dimension]

    def dimensions_present(self) -> List[str]:
        seen: List[str] = []
        for it in self.items:
            if it.dimension not in seen:
                seen.append(it.dimension)
        return seen

    @classmethod
    def synthetic(
        cls,
        *,
        dimensions: Sequence[str] = DIMENSIONS,
        n_per_dim: int = 5,
        seed: int = 0,
    ) -> "ItemBank":
        """Build a bank with KNOWN, deterministic 3PL parameters.

        Difficulties span roughly [-2, 2]; discriminations are moderate-to-high so items
        carry information; guessing is a small constant. Deterministic given ``seed``.
        """
        rng = np.random.default_rng(seed)
        items: List[ProbeItem] = []
        for dim in dimensions:
            # difficulties evenly spread so the bank covers a range of ability
            bs = np.linspace(-2.0, 2.0, n_per_dim)
            for k in range(n_per_dim):
                a = float(round(rng.uniform(0.8, 2.0), 4))
                b = float(round(float(bs[k]), 4))
                c = 0.2 if dim in ("instruction_following", "tool_call_reliability") else 0.0
                items.append(
                    ProbeItem(
                        item_id=f"{dim}-{k}",
                        dimension=dim,
                        a=a,
                        b=b,
                        c=c,
                        prompt=f"[synthetic probe] {dim} #{k}",
                        scorer=dim,
                    )
                )
        return cls(items=items, calibrated=True)

    @staticmethod
    def filter_quality(
        items: Sequence[ProbeItem],
        responses: Dict[str, Sequence[int]],
        *,
        min_var: float = 0.01,
        max_acc: float = 0.95,
        min_point_biserial: float = 0.1,
    ) -> List[ProbeItem]:
        """Keep only well-behaved calibration items.

        ``responses`` maps item_id -> a 0/1 response vector across the SAME ordered set of
        calibration subjects. An item is kept iff:
          * response variance >= ``min_var`` (drops no-variance items),
          * accuracy (mean correct) <= ``max_acc`` (drops near-ceiling items), and
          * point-biserial correlation with the subject REST-score >= ``min_point_biserial``
            (drops non-discriminating / reverse items).
        Items without response data are dropped.
        """
        item_ids = [it.item_id for it in items if it.item_id in responses]
        if not item_ids:
            return []
        matrix = np.array([np.asarray(responses[i], dtype=float) for i in item_ids])  # (I, S)
        totals = matrix.sum(axis=0)  # (S,) total score per subject

        kept: List[ProbeItem] = []
        by_id = {it.item_id: it for it in items}
        for row, iid in zip(matrix, item_ids):
            acc = float(row.mean())
            var = float(row.var())
            if var < min_var:
                continue
            if acc > max_acc:
                continue
            # point-biserial vs the REST-score (total minus this item) to avoid spurious
            # self-correlation.
            rest = totals - row
            if rest.std() == 0.0 or row.std() == 0.0:
                continue
            pb = float(np.corrcoef(row, rest)[0, 1])
            if pb < min_point_biserial:
                continue
            kept.append(by_id[iid])
        return kept
