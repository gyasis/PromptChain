"""cat — Computerized Adaptive Testing item selection / SE / stop rule (F2).

Pure, deterministic. See specs/013-model-profiler/research.md (D1).
"""
from __future__ import annotations

from typing import Optional, Sequence, Set

import numpy as np

from promptchain.profiler import irt
from promptchain.profiler.item_bank import ItemBank, ProbeItem


def select_next_item(
    theta_hat: float,
    bank: ItemBank,
    administered: Set[str],
) -> Optional[ProbeItem]:
    """Return the un-administered item with maximum Fisher information at theta_hat.

    Returns None if no items remain.
    """
    candidates = [it for it in bank.items if it.item_id not in administered]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda it: irt.fisher_information(theta_hat, it.a, it.b, it.c),
    )


def standard_error(
    theta_hat: float,
    administered_items: Sequence[ProbeItem],
) -> float:
    """SE(theta_hat) = 1/sqrt(sum I_i). Guards empty / zero-information sets."""
    total_info = sum(
        irt.fisher_information(theta_hat, it.a, it.b, it.c)
        for it in administered_items
    )
    if total_info <= 1e-12:
        return 1e6
    return float(1.0 / np.sqrt(total_info))


def cat_should_stop(
    se: float,
    n: int,
    *,
    tau: float = 0.3,
    max_items: int = 30,
    min_items: int = 5,
) -> bool:
    """Stop when (n >= min_items and se <= tau) or (n >= max_items)."""
    if n >= max_items:
        return True
    if n >= min_items and se <= tau:
        return True
    return False
