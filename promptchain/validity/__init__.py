"""promptchain.validity — the agent-facing EXPERIMENT-VALIDITY toolkit (issue #40).

Brings the validity work "home" under one cohesive, agent-specific namespace. Two layers:
  • PROCEDURAL checks (validity_suite): technique_fired · no_regression · harness_faithful ·
    no_silent_defaults · variance_bounded · negative_control · monotonic · ValiditySuite runner.
  • STATISTICAL inference (validity_stats): mcnemar · wilson_ci · holm_bonferroni · min_detectable_effect
    · cohens_h/d · cohens_kappa · bootstrap_ci · compare_paired_binary · (lazy scipy) rank tests.

Plus an OKF knowledge bundle at `promptchain/validity/okf/` — the AGENT-READABLE instruction set
(when/why to run each check), so an agent can both CALL the checks and READ the reasoning. Point an OKF
consumer at that dir. `okf_path()` returns it.

The implementations live in promptchain.utils.{validity_suite,validity_stats}; this package is the
convenient front door. Run the checks BEFORE trusting any experiment's aggregate score.
"""
import os

from promptchain.utils.validity_suite import (  # procedural
    Check, ValiditySuite,
    technique_fired, no_regression, above_noise, harness_faithful,
    no_silent_defaults, variance_bounded, negative_control, monotonic,
)
from promptchain.utils.validity_stats import (  # statistical
    mcnemar, wilson_ci, holm_bonferroni, min_detectable_effect,
    cohens_h, cohens_d, cohens_kappa, bootstrap_ci, compare_paired_binary,
    wilcoxon_signed_rank, mann_whitney_u,
)


def okf_path():
    """Absolute path to the OKF instruction-set bundle (the agent-readable validity knowledge)."""
    return os.path.join(os.path.dirname(__file__), "okf")


__all__ = [
    "Check", "ValiditySuite",
    "technique_fired", "no_regression", "above_noise", "harness_faithful",
    "no_silent_defaults", "variance_bounded", "negative_control", "monotonic",
    "mcnemar", "wilson_ci", "holm_bonferroni", "min_detectable_effect",
    "cohens_h", "cohens_d", "cohens_kappa", "bootstrap_ci", "compare_paired_binary",
    "wilcoxon_signed_rank", "mann_whitney_u", "okf_path",
]
