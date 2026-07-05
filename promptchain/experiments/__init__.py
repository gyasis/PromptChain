"""promptchain.experiments — the technique graduation harness (spec-015).

The GATE (`run_gate`) is the turnstile: a technique piloted on the experimental bench
(EnhancedAgenticStepProcessor) may graduate to a production `AgenticStepProcessor` param ONLY by passing
this gate on a held-out set. See `promptchain/experiments/README.md` and specs/015-technique-graduation-pipeline/.
"""
from .gate import run_gate, GateReport, split

__all__ = ["run_gate", "GateReport", "split"]
