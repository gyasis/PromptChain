# Contract: Model Profiler public API surface

The public surface exported from `promptchain.profiler`. Async-first with sync wrappers (Principle
VI). Pure-math functions are synchronous and deterministic.

## Pure-math core (synchronous, deterministic — `irt`, `cat`, `composite`)

```python
# promptchain.profiler.irt
def prob_3pl(theta: float, a: float, b: float, c: float) -> float
    """P_i(θ) = c + (1−c)·σ(a(θ−b)).  Vectorized variant accepts np arrays."""

def fisher_information(theta: float, a: float, b: float, c: float) -> float
    """I_i(θ) = a²(P−c)²(1−P)/[P(1−c)²]."""

def estimate_theta_eap(responses, items, *, grid=None, prior=(0.0, 1.0)) -> tuple[float, float]
    """EAP θ̂ and SE over a θ grid with N(prior) prior. Returns (theta_hat, se)."""

def estimate_theta_wle(responses, items) -> tuple[float, float]
    """Warm's weighted-likelihood estimate; fallback for all-correct/all-incorrect."""

def estimate_theta(responses, items) -> tuple[float, float]
    """EAP, switching to WLE at the all-right/all-wrong extremes. The entry point."""

# promptchain.profiler.cat
def select_next_item(theta_hat: float, bank, administered: set[str]) -> ProbeItem
    """Pick the unused item with max Fisher information at theta_hat."""

def standard_error(theta_hat: float, administered_items) -> float
    """SE(θ̂) = 1/√(Σ I_i(θ̂))."""

def cat_should_stop(se: float, n: int, *, tau=0.3, max_items=30, min_items=5) -> bool
    """Stop when se ≤ tau (after min_items) or n ≥ max_items."""

# promptchain.profiler.composite
def capability(theta_hat: float) -> float                 # C = σ(θ̂)
def calibration_k(ece: float, *, sc=None, sem_entropy=None) -> float   # K = 1 − ECE (or weighted)
def cost_penalty(latency, latency_max, cost, cost_ref) -> float        # F
def omega(C: float, K: float, F: float) -> float          # Ω = 0.7·C·K − 0.3·F
def derive_jacket(profile_inputs) -> Jacket               # (Ω, θ̂, SE, C, deg_turn, ctx) → Jacket bands + escalation
```

## Harness + store (async core, sync wrappers — `probe`, `store`)

```python
# promptchain.profiler  (re-exported)
class ModelProfiler:
    def __init__(self, *, store_path=None, transcript_dir=None, item_bank=None): ...

    async def run_probe_async(self, model_id: str, *, bank=None, model_runner=None,
                              dimensions=None, persist=True) -> CapabilityProfile:
        """Run the CAT probe: per selected item, run an ISOLATED PromptChain session with the F1
        TranscriptEmitter attached (so each trial is an F1 transcript carrying `model`), score it,
        update θ̂/SE per dimension, stop at SE≤τ. Compute capability→tier/budget (US1); Ω+jacket
        (US2). Persist + return the CapabilityProfile. `model_runner` is injectable for tests
        (a fake model)."""

    def run_probe(self, model_id, **kw) -> CapabilityProfile      # sync wrapper (asyncio.run)

    def get_profile(self, model_id: str) -> CapabilityProfile | None
    def refine(self, model_id: str, session_metrics: dict, *, lam=0.2) -> CapabilityProfile  # EWMA; no-op if metrics empty
    def jacket_fit(self, model_id, jackets, *, bank=None, baseline=None) -> dict  # US3: Δθ per jacket → best
```

## Guarantees / invariants

- **Determinism**: the pure-math functions are deterministic; given the same responses + bank,
  `estimate_theta` returns the same θ̂ (reproducible profiles, SC-002).
- **Isolation**: each probe trial is a fresh PromptChain session — no shared context (FR-006).
- **F1 conformance**: trial transcripts are schema-valid F1 transcripts with `model` set; the
  profiler adds NO required transcript field (FR-007 / research D7).
- **Uncalibrated bank**: `run_probe` against an uncalibrated/empty bank raises a clear error
  rather than emitting a bogus θ̂ (edge case).
- **Graceful US3**: `jacket_fit` / GEPA reuse degrade to a no-op (returning the probe-derived
  jacket) when SIO experiment/optimize or telemetry are unavailable.
- **Injectable model**: `model_runner` lets integration tests drive the probe with a fake model
  (no live calls); the offline live smoke uses a real LAN ollama model.
