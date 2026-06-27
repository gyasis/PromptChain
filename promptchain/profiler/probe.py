"""probe — the Model Profiler CAT harness (F2, US1).

``ModelProfiler.run_probe`` runs a Computerized Adaptive Testing (CAT) probe over a
calibrated ``ItemBank``: per selected item it runs the target model (injectable for
tests), scores the reply, updates the per-dimension ability estimate (θ̂/SE via
``irt.estimate_theta``), and stops at SE ≤ τ. Each trial is written as a schema-valid
F1 transcript (via the real ``TranscriptEmitter``) carrying the probed model id, so the
probe stream is mineable by SIO (FR-007).

The composite Ω + jacket derivation is US2 and intentionally left None here.

See specs/013-model-profiler/contracts/profiler-api.md.
"""
from __future__ import annotations

import asyncio
import inspect
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from promptchain.observability.transcript_emitter import (
    TranscriptEmitter,
    TranscriptEmitterConfig,
)
from promptchain.profiler import cat, composite, irt, scoring
from promptchain.profiler.item_bank import (
    CAPABILITY_DIMENSIONS,
    ItemBank,
    ProbeItem,
)
from promptchain.profiler.jacket import (
    CapabilityProfile,
    ProbeResponse,
    SkillEstimate,
)
from promptchain.profiler.store import ProfileStore
from promptchain.utils.execution_events import ExecutionEvent, ExecutionEventType

# Tier bands keyed off the aggregate capability C. (capability_floor, tier, budget_tokens)
_TIER_BANDS = (
    (0.75, "lean", 2000),
    (0.55, "standard", 4000),
    (0.35, "rich", 8000),
    (0.0, "max-rich", 16000),
)

# SE precision target — mirrors cat.cat_should_stop's tau.
_SE_TAU = 0.3


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _tier_for_capability(c: float) -> tuple[str, int]:
    for floor, tier, budget in _TIER_BANDS:
        if c >= floor:
            return tier, budget
    return "max-rich", 16000


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


class ModelProfiler:
    """Probe a model's per-dimension ability and persist a CapabilityProfile."""

    def __init__(
        self,
        *,
        store_path=None,
        transcript_dir=None,
        item_bank: Optional[ItemBank] = None,
    ) -> None:
        self.store_path = Path(
            store_path
            if store_path is not None
            else Path.home() / ".promptchain" / "model_profiles.json"
        )
        self.transcript_dir = Path(
            transcript_dir
            if transcript_dir is not None
            else Path.home() / ".promptchain" / "transcripts"
        )
        self.item_bank = item_bank

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def run_probe_async(
        self,
        model_id: str,
        *,
        bank: Optional[ItemBank] = None,
        model_runner: Optional[Callable] = None,
        scorer: Optional[Callable] = None,
        dimensions: Optional[Sequence[str]] = None,
        persist: bool = True,
    ) -> CapabilityProfile:
        bank = bank if bank is not None else self.item_bank
        if bank is None or not getattr(bank, "calibrated", False):
            raise ValueError(
                "ModelProfiler.run_probe requires a calibrated, non-empty item bank "
                "(uncalibrated/empty item bank rejected)."
            )

        runner = model_runner if model_runner is not None else self._default_runner(model_id)
        score = scorer if scorer is not None else scoring.score_response
        dims = list(dimensions) if dimensions else bank.dimensions_present()

        project = _slug(model_id)
        emitter = TranscriptEmitter(
            TranscriptEmitterConfig(
                enabled=True, base_dir=Path(self.transcript_dir), project=project
            )
        )

        skills: dict[str, SkillEstimate] = {}
        for dim in dims:
            est = await self._probe_dimension(
                model_id, dim, bank, runner, score, emitter, project
            )
            if est is not None:
                skills[dim] = est

        # Aggregate capability over the IRT capability dimensions present (fallback: all).
        cap_dims = [d for d in skills if d in CAPABILITY_DIMENSIONS]
        if cap_dims:
            capability = sum(skills[d].capability for d in cap_dims) / len(cap_dims)
        elif skills:
            capability = sum(e.capability for e in skills.values()) / len(skills)
        else:
            capability = 0.0

        tier, budget = _tier_for_capability(capability)

        # US2: composite Ω + jacket. With no measured ECE/cost on a local probe, K
        # defaults to 1-ECE=1.0 (refined from telemetry in US3) and the cost penalty
        # F=0; both fields are persisted so F3 can read them.
        cap_thetas = [skills[d].theta for d in cap_dims] or [e.theta for e in skills.values()]
        mean_theta = sum(cap_thetas) / len(cap_thetas) if cap_thetas else 0.0
        worst_se = max((e.se for e in skills.values()), default=float("inf"))
        # degradation_turn / effective_context are raw values from the non-capability
        # scorers; not threaded through the capability CAT yet → left None (optional fields).
        degradation_turn = None
        effective_context = None
        calibration_k = composite.calibration_k(0.0)
        cost_penalty_f = 0.0
        omega_value = composite.omega(capability, calibration_k, cost_penalty_f)
        jacket = composite.derive_jacket(
            omega=omega_value,
            theta=mean_theta,
            se=worst_se,
            capability=capability,
            degradation_turn=degradation_turn,
            effective_context=effective_context,
        )

        now = datetime.now(timezone.utc).isoformat()
        profile = CapabilityProfile(
            model_id=model_id,
            skills=skills,
            capability=capability,
            recommended_tier=tier,
            budget_tokens=budget,
            effective_context=effective_context,
            degradation_turn=degradation_turn,
            omega=omega_value,
            calibration_k=calibration_k,
            cost_penalty_f=cost_penalty_f,
            jacket=jacket,
            n_observations=sum(e.n_items for e in skills.values()),
            created_ts=now,
            updated_ts=now,
            schema_version=1,
        )

        if persist:
            ProfileStore(self.store_path).upsert(profile)
        return profile

    def run_probe(self, model_id: str, **kw) -> CapabilityProfile:
        """Sync wrapper around :meth:`run_probe_async`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_probe_async(model_id, **kw))
        # A loop is already running — execute in a dedicated thread with its own loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda: asyncio.run(self.run_probe_async(model_id, **kw))
            ).result()

    def get_profile(self, model_id: str) -> Optional[CapabilityProfile]:
        return ProfileStore(self.store_path).get(model_id)

    # ------------------------------------------------------------------ #
    # CAT loop for one dimension
    # ------------------------------------------------------------------ #
    async def _probe_dimension(
        self,
        model_id: str,
        dim: str,
        bank: ItemBank,
        runner: Callable,
        score: Callable,
        emitter: TranscriptEmitter,
        project: str,
    ) -> Optional[SkillEstimate]:
        dim_items = bank.by_dimension(dim)
        if not dim_items:
            return None
        # Wrap the dimension's items in an ItemBank so cat.select_next_item can iterate
        # its .items; the calibrated flag is irrelevant for selection.
        dim_bank = ItemBank(items=list(dim_items))

        theta_hat = 0.0
        se = float("inf")
        administered: set[str] = set()
        administered_items: list[ProbeItem] = []
        responses: list[int] = []

        while True:
            item = cat.select_next_item(theta_hat, dim_bank, administered)
            if item is None:
                break

            trial_index = len(responses)
            reply = await self._run_one(runner, item)
            correct, raw = score(item, reply)

            await self._emit_trial(
                emitter, project, model_id, dim, item, trial_index, reply
            )

            responses.append(1 if correct else 0)
            administered.add(item.item_id)
            administered_items.append(item)

            theta_hat, se = irt.estimate_theta(responses, administered_items)
            n = len(responses)
            if cat.cat_should_stop(se, n):
                break

        if not administered_items:
            return None

        capability = _sigmoid(theta_hat)
        return SkillEstimate(
            dimension=dim,
            theta=theta_hat,
            se=se,
            n_items=len(responses),
            capability=capability,
            low_precision=(se > _SE_TAU),
        )

    # ------------------------------------------------------------------ #
    # Trial execution + transcript emission
    # ------------------------------------------------------------------ #
    @staticmethod
    async def _run_one(runner: Callable, item: ProbeItem) -> dict:
        """Call *runner* (sync or async) for one item; never raise out of a trial."""
        try:
            result = runner(item)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                result = {"text": str(result), "usage": None, "latency_ms": None}
            return result
        except Exception as exc:  # one failed trial must not crash the probe
            return {"text": "", "usage": None, "latency_ms": None, "error": str(exc)}

    async def _emit_trial(
        self,
        emitter: TranscriptEmitter,
        project: str,
        model_id: str,
        dim: str,
        item: ProbeItem,
        trial_index: int,
        reply: dict,
    ) -> None:
        """Drive the F1 TranscriptEmitter to write ONE schema-valid trial transcript."""
        session_id = f"{project}-{dim}-{item.item_id}-{trial_index}"
        usage = reply.get("usage")
        latency = reply.get("latency_ms")
        total_tokens = (usage or {}).get("total_tokens")

        await emitter.handle_event(
            ExecutionEvent(
                ExecutionEventType.CHAIN_START,
                metadata={"chain_id": session_id, "project": project},
            )
        )
        await emitter.handle_event(
            ExecutionEvent(
                ExecutionEventType.MODEL_CALL_END,
                model_name=model_id,
                metadata={
                    "chain_id": session_id,
                    "call_id": f"model-{session_id}",
                    "usage": usage,
                    "execution_time_ms": latency,
                },
            )
        )
        await emitter.handle_event(
            ExecutionEvent(
                ExecutionEventType.CHAIN_END,
                metadata={
                    "chain_id": session_id,
                    "total_tokens": total_tokens,
                    "execution_time_ms": latency,
                },
            )
        )

    # ------------------------------------------------------------------ #
    # Default (live) model runner — exercised by the W11 live smoke, not the suite.
    # ------------------------------------------------------------------ #
    def _default_runner(self, model_id: str) -> Callable:
        async def _default_model_runner(item: ProbeItem) -> dict:
            import time

            import litellm

            t0 = time.perf_counter()
            try:
                resp = await litellm.acompletion(
                    model=model_id,
                    messages=[{"role": "user", "content": item.prompt or item.item_id}],
                )
                dt = (time.perf_counter() - t0) * 1000.0
                text = resp.choices[0].message.content or ""
                u = getattr(resp, "usage", None)
                if u is not None:
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(u, "total_tokens", 0) or 0,
                    }
                else:
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                return {"text": text, "usage": usage, "latency_ms": dt}
            except Exception as exc:
                dt = (time.perf_counter() - t0) * 1000.0
                return {
                    "text": "",
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "latency_ms": dt,
                    "error": str(exc),
                }

        return _default_model_runner
