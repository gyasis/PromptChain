"""Unit tests for LightRAGBranchingThoughts.

The pattern queries LightRAG once per branch mode (local / global / hybrid),
asks an LLM judge to score the resulting hypotheses and keeps the best one.
Both the retrieval backend and ``litellm.acompletion`` are mocked; the judge
model is named by the test, never defaulted.

Already covered by tests/test_lightrag_branching.py (not repeated here):
initialisation, per-mode queries, judge-prompt and judge-output parsing,
best-hypothesis selection, the full ``execute`` flow.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptchain.integrations.lightrag.branching import (
    BranchingConfig,
    Hypothesis,
    LightRAGBranchingThoughts,
)
from promptchain.patterns.base import BasePattern

JUDGE_MODEL = "test-provider/judge-model"


@pytest.fixture
def core():
    mock = AsyncMock()
    mock.local_query.return_value = "local answer"
    mock.global_query.return_value = "global answer"
    mock.hybrid_query.return_value = {"response": "hybrid answer"}
    return mock


def make_pattern(core, **config_kwargs):
    config_kwargs.setdefault("judge_model", JUDGE_MODEL)
    with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", True):
        return LightRAGBranchingThoughts(
            lightrag_core=core, config=BranchingConfig(**config_kwargs)
        )


def judge_response(scores_by_mode):
    """Build an acompletion side effect that scores hypotheses by their mode."""

    async def completion(*args, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        evaluations = []
        for block in prompt.split("Hypothesis ")[1:]:
            if not block.startswith(tuple("0123456789")):
                continue
            hid = block.split("ID: ")[1].split(",")[0]
            mode = block.split("Mode: ")[1].split(")")[0]
            evaluations.append(
                {
                    "hypothesis_id": hid,
                    "score": scores_by_mode[mode],
                    "reasoning": f"{mode} judged",
                    "strengths": [],
                    "weaknesses": [],
                }
            )
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps({"evaluations": evaluations})
        return response

    return completion


@pytest.mark.asyncio
class TestModeSelection:
    @pytest.mark.parametrize(
        "mode,method",
        [("local", "local_query"), ("global", "global_query"), ("hybrid", "hybrid_query")],
    )
    async def test_single_mode_queries_only_that_mode(self, core, mode, method):
        pattern = make_pattern(core, hypothesis_count=2, modes=[mode])

        hypotheses = await pattern._generate_hypotheses("question", 2)

        assert [h.mode for h in hypotheses] == [mode, mode]
        assert getattr(core, method).await_count == 2
        other = {"local_query", "global_query", "hybrid_query"} - {method}
        for name in other:
            getattr(core, name).assert_not_awaited()

    async def test_modes_are_cycled_to_reach_the_count(self, core):
        pattern = make_pattern(core, modes=["local", "global"])

        hypotheses = await pattern._generate_hypotheses("question", 5)

        assert [h.mode for h in hypotheses] == [
            "local", "global", "local", "global", "local"
        ]

    async def test_count_below_mode_count_truncates(self, core):
        pattern = make_pattern(core)

        hypotheses = await pattern._generate_hypotheses("question", 2)

        assert [h.mode for h in hypotheses] == ["local", "global"]

    async def test_default_modes_cover_local_global_hybrid(self, core):
        pattern = make_pattern(core)

        hypotheses = await pattern._generate_hypotheses("question", 3)

        assert [h.mode for h in hypotheses] == ["local", "global", "hybrid"]

    async def test_dict_result_text_is_used_as_reasoning(self, core):
        pattern = make_pattern(core, modes=["hybrid"])

        hypotheses = await pattern._generate_hypotheses("question", 1)

        assert hypotheses[0].reasoning == "hybrid answer"


class TestConfigValidation:
    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid modes"):
            BranchingConfig(modes=["local", "bogus"])

    def test_empty_modes_are_rejected(self):
        with pytest.raises(ValueError, match="modes must not be empty"):
            BranchingConfig(modes=[])

    @pytest.mark.parametrize("count", [0, -1])
    def test_hypothesis_count_must_be_positive(self, count):
        with pytest.raises(ValueError, match="hypothesis_count must be at least 1"):
            BranchingConfig(hypothesis_count=count)

    def test_modes_default_to_none(self):
        assert BranchingConfig().modes is None


@pytest.mark.asyncio
class TestJudging:
    async def test_judge_scores_follow_the_llm_output(self, core):
        pattern = make_pattern(core)
        hypotheses = await pattern._generate_hypotheses("question", 3)

        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            llm.side_effect = judge_response(
                {"local": 0.9, "global": 0.7, "hybrid": 0.8}
            )
            scores = await pattern._judge_hypotheses("question", hypotheses)

        by_mode = {
            h.mode: s.score
            for h in hypotheses
            for s in scores
            if s.hypothesis_id == h.hypothesis_id
        }
        assert by_mode == {"local": 0.9, "global": 0.7, "hybrid": 0.8}
        assert llm.await_count == 1
        assert llm.await_args.kwargs["model"] == JUDGE_MODEL

    async def test_judge_failure_falls_back_to_default_confidence(self, core):
        pattern = make_pattern(core)
        hypotheses = await pattern._generate_hypotheses("question", 3)

        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            llm.side_effect = RuntimeError("judge unavailable")
            scores = await pattern._judge_hypotheses("question", hypotheses)

        assert [s.score for s in scores] == [h.confidence for h in hypotheses]
        assert all("unavailable" in s.reasoning for s in scores)

    async def test_best_scored_mode_is_selected(self, core):
        pattern = make_pattern(core)

        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            llm.side_effect = judge_response(
                {"local": 0.5, "global": 0.95, "hybrid": 0.6}
            )
            result = await pattern.execute(problem="question")

        assert result.success is True
        assert result.selected_hypothesis.mode == "global"
        assert result.result == "global answer"
        assert "0.95" in result.selection_reasoning


@pytest.mark.asyncio
class TestFailureHandling:
    async def test_no_hypotheses_skips_the_judge_and_warns(self, core):
        core.local_query.side_effect = core.global_query.side_effect = RuntimeError("x")
        core.hybrid_query.side_effect = RuntimeError("x")
        pattern = make_pattern(core)

        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            result = await pattern.execute(problem="question")

        llm.assert_not_awaited()
        assert result.hypotheses == []
        assert result.selected_hypothesis is None
        assert "every query mode failed" in result.metadata["warning"]

    async def test_one_failing_mode_keeps_the_others(self, core):
        core.global_query.side_effect = RuntimeError("global index offline")
        pattern = make_pattern(core)

        hypotheses = await pattern._generate_hypotheses("question", 3)

        assert [h.mode for h in hypotheses] == ["local", "hybrid"]


@pytest.mark.asyncio
class TestEvents:
    async def test_event_sequence_for_a_full_run(self, core, event_collector):
        pattern = make_pattern(core)
        pattern.add_event_handler(event_collector.collect)

        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            llm.side_effect = judge_response(
                {"local": 0.5, "global": 0.9, "hybrid": 0.6}
            )
            await pattern.execute(problem="question")

        types = [e["type"] for e in event_collector.get_events()]
        assert types[0] == "pattern.branching.started"
        assert types.index("pattern.branching.generating") < types.index(
            "pattern.branching.judging"
        )
        assert types.index("pattern.branching.judging") < types.index(
            "pattern.branching.selected"
        )
        assert types[-1] == "pattern.branching.completed"
        assert types.count("pattern.branching.hypothesis_generated") == 3

    async def test_generating_event_lists_the_modes(self, core, event_collector):
        pattern = make_pattern(core, modes=["local", "hybrid"])
        pattern.add_event_handler(event_collector.collect)

        await pattern._generate_hypotheses("question", 2)

        event = event_collector.get_events("pattern.branching.generating")[0]
        assert event["data"]["modes"] == ["local", "hybrid"]

    async def test_selected_event_reports_mode_and_score(self, core, event_collector):
        pattern = make_pattern(core)
        pattern.add_event_handler(event_collector.collect)
        hypothesis = Hypothesis("h1", "approach", "reasoning", 0.7, "global")
        from promptchain.integrations.lightrag.branching import HypothesisScore

        pattern._select_best_hypothesis(
            [hypothesis], [HypothesisScore("h1", 0.88, "good")]
        )

        data = event_collector.get_events("pattern.branching.selected")[0]["data"]
        assert data["mode"] == "global"
        assert data["score"] == 0.88

    async def test_no_events_when_disabled(self, core, event_collector):
        pattern = make_pattern(core, emit_events=False)
        pattern.add_event_handler(event_collector.collect)

        await pattern._generate_hypotheses("question", 3)

        assert event_collector.get_events() == []


@pytest.mark.asyncio
class TestPatternInterface:
    async def test_implements_base_pattern_contract(self, core):
        pattern = make_pattern(core)

        assert isinstance(pattern, BasePattern)
        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            llm.side_effect = judge_response({"local": 0.5, "global": 0.6, "hybrid": 0.7})
            result = await pattern.execute(problem="question")

        assert result.pattern_id == pattern.config.pattern_id
        assert result.to_dict()["selected_hypothesis"]["mode"] == "hybrid"

    async def test_hypothesis_count_override_beats_config(self, core):
        pattern = make_pattern(core, hypothesis_count=3)

        with patch.object(
            pattern._litellm, "acompletion", new_callable=AsyncMock
        ) as llm:
            llm.side_effect = judge_response({"local": 0.5, "global": 0.6, "hybrid": 0.7})
            result = await pattern.execute(problem="question", hypothesis_count=2)

        assert len(result.hypotheses) == 2
