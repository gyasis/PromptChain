"""Unit tests for LightRAGMultiHop.

The pattern optionally decomposes a question into sub-questions (one LLM
call), hands the question to ``SearchInterface.agentic_search`` and
synthesises the returned answer. ``agentic_search`` is a single opaque call:
it returns ``answer`` and ``steps_taken`` and nothing per step. The search
interface is an ``AsyncMock`` and ``litellm.acompletion`` is patched, so no
network call happens.

Already covered by tests/test_multi_hop_pattern.py (not repeated here):
initialisation, decomposition parsing, synthesis, unanswered aspects,
decompose-first on/off, the error path, the Blackboard hand-off.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptchain.integrations.lightrag.multi_hop import (
    LightRAGMultiHop,
    MultiHopConfig,
    MultiHopResult,
)
from promptchain.patterns.base import BasePattern

SYNTH_MODEL = "test-provider/synth-model"


@pytest.fixture
def search():
    mock = AsyncMock()
    mock.agentic_search.return_value = {"answer": "the answer", "steps_taken": 3}
    return mock


def make_pattern(search, **config_kwargs):
    config_kwargs.setdefault("decompose_first", False)
    with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
        return LightRAGMultiHop(search, MultiHopConfig(**config_kwargs))


def llm_returning(payload):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(payload)
    return AsyncMock(return_value=response)


@pytest.mark.asyncio
class TestAgenticSearchDelegation:
    async def test_question_and_hop_budget_reach_agentic_search(self, search):
        pattern = make_pattern(search, max_hops=4)

        result = await pattern.execute(question="Who built X?")

        search.agentic_search.assert_awaited_once()
        kwargs = search.agentic_search.await_args.kwargs
        assert kwargs["query"] == "Who built X?"
        assert kwargs["max_steps"] == 4
        assert "Who built X?" in kwargs["objective"]
        assert result.unified_answer == "the answer"

    async def test_hop_count_comes_from_the_search_result(self, search):
        pattern = make_pattern(search, max_hops=5)

        result = await pattern.execute(question="q")

        assert result.hops_executed == 3
        assert result.metadata["max_hops_configured"] == 5

    async def test_execute_kwarg_overrides_configured_max_hops(self, search):
        pattern = make_pattern(search, max_hops=5)

        result = await pattern.execute(question="q", max_hops=2)

        assert search.agentic_search.await_args.kwargs["max_steps"] == 2
        assert result.metadata["max_hops_configured"] == 2

    async def test_sub_questions_are_listed_in_the_objective(self, search):
        pattern = make_pattern(search, decompose_first=True, synthesizer_model=SYNTH_MODEL)
        payload = {
            "sub_questions": [
                {"question_id": "sq1", "question_text": "Who is the inventor?"},
                {"question_id": "sq2", "question_text": "Where was the inventor born?"},
            ]
        }

        with patch("litellm.acompletion", new=llm_returning(payload)) as llm:
            result = await pattern.execute(question="Where was the inventor born?")

        assert llm.await_args.kwargs["model"] == SYNTH_MODEL
        objective = search.agentic_search.await_args.kwargs["objective"]
        assert "- Who is the inventor?" in objective
        assert [sq.question_id for sq in result.sub_questions] == ["sq1", "sq2"]
        assert all(sq.answer == "the answer" for sq in result.sub_questions)

    async def test_no_decomposition_means_no_llm_call(self, search):
        pattern = make_pattern(search, decompose_first=False)

        with patch("litellm.acompletion", new=AsyncMock()) as llm:
            result = await pattern.execute(question="q")

        llm.assert_not_awaited()
        assert result.sub_questions == []
        assert result.sub_answers == {"main": "the answer"}

    async def test_max_sub_questions_caps_decomposition(self, search):
        pattern = make_pattern(search, decompose_first=True, max_sub_questions=1)
        payload = {
            "sub_questions": [
                {"question_id": f"sq{i}", "question_text": f"Q{i}"} for i in range(4)
            ]
        }

        with patch("litellm.acompletion", new=llm_returning(payload)):
            result = await pattern.execute(question="q")

        assert len(result.sub_questions) == 1


class TestConfigValidation:
    @pytest.mark.parametrize("hops", [0, -3])
    def test_max_hops_must_be_positive(self, hops):
        with pytest.raises(ValueError, match="max_hops must be at least 1"):
            MultiHopConfig(max_hops=hops)

    def test_negative_sub_question_cap_is_rejected(self):
        with pytest.raises(ValueError, match="max_sub_questions must not be negative"):
            MultiHopConfig(max_sub_questions=-1)

    def test_zero_sub_question_cap_is_allowed(self):
        assert MultiHopConfig(max_sub_questions=0).max_sub_questions == 0

    def test_defaults_stay_valid(self):
        config = MultiHopConfig()
        assert (config.max_hops, config.max_sub_questions) == (5, 5)


@pytest.mark.asyncio
class TestEvents:
    async def test_event_order_for_a_run_without_decomposition(
        self, search, event_collector
    ):
        pattern = make_pattern(search)
        pattern.add_event_handler(event_collector.collect)

        await pattern.execute(question="q")

        assert [e["type"] for e in event_collector.get_events()] == [
            "pattern.multi_hop.started",
            "pattern.multi_hop.hop_completed",
            "pattern.multi_hop.synthesizing",
            "pattern.multi_hop.completed",
        ]

    async def test_hop_completed_event_reports_hops(self, search, event_collector):
        pattern = make_pattern(search)
        pattern.add_event_handler(event_collector.collect)

        await pattern.execute(question="q")

        data = event_collector.get_events("pattern.multi_hop.hop_completed")[0]["data"]
        assert data["hops_executed"] == 3
        assert data["num_sub_answers"] == 1

    async def test_decomposed_event_lists_sub_questions(self, search, event_collector):
        pattern = make_pattern(search, decompose_first=True)
        pattern.add_event_handler(event_collector.collect)
        payload = [{"question_id": "sq1", "question_text": "Q1"}]

        with patch("litellm.acompletion", new=llm_returning(payload)):
            await pattern.execute(question="q")

        data = event_collector.get_events("pattern.multi_hop.decomposed")[0]["data"]
        assert data["sub_questions"] == ["Q1"]

    async def test_completed_event_counts_unanswered_aspects(
        self, search, event_collector
    ):
        search.agentic_search.return_value = {"answer": "", "steps_taken": 1}
        pattern = make_pattern(search, decompose_first=True)
        pattern.add_event_handler(event_collector.collect)
        payload = [{"question_id": "sq1", "question_text": "Q1"}]

        with patch("litellm.acompletion", new=llm_returning(payload)):
            result = await pattern.execute(question="q")

        completed = event_collector.get_events("pattern.multi_hop.completed")[0]
        assert completed["data"]["num_unanswered"] == 1
        assert result.unanswered_aspects == ["Q1"]


@pytest.mark.asyncio
class TestPatternInterface:
    async def test_implements_base_pattern_contract(self, search):
        pattern = make_pattern(search)

        assert isinstance(pattern, BasePattern)
        result = await pattern.execute(question="q")
        assert isinstance(result, MultiHopResult)
        assert result.success is True
        assert result.original_question == "q"
        assert result.result == result.unified_answer

    async def test_execute_with_timeout_wraps_the_result(self, search):
        pattern = make_pattern(search)

        result = await pattern.execute_with_timeout(question="q")

        assert result.success is True
        assert result.unified_answer == "the answer"
