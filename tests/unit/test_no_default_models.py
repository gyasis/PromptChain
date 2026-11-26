"""No library default model: a required model that is missing raises a clear
error naming the parameter; an optional model that is missing leaves its feature
off. Everything runs offline."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

from promptchain.integrations.lightrag.branching import (
    BranchingConfig, Hypothesis, LightRAGBranchingThoughts)
from promptchain.integrations.lightrag.core import (LightRAGConfig,
                                                    LightRAGIntegration)
from promptchain.integrations.lightrag.multi_hop import (LightRAGMultiHop,
                                                         MultiHopConfig)
from promptchain.cli.models.agent_config import Agent
from promptchain.cli.models.config import Config, ContextManagementConfig
from promptchain.cli.models.orchestration_config import RouterConfig
from promptchain.cli.models.session import Session
from promptchain.cli.session_manager import SessionManager
from promptchain.cli.utils.agent_templates import create_from_template
from promptchain.cli.utils.model_required import (ModelNotSelectedError,
                                                  require_model)
from promptchain.patterns.agent_comms import agent, orchestrator
from promptchain.utils.agentic_step_processor import AgenticStepProcessor
from promptchain.utils.chain_executor import (ChainExecutionError,
                                              ChainExecutor)
from promptchain.utils.chain_models import ChainDefinition, ChainStepDefinition
from promptchain.utils.history_summarizer import HistorySummarizer
from promptchain.utils.orchestrator_supervisor import OrchestratorSupervisor
from promptchain.utils.prompt_engineer import PromptEngineer

MODEL = "ollama_chat/glm-5.3-flash:cloud"


class TestRequiredModelRaises:
    def test_llm_agent(self):
        with pytest.raises(ValueError, match="requires `model`"):
            agent("A", "an engineer")

    def test_orchestrator(self):
        with pytest.raises(ValueError, match="requires `model`"):
            orchestrator("M")

    def test_history_summarizer(self):
        with pytest.raises(ValueError, match="requires `model`"):
            HistorySummarizer()

    def test_orchestrator_supervisor(self):
        with pytest.raises(ValueError, match="model_name"):
            OrchestratorSupervisor(agent_descriptions={"a": "b"})

    def test_prompt_engineer(self):
        with pytest.raises(ValueError, match="evaluator_model"):
            PromptEngineer()

    def test_lightrag_config_validated_at_use(self):
        cfg = LightRAGConfig()
        assert cfg.model_name is None and cfg.embedding_model is None
        with patch("promptchain.integrations.lightrag.core.LIGHTRAG_AVAILABLE", True):
            integration = LightRAGIntegration(cfg)
        with pytest.raises(ValueError, match="model_name"):
            integration._ensure_initialized()
        integration.config.model_name = MODEL
        with pytest.raises(ValueError, match="embedding_model"):
            integration._ensure_initialized()

    def test_router_config_model_unset_is_allowed_until_used(self):
        assert RouterConfig().model is None

    def test_template_needs_model(self):
        with pytest.raises(ModelNotSelectedError, match="Pick one"):
            create_from_template("researcher", "r")
        assert create_from_template("researcher", "r", model_override=MODEL).model_name == MODEL

    def test_chain_executor_without_any_model(self):
        step = ChainStepDefinition(id="s1", type="prompt", content="hi")
        chain = ChainDefinition(vin="v", model="m", version="v1", steps=[step])
        assert chain.llm_model is None
        executor = ChainExecutor()
        with pytest.raises(ChainExecutionError, match="No LLM model"):
            executor._resolve_model(chain, step)
        assert ChainExecutor(default_model=MODEL)._resolve_model(chain, step) == MODEL

    def test_agentchain_synthesizer_needs_model(self):
        from promptchain.utils.agent_chain import AgentChain
        from promptchain.utils.promptchaining import PromptChain

        chains = {"a": PromptChain(models=[MODEL], instructions=["{input}"])}
        with pytest.raises(ValueError, match="synthesizer_config requires a 'model'"):
            AgentChain(agents=chains, agent_descriptions={"a": "x"},
                       execution_mode="broadcast",
                       synthesizer_config={"prompt": "p {agent_responses}"})


class TestOptionalModelIsOff:
    def test_branching_judge_off_uses_confidence(self):
        assert BranchingConfig().judge_model is None
        with patch("promptchain.integrations.lightrag.branching.LIGHTRAG_AVAILABLE", True):
            pattern = LightRAGBranchingThoughts(MagicMock(), BranchingConfig())
        hyp = Hypothesis(hypothesis_id="h1", approach="a", reasoning="r",
                         confidence=0.4, mode="local")
        scores = asyncio.run(pattern._judge_hypotheses("p", [hyp]))
        assert scores[0].score == 0.4

    def test_multi_hop_synthesizer_off(self):
        assert MultiHopConfig().synthesizer_model is None
        with patch("promptchain.integrations.lightrag.multi_hop.HYBRIDRAG_AVAILABLE", True):
            pattern = LightRAGMultiHop(MagicMock(), MultiHopConfig())
        assert asyncio.run(pattern._decompose_question("q")) == []
        answer = asyncio.run(pattern._synthesize_answer(
            "q", [], {"sq1": "one", "sq2": "two"}))
        assert answer == "one\n\ntwo"

    def test_history_summarization_off_without_model(self):
        asp = AgenticStepProcessor(objective="o", max_internal_steps=2,
                                   model_name=MODEL, history_mode="progressive")
        assert asp.summarizer_model is None
        assert asp._should_summarize(iteration=4, current_tokens=10**9) is False
        asp2 = AgenticStepProcessor(objective="o", max_internal_steps=2,
                                    model_name=MODEL, history_mode="progressive",
                                    summarizer_model=MODEL)
        assert asp2._should_summarize(iteration=4, current_tokens=0) is True

    def test_context_management_summarizer_unset(self):
        assert ContextManagementConfig().summarizer_model is None


class TestCliHoldsNoModel:
    def test_config_default_model_unset(self):
        cfg = Config()
        assert cfg.default_model is None
        cfg.validate()

    def test_session_without_model_round_trips(self, tmp_path):
        manager = SessionManager(sessions_dir=tmp_path)
        session = manager.create_session("nomodel", working_directory=tmp_path)
        assert session.default_model is None
        assert session.agents["default"].model_name is None
        loaded = manager.load_session("nomodel")
        assert loaded.default_model is None
        assert loaded.agents["default"].model_name is None

    def test_session_keeps_stored_model(self, tmp_path):
        manager = SessionManager(sessions_dir=tmp_path)
        manager.create_session("withmodel", working_directory=tmp_path,
                               default_model=MODEL)
        assert manager.load_session("withmodel").default_model == MODEL

    def test_require_model_message_tells_user_what_to_do(self):
        with pytest.raises(ModelNotSelectedError, match="/model"):
            require_model(None, "", context="Router")
        assert require_model(None, MODEL) == MODEL

    def test_agent_rejects_bad_format_only_when_set(self):
        assert Agent(name="a", model_name=None).model_name is None
        assert Agent(name="a", model_name="").model_name is None
        with pytest.raises(ValueError):
            Agent(name="a", model_name="nomodelslash")
