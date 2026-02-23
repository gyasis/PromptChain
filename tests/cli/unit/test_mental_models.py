"""
Unit tests for Mental Models system (US7 T076-T120).

Tests cover:
- SpecializationType enum
- AgentSpecialization dataclass
- MentalModel dataclass
- MentalModelManager
- create_default_model factory function
"""

import pytest
from datetime import datetime
from typing import List

from promptchain.cli.models.mental_models import (
    SpecializationType,
    AgentSpecialization,
    MentalModel,
    MentalModelManager,
    create_default_model,
    DEFAULT_SPECIALIZATION_CAPABILITIES
)


# ============================================================================
# SpecializationType Tests
# ============================================================================

class TestSpecializationType:
    """Tests for SpecializationType enum."""

    def test_all_enum_values_exist(self):
        """Test that all expected specialization types are defined."""
        expected_types = [
            "CODE_ANALYSIS",
            "CODE_GENERATION",
            "FILE_OPERATIONS",
            "SEARCH_DISCOVERY",
            "TESTING",
            "DOCUMENTATION",
            "DEBUGGING",
            "ARCHITECTURE",
            "DATA_PROCESSING",
            "API_INTEGRATION",
            "SECURITY",
            "PERFORMANCE",
            "UI_UX",
            "DEVOPS",
            "GENERAL"
        ]

        for type_name in expected_types:
            assert hasattr(SpecializationType, type_name)

    def test_string_conversion(self):
        """Test that enum values convert to expected strings."""
        assert SpecializationType.CODE_ANALYSIS.value == "code_analysis"
        assert SpecializationType.TESTING.value == "testing"
        assert SpecializationType.GENERAL.value == "general"

    def test_enum_construction_from_string(self):
        """Test creating enum from string value."""
        spec = SpecializationType("code_analysis")
        assert spec == SpecializationType.CODE_ANALYSIS

        spec = SpecializationType("testing")
        assert spec == SpecializationType.TESTING


# ============================================================================
# AgentSpecialization Tests
# ============================================================================

class TestAgentSpecialization:
    """Tests for AgentSpecialization dataclass."""

    def test_creation_with_defaults(self):
        """Test creating specialization with default values."""
        spec = AgentSpecialization(specialization=SpecializationType.CODE_ANALYSIS)

        assert spec.specialization == SpecializationType.CODE_ANALYSIS
        assert spec.proficiency == 0.5
        assert spec.related_capabilities == []
        assert spec.experience_count == 0

    def test_creation_with_custom_values(self):
        """Test creating specialization with custom values."""
        spec = AgentSpecialization(
            specialization=SpecializationType.TESTING,
            proficiency=0.8,
            related_capabilities=["test_runner", "terminal_execute"],
            experience_count=10
        )

        assert spec.specialization == SpecializationType.TESTING
        assert spec.proficiency == 0.8
        assert spec.related_capabilities == ["test_runner", "terminal_execute"]
        assert spec.experience_count == 10

    @pytest.mark.parametrize("invalid_proficiency", [-0.1, 1.1, 2.0, -1.0])
    def test_proficiency_bounds_validation(self, invalid_proficiency):
        """Test that proficiency must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="Proficiency must be 0.0-1.0"):
            AgentSpecialization(
                specialization=SpecializationType.GENERAL,
                proficiency=invalid_proficiency
            )

    @pytest.mark.parametrize("valid_proficiency", [0.0, 0.5, 1.0, 0.25, 0.99])
    def test_proficiency_bounds_valid(self, valid_proficiency):
        """Test that valid proficiency values are accepted."""
        spec = AgentSpecialization(
            specialization=SpecializationType.GENERAL,
            proficiency=valid_proficiency
        )
        assert spec.proficiency == valid_proficiency

    def test_record_experience_success(self):
        """Test recording successful task completion increases proficiency."""
        spec = AgentSpecialization(
            specialization=SpecializationType.CODE_ANALYSIS,
            proficiency=0.5
        )

        initial_proficiency = spec.proficiency
        spec.record_experience(success=True)

        assert spec.experience_count == 1
        assert spec.proficiency > initial_proficiency
        assert spec.proficiency <= 1.0

    def test_record_experience_failure(self):
        """Test recording failed task completion decreases proficiency."""
        spec = AgentSpecialization(
            specialization=SpecializationType.CODE_ANALYSIS,
            proficiency=0.5
        )

        initial_proficiency = spec.proficiency
        spec.record_experience(success=False)

        assert spec.experience_count == 1
        assert spec.proficiency < initial_proficiency
        assert spec.proficiency >= 0.0

    def test_record_experience_diminishing_returns(self):
        """Test that proficiency gains diminish as it approaches 1.0."""
        spec = AgentSpecialization(
            specialization=SpecializationType.GENERAL,
            proficiency=0.5
        )

        # First success
        spec.record_experience(success=True)
        first_gain = spec.proficiency - 0.5

        # Second success at higher proficiency
        current_prof = spec.proficiency
        spec.record_experience(success=True)
        second_gain = spec.proficiency - current_prof

        # Second gain should be smaller
        assert second_gain < first_gain

    def test_record_experience_caps_at_one(self):
        """Test that proficiency never exceeds 1.0."""
        spec = AgentSpecialization(
            specialization=SpecializationType.GENERAL,
            proficiency=0.95
        )

        # Many successes
        for _ in range(100):
            spec.record_experience(success=True)

        # Use approximate comparison for floating point
        assert spec.proficiency >= 0.9999
        assert spec.proficiency <= 1.0
        assert spec.experience_count == 100

    def test_record_experience_floors_at_zero(self):
        """Test that proficiency never goes below 0.0."""
        spec = AgentSpecialization(
            specialization=SpecializationType.GENERAL,
            proficiency=0.05
        )

        # Many failures
        for _ in range(10):
            spec.record_experience(success=False)

        assert spec.proficiency == 0.0
        assert spec.experience_count == 10

    def test_to_dict_serialization(self):
        """Test converting specialization to dictionary."""
        spec = AgentSpecialization(
            specialization=SpecializationType.TESTING,
            proficiency=0.75,
            related_capabilities=["test_runner", "terminal_execute"],
            experience_count=5
        )

        result = spec.to_dict()

        assert result == {
            "specialization": "testing",
            "proficiency": 0.75,
            "related_capabilities": ["test_runner", "terminal_execute"],
            "experience_count": 5
        }

    def test_from_dict_deserialization(self):
        """Test creating specialization from dictionary."""
        data = {
            "specialization": "code_generation",
            "proficiency": 0.9,
            "related_capabilities": ["file_write", "code_generation"],
            "experience_count": 20
        }

        spec = AgentSpecialization.from_dict(data)

        assert spec.specialization == SpecializationType.CODE_GENERATION
        assert spec.proficiency == 0.9
        assert spec.related_capabilities == ["file_write", "code_generation"]
        assert spec.experience_count == 20

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {
            "specialization": "debugging"
        }

        spec = AgentSpecialization.from_dict(data)

        assert spec.specialization == SpecializationType.DEBUGGING
        assert spec.proficiency == 0.5
        assert spec.related_capabilities == []
        assert spec.experience_count == 0

    def test_round_trip_serialization(self):
        """Test that to_dict/from_dict is reversible."""
        original = AgentSpecialization(
            specialization=SpecializationType.ARCHITECTURE,
            proficiency=0.65,
            related_capabilities=["code_analysis", "file_read"],
            experience_count=15
        )

        data = original.to_dict()
        restored = AgentSpecialization.from_dict(data)

        assert restored.specialization == original.specialization
        assert restored.proficiency == original.proficiency
        assert restored.related_capabilities == original.related_capabilities
        assert restored.experience_count == original.experience_count


# ============================================================================
# MentalModel Tests
# ============================================================================

class TestMentalModel:
    """Tests for MentalModel dataclass."""

    def test_creation_minimal(self):
        """Test creating mental model with just agent name."""
        model = MentalModel(agent_name="test_agent")

        assert model.agent_name == "test_agent"
        assert model.specializations == []
        assert model.known_agents == {}
        assert model.task_history == []
        assert isinstance(model.created_at, float)
        assert isinstance(model.updated_at, float)

    def test_creation_with_timestamp(self):
        """Test that created_at and updated_at are set."""
        before = datetime.now().timestamp()
        model = MentalModel(agent_name="test")
        after = datetime.now().timestamp()

        assert before <= model.created_at <= after
        assert before <= model.updated_at <= after

    def test_add_specialization_basic(self):
        """Test adding a specialization."""
        model = MentalModel(agent_name="test")

        spec = model.add_specialization(SpecializationType.CODE_ANALYSIS)

        assert len(model.specializations) == 1
        assert spec in model.specializations
        assert spec.specialization == SpecializationType.CODE_ANALYSIS
        assert spec.proficiency == 0.5

    def test_add_specialization_with_params(self):
        """Test adding specialization with custom parameters."""
        model = MentalModel(agent_name="test")

        spec = model.add_specialization(
            spec_type=SpecializationType.TESTING,
            proficiency=0.8,
            capabilities=["test_runner", "terminal_execute"]
        )

        assert spec.proficiency == 0.8
        assert spec.related_capabilities == ["test_runner", "terminal_execute"]

    def test_add_specialization_updates_timestamp(self):
        """Test that adding specialization updates updated_at."""
        model = MentalModel(agent_name="test")
        initial_updated = model.updated_at

        import time
        time.sleep(0.01)  # Ensure timestamp difference

        model.add_specialization(SpecializationType.GENERAL)

        assert model.updated_at > initial_updated

    def test_add_multiple_specializations(self):
        """Test adding multiple specializations."""
        model = MentalModel(agent_name="test")

        model.add_specialization(SpecializationType.CODE_ANALYSIS)
        model.add_specialization(SpecializationType.TESTING)
        model.add_specialization(SpecializationType.DEBUGGING)

        assert len(model.specializations) == 3

    def test_get_specialization_exists(self):
        """Test getting an existing specialization."""
        model = MentalModel(agent_name="test")
        added_spec = model.add_specialization(SpecializationType.TESTING, proficiency=0.75)

        found_spec = model.get_specialization(SpecializationType.TESTING)

        assert found_spec is not None
        assert found_spec.specialization == SpecializationType.TESTING
        assert found_spec.proficiency == 0.75

    def test_get_specialization_not_exists(self):
        """Test getting a non-existent specialization returns None."""
        model = MentalModel(agent_name="test")
        model.add_specialization(SpecializationType.TESTING)

        result = model.get_specialization(SpecializationType.CODE_ANALYSIS)

        assert result is None

    def test_learn_about_agent(self):
        """Test learning about another agent's capabilities."""
        model = MentalModel(agent_name="agent1")

        model.learn_about_agent("agent2", ["file_read", "file_write"])

        assert "agent2" in model.known_agents
        assert model.known_agents["agent2"] == ["file_read", "file_write"]

    def test_learn_about_multiple_agents(self):
        """Test learning about multiple agents."""
        model = MentalModel(agent_name="agent1")

        model.learn_about_agent("agent2", ["file_read"])
        model.learn_about_agent("agent3", ["test_runner"])

        assert len(model.known_agents) == 2
        assert model.known_agents["agent2"] == ["file_read"]
        assert model.known_agents["agent3"] == ["test_runner"]

    def test_learn_about_agent_overwrites(self):
        """Test that learning about same agent overwrites previous data."""
        model = MentalModel(agent_name="agent1")

        model.learn_about_agent("agent2", ["file_read"])
        model.learn_about_agent("agent2", ["file_write"])

        assert model.known_agents["agent2"] == ["file_write"]

    def test_forget_agent(self):
        """Test forgetting an agent (manual removal from known_agents)."""
        model = MentalModel(agent_name="agent1")
        model.learn_about_agent("agent2", ["file_read"])

        # Manual forget operation
        del model.known_agents["agent2"]

        assert "agent2" not in model.known_agents

    def test_record_task_outcome(self):
        """Test recording a task outcome."""
        model = MentalModel(agent_name="test")

        model.record_task_outcome(
            task_type="code_analysis",
            success=True,
            capabilities_used=["file_read", "code_search"]
        )

        assert len(model.task_history) == 1
        outcome = model.task_history[0]
        assert outcome["task_type"] == "code_analysis"
        assert outcome["success"] is True
        assert outcome["capabilities_used"] == ["file_read", "code_search"]
        assert "timestamp" in outcome

    def test_record_task_outcome_multiple(self):
        """Test recording multiple task outcomes."""
        model = MentalModel(agent_name="test")

        model.record_task_outcome("task1", True, ["cap1"])
        model.record_task_outcome("task2", False, ["cap2"])
        model.record_task_outcome("task3", True, ["cap3"])

        assert len(model.task_history) == 3

    def test_record_task_outcome_limits_history(self):
        """Test that task history is limited to 100 entries."""
        model = MentalModel(agent_name="test")

        # Add 150 tasks
        for i in range(150):
            model.record_task_outcome(f"task{i}", True, ["cap"])

        assert len(model.task_history) == 100
        # Should keep the last 100
        assert model.task_history[0]["task_type"] == "task50"
        assert model.task_history[-1]["task_type"] == "task149"

    def test_calculate_task_fitness_no_requirements(self):
        """Test fitness calculation with no required capabilities."""
        model = MentalModel(agent_name="test")
        model.add_specialization(SpecializationType.GENERAL, capabilities=["file_read"])

        fitness = model.calculate_task_fitness([])

        assert fitness == 0.5  # Default for unspecified

    def test_calculate_task_fitness_perfect_match(self):
        """Test fitness calculation with perfect capability match."""
        model = MentalModel(agent_name="test")
        model.add_specialization(
            SpecializationType.CODE_ANALYSIS,
            capabilities=["file_read", "code_search", "ripgrep_search"]
        )

        required = ["file_read", "code_search"]
        fitness = model.calculate_task_fitness(required)

        assert fitness == 1.0  # 2/2 match

    def test_calculate_task_fitness_partial_match(self):
        """Test fitness calculation with partial capability match."""
        model = MentalModel(agent_name="test")
        model.add_specialization(
            SpecializationType.CODE_ANALYSIS,
            capabilities=["file_read", "code_search"]
        )

        required = ["file_read", "code_search", "file_write", "terminal_execute"]
        fitness = model.calculate_task_fitness(required)

        assert fitness == 0.5  # 2/4 match

    def test_calculate_task_fitness_no_match(self):
        """Test fitness calculation with no capability match."""
        model = MentalModel(agent_name="test")
        model.add_specialization(
            SpecializationType.CODE_ANALYSIS,
            capabilities=["file_read", "code_search"]
        )

        required = ["database_query", "api_call"]
        fitness = model.calculate_task_fitness(required)

        assert fitness == 0.0  # 0/2 match

    def test_calculate_task_fitness_multiple_specializations(self):
        """Test fitness calculation aggregates across all specializations."""
        model = MentalModel(agent_name="test")
        model.add_specialization(
            SpecializationType.CODE_ANALYSIS,
            capabilities=["file_read", "code_search"]
        )
        model.add_specialization(
            SpecializationType.TESTING,
            capabilities=["test_runner", "terminal_execute"]
        )

        required = ["file_read", "test_runner"]
        fitness = model.calculate_task_fitness(required)

        assert fitness == 1.0  # 2/2 match across both specs

    def test_suggest_agent_for_task_good_match(self):
        """Test suggesting agent with good capability match."""
        model = MentalModel(agent_name="agent1")
        model.learn_about_agent("agent2", ["file_read", "code_search", "file_write"])

        suggestion = model.suggest_agent_for_task(["file_read", "code_search"])

        assert suggestion == "agent2"

    def test_suggest_agent_for_task_below_threshold(self):
        """Test that low matches return None."""
        model = MentalModel(agent_name="agent1")
        model.learn_about_agent("agent2", ["file_read"])

        # Only 1/5 match = 0.2, below 0.3 threshold
        suggestion = model.suggest_agent_for_task([
            "file_read", "code_search", "file_write", "terminal_execute", "test_runner"
        ])

        assert suggestion is None

    def test_suggest_agent_for_task_best_of_multiple(self):
        """Test suggesting best agent when multiple known."""
        model = MentalModel(agent_name="agent1")
        model.learn_about_agent("agent2", ["file_read"])
        model.learn_about_agent("agent3", ["file_read", "code_search", "file_write"])

        suggestion = model.suggest_agent_for_task(["file_read", "code_search"])

        assert suggestion == "agent3"  # Better match

    def test_suggest_agent_for_task_no_known_agents(self):
        """Test suggesting agent when none are known."""
        model = MentalModel(agent_name="agent1")

        suggestion = model.suggest_agent_for_task(["file_read"])

        assert suggestion is None

    def test_to_dict_serialization(self):
        """Test converting mental model to dictionary."""
        model = MentalModel(agent_name="test_agent")
        model.add_specialization(SpecializationType.TESTING, proficiency=0.8)
        model.learn_about_agent("other_agent", ["file_read"])
        model.record_task_outcome("task1", True, ["cap1"])

        result = model.to_dict()

        assert result["agent_name"] == "test_agent"
        assert len(result["specializations"]) == 1
        assert result["specializations"][0]["specialization"] == "testing"
        assert "other_agent" in result["known_agents"]
        assert len(result["task_history"]) == 1
        assert "created_at" in result
        assert "updated_at" in result

    def test_from_dict_deserialization(self):
        """Test creating mental model from dictionary."""
        data = {
            "agent_name": "test_agent",
            "specializations": [
                {
                    "specialization": "code_analysis",
                    "proficiency": 0.7,
                    "related_capabilities": ["file_read"],
                    "experience_count": 5
                }
            ],
            "known_agents": {"agent2": ["file_write"]},
            "task_history": [{"task_type": "task1", "success": True}],
            "created_at": 1000.0,
            "updated_at": 2000.0
        }

        model = MentalModel.from_dict(data)

        assert model.agent_name == "test_agent"
        assert len(model.specializations) == 1
        assert model.specializations[0].specialization == SpecializationType.CODE_ANALYSIS
        assert "agent2" in model.known_agents
        assert len(model.task_history) == 1
        assert model.created_at == 1000.0
        assert model.updated_at == 2000.0

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {
            "agent_name": "minimal_agent"
        }

        model = MentalModel.from_dict(data)

        assert model.agent_name == "minimal_agent"
        assert model.specializations == []
        assert model.known_agents == {}
        assert model.task_history == []
        assert isinstance(model.created_at, float)
        assert isinstance(model.updated_at, float)

    def test_round_trip_serialization(self):
        """Test that to_dict/from_dict is reversible."""
        original = MentalModel(agent_name="agent1")
        original.add_specialization(SpecializationType.TESTING, proficiency=0.9)
        original.add_specialization(SpecializationType.DEBUGGING, proficiency=0.6)
        original.learn_about_agent("agent2", ["file_read", "file_write"])
        original.record_task_outcome("task1", True, ["cap1", "cap2"])

        data = original.to_dict()
        restored = MentalModel.from_dict(data)

        assert restored.agent_name == original.agent_name
        assert len(restored.specializations) == len(original.specializations)
        assert restored.known_agents == original.known_agents
        assert len(restored.task_history) == len(original.task_history)

    def test_repr_format(self):
        """Test string representation of mental model."""
        model = MentalModel(agent_name="test_agent")
        model.add_specialization(SpecializationType.CODE_ANALYSIS)
        model.add_specialization(SpecializationType.TESTING)
        model.learn_about_agent("agent2", ["file_read"])

        repr_str = repr(model)

        assert "test_agent" in repr_str
        assert "code_analysis" in repr_str
        assert "testing" in repr_str
        assert "known_agents=1" in repr_str


# ============================================================================
# MentalModelManager Tests
# ============================================================================

class TestMentalModelManager:
    """Tests for MentalModelManager."""

    def test_initialization(self):
        """Test manager initializes with empty models."""
        manager = MentalModelManager()

        assert manager._models == {}

    def test_get_or_create_new(self):
        """Test getting or creating a new model."""
        manager = MentalModelManager()

        model = manager.get_or_create("agent1")

        assert model.agent_name == "agent1"
        assert "agent1" in manager._models

    def test_get_or_create_existing(self):
        """Test getting existing model doesn't create new one."""
        manager = MentalModelManager()

        model1 = manager.get_or_create("agent1")
        model1.add_specialization(SpecializationType.TESTING)

        model2 = manager.get_or_create("agent1")

        assert model1 is model2
        assert len(model2.specializations) == 1

    def test_get_exists(self):
        """Test getting an existing model."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")

        model = manager.get("agent1")

        assert model is not None
        assert model.agent_name == "agent1"

    def test_get_not_exists(self):
        """Test getting non-existent model returns None."""
        manager = MentalModelManager()

        result = manager.get("nonexistent")

        assert result is None

    def test_update_new_model(self):
        """Test updating with a new model."""
        manager = MentalModelManager()
        model = MentalModel(agent_name="agent1")
        model.add_specialization(SpecializationType.TESTING)

        manager.update(model)

        retrieved = manager.get("agent1")
        assert retrieved is not None
        assert len(retrieved.specializations) == 1

    def test_update_existing_model(self):
        """Test updating an existing model."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")

        updated_model = MentalModel(agent_name="agent1")
        updated_model.add_specialization(SpecializationType.CODE_GENERATION)

        manager.update(updated_model)

        retrieved = manager.get("agent1")
        assert len(retrieved.specializations) == 1
        assert retrieved.specializations[0].specialization == SpecializationType.CODE_GENERATION

    def test_delete_model(self):
        """Test deleting a model (manual operation)."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")

        # Manual delete
        del manager._models["agent1"]

        assert manager.get("agent1") is None

    def test_list_agents_empty(self):
        """Test listing agents when none exist."""
        manager = MentalModelManager()

        agents = manager.list_agents()

        assert agents == []

    def test_list_agents_multiple(self):
        """Test listing multiple agents."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")
        manager.get_or_create("agent2")
        manager.get_or_create("agent3")

        agents = manager.list_agents()

        assert len(agents) == 3
        assert set(agents) == {"agent1", "agent2", "agent3"}

    def test_find_best_agent_single(self):
        """Test finding best agent with single candidate."""
        manager = MentalModelManager()
        model = manager.get_or_create("agent1")
        model.add_specialization(
            SpecializationType.CODE_ANALYSIS,
            capabilities=["file_read", "code_search"]
        )

        best = manager.find_best_agent(["file_read", "code_search"])

        assert best == "agent1"

    def test_find_best_agent_multiple_candidates(self):
        """Test finding best agent among multiple candidates."""
        manager = MentalModelManager()

        # Agent with partial match
        model1 = manager.get_or_create("agent1")
        model1.add_specialization(
            SpecializationType.GENERAL,
            capabilities=["file_read"]
        )

        # Agent with better match
        model2 = manager.get_or_create("agent2")
        model2.add_specialization(
            SpecializationType.CODE_ANALYSIS,
            capabilities=["file_read", "code_search", "ripgrep_search"]
        )

        best = manager.find_best_agent(["file_read", "code_search"])

        assert best == "agent2"

    def test_find_best_agent_below_threshold(self):
        """Test that low matches return None."""
        manager = MentalModelManager()
        model = manager.get_or_create("agent1")
        model.add_specialization(
            SpecializationType.GENERAL,
            capabilities=["file_read"]
        )

        # Only 1/5 = 0.2, below 0.3 threshold
        best = manager.find_best_agent([
            "file_read", "code_search", "file_write", "terminal_execute", "test_runner"
        ])

        assert best is None

    def test_find_best_agent_no_agents(self):
        """Test finding best agent when none exist."""
        manager = MentalModelManager()

        best = manager.find_best_agent(["file_read"])

        assert best is None

    def test_broadcast_agent_discovery(self):
        """Test broadcasting agent capabilities to all others."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")
        manager.get_or_create("agent2")
        manager.get_or_create("agent3")

        manager.broadcast_agent_discovery("new_agent", ["file_read", "file_write"])

        # All existing agents should know about new_agent
        assert "new_agent" in manager.get("agent1").known_agents
        assert "new_agent" in manager.get("agent2").known_agents
        assert "new_agent" in manager.get("agent3").known_agents

        # new_agent itself shouldn't be in the list
        assert manager.get("new_agent") is None

    def test_broadcast_agent_discovery_excludes_self(self):
        """Test that broadcast doesn't tell agent about itself."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")
        manager.get_or_create("agent2")

        manager.broadcast_agent_discovery("agent1", ["file_read"])

        # agent1 should not know about itself
        assert "agent1" not in manager.get("agent1").known_agents
        # But agent2 should know about agent1
        assert "agent1" in manager.get("agent2").known_agents

    def test_export_all_empty(self):
        """Test exporting when no models exist."""
        manager = MentalModelManager()

        exported = manager.export_all()

        assert exported == {}

    def test_export_all_multiple(self):
        """Test exporting multiple models."""
        manager = MentalModelManager()
        manager.get_or_create("agent1").add_specialization(SpecializationType.TESTING)
        manager.get_or_create("agent2").add_specialization(SpecializationType.CODE_ANALYSIS)

        exported = manager.export_all()

        assert len(exported) == 2
        assert "agent1" in exported
        assert "agent2" in exported
        assert exported["agent1"]["agent_name"] == "agent1"
        assert exported["agent2"]["agent_name"] == "agent2"

    def test_import_all(self):
        """Test importing models."""
        data = {
            "agent1": {
                "agent_name": "agent1",
                "specializations": [
                    {
                        "specialization": "testing",
                        "proficiency": 0.8,
                        "related_capabilities": ["test_runner"],
                        "experience_count": 5
                    }
                ],
                "known_agents": {},
                "task_history": [],
                "created_at": 1000.0,
                "updated_at": 2000.0
            },
            "agent2": {
                "agent_name": "agent2",
                "specializations": [],
                "known_agents": {"agent1": ["test_runner"]},
                "task_history": [],
                "created_at": 1000.0,
                "updated_at": 2000.0
            }
        }

        manager = MentalModelManager()
        manager.import_all(data)

        assert len(manager._models) == 2
        assert manager.get("agent1") is not None
        assert manager.get("agent2") is not None
        assert len(manager.get("agent1").specializations) == 1
        assert "agent1" in manager.get("agent2").known_agents

    def test_export_import_round_trip(self):
        """Test that export/import preserves all data."""
        manager1 = MentalModelManager()
        manager1.get_or_create("agent1").add_specialization(SpecializationType.TESTING, proficiency=0.9)
        manager1.get_or_create("agent2").learn_about_agent("agent1", ["test_runner"])

        exported = manager1.export_all()

        manager2 = MentalModelManager()
        manager2.import_all(exported)

        assert len(manager2._models) == 2
        assert manager2.get("agent1").specializations[0].proficiency == 0.9
        assert "agent1" in manager2.get("agent2").known_agents

    def test_clear(self):
        """Test clearing all models."""
        manager = MentalModelManager()
        manager.get_or_create("agent1")
        manager.get_or_create("agent2")

        manager.clear()

        assert manager._models == {}
        assert manager.list_agents() == []


# ============================================================================
# create_default_model Tests
# ============================================================================

class TestCreateDefaultModel:
    """Tests for create_default_model factory function."""

    def test_default_general_specialization(self):
        """Test that default model gets GENERAL specialization."""
        model = create_default_model("test_agent")

        assert model.agent_name == "test_agent"
        assert len(model.specializations) == 1
        assert model.specializations[0].specialization == SpecializationType.GENERAL

    def test_default_proficiency(self):
        """Test that default proficiency is 0.5."""
        model = create_default_model("test_agent")

        assert model.specializations[0].proficiency == 0.5

    def test_default_capabilities_assigned(self):
        """Test that default GENERAL capabilities are assigned."""
        model = create_default_model("test_agent")

        expected_caps = DEFAULT_SPECIALIZATION_CAPABILITIES[SpecializationType.GENERAL]
        assert model.specializations[0].related_capabilities == expected_caps

    def test_single_custom_specialization(self):
        """Test creating model with single custom specialization."""
        model = create_default_model(
            "test_agent",
            specializations=[SpecializationType.TESTING]
        )

        assert len(model.specializations) == 1
        assert model.specializations[0].specialization == SpecializationType.TESTING

    def test_multiple_custom_specializations(self):
        """Test creating model with multiple specializations."""
        model = create_default_model(
            "test_agent",
            specializations=[
                SpecializationType.CODE_ANALYSIS,
                SpecializationType.TESTING,
                SpecializationType.DEBUGGING
            ]
        )

        assert len(model.specializations) == 3
        spec_types = [s.specialization for s in model.specializations]
        assert SpecializationType.CODE_ANALYSIS in spec_types
        assert SpecializationType.TESTING in spec_types
        assert SpecializationType.DEBUGGING in spec_types

    @pytest.mark.parametrize("spec_type", [
        SpecializationType.CODE_ANALYSIS,
        SpecializationType.CODE_GENERATION,
        SpecializationType.FILE_OPERATIONS,
        SpecializationType.SEARCH_DISCOVERY,
        SpecializationType.TESTING,
        SpecializationType.DOCUMENTATION,
        SpecializationType.DEBUGGING,
        SpecializationType.ARCHITECTURE
    ])
    def test_specialization_gets_correct_capabilities(self, spec_type):
        """Test that each specialization type gets correct default capabilities."""
        model = create_default_model("test_agent", specializations=[spec_type])

        expected_caps = DEFAULT_SPECIALIZATION_CAPABILITIES.get(spec_type, [])
        actual_caps = model.specializations[0].related_capabilities

        assert actual_caps == expected_caps

    def test_empty_specializations_list_uses_default(self):
        """Test that empty list defaults to GENERAL."""
        model = create_default_model("test_agent", specializations=[])

        assert len(model.specializations) == 1
        assert model.specializations[0].specialization == SpecializationType.GENERAL


# ============================================================================
# Integration Tests
# ============================================================================

class TestMentalModelsIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_single_agent(self):
        """Test complete workflow for a single agent."""
        manager = MentalModelManager()

        # Create agent with custom model
        model = create_default_model(
            "code_agent",
            specializations=[SpecializationType.CODE_ANALYSIS, SpecializationType.TESTING]
        )
        manager.update(model)

        # Record some experience
        code_spec = model.get_specialization(SpecializationType.CODE_ANALYSIS)
        code_spec.record_experience(success=True)
        code_spec.record_experience(success=True)

        # Record task outcome
        model.record_task_outcome("analysis_task", True, ["file_read", "code_search"])

        # Check fitness
        fitness = model.calculate_task_fitness(["file_read", "code_search"])

        assert fitness > 0.5
        assert code_spec.experience_count == 2
        assert len(model.task_history) == 1

    def test_multi_agent_collaboration(self):
        """Test multiple agents learning about each other."""
        manager = MentalModelManager()

        # Create specialized agents
        analyzer = create_default_model("analyzer", [SpecializationType.CODE_ANALYSIS])
        tester = create_default_model("tester", [SpecializationType.TESTING])
        writer = create_default_model("writer", [SpecializationType.DOCUMENTATION])

        manager.update(analyzer)
        manager.update(tester)
        manager.update(writer)

        # Broadcast capabilities
        analyzer_caps = analyzer.specializations[0].related_capabilities
        manager.broadcast_agent_discovery("analyzer", analyzer_caps)

        tester_caps = tester.specializations[0].related_capabilities
        manager.broadcast_agent_discovery("tester", tester_caps)

        # Check that agents know each other
        assert "analyzer" in manager.get("tester").known_agents
        assert "analyzer" in manager.get("writer").known_agents
        assert "tester" in manager.get("analyzer").known_agents

    def test_agent_selection_workflow(self):
        """Test selecting best agent for a task."""
        manager = MentalModelManager()

        # Create agents with different capabilities
        generalist = create_default_model("generalist", [SpecializationType.GENERAL])
        specialist = create_default_model("specialist", [SpecializationType.CODE_ANALYSIS])

        manager.update(generalist)
        manager.update(specialist)

        # Find best for code analysis task
        best = manager.find_best_agent(["file_read", "code_search", "ripgrep_search"])

        # Specialist should be better match
        assert best == "specialist"

    def test_persistence_workflow(self):
        """Test full persistence cycle."""
        # Create manager with data
        manager1 = MentalModelManager()
        model1 = create_default_model("agent1", [SpecializationType.TESTING])
        model1.record_task_outcome("test_task", True, ["test_runner"])
        manager1.update(model1)

        model2 = create_default_model("agent2", [SpecializationType.CODE_ANALYSIS])
        manager1.update(model2)

        # Export
        exported = manager1.export_all()

        # Create new manager and import
        manager2 = MentalModelManager()
        manager2.import_all(exported)

        # Verify all data preserved
        restored1 = manager2.get("agent1")
        assert restored1 is not None
        assert len(restored1.task_history) == 1
        assert restored1.specializations[0].specialization == SpecializationType.TESTING

        restored2 = manager2.get("agent2")
        assert restored2 is not None
        assert restored2.specializations[0].specialization == SpecializationType.CODE_ANALYSIS
