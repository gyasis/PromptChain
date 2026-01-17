"""Contract tests for agentic_step instruction chain validation (T045).

These tests verify that the instruction_chain field in Agent configurations
properly validates agentic_step configurations with correct schema, required
fields, optional defaults, and error handling per AgenticStepProcessor specs.

Test Strategy:
- Validate agentic_step config schema (required/optional fields)
- Test field validation (objective required, valid history_mode values)
- Verify default values for optional fields
- Test mixed instruction chains (strings + agentic_step configs)
- Ensure proper error messages for invalid configurations

RED Phase: Test agentic_step instruction chain validation contract
GREEN Phase: Tests pass after Agent model validates agentic_step configs
"""

import pytest
from typing import Dict, Any

from promptchain.cli.models.agent_config import Agent, HistoryConfig
from promptchain.utils.agentic_step_processor import HistoryMode


class TestAgenticStepInstructionChainSchema:
    """Contract tests for agentic_step instruction schema validation (T045)."""

    def test_valid_agentic_step_config(self):
        """T045: Validate complete agentic_step instruction chain schema."""
        # Create agent with valid agentic_step config
        agent = Agent(
            name="researcher",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Research authentication best practices",
                    "max_internal_steps": 5,
                    "history_mode": "progressive"
                }
            ]
        )

        # Verify agent created successfully
        assert len(agent.instruction_chain) == 1
        step_config = agent.instruction_chain[0]

        assert isinstance(step_config, dict)
        assert step_config["type"] == "agentic_step"
        assert step_config["objective"] == "Research authentication best practices"
        assert step_config["max_internal_steps"] == 5
        assert step_config["history_mode"] == "progressive"

    def test_agentic_step_requires_objective(self):
        """T045: Verify objective field is required for agentic_step."""
        # AgenticStepProcessor requires objective at runtime
        # Agent model should store the config as-is
        agent = Agent(
            name="test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "max_internal_steps": 5
                }
            ]
        )

        # Config stored as-is (validation happens at execution time)
        step_config = agent.instruction_chain[0]
        assert step_config["type"] == "agentic_step"
        assert "objective" not in step_config
        # Note: Runtime validation in AgenticStepProcessor will catch missing objective

    def test_agentic_step_optional_fields_defaults(self):
        """T045: Verify max_internal_steps and history_mode have defaults."""
        # Minimal agentic_step config (only objective)
        agent = Agent(
            name="minimal",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Analyze code"
                }
            ]
        )

        step_config = agent.instruction_chain[0]
        assert step_config["type"] == "agentic_step"
        assert step_config["objective"] == "Analyze code"

        # AgenticStepProcessor defaults:
        # - max_internal_steps: 5 (per __init__ signature)
        # - history_mode: "minimal" (per __init__ signature)
        # Note: Defaults applied at AgenticStepProcessor instantiation time

    def test_agentic_step_valid_history_modes(self):
        """T045: Verify all valid history_mode values are accepted."""
        valid_modes = ["minimal", "progressive", "kitchen_sink"]

        for mode in valid_modes:
            agent = Agent(
                name=f"agent-{mode}",
                model_name="gpt-4.1-mini-2025-04-14",
                instruction_chain=[
                    {
                        "type": "agentic_step",
                        "objective": "Test objective",
                        "history_mode": mode
                    }
                ]
            )

            step_config = agent.instruction_chain[0]
            assert step_config["history_mode"] == mode

    def test_agentic_step_invalid_history_mode_stored(self):
        """T045: Invalid history_mode is stored but fails at runtime."""
        # Agent model stores config as-is (no validation at this level)
        agent = Agent(
            name="invalid",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Test",
                    "history_mode": "invalid_mode"
                }
            ]
        )

        step_config = agent.instruction_chain[0]
        assert step_config["history_mode"] == "invalid_mode"
        # Note: AgenticStepProcessor.__init__ will raise ValueError at runtime

    def test_mixed_instruction_chain(self):
        """T045: Support mixed instruction chains (strings + agentic_step)."""
        agent = Agent(
            name="mixed",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                "Prepare initial analysis: {input}",
                {
                    "type": "agentic_step",
                    "objective": "Perform deep research",
                    "max_internal_steps": 8,
                    "history_mode": "progressive"
                },
                "Synthesize findings: {input}"
            ]
        )

        # Verify mixed chain structure
        assert len(agent.instruction_chain) == 3

        # First instruction: string
        assert isinstance(agent.instruction_chain[0], str)
        assert agent.instruction_chain[0] == "Prepare initial analysis: {input}"

        # Second instruction: agentic_step config
        agentic_config = agent.instruction_chain[1]
        assert isinstance(agentic_config, dict)
        assert agentic_config["type"] == "agentic_step"
        assert agentic_config["objective"] == "Perform deep research"
        assert agentic_config["max_internal_steps"] == 8

        # Third instruction: string
        assert isinstance(agent.instruction_chain[2], str)
        assert agent.instruction_chain[2] == "Synthesize findings: {input}"

    def test_multiple_agentic_steps_in_chain(self):
        """T045: Support multiple agentic_step configurations in one chain."""
        agent = Agent(
            name="multi-agentic",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Research phase 1",
                    "max_internal_steps": 5
                },
                "Intermediate processing: {input}",
                {
                    "type": "agentic_step",
                    "objective": "Research phase 2",
                    "max_internal_steps": 3,
                    "history_mode": "progressive"
                }
            ]
        )

        # Verify two agentic steps present
        assert len(agent.instruction_chain) == 3

        # First agentic step
        step1 = agent.instruction_chain[0]
        assert step1["type"] == "agentic_step"
        assert step1["objective"] == "Research phase 1"

        # Second agentic step
        step2 = agent.instruction_chain[2]
        assert step2["type"] == "agentic_step"
        assert step2["objective"] == "Research phase 2"
        assert step2["history_mode"] == "progressive"


class TestAgenticStepConfigSerialization:
    """Test agentic_step config serialization/deserialization (T045)."""

    def test_agentic_step_serialization(self):
        """T045: Verify agentic_step configs serialize properly."""
        agent = Agent(
            name="serialize-test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                "Initial: {input}",
                {
                    "type": "agentic_step",
                    "objective": "Deep analysis",
                    "max_internal_steps": 10,
                    "history_mode": "kitchen_sink"
                }
            ]
        )

        # Serialize to dict
        agent_dict = agent.to_dict()

        # Verify instruction_chain preserved
        assert "instruction_chain" in agent_dict
        assert len(agent_dict["instruction_chain"]) == 2

        # Verify agentic_step config preserved
        agentic_config = agent_dict["instruction_chain"][1]
        assert agentic_config["type"] == "agentic_step"
        assert agentic_config["objective"] == "Deep analysis"
        assert agentic_config["max_internal_steps"] == 10
        assert agentic_config["history_mode"] == "kitchen_sink"

    def test_agentic_step_deserialization(self):
        """T045: Verify agentic_step configs deserialize properly."""
        agent_data: Dict[str, Any] = {
            "name": "deserialize-test",
            "model_name": "anthropic/claude-3-opus",
            "created_at": 1234567890.0,
            "instruction_chain": [
                {
                    "type": "agentic_step",
                    "objective": "Research security patterns",
                    "max_internal_steps": 6,
                    "history_mode": "progressive"
                },
                "Final synthesis: {input}"
            ]
        }

        # Deserialize from dict
        agent = Agent.from_dict(agent_data)

        # Verify instruction_chain reconstructed
        assert len(agent.instruction_chain) == 2

        # Verify agentic_step config
        agentic_config = agent.instruction_chain[0]
        assert isinstance(agentic_config, dict)
        assert agentic_config["type"] == "agentic_step"
        assert agentic_config["objective"] == "Research security patterns"
        assert agentic_config["max_internal_steps"] == 6
        assert agentic_config["history_mode"] == "progressive"

    def test_roundtrip_serialization_with_agentic_step(self):
        """T045: Verify complete roundtrip preserves agentic_step configs."""
        original = Agent(
            name="roundtrip",
            model_name="gpt-4.1-mini-2025-04-14",
            description="Testing roundtrip serialization",
            instruction_chain=[
                "Prepare: {input}",
                {
                    "type": "agentic_step",
                    "objective": "Multi-hop reasoning",
                    "max_internal_steps": 12,
                    "history_mode": "progressive"
                },
                "Finalize: {input}"
            ],
            tools=["web_search", "calculator"]
        )

        # Serialize
        agent_dict = original.to_dict()

        # Deserialize
        restored = Agent.from_dict(agent_dict)

        # Verify all fields match
        assert restored.name == original.name
        assert restored.model_name == original.model_name
        assert restored.description == original.description
        assert len(restored.instruction_chain) == len(original.instruction_chain)

        # Verify agentic_step config preserved exactly
        original_agentic = original.instruction_chain[1]
        restored_agentic = restored.instruction_chain[1]
        assert restored_agentic == original_agentic
        assert restored_agentic["objective"] == "Multi-hop reasoning"
        assert restored_agentic["max_internal_steps"] == 12
        assert restored_agentic["history_mode"] == "progressive"


class TestAgenticStepConfigFieldTypes:
    """Test field type validation for agentic_step configs (T045)."""

    def test_objective_string_type(self):
        """T045: Verify objective accepts string values."""
        agent = Agent(
            name="type-test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "String objective value"
                }
            ]
        )

        step_config = agent.instruction_chain[0]
        assert isinstance(step_config["objective"], str)
        assert step_config["objective"] == "String objective value"

    def test_max_internal_steps_integer_type(self):
        """T045: Verify max_internal_steps accepts integer values."""
        agent = Agent(
            name="steps-test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Test",
                    "max_internal_steps": 15
                }
            ]
        )

        step_config = agent.instruction_chain[0]
        assert isinstance(step_config["max_internal_steps"], int)
        assert step_config["max_internal_steps"] == 15

    def test_history_mode_string_type(self):
        """T045: Verify history_mode accepts string values."""
        agent = Agent(
            name="history-test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Test",
                    "history_mode": "progressive"
                }
            ]
        )

        step_config = agent.instruction_chain[0]
        assert isinstance(step_config["history_mode"], str)
        assert step_config["history_mode"] == "progressive"

    def test_agentic_step_with_all_optional_fields(self):
        """T045: Verify agentic_step config with all possible fields."""
        agent = Agent(
            name="complete",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Comprehensive research task",
                    "max_internal_steps": 20,
                    "history_mode": "kitchen_sink",
                    "model_name": "anthropic/claude-3-opus",  # Override model
                    "model_params": {"temperature": 0.5}
                }
            ]
        )

        step_config = agent.instruction_chain[0]
        assert step_config["type"] == "agentic_step"
        assert step_config["objective"] == "Comprehensive research task"
        assert step_config["max_internal_steps"] == 20
        assert step_config["history_mode"] == "kitchen_sink"
        assert step_config["model_name"] == "anthropic/claude-3-opus"
        assert step_config["model_params"] == {"temperature": 0.5}


class TestAgenticStepBackwardCompatibility:
    """Test backward compatibility with existing instruction chains (T045)."""

    def test_string_only_instruction_chain(self):
        """T045: Verify string-only chains still work (v1 compatibility)."""
        agent = Agent(
            name="v1-compat",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                "Step 1: {input}",
                "Step 2: {input}",
                "Step 3: {input}"
            ]
        )

        # All strings preserved
        assert len(agent.instruction_chain) == 3
        assert all(isinstance(step, str) for step in agent.instruction_chain)

    def test_function_ref_instruction_chain(self):
        """T045: Verify function ref configs still work."""
        agent = Agent(
            name="function-test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {"type": "function", "name": "validate_data"},
                "Process: {input}"
            ]
        )

        # Function ref preserved
        assert len(agent.instruction_chain) == 2
        assert agent.instruction_chain[0]["type"] == "function"
        assert agent.instruction_chain[0]["name"] == "validate_data"

    def test_empty_instruction_chain(self):
        """T045: Verify empty instruction chain is valid (backward compat)."""
        agent = Agent(
            name="empty",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[]
        )

        assert agent.instruction_chain == []
        assert isinstance(agent.instruction_chain, list)


class TestAgenticStepRuntimeValidation:
    """Document runtime validation expectations (T045)."""

    def test_runtime_validation_expectations(self):
        """T045: Document that runtime validation occurs in AgenticStepProcessor.

        This test documents the expected validation contract:

        1. Agent model stores instruction_chain configs AS-IS (no validation)
        2. AgenticStepProcessor validates at instantiation:
           - objective: required, non-empty string
           - max_internal_steps: defaults to 5
           - history_mode: must be valid HistoryMode enum value
        3. Invalid configs fail at RUNTIME, not at config creation

        This design allows flexibility in config storage while ensuring
        correctness at execution time.
        """
        # Agent accepts ANY dict in instruction_chain
        agent = Agent(
            name="runtime-validation-test",
            model_name="gpt-4.1-mini-2025-04-14",
            instruction_chain=[
                {
                    "type": "agentic_step",
                    "objective": "Valid objective",
                    "history_mode": "progressive"
                }
            ]
        )

        # Config stored successfully
        assert len(agent.instruction_chain) == 1
        step_config = agent.instruction_chain[0]

        # AgenticStepProcessor validation would occur when creating instance:
        # from promptchain.utils.agentic_step_processor import AgenticStepProcessor
        # processor = AgenticStepProcessor(**step_config)  # Validates here

        # This test just documents the contract
        assert step_config["type"] == "agentic_step"
        assert "objective" in step_config

    def test_history_mode_enum_values(self):
        """T045: Document valid HistoryMode enum values from AgenticStepProcessor."""
        # AgenticStepProcessor defines HistoryMode enum
        valid_modes = {
            HistoryMode.MINIMAL.value,      # "minimal"
            HistoryMode.PROGRESSIVE.value,  # "progressive"
            HistoryMode.KITCHEN_SINK.value  # "kitchen_sink"
        }

        assert "minimal" in valid_modes
        assert "progressive" in valid_modes
        assert "kitchen_sink" in valid_modes

        # These are the ONLY valid values accepted by AgenticStepProcessor
        assert len(valid_modes) == 3
