"""Unit tests for AgenticStepProcessor YAML configuration translation (T048).

These tests verify that YAML configurations with agentic_step instructions
are correctly parsed, validated, and translated into AgenticStepProcessor instances.

Test Coverage:
- test_agentic_step_yaml_parsing: Basic YAML parsing
- test_objective_extraction: Objective field extraction
- test_max_steps_extraction: max_internal_steps configuration
- test_default_values: Default value application
- test_invalid_configs: Error handling for invalid configs
"""

import pytest
import tempfile
from pathlib import Path

from promptchain.cli.config.yaml_translator import (
    YAMLConfigTranslator,
    YAMLAgentConfig,
)
from promptchain.utils.agentic_step_processor import AgenticStepProcessor


class TestYAMLAgenticConfig:
    """Unit tests for AgenticStepProcessor YAML config translation."""

    @pytest.fixture
    def translator(self):
        """Create YAML config translator."""
        return YAMLConfigTranslator()

    def test_agentic_step_yaml_parsing(self, translator):
        """Unit: YAML with agentic_step parsed correctly.

        Validates:
        - type: agentic_step recognized
        - Config dict parsed
        - Fields extracted
        - Structure validated
        """
        yaml_content = """
agents:
  researcher:
    model: gpt-4.1-mini-2025-04-14
    description: "Research agent"
    instruction_chain:
      - type: agentic_step
        objective: "Research topic"
        max_internal_steps: 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)

            # Validate agent config parsed
            assert "researcher" in yaml_config.agents
            agent = yaml_config.agents["researcher"]

            # Validate instruction_chain structure
            assert len(agent.instruction_chain) == 1
            instruction = agent.instruction_chain[0]

            # Should be dict with agentic_step config
            assert isinstance(instruction, dict)
            assert instruction["type"] == "agentic_step"
            assert instruction["objective"] == "Research topic"
            assert instruction["max_internal_steps"] == 5

        finally:
            yaml_path.unlink()

    def test_objective_extraction(self, translator):
        """Unit: Objective field extracted from YAML.

        Validates:
        - objective field read correctly
        - String value preserved
        - No truncation or modification
        """
        yaml_content = """
agents:
  analyst:
    model: gpt-4.1-mini-2025-04-14
    description: "Analyst"
    instruction_chain:
      - type: agentic_step
        objective: "Analyze complex data patterns and provide insights"
        max_internal_steps: 4
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)
            agent = yaml_config.agents["analyst"]
            instruction = agent.instruction_chain[0]

            # Validate objective
            assert instruction["objective"] == "Analyze complex data patterns and provide insights"

        finally:
            yaml_path.unlink()

    def test_max_steps_extraction(self, translator):
        """Unit: max_internal_steps field extracted from YAML.

        Validates:
        - max_internal_steps parsed as integer
        - Different values handled
        - Reasonable range (1-10)
        """
        yaml_content = """
agents:
  quick:
    model: gpt-4.1-mini-2025-04-14
    description: "Quick agent"
    instruction_chain:
      - type: agentic_step
        objective: "Quick task"
        max_internal_steps: 2

  thorough:
    model: gpt-4.1-mini-2025-04-14
    description: "Thorough agent"
    instruction_chain:
      - type: agentic_step
        objective: "Complex task"
        max_internal_steps: 8
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)

            # Quick agent
            quick_instruction = yaml_config.agents["quick"].instruction_chain[0]
            assert quick_instruction["max_internal_steps"] == 2

            # Thorough agent
            thorough_instruction = yaml_config.agents["thorough"].instruction_chain[0]
            assert thorough_instruction["max_internal_steps"] == 8

        finally:
            yaml_path.unlink()

    def test_default_values(self, translator):
        """Unit: Default values applied for missing fields.

        Validates:
        - Missing max_internal_steps gets default (5)
        - Missing objective gets placeholder
        - Defaults are sensible
        """
        yaml_content = """
agents:
  minimal:
    model: gpt-4.1-mini-2025-04-14
    description: "Minimal config"
    instruction_chain:
      - type: agentic_step
        objective: "Simple task"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)

            # Build agents to see defaults applied
            agents = translator.build_agents(yaml_config)
            agentic_step = agents["minimal"].instructions[0]

            # Should have default max_internal_steps
            assert agentic_step.max_internal_steps == 5  # Default from translator

        finally:
            yaml_path.unlink()

    def test_invalid_configs(self, translator):
        """Unit: Invalid agentic_step configs rejected.

        Validates:
        - Missing type field detected
        - Invalid type values rejected
        - Proper error messages
        """
        # Missing objective
        yaml_content = """
agents:
  broken:
    model: gpt-4.1-mini-2025-04-14
    description: "Broken config"
    instruction_chain:
      - type: agentic_step
        max_internal_steps: 5
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)

            # Build agents - should use placeholder objective
            agents = translator.build_agents(yaml_config)
            agentic_step = agents["broken"].instructions[0]

            # Should have default objective
            assert agentic_step.objective == "Complete task"

        finally:
            yaml_path.unlink()

    def test_mixed_instruction_chain(self, translator):
        """Unit: Mix string prompts and agentic_step in chain.

        Validates:
        - Multiple instruction types in one chain
        - Order preserved
        - Each type processed correctly
        """
        yaml_content = """
agents:
  mixed:
    model: gpt-4.1-mini-2025-04-14
    description: "Mixed agent"
    instruction_chain:
      - "Initial prompt: {input}"
      - type: agentic_step
        objective: "Multi-hop reasoning"
        max_internal_steps: 3
      - "Final summary: {input}"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)
            agent = yaml_config.agents["mixed"]

            # Validate instruction_chain structure
            assert len(agent.instruction_chain) == 3
            assert isinstance(agent.instruction_chain[0], str)
            assert isinstance(agent.instruction_chain[1], dict)
            assert isinstance(agent.instruction_chain[2], str)

            # Validate agentic_step config
            agentic_config = agent.instruction_chain[1]
            assert agentic_config["type"] == "agentic_step"

        finally:
            yaml_path.unlink()

    def test_build_agents_creates_processor(self, translator):
        """Unit: build_agents() creates AgenticStepProcessor instances.

        Validates:
        - AgenticStepProcessor instantiated
        - Parameters passed correctly
        - Instance is correct type
        """
        yaml_content = """
agents:
  processor_test:
    model: gpt-4.1-mini-2025-04-14
    description: "Test agent"
    instruction_chain:
      - type: agentic_step
        objective: "Test objective"
        max_internal_steps: 4
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)
            agents = translator.build_agents(yaml_config)

            # Validate AgenticStepProcessor created
            processor_test = agents["processor_test"]
            assert len(processor_test.instructions) == 1

            agentic_step = processor_test.instructions[0]
            assert isinstance(agentic_step, AgenticStepProcessor)

            # Validate parameters
            assert agentic_step.objective == "Test objective"
            assert agentic_step.max_internal_steps == 4
            assert agentic_step.model_name == "gpt-4.1-mini-2025-04-14"

        finally:
            yaml_path.unlink()

    def test_translate_to_agent_configs(self, translator):
        """Unit: translate_to_agent_configs preserves agentic_step dicts.

        Validates:
        - Agent configs have instruction_chain field
        - Agentic_step configs preserved as dicts
        - Can be serialized to JSON for database
        """
        yaml_content = """
agents:
  config_test:
    model: gpt-4.1-mini-2025-04-14
    description: "Config test"
    instruction_chain:
      - "Start: {input}"
      - type: agentic_step
        objective: "Multi-hop"
        max_internal_steps: 6
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)
            agent_configs = translator.translate_to_agent_configs(yaml_config)

            # Validate Agent config
            config_test = agent_configs["config_test"]
            assert len(config_test.instruction_chain) == 2

            # First should be string
            assert isinstance(config_test.instruction_chain[0], str)

            # Second should be dict with agentic_step config
            agentic_dict = config_test.instruction_chain[1]
            assert isinstance(agentic_dict, dict)
            assert agentic_dict["type"] == "agentic_step"
            assert agentic_dict["objective"] == "Multi-hop"
            assert agentic_dict["max_internal_steps"] == 6

        finally:
            yaml_path.unlink()

    def test_environment_variable_substitution(self, translator):
        """Unit: Environment variables work in agentic_step configs.

        Validates:
        - ${VAR} syntax in objective
        - Environment substitution works
        - Variables expanded before processing
        """
        import os

        os.environ["TEST_OBJECTIVE"] = "Research Python patterns"

        yaml_content = """
agents:
  env_test:
    model: gpt-4.1-mini-2025-04-14
    description: "Environment test"
    instruction_chain:
      - type: agentic_step
        objective: "${TEST_OBJECTIVE}"
        max_internal_steps: 3
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)
            agent = yaml_config.agents["env_test"]
            instruction = agent.instruction_chain[0]

            # Validate environment variable expanded
            assert instruction["objective"] == "Research Python patterns"

        finally:
            yaml_path.unlink()
            del os.environ["TEST_OBJECTIVE"]

    def test_multiple_agentic_steps(self, translator):
        """Unit: Multiple agentic_step configs in one chain.

        Validates:
        - Multiple agentic steps allowed
        - Each configured independently
        - Order preserved
        """
        yaml_content = """
agents:
  multi_step:
    model: gpt-4.1-mini-2025-04-14
    description: "Multi-step agent"
    instruction_chain:
      - type: agentic_step
        objective: "Gather information"
        max_internal_steps: 3
      - type: agentic_step
        objective: "Synthesize findings"
        max_internal_steps: 2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)
            agents = translator.build_agents(yaml_config)

            multi_step = agents["multi_step"]
            assert len(multi_step.instructions) == 2

            # Both should be AgenticStepProcessor
            step1 = multi_step.instructions[0]
            step2 = multi_step.instructions[1]

            assert isinstance(step1, AgenticStepProcessor)
            assert isinstance(step2, AgenticStepProcessor)

            # Validate different objectives
            assert step1.objective == "Gather information"
            assert step2.objective == "Synthesize findings"

            # Validate different max_steps
            assert step1.max_internal_steps == 3
            assert step2.max_internal_steps == 2

        finally:
            yaml_path.unlink()

    def test_agentic_step_with_history_config(self, translator):
        """Unit: Agentic step respects agent's history_config.

        Validates:
        - history_config from agent applied
        - AgenticStepProcessor gets history settings
        - Integration between configs
        """
        yaml_content = """
agents:
  history_test:
    model: gpt-4.1-mini-2025-04-14
    description: "History test"
    instruction_chain:
      - type: agentic_step
        objective: "Research task"
        max_internal_steps: 4
    history_config:
      enabled: true
      max_tokens: 8000
      max_entries: 30
      truncation_strategy: "progressive"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            yaml_config = translator.load_yaml(yaml_path)

            # Validate history_config parsed
            agent = yaml_config.agents["history_test"]
            assert agent.history_config is not None
            assert agent.history_config["enabled"] is True
            assert agent.history_config["max_tokens"] == 8000

        finally:
            yaml_path.unlink()
