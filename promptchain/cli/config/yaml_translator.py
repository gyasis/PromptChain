"""YAML configuration translator for PromptChain CLI.

This module translates declarative YAML configurations into PromptChain
runtime objects (AgentChain, PromptChain, MCPServerConfig, etc.).

Supports:
- Environment variable substitution (${VAR_NAME})
- Agent configuration translation
- MCP server configuration translation
- Orchestration settings translation
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from promptchain import PromptChain
from promptchain.utils.agent_chain import AgentChain
from promptchain.utils.agentic_step_processor import AgenticStepProcessor

from ..models.agent_config import Agent, HistoryConfig
from ..models.mcp_config import MCPServerConfig
from ..models.orchestration_config import OrchestrationConfig, RouterConfig
from ..security.yaml_validator import YAMLValidator, ValidationError
from ..security.input_sanitizer import InputSanitizer


@dataclass
class YAMLSessionConfig:
    """Session configuration from YAML."""

    auto_save_interval: int = 300
    max_history_entries: int = 50
    working_directory: Optional[str] = None


@dataclass
class YAMLPreferencesConfig:
    """User preferences from YAML."""

    verbose: bool = False
    theme: str = "dark"
    show_token_usage: bool = True
    show_reasoning_steps: bool = True


@dataclass
class YAMLOrchestrationConfig:
    """Orchestration configuration from YAML."""

    execution_mode: str = "router"
    default_agent: Optional[str] = None
    router: Optional[Dict[str, Any]] = None


@dataclass
class YAMLAgentConfig:
    """Agent configuration from YAML."""

    model: str
    description: str
    instruction_chain: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    history_config: Optional[Dict[str, Any]] = None


@dataclass
class YAMLConfig:
    """Parsed and validated YAML configuration."""

    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    agents: Dict[str, YAMLAgentConfig] = field(default_factory=dict)
    orchestration: YAMLOrchestrationConfig = field(
        default_factory=YAMLOrchestrationConfig
    )
    session: YAMLSessionConfig = field(default_factory=YAMLSessionConfig)
    preferences: YAMLPreferencesConfig = field(default_factory=YAMLPreferencesConfig)


class YAMLConfigTranslator:
    """Translates YAML configuration to PromptChain runtime objects.

    Handles:
    - Loading and parsing YAML files
    - Environment variable substitution
    - Translation to Agent, MCPServerConfig, OrchestrationConfig
    - Building AgentChain from YAML configuration
    """

    def __init__(self):
        """Initialize YAML config translator."""
        self.env_var_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
        self.yaml_validator = YAMLValidator()  # T109: Security validator
        self.input_sanitizer = InputSanitizer()  # T109: Input sanitizer

    def substitute_env_vars(self, value: Any) -> Any:
        """Recursively substitute environment variables in configuration values.

        Supports ${VAR_NAME} syntax for environment variable substitution.

        Args:
            value: Configuration value (str, dict, list, or other)

        Returns:
            Any: Value with environment variables substituted
        """
        if isinstance(value, str):
            # Replace ${VAR_NAME} with environment variable value
            def replacer(match):
                var_name = match.group(1)
                env_value = os.environ.get(var_name)
                if env_value is None:
                    raise ValueError(
                        f"Environment variable {var_name} not found (referenced as ${{{var_name}}})"
                    )
                return env_value

            return self.env_var_pattern.sub(replacer, value)

        elif isinstance(value, dict):
            return {k: self.substitute_env_vars(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self.substitute_env_vars(item) for item in value]

        else:
            return value

    def load_yaml(self, path: Union[str, Path]) -> YAMLConfig:
        """Load and parse YAML configuration file with security validation.

        Args:
            path: Path to YAML configuration file

        Returns:
            YAMLConfig: Parsed configuration object

        Raises:
            FileNotFoundError: If YAML file not found
            yaml.YAMLError: If YAML parsing fails
            ValueError: If environment variable substitution fails
            ValidationError: If security validation fails (T109)
        """
        path = Path(path)

        # T109: Security validation BEFORE processing
        # Prevents YAML injection, YAML bombs, and malicious content
        raw_config = self.yaml_validator.load_and_validate(path)

        if raw_config is None:
            raw_config = {}

        # Substitute environment variables
        config_data = self.substitute_env_vars(raw_config)

        # Parse into structured config
        return self._parse_yaml_config(config_data)

    def _parse_yaml_config(self, data: Dict[str, Any]) -> YAMLConfig:
        """Parse raw YAML data into YAMLConfig object with input validation.

        Args:
            data: Raw YAML data dictionary

        Returns:
            YAMLConfig: Structured configuration object

        Raises:
            ValidationError: If input validation fails (T109)
        """
        # Parse agents
        agents = {}
        for agent_name, agent_data in data.get("agents", {}).items():
            # T109: Validate agent name
            validated_name = self.input_sanitizer.validate_name(agent_name, context="agent name")

            # T109: Validate model name
            validated_model = self.input_sanitizer.validate_model_name(agent_data["model"])

            # T109: Validate instruction chain
            instruction_chain = agent_data.get("instruction_chain", ["{input}"])
            validated_instructions = self.input_sanitizer.validate_instruction_chain(instruction_chain)

            agents[agent_name] = YAMLAgentConfig(
                model=validated_model,
                description=agent_data.get("description", ""),
                instruction_chain=validated_instructions,
                tools=agent_data.get("tools", []),
                history_config=agent_data.get("history_config"),
            )

        # Parse orchestration
        orch_data = data.get("orchestration", {})
        orchestration = YAMLOrchestrationConfig(
            execution_mode=orch_data.get("execution_mode", "router"),
            default_agent=orch_data.get("default_agent"),
            router=orch_data.get("router"),
        )

        # Parse session
        session_data = data.get("session", {})
        session = YAMLSessionConfig(
            auto_save_interval=session_data.get("auto_save_interval", 300),
            max_history_entries=session_data.get("max_history_entries", 50),
            working_directory=session_data.get("working_directory"),
        )

        # Parse preferences
        pref_data = data.get("preferences", {})
        preferences = YAMLPreferencesConfig(
            verbose=pref_data.get("verbose", False),
            theme=pref_data.get("theme", "dark"),
            show_token_usage=pref_data.get("show_token_usage", True),
            show_reasoning_steps=pref_data.get("show_reasoning_steps", True),
        )

        return YAMLConfig(
            mcp_servers=data.get("mcp_servers", []),
            agents=agents,
            orchestration=orchestration,
            session=session,
            preferences=preferences,
        )

    def _create_agentic_step_processor(
        self,
        config: Dict[str, Any],
        default_model: Optional[str] = None
    ) -> AgenticStepProcessor:
        """Create AgenticStepProcessor from YAML agentic_step config (T050).

        Args:
            config: Agentic step configuration dictionary from YAML
            default_model: Optional default model to use if not specified in config

        Returns:
            AgenticStepProcessor: Configured agentic step processor instance

        Raises:
            ValueError: If history_mode is invalid

        Example YAML config:
            ```yaml
            type: agentic_step
            objective: "Research {topic} comprehensively"
            max_internal_steps: 8
            history_mode: progressive
            model_name: anthropic/claude-3-opus  # Optional, overrides default

            # Phase 1: Two-Tier Routing (Cost Optimization)
            enable_two_tier_routing: true
            fallback_model: anthropic/claude-3-haiku

            # Phase 2: Blackboard Architecture (Token Optimization)
            enable_blackboard: true

            # Phase 3: Safety & Reliability (Error Reduction)
            enable_cove: true
            cove_confidence_threshold: 0.7
            enable_checkpointing: true

            # Phase 4: TAO Loop + Transparent Reasoning
            enable_tao_loop: true
            enable_dry_run: true
            ```
        """
        # Extract objective with placeholder for missing value (per test expectations)
        objective = config.get("objective", "Complete task")

        # Extract optional fields with defaults using AgenticConfig defaults
        # Default: 15 internal steps, progressive history mode
        from ..models.config import AgenticConfig
        agentic_defaults = AgenticConfig()
        max_internal_steps = config.get("max_internal_steps", agentic_defaults.default_max_internal_steps)
        history_mode = config.get("history_mode", agentic_defaults.history_mode)
        model_name = config.get("model_name", default_model)  # Optional override
        model_params = config.get("model_params", None)  # Optional model parameters

        # Phase 1: Two-Tier Routing parameters (v0.4.3+)
        enable_two_tier_routing = config.get("enable_two_tier_routing", False)
        fallback_model = config.get("fallback_model", None)

        # Phase 2: Blackboard Architecture parameters (v0.4.3+)
        enable_blackboard = config.get("enable_blackboard", False)

        # Phase 3: Safety & Reliability parameters (v0.4.3+)
        enable_cove = config.get("enable_cove", False)
        cove_confidence_threshold = config.get("cove_confidence_threshold", 0.7)
        enable_checkpointing = config.get("enable_checkpointing", False)

        # Phase 4: TAO Loop + Dry Run parameters (v0.4.3+)
        enable_tao_loop = config.get("enable_tao_loop", False)
        enable_dry_run = config.get("enable_dry_run", False)

        # Validate history_mode if provided
        valid_modes = ["minimal", "progressive", "kitchen_sink"]
        if history_mode not in valid_modes:
            raise ValueError(
                f"Invalid history_mode '{history_mode}' in agentic_step config. "
                f"Valid modes: {', '.join(valid_modes)}"
            )

        # Create AgenticStepProcessor instance with all Phase 1-4 features
        processor = AgenticStepProcessor(
            objective=objective,
            max_internal_steps=max_internal_steps,
            model_name=model_name,
            model_params=model_params,
            history_mode=history_mode,

            # Phase 1: Two-Tier Routing
            enable_two_tier_routing=enable_two_tier_routing,
            fallback_model=fallback_model,

            # Phase 2: Blackboard Architecture
            enable_blackboard=enable_blackboard,

            # Phase 3: Safety & Reliability
            enable_cove=enable_cove,
            cove_confidence_threshold=cove_confidence_threshold,
            enable_checkpointing=enable_checkpointing,

            # Phase 4: TAO Loop + Transparent Reasoning
            enable_tao_loop=enable_tao_loop,
            enable_dry_run=enable_dry_run,
        )

        return processor

    def _build_instruction_chain(
        self,
        agent_config: YAMLAgentConfig
    ) -> List[Union[str, AgenticStepProcessor]]:
        """Build instruction chain with support for strings and AgenticStepProcessor (T049).

        Processes instruction chain entries and converts agentic_step configs
        into AgenticStepProcessor instances while preserving string instructions.

        Args:
            agent_config: YAML agent configuration containing instruction_chain

        Returns:
            List[Union[str, AgenticStepProcessor]]: Processed instruction chain

        Raises:
            ValueError: If instruction format is invalid
        """
        instructions = []

        for instruction in agent_config.instruction_chain:
            if isinstance(instruction, str):
                # String instruction: preserve as-is (template variables intact)
                instructions.append(instruction)

            elif isinstance(instruction, dict):
                if instruction.get("type") == "agentic_step":
                    # Create AgenticStepProcessor from config
                    processor = self._create_agentic_step_processor(
                        config=instruction,
                        default_model=agent_config.model
                    )
                    instructions.append(processor)
                else:
                    # Other instruction types (e.g., function refs) - preserve as-is
                    # Allows future extension for other instruction types
                    instructions.append(instruction)

            else:
                raise ValueError(
                    f"Invalid instruction format in agent '{agent_config.model}': {type(instruction)}. "
                    "Expected string or dict with 'type' field."
                )

        return instructions

    def build_agents(
        self, yaml_config: YAMLConfig
    ) -> Dict[str, PromptChain]:
        """Convert YAML agent configurations to PromptChain instances (T017).

        Args:
            yaml_config: Parsed YAML configuration

        Returns:
            Dict[str, PromptChain]: Dictionary of agent name to PromptChain instance
        """
        agents = {}

        for agent_name, agent_config in yaml_config.agents.items():
            # Build instruction chain using new processor (T049)
            instructions = self._build_instruction_chain(agent_config)

            # Create PromptChain instance
            chain = PromptChain(
                models=[agent_config.model],
                instructions=instructions,
                verbose=False,  # Controlled by preferences
            )

            agents[agent_name] = chain

        return agents

    def build_agent_chain(
        self, yaml_config: YAMLConfig, agents: Dict[str, PromptChain]
    ) -> AgentChain:
        """Convert YAML orchestration config to AgentChain (T018).

        Args:
            yaml_config: Parsed YAML configuration
            agents: Dictionary of PromptChain instances

        Returns:
            AgentChain: Configured agent orchestrator
        """
        # Build agent descriptions for router
        agent_descriptions = {
            name: yaml_config.agents[name].description for name in agents.keys()
        }

        # Build router configuration
        router_config = None
        if yaml_config.orchestration.execution_mode == "router":
            router_data = yaml_config.orchestration.router or {}
            router_config = {
                "models": [router_data.get("model", "openai/gpt-4o-mini")],
                "instructions": [None, "{input}"],
            }

            # Add decision prompt if specified
            if router_data.get("decision_prompt"):
                router_config["decision_prompt_templates"] = {
                    "single_agent_dispatch": router_data["decision_prompt"]
                }

        # Build AgentChain
        agent_chain = AgentChain(
            agents=agents,
            agent_descriptions=agent_descriptions,
            execution_mode=yaml_config.orchestration.execution_mode,
            router=router_config,
            verbose=yaml_config.preferences.verbose,
        )

        return agent_chain

    def build_mcp_servers(
        self, yaml_config: YAMLConfig
    ) -> List[MCPServerConfig]:
        """Parse MCP server configurations from YAML (T019).

        Args:
            yaml_config: Parsed YAML configuration

        Returns:
            List[MCPServerConfig]: List of MCP server configurations
        """
        mcp_servers = []

        for server_data in yaml_config.mcp_servers:
            server_config = MCPServerConfig(
                id=server_data["id"],
                type=server_data["type"],
                command=server_data.get("command"),
                args=server_data.get("args", []),
                url=server_data.get("url"),
                auto_connect=server_data.get("auto_connect", True),
            )
            mcp_servers.append(server_config)

        return mcp_servers

    def build_orchestration_config(
        self, yaml_config: YAMLConfig
    ) -> OrchestrationConfig:
        """Build OrchestrationConfig from YAML configuration.

        Args:
            yaml_config: Parsed YAML configuration

        Returns:
            OrchestrationConfig: Orchestration configuration object
        """
        # Build router config if in router mode
        router_config = None
        if yaml_config.orchestration.execution_mode == "router":
            router_data = yaml_config.orchestration.router or {}
            router_config = RouterConfig(
                model=router_data.get("model", "openai/gpt-4o-mini"),
                timeout_seconds=router_data.get("timeout_seconds", 10),
            )

            # Use custom decision prompt if provided
            if router_data.get("decision_prompt"):
                router_config.decision_prompt_template = router_data["decision_prompt"]

        # Build orchestration config
        return OrchestrationConfig(
            execution_mode=yaml_config.orchestration.execution_mode,
            default_agent=yaml_config.orchestration.default_agent,
            router_config=router_config,
            auto_include_history=True,  # Default to enabled
        )

    def translate_to_agent_configs(
        self, yaml_config: YAMLConfig
    ) -> Dict[str, Agent]:
        """Translate YAML agent configs to Agent dataclass instances.

        Args:
            yaml_config: Parsed YAML configuration

        Returns:
            Dict[str, Agent]: Dictionary of agent name to Agent config
        """
        agent_configs = {}

        for agent_name, yaml_agent in yaml_config.agents.items():
            # Build history config
            history_config = None
            if yaml_agent.history_config:
                history_config = HistoryConfig(
                    enabled=yaml_agent.history_config.get("enabled", True),
                    max_tokens=yaml_agent.history_config.get("max_tokens", 4000),
                    max_entries=yaml_agent.history_config.get("max_entries", 20),
                    truncation_strategy=yaml_agent.history_config.get(
                        "truncation_strategy", "oldest_first"
                    ),
                    include_types=yaml_agent.history_config.get("include_types"),
                    exclude_sources=yaml_agent.history_config.get("exclude_sources"),
                )

            # Create Agent config
            agent_config = Agent(
                name=agent_name,
                model_name=yaml_agent.model,
                description=yaml_agent.description,
                instruction_chain=yaml_agent.instruction_chain,
                tools=yaml_agent.tools,
                history_config=history_config,
            )

            agent_configs[agent_name] = agent_config

        return agent_configs


def load_config_with_precedence(
    cli_config_path: Optional[str] = None,
) -> Optional[YAMLConfig]:
    """Load YAML config with precedence rules (T021).

    Precedence order (highest to lowest):
    1. CLI argument (--config path)
    2. Project-level .promptchain.yml
    3. User-level ~/.promptchain/config.yml
    4. None (use defaults)

    Args:
        cli_config_path: Optional path from CLI argument

    Returns:
        Optional[YAMLConfig]: Loaded configuration or None if no config found
    """
    translator = YAMLConfigTranslator()

    # 1. CLI argument (highest precedence)
    if cli_config_path:
        cli_path = Path(cli_config_path)
        if cli_path.exists():
            return translator.load_yaml(cli_path)
        else:
            raise FileNotFoundError(f"CLI config file not found: {cli_config_path}")

    # 2. Project-level .promptchain.yml
    project_config = Path.cwd() / ".promptchain.yml"
    if project_config.exists():
        return translator.load_yaml(project_config)

    # 3. User-level ~/.promptchain/config.yml
    user_config = Path.home() / ".promptchain" / "config.yml"
    if user_config.exists():
        return translator.load_yaml(user_config)

    # 4. No config found, use defaults
    return None
