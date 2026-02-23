"""YAML configuration validator using JSON Schema.

This module validates YAML configuration files against the JSON schema
defined in specs/002-cli-orchestration/contracts/yaml-config-schema.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception  # Fallback for type hints


class YAMLConfigValidator:
    """Validates YAML configuration against JSON schema.

    Uses jsonschema library to validate configuration structure,
    types, and constraints defined in yaml-config-schema.json.
    """

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize validator with JSON schema.

        Args:
            schema_path: Path to JSON schema file (default: auto-detect from specs/)
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema library not available. "
                "Install with: pip install jsonschema>=4.17"
            )

        # Auto-detect schema path if not provided
        if schema_path is None:
            # Try to find schema in specs directory
            possible_paths = [
                Path(__file__).parent.parent.parent.parent
                / "specs"
                / "002-cli-orchestration"
                / "contracts"
                / "yaml-config-schema.json",
                Path.cwd() / "specs" / "002-cli-orchestration" / "contracts" / "yaml-config-schema.json",
            ]

            for path in possible_paths:
                if path.exists():
                    schema_path = path
                    break

            if schema_path is None:
                # Use inline minimal schema as fallback
                self.schema = self._get_minimal_schema()
                return

        # Load schema from file
        with open(schema_path, "r") as f:
            self.schema = json.load(f)

        # Create validator
        self.validator = Draft7Validator(self.schema)

    def _get_minimal_schema(self) -> Dict[str, Any]:
        """Get minimal inline schema when schema file not found.

        Returns:
            Dict[str, Any]: Minimal JSON schema for YAML config
        """
        return {
            "type": "object",
            "properties": {
                "mcp_servers": {"type": "array"},
                "agents": {"type": "object"},
                "orchestration": {
                    "type": "object",
                    "properties": {
                        "execution_mode": {
                            "type": "string",
                            "enum": ["router", "pipeline", "round-robin", "broadcast"],
                        }
                    },
                },
                "session": {"type": "object"},
                "preferences": {"type": "object"},
            },
        }

    def validate(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration data against schema.

        Args:
            config_data: Parsed YAML configuration data

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        if not JSONSCHEMA_AVAILABLE:
            return []  # Skip validation if jsonschema not installed

        errors = []

        for error in self.validator.iter_errors(config_data):
            # Format error message with path
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            message = f"{path}: {error.message}"
            errors.append(message)

        return errors

    def validate_and_raise(self, config_data: Dict[str, Any]):
        """Validate configuration and raise exception if invalid.

        Args:
            config_data: Parsed YAML configuration data

        Raises:
            ValidationError: If configuration is invalid
        """
        errors = self.validate(config_data)

        if errors:
            error_message = "YAML configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValidationError(error_message)

    def validate_agent_config(self, agent_name: str, agent_config: Dict[str, Any]) -> List[str]:
        """Validate individual agent configuration.

        Args:
            agent_name: Name of the agent
            agent_config: Agent configuration dictionary

        Returns:
            List[str]: List of validation error messages
        """
        errors = []

        # Required fields
        if "model" not in agent_config:
            errors.append(f"Agent '{agent_name}': missing required field 'model'")

        if "description" not in agent_config:
            errors.append(f"Agent '{agent_name}': missing required field 'description'")

        # Validate instruction_chain
        if "instruction_chain" in agent_config:
            if not isinstance(agent_config["instruction_chain"], list):
                errors.append(
                    f"Agent '{agent_name}': instruction_chain must be a list"
                )
            else:
                for i, instruction in enumerate(agent_config["instruction_chain"]):
                    if isinstance(instruction, dict):
                        # Validate agentic_step format
                        if instruction.get("type") == "agentic_step":
                            if "objective" not in instruction:
                                errors.append(
                                    f"Agent '{agent_name}': agentic_step at index {i} missing 'objective'"
                                )
                    elif not isinstance(instruction, str):
                        errors.append(
                            f"Agent '{agent_name}': instruction at index {i} must be string or dict"
                        )

        # Validate history_config
        if "history_config" in agent_config:
            history_config = agent_config["history_config"]
            if not isinstance(history_config, dict):
                errors.append(f"Agent '{agent_name}': history_config must be an object")
            else:
                # Validate max_tokens range
                if "max_tokens" in history_config:
                    max_tokens = history_config["max_tokens"]
                    if not isinstance(max_tokens, int) or not (100 <= max_tokens <= 16000):
                        errors.append(
                            f"Agent '{agent_name}': history_config.max_tokens must be 100-16000"
                        )

                # Validate max_entries range
                if "max_entries" in history_config:
                    max_entries = history_config["max_entries"]
                    if not isinstance(max_entries, int) or not (1 <= max_entries <= 200):
                        errors.append(
                            f"Agent '{agent_name}': history_config.max_entries must be 1-200"
                        )

                # Validate truncation_strategy
                if "truncation_strategy" in history_config:
                    strategy = history_config["truncation_strategy"]
                    if strategy not in ["oldest_first", "keep_last"]:
                        errors.append(
                            f"Agent '{agent_name}': history_config.truncation_strategy must be 'oldest_first' or 'keep_last'"
                        )

        return errors

    def validate_mcp_server_config(self, server_config: Dict[str, Any]) -> List[str]:
        """Validate individual MCP server configuration.

        Args:
            server_config: MCP server configuration dictionary

        Returns:
            List[str]: List of validation error messages
        """
        errors = []

        # Required fields
        if "id" not in server_config:
            errors.append("MCP server: missing required field 'id'")

        if "type" not in server_config:
            errors.append("MCP server: missing required field 'type'")
        elif server_config["type"] not in ["stdio", "http"]:
            errors.append("MCP server: type must be 'stdio' or 'http'")

        # Type-specific validation
        if server_config.get("type") == "stdio":
            if "command" not in server_config:
                errors.append(
                    f"MCP server '{server_config.get('id', 'unknown')}': stdio type requires 'command' field"
                )

        if server_config.get("type") == "http":
            if "url" not in server_config:
                errors.append(
                    f"MCP server '{server_config.get('id', 'unknown')}': http type requires 'url' field"
                )

        return errors

    def validate_orchestration_config(self, orch_config: Dict[str, Any]) -> List[str]:
        """Validate orchestration configuration.

        Args:
            orch_config: Orchestration configuration dictionary

        Returns:
            List[str]: List of validation error messages
        """
        errors = []

        # Validate execution_mode
        if "execution_mode" in orch_config:
            mode = orch_config["execution_mode"]
            valid_modes = ["router", "pipeline", "round-robin", "broadcast"]
            if mode not in valid_modes:
                errors.append(
                    f"orchestration.execution_mode must be one of {valid_modes}, got: {mode}"
                )

            # Router mode requires router config
            if mode == "router" and "router" not in orch_config:
                errors.append(
                    "orchestration.execution_mode='router' requires 'router' configuration"
                )

        # Validate router config
        if "router" in orch_config:
            router_config = orch_config["router"]
            if not isinstance(router_config, dict):
                errors.append("orchestration.router must be an object")
            else:
                # Validate model
                if "model" in router_config and not isinstance(router_config["model"], str):
                    errors.append("orchestration.router.model must be a string")

        return errors


def validate_yaml_config(
    config_data: Dict[str, Any], schema_path: Optional[Path] = None
) -> List[str]:
    """Validate YAML configuration data against schema.

    Convenience function for quick validation.

    Args:
        config_data: Parsed YAML configuration data
        schema_path: Optional path to JSON schema file

    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    if not JSONSCHEMA_AVAILABLE:
        # Graceful degradation: perform basic validation without jsonschema
        errors = []

        # Check basic structure
        if not isinstance(config_data, dict):
            errors.append("Configuration must be an object")
            return errors

        # Validate agents
        if "agents" in config_data:
            if not isinstance(config_data["agents"], dict):
                errors.append("agents must be an object")
            else:
                validator = YAMLConfigValidator(schema_path)
                for agent_name, agent_config in config_data["agents"].items():
                    errors.extend(
                        validator.validate_agent_config(agent_name, agent_config)
                    )

        # Validate MCP servers
        if "mcp_servers" in config_data:
            if not isinstance(config_data["mcp_servers"], list):
                errors.append("mcp_servers must be an array")
            else:
                validator = YAMLConfigValidator(schema_path)
                for server_config in config_data["mcp_servers"]:
                    errors.extend(validator.validate_mcp_server_config(server_config))

        # Validate orchestration
        if "orchestration" in config_data:
            validator = YAMLConfigValidator(schema_path)
            errors.extend(
                validator.validate_orchestration_config(config_data["orchestration"])
            )

        return errors

    # Use full jsonschema validation
    validator = YAMLConfigValidator(schema_path)
    return validator.validate(config_data)
