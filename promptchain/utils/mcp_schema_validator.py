"""
MCP Schema Validator for PromptChain Tool Hijacker

This module provides JSON Schema validation capabilities for MCP tool parameters,
ensuring parameter correctness and type safety for direct tool execution.
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


class MCPSchemaValidator:
    """
    JSON Schema validator for MCP tool parameters.

    This class provides:
    - JSON Schema validation for tool parameters
    - Type checking and conversion
    - Required parameter validation
    - Custom validation rules
    - Schema introspection and analysis
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize MCP Schema Validator.

        Args:
            verbose: Enable debug output
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        if verbose:
            self.logger.setLevel(logging.DEBUG)

        # Try to import jsonschema for advanced validation
        self._jsonschema_available = False
        try:
            import jsonschema
            from jsonschema import Draft7Validator, ValidationError, validate

            self._jsonschema = jsonschema
            self._validate = validate
            self._ValidationError = ValidationError
            self._Draft7Validator = Draft7Validator
            self._jsonschema_available = True

            if verbose:
                self.logger.debug(
                    "jsonschema library available for advanced validation"
                )

        except ImportError:
            if verbose:
                self.logger.warning(
                    "jsonschema library not available, using basic validation"
                )

        # Schema cache for performance
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._compiled_validators: Dict[str, Any] = {}

    def validate_parameters(
        self,
        tool_name: str,
        schema: Dict[str, Any],
        parameters: Dict[str, Any],
        strict: bool = True,
    ) -> None:
        """
        Validate parameters against a JSON schema.

        Args:
            tool_name: Name of the tool (for error messages)
            schema: JSON schema for the tool's parameters
            parameters: Parameters to validate
            strict: Whether to enforce strict validation

        Raises:
            SchemaValidationError: If validation fails
        """
        if not schema:
            if self.verbose:
                self.logger.debug(
                    f"No schema provided for {tool_name}, skipping validation"
                )
            return

        try:
            # Extract the parameters schema from the tool schema
            param_schema = self._extract_parameter_schema(schema)

            if not param_schema:
                if self.verbose:
                    self.logger.debug(f"No parameter schema found for {tool_name}")
                return

            # Use advanced validation if available
            if self._jsonschema_available:
                self._validate_with_jsonschema(tool_name, param_schema, parameters)
            else:
                self._validate_basic(tool_name, param_schema, parameters, strict)

            if self.verbose:
                self.logger.debug(f"Parameters validated successfully for {tool_name}")

        except SchemaValidationError:
            raise
        except Exception as e:
            raise SchemaValidationError(f"Validation error for {tool_name}: {e}")

    def _extract_parameter_schema(
        self, tool_schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Extract parameter schema from tool schema.

        Args:
            tool_schema: Complete tool schema

        Returns:
            Parameter schema or None if not found
        """
        # Handle different schema formats
        if "function" in tool_schema and "parameters" in tool_schema["function"]:
            return tool_schema["function"]["parameters"]
        elif "parameters" in tool_schema:
            return tool_schema["parameters"]
        elif "inputSchema" in tool_schema:
            return tool_schema["inputSchema"]

        return None

    def _validate_with_jsonschema(
        self, tool_name: str, param_schema: Dict[str, Any], parameters: Dict[str, Any]
    ) -> None:
        """
        Validate using the jsonschema library.

        Args:
            tool_name: Name of the tool
            param_schema: Parameter schema
            parameters: Parameters to validate

        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            # Use compiled validator for performance if available
            validator_key = (
                f"{tool_name}_{hash(json.dumps(param_schema, sort_keys=True))}"
            )

            if validator_key not in self._compiled_validators:
                self._compiled_validators[validator_key] = self._Draft7Validator(
                    param_schema
                )

            validator = self._compiled_validators[validator_key]

            # Validate parameters
            errors = sorted(validator.iter_errors(parameters), key=lambda e: e.path)

            if errors:
                error_messages = []
                for error in errors:
                    path = (
                        ".".join(str(p) for p in error.path) if error.path else "root"
                    )
                    error_messages.append(f"{path}: {error.message}")

                raise SchemaValidationError(
                    f"Schema validation failed for {tool_name}:\n"
                    + "\n".join(error_messages)
                )

        except self._ValidationError as e:
            path = ".".join(str(p) for p in e.path) if e.path else "root"
            raise SchemaValidationError(
                f"Schema validation failed for {tool_name} at {path}: {e.message}"
            )

    def _validate_basic(
        self,
        tool_name: str,
        param_schema: Dict[str, Any],
        parameters: Dict[str, Any],
        strict: bool,
    ) -> None:
        """
        Basic validation without jsonschema library.

        Args:
            tool_name: Name of the tool
            param_schema: Parameter schema
            parameters: Parameters to validate
            strict: Whether to enforce strict validation

        Raises:
            SchemaValidationError: If validation fails
        """
        schema_type = param_schema.get("type", "object")

        if schema_type == "object":
            self._validate_object(tool_name, param_schema, parameters, strict)
        elif schema_type == "array":
            self._validate_array(tool_name, param_schema, parameters, strict)
        else:
            # Simple type validation
            self._validate_simple_type(tool_name, schema_type, parameters, strict)

    def _validate_object(
        self,
        tool_name: str,
        schema: Dict[str, Any],
        parameters: Dict[str, Any],
        strict: bool,
    ) -> None:
        """Validate object parameters."""
        if not isinstance(parameters, dict):
            raise SchemaValidationError(
                f"Parameters for {tool_name} must be an object, got {type(parameters)}"
            )

        # Check required properties
        required = schema.get("required", [])
        missing = [prop for prop in required if prop not in parameters]
        if missing:
            raise SchemaValidationError(
                f"Missing required parameters for {tool_name}: {missing}"
            )

        # Validate properties
        properties = schema.get("properties", {})

        for prop_name, prop_value in parameters.items():
            if prop_name in properties:
                prop_schema = properties[prop_name]
                try:
                    self._validate_property(
                        tool_name, prop_name, prop_schema, prop_value, strict
                    )
                except SchemaValidationError as e:
                    raise SchemaValidationError(
                        f"Parameter {prop_name} in {tool_name}: {e}"
                    )
            elif strict:
                # Additional properties not allowed in strict mode
                additional_properties = schema.get("additionalProperties", True)
                if not additional_properties:
                    raise SchemaValidationError(
                        f"Additional property '{prop_name}' not allowed for {tool_name}"
                    )

    def _validate_array(
        self, tool_name: str, schema: Dict[str, Any], parameters: Any, strict: bool
    ) -> None:
        """Validate array parameters."""
        if not isinstance(parameters, list):
            raise SchemaValidationError(
                f"Parameters for {tool_name} must be an array, got {type(parameters)}"
            )

        # Validate array length
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")

        if min_items is not None and len(parameters) < min_items:
            raise SchemaValidationError(
                f"Array for {tool_name} must have at least {min_items} items"
            )

        if max_items is not None and len(parameters) > max_items:
            raise SchemaValidationError(
                f"Array for {tool_name} must have at most {max_items} items"
            )

        # Validate items
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(parameters):
                try:
                    self._validate_property(
                        tool_name, f"[{i}]", items_schema, item, strict
                    )
                except SchemaValidationError as e:
                    raise SchemaValidationError(f"Array item {i} in {tool_name}: {e}")

    def _validate_property(
        self,
        tool_name: str,
        prop_name: str,
        schema: Dict[str, Any],
        value: Any,
        strict: bool,
    ) -> None:
        """Validate a single property."""
        expected_type = schema.get("type")

        if expected_type:
            if not self._check_type(value, expected_type):
                raise SchemaValidationError(
                    f"Property {prop_name} must be of type {expected_type}, got {type(value).__name__}"
                )

        # Validate enum values
        enum_values = schema.get("enum")
        if enum_values and value not in enum_values:
            raise SchemaValidationError(
                f"Property {prop_name} must be one of {enum_values}, got {value}"
            )

        # Type-specific validations
        if expected_type == "string":
            self._validate_string(prop_name, schema, value)
        elif expected_type == "number" or expected_type == "integer":
            self._validate_number(prop_name, schema, value)
        elif expected_type == "array":
            self._validate_array(tool_name, schema, value, strict)
        elif expected_type == "object":
            self._validate_object(tool_name, schema, value, strict)

    def _validate_string(
        self, prop_name: str, schema: Dict[str, Any], value: str
    ) -> None:
        """Validate string properties."""
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")

        if min_length is not None and len(value) < min_length:
            raise SchemaValidationError(
                f"String {prop_name} must be at least {min_length} characters"
            )

        if max_length is not None and len(value) > max_length:
            raise SchemaValidationError(
                f"String {prop_name} must be at most {max_length} characters"
            )

        if pattern:
            import re

            if not re.match(pattern, value):
                raise SchemaValidationError(
                    f"String {prop_name} does not match pattern {pattern}"
                )

    def _validate_number(
        self, prop_name: str, schema: Dict[str, Any], value: Union[int, float]
    ) -> None:
        """Validate number properties."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        multiple_of = schema.get("multipleOf")

        if minimum is not None and value < minimum:
            raise SchemaValidationError(f"Number {prop_name} must be >= {minimum}")

        if maximum is not None and value > maximum:
            raise SchemaValidationError(f"Number {prop_name} must be <= {maximum}")

        if exclusive_minimum is not None and value <= exclusive_minimum:
            raise SchemaValidationError(
                f"Number {prop_name} must be > {exclusive_minimum}"
            )

        if exclusive_maximum is not None and value >= exclusive_maximum:
            raise SchemaValidationError(
                f"Number {prop_name} must be < {exclusive_maximum}"
            )

        if multiple_of is not None and value % multiple_of != 0:
            raise SchemaValidationError(
                f"Number {prop_name} must be a multiple of {multiple_of}"
            )

    def _validate_simple_type(
        self, tool_name: str, expected_type: str, value: Any, strict: bool
    ) -> None:
        """Validate simple type."""
        if not self._check_type(value, expected_type):
            raise SchemaValidationError(
                f"Value for {tool_name} must be of type {expected_type}, got {type(value).__name__}"
            )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow it

        return isinstance(value, expected_python_type)  # type: ignore[arg-type]

    def get_schema_info(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information from a schema.

        Args:
            schema: JSON schema

        Returns:
            Schema information dictionary
        """
        param_schema = self._extract_parameter_schema(schema)

        if not param_schema:
            return {"has_schema": False}

        info = {
            "has_schema": True,
            "type": param_schema.get("type", "unknown"),
            "required_params": param_schema.get("required", []),
            "optional_params": [],
            "properties": {},
        }

        # Extract property information
        properties = param_schema.get("properties", {})
        required = set(param_schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            info["properties"][prop_name] = {
                "type": prop_schema.get("type", "unknown"),
                "required": prop_name in required,
                "description": prop_schema.get("description", ""),
                "enum": prop_schema.get("enum"),
                "default": prop_schema.get("default"),
            }

            if prop_name not in required:
                info["optional_params"].append(prop_name)

        return info

    def suggest_parameter_fixes(
        self, tool_name: str, schema: Dict[str, Any], parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Suggest fixes for parameter validation errors.

        Args:
            tool_name: Name of the tool
            schema: Tool schema
            parameters: Parameters that failed validation

        Returns:
            List of suggested fixes
        """
        suggestions: List[str] = []

        try:
            self.validate_parameters(tool_name, schema, parameters)
            return suggestions  # No errors, no suggestions needed
        except SchemaValidationError:
            pass  # Expected, we want to analyze the errors

        param_schema = self._extract_parameter_schema(schema)
        if not param_schema:
            return suggestions

        # Check for missing required parameters
        required = param_schema.get("required", [])
        missing = [prop for prop in required if prop not in parameters]

        for missing_prop in missing:
            suggestions.append(f"Add required parameter: {missing_prop}")

        # Check for type mismatches
        properties = param_schema.get("properties", {})

        for prop_name, prop_value in parameters.items():
            if prop_name in properties:
                prop_schema = properties[prop_name]
                expected_type = prop_schema.get("type")

                if expected_type and not self._check_type(prop_value, expected_type):
                    suggestions.append(
                        f"Convert {prop_name} to {expected_type} (currently {type(prop_value).__name__})"
                    )

        # Check for additional properties
        additional_properties = param_schema.get("additionalProperties", True)
        if not additional_properties:
            extra_props = [prop for prop in parameters.keys() if prop not in properties]
            for extra_prop in extra_props:
                suggestions.append(f"Remove unexpected parameter: {extra_prop}")

        return suggestions

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._schema_cache.clear()
        self._compiled_validators.clear()

        if self.verbose:
            self.logger.debug("Schema validation cache cleared")
