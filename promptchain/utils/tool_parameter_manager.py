"""
Tool Parameter Manager for PromptChain Tool Hijacker

This module provides comprehensive parameter management capabilities including
static parameter storage, dynamic parameter merging, transformation functions,
and validation logic for MCP tool execution.
"""

import logging
from typing import Dict, Any, Callable, Optional, Union, List
from copy import deepcopy
import json


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""
    pass


class ParameterTransformationError(Exception):
    """Raised when parameter transformation fails."""
    pass


class ToolParameterManager:
    """
    Manages static and dynamic parameters for MCP tools.
    
    This class provides:
    - Static parameter storage and management
    - Parameter merging (static + dynamic)
    - Parameter transformation functions
    - Parameter validation
    - Default value handling
    - Parameter templates and substitution
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Tool Parameter Manager.
        
        Args:
            verbose: Enable debug output
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        # Parameter storage
        self.static_params: Dict[str, Dict[str, Any]] = {}  # tool_name -> {param: value}
        self.default_params: Dict[str, Dict[str, Any]] = {}  # tool_name -> {param: default_value}
        self.required_params: Dict[str, List[str]] = {}  # tool_name -> [required_param_names]
        
        # Transformation functions
        self.transformers: Dict[str, Dict[str, Callable]] = {}  # tool_name -> {param: transformer_func}
        self.validators: Dict[str, Dict[str, Callable]] = {}  # tool_name -> {param: validator_func}
        
        # Parameter templates
        self.templates: Dict[str, Dict[str, str]] = {}  # tool_name -> {param: template_string}
        
        # Global settings
        self.global_transformers: Dict[str, Callable] = {}  # param_name -> transformer (applies to all tools)
        self.global_validators: Dict[str, Callable] = {}  # param_name -> validator (applies to all tools)
    
    def set_static_params(self, tool_name: str, **params) -> None:
        """
        Set static parameters for a tool.
        
        Static parameters are applied to every call of the tool unless explicitly overridden.
        
        Args:
            tool_name: Name of the tool
            **params: Parameter key-value pairs
        """
        if tool_name not in self.static_params:
            self.static_params[tool_name] = {}
        
        self.static_params[tool_name].update(params)
        
        if self.verbose:
            self.logger.debug(f"Set static params for {tool_name}: {params}")
    
    def get_static_params(self, tool_name: str) -> Dict[str, Any]:
        """
        Get static parameters for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of static parameters
        """
        return self.static_params.get(tool_name, {}).copy()
    
    def remove_static_param(self, tool_name: str, param_name: str) -> bool:
        """
        Remove a static parameter for a tool.
        
        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter to remove
            
        Returns:
            True if parameter was removed, False if it didn't exist
        """
        if tool_name in self.static_params and param_name in self.static_params[tool_name]:
            del self.static_params[tool_name][param_name]
            
            if self.verbose:
                self.logger.debug(f"Removed static param {param_name} from {tool_name}")
            
            return True
        return False
    
    def clear_static_params(self, tool_name: str) -> None:
        """
        Clear all static parameters for a tool.
        
        Args:
            tool_name: Name of the tool
        """
        if tool_name in self.static_params:
            del self.static_params[tool_name]
            
            if self.verbose:
                self.logger.debug(f"Cleared all static params for {tool_name}")
    
    def set_default_params(self, tool_name: str, **params) -> None:
        """
        Set default parameters for a tool.
        
        Default parameters are used when no static or dynamic value is provided.
        
        Args:
            tool_name: Name of the tool
            **params: Parameter key-value pairs
        """
        if tool_name not in self.default_params:
            self.default_params[tool_name] = {}
        
        self.default_params[tool_name].update(params)
        
        if self.verbose:
            self.logger.debug(f"Set default params for {tool_name}: {params}")
    
    def set_required_params(self, tool_name: str, param_names: List[str]) -> None:
        """
        Set required parameters for a tool.
        
        Args:
            tool_name: Name of the tool
            param_names: List of required parameter names
        """
        self.required_params[tool_name] = param_names.copy()
        
        if self.verbose:
            self.logger.debug(f"Set required params for {tool_name}: {param_names}")
    
    def add_transformer(self, tool_name: str, param_name: str, transformer: Callable) -> None:
        """
        Add parameter transformation function for a specific tool and parameter.
        
        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter
            transformer: Function that transforms the parameter value
        """
        if tool_name not in self.transformers:
            self.transformers[tool_name] = {}
        
        self.transformers[tool_name][param_name] = transformer
        
        if self.verbose:
            self.logger.debug(f"Added transformer for {tool_name}.{param_name}")
    
    def add_global_transformer(self, param_name: str, transformer: Callable) -> None:
        """
        Add global parameter transformation function.
        
        Global transformers are applied to all tools that have the specified parameter.
        
        Args:
            param_name: Name of the parameter
            transformer: Function that transforms the parameter value
        """
        self.global_transformers[param_name] = transformer
        
        if self.verbose:
            self.logger.debug(f"Added global transformer for parameter: {param_name}")
    
    def add_validator(self, tool_name: str, param_name: str, validator: Callable) -> None:
        """
        Add parameter validation function for a specific tool and parameter.
        
        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter
            validator: Function that validates the parameter value (returns bool)
        """
        if tool_name not in self.validators:
            self.validators[tool_name] = {}
        
        self.validators[tool_name][param_name] = validator
        
        if self.verbose:
            self.logger.debug(f"Added validator for {tool_name}.{param_name}")
    
    def add_global_validator(self, param_name: str, validator: Callable) -> None:
        """
        Add global parameter validation function.
        
        Args:
            param_name: Name of the parameter
            validator: Function that validates the parameter value (returns bool)
        """
        self.global_validators[param_name] = validator
        
        if self.verbose:
            self.logger.debug(f"Added global validator for parameter: {param_name}")
    
    def set_parameter_template(self, tool_name: str, param_name: str, template: str) -> None:
        """
        Set parameter template for dynamic value substitution.
        
        Templates support variable substitution using {variable_name} syntax.
        
        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter
            template: Template string with {variable} placeholders
        """
        if tool_name not in self.templates:
            self.templates[tool_name] = {}
        
        self.templates[tool_name][param_name] = template
        
        if self.verbose:
            self.logger.debug(f"Set template for {tool_name}.{param_name}: {template}")
    
    def merge_params(self, tool_name: str, **dynamic_params) -> Dict[str, Any]:
        """
        Merge static, default, and dynamic parameters for a tool.
        
        Priority order (highest to lowest):
        1. Dynamic parameters (provided at call time)
        2. Static parameters (configured for the tool)
        3. Default parameters (fallback values)
        
        Args:
            tool_name: Name of the tool
            **dynamic_params: Dynamic parameters provided at call time
            
        Returns:
            Merged parameter dictionary
        """
        # Start with defaults
        merged = deepcopy(self.default_params.get(tool_name, {}))
        
        # Add static parameters (override defaults)
        static = self.static_params.get(tool_name, {})
        merged.update(static)
        
        # Add dynamic parameters (override static)
        merged.update(dynamic_params)
        
        if self.verbose:
            self.logger.debug(f"Merged params for {tool_name}: defaults={len(self.default_params.get(tool_name, {}))}, "
                             f"static={len(static)}, dynamic={len(dynamic_params)}, final={len(merged)}")
        
        return merged
    
    def apply_templates(self, tool_name: str, params: Dict[str, Any], 
                       template_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply parameter templates to substitute dynamic values.
        
        Args:
            tool_name: Name of the tool
            params: Parameter dictionary
            template_vars: Variables available for template substitution
            
        Returns:
            Parameters with templates applied
        """
        if tool_name not in self.templates:
            return params
        
        template_vars = template_vars or {}
        result = params.copy()
        
        for param_name, template in self.templates[tool_name].items():
            if param_name in result:
                try:
                    # Apply template substitution
                    result[param_name] = template.format(**template_vars)
                    
                    if self.verbose:
                        self.logger.debug(f"Applied template to {tool_name}.{param_name}")
                        
                except KeyError as e:
                    self.logger.warning(f"Template variable not found: {e}")
                except Exception as e:
                    self.logger.error(f"Template application failed: {e}")
        
        return result
    
    def transform_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply parameter transformations.
        
        Args:
            tool_name: Name of the tool
            params: Parameter dictionary
            
        Returns:
            Transformed parameters
            
        Raises:
            ParameterTransformationError: If transformation fails
        """
        result = params.copy()
        
        # Apply tool-specific transformers
        tool_transformers = self.transformers.get(tool_name, {})
        for param_name, transformer in tool_transformers.items():
            if param_name in result:
                try:
                    original_value = result[param_name]
                    result[param_name] = transformer(original_value)
                    
                    if self.verbose:
                        self.logger.debug(f"Transformed {tool_name}.{param_name}: {original_value} -> {result[param_name]}")
                        
                except Exception as e:
                    raise ParameterTransformationError(
                        f"Transformation failed for {tool_name}.{param_name}: {e}"
                    )
        
        # Apply global transformers
        for param_name, transformer in self.global_transformers.items():
            if param_name in result:
                try:
                    original_value = result[param_name]
                    result[param_name] = transformer(original_value)
                    
                    if self.verbose:
                        self.logger.debug(f"Applied global transform to {param_name}: {original_value} -> {result[param_name]}")
                        
                except Exception as e:
                    raise ParameterTransformationError(
                        f"Global transformation failed for {param_name}: {e}"
                    )
        
        return result
    
    def validate_params(self, tool_name: str, params: Dict[str, Any]) -> None:
        """
        Validate parameters against configured validators and requirements.
        
        Args:
            tool_name: Name of the tool
            params: Parameter dictionary
            
        Raises:
            ParameterValidationError: If validation fails
        """
        # Check required parameters
        required = self.required_params.get(tool_name, [])
        missing_params = [param for param in required if param not in params]
        
        if missing_params:
            raise ParameterValidationError(
                f"Missing required parameters for {tool_name}: {missing_params}"
            )
        
        # Apply tool-specific validators
        tool_validators = self.validators.get(tool_name, {})
        for param_name, validator in tool_validators.items():
            if param_name in params:
                try:
                    if not validator(params[param_name]):
                        raise ParameterValidationError(
                            f"Validation failed for {tool_name}.{param_name}: {params[param_name]}"
                        )
                        
                    if self.verbose:
                        self.logger.debug(f"Validated {tool_name}.{param_name}")
                        
                except Exception as e:
                    if isinstance(e, ParameterValidationError):
                        raise
                    raise ParameterValidationError(
                        f"Validator error for {tool_name}.{param_name}: {e}"
                    )
        
        # Apply global validators
        for param_name, validator in self.global_validators.items():
            if param_name in params:
                try:
                    if not validator(params[param_name]):
                        raise ParameterValidationError(
                            f"Global validation failed for {param_name}: {params[param_name]}"
                        )
                        
                    if self.verbose:
                        self.logger.debug(f"Applied global validation to {param_name}")
                        
                except Exception as e:
                    if isinstance(e, ParameterValidationError):
                        raise
                    raise ParameterValidationError(
                        f"Global validator error for {param_name}: {e}"
                    )
    
    def process_params(self, tool_name: str, template_vars: Optional[Dict[str, Any]] = None, 
                      **dynamic_params) -> Dict[str, Any]:
        """
        Complete parameter processing pipeline.
        
        This method:
        1. Merges static, default, and dynamic parameters
        2. Applies templates
        3. Transforms parameters
        4. Validates parameters
        
        Args:
            tool_name: Name of the tool
            template_vars: Variables for template substitution
            **dynamic_params: Dynamic parameters
            
        Returns:
            Fully processed parameters
            
        Raises:
            ParameterValidationError: If validation fails
            ParameterTransformationError: If transformation fails
        """
        # Step 1: Merge parameters
        merged = self.merge_params(tool_name, **dynamic_params)
        
        # Step 2: Apply templates
        if template_vars:
            merged = self.apply_templates(tool_name, merged, template_vars)
        
        # Step 3: Transform parameters
        transformed = self.transform_params(tool_name, merged)
        
        # Step 4: Validate parameters
        self.validate_params(tool_name, transformed)
        
        return transformed
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool configuration dictionary
        """
        return {
            "static_params": self.static_params.get(tool_name, {}),
            "default_params": self.default_params.get(tool_name, {}),
            "required_params": self.required_params.get(tool_name, []),
            "transformers": list(self.transformers.get(tool_name, {}).keys()),
            "validators": list(self.validators.get(tool_name, {}).keys()),
            "templates": self.templates.get(tool_name, {})
        }
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export complete parameter manager configuration.
        
        Note: Transformer and validator functions are not serializable,
        only their presence is indicated.
        
        Returns:
            Configuration dictionary
        """
        return {
            "static_params": deepcopy(self.static_params),
            "default_params": deepcopy(self.default_params),
            "required_params": deepcopy(self.required_params),
            "templates": deepcopy(self.templates),
            "tool_transformers": {
                tool: list(transformers.keys()) 
                for tool, transformers in self.transformers.items()
            },
            "tool_validators": {
                tool: list(validators.keys())
                for tool, validators in self.validators.items()
            },
            "global_transformers": list(self.global_transformers.keys()),
            "global_validators": list(self.global_validators.keys())
        }


# Common parameter transformers
class CommonTransformers:
    """Collection of commonly used parameter transformers."""
    
    @staticmethod
    def clamp_float(min_val: float = 0.0, max_val: float = 1.0) -> Callable:
        """Create a transformer that clamps float values to a range."""
        def transformer(value):
            return max(min_val, min(max_val, float(value)))
        return transformer
    
    @staticmethod
    def clamp_int(min_val: int = 0, max_val: int = 100) -> Callable:
        """Create a transformer that clamps integer values to a range."""
        def transformer(value):
            return max(min_val, min(max_val, int(value)))
        return transformer
    
    @staticmethod
    def to_string() -> Callable:
        """Transformer that converts values to strings."""
        return lambda value: str(value)
    
    @staticmethod
    def to_lowercase() -> Callable:
        """Transformer that converts strings to lowercase."""
        return lambda value: str(value).lower()
    
    @staticmethod
    def to_uppercase() -> Callable:
        """Transformer that converts strings to uppercase."""
        return lambda value: str(value).upper()
    
    @staticmethod
    def strip_whitespace() -> Callable:
        """Transformer that strips whitespace from strings."""
        return lambda value: str(value).strip()
    
    @staticmethod
    def truncate_string(max_length: int) -> Callable:
        """Create a transformer that truncates strings to max length."""
        def transformer(value):
            s = str(value)
            return s[:max_length] if len(s) > max_length else s
        return transformer


# Common parameter validators
class CommonValidators:
    """Collection of commonly used parameter validators."""
    
    @staticmethod
    def is_float_in_range(min_val: float = 0.0, max_val: float = 1.0) -> Callable:
        """Create a validator for float values in a range."""
        def validator(value):
            try:
                f_val = float(value)
                return min_val <= f_val <= max_val
            except (ValueError, TypeError):
                return False
        return validator
    
    @staticmethod
    def is_int_in_range(min_val: int = 0, max_val: int = 100) -> Callable:
        """Create a validator for integer values in a range."""
        def validator(value):
            try:
                i_val = int(value)
                return min_val <= i_val <= max_val
            except (ValueError, TypeError):
                return False
        return validator
    
    @staticmethod
    def is_non_empty_string() -> Callable:
        """Validator for non-empty strings."""
        def validator(value):
            return isinstance(value, str) and len(value.strip()) > 0
        return validator
    
    @staticmethod
    def is_string_max_length(max_length: int) -> Callable:
        """Create a validator for string maximum length."""
        def validator(value):
            try:
                return len(str(value)) <= max_length
            except (ValueError, TypeError):
                return False
        return validator
    
    @staticmethod
    def is_in_choices(choices: List[Any]) -> Callable:
        """Create a validator for value in allowed choices."""
        def validator(value):
            return value in choices
        return validator
    
    @staticmethod
    def matches_pattern(pattern: str) -> Callable:
        """Create a validator for regex pattern matching."""
        import re
        compiled_pattern = re.compile(pattern)
        
        def validator(value):
            try:
                return compiled_pattern.match(str(value)) is not None
            except (ValueError, TypeError):
                return False
        return validator