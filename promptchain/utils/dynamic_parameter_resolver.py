"""
Dynamic Parameter Resolver for MCP Tool Hijacker
================================================

This module enables dynamic parameter passing between PromptChain steps,
allowing MCP tool outputs to be used as inputs for subsequent steps.

Key Features:
- Variable substitution from previous step outputs
- JSON path extraction (e.g., "results[0].id")
- Type conversion and validation
- Error handling for missing/invalid data
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
import logging
from copy import deepcopy


class ParameterResolutionError(Exception):
    """Raised when parameter resolution fails."""
    pass


class DynamicParameterResolver:
    """
    Resolves dynamic parameters by extracting values from previous step outputs.
    
    Supports:
    - JSON path notation: {previous.results[0].id}
    - Array indexing: {previous.data[0]}  
    - Nested object access: {previous.metadata.title}
    - Default values: {previous.missing|default_value}
    - Type conversion: {previous.count:int}
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize parameter resolver.
        
        Args:
            verbose: Enable debug logging
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        
        # Pattern to match parameter placeholders
        self.param_pattern = re.compile(r'\{([^}]+)\}')
        
        # Step outputs storage (step_index -> output)
        self.step_outputs: Dict[int, Any] = {}
        
        # Named step outputs (step_name -> output) 
        self.named_outputs: Dict[str, Any] = {}
    
    def store_step_output(self, step_index: int, output: Any, step_name: Optional[str] = None):
        """
        Store output from a step for later parameter resolution.
        
        Args:
            step_index: Index of the step (0-based)
            output: Output data from the step
            step_name: Optional name for the step
        """
        self.step_outputs[step_index] = output
        
        if step_name:
            self.named_outputs[step_name] = output
        
        if self.verbose:
            self.logger.debug(f"Stored output for step {step_index}: {type(output)} "
                             f"{'(' + step_name + ')' if step_name else ''}")
    
    def resolve_parameters(self, params: Dict[str, Any], current_step: int) -> Dict[str, Any]:
        """
        Resolve dynamic parameters using previous step outputs.
        
        Args:
            params: Parameter dictionary that may contain placeholders
            current_step: Current step index (for determining "previous")
            
        Returns:
            Parameters with resolved values
            
        Raises:
            ParameterResolutionError: If resolution fails
        """
        if not params:
            return params
        
        resolved_params = deepcopy(params)
        
        for param_name, param_value in resolved_params.items():
            if isinstance(param_value, str):
                resolved_params[param_name] = self._resolve_string_parameter(
                    param_value, current_step
                )
            elif isinstance(param_value, dict):
                resolved_params[param_name] = self._resolve_dict_parameter(
                    param_value, current_step
                )
            elif isinstance(param_value, list):
                resolved_params[param_name] = self._resolve_list_parameter(
                    param_value, current_step
                )
        
        if self.verbose:
            self.logger.debug(f"Resolved parameters: {resolved_params}")
        
        return resolved_params
    
    def _resolve_string_parameter(self, param_value: str, current_step: int) -> Any:
        """Resolve string parameter with placeholders."""
        
        # Find all placeholders in the string
        matches = self.param_pattern.findall(param_value)
        
        if not matches:
            return param_value
        
        resolved_value = param_value
        
        for match in matches:
            placeholder = f"{{{match}}}"
            resolved_val = self._extract_value_from_path(match, current_step)
            
            # If the entire string is just this placeholder, return the actual value
            if param_value == placeholder:
                return resolved_val
            
            # Otherwise, substitute in the string
            resolved_value = resolved_value.replace(placeholder, str(resolved_val))
        
        return resolved_value
    
    def _resolve_dict_parameter(self, param_dict: Dict[str, Any], current_step: int) -> Dict[str, Any]:
        """Resolve dictionary parameter recursively."""
        resolved_dict = {}
        
        for key, value in param_dict.items():
            if isinstance(value, str):
                resolved_dict[key] = self._resolve_string_parameter(value, current_step)
            elif isinstance(value, dict):
                resolved_dict[key] = self._resolve_dict_parameter(value, current_step)
            elif isinstance(value, list):
                resolved_dict[key] = self._resolve_list_parameter(value, current_step)
            else:
                resolved_dict[key] = value
        
        return resolved_dict
    
    def _resolve_list_parameter(self, param_list: List[Any], current_step: int) -> List[Any]:
        """Resolve list parameter recursively."""
        resolved_list = []
        
        for item in param_list:
            if isinstance(item, str):
                resolved_list.append(self._resolve_string_parameter(item, current_step))
            elif isinstance(item, dict):
                resolved_list.append(self._resolve_dict_parameter(item, current_step))
            elif isinstance(item, list):
                resolved_list.append(self._resolve_list_parameter(item, current_step))
            else:
                resolved_list.append(item)
        
        return resolved_list
    
    def _extract_value_from_path(self, path: str, current_step: int) -> Any:
        """
        Extract value using JSON path notation.
        
        Supported formats:
        - previous.key
        - previous.results[0].id
        - step1.data.title
        - previous.missing|default_value
        - previous.count:int
        """
        # Handle default values
        if '|' in path:
            path, default_value = path.split('|', 1)
            path = path.strip()
            default_value = default_value.strip()
        else:
            default_value = None
        
        # Handle type conversion
        if ':' in path:
            path, type_hint = path.rsplit(':', 1)
            path = path.strip()
            type_hint = type_hint.strip()
        else:
            type_hint = None
        
        # Determine source data
        if path.startswith('previous.'):
            if current_step == 0:
                if default_value is not None:
                    return self._convert_type(default_value, type_hint)
                raise ParameterResolutionError(f"No previous step available for step 0")
            
            source_data = self.step_outputs.get(current_step - 1)
            path = path[9:]  # Remove "previous."
            
        elif path.startswith('step'):
            # Extract step number: step1.data -> step_num=1, path=data
            step_match = re.match(r'step(\d+)\.(.+)', path)
            if step_match:
                step_num = int(step_match.group(1))
                path = step_match.group(2)
                source_data = self.step_outputs.get(step_num)
            else:
                raise ParameterResolutionError(f"Invalid step reference: {path}")
        
        else:
            # Check if it's a named step
            if '.' in path:
                step_name, remaining_path = path.split('.', 1)
                if step_name in self.named_outputs:
                    source_data = self.named_outputs[step_name]
                    path = remaining_path
                else:
                    raise ParameterResolutionError(f"Unknown step name: {step_name}")
            else:
                raise ParameterResolutionError(f"Invalid path format: {path}")
        
        if source_data is None:
            if default_value is not None:
                return self._convert_type(default_value, type_hint)
            raise ParameterResolutionError(f"No data available for path: {path}")
        
        # Extract value using path
        try:
            value = self._extract_nested_value(source_data, path)
            
            if value is None and default_value is not None:
                value = default_value
            
            return self._convert_type(value, type_hint)
            
        except Exception as e:
            if default_value is not None:
                return self._convert_type(default_value, type_hint)
            raise ParameterResolutionError(f"Failed to extract value from path '{path}': {e}")
    
    def _extract_nested_value(self, data: Any, path: str) -> Any:
        """Extract nested value using dot notation and array indexing."""
        
        if not path:
            return data
        
        # Handle array indexing: results[0] or results[0].title
        if '[' in path:
            # Split on first bracket
            before_bracket = path.split('[', 1)[0]
            bracket_content = path.split('[', 1)[1]
            
            if ']' not in bracket_content:
                raise ValueError(f"Invalid array syntax in path: {path}")
            
            index_str = bracket_content.split(']', 1)[0]
            after_bracket = bracket_content.split(']', 1)[1]
            
            # Get the array
            if before_bracket:
                array_data = self._extract_nested_value(data, before_bracket)
            else:
                array_data = data
            
            if not isinstance(array_data, (list, tuple)):
                raise ValueError(f"Expected array for indexing, got {type(array_data)}")
            
            # Get array index
            try:
                index = int(index_str)
                if index < 0:
                    index = len(array_data) + index  # Support negative indexing
                
                if index >= len(array_data):
                    raise IndexError(f"Index {index} out of range for array of length {len(array_data)}")
                
                indexed_value = array_data[index]
                
            except ValueError:
                raise ValueError(f"Invalid array index: {index_str}")
            
            # Continue with remaining path
            if after_bracket.startswith('.'):
                after_bracket = after_bracket[1:]  # Remove leading dot
            
            if after_bracket:
                return self._extract_nested_value(indexed_value, after_bracket)
            else:
                return indexed_value
        
        # Handle simple dot notation
        if '.' in path:
            key, remaining = path.split('.', 1)
        else:
            key, remaining = path, ""
        
        # Extract key
        if isinstance(data, dict):
            value = data.get(key)
        else:
            # Try to get attribute
            try:
                value = getattr(data, key)
            except AttributeError:
                raise ValueError(f"Key '{key}' not found in {type(data)}")
        
        if remaining:
            return self._extract_nested_value(value, remaining)
        else:
            return value
    
    def _convert_type(self, value: Any, type_hint: Optional[str]) -> Any:
        """Convert value to specified type."""
        if type_hint is None:
            return value
        
        type_hint = type_hint.lower()
        
        try:
            if type_hint == 'int':
                return int(value)
            elif type_hint == 'float':
                return float(value)
            elif type_hint == 'bool':
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif type_hint == 'str':
                return str(value)
            elif type_hint == 'json':
                if isinstance(value, str):
                    return json.loads(value)
                return value
            else:
                self.logger.warning(f"Unknown type hint: {type_hint}")
                return value
        
        except Exception as e:
            self.logger.warning(f"Type conversion failed for {value} to {type_hint}: {e}")
            return value
    
    def clear_outputs(self):
        """Clear all stored step outputs."""
        self.step_outputs.clear()
        self.named_outputs.clear()
    
    def get_available_variables(self) -> Dict[str, Any]:
        """Get summary of available variables for debugging."""
        return {
            'step_outputs': {k: type(v).__name__ for k, v in self.step_outputs.items()},
            'named_outputs': {k: type(v).__name__ for k, v in self.named_outputs.items()}
        }