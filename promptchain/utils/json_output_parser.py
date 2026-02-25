"""
JSON Output Parser for MCP Tool Hijacker
========================================

This module provides utilities to extract specific values from MCP tool JSON outputs,
enabling seamless step-to-step parameter passing in PromptChain workflows.

Key Features:
- Extract values using JSON path notation (e.g., "results[0].id")
- Handle nested objects and arrays
- Provide default values for missing keys
- Type conversion utilities
- Common extraction patterns for MCP responses
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
import logging


class JSONParseError(Exception):
    """Raised when JSON parsing fails."""
    pass


logger = logging.getLogger(__name__)

# Sentinel used to distinguish "caller passed no default" from passing None.
_SENTINEL = object()


class JSONOutputParser:
    """
    Parse and extract values from MCP tool JSON outputs.

    Supports:
    - Simple JSON text parsing: extract(text) → parsed object or self.default
    - JSON path extraction: extract(data, path="results[0].id")
    - Default values: extract(data, path="missing_key", default="fallback")
    - Type conversion: extract(data, path="count", convert_type=int)
    - Batch extraction: extract_multiple({"id": "results[0].id", ...})

    The ``default`` parameter supplied at construction time is used as the
    fallback whenever ``extract()`` is called without an explicit per-call
    default AND whenever an unexpected exception occurs.  ``extract()`` is
    guaranteed never to propagate an exception to its caller.
    """

    def __init__(self, default: Any = None, verbose: bool = False):
        """
        Initialize JSON output parser.

        Args:
            default: Value returned when extraction fails and no per-call
                     default is provided.  Defaults to ``None``.
            verbose: Enable debug logging.
        """
        self.default = default
        self.verbose = verbose
        self.logger = logger
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def extract(self, data: Any, path: str = "", default: Any = _SENTINEL,
                convert_type: Optional[type] = None) -> Any:
        """
        Extract value from JSON data using path notation.

        This method is guaranteed to **never** raise an exception.  When
        anything goes wrong the per-call ``default`` is returned if supplied,
        otherwise ``self.default`` (set at construction time) is returned and
        a WARNING is emitted.

        Args:
            data: JSON data (dict, list, or raw string to parse).  When
                  ``path`` is empty the parsed ``data`` itself is returned.
            path: Dot/bracket path to extract
                  (e.g. ``"results[0].metadata.title"``).  Pass an empty
                  string (the default) to return the fully-parsed ``data``.
            default: Override the instance-level default for this call only.
            convert_type: Type to convert the result to (e.g. ``int``).

        Returns:
            Extracted value, or the applicable default on any failure.
        """
        # Resolve which default applies for this call.
        call_default = self.default if default is _SENTINEL else default

        try:
            # Parse JSON string if needed.
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    if call_default is not None:
                        return call_default
                    raise JSONParseError(f"Invalid JSON string: {data!r}")

            # When no path is requested, return the parsed data directly.
            if not path:
                return data

            # Extract value using path.
            value = self._extract_nested_value(data, path)

            if value is None and call_default is not None:
                value = call_default

            # Convert type if requested.
            if convert_type and value is not None:
                value = self._convert_type(value, convert_type)

            if self.verbose:
                self.logger.debug(f"Extracted '{path}': {value}")

            return value

        except Exception as e:  # catch ALL exceptions — never propagate
            self.logger.warning(
                f"JSONOutputParser.extract() failed on input: {data!r}. "
                f"Error: {e}. Returning default: {call_default!r}"
            )
            return call_default
    
    def extract_multiple(self, data: Any, extractions: Dict[str, str], 
                        defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract multiple values in one operation.
        
        Args:
            data: JSON data to extract from
            extractions: Map of output_key -> json_path
            defaults: Default values for each key
            
        Returns:
            Dictionary with extracted values
        """
        defaults = defaults or {}
        results = {}
        
        for output_key, json_path in extractions.items():
            default_val = defaults.get(output_key)
            try:
                results[output_key] = self.extract(data, json_path, default=default_val)
            except JSONParseError:
                if default_val is not None:
                    results[output_key] = default_val
                else:
                    # Skip missing values without defaults
                    continue
        
        return results
    
    def _extract_nested_value(self, data: Any, path: str) -> Any:
        """Extract value using dot notation and array indexing."""
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
    
    def _convert_type(self, value: Any, target_type: type) -> Any:
        """Convert value to target type."""
        try:
            if target_type == bool and isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return target_type(value)
        except Exception as e:
            self.logger.warning(f"Type conversion failed for {value} to {target_type}: {e}")
            return value


class CommonExtractions:
    """Common extraction patterns for MCP responses."""
    
    @staticmethod
    def deeplake_results(data: Any) -> List[Dict[str, Any]]:
        """Extract results array from DeepLake response."""
        parser = JSONOutputParser()
        return parser.extract(data, "results", default=[])
    
    @staticmethod
    def deeplake_first_result(data: Any) -> Optional[Dict[str, Any]]:
        """Extract first result from DeepLake response."""
        parser = JSONOutputParser()
        return parser.extract(data, "results[0]", default=None)
    
    @staticmethod
    def deeplake_document_ids(data: Any) -> List[str]:
        """Extract all document IDs from DeepLake results."""
        parser = JSONOutputParser()
        results = parser.extract(data, "results", default=[])
        return [result.get("id") for result in results if result.get("id")]
    
    @staticmethod
    def deeplake_document_texts(data: Any) -> List[str]:
        """Extract all document texts from DeepLake results."""
        parser = JSONOutputParser()
        results = parser.extract(data, "results", default=[])
        return [result.get("text", "") for result in results]
    
    @staticmethod
    def deeplake_titles(data: Any) -> List[str]:
        """Extract all document titles from DeepLake results."""
        parser = JSONOutputParser()
        results = parser.extract(data, "results", default=[])
        titles = []
        for result in results:
            metadata = result.get("metadata", {})
            title = metadata.get("title", "Unknown Title")
            titles.append(title)
        return titles
    
    @staticmethod
    def create_template_vars(data: Any, prefix: str = "previous") -> Dict[str, Any]:
        """
        Create template variables from MCP output for step chaining.
        
        Args:
            data: MCP tool output (usually JSON)
            prefix: Prefix for template variables (e.g., "previous", "step1")
            
        Returns:
            Dictionary suitable for use as template_vars
        """
        if not isinstance(data, dict):
            return {prefix: data}
        
        template_vars: Dict[str, Any] = {prefix: data}
        
        # Add common shortcuts for DeepLake responses
        if "results" in data and isinstance(data["results"], list):
            template_vars[f"{prefix}_results"] = data["results"]
            
            if data["results"]:
                first_result = data["results"][0]
                template_vars[f"{prefix}_first"] = first_result
                
                # Add shortcuts for common fields
                if "id" in first_result:
                    template_vars[f"{prefix}_first_id"] = first_result["id"]
                if "text" in first_result:
                    template_vars[f"{prefix}_first_text"] = first_result["text"]
                if "metadata" in first_result and "title" in first_result["metadata"]:
                    template_vars[f"{prefix}_first_title"] = first_result["metadata"]["title"]
        
        return template_vars


# Convenience functions for common use cases
def extract_value(data: Any, path: str, default: Any = None, convert_type: Optional[type] = None) -> Any:
    """Convenience function for single value extraction."""
    parser = JSONOutputParser()
    return parser.extract(data, path, default, convert_type)


def extract_multiple(data: Any, extractions: Dict[str, str], defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for multiple value extraction."""
    parser = JSONOutputParser()
    return parser.extract_multiple(data, extractions, defaults)


def create_step_template_vars(step_output: Any, step_name: str = "previous") -> Dict[str, Any]:
    """Create template variables from step output for use in next step."""
    return CommonExtractions.create_template_vars(step_output, step_name)