#!/usr/bin/env python3
"""
Custom JSON Encoder for MCP Server
=================================
Handles serialization of complex objects for MCP protocol compliance.
"""

import json
import logging
from typing import Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class MCPJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MCP server responses."""
    
    def default(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to JSON-compatible types."""
        
        # Handle numpy types
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle sets
        elif isinstance(obj, set):
            return list(obj)
        
        # Handle complex numbers
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        
        # Handle objects with __dict__ (custom classes)
        elif hasattr(obj, '__dict__'):
            return {
                "_type": obj.__class__.__name__,
                "data": {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            }
        
        # Handle objects with to_dict method
        elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        
        # Handle objects with __str__ method as fallback
        elif hasattr(obj, '__str__'):
            return str(obj)
        
        # Final fallback
        try:
            return super().default(obj)
        except TypeError:
            logger.warning(f"Could not serialize object of type {type(obj)}: {obj}")
            return f"<non-serializable: {type(obj).__name__}>"

def safe_json_serialize(data: Any, max_depth: int = 10) -> dict:
    """
    Safely serialize complex data structures for MCP protocol.
    
    Args:
        data: Data to serialize
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        JSON-serializable dictionary
    """
    if max_depth <= 0:
        return "<max_depth_reached>"
    
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Ensure key is string
            str_key = str(key) if not isinstance(key, str) else key
            result[str_key] = safe_json_serialize(value, max_depth - 1)
        return result
    
    elif isinstance(data, (list, tuple)):
        return [safe_json_serialize(item, max_depth - 1) for item in data]
    
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    
    else:
        # Use the custom encoder for complex objects
        try:
            encoder = MCPJSONEncoder()
            return encoder.default(data)
        except Exception as e:
            logger.warning(f"Failed to serialize {type(data)}: {e}")
            return f"<serialization_error: {type(data).__name__}>"

def prepare_mcp_response(response_data: dict) -> dict:
    """
    Prepare a response dictionary for MCP protocol serialization.
    
    Args:
        response_data: Raw response data
        
    Returns:
        MCP-compatible response dictionary
    """
    try:
        # First pass: deep copy and sanitize
        sanitized = safe_json_serialize(response_data)
        
        # Second pass: test JSON serialization
        json_str = json.dumps(sanitized, cls=MCPJSONEncoder, ensure_ascii=False)
        
        # Third pass: verify round-trip
        verified = json.loads(json_str)
        
        logger.debug(f"Successfully prepared MCP response: {len(json_str)} chars")
        return verified
        
    except Exception as e:
        logger.error(f"Failed to prepare MCP response: {e}")
        return {
            "success": False,
            "error": f"Response serialization failed: {str(e)}",
            "original_keys": list(response_data.keys()) if isinstance(response_data, dict) else "non-dict"
        }