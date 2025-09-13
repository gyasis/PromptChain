#!/usr/bin/env python3
"""
Athena LightRAG Exceptions
==========================
Custom exceptions for comprehensive error handling.

Author: Athena LightRAG System
Date: 2025-09-08
"""

from typing import Optional, Dict, Any


class AthenaLightRAGException(Exception):
    """Base exception for all Athena LightRAG errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class DatabaseNotFoundError(AthenaLightRAGException):
    """Raised when LightRAG database is not found."""
    
    def __init__(self, working_dir: str):
        super().__init__(
            f"LightRAG database not found at {working_dir}. Run ingestion first.",
            error_code="DB_NOT_FOUND",
            details={"working_dir": working_dir}
        )


class ConfigurationError(AthenaLightRAGException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_field: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIG_ERROR",
            details={"config_field": config_field} if config_field else {}
        )


class QueryExecutionError(AthenaLightRAGException):
    """Raised when query execution fails."""
    
    def __init__(
        self, 
        message: str, 
        query: str, 
        mode: str, 
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            error_code="QUERY_ERROR",
            details={
                "query": query,
                "mode": mode,
                "original_error": str(original_error) if original_error else None
            }
        )
        self.original_error = original_error


class ContextExtractionError(AthenaLightRAGException):
    """Raised when context extraction fails."""
    
    def __init__(
        self, 
        message: str, 
        query: str, 
        context_type: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            error_code="CONTEXT_ERROR",
            details={
                "query": query,
                "context_type": context_type,
                "original_error": str(original_error) if original_error else None
            }
        )


class SQLGenerationError(AthenaLightRAGException):
    """Raised when SQL generation fails."""
    
    def __init__(
        self, 
        message: str, 
        natural_query: str,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            error_code="SQL_GEN_ERROR",
            details={
                "natural_query": natural_query,
                "original_error": str(original_error) if original_error else None
            }
        )


class AgenticReasoningError(AthenaLightRAGException):
    """Raised when agentic reasoning fails."""
    
    def __init__(
        self, 
        message: str, 
        objective: str,
        step_number: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            error_code="AGENTIC_ERROR",
            details={
                "objective": objective,
                "step_number": step_number,
                "original_error": str(original_error) if original_error else None
            }
        )


class MCPServerError(AthenaLightRAGException):
    """Raised when MCP server operations fail."""
    
    def __init__(
        self, 
        message: str, 
        tool_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message,
            error_code="MCP_ERROR",
            details={
                "tool_name": tool_name,
                "original_error": str(original_error) if original_error else None
            }
        )


class ValidationError(AthenaLightRAGException):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field_name: str,
        field_value: Any,
        expected_type: Optional[str] = None
    ):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            details={
                "field_name": field_name,
                "field_value": str(field_value),
                "expected_type": expected_type
            }
        )


class TokenLimitExceededError(AthenaLightRAGException):
    """Raised when token limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        current_tokens: int,
        max_tokens: int,
        operation: str
    ):
        super().__init__(
            message,
            error_code="TOKEN_LIMIT_ERROR",
            details={
                "current_tokens": current_tokens,
                "max_tokens": max_tokens,
                "operation": operation
            }
        )