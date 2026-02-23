"""Comprehensive error handling for PromptChain CLI.

This module provides centralized error handling with:
- User-friendly error messages
- Specific error recovery strategies
- Auto-retry with exponential backoff
- Error logging and tracking (T143: JSONL session logs)
- Graceful degradation

Task T141: Global Error Handler Implementation
Task T143: Error logging to JSONL session logs
"""

import asyncio
import logging
import os
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .models import Message


class ErrorSeverity(Enum):
    """Error severity levels."""

    INFO = "info"  # Informational, no action needed
    WARNING = "warning"  # Warning, operation continues
    ERROR = "error"  # Error, operation failed but app continues
    CRITICAL = "critical"  # Critical, app may need to exit


class ErrorCategory(Enum):
    """Error categories for specific handling."""

    API_AUTH = "api_auth"  # API authentication/authorization
    API_RATE_LIMIT = "api_rate_limit"  # Rate limiting
    API_TIMEOUT = "api_timeout"  # Request timeout
    NETWORK = "network"  # Network connectivity
    FILE_IO = "file_io"  # File operations
    PERMISSION = "permission"  # Permission errors
    VALIDATION = "validation"  # Input validation
    SESSION = "session"  # Session management
    AGENT = "agent"  # Agent initialization/execution
    MCP = "mcp"  # MCP server errors
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class ErrorContext:
    """Context information for an error.

    Attributes:
        error: The exception that occurred
        category: Error category for specific handling
        severity: Error severity level
        operation: What operation was being performed
        user_message: User-friendly error message
        recovery_hint: Suggestion for recovery
        metadata: Additional context data
        traceback_str: Full traceback string
        timestamp: When error occurred
    """

    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    operation: str
    user_message: str
    recovery_hint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    traceback_str: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
        if self.traceback_str is None:
            self.traceback_str = traceback.format_exc()


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        retryable_categories: Error categories that should trigger retry
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_categories: Optional[List[ErrorCategory]] = None

    def __post_init__(self):
        if self.retryable_categories is None:
            self.retryable_categories = [
                ErrorCategory.API_RATE_LIMIT,
                ErrorCategory.API_TIMEOUT,
                ErrorCategory.NETWORK,
            ]


class ErrorHandler:
    """Centralized error handling with recovery strategies.

    Features:
    - Error classification and categorization
    - User-friendly message generation
    - Auto-retry with exponential backoff
    - Error logging and tracking (in-memory + JSONL session logs)
    - Context-aware recovery hints
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        debug_mode: bool = False,
        session_id: Optional[str] = None,
        sessions_dir: Optional[Path] = None,
    ):
        """Initialize error handler.

        Args:
            retry_config: Retry configuration (default: RetryConfig())
            debug_mode: Whether to include full tracebacks
            session_id: Session ID for JSONL error logging (T143)
            sessions_dir: Base sessions directory (default: ~/.promptchain/sessions)
        """
        self.retry_config = retry_config or RetryConfig()
        self.debug_mode = debug_mode or os.getenv("PROMPTCHAIN_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )

        # Error history for tracking (in-memory)
        self.error_history: List[ErrorContext] = []
        self.max_history = 100

        # Setup logging
        self.logger = logging.getLogger("promptchain.error_handler")

        # JSONL error logger for session (T143)
        self.error_logger = None
        if session_id:
            from .utils.error_logger import create_error_logger

            self.error_logger = create_error_logger(session_id, sessions_dir)

    def classify_error(self, error: Exception, operation: str = "") -> ErrorContext:
        """Classify an error and create error context.

        Args:
            error: The exception to classify
            operation: What operation was being performed

        Returns:
            ErrorContext with classification and user message
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check specific exception types first (more specific than string matching)
        # Note: TimeoutError, ConnectionError, PermissionError are all OSError subclasses
        # So check them BEFORE the generic OSError check

        # Timeout Errors (subclass of OSError, so check first)
        if isinstance(error, TimeoutError):
            return self._create_timeout_error(error, operation)

        # Network Errors (ConnectionError is subclass of OSError)
        elif isinstance(error, ConnectionError):
            return self._create_network_error(error, operation)

        # Permission Errors (subclass of OSError)
        elif isinstance(error, PermissionError):
            return self._create_permission_error(error, operation)

        # File I/O Errors (check after more specific OSError subclasses)
        elif isinstance(error, (FileNotFoundError, IsADirectoryError, OSError)):
            return self._create_file_error(error, operation)

        # Now check string patterns (order matters - most specific first)

        # API Authentication Errors
        elif any(
            keyword in error_str
            for keyword in ["api_key", "api key", "authentication", "unauthorized", "401", "403"]
        ):
            return self._create_api_auth_error(error, operation)

        # Rate Limit Errors
        elif any(keyword in error_str for keyword in ["rate limit", "429", "quota"]):
            return self._create_rate_limit_error(error, operation)

        # Timeout Errors
        elif any(
            keyword in error_str for keyword in ["timeout", "timed out", "time out"]
        ):
            return self._create_timeout_error(error, operation)

        # Network Errors
        elif any(
            keyword in error_str
            for keyword in [
                "connection",
                "network",
                "unreachable",
                "dns",
                "socket",
            ]
        ):
            return self._create_network_error(error, operation)

        # Model/API Errors
        elif any(
            keyword in error_str
            for keyword in [
                "model not found",
                "invalid model",
                "model not available",
            ]
        ):
            return self._create_model_error(error, operation)

        # Session Errors (check operation context)
        elif "session" in operation.lower():
            return self._create_session_error(error, operation)

        # Agent Errors (check operation context)
        elif "agent" in operation.lower():
            return self._create_agent_error(error, operation)

        # MCP Server Errors
        elif any(keyword in error_str for keyword in ["mcp", "server", "tool"]):
            return self._create_mcp_error(error, operation)

        # Validation Errors (fallback for ValueError/TypeError/AssertionError)
        elif isinstance(error, (ValueError, AssertionError, TypeError)):
            return self._create_validation_error(error, operation)

        # Unknown Errors
        else:
            return self._create_unknown_error(error, operation)

    def _create_api_auth_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for API authentication errors."""
        error_str = str(error).lower()

        # Try to detect provider
        provider_hint = ""
        if "openai" in error_str:
            provider_hint = "\n\nSet environment variable:\n  export OPENAI_API_KEY=your_key"
        elif "anthropic" in error_str or "claude" in error_str:
            provider_hint = (
                "\n\nSet environment variable:\n  export ANTHROPIC_API_KEY=your_key"
            )
        elif "google" in error_str or "gemini" in error_str:
            provider_hint = (
                "\n\nSet environment variable:\n  export GOOGLE_API_KEY=your_key"
            )

        return ErrorContext(
            error=error,
            category=ErrorCategory.API_AUTH,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]API Authentication Failed[/bold red]\n\n"
                f"The API key for your model is missing or invalid.{provider_hint}\n\n"
                f"Add your API keys to a .env file or export as environment variables."
            ),
            recovery_hint=(
                "1. Check .env file for API key\n"
                "2. Verify API key is valid\n"
                "3. Try using a different model"
            ),
            metadata={"provider_detected": "openai" in error_str or "anthropic" in error_str},
        )

    def _create_rate_limit_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for rate limit errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.API_RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            operation=operation,
            user_message=(
                f"[bold yellow]Rate Limit Exceeded[/bold yellow]\n\n"
                f"You've hit the API rate limit. The request will be retried automatically.\n\n"
                f"Tips:\n"
                f"  • Automatic retry in progress (with exponential backoff)\n"
                f"  • Use a different model with higher limits\n"
                f"  • Upgrade your API plan for higher rate limits"
            ),
            recovery_hint="Wait for automatic retry or switch to a different model",
            metadata={"retryable": True},
        )

    def _create_timeout_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for timeout errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.API_TIMEOUT,
            severity=ErrorSeverity.WARNING,
            operation=operation,
            user_message=(
                f"[bold yellow]Request Timed Out[/bold yellow]\n\n"
                f"The request took too long to complete. Retrying automatically...\n\n"
                f"Possible causes:\n"
                f"  • API server is slow\n"
                f"  • Large request or response\n"
                f"  • Network congestion"
            ),
            recovery_hint="Wait for automatic retry or simplify your request",
            metadata={"retryable": True},
        )

    def _create_network_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for network errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Network Error[/bold red]\n\n"
                f"Could not connect to the API server.\n\n"
                f"Possible causes:\n"
                f"  • No internet connection\n"
                f"  • API server is down\n"
                f"  • Firewall blocking connection\n"
                f"  • DNS resolution failed\n\n"
                f"Check your internet connection and try again."
            ),
            recovery_hint="Verify internet connection and API server status",
            metadata={"retryable": True},
        )

    def _create_file_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for file I/O errors."""
        filename = getattr(error, "filename", "unknown file")

        return ErrorContext(
            error=error,
            category=ErrorCategory.FILE_IO,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]File Error[/bold red]\n\n"
                f"Could not access: {filename}\n\n"
                f"Current directory: {os.getcwd()}\n\n"
                f"Check the file path and try again.\n"
                f"Use @path/to/file syntax for file references."
            ),
            recovery_hint="Verify file path and check file permissions",
            metadata={"filename": str(filename)},
        )

    def _create_permission_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for permission errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Permission Denied[/bold red]\n\n"
                f"Do not have permission to access the requested resource.\n\n"
                f"Check file/directory permissions and try again."
            ),
            recovery_hint="Check permissions with: ls -l",
        )

    def _create_model_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for model not found errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Model Not Available[/bold red]\n\n"
                f"The specified model could not be found or accessed.\n\n"
                f"Common causes:\n"
                f"  • Model name typo (check spelling)\n"
                f"  • Model not available in your region\n"
                f"  • Model requires different API tier\n\n"
                f"Check LiteLLM documentation for available models:\n"
                f"https://docs.litellm.ai/docs/providers"
            ),
            recovery_hint="Use /agent create with a different model",
        )

    def _create_session_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for session errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.SESSION,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Session Error[/bold red]\n\n"
                f"Failed to {operation}.\n\n"
                f"Error: {str(error)}\n\n"
                f"Session data may be corrupted or inaccessible."
            ),
            recovery_hint="Try creating a new session or check file permissions",
        )

    def _create_agent_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for agent errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.AGENT,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Agent Error[/bold red]\n\n"
                f"Failed to {operation}.\n\n"
                f"Error: {str(error)}\n\n"
                f"Try recreating the agent or using a different model."
            ),
            recovery_hint="Use /agent create to create a new agent",
        )

    def _create_mcp_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for MCP server errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.MCP,
            severity=ErrorSeverity.WARNING,
            operation=operation,
            user_message=(
                f"[bold yellow]MCP Server Error[/bold yellow]\n\n"
                f"Failed to communicate with MCP server.\n\n"
                f"Error: {str(error)}\n\n"
                f"The operation will continue without MCP tools."
            ),
            recovery_hint="Check MCP server configuration and restart if needed",
            metadata={"degraded_mode": True},
        )

    def _create_validation_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for validation errors."""
        return ErrorContext(
            error=error,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Invalid Input[/bold red]\n\n"
                f"The input provided is invalid.\n\n"
                f"Error: {str(error)}\n\n"
                f"Check your input and try again."
            ),
            recovery_hint="Review the command syntax with /help",
        )

    def _create_unknown_error(self, error: Exception, operation: str) -> ErrorContext:
        """Create context for unknown errors."""
        traceback_info = ""
        if self.debug_mode:
            traceback_info = f"\n\nTraceback:\n{traceback.format_exc()}"

        return ErrorContext(
            error=error,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            operation=operation,
            user_message=(
                f"[bold red]Unexpected Error[/bold red]\n\n"
                f"{type(error).__name__}: {str(error)}\n\n"
                f"If this persists, please report at:\n"
                f"https://github.com/your-org/promptchain/issues{traceback_info}"
            ),
            recovery_hint="Enable debug mode: export PROMPTCHAIN_DEBUG=1",
        )

    def create_error_message(self, context: ErrorContext) -> Message:
        """Create a Message object from error context.

        Args:
            context: Error context

        Returns:
            Message object for display
        """
        # Add debug info if enabled
        content = context.user_message
        if self.debug_mode and context.traceback_str:
            content += f"\n\n[dim]Traceback:\n{context.traceback_str}[/dim]"

        return Message(
            role="system",
            content=content,
            metadata={
                "error": True,
                "error_type": type(context.error).__name__,
                "error_category": context.category.value,
                "error_severity": context.severity.value,
                "error_message": str(context.error),
                "operation": context.operation,
                "recovery_hint": context.recovery_hint,
                "timestamp": context.timestamp,
                "traceback": context.traceback_str if self.debug_mode else None,
                **(context.metadata or {}),
            },
        )

    async def handle_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs,
    ) -> Any:
        """Execute operation with automatic retry on transient errors.

        Args:
            operation: Async function to execute
            operation_name: Human-readable operation name
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        last_context = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                else:
                    return operation(*args, **kwargs)

            except Exception as e:
                last_error = e
                last_context = self.classify_error(e, operation_name)

                # Check if error is retryable
                retryable_cats = self.retry_config.retryable_categories or []
                if last_context.category not in retryable_cats:
                    # Not retryable, fail immediately
                    raise

                # Log retry attempt
                if attempt < self.retry_config.max_attempts - 1:
                    delay = min(
                        self.retry_config.base_delay
                        * (self.retry_config.exponential_base**attempt),
                        self.retry_config.max_delay,
                    )

                    self.logger.warning(
                        f"Retry attempt {attempt + 1}/{self.retry_config.max_attempts} "
                        f"for {operation_name} after {delay:.1f}s delay. "
                        f"Error: {str(e)}"
                    )

                    # Wait before retrying
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    self.logger.error(
                        f"All {self.retry_config.max_attempts} retry attempts failed "
                        f"for {operation_name}. Error: {str(e)}"
                    )
                    raise

        # Should not reach here, but raise last error just in case
        if last_error:
            raise last_error

    def track_error(self, context: ErrorContext):
        """Track error in history (in-memory + JSONL session log).

        Args:
            context: Error context to track
        """
        # Add to in-memory history
        self.error_history.append(context)

        # Trim history if too large
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history :]

        # Log to JSONL session file (T143)
        if self.error_logger:
            self.error_logger.log_error(
                error=context.error,
                context=context.operation,
                metadata={
                    "category": context.category.value,
                    "severity": context.severity.value,
                    "user_message": context.user_message,
                    "recovery_hint": context.recovery_hint,
                    **(context.metadata or {}),
                },
                include_traceback=self.debug_mode,
            )

    def get_recent_errors(
        self, count: int = 10, category: Optional[ErrorCategory] = None
    ) -> List[ErrorContext]:
        """Get recent errors from history.

        Args:
            count: Number of errors to return
            category: Filter by category (optional)

        Returns:
            List of recent error contexts
        """
        errors = self.error_history

        if category:
            errors = [e for e in errors if e.category == category]

        return errors[-count:]
