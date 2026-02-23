"""Tests for comprehensive error handler.

Tests cover:
- Error classification and categorization
- User-friendly message generation
- Auto-retry with exponential backoff
- Error tracking and history
- Recovery hints

Task T141: Global Error Handler Tests
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from promptchain.cli.error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
    RetryConfig,
)


@pytest.fixture
def error_handler():
    """Create error handler for testing."""
    return ErrorHandler(
        retry_config=RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Fast retry for tests
            max_delay=1.0,
        ),
        debug_mode=False,
    )


@pytest.fixture
def debug_error_handler():
    """Create error handler with debug mode enabled."""
    return ErrorHandler(
        retry_config=RetryConfig(max_attempts=2),
        debug_mode=True,
    )


class TestErrorClassification:
    """Test error classification and categorization."""

    def test_api_auth_error_openai(self, error_handler):
        """Test classification of OpenAI API key error."""
        error = ValueError("Invalid API key provided for OpenAI")
        context = error_handler.classify_error(error, "calling OpenAI API")

        assert context.category == ErrorCategory.API_AUTH
        assert context.severity == ErrorSeverity.ERROR
        assert "API Authentication Failed" in context.user_message
        assert "OPENAI_API_KEY" in context.user_message

    def test_api_auth_error_anthropic(self, error_handler):
        """Test classification of Anthropic API key error."""
        error = Exception("Authentication failed: Invalid Anthropic API key")
        context = error_handler.classify_error(error, "calling Claude")

        assert context.category == ErrorCategory.API_AUTH
        assert "ANTHROPIC_API_KEY" in context.user_message

    def test_rate_limit_error(self, error_handler):
        """Test classification of rate limit error."""
        error = Exception("Rate limit exceeded (429)")
        context = error_handler.classify_error(error, "API call")

        assert context.category == ErrorCategory.API_RATE_LIMIT
        assert context.severity == ErrorSeverity.WARNING
        assert "Rate Limit Exceeded" in context.user_message
        assert context.metadata.get("retryable") is True

    def test_timeout_error(self, error_handler):
        """Test classification of timeout error."""
        error = TimeoutError("Request timed out after 30 seconds")
        context = error_handler.classify_error(error, "API call")

        assert context.category == ErrorCategory.API_TIMEOUT
        assert context.severity == ErrorSeverity.WARNING
        assert "Timed Out" in context.user_message

    def test_network_error(self, error_handler):
        """Test classification of network error."""
        error = ConnectionError("Failed to establish connection")
        context = error_handler.classify_error(error, "API call")

        assert context.category == ErrorCategory.NETWORK
        assert "Network Error" in context.user_message

    def test_file_not_found_error(self, error_handler):
        """Test classification of file not found error."""
        error = FileNotFoundError("test.txt")
        error.filename = "test.txt"
        context = error_handler.classify_error(error, "loading file")

        assert context.category == ErrorCategory.FILE_IO
        assert "File Error" in context.user_message
        assert "test.txt" in context.user_message

    def test_permission_error(self, error_handler):
        """Test classification of permission error."""
        error = PermissionError("Permission denied")
        context = error_handler.classify_error(error, "accessing file")

        assert context.category == ErrorCategory.PERMISSION
        assert "Permission Denied" in context.user_message

    def test_model_not_found_error(self, error_handler):
        """Test classification of model not found error."""
        error = ValueError("Model not found: invalid-model-name")
        context = error_handler.classify_error(error, "initializing agent")

        assert context.category == ErrorCategory.VALIDATION
        assert "Model Not Available" in context.user_message

    def test_validation_error(self, error_handler):
        """Test classification of validation error.

        Note: When operation contains 'agent', it's classified as AGENT error
        even if it's a ValueError. Use different operation for pure validation test.
        """
        error = ValueError("Invalid input: name must not be empty")
        context = error_handler.classify_error(error, "validating input")

        assert context.category == ErrorCategory.VALIDATION
        assert "Invalid Input" in context.user_message

    def test_unknown_error(self, error_handler):
        """Test classification of unknown error."""
        error = RuntimeError("Unexpected runtime error")
        context = error_handler.classify_error(error, "unknown operation")

        assert context.category == ErrorCategory.UNKNOWN
        assert "Unexpected Error" in context.user_message


class TestMessageCreation:
    """Test error message creation."""

    def test_create_message_basic(self, error_handler):
        """Test creating message from error context."""
        error = ValueError("Test error")
        context = error_handler.classify_error(error, "test operation")
        message = error_handler.create_error_message(context)

        assert message.role == "system"
        assert message.metadata["error"] is True
        assert message.metadata["error_type"] == "ValueError"
        assert message.metadata["error_category"] == context.category.value

    def test_create_message_with_debug(self, debug_error_handler):
        """Test creating message with debug mode enabled."""
        error = ValueError("Test error")
        context = debug_error_handler.classify_error(error, "test")
        message = debug_error_handler.create_error_message(context)

        # Debug mode should include traceback
        assert message.metadata["traceback"] is not None
        assert "Traceback" in message.content

    def test_create_message_without_debug(self, error_handler):
        """Test creating message without debug mode."""
        error = ValueError("Test error")
        context = error_handler.classify_error(error, "test")
        message = error_handler.create_error_message(context)

        # No debug mode means no traceback
        assert message.metadata["traceback"] is None


class TestRetryMechanism:
    """Test auto-retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, error_handler):
        """Test successful operation on first attempt (no retry)."""
        mock_operation = AsyncMock(return_value="success")

        result = await error_handler.handle_with_retry(
            mock_operation, "test operation"
        )

        assert result == "success"
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self, error_handler):
        """Test successful operation after transient failures."""
        # Fail twice, then succeed
        mock_operation = AsyncMock(
            side_effect=[
                ConnectionError("Network error"),
                ConnectionError("Network error"),
                "success",
            ]
        )

        result = await error_handler.handle_with_retry(
            mock_operation, "test operation"
        )

        assert result == "success"
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_all_attempts_fail(self, error_handler):
        """Test all retry attempts fail."""
        mock_operation = AsyncMock(side_effect=ConnectionError("Network error"))

        with pytest.raises(ConnectionError):
            await error_handler.handle_with_retry(mock_operation, "test operation")

        # Should try max_attempts times
        assert mock_operation.call_count == error_handler.retry_config.max_attempts

    @pytest.mark.asyncio
    async def test_retry_non_retryable_error(self, error_handler):
        """Test non-retryable error fails immediately."""
        # ValueError is not in retryable_categories
        mock_operation = AsyncMock(side_effect=ValueError("Bad input"))

        with pytest.raises(ValueError):
            await error_handler.handle_with_retry(mock_operation, "test operation")

        # Should fail immediately, no retries
        assert mock_operation.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self, error_handler):
        """Test exponential backoff between retries."""
        mock_operation = AsyncMock(
            side_effect=[
                TimeoutError("Timeout"),
                TimeoutError("Timeout"),
                "success",
            ]
        )

        start_time = time.time()
        result = await error_handler.handle_with_retry(
            mock_operation, "test operation"
        )
        elapsed = time.time() - start_time

        assert result == "success"
        # Should have delays: 0.1s, 0.2s (exponential)
        # Total minimum delay: ~0.3s
        assert elapsed >= 0.3

    @pytest.mark.asyncio
    async def test_retry_rate_limit_error(self, error_handler):
        """Test rate limit errors are retryable."""
        mock_operation = AsyncMock(
            side_effect=[
                Exception("Rate limit exceeded (429)"),
                "success",
            ]
        )

        result = await error_handler.handle_with_retry(
            mock_operation, "API call"
        )

        assert result == "success"
        assert mock_operation.call_count == 2


class TestErrorTracking:
    """Test error tracking and history."""

    def test_track_error(self, error_handler):
        """Test tracking errors in history."""
        error = ValueError("Test error")
        context = error_handler.classify_error(error, "test")

        error_handler.track_error(context)

        assert len(error_handler.error_history) == 1
        assert error_handler.error_history[0] == context

    def test_track_multiple_errors(self, error_handler):
        """Test tracking multiple errors."""
        for i in range(5):
            error = ValueError(f"Error {i}")
            context = error_handler.classify_error(error, f"operation {i}")
            error_handler.track_error(context)

        assert len(error_handler.error_history) == 5

    def test_track_error_max_history(self, error_handler):
        """Test error history is limited to max_history."""
        error_handler.max_history = 10

        # Track more than max_history errors
        for i in range(15):
            error = ValueError(f"Error {i}")
            context = error_handler.classify_error(error, f"operation {i}")
            error_handler.track_error(context)

        # Should only keep last 10
        assert len(error_handler.error_history) == 10
        # Should keep most recent
        assert "Error 14" in str(error_handler.error_history[-1].error)

    def test_get_recent_errors(self, error_handler):
        """Test retrieving recent errors."""
        for i in range(10):
            error = ValueError(f"Error {i}")
            context = error_handler.classify_error(error, f"operation {i}")
            error_handler.track_error(context)

        recent = error_handler.get_recent_errors(count=3)

        assert len(recent) == 3
        # Should get most recent
        assert "Error 9" in str(recent[-1].error)

    def test_get_recent_errors_by_category(self, error_handler):
        """Test filtering recent errors by category."""
        # Track different types of errors
        error_handler.track_error(
            error_handler.classify_error(
                ConnectionError("Network error"), "network op"
            )
        )
        error_handler.track_error(
            error_handler.classify_error(ValueError("Bad input"), "validation op")
        )
        error_handler.track_error(
            error_handler.classify_error(
                ConnectionError("Another network error"), "network op 2"
            )
        )

        network_errors = error_handler.get_recent_errors(
            count=10, category=ErrorCategory.NETWORK
        )

        assert len(network_errors) == 2
        assert all(e.category == ErrorCategory.NETWORK for e in network_errors)


class TestRecoveryHints:
    """Test recovery hint generation."""

    def test_api_auth_recovery_hint(self, error_handler):
        """Test recovery hint for API auth error."""
        error = ValueError("Invalid API key")
        context = error_handler.classify_error(error, "API call")

        assert context.recovery_hint is not None
        assert "API key" in context.recovery_hint

    def test_file_error_recovery_hint(self, error_handler):
        """Test recovery hint for file error."""
        error = FileNotFoundError("test.txt")
        error.filename = "test.txt"
        context = error_handler.classify_error(error, "loading file")

        assert context.recovery_hint is not None
        assert "file path" in context.recovery_hint.lower()

    def test_rate_limit_recovery_hint(self, error_handler):
        """Test recovery hint for rate limit error."""
        error = Exception("Rate limit exceeded")
        context = error_handler.classify_error(error, "API call")

        assert context.recovery_hint is not None
        assert "retry" in context.recovery_hint.lower()


class TestErrorContextMetadata:
    """Test error context metadata."""

    def test_error_context_timestamp(self, error_handler):
        """Test error context includes timestamp."""
        error = ValueError("Test error")
        context = error_handler.classify_error(error, "test")

        assert context.timestamp > 0
        assert context.timestamp <= time.time()

    def test_error_context_traceback(self, error_handler):
        """Test error context includes traceback."""
        try:
            # Create a real traceback by raising the error
            raise ValueError("Test error")
        except ValueError as error:
            context = error_handler.classify_error(error, "test")

        assert context.traceback_str is not None
        assert "Traceback" in context.traceback_str or "ValueError" in context.traceback_str

    def test_error_context_metadata_populated(self, error_handler):
        """Test error context metadata is populated."""
        error = ValueError("Rate limit exceeded")
        context = error_handler.classify_error(error, "API call")

        assert context.metadata is not None
        assert isinstance(context.metadata, dict)
