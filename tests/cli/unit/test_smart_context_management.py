"""Unit tests for smart context management components.

Tests for HistorySummarizer, EphemeralToolExecutor, and ContextManagementConfig.
"""

import asyncio
import pytest
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

from promptchain.cli.models.config import ContextManagementConfig, Config
from promptchain.utils.history_summarizer import (
    HistorySummarizer,
    SummarizationResult,
    SUMMARIZATION_PROMPT,
)
from promptchain.utils.ephemeral_executor import (
    EphemeralToolExecutor,
    EphemeralResult,
    DEFAULT_HEAVY_TOOL_PATTERNS,
    _estimate_tokens,
    _truncate_output,
)


class TestContextManagementConfig:
    """Tests for ContextManagementConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ContextManagementConfig()

        # Summarization defaults
        assert config.summarizer_model == "openai/gpt-4.1-mini-2025-04-14"
        assert config.summarize_every_n_iterations == 5
        assert config.summarize_token_threshold == 0.7
        assert config.max_summary_tokens == 500
        assert config.preserve_last_n_turns == 2

        # Ephemeral defaults
        assert config.ephemeral_enabled is True
        assert config.ephemeral_file_threshold_kb == 10
        assert config.ephemeral_response_threshold_kb == 5
        assert config.ephemeral_timeout_seconds == 300
        assert config.capture_full_errors is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ContextManagementConfig(
            summarizer_model="anthropic/claude-3-haiku-20240307",
            summarize_every_n_iterations=3,
            summarize_token_threshold=0.5,
            max_summary_tokens=1000,
            preserve_last_n_turns=4,
            ephemeral_enabled=False,
            ephemeral_file_threshold_kb=50,
            ephemeral_timeout_seconds=600,
        )

        assert config.summarizer_model == "anthropic/claude-3-haiku-20240307"
        assert config.summarize_every_n_iterations == 3
        assert config.summarize_token_threshold == 0.5
        assert config.max_summary_tokens == 1000
        assert config.preserve_last_n_turns == 4
        assert config.ephemeral_enabled is False
        assert config.ephemeral_file_threshold_kb == 50
        assert config.ephemeral_timeout_seconds == 600

    def test_config_serialization(self):
        """Test that config can be serialized to dict."""
        config = ContextManagementConfig()
        data = asdict(config)

        assert "summarizer_model" in data
        assert "ephemeral_enabled" in data
        assert data["summarize_token_threshold"] == 0.7


class TestContextManagementConfigValidation:
    """Tests for Config.validate() with context_management settings."""

    def test_valid_config_passes_validation(self):
        """Test that valid config passes validation."""
        config = Config()
        config.validate()  # Should not raise

    def test_invalid_summarize_every_n_iterations(self):
        """Test validation fails for invalid summarize_every_n_iterations."""
        config = Config()
        config.context_management.summarize_every_n_iterations = 0

        with pytest.raises(ValueError, match="summarize_every_n_iterations"):
            config.validate()

    def test_invalid_summarize_token_threshold_low(self):
        """Test validation fails for token threshold below 0.1."""
        config = Config()
        config.context_management.summarize_token_threshold = 0.05

        with pytest.raises(ValueError, match="summarize_token_threshold"):
            config.validate()

    def test_invalid_summarize_token_threshold_high(self):
        """Test validation fails for token threshold above 1.0."""
        config = Config()
        config.context_management.summarize_token_threshold = 1.5

        with pytest.raises(ValueError, match="summarize_token_threshold"):
            config.validate()

    def test_invalid_max_summary_tokens_low(self):
        """Test validation fails for max_summary_tokens below 100."""
        config = Config()
        config.context_management.max_summary_tokens = 50

        with pytest.raises(ValueError, match="max_summary_tokens"):
            config.validate()

    def test_invalid_max_summary_tokens_high(self):
        """Test validation fails for max_summary_tokens above 2000."""
        config = Config()
        config.context_management.max_summary_tokens = 3000

        with pytest.raises(ValueError, match="max_summary_tokens"):
            config.validate()

    def test_invalid_preserve_last_n_turns(self):
        """Test validation fails for invalid preserve_last_n_turns."""
        config = Config()
        config.context_management.preserve_last_n_turns = 15

        with pytest.raises(ValueError, match="preserve_last_n_turns"):
            config.validate()

    def test_invalid_ephemeral_file_threshold(self):
        """Test validation fails for invalid ephemeral_file_threshold_kb."""
        config = Config()
        config.context_management.ephemeral_file_threshold_kb = 0

        with pytest.raises(ValueError, match="ephemeral_file_threshold_kb"):
            config.validate()

    def test_invalid_ephemeral_timeout_low(self):
        """Test validation fails for timeout below 10 seconds."""
        config = Config()
        config.context_management.ephemeral_timeout_seconds = 5

        with pytest.raises(ValueError, match="ephemeral_timeout_seconds"):
            config.validate()

    def test_invalid_ephemeral_timeout_high(self):
        """Test validation fails for timeout above 3600 seconds."""
        config = Config()
        config.context_management.ephemeral_timeout_seconds = 7200

        with pytest.raises(ValueError, match="ephemeral_timeout_seconds"):
            config.validate()


class TestSummarizationResult:
    """Tests for SummarizationResult dataclass."""

    def test_savings_percent_calculation(self):
        """Test token savings percentage calculation."""
        result = SummarizationResult(
            summary="Test summary",
            original_tokens=1000,
            summary_tokens=100,
            entries_summarized=10,
            entries_preserved=2,
        )

        assert result.savings_percent == 90.0

    def test_savings_percent_zero_original(self):
        """Test savings percent with zero original tokens."""
        result = SummarizationResult(
            summary="",
            original_tokens=0,
            summary_tokens=0,
            entries_summarized=0,
            entries_preserved=0,
        )

        assert result.savings_percent == 0.0


class TestHistorySummarizer:
    """Tests for HistorySummarizer class."""

    def test_initialization(self):
        """Test summarizer initialization."""
        summarizer = HistorySummarizer()

        assert summarizer.model == "openai/gpt-4.1-mini-2025-04-14"
        assert summarizer.max_summary_tokens == 500
        assert summarizer.encoding is not None

    def test_initialization_custom_model(self):
        """Test summarizer with custom model."""
        summarizer = HistorySummarizer(
            model="anthropic/claude-3-haiku-20240307",
            max_summary_tokens=1000,
        )

        assert summarizer.model == "anthropic/claude-3-haiku-20240307"
        assert summarizer.max_summary_tokens == 1000

    def test_count_tokens_empty_string(self):
        """Test token counting for empty string."""
        summarizer = HistorySummarizer()
        assert summarizer.count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Test token counting for simple text."""
        summarizer = HistorySummarizer()
        count = summarizer.count_tokens("Hello, world!")
        assert count > 0

    def test_format_entry_user(self):
        """Test formatting user message entry."""
        summarizer = HistorySummarizer()
        entry = {"role": "user", "content": "What is the weather?"}
        formatted = summarizer._format_entry(entry)

        assert "[USER]" in formatted
        assert "What is the weather?" in formatted

    def test_format_entry_assistant(self):
        """Test formatting assistant message entry."""
        summarizer = HistorySummarizer()
        entry = {"role": "assistant", "content": "The weather is sunny."}
        formatted = summarizer._format_entry(entry)

        assert "[ASSISTANT]" in formatted
        assert "The weather is sunny." in formatted

    def test_format_entry_tool(self):
        """Test formatting tool result entry."""
        summarizer = HistorySummarizer()
        entry = {
            "role": "tool",
            "name": "search",
            "content": "Found 5 results",
        }
        formatted = summarizer._format_entry(entry)

        assert "[TOOL:search]" in formatted
        assert "Found 5 results" in formatted

    def test_format_entry_tool_truncation(self):
        """Test tool result truncation for long content."""
        summarizer = HistorySummarizer()
        long_content = "x" * 1000
        entry = {
            "role": "tool",
            "name": "read_file",
            "content": long_content,
        }
        formatted = summarizer._format_entry(entry)

        assert len(formatted) < 600  # Truncated
        assert "truncated" in formatted

    def test_format_entry_tool_calls(self):
        """Test formatting assistant with tool calls."""
        summarizer = HistorySummarizer()
        entry = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search"}},
                {"function": {"name": "read"}},
            ],
        }
        formatted = summarizer._format_entry(entry)

        assert "[ASSISTANT]" in formatted
        assert "search" in formatted
        assert "read" in formatted

    def test_create_summary_message(self):
        """Test creating summary message for history."""
        summarizer = HistorySummarizer()
        summary = "Key points:\n- User asked about weather\n- Agent provided forecast"
        message = summarizer.create_summary_message(summary)

        assert message["role"] == "system"
        assert "[Summary of previous conversation]" in message["content"]
        assert summary in message["content"]

    def test_estimate_tokens_after_summarization_no_entries(self):
        """Test token estimation with no entries."""
        summarizer = HistorySummarizer()
        result = summarizer.estimate_tokens_after_summarization([], preserve_last_n=2)
        assert result == 0

    def test_estimate_tokens_after_summarization_few_entries(self):
        """Test token estimation with entries <= preserve_last_n."""
        summarizer = HistorySummarizer()
        entries = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = summarizer.estimate_tokens_after_summarization(entries, preserve_last_n=2)
        assert result > 0

    @pytest.mark.asyncio
    async def test_summarize_history_no_entries(self):
        """Test summarization with no entries to summarize."""
        summarizer = HistorySummarizer()
        entries = [
            {"role": "user", "content": "Hello"},
        ]
        result = await summarizer.summarize_history(entries, preserve_last_n=2)

        assert result.summary == ""
        assert result.entries_summarized == 0
        assert result.entries_preserved == 1

    @pytest.mark.asyncio
    async def test_summarize_history_with_mock(self):
        """Test summarization with mocked LLM call."""
        summarizer = HistorySummarizer()

        entries = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "How do I install it?"},
            {"role": "assistant", "content": "You can use apt or download from python.org."},
            {"role": "user", "content": "What about pip?"},
        ]

        with patch("litellm.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Summary: Python discussion with install instructions."
            mock_acompletion.return_value = mock_response

            result = await summarizer.summarize_history(entries, preserve_last_n=2)

            assert result.entries_summarized == 3  # First 3 entries summarized
            assert result.entries_preserved == 2  # Last 2 preserved
            assert "Summary" in result.summary


class TestEphemeralHelpers:
    """Tests for ephemeral executor helper functions."""

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty string."""
        assert _estimate_tokens("") == 0

    def test_estimate_tokens_simple(self):
        """Test token estimation for simple text."""
        # Rough estimate: ~4 chars per token
        assert _estimate_tokens("hello world") == 2  # 11 chars / 4 = 2

    def test_truncate_output_short(self):
        """Test truncation doesn't modify short text."""
        text = "Short text"
        result = _truncate_output(text, max_chars=100)
        assert result == text

    def test_truncate_output_long(self):
        """Test truncation of long text."""
        text = "x" * 2000
        result = _truncate_output(text, max_chars=100)

        assert len(result) < 2000
        assert "truncated" in result


class TestEphemeralResult:
    """Tests for EphemeralResult dataclass."""

    def test_success_result(self):
        """Test successful result creation."""
        result = EphemeralResult(
            success=True,
            summary="[SUCCESS] Operation completed",
            execution_time=1.5,
            original_output_tokens=500,
            tool_name="docker_build",
            was_ephemeral=True,
        )

        assert result.success is True
        assert result.error_details is None
        assert result.execution_time == 1.5
        assert result.was_ephemeral is True

    def test_failure_result(self):
        """Test failed result creation."""
        result = EphemeralResult(
            success=False,
            summary="[FAILED] Docker build failed",
            error_details="ModuleNotFoundError: No module named 'foo'",
            tool_name="docker_build",
        )

        assert result.success is False
        assert result.error_details is not None


class TestEphemeralToolExecutor:
    """Tests for EphemeralToolExecutor class."""

    def test_initialization_defaults(self):
        """Test executor initialization with defaults."""
        executor = EphemeralToolExecutor()

        assert executor.timeout == 300
        assert executor.file_size_threshold_kb == 10
        assert executor.summarize_success is True
        assert executor.capture_full_errors is True

    def test_initialization_custom(self):
        """Test executor initialization with custom values."""
        executor = EphemeralToolExecutor(
            timeout=600,
            file_size_threshold_kb=50,
            summarize_success=False,
        )

        assert executor.timeout == 600
        assert executor.file_size_threshold_kb == 50
        assert executor.summarize_success is False

    def test_is_heavy_tool_docker(self):
        """Test detection of docker tools as heavy."""
        executor = EphemeralToolExecutor()

        assert executor.is_heavy_tool("docker_build") is True
        assert executor.is_heavy_tool("docker_run") is True
        assert executor.is_heavy_tool("DOCKER_EXEC") is True  # Case insensitive

    def test_is_heavy_tool_uv(self):
        """Test detection of uv tools as heavy."""
        executor = EphemeralToolExecutor()

        assert executor.is_heavy_tool("uv_sync") is True
        assert executor.is_heavy_tool("uv_install") is True
        assert executor.is_heavy_tool("pip_install") is True

    def test_is_heavy_tool_code_execution(self):
        """Test detection of code execution tools as heavy."""
        executor = EphemeralToolExecutor()

        assert executor.is_heavy_tool("execute_code") is True
        assert executor.is_heavy_tool("run_code") is True
        assert executor.is_heavy_tool("sandbox_execute") is True

    def test_is_heavy_tool_non_heavy(self):
        """Test detection of non-heavy tools."""
        executor = EphemeralToolExecutor()

        assert executor.is_heavy_tool("get_weather") is False
        assert executor.is_heavy_tool("search") is False
        assert executor.is_heavy_tool("calculate") is False

    def test_summarize_docker_success(self):
        """Test summarization of successful docker output."""
        executor = EphemeralToolExecutor()

        output = """
        Step 1/10: FROM python:3.9
        Step 2/10: WORKDIR /app
        Successfully built abc123def456
        Successfully tagged myapp:latest
        """

        summary = executor._summarize_docker_output(output)

        assert "[SUCCESS]" in summary
        assert "tagged" in summary.lower() or "built" in summary.lower()

    def test_summarize_docker_failure(self):
        """Test summarization of failed docker output."""
        executor = EphemeralToolExecutor()

        output = """
        Step 1/10: FROM python:3.9
        Step 2/10: COPY requirements.txt .
        ERROR: Could not find requirements.txt
        The command failed with exit code 1
        """

        summary = executor._summarize_docker_output(output)

        assert "[FAILED]" in summary

    def test_summarize_uv_success(self):
        """Test summarization of successful uv output."""
        executor = EphemeralToolExecutor()

        output = """
        Resolved 15 packages in 1.2s
        Adding requests==2.31.0
        Adding numpy==1.24.0
        Adding pandas==2.0.0
        Installed 15 packages in 2.5s
        """

        summary = executor._summarize_uv_output(output)

        assert "[SUCCESS]" in summary
        assert "package" in summary.lower()

    def test_summarize_code_execution_success(self):
        """Test summarization of successful code execution."""
        executor = EphemeralToolExecutor()

        output = "Hello, World!\nResult: 42"

        summary = executor._summarize_code_output(output)

        assert "[SUCCESS]" in summary
        assert "Hello, World!" in summary

    def test_summarize_code_execution_error(self):
        """Test summarization of failed code execution."""
        executor = EphemeralToolExecutor()

        output = """
        Traceback (most recent call last):
          File "test.py", line 5, in <module>
            raise ValueError("Something went wrong")
        ValueError: Something went wrong
        """

        summary = executor._summarize_code_output(output)

        assert "[FAILED]" in summary
        assert "ValueError" in summary

    def test_summarize_file_read(self):
        """Test summarization of file read."""
        executor = EphemeralToolExecutor()

        content = """
def hello():
    print("Hello")

class MyClass:
    pass

def world():
    print("World")
"""

        summary = executor._summarize_file_read(content, "/path/to/file.py")

        assert "[FILE READ]" in summary
        assert "file.py" in summary
        assert "lines" in summary.lower()

    def test_get_metrics(self):
        """Test metrics retrieval."""
        executor = EphemeralToolExecutor()

        metrics = executor.get_metrics()

        assert "total_executions" in metrics
        assert "total_tokens_saved" in metrics
        assert metrics["total_executions"] == 0

    def test_reset_metrics(self):
        """Test metrics reset."""
        executor = EphemeralToolExecutor()
        executor._total_executions = 10
        executor._total_tokens_saved = 5000

        executor.reset_metrics()

        assert executor._total_executions == 0
        assert executor._total_tokens_saved == 0

    @pytest.mark.asyncio
    async def test_execute_non_ephemeral(self):
        """Test execution of non-ephemeral tool."""
        executor = EphemeralToolExecutor()

        async def simple_tool(name: str, args: dict) -> str:
            return "Simple result"

        result = await executor.execute(
            tool_name="get_weather",
            tool_args={"location": "NYC"},
            tool_executor=simple_tool,
        )

        assert result.success is True
        assert result.summary == "Simple result"
        assert result.was_ephemeral is False

    @pytest.mark.asyncio
    async def test_execute_ephemeral(self):
        """Test execution of ephemeral tool."""
        executor = EphemeralToolExecutor()

        async def docker_tool(name: str, args: dict) -> str:
            return "Successfully built abc123\nSuccessfully tagged myapp:latest"

        result = await executor.execute(
            tool_name="docker_build",
            tool_args={"dockerfile": "."},
            tool_executor=docker_tool,
        )

        assert result.success is True
        assert result.was_ephemeral is True
        assert "[SUCCESS]" in result.summary

    @pytest.mark.asyncio
    async def test_execute_force_ephemeral(self):
        """Test forced ephemeral execution."""
        executor = EphemeralToolExecutor()

        async def normal_tool(name: str, args: dict) -> str:
            return "x" * 1000  # Long output

        result = await executor.execute(
            tool_name="custom_tool",
            tool_args={},
            tool_executor=normal_tool,
            force_ephemeral=True,
        )

        assert result.was_ephemeral is True
        assert len(result.summary) < 1000

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test timeout handling."""
        executor = EphemeralToolExecutor(timeout=1)  # 1 second timeout

        async def slow_tool(name: str, args: dict) -> str:
            await asyncio.sleep(5)  # 5 seconds - will timeout
            return "Result"

        result = await executor.execute(
            tool_name="slow_tool",
            tool_args={},
            tool_executor=slow_tool,
        )

        assert result.success is False
        assert "[TIMEOUT]" in result.summary

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        """Test exception handling."""
        executor = EphemeralToolExecutor()

        async def failing_tool(name: str, args: dict) -> str:
            raise ValueError("Tool execution failed")

        result = await executor.execute(
            tool_name="failing_tool",
            tool_args={},
            tool_executor=failing_tool,
        )

        assert result.success is False
        assert "[FAILED]" in result.summary
        assert "Tool execution failed" in result.error_details


class TestDefaultHeavyToolPatterns:
    """Tests for default heavy tool pattern set."""

    def test_docker_patterns_present(self):
        """Test docker patterns are in defaults."""
        assert "docker_*" in DEFAULT_HEAVY_TOOL_PATTERNS

    def test_uv_patterns_present(self):
        """Test uv patterns are in defaults."""
        assert "uv_*" in DEFAULT_HEAVY_TOOL_PATTERNS

    def test_code_execution_patterns_present(self):
        """Test code execution patterns are in defaults."""
        assert "execute_code" in DEFAULT_HEAVY_TOOL_PATTERNS
        assert "sandbox_execute" in DEFAULT_HEAVY_TOOL_PATTERNS


class TestSummarizationPrompt:
    """Tests for the summarization prompt template."""

    def test_prompt_contains_key_sections(self):
        """Test prompt template has required sections."""
        assert "Key decisions" in SUMMARIZATION_PROMPT
        assert "Important results" in SUMMARIZATION_PROMPT
        assert "Errors and failures" in SUMMARIZATION_PROMPT
        assert "Progress milestones" in SUMMARIZATION_PROMPT
        assert "Unresolved items" in SUMMARIZATION_PROMPT

    def test_prompt_has_placeholders(self):
        """Test prompt template has format placeholders."""
        assert "{history}" in SUMMARIZATION_PROMPT
        assert "{max_tokens}" in SUMMARIZATION_PROMPT
