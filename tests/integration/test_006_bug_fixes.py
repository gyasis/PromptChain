"""
Integration tests for 006-promptchain-improvements: Critical Bug Fixes (US1).

Covers:
- test_gemini_debug_correct_params: mock MCP, assert error_message key present
- test_gemini_brainstorm_no_num_ideas: mock MCP, assert num_ideas absent
- test_ask_gemini_prompt_param: mock MCP, assert prompt key present
- test_event_loop_no_crash_in_tui_context: simulate Textual loop, no RuntimeError
- test_json_parser_malformed_returns_default: invalid JSON -> default, no exception
- test_mlflow_shutdown_bounded: mock unresponsive queue, returns within 3s
- test_config_cache_no_disk_read_on_second_call: file open count == 1
- test_verification_result_deep_copy: mutate result, original cache unchanged

Branch: 006-promptchain-improvements
Success Criteria: SC-001 (zero errors on Gemini tools, TUI, JSON parser, MLflow)

NOTE ON T008-T010 RED/GREEN STATUS
-----------------------------------
Commit 86b5a4b already applied all Gemini parameter fixes.  The current
production code uses the correct parameter names:
  - gemini_debug  -> error_message  (not error_context)
  - gemini_brainstorm -> no num_ideas argument
  - ask_gemini    -> prompt          (not question)

As a result T008, T009, T010 are GREEN in the current state of the branch.
The test assertions below verify the CORRECT (fixed) behavior.  If a
regression re-introduces the wrong parameter name these tests will turn RED
and catch the regression immediately.
"""
import asyncio
import copy
import time
import threading
import queue as stdlib_queue
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_mcp_helper(return_value: str = "ok") -> MagicMock:
    """Return a mock MCPHelper whose call_mcp_tool is an AsyncMock."""
    helper = MagicMock()
    helper.call_mcp_tool = AsyncMock(return_value=return_value)
    return helper


# ---------------------------------------------------------------------------
# T008 – T010: Gemini MCP tool parameter correctness
# ---------------------------------------------------------------------------

class TestGeminiBugFixes:
    """Tests for Gemini MCP tool parameter fixes (BUG-017, BUG-018, BUG-019).

    Each test exercises the actual production method and then inspects the
    ``arguments`` dict that was forwarded to ``mcp_helper.call_mcp_tool``.
    The assertions verify the CORRECT (fixed) parameter names are used.
    """

    # ------------------------------------------------------------------
    # T008 – gemini_debug must receive error_message, NOT error_context
    # ------------------------------------------------------------------

    def test_gemini_debug_correct_params(self):
        """
        T008 – verify_tool_result must pass error_message to gemini_debug.

        BUG-019: The original code passed error_context instead of
        error_message.  This test asserts the correct key is present and the
        wrong key is absent.

        Expected state: GREEN (bug fixed in commit 86b5a4b).
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            GeminiReasoningAugmentor,
            VerificationResult,
        )

        mock_helper = _make_mock_mcp_helper(return_value="looks good")

        augmentor = GeminiReasoningAugmentor(mcp_helper=mock_helper)

        # verify_tool_result is on GeminiReasoningAugmentor, but the actual
        # gemini_debug call lives inside the standalone verify_tool_result
        # method attached to that object.  We drive it directly.
        result = asyncio.run(
            augmentor.verify_tool_result(
                tool_name="search_tool",
                tool_result='{"status": "ok", "data": "some result"}',
                expected_outcome="return search results",
            )
        )

        # Verify the MCP helper was called at all
        assert mock_helper.call_mcp_tool.called, (
            "call_mcp_tool was never invoked – verify_tool_result did nothing"
        )

        # Find the gemini_debug call
        gemini_debug_calls = [
            c for c in mock_helper.call_mcp_tool.call_args_list
            if c.kwargs.get("tool_name") == "gemini_debug"
            or (c.args and len(c.args) > 1 and c.args[1] == "gemini_debug")
        ]

        assert gemini_debug_calls, (
            "gemini_debug was never called – the method may have changed its tool name"
        )

        # Inspect the arguments dict passed to gemini_debug
        call_obj = gemini_debug_calls[0]
        # Support both positional and keyword argument styles
        if "arguments" in call_obj.kwargs:
            args_dict: Dict[str, Any] = call_obj.kwargs["arguments"]
        else:
            # Positional: call_mcp_tool(server_id, tool_name, arguments)
            args_dict = call_obj.args[2] if len(call_obj.args) > 2 else {}

        assert "error_message" in args_dict, (
            f"gemini_debug must receive 'error_message' key, "
            f"but got keys: {list(args_dict.keys())}.  "
            f"BUG-019 regression: code is using wrong parameter name."
        )
        assert "error_context" not in args_dict, (
            f"gemini_debug must NOT receive 'error_context' key – "
            f"this is the buggy parameter name from BUG-019."
        )

    # ------------------------------------------------------------------
    # T009 – gemini_brainstorm must NOT include num_ideas
    # ------------------------------------------------------------------

    def test_gemini_brainstorm_no_num_ideas(self):
        """
        T009 – _brainstorm_options must NOT pass num_ideas to gemini_brainstorm.

        BUG-018: The original code injected a num_ideas key that is not part of
        the gemini_brainstorm MCP tool schema, causing the tool to reject the
        call.  This test asserts the key is absent.

        Expected state: GREEN (bug fixed in commit 86b5a4b).
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            GeminiReasoningAugmentor,
            DecisionComplexity,
        )

        mock_helper = _make_mock_mcp_helper(return_value="- option A\n- option B")

        augmentor = GeminiReasoningAugmentor(mcp_helper=mock_helper)

        # Drive the MODERATE complexity path which calls _brainstorm_options
        result = asyncio.run(
            augmentor.augment_decision_making(
                objective="choose the best retrieval strategy",
                current_context="We have a RAG pipeline",
                decision_point="Should we use dense or sparse retrieval?",
                complexity=DecisionComplexity.MODERATE,
            )
        )

        # Find the gemini_brainstorm call
        brainstorm_calls = [
            c for c in mock_helper.call_mcp_tool.call_args_list
            if c.kwargs.get("tool_name") == "gemini_brainstorm"
            or (c.args and len(c.args) > 1 and c.args[1] == "gemini_brainstorm")
        ]

        assert brainstorm_calls, (
            "gemini_brainstorm was never called during MODERATE complexity path"
        )

        call_obj = brainstorm_calls[0]
        if "arguments" in call_obj.kwargs:
            args_dict = call_obj.kwargs["arguments"]
        else:
            args_dict = call_obj.args[2] if len(call_obj.args) > 2 else {}

        assert "num_ideas" not in args_dict, (
            f"gemini_brainstorm must NOT receive 'num_ideas' key – "
            f"this is the invalid parameter from BUG-018.  "
            f"Keys received: {list(args_dict.keys())}"
        )

    # ------------------------------------------------------------------
    # T010 – ask_gemini must receive prompt, NOT question
    # ------------------------------------------------------------------

    def test_ask_gemini_prompt_param(self):
        """
        T010 – _quick_ask must pass prompt to ask_gemini, NOT question.

        BUG-017: The original code used the key 'question' which is not the
        parameter name expected by the ask_gemini MCP tool; 'prompt' is
        correct.  This test asserts 'prompt' is present and 'question' absent.

        Expected state: GREEN (bug fixed in commit 86b5a4b).
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            GeminiReasoningAugmentor,
            DecisionComplexity,
        )

        mock_helper = _make_mock_mcp_helper(return_value="use dense retrieval")

        augmentor = GeminiReasoningAugmentor(mcp_helper=mock_helper)

        # Drive the SIMPLE complexity path which calls _quick_ask -> ask_gemini
        result = asyncio.run(
            augmentor.augment_decision_making(
                objective="pick a model",
                current_context="minimal context",
                decision_point="Which embedding model to use?",
                complexity=DecisionComplexity.SIMPLE,
            )
        )

        ask_gemini_calls = [
            c for c in mock_helper.call_mcp_tool.call_args_list
            if c.kwargs.get("tool_name") == "ask_gemini"
            or (c.args and len(c.args) > 1 and c.args[1] == "ask_gemini")
        ]

        assert ask_gemini_calls, (
            "ask_gemini was never called during SIMPLE complexity path"
        )

        call_obj = ask_gemini_calls[0]
        if "arguments" in call_obj.kwargs:
            args_dict = call_obj.kwargs["arguments"]
        else:
            args_dict = call_obj.args[2] if len(call_obj.args) > 2 else {}

        assert "prompt" in args_dict, (
            f"ask_gemini must receive 'prompt' key, "
            f"but got keys: {list(args_dict.keys())}.  "
            f"BUG-017 regression: code is using wrong parameter name."
        )
        assert "question" not in args_dict, (
            f"ask_gemini must NOT receive 'question' key – "
            f"this is the buggy parameter name from BUG-017."
        )


# ---------------------------------------------------------------------------
# T011: Event loop crash prevention in TUI context
# ---------------------------------------------------------------------------

class TestEventLoopFixes:
    """Tests for TUI asyncio event loop crash fixes (BUG-001).

    Issue #2: Calling asyncio.run() from inside an already-running event loop
    raises RuntimeError.  The event_loop_manager module must detect this
    condition and raise a clear error rather than crashing the TUI.
    """

    def test_event_loop_no_crash_in_tui_context(self):
        """
        T011 – run_async_in_context raises RuntimeError with a helpful message
        when called from within an already-running event loop, rather than
        propagating a bare 'This event loop is already running' error.

        The TUI (Textual) runs its own event loop.  Pattern command handlers
        that previously called asyncio.run() directly would crash.  After the
        fix, run_async_in_context detects the running loop and raises a clear,
        actionable RuntimeError instructing the caller to use 'await' instead.

        Expected state: depends on whether BUG-001 is fixed.
        If event_loop_manager.py exists and the guard is in place -> GREEN.
        """
        from promptchain.cli.utils.event_loop_manager import run_async_in_context

        async def _driver():
            """This coroutine simulates the TUI context (loop already running)."""
            async def _inner():
                return "inner result"

            # Create the coroutine object before passing it to run_async_in_context
            # so we can close it explicitly if the call raises (avoids the
            # "coroutine was never awaited" RuntimeWarning from the garbage collector).
            coro = _inner()

            # Calling run_async_in_context from inside a running loop should
            # raise RuntimeError with a message directing the caller to use await.
            with pytest.raises(RuntimeError) as exc_info:
                run_async_in_context(coro)

            # Explicitly close the unawaited coroutine to suppress the GC warning.
            coro.close()

            error_message = str(exc_info.value).lower()
            # The error should be informative – not a bare asyncio crash
            assert any(
                keyword in error_message
                for keyword in ("already running", "tui", "await", "loop")
            ), (
                f"RuntimeError message should explain the TUI context requirement, "
                f"but got: {exc_info.value!r}"
            )

        asyncio.run(_driver())

    def test_is_event_loop_running_detects_running_loop(self):
        """
        Verify is_event_loop_running() returns True inside a running loop
        and False outside one.
        """
        from promptchain.cli.utils.event_loop_manager import is_event_loop_running

        # Outside any loop
        assert is_event_loop_running() is False, (
            "is_event_loop_running() should return False when no loop is running"
        )

        results: list = []

        async def _check():
            results.append(is_event_loop_running())

        asyncio.run(_check())

        assert results == [True], (
            "is_event_loop_running() should return True inside a running coroutine"
        )


# ---------------------------------------------------------------------------
# T012: JSONOutputParser malformed input returns default without raising
# ---------------------------------------------------------------------------

class TestJSONParserRobustness:
    """Tests for JSONOutputParser silent failure prevention (FR-003)."""

    def test_json_parser_malformed_returns_default(self):
        """
        T012 – extract() on completely invalid JSON returns the per-call
        default and never raises an exception.

        Expected state: GREEN (FR-003 fix already applied).
        """
        from promptchain.utils.json_output_parser import JSONOutputParser

        parser = JSONOutputParser()

        result = parser.extract("not valid json {{{{", path="key", default="fallback")

        assert result == "fallback", (
            f"Expected default 'fallback' for malformed JSON, got {result!r}"
        )

    def test_json_parser_malformed_no_exception(self):
        """
        T012b – extract() must never propagate an exception regardless of input.
        """
        from promptchain.utils.json_output_parser import JSONOutputParser

        parser = JSONOutputParser(default="safe_default")

        # These would crash a naive json.loads() call
        bad_inputs = [
            "not valid json {{{{",
            "{bad json}",
            "[unclosed",
            "}{",
            "",
            "null null null",
            "\x00\x01\x02",
        ]

        for bad_input in bad_inputs:
            try:
                result = parser.extract(bad_input, path="any_key", default="safe")
                # Must return a value, not raise
                assert result == "safe", (
                    f"Expected 'safe' default for input {bad_input!r}, got {result!r}"
                )
            except Exception as exc:
                pytest.fail(
                    f"JSONOutputParser.extract() raised {type(exc).__name__} "
                    f"for input {bad_input!r}: {exc}"
                )

    def test_json_parser_valid_json_extracts_correctly(self):
        """
        T012c – extract() on valid JSON with a real path works correctly.
        This guards against overly defensive implementations that always
        return the default.
        """
        from promptchain.utils.json_output_parser import JSONOutputParser

        parser = JSONOutputParser()
        data = '{"status": "ok", "count": 42}'

        assert parser.extract(data, path="status", default="missing") == "ok"
        assert parser.extract(data, path="count", default=0) == 42
        assert parser.extract(data, path="missing_key", default="default_val") == "default_val"


# ---------------------------------------------------------------------------
# T013: BackgroundLogger.shutdown() is bounded even when queue hangs
# ---------------------------------------------------------------------------

class TestMLflowShutdown:
    """Tests for MLflow queue bounded shutdown (FR-004)."""

    def test_mlflow_shutdown_bounded(self):
        """
        T013 – BackgroundLogger.shutdown(timeout=T) must return within a
        reasonable bound even when the worker thread is sluggish.

        The implementation calls flush(timeout=T) then worker.join(timeout=T),
        so the theoretical worst-case wall time is 2*T.  This test uses T=1.0
        and asserts the total elapsed time stays below 2*T + 0.5s = 2.5s.
        An unbounded (pre-fix) implementation would block indefinitely; the
        fix ensures the call always returns.

        Expected state: GREEN (FR-004 fix already applied).
        """
        from promptchain.observability.queue import BackgroundLogger

        shutdown_timeout = 1.0  # seconds
        max_allowed = shutdown_timeout * 2 + 0.5  # 2.5s ceiling

        # Patch use_background_logging so the worker thread is started
        with patch(
            "promptchain.observability.queue.use_background_logging",
            return_value=True,
        ):
            bg_logger = BackgroundLogger(maxsize=10)

        # Replace flush() with a version that deliberately consumes the full
        # timeout budget, simulating an unresponsive MLflow server.
        # It sleeps for exactly the timeout value (the maximum the real
        # flush() implementation would wait) then returns False.
        def slow_flush(timeout=None):
            sleep_duration = timeout if timeout is not None else shutdown_timeout
            time.sleep(sleep_duration)
            return False

        bg_logger.flush = slow_flush

        start = time.perf_counter()
        bg_logger.shutdown(timeout=shutdown_timeout)
        elapsed = time.perf_counter() - start

        assert elapsed < max_allowed, (
            f"shutdown(timeout={shutdown_timeout}) took {elapsed:.2f}s, "
            f"expected < {max_allowed:.1f}s (2*timeout + 0.5s guard).  "
            f"The bounded-timeout fix is not working – shutdown may be "
            f"blocking indefinitely."
        )

    def test_mlflow_shutdown_flushes_queue_before_stop(self):
        """
        T013b – shutdown() calls flush() BEFORE joining the worker thread so
        that already-queued items are not discarded.
        """
        from promptchain.observability.queue import BackgroundLogger

        call_order: List[str] = []

        with patch(
            "promptchain.observability.queue.use_background_logging",
            return_value=True,
        ):
            logger = BackgroundLogger(maxsize=10)

        original_flush = logger.flush

        def tracking_flush(timeout=None):
            call_order.append("flush")
            return original_flush(timeout=timeout)

        original_join = logger.worker.join if logger.worker else None

        def tracking_join(timeout=None):
            call_order.append("join")
            if original_join:
                original_join(timeout=timeout)

        logger.flush = tracking_flush
        if logger.worker:
            logger.worker.join = tracking_join

        logger.shutdown(timeout=1.0)

        assert "flush" in call_order, "flush() was never called during shutdown()"
        if "join" in call_order:
            flush_idx = call_order.index("flush")
            join_idx = call_order.index("join")
            assert flush_idx < join_idx, (
                f"flush() must be called BEFORE worker.join(), "
                f"but order was {call_order}"
            )


# ---------------------------------------------------------------------------
# T014: get_observability_config() reads the file at most once (cache hit)
# ---------------------------------------------------------------------------

class TestConfigCache:
    """Tests for observability config caching (FR-005)."""

    def test_config_cache_no_disk_read_on_second_call(self):
        """
        T014 – When no config file exists, get_observability_config() must
        read the filesystem at most once across two successive calls.
        The second call must be served entirely from the in-memory cache.

        Uses patch('builtins.open') to count how many times open() is invoked.

        Expected state: GREEN (FR-005 fix already applied).
        """
        import promptchain.observability.config as cfg_module

        # Reset cache before the test to guarantee a clean slate
        cfg_module.clear_config_cache()

        open_call_count = [0]
        original_open = open

        def counting_open(*args, **kwargs):
            open_call_count[0] += 1
            return original_open(*args, **kwargs)

        # Patch _get_config_file_path to simulate "no config file on disk"
        # so we don't accidentally read the developer's real config
        with patch.object(
            cfg_module, "_get_config_file_path", return_value=None
        ), patch("builtins.open", side_effect=counting_open):
            cfg_module.clear_config_cache()  # clear again inside the patch

            _ = cfg_module.get_observability_config()
            _ = cfg_module.get_observability_config()

        assert open_call_count[0] <= 1, (
            f"open() was called {open_call_count[0]} times across two "
            f"get_observability_config() calls.  Expected at most 1 call "
            f"(second call should use cache).  FR-005 cache is not working."
        )

    def test_config_cache_returns_consistent_values(self):
        """
        T014b – Two consecutive calls to get_observability_config() return
        identical results (content is stable, not re-read each time).
        """
        import promptchain.observability.config as cfg_module

        with patch.object(cfg_module, "_get_config_file_path", return_value=None):
            cfg_module.clear_config_cache()
            first = cfg_module.get_observability_config()
            second = cfg_module.get_observability_config()

        assert first == second, (
            f"Two calls to get_observability_config() returned different values: "
            f"{first} vs {second}"
        )

    def test_config_cache_clear_forces_reload(self):
        """
        T014c – After clear_config_cache(), the next call re-evaluates
        configuration (open may be called again).
        """
        import promptchain.observability.config as cfg_module

        with patch.object(cfg_module, "_get_config_file_path", return_value=None):
            cfg_module.clear_config_cache()
            _ = cfg_module.get_observability_config()

            # Clearing cache should reset internal state
            cfg_module.clear_config_cache()

            # Internal cache vars should be None after clear
            assert cfg_module._config_cache is None, (
                "_config_cache should be None immediately after clear_config_cache()"
            )


# ---------------------------------------------------------------------------
# T015: LogicVerifier.verify_tool_selection() returns deep copies from cache
# ---------------------------------------------------------------------------

class TestVerificationCache:
    """Tests for verification result deep copy (FR-006, BUG-009)."""

    def test_verification_result_deep_copy(self):
        """
        T015 – verify_tool_selection() must return a deep copy of cached
        VerificationResult objects.  Mutating the returned result must NOT
        alter the entry stored in the internal cache.

        BUG-009: shallow copies of cached verification results meant that
        downstream code mutating the returned VerificationResult would
        corrupt the cache entry, causing incorrect behaviour on subsequent
        calls with the same (tool, objective) pair.

        Expected state: GREEN (copy.deepcopy fix already applied).
        """
        from promptchain.utils.enhanced_agentic_step_processor import (
            LogicVerifier,
            VerificationResult,
        )

        mock_helper = _make_mock_mcp_helper(
            return_value='{"documents": [], "results": []}'
        )

        verifier = LogicVerifier(mcp_helper=mock_helper)

        objective = "find all Python files in the repo"
        tool_name = "search_files"

        # Populate the cache via the public API
        result1: VerificationResult = asyncio.run(
            verifier.verify_tool_selection(
                objective=objective,
                tool_name=tool_name,
                tool_args={"pattern": "*.py"},
                context=[],
            )
        )

        # The cache entry key uses hash(objective)
        cache_key = f"{tool_name}:{hash(objective)}"
        assert cache_key in verifier.verification_cache, (
            f"Cache key {cache_key!r} not found.  The result was not cached."
        )

        # Get the cached entry directly (for comparison)
        cached_before = verifier.verification_cache[cache_key]
        original_warnings = list(cached_before.warnings)

        # Retrieve via the public API a second time (should be a cache hit)
        result2: VerificationResult = asyncio.run(
            verifier.verify_tool_selection(
                objective=objective,
                tool_name=tool_name,
                tool_args={"pattern": "*.py"},
                context=[],
            )
        )

        # Mutate the second returned result
        result2.warnings.append("INJECTED_BY_TEST_MUTATION")

        # The cache entry must be unchanged
        cached_after = verifier.verification_cache[cache_key]

        assert cached_after.warnings == original_warnings, (
            f"Mutating the returned VerificationResult corrupted the cache entry.  "
            f"Cache warnings before mutation: {original_warnings!r}.  "
            f"Cache warnings after mutation: {cached_after.warnings!r}.  "
            f"BUG-009 / FR-006 deep copy fix is not working."
        )

    def test_verification_cache_hit_avoids_mcp_call(self):
        """
        T015b – On the second call with the same (tool, objective) pair, the
        MCPHelper must NOT be invoked again (cache hit, no network call).
        """
        from promptchain.utils.enhanced_agentic_step_processor import LogicVerifier

        mock_helper = _make_mock_mcp_helper(
            return_value='{"documents": [], "results": []}'
        )

        verifier = LogicVerifier(mcp_helper=mock_helper)

        objective = "list all markdown files"
        tool_name = "list_files"
        tool_args = {"extension": ".md"}

        # First call – populates cache
        asyncio.run(
            verifier.verify_tool_selection(
                objective=objective,
                tool_name=tool_name,
                tool_args=tool_args,
                context=[],
            )
        )

        first_call_count = mock_helper.call_mcp_tool.call_count

        # Second call – should be served from cache
        asyncio.run(
            verifier.verify_tool_selection(
                objective=objective,
                tool_name=tool_name,
                tool_args=tool_args,
                context=[],
            )
        )

        second_call_count = mock_helper.call_mcp_tool.call_count

        assert second_call_count == first_call_count, (
            f"call_mcp_tool was invoked again on a cache hit "
            f"(count went from {first_call_count} to {second_call_count}).  "
            f"The cache is not being consulted before making MCP calls."
        )
