import asyncio
import json
import logging
import warnings
from datetime import datetime
from enum import Enum
from typing import (TYPE_CHECKING, Any, Awaitable, Callable, Dict, List,
                    Optional, Union)

from .agentic_step_result import AgenticStepResult, StepExecutionMetadata

if TYPE_CHECKING:
    from .history_summarizer import HistorySummarizer

logger = logging.getLogger(__name__)

# MLflow tracing support (optional)
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate token count for a list of messages.
    Uses simple approximation: ~4 characters per token.
    For more accuracy, integrate tiktoken library.

    Args:
        messages: List of message dictionaries

    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict):
            # Count content
            content = msg.get("content", "")
            if content:
                total_chars += len(str(content))
            # Count tool calls if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                total_chars += len(str(tool_calls))
    return total_chars // 4  # Approximate: 4 chars per token


class HistoryMode(str, Enum):
    """History accumulation modes for AgenticStepProcessor."""

    MINIMAL = "minimal"  # Only keep last assistant + tool results (default, backward compatible)
    PROGRESSIVE = (
        "progressive"  # Accumulate assistant messages + tool results progressively
    )
    KITCHEN_SINK = (
        "kitchen_sink"  # Keep everything - all reasoning, tool calls, results
    )


def get_function_name_from_tool_call(tool_call) -> Optional[str]:
    """
    Safely extract function name from different tool_call object formats.

    Args:
        tool_call: A tool call object from various LLM providers

    Returns:
        The function name or None if not found
    """
    try:
        # If it's a dictionary (like from OpenAI direct API)
        if isinstance(tool_call, dict):
            function_obj = tool_call.get("function", {})
            if isinstance(function_obj, dict):
                return function_obj.get("name")
        # If it's an object with attributes (like from LiteLLM)
        else:
            function_obj = getattr(tool_call, "function", None)
            if function_obj:
                # Direct attribute access
                if hasattr(function_obj, "name"):
                    return function_obj.name
                # Dictionary-like access for nested objects
                elif isinstance(function_obj, dict):
                    return function_obj.get("name")
                # Get function name through __dict__
                elif hasattr(function_obj, "__dict__"):
                    return function_obj.__dict__.get("name")
        # Last resort: try to convert to dictionary if possible
        if hasattr(tool_call, "model_dump"):
            model_dump = tool_call.model_dump()
            if isinstance(model_dump, dict) and "function" in model_dump:
                function_dict = model_dump["function"]
                if isinstance(function_dict, dict):
                    return function_dict.get("name")
    except Exception as e:
        logger.warning(f"Error extracting function name: {e}")

    return None  # Return None if we couldn't extract the name


class AgenticStepProcessor:
    """
    Represents a single step within a PromptChain that executes an internal
    agentic loop to achieve its objective, potentially involving multiple
    LLM calls and tool executions within this single step.
    Optionally, a specific model can be set for this step; otherwise, the chain's default model is used.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️  CRITICAL: INTERNAL HISTORY ISOLATION PATTERN (v0.4.2)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    The `conversation_history` maintained by AgenticStepProcessor is STRICTLY INTERNAL
    to its reasoning loop. This history is **NEVER** exposed to other agents in
    multi-agent workflows.

    **What Gets Exposed to Other Agents:**
    - ✅ Only the final output (`final_answer` from the reasoning loop)
    - ✅ High-level metadata (steps executed, tools called, execution time)

    **What NEVER Gets Exposed:**
    - ❌ Internal reasoning steps and intermediate LLM calls
    - ❌ Tool call details and results (unless in final answer)
    - ❌ Internal conversation history (progressive/kitchen_sink modes)
    - ❌ Clarification attempts and recovery logic

    **Why This Matters:**
    This isolation prevents **token explosion** in multi-agent plans. If each agent's
    internal reasoning history was exposed to subsequent agents, the context would
    grow exponentially with each step, quickly exceeding model limits and incurring
    massive costs.

    **Example Token Savings:**
    ```
    WITHOUT Isolation (BAD):
    Step 1: Agent A internal reasoning (2000 tokens)
    Step 2: Agent B receives Agent A's internal reasoning + does own (4000 tokens total)
    Step 3: Agent C receives A's + B's internal reasoning + does own (8000 tokens total)
    → Token explosion! Exponential growth!

    WITH Isolation (GOOD):
    Step 1: Agent A internal reasoning (2000 tokens) → outputs answer (100 tokens)
    Step 2: Agent B receives only Agent A's answer (100 tokens) + does own reasoning (2000 tokens)
    Step 3: Agent C receives only B's answer (100 tokens) + does own reasoning (2000 tokens)
    → Linear growth! Sustainable!
    ```

    **For Multi-Agent Context Sharing:**
    To expose conversation context to agents in multi-agent plans, use the
    per-agent history configuration in AgentChain:

    ```python
    agent_history_configs = {
        "my_agent": {
            "enabled": True,
            "max_tokens": 4000,
            "truncation_strategy": "keep_last"
        }
    }
    ```

    This provides **conversation-level** history (user inputs, agent outputs),
    NOT internal reasoning steps from AgenticStepProcessor.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(
        self,
        objective: str,
        max_internal_steps: int = 5,
        model_name: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        history_mode: str = "minimal",
        max_context_tokens: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        user_input_queue: Optional[asyncio.Queue] = None,
        streaming_callback: Optional[Callable[[str, str], None]] = None,
        step_timeout: Optional[
            float
        ] = 120.0,  # Timeout per LLM call in seconds (default 2 min)
        # Smart context management parameters
        enable_summarization: bool = True,  # Enable history summarization
        summarize_every_n: int = 5,  # Summarize after N iterations
        summarize_token_threshold: float = 0.7,  # Summarize when tokens exceed this % of max
        summarizer_model: str = "openai/gpt-4.1-mini-2025-04-14",  # Model for summarization
        preserve_last_n_turns: int = 2,  # Keep last N turns verbatim when summarizing
        # Two-tier model routing parameters (Phase 1: Quick Wins)
        fallback_model: Optional[
            str
        ] = None,  # Fast/cheap model for simple tasks (e.g., "openai/gpt-4o-mini")
        enable_two_tier_routing: bool = False,  # Enable intelligent model routing (default: off for backward compatibility)
        # Blackboard Architecture parameters (Phase 2: Token Optimization - 80% reduction)
        enable_blackboard: bool = False,  # Enable Blackboard structured state (default: off for backward compatibility)
        # Chain of Verification + Checkpointing parameters (Phase 3: Safety & Reliability - 50% error reduction)
        enable_cove: bool = False,  # Enable Chain of Verification (pre-execution validation)
        enable_checkpointing: bool = False,  # Enable epistemic checkpointing (stuck state detection)
        cove_confidence_threshold: float = 0.7,  # Minimum confidence to execute tool (0.0 to 1.0)
        # TAO Loop + Dry Runs parameters (Phase 4: Transparent Reasoning - explicit Think-Act-Observe)
        enable_tao_loop: bool = False,  # Enable explicit Think-Act-Observe loop (default: off, uses implicit ReAct)
        enable_dry_run: bool = False,  # Enable tool outcome prediction before execution (transparent reasoning)
    ):
        """
        Initializes the agentic step.

        Args:
            objective: The specific goal or instruction for this agentic step.
                       This will be used to guide the internal loop.
            max_internal_steps: Maximum number of internal iterations (LLM calls)
                                to prevent infinite loops.
            model_name: (Optional) The model to use for this agentic step. If None, defaults to the chain's first model.
            model_params: (Optional) Dictionary of parameters to pass to the model, such as tool_choice settings.
            history_mode: (Optional) History accumulation mode. Options:
                         - "minimal" (default): Only keep last assistant + tool results (backward compatible)
                           ⚠️  DEPRECATION NOTICE: This mode may be deprecated in future versions.
                           Consider using "progressive" for better multi-hop reasoning.
                         - "progressive": Accumulate assistant messages + tool results progressively (RECOMMENDED)
                         - "kitchen_sink": Keep everything - all reasoning, tool calls, results
            max_context_tokens: (Optional) Maximum context tokens before warning. Default is None (no limit).
            progress_callback: (Optional) Callback function called at each reasoning step (T052).
                             Signature: callback(current_step: int, max_steps: int, status: str)
                             Enables real-time progress updates for TUI integration.
            user_input_queue: (Optional) Async queue for receiving user input during execution.
                             Enables REACT-style interaction where users can provide input mid-execution.
                             Messages from queue are injected as user messages in the conversation.
            streaming_callback: (Optional) Callback for streaming reasoning output in real-time.
                              Signature: callback(event_type: str, content: str)
                              Event types: "thinking", "tool_call", "tool_result", "answer"
            step_timeout: (Optional) Timeout for each LLM call in seconds (default: 120.0 = 2 min).
                         Prevents hanging when LLM or tools don't respond. Set to None to disable.
            enable_summarization: Enable smart history summarization (default: True)
            summarize_every_n: Summarize after every N iterations (default: 5)
            summarize_token_threshold: Summarize when tokens exceed this % of max_context_tokens (default: 0.7)
            summarizer_model: LLM model for summarization (default: gpt-4.1-mini)
            preserve_last_n_turns: Keep last N turns verbatim when summarizing (default: 2)
            fallback_model: (Optional) Fast/cheap model for simple tasks (e.g., "openai/gpt-4o-mini").
                          When enable_two_tier_routing is True, simple tasks route to this model.
                          Expected cost: ~10x cheaper, latency: ~2-3x faster than primary model.
            enable_two_tier_routing: Enable intelligent model routing (default: False for backward compatibility).
                                   When True, classifies task complexity and routes simple tasks to fallback_model.
                                   Achieves 40-50% cost savings on typical workloads with no quality degradation.
            enable_blackboard: Enable Blackboard Architecture (default: False for backward compatibility).
                             When True, replaces linear chat history with structured state management.
                             Achieves 70-80% token reduction (5000 → 1000 tokens) while preserving all critical context.
                             Recommended for tasks requiring many iterations or extensive reasoning.
            enable_cove: Enable Chain of Verification (default: False for backward compatibility).
                       When True, LLM verifies tool calls before execution by considering assumptions and risks.
                       Achieves 40-50% error reduction with only ~10% overhead.
                       Particularly effective for destructive operations (delete, overwrite, API calls).
            enable_checkpointing: Enable epistemic checkpointing (default: False for backward compatibility).
                                When True, creates snapshots and detects stuck states (same tool 3+ times).
                                Automatically rolls back to previous checkpoint when stuck detected.
                                Prevents wasted computation and infinite loops.
            cove_confidence_threshold: Minimum confidence required to execute tool (default: 0.7).
                                     Tools with verification confidence below this threshold are skipped.
                                     Range: 0.0 (execute everything) to 1.0 (only execute very confident calls).
                                     Recommended: 0.7 for balance, 0.9 for critical operations.
        """
        if not objective or not isinstance(objective, str):
            raise ValueError("Objective must be a non-empty string.")

        # Validate history_mode
        if history_mode not in [mode.value for mode in HistoryMode]:
            raise ValueError(
                f"Invalid history_mode: {history_mode}. Must be one of {[mode.value for mode in HistoryMode]}"
            )

        self.objective = objective
        self.max_internal_steps = max_internal_steps
        self.model_name = model_name
        self.model_params = model_params or {}
        self.history_mode = history_mode
        self.max_context_tokens = max_context_tokens
        self.progress_callback = (
            progress_callback  # T052: Progress callback for TUI updates
        )
        self.user_input_queue = (
            user_input_queue  # Async queue for mid-execution user input
        )
        self.streaming_callback = streaming_callback  # Real-time streaming of reasoning
        self.step_timeout = step_timeout  # Timeout for each LLM call
        self.conversation_history: List[Dict[str, Any]] = (
            []
        )  # Accumulated history for progressive/kitchen_sink modes
        self._interrupt_requested = False  # Flag for graceful interruption

        # Token tracking for real-time display
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Smart context management
        self.enable_summarization = enable_summarization
        self.summarize_every_n = summarize_every_n
        self.summarize_token_threshold = summarize_token_threshold
        self.summarizer_model = summarizer_model
        self.preserve_last_n_turns = preserve_last_n_turns
        self._summarizer: Optional["HistorySummarizer"] = None  # Lazy init
        self._summarization_count = 0  # Track how many times summarization triggered
        self._last_summary: Optional[str] = None  # Store last generated summary

        # Two-tier model routing (Phase 1: Quick Wins - 40% cost savings)
        self.fallback_model = fallback_model
        self.enable_two_tier_routing = enable_two_tier_routing
        self._fast_model_count = 0  # Track how many times fast model used
        self._slow_model_count = 0  # Track how many times slow model used

        # Validate two-tier routing configuration
        if self.enable_two_tier_routing and not self.fallback_model:
            logger.warning(
                "⚠️  Two-tier routing enabled but no fallback_model specified. "
                "Disabling two-tier routing. Set fallback_model (e.g., 'openai/gpt-4o-mini') to enable."
            )
            self.enable_two_tier_routing = False

        # Blackboard Architecture (Phase 2: Token Optimization - 70-80% reduction)
        self.enable_blackboard = enable_blackboard
        self.blackboard = None  # Will be initialized on first use
        if self.enable_blackboard:
            from promptchain.utils.blackboard import Blackboard

            self.blackboard = Blackboard(
                objective=objective,
                max_plan_items=10,
                max_facts=20,
                max_observations=15,
            )
            logger.info(
                "✅ Blackboard Architecture enabled. "
                "Expected: 70-80% token reduction (5000 → 1000 tokens)."
            )

        # Chain of Verification (Phase 3: Safety & Reliability - 40-50% error reduction)
        self.enable_cove = enable_cove
        self.cove_confidence_threshold = cove_confidence_threshold
        self.cove_verifier: Optional[Any] = None  # Lazy init - created on first use
        if self.enable_cove:
            logger.info(
                f"✅ Chain of Verification enabled (threshold={cove_confidence_threshold:.2f}). "
                "Expected: 40-50% error reduction with ~10% overhead."
            )

        # Epistemic Checkpointing (Phase 3: Stuck State Detection & Rollback)
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_manager = None
        if self.enable_checkpointing:
            from promptchain.utils.checkpoint_manager import CheckpointManager

            self.checkpoint_manager = CheckpointManager(stuck_threshold=3)
            logger.info(
                "✅ Epistemic Checkpointing enabled (stuck_threshold=3). "
                "Automatic rollback when same tool called 3+ times."
            )

        # TAO Loop + Dry Runs (Phase 4: Transparent Reasoning)
        self.enable_tao_loop = enable_tao_loop
        self.enable_dry_run = enable_dry_run
        self.dry_run_predictor: Optional[Any] = None  # Lazy init - created on first use
        if self.enable_tao_loop:
            logger.info(
                "✅ TAO Loop enabled (explicit Think-Act-Observe phases). "
                "Provides transparent reasoning structure."
            )
        if self.enable_dry_run:
            logger.info(
                "✅ Dry Run Prediction enabled. "
                "Tool outcomes predicted before execution (~10-15% overhead)."
            )

        # BUG-023 fix: Use warnings.warn() instead of logger.info() for deprecation warnings
        if self.history_mode == HistoryMode.MINIMAL.value:
            warnings.warn(
                "Using 'minimal' history mode (default). This mode may be deprecated in future versions. "
                "Consider using 'progressive' mode for better multi-hop reasoning capabilities.",
                DeprecationWarning,
                stacklevel=2,
            )

        logger.debug(
            f"AgenticStepProcessor initialized with objective: {self.objective[:100]}... "
            f"Model: {self.model_name}, Params: {self.model_params}, "
            f"History Mode: {self.history_mode}, Max Tokens: {self.max_context_tokens}"
        )

    @property
    def current_step(self) -> int:
        """Return the number of reasoning steps executed so far.

        This property allows tests and external code to track progress
        through the internal reasoning loop. Returns 0 if no steps have
        been executed yet.

        Returns:
            int: Number of internal reasoning steps completed
        """
        return getattr(self, "steps_executed", 0)

    @property
    def internal_history(self) -> List[Dict[str, Any]]:
        """Return the internal reasoning history for inspection.

        This property exposes the full internal conversation history
        that occurred during the reasoning loop. Useful for debugging
        and observability.

        Note: This is separate from conversation_history (which is used
        for progressive/kitchen_sink history modes). This returns the
        complete internal_history tracked during execution.

        Returns:
            List[Dict[str, Any]]: Internal conversation history, empty list if not yet executed
        """
        return getattr(self, "_internal_history", [])

    async def _check_user_input(
        self, internal_history: List[Dict], llm_history: List[Dict]
    ) -> Optional[str]:
        """Check for and process user input from the queue.

        Non-blocking check - returns immediately if no input available.
        If input is found, injects it into the conversation history.

        Returns:
            User input string if available, None otherwise
        """
        if not self.user_input_queue:
            return None

        try:
            # Non-blocking get
            user_input = self.user_input_queue.get_nowait()

            if user_input:
                # Log the mid-execution input
                logger.info(f"Received mid-execution user input: {user_input[:100]}...")

                # Inject as user message
                user_msg = {
                    "role": "user",
                    "content": f"[User interrupt]: {user_input}",
                }
                internal_history.append(user_msg)
                llm_history.append(user_msg)

                # Stream the input event
                self._stream_event("user_input", user_input)

                return user_input
        except asyncio.QueueEmpty:
            pass
        except Exception as e:
            logger.warning(f"Error checking user input queue: {e}")

        return None

    def _stream_event(self, event_type: str, content: str) -> None:
        """Stream an event to the streaming callback if configured.

        Args:
            event_type: Type of event ("thinking", "tool_call", "tool_result", "answer", "user_input")
            content: Content to stream
        """
        if self.streaming_callback:
            try:
                self.streaming_callback(event_type, content)
            except Exception as e:
                logger.warning(f"Error in streaming callback: {e}")

    def _emit_token_usage(self, response) -> None:
        """Extract token usage from LLM response and emit to streaming callback.

        Args:
            response: LLM response object (LiteLLM ModelResponse or dict)
        """
        if not self.streaming_callback:
            return

        try:
            # Extract usage from response - try multiple approaches
            usage = None

            # Method 1: Direct attribute access (LiteLLM ModelResponse)
            if hasattr(response, "usage"):
                usage = response.usage
            # Method 2: Dict access
            elif isinstance(response, dict) and "usage" in response:
                usage = response["usage"]
            # Method 3: model_dump() for Pydantic objects
            elif hasattr(response, "model_dump"):
                try:
                    response_dict = response.model_dump()
                    usage = response_dict.get("usage")
                except Exception:
                    pass

            if usage:
                # Extract token counts
                prompt_tokens = 0
                completion_tokens = 0

                if hasattr(usage, "prompt_tokens"):
                    prompt_tokens = usage.prompt_tokens or 0
                elif isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)

                if hasattr(usage, "completion_tokens"):
                    completion_tokens = usage.completion_tokens or 0
                elif isinstance(usage, dict):
                    completion_tokens = usage.get("completion_tokens", 0)

                # Accumulate totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

                # Emit token event
                token_data = json.dumps(
                    {
                        "prompt_tokens": self.total_prompt_tokens,
                        "completion_tokens": self.total_completion_tokens,
                        "last_prompt": prompt_tokens,
                        "last_completion": completion_tokens,
                    }
                )
                self._stream_event("tokens", token_data)
                logger.debug(
                    f"Emitted token usage: prompt={self.total_prompt_tokens}, completion={self.total_completion_tokens}"
                )

        except Exception as e:
            logger.debug(f"Failed to emit token usage: {e}")

    def request_interrupt(self) -> None:
        """Request graceful interruption of the current execution.

        The processor will stop at the next safe point and return partial results.
        """
        self._interrupt_requested = True
        logger.info("Interrupt requested - will stop at next safe point")

    def _get_summarizer(self) -> "HistorySummarizer":
        """Get or create the history summarizer (lazy initialization).

        Returns:
            HistorySummarizer instance
        """
        if self._summarizer is None:
            from .history_summarizer import HistorySummarizer

            self._summarizer = HistorySummarizer(
                model=self.summarizer_model,
                max_summary_tokens=500,
            )
        return self._summarizer

    def _should_summarize(self, iteration: int, current_tokens: int) -> bool:
        """Check if history should be summarized based on triggers.

        Triggers on EITHER:
        1. After every N iterations (summarize_every_n)
        2. When tokens exceed threshold % of max_context_tokens

        Args:
            iteration: Current iteration number (0-indexed)
            current_tokens: Current estimated token count

        Returns:
            True if summarization should trigger
        """
        if not self.enable_summarization:
            return False

        # Only applicable for progressive/kitchen_sink modes
        if self.history_mode == HistoryMode.MINIMAL.value:
            return False

        # Trigger 1: Iteration-based (after every N iterations, starting from iteration N)
        iteration_trigger = (iteration > 0) and (
            (iteration + 1) % self.summarize_every_n == 0
        )

        # Trigger 2: Token-based (when exceeding threshold)
        token_trigger = False
        if self.max_context_tokens and current_tokens > 0:
            threshold = self.max_context_tokens * self.summarize_token_threshold
            token_trigger = current_tokens >= threshold

        should_summarize = iteration_trigger or token_trigger

        if should_summarize:
            trigger_reason = []
            if iteration_trigger:
                trigger_reason.append(f"iteration {iteration + 1}")
            if token_trigger:
                trigger_reason.append(f"tokens {current_tokens} >= {threshold:.0f}")
            logger.info(f"Summarization triggered: {', '.join(trigger_reason)}")

        return should_summarize

    async def _summarize_conversation_history(self) -> bool:
        """Summarize the accumulated conversation history.

        Compresses self.conversation_history while preserving recent turns.
        Updates self.conversation_history with summarized version.

        Returns:
            True if summarization was performed, False otherwise
        """
        if len(self.conversation_history) <= self.preserve_last_n_turns:
            logger.debug("Not enough history to summarize")
            return False

        try:
            summarizer = self._get_summarizer()

            # Stream summarization event
            self._stream_event("thinking", "Compressing conversation history...")

            result = await summarizer.summarize_history(
                entries=self.conversation_history,
                preserve_last_n=self.preserve_last_n_turns,
            )

            if result.summary:
                # Create summary message
                summary_msg = {
                    "role": "system",
                    "content": f"[Summary of previous {result.entries_summarized} exchanges]\n{result.summary}",
                }

                # Preserve recent turns
                preserved_entries = (
                    self.conversation_history[-self.preserve_last_n_turns :]
                    if self.preserve_last_n_turns > 0
                    else []
                )

                # Replace history with summary + preserved turns
                self.conversation_history = [summary_msg] + preserved_entries

                # Track summarization
                self._summarization_count += 1
                self._last_summary = result.summary

                logger.info(
                    f"History summarized: {result.original_tokens} -> {result.summary_tokens} tokens "
                    f"({result.savings_percent:.1f}% reduction)"
                )

                # Stream completion
                self._stream_event(
                    "thinking",
                    f"History compressed: {result.savings_percent:.0f}% reduction",
                )

                return True
            else:
                logger.warning("Summarization returned empty summary")
                return False

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Continue without summarization - don't break execution
            return False

    def _classify_task_complexity(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], step_num: int
    ) -> str:
        """Classify task complexity to route between fast/slow models.

        This heuristic classifier decides whether a task is "simple" enough
        for the fallback model or requires the primary model's capabilities.

        Classification Heuristics (Research-Based):
        1. Tool-free reasoning → Simple (no execution risk)
        2. Single tool with <3 args → Simple (straightforward operation)
        3. Multiple tools or complex args → Complex (requires careful reasoning)
        4. Early iterations (1-2) → Complex (planning phase)
        5. Later iterations with progress → Simple (execution phase)

        Args:
            messages: Current conversation history
            tools: Available tools for this call
            step_num: Current iteration number (0-indexed)

        Returns:
            "simple" or "complex"
        """
        # Early iterations (planning phase) - use primary model
        if step_num < 2:
            logger.debug(
                f"[Two-Tier] Step {step_num}: Early planning phase → COMPLEX (use primary model)"
            )
            return "complex"

        # Check if this is likely a tool-free reasoning step
        last_message = messages[-1] if messages else {}
        last_content = last_message.get("content", "")

        # Patterns indicating complex reasoning
        complex_patterns = [
            "think",
            "analyze",
            "plan",
            "strategy",
            "approach",
            "why",
            "how",
            "multiple",
            "several",
            "compare",
        ]

        if any(pattern in last_content.lower() for pattern in complex_patterns):
            logger.debug(
                f"[Two-Tier] Step {step_num}: Complex reasoning detected → COMPLEX"
            )
            return "complex"

        # Simple patterns (execution, status updates)
        simple_patterns = [
            "list",
            "get",
            "show",
            "display",
            "status",
            "next",
            "continue",
            "done",
            "complete",
        ]

        if any(pattern in last_content.lower() for pattern in simple_patterns):
            logger.debug(
                f"[Two-Tier] Step {step_num}: Simple execution detected → SIMPLE (use fallback)"
            )
            return "simple"

        # Default: Later iterations likely executing plan → Simple
        logger.debug(
            f"[Two-Tier] Step {step_num}: Late iteration, default → SIMPLE (use fallback)"
        )
        return "simple"

    async def _smart_llm_call(
        self,
        llm_runner: Callable[..., Awaitable[Any]],
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str,
        step_num: int,
        model_override: Optional[str] = None,
    ) -> Any:
        """Intelligently route LLM call to fast or slow model based on complexity.

        This wrapper implements two-tier model routing:
        - Simple tasks → fallback_model (fast/cheap)
        - Complex tasks → primary model (slow/expensive)

        Args:
            llm_runner: LLM call function
            messages: Conversation history
            tools: Available tools
            tool_choice: Tool selection strategy
            step_num: Current iteration number
            model_override: Optional model override (bypasses routing)

        Returns:
            LLM response message
        """
        # If two-tier routing disabled, use primary model
        if not self.enable_two_tier_routing:
            return await llm_runner(
                messages=messages, tools=tools, tool_choice=tool_choice
            )

        # If model override specified, respect it
        if model_override:
            logger.debug(f"[Two-Tier] Using model override: {model_override}")
            return await llm_runner(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                model=model_override,
            )

        # Classify task complexity
        complexity = self._classify_task_complexity(messages, tools, step_num)

        # Route based on complexity
        if complexity == "simple" and self.fallback_model:
            # Use fast/cheap model
            self._fast_model_count += 1
            logger.info(
                f"[Two-Tier] Step {step_num}: SIMPLE task → using fallback model '{self.fallback_model}' "
                f"(saved {self._fast_model_count}/{self._fast_model_count + self._slow_model_count} calls)"
            )
            return await llm_runner(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                model=self.fallback_model,
            )
        else:
            # Use primary model
            self._slow_model_count += 1
            logger.info(
                f"[Two-Tier] Step {step_num}: COMPLEX task → using primary model '{self.model_name or 'default'}'"
            )
            return await llm_runner(
                messages=messages, tools=tools, tool_choice=tool_choice
            )

    def _ensure_cove_verifier(self, llm_runner: Callable[..., Awaitable[Any]]):
        """
        Lazy initialization of CoVe verifier.

        Args:
            llm_runner: LLM runner function to use for verification
        """
        if self.enable_cove and self.cove_verifier is None:
            from promptchain.utils.verification import CoVeVerifier

            # Use fallback model for verification (cheaper/faster) or primary if not available
            verification_model = (
                self.fallback_model if self.fallback_model else self.model_name
            ) or ""
            self.cove_verifier = CoVeVerifier(llm_runner, verification_model)
            logger.info(f"[CoVe] Initialized verifier with model: {verification_model}")

    async def run_async(
        self,
        initial_input: str,
        available_tools: List[Dict[str, Any]],
        llm_runner: Callable[
            ..., Awaitable[Any]
        ],  # Should return LiteLLM-like response message
        tool_executor: Callable[
            [Any], Awaitable[str]
        ],  # Takes tool_call, returns result string
        return_metadata: bool = False,  # NEW: Return metadata instead of just string
        callback_manager: Optional[
            Any
        ] = None,  # ✅ NEW (v0.4.2): Observability support
        user_input_queue: Optional[
            asyncio.Queue
        ] = None,  # Runtime override for mid-execution input
        streaming_callback: Optional[
            Callable[[str, str], None]
        ] = None,  # Runtime override for streaming
        step_timeout: Optional[
            float
        ] = None,  # Runtime override for timeout (default: use init value)
    ) -> Union[str, AgenticStepResult]:
        """
        Executes the internal agentic loop for this step with optional metadata return.

        Args:
            initial_input: The input string received from the previous PromptChain step.
            available_tools: List of tool schemas available for this step.
            llm_runner: An async function to call the LLM.
                        Expected signature: llm_runner(messages, tools, tool_choice) -> response_message
            tool_executor: An async function to execute a tool call.
                           Expected signature: tool_executor(tool_call) -> result_content_string
            return_metadata: If True, return AgenticStepResult with full execution metadata.
                           If False, return just the final answer string (backward compatible).

        Returns:
            - str: Just final answer (default, backward compatible)
            - AgenticStepResult: Full execution metadata when return_metadata=True
        """
        # Apply runtime overrides for queue, streaming, and timeout (if provided)
        if user_input_queue is not None:
            self.user_input_queue = user_input_queue
        if streaming_callback is not None:
            self.streaming_callback = streaming_callback
        if step_timeout is not None:
            self.step_timeout = step_timeout

        # Reset interrupt flag for new execution
        self._interrupt_requested = False

        # Store available tools as an attribute for inspection/debugging
        self.available_tools = available_tools

        # Log the tool names for proof
        tool_names = [t["function"]["name"] for t in available_tools if "function" in t]
        logger.info(f"[AgenticStepProcessor] Available tools at start: {tool_names}")

        logger.info(f"Starting agentic step. Objective: {self.objective[:100]}...")
        logger.debug(f"Initial input: {initial_input[:100]}...")

        # Metadata tracking variables
        execution_start_time = datetime.now()
        steps_metadata: List[StepExecutionMetadata] = []
        total_tools_called = 0
        total_tokens_used = 0
        execution_errors: List[str] = []
        execution_warnings: List[str] = []

        # Full internal history for debugging/tracking (not sent to LLM)
        internal_history: List[Dict[str, Any]] = []
        # Minimal valid LLM message history (sent to LLM)
        llm_history: List[Dict[str, Any]] = []

        # Start with the objective and initial input
        system_prompt = f"""Your goal is to achieve the following objective: {self.objective}

REACT WORKFLOW (Reason-Act-Observe):
You MUST follow this pattern for EVERY request:

STEP 1 - THINK: Analyze the request
- What is the user actually asking for?
- What type of task is this?
- What information or actions are needed?

STEP 2 - PLAN: Use task_list_write_tool to create your plan
IMMEDIATELY call task_list_write_tool with your planned tasks:
```
task_list_write_tool([
  {{"content": "Task description", "status": "pending", "activeForm": "Doing task..."}},
  {{"content": "Second task", "status": "pending", "activeForm": "Doing second task..."}}
])
```
This MUST be your FIRST tool call - always create a plan before executing.

STEP 3 - ACT: Execute one task at a time
- Mark task as "in_progress" when starting
- Use the appropriate tool for the current task
- Wait for the result

STEP 4 - OBSERVE: Check result and update plan
- Did it succeed? Mark task "completed"
- Error? Add recovery tasks to the list
- New information? Adjust remaining tasks

STEP 5 - REPEAT: Continue until all tasks complete

AVAILABLE TOOLS (choose correctly):
- task_list_write_tool: REQUIRED first - create/update your task plan
- ripgrep_search: Search LOCAL files and code (NOT for web searches!)
- file_read/file_write/file_edit: File operations
- terminal_execute: Run shell commands
- list_directory/create_directory: Directory operations
- sandbox_*: Safe code execution environment

MCP TOOLS (external services via Model Context Protocol):
- mcp_gemini_gemini_research: WEB SEARCH - Use this to search the internet for current information
- mcp_gemini_ask_gemini: Ask Gemini general questions or get a second opinion
- mcp_gemini_gemini_brainstorm: Creative brainstorming and idea generation
- mcp_gemini_gemini_debug: Debug errors with Gemini's help
- mcp_gemini_gemini_code_review: Get code review feedback

IMPORTANT TOOL SELECTION:
- "search the web/internet/online" -> USE mcp_gemini_gemini_research (NOT ripgrep!)
- "find packages/libraries/docs online" -> USE mcp_gemini_gemini_research
- "find in files/code locally" -> Use ripgrep_search
- "read a file" -> Use file_read
- "run a command" -> Use terminal_execute
- "run a script" -> Use terminal_execute('python /path/to/script.py')

EXECUTING SCRIPTS - ALWAYS RUN WHEN ASKED:
When the user asks you to "run", "execute", or "create and run" a script:
1. Create the script file with file_write
2. THEN run it with terminal_execute('python /absolute/path/to/script.py')
3. Show the actual OUTPUT from the script execution
4. DO NOT just create the script and tell the user to run it themselves
5. If the script generates files (HTML, images, etc.), report the output paths

Example workflow for "create a Plotly chart":
1. file_write('/path/to/chart.py', code_content)
2. terminal_execute('python /path/to/chart.py')
3. Show the script output AND confirm the generated file exists

PATH HANDLING - ABSOLUTE PATHS REQUIRED:
ALWAYS use and return FULL ABSOLUTE PATHS (e.g., /home/user/project/file.py), NEVER relative paths (e.g., ../file.py, ./src).

Tools for path operations:
- resolve_path: Convert ANY path to absolute path (use before reporting paths to user)
- find_paths: Find files/dirs and get their absolute paths
- get_cwd: Get current working directory
- path_info: Check if path exists and get full info

SECURITY MODES - Path boundary handling depends on session security mode:
- STRICT: All outside-directory paths require explicit confirmation. Tool returns `requires_confirmation: true`.
- DEFAULT: First access shows warning, then auto-allows subsequent access. Tool returns warning message once.
- TRUSTED: No boundary warnings - all paths accessible without confirmation.

When you receive a path warning or `requires_confirmation: true`:
1. Inform user the path is outside working directory
2. In STRICT mode: Wait for explicit confirmation before proceeding
3. In DEFAULT/TRUSTED mode: Proceed but mention the location to user

When reporting file/directory locations to the user:
- WRONG: "Found it at ../hybridrag"
- CORRECT: "Found it at /home/user/Documents/code/hybridrag"

CRITICAL:
- ALWAYS call task_list_write_tool FIRST to show your plan
- Update task status as you work (pending -> in_progress -> completed)
- If a task fails, add a recovery task and continue
- Only give final answer after completing all tasks with real tool results

FINAL ANSWER REQUIREMENTS:
- Your final answer MUST include the FULL content/information from tool results
- DO NOT just say "I have explained" or "I have provided" - actually SHOW the information
- If a tool returns documentation, code examples, or explanations - include that content in your response
- If you used mcp_gemini_ask_gemini or similar - include the actual answer Gemini gave, not a summary
- The user cannot see tool results directly - they only see YOUR final response
- WRONG: "I have explained how to add MCP servers" (user sees nothing useful)
- CORRECT: Include the actual steps, code, and explanations from the tool response"""
        system_message = {"role": "system", "content": system_prompt}
        user_message = (
            {
                "role": "user",
                "content": f"The initial input for this step is: {initial_input}",
            }
            if initial_input
            else None
        )
        internal_history.append(system_message)
        if user_message:
            internal_history.append(user_message)
        # Initialize llm_history for first call
        llm_history = [system_message]
        if user_message:
            llm_history.append(user_message)

        final_answer = None
        last_tool_call_assistant_msg = None
        last_tool_msgs: List[Dict[str, Any]] = []
        clarification_attempts = 0
        max_clarification_attempts = 3

        for step_num in range(self.max_internal_steps):
            # Check for interrupt request at start of each step
            if self._interrupt_requested:
                logger.info("Interrupt detected - stopping execution gracefully")
                execution_warnings.append("Execution interrupted by user")
                final_answer = (
                    "[Execution interrupted by user. Partial results may be available.]"
                )
                break

            logger.info(
                f"Agentic step internal iteration {step_num + 1}/{self.max_internal_steps}"
            )

            # T052: Invoke progress callback at start of each step
            if self.progress_callback:
                self.progress_callback(
                    step_num + 1, self.max_internal_steps, "Reasoning..."
                )

            # Phase 2: Increment Blackboard iteration counter
            if self.enable_blackboard and self.blackboard:
                self.blackboard.increment_iteration()

            # Phase 3: Create checkpoint for potential rollback
            if (
                self.enable_checkpointing
                and self.checkpoint_manager
                and self.blackboard
            ):
                snapshot_id = self.blackboard.snapshot()
                confidence = self.blackboard.get_state().get("confidence", 1.0)
                self.checkpoint_manager.create_checkpoint(
                    iteration=step_num,
                    blackboard_snapshot=snapshot_id,
                    confidence=confidence,
                )

            # Stream thinking event
            self._stream_event(
                "thinking",
                f"Step {step_num + 1}: Analyzing and planning next action...",
            )

            # Track metadata for this step
            step_start_time = datetime.now()
            step_tool_calls: List[Dict[str, Any]] = []
            step_clarification_attempts = 0
            step_error: Optional[str] = None

            # Note: Event emission moved to END of iteration to capture actual activity

            # Phase 4: TAO Loop - Explicit Think-Act-Observe (if enabled)
            if self.enable_tao_loop:
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # TAO LOOP EXECUTION (Phase 4: Transparent Reasoning)
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                logger.info(f"[TAO] Starting TAO loop for iteration {step_num}")

                # Build LLM history for this iteration
                if self.enable_blackboard and self.blackboard:
                    blackboard_summary = self.blackboard.to_prompt()
                    llm_history_for_tao = [
                        system_message,
                        {
                            "role": "user",
                            "content": f"{user_message['content'] if user_message else ''}\n\nCURRENT STATE:\n{blackboard_summary}",
                        },
                    ]
                elif self.history_mode == HistoryMode.MINIMAL.value:
                    llm_history_for_tao = [system_message]
                    if user_message:
                        llm_history_for_tao.append(user_message)
                else:
                    llm_history_for_tao = [system_message] + internal_history[-10:]

                # THINK phase: LLM reasons about next action
                think_result = await self._tao_think_phase(
                    llm_runner=llm_runner,
                    llm_history=llm_history_for_tao,
                    iteration_count=step_num,
                    available_tools=available_tools,
                )

                response_message = think_result["response"]
                tool_calls = think_result["tool_calls"]
                has_tool_calls = think_result["has_tool_calls"]

                # Track token usage
                self._emit_token_usage(response_message)

                # Add assistant's response to history
                internal_history.append(
                    response_message
                    if isinstance(response_message, dict)
                    else {"role": "assistant", "content": str(response_message)}
                )

                # DEBUG: Log the values
                logger.debug(
                    f"[TAO] DEBUG: has_tool_calls={has_tool_calls}, tool_calls={tool_calls}, len={len(tool_calls) if tool_calls else 0}"
                )

                if has_tool_calls and tool_calls:
                    # ACT phase: Execute tools (reuse existing tool execution code)
                    # Note: We'll use the existing tool execution infrastructure below
                    # but wrap it with TAO observation

                    # Store original tool calls for TAO observe phase
                    tao_tool_calls = tool_calls
                    logger.info(
                        f"[TAO] Stored {len(tool_calls)} tool calls for execution"
                    )

                    # Continue to existing tool execution code (lines 1090+)
                    # The existing CoVe verification and tool execution will handle this
                    pass  # Execution continues below in existing loop
                else:
                    # No tool calls - check for completion
                    logger.warning(
                        f"[TAO] No tool calls to execute: has_tool_calls={has_tool_calls}, tool_calls={tool_calls}"
                    )
                    tool_calls = None

                # The TAO observe phase will be called after tool execution (after line 1315)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # STANDARD ReAct LOOP (backward compatible, default behavior)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            while True:
                # TAO Loop bypass: If TAO is enabled, skip ReAct loop (TAO already executed THINK phase above)
                if self.enable_tao_loop:
                    # TAO has already set tool_calls and response_message above
                    # Break out of while True loop to proceed to tool execution
                    break

                # Check for user input before LLM call
                user_input = await self._check_user_input(internal_history, llm_history)
                if user_input:
                    logger.info(
                        f"Processing user input mid-execution: {user_input[:50]}..."
                    )
                    # Continue loop to process the new input with LLM
                # Prepare minimal valid messages for LLM
                messages_for_llm = llm_history[:]
                # Log the full structure of messages and tools for debugging
                logger.debug(
                    f"Messages sent to LLM: {json.dumps(messages_for_llm, indent=2)}"
                )
                logger.debug(
                    f"Tools sent to LLM: {json.dumps(available_tools, indent=2)}"
                )
                try:
                    # Call the LLM to decide next action or give final answer
                    logger.debug("Calling LLM for next action/final answer...")

                    # Get tool_choice from model_params if available, otherwise default to "auto"
                    tool_choice = self.model_params.get("tool_choice", "auto")

                    # Use smart LLM routing (two-tier model selection based on complexity)
                    # This wraps the llm_runner and intelligently routes to fast/slow model
                    llm_coro = self._smart_llm_call(
                        llm_runner=llm_runner,
                        messages=messages_for_llm,
                        tools=available_tools,
                        tool_choice=tool_choice,
                        step_num=step_num,
                    )

                    if self.step_timeout:
                        try:
                            response_message = await asyncio.wait_for(
                                llm_coro, timeout=self.step_timeout
                            )
                        except asyncio.TimeoutError:
                            timeout_msg = (
                                f"LLM call timed out after {self.step_timeout} seconds"
                            )
                            logger.error(timeout_msg)
                            self._stream_event("error", timeout_msg)
                            execution_errors.append(timeout_msg)
                            final_answer = f"[Error: {timeout_msg}. Please try again or simplify your request.]"
                            break
                    else:
                        response_message = await llm_coro

                    # Extract and emit token usage for real-time display
                    self._emit_token_usage(response_message)

                    # Add more detailed debugging
                    if isinstance(response_message, dict):
                        logger.debug(
                            f"Response message (dict): {json.dumps(response_message)}"
                        )
                    else:
                        logger.debug(
                            f"Response message (object): type={type(response_message)}, dir={dir(response_message)}"
                        )
                        if hasattr(response_message, "model_dump"):
                            logger.debug(
                                f"Response message dump: {json.dumps(response_message.model_dump())}"
                            )

                    # Append assistant's response (thought process + potential tool call) to internal history
                    internal_history.append(
                        response_message
                        if isinstance(response_message, dict)
                        else {"role": "assistant", "content": str(response_message)}
                    )

                    # Enhanced tool call detection - try multiple approaches
                    tool_calls = None

                    # Method 1: Try standard attribute/key access
                    if isinstance(response_message, dict):
                        tool_calls = response_message.get("tool_calls")
                    else:
                        tool_calls = getattr(response_message, "tool_calls", None)

                    # Method 2: For LiteLLM objects, try nested access via model_dump
                    if tool_calls is None and hasattr(response_message, "model_dump"):
                        try:
                            response_dict = response_message.model_dump()
                            if isinstance(response_dict, dict):
                                tool_calls = response_dict.get("tool_calls")
                        except Exception as dump_err:
                            logger.warning(f"Error accessing model_dump: {dump_err}")

                    # Log the detected tool calls for debugging
                    if tool_calls:
                        logger.debug(f"Detected tool calls: {tool_calls}")

                    if tool_calls:
                        clarification_attempts = 0  # Reset on tool call
                        logger.info(f"LLM requested {len(tool_calls)} tool(s).")

                        # T052: Progress callback for tool calling phase with tool names
                        tool_names = []
                        for tc in tool_calls:
                            fn = get_function_name_from_tool_call(tc)
                            if fn:
                                tool_names.append(fn)
                        tool_names_str = ", ".join(tool_names[:3])  # Show first 3
                        if len(tool_names) > 3:
                            tool_names_str += f" +{len(tool_names) - 3} more"

                        if self.progress_callback:
                            self.progress_callback(
                                step_num + 1,
                                self.max_internal_steps,
                                (
                                    f"Calling: {tool_names_str}"
                                    if tool_names_str
                                    else f"Calling {len(tool_calls)} tool(s)"
                                ),
                            )

                        # Prepare new minimal LLM history: system, user, assistant (with tool_calls), tool(s)
                        last_tool_call_assistant_msg = (
                            response_message
                            if isinstance(response_message, dict)
                            else {"role": "assistant", "content": str(response_message)}
                        )
                        last_tool_msgs = []

                        # Phase 3: Chain of Verification (CoVe) - verify tools before execution
                        if self.enable_cove:
                            # Lazy init CoVe verifier
                            self._ensure_cove_verifier(llm_runner)
                            assert (
                                self.cove_verifier is not None
                            )  # guaranteed by _ensure_cove_verifier

                            verified_tool_calls = []
                            for tool_call in tool_calls:
                                function_name = get_function_name_from_tool_call(
                                    tool_call
                                )
                                if not function_name:
                                    verified_tool_calls.append(
                                        tool_call
                                    )  # Can't verify, allow it
                                    continue

                                # Parse arguments
                                if isinstance(tool_call, dict):
                                    tool_call_args = tool_call.get("function", {}).get(
                                        "arguments", "{}"
                                    )
                                else:
                                    func_obj = getattr(tool_call, "function", None)
                                    tool_call_args = (
                                        getattr(func_obj, "arguments", "{}")
                                        if func_obj
                                        else "{}"
                                    )

                                # Parse JSON args
                                try:
                                    tool_args_dict = (
                                        json.loads(tool_call_args)
                                        if isinstance(tool_call_args, str)
                                        else tool_call_args
                                    )
                                except json.JSONDecodeError:
                                    tool_args_dict = {}

                                # Get context for verification
                                context = (
                                    self.blackboard.to_prompt()
                                    if (self.enable_blackboard and self.blackboard)
                                    else "No context available"
                                )

                                # Verify tool call
                                verification = (
                                    await self.cove_verifier.verify_tool_call(
                                        tool_name=function_name,
                                        tool_args=tool_args_dict,
                                        context=context,
                                        available_tools=available_tools,
                                    )
                                )

                                # Check confidence threshold
                                if (
                                    verification.should_execute
                                    and verification.confidence
                                    >= self.cove_confidence_threshold
                                ):
                                    # Apply suggested modifications if any
                                    if verification.suggested_modifications:
                                        # Merge modifications into tool args
                                        modified_args = {
                                            **tool_args_dict,
                                            **verification.suggested_modifications,
                                        }
                                        # Update tool call arguments
                                        if isinstance(tool_call, dict):
                                            tool_call["function"]["arguments"] = (
                                                json.dumps(modified_args)
                                            )
                                        logger.info(
                                            f"[CoVe] Applied modifications to {function_name}"
                                        )

                                    verified_tool_calls.append(tool_call)
                                    logger.info(
                                        f"[CoVe] ✅ Approved {function_name} (confidence={verification.confidence:.2f})"
                                    )
                                else:
                                    # Log skipped tool
                                    logger.warning(
                                        f"[CoVe] ❌ Skipped {function_name} "
                                        f"(confidence={verification.confidence:.2f}, threshold={self.cove_confidence_threshold:.2f})"
                                    )
                                    if self.blackboard:
                                        self.blackboard.add_observation(
                                            f"Skipped {function_name}: {verification.verification_reasoning[:100]}"
                                        )

                            # Replace tool calls with verified subset
                            tool_calls = verified_tool_calls
                            logger.info(
                                f"[CoVe] Verified {len(verified_tool_calls)}/{len(tool_calls)} tool calls"
                            )

                        # TAO ACT phase: Wrap tool execution with MLflow span (if TAO enabled)
                        if self.enable_tao_loop and MLFLOW_AVAILABLE:
                            act_span_context = mlflow.start_span(
                                name=f"TAO_ACT_iter_{step_num}", span_type="TOOL"
                            )
                            act_span = act_span_context.__enter__()
                        else:
                            act_span_context = None
                            act_span = None

                        try:
                            if self.enable_tao_loop:
                                logger.info(
                                    f"[TAO] ⚡ ACT phase (iteration {step_num}, {len(tool_calls)} tool calls)"
                                )

                            # Execute tools sequentially for simplicity
                            for tool_call in tool_calls:
                                # Extract tool call ID and function name using the helper function
                                if isinstance(tool_call, dict):
                                    tool_call_id = tool_call.get("id")
                                    tool_call_args = tool_call.get("function", {}).get(
                                        "arguments", {}
                                    )
                            else:
                                tool_call_id = getattr(tool_call, "id", None)
                                func_obj = getattr(tool_call, "function", None)
                                tool_call_args = (
                                    getattr(func_obj, "arguments", {})
                                    if func_obj
                                    else {}
                                )

                            function_name = get_function_name_from_tool_call(tool_call)

                            if not function_name:
                                logger.error(
                                    f"Could not extract function name from tool call: {tool_call}"
                                )
                                tool_result_content = json.dumps(
                                    {
                                        "error": "Could not determine function name for tool execution"
                                    }
                                )
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": "unknown_function",
                                    "content": tool_result_content,
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                                # Track failed tool call
                                step_tool_calls.append(
                                    {
                                        "name": "unknown_function",
                                        "args": {},
                                        "result": tool_result_content,
                                        "time_ms": 0,
                                        "error": "Could not determine function name",
                                    }
                                )
                                continue

                            logger.info(
                                f"Executing tool: {function_name} (ID: {tool_call_id})"
                            )

                            # Stream tool call event
                            self._stream_event(
                                "tool_call",
                                f"{function_name}: {str(tool_call_args)[:200]}",
                            )

                            tool_exec_start = datetime.now()
                            try:
                                # Use the provided executor callback with timeout
                                tool_coro = tool_executor(tool_call)
                                if self.step_timeout:
                                    try:
                                        tool_result_content = await asyncio.wait_for(
                                            tool_coro, timeout=self.step_timeout
                                        )
                                    except asyncio.TimeoutError:
                                        tool_exec_time = (
                                            datetime.now() - tool_exec_start
                                        ).total_seconds() * 1000
                                        timeout_msg = f"Tool {function_name} timed out after {self.step_timeout}s"
                                        logger.error(timeout_msg)
                                        self._stream_event("error", timeout_msg)
                                        tool_result_content = f"Error: {timeout_msg}"
                                        tool_msg = {
                                            "role": "tool",
                                            "tool_call_id": tool_call_id,
                                            "name": function_name,
                                            "content": tool_result_content,
                                        }
                                        internal_history.append(tool_msg)
                                        last_tool_msgs.append(tool_msg)
                                        step_tool_calls.append(
                                            {
                                                "name": function_name,
                                                "args": tool_call_args,
                                                "result": tool_result_content,
                                                "time_ms": tool_exec_time,
                                                "error": timeout_msg,
                                            }
                                        )
                                        execution_errors.append(timeout_msg)
                                        continue  # Skip to next tool
                                else:
                                    tool_result_content = await tool_coro

                                tool_exec_time = (
                                    datetime.now() - tool_exec_start
                                ).total_seconds() * 1000
                                logger.info(
                                    f"Tool {function_name} executed successfully."
                                )
                                logger.debug(
                                    f"Tool result content: {tool_result_content[:150]}..."
                                )

                                # Stream tool result event
                                result_preview = str(tool_result_content)[:500]
                                self._stream_event(
                                    "tool_result",
                                    f"{function_name} completed: {result_preview}",
                                )

                                # Append tool result to internal history
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": function_name,
                                    "content": tool_result_content,
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                                # Track successful tool call
                                step_tool_calls.append(
                                    {
                                        "name": function_name,
                                        "args": tool_call_args,
                                        "result": tool_result_content,
                                        "time_ms": tool_exec_time,
                                    }
                                )
                                total_tools_called += 1
                            except Exception as tool_exec_error:
                                tool_exec_time = (
                                    datetime.now() - tool_exec_start
                                ).total_seconds() * 1000
                                logger.error(
                                    f"Error executing tool {function_name}: {tool_exec_error}",
                                    exc_info=True,
                                )
                                error_msg = str(tool_exec_error)
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": function_name,
                                    "content": f"Error executing tool: {error_msg}",
                                }
                                internal_history.append(tool_msg)
                                last_tool_msgs.append(tool_msg)
                                # Track failed tool call
                                step_tool_calls.append(
                                    {
                                        "name": function_name,
                                        "args": tool_call_args,
                                        "result": f"Error: {error_msg}",
                                        "time_ms": tool_exec_time,
                                        "error": error_msg,
                                    }
                                )
                                execution_errors.append(
                                    f"Tool {function_name}: {error_msg}"
                                )

                            # Log ACT phase metrics to MLflow span
                            if self.enable_tao_loop and act_span:
                                act_span.set_attribute("iteration", step_num)
                                act_span.set_attribute(
                                    "tools_executed", len(last_tool_msgs)
                                )
                                act_span.set_attribute(
                                    "total_tool_calls", len(tool_calls)
                                )
                                logger.info(
                                    f"[TAO] ACT complete: {len(last_tool_msgs)} tools executed"
                                )

                        finally:
                            # Close ACT phase MLflow span
                            if act_span_context:
                                act_span_context.__exit__(None, None, None)

                        # T052: Progress callback for synthesis phase
                        if self.progress_callback:
                            self.progress_callback(
                                step_num + 1,
                                self.max_internal_steps,
                                "Synthesizing results...",
                            )

                        # Phase 2: Update Blackboard with tool execution results
                        if self.enable_blackboard and self.blackboard:
                            for tool_msg in last_tool_msgs:
                                # Extract tool name from the tool call
                                # Tool messages have tool_call_id, we need to match with tool_calls
                                for tool_call in tool_calls:
                                    if tool_msg.get("tool_call_id") == tool_call.get(
                                        "id"
                                    ):
                                        function_name = tool_call.get(
                                            "function", {}
                                        ).get("name", "unknown")
                                        result_content = tool_msg.get("content", "")
                                        self.blackboard.store_tool_result(
                                            function_name, result_content
                                        )
                                        self.blackboard.add_observation(
                                            f"Executed {function_name}"
                                        )
                                        break
                            logger.debug(
                                f"[Blackboard] Updated with {len(last_tool_msgs)} tool results"
                            )

                        # Phase 3: Stuck state detection and rollback (after tool execution)
                        if self.enable_checkpointing and self.checkpoint_manager:
                            # Record tool execution for stuck state detection
                            for tool_call in tool_calls:
                                function_name = tool_call.get("function", {}).get(
                                    "name"
                                )
                                if function_name:
                                    self.checkpoint_manager.record_tool_execution(
                                        function_name
                                    )

                            # Check for stuck state (same tool called 3+ times)
                            if self.checkpoint_manager.is_stuck():
                                logger.warning(
                                    "[Checkpoint] ⚠️  Stuck state detected - initiating rollback"
                                )

                                # Get rollback checkpoint
                                rollback_cp = (
                                    self.checkpoint_manager.get_rollback_checkpoint()
                                )
                                if rollback_cp and self.blackboard:
                                    # Rollback blackboard state
                                    self.blackboard.rollback(
                                        rollback_cp.blackboard_snapshot
                                    )
                                    self.blackboard.add_error(
                                        "Detected stuck state, rolled back to checkpoint"
                                    )

                                    # Log rollback details
                                    logger.info(
                                        f"[Checkpoint] ↶ Rolled back to iteration {rollback_cp.iteration} "
                                        f"(checkpoint {rollback_cp.checkpoint_id})"
                                    )
                                    self._stream_event(
                                        "checkpoint_rollback",
                                        f"Stuck state detected, rolled back to iteration {rollback_cp.iteration}",
                                    )
                                elif rollback_cp:
                                    logger.warning(
                                        "[Checkpoint] Rollback checkpoint found but Blackboard not enabled - cannot rollback"
                                    )
                                else:
                                    logger.warning(
                                        "[Checkpoint] Stuck state detected but no checkpoint available for rollback"
                                    )

                        # Phase 4: TAO OBSERVE phase (if TAO enabled and tools were executed)
                        if (
                            self.enable_tao_loop
                            and "tao_tool_calls" in locals()
                            and tao_tool_calls
                        ):
                            # Prepare results for OBSERVE phase
                            tao_act_results = []
                            for i, tool_call in enumerate(tao_tool_calls):
                                tool_name = tool_call.get("function", {}).get(
                                    "name", "unknown"
                                )
                                # Get corresponding result from last_tool_msgs
                                if i < len(last_tool_msgs):
                                    result = last_tool_msgs[i].get(
                                        "content", "No result"
                                    )
                                else:
                                    result = "No result available"

                                tao_act_results.append(
                                    {
                                        "tool_name": tool_name,
                                        "tool_args": tool_call.get("function", {}).get(
                                            "arguments", {}
                                        ),
                                        "result": result,
                                        "prediction": None,  # Would be set by ACT phase if dry_run enabled
                                        "prediction_accuracy": None,
                                    }
                                )

                            # Call OBSERVE phase to synthesize observations
                            observation_summary = await self._tao_observe_phase(
                                act_results=tao_act_results, iteration_count=step_num
                            )

                            logger.info(
                                f"[TAO] Complete: THINK → ACT → OBSERVE cycle finished for iteration {step_num}"
                            )

                        # Build llm_history based on history_mode
                        # Phase 2: Use Blackboard if enabled (70-80% token reduction)
                        if self.enable_blackboard and self.blackboard:
                            # Use Blackboard structured state instead of linear history
                            blackboard_summary = self.blackboard.to_prompt()
                            llm_history = [
                                system_message,
                                {
                                    "role": "user",
                                    "content": f"{user_message['content'] if user_message else ''}\n\nCURRENT STATE:\n{blackboard_summary}",
                                },
                            ]
                            logger.debug(
                                f"[Blackboard] Using structured state ({len(blackboard_summary)} chars)"
                            )
                        elif self.history_mode == HistoryMode.MINIMAL.value:
                            # MINIMAL: Only keep last assistant + tool results (original behavior)
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.append(last_tool_call_assistant_msg)
                            llm_history.extend(last_tool_msgs)

                        elif self.history_mode == HistoryMode.PROGRESSIVE.value:
                            # PROGRESSIVE: Accumulate assistant messages + tool results
                            # Add to conversation history
                            self.conversation_history.append(
                                last_tool_call_assistant_msg
                            )
                            self.conversation_history.extend(last_tool_msgs)

                            # Build history: system + user + accumulated conversation
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.extend(self.conversation_history)

                        elif self.history_mode == HistoryMode.KITCHEN_SINK.value:
                            # KITCHEN_SINK: Keep everything including user messages between iterations
                            self.conversation_history.append(
                                last_tool_call_assistant_msg
                            )
                            self.conversation_history.extend(last_tool_msgs)

                            # Build history: system + user + accumulated everything
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.extend(self.conversation_history)

                        # Smart context management: Check if summarization needed
                        estimated_tokens = estimate_tokens(llm_history)

                        if self._should_summarize(step_num, estimated_tokens):
                            # Summarize accumulated history
                            summarized = await self._summarize_conversation_history()
                            if summarized:
                                # Rebuild llm_history with summarized conversation
                                llm_history = [system_message]
                                if user_message:
                                    llm_history.append(user_message)
                                llm_history.extend(self.conversation_history)

                                # Log the new token count
                                new_tokens = estimate_tokens(llm_history)
                                logger.info(
                                    f"Post-summarization tokens: {new_tokens} (was {estimated_tokens})"
                                )
                        elif (
                            self.max_context_tokens
                            and estimated_tokens > self.max_context_tokens
                        ):
                            # Fallback warning if summarization not enabled/available
                            logger.warning(
                                f"Context size ({estimated_tokens} tokens) exceeds max_context_tokens "
                                f"({self.max_context_tokens}). Consider enabling summarization or "
                                f"using 'minimal' history_mode. History mode: {self.history_mode}"
                            )

                        # After tool(s) executed, continue inner while loop (do not increment step_num)
                        continue
                    else:
                        # No tool calls, check for content or clarify
                        final_answer_content = None

                        # Try multiple approaches to get content
                        if isinstance(response_message, dict):
                            final_answer_content = response_message.get("content")
                        else:
                            final_answer_content = getattr(
                                response_message, "content", None
                            )

                        if final_answer_content is not None:
                            clarification_attempts = 0  # Reset on final answer
                            logger.info("LLM provided final answer for the step.")
                            final_answer = str(final_answer_content)

                            # Stream the final answer
                            self._stream_event("answer", final_answer)

                            # Phase 2: Mark step complete in Blackboard
                            if self.enable_blackboard and self.blackboard:
                                self.blackboard.mark_step_complete(
                                    f"Iteration {step_num + 1}: Completed successfully"
                                )

                            # T052: Progress callback for completion
                            if self.progress_callback:
                                self.progress_callback(
                                    step_num + 1, self.max_internal_steps, "Complete"
                                )

                            break  # Break inner while loop, increment step_num
                        else:
                            clarification_attempts += 1
                            step_clarification_attempts += 1
                            logger.warning(
                                f"LLM did not request tools and did not provide content. Clarification attempt {clarification_attempts}/{max_clarification_attempts}."
                            )
                            # Add a message indicating confusion or request for clarification
                            clarification_msg = {
                                "role": "user",
                                "content": "Please either call a tool or provide the final answer.",
                            }
                            internal_history.append(clarification_msg)
                            # For next LLM call, send: system, user, clarification
                            llm_history = [system_message]
                            if user_message:
                                llm_history.append(user_message)
                            llm_history.append(clarification_msg)
                            if clarification_attempts >= max_clarification_attempts:
                                logger.error(
                                    f"Agentic step exceeded {max_clarification_attempts} clarification attempts without tool call or answer. Breaking loop."
                                )
                                final_answer = "Error: LLM did not call a tool or provide an answer after multiple attempts."
                                step_error = final_answer
                                execution_errors.append(final_answer)
                                break
                            continue
                except Exception as llm_error:
                    logger.error(
                        f"Error during LLM call in agentic step: {llm_error}",
                        exc_info=True,
                    )
                    error_msg = f"Error during agentic step processing: {llm_error}"
                    final_answer = error_msg
                    step_error = error_msg
                    execution_errors.append(error_msg)
                    break  # Break inner while loop, increment step_num

            # Create step metadata after inner while loop completes
            step_execution_time = (
                datetime.now() - step_start_time
            ).total_seconds() * 1000
            step_tokens = estimate_tokens(llm_history)
            total_tokens_used += step_tokens

            step_metadata = StepExecutionMetadata(
                step_number=step_num + 1,
                tool_calls=step_tool_calls,
                tokens_used=step_tokens,
                execution_time_ms=step_execution_time,
                clarification_attempts=step_clarification_attempts,
                error=step_error,
            )
            steps_metadata.append(step_metadata)

            # ✅ OBSERVABILITY (v0.4.2): Emit event AFTER iteration completes with actual activity
            if (
                callback_manager
                and hasattr(callback_manager, "has_callbacks")
                and callback_manager.has_callbacks()
            ):
                from .execution_events import (ExecutionEvent,
                                               ExecutionEventType)

                # Get the assistant's thought/decision from this iteration
                assistant_thought = None
                for msg in reversed(internal_history):
                    if msg.get("role") == "assistant":
                        assistant_thought = msg.get("content", "")
                        if not assistant_thought and msg.get("tool_calls"):
                            assistant_thought = (
                                f"[Called {len(msg.get('tool_calls') or [])} tool(s)]"
                            )
                        break

                # ✅ Calculate internal history tokens for observability (v0.4.2)
                internal_history_tokens = 0
                try:
                    import tiktoken

                    enc = tiktoken.encoding_for_model("gpt-4")
                    # Serialize internal history to string for token counting
                    history_str = json.dumps(internal_history)
                    internal_history_tokens = len(enc.encode(history_str))
                except Exception:
                    # Fallback to character-based estimate
                    history_str = json.dumps(internal_history)
                    internal_history_tokens = len(history_str) // 4

                await callback_manager.emit(
                    ExecutionEvent(
                        event_type=ExecutionEventType.AGENTIC_INTERNAL_STEP,
                        timestamp=datetime.now(),
                        metadata={
                            "objective": self.objective,
                            "iteration": step_num + 1,
                            "max_iterations": self.max_internal_steps,
                            "assistant_thought": assistant_thought or "[No output]",
                            "tool_calls": step_tool_calls,
                            "tools_called_count": len(step_tool_calls),
                            "execution_time_ms": step_execution_time,
                            "tokens_used": step_tokens,
                            "has_final_answer": final_answer is not None,
                            "error": step_error,
                            # ✅ NEW (v0.4.2): Full internal conversation history for observability
                            "internal_history": internal_history,  # Complete conversation at this iteration
                            "internal_history_length": len(
                                internal_history
                            ),  # Number of messages
                            "internal_history_tokens": internal_history_tokens,  # Token count
                            "llm_history_sent": llm_history,  # What was actually sent to LLM (might differ in progressive mode)
                        },
                    )
                )

            # If we have a final answer or error, break outer for loop
            if final_answer is not None:
                break

        # After loop finishes (or breaks)
        if final_answer is None:
            logger.warning(
                f"Agentic step reached max iterations ({self.max_internal_steps}) without a final answer."
            )
            execution_warnings.append(
                f"Reached max iterations ({self.max_internal_steps}) without final answer"
            )
            # Try to return the last assistant message with content, if any
            last_content = None
            for msg in reversed(internal_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_content = msg["content"]
                    break
            if last_content:
                final_answer = last_content
            else:
                final_answer = "No tool was called, and the LLM did not provide an answer. Please try rephrasing your request or check tool availability."

        logger.info(f"Agentic step finished. Final output: {final_answer[:150]}...")

        # Calculate total execution time
        total_execution_time = (
            datetime.now() - execution_start_time
        ).total_seconds() * 1000

        # ✅ Store execution metadata as instance attributes for event emission and test access
        self.steps_executed = len(steps_metadata)
        self.total_tools_called = total_tools_called
        self.total_tokens_used = total_tokens_used
        self.execution_time_ms = total_execution_time
        self.max_steps_reached = len(steps_metadata) >= self.max_internal_steps
        self._internal_history = (
            internal_history  # Store for internal_history property access
        )

        # Log two-tier routing statistics
        if self.enable_two_tier_routing:
            total_calls = self._fast_model_count + self._slow_model_count
            if total_calls > 0:
                savings_pct = (self._fast_model_count / total_calls) * 100
                logger.info(
                    f"[Two-Tier Summary] Fast model: {self._fast_model_count}/{total_calls} calls ({savings_pct:.1f}% routed), "
                    f"Estimated cost savings: ~{savings_pct * 0.9:.1f}% (assuming 10x cost difference)"
                )

        # Return metadata if requested, otherwise just the answer
        if return_metadata:
            return AgenticStepResult(
                final_answer=final_answer,
                total_steps=len(steps_metadata),
                max_steps_reached=(len(steps_metadata) >= self.max_internal_steps),
                objective_achieved=(
                    final_answer is not None
                    and not any(steps_metadata[-1].error for _ in [1] if steps_metadata)
                ),
                steps=steps_metadata,
                total_tools_called=total_tools_called,
                total_tokens_used=total_tokens_used,
                total_execution_time_ms=total_execution_time,
                history_mode=self.history_mode,
                max_internal_steps=self.max_internal_steps,
                model_name=self.model_name,
                errors=execution_errors,
                warnings=execution_warnings,
            )
        else:
            # Backward compatible: just return the string
            return final_answer

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAO LOOP METHODS (Phase 4: Explicit Think-Act-Observe)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _tao_think_phase(
        self,
        llm_runner: Callable[..., Awaitable[Any]],
        llm_history: List[Dict],
        iteration_count: int,
        available_tools: List[Dict],
    ) -> Dict[str, Any]:
        """
        THINK phase: LLM reasons about next action.

        Explicitly prompts the LLM to think about:
        1. What is the next best action to achieve the objective?
        2. What information do we need?
        3. What are the potential risks?
        4. Which tool should we use (if any)?

        Args:
            llm_runner: Async function that calls LLM
            llm_history: Current conversation history
            iteration_count: Current iteration number
            available_tools: List of available tool schemas

        Returns:
            Dict with 'response', 'tool_calls', and 'has_tool_calls'
        """
        logger.info(f"[TAO] 🧠 THINK phase (iteration {iteration_count})")

        # Start MLflow span for observability
        if MLFLOW_AVAILABLE:
            span_context = mlflow.start_span(
                name=f"TAO_THINK_iter_{iteration_count}", span_type="CHAIN"
            )
            span = span_context.__enter__()
        else:
            span_context = None
            span = None

        try:
            # Add explicit thinking instruction to history
            thinking_history = llm_history.copy()

            # Build tool usage guidance if tools are available
            if available_tools:
                tool_names = [
                    tool.get("function", {}).get("name", "unknown")
                    for tool in available_tools
                ]
                tool_list = "\n".join([f"   - {name}" for name in tool_names])
                tool_guidance = (
                    f"\nAVAILABLE TOOLS:\n{tool_list}\n\n"
                    "YOU MUST USE THESE TOOLS to gather information and accomplish the objective. "
                    "DO NOT try to answer without using tools first.\n"
                )
            else:
                tool_guidance = ""

            thinking_history.append(
                {
                    "role": "system",
                    "content": (
                        "THINK CAREFULLY:\n"
                        "1. What is the next best action to achieve the objective?\n"
                        "2. What information do we currently have?\n"
                        "3. What information do we still need?\n"
                        "4. What tool should we use to get that information?\n"
                        f"{tool_guidance}"
                        "If you need information, call a tool now. Do not provide an answer without using tools first."
                    ),
                }
            )

            # Call LLM for thinking/reasoning
            logger.debug(
                f"[TAO] DEBUG: Calling LLM with {len(available_tools) if available_tools else 0} tools"
            )
            response = await llm_runner(
                messages=thinking_history,
                tools=available_tools if available_tools else None,
            )

            # DEBUG: Log full response to understand structure
            logger.debug(f"[TAO] DEBUG: Response type: {type(response)}")
            logger.debug(f"[TAO] DEBUG: Response: {response}")

            # Extract tool calls if present
            tool_calls = []
            if isinstance(response, dict):
                tool_calls = response.get("tool_calls", [])
                logger.debug(f"[TAO] DEBUG: tool_calls from dict: {tool_calls}")
            else:
                tool_calls = getattr(response, "tool_calls", [])
                logger.debug(f"[TAO] DEBUG: tool_calls from attr: {tool_calls}")
                # Also check if it's in content
                content = getattr(response, "content", None)
                if content:
                    logger.debug(f"[TAO] DEBUG: Response content: {content}")

            has_tool_calls = bool(tool_calls and len(tool_calls) > 0)

            logger.info(
                f"[TAO] THINK complete: tool_calls={len(tool_calls) if has_tool_calls else 0}"
            )

            # Log to MLflow span
            if span:
                span.set_attribute("iteration", iteration_count)
                span.set_attribute(
                    "tool_calls_count", len(tool_calls) if has_tool_calls else 0
                )
                span.set_attribute("has_tool_calls", has_tool_calls)

            return {
                "response": response,
                "tool_calls": tool_calls,
                "has_tool_calls": has_tool_calls,
            }
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    async def _tao_act_phase(
        self,
        tool_calls: List[Dict],
        llm_runner: Callable[..., Awaitable[Any]],
        iteration_count: int,
    ) -> List[Dict[str, Any]]:
        """
        ACT phase: Execute tools with optional dry run prediction.

        For each tool call:
        1. Optionally predict outcome (if dry_run enabled)
        2. Execute the tool
        3. Optionally compare prediction to actual (if dry_run enabled)
        4. Collect results

        Args:
            tool_calls: List of tool calls to execute
            llm_runner: Async function that calls LLM
            iteration_count: Current iteration number

        Returns:
            List of dicts with tool execution results, each containing:
            - tool_name: Name of the tool
            - tool_args: Arguments passed
            - result: Tool execution result
            - prediction: DryRunPrediction if dry_run enabled, else None
            - prediction_accuracy: Similarity score if dry_run enabled, else None
        """
        logger.info(
            f"[TAO] ⚡ ACT phase (iteration {iteration_count}, {len(tool_calls)} tool calls)"
        )

        # Start MLflow span for observability
        if MLFLOW_AVAILABLE:
            span_context = mlflow.start_span(
                name=f"TAO_ACT_iter_{iteration_count}", span_type="TOOL"
            )
            span = span_context.__enter__()
        else:
            span_context = None
            span = None

        try:
            # Initialize dry run predictor if needed (lazy init)
            if self.enable_dry_run and self.dry_run_predictor is None:
                from promptchain.utils.dry_run import DryRunPredictor

                # Use fallback model for predictions if available (cheaper/faster)
                prediction_model = (
                    self.fallback_model if self.fallback_model else self.model_name
                ) or ""
                self.dry_run_predictor = DryRunPredictor(llm_runner, prediction_model)
                logger.info(
                    f"[DryRun] Initialized predictor with model: {prediction_model}"
                )

            results = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

                try:
                    tool_args = (
                        json.loads(tool_args_str)
                        if isinstance(tool_args_str, str)
                        else tool_args_str
                    )
                except json.JSONDecodeError:
                    tool_args = {"raw_args": tool_args_str}

                # Optional: Dry run prediction
                prediction = None
                if self.enable_dry_run and self.dry_run_predictor:
                    context = (
                        self.blackboard.to_prompt()
                        if (self.enable_blackboard and self.blackboard)
                        else "No context available"
                    )

                    prediction = await self.dry_run_predictor.predict_outcome(
                        tool_name=tool_name, tool_args=tool_args, context=context
                    )

                    logger.info(
                        f"[DryRun] 🔮 Predicted {tool_name}: "
                        f"{prediction.predicted_output[:100]}... "
                        f"(confidence={prediction.confidence:.2f})"
                    )

                # NOTE: Actual tool execution would happen here in the full integration
                # For now, we're just building the structure for TAO phase methods
                # The actual execution will be integrated in Phase 4.4

                result_dict = {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": None,  # Will be filled by actual execution
                    "prediction": prediction,
                    "prediction_accuracy": None,  # Will be calculated after execution
                }

                results.append(result_dict)

            logger.info(f"[TAO] ACT complete: {len(results)} tools processed")

            # Log to MLflow span
            if span:
                span.set_attribute("iteration", iteration_count)
                span.set_attribute("tool_calls_count", len(tool_calls))
                span.set_attribute("tools_processed", len(results))

            return results
        finally:
            if span_context:
                span_context.__exit__(None, None, None)

    async def _tao_observe_phase(
        self, act_results: List[Dict[str, Any]], iteration_count: int
    ) -> str:
        """
        OBSERVE phase: Synthesize observations from action results.

        Processes results from ACT phase:
        1. Summarizes what happened
        2. Compares predictions to actual (if dry_run enabled)
        3. Updates Blackboard with observations (if enabled)
        4. Returns observation summary for next iteration

        Args:
            act_results: Results from ACT phase
            iteration_count: Current iteration number

        Returns:
            Observation summary string
        """
        logger.info(f"[TAO] 👁️  OBSERVE phase (iteration {iteration_count})")

        # Start MLflow span for observability
        if MLFLOW_AVAILABLE:
            span_context = mlflow.start_span(
                name=f"TAO_OBSERVE_iter_{iteration_count}", span_type="CHAIN"
            )
            span = span_context.__enter__()
        else:
            span_context = None
            span = None

        try:
            observations = []

            for result in act_results:
                tool_name = result["tool_name"]
                actual_result = result.get("result", "No result available")
                prediction = result.get("prediction")

                # Format observation
                obs_text = f"Executed {tool_name}"

                # Add result preview
                if actual_result and actual_result != "No result available":
                    result_preview = str(actual_result)[:200]
                    obs_text += f": {result_preview}"
                    if len(str(actual_result)) > 200:
                        obs_text += "..."

                # Compare prediction to actual if dry_run enabled
                if prediction and actual_result and self.dry_run_predictor:
                    similarity = self.dry_run_predictor.compare_prediction_to_actual(
                        prediction, str(actual_result)
                    )
                    result["prediction_accuracy"] = similarity

                    if similarity > 0.7:
                        accuracy_emoji = "✅"
                    elif similarity > 0.4:
                        accuracy_emoji = "⚠️"
                    else:
                        accuracy_emoji = "❌"

                    logger.info(
                        f"[DryRun] {accuracy_emoji} Prediction accuracy for {tool_name}: {similarity:.2f}"
                    )

                    obs_text += f" [Prediction accuracy: {similarity:.2f}]"

                observations.append(obs_text)

                # Update Blackboard if enabled
                if self.enable_blackboard and self.blackboard and actual_result:
                    self.blackboard.add_observation(obs_text)

            observation_summary = "\n".join(observations)
            logger.info(
                f"[TAO] OBSERVE complete: {len(observations)} observations synthesized"
            )

            # Log to MLflow span
            if span:
                span.set_attribute("iteration", iteration_count)
                span.set_attribute("observations_count", len(observations))

            return observation_summary
        finally:
            if span_context:
                span_context.__exit__(None, None, None)
