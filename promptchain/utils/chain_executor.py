"""
ChainExecutor - Executes chain definitions with strict guardrails.

Converts ChainDefinition to PromptChain instructions and executes
with full guardrail enforcement, nested chain resolution, and
execution tracking.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from .chain_factory import ChainFactory, ChainFactoryError, ChainNotFoundError
from .chain_models import (ChainDefinition, ChainExecutionRecord, ChainMode,
                           ChainStepDefinition, StepType, ValidationResult)

logger = logging.getLogger(__name__)


class ChainExecutionError(Exception):
    """Error during chain execution."""

    def __init__(
        self,
        message: str,
        step_id: Optional[str] = None,
        chain_vin: Optional[str] = None,
    ):
        self.step_id = step_id
        self.chain_vin = chain_vin
        super().__init__(message)


class ChainTimeoutError(ChainExecutionError):
    """Chain execution timed out."""

    pass


class ChainGuardFailure(ChainExecutionError):
    """Guardrail check failed during execution."""

    pass


class ChainExecutor:
    """Executes chain definitions with strict guardrails.

    The executor converts ChainDefinitions into PromptChain instances
    and executes them step-by-step with:
    - Guardrail enforcement (forbidden patterns, step limits, timeouts)
    - Nested chain resolution and execution
    - Execution tracking and analytics
    - Error handling and recovery

    Usage:
        factory = ChainFactory()
        executor = ChainExecutor(factory)

        chain = factory.create("query-optimizer")
        result = await executor.execute(chain, "my input query")
    """

    def __init__(
        self,
        factory: Optional[ChainFactory] = None,
        default_model: str = "openai/gpt-4.1-mini-2025-04-14",
        verbose: bool = False,
        track_executions: bool = True,
        max_nested_depth: int = 5,
    ):
        """Initialize ChainExecutor.

        Args:
            factory: ChainFactory for resolving nested chains
            default_model: Default LLM model for prompt steps
            verbose: Enable verbose logging
            track_executions: Track execution records for analytics
            max_nested_depth: Maximum chain nesting depth
        """
        self.factory = factory or ChainFactory()
        self.default_model = default_model
        self.verbose = verbose
        self.track_executions = track_executions
        self.max_nested_depth = max_nested_depth

        # Execution state
        self._current_depth = 0
        self._execution_records: List[ChainExecutionRecord] = []

        # Registered functions for function steps
        self._registered_functions: Dict[str, Callable] = {}

    def register_function(self, name: str, func: Callable[[str], str]):
        """Register a function for use in function steps.

        Args:
            name: Function name (must match function_name in step)
            func: Function that takes string input and returns string output
        """
        self._registered_functions[name] = func

    # =========================================================================
    # Main Execution Methods
    # =========================================================================

    async def execute_async(
        self,
        chain: ChainDefinition,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a chain asynchronously.

        Args:
            chain: ChainDefinition to execute
            input_data: Input string for the chain
            context: Optional context dict for variable substitution

        Returns:
            Final output string

        Raises:
            ChainExecutionError: If execution fails
            ChainTimeoutError: If execution times out
            ChainGuardFailure: If guardrails are violated
        """
        start_time = time.time()
        self._current_depth += 1

        # Check nesting depth
        if self._current_depth > self.max_nested_depth:
            self._current_depth -= 1
            raise ChainExecutionError(
                f"Maximum nesting depth ({self.max_nested_depth}) exceeded",
                chain_vin=chain.vin,
            )

        # Validate chain before execution
        validation = self.factory.validate(chain)
        if not validation.passed:
            self._current_depth -= 1
            errors = [i.message for i in validation.issues if i.severity == "error"]
            raise ChainGuardFailure(
                f"Chain validation failed: {'; '.join(errors)}", chain_vin=chain.vin
            )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_chain_steps(chain, input_data, context),
                timeout=chain.guardrails.timeout_seconds,
            )

            # Track execution
            if self.track_executions:
                self._record_execution(
                    chain=chain,
                    input_data=input_data,
                    output=result,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    success=True,
                )

            return result

        except asyncio.TimeoutError:
            self._current_depth -= 1
            raise ChainTimeoutError(
                f"Chain execution timed out after {chain.guardrails.timeout_seconds}s",
                chain_vin=chain.vin,
            )
        except Exception as e:
            self._current_depth -= 1
            if self.track_executions:
                self._record_execution(
                    chain=chain,
                    input_data=input_data,
                    output=None,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    success=False,
                    error=str(e),
                )
            raise
        finally:
            self._current_depth -= 1

    def execute(
        self,
        chain: ChainDefinition,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a chain synchronously.

        Args:
            chain: ChainDefinition to execute
            input_data: Input string for the chain
            context: Optional context dict for variable substitution

        Returns:
            Final output string
        """
        return asyncio.run(self.execute_async(chain, input_data, context))

    async def _execute_chain_steps(
        self,
        chain: ChainDefinition,
        input_data: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute chain steps sequentially.

        Args:
            chain: ChainDefinition to execute
            input_data: Current input
            context: Variable context

        Returns:
            Final output after all steps
        """
        current_input = input_data
        context = context or {}
        context["input"] = input_data

        if self.verbose:
            logger.info(f"Executing chain: {chain.vin} ({len(chain.steps)} steps)")

        for i, step in enumerate(chain.steps):
            if self.verbose:
                logger.debug(
                    f"Step {i+1}/{len(chain.steps)}: {step.id} ({step.type.value})"
                )

            # Check for forbidden patterns in current input
            self._check_forbidden_patterns(current_input, chain, step.id)

            # Execute step based on type
            if step.type == StepType.PROMPT:
                current_input = await self._execute_prompt_step(
                    step, current_input, chain, context
                )

            elif step.type == StepType.CHAIN:
                current_input = await self._execute_chain_step(
                    step, current_input, context
                )

            elif step.type == StepType.FUNCTION:
                current_input = await self._execute_function_step(step, current_input)

            elif step.type == StepType.AGENTIC:
                if chain.mode == ChainMode.STRICT:
                    raise ChainGuardFailure(
                        "Agentic steps not allowed in strict mode",
                        step_id=step.id,
                        chain_vin=chain.vin,
                    )
                current_input = await self._execute_agentic_step(
                    step, current_input, chain
                )

            # Update context
            context["input"] = current_input
            context[f"step_{step.id}_output"] = current_input

        return current_input

    # =========================================================================
    # Step Execution Methods
    # =========================================================================

    async def _execute_prompt_step(
        self,
        step: ChainStepDefinition,
        input_data: str,
        chain: ChainDefinition,
        context: Dict[str, Any],
    ) -> str:
        """Execute a prompt step using LLM.

        Args:
            step: Step definition
            input_data: Current input
            chain: Parent chain (for model config)
            context: Variable context

        Returns:
            LLM response
        """
        # Import here to avoid circular imports
        from .preprompt import PrePrompt
        from .promptchaining import PromptChain

        # Get prompt content
        if step.prompt_id:
            # Load from PrePrompt
            preprompt = PrePrompt()
            prompt_template = preprompt.load(step.prompt_id)
        elif step.content:
            prompt_template = step.content
        else:
            raise ChainExecutionError(
                "Prompt step has no prompt_id or content",
                step_id=step.id,
                chain_vin=chain.vin,
            )

        # Substitute variables
        prompt = prompt_template.format(**context)

        # Create minimal PromptChain for single step
        mini_chain = PromptChain(
            models=[chain.llm_model], instructions=[prompt], verbose=self.verbose
        )

        result = await mini_chain.process_prompt_async(input_data)
        return result

    async def _execute_chain_step(
        self, step: ChainStepDefinition, input_data: str, context: Dict[str, Any]
    ) -> str:
        """Execute a nested chain step.

        Args:
            step: Step definition with chain_id
            input_data: Current input
            context: Variable context

        Returns:
            Nested chain output
        """
        if not step.chain_id:
            raise ChainExecutionError("Chain step missing chain_id", step_id=step.id)

        try:
            nested_chain = self.factory.resolve(step.chain_id)
        except ChainNotFoundError as e:
            raise ChainExecutionError(
                f"Nested chain not found: {step.chain_id}", step_id=step.id
            ) from e

        # Recursive execution
        result = await self.execute_async(nested_chain, input_data, context)
        return result

    async def _execute_function_step(
        self, step: ChainStepDefinition, input_data: str
    ) -> str:
        """Execute a registered function step.

        Args:
            step: Step definition with function_name
            input_data: Current input

        Returns:
            Function output
        """
        if not step.function_name:
            raise ChainExecutionError(
                "Function step missing function_name", step_id=step.id
            )

        if step.function_name not in self._registered_functions:
            raise ChainExecutionError(
                f"Function not registered: {step.function_name}", step_id=step.id
            )

        func = self._registered_functions[step.function_name]

        # Execute function (support both sync and async)
        if asyncio.iscoroutinefunction(func):
            result = await func(input_data)
        else:
            result = func(input_data)

        return str(result)

    async def _execute_agentic_step(
        self, step: ChainStepDefinition, input_data: str, chain: ChainDefinition
    ) -> str:
        """Execute an AgenticStepProcessor step (hybrid mode only).

        Args:
            step: Step definition with objective
            input_data: Current input
            chain: Parent chain

        Returns:
            Agentic processor output
        """
        from .agentic_step_processor import AgenticStepProcessor

        if not step.objective:
            raise ChainExecutionError(
                "Agentic step missing objective", step_id=step.id, chain_vin=chain.vin
            )

        # Create AgenticStepProcessor
        processor = AgenticStepProcessor(  # type: ignore[call-arg]
            objective=step.objective,
            max_internal_steps=step.max_steps or 5,
            model_name=chain.llm_model,
            verbose=self.verbose,
        )

        # Run processor
        result = await processor.run_async(input_data)  # type: ignore[call-arg]

        # Extract output from result
        if hasattr(result, "final_output"):
            return result.final_output
        return str(result)

    # =========================================================================
    # Guardrail Checks
    # =========================================================================

    def _check_forbidden_patterns(
        self, content: str, chain: ChainDefinition, step_id: Optional[str] = None
    ):
        """Check content for forbidden patterns.

        Args:
            content: Content to check
            chain: Chain with guardrails
            step_id: Current step ID for error reporting

        Raises:
            ChainGuardFailure: If forbidden pattern found
        """
        for pattern in chain.guardrails.forbidden_patterns:
            if pattern.lower() in content.lower():
                raise ChainGuardFailure(
                    f"Forbidden pattern detected: '{pattern}'",
                    step_id=step_id,
                    chain_vin=chain.vin,
                )

    # =========================================================================
    # Execution Tracking
    # =========================================================================

    def _record_execution(
        self,
        chain: ChainDefinition,
        input_data: str,
        output: Optional[str],
        execution_time_ms: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record chain execution for analytics.

        Args:
            chain: Executed chain
            input_data: Input string
            output: Output string (None if failed)
            execution_time_ms: Execution time
            success: Whether execution succeeded
            error: Error message if failed
        """
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()[:16]

        record = ChainExecutionRecord(
            vin=chain.vin,
            input_hash=input_hash,
            output=output,
            execution_time_ms=execution_time_ms,
            success=success,
            error_message=error,
            steps_executed=len(chain.steps),
        )

        self._execution_records.append(record)

        if self.verbose:
            status = "SUCCESS" if success else f"FAILED: {error}"
            logger.info(f"Chain {chain.vin}: {status} ({execution_time_ms}ms)")

    def get_execution_records(self) -> List[ChainExecutionRecord]:
        """Get all execution records."""
        return self._execution_records.copy()

    def clear_execution_records(self):
        """Clear execution records."""
        self._execution_records.clear()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def execute_by_id_async(
        self, chain_ref: str, input_data: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a chain by reference (model:version or VIN).

        Args:
            chain_ref: Chain reference (e.g., "query-optimizer:v1.0")
            input_data: Input string
            context: Optional context dict

        Returns:
            Chain output
        """
        chain = self.factory.resolve(chain_ref)
        return await self.execute_async(chain, input_data, context)

    def execute_by_id(
        self, chain_ref: str, input_data: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a chain by reference synchronously."""
        return asyncio.run(self.execute_by_id_async(chain_ref, input_data, context))


# =========================================================================
# Helper Functions
# =========================================================================


def create_executor(
    factory: Optional[ChainFactory] = None, verbose: bool = False
) -> ChainExecutor:
    """Create a ChainExecutor with sensible defaults.

    Args:
        factory: Optional ChainFactory (creates new if None)
        verbose: Enable verbose logging

    Returns:
        Configured ChainExecutor
    """
    return ChainExecutor(factory=factory or ChainFactory(), verbose=verbose)
