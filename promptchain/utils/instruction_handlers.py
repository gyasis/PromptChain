"""
Instruction Handlers for PromptChain.

Clean architecture for handling different instruction types:
- CallableHandler: Python functions
- AgenticHandler: AgenticStepProcessor instances
- ChainHandler: ChainCall or chain: prefix instructions
- ModelHandler: LLM prompts (string instructions)

This replaces the messy if-elif-else blocks with a registry pattern.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .promptchaining import PromptChain
    from .agentic_step_processor import AgenticStepProcessor
    from .chain_models import ChainCall
    from .events import ExecutionEvent, ExecutionEventType, CallbackManager

logger = logging.getLogger(__name__)


class InstructionResult:
    """Result from executing an instruction."""

    def __init__(
        self,
        output: str,
        step_type: str,
        execution_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        self.output = output
        self.step_type = step_type
        self.execution_time_ms = execution_time_ms
        self.metadata = metadata or {}
        self.success = success
        self.error = error


class InstructionContext:
    """Context passed to instruction handlers."""

    def __init__(
        self,
        step_num: int,
        input_content: str,
        current_messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        callback_manager: 'CallbackManager',
        llm_runner: Callable,
        tool_executor: Callable,
        verbose: bool = False,
        full_history: bool = False,
        user_input_queue: Optional[asyncio.Queue] = None,
        streaming_callback: Optional[Callable] = None,
        # Model-specific context
        model_name: Optional[str] = None,
        model_params: Optional[Dict[str, Any]] = None,
        mcp_helper: Optional[Any] = None,
        preprompter: Optional[Any] = None,
    ):
        self.step_num = step_num
        self.input_content = input_content
        self.current_messages = current_messages
        self.available_tools = available_tools
        self.callback_manager = callback_manager
        self.llm_runner = llm_runner
        self.tool_executor = tool_executor
        self.verbose = verbose
        self.full_history = full_history
        self.user_input_queue = user_input_queue
        self.streaming_callback = streaming_callback
        self.model_name = model_name
        self.model_params = model_params
        self.mcp_helper = mcp_helper
        self.preprompter = preprompter


class InstructionHandler(ABC):
    """Base class for instruction handlers."""

    @abstractmethod
    def can_handle(self, instruction: Any) -> bool:
        """Check if this handler can process the instruction."""
        pass

    @abstractmethod
    async def handle(
        self,
        instruction: Any,
        context: InstructionContext
    ) -> InstructionResult:
        """Execute the instruction and return result."""
        pass


class CallableHandler(InstructionHandler):
    """Handler for Python callable instructions."""

    def can_handle(self, instruction: Any) -> bool:
        return callable(instruction) and not hasattr(instruction, 'objective')

    async def handle(
        self,
        instruction: Callable,
        context: InstructionContext
    ) -> InstructionResult:
        step_type = "function"
        start_time = datetime.now()

        if context.verbose:
            func_name = getattr(instruction, '__name__', str(instruction))
            logger.debug(f"Executing function step: {func_name}")

        try:
            exec_start = time.time()

            # Execute function (support both sync and async)
            if asyncio.iscoroutinefunction(instruction):
                output = await instruction(context.input_content)
            else:
                output = instruction(context.input_content)

            # Ensure string output
            if not isinstance(output, str):
                logger.warning(f"Function returned non-string: {type(output)}")
                output = str(output)

            exec_time = time.time() - exec_start
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            return InstructionResult(
                output=output,
                step_type=step_type,
                execution_time_ms=duration_ms,
                metadata={
                    "function_name": getattr(instruction, '__name__', 'anonymous'),
                    "result_length": len(output)
                }
            )

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error in function step: {e}", exc_info=context.verbose)

            return InstructionResult(
                output="",
                step_type=step_type,
                execution_time_ms=duration_ms,
                success=False,
                error=str(e)
            )


class AgenticHandler(InstructionHandler):
    """Handler for AgenticStepProcessor instructions."""

    def can_handle(self, instruction: Any) -> bool:
        return hasattr(instruction, 'objective') and hasattr(instruction, 'run_async')

    async def handle(
        self,
        instruction: 'AgenticStepProcessor',
        context: InstructionContext
    ) -> InstructionResult:
        step_type = "agentic"
        start_time = datetime.now()

        if context.verbose:
            logger.debug(f"Executing agentic step: {instruction.objective[:100]}...")

        try:
            from .events import ExecutionEvent, ExecutionEventType

            # Emit start event
            await context.callback_manager.emit(
                ExecutionEvent(
                    event_type=ExecutionEventType.AGENTIC_STEP_START,
                    timestamp=start_time,
                    step_number=context.step_num,
                    metadata={
                        "objective": instruction.objective[:200],
                        "max_internal_steps": instruction.max_internal_steps
                    }
                )
            )

            exec_start = time.time()

            # Execute agentic step
            output = await instruction.run_async(
                initial_input=context.input_content,
                available_tools=context.available_tools,
                llm_runner=context.llm_runner,
                tool_executor=context.tool_executor,
                callback_manager=context.callback_manager,
                user_input_queue=context.user_input_queue,
                streaming_callback=context.streaming_callback
            )

            # Ensure string output
            if not isinstance(output, str):
                output = str(output)

            exec_time = time.time() - exec_start
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Emit end event
            await context.callback_manager.emit(
                ExecutionEvent(
                    event_type=ExecutionEventType.AGENTIC_STEP_END,
                    timestamp=end_time,
                    step_number=context.step_num,
                    metadata={
                        "objective": instruction.objective[:200],
                        "execution_time_ms": duration_ms,
                        "steps_executed": getattr(instruction, 'steps_executed', None),
                        "result_length": len(output),
                        "success": True
                    }
                )
            )

            return InstructionResult(
                output=output,
                step_type=step_type,
                execution_time_ms=duration_ms,
                metadata={
                    "objective": instruction.objective[:200],
                    "steps_executed": getattr(instruction, 'steps_executed', None)
                }
            )

        except Exception as e:
            from .events import ExecutionEvent, ExecutionEventType

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            await context.callback_manager.emit(
                ExecutionEvent(
                    event_type=ExecutionEventType.AGENTIC_STEP_ERROR,
                    timestamp=end_time,
                    step_number=context.step_num,
                    metadata={
                        "objective": instruction.objective[:200],
                        "error": str(e),
                        "execution_time_ms": duration_ms
                    }
                )
            )

            logger.error(f"Error in agentic step: {e}", exc_info=context.verbose)
            raise  # Re-raise to maintain existing behavior


class ChainHandler(InstructionHandler):
    """Handler for ChainCall instructions and chain: prefix strings."""

    def can_handle(self, instruction: Any) -> bool:
        # Handle ChainCall class
        if hasattr(instruction, 'chain_ref'):
            return True
        # Handle chain: prefix string
        if isinstance(instruction, str) and instruction.startswith("chain:"):
            return True
        return False

    def _get_chain_ref(self, instruction: Any) -> str:
        """Extract chain reference from instruction."""
        if hasattr(instruction, 'chain_ref'):
            return instruction.chain_ref
        if isinstance(instruction, str) and instruction.startswith("chain:"):
            return instruction[6:]  # Remove "chain:" prefix
        raise ValueError(f"Cannot extract chain ref from: {instruction}")

    async def handle(
        self,
        instruction: Any,
        context: InstructionContext
    ) -> InstructionResult:
        step_type = "chain"
        chain_ref = self._get_chain_ref(instruction)
        start_time = datetime.now()

        if context.verbose:
            logger.debug(f"Executing chain: {chain_ref}")

        try:
            from .chain_executor import ChainExecutor
            from .chain_factory import ChainFactory
            from .events import ExecutionEvent, ExecutionEventType

            # Create executor
            factory = ChainFactory()
            executor = ChainExecutor(factory=factory, verbose=context.verbose)

            # Resolve and execute chain
            chain_def = factory.resolve(chain_ref)

            exec_start = time.time()
            output = await executor.execute_async(
                chain=chain_def,
                input_data=context.input_content
            )
            exec_time = time.time() - exec_start

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Emit end event
            await context.callback_manager.emit(
                ExecutionEvent(
                    event_type=ExecutionEventType.STEP_END,
                    timestamp=end_time,
                    step_number=context.step_num,
                    metadata={
                        "chain_ref": chain_ref,
                        "chain_vin": chain_def.vin,
                        "execution_time_ms": duration_ms,
                        "result_length": len(output) if output else 0,
                        "success": True
                    }
                )
            )

            return InstructionResult(
                output=output,
                step_type=step_type,
                execution_time_ms=duration_ms,
                metadata={
                    "chain_ref": chain_ref,
                    "chain_vin": chain_def.vin
                }
            )

        except Exception as e:
            from .events import ExecutionEvent, ExecutionEventType

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            await context.callback_manager.emit(
                ExecutionEvent(
                    event_type=ExecutionEventType.STEP_ERROR,
                    timestamp=end_time,
                    step_number=context.step_num,
                    metadata={
                        "chain_ref": chain_ref,
                        "error": str(e),
                        "execution_time_ms": duration_ms
                    }
                )
            )

            logger.error(f"Error executing chain: {e}", exc_info=context.verbose)
            raise


class InstructionHandlerRegistry:
    """Registry of instruction handlers with priority-based dispatch.

    Usage:
        registry = InstructionHandlerRegistry()
        # Handlers are registered by default

        # To handle an instruction:
        handler = registry.get_handler(instruction)
        if handler:
            result = await handler.handle(instruction, context)
    """

    def __init__(self):
        # Handlers in priority order (first match wins)
        self._handlers: List[InstructionHandler] = [
            CallableHandler(),
            AgenticHandler(),
            ChainHandler(),
            # ModelHandler is NOT registered here - it's the fallback
            # handled by PromptChain's existing model execution logic
        ]

    def register(self, handler: InstructionHandler, priority: int = -1):
        """Register a handler at specified priority (0 = highest)."""
        if priority < 0 or priority >= len(self._handlers):
            self._handlers.append(handler)
        else:
            self._handlers.insert(priority, handler)

    def get_handler(self, instruction: Any) -> Optional[InstructionHandler]:
        """Find handler for instruction (first match wins)."""
        for handler in self._handlers:
            if handler.can_handle(instruction):
                return handler
        return None  # No handler found - use default model execution

    def can_handle(self, instruction: Any) -> bool:
        """Check if any handler can process this instruction."""
        return self.get_handler(instruction) is not None


# Global registry instance
_default_registry: Optional[InstructionHandlerRegistry] = None


def get_instruction_registry() -> InstructionHandlerRegistry:
    """Get the default instruction handler registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = InstructionHandlerRegistry()
    return _default_registry
